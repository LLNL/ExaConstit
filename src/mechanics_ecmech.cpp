#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "ECMech_cases.h"
#include "ECMech_evptnWrap.h"
#include "mechanics_model.hpp"
#include "mechanics_log.hpp"
#include "mechanics_ecmech.hpp"
#include "BCManager.hpp"
#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr
#include "RAJA/RAJA.hpp"
#include "mechanics_kernels.hpp"

using namespace mfem;
using namespace std;
using namespace ecmech;

namespace {

// Sets-up everything for the kernel
void kernel_setup(const int npts, const int nstatev,
                  const double dt, const double temp_k, const double* vel_grad_array,
                  const double* stress_array, const double* state_vars_array,
                  double* stress_svec_p_array, double* d_svec_p_array,
                  double* w_vec_array, double* vol_ratio_array,
                  double* eng_int_array, double* tempk_array, double* dEff)
{
   // vgrad is kinda a pain to deal with as a raw 1d array, so we're
   // going to just use a RAJA view here. The data is taken to be in col. major format.
   // It might be nice to eventually create a type alias for the below or
   // maybe something like it.

   const int ind_int_eng = nstatev - ecmech::ne;
   const int ind_vols = ind_int_eng - 1;

   const int DIM = 3;
   std::array<RAJA::idx_t, DIM> perm {{ 2, 1, 0 } };
   RAJA::Layout<DIM> layout = RAJA::make_permuted_layout({{ ecmech::ndim, ecmech::ndim, npts } }, perm);
   RAJA::View<const double, RAJA::Layout<DIM, RAJA::Index_type, 0> > vgrad_view(vel_grad_array, layout);

   MFEM_FORALL(i_pts, npts, {
         // Might want to eventually set these all up using RAJA views. It might simplify
         // things later on.
         // These are our inputs
         const double* state_vars = &(state_vars_array[i_pts * nstatev]);
         const double* stress = &(stress_array[i_pts * ecmech::nsvec]);
         // Here is all of our ouputs
         double* eng_int = &(eng_int_array[i_pts * ecmech::ne]);
         double* w_vec = &(w_vec_array[i_pts * ecmech::nwvec]);
         double* vol_ratio = &(vol_ratio_array[i_pts * ecmech::nvr]);
         // A few variables are set up as the 6-vec deviatoric + tr(tens) values
         int ind_svecp = i_pts * ecmech::nsvp;
         double* stress_svec_p = &(stress_svec_p_array[ind_svecp]);
         double* d_svec_p = &(d_svec_p_array[ind_svecp]);

         tempk_array[i_pts] = temp_k;

         for (int i = 0; i < ecmech::ne; i++) {
            eng_int[i] = state_vars[ind_int_eng + i];
         }

         // Here we have the skew portion of our velocity gradient as represented as an
         // axial vector.
         w_vec[0] = 0.5 * (vgrad_view(2, 1, i_pts) - vgrad_view(1, 2, i_pts));
         w_vec[1] = 0.5 * (vgrad_view(0, 2, i_pts) - vgrad_view(2, 0, i_pts));
         w_vec[2] = 0.5 * (vgrad_view(1, 0, i_pts) - vgrad_view(0, 1, i_pts));

         // Really we're looking at the negative of J but this will do...
         double d_mean = -ecmech::onethird * (vgrad_view(0, 0, i_pts) + vgrad_view(1, 1, i_pts) + vgrad_view(2, 2, i_pts));
         // The 1st 6 components are the symmetric deviatoric portion of our velocity gradient
         // The last value is simply the trace of the deformation rate
         d_svec_p[0] = vgrad_view(0, 0, i_pts) + d_mean;
         d_svec_p[1] = vgrad_view(1, 1, i_pts) + d_mean;
         d_svec_p[2] = vgrad_view(2, 2, i_pts) + d_mean;
         d_svec_p[3] = 0.5 * (vgrad_view(2, 1, i_pts) + vgrad_view(1, 2, i_pts));
         d_svec_p[4] = 0.5 * (vgrad_view(2, 0, i_pts) + vgrad_view(0, 2, i_pts));
         d_svec_p[5] = 0.5 * (vgrad_view(1, 0, i_pts) + vgrad_view(0, 1, i_pts));
         d_svec_p[6] = -3.0 * d_mean;

         double d_vecd_sm[ecmech::ntvec];
         ecmech::svecToVecd(d_vecd_sm, d_svec_p);
         dEff[i_pts] = ecmech::vecd_Deff(d_vecd_sm);

         vol_ratio[0] = state_vars[ind_vols];
         vol_ratio[1] = vol_ratio[0] * exp(d_svec_p[ecmech::iSvecP] * dt);
         vol_ratio[3] = vol_ratio[1] - vol_ratio[0];
         vol_ratio[2] = vol_ratio[3] / (dt * 0.5 * (vol_ratio[0] + vol_ratio[1]));

         for (int i = 0; i < ecmech::nsvec; i++) {
            stress_svec_p[i] = stress[i];
         }

         double stress_mean = -ecmech::onethird * (stress[0] + stress[1] + stress[2]);
         stress_svec_p[0] += stress_mean;
         stress_svec_p[1] += stress_mean;
         stress_svec_p[2] += stress_mean;
         stress_svec_p[ecmech::iSvecP] = stress_mean;
      }); // end of npts loop
} // end of set-up func

// Retrieves the stress and reorders it into the desired 6 vec format. A copy of that vector
// is sent back to the CPU for the time being. It also stores all of the state variables into their
// appropriate vector. Finally, it saves off the material tangent stiffness vector. In the future,
// if PA is used then the 4D 3x3x3x3 tensor is saved off rather than the 6x6 2D matrix.
void kernel_postprocessing(const int npts, const int nstatev, const double dt, const double* dEff,
                           const double* stress_svec_p_array, const double* vol_ratio_array,
                           const double* eng_int_array, const double* beg_state_vars_array,
                           double* state_vars_array, double* stress_array,
                           double* ddsdde_array, Assembly assembly)
{
   const int ind_int_eng = nstatev - ecmech::ne;
   const int ind_pl_work = ecmech::evptn::iHistA_flowStr;
   const int ind_vols = ind_int_eng - 1;

   MFEM_FORALL(i_pts, npts, {
         // These are our outputs
         double* state_vars = &(state_vars_array[i_pts * nstatev]);
         const double* beg_state_vars = &(beg_state_vars_array[i_pts * nstatev]);
         double* stress = &(stress_array[i_pts * ecmech::nsvec]);
         // Here is all of our ouputs
         const double* eng_int = &(eng_int_array[i_pts * ecmech::ne]);
         const double* vol_ratio = &(vol_ratio_array[i_pts * ecmech::nvr]);
         // A few variables are set up as the 6-vec deviatoric + tr(tens) values
         int ind_svecp = i_pts * ecmech::nsvp;
         const double* stress_svec_p = &(stress_svec_p_array[ind_svecp]);

         // We need to update our state variables to include the volume ratio and
         // internal energy portions
         state_vars[ind_vols] = vol_ratio[1];
         for (int i = 0; i < ecmech::ne; i++) {
            state_vars[ind_int_eng + i] = eng_int[i];
         }

         if(dEff[i_pts] > ecmech::idp_tiny_sqrt) {
            state_vars[ind_pl_work] *= dEff[i_pts] * dt;
         } else {
            state_vars[ind_pl_work] = 0.0;
         }
         state_vars[ind_pl_work] += beg_state_vars[ind_pl_work];

         // Here we're converting back from our deviatoric + pressure representation of our
         // Cauchy stress back to the Voigt notation of stress.
         double stress_mean = -stress_svec_p[ecmech::iSvecP];
         for (int i = 0; i < ecmech::nsvec; i++) {
            stress[i] = stress_svec_p[i];
         }

         stress[0] += stress_mean;
         stress[1] += stress_mean;
         stress[2] += stress_mean;
      }); // end of npts loop

   // No need to transpose this if running on the GPU and doing EA
   if ((assembly == Assembly::EA) and mfem::Device::Allows(Backend::DEVICE_MASK)) { return; }
   else
   {
      // std::cout << "rotate tan stiffness mat" << std::endl;
      MFEM_FORALL(i_pts, npts, {
         // ExaCMech saves this in Row major, so we need to get out the transpose.
         // The good thing is we can do this all in place no problem.
         double* ddsdde = &(ddsdde_array[i_pts * ecmech::nsvec * ecmech::nsvec]);
         for (int i = 0; i < ecmech::nsvec; ++i) {
            for (int j = i + 1; j < ecmech::nsvec; ++j) {
               double tmp = ddsdde[(ecmech::nsvec * j) +i];
               ddsdde[(ecmech::nsvec * j) +i] = ddsdde[(ecmech::nsvec * i) +j];
               ddsdde[(ecmech::nsvec * i) +j] = tmp;
            }
         }
      });
   }
} // end of post-processing func

// The different CPU, OpenMP, and GPU kernels aren't needed here, since they're
// defined in ExaCMech itself.
void kernel(const ecmech::matModelBase* mat_model_base,
            const int npts, const double dt, double* state_vars_array,
            double* stress_svec_p_array, double* d_svec_p_array,
            double* w_vec_array, double* ddsdde_array,
            double* vol_ratio_array, double* eng_int_array,
            double* tempk_array, double* sdd_array)
{
  mat_model_base->getResponseECM(dt, d_svec_p_array, w_vec_array, vol_ratio_array,
                               eng_int_array, stress_svec_p_array, state_vars_array,
                               tempk_array, sdd_array, ddsdde_array, npts);
}

} // End private namespace

// Our model set-up makes use of several preprocessing kernels,
// the actual material model kernel, and finally a post-processing kernel.
void ExaCMechModel::ModelSetup(const int nqpts, const int nelems, const int /*space_dim*/,
                               const int nnodes, const Vector &jacobian,
                               const Vector &loc_grad, const Vector &vel)
{
   const int nstatev = numStateVars;

   const double *jacobian_array = jacobian.Read();
   const double *loc_grad_array = loc_grad.Read();
   const double *vel_array = vel.Read();

   // Here we call an initialization function which sets the end step stress
   // and state variable variables to the initial time step values.
   // Then the pointer to the underlying data array is returned and
   // operated on to those end time step variables
   double* state_vars_array = StateVarsSetup();
   const double *state_vars_beg = matVars0->Read();
   double* stress_array = StressSetup();
   // If we require a 4D tensor for PA applications then we might
   // need to use something other than this for our applications.
   QuadratureFunction* matGrad_qf = matGrad;
   *matGrad_qf = 0.0;
   double* ddsdde_array = matGrad_qf->ReadWrite();
   // All of these variables are stored on the material model class using
   // the vector class.
   *vel_grad_array = 0.0;
   double* vel_grad_array_data = vel_grad_array->ReadWrite();
   double* stress_svec_p_array_data = stress_svec_p_array->ReadWrite();
   double* d_svec_p_array_data = d_svec_p_array->ReadWrite();
   double* w_vec_array_data = w_vec_array->ReadWrite();
   double* vol_ratio_array_data = vol_ratio_array->ReadWrite();
   double* eng_int_array_data = eng_int_array->ReadWrite();
   double* tempk_array_data = tempk_array->ReadWrite();
   double* sdd_array_data = sdd_array->ReadWrite();

   const int npts = nqpts * nelems;
   double* dEff = eff_def_rate->Write();

   // If we're on the initial step we need to first calculate a
   // solution where our vgrad is the 0 tensor across the entire
   // body. After we obtain this we calculate our actual velocity
   // gradient and use that to obtain the appropriate stress field
   // but don't calculate the material tangent stiffness tensor.
   // Any other step is much simpler, and we just calculate the
   // velocity gradient, run our model, and then obtain our material
   // tangent stiffness matrix.
   CALI_MARK_BEGIN("ecmech_setup");
   exaconstit::kernel::grad_calc(nqpts, nelems, nnodes, jacobian_array, loc_grad_array,
                                 vel_array, vel_grad_array_data);

   kernel_setup(npts, nstatev, dt, temp_k, vel_grad_array_data,
                stress_array, state_vars_array, stress_svec_p_array_data,
                d_svec_p_array_data, w_vec_array_data,
                vol_ratio_array_data, eng_int_array_data, tempk_array_data, dEff);
   CALI_MARK_END("ecmech_setup");
   CALI_MARK_BEGIN("ecmech_kernel");
   kernel(mat_model_base, npts, dt, state_vars_array,
            stress_svec_p_array_data, d_svec_p_array_data, w_vec_array_data,
            ddsdde_array, vol_ratio_array_data, eng_int_array_data,
            tempk_array_data, sdd_array_data);
   CALI_MARK_END("ecmech_kernel");

   CALI_MARK_BEGIN("ecmech_postprocessing");
   kernel_postprocessing(npts, nstatev, dt, dEff, stress_svec_p_array_data,
                         vol_ratio_array_data, eng_int_array_data, state_vars_beg, state_vars_array,
                         stress_array, ddsdde_array, assembly);
   CALI_MARK_END("ecmech_postprocessing");
} // End of ModelSetup function
