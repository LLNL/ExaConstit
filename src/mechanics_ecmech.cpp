#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "ECMech_cases.h"
#include "ECMech_evptnWrap.h"
#include "mechanics_integrators.hpp"
#include "mechanics_ecmech.hpp"
#include "BCManager.hpp"
#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr
#include "RAJA/RAJA.hpp"

using namespace mfem;
using namespace std;
using namespace ecmech;

namespace {
// Performs all the calculations related to calculating the velocity gradient
// vel_grad_array should be set to 0.0 outside of this function.
void kernel_vgrad_calc(const int nqpts, const int nelems, const int nnodes,
                       const double *jacobian_data, const double *loc_grad_data,
                       const double *vel_data, double* vel_grad_array)
{
   const int DIM4 = 4;
   const int DIM3 = 3;
   const int DIM2 = 2;
   std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
   std::array<RAJA::idx_t, DIM3> perm3{{ 2, 1, 0 } };
   std::array<RAJA::idx_t, DIM2> perm2{{ 1, 0 } };

   const int dim = 3;
   const int space_dim2 = dim * dim;

   // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
   RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
   RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian_data, layout_jacob);
   // vgrad
   RAJA::Layout<DIM4> layout_vgrad = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
   RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > vgrad_view(vel_grad_array, layout_vgrad);
   // velocity
   RAJA::Layout<DIM3> layout_vel = RAJA::make_permuted_layout({{ nnodes, dim, nelems } }, perm3);
   RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > vel_view(vel_data, layout_vel);
   // loc_grad
   RAJA::Layout<DIM3> layout_loc_grad = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
   RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > loc_grad_view(loc_grad_data, layout_loc_grad);

   RAJA::Layout<DIM2> layout_jinv = RAJA::make_permuted_layout({{ dim, dim } }, perm2);

   MFEM_FORALL(i_elems, nelems, {
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
            const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
            const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
            const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
            const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
            const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
            const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
            const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
            const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
            const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                /* */ J21 * (J12 * J33 - J32 * J13) +
                                /* */ J31 * (J12 * J23 - J22 * J13);
            const double c_detJ = 1.0 / detJ;
            // adj(J)
            const double A11 = c_detJ * ((J22 * J33) - (J23 * J32));
            const double A12 = c_detJ * ((J32 * J13) - (J12 * J33));
            const double A13 = c_detJ * ((J12 * J23) - (J22 * J13));
            const double A21 = c_detJ * ((J31 * J23) - (J21 * J33));
            const double A22 = c_detJ * ((J11 * J33) - (J13 * J31));
            const double A23 = c_detJ * ((J21 * J13) - (J11 * J23));
            const double A31 = c_detJ * ((J21 * J32) - (J31 * J22));
            const double A32 = c_detJ * ((J31 * J12) - (J11 * J32));
            const double A33 = c_detJ * ((J11 * J22) - (J12 * J21));
            const double A[space_dim2] = { A11, A21, A31, A12, A22, A32, A13, A23, A33 };
            // Raja view to make things easier again

            RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > jinv_view(&A[0], layout_jinv);
            // All of the data down below is ordered in column major order
            for (int t = 0; t < dim; t++) {
               for (int s = 0; s < dim; s++) {
                  for (int r = 0; r < nnodes; r++) {
                     for (int q = 0; q < dim; q++) {
                        vgrad_view(q, t, j_qpts, i_elems) += vel_view(r, q, i_elems) *
                                                             loc_grad_view(r, s, j_qpts) * jinv_view(s, t);
                     }
                  }
               }
            } // End of loop used to calculate velocity gradient
         } // end of forall loop for quadrature points
      }); // end of forall loop for number of elements
} // end of vgrad calc func

// Sets-up everything for the kernel
void kernel_setup(const int npts, const int nstatev,
                  const double dt, const double temp_k, const double* vel_grad_array,
                  const double* stress_array, const double* state_vars_array,
                  double* stress_svec_p_array, double* d_svec_p_array,
                  double* w_vec_array, double* vol_ratio_array,
                  double* eng_int_array, double* tempk_array)
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
void kernel_postprocessing(const int npts, const int nstatev,
                           const double* stress_svec_p_array, const double* vol_ratio_array,
                           const double* eng_int_array, double* state_vars_array,
                           double* stress_array, double* ddsdde_array)
{
   const int ind_int_eng = nstatev - ecmech::ne;
   const int ind_vols = ind_int_eng - 1;

   MFEM_FORALL(i_pts, npts, {
         // These are our outputs
         double* state_vars = &(state_vars_array[i_pts * nstatev]);
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
   mat_model_base->getResponse(dt, d_svec_p_array, w_vec_array, vol_ratio_array,
                               eng_int_array, stress_svec_p_array, state_vars_array,
                               tempk_array, sdd_array, ddsdde_array, npts);
}

// Same as the above kernel but the material tangent stiffness matrix is not saved off
void kernel_init(const ecmech::matModelBase* mat_model_base,
                 const int npts, const double dt, double* state_vars_array,
                 double* stress_svec_p_array, double* d_svec_p_array,
                 double* w_vec_array, double* vol_ratio_array,
                 double* eng_int_array, double* tempk_array,
                 double* sdd_array)
{
   mat_model_base->getResponse(dt, d_svec_p_array, w_vec_array, vol_ratio_array,
                               eng_int_array, stress_svec_p_array, state_vars_array,
                               tempk_array, sdd_array, nullptr, npts);
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

   // If we're on the initial step we need to first calculate a
   // solution where our vgrad is the 0 tensor across the entire
   // body. After we obtain this we calculate our actual velocity
   // gradient and use that to obtain the appropriate stress field
   // but don't calculate the material tangent stiffness tensor.
   // Any other step is much simpler, and we just calculate the
   // velocity gradient, run our model, and then obtain our material
   // tangent stiffness matrix.

   kernel_vgrad_calc(nqpts, nelems, nnodes, jacobian_array, loc_grad_array,
                     vel_array, vel_grad_array_data);

   kernel_setup(npts, nstatev, dt, temp_k, vel_grad_array_data,
                stress_array, state_vars_array, stress_svec_p_array_data,
                d_svec_p_array_data, w_vec_array_data,
                vol_ratio_array_data, eng_int_array_data, tempk_array_data);

   if (init_step) {
      // Initially set the velocity gradient to being 0.0

      *d_svec_p_array = 0.0;
      *w_vec_array = 0.0;

      d_svec_p_array_data = d_svec_p_array->ReadWrite();
      w_vec_array_data = w_vec_array->ReadWrite();

      kernel(mat_model_base, npts, dt, state_vars_array,
             stress_svec_p_array_data, d_svec_p_array_data, w_vec_array_data,
             ddsdde_array, vol_ratio_array_data, eng_int_array_data,
             tempk_array_data, sdd_array_data);

      kernel_setup(npts, nstatev, dt, temp_k, vel_grad_array_data,
                   stress_array, state_vars_array, stress_svec_p_array_data,
                   d_svec_p_array_data, w_vec_array_data,
                   vol_ratio_array_data, eng_int_array_data, tempk_array_data);

      kernel_init(mat_model_base, npts, dt, state_vars_array,
                  stress_svec_p_array_data, d_svec_p_array_data, w_vec_array_data,
                  vol_ratio_array_data, eng_int_array_data, tempk_array_data, sdd_array_data);
   }
   else {
      kernel(mat_model_base, npts, dt, state_vars_array,
             stress_svec_p_array_data, d_svec_p_array_data, w_vec_array_data,
             ddsdde_array, vol_ratio_array_data, eng_int_array_data,
             tempk_array_data, sdd_array_data);
   } // endif

   kernel_postprocessing(npts, nstatev, stress_svec_p_array_data,
                         vol_ratio_array_data, eng_int_array_data, state_vars_array,
                         stress_array, ddsdde_array);
} // End of ModelSetup function