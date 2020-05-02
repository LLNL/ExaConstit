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

void ExaCMechModel::UpdateModelVars(){}

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
   QuadratureFunction* matGrad_qf = matGrad.GetQuadFunction();
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

VoceFCCModel::VoceFCCModel(mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
                           mfem::QuadratureFunction *_q_matGrad, mfem::QuadratureFunction *_q_matVars0,
                           mfem::QuadratureFunction *_q_matVars1,
                           mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                           mfem::Vector *_props, int _nProps, int _nStateVars, double _temp_k,
                           ecmech::ExecutionStrategy _accel, bool _PA) :
   ExaCMechModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
                 _beg_coords, _end_coords, _props, _nProps, _nStateVars, _temp_k,
                 _accel, _PA)
{
   // For FCC material models we have the following state variables
   // and their number of components
   // effective shear rate(1), effective shear(1), flow strength(1), n_evals(1), deviatoric elastic strain(5),
   // quaternions(4), h(Kinetics::nH), gdot(SlipGeom::nslip), relative volume(1),
   // internal energy(ecmech::ne)
   int num_state_vars = ecmech::matModelEvptn_FCC_A::numHist + ecmech::ne + 1;

   std::vector<unsigned int> strides;
   // Deformation rate stride
   strides.push_back(ecmech::nsvp);
   // Spin rate stride
   strides.push_back(ecmech::ndim);
   // Volume ratio stride
   strides.push_back(ecmech::nvr);
   // Internal energy stride
   strides.push_back(ecmech::ne);
   // Stress vector stride
   strides.push_back(ecmech::nsvp);
   // History variable stride
   strides.push_back(num_state_vars);
   // Temperature stride
   strides.push_back(1);
   // SDD stride
   strides.push_back(ecmech::nsdd);

   mat_model = new ecmech::matModelEvptn_FCC_A(strides.data(), strides.size());

   mat_model_base = dynamic_cast<ecmech::matModelBase*>(mat_model);

   ind_dp_eff = ecmech::evptn::iHistA_shrateEff;
   ind_eql_pl_strain = ecmech::evptn::iHistA_shrEff;
   ind_flow_stress = ecmech::evptn::iHistA_flowStr;
   ind_num_evals = ecmech::evptn::iHistA_nFEval;
   ind_dev_elas_strain = ecmech::evptn::iHistLbE;
   ind_quats = ecmech::evptn::iHistLbQ;
   ind_hardness = ecmech::evptn::iHistLbH;

   ind_gdot = mat_model->iHistLbGdot;
   // This will always be 1 for this class
   num_hardness = mat_model->nH;
   // This will always be 12 for this class
   num_slip = mat_model->nslip;
   // The number of vols -> we actually only need to save the previous time step value
   // instead of all 4 values used in the evalModel. The rest can be calculated from
   // this value.
   num_vols = 1;
   ind_vols = ind_gdot + num_slip;
   // The number of internal energy variables -> currently 1
   num_int_eng = 1;
   ind_int_eng = ind_vols + num_vols;
   // Params start off with:
   // initial density, heat capacity at constant volume, and a tolerance param
   // Params then include Elastic constants:
   // c11, c12, c44 for Cubic crystals
   // Params then include the following:
   // shear modulus, m parameter seen in slip kinetics, gdot_0 term found in slip kinetic eqn,
   // hardening coeff. defined for g_crss evolution eqn, initial CRSS value,
   // initial CRSS saturation strength, CRSS saturation strength scaling exponent,
   // CRSS saturation strength rate scaling coeff, tausi -> hdn_init (not used)
   // Params then include the following:
   // the Grüneisen parameter, reference internal energy

   // Opts and strs are just empty vectors of int and strings
   std::vector<double> params;
   std::vector<int> opts;
   std::vector<std::string> strs;
   // 9 terms come from hardening law, 3 terms come from elastic modulus, 4 terms are related to EOS params,
   // 1 terms is related to solving tolerances == 17 total params
   MFEM_ASSERT(matProps->Size() == ecmech::matModelEvptn_FCC_A::nParams,
               "Properties did not contain " << ecmech::matModelEvptn_FCC_A::nParams <<
               " parameters for Voce FCC model.");

   for (int i = 0; i < matProps->Size(); i++) {
      params.push_back(matProps->Elem(i));
   }

   // We really shouldn't see this change over time at least for our applications.
   mat_model_base->initFromParams(opts, params, strs);
   //
   mat_model_base->complete();
   mat_model_base->setExecutionStrategy(accel);

   std::vector<double> histInit;
   {
      std::vector<std::string> names;
      std::vector<bool>        plot;
      std::vector<bool>        state;
      mat_model_base->getHistInfo(names, histInit, plot, state);
   }

   init_state_vars(_q_matVars0, histInit);
}

void VoceFCCModel::init_state_vars(mfem::QuadratureFunction *_q_matVars0, std::vector<double> hist_init)
{
   double histInit_vec[ecmech::matModelEvptn_FCC_A::numHist];
   assert(hist_init.size() == ecmech::matModelEvptn_FCC_A::numHist);

   for (uint i = 0; i < hist_init.size(); i++) {
      histInit_vec[i] = hist_init.at(i);
   }

   double* state_vars = _q_matVars0->ReadWrite();

   int qf_size = (_q_matVars0->Size()) / (_q_matVars0->GetVDim());

   int vdim = _q_matVars0->GetVDim();

   MFEM_FORALL(i, qf_size, {
      const int ind = i * vdim;

      state_vars[ind + ind_dp_eff] = histInit_vec[ind_dp_eff];
      state_vars[ind + ind_eql_pl_strain] = histInit_vec[ind_eql_pl_strain];
      state_vars[ind + ind_flow_stress] = histInit_vec[ind_flow_stress];
      state_vars[ind + ind_num_evals] = histInit_vec[ind_num_evals];
      state_vars[ind + ind_hardness] = histInit_vec[ind_hardness];
      state_vars[ind + ind_vols] = 1.0;

      for (int j = 0; j < ecmech::ne; j++) {
         state_vars[ind + ind_int_eng] = 0.0;
      }

      for (int j = 0; j < 5; j++) {
         state_vars[ind + ind_dev_elas_strain + j] = histInit_vec[ind_dev_elas_strain + j];
      }

      for (int j = 0; j < num_slip; j++) {
         state_vars[ind + ind_gdot + j] = histInit_vec[ind_gdot + j];
      }
   });
}

KinKMBalDDFCCModel::KinKMBalDDFCCModel(mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
                                       mfem::QuadratureFunction *_q_matGrad, mfem::QuadratureFunction *_q_matVars0,
                                       mfem::QuadratureFunction *_q_matVars1,
                                       mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                                       mfem::Vector *_props, int _nProps, int _nStateVars, double _temp_k,
                                       ecmech::ExecutionStrategy _accel, bool _PA) :
   ExaCMechModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
                 _beg_coords, _end_coords, _props, _nProps, _nStateVars, _temp_k,
                 _accel, _PA)
{
   // For FCC material models we have the following state variables
   // and their number of components
   // effective shear rate(1), effective shear(1), flow strength(1), n_evals(1), deviatoric elastic strain(5),
   // quaternions(4), h(Kinetics::nH), gdot(SlipGeom::nslip), relative volume(1),
   // internal energy(ecmech::ne)
   int num_state_vars = ecmech::matModelEvptn_FCC_B::numHist + ecmech::ne + 1;

   std::vector<unsigned int> strides;
   // Deformation rate stride
   strides.push_back(ecmech::nsvp);
   // Spin rate stride
   strides.push_back(ecmech::ndim);
   // Volume ratio stride
   strides.push_back(ecmech::nvr);
   // Internal energy stride
   strides.push_back(ecmech::ne);
   // Stress vector stride
   strides.push_back(ecmech::nsvp);
   // History variable stride
   strides.push_back(num_state_vars);
   // Temperature stride
   strides.push_back(1);
   // SDD stride
   strides.push_back(ecmech::nsdd);

   mat_model = new ecmech::matModelEvptn_FCC_B(strides.data(), strides.size());

   mat_model_base = dynamic_cast<ecmech::matModelBase*>(mat_model);

   ind_dp_eff = ecmech::evptn::iHistA_shrateEff;
   ind_eql_pl_strain = ecmech::evptn::iHistA_shrEff;
   ind_flow_stress = ecmech::evptn::iHistA_flowStr;
   ind_num_evals = ecmech::evptn::iHistA_nFEval;
   ind_dev_elas_strain = ecmech::evptn::iHistLbE;
   ind_quats = ecmech::evptn::iHistLbQ;
   ind_hardness = ecmech::evptn::iHistLbH;

   ind_gdot = mat_model->iHistLbGdot;
   // This will always be 1 for this class
   num_hardness = mat_model->nH;
   // This will always be 12 for this class
   num_slip = mat_model->nslip;
   // The number of vols -> we actually only need to save the previous time step value
   // instead of all 4 values used in the evalModel. The rest can be calculated from
   // this value.
   num_vols = 1;
   ind_vols = ind_gdot + num_slip;
   // The number of internal energy variables -> currently 1
   num_int_eng = ecmech::ne;
   ind_int_eng = ind_vols + num_vols;

   // Params start off with:
   // initial density, heat capacity at constant volume, and a tolerance param
   // Params then include Elastic constants:
   // c11, c12, c44 for Cubic crystals
   // Params then include the following:
   // reference shear modulus, reference temperature, g_0 * b^3 / \kappa where b is the
   // magnitude of the burger's vector and \kappa is Boltzmann's constant, Peierls barrier,
   // MTS curve shape parameter (p), MTS curve shape parameter (q), reference thermally activated
   // slip rate, reference drag limited slip rate, drag reference stress, slip resistance const (g_0),
   // slip resistance const (s), dislocation density production constant (k_1),
   // dislocation density production constant (k_{2_0}), dislocation density exponential constant,
   // reference net slip rate constant, reference relative dislocation density
   // Params then include the following:
   // the Grüneisen parameter, reference internal energy

   // Opts and strs are just empty vectors of int and strings
   std::vector<double> params;
   std::vector<int> opts;
   std::vector<std::string> strs;
   // 16 terms come from hardening law, 3 terms come from elastic modulus, 4 terms are related to EOS params,
   // 1 terms is related to solving tolerances == 24 total params
   MFEM_ASSERT(matProps->Size() == ecmech::matModelEvptn_FCC_B::nParams,
               "Properties need " << ecmech::matModelEvptn_FCC_B::nParams <<
               " parameters for FCC MTS like hardening model");

   for (int i = 0; i < matProps->Size(); i++) {
      params.push_back(matProps->Elem(i));
   }

   // We really shouldn't see this change over time at least for our applications.
   mat_model_base->initFromParams(opts, params, strs);
   //
   mat_model_base->complete();
   mat_model_base->setExecutionStrategy(accel);


   std::vector<double> histInit;
   {
      std::vector<std::string> names;
      std::vector<bool>        plot;
      std::vector<bool>        state;
      mat_model_base->getHistInfo(names, histInit, plot, state);
   }
   init_state_vars(_q_matVars0, histInit);
}

void KinKMBalDDFCCModel::init_state_vars(mfem::QuadratureFunction *_q_matVars0, std::vector<double> hist_init)
{
   double histInit_vec[ecmech::matModelEvptn_FCC_B::numHist];
   assert(hist_init.size() == ecmech::matModelEvptn_FCC_B::numHist);

   for (uint i = 0; i < hist_init.size(); i++) {
      histInit_vec[i] = hist_init.at(i);
   }

   double* state_vars = _q_matVars0->ReadWrite();

   int qf_size = (_q_matVars0->Size()) / (_q_matVars0->GetVDim());

   int vdim = _q_matVars0->GetVDim();

   MFEM_FORALL(i, qf_size, {
      const int ind = i * vdim;

      state_vars[ind + ind_dp_eff] = histInit_vec[ind_dp_eff];
      state_vars[ind + ind_eql_pl_strain] = histInit_vec[ind_eql_pl_strain];
      state_vars[ind + ind_flow_stress] = histInit_vec[ind_flow_stress];
      state_vars[ind + ind_num_evals] = histInit_vec[ind_num_evals];
      state_vars[ind + ind_hardness] = histInit_vec[ind_hardness];
      state_vars[ind + ind_vols] = 1.0;

      for (int j = 0; j < ecmech::ne; j++) {
         state_vars[ind + ind_int_eng] = 0.0;
      }

      for (int j = 0; j < ecmech::ntvec; j++) {
         state_vars[ind + ind_dev_elas_strain + j] = histInit_vec[ind_dev_elas_strain + j];
      }

      for (int j = 0; j < num_slip; j++) {
         state_vars[ind + ind_gdot + j] = histInit_vec[ind_gdot + j];
      }
   });
}