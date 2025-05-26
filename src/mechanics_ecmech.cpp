#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "ECMech_cases.h"
#include "ECMech_const.h"

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
                           double* ddsdde_array, AssemblyType assembly)
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
   if ((assembly == AssemblyType::EA) and mfem::Device::Allows(Backend::DEVICE_MASK)) { return; }
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


ExaCMechModel::ExaCMechModel(
               mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
               mfem::QuadratureFunction *_q_matGrad, mfem::QuadratureFunction *_q_matVars0,
               mfem::QuadratureFunction *_q_matVars1,
               mfem::Vector *_props, int _nProps, int _nStateVars, double _temp_k,
               ecmech::ExecutionStrategy _accel, std::string mat_model_name,
               SimulationState& sim_state
               ) :
         ExaModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
                  _props, _nProps, _nStateVars, sim_state),
         temp_k(_temp_k), accel(_accel)
{
   setup_data_structures();
   setup_model(mat_model_name);
}

void ExaCMechModel::setup_data_structures() {
   // First find the total number of points that we're dealing with so nelems * nqpts
   const int vdim = stress0->GetVDim();
   const int size = stress0->Size();
   const int npts = size / vdim;
   // Now initialize all of the vectors that we'll be using with our class
   vel_grad_array = new mfem::Vector(npts * ecmech::ndim * ecmech::ndim, mfem::Device::GetMemoryType());
   eng_int_array = new mfem::Vector(npts * ecmech::ne, mfem::Device::GetMemoryType());
   w_vec_array = new mfem::Vector(npts * ecmech::nwvec, mfem::Device::GetMemoryType());
   vol_ratio_array = new mfem::Vector(npts * ecmech::nvr, mfem::Device::GetMemoryType());
   stress_svec_p_array = new mfem::Vector(npts * ecmech::nsvp, mfem::Device::GetMemoryType());
   d_svec_p_array = new mfem::Vector(npts * ecmech::nsvp, mfem::Device::GetMemoryType());
   tempk_array = new mfem::Vector(npts, mfem::Device::GetMemoryType());
   sdd_array = new mfem::Vector(npts * ecmech::nsdd, mfem::Device::GetMemoryType());
   eff_def_rate = new mfem::Vector(npts, mfem::Device::GetMemoryType());
   // If we're using a Device we'll want all of these vectors on it and staying there.
   // Also, note that UseDevice() only returns a boolean saying if it's on the device or not
   // rather than telling the vector whether or not it needs to lie on the device.
   vel_grad_array->UseDevice(true); *vel_grad_array = 0.0;
   eng_int_array->UseDevice(true); *eng_int_array = 0.0;
   w_vec_array->UseDevice(true); *w_vec_array = 0.0;
   vol_ratio_array->UseDevice(true); *vol_ratio_array = 0.0;
   stress_svec_p_array->UseDevice(true); *stress_svec_p_array = 0.0;
   d_svec_p_array->UseDevice(true); *d_svec_p_array = 0.0;
   tempk_array->UseDevice(true); *tempk_array = 0.0;
   sdd_array->UseDevice(true); *sdd_array = 0.0;
   eff_def_rate->UseDevice(true); *eff_def_rate = 0.0;
}

void ExaCMechModel::setup_model(std::string mat_model_name) {
   // First aspect is setting up our various map structures
   index_map =  ecmech::modelParamIndexMap(mat_model_name);
   // additional terms we need to add
   index_map["num_volumes"] = 1;
   index_map["index_volume"] = index_map["index_slip_rates"] + index_map["num_slip_system"];
   index_map["num_internal_energy"] = ecmech::ne;
   index_map["index_internal_energy"] = index_map["index_volume"] + index_map["num_volumes"];

   {
      std::string s_shrateEff = "shrateEff";
      std::string s_shrEff = "shrEff";
      std::string s_pl_work = "pl_work";
      std::string s_quats = "quats";
      std::string s_gdot = "gdot";
      std::string s_hard = "hardness";
      std::string s_ieng = "int_eng";
      std::string s_rvol = "rel_vol";
      std::string s_est  = "elas_strain";

      std::pair<int, int>  i_sre = std::make_pair(index_map["index_effective_shear_rate"], 1);
      std::pair<int, int>  i_se = std::make_pair(index_map["index_effective_shear"], 1);
      std::pair<int, int>  i_plw = std::make_pair(index_map["index_flow_strength"], 1);
      std::pair<int, int>  i_q = std::make_pair(index_map["index_lattice_ori"], 4);
      std::pair<int, int>  i_g = std::make_pair(index_map["index_slip_rates"], index_map["num_slip_system"]);
      std::pair<int, int>  i_h = std::make_pair(index_map["index_hardness"], index_map["num_hardening"]);
      std::pair<int, int>  i_en = std::make_pair(index_map["index_internal_energy"], ecmech::ne);
      std::pair<int, int>  i_rv = std::make_pair(index_map["index_volume"], 1);
      std::pair<int, int>  i_est = std::make_pair(index_map["index_dev_elas_strain"], ecmech::ntvec);

      qf_mapping[s_shrateEff] = i_sre;
      qf_mapping[s_shrEff] = i_se;
      qf_mapping[s_pl_work] = i_plw;
      qf_mapping[s_quats] = i_q;
      qf_mapping[s_gdot] = i_g;
      qf_mapping[s_hard] = i_h;
      qf_mapping[s_ieng] = i_en;
      qf_mapping[s_rvol] = i_rv;
      qf_mapping[s_est] = i_est;
   }

   // Now we can create our model
   mat_model_base = ecmech::makeMatModel(mat_model_name);
   // and update our model strides from the default values
   size_t num_state_vars = index_map["num_hist"] + ecmech::ne + 1;
   std::vector<size_t> strides;
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
   // Update our stride values from the default as our history strides are different
   mat_model_base->updateStrides(strides);

   // Now get out the parameters to instantiate our history variables
   // Opts and strs are just empty vectors of int and strings
   std::vector<double> params;
   std::vector<int> opts;
   std::vector<std::string> strs;

   for (int i = 0; i < matProps->Size(); i++) {
      params.push_back(matProps->Elem(i));
   }

   // We really shouldn't see this change over time at least for our applications.
   mat_model_base->initFromParams(opts, params, strs);
   mat_model_base->complete();
   mat_model_base->setExecutionStrategy(accel);

   std::vector<double> histInit;
   {
      std::vector<std::string> names;
      std::vector<bool>        plot;
      std::vector<bool>        state;
      mat_model_base->getHistInfo(names, histInit, plot, state);
   }

   init_state_vars(histInit);
}

void ExaCMechModel::init_state_vars(std::vector<double> hist_init)
{
   mfem::Vector histInit(index_map["num_hist"], mfem::Device::GetMemoryType());
   histInit.UseDevice(true); histInit.HostReadWrite();
   assert(hist_init.size() == index_map["num_hist"]);

   for (uint i = 0; i < hist_init.size(); i++) {
      histInit(i) = hist_init.at(i);
   }

   const double* histInit_vec = histInit.Read(); 
   double* state_vars = matVars0->ReadWrite();

   const size_t qf_size = (matVars0->Size()) / (matVars0->GetVDim());

   const size_t vdim = matVars0->GetVDim();

   const size_t ind_dp_eff = index_map["index_effective_shear_rate"];
   const size_t ind_eql_pl_strain = index_map["index_effective_shear"];
   const size_t ind_pl_work = index_map["index_flow_strength"];
   const size_t ind_num_evals = index_map["index_num_func_evals"];
   const size_t ind_hardness = index_map["index_hardness"];
   const size_t ind_vols = index_map["index_volume"];
   const size_t ind_int_eng = index_map["index_internal_energy"];
   const size_t ind_dev_elas_strain = index_map["index_dev_elas_strain"];
   const size_t ind_gdot = index_map["index_slip_rates"];
   const size_t num_slip = index_map["num_slip_system"];
   const size_t num_hardness = index_map["num_hardening"];

   mfem::MFEM_FORALL(i, qf_size, {
      const size_t ind = i * vdim;

      state_vars[ind + ind_dp_eff] = histInit_vec[ind_dp_eff];
      state_vars[ind + ind_eql_pl_strain] = histInit_vec[ind_eql_pl_strain];
      state_vars[ind + ind_pl_work] = histInit_vec[ind_pl_work];
      state_vars[ind + ind_num_evals] = histInit_vec[ind_num_evals];
      state_vars[ind + ind_vols] = 1.0;

      for (size_t j = 0; j < num_hardness; j++) {
         state_vars[ind + ind_hardness + j] = histInit_vec[ind_hardness + j];
      }

      for (size_t j = 0; j < ecmech::ne; j++) {
         state_vars[ind + ind_int_eng + j] = 0.0;
      }

      for (size_t j = 0; j < ecmech::ntvec; j++) {
         state_vars[ind + ind_dev_elas_strain + j] = histInit_vec[ind_dev_elas_strain + j];
      }

      for (size_t j = 0; j < num_slip; j++) {
         state_vars[ind + ind_gdot + j] = histInit_vec[ind_gdot + j];
      }
   });
}

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
