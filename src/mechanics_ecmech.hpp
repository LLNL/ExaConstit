#ifndef MECHANICS_ECMECH
#define MECHANICS_ECMECH

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "ECMech_cases.h"
#include "ECMech_evptnWrap.h"
#include "ECMech_const.h"
#include "mechanics_integrators.hpp"

// using namespace mfem;
// using namespace ecmech;

// Base class for all of our ExaCMechModels.
class ExaCMechModel : public ExaModel
{
   protected:

      // Current temperature in Kelvin degrees
      double temp_k;
      // The indices of our history variables which will be quite useful for
      // post processing of the results.
      // These will be set during the initialization stage of things
      // Initial values for these are passed in through the state variable file.
      // This file will include everything associated with the history variables
      // along with the relative volume value and the initial internal energy value.
      // For most purposes the relative volume should be set to 1 and the initial internal
      // energy should be set to 0.
      int ind_dp_eff, ind_eql_pl_strain, ind_num_evals, ind_dev_elas_strain;
      int ind_quats, ind_hardness, ind_gdot, ind_vols, ind_int_eng;
      // number of hardness variables and number of slip systems
      // these are available in the mat_model_base class as well but I thought
      // it might not hurt to have these explicitly declared.
      int num_hardness, num_slip, num_vols, num_int_eng;
      // Our total number of state variables for the FCC Voce and KinKMBalDD model should be equal to
      // 3+5+1+12+2 = 27 with 4 supplied from quaternion sets so 23 should be in the state variable file.
      virtual void class_instantiation() = 0;

      // A pointer to our actual material model class that ExaCMech uses.
      // The childern classes to this class will have also have another variable
      // that actually contains the real material model that is then dynamically casted
      // to this base class during the instantiation of the class.
      ecmech::matModelBase* mat_model_base;
      // We might also want a mfem::QuadratureFunctionCoefficientmfem::Vector class for our accumulated gammadots called gamma
      // We would update this using a pretty simple integration scheme so gamma += delta_t * gammadot
      // Quadraturemfem::VectorFunctionCoefficient gamma;

      // Our accelartion that we are making use of.
      ecmech::Accelerator accel;

      // Temporary variables that we'll be making use of when running our
      // models.
      mfem::Vector *vel_grad_array;
      mfem::Vector *eng_int_array;
      mfem::Vector *w_vec_array;
      mfem::Vector *vol_ratio_array;
      mfem::Vector *stress_svec_p_array;
      mfem::Vector *d_svec_p_array;
      mfem::Vector *tempk_array;
      mfem::Vector *sdd_array;

   public:
      ExaCMechModel(mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
                    mfem::QuadratureFunction *_q_matGrad, mfem::QuadratureFunction *_q_matVars0,
                    mfem::QuadratureFunction *_q_matVars1,
                    mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                    mfem::Vector *_props, int _nProps, int _nStateVars, double _temp_k,
                    ecmech::Accelerator _accel, bool _PA) :
         ExaModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
                  _beg_coords, _end_coords, _props, _nProps, _nStateVars, _PA),
         temp_k(_temp_k), accel(_accel)
      {
         // First find the total number of points that we're dealing with so nelems * nqpts
         const int vdim = _q_stress0->GetVDim();
         const int size = _q_stress0->Size();
         const int npts = size / vdim;
         // Now initialize all of the vectors that we'll be using with our class
         vel_grad_array = new mfem::Vector(npts * ecmech::ndim * ecmech::ndim);
         eng_int_array = new mfem::Vector(npts * ecmech::ne);
         w_vec_array = new mfem::Vector(npts * ecmech::nwvec);
         vol_ratio_array = new mfem::Vector(npts * ecmech::nvr);
         stress_svec_p_array = new mfem::Vector(npts * ecmech::nsvp);
         d_svec_p_array = new mfem::Vector(npts * ecmech::nsvp);
         tempk_array = new mfem::Vector(npts);
         sdd_array = new mfem::Vector(npts * ecmech::nsdd);
      }

      virtual ~ExaCMechModel()
      {
         delete vel_grad_array;
         delete eng_int_array;
         delete w_vec_array;
         delete vol_ratio_array;
         delete stress_svec_p_array;
         delete d_svec_p_array;
         delete tempk_array;
         delete sdd_array;
      }

      virtual void ModelSetup(const int nqpts, const int nelems, const int space_dim,
                              const int nnodes, const mfem::Vector &jacobian,
                              const mfem::Vector &loc_grad, const mfem::Vector &vel);

      virtual void UpdateModelVars();
};

// A linear isotropic Voce hardening model with a power law formulation for the slip kinetics
// This model generally can do a decent job of capturing the material behavior in strain rates
// that are a bit lower where thermally activated slip is a more appropriate approximation.
// Generally, you'll find that if fitted to capture the elastic plastic transition it will miss the later
// plastic behavior of the material. However, if it is fitted to the general macroscopic stress-strain
// curve and the d\sigma / d \epsilon_e vs epsilon it will miss the elastic-plastic regime.
// Based on far-field high energy x-ray diffraction (ff-HEXD) data, this model is capable of capture 1st
// order behaviors of the distribution of elastic intragrain heterogeneity. However, it fails to capture
// transient behaviors of these distributions as seen in  http://doi.org/10.7298/X4JM27SD and
// http://doi.org/10.1088/1361-651x/aa6dc5 for fatigue applications.

// A good reference for the Voce implementation can be found in:
// section 2.1 https://doi.org/10.1016/S0045-7825(98)00034-6
// section 2.1 https://doi.org/10.1016/j.ijplas.2007.03.004
// Basics for how to fit such a model can be found here:
// https://doi.org/10.1016/S0921-5093(01)01174-1 . Although, it should be noted
// that this is more for the MTS model it can be adapted to the Voce model by taking into
// account that the m parameter determines the rate sensitivity. So, the more rate insensitive
// the material is the closer this will be to 0. It is also largely responsible for how sharp
// the knee of the macroscopic stress-strain curve is. You'll often see OFHC copper around 0.01 - 0.02.
// The exponent to the Voce hardening law can be determined by what ordered function best
// fits the d\sigma / d \epsilon_e vs epsilon curve. The initial CRSS term best determines
// when the material starts to plastically deform. The saturation CRSS term determines pretty
// much how much the material is able to harden. The hardening coeff. for CRSS best determines
// the rate at which the material hardens so larger values lead to a quicker hardening of the material.
// The saturation CRSS isn't seen as constant in several papers involving this model.
// Since, they are often only examining things for a specific strain rate and temperature.
// I don't have a lot of personal experience with fitting this, so I can't provide
// good guidance on how to properly fit this parameter.
//
class VoceFCCModel : public ExaCMechModel
{
   protected:

      // We can define our class instantiation using the following
      virtual void class_instantiation()
      {
         // We have 23 state variables plus the 4 from quaternions for
         // a total of 27 for FCC materials using either the
         // voce or mts model.
         // They are in order:
         // dp_eff(1), eq_pl_strain(2), n_evals(3), dev. elastic strain(4-8),
         // quats(9-12), h(13), gdot(14-25), rel_vol(26), int_eng(27)
         int num_state_vars = 27;

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
         MFEM_ASSERT(matProps->Size() == 17, "Properties did not contain 17 parameters for Voce FCC model.");

         for (int i = 0; i < matProps->Size(); i++) {
            params.push_back(matProps->Elem(i));
         }

         // We really shouldn't see this change over time at least for our applications.
         mat_model_base->initFromParams(opts, params, strs);
         //
         mat_model_base->complete();

         std::vector<double> histInit_vec;
         {
            std::vector<std::string> names;
            std::vector<bool>        plot;
            std::vector<bool>        state;
            mat_model_base->getHistInfo(names, histInit_vec, plot, state);
         }

         mfem::QuadratureFunction* _state_vars = matVars0.GetQuadFunction();
         double* state_vars = _state_vars->GetData();

         int qf_size = (_state_vars->Size()) / (_state_vars->GetVDim());

         int vdim = _state_vars->GetVDim();
         int ind = 0;

         for (int i = 0; i < qf_size; i++) {
            ind = i * vdim;

            state_vars[ind + ind_dp_eff] = histInit_vec[ind_dp_eff];
            state_vars[ind + ind_eql_pl_strain] = histInit_vec[ind_eql_pl_strain];
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
         }
      }

      // This is a type alias for:
      // evptn::matModel< SlipGeomFCC, KineticsVocePL, evptn::ThermoElastNCubic, EosModelConst<false> >
      // We also need to store the full class instantiation of the class on our own class.
      ecmech::matModelEvptn_FCC_A *mat_model;

   // We might also want a mfem::QuadratureFunctionCoefficientmfem::Vector class for our accumulated gammadots called gamma
   // We would update this using a pretty simple integration scheme so gamma += delta_t * gammadot
   // Quadraturemfem::VectorFunctionCoefficient gamma;

   public:
      VoceFCCModel(mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
                   mfem::QuadratureFunction *_q_matGrad, mfem::QuadratureFunction *_q_matVars0,
                   mfem::QuadratureFunction *_q_matVars1,
                   mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                   mfem::Vector *_props, int _nProps, int _nStateVars, double _temp_k,
                   ecmech::Accelerator _accel, bool _PA) :
         ExaCMechModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
                       _beg_coords, _end_coords, _props, _nProps, _nStateVars, _temp_k,
                       _accel, _PA)
      {
         class_instantiation();
      }

      virtual ~VoceFCCModel()
      {
         delete mat_model;
      }
};

// A class with slip and hardening kinetics based on a single Kocks-Mecking dislocation density
// balanced thermally activated MTS-like slip kinetics with phonon drag effects.
// See papers https://doi.org/10.1088/0965-0393/17/3/035003 (Section 2 - 2.3)
// and https://doi.org/10.1063/1.4792227  (Section 3 up to the intro of the twinning kinetics ~ eq 8)
// for more info on this particular style of models.
// This model includes a combination of the above two see the actual implementation of
// ExaCMech ECMech_kinetics_KMBalD.h file for the actual specifics.
//
// This model is much more complicated than the simple Voce hardening model and power law slip kinetics
// seen above. However, it is capable of capturing the behavior of the material over a wide range of
// not only strain rates but also temperature ranges. The thermal activated slip kinetics is more or less
// what the slip kinetic power law used with the Voce hardening model approximates as seen in:
// https://doi.org/10.1016/0079-6425(75)90007-9 and more specifically the Emperical Law section eqns:
// 34h - 34s. It should be noted though that this was based on work for FCC materials.
// The classical MTS model can be seen here towards its application towards copper for historical context:
// https://doi.org/10.1016/0001-6160(88)90030-2
// An incredibly detailed overview of the thermally activated slip mechanisms can
// be found in https://doi.org/10.1016/S0079-6425(02)00003-8 . The conclusions provide a nice
// overview for how several of the parameters can be fitted for this model. Sections 2.3 - 3.4
// also go a bit more in-depth into the basis for why the fits are done the way they are
// conducted.
// The phonon drag contribution has shown to really start to play a role at strain rates
// 10^3 and above. A bit of a review on this topic can be found in https://doi.org/10.1016/0001-6160(87)90285-9 .
// It should be noted that the model implemented here does not follow the same formulation
// provided in that paper. This model can be thought of as a simplified version.
class KinKMBalDDFCCModel : public ExaCMechModel
{
   protected:

      // We can define our class instantiation using the following
      virtual void class_instantiation()
      {
         // We have 23 state variables plus the 4 from quaternions for
         // a total of 27 for FCC materials using either the
         // voce or mts model.
         // They are in order:
         // dp_eff(1), eq_pl_strain(2), n_evals(3), dev. elastic strain(4-8),
         // quats(9-12), h(13), gdot(14-25), rel_vol(26), int_eng(27)
         int num_state_vars = 27;

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
         MFEM_ASSERT(matProps->Size() == 24, "Properties need 24 parameters for FCC MTS like hardening model");

         for (int i = 0; i < matProps->Size(); i++) {
            params.push_back(matProps->Elem(i));
         }

         // We really shouldn't see this change over time at least for our applications.
         mat_model_base->initFromParams(opts, params, strs);
         //
         mat_model_base->complete();

         std::vector<double> histInit_vec;
         {
            std::vector<std::string> names;
            std::vector<bool>        plot;
            std::vector<bool>        state;
            mat_model_base->getHistInfo(names, histInit_vec, plot, state);
         }

         mfem::QuadratureFunction* _state_vars = matVars0.GetQuadFunction();
         double* state_vars = _state_vars->GetData();

         int qf_size = (_state_vars->Size()) / (_state_vars->GetVDim());

         int vdim = _state_vars->GetVDim();
         int ind = 0;

         for (int i = 0; i < qf_size; i++) {
            ind = i * vdim;

            state_vars[ind + ind_dp_eff] = histInit_vec[ind_dp_eff];
            state_vars[ind + ind_eql_pl_strain] = histInit_vec[ind_eql_pl_strain];
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
         }
      }

      // This is a type alias for:
      // evptn::matModel< SlipGeomFCC, Kin_KMBalD_FFF, evptn::ThermoElastNCubic, EosModelConst<false> >
      // where Kin_KMBalD_FFF is further a type alias for:
      // KineticsKMBalD< false, false, false > Kin_KMBalD_FFF;
      // We also need to store the full class instantiation of the class on our own class.
      ecmech::matModelEvptn_FCC_B *mat_model;

   // We might also want a mfem::QuadratureFunctionCoefficientmfem::Vector class for our accumulated gammadots called gamma
   // We would update this using a pretty simple integration scheme so gamma += delta_t * gammadot
   // Quadraturemfem::VectorFunctionCoefficient gamma;

   public:
      KinKMBalDDFCCModel(mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
                         mfem::QuadratureFunction *_q_matGrad, mfem::QuadratureFunction *_q_matVars0,
                         mfem::QuadratureFunction *_q_matVars1,
                         mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                         mfem::Vector *_props, int _nProps, int _nStateVars, double _temp_k,
                         ecmech::Accelerator _accel, bool _PA) :
         ExaCMechModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
                       _beg_coords, _end_coords, _props, _nProps, _nStateVars, _temp_k,
                       _accel, _PA)
      {
         class_instantiation();
      }

      virtual ~KinKMBalDDFCCModel()
      {
         delete mat_model;
      }
};

#endif
