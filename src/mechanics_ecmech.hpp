#ifndef MECHANICS_ECMECH
#define MECHANICS_ECMECH

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "ECMech_cases.h"
#include "ECMech_evptnWrap.h"
#include "ECMech_const.h"
#include "mechanics_model.hpp"

/// Base class for all of our ExaCMechModels.
class ExaCMechModel : public ExaModel
{
   protected:

      // Current temperature in Kelvin degrees
      double temp_k;

      // A pointer to our actual material model class that ExaCMech uses.
      // The childern classes to this class will have also have another variable
      // that actually contains the real material model that is then dynamically casted
      // to this base class during the instantiation of the class.
      ecmech::matModelBase* mat_model_base;

      // Our accelartion that we are making use of.
      ecmech::ExecutionStrategy accel;

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
                    ecmech::ExecutionStrategy _accel, bool _PA) :
         ExaModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
                  _beg_coords, _end_coords, _props, _nProps, _nStateVars, _PA),
         temp_k(_temp_k), accel(_accel)
      {
         // First find the total number of points that we're dealing with so nelems * nqpts
         const int vdim = _q_stress0->GetVDim();
         const int size = _q_stress0->Size();
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

      /** This model takes in the velocity, det(jacobian), and local_grad/jacobian.
       *  It then computes velocity gradient symm and skw tensors and passes
       *  that to our material model in order to get out our Cauchy stress and
       * the material tangent matrix (d \sigma / d Vgrad_{sym}). It also
       * updates all of the state variables that live at the quadrature pts.
       */
      virtual void ModelSetup(const int nqpts, const int nelems, const int /*space_dim*/,
                              const int nnodes, const mfem::Vector &jacobian,
                              const mfem::Vector &loc_grad, const mfem::Vector &vel);

      /// If we needed to do anything to our state variables once things are solved
      /// for we do that here.
      virtual void UpdateModelVars(){}
};

/// A generic templated class that takes in a typedef of the crystal model that
/// we want to use from ExaCMech.
template<typename ecmechXtal>
class ECMechXtalModel : public ExaCMechModel
{
   protected:
      ecmechXtal *mat_model;
      // Just various indices that we share during initialization
      // in the future these could probably be eliminated all together
      int ind_dp_eff, ind_eql_pl_strain, ind_pl_work, ind_num_evals, ind_dev_elas_strain;
      int ind_quats, ind_hardness, ind_gdot, ind_vols, ind_int_eng;
      int num_hardness, num_slip, num_vols, num_int_eng;

   // Note to self: we might want to in the future add support for the calculation
   // of D^p_{eff} and \int D^p_{eff} dt for post processing needs

   public:
      ECMechXtalModel(mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
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
         // effective shear rate(1), effective shear(1), flow strength(1), n_evals(1),
         // deviatoric elastic strain(5), quaternions(4), h(Kinetics::nH),
         // gdot(SlipGeom::nslip), relative volume(1), internal energy(ecmech::ne)
         int num_state_vars = ecmechXtal::numHist + ecmech::ne + 1;

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

         mat_model = new ecmechXtal(strides.data(), strides.size());

         mat_model_base = dynamic_cast<ecmech::matModelBase*>(mat_model);

         ind_dp_eff = ecmech::evptn::iHistA_shrateEff;
         ind_eql_pl_strain = ecmech::evptn::iHistA_shrEff;
         ind_pl_work = ecmech::evptn::iHistA_flowStr;
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
         {
            std::string s_shrateEff = "shrateEff";
            std::string s_shrEff = "shrEff";
            std::string s_pl_work = "pl_work";
            std::string s_quats = "quats";
            std::string s_gdot = "gdot";
            std::string s_hard = "hardness";
            std::string s_ieng = "int_eng";
            std::string s_rvol = "rel_vol";

            std::pair<int, int>  i_sre = std::make_pair(ind_dp_eff, 1);
            std::pair<int, int>  i_se = std::make_pair(ind_eql_pl_strain, 1);
            std::pair<int, int>  i_plw = std::make_pair(ind_pl_work, 1);
            std::pair<int, int>  i_q = std::make_pair(ind_quats, 4);
            std::pair<int, int>  i_g = std::make_pair(ind_gdot, num_slip);
            std::pair<int, int>  i_h = std::make_pair(ind_hardness, num_hardness);
            std::pair<int, int>  i_en = std::make_pair(ind_int_eng, ecmech::ne);
            std::pair<int, int>  i_rv = std::make_pair(ind_vols, 1);

            qf_mapping[s_shrateEff] = i_sre;
            qf_mapping[s_shrEff] = i_se;
            qf_mapping[s_pl_work] = i_plw;
            qf_mapping[s_quats] = i_q;
            qf_mapping[s_gdot] = i_g;
            qf_mapping[s_hard] = i_h;
            qf_mapping[s_ieng] = i_en;
            qf_mapping[s_rvol] = i_rv;
         }

         // Opts and strs are just empty vectors of int and strings
         std::vector<double> params;
         std::vector<int> opts;
         std::vector<std::string> strs;

         MFEM_ASSERT(matProps->Size() == ecmechXtal::nParams,
                     "Properties did not contain " << ecmechXtal::nParams <<
                     " parameters for Voce model.");

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

         init_state_vars(_q_matVars0, histInit);
      }

      /// This really shouldn't be used. It's only public due to the internal
      /// MFEM_FORALL requiring it to be public
      void init_state_vars(mfem::QuadratureFunction *_q_matVars0, std::vector<double> hist_init)
      {
         double histInit_vec[ecmechXtal::numHist];
         assert(hist_init.size() == ecmechXtal::numHist);

         for (uint i = 0; i < hist_init.size(); i++) {
            histInit_vec[i] = hist_init.at(i);
         }

         double* state_vars = _q_matVars0->ReadWrite();

         int qf_size = (_q_matVars0->Size()) / (_q_matVars0->GetVDim());

         int vdim = _q_matVars0->GetVDim();

         mfem::MFEM_FORALL(i, qf_size, {
            const int ind = i * vdim;

            state_vars[ind + ind_dp_eff] = histInit_vec[ind_dp_eff];
            state_vars[ind + ind_eql_pl_strain] = histInit_vec[ind_eql_pl_strain];
            state_vars[ind + ind_pl_work] = histInit_vec[ind_pl_work];
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

      virtual ~ECMechXtalModel()
      {
         delete mat_model;
      }
};

/** @brief A linear isotropic Voce hardening model with a power law formulation
 *         for the slip kinetics.
 *
 * This model generally can do a decent job of capturing the material behavior in strain rates
 * that are a bit lower where thermally activated slip is a more appropriate approximation.
 * Generally, you'll find that if fitted to capture the elastic plastic transition
 * it will miss the later plastic behavior of the material. However if it is fitted
 * to the general macroscopic stress-strain and d\sigma / d \epsilon_e vs epsilon
 * curve, it will miss the elastic-plastic regime. Based on far-field high energy
 * x-ray diffraction (ff-HEXD) data, this model is capable of capture 1st order
 * behaviors of the distribution of elastic intragrain heterogeneity. However,
 * it fails to capture transient behaviors of these distributions as seen in
 * http://doi.org/10.7298/X4JM27SD and http://doi.org/10.1088/1361-651x/aa6dc5
 * for fatigue applications.
 *
 * A good reference for the Voce implementation can be found in:
 * section 2.1 https://doi.org/10.1016/S0045-7825(98)00034-6
 * section 2.1 https://doi.org/10.1016/j.ijplas.2007.03.004
 * Basics for how to fit such a model can be found here:
 * https://doi.org/10.1016/S0921-5093(01)01174-1 . Although, it should be noted
 * that this is more for the MTS model it can be adapted to the Voce model by taking into
 * account that the m parameter determines the rate sensitivity. So, the more rate insensitive
 * the material is the closer this will be to 0. The exponent to the Voce
 * hardening law can be determined by what ordered function best fits the
 * $\frac{d\sigma}{d\epsilon_e} \text{vs} \epsilon$ curve.
 * The initial CRSS term best determines when the material starts to plastically deform.
 * The saturation CRSS term determines pretty much how much the material is able
 * to harden. The hardening coeff. for CRSS best determines the rate at which the
 * material hardens so larger values lead to a quicker hardening of the material.
 *
 * Params start off with:
 * initial density, heat capacity at constant volume, and a tolerance param
 * Params then include Elastic constants:
 * c11, c12, c44 for Cubic crystals
 * Params then include the following:
 * shear modulus, m parameter seen in slip kinetics, gdot_0 term found in slip kinetic eqn,
 * hardening coeff. defined for g_crss evolution eqn, initial CRSS value,
 * initial CRSS saturation strength, CRSS saturation strength scaling exponent,
 * CRSS saturation strength rate scaling coeff, tausi -> hdn_init (not used)
 * Params then include the following:
 * the Grüneisen parameter, reference internal energy
 */
typedef ECMechXtalModel<ecmech::matModelEvptn_FCC_A> VoceFCCModel;
typedef ECMechXtalModel<ecmech::matModelEvptn_FCC_AH> VoceNLFCCModel;
typedef ECMechXtalModel<ecmech::evptn::matModel<ecmech::SlipGeom_BCC_A, ecmech::Kin_FCC_A, 
                        ecmech::evptn::ThermoElastNCubic, ecmech::EosModelConst<false>>>
                        VoceBCCModel;
typedef ECMechXtalModel<ecmech::evptn::matModel<ecmech::SlipGeom_BCC_A, ecmech::Kin_FCC_AH, 
                        ecmech::evptn::ThermoElastNCubic, ecmech::EosModelConst<false>>>
                        VoceNLBCCModel;

/** @brief A class with slip and hardening kinetics based on a single Kocks-Mecking dislocation density
 *   balanced thermally activated MTS-like slip kinetics with phonon drag effects.
 *
 * See papers https://doi.org/10.1088/0965-0393/17/3/035003 (Section 2 - 2.3)
 * and https://doi.org/10.1063/1.4792227  (Section 3 up to the intro of the twinning kinetics ~ eq 8)
 * for more info on this particular style of models.
 * This model includes a combination of the above two see the actual implementation of
 * ExaCMech ECMech_kinetics_KMBalD.h file for the actual specifics.
 *
 * This model is much more complicated than the simple Voce hardening model and power law slip kinetics
 * seen above. However, it is capable of capturing the behavior of the material over a wide range of
 * not only strain rates but also temperature ranges. The thermal activated slip kinetics is more or less
 * what the slip kinetic power law used with the Voce hardening model approximates as seen in:
 * https://doi.org/10.1016/0079-6425(75)90007-9 and more specifically the Emperical Law section eqns:
 * 34h - 34s. It should be noted though that this was based on work for FCC materials.
 * The classical MTS model can be seen here towards its application towards copper for historical context:
 * https://doi.org/10.1016/0001-6160(88)90030-2
 *
 * An incredibly detailed overview of the thermally activated slip mechanisms can
 * be found in https://doi.org/10.1016/S0079-6425(02)00003-8 . The conclusions provide a nice
 * overview for how several of the parameters can be fitted for this model. Sections 2.3 - 3.4
 * also go a bit more in-depth into the basis for why the fits are done the way they are
 * conducted.
 * The phonon drag contribution has shown to really start to play a role at strain rates
 * 10^3 and above. A bit of a review on this topic can be found in https://doi.org/10.1016/0001-6160(87)90285-9 .
 * It should be noted that the model implemented here does not follow the same formulation
 * provided in that paper. This model can be thought of as a simplified version.
 *
 * Params start off with:
 * initial density, heat capacity at constant volume, and a tolerance param
 * Params then include Elastic constants:
 * (c11, c12, c44 for Cubic crystals) or (c11, c12, c13, c33, and c44 for Hexagonal Crystals)
 * Params then include the following:
 * reference shear modulus, reference temperature, g_0 * b^3 / \kappa where b is the
 * magnitude of the burger's vector and \kappa is Boltzmann's constant**,
 * Peierls barrier, MTS curve shape parameter (p), MTS curve shape parameter (q),
 * reference thermally activated slip rate, reference drag limited slip rate,
 * drag reference stress, slip resistance const (g_0)**, slip resistance const (s)**,
 * dislocation density production constant (k_1), dislocation density production
 * constant (k_{2_0}), dislocation density exponential constant,
 * reference net slip rate constant, reference relative dislocation density
 * Params then include the following:
 * the Grüneisen parameter, reference internal energy
 */
typedef ECMechXtalModel<ecmech::matModelEvptn_FCC_B> KinKMBalDDFCCModel;
/// See documentation related to KinKMBalDDFCCModel
typedef ECMechXtalModel<ecmech::matModelEvptn_HCP_A> KinKMBalDDHCPModel;
typedef ECMechXtalModel<ecmech::matModelEvptn_BCC_A> KinKMbalDDBCCModel;

#endif
