#ifndef MECHANICS_ECMECH
#define MECHANICS_ECMECH

#include "mfem.hpp"
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

      virtual void ModelSetup(const int nqpts, const int nelems, const int /*space_dim*/,
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
      virtual void class_instantiation() {}

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
                   ecmech::Accelerator _accel, bool _PA);

      void init_state_vars(mfem::QuadratureFunction *_q_matVars0, std::vector<double> hist_init);

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
      virtual void class_instantiation(){}

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
                         ecmech::Accelerator _accel, bool _PA);

      void init_state_vars(mfem::QuadratureFunction *_q_matVars0, std::vector<double> hist_init);

      virtual ~KinKMBalDDFCCModel()
      {
         delete mat_model;
      }
};

#endif
