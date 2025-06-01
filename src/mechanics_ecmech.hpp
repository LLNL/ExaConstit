#pragma once

#include "mfem.hpp"
#include "ECMech_const.h"
#include "ECMech_matModelBase.h"
#include "mechanics_model.hpp"

/// Base class for all of our ExaCMechModels.
/// 
/// KEY ARCHITECTURAL CHANGE: This class no longer takes QuadratureFunction pointers
/// in its constructor. Instead, it receives a region identifier and accesses all
/// QuadratureFunctions through the SimulationState interface. This enables:
/// 1. Better encapsulation - the model doesn't manage QF lifetimes
/// 2. Multi-material support - each model instance knows its region
/// 3. Dynamic access - models can access different QFs based on runtime conditions
/// 4. Simplified construction - much fewer constructor parameters
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

      // RETAINED: Temporary variables that we'll be making use of when running our models.
      // These are working space arrays specific to the ExaCMech model execution,
      // not data storage, so they remain as member variables
      mfem::Vector *vel_grad_array;
      mfem::Vector *eng_int_array;
      mfem::Vector *w_vec_array;
      mfem::Vector *vol_ratio_array;
      mfem::Vector *stress_svec_p_array;
      mfem::Vector *d_svec_p_array;
      mfem::Vector *tempk_array;
      mfem::Vector *sdd_array;
      mfem::Vector *eff_def_rate;

      // Mapping from variable names to their locations within the state variable vector
      // This is ExaCMech-specific and helps locate variables within the large state vector
      std::map<std::string, size_t> index_map;

   public:
      // NEW CONSTRUCTOR: Much simpler parameter list focused on essential ExaCMech-specific info
      // 
      // Parameters:
      // - region: Which material region this model manages (key for SimulationState access)
      // - nProps: Number of material properties
      // - nStateVars: Number of state variables  
      // - temp_k: Temperature in Kelvin
      // - accel: Execution strategy (CPU/OpenMP/GPU)
      // - mat_model_name: ExaCMech model name (e.g., "FCC_PowerVoce")
      // - sim_state: Reference to simulation state for data access
      //
      // REMOVED PARAMETERS (now accessed through SimulationState):
      // - All QuadratureFunction pointers (_q_stress0, _q_stress1, etc.)
      // - mfem::Vector *_props (material properties)
      ExaCMechModel(const int region, int nStateVars, 
                    double temp_k, ecmech::ExecutionStrategy accel, 
                    const std::string& mat_model_name, 
                    SimulationState& sim_state);

      // Destructor unchanged - still needs to clean up working arrays and model
      ~ExaCMechModel()
      {
         delete vel_grad_array;
         delete eng_int_array;
         delete w_vec_array;
         delete vol_ratio_array;
         delete stress_svec_p_array;
         delete d_svec_p_array;
         delete tempk_array;
         delete sdd_array;
         delete eff_def_rate;
         delete mat_model_base;
      }

      // UNCHANGED: These methods remain the same since they work with internal data structures
      void setup_data_structures();
      void setup_model(const std::string& mat_model_name);
      void init_state_vars(std::vector<double> hist_init);

      /** This model takes in the velocity, det(jacobian), and local_grad/jacobian.
       *  It then computes velocity gradient symm and skw tensors and passes
       *  that to our material model in order to get out our Cauchy stress and
       * the material tangent matrix (d \sigma / d Vgrad_{sym}). It also
       * updates all of the state variables that live at the quadrature pts.
       * 
       * IMPLEMENTATION NOTE: This method's signature remains unchanged, but internally
       * it will use the new accessor methods to get QuadratureFunctions from SimulationState
       */
      void ModelSetup(const int nqpts, const int nelems, const int /*space_dim*/,
                      const int nnodes, const mfem::Vector &jacobian,
                      const mfem::Vector &loc_grad, const mfem::Vector &vel) override;

      /// If we needed to do anything to our state variables once things are solved
      /// for we do that here.
      /// UNCHANGED: This method doesn't directly access QuadratureFunctions
      virtual void UpdateModelVars() override {}
      
      /// UNCHANGED: This method doesn't access QuadratureFunctions
      void calcDpMat(mfem::QuadratureFunction &/* DpMat */) const override {}
};