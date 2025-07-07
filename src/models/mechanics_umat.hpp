#ifndef MECHANICS_UMAT
#define MECHANICS_UMAT

#include "mfem.hpp"
#include "mechanics_model.hpp"
#include "userumat.h"

/// Abaqus Umat class.
/// 
/// KEY ARCHITECTURAL CHANGE: This class no longer takes QuadratureFunction pointers
/// in its constructor. Instead, it receives a region identifier and accesses all
/// QuadratureFunctions through the SimulationState interface. This enables:
/// 1. Better encapsulation - the model doesn't manage QF lifetimes
/// 2. Multi-material support - each model instance knows its region  
/// 3. Dynamic access - models can access different QFs based on runtime conditions
/// 4. Simplified construction - much fewer constructor parameters
class AbaqusUmatModel : public ExaModel
{
   protected:

      // add member variables.
      double elemLength;

      // RETAINED: The initial local shape function gradients.
      // These are working space specific to UMAT models, so they remain as member variables
      mfem::QuadratureFunction loc0_sf_grad;

      // RETAINED: The incremental deformation gradients.
      // These are working space specific to UMAT models, so they remain as member variables
      mfem::QuadratureFunction incr_def_grad;

      // RETAINED: The end step deformation gradients.  
      // These are working space specific to UMAT models, so they remain as member variables
      mfem::QuadratureFunction end_def_grad;

      // REMOVED: mfem::QuadratureFunction *defGrad0;
      // This is now accessed through SimulationState using GetDefGrad0()

      // pointer to umat function
      // we really don't use this in the code
      void (*umatp)(double[6], double[], double[36],
                    double*, double*, double*, double*,
                    double[6], double[6], double*,
                    double[6], double[6], double[2],
                    double*, double*, double*, double*,
                    double*, double*, int*, int*, int*,
                    int *, double[], int*, double[3],
                    double[9], double*, double*,
                    double[9], double[9], int*, int*,
                    int*, int*, int*, int*);

      // Calculates the incremental versions of the strain measures that we're given
      // above
      void CalcLogStrainIncrement(mfem::DenseMatrix &dE, const mfem::DenseMatrix &Jpt);
      void CalcEulerianStrainIncr(mfem::DenseMatrix& dE, const mfem::DenseMatrix &Jpt);
      void CalcLagrangianStrainIncr(mfem::DenseMatrix& dE, const mfem::DenseMatrix &Jpt);

      // calculates the element length
      void CalcElemLength(const double elemVol);

      void init_loc_sf_grads(std::shared_ptr<mfem::ParFiniteElementSpace> fes);
      void init_incr_end_def_grad();

      // For when the ParFinitieElementSpace is stored on the class...
      virtual void calc_incr_end_def_grad(const mfem::ParGridFunction &x0);

   public:
      // NEW CONSTRUCTOR: Much simpler parameter list focused on essential UMAT-specific info
      // 
      // Parameters:
      // - region: Which material region this model manages (key for SimulationState access)
      // - nProps: Number of material properties
      // - nStateVars: Number of state variables
      // - sim_state: Reference to simulation state for data access
      //
      // REMOVED PARAMETERS (now accessed through SimulationState):
      // - All QuadratureFunction pointers (_q_stress0, _q_stress1, etc.) 
      // - mfem::QuadratureFunction *_q_defGrad0 (deformation gradient)
      // - mfem::Vector *_props (material properties)
      AbaqusUmatModel(const int region, int nStateVars, 
                      SimulationState& sim_state);

      virtual ~AbaqusUmatModel() { }

      // NEW: Helper method to get defGrad0 from SimulationState
      // This replaces the direct member variable access and enables dynamic access
      // to the correct region-specific deformation gradient data
      std::shared_ptr<mfem::expt::PartialQuadratureFunction> GetDefGrad0();

      // UNCHANGED: These methods remain the same since they work with internal data or don't access QFs directly
      virtual void UpdateModelVars() override;

      virtual void ModelSetup(const int nqpts, const int nelems, const int space_dim,
                              const int /*nnodes*/, const mfem::Vector &jacobian,
                              const mfem::Vector & /*loc_grad*/, const mfem::Vector &vel) override;
};

#endif