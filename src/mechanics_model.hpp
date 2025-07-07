#ifndef MECHANICS_MODEL
#define MECHANICS_MODEL

#include "options/option_parser_v2.hpp"
#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <utility>
#include <unordered_map>
#include <string>

/// free function to compute the beginning step deformation gradient to store
/// on a quadrature function
void computeDefGrad(mfem::QuadratureFunction *qf, mfem::ParFiniteElementSpace *fes,
                    mfem::Vector &x0);

class ExaModel
{
   public:
      int numStateVars;
   protected:
      // NEW: Region identifier for this model instance
      // This tells the model which region's data to access from SimulationState
      int m_region;

      AssemblyType assembly;
      // Temporary fix just to make sure things work - keep for PA assembly
      mfem::Vector matGradPA;

      SimulationState& m_sim_state;
   // ---------------------------------------------------------------------------

   public:
      // Constructor only takes region and basic info
      // The region parameter tells this model instance which material region 
      // it's responsible for, allowing it to access the correct data from SimulationState
      ExaModel(const int region, int nStateVars, SimulationState& sim_state);

      virtual ~ExaModel() { }
      
      // Helper method to get material properties for this region
      // This replaces direct access to the matProps vector
      const std::vector<double>& GetMaterialProperties() const;

      /** @brief This function is responsible for running the entire model and will be the
      *   external function that other classes/people can call.
      *
      *   It will consist of 3 stages/kernels:
      *   1.) A set-up kernel/stage that computes all of the needed values for the material model
      *   2.) A kernel that runs the material model (an t = 0 version of this will exist as well)
      *   3.) A post-processing kernel/stage that does everything after the kernel
      *   e.g. All of the data is put back into the correct format here and re-arranged as needed
      *   By having this function, we only need to ever right one integrator for everything.
      *   It also allows us to run these models on the GPU even if the rest of the assembly operation
      *   can't be there yet. If UMATs are used then these operations won't occur on the GPU.
      *
      *   We'll need to supply the number of quadrature pts, number of elements, the dimension
      *   of the space we're working with, the number of nodes for an element, the jacobian associated
      *   with the transformation from the reference element to the local element, the quadrature integration wts,
      *   and the velocity field at the elemental level (space_dim * nnodes * nelems).
      */
      virtual void ModelSetup(const int nqpts, const int nelems, const int space_dim,
                              const int nnodes, const mfem::Vector &jacobian,
                              const mfem::Vector &loc_grad, const mfem::Vector &vel) = 0;

      /// routine to update the beginning step deformation gradient. This must
      /// be written by a model class extension to update whatever else
      /// may be required for that particular model
      virtual void UpdateModelVars() = 0;
};

#endif