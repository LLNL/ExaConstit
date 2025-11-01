#ifndef MECHANICS_MODEL
#define MECHANICS_MODEL

#include "options/option_parser_v2.hpp"
#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <string>
#include <unordered_map>
#include <utility>

/**
 * @brief Computes the beginning step deformation gradient and stores it in a quadrature function
 *
 * @param qf Pointer to the quadrature function where deformation gradient will be stored
 * @param fes Pointer to the parallel finite element space for the mesh
 * @param x0 Current nodal coordinates vector
 *
 * @details This function computes the incremental deformation gradient at each quadrature point by:
 * 1. Looping over all elements in the finite element space
 * 2. For each element, computing shape function gradients and element Jacobians
 * 3. Computing the incremental deformation gradient as the transpose multiplication of element
 * coordinates with shape function gradients
 * 4. Updating the beginning step deformation gradient by multiplying with the previous gradient
 * 5. Storing results in the provided quadrature function
 */
void computeDefGrad(mfem::QuadratureFunction* qf,
                    mfem::ParFiniteElementSpace* fes,
                    mfem::Vector& x0);

/**
 * @brief Base class for all material constitutive models in ExaConstit
 *
 * @details Provides a unified interface for different material model implementations
 * including ExaCMech crystal plasticity models and UMAT interfaces. This class enables
 * multi-material simulations by using region identifiers to access appropriate data
 * subsets from SimulationState.
 *
 * The class follows a three-stage execution pattern:
 * 1. Setup kernel: Computes kinematic quantities needed by the material model
 * 2. Material kernel: Executes the actual constitutive model calculations
 * 3. Post-processing kernel: Formats and stores results in appropriate data structures
 */
class ExaModel {
public:
    /** @brief Number of state variables required by this material model */
    int num_state_vars;

protected:
    /** @brief Region identifier for this model instance - used to access region-specific data from
     * SimulationState */
    int m_region;
    /** @brief Assembly type specification (Full Assembly, Partial Assembly, or Element Assembly) */
    AssemblyType assembly;
    /** @brief Reference to simulation state for accessing quadrature functions and other simulation
     * data */
    std::shared_ptr<SimulationState> m_sim_state;
    // ---------------------------------------------------------------------------

public:
    /**
     * @brief Construct a base ExaModel with region-specific capabilities
     *
     * @param region Material region identifier that this model instance manages
     * @param n_state_vars Number of state variables required by this material model
     * @param sim_state Reference to the simulation state for accessing region-specific data
     *
     * @details The region parameter enables multi-material simulations by allowing each
     * model instance to access the correct data subset from SimulationState.
     */
    ExaModel(const int region, int n_state_vars, std::shared_ptr<SimulationState> sim_state);

    /**
     * @brief Virtual destructor to ensure proper cleanup of derived class resources
     */
    virtual ~ExaModel() {}

    /**
     * @brief Get material properties for this model's region
     *
     * @return Constant reference to the material properties vector for this model's region
     *
     * @details Provides access to region-specific material properties, replacing direct
     * access to material property vectors. This enables proper encapsulation and
     * multi-material support.
     */
    const std::vector<double>& GetMaterialProperties() const;

    /**
     * @brief Returns material model region id
     *
     * @return material model region id
     */
    int GetRegionID() const {
        return m_region;
    }

    /**
     * @brief Main material model execution method - must be implemented by all derived classes
     *
     * @param nqpts Number of quadrature points per element
     * @param nelems Number of elements in this batch
     * @param space_dim Spatial dimension (2D or 3D)
     * @param nnodes Number of nodes per element
     * @param jacobian Jacobian transformation matrices for elements
     * @param loc_grad Local gradient operators
     * @param vel Velocity field at elemental level (space_dim * nnodes * nelems)
     *
     * @details This function is responsible for running the entire model and consists of 3
     * stages/kernels:
     * 1. A set-up kernel/stage that computes all of the needed values for the material model
     * 2. A kernel that runs the material model (a t = 0 version of this will exist as well)
     * 3. A post-processing kernel/stage that does everything after the kernel
     *
     * By having this unified function, we only need to write one integrator for everything.
     * It also allows us to run these models on the GPU even if the rest of the assembly
     * operation can't be there yet. If UMATs are used then these operations won't occur on the GPU.
     */
    virtual void ModelSetup(const int nqpts,
                            const int nelems,
                            const int space_dim,
                            const int nnodes,
                            const mfem::Vector& jacobian,
                            const mfem::Vector& loc_grad,
                            const mfem::Vector& vel) = 0;

    /**
     * @brief Update model state variables after a converged solution
     *
     * @details Default implementation does nothing; derived classes override if state
     * variable updates are needed after successful solution convergence.
     */
    virtual void UpdateModelVars() = 0;
};

#endif