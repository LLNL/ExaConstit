#pragma once

#include "models/mechanics_model.hpp"
#include "sim_state/simulation_state.hpp"

#include <memory>
#include <vector>

/**
 * @brief Multi material model that coordinates multiple region-specific models
 * 
 * This class implements the Composite design pattern to manage multiple material
 * models within a single simulation. From the outside, it looks and behaves exactly
 * like any other ExaModel, but internally it coordinates multiple "child" models
 * that handle different material regions.
 * 
 * Key responsibilities:
 * 1. Maintain mapping between elements and material regions
 * 2. Route global simulation data to appropriate child models
 * 3. Coordinate child model execution during setup phases
 * 4. Aggregate results from child models into global data structures
 * 5. Present a unified ExaModel interface to external code
 * 
 * This design eliminates the need for external classes (like NonlinearMechOperator)
 * to understand or manage multiple models, significantly simplifying the overall
 * architecture while maintaining full flexibility for multi-material simulations.
 */
class MultiExaModel : public ExaModel
{
private:
    // Child models - one for each material region
    std::vector<std::unique_ptr<ExaModel>> m_child_models;
    
    // Number of regions in this simulation
    int m_num_regions;
    
public:
    /**
     * @brief Construct a composite model from simulation options
     * 
     * This constructor analyzes the ExaOptions to determine how many regions
     * are needed, creates appropriate child models for each region, and sets up
     * all the internal data structures for efficient region management.
     * 
     * @param sim_state Reference to simulation state for data access
     * @param options Simulation options containing material definitions
     */
    MultiExaModel(SimulationState& sim_state, const ExaOptions& options);
    
    /**
     * @brief Destructor - child models are automatically cleaned up by unique_ptr
     */
    virtual ~MultiExaModel() = default;
    
    // ExaModel interface implementation
    // These methods coordinate the child models while presenting a unified interface
    
    /**
     * @brief Main model setup method - coordinates all child models
     * 
     * This method receives global simulation data and internally:
     * 1. Extracts region-specific subsets of the data
     * 2. Calls each child model with its appropriate data subset
     * 3. Coordinates result collection back to global data structures
     * 
     * From the caller's perspective, this looks identical to any single model setup.
     */
    virtual void ModelSetup(const int nqpts, const int nelems, const int space_dim,
                            const int nnodes, const mfem::Vector &jacobian,
                            const mfem::Vector &loc_grad, const mfem::Vector &vel) override;
    
    /**
     * @brief Update all child models' state variables
     * 
     * This coordinates the state variable updates across all regions,
     * ensuring that beginning-of-step values are properly synchronized.
     */
    virtual void UpdateModelVars() override;

    // Additional methods for region management and introspection
    
    /**
     * @brief Get the number of material regions
     */
    int GetNumberOfRegions() const { return m_child_models.size(); }
    
    /**
     * @brief Get a specific child model (for advanced use cases)
     * 
     * This allows external code to access specific region models if needed,
     * though in most cases the composite interface should be sufficient.
     */
    ExaModel* GetChildModel(int region_idx) const;
    
private:
    // Internal setup and coordination methods
    
    /**
     * @brief Create child models for each region
     * 
     * This analyzes the material options and creates appropriate ExaModel
     * instances (ExaCMech, UMAT, etc.) for each defined material region.
     */
    void CreateChildModels(const ExaOptions& options);
    
    /**
     * @brief Setup and execute a specific child model
     * 
     * This calls the child model for a specific region, letting SimulationState
     * handle all the data routing and region-specific data management.
     */
    bool SetupChildModel(int region_idx, const int nqpts, const int nelems, 
                        const int space_dim, const int nnodes, 
                        const mfem::Vector &jacobian, const mfem::Vector &loc_grad, 
                        const mfem::Vector &vel) const;
    
    /**
     * @brief Error handling and validation across regions
     * 
     * This method uses MPI collective operations to ensure that if any
     * child model fails on any processor, the entire simulation knows
     * about it and can respond appropriately.
     */
    bool ValidateAllRegionsSucceeded(const std::vector<bool>& region_success) const;
};