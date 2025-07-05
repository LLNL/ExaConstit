#include "mechanics_multi_model.hpp"
#include "mechanics_ecmech.hpp"
#include "mechanics_umat.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "mfem_expt/partial_qfunc.hpp"
#include "mechanics_log.hpp"

#include <stdexcept>
#include <algorithm>

MultiExaModel::MultiExaModel(SimulationState& sim_state, const ExaOptions& options)
    : ExaModel(-1, 0, sim_state)  // Region -1, nStateVars computed later
{
    CALI_CXX_MARK_SCOPE("composite_model_construction");
    
    // The construction is now beautifully simple because SimulationState
    // already handles all the complex region management for us
    
    m_num_regions = sim_state.GetNumberOfRegions();
    
    // Create specialized models for each region
    CreateChildModels(options);
    
    // Update our base class state variable count to reflect the maximum needed
    // across all child models. This ensures compatibility with existing interfaces.
    int max_state_vars = 0;
    for (const auto& child : m_child_models) {
        max_state_vars = std::max(max_state_vars, child->numStateVars);
    }
    numStateVars = max_state_vars;
}

void MultiExaModel::CreateChildModels(const ExaOptions& options)
{
    // Create specialized material models for each region
    // SimulationState already knows about regions, so we just create models
    
    m_child_models.reserve(options.materials.size());
    
    for (size_t region_idx = 0; region_idx < options.materials.size(); ++region_idx) {
        const auto& material = options.materials[region_idx];
        
        // Create the appropriate model type based on material specification
        std::unique_ptr<ExaModel> child_model;
        
        if (material.mech_type == MechType::UMAT) {
            // Create UMAT model for this region
            child_model = std::make_unique<AbaqusUmatModel>(
                region_idx,                    // This model handles this specific region
                material.state_vars.num_vars,  // State variables for this material
                m_sim_state                    // Shared simulation state
            );
        }
        else if (material.mech_type == MechType::EXACMECH) {
            // Create ExaCMech model for this region
            
            // Determine execution strategy based on global solver settings
            ecmech::ExecutionStrategy accel = ecmech::ExecutionStrategy::CPU;
            if (options.solvers.rtmodel == RTModel::OPENMP) {
                accel = ecmech::ExecutionStrategy::OPENMP;
            }
            else if (options.solvers.rtmodel == RTModel::GPU) {
                accel = ecmech::ExecutionStrategy::GPU;
            }
            
            // Extract the material model name from the options
            std::string model_name = material.model.exacmech ? 
                                   material.model.exacmech->shortcut : "";

            child_model = std::make_unique<ExaCMechModel>(
                region_idx,                    // Region this model handles
                material.state_vars.num_vars,  // State variables
                material.temperature,          // Operating temperature
                accel,                        // Execution strategy (CPU/GPU)
                model_name,                   // ExaCMech model type
                m_sim_state                   // Shared simulation state
            );
        }
        else {
            throw std::runtime_error("Unknown material type for region " + std::to_string(region_idx));
        }
        
        if (!child_model) {
            throw std::runtime_error("Failed to create material model for region " + std::to_string(region_idx));
        }
        m_child_models.push_back(std::move(child_model));
    }
}

void MultiExaModel::ModelSetup(const int nqpts, const int nelems, const int space_dim,
                                  const int nnodes, const mfem::Vector &jacobian,
                                  const mfem::Vector &loc_grad, const mfem::Vector &vel)
{
    CALI_CXX_MARK_SCOPE("composite_model_setup");
    
    // This is now incredibly simple because SimulationState handles all the complexity!
    // Each child model automatically gets the right data for its region through SimulationState
    
    // Process each region - each child model operates on its own region's data
    std::vector<bool> region_success(m_child_models.size());
    
    for (size_t region_idx = 0; region_idx < m_child_models.size(); ++region_idx) {
        region_success[region_idx] = SetupChildModel(
            region_idx, nqpts, nelems, space_dim, nnodes, jacobian, loc_grad, vel);
    }
    
    // Verify that all regions completed successfully across all MPI processes
    if (!ValidateAllRegionsSucceeded(region_success)) {
        throw std::runtime_error("One or more material regions failed during setup");
    }
    
    // No need for explicit result aggregation - SimulationState handles this automatically
    // through the PartialQuadratureFunction system when child models write their results
}

bool MultiExaModel::SetupChildModel(int region_idx, const int nqpts, const int nelems, 
                                       const int space_dim, const int nnodes,
                                       const mfem::Vector &jacobian, const mfem::Vector &loc_grad, 
                                       const mfem::Vector &vel) const
{
    CALI_CXX_MARK_SCOPE("composite_setup_child");
    
    try {
        // The beauty of this design: we just call the child model with the region index
        // SimulationState automatically routes the right data to the right model!
        auto& child_model = m_child_models[region_idx];
        
        // The child model uses its region_idx to get region-specific data from SimulationState
        // This is much cleaner than manually extracting and routing data
        child_model->ModelSetup(nqpts, nelems, space_dim, nnodes, jacobian, loc_grad, vel);
        
        return true;
    }
    catch (const std::exception& e) {
        MFEM_WARNING("Region " + std::to_string(region_idx) + " failed: " + e.what());
        return false;
    }
    catch (...) {
        MFEM_WARNING("Region " + std::to_string(region_idx) + " failed with unknown error");
        return false;
    }
}

bool MultiExaModel::ValidateAllRegionsSucceeded(const std::vector<bool>& region_success) const
{
    // Check if all regions succeeded on this processor
    bool local_success = std::all_of(region_success.begin(), region_success.end(),
                                    [](bool success) { return success; });
    
    // Use MPI collective operation to ensure global consistency
    bool global_success = false;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    
    return global_success;
}

void MultiExaModel::UpdateModelVars()
{
    // Coordinate state variable updates across all child models
    for (auto& child : m_child_models) {
        child->UpdateModelVars();
    }
}

void MultiExaModel::UpdateStateVars()
{
    // Coordinate state variable updates across all child models
    for (auto& child : m_child_models) {
        child->UpdateStateVars();
    }
}

// Utility methods for external access
ExaModel* MultiExaModel::GetChildModel(int region_idx) const
{
    if (region_idx < 0 || region_idx >= static_cast<int>(m_child_models.size())) {
        return nullptr;
    }
    return m_child_models[region_idx].get();
}