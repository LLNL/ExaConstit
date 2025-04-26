#pragma once

#include "mfem.hpp"
#include "mechanics_kernels.hpp"
#include "ECMech_const.h"
#include "option_types.hpp"
#include "sim_state/simulation_state.hpp"
#include "projection_traits_v2.hpp"

/**
 * @brief PostProcessingDriver handles all post-processing operations for ExaConstit simulations
 * 
 * This class manages:
 * 1. Projection of quadrature data to grid functions for visualization
 * 2. Calculation of volume-averaged quantities
 * 3. Registration of data with visualization collections
 * 4. Output of post-processed data at specified intervals
 */
class PostProcessingDriver {
public:
    /**
     * @brief Construct a new PostProcessingDriver
     * 
     * @param sim_state Reference to global simulation state
     * @param options Simulation options
     */
    PostProcessingDriver(SimulationState& sim_state, ExaOptions& options);
    
    /**
     * @brief Destructor
     */
    ~PostProcessingDriver() = default;
    
    /**
     * @brief Update post-processing data for current step
     * 
     * @param step Current time step
     * @param time Current simulation time
     */
    void Update(const int step, const double time);
    
    /**
     * @brief Calculate and output volume-averaged quantities
     * 
     * @param time Current simulation time
     */
    void PrintVolValues(const double time);
    
    /**
     * @brief Update data collections with current projection data
     * 
     * @param step Current time step
     * @param time Current simulation time
     */
    void UpdateDataCollections(const int step, const double time);
    
    /**
     * @brief Enable or disable a projection for a specific phase
     * 
     * @param field_name Name of the field
     * @param phase Phase index
     * @param enable Whether to enable the projection
     */
    void EnableProjection(const std::string& field_name, int phase, bool enable = true);
    
    /**
     * @brief Enable or disable a projection for all phases
     * 
     * @param field_name Name of the field
     * @param enable Whether to enable the projection
     */
    void EnableProjection(const std::string& field_name, bool enable = true);
    
    /**
     * @brief Enable all projections compatible with the current model types
     */
    void EnableAllProjections();
    
    /**
     * @brief Get list of available projections
     * 
     * @return Vector of pairs with field name and display name
     */
    std::vector<std::pair<std::string, std::string>> GetAvailableProjections() const;
    
private:
    // Registration structures for projections and volume calculations
    struct ProjectionRegistration {
        std::string field_name;                      // Field identifier
        std::string display_name;                    // User-friendly name
        ProjectionTraits::ModelCompatibility model_compatibility; // Compatible models
        std::vector<bool> user_requested;            // Per-phase enabled flag
        std::function<void(int)> projection_function; // Function to call for projection
    };
    
    /**
     * @brief Register a simple projection for a component-based field
     * 
     * @tparam ProjectionType Type of projection trait
     * @param field_name Name of the field
     * @param display_name Display name for UI
     * @param default_enabled Whether the projection is enabled by default
     */
    template<typename ProjectionType>
    void RegisterSimpleProjection(
        const std::string& field_name, 
        const std::string& display_name,
        bool default_enabled = true
    );
    
    /**
     * @brief Register a special projection that calculates derived values
     * 
     * @tparam ProjectionType Type of projection trait
     * @param source_field Source field name (input)
     * @param target_field Target field name (output)
     * @param display_name Display name for UI
     * @param default_enabled Whether the projection is enabled by default
     */
    template<typename ProjectionType>
    void RegisterSpecialProjection(
        const std::string& source_field, 
        const std::string& target_field,
        const std::string& display_name,
        bool default_enabled = true
    );
    
    /**
     * @brief Register a direct geometry projection
     * 
     * @tparam ProjectionType Type of projection trait
     * @param field_name Name of the field
     * @param display_name Display name for UI
     * @param default_enabled Whether the projection is enabled by default
     */
    template<typename ProjectionType>
    void RegisterGeometryProjection(
        const std::string& field_name, 
        const std::string& display_name,
        bool default_enabled = true
    );
    
    /**
     * @brief Register elastic strain projection
     * 
     * @param strain_field Strain field name
     * @param vol_field Volume field name
     * @param display_name Display name for UI
     * @param default_enabled Whether the projection is enabled by default
     */
    void RegisterElasticStrainProjection(
        const std::string& strain_field,
        const std::string& vol_field,
        const std::string& display_name,
        bool default_enabled = true
    );
    
    /**
     * @brief Register a volume average calculation function
     * 
     * @param name Name of the calculation
     * @param display_name Display name for UI
     * @param avg_function Function to calculate the average
     * @param enabled Whether enabled by default
     */
    void RegisterVolumeAverageFunction(
        const std::string& name,
        const std::string& display_name,
        std::function<void(const int, const double)> avg_function,
        bool enabled = true
    );
    
    /**
     * @brief Execute a simple projection
     * 
     * @tparam ProjectionType Type of projection trait
     * @param field_name Field name
     * @param phase Phase index
     */
    template<typename ProjectionType>
    void ExecuteSimpleProjection(const std::string& field_name, int phase);
    
    /**
     * @brief Execute a special projection
     * 
     * @tparam ProjectionType Type of projection trait
     * @param source_field Source field name
     * @param target_field Target field name
     * @param phase Phase index
     */
    template<typename ProjectionType>
    void ExecuteSpecialProjection(
        const std::string& source_field,
        const std::string& target_field,
        int phase
    );
    
    /**
     * @brief Execute a geometry projection
     * 
     * @tparam ProjectionType Type of projection trait
     * @param field_name Field name
     * @param phase Phase index
     */
    template<typename ProjectionType>
    void ExecuteGeometryProjection(const std::string& field_name, int phase);
    
    /**
     * @brief Execute an elastic strain projection
     * 
     * @param strain_field Strain field name
     * @param vol_field Volume field name
     * @param phase Phase index
     */
    void ExecuteElasticStrainProjection(
        const std::string& strain_field,
        const std::string& vol_field,
        int phase
    );
    
    /**
     * @brief Initialize data collections for visualization
     * 
     * @param options Simulation options
     */
    void InitializeDataCollections(ExaOptions& options);
    
    // Volume average calculation methods
    void VolumeAvgStress(const int phase, const double time);
    void VolumeAvgEulerStrain(const int phase, const double time);
    void VolumeAvgDefGrad(const int phase, const double time);
    void VolumePlWork(const int phase, const double time);
    void VolumeAvgElasticStrain(const int phase, const double time);
    
    // Projection methods (implementations use trait templates)
    void ProjectCentroid(const int phase);
    void ProjectVolume(const int phase);
    void ProjectModelStress(const int phase);
    void ProjectVonMisesStress(const int phase);
    void ProjectHydroStress(const int phase);
    void ProjectDpEff(const int phase);
    void ProjectEffPlasticStrain(const int phase);
    void ProjectShearRate(const int phase);
    void ProjectOrientation(const int phase);
    void ProjectH(const int phase);
    void ProjectElasticStrains(const int phase);
    
    // Calculate element average values from quadrature function
    void CalcElementAvg(mfem::Vector* elemVal, const mfem::QuadratureFunction* qf);
    
    // Helper to get quadrature function size
    size_t GetQuadratureFunctionSize() const;
    
private:
    // Reference to simulation state
    const SimulationState& m_sim_state;
    
    // MPI rank
    const int m_mpi_rank;
    
    // Model types for each phase
    std::vector<MechType> m_phase_mech_types;
    
    // Buffer for element-averaged values
    std::unique_ptr<mfem::Vector> m_evec;
    
    // Base path for output files
    std::string m_avg_filepath_base;
    
    // Maps for grid functions and data collections
    std::map<std::string, std::unique_ptr<mfem::ParGridFunction>> m_map_gfs;
    std::map<std::string, std::unique_ptr<mfem::DataCollection>> m_map_dcs;
    
    // Registered projections and volume calculations
    std::vector<ProjectionRegistration> m_registered_projections;
    std::map<std::string, std::function<void(const int, const double)>> m_map_avg_fcns;
    std::map<std::string, std::string> m_map_avg_names;
    std::map<std::string, bool> m_map_avg_enabled;
};

// Template implementations

template<typename ProjectionType>
void PostProcessingDriver::RegisterSimpleProjection(
    const std::string& field_name, 
    const std::string& display_name,
    bool default_enabled
) {
    // Get model compatibility from the projection trait
    auto compatibility = ProjectionType::GetModelCompatibility();
    
    // Create function object for this projection
    auto projection_func = [this, field_name](int phase) {
        // Skip if incompatible with this phase's model
        if ((compatibility == ProjectionTraits::ModelCompatibility::EXACMECH_ONLY && 
            m_phase_mech_types[phase] != MechType::EXACMECH) ||
            (compatibility == ProjectionTraits::ModelCompatibility::UMAT_ONLY && 
            m_phase_mech_types[phase] != MechType::UMAT)) {
            return;
        }
        
        this->ExecuteSimpleProjection<ProjectionType>(field_name, phase);
    };
    
    // Initialize per-phase enabled flags
    std::vector<bool> phase_enabled(m_sim_state.GetNumberOfPhases(), default_enabled);
    
    // Register the projection
    m_registered_projections.push_back({
        field_name,
        display_name,
        compatibility,
        phase_enabled,
        projection_func
    });
}

template<typename ProjectionType>
void PostProcessingDriver::RegisterSpecialProjection(
    const std::string& source_field, 
    const std::string& target_field,
    const std::string& display_name,
    bool default_enabled
) {
    // Get model compatibility from the projection trait
    auto compatibility = ProjectionType::GetModelCompatibility();
    
    // Create function object for this projection
    auto projection_func = [this, source_field, target_field](int phase) {
        // Skip if incompatible with this phase's model
        if ((compatibility == ProjectionTraits::ModelCompatibility::EXACMECH_ONLY && 
            m_phase_mech_types[phase] != MechType::EXACMECH) ||
            (compatibility == ProjectionTraits::ModelCompatibility::UMAT_ONLY && 
            m_phase_mech_types[phase] != MechType::UMAT)) {
            return;
        }
        
        this->ExecuteSpecialProjection<ProjectionType>(source_field, target_field, phase);
    };
    
    // Initialize per-phase enabled flags
    std::vector<bool> phase_enabled(m_sim_state.GetNumberOfPhases(), default_enabled);
    
    // Register the projection
    m_registered_projections.push_back({
        target_field,
        display_name,
        compatibility,
        phase_enabled,
        projection_func
    });
}

template<typename ProjectionType>
void PostProcessingDriver::RegisterGeometryProjection(
    const std::string& field_name, 
    const std::string& display_name,
    bool default_enabled
) {
    // Get model compatibility from the projection trait
    auto compatibility = ProjectionType::GetModelCompatibility();
    
    // Create function object for this projection
    auto projection_func = [this, field_name](int phase) {
        this->ExecuteGeometryProjection<ProjectionType>(field_name, phase);
    };
    
    // Initialize per-phase enabled flags
    std::vector<bool> phase_enabled(m_sim_state.GetNumberOfPhases(), default_enabled);
    
    // Register the projection
    m_registered_projections.push_back({
        field_name,
        display_name,
        compatibility,
        phase_enabled,
        projection_func
    });
}

template<typename ProjectionType>
void PostProcessingDriver::ExecuteSimpleProjection(const std::string& field_name, int phase) {
    auto field_map_name = m_sim_state.GetQuadratureFunctionMapName(field_name, phase);
    
    // Get state pair info
    auto state_pair = m_sim_state.GetQuadratureFunctionStatePair(field_map_name, phase);
    
    // Get the grid function to project to
    auto& grid_function = *m_map_gfs[field_map_name];
    
    // Project the component
    mfem::VectorQuadratureFunctionCoefficient qfvc(*m_evec);
    ProjectionTraits::ProjectionTrait<ProjectionType>::SelectComponent(qfvc, state_pair);
    grid_function.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
    
    // Apply any post-processing
    ProjectionTraits::ProjectionTrait<ProjectionType>::PostProcess(grid_function);
}

template<typename ProjectionType>
void PostProcessingDriver::ExecuteSpecialProjection(
    const std::string& source_field,
    const std::string& target_field,
    int phase
) {
    auto source_name = m_sim_state.GetQuadratureFunctionMapName(source_field, phase);
    auto target_name = m_sim_state.GetQuadratureFunctionMapName(target_field, phase);
    
    // Get the grid functions
    auto& source_gf = *m_map_gfs[source_name];
    auto& target_gf = *m_map_gfs[target_name];
    
    // Execute the specialized projection
    ProjectionType::PostProcess(source_gf, target_gf);
}

template<typename ProjectionType>
void PostProcessingDriver::ExecuteGeometryProjection(const std::string& field_name, int phase) {
    auto field_map_name = m_sim_state.GetQuadratureFunctionMapName(field_name, phase);
    
    // Get the grid function
    auto& grid_function = *m_map_gfs[field_map_name];
    
    // Execute specialized geometry projection
    ProjectionType::Project(
        m_sim_state.GetMeshParFiniteElementSpace(),
        grid_function
    );
}