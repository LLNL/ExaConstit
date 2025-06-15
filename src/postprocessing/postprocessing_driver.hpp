#pragma once

#include "mfem.hpp"
#include "mechanics_kernels.hpp"
#include "ECMech_const.h"
#include "sim_state/simulation_state.hpp"
#include "projection_traits_v2.hpp"

// Forward declaration to avoid circular includes
class PostProcessingFileManager;

/**
 * @brief PostProcessingDriver handles all post-processing operations for ExaConstit simulations
 * 
 * This class manages:
 * 1. Projection of quadrature data to grid functions for visualization
 * 2. Calculation of volume-averaged quantities (global and per-region)
 * 3. Registration of data with visualization collections
 * 4. Output of post-processed data at specified intervals
 * 5. Multi-material support with region-specific and combined visualizations
 */
class PostProcessingDriver {
public:
    /**
     * @brief Aggregation mode for multi-region data
     */
    enum class AggregationMode {
        PER_REGION,    // Process each region separately
        GLOBAL_COMBINED, // Combine all regions into global fields
        BOTH           // Both per-region and global combined
    };

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
    ~PostProcessingDriver();
    
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
     * @param mode Aggregation mode (default: BOTH)
     */
    void PrintVolValues(const double time, AggregationMode mode = AggregationMode::BOTH);
    
    /**
     * @brief Update data collections with current projection data
     * 
     * @param step Current time step
     * @param time Current simulation time
     */
    void UpdateDataCollections(const int step, const double time);
    
    /**
     * @brief Enable or disable a projection for a specific region
     * 
     * @param field_name Name of the field
     * @param region region index
     * @param enable Whether to enable the projection
     */
    void EnableProjection(const std::string& field_name, int region, bool enable = true);
    
    /**
     * @brief Enable or disable a projection for all regions
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
    
    /**
     * @brief Set aggregation mode for multi-region processing
     */
    void SetAggregationMode(AggregationMode mode) { m_aggregation_mode = mode; }
    
    /**
     * @brief Check if output should occur at this step (respects ExaOptions frequency)
     * 
     * @param step Current step number
     * @return true if output should occur
     */
    bool ShouldOutputAtStep(int step) const;

private:
    // Registration structures for projections and volume calculations
    struct ProjectionRegistration {
        std::string field_name;                      // Field identifier
        std::string display_name;                    // User-friendly name
        ProjectionTraits::ModelCompatibility model_compatibility; // Compatible models
        std::vector<bool> region_enabled;            // Per-region enabled flags
        std::function<void(int)> projection_func;    // Function to execute projection
        bool supports_global_aggregation = true;    // Can be aggregated globally
    };

    struct VolumeAverageRegistration {
        std::string calc_name;                       // Calculation identifier
        std::string display_name;                    // User-friendly name
        ProjectionTraits::ModelCompatibility model_compatibility; // Compatible models
        std::vector<bool> region_enabled;            // Per-region enabled flags
        std::function<void(int, double)> region_func; // Per-region calculation
        std::function<void(double)> global_func;     // Global aggregation function
        bool has_global_aggregation = true;         // Whether global calc is available
    };

    // Template registration methods
    template<typename ProjectionType>
    void RegisterSimpleProjection(
        const std::string& field_name, 
        const std::string& display_name,
        bool default_enabled = false,
        bool supports_global = true
    );
    
    template<typename ProjectionType>
    void RegisterSpecialProjection(
        const std::string& source_field, 
        const std::string& target_field,
        const std::string& display_name,
        bool default_enabled = false,
        bool supports_global = true
    );
    
    template<typename ProjectionType>
    void RegisterGeometryProjection(
        const std::string& field_name, 
        const std::string& display_name,
        bool default_enabled = false,
        bool supports_global = true
    );
    
    /**
     * @brief Register a volume average calculation
     * 
     * @param calc_name Name of the calculation
     * @param display_name Display name for UI
     * @param region_func Per-region calculation function
     * @param global_func Global aggregation function (optional)
     * @param enabled Whether enabled by default
     */
    void RegisterVolumeAverageFunction(
        const std::string& calc_name,
        const std::string& display_name,
        std::function<void(const int, const double)> region_func,
        std::function<void(const double)> global_func = nullptr,
        bool enabled = true
    );
    
    // Template execution methods
    template<typename ProjectionType>
    void ExecuteSimpleProjection(const std::string& field_name, int region);
    
    template<typename ProjectionType>
    void ExecuteSpecialProjection(
        const std::string& source_field,
        const std::string& target_field,
        int region
    );
    
    template<typename ProjectionType>
    void ExecuteGeometryProjection(const std::string& field_name, int region);
    
    // Global aggregation methods
    void ExecuteGlobalProjection(const std::string& field_name);
    void CombineRegionDataToGlobal(const std::string& field_name);
    
    /**
     * @brief Execute an elastic strain projection
     * 
     * @param strain_field Strain field name
     * @param vol_field Volume field name
     * @param region region index
     */
    void ExecuteElasticStrainProjection(
        const std::string& strain_field,
        const std::string& vol_field,
        int region
    );
    
    /**
     * @brief Initialize data collections for visualization
     * 
     * @param options Simulation options
     */
    void InitializeDataCollections(ExaOptions& options);
    
    /**
     * @brief Initialize grid functions for all registered projections
     */
    void InitializeGridFunctions();
    
    /**
     * @brief Check if a region has the required quadrature function
     */
    bool RegionHasQuadratureFunction(const std::string& field_name, int region) const;
    
    /**
     * @brief Get all active regions for a given field
     */
    std::vector<int> GetActiveRegionsForField(const std::string& field_name) const;
    
    // Volume average calculation methods (per-region)
    void VolumeAvgStress(const int region, const double time);
    void VolumeAvgEulerStrain(const int region, const double time);
    void VolumeAvgDefGrad(const int region, const double time);
    void VolumePlWork(const int region, const double time);
    void VolumeAvgElasticStrain(const int region, const double time);
    
    // Global volume average calculations
    void GlobalVolumeAvgStress(const double time);
    void GlobalVolumeAvgEulerStrain(const double time);
    void GlobalVolumeAvgDefGrad(const double time);
    void GlobalVolumePlWork(const double time);
    void GlobalVolumeAvgElasticStrain(const double time);
    
    // Projection methods (per-region implementations)
    void ProjectCentroid(const int region);
    void ProjectVolume(const int region);
    void ProjectModelStress(const int region);
    void ProjectVonMisesStress(const int region);
    void ProjectHydroStress(const int region);
    void ProjectDpEff(const int region);
    void ProjectEffPlasticStrain(const int region);
    void ProjectShearRate(const int region);
    void ProjectOrientation(const int region);
    void ProjectH(const int region);
    void ProjectElasticStrains(const int region);
    
    // Calculate element average values from partial quadrature function
    void CalcElementAvg(mfem::expt::PartialQuadratureFunction* elemVal, 
                       const mfem::expt::PartialQuadratureFunction* qf);
    
    // Calculate element average across regions for global aggregation
    void CalcGlobalElementAvg(mfem::Vector* elemVal, 
                             const std::string& field_name);
    
    // Helper to get quadrature function size
    size_t GetQuadratureFunctionSize() const;
    
    // Helper to get the appropriate grid function name
    std::string GetGridFunctionName(const std::string& field_name, int region = -1) const;
    
    // Default projection and volume calculation registration
    void RegisterDefaultProjections();
    void RegisterDefaultVolumeCalculations();
    
private:
    // Reference to simulation state
    SimulationState& m_sim_state;
    
    // MPI rank
    int m_mpi_rank;
    
    // Model types for each region
    std::vector<MechType> m_region_model_types;
    
    // Number of regions
    int m_num_regions;
    
    // Current aggregation mode
    AggregationMode m_aggregation_mode;
    
    // Buffer for element-averaged values (one per region + global)
    std::vector<std::unique_ptr<mfem::expt::PartialQuadratureFunction>> m_region_evec;
    std::unique_ptr<mfem::Vector> m_global_evec;
    
    // File manager for proper ExaOptions-compliant output
    std::unique_ptr<PostProcessingFileManager> m_file_manager;
    
    // Maps for grid functions and data collections
    std::map<std::string, std::unique_ptr<mfem::ParGridFunction>> m_map_gfs;
    std::map<std::string, std::unique_ptr<mfem::DataCollection>> m_map_dcs;
    
    // Registered projections and volume calculations
    std::vector<ProjectionRegistration> m_registered_projections;
    std::vector<VolumeAverageRegistration> m_registered_volume_calcs;

    bool enable_visualization;
};

// Template implementations

template<typename ProjectionType>
void PostProcessingDriver::RegisterSimpleProjection(
    const std::string& field_name, 
    const std::string& display_name,
    bool default_enabled,
    bool supports_global
) {
    // Get model compatibility from the projection trait
    auto compatibility = ProjectionType::GetModelCompatibility();
    
    // Create function object for this projection
    auto projection_func = [this, field_name, compatibility](int region) {
        // Skip if incompatible with this region's model
        if ((compatibility == ProjectionTraits::ModelCompatibility::EXACMECH_ONLY && 
            m_region_model_types[region] != MechType::EXACMECH) ||
            (compatibility == ProjectionTraits::ModelCompatibility::UMAT_ONLY && 
            m_region_model_types[region] != MechType::UMAT)) {
            return;
        }
        
        // Skip if region doesn't have the required quadrature function
        if (!RegionHasQuadratureFunction(field_name, region)) {
            return;
        }
        
        this->ExecuteSimpleProjection<ProjectionType>(field_name, region);
    };
    
    // Initialize per-region enabled flags
    std::vector<bool> region_enabled(m_num_regions, default_enabled);
    
    // Register the projection
    m_registered_projections.push_back({
        field_name,
        display_name,
        compatibility,
        region_enabled,
        projection_func,
        supports_global
    });
}

template<typename ProjectionType>
void PostProcessingDriver::RegisterSpecialProjection(
    const std::string& source_field, 
    const std::string& target_field,
    const std::string& display_name,
    bool default_enabled,
    bool supports_global
) {
    // Get model compatibility from the projection trait
    auto compatibility = ProjectionType::GetModelCompatibility();
    
    // Create function object for this projection
    auto projection_func = [this, source_field, target_field, compatibility](int region) {
        // Skip if incompatible with this region's model
        if ((compatibility == ProjectionTraits::ModelCompatibility::EXACMECH_ONLY && 
            m_region_model_types[region] != MechType::EXACMECH) ||
            (compatibility == ProjectionTraits::ModelCompatibility::UMAT_ONLY && 
            m_region_model_types[region] != MechType::UMAT)) {
            return;
        }
        
        // Skip if region doesn't have the required quadrature functions
        if (!RegionHasQuadratureFunction(source_field, region)) {
            return;
        }
        
        this->ExecuteSpecialProjection<ProjectionType>(source_field, target_field, region);
    };
    
    // Initialize per-region enabled flags
    std::vector<bool> region_enabled(m_num_regions, default_enabled);
    
    // Register the projection
    m_registered_projections.push_back({
        target_field,
        display_name,
        compatibility,
        region_enabled,
        projection_func,
        supports_global
    });
}

template<typename ProjectionType>
void PostProcessingDriver::RegisterGeometryProjection(
    const std::string& field_name, 
    const std::string& display_name,
    bool default_enabled,
    bool supports_global
) {
    // Get model compatibility from the projection trait
    auto compatibility = ProjectionType::GetModelCompatibility();
    
    // Create function object for this projection
    auto projection_func = [this, field_name](int region) {
        this->ExecuteGeometryProjection<ProjectionType>(field_name, region);
    };
    
    // Initialize per-region enabled flags
    std::vector<bool> region_enabled(m_num_regions, default_enabled);
    
    // Register the projection
    m_registered_projections.push_back({
        field_name,
        display_name,
        compatibility,
        region_enabled,
        projection_func,
        supports_global
    });
}

template<typename ProjectionType>
void PostProcessingDriver::ExecuteSimpleProjection(const std::string& field_name, int region) {
    auto field_map_name = GetGridFunctionName(field_name, region);
    
    // Get the partial quadrature function for this region
    auto pqf = m_sim_state.GetQuadratureFunction(field_name, region);
    if (!pqf) {
        return; // This region doesn't have this quadrature function
    }
    
    // Get state pair info for the partial quadrature function
    auto& region_evec = *m_region_evec[region];
    
    // Calculate element averages for this region's data
    CalcElementAvg(&region_evec, pqf.get());
    
    // Get the grid function to project to
    auto& grid_function = *m_map_gfs[field_map_name];
    
    // Project the component using the region-specific element averages
    mfem::VectorQuadratureFunctionCoefficient qfvc(region_evec);
    auto [index, length] = m_sim_state.GetQuadratureFunctionStatePair(field_name, region);

    if (index == -1) {
        index = 0;
        length = pqf->GetVDim();
    }

    ProjectionTraits::ProjectionTrait<ProjectionType>::SelectComponent(qfvc, index, length);
    grid_function.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
    
    // Apply any post-processing
    ProjectionTraits::ProjectionTrait<ProjectionType>::PostProcess(grid_function);
}

template<typename ProjectionType>
void PostProcessingDriver::ExecuteSpecialProjection(
    const std::string& source_field,
    const std::string& target_field,
    int region
) {
    auto source_name = GetGridFunctionName(source_field, region);
    auto target_name = GetGridFunctionName(target_field, region);
    
    // Get the grid functions
    auto& source_gf = *m_map_gfs[source_name];
    auto& target_gf = *m_map_gfs[target_name];
    
    // Execute the specialized projection
    ProjectionType::PostProcess(source_gf, target_gf);
}

template<typename ProjectionType>
void PostProcessingDriver::ExecuteGeometryProjection(const std::string& field_name, int region) {
    auto field_map_name = GetGridFunctionName(field_name, region);
    
    // Get the grid function
    auto& grid_function = *m_map_gfs[field_map_name];
    
    // Execute specialized geometry projection
    // Note: Geometry projections typically don't depend on region-specific data
    ProjectionType::Project(
        m_sim_state.GetMeshParFiniteElementSpace().get(),
        grid_function
    );
}