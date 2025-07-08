#pragma once

#include "utilities/mechanics_kernels.hpp"
#include "sim_state/simulation_state.hpp"
#include "postprocessing/projection_class.hpp"

#include "mfem.hpp"
#include "ECMech_const.h"

// Forward declaration to avoid circular includes
class PostProcessingFileManager;

class LatticeTypeCubic;
template<class LatticeType>
class LightUp;
using LightUpCubic = LightUp<LatticeTypeCubic>;

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

    // Returns a pointer to a ParFiniteElementSpace (PFES) that's ordered according to VDIMs
    // and makes use of an L2 FiniteElementCollection
    // If the vdim is not in the internal mapping than a new PFES will be created
    std::shared_ptr<mfem::ParFiniteElementSpace> GetParFiniteElementSpace(const int region, const int vdim);

private:
    // Registration structures for projections and volume calculations
    struct ProjectionRegistration {
        std::string field_name;                      // Field identifier
        std::string display_name;                    // User-friendly name
        ProjectionTraits::ModelCompatibility model_compatibility; // Compatible models
        std::vector<bool> region_enabled;            // Per-region enabled flags
        std::vector<std::shared_ptr<ProjectionBase>> projection_class;    // Function to execute projection
        std::vector<int> region_length;
        bool supports_global_aggregation = false;    // Can be aggregated globally
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

    // Global aggregation methods
    void ExecuteGlobalProjection(const std::string& field_name);
    void CombineRegionDataToGlobal(const std::string& field_name);

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
    void VolumeEPS(const int region, const double time);
    void VolumeAvgElasticStrain(const int region, const double time);
    
    // Global volume average calculations
    void GlobalVolumeAvgStress(const double time);
    void GlobalVolumeAvgEulerStrain(const double time);
    void GlobalVolumeAvgDefGrad(const double time);
    void GlobalVolumePlWork(const double time);
    void GlobalVolumeEPS(const double time);
    void GlobalVolumeAvgElasticStrain(const double time);

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

    void UpdateFields(const int step, const double time);

    // Default projection and volume calculation registration
    void RegisterDefaultProjections();
    void RegisterDefaultVolumeCalculations();
    void RegisterProjection(const std::string& field);

    void InitializeLightUpAnalysis();
    void UpdateLightUpAnalysis();

private:
    // Reference to simulation state
    SimulationState& m_sim_state;
    
    // MPI rank
    int m_mpi_rank;
    int m_num_mpi_rank;

    
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
    std::map<int, std::map<int, std::shared_ptr<mfem::ParFiniteElementSpace>>> m_map_pfes;
    std::map<int, std::shared_ptr<mfem::ParMesh>> m_map_submesh;
    std::map<int, mfem::Array<int>> m_map_pqs2submesh;

    std::map<std::string, std::shared_ptr<mfem::ParGridFunction>> m_map_gfs;
    std::map<std::string, std::unique_ptr<mfem::DataCollection>> m_map_dcs;
    
    // Registered projections and volume calculations
    std::vector<ProjectionRegistration> m_registered_projections;
    std::vector<VolumeAverageRegistration> m_registered_volume_calcs;

    bool enable_visualization;

    // All light-up options that we might want to have
    std::vector<std::unique_ptr<LightUpCubic>> light_up_instances;
};
