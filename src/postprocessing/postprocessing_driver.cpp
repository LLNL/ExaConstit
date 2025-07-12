#include "postprocessing_driver.hpp"
#include "postprocessing_file_manager.hpp"
#include "postprocessing/projection_class.hpp"
#include "postprocessing/mechanics_lightup.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"

#include "SNLS_linalg.h"
#include "ECMech_const.h"
#include "mechanics_lightup.hpp"

#include <filesystem>
namespace fs = std::filesystem;

namespace {

/**
 * @brief Generic registration template for projection types
 * 
 * @tparam T Projection class type to register
 * @param region_model_types Vector of material model types per region
 * @return Vector of shared projection instances, one per region plus global
 * 
 * Creates projection instances for all regions plus one additional
 * global instance. Each projection is wrapped in a shared_ptr for
 * efficient memory management and polymorphic behavior.
 * 
 * The template design enables type-safe registration of any
 * projection class derived from ProjectionBase.
 */
template<class T>
std::vector<std::shared_ptr<ProjectionBase>>
RegisterGeneric(const std::vector<MechType>& region_model_types)
{
    std::vector<std::shared_ptr<ProjectionBase>> base;
    const size_t num_regions = region_model_types.size() + 1;
    for (size_t i = 0; i < num_regions; i++) {
        base.emplace_back(std::make_shared<T>());
    }
    return base;
}

/**
 * @brief Register centroid projections for all regions
 * 
 * @param region_model_types Vector of material model types per region
 * @return Vector of CentroidProjection instances
 * 
 * Creates centroid projection instances that compute geometric
 * centroids of mesh elements. Compatible with all material model
 * types as it depends only on mesh geometry.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterCentroid(const std::vector<MechType>& region_model_types)
{
    return RegisterGeneric<CentroidProjection>(region_model_types);
}

/**
 * @brief Register volume projections for all regions
 * 
 * @param region_model_types Vector of material model types per region
 * @return Vector of VolumeProjection instances
 * 
 * Creates volume projection instances that compute element volumes
 * from integration of geometric determinants. Provides essential
 * geometric information for visualization and volume averaging.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterVolume(const std::vector<MechType>& region_model_types)
{
    return RegisterGeneric<VolumeProjection>(region_model_types);
}

/**
 * @brief Register Cauchy stress projections for all regions
 * 
 * @param region_model_types Vector of material model types per region
 * @return Vector of CauchyStressProjection instances
 * 
 * Creates projections for full Cauchy stress tensor (6 components
 * in Voigt notation). Compatible with all material models that
 * provide stress state information.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterCauchyStress(const std::vector<MechType>& region_model_types)
{
    return RegisterGeneric<CauchyStressProjection>(region_model_types);
}

/**
 * @brief Register Von Mises stress projections for all regions
 * 
 * @param region_model_types Vector of material model types per region
 * @return Vector of VonMisesStressProjection instances
 * 
 * Creates projections that compute Von Mises equivalent stress
 * from the Cauchy stress tensor. Provides scalar stress measure
 * commonly used for yield and failure analysis.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterVMStress(const std::vector<MechType>& region_model_types)
{
    return RegisterGeneric<VonMisesStressProjection>(region_model_types);
}

/**
 * @brief Register hydrostatic stress projections for all regions
 * 
 * @param region_model_types Vector of material model types per region
 * @return Vector of HydrostaticStressProjection instances
 * 
 * Creates projections that compute hydrostatic (mean) stress
 * component. Essential for analyzing volumetric deformation
 * and pressure-dependent material behavior.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterHydroStress(const std::vector<MechType>& region_model_types)
{
    return RegisterGeneric<HydrostaticStressProjection>(region_model_types);
}

/**
 * @brief Register all state variables projections for all regions
 * 
 * @param region_model_types Vector of material model types per region
 * @return Vector of AllStateVariablesProjection instances
 * 
 * Creates projections that output all available state variables
 * for debugging and detailed analysis. State variable count and
 * interpretation depend on the specific material model.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterAllState(const std::vector<MechType>& region_model_types)
{
    return RegisterGeneric<AllStateVariablesProjection>(region_model_types);
}

/**
 * @brief Generic registration template for ECMech-specific projections
 * 
 * @tparam T ECMech projection class type
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @param key State variable key name for ECMech lookup
 * @return Vector of ECMech projection instances
 * 
 * Creates ECMech-specific projections with automatic state variable
 * index resolution. Non-ECMech regions receive dummy projections
 * with invalid indices. The maximum state variable length across
 * all regions is tracked for consistent vector dimensions.
 */
template<class T>
std::vector<std::shared_ptr<ProjectionBase>>
RegisterECMech(const SimulationState& sim_state, const std::vector<MechType>& region_model_types, const std::string key)
{
    std::vector<std::shared_ptr<ProjectionBase>> base;
    const size_t num_regions = region_model_types.size();
    int max_length = -1;
    for (size_t i = 0; i < num_regions; i++) {
        if (region_model_types[i] != MechType::EXACMECH) {
            // Need to do a basic guard against non-ecmech models
            base.emplace_back(std::make_shared<T>("", -1, -1));
            continue;
        }
        auto [index, length] = sim_state.GetQuadratureFunctionStatePair(key, i);
        base.emplace_back(std::make_shared<T>(key, index, length));
        max_length = (max_length < length) ? length : max_length;

    }

    if (base[0]->CanAggregateGlobally()) {
        base.emplace_back(std::make_shared<T>(key, 0, max_length));
    }

    return base;
}

/**
 * @brief Register DpEff (effective plastic strain rate) projections for ExaCMech
 * 
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of DpEffProjection instances
 * 
 * Creates DpEffProjection instances for regions with ExaCMech material models.
 * Uses the "eq_pl_strain_rate" state variable key to access effective plastic
 * strain rate data. Non-ExaCMech regions receive dummy projections.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterDpEffProjection(const SimulationState& sim_state, const std::vector<MechType>& region_model_types)
{
    std::string key = "eq_pl_strain_rate";
    return RegisterECMech<DpEffProjection>(sim_state, region_model_types, key);
}

/**
 * @brief Register crystal orientation projections for ExaCMech
 * 
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of XtalOrientationProjection instances
 * 
 * Creates crystal orientation projection instances using the "quats" state
 * variable key to access quaternion orientation data. Only compatible with
 * ExaCMech material models that provide crystal orientation information.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterXtalOriProjection(const SimulationState& sim_state, const std::vector<MechType>& region_model_types)
{
    std::string key = "quats";
    return RegisterECMech<XtalOrientationProjection>(sim_state, region_model_types, key);
}

/**
 * @brief Register elastic strain projections for ExaCMech
 * 
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of ElasticStrainProjection instances
 * 
 * Creates elastic strain projection instances using the "elastic_strain" state
 * variable key. Handles coordinate transformations and tensor reconstruction
 * for ExaCMech elastic strain data.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterElasticStrainProjection(const SimulationState& sim_state, const std::vector<MechType>& region_model_types)
{
    std::string key = "elastic_strain";
    return RegisterECMech<ElasticStrainProjection>(sim_state, region_model_types, key);
}

/**
 * @brief Register hardness projections for ExaCMech
 * 
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of HardnessProjection instances
 * 
 * Creates hardness projection instances using the "hardness" state variable
 * key. Includes post-processing to ensure non-negative hardness values
 * suitable for visualization and analysis.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterHardnessProjection(const SimulationState& sim_state, const std::vector<MechType>& region_model_types)
{
    std::string key = "hardness";
    return RegisterECMech<HardnessProjection>(sim_state, region_model_types, key);
}

/**
 * @brief Register shear rate projections for ExaCMech
 * 
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of ShearingRateProjection instances
 * 
 * Creates shear rate projection instances using the "shear_rate" state
 * variable key. Provides access to macroscopic shear rate data for
 * rate-dependent analysis and deformation characterization.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterShearRateProjection(const SimulationState& sim_state, const std::vector<MechType>& region_model_types)
{
    std::string key = "shear_rate";
    return RegisterECMech<ShearingRateProjection>(sim_state, region_model_types, key);
}
}

void PostProcessingDriver::RegisterProjection(
    const std::string& field)
{

    std::vector<std::shared_ptr<ProjectionBase>> projection_class;

    if (field == "centroid") {
        projection_class = RegisterCentroid(m_region_model_types);
    }
    else if (field == "volume") {
        projection_class = RegisterVolume(m_region_model_types);
    }
    else if (field == "cauchy") {
        projection_class = RegisterCauchyStress(m_region_model_types);
    }
    else if (field == "von_mises") {
        projection_class = RegisterVMStress(m_region_model_types);
    }
    else if (field == "hydro") {
        projection_class = RegisterHydroStress(m_region_model_types);
    }
    else if (field == "all_state") {
        projection_class = RegisterAllState(m_region_model_types);
    }
    else if (field == "dpeff") {
        projection_class = RegisterDpEffProjection(m_sim_state, m_region_model_types);
    }
    else if (field == "xtal_ori") {
        projection_class = RegisterXtalOriProjection(m_sim_state, m_region_model_types);
    }
    else if (field == "elastic_strain") {
        projection_class = RegisterElasticStrainProjection(m_sim_state, m_region_model_types);
    }
    else if (field == "hardness") {
        projection_class = RegisterHardnessProjection(m_sim_state, m_region_model_types);
    }
    else if (field == "shear_rate") {
        projection_class = RegisterShearRateProjection(m_sim_state, m_region_model_types);
    }
    else {
        return;
    }

    std::string field_name = field;
    std::string display_name = projection_class[0]->GetDisplayName();
    using PTMC = ProjectionTraits::ModelCompatibility;
    PTMC model_compatibility = projection_class[0]->model;
    bool supports_global_aggregation = projection_class[0]->CanAggregateGlobally();

    std::vector<bool> region_enabled;
    std::vector<int> region_length;

    for (size_t i = 0; i < m_region_model_types.size(); i++) {
        const auto model = m_region_model_types[i];
        const auto project_model = projection_class[i]->model;
        region_length.push_back(projection_class[i]->GetVectorDimension());
        if (project_model == PTMC::EXACMECH_ONLY && model == MechType::EXACMECH) {
            region_enabled.push_back(true);
        }
        else if (project_model == PTMC::EXACMECH_ONLY && model == MechType::UMAT) {
            region_enabled.push_back(false);
        }
        else if (project_model == PTMC::UMAT_ONLY && model == MechType::EXACMECH)
        {
            region_enabled.push_back(false);
        }
        else if (project_model == PTMC::UMAT_ONLY && model == MechType::UMAT)
        {
            region_enabled.push_back(true);
        }
        else if (project_model == PTMC::ALL_MODELS) {
            region_enabled.push_back(true);
        } else {
            region_enabled.push_back(false);
        }
    }
    if (supports_global_aggregation) {
        region_enabled.push_back(true);
        region_length.push_back(projection_class[m_region_model_types.size()]->GetVectorDimension());
    }

    // Register the projection
    m_registered_projections.push_back({
        field_name,
        display_name,
        model_compatibility,
        region_enabled,
        projection_class,
        region_length,
        supports_global_aggregation
    });
}

PostProcessingDriver::PostProcessingDriver(SimulationState& sim_state, ExaOptions& options)
    : m_sim_state(sim_state),
      m_mpi_rank(0),
      m_num_regions(sim_state.GetNumberOfRegions()),
      m_aggregation_mode(AggregationMode::BOTH),
      enable_visualization(options.visualization.visit || 
                           options.visualization.conduit || 
                           options.visualization.paraview || 
                           options.visualization.adios2)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &m_mpi_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &m_num_mpi_rank);
    
    // Initialize file manager with proper ExaOptions handling
    m_file_manager = std::make_unique<PostProcessingFileManager>(options, m_mpi_rank);
    
    // Ensure output directory exists
    if (!m_file_manager->EnsureOutputDirectoryExists()) {
        if (m_mpi_rank == 0) {
            std::cerr << "Warning: Failed to create output directory. Volume averaging may fail." << std::endl;
        }
    }
    
    // Initialize region-specific data structures
    m_region_model_types.resize(m_num_regions);
    m_region_evec.resize(m_num_regions);
    
    int max_vdim = 0;
    // Get model types for each region
    for (int region = 0; region < m_num_regions; ++region) {
        m_region_model_types[region] = sim_state.GetRegionModelType(region);
        // Initialize region-specific element average buffer
        if (auto pqf = sim_state.GetQuadratureFunction("cauchy_stress_end", region)) {
            // Find maximum vdim across all possible quadrature functions for this region
            for (const auto& field_name : {"cauchy_stress_end", "state_var_end", "von_mises", "kinetic_grads"}) {
                if (auto qf = sim_state.GetQuadratureFunction(field_name, region)) {
                    max_vdim = std::max(max_vdim, qf->GetVDim());
                }
            }
            // Create element average buffer with maximum dimension needed
            m_region_evec[region] = std::make_unique<mfem::expt::PartialQuadratureFunction>(
                pqf->GetPartialSpaceShared(), max_vdim);
        }
    }
    
    // Initialize global element average buffer
    auto fe_space = sim_state.GetMeshParFiniteElementSpace();
    int global_max_vdim = max_vdim; // Accommodate stress tensors and other multi-component fields
    m_global_evec = std::make_unique<mfem::Vector>(global_max_vdim * fe_space->GetNE());
    m_global_evec->UseDevice(true);
    
    // Register default projections and volume calculations
    RegisterDefaultVolumeCalculations();
    
    // Initialize grid functions and data collections
    if (enable_visualization) {
        auto mesh = m_sim_state.getMesh();
        if (m_num_regions == 1) {
            auto l2g = sim_state.GetQuadratureFunction("cauchy_stress_end", 0)->GetPartialSpaceShared()->getLocal2Global();
            mfem::Array<int> pqs2submesh(l2g);
            m_map_pqs2submesh.emplace(0, std::move(pqs2submesh));
            m_map_submesh.emplace(0, mesh);
        }
        else {
            for (int region = 0; region < m_num_regions; ++region) {
                auto pqs = sim_state.GetQuadratureFunction("cauchy_stress_end", region)->GetPartialSpaceShared();
                auto l2g = pqs->getLocal2Global();
                mfem::Array<int> pqs2submesh(l2g.Size());
    
                mfem::Array<int> domain(1);
                domain[0] = region + 1;    
                auto submesh = mfem::ParSubMesh::CreateFromDomain(*mesh.get(), domain);
    
                for (int i = 0; i < l2g.Size(); i++) {
                    pqs2submesh[i] = submesh.GetSubMeshElementFromParent(l2g[i]);
                }
                auto submesh_ptr = std::make_shared<mfem::ParSubMesh>(std::move(submesh));
                m_map_pqs2submesh.emplace(region, std::move(pqs2submesh));
                m_map_submesh.emplace(region, std::move(submesh_ptr));
            }
        }

        RegisterDefaultProjections();
        InitializeGridFunctions();
        InitializeDataCollections(options);
    }

    InitializeLightUpAnalysis();
}

std::shared_ptr<mfem::ParFiniteElementSpace> PostProcessingDriver::GetParFiniteElementSpace(const int region, const int vdim)
{
    if (!enable_visualization) { return std::shared_ptr<mfem::ParFiniteElementSpace>(); }

    if (m_map_pfes.find(region) == m_map_pfes.end())
    {
        m_map_pfes.emplace(region, std::map<int, std::shared_ptr<mfem::ParFiniteElementSpace>>());
    }

    if (m_map_pfes[region].find(vdim) == m_map_pfes[region].end())
    {
        auto mesh = m_map_submesh[region];
        const int space_dim = mesh->SpaceDimension();
        std::string l2_fec_str = "L2_" + std::to_string(space_dim) + "D_P" + std::to_string(0);
        auto l2_fec = m_sim_state.GetFiniteElementCollection(l2_fec_str);
        auto value = std::make_shared<mfem::ParFiniteElementSpace>(mesh.get(), l2_fec.get(), vdim, mfem::Ordering::byVDIM);
        m_map_pfes[region].emplace(vdim, std::move(value));
    }
    return m_map_pfes[region][vdim];
}

void PostProcessingDriver::UpdateFields([[maybe_unused]] const int step, [[maybe_unused]] const double time) {
    for (int region = 0; region < m_num_regions; ++region) {
        auto state_qf_avg = m_sim_state.GetQuadratureFunction("state_var_avg", region);
        auto state_qf_end = m_sim_state.GetQuadratureFunction("state_var_end", region);
        CalcElementAvg(state_qf_avg.get(), state_qf_end.get());
        auto cauchy_qf_avg = m_sim_state.GetQuadratureFunction("cauchy_stress_avg", region);
        auto cauchy_qf_end = m_sim_state.GetQuadratureFunction("cauchy_stress_end", region);
        CalcElementAvg(cauchy_qf_avg.get(), cauchy_qf_end.get());
    }

    // Execute projections based on aggregation mode
    if (m_aggregation_mode == AggregationMode::PER_REGION || 
        m_aggregation_mode == AggregationMode::BOTH) {

        // Process each region separately
        for (int region = 0; region < m_num_regions; ++region) {
            auto qpts2mesh = m_map_pqs2submesh[region];
            for (auto& reg : m_registered_projections) {
                if (reg.region_enabled[region]) {
                    const auto gf_name = GetGridFunctionName(reg.field_name, region);
                    auto& grid_func = m_map_gfs[gf_name];
                    reg.projection_class[region]->Execute(m_sim_state, grid_func, qpts2mesh, region);
                }
            }
        }
    }

    if (m_aggregation_mode == AggregationMode::GLOBAL_COMBINED || 
        m_aggregation_mode == AggregationMode::BOTH) {
        
        // Execute global aggregated projections
        for (auto& reg : m_registered_projections) {
            if (reg.supports_global_aggregation) {
                ExecuteGlobalProjection(reg.field_name);
            }
        }
    }
}

void PostProcessingDriver::Update(const int step, const double time) {
    CALI_CXX_MARK_SCOPE("postprocessing_update");
    UpdateFields(step, time);
    // Check if we should output volume averages at this step
    if (ShouldOutputAtStep(step)) {
        PrintVolValues(time, m_aggregation_mode);
    }
    
    // Update data collections for visualization
    if (enable_visualization) {
        UpdateDataCollections(step, time);
    }

    if (light_up_instances.size() > 0) {
        UpdateLightUpAnalysis();
    }
}

PostProcessingDriver::~PostProcessingDriver() = default;

void PostProcessingDriver::PrintVolValues(const double time, AggregationMode mode) {
    CALI_CXX_MARK_SCOPE("postprocessing_vol_values");
    
    if (mode == AggregationMode::PER_REGION || mode == AggregationMode::BOTH) {
        // Calculate per-region volume averages
        for (int region = 0; region < m_num_regions; ++region) {
            for (auto& reg : m_registered_volume_calcs) {
                if (reg.region_enabled[region]) {
                    reg.region_func(region, time);
                }
            }
        }
    }
    
    if (mode == AggregationMode::GLOBAL_COMBINED || mode == AggregationMode::BOTH) {
        // Calculate global aggregated volume averages
        for (auto& reg : m_registered_volume_calcs) {
            if (reg.has_global_aggregation && reg.global_func) {
                reg.global_func(time);
            }
        }
    }
}

bool PostProcessingDriver::ShouldOutputAtStep(int step) const {
    return m_file_manager->ShouldOutputAtStep(step);
}

void PostProcessingDriver::VolumeAvgStress(const int region, const double time) {
    auto stress_pqf = m_sim_state.GetQuadratureFunction("cauchy_stress_end", region);
    if (!stress_pqf) {
        return; // This region doesn't have stress data
    }
    
    // Calculate volume-averaged stress for this region
    mfem::Vector avg_stress(6); // Symmetric stress tensor
    
    double total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
        stress_pqf.get(), avg_stress, 6, m_sim_state.getOptions().solvers.rtmodel);
    
    // Output to region-specific file using file manager
    if (m_mpi_rank == 0) {
        auto region_name = m_sim_state.GetRegionName(region);
        auto filepath = m_file_manager->GetVolumeAverageFilePath("stress", region, region_name);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("stress");
            }
            
            *file << time << " " << total_volume;
            for (int i = 0; i < 6; ++i) {
                *file << " " << avg_stress[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::GlobalVolumeAvgStress(const double time) {
    mfem::Vector global_avg_stress(6);
    global_avg_stress = 0.0;
    double global_volume = 0.0;
    
    // Accumulate contributions from all regions
    for (int region = 0; region < m_num_regions; ++region) {
        auto stress_pqf = m_sim_state.GetQuadratureFunction("cauchy_stress_end", region);
        if (!stress_pqf) {
            continue;
        }
        
        mfem::Vector region_stress(6);
        
        double region_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
            stress_pqf.get(), region_stress, 6, m_sim_state.getOptions().solvers.rtmodel);
        
        // Volume-weighted average
        for (int i = 0; i < 6; ++i) {
            global_avg_stress[i] += region_stress[i] * region_volume;
        }
        global_volume += region_volume;
    }
    
    // Normalize by total volume
    if (global_volume > 0.0) {
        global_avg_stress /= global_volume;
    }
    
    // Output to global file
    if (m_mpi_rank == 0) {
        auto filepath = m_file_manager->GetVolumeAverageFilePath("stress", -1);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("stress");
            }
            
            *file << time << " " << global_volume;
            for (int i = 0; i < 6; ++i) {
                *file << " " << global_avg_stress[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::VolumeAvgDefGrad(const int region, const double time) {
    auto def_grad_pqf = m_sim_state.GetQuadratureFunction("kinetic_grads", region);
    auto def_grad_global = m_sim_state.GetQuadratureFunction("kinetic_grads", -1);

    if (!def_grad_pqf) {
        return;
    }
    
    def_grad_pqf->operator=(*dynamic_cast<mfem::QuadratureFunction*>(def_grad_global.get()));

    // Calculate volume-averaged deformation gradient for this region
    mfem::Vector avg_def_grad(9); // 3x3 tensor
    
    double total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
        def_grad_pqf.get(), avg_def_grad, 9, m_sim_state.getOptions().solvers.rtmodel);
    
    // Output to region-specific file using file manager
    if (m_mpi_rank == 0) {
        auto region_name = m_sim_state.GetRegionName(region);
        auto filepath = m_file_manager->GetVolumeAverageFilePath("def_grad", region, region_name);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("def_grad");
            }
            
            *file << time << " " << total_volume;
            for (int i = 0; i < 9; ++i) {
                *file << " " << avg_def_grad[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::GlobalVolumeAvgDefGrad(const double time) {
    mfem::Vector global_avg_def_grad(9);
    global_avg_def_grad = 0.0;
    double global_volume = 0.0;
    auto def_grad_global = m_sim_state.GetQuadratureFunction("kinetic_grads", -1);

    // Accumulate contributions from all regions
    for (int region = 0; region < m_num_regions; ++region) {
        auto def_grad_pqf = m_sim_state.GetQuadratureFunction("kinetic_grads", region);    
        if (!def_grad_pqf) {
            continue;
        }
        
        def_grad_pqf->operator=(*dynamic_cast<mfem::QuadratureFunction*>(def_grad_global.get()));
        
        mfem::Vector region_def_grad(9);
        
        double region_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
            def_grad_pqf.get(), region_def_grad, 9, m_sim_state.getOptions().solvers.rtmodel);
        
        // Volume-weighted average
        for (int i = 0; i < 9; ++i) {
            global_avg_def_grad[i] += region_def_grad[i] * region_volume;
        }
        global_volume += region_volume;
    }
    
    // Normalize by total volume
    if (global_volume > 0.0) {
        global_avg_def_grad /= global_volume;
    }
    
    // Output to global file
    if (m_mpi_rank == 0) {
        auto filepath = m_file_manager->GetVolumeAverageFilePath("def_grad", -1);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("def_grad");
            }
            
            *file << time << " " << global_volume;
            for (int i = 0; i < 9; ++i) {
                *file << " " << global_avg_def_grad[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::VolumePlWork(const int region, const double time) {
    auto pl_work_pqf = m_sim_state.GetQuadratureFunction("scalar", region);
    if (!pl_work_pqf) {
        return;
    }

    auto state_vars = m_sim_state.GetQuadratureFunction("state_var_end", region)->Read();
    const int vdim = m_sim_state.GetQuadratureFunction("state_var_end", region)->GetVDim();
    const int pl_work_ind = m_sim_state.GetQuadratureFunctionStatePair("plastic_work", region).first;
    auto data = pl_work_pqf->Write();

    mfem::forall(pl_work_pqf->Size(), [=] MFEM_HOST_DEVICE (int i) {
        data[i] = state_vars[i * vdim + pl_work_ind];
    });
    
    // Calculate volume-averaged plastic work for this region
    mfem::Vector avg_pl_work(1); // Scalar quantity
    
    double total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<false>(
        pl_work_pqf.get(), avg_pl_work, 1, m_sim_state.getOptions().solvers.rtmodel);
    
    // Output to region-specific file using file manager
    if (m_mpi_rank == 0) {
        auto region_name = m_sim_state.GetRegionName(region);
        auto filepath = m_file_manager->GetVolumeAverageFilePath("plastic_work", region, region_name);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("plastic_work");
            }
            
            *file << time << " " << total_volume << " " << avg_pl_work[0] << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::GlobalVolumePlWork(const double time) {
    double global_avg_pl_work = 0.0;
    double global_volume = 0.0;
    
    // Accumulate contributions from all regions
    for (int region = 0; region < m_num_regions; ++region) {
        auto pl_work_pqf = m_sim_state.GetQuadratureFunction("scalar", region);
        if (!pl_work_pqf) {
            continue;
        }

        auto state_vars = m_sim_state.GetQuadratureFunction("state_var_end", region)->Read();
        const int vdim = m_sim_state.GetQuadratureFunction("state_var_end", region)->GetVDim();
        const int pl_work_ind = m_sim_state.GetQuadratureFunctionStatePair("plastic_work", region).first;
        auto data = pl_work_pqf->Write();
    
        mfem::forall(pl_work_pqf->Size(), [=] MFEM_HOST_DEVICE (int i) {
            data[i] = state_vars[i * vdim + pl_work_ind];
        });
        
        mfem::Vector region_pl_work(1);
        
        double region_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<false>(
            pl_work_pqf.get(), region_pl_work, 1, m_sim_state.getOptions().solvers.rtmodel);
        
        // Volume-weighted average
        global_avg_pl_work += region_pl_work[0] * region_volume;
        global_volume += region_volume;
    }
    
    // Normalize by total volume
    if (global_volume > 0.0) {
        global_avg_pl_work /= global_volume;
    }
    
    // Output to global file
    if (m_mpi_rank == 0) {
        auto filepath = m_file_manager->GetVolumeAverageFilePath("plastic_work", -1);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("plastic_work");
            }
            
            *file << time << " " << global_volume << " " << global_avg_pl_work << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::VolumeEPS(const int region, const double time) {
    auto eps_pqf = m_sim_state.GetQuadratureFunction("scalar", region);
    if (!eps_pqf) {
        return;
    }

    auto state_vars = m_sim_state.GetQuadratureFunction("state_var_end", region)->Read();
    const int vdim = m_sim_state.GetQuadratureFunction("state_var_end", region)->GetVDim();
    const int eps_ind = m_sim_state.GetQuadratureFunctionStatePair("eq_pl_strain", region).first;
    auto data = eps_pqf->Write();

    mfem::forall(eps_pqf->Size(), [=] MFEM_HOST_DEVICE (int i) {
        data[i] = state_vars[i * vdim + eps_ind];
    });
    
    // Calculate volume-averaged equivalent plastic strain for this region
    mfem::Vector avg_eps(1); // Scalar quantity
    
    double total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
        eps_pqf.get(), avg_eps, 1, m_sim_state.getOptions().solvers.rtmodel);
    
    // Output to region-specific file using file manager
    if (m_mpi_rank == 0) {
        auto region_name = m_sim_state.GetRegionName(region);
        auto filepath = m_file_manager->GetVolumeAverageFilePath("eq_pl_strain", region, region_name);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("eq_pl_strain");
            }
            
            *file << time << " " << total_volume << " " << avg_eps[0] << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::GlobalVolumeEPS(const double time) {
    double global_avg_eps = 0.0;
    double global_volume = 0.0;
    
    // Accumulate contributions from all regions
    for (int region = 0; region < m_num_regions; ++region) {
        auto eps_pqf = m_sim_state.GetQuadratureFunction("scalar", region);
        if (!eps_pqf) {
            continue;
        }
    
        auto state_vars = m_sim_state.GetQuadratureFunction("state_var_end", region)->Read();
        const int vdim = m_sim_state.GetQuadratureFunction("state_var_end", region)->GetVDim();
        const int eps_ind = m_sim_state.GetQuadratureFunctionStatePair("eq_pl_strain", region).first;
        auto data = eps_pqf->Write();
    
        mfem::forall(eps_pqf->Size(), [=] MFEM_HOST_DEVICE (int i) {
            data[i] = state_vars[i * vdim + eps_ind];
        });
        
        mfem::Vector region_eq_pl_strain(1);
        
        double region_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
            eps_pqf.get(), region_eq_pl_strain, 1, m_sim_state.getOptions().solvers.rtmodel);
        
        // Volume-weighted average
        global_avg_eps += region_eq_pl_strain[0] * region_volume;
        global_volume += region_volume;
    }
    
    // Normalize by total volume
    if (global_volume > 0.0) {
        global_avg_eps /= global_volume;
    }
    
    // Output to global file
    if (m_mpi_rank == 0) {
        auto filepath = m_file_manager->GetVolumeAverageFilePath("eq_pl_strain", -1);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("eq_pl_strain");
            }
            
            *file << time << " " << global_volume << " " << global_avg_eps << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::VolumeAvgEulerStrain(const int region, const double time) {
    auto euler_strain_global = m_sim_state.GetQuadratureFunction("kinetic_grads", -1);
    auto euler_strain_pqf = m_sim_state.GetQuadratureFunction("kinetic_grads", region);    
    if (!euler_strain_pqf) {
        return;
    }
    
    euler_strain_pqf->operator=(*dynamic_cast<mfem::QuadratureFunction*>(euler_strain_global.get()));
    
    mfem::Vector avg_def_grad(9);
    mfem::Vector avg_euler_strain(6);

    
    double total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
        euler_strain_pqf.get(), avg_def_grad, 9, m_sim_state.getOptions().solvers.rtmodel);

    {
        mfem::DenseMatrix euler_strain(3, 3);
        mfem::DenseMatrix def_grad(avg_def_grad.HostReadWrite(), 3, 3);
        int dim = 3;
        mfem::DenseMatrix Finv(dim), Binv(dim);
        double half = 1.0 / 2.0;

        mfem::CalcInverse(def_grad, Finv);
        mfem::MultAtB(Finv, Finv, Binv);
     
        euler_strain = 0.0;
     
        for (int j = 0; j < dim; j++) {
           for (int i = 0; i < dim; i++) {
            euler_strain(i, j) -= half * Binv(i, j);
           }
     
           euler_strain(j, j) += half;
        }

        avg_euler_strain(0) = euler_strain(0, 0);
        avg_euler_strain(1) = euler_strain(1, 1);
        avg_euler_strain(2) = euler_strain(2, 2);
        avg_euler_strain(3) = euler_strain(1, 2);
        avg_euler_strain(4) = euler_strain(0, 2);
        avg_euler_strain(5) = euler_strain(0, 1);
    }
    
    if (m_mpi_rank == 0) {
        auto region_name = m_sim_state.GetRegionName(region);
        auto filepath = m_file_manager->GetVolumeAverageFilePath("euler_strain", region, region_name);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("euler_strain");
            }
            
            *file << time << " " << total_volume;
            for (int i = 0; i < 6; ++i) {
                *file << " " << avg_euler_strain[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::GlobalVolumeAvgEulerStrain(const double time) {
    mfem::Vector global_avg_euler_strain(6);
    global_avg_euler_strain = 0.0;
    double global_volume = 0.0;
    auto euler_strain_global = m_sim_state.GetQuadratureFunction("kinetic_grads", -1);

    for (int region = 0; region < m_num_regions; ++region) {

        auto euler_strain_pqf = m_sim_state.GetQuadratureFunction("kinetic_grads", region);    
        if (!euler_strain_pqf) {
            continue;
        }
        
        euler_strain_pqf->operator=(*dynamic_cast<mfem::QuadratureFunction*>(euler_strain_global.get()));
        
        mfem::Vector avg_def_grad(9);
        mfem::Vector region_euler_strain(6);
        
        double region_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
            euler_strain_pqf.get(), avg_def_grad, 9, m_sim_state.getOptions().solvers.rtmodel);
    
        {
            mfem::DenseMatrix euler_strain(3, 3);
            mfem::DenseMatrix def_grad(avg_def_grad.HostReadWrite(), 3, 3);
            int dim = 3;
            mfem::DenseMatrix Finv(dim), Binv(dim);
            double half = 1.0 / 2.0;
    
            mfem::CalcInverse(def_grad, Finv);
            mfem::MultAtB(Finv, Finv, Binv);
         
            euler_strain = 0.0;
         
            for (int j = 0; j < dim; j++) {
               for (int i = 0; i < dim; i++) {
                euler_strain(i, j) -= half * Binv(i, j);
               }
         
               euler_strain(j, j) += half;
            }
    
            region_euler_strain(0) = euler_strain(0, 0);
            region_euler_strain(1) = euler_strain(1, 1);
            region_euler_strain(2) = euler_strain(2, 2);
            region_euler_strain(3) = euler_strain(1, 2);
            region_euler_strain(4) = euler_strain(0, 2);
            region_euler_strain(5) = euler_strain(0, 1);
        }
        
        for (int i = 0; i < 6; ++i) {
            global_avg_euler_strain[i] += region_euler_strain[i] * region_volume;
        }
        global_volume += region_volume;
    }
    
    if (global_volume > 0.0) {
        global_avg_euler_strain /= global_volume;
    }
    
    if (m_mpi_rank == 0) {
        auto filepath = m_file_manager->GetVolumeAverageFilePath("euler_strain", -1);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("euler_strain");
            }
            
            *file << time << " " << global_volume;
            for (int i = 0; i < 6; ++i) {
                *file << " " << global_avg_euler_strain[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::VolumeAvgElasticStrain(const int region, const double time) {
    if ( m_region_model_types[region] != MechType::EXACMECH) {
        return;
    }

    auto elastic_strain_pqf = m_sim_state.GetQuadratureFunction("kinetic_grads", region);    
    if (!elastic_strain_pqf) {
        return;
    }

    auto state_vars = m_sim_state.GetQuadratureFunction("state_var_end", region)->Read();
    const int vdim = m_sim_state.GetQuadratureFunction("state_var_end", region)->GetVDim();
    const int ne = m_sim_state.GetQuadratureFunction("state_var_end", region)->GetSpaceShared()->GetNE();
    const int estrain_ind = m_sim_state.GetQuadratureFunctionStatePair("elastic_strain", region).first;
    const int quats_ind = m_sim_state.GetQuadratureFunctionStatePair("quats", region).first;
    const int rel_vol_ind = m_sim_state.GetQuadratureFunctionStatePair("relative_volume", region).first;
    elastic_strain_pqf->operator=(0.0);
    auto data = elastic_strain_pqf->Write();
    
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int i) {
        const auto strain_lat = &state_vars[i * vdim + estrain_ind];
        const auto quats = &state_vars[i * vdim + quats_ind];
        const auto rel_vol = state_vars[i * vdim + rel_vol_ind];
        double* strain = &data[i * 9];

        {
            double strainm[3 * 3] = {};
            double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
            const double t1 = ecmech::sqr2i * strain_lat[0];
            const double t2 = ecmech::sqr6i * strain_lat[1];
            //
            // Volume strain is ln(V^e_mean) term aka ln(relative volume)
            // Our plastic deformation has a det(1) aka no change in volume change
            const double elas_vol_strain = log(rel_vol);
            // We output elastic strain formulation such that the relationship
            // between V^e and \varepsilon is just V^e = I + \varepsilon
            strain_m[0][0] = (t1 - t2) + elas_vol_strain; // 11
            strain_m[1][1] = (-t1 - t2) + elas_vol_strain ; // 22
            strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
            strain_m[1][2] = ecmech::sqr2i * strain_lat[4]; // 23
            strain_m[2][0] = ecmech::sqr2i * strain_lat[3]; // 31
            strain_m[0][1] = ecmech::sqr2i * strain_lat[2]; // 12

            strain_m[2][1] = strain_m[1][2];
            strain_m[0][2] = strain_m[2][0];
            strain_m[1][0] = strain_m[0][1];

            double rmat[3 * 3] = {};
            double strain_samp[3 * 3] = {};            

            quat2rmat(quats, rmat);
            snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);

            strain_m[0] = &strain_samp[0];
            strain_m[1] = &strain_samp[3];
            strain_m[2] = &strain_samp[6];
            strain[0] = strain_m[0][0];
            strain[1] = strain_m[1][1];
            strain[2] = strain_m[2][2];
            strain[3] = strain_m[1][2];
            strain[4] = strain_m[0][2];
            strain[5] = strain_m[0][1];
            strain[6] = 0.0;
            strain[7] = 0.0;
            strain[8] = 0.0;
        }
    });
    
    mfem::Vector avg_elastic_strain(9); // 3x3 tensor
    
    double total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
        elastic_strain_pqf.get(), avg_elastic_strain, 9, m_sim_state.getOptions().solvers.rtmodel);
    
    if (m_mpi_rank == 0) {
        auto region_name = m_sim_state.GetRegionName(region);
        auto filepath = m_file_manager->GetVolumeAverageFilePath("elastic_strain", region, region_name);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("elastic_strain");
            }
            
            *file << time << " " << total_volume;
            for (int i = 0; i < 6; ++i) {
                *file << " " << avg_elastic_strain[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::GlobalVolumeAvgElasticStrain(const double time) {
    mfem::Vector global_avg_elastic_strain(9);
    global_avg_elastic_strain = 0.0;
    double global_volume = 0.0;
    
    for (int region = 0; region < m_num_regions; ++region) {
        if ( m_region_model_types[region] != MechType::EXACMECH) {
            continue;
        }
    
        auto elastic_strain_pqf = m_sim_state.GetQuadratureFunction("kinetic_grads", region);    
        if (!elastic_strain_pqf) {
            continue;
        }
    
        auto state_vars = m_sim_state.GetQuadratureFunction("state_var_end", region)->Read();
        const int vdim = m_sim_state.GetQuadratureFunction("state_var_end", region)->GetVDim();
        const int ne = m_sim_state.GetQuadratureFunction("state_var_end", region)->GetSpaceShared()->GetNE();
        const int estrain_ind = m_sim_state.GetQuadratureFunctionStatePair("elastic_strain", region).first;
        const int quats_ind = m_sim_state.GetQuadratureFunctionStatePair("quats", region).first;
        const int rel_vol_ind = m_sim_state.GetQuadratureFunctionStatePair("relative_volume", region).first;
    
        elastic_strain_pqf->operator=(0.0);
        auto data = elastic_strain_pqf->Write();
        
        mfem::forall(ne, [=] MFEM_HOST_DEVICE (int i) {
            const auto strain_lat = &state_vars[i * vdim + estrain_ind];
            const auto quats = &state_vars[i * vdim + quats_ind];
            const auto rel_vol = state_vars[i * vdim + rel_vol_ind];
            double* strain = &data[i * 9];
    
            {
                double strainm[3 * 3] = {};
                double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
                const double t1 = ecmech::sqr2i * strain_lat[0];
                const double t2 = ecmech::sqr6i * strain_lat[1];
                //
                // Volume strain is ln(V^e_mean) term aka ln(relative volume)
                // Our plastic deformation has a det(1) aka no change in volume change
                const double elas_vol_strain = log(rel_vol);
                // We output elastic strain formulation such that the relationship
                // between V^e and \varepsilon is just V^e = I + \varepsilon
                strain_m[0][0] = (t1 - t2) + elas_vol_strain; // 11
                strain_m[1][1] = (-t1 - t2) + elas_vol_strain ; // 22
                strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
                strain_m[1][2] = ecmech::sqr2i * strain_lat[4]; // 23
                strain_m[2][0] = ecmech::sqr2i * strain_lat[3]; // 31
                strain_m[0][1] = ecmech::sqr2i * strain_lat[2]; // 12
    
                strain_m[2][1] = strain_m[1][2];
                strain_m[0][2] = strain_m[2][0];
                strain_m[1][0] = strain_m[0][1];
    
                double rmat[3 * 3] = {};
                double strain_samp[3 * 3] = {};            
    
                quat2rmat(quats, rmat);
                snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);
    
                strain_m[0] = &strain_samp[0];
                strain_m[1] = &strain_samp[3];
                strain_m[2] = &strain_samp[6];
                strain[0] = strain_m[0][0];
                strain[1] = strain_m[1][1];
                strain[2] = strain_m[2][2];
                strain[3] = strain_m[1][2];
                strain[4] = strain_m[0][2];
                strain[5] = strain_m[0][1];
                strain[6] = 0.0;
                strain[7] = 0.0;
                strain[8] = 0.0;
            }
        });
        
        mfem::Vector region_elastic_strain(9);
        
        double region_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
            elastic_strain_pqf.get(), region_elastic_strain, 9, m_sim_state.getOptions().solvers.rtmodel);
        
        for (int i = 0; i < 9; ++i) {
            global_avg_elastic_strain[i] += region_elastic_strain[i] * region_volume;
        }
        global_volume += region_volume;
    }
    
    if (global_volume > 0.0) {
        global_avg_elastic_strain /= global_volume;
    }
    
    if (m_mpi_rank == 0) {
        auto filepath = m_file_manager->GetVolumeAverageFilePath("elastic_strain", -1);
        
        bool file_exists = fs::exists(filepath);
        auto file = m_file_manager->CreateOutputFile(filepath, true);
        
        if (file && file->is_open()) {
            if (!file_exists) {
                *file << m_file_manager->GetVolumeAverageHeader("elastic_strain");
            }
            
            *file << time << " " << global_volume;
            for (int i = 0; i < 6; ++i) {
                *file << " " << global_avg_elastic_strain[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::RegisterDefaultProjections()
{
    RegisterProjection("centroid");
    RegisterProjection("volume");
    RegisterProjection("cauchy");
    RegisterProjection("von_mises");
    RegisterProjection("hydro");
    RegisterProjection("dpeff");
    RegisterProjection("xtal_ori");
    RegisterProjection("elastic_strain");
    RegisterProjection("hardness");
    RegisterProjection("shear_rate");
}

void PostProcessingDriver::RegisterDefaultVolumeCalculations() {
    // Register volume average calculations with both per-region and global variants
    // Only register if the corresponding option is enabled in ExaOptions
    
    const auto& vol_opts = m_sim_state.getOptions().post_processing.volume_averages;
    
    if (vol_opts.enabled && vol_opts.stress) {
        RegisterVolumeAverageFunction(
            "stress", "Volume Average Stress",
            [this](int region, double time) { VolumeAvgStress(region, time); },
            [this](double time) { GlobalVolumeAvgStress(time); },
            true
        );
    }
    
    if (vol_opts.enabled && vol_opts.def_grad) {
        RegisterVolumeAverageFunction(
            "def_grad", "Volume Average Deformation Gradient",
            [this](int region, double time) { VolumeAvgDefGrad(region, time); },
            [this](double time) { GlobalVolumeAvgDefGrad(time); },
            true
        );
    }
    
    if (vol_opts.enabled && vol_opts.euler_strain) {
        RegisterVolumeAverageFunction(
            "euler_strain", "Volume Average Euler Strain",
            [this](int region, double time) { VolumeAvgEulerStrain(region, time); },
            [this](double time) { GlobalVolumeAvgEulerStrain(time); },
            true
        );
    }
    
    if (vol_opts.enabled && vol_opts.plastic_work) {
        RegisterVolumeAverageFunction(
            "plastic_work", "Volume Plastic Work",
            [this](int region, double time) { VolumePlWork(region, time); },
            [this](double time) { GlobalVolumePlWork(time); },
            true
        );
    }

    if (vol_opts.enabled && vol_opts.eq_pl_strain) {
        RegisterVolumeAverageFunction(
            "equivalent_plastic_strain", "Volume Equivalent Plastic Strain",
            [this](int region, double time) { VolumeEPS(region, time); },
            [this](double time) { GlobalVolumeEPS(time); },
            true
        );
    }

    if (vol_opts.enabled && vol_opts.elastic_strain) {
        RegisterVolumeAverageFunction(
            "elastic_strain", "Volume Average Elastic Strain",
            [this](int region, double time) { VolumeAvgElasticStrain(region, time); },
            [this](double time) { GlobalVolumeAvgElasticStrain(time); },
            true
        );
    }
}

void PostProcessingDriver::RegisterVolumeAverageFunction(
    const std::string& calc_name,
    const std::string& display_name,
    std::function<void(const int, const double)> region_func,
    std::function<void(const double)> global_func,
    bool enabled
) {
    std::vector<bool> region_enabled(m_num_regions, enabled);
    
    m_registered_volume_calcs.push_back({
        calc_name,
        display_name,
        ProjectionTraits::ModelCompatibility::ALL_MODELS, // Default compatibility
        region_enabled,
        region_func,
        global_func,
        (global_func != nullptr)
    });
}

bool PostProcessingDriver::RegionHasQuadratureFunction(const std::string& field_name, int region) const {
    return m_sim_state.GetQuadratureFunction(field_name, region) != nullptr;
}

std::vector<int> PostProcessingDriver::GetActiveRegionsForField(const std::string& field_name) const {
    std::vector<int> active_regions;

    auto find_lambda = [&](const int region)->bool {
        const auto gf_name = this->GetGridFunctionName(field_name, region);
        return (this->m_map_gfs.find(gf_name) != this->m_map_gfs.end());
    };

    for (int region = 0; region < m_num_regions; ++region) {
        active_regions.push_back(find_lambda(region));
    }
    return active_regions;
}

std::string PostProcessingDriver::GetGridFunctionName(const std::string& field_name, int region) const {
    if (region == -1) {
        return field_name + "_global";
    } else {
        return field_name + "_region_" + std::to_string(region);
    }
}

void PostProcessingDriver::ExecuteGlobalProjection(const std::string& field_name) {
    if (m_num_regions == 1) { return; }
    // Get all active regions for this field
    auto active_regions = GetActiveRegionsForField(field_name);
    if (active_regions.empty()) {
        return;
    }
    // Combine region data into global grid function
    CombineRegionDataToGlobal(field_name);
}

void PostProcessingDriver::CombineRegionDataToGlobal(const std::string& field_name) {
    auto global_gf_name = GetGridFunctionName(field_name, -1); // -1 indicates global
    auto& global_gf = *m_map_gfs[global_gf_name];

    // Initialize global grid function to zero
    global_gf = 0.0;

    // Get active regions for this field
    auto active_regions = GetActiveRegionsForField(field_name);

    int index = 0;
    for (const auto active : active_regions) {
        if (active) {
            auto submesh = std::dynamic_pointer_cast<mfem::ParSubMesh>(m_map_submesh[index]);
            if (submesh) {
                auto gf_name = GetGridFunctionName(field_name, index); // -1 indicates global
                auto& gf = *m_map_gfs[gf_name];
                submesh->Transfer(gf, global_gf);
            }
        }
        index += 1;
    }
}

void PostProcessingDriver::CalcElementAvg(mfem::expt::PartialQuadratureFunction* elemVal, 
                                         const mfem::expt::PartialQuadratureFunction* qf) {
    CALI_CXX_MARK_SCOPE("calc_element_avg_partial");
    
    auto pqs = qf->GetPartialSpaceShared();
    auto mesh = pqs->GetMeshShared();
    const mfem::FiniteElement &el = *m_sim_state.GetMeshParFiniteElementSpace()->GetFE(0);
    const mfem::IntegrationRule *ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    
    const int nqpts = ir->GetNPoints();
    const int vdim = qf->GetVDim();
    const int NE = pqs->GetNE(); // Number of elements in this PARTIAL space (key difference!)
    
    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors *geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);
    
    // KEY DIFFERENCE: Get the local-to-global element mapping for partial space
    auto l2g = pqs->getLocal2Global().Read();           // Maps local element index to global element index
    auto loc_offsets = pqs->getOffsets().Read();        // Offsets for local data layout
    // auto global_offsets = (pqs->getGlobalOffset().Size() > 1) ? 
    //                        pqs->getGlobalOffset().Read() : loc_offsets; // Offsets for global data layout
    
    auto qf_data = qf->Read();        // Partial quadrature function data (only for this region!)
    auto elem_data = elemVal->ReadWrite();  // Element averages output (only for this region!)
    auto j_data = geom->detJ.Read();  // Global geometric factors
    
    // Zero out element averages
    *elemVal = 0.0;
    
    // KEY DIFFERENCE: Process only the elements that exist in this partial space
    // The old version processed ALL elements (0 to nelems-1)
    // The new version processes only local elements (0 to NE-1) and maps to global indices
    mfem::forall(NE, [=] MFEM_HOST_DEVICE (int ie) {
        const int global_elem = l2g[ie];              // Map local element to global element
        const int local_offset = loc_offsets[ie];     // Offset into local data array
        // const int global_offset = global_offsets[global_elem]; // Offset into global layout
        const int npts_elem = loc_offsets[ie + 1] - local_offset; // Number of qpts for this element
        
        double vol = 0.0;
        
        // Calculate volume and weighted sum using actual quadrature points for this element
        for (int iq = 0; iq < npts_elem; ++iq) {
            // Use global element index for geometric factors (j_data)
            const double wt = j_data[global_elem * nqpts + iq] * W[iq];
            vol += wt;
            
            for (int iv = 0; iv < vdim; ++iv) {
                // Use local data layout for quadrature function values
                const int local_idx = local_offset * vdim + iq * vdim + iv;
                const double val = qf_data[local_idx];
                
                // Store in local element index (ie, not global_elem!)
                elem_data[ie * vdim + iv] += val * wt;
            }
        }
        
        // Normalize by volume to get element average
        const double inv_vol = 1.0 / vol;
        for (int iv = 0; iv < vdim; ++iv) {
            elem_data[ie * vdim + iv] *= inv_vol;
        }
    });
}

void PostProcessingDriver::CalcGlobalElementAvg(mfem::Vector* elemVal, 
                                               const std::string& field_name) {
    CALI_CXX_MARK_SCOPE("calc_global_element_avg");
    
    auto fe_space = m_sim_state.GetMeshParFiniteElementSpace();
    const int nelems = fe_space->GetNE();
    
    // Find the vector dimension by checking the first available region
    int vdim = 1;
    for (int region = 0; region < m_num_regions; ++region) {
        if (auto pqf = m_sim_state.GetQuadratureFunction(field_name, region)) {
            if (vdim < pqf->GetVDim()) {
                vdim = pqf->GetVDim();
            }
        }
    }
    
    // Ensure elemVal is sized correctly
    if (elemVal->Size() != vdim * nelems) {
        elemVal->SetSize(vdim * nelems);
        elemVal->UseDevice(true);
    }
    
    // Initialize to zero
    *elemVal = 0.0;
    double* global_data = elemVal->ReadWrite();
    
    // Accumulate from all regions
    for (int region = 0; region < m_num_regions; ++region) {
        auto pqf = m_sim_state.GetQuadratureFunction(field_name, region);
        if (!pqf || !m_region_evec[region]) {
            continue;
        }
        
        // Calculate element averages for this region
        CalcElementAvg(m_region_evec[region].get(), pqf.get());
        
        // Add this region's contribution to global averages
        auto pqs = pqf->GetPartialSpaceShared();
        auto l2g = pqs->getLocal2Global().Read();
        auto region_data = m_region_evec[region]->Read();
        const int NE_region = pqs->GetNE();
        const int local_vdim = pqf->GetVDim();
        
        mfem::forall(NE_region, [=] MFEM_HOST_DEVICE (int ie) {
            const int global_elem = l2g[ie];
            for (int iv = 0; iv < local_vdim; ++iv) {
                global_data[global_elem * vdim + iv] = region_data[ie * local_vdim + iv];
            }
        });
    }
}

void PostProcessingDriver::InitializeGridFunctions() {
    for (auto& reg : m_registered_projections) {
        // Create per-region grid functions
        int max_vdim = 0;
        if (m_aggregation_mode == AggregationMode::PER_REGION || 
            m_aggregation_mode == AggregationMode::BOTH) {
            for (int region = 0; region < m_num_regions; ++region) {
                if (reg.region_enabled[region]) {
                    const auto gf_name = GetGridFunctionName(reg.field_name, region);
                    // Determine vector dimension from quadrature function
                    const int vdim = reg.region_length[region];
                    max_vdim = (vdim > max_vdim) ? vdim : max_vdim;
                    auto fe_space = GetParFiniteElementSpace(region, vdim);
                    m_map_gfs.emplace(gf_name, std::make_shared<mfem::ParGridFunction>(
                        fe_space.get()));
                    m_map_gfs[gf_name]->operator=(0.0);
                }
            }
        }
        // Create global grid functions
        if (reg.supports_global_aggregation && 
            (m_aggregation_mode == AggregationMode::GLOBAL_COMBINED || 
             m_aggregation_mode == AggregationMode::BOTH) && (m_num_regions > 1)) {

            if (max_vdim < 1) {
                for (int region = 0; region < m_num_regions; ++region) {
                    if (reg.region_enabled[region]) {
                        const auto gf_name = GetGridFunctionName(reg.field_name, region);
                        // Determine vector dimension from quadrature function
                        const int vdim = reg.region_length[region];
                        max_vdim = (vdim > max_vdim) ? vdim : max_vdim;
                    }
                }
            }

            auto gf_name = GetGridFunctionName(reg.field_name, -1);
            auto fe_space = m_sim_state.GetParFiniteElementSpace(max_vdim);
            m_map_gfs.emplace(gf_name, std::make_shared<mfem::ParGridFunction>(
                fe_space.get()));
            m_map_gfs[gf_name]->operator=(0.0);
        }
    }

    UpdateFields(m_sim_state.getSimulationCycle(), m_sim_state.getTime());
}

void PostProcessingDriver::InitializeDataCollections(ExaOptions& options) {
    auto output_dir_base = m_file_manager->GetVizDirectory();
    std::string visit_key = "visit_";
    std::string paraview_key = "paraview_";
#if defined(MFEM_USE_ADIOS2)
    std::string adios2_key = "adios2_";
#endif

    if (m_aggregation_mode == AggregationMode::PER_REGION || 
        m_aggregation_mode == AggregationMode::BOTH) {
        for (int region = 0; region < m_num_regions; ++region) {
            auto mesh = m_map_submesh[region];
            std::string region_postfix = "region_" + std::to_string(region);
            std::string output_dir = output_dir_base + region_postfix + "/" + m_file_manager->GetBaseFilename();
            m_file_manager->EnsureDirectoryExists(output_dir);
            std::vector<std::string> dcs_keys; 
            if (options.visualization.visit) {
                std::string key = visit_key + region_postfix;
                m_map_dcs.emplace(key, std::make_unique<mfem::VisItDataCollection>(output_dir, mesh.get()));
                m_map_dcs[key]->SetPrecision(10);
                dcs_keys.push_back(key);
            }
            if (options.visualization.paraview) {
                std::string key = paraview_key + region_postfix;
                m_map_dcs.emplace(key, std::make_unique<mfem::ParaViewDataCollection>(output_dir, mesh.get()));
                auto& paraview = *(dynamic_cast<mfem::ParaViewDataCollection*>(m_map_dcs[key].get()));
                paraview.SetLevelsOfDetail(options.mesh.order);
                paraview.SetDataFormat(mfem::VTKFormat::BINARY);
                paraview.SetHighOrderOutput(false);
                dcs_keys.push_back(key);
            }
#ifdef MFEM_USE_ADIOS2
            if (options.visualization.adios2) {
                const std::string basename = output_dir + ".bp";
                std::string key = adios2_key + region_postfix;
                m_map_dcs.emplace(key, std::make_unique<mfem::ADIOS2DataCollection>(MPI_COMM_WORLD, basename, mesh.get()));
                auto& adios2 = *(dynamic_cast<mfem::ADIOS2DataCollection*>(m_map_dcs[key].get()));
                adios2.SetParameter("SubStreams", std::to_string(m_num_mpi_rank / 2));
                dcs_keys.push_back(key);
            }
#endif
            for (auto& dcs_key : dcs_keys) {
                auto& dcs = m_map_dcs[dcs_key];
                for (auto& [key, value] : m_map_gfs) {
                    if (key.find(region_postfix) != std::string::npos) {
                        dcs->RegisterField(key, value.get());
                    }
                }
                dcs->SetCycle(0);
                dcs->SetTime(0.0);
                dcs->Save();
            }
        }
    }

    if ((m_aggregation_mode == AggregationMode::GLOBAL_COMBINED || 
        m_aggregation_mode == AggregationMode::BOTH) &&
        (m_num_regions > 1)) {

        auto mesh = m_sim_state.getMesh();

        std::string region_postfix = "global";
        std::string output_dir = output_dir_base + region_postfix + "/" + m_file_manager->GetBaseFilename();
        m_file_manager->EnsureDirectoryExists(output_dir);
        std::vector<std::string> dcs_keys; 
        if (options.visualization.visit) {
            std::string key = visit_key + region_postfix;
            m_map_dcs.emplace(key, std::make_unique<mfem::VisItDataCollection>(output_dir, mesh.get()));
            m_map_dcs[key]->SetPrecision(10);
            dcs_keys.push_back(key);
        }
        if (options.visualization.paraview) {
            std::string key = paraview_key + region_postfix;
            m_map_dcs.emplace(key, std::make_unique<mfem::ParaViewDataCollection>(output_dir, mesh.get()));
            auto& paraview = *(dynamic_cast<mfem::ParaViewDataCollection*>(m_map_dcs[key].get()));
            paraview.SetLevelsOfDetail(options.mesh.order);
            paraview.SetDataFormat(mfem::VTKFormat::BINARY);
            paraview.SetHighOrderOutput(false);
            dcs_keys.push_back(key);
        }
#ifdef MFEM_USE_ADIOS2
        if (options.visualization.adios2) {
            const std::string basename = output_dir + ".bp";
            std::string key = adios2_key + region_postfix;
            m_map_dcs.emplace(key, std::make_unique<mfem::ADIOS2DataCollection>(MPI_COMM_WORLD, basename, mesh.get()));
            auto& adios2 = *(dynamic_cast<mfem::ADIOS2DataCollection*>(m_map_dcs[key].get()));
            adios2.SetParameter("SubStreams", std::to_string(m_num_mpi_rank / 2));
            dcs_keys.push_back(key);
        }
#endif

        for (auto& dcs_key : dcs_keys) {
            auto& dcs = m_map_dcs[dcs_key];
            for (auto& [key, value] : m_map_gfs) {
                if (key.find(region_postfix) != std::string::npos) {
                    dcs->RegisterField(key, value.get());
                }
            }
            dcs->SetCycle(0);
            dcs->SetTime(0.0);
            dcs->Save();
        }

    }
}

void PostProcessingDriver::UpdateDataCollections(const int step, const double time) {
    for (auto& [tmp, dcs] : m_map_dcs) {
        dcs->SetCycle(step);
        dcs->SetTime(time);
        dcs->Save();
    }
}

void PostProcessingDriver::InitializeLightUpAnalysis() {
    auto options = m_sim_state.getOptions();
    // Clear any existing instances
    light_up_instances.clear();
    
    // Get enabled light_up configurations
    auto enabled_configs = options.post_processing.get_enabled_light_up_configs();
    
    if (!enabled_configs.empty()) {
        std::cout << "Initializing LightUp analysis for " << enabled_configs.size() 
                  << " material(s)" << std::endl;
    }
    
    // Create LightUp instance for each enabled configuration
    for (const auto& light_config : enabled_configs) {
        if (!light_config.region_id.has_value()) {
            std::cerr << "Error: LightUp config for material '" << light_config.material_name 
                      << "' has unresolved region_id" << std::endl;
            continue;
        }
        
        int region_id = light_config.region_id.value();
        
        std::cout << "  Creating LightUp for material '" << light_config.material_name 
                  << "' (region " << region_id << ")" << std::endl;

        std::string lattice_basename = m_file_manager->GetOutputDirectory() + light_config.lattice_basename;
        
        auto light_up_instance = std::make_unique<LightUpCubic>(
                                    light_config.hkl_directions,
                                    light_config.distance_tolerance,
                                    light_config.sample_direction,
                                    m_sim_state.GetMeshParFiniteElementSpace().get(),
                                    m_sim_state.GetQuadratureFunction("cauchy_stress_end", region_id)->GetPartialSpaceShared(),
                                    m_sim_state,
                                    region_id,  // Use the resolved region_id
                                    options.solvers.rtmodel,
                                    lattice_basename,
                                    light_config.lattice_parameters
                                );
        
        light_up_instances.push_back(std::move(light_up_instance));
    }
}

void PostProcessingDriver::UpdateLightUpAnalysis()
{
   // Update all LightUp instances
   for (auto& light_up : light_up_instances) {
      const int region_id = light_up->get_region_id();

      auto state_vars = m_sim_state.GetQuadratureFunction("state_var_end", region_id);
      auto stress = m_sim_state.GetQuadratureFunction("cauchy_stress_end", region_id);

      light_up->calculate_lightup_data(state_vars, stress);
   }
}

void PostProcessingDriver::EnableProjection(const std::string& field_name, int region, bool enable) {
    for (auto& reg : m_registered_projections) {
        if (reg.field_name == field_name && region < static_cast<int>(reg.region_enabled.size())) {
            reg.region_enabled[region] = enable;
        }
    }
}

void PostProcessingDriver::EnableProjection(const std::string& field_name, bool enable) {
    for (auto& reg : m_registered_projections) {
        if (reg.field_name == field_name) {
            std::fill(reg.region_enabled.begin(), reg.region_enabled.end(), enable);
        }
    }
}

void PostProcessingDriver::EnableAllProjections() {
    for (auto& reg : m_registered_projections) {
        for (int region = 0; region < m_num_regions; ++region) {
            // Check compatibility with region's model type
            bool compatible = true;
            if (reg.model_compatibility == ProjectionTraits::ModelCompatibility::EXACMECH_ONLY &&
                m_region_model_types[region] != MechType::EXACMECH) {
                compatible = false;
            }
            if (reg.model_compatibility == ProjectionTraits::ModelCompatibility::UMAT_ONLY &&
                m_region_model_types[region] != MechType::UMAT) {
                compatible = false;
            }

            // Only enable if compatible and has required data
            if (compatible) {
                reg.region_enabled[region] = true;
            }
        }
    }
}

std::vector<std::pair<std::string, std::string>> PostProcessingDriver::GetAvailableProjections() const {
    std::vector<std::pair<std::string, std::string>> available;
    for (const auto& reg : m_registered_projections) {
        available.emplace_back(reg.field_name, reg.display_name);
    }
    return available;
}

size_t PostProcessingDriver::GetQuadratureFunctionSize() const {
    // Return size based on one of the region quadrature functions
    for (int region = 0; region < m_num_regions; ++region) {
        if (auto pqf = m_sim_state.GetQuadratureFunction("cauchy_stress_end", region)) {
            return pqf->GetSpaceShared()->GetSize();
        }
    }
    return 0;
}