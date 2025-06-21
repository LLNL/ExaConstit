#include "postprocessing_driver.hpp"
#include "postprocessing_file_manager.hpp"
#include "mechanics_kernels.hpp"
#include "mechanics_log.hpp"

#include <filesystem>
namespace fs = std::filesystem;

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
    
    // Get model types for each region
    for (int region = 0; region < m_num_regions; ++region) {
        m_region_model_types[region] = sim_state.GetRegionModelType(region);
        
        // Initialize region-specific element average buffer
        if (auto pqf = sim_state.GetQuadratureFunction("cauchy_stress_end", region)) {
            int max_vdim = 0;
            // Find maximum vdim across all possible quadrature functions for this region
            for (const auto& field_name : {"cauchy_stress_end", "state_var_end", "von_mises"}) {
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
    int global_max_vdim = 9; // Accommodate stress tensors and other multi-component fields
    m_global_evec = std::make_unique<mfem::Vector>(global_max_vdim * fe_space->GetNE());
    m_global_evec->UseDevice(true);
    
    // Register default projections and volume calculations
    RegisterDefaultVolumeCalculations();
    
    // Initialize grid functions and data collections
    if (enable_visualization) {
        InitializeGridFunctions();
        RegisterDefaultProjections();
        InitializeDataCollections(options);
    }
}

void PostProcessingDriver::Update(const int step, const double time) {
    CALI_CXX_MARK_SCOPE("postprocessing_update");
    
    // Execute projections based on aggregation mode
    if (m_aggregation_mode == AggregationMode::PER_REGION || 
        m_aggregation_mode == AggregationMode::BOTH) {
        
        // Process each region separately
        for (int region = 0; region < m_num_regions; ++region) {
            for (auto& reg : m_registered_projections) {
                if (reg.region_enabled[region]) {
                    reg.projection_func(region);
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
    
    // Check if we should output volume averages at this step
    if (ShouldOutputAtStep(step)) {
        PrintVolValues(time, m_aggregation_mode);
    }
    
    // Update data collections for visualization
    if (enable_visualization) {
        UpdateDataCollections(step, time);
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
            return;
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
    auto pl_work_pqf = m_sim_state.GetQuadratureFunction("plastic_work", region);
    if (!pl_work_pqf) {
        return;
    }
    
    // Calculate volume-averaged plastic work for this region
    mfem::Vector avg_pl_work(1); // Scalar quantity
    
    double total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
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
        auto pl_work_pqf = m_sim_state.GetQuadratureFunction("plastic_work", region);
        if (!pl_work_pqf) {
            continue;
        }
        
        mfem::Vector region_pl_work(1);
        
        double region_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
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
            return;
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
    auto elastic_strain_pqf = m_sim_state.GetQuadratureFunction("elastic_strain_end", region);
    if (!elastic_strain_pqf) {
        return;
    }
    
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
            for (int i = 0; i < 9; ++i) {
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
        auto elastic_strain_pqf = m_sim_state.GetQuadratureFunction("elastic_strain_end", region);
        if (!elastic_strain_pqf) {
            continue;
        }
        
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
            for (int i = 0; i < 9; ++i) {
                *file << " " << global_avg_elastic_strain[i];
            }
            *file << "\n" << std::flush;
        }
    }
}

void PostProcessingDriver::RegisterDefaultProjections() {
    // Register standard projections with multi-material support
    
    // Stress-related projections (available for all material types)
    RegisterSimpleProjection<ProjectionTraits::ModelStressProjection>(
        "cauchy_stress_end", "Model Stress", true, true);
    
    RegisterSpecialProjection<ProjectionTraits::VonMisesProjection>(
        "cauchy_stress_end", "von_mises", "Von Mises Stress", true, true);
    
    RegisterSpecialProjection<ProjectionTraits::HydroStressProjection>(
        "cauchy_stress_end", "hydro_stress", "Hydrostatic Stress", true, true);
    
    // Geometry projections (always available)
    RegisterGeometryProjection<ProjectionTraits::CentroidProjection>(
        "centroid", "Element Centroid", true, true);
    
    RegisterGeometryProjection<ProjectionTraits::VolumeProjection>(
        "volume", "Element Volume", true, true);
    
    // ExaCMech-specific projections
    RegisterSimpleProjection<ProjectionTraits::DpEffProjection>(
        "dp_eff", "Effective Plastic Strain Rate", false, true);
    
    RegisterSimpleProjection<ProjectionTraits::EffPlasticStrainProjection>(
        "eff_plastic_strain", "Effective Plastic Strain", false, true);
    
    RegisterSimpleProjection<ProjectionTraits::ShearRateProjection>(
        "shear_rate", "Shear Rate", false, true);
    
    RegisterSimpleProjection<ProjectionTraits::OrientationProjection>(
        "orientation", "Crystal Orientation", false, false); // Orientations don't aggregate well
    
    RegisterSimpleProjection<ProjectionTraits::HProjection>(
        "hardness", "Hardness Parameter", false, true);
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
    for (int region = 0; region < m_num_regions; ++region) {
        if (RegionHasQuadratureFunction(field_name, region)) {
            active_regions.push_back(region);
        }
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
    
    // Calculate global element averages from all regions
    CalcGlobalElementAvg(m_global_evec.get(), field_name);
    
    // Project to global grid function
    // Note: This assumes compatible vector dimensions across regions
    auto fe_space = m_sim_state.GetMeshParFiniteElementSpace();
    const int vdim = global_gf.VectorDim();
    
    // Create a temporary quadrature function for the global data
    auto mesh = fe_space->GetMesh();
    const mfem::FiniteElement &el = *fe_space->GetFE(0);
    const mfem::IntegrationRule *ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    auto temp_qspace = std::make_shared<mfem::QuadratureSpace>(mesh, *ir);
    mfem::QuadratureFunction temp_qf(temp_qspace, vdim);
    
    // Convert element averages back to quadrature function format
    // This is a simplified approach - in practice you might want more sophisticated interpolation
    const int nqpts = ir->GetNPoints();
    const int nelems = fe_space->GetNE();
    
    double* qf_data = temp_qf.ReadWrite();
    const double* elem_data = m_global_evec->Read();
    
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
        for (int iq = 0; iq < nqpts; ++iq) {
            for (int iv = 0; iv < vdim; ++iv) {
                qf_data[ie * nqpts * vdim + iq * vdim + iv] = elem_data[ie * vdim + iv];
            }
        }
    });
    
    // Project to grid function
    mfem::VectorQuadratureFunctionCoefficient qfvc(temp_qf);
    global_gf.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
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
            vdim = pqf->GetVDim();
            break;
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
        
        mfem::forall(NE_region, [=] MFEM_HOST_DEVICE (int ie) {
            const int global_elem = l2g[ie];
            for (int iv = 0; iv < vdim; ++iv) {
                global_data[global_elem * vdim + iv] = region_data[ie * vdim + iv];
            }
        });
    }
}

void PostProcessingDriver::InitializeGridFunctions() {    
    for (auto& reg : m_registered_projections) {
        // Create per-region grid functions
        if (m_aggregation_mode == AggregationMode::PER_REGION || 
            m_aggregation_mode == AggregationMode::BOTH) {
            
            for (int region = 0; region < m_num_regions; ++region) {
                if (RegionHasQuadratureFunction(reg.field_name, region)) {
                    auto gf_name = GetGridFunctionName(reg.field_name, region);
                    
                    // Determine vector dimension from quadrature function
                    auto pqf = m_sim_state.GetQuadratureFunction(reg.field_name, region);
                    int vdim = pqf ? pqf->GetVDim() : 1;
                    auto fe_space = m_sim_state.GetParFiniteElementSpace(vdim);

                    m_map_gfs[gf_name] = std::make_unique<mfem::ParGridFunction>(
                        fe_space.get());
                }
            }
        }
        
        // Create global grid functions
        if (reg.supports_global_aggregation && 
            (m_aggregation_mode == AggregationMode::GLOBAL_COMBINED || 
             m_aggregation_mode == AggregationMode::BOTH)) {
            
            auto gf_name = GetGridFunctionName(reg.field_name, -1);
            
            // Find vdim from any active region
            int vdim = 1;
            for (int region = 0; region < m_num_regions; ++region) {
                if (auto pqf = m_sim_state.GetQuadratureFunction(reg.field_name, region)) {
                    vdim = pqf->GetVDim();
                    break;
                }
            }

            // Determine vector dimension from quadrature function
            auto fe_space = m_sim_state.GetParFiniteElementSpace(vdim);
            m_map_gfs[gf_name] = std::make_unique<mfem::ParGridFunction>(
                fe_space.get());
        }
    }
}

void PostProcessingDriver::InitializeDataCollections([[maybe_unused]] ExaOptions& options) {
    // Initialize data collections for visualization
    // Implementation would depend on specific visualization needs
    // For now, this is a placeholder
}

void PostProcessingDriver::UpdateDataCollections([[maybe_unused]] const int step, [[maybe_unused]] const double time) {
    // Update data collections with current grid function data
    // Implementation would depend on specific visualization needs
    // For now, this is a placeholder
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
            if (compatible && RegionHasQuadratureFunction(reg.field_name, region)) {
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

// Placeholder implementations for projection methods
void PostProcessingDriver::ProjectCentroid([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectVolume([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectModelStress([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectVonMisesStress([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectHydroStress([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectDpEff([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectEffPlasticStrain([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectShearRate([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectOrientation([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectH([[maybe_unused]] const int region) {}
void PostProcessingDriver::ProjectElasticStrains([[maybe_unused]] const int region) {}
void PostProcessingDriver::ExecuteElasticStrainProjection([[maybe_unused]] const std::string& strain_field, [[maybe_unused]] const std::string& vol_field, [[maybe_unused]]  int region) {}