#include "postprocessing_driver.hpp"
#include "mechanics_kernels.hpp"
#include "sim_state/simulation_state.hpp"

PostProcessingDriver::PostProcessingDriver(SimulationState& sim_state, ExaOptions& options)
    : m_sim_state(sim_state),
      m_mpi_rank(0),
      m_avg_filepath_base(options.basename + "/")
{
    MPI_Comm_rank(MPI_COMM_WORLD, &m_mpi_rank);
    
    // Simplify visualization check
    enable_visualization = 
        options.visualization.visit || options.visualization.conduit || options.visualization.paraview || options.visualization.adios2;
    
    // Initialize region model types
    m_region_mech_types.resize(m_sim_state.GetNumberOfRegions());
    for (int region = 0; region < m_sim_state.GetNumberOfRegions(); region++) {
        m_region_mech_types[region] = m_sim_state.GetRegionModelType(region);
    }
    
    // Initialize m_evec for element averaging if visualization is enabled
    if (enable_visualization) {
        auto qf_size = GetQuadratureFunctionSize();
        m_evec = std::make_unique<mfem::expt::PartialQuadratureFunction>(m_sim_state.getGlobalVizQuadSpace(), qf_size);
        m_evec->UseDevice(true);
    }
    
    // Register standard projections for all model types
    RegisterSimpleProjection<ProjectionTraits::ProjectionTrait<void>>(
        "displacement", "Displacement", enable_visualization);
    
    RegisterSimpleProjection<ProjectionTraits::ProjectionTrait<void>>(
        "velocity", "Velocity", enable_visualization);
    
    // Use GeometryProjection for Centroid and Volume since they work directly with the mesh
    RegisterGeometryProjection<ProjectionTraits::CentroidTrait>(
        "centroid", "Element Centroid", enable_visualization);
    
    RegisterGeometryProjection<ProjectionTraits::VolumeTrait>(
        "volume", "Element Volume", enable_visualization);
    
    RegisterSimpleProjection<ProjectionTraits::ProjectionTrait<void>>(
        "cauchy_stress_end", "Stress", enable_visualization);
    
    RegisterSpecialProjection<ProjectionTraits::VonMisesStressTrait>(
        "cauchy_stress_end", "von_mises_stress", "Von Mises Stress", enable_visualization);
    
    RegisterSpecialProjection<ProjectionTraits::HydroStressTrait>(
        "cauchy_stress_end", "hydrostatic_stress", "Hydrostatic Stress", enable_visualization);
    
    // Register region-specific projections based on model type
    for (int region = 0; region < m_sim_state.GetNumberOfRegions(); region++) {
        if (m_region_mech_types[region] == MechType::EXACMECH) {
            // ExaCMech-specific projections
            // Note: These will only be enabled for this specific region
            RegisterSimpleProjection<ProjectionTraits::DpEffTrait>(
                "effective_plastic_deformation_rate", "Effective Plastic Rate", enable_visualization);
            
            RegisterSimpleProjection<ProjectionTraits::EffPlasticStrainTrait>(
                "effective_plastic_deformation", "Effective Plastic Strain", enable_visualization);
            
            RegisterSimpleProjection<ProjectionTraits::ShearRateTrait>(
                "plastic_shearing_rate", "Shear Rate", enable_visualization);
            
            RegisterSimpleProjection<ProjectionTraits::OrientationTrait>(
                "lattice_orientation", "Lattice Orientation", enable_visualization);
            
            RegisterSimpleProjection<ProjectionTraits::HardnessTrait>(
                "hardness", "Hardness", enable_visualization);
            
            RegisterElasticStrainProjection(
                "lattice_elastic_strain", "relative_volume", "Elastic Strain", enable_visualization);
        }
    }

    // Register volume averaging methods
    RegisterVolumeAverageFunction(
        "avg_stress", "Average Stress",
        [this](const int region, const double time) { 
            this->VolumeAvgStress(region, time); 
        },
        true);

    // Register volume average calculations if requested
    if (options.post_processing.volume_averages.additional_avgs) {
            
        RegisterVolumeAverageFunction(
            "avg_euler_strain", "Average Euler Strain",
            [this](const int region, const double time) { 
                this->VolumeAvgEulerStrain(region, time); 
            },
            true);
            
        RegisterVolumeAverageFunction(
            "avg_def_grad", "Average Deformation Gradient",
            [this](const int region, const double time) { 
                this->VolumeAvgDefGrad(region, time); 
            },
            true);
            
        // ExaCMech-specific volume averages
        RegisterVolumeAverageFunction(
            "avg_pl_work", "Average Plastic Work",
            [this](const int region, const double time) { 
                this->VolumePlWork(region, time); 
            }, 
            true);
            
        RegisterVolumeAverageFunction(
            "avg_elastic_strain", "Average Elastic Strain",
            [this](const int region, const double time) { 
                this->VolumeAvgElasticStrain(region, time); 
            },
            true);
    }
    
    // Initialize visualization data collections
    InitializeDataCollections(options);
}

void PostProcessingDriver::RegisterVolumeAverageFunction(
    const std::string& name,
    const std::string& display_name,
    std::function<void(const int, const double)> avg_function,
    bool enabled
) {
    std::cout << "name: " << name << std::endl;
    std::cout << "display_name: " << display_name << std::endl;
    std::cout << "enabled: " << enabled << std::endl;

    m_map_avg_fcns[name] = avg_function;
    m_map_avg_names[name] = display_name;
    m_map_avg_enabled[name] = enabled;
}

void PostProcessingDriver::RegisterElasticStrainProjection(
    const std::string& strain_field,
    const std::string& vol_field,
    const std::string& display_name,
    bool default_enabled
) {
    // ExaCMech compatibility check
    auto compatibility = ProjectionTraits::ModelCompatibility::EXACMECH_ONLY;
    
    auto projection_func = [this, strain_field, vol_field](int region) {
        // Skip if incompatible with this region's model
        if (m_region_mech_types[region] != MechType::EXACMECH) {
            return;
        }
        
        this->ExecuteElasticStrainProjection(strain_field, vol_field, region);
    };
    
    // Initialize per-region enabled flags
    std::vector<bool> region_enabled(m_sim_state.GetNumberOfRegions(), default_enabled);
    
    // Register the projection
    m_registered_projections.push_back({
        strain_field,
        display_name,
        compatibility,
        region_enabled,
        projection_func
    });
}

void PostProcessingDriver::ExecuteElasticStrainProjection(
    const std::string& strain_field,
    const std::string& vol_field,
    int region
) {
    auto strain_name = m_sim_state.GetQuadratureFunctionMapName(strain_field, region);
    auto vol_name = m_sim_state.GetQuadratureFunctionMapName(vol_field, region);
    
    // Get component info
    auto strain_pair = m_sim_state.GetQuadratureFunctionStatePair(strain_name, region);
    auto vol_pair = m_sim_state.GetQuadratureFunctionStatePair(vol_name, region);
    
    // Get grid function
    auto& estrain = *m_map_gfs[strain_name];
    
    // Execute specialized projection
    ProjectionTraits::ElasticStrainTrait::PostProcess(
        estrain, *m_evec, strain_pair, vol_pair);
}

void PostProcessingDriver::Update(const int step, const double time) {
    PrintVolValues(time);
    if (enable_visualization) {
        UpdateDataCollections(step, time);
    }
}

void PostProcessingDriver::PrintVolValues(const double time) {    
    // Execute all enabled volume average calculations
    for (auto& [name, func] : m_map_avg_fcns) {
        std::cout << "name_func: " << name << std::endl;
        // Skip disabled calculations
        if (!m_map_avg_enabled[name]) {
            continue;
        }

        func(-1, time);

        /*
        // Execute volume average calculation for each region
        for (int region = 0; region < m_sim_state.GetNumberOfRegions(); region++) {
            std::cout << "region: " << region << std::endl;
            // ExaCMech-specific check
            if ((name.find("pl_work") != std::string::npos || 
                 name.find("elastic_strain") != std::string::npos) &&
                m_region_mech_types[region] != MechType::EXACMECH) {
                continue;
            }
            
            // Call the volume average function
            func(region, time);
        }
        */
    }
}

void PostProcessingDriver::UpdateDataCollections(const int step, const double time) {    
    // Only calculate element averages if we have registered projections
    if (!m_registered_projections.empty()) {
        const auto mat_vars0 = m_sim_state.GetQuadratureFunction("state_var_beg", 0);
        std::cout << "evec_ptr" << m_evec.get() << std::endl;
        std::cout << "mat_vars0_ptr" << mat_vars0.get() << std::endl;
        CalcElementAvg(m_evec.get(), mat_vars0.get());
    }
    
    // Execute only user-requested projections for each region
    for (int region = 0; region < m_sim_state.GetNumberOfRegions(); region++) {
        for (const auto& reg : m_registered_projections) {
            // Skip if not requested by user for this region
            if (region >= (int) reg.user_requested.size() || !reg.user_requested[region]) {
                continue;
            }
            
            // Skip if incompatible with this region's model type
            auto compatibility = reg.model_compatibility;
            if ((compatibility == ProjectionTraits::ModelCompatibility::EXACMECH_ONLY && 
                m_region_mech_types[region] != MechType::EXACMECH) ||
                (compatibility == ProjectionTraits::ModelCompatibility::UMAT_ONLY && 
                m_region_mech_types[region] != MechType::UMAT)) {
                continue;
            }
            
            // Execute the projection
            reg.projection_function(region);
        }
    }
    
    // Update all data collections
    for (auto& [name, dc] : m_map_dcs) {
        dc->SetCycle(step);
        dc->SetTime(time);
        dc->Save();
    }
}

void PostProcessingDriver::EnableProjection(const std::string& field_name, int region, bool enable) {
    for (auto& reg : m_registered_projections) {
        if (reg.field_name == field_name && region < (int) reg.user_requested.size()) {
            reg.user_requested[region] = enable;
            break;
        }
    }
}

void PostProcessingDriver::EnableProjection(const std::string& field_name, bool enable) {
    for (auto& reg : m_registered_projections) {
        if (reg.field_name == field_name) {
            std::fill(reg.user_requested.begin(), reg.user_requested.end(), enable);
            break;
        }
    }
}

void PostProcessingDriver::EnableAllProjections() {
    for (auto& reg : m_registered_projections) {
        std::fill(reg.user_requested.begin(), reg.user_requested.end(), true);
    }
}

std::vector<std::pair<std::string, std::string>> PostProcessingDriver::GetAvailableProjections() const {
    std::vector<std::pair<std::string, std::string>> result;
    for (const auto& reg : m_registered_projections) {
        result.push_back({reg.field_name, reg.display_name});
    }
    return result;
}

void PostProcessingDriver::CalcElementAvg(mfem::expt::PartialQuadratureFunction* elemVal, const mfem::expt::PartialQuadratureFunction* qf) {
    ProjectionTraits::ProjectionTrait<void>::CalcElementAvg(*elemVal, *qf, m_sim_state.GetMeshParFiniteElementSpace().get());
}

size_t PostProcessingDriver::GetQuadratureFunctionSize() const {

    // Determine size based on quad function vdim and number of elements
    int max_nelems = m_sim_state.getMesh()->GetNE();
    int max_vdims = -1;

    for (int region = 0; region < m_sim_state.GetNumberOfRegions(); region++) {
        const auto mat_vars0 = m_sim_state.GetQuadratureFunction("state_var_beg", region);        
        const int vdim = mat_vars0->GetVDim();
        max_vdims = (vdim > max_vdims) ? vdim : max_vdims;
    }

    return static_cast<size_t>(max_nelems * max_vdims);
}

void PostProcessingDriver::InitializeDataCollections(ExaOptions& options) {
    // Create appropriate data collections based on options
    // This implementation depends on the details of ExaOptions
    // and would include code like:
    
    if (options.visualization.visit) {
        auto visit_dc = std::make_unique<mfem::VisItDataCollection>(options.basename, m_sim_state.getMesh().get());
        visit_dc->SetPrecision(12);
        m_map_dcs["visit"] = std::move(visit_dc);
    }
    
    if (options.visualization.paraview) {
        auto paraview_dc = std::make_unique<mfem::ParaViewDataCollection>(options.basename, m_sim_state.getMesh().get());
        paraview_dc->SetLevelsOfDetail(options.mesh.order);
        paraview_dc->SetDataFormat(mfem::VTKFormat::BINARY);
        paraview_dc->SetHighOrderOutput(false);
        m_map_dcs["paraview"] = std::move(paraview_dc);
    }
    
    // Similar for Conduit and ADIOS2
    
    // Register fields with data collections
    for (auto& [name, dc] : m_map_dcs) {
        for (const auto& reg : m_registered_projections) {
            if (std::any_of(reg.user_requested.begin(), reg.user_requested.end(), 
                           [](bool b) { return b; })) {
                for (int region = 0; region < m_sim_state.GetNumberOfRegions(); region++) {
                    if (region < (int) reg.user_requested.size() && reg.user_requested[region]) {
                        auto field_name = m_sim_state.GetQuadratureFunctionMapName(reg.field_name, region);
                        dc->RegisterField(field_name, m_map_gfs[field_name].get());
                    }
                }
            }
        }
    }
}

// Volume average calculation methods
void PostProcessingDriver::VolumeAvgStress(const int region, [[maybe_unused]] const double time) {    
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_beg", region);
    const auto qf_val = m_sim_state.GetQuadratureFunction(qf_name);

    // Calculate volume average
    mfem::Vector vol_avg_quant(6);
    vol_avg_quant = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(
        m_sim_state.GetMeshParFiniteElementSpace().get(), 
        qf_val.get(), 
        vol_avg_quant, 
        6, 
        m_sim_state.class_device);

    // Only rank 0 writes to file
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_" + 
            m_sim_state.GetQuadratureFunctionMapName("cauchy_stress", region) + ".txt";
        file.open(file_name, std::ios_base::app);
        
//        file.setf(std::ios::scientific);
        file.setf(std::ios::showpoint);
        file.precision(8);
        
        vol_avg_quant.Print(file, 6);
    }
}

void PostProcessingDriver::VolumeAvgDefGrad(const int region, [[maybe_unused]] const double time) {    
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("kinetic_grads", -1);
    const auto qf_val = m_sim_state.GetQuadratureFunction("kinetic_grads", -1);
    
    mfem::Vector vol_avg_dgrad(qf_val->GetVDim());
    vol_avg_dgrad = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(
        m_sim_state.GetMeshParFiniteElementSpace().get(), 
        qf_val.get(), 
        vol_avg_dgrad, 
        vol_avg_dgrad.Size(), 
        m_sim_state.class_device);
    
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_" + 
            m_sim_state.GetQuadratureFunctionMapName("def_grad", region) + ".txt";
        file.open(file_name, std::ios_base::app);
        
//        file.setf(std::ios::scientific);
        file.setf(std::ios::showpoint);
        file.precision(8);
        
        vol_avg_dgrad.Print(file, vol_avg_dgrad.Size());
    }
}

void PostProcessingDriver::VolumeAvgEulerStrain(const int region, [[maybe_unused]] const double time) {    
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("kinetic_grads", -1);
    const auto qf_val = m_sim_state.GetQuadratureFunction("kinetic_grads", -1);
    
    mfem::Vector vol_avg_dgrad(qf_val->GetVDim());
    vol_avg_dgrad = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(
        m_sim_state.GetMeshParFiniteElementSpace().get(), 
        qf_val.get(), 
        vol_avg_dgrad, 
        vol_avg_dgrad.Size(), 
        m_sim_state.class_device);
    
    // Eulerian strain calculation
    mfem::DenseMatrix estrain(3, 3);
    {
        mfem::DenseMatrix def_grad(vol_avg_dgrad.HostReadWrite(), 3, 3);
        
        // Calculate Eulerian strain: e = 1/2(I - F^(-t)F^(-1))
        const int dim = 3;
        mfem::DenseMatrix Finv(dim), Binv(dim);
        const double half = 1.0 / 2.0;
        
        mfem::CalcInverse(def_grad, Finv);
        mfem::MultAtB(Finv, Finv, Binv);
        
        estrain = 0.0;
        for (int j = 0; j < dim; j++) {
            for (int i = 0; i < dim; i++) {
                estrain(i, j) -= half * Binv(i, j);
            }
            estrain(j, j) += half;
        }
    }
    
    // Convert to Voigt notation
    mfem::Vector euler_strain(6);
    euler_strain(0) = estrain(0, 0);
    euler_strain(1) = estrain(1, 1);
    euler_strain(2) = estrain(2, 2);
    euler_strain(3) = estrain(1, 2);
    euler_strain(4) = estrain(0, 2);
    euler_strain(5) = estrain(0, 1);
    
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_" + 
            m_sim_state.GetQuadratureFunctionMapName("euler_strain", region) + ".txt";
        file.open(file_name, std::ios_base::app);
        
//        file.setf(std::ios::scientific);
        file.setf(std::ios::showpoint);
        file.precision(8);
        
        euler_strain.Print(file, euler_strain.Size());
    }
}

void PostProcessingDriver::VolumeAvgElasticStrain(const int region, [[maybe_unused]] const double time) {
    if (m_region_mech_types[region] != MechType::EXACMECH) {
        return;
    }
    
    // Implementation would calculate volume-averaged elastic strain
    // for ExaCMech models
    MFEM_WARNING("Volume average elastic strain not fully implemented yet");
}

void PostProcessingDriver::VolumePlWork(const int region, [[maybe_unused]] const double time) {
    if (m_region_mech_types[region] != MechType::EXACMECH) {
        return;
    }
    
    const auto qf_val = m_sim_state.GetQuadratureFunction("state_var_beg", region);
    
    std::string s_pl_work = "pl_work";
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(s_pl_work, region);
    
    // Calculate volume average of plastic work
    mfem::Vector state_var(qf_val->GetVDim());
    state_var = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<false>(
        m_sim_state.GetMeshParFiniteElementSpace().get(), 
        qf_val.get(), 
        state_var, 
        state_var.Size(), 
        m_sim_state.class_device);
    
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_pl_work_" + std::to_string(region) + ".txt";
        file.open(file_name, std::ios_base::app);
        
//        file.setf(std::ios::scientific);
        file.setf(std::ios::showpoint);
        file.precision(8);
        
        file << state_var[pair.first] << std::endl;
    }
}

// Individual projection methods
void PostProcessingDriver::ProjectCentroid(const int region) {
    auto field_name = m_sim_state.GetQuadratureFunctionMapName("centroid", region);
    
    ProjectionTraits::CentroidTrait::Project(
        m_sim_state.GetMeshParFiniteElementSpace().get(),
        *m_map_gfs[field_name]);
}

void PostProcessingDriver::ProjectVolume(const int region) {
    auto field_name = m_sim_state.GetQuadratureFunctionMapName("volume", region);
    
    ProjectionTraits::VolumeTrait::Project(
        m_sim_state.GetMeshParFiniteElementSpace().get(),
        *m_map_gfs[field_name]);
}

void PostProcessingDriver::ProjectModelStress(const int region) {
    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", region);
    auto csi = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_init", region);
    auto stress_q = m_sim_state.GetQuadratureFunction("cauchy_stress_init", region);
    
    // Use element averaging
    ProjectionTraits::ProjectionTrait<void>::ProjectQFToGF(
        *stress_q, *m_map_gfs[cse], *m_evec);
}

void PostProcessingDriver::ProjectVonMisesStress(const int region) {
    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", region);
    auto vms = m_sim_state.GetQuadratureFunctionMapName("von_mises_stress", region);
    
    ProjectionTraits::VonMisesStressTrait::PostProcess(
        *m_map_gfs[cse], *m_map_gfs[vms]);
}

void PostProcessingDriver::ProjectHydroStress(const int region) {
    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", region);
    auto hs = m_sim_state.GetQuadratureFunctionMapName("hydrostatic_stress", region);
    
    ProjectionTraits::HydroStressTrait::PostProcess(
        *m_map_gfs[cse], *m_map_gfs[hs]);
}

void PostProcessingDriver::ProjectDpEff(const int region) {
    if (m_region_mech_types[region] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "effective_plastic_deformation_rate", region);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair("effective_plastic_deformation_rate", region);
    
    ProjectionTraits::ProjectionTrait<ProjectionTraits::DpEffTrait>::ProjectComponent(
        *m_evec, *m_map_gfs[field_name], pair);
}

void PostProcessingDriver::ProjectEffPlasticStrain(const int region) {
    if (m_region_mech_types[region] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "effective_plastic_deformation", region);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair("effective_plastic_deformation", region);
    
    ProjectionTraits::ProjectionTrait<ProjectionTraits::EffPlasticStrainTrait>::ProjectComponent(
        *m_evec, *m_map_gfs[field_name], pair);
}

void PostProcessingDriver::ProjectShearRate(const int region) {
    if (m_region_mech_types[region] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "plastic_shearing_rate", region);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair("plastic_shearing_rate", region);
    
    ProjectionTraits::ProjectionTrait<ProjectionTraits::ShearRateTrait>::ProjectComponent(
        *m_evec, *m_map_gfs[field_name], pair);
}

void PostProcessingDriver::ProjectOrientation(const int region) {
    if (m_region_mech_types[region] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "lattice_orientation", region);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair("lattice_orientation", region);
    
    // Project the component
    ProjectionTraits::ProjectionTrait<ProjectionTraits::OrientationTrait>::ProjectComponent(
        *m_evec, *m_map_gfs[field_name], pair);
    
    // Apply normalization post-processing
    ProjectionTraits::OrientationTrait::PostProcess(*m_map_gfs[field_name]);
}

void PostProcessingDriver::ProjectH(const int region) {
    if (m_region_mech_types[region] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "hardness", region);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair("hardness", region);
    
    ProjectionTraits::ProjectionTrait<ProjectionTraits::HardnessTrait>::ProjectComponent(
        *m_evec, *m_map_gfs[field_name], pair);
}

void PostProcessingDriver::ProjectElasticStrains(const int region) {
    if (m_region_mech_types[region] != MechType::EXACMECH) return;
    
    auto s_estrain = m_sim_state.GetQuadratureFunctionMapName("lattice_elastic_strain", region);
    auto s_rvol = m_sim_state.GetQuadratureFunctionMapName("relative_volume", region);
    
    auto estrain_pair = m_sim_state.GetQuadratureFunctionStatePair("lattice_elastic_strain", region);
    auto rvol_pair = m_sim_state.GetQuadratureFunctionStatePair("relative_volume", region);
    
    // Apply the transformation
    ProjectionTraits::ElasticStrainTrait::PostProcess(
        *m_map_gfs[s_estrain], *m_evec, estrain_pair, rvol_pair);
}