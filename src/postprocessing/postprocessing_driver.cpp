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
    const bool enable_visualization = 
        options.visit || options.conduit || options.paraview || options.adios2;
    
    // Initialize phase model types
    m_phase_mech_types.resize(m_sim_state.GetNumberOfPhases());
    for (int phase = 0; phase < m_sim_state.GetNumberOfPhases(); phase++) {
        m_phase_mech_types[phase] = m_sim_state.GetPhaseModelType(phase);
    }
    
    // Initialize m_evec for element averaging if visualization is enabled
    if (enable_visualization) {
        auto qf_size = GetQuadratureFunctionSize();
        m_evec = std::make_unique<mfem::Vector>(qf_size);
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
    
    // Register phase-specific projections based on model type
    for (int phase = 0; phase < m_sim_state.GetNumberOfPhases(); phase++) {
        if (m_phase_mech_types[phase] == MechType::EXACMECH) {
            // ExaCMech-specific projections
            // Note: These will only be enabled for this specific phase
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
    
    // Register volume average calculations if requested
    if (options.additional_avgs) {
        // Register volume averaging methods
        RegisterVolumeAverageFunction(
            "avg_stress", "Average Stress",
            [this](const int phase, const double time) { 
                this->VolumeAvgStress(phase, time); 
            },
            true);
            
        RegisterVolumeAverageFunction(
            "avg_euler_strain", "Average Euler Strain",
            [this](const int phase, const double time) { 
                this->VolumeAvgEulerStrain(phase, time); 
            },
            true);
            
        RegisterVolumeAverageFunction(
            "avg_def_grad", "Average Deformation Gradient",
            [this](const int phase, const double time) { 
                this->VolumeAvgDefGrad(phase, time); 
            },
            true);
            
        // ExaCMech-specific volume averages
        RegisterVolumeAverageFunction(
            "avg_pl_work", "Average Plastic Work",
            [this](const int phase, const double time) { 
                this->VolumePlWork(phase, time); 
            }, 
            true);
            
        RegisterVolumeAverageFunction(
            "avg_elastic_strain", "Average Elastic Strain",
            [this](const int phase, const double time) { 
                this->VolumeAvgElasticStrain(phase, time); 
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
    
    auto projection_func = [this, strain_field, vol_field](int phase) {
        // Skip if incompatible with this phase's model
        if (m_phase_mech_types[phase] != MechType::EXACMECH) {
            return;
        }
        
        this->ExecuteElasticStrainProjection(strain_field, vol_field, phase);
    };
    
    // Initialize per-phase enabled flags
    std::vector<bool> phase_enabled(m_sim_state.GetNumberOfPhases(), default_enabled);
    
    // Register the projection
    m_registered_projections.push_back({
        strain_field,
        display_name,
        compatibility,
        phase_enabled,
        projection_func
    });
}

void PostProcessingDriver::ExecuteElasticStrainProjection(
    const std::string& strain_field,
    const std::string& vol_field,
    int phase
) {
    auto strain_name = m_sim_state.GetQuadratureFunctionMapName(strain_field, phase);
    auto vol_name = m_sim_state.GetQuadratureFunctionMapName(vol_field, phase);
    
    // Get component info
    auto strain_pair = m_sim_state.GetQuadratureFunctionStatePair(strain_name, phase);
    auto vol_pair = m_sim_state.GetQuadratureFunctionStatePair(vol_name, phase);
    
    // Get grid function
    auto& estrain = *m_map_gfs[strain_name];
    
    // Execute specialized projection
    ProjectionTraits::ElasticStrainTrait::PostProcess(
        estrain, *m_evec, strain_pair, vol_pair);
}

void PostProcessingDriver::Update(const int step, const double time) {
    PrintVolValues(time);
    UpdateDataCollections(step, time);
}

void PostProcessingDriver::PrintVolValues(const double time) {
    CALI_CXX_MARK_SCOPE("print_vol_values");
    
    // Execute all enabled volume average calculations
    for (auto& [name, func] : m_map_avg_fcns) {
        // Skip disabled calculations
        if (!m_map_avg_enabled[name]) {
            continue;
        }
        
        // Execute volume average calculation for each phase
        for (int phase = 0; phase < m_sim_state.GetNumberOfPhases(); phase++) {
            // ExaCMech-specific check
            if ((name.find("pl_work") != std::string::npos || 
                 name.find("elastic_strain") != std::string::npos) &&
                m_phase_mech_types[phase] != MechType::EXACMECH) {
                continue;
            }
            
            // Call the volume average function
            func(phase, time);
        }
    }
}

void PostProcessingDriver::UpdateDataCollections(const int step, const double time) {
    CALI_CXX_MARK_SCOPE("update_data_collections");
    
    // Only calculate element averages if we have registered projections
    if (!m_registered_projections.empty()) {
        auto mat_vars_0_name = m_sim_state.GetQuadratureFunctionMapName("state_variables_0", 0);
        const auto mat_vars0 = m_sim_state.GetQuadratureFunction(mat_vars_0_name);
        CalcElementAvg(m_evec.get(), mat_vars0.get());
    }
    
    // Execute only user-requested projections for each phase
    for (int phase = 0; phase < m_sim_state.GetNumberOfPhases(); phase++) {
        for (const auto& reg : m_registered_projections) {
            // Skip if not requested by user for this phase
            if (phase >= reg.user_requested.size() || !reg.user_requested[phase]) {
                continue;
            }
            
            // Skip if incompatible with this phase's model type
            auto compatibility = reg.model_compatibility;
            if ((compatibility == ProjectionTraits::ModelCompatibility::EXACMECH_ONLY && 
                m_phase_mech_types[phase] != MechType::EXACMECH) ||
                (compatibility == ProjectionTraits::ModelCompatibility::UMAT_ONLY && 
                m_phase_mech_types[phase] != MechType::UMAT)) {
                continue;
            }
            
            // Execute the projection
            reg.projection_function(phase);
        }
    }
    
    // Update all data collections
    for (auto& [name, dc] : m_map_dcs) {
        dc->SetCycle(step);
        dc->SetTime(time);
        dc->Save();
    }
}

void PostProcessingDriver::EnableProjection(const std::string& field_name, int phase, bool enable) {
    for (auto& reg : m_registered_projections) {
        if (reg.field_name == field_name && phase < reg.user_requested.size()) {
            reg.user_requested[phase] = enable;
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

void PostProcessingDriver::CalcElementAvg(mfem::Vector* elemVal, const mfem::QuadratureFunction* qf) {
    ProjectionTraits::ProjectionTrait<void>::CalcElementAvg(elemVal, *qf);
}

size_t PostProcessingDriver::GetQuadratureFunctionSize() const {
    // Determine size based on quad function vdim and number of elements
    auto mat_vars_0_name = m_sim_state.GetQuadratureFunctionMapName("state_variables_0", 0);
    const auto mat_vars0 = m_sim_state.GetQuadratureFunction(mat_vars_0_name);
    
    const int nelems = m_sim_state.GetMeshParFiniteElementSpace()->GetNE();
    const int vdim = mat_vars0->GetVDim();
    
    return static_cast<size_t>(nelems * vdim);
}

void PostProcessingDriver::InitializeDataCollections(ExaOptions& options) {
    // Create appropriate data collections based on options
    // This implementation depends on the details of ExaOptions
    // and would include code like:
    
    if (options.visit) {
        auto visit_dc = std::make_unique<mfem::VisItDataCollection>(options.basename, m_sim_state.GetMesh());
        visit_dc->SetPrecision(12);
        m_map_dcs["visit"] = std::move(visit_dc);
    }
    
    if (options.paraview) {
        auto paraview_dc = std::make_unique<mfem::ParaViewDataCollection>(options.basename, m_sim_state.GetMesh());
        paraview_dc->SetLevelsOfDetail(options.order);
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
                for (int phase = 0; phase < m_sim_state.GetNumberOfPhases(); phase++) {
                    if (phase < reg.user_requested.size() && reg.user_requested[phase]) {
                        auto field_name = m_sim_state.GetQuadratureFunctionMapName(reg.field_name, phase);
                        dc->RegisterField(field_name, m_map_gfs[field_name].get());
                    }
                }
            }
        }
    }
}

// Volume average calculation methods
void PostProcessingDriver::VolumeAvgStress(const int phase, const double time) {
    CALI_CXX_MARK_SCOPE("avg_stress_computation");
    
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_init", phase);
    const auto qf_val = m_sim_state.GetQuadratureFunction(qf_name);

    // Calculate volume average
    mfem::Vector vol_avg_quant(6);
    vol_avg_quant = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(
        *m_sim_state.GetMeshParFiniteElementSpace(), 
        qf_val.get(), 
        vol_avg_quant, 
        6, 
        m_sim_state.class_device);

    // Only rank 0 writes to file
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_" + 
            m_sim_state.GetQuadratureFunctionMapName("cauchy_stress", phase) + ".txt";
        file.open(file_name, std::ios_base::app);
        
        file.setf(std::ios::fixed);
        file.setf(std::ios::showpoint);
        file.precision(8);
        
        vol_avg_quant.Print(file, 6);
    }
}

void PostProcessingDriver::VolumeAvgDefGrad(const int phase, const double time) {
    CALI_CXX_MARK_SCOPE("avg_def_grad_computation");
    
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("deformation_gradient_init", phase);
    const auto qf_val = m_sim_state.GetQuadratureFunction(qf_name);
    
    mfem::Vector vol_avg_dgrad(qf_val->GetVDim());
    vol_avg_dgrad = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(
        *m_sim_state.GetMeshParFiniteElementSpace(), 
        qf_val.get(), 
        vol_avg_dgrad, 
        vol_avg_dgrad.Size(), 
        m_sim_state.class_device);
    
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_" + 
            m_sim_state.GetQuadratureFunctionMapName("def_grad", phase) + ".txt";
        file.open(file_name, std::ios_base::app);
        
        file.setf(std::ios::fixed);
        file.setf(std::ios::showpoint);
        file.precision(8);
        
        vol_avg_dgrad.Print(file, vol_avg_dgrad.Size());
    }
}

void PostProcessingDriver::VolumeAvgEulerStrain(const int phase, const double time) {
    CALI_CXX_MARK_SCOPE("avg_eul_strain_computation");
    
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("deformation_gradient_init", phase);
    const auto qf_val = m_sim_state.GetQuadratureFunction(qf_name);
    
    mfem::Vector vol_avg_dgrad(qf_val->GetVDim());
    vol_avg_dgrad = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(
        *m_sim_state.GetMeshParFiniteElementSpace(), 
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
            m_sim_state.GetQuadratureFunctionMapName("euler_strain", phase) + ".txt";
        file.open(file_name, std::ios_base::app);
        
        file.setf(std::ios::fixed);
        file.setf(std::ios::showpoint);
        file.precision(8);
        
        euler_strain.Print(file, euler_strain.Size());
    }
}

void PostProcessingDriver::VolumeAvgElasticStrain(const int phase, const double time) {
    if (m_phase_mech_types[phase] != MechType::EXACMECH) {
        return;
    }
    
    // Implementation would calculate volume-averaged elastic strain
    // for ExaCMech models
    MFEM_WARNING("Volume average elastic strain not fully implemented yet");
}

void PostProcessingDriver::VolumePlWork(const int phase, const double time) {
    if (m_phase_mech_types[phase] != MechType::EXACMECH) {
        return;
    }
    
    CALI_CXX_MARK_SCOPE("vol_pl_work_computation");
    
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("state_variables_0", phase);
    const auto qf_val = m_sim_state.GetQuadratureFunction(qf_name);
    
    std::string s_pl_work = "pl_work";
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(s_pl_work, phase);
    
    // Calculate volume average of plastic work
    mfem::Vector state_var(qf_val->GetVDim());
    state_var = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<false>(
        *m_sim_state.GetMeshParFiniteElementSpace(), 
        qf_val.get(), 
        state_var, 
        state_var.Size(), 
        m_sim_state.class_device);
    
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_pl_work_" + std::to_string(phase) + ".txt";
        file.open(file_name, std::ios_base::app);
        
        file.setf(std::ios::fixed);
        file.setf(std::ios::showpoint);
        file.precision(8);
        
        file << state_var[pair.first] << std::endl;
    }
}

// Individual projection methods
void PostProcessingDriver::ProjectCentroid(const int phase) {
    auto field_name = m_sim_state.GetQuadratureFunctionMapName("centroid", phase);
    
    ProjectionTraits::CentroidTrait::Project(
        m_sim_state.GetMeshParFiniteElementSpace(),
        *m_map_gfs[field_name]);
}

void PostProcessingDriver::ProjectVolume(const int phase) {
    auto field_name = m_sim_state.GetQuadratureFunctionMapName("volume", phase);
    
    ProjectionTraits::VolumeTrait::Project(
        m_sim_state.GetMeshParFiniteElementSpace(),
        *m_map_gfs[field_name]);
}

void PostProcessingDriver::ProjectModelStress(const int phase) {
    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", phase);
    auto csi = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_init", phase);
    auto stress_q = m_sim_state.GetQuadratureFunction(csi, phase);
    
    // Use element averaging
    ProjectionTraits::ProjectionTrait<void>::ProjectQFToGF(
        stress_q.get(), *m_map_gfs[cse], m_evec.get());
}

void PostProcessingDriver::ProjectVonMisesStress(const int phase) {
    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", phase);
    auto vms = m_sim_state.GetQuadratureFunctionMapName("von_mises_stress", phase);
    
    ProjectionTraits::VonMisesStressTrait::PostProcess(
        *m_map_gfs[cse], *m_map_gfs[vms]);
}

void PostProcessingDriver::ProjectHydroStress(const int phase) {
    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", phase);
    auto hs = m_sim_state.GetQuadratureFunctionMapName("hydrostatic_stress", phase);
    
    ProjectionTraits::HydroStressTrait::PostProcess(
        *m_map_gfs[cse], *m_map_gfs[hs]);
}

void PostProcessingDriver::ProjectDpEff(const int phase) {
    if (m_phase_mech_types[phase] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "effective_plastic_deformation_rate", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(field_name, phase);
    
    ProjectionTraits::ProjectionTrait<ProjectionTraits::DpEffTrait>::ProjectComponent(
        m_evec.get(), *m_map_gfs[field_name], pair);
}

void PostProcessingDriver::ProjectEffPlasticStrain(const int phase) {
    if (m_phase_mech_types[phase] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "effective_plastic_deformation", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(field_name, phase);
    
    ProjectionTraits::ProjectionTrait<ProjectionTraits::EffPlasticStrainTrait>::ProjectComponent(
        m_evec.get(), *m_map_gfs[field_name], pair);
}

void PostProcessingDriver::ProjectShearRate(const int phase) {
    if (m_phase_mech_types[phase] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "plastic_shearing_rate", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(field_name, phase);
    
    ProjectionTraits::ProjectionTrait<ProjectionTraits::ShearRateTrait>::ProjectComponent(
        m_evec.get(), *m_map_gfs[field_name], pair);
}

void PostProcessingDriver::ProjectOrientation(const int phase) {
    if (m_phase_mech_types[phase] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "lattice_orientation", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(field_name, phase);
    
    // Project the component
    ProjectionTraits::ProjectionTrait<ProjectionTraits::OrientationTrait>::ProjectComponent(
        m_evec.get(), *m_map_gfs[field_name], pair);
    
    // Apply normalization post-processing
    ProjectionTraits::OrientationTrait::PostProcess(*m_map_gfs[field_name]);
}

void PostProcessingDriver::ProjectH(const int phase) {
    if (m_phase_mech_types[phase] != MechType::EXACMECH) return;
    
    auto field_name = m_sim_state.GetQuadratureFunctionMapName(
        "hardness", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(field_name, phase);
    
    ProjectionTraits::ProjectionTrait<ProjectionTraits::HardnessTrait>::ProjectComponent(
        m_evec.get(), *m_map_gfs[field_name], pair);
}

void PostProcessingDriver::ProjectElasticStrains(const int phase) {
    if (m_phase_mech_types[phase] != MechType::EXACMECH) return;
    
    auto s_estrain = m_sim_state.GetQuadratureFunctionMapName("lattice_elastic_strain", phase);
    auto s_rvol = m_sim_state.GetQuadratureFunctionMapName("relative_volume", phase);
    
    auto estrain_pair = m_sim_state.GetQuadratureFunctionStatePair(s_estrain, phase);
    auto rvol_pair = m_sim_state.GetQuadratureFunctionStatePair(s_rvol, phase);
    
    // Apply the transformation
    ProjectionTraits::ElasticStrainTrait::PostProcess(
        *m_map_gfs[s_estrain], *m_evec, estrain_pair, rvol_pair);
}