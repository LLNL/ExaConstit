#include "postprocessing_driver.hpp"
#include "mechanics_kernels.hpp"

void SimulationState::UpdateModel()
{
    for (auto pair : m_model_update_qf_pairs) {
        auto beg = m_map_qfs[*pair.first];
        auto end = m_map_qfs[*pair.second];
        beg.get()->Swap(*end.get());
    }
}

void PostProcessingDriver::Update(const int step, const double time)
{
    PrintVolValues(time);
    UpdateDataCollections(step, time);
}

void PostProcessingDriver::PrintVolValues(const double time)
{
    // Do all of our volume print values
    for (auto it : m_map_avg_fcns) {
        auto &vol_print_func = *it.second;
        vol_print_func(time);
    }
}


void PostProcessingDriver::UpdateDataCollections(const int step, const double time)
{
    auto mat_vars_0_name = m_sim_state.GetQuadratureFunctionMapName("state_variables_0", 0);
    const auto mat_vars0 = m_sim_state.GetQuadratureFunction(mat_vars_0_name);
    CalcElementAvg(*m_evec, mat_vars0);
    // Update all of our gridfunctions
    for (auto it : m_map_gfs_fcns) {
        auto &gf_update_func = *it.second;
        gf_update_func();
    }
    // Update all of our data collections now
    for (auto it : m_map_dcs) {
        auto &data_collection = *it.second;
        data_collection->SetCycle(step);
        data_collection->SetTime(time);
        data_collection->Save();
    }
}

void PostProcessingDriver::VolumeAvgStress(const int /* phase */, const double /* time */)
{
    CALI_CXX_MARK_SCOPE("avg_stress_computation");
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_init", -1);
    const auto qf_val = m_sim_state.GetQuadratureFunction(qf_name);

    // Here we're getting the average stress value
    mfem::Vector vol_avg_quant(6);
    vol_avg_quant = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(*m_sim_state.GetMeshParFiniteElementSpace(), qf_val, vol_avg_quant, 6, m_sim_state.class_device);

    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::showpoint);
    std::cout.precision(8);

    // Now we're going to save off the average stress tensor to a file
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_" + m_sim_state.GetQuadratureFunctionMapName("cauchy_stress") + ".txt";
        file.open(file_name, std::ios_base::app);
        vol_avg_quant.Print(file, 6);
    }
}

void PostProcessingDriver::VolumeAvgDefGrad(const double /* time */)
{
    CALI_CXX_MARK_SCOPE("avg_def_grad_computation");
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("deformation_gradient_init", -1);
    const auto qf_val = m_sim_state.GetQuadratureFunction(qf_name);

    mfem::Vector vol_avg_dgrad(qf_val->GetVDim());

    vol_avg_dgrad = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(*m_sim_state.GetMeshParFiniteElementSpace(), qf_val, vol_avg_dgrad, vol_avg_dgrad.Size(), m_sim_state.class_device);

    // Now we're going to save off the average stress tensor to a file
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_" + m_sim_state.GetQuadratureFunctionMapName("def_grad") + ".txt";
        file.open(file_name, std::ios_base::app);
        euler_strain.Print(file, euler_strain.Size());
    }
    
}

void PostProcessingDriver::VolumeAvgEulerStrain(const int /* phase */, const double /* time */)
{
    CALI_CXX_MARK_SCOPE("avg_eul_strain_computation");
    auto qf_name = m_sim_state.GetQuadratureFunctionMapName("deformation_gradient_init", -1);
    const auto qf_val = m_sim_state.GetQuadratureFunction(qf_name);

    mfem::Vector vol_avg_dgrad(qf_val->GetVDim());

    vol_avg_dgrad = 0.0;
    exaconstit::kernel::ComputeVolAvgTensor<true>(*m_sim_state.GetMeshParFiniteElementSpace(), qf_val, vol_avg_dgrad, vol_avg_dgrad.Size(), m_sim_state.class_device);

    // Eulerian strain calculation
    mfem::DenseMatrix estrain(3, 3);
    {
        mfem::DenseMatrix def_grad(dgrad.HostReadWrite(), 3, 3);
        // Would be nice if we could just do this but maybe we should create more kernels for users...
        // ExaModel::CalcEulerianStrain(estrain, def_grad);

        /// Eulerian is simply e = 1/2(I - F^(-t)F^(-1))
        const int dim = 3;
        mfem::DenseMatrix Finv(dim), Binv(dim);
        double half = 1.0 / 2.0;

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

    mfem::Vector euler_strain(6);
    euler_strain(0) = estrain(0, 0);
    euler_strain(1) = estrain(1, 1);
    euler_strain(2) = estrain(2, 2);
    euler_strain(3) = estrain(1, 2);
    euler_strain(4) = estrain(0, 2);
    euler_strain(5) = estrain(0, 1);

    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::showpoint);
    std::cout.precision(8);

    // Now we're going to save off the average stress tensor to a file
    if (m_mpi_rank == 0) {
        std::ofstream file;
        std::string file_name = m_avg_filepath_base + "avg_" + m_sim_state.GetQuadratureFunctionMapName("euler_strain") + ".txt";
        file.open(file_name, std::ios_base::app);
        euler_strain.Print(file, euler_strain.Size());
    }
}

void PostProcessingDriver::VolumeAvgElasticStrain(const int phase, const double /* time */)
{
    MFEM_WARNING("Volume average elastic strain not implemented yet");   
}

void PostProcessingDriver::VolumePlWork(const int phase, const double /* time */)
{
    
}



void PostProcessingDriver::PrintVolValues(const double time)
{
    {
        CALI_CXX_MARK_SCOPE("avg_stress_computation");
        // Here we're getting the average stress value
        mfem::Vector stress(6);
        stress = 0.0;

        const mfem::QuadratureFunction *qstress = model->GetStress0();

        exaconstit::kernel::ComputeVolAvgTensor<true>(*m_sim_state.GetMeshParFiniteElementSpace(), qstress, stress, 6, m_sim_state.class_device);

        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::showpoint);
        std::cout.precision(8);

        // Now we're going to save off the average stress tensor to a file
        if (my_id == 0) {
            std::ofstream file;
            file.open(avg_stress_fname, std::ios_base::app);
            stress.Print(file, 6);
        }
    }

    if (mech_type == MechType::EXACMECH && additional_avgs) {
        CALI_CXX_MARK_SCOPE("extra_avgs_computations");
        const mfem::QuadratureFunction *qstate_var = model->GetMatVars0();
        // Here we're getting the average stress value
        mfem::Vector state_var(qstate_var->GetVDim());
        state_var = 0.0;

        std::string s_pl_work = "pl_work";
        auto qf_mapping = GetQuadratureFunctionStatePair();
        auto pair = qf_mapping->find(s_pl_work)->second;

        exaconstit::kernel::ComputeVolAvgTensor<false>(fes, qstate_var, state_var, state_var.Size(), m_sim_state.class_device);

        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::showpoint);
        std::cout.precision(8);

        int my_id;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
        // Now we're going to save off the average stress tensor to a file
        if (my_id == 0) {
            std::ofstream file;
            file.open(avg_pl_work_fname, std::ios_base::app);
            file << state_var[pair.first] << std::endl;
        }
        mech_operator->CalculateDeformationGradient(def_grad);
    }

    if (additional_avgs)
    {
        CALI_CXX_MARK_SCOPE("extra_avgs_def_grad_computation");
        const mfem::QuadratureFunction *qstate_var = &def_grad;
        // Here we're getting the average stress value
        mfem::Vector dgrad(qstate_var->GetVDim());
        dgrad = 0.0;

        exaconstit::kernel::ComputeVolAvgTensor<true>(fes, qstate_var, dgrad, dgrad.Size(), m_sim_state.class_device);

        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::showpoint);
        std::cout.precision(8);

        int my_id;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
        // Now we're going to save off the average stress tensor to a file
        if (my_id == 0) {
            std::ofstream file;
            file.open(avg_def_grad_fname, std::ios_base::app);
            dgrad.Print(file, dgrad.Size());
        }
    }

    if (mech_type == MechType::EXACMECH && additional_avgs) {
        CALI_CXX_MARK_SCOPE("extra_avgs_dp_tensor_computation");

        model->calcDpMat(def_grad);
        const mfem::QuadratureFunction *qstate_var = &def_grad;
        // Here we're getting the average stress value
        mfem::Vector dgrad(qstate_var->GetVDim());
        dgrad = 0.0;

        exaconstit::kernel::ComputeVolAvgTensor<true>(fes, qstate_var, dgrad, dgrad.Size(), m_sim_state.class_device);

        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::showpoint);
        std::cout.precision(8);

        mfem::Vector dpgrad(6);
        dpgrad(0) = dgrad(0);
        dpgrad(1) = dgrad(4);
        dpgrad(2) = dgrad(8);
        dpgrad(3) = dgrad(5);
        dpgrad(4) = dgrad(2);
        dpgrad(5) = dgrad(1);

        int my_id;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
        // Now we're going to save off the average dp tensor to a file
        if (my_id == 0) {
            std::ofstream file;
            file.open(avg_dp_tensor_fname, std::ios_base::app);
            dpgrad.Print(file, dpgrad.Size());
        }
    }

    if(postprocessing) {
        CalcElementAvg(m_evec, model->GetMatVars0());
    }
}

void PostProcessingDriver::CalcElementAvg(mfem::Vector* elemVal, const mfem::QuadratureFunction& qf)
{
    const mfem::FiniteElement &el = *(m_sim_state.GetFiniteElementSpace()->GetFE(0));
    const mfem::IntegrationRule *ir = m_sim_state.GetIntegrationRule();

    const int nqpts = ir->GetNPoints();
    const int nelems = m_sim_state.GetFiniteElementSpace()->GetNE();
    const int vdim = m_sim_state.GetMesh()->SpaceDimension(); 

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors *geom = m_sim_state.GetMesh()->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);

    const int DIM2 = 2;
    const int DIM3 = 3;
    std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
    std::array<RAJA::idx_t, DIM3> perm3 {{2, 1, 0}};

    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);
    RAJA::Layout<DIM2> layout_ev = RAJA::make_permuted_layout({{ vdim, nelems } }, perm2);
    RAJA::Layout<DIM3> layout_qf = RAJA::make_permuted_layout({{vdim, nqpts, nelems}}, perm3);

    (*elemVal) = 0.0;

    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > qf_view(qf.Read(), layout_qf);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > ev_view(elemVal->ReadWrite(), layout_ev);

    MFEM_FORALL(i, nelems, {
        double vol = 0.0;
        for(int j = 0; j < nqpts; j++) {
            const double wts = j_view(j, i) * W[j];
            vol += wts;
            for(int k = 0; k < vdim; k++) {
            ev_view(k, i) += qf_view(k, j, i) * wts;
            }
        }
        const double ivol = 1.0 / vol;
        for(int k = 0; k < vdim; k++) {
            ev_view(k, i) *= ivol;
        }
    });
}

void PostProcessingDriver::ProjectCentroid(const int /* phase */)
{
    const mfem::FiniteElement &el = *(m_sim_state.GetFiniteElementSpace()->GetFE(0));
    const mfem::IntegrationRule *ir = m_sim_state.GetIntegrationRule();

    const int nqpts = ir->GetNPoints();
    const int nelems = m_sim_state.GetFiniteElementSpace()->GetNE();
    const int vdim = m_sim_state.GetMesh()->SpaceDimension(); 

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors *geom = m_sim_state.GetMesh()->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);

    const int DIM2 = 2;
    const int DIM3 = 3;
    std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
    std::array<RAJA::idx_t, DIM3> perm3 {{2, 1, 0}};

    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);
    RAJA::Layout<DIM2> layout_ev = RAJA::make_permuted_layout({{ vdim, nelems } }, perm2);
    RAJA::Layout<DIM3> layout_qf = RAJA::make_permuted_layout({{nqpts, vdim, nelems}}, perm3);

    auto state_name = m_sim_state.GetQuadratureFunctionMapName("centroid", -1);
    mfem::ParGridFunction &centroid = *m_map_gfs[state_name]; 
    centroid = 0.0;

    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > x_view(geom->X.Read(), layout_qf);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > ev_view(centroid.ReadWrite(), layout_ev);

    MFEM_FORALL(i, nelems, {
        double vol = 0.0;
        for(int j = 0; j < nqpts; j++) {
            const double wts = j_view(j, i) * W[j];
            vol += wts;
            for(int k = 0; k < vdim; k++) {
            ev_view(k, i) += x_view(j, k, i) * wts;
            }
        }
        const double ivol = 1.0 / vol;
        for(int k = 0; k < vdim; k++) {
            ev_view(k, i) *= ivol;
        }
    });
}

void PostProcessingDriver::ProjectVolume(const int /* phase */)
{
    const mfem::FiniteElement &el = *m_sim_state.GetFiniteElementSpace()->GetFE(0);
    const mfem::IntegrationRule *ir = m_sim_state.GetIntegrationRule();

    const int nqpts = ir->GetNPoints();
    const int nelems = m_sim_state.GetFiniteElementSpace()->GetNE();

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors *geom = m_sim_state.GetMesh()->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);

    auto state_name = m_sim_state.GetQuadratureFunctionMapName("volume", -1);
    mfem::ParGridFunction &vol = *m_map_gfs[state_name];
    double *vol_data = vol.ReadWrite();
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);

    MFEM_FORALL(i, nelems, {
        vol_data[i] = 0.0;
        for(int j = 0; j < nqpts; j++) {
            vol_data[i] += j_view(j, i) * W[j];
        }
    });
}

void PostProcessingDriver::ProjectModelStress(const int /* phase */)
{
    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", -1);
    mfem::ParGridFunction* s = m_map_gfs[cse].get();
    auto csi = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_init", -1);
    auto stress_q = m_sim_state.GetQuadratureFunction(csi, -1);
    CalcElementAvg(s, *stress_q);
}

void PostProcessingDriver::ProjectVonMisesStress(const int /* phase */)
{  
    const int npts = vm.Size();

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2{{ 1, 0 } };

    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", -1);
    const mfem::ParGridFunction& s = *m_map_gfs[cse];
    auto vms = m_sim_state.GetQuadratureFunctionMapName("von_mises_stress", -1);
    mfem::ParGridFunction& vm = *m_map_gfs[vms];

    RAJA::Layout<DIM2> layout_stress = RAJA::make_permuted_layout({{ 6, npts } }, perm2);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > stress_view(s.Read(), layout_stress);
    double *vm_data = vm.ReadWrite();

    MFEM_FORALL(i, npts, {
        double term1 = stress_view(0, i) - stress_view(1, i);
        double term2 = stress_view(1, i) - stress_view(2, i);
        double term3 = stress_view(2, i) - stress_view(0, i);
        double term4 = stress_view(3, i) * stress_view(3, i)
                        + stress_view(4, i) * stress_view(4, i)
                        + stress_view(5, i) * stress_view(5, i);

        term1 *= term1;
        term2 *= term2;
        term3 *= term3;
        term4 *= 6.0;

        vm_data[i] = sqrt(0.5 * (term1 + term2 + term3 + term4));
    });

}

void PostProcessingDriver::ProjectHydroStress(const int /* phase */)
{
    const int npts = hss.Size();

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2{{ 1, 0 } };

    auto cse = m_sim_state.GetQuadratureFunctionMapName("cauchy_stress_end", -1);
    const mfem::ParGridFunction& s = *m_map_gfs[cse];
    auto hs = m_sim_state.GetQuadratureFunctionMapName("hydrostatic_stress", -1);
    mfem::ParGridFunction& hss = *m_map_gfs[hs];

    RAJA::Layout<DIM2> layout_stress = RAJA::make_permuted_layout({{ 6, npts } }, perm2);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > stress_view(s.Read(), layout_stress);
    double* hydro = hss.ReadWrite();

    const double one_third = 1.0 / 3.0;

    MFEM_FORALL(i, npts, {
        hydro[i] = one_third * (stress_view(0, i) + stress_view(1, i) + stress_view(2, i));
    });

   
}

// These next group of Project* functions are only available with ExaCMech type models
// Need to figure out a smart way to get all of the indices that I want for down below
// that go with ExaModel
void PostProcessingDriver::ProjectDpEff(const int phase)
{
    auto s_shrateEff = m_sim_state.GetQuadratureFunctionMapName("effective_plastic_deformation_rate", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(s_shrateEff, phase);

    mfem::VectorQuadratureFunctionCoefficient qfvc(*m_evec);
    qfvc.SetComponent(pair.first, pair.second);

    mfem::ParGridFunction& dpeff = *m_map_gfs[s_shrateEff];
    dpeff.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
}

void PostProcessingDriver::ProjectEffPlasticStrain(const int phase)
{
    auto s_shrEff = m_sim_state.GetQuadratureFunctionMapName("effective_plastic_deformation", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(s_shrEff, phase);

    mfem::VectorQuadratureFunctionCoefficient qfvc(*m_evec);
    qfvc.SetComponent(pair.first, pair.second);
    
    mfem::ParGridFunction& pleff = *m_map_gfs[s_shrEff];
    pleff.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
}

void PostProcessingDriver::ProjectShearRate(const int phase)
{

    auto s_gdot = m_sim_state.GetQuadratureFunctionMapName("plastic_shearing_rate", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(s_gdot, phase);

    mfem::VectorQuadratureFunctionCoefficient qfvc(*m_evec);
    qfvc.SetComponent(pair.first, pair.second);

    mfem::ParGridFunction& gdot = *m_map_gfs[s_gdot];
    gdot.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
}

// This one requires that the orientations be made unit normals afterwards
void PostProcessingDriver::ProjectOrientation(const int phase)
{
    auto s_quats = m_sim_state.GetQuadratureFunctionMapName("lattice_orientation", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(s_quats, phase);

    mfem::VectorQuadratureFunctionCoefficient qfvc(*m_evec);
    qfvc.SetComponent(pair.first, pair.second);

    mfem::ParGridFunction& quats = *m_map_gfs[s_quats];
    quats.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);

    // The below is normalizing the quaternion since it most likely was not
    // returned normalized
    int _size = quats.Size();
    int size = _size / 4;

    double norm = 0;
    double inv_norm = 0;
    int index = 0;

    for (int i = 0; i < size; i++) {
        index = i * 4;

        norm = quats(index + 0) * quats(index + 0);
        norm += quats(index + 1) * quats(index + 1);
        norm += quats(index + 2) * quats(index + 2);
        norm += quats(index + 3) * quats(index + 3);

        inv_norm = 1.0 / sqrt(norm);

        for (int j = 0; j < 4; j++) {
        quats(index + j) *= inv_norm;
        }
    }
}

// Here this can be either the CRSS for a voce model or relative dislocation density
// value for the MTS model.
void PostProcessingDriver::ProjectH(const int phase)
{
    std::string s_hard = "hardness";
    auto s_hard = m_sim_state.GetQuadratureFunctionMapName("hardness", phase);
    auto pair = m_sim_state.GetQuadratureFunctionStatePair(s_hard, phase);

    mfem::VectorQuadratureFunctionCoefficient qfvc(*m_evec);
    qfvc.SetComponent(pair.first, pair.second);

    mfem::ParGridFunction& h = *m_map_gfs[s_hard];
    h.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
}

// This one requires that the deviatoric strain be converted from 5d rep to 6d
// and have vol. contribution added.
void PostProcessingDriver::ProjectElasticStrains(const int phase)
{

    auto s_estrain = m_sim_state.GetQuadratureFunctionMapName("lattice_elastic_strain", phase);
    auto s_rvol = m_sim_state.GetQuadratureFunctionMapName("relative_volume", phase);
    auto espair = m_sim_state.GetQuadratureFunctionStatePair(s_estrain, phase);
    auto rvpair = m_sim_state.GetQuadratureFunctionStatePair(s_rvol, phase);

    const int e_offset = espair.first;
    const int rv_offset = rvpair.first;

    int _size = estrain.Size();
    int nelems = _size / 6;

    mfem::ParGridFunction& estrain = *m_map_gfs[s_estrain];
    auto data_estrain = mfem::Reshape(estrain.HostReadWrite(), 6, nelems);
    auto data_evec = mfem::Reshape(m_evec->HostReadWrite(), m_evec->GetVDim(), nelems);
    // The below is outputting the full elastic strain in the crystal ref frame
    // We'd only stored the 5d deviatoric elastic strain, so we need to convert
    // it over to the 6d version and add in the volume elastic strain contribution.
    for (int i = 0; i < nelems; i++) {
        const double t1 = ecmech::sqr2i * data_evec(0 + e_offset, i);
        const double t2 = ecmech::sqr6i * data_evec(1 + e_offset, i);
        //
        // Volume strain is ln(V^e_mean) term aka ln(relative volume)
        // Our plastic deformation has a det(1) aka no change in volume change
        const double elas_vol_strain = log(data_evec(rv_offset, i));
        // We output elastic strain formulation such that the relationship
        // between V^e and \varepsilon is just V^e = I + \varepsilon
        data_estrain(0, i) = (t1 - t2) + elas_vol_strain; // 11
        data_estrain(1, i) = (-t1 - t2) + elas_vol_strain ; // 22
        data_estrain(2, i) = ecmech::sqr2b3 * data_evec(1 + e_offset, i) + elas_vol_strain; // 33
        data_estrain(3, i) = ecmech::sqr2i * data_evec(4 + e_offset, i); // 23
        data_estrain(4, i) = ecmech::sqr2i * data_evec(3 + e_offset, i); // 31
        data_estrain(5, i) = ecmech::sqr2i * data_evec(2 + e_offset, i); // 12
    }
}
