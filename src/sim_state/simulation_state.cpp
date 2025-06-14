#include "simulation_state.hpp"

namespace {

void setupBoundaryConditions(ExaOptions& options) {
    BCManager& bcm = BCManager::getInstance();
    auto& bcs_opts = options.boundary_conditions;
    bcm.init(bcs_opts.time_info.cycles, bcs_opts.map_ess_vel, bcs_opts.map_ess_vgrad, bcs_opts.map_ess_comp,
        bcs_opts.map_ess_id);
}

void setBdrConditions(mfem::Mesh& mesh)
{
   // modify MFEM auto cuboidal hex mesh generation boundary
   // attributes to correspond to correct ExaConstit boundary conditions.
   // Look at ../../mesh/mesh.cpp Make3D() to see how boundary attributes
   // are set and modify according to ExaConstit convention

   // loop over boundary elements
   for (int i = 0; i<mesh.GetNBE(); ++i) {
      int bdrAttr = mesh.GetBdrAttribute(i);

      switch (bdrAttr) {
         // note, srw wrote SetBdrAttribute() in ../../mesh/mesh.hpp
         case 1:
            mesh.SetBdrAttribute(i, 1); // bottom
            break;
         case 2:
            mesh.SetBdrAttribute(i, 3); // front
            break;
         case 3:
            mesh.SetBdrAttribute(i, 5); // right
            break;
         case 4:
            mesh.SetBdrAttribute(i, 6); // back
            break;
         case 5:
            mesh.SetBdrAttribute(i, 2); // left
            break;
         case 6:
            mesh.SetBdrAttribute(i, 4); // top
            break;
      }
   }

   return;
}

void setElementGrainIDs(mfem::Mesh& mesh, const mfem::Vector& grainMap, int ncols, int offset)
{
   // after a call to reorderMeshElements, the elements in the serial
   // MFEM mesh should be ordered the same as the input grainMap
   // vector. Set the element attribute to the grain id. This vector
   // has stride of 4 with the id in the 3rd position indexing from 0

   const double* data = grainMap.HostRead();

   // loop over elements
   for (int i = 0; i<mesh.GetNE(); ++i) {
      mesh.SetAttribute(i, data[ncols * i + offset]);
   }

   return;
}

// Projects the element attribute to GridFunction nodes
// This also assumes this the GridFunction is an L2 FE space
void projectElemAttr2GridFunc(std::shared_ptr<mfem::Mesh> mesh, std::shared_ptr<mfem::ParGridFunction> elem_attr) {
   // loop over elementsQ
   elem_attr->HostRead();
   mfem::ParFiniteElementSpace *pfes = elem_attr->ParFESpace();
   mfem::Array<int> vdofs;
   for (int i = 0; i < mesh->GetNE(); ++i) {
      pfes->GetElementVDofs(i, vdofs);
      const double ea = static_cast<double>(mesh->GetAttribute(i));
      elem_attr->SetSubVector(vdofs, ea);
   }
}

std::shared_ptr<mfem::ParMesh> makeMesh(ExaOptions& options, const int my_id)
{
    mfem::Mesh mesh;
    if (options.mesh.mesh_type == MeshType::FILE) {

        if (my_id == 0) {
            std::cout << "Opening mesh file: " << options.mesh.mesh_file << std::endl;
        }

        mesh = mfem::Mesh(options.mesh.mesh_file.c_str(), 1, 1, true);
    }
    // We're using the auto mesh generator
    else {
        if (options.mesh.nxyz[0] <= 0 || options.mesh.mxyz[0] <= 0) {
            std::cerr << std::endl << "Must input mesh geometry/discretization for hex_mesh_gen" << std::endl;
        }

        if (my_id == 0) {
            std::cout << "Using mfem's hex mesh generator" << std::endl;
        }

        // use constructor to generate a 3D cuboidal mesh with 8 node hexes
        // The false at the end is to tell the inline mesh generator to use the lexicographic ordering of the mesh
        // The newer space-filling ordering option that was added in the pre-okina tag of MFEM resulted in a noticeable divergence
        // of the material response for a monotonic tension test using symmetric boundary conditions out to 1% strain.
        mesh =
            mfem::Mesh::MakeCartesian3D(options.mesh.nxyz[0], options.mesh.nxyz[1], options.mesh.nxyz[2], mfem::Element::HEXAHEDRON, 
                options.mesh.mxyz[0], options.mesh.mxyz[1], options.mesh.mxyz[2], false);
        // read in the grain map if using a MFEM auto generated cuboidal mesh
        if (options.grain_file) {
            std::ifstream gfile(options.grain_file->c_str());
            if (!gfile && my_id == 0) {
                std::cerr << std::endl << "Cannot open grain map file: " << options.grain_file->c_str() << std::endl;
            }

            const int gmap_size = mesh.GetNE();
            mfem::Vector gmap(gmap_size);
            gmap.Load(gfile, gmap_size);
            gfile.close();

            // set grain ids as element attributes on the mesh
            // The offset of where the grain index is located is
            // location - 1.
            ::setElementGrainIDs(mesh, gmap, 1, 0);
        }
        //// reorder elements to conform to ordering convention in grain map file
        // No longer needed for the CA stuff. It's now ordered as X->Y->Z
        // reorderMeshElements(mesh, &toml_opt.nxyz[0]);

        // reset boundary conditions from
        ::setBdrConditions(mesh);
    }

    // We need to check to see if our provided mesh has a different order than
    // the order provided. If we see a difference we either increase our order seen
    // in the options file or we increase the mesh ordering. I'm pretty sure this
    // was causing a problem earlier with our auto-generated mesh and if we wanted
    // to use a higher order FE space.
    // So we can't really do the GetNodalFESpace it appears if we're given
    // an initial mesh. It looks like NodalFESpace is initially set to
    // NULL and only if we swap the mesh nodes does this actually
    // get set...
    // So, we're just going to set the mesh order to at least be 1. Although,
    // I would like to see this change sometime in the future.
    int mesh_order = 1;
    if (mesh_order > options.mesh.order) {
        options.mesh.order = mesh_order;
    }
    if (mesh_order <= options.mesh.order) {
        if (my_id == 0) {
            std::cout << "Increasing mesh order of the mesh to " << options.mesh.order << std::endl;
        }
        mesh_order = options.mesh.order;
        mesh.SetCurvature(mesh_order);
    }

    // mesh refinement if specified in input
    for (int lev = 0; lev < options.mesh.ref_ser; lev++) {
        mesh.UniformRefinement();
    }

    std::shared_ptr<mfem::ParMesh> pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, mesh);

    for (int lev = 0; lev < options.mesh.ref_par; lev++) {
        pmesh->UniformRefinement();
    }
    pmesh->SetAttributes();

    return pmesh;
}

std::map<int, int>
create_grains_to_map(const ExaOptions& options, const mfem::Array<int>& grains)
{
    std::map<int, int> grain2regions;

    if (!options.region_mapping_file) {
        for (const auto& item: grains) {
            const int key = item;
            grain2regions.emplace(key, 1);
        }
    }
    else {
        std::ifstream file(*options.region_mapping_file);

        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << *options.region_mapping_file << std::endl;
        }

        std::string line;
        int key, value;
        size_t lineNumber = 0;

        while (std::getline(file, line)) {
            ++lineNumber;
            if (line.empty()) {
                continue; // Skip empty lines
            }
            std::istringstream iss(line);
            if (!(iss >> key >> value)) {
                std::cerr << "Error reading data on line " << lineNumber << " key " << key << " line " << line << std::endl;
                continue;
            }
            // Insert into the map
            // Since keys are assumed to be unique, this won't overwrite any existing entry.
            grain2regions.emplace(key, value);
        }
        file.close();
    }

    return grain2regions;
}

/**
 * @brief Helper function to initialize deformation gradient QuadratureFunction to identity
 */
void initializeDeformationGradientToIdentity(mfem::expt::PartialQuadratureFunction& defGrad) {
    // This function would need to be implemented to properly initialize
    // a 9-component QuadratureFunction representing 3x3 identity matrices
    // at each quadrature point
    
    double* data = defGrad.HostReadWrite();
    const int npts = defGrad.Size() / defGrad.GetVDim();
    
    // Initialize each 3x3 matrix to identity
    for (int i = 0; i < npts; i++) {
        double* mat = &data[i * 9];
        // Set to identity: [1,0,0,0,1,0,0,0,1]
        mat[0] = 1.0; mat[1] = 0.0; mat[2] = 0.0;  // first row
        mat[3] = 0.0; mat[4] = 1.0; mat[5] = 0.0;  // second row  
        mat[6] = 0.0; mat[7] = 0.0; mat[8] = 1.0;  // third row
    }
}

} // end namespace

SimulationState::SimulationState(ExaOptions& options) : m_time_manager(options), m_options(options), class_device(options.solvers.rtmodel) 
{
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    m_time_manager = TimeManagement(options);
    m_mesh = ::makeMesh(options, my_id);
    ::setupBoundaryConditions(options);
    // m_bc_manager = BCManager::getInstance();

    // Set-up the mesh FEC and PFES
    {
        const int space_dim = m_mesh->SpaceDimension();
        std::string mesh_fec_str = "H1_" + std::to_string(space_dim) + "D_P" + std::to_string(options.mesh.order); 
        m_map_fec[mesh_fec_str] = std::make_shared<mfem::H1_FECollection>(options.mesh.order, space_dim);
        m_mesh_fes = std::make_shared<mfem::ParFiniteElementSpace>(m_mesh.get(), m_map_fec[mesh_fec_str].get(), space_dim);
    }

    // Set-up our various mesh nodes / mesh QoI and
    // primal variables
    {
        // Create our mesh nodes
        m_mesh_nodes["mesh_current"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());
        // Create our mesh nodes
        m_mesh_nodes["mesh_t_beg"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());
        // Create our mesh nodes
        m_mesh_nodes["mesh_ref"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());

        // Set them to the current default vaules
        m_mesh->GetNodes(*m_mesh_nodes["mesh_current"]);
        (*m_mesh_nodes["mesh_t_beg"]) = *m_mesh_nodes["mesh_current"];
        (*m_mesh_nodes["mesh_ref"]) = *m_mesh_nodes["mesh_current"];

        {
            mfem::GridFunction *nodes = m_mesh_nodes["mesh_current"].get(); // set a nodes grid function to global current configuration
            int owns_nodes = 0;
            m_mesh->SwapNodes(nodes, owns_nodes); // m_mesh has current configuration nodes
            delete nodes;
        }

        m_mesh_qoi_nodes["displacement"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());

        m_mesh_qoi_nodes["velocity"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());

        (*m_mesh_qoi_nodes["displacement"]) = 0.0;
        (*m_mesh_qoi_nodes["velocity"]) = 0.0;
        // This is our velocity field
        m_primal_field = std::make_shared<mfem::Vector>(m_mesh_fes->TrueVSize()); m_primal_field->UseDevice(true);
        m_primal_field_prev = std::make_shared<mfem::Vector>(m_mesh_fes->TrueVSize()); m_primal_field_prev->UseDevice(true);
        (*m_primal_field) = 0.0;
        (*m_primal_field_prev) = 0.0;
    }

    {
        const int space_dim = m_mesh->SpaceDimension();
        std::string l2_fec_str = "L2_" + std::to_string(space_dim) + "D_P" + std::to_string(0);
        m_map_fec[l2_fec_str] = std::make_shared<mfem::L2_FECollection>(0, space_dim);
    }

    // Global QuadratureSpace Setup and QFs
    const int int_order = 2 * options.mesh.order + 1;
    {
        mfem::Array<bool> global_index;
        m_map_qs["global"] = std::make_shared<mfem::expt::PartialQuadratureSpace>(m_mesh, int_order, global_index);

        m_map_qs["global_ord_0"] = std::make_shared<mfem::expt::PartialQuadratureSpace>(m_mesh, 1, global_index);

        m_map_qfs["cauchy_stress_beg"] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs["global"], 6, 0.0);
        m_map_qfs["cauchy_stress_end"] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs["global"], 6, 0.0);

        m_map_qfs["cauchy_stress_beg"]->operator=(0.0);
        m_map_qfs["cauchy_stress_end"]->operator=(0.0);

        m_model_update_qf_pairs.push_back(std::make_pair("cauchy_stress_beg", "cauchy_stress_end"));

        auto kinetic_grads_name = GetQuadratureFunctionMapName("kinetic_grads", -1);
        m_map_qfs[kinetic_grads_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs["global"], 9, 0.0);
        ::initializeDeformationGradientToIdentity(*m_map_qfs[kinetic_grads_name]);

        auto tangent_stiffness_name = GetQuadratureFunctionMapName("tangent_stiffness", -1);
        m_map_qfs[tangent_stiffness_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs["global"], 36, 0.0);

    }

    // Material state variable and qspace setup
    {
        // Update our options file with the correct number of state variables now
        UpdateExaOptionsWithOrientationCounts();
        // create our region map now
        const int loc_nelems = m_mesh->GetNE();
        mfem::Array2D<bool> region_map(options.materials.size(), loc_nelems);
        region_map = false;
        m_grains = std::make_shared<mfem::Array<int>>(loc_nelems);

        for (int i = 0; i < loc_nelems; i++) {
            m_grains->operator[](i) = m_mesh->GetAttribute(i);
        }

        const auto grains2region = ::create_grains_to_map(options, (*m_grains));

        for (int i = 0; i < loc_nelems; i++) {
            const int grain_id = m_grains->operator[](i);
            const int region_id = grains2region.at(grain_id);
            m_mesh->SetAttribute(i, region_id);
            region_map(region_id - 1, i) = true;
        }

        // update all of our attributes
        m_mesh->SetAttributes();

        for (auto matl : options.materials) {
            const int region_id = matl.region_id;
            m_region_material_type.push_back(matl.mech_type);
            m_material_name_region.push_back(std::make_pair(matl.material_name, region_id));
            std::string qspace_name = GetRegionName(region_id);

            m_material_properties.emplace(qspace_name, matl.properties.properties);
            mfem::Array<bool> loc_index(region_map.GetRow(region_id), loc_nelems, false);

            m_map_qs[qspace_name] = std::make_shared<mfem::expt::PartialQuadratureSpace>(m_mesh, int_order, loc_index);

            auto state_var_beg_name = GetQuadratureFunctionMapName("state_var_beg", region_id);
            auto state_var_end_name = GetQuadratureFunctionMapName("state_var_end", region_id);
            auto cauchy_stress_beg_name = GetQuadratureFunctionMapName("cauchy_stress_beg", region_id);
            auto cauchy_stress_end_name = GetQuadratureFunctionMapName("cauchy_stress_end", region_id);
            auto tangent_stiffness_name = GetQuadratureFunctionMapName("tangent_stiffness", region_id);
            auto vm_name = GetQuadratureFunctionMapName("von_mises", region_id);


            m_map_qfs[state_var_beg_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], matl.state_vars.num_vars, 0.0);
            m_map_qfs[state_var_end_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], matl.state_vars.num_vars, 0.0);
            m_map_qfs[cauchy_stress_beg_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], 6, 0.0);
            m_map_qfs[cauchy_stress_end_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], 6, 0.0);
            m_map_qfs[tangent_stiffness_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], 36, 0.0);
            m_map_qfs[vm_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], 1, 0.0);

            m_map_qfs[state_var_beg_name]->operator=(0.0);
            m_map_qfs[state_var_end_name]->operator=(0.0);
            m_map_qfs[cauchy_stress_beg_name]->operator=(0.0);
            m_map_qfs[cauchy_stress_end_name]->operator=(0.0);
            m_map_qfs[tangent_stiffness_name]->operator=(0.0);
            m_map_qfs[vm_name]->operator=(0.0);

            if (matl.mech_type == MechType::UMAT) {
                auto def_grad_name = GetQuadratureFunctionMapName("def_grad_beg", region_id);
                m_map_qfs[def_grad_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], 9, 0.0);
                ::initializeDeformationGradientToIdentity(*m_map_qfs[def_grad_name]);
            }

            m_model_update_qf_pairs.push_back(std::make_pair(state_var_beg_name, state_var_end_name));
            m_model_update_qf_pairs.push_back(std::make_pair(cauchy_stress_beg_name, cauchy_stress_end_name));

        }
    }
    InitializeStateVariables();
}

// Modified InitializeStateVariables to load shared orientation data first
void SimulationState::InitializeStateVariables() {
    // Create grain to region mapping
    std::map<int, int> grains2region = create_grains_to_map(m_options, *m_grains);
    
    // First, load shared orientation data if any material needs it
    for (const auto& material : m_options.materials) {
        if (material.grain_info.has_value() && material.grain_info->orientation_file.has_value()) {
            if (!LoadSharedOrientationData(material.grain_info->orientation_file.value(), 
                                         material.grain_info->num_grains)) {
                if (my_id == 0) {
                    std::cerr << "Failed to load shared orientation data from " 
                              << material.grain_info->orientation_file.value() << std::endl;
                }
                return;
            }
            break; // Only need to load once since it's shared
        }
    }
    
    // Initialize state variables for each material region
    for (size_t i = 0; i < m_options.materials.size(); ++i) {
        const auto& material = m_options.materials[i];
        InitializeRegionStateVariables(material.region_id, material, grains2region);
    }
    
    // Clean up shared orientation data after all regions are initialized
    CleanupSharedOrientationData();
}

// Refactored InitializeRegionStateVariables function
void SimulationState::InitializeRegionStateVariables(int region_id, 
                                                    const MaterialOptions& material,
                                                    const std::map<int, int>& grains2region) {
    // Get the state variable QuadratureFunction for this region
    auto state_var_beg_name = GetQuadratureFunctionMapName("state_var_beg", region_id);
    auto state_var_qf = m_map_qfs[state_var_beg_name];
    
    // Get the QuadratureSpace for this region
    std::string qspace_name = GetRegionName(region_id);
    auto qspace = m_map_qs[qspace_name];
    // Prepare orientation data for this region (convert from shared quaternions if needed)
    OrientationConfig orientation_config = PrepareOrientationForRegion(material);

    // Calculate effective state variable count (includes orientations)
    const int effective_state_var_size = material.state_vars.num_vars;
    const int base_state_var_size = material.state_vars.num_vars - orientation_config.stride;
    
    // Load base state variable initial values
    std::vector<double> state_var_data;
    if (!material.state_vars.initial_values.empty()) {
        state_var_data = material.state_vars.initial_values;
    } else if (!material.state_vars.state_file.empty()) {
        // Load from file if not already loaded
        std::ifstream file(material.state_vars.state_file);
        if (!file.is_open()) {
            if (my_id == 0) {
                std::cerr << "Error: Cannot open state variables file: " 
                          << material.state_vars.state_file << std::endl;
            }
            return;
        }
        
        double value;
        while (file >> value) {
            state_var_data.push_back(value);
        }
        file.close();
    }
    
    // Validate state variable data size
    if (state_var_data.size() != static_cast<size_t>(base_state_var_size) ) {
        if (my_id == 0) {
            std::cerr << "Warning: State variable data size (" << state_var_data.size() 
                      << ") doesn't match expected size (" << base_state_var_size 
                      << ") for material " << material.material_name << std::endl;
        }
    }
    
    // Get the data pointer for the QuadratureFunction
    double* qf_data = state_var_qf->HostReadWrite();
    const int qf_vdim = state_var_qf->GetVDim();
    
    // Validate that our total size matches
    if (qf_vdim != effective_state_var_size) {
        if (my_id == 0) {
            std::cerr << "Error: QuadratureFunction vdim (" << qf_vdim 
                      << ") doesn't match effective total size (" << effective_state_var_size 
                      << ") for material " << material.material_name << std::endl;
        }
        return;
    }
    
    // Get the local to global element mapping for this region
    const auto& local2global = qspace->getLocal2Global();
    const int num_local_elements = qspace->getNumLocalElements();
    
    // Loop over local elements in this region
    for (int local_elem = 0; local_elem < num_local_elements; ++local_elem) {
        const int global_elem = local2global[local_elem];
        
        // Get the grain ID for this element (before region mapping)
        const int grain_id = m_grains->operator[](global_elem);
        
        // Verify this element belongs to the current region
        const int elem_region = grains2region.at(grain_id);
        if (elem_region != (region_id + 1)) { // grains2region uses 1-based indexing
            continue; // Skip elements that don't belong to this region
        }
        
        // Get the integration rule for this element
        const mfem::IntegrationRule* ir = &(state_var_qf->GetSpace()->GetIntRule(local_elem));
        const int num_qpts = ir->GetNPoints();
        
        // Loop over quadrature points in this element
        for (int qpt = 0; qpt < num_qpts; ++qpt) {
            // Calculate the base index for this quadrature point's data
            const int qpt_base_index = (local_elem * num_qpts + qpt) * qf_vdim;
            
            // Fill state variables
            int state_var_idx = 0;
            for (int k = 0; k < qf_vdim; ++k) {
                // Check if this component is NOT orientation data
                if (orientation_config.is_valid && 
                    k > orientation_config.offset_start && k < orientation_config.offset_end) {
                    // Skip orientation components - they'll be filled separately
                    continue;
                } else {
                    // This is state variable data
                    double var_data = 0.0;
                    if (state_var_idx < static_cast<int>(state_var_data.size())) {
                        var_data = state_var_data[state_var_idx];
                    } else if (my_id == 0) {
                        std::cerr << "Warning: Missing state variable data, component " 
                                  << state_var_idx << std::endl;
                    }
                    qf_data[qpt_base_index + k] = var_data;
                    state_var_idx++;
                }
            }
            
            // Fill orientation data (converted to format required by this material)
            FillOrientationData(qf_data, qpt_base_index, qf_vdim, grain_id, orientation_config);
        }
    }
    
    if (my_id == 0) {
        std::cout << "Initialized state variables for material " << material.material_name 
                  << " (region " << region_id << ")" << std::endl;
        if (orientation_config.is_valid) {
            const auto& grain_info = material.grain_info.value();
            std::string format_name = "custom";
            if (grain_info.ori_type == OriType::QUAT) format_name = "quaternions";
            else if (grain_info.ori_type == OriType::EULER) format_name = "Euler angles";
            else if (grain_info.ori_type == OriType::CUSTOM && orientation_config.stride == 9) format_name = "rotation matrices";
            
            std::cout << "  - Converted orientation data to " << format_name 
                      << " (stride " << orientation_config.stride << ")" << std::endl;
        }
    }
}

// Additional utility function to update ExaOptions with correct state variable counts
void SimulationState::UpdateExaOptionsWithOrientationCounts() {
    for (auto& material : m_options.materials) {
        int effective_count = CalculateEffectiveStateVarCount(material);
        if (effective_count != material.state_vars.num_vars) {
            if (my_id == 0) {
                std::cout << "Updated state variable count for material " 
                          << material.material_name << " from " 
                          << material.state_vars.num_vars << " to " 
                          << effective_count << " (includes orientations)" << std::endl;
            }
            material.state_vars.num_vars = effective_count;
        }
    }
}

int SimulationState::CalculateEffectiveStateVarCount(const MaterialOptions& material) {
    int base_count = material.state_vars.num_vars;
    
    if (material.grain_info.has_value()) {
        const auto& grain_info = material.grain_info.value();
        
        // Add orientation variables based on what format this material needs
        int orientation_vars = 0;
        if (grain_info.ori_type == OriType::QUAT) {
            orientation_vars = 4;  // Quaternions
        } else if (grain_info.ori_type == OriType::EULER) {
            orientation_vars = 3;  // Euler angles
        } else if (grain_info.ori_type == OriType::CUSTOM) {
            orientation_vars = grain_info.ori_stride;  // Custom stride
        }
        
        base_count += orientation_vars;
    }
    
    return base_count;
}

bool SimulationState::LoadSharedOrientationData(const std::string& orientation_file, int num_grains) {
    if (m_shared_orientation_data.is_loaded) {
        // Already loaded, just verify grain count matches
        if (m_shared_orientation_data.num_grains != num_grains) {
            if (my_id == 0) {
                std::cerr << "Error: Grain count mismatch. Expected " << num_grains 
                          << " but shared data has " << m_shared_orientation_data.num_grains << std::endl;
            }
            return false;
        }
        return true;
    }
    
    std::ifstream orient_file(orientation_file);
    if (!orient_file.is_open()) {
        if (my_id == 0) {
            std::cerr << "Error: Cannot open orientation file: " << orientation_file << std::endl;
        }
        return false;
    }
    
    // Load unit quaternions (passive rotations from crystal to sample reference)
    const int expected_size = 4 * num_grains;  // Always 4 components per quaternion
    m_shared_orientation_data.quaternions.reserve(expected_size);
    
    double value;
    while (orient_file >> value && m_shared_orientation_data.quaternions.size() < static_cast<size_t>(expected_size)) {
        m_shared_orientation_data.quaternions.push_back(value);
    }
    orient_file.close();
    
    if (m_shared_orientation_data.quaternions.size() != static_cast<size_t>(expected_size)) {
        if (my_id == 0) {
            std::cerr << "Error: Orientation file size (" << m_shared_orientation_data.quaternions.size() 
                      << ") doesn't match expected size (" << expected_size 
                      << ") for " << num_grains << " grains" << std::endl;
        }
        m_shared_orientation_data.quaternions.clear();
        return false;
    }
    
    // Validate that quaternions are properly normalized
    for (int i = 0; i < num_grains; ++i) {
        const int base_idx = i * 4;
        const double w = m_shared_orientation_data.quaternions[base_idx];
        const double x = m_shared_orientation_data.quaternions[base_idx + 1];
        const double y = m_shared_orientation_data.quaternions[base_idx + 2];
        const double z = m_shared_orientation_data.quaternions[base_idx + 3];
        
        const double norm = sqrt(w*w + x*x + y*y + z*z);
        const double tolerance = 1e-6;
        
        if (fabs(norm - 1.0) > tolerance) {
            if (my_id == 0) {
                std::cerr << "Warning: Quaternion " << i << " is not normalized (norm = " 
                          << norm << "). Normalizing..." << std::endl;
            }
            // Normalize the quaternion
            m_shared_orientation_data.quaternions[base_idx] = w / norm;
            m_shared_orientation_data.quaternions[base_idx + 1] = x / norm;
            m_shared_orientation_data.quaternions[base_idx + 2] = y / norm;
            m_shared_orientation_data.quaternions[base_idx + 3] = z / norm;
        }
    }
    
    m_shared_orientation_data.num_grains = num_grains;
    m_shared_orientation_data.is_loaded = true;
    
    if (my_id == 0) {
        std::cout << "Loaded shared orientation data: " << num_grains 
                  << " unit quaternions (passive rotations)" << std::endl;
    }
    
    return true;
}

void SimulationState::CleanupSharedOrientationData() {
    m_shared_orientation_data.quaternions.clear();
    m_shared_orientation_data.quaternions.shrink_to_fit();
    m_shared_orientation_data.num_grains = 0;
    m_shared_orientation_data.is_loaded = false;
    
    if (my_id == 0) {
        std::cout << "Cleaned up shared orientation data to free memory" << std::endl;
    }
}

std::vector<double> SimulationState::ConvertQuaternionsToEuler(const std::vector<double>& quaternions, int num_grains) {
    std::vector<double> euler_angles;
    euler_angles.reserve(num_grains * 3);

    auto bunge_func = [](const double* const quat, double* bunge_ang) {
        // below is equivalent to std::sqrt(std::numeric_limits<T>::epsilon);
        constexpr double tol = 1.4901161193847656e-08;
        const auto q03 = quat[0] * quat[0] + quat[3] * quat[3];
        const auto q12 = quat[1] * quat[1] + quat[2] * quat[2];
        const auto xi = std::sqrt(q03 * q12);
        //We get to now go through all of the different cases that this might break down into
        if (std::abs(xi) < tol && std::abs(q12) < tol) {
            bunge_ang[0] = std::atan2(-2.0 * quat[0] * quat[3], quat[0] * quat[0] - quat[3] * quat[3]);
            //All of the other values are zero
        }else if (std::abs(xi) < tol && std::abs(q03) < tol) {
            bunge_ang[0] = std::atan2(2.0 * quat[1] * quat[2], quat[1] * quat[1] - quat[2] * quat[2]);
            bunge_ang[1] = 3.141592653589793;
            //The other value is zero
        }else{
            const double inv_xi = 1.0 / xi;
            //The atan2 terms are pretty long so we're breaking it down into a couple of temp terms
            const double t1 = inv_xi * (quat[1] * quat[3] - quat[0] * quat[2]);
            const double t2 = inv_xi * (-quat[0] * quat[1] - quat[2] * quat[3]);
            //We can now assign the first two bunge angles
            bunge_ang[0] = std::atan2(t1, t2);
            bunge_ang[1] = std::atan2(2.0 * xi, q03 - q12);
            //Once again these terms going into the atan2 term are pretty long
            {
                const double t1 = inv_xi * (quat[0] * quat[2] + quat[1] * quat[3]);
                const double t2 = inv_xi * (quat[2] * quat[3] - quat[0] * quat[1]);
                //We can finally find the final bunge angle
                bunge_ang[2] = std::atan2(t1, t2);
            }
        }
    };
    
    for (int i = 0; i < num_grains; ++i) {
        const int base_idx = i * 4;
        const double* const quat = &(quaternions.data()[base_idx]);
        double bunge[3] = {};

        bunge_func(quat, bunge);
        
        euler_angles.push_back(bunge[0]);
        euler_angles.push_back(bunge[1]);
        euler_angles.push_back(bunge[2]);
    }
    
    return euler_angles;
}

std::vector<double> SimulationState::ConvertQuaternionsToMatrix(const std::vector<double>& quaternions, int num_grains) {
    std::vector<double> matrices;
    matrices.reserve(num_grains * 9);
    
    for (int i = 0; i < num_grains; ++i) {
        const int base_idx = i * 4;
        const double* const quat = &(quaternions.data()[base_idx]);

        const double qbar =  quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

        // Row-major order: [r11, r12, r13, r21, r22, r23, r31, r32, r33]
        matrices.push_back(qbar + 2.0 * quat[1] * quat[1]);
        matrices.push_back(2.0 * (quat[1] * quat[2] - quat[0] * quat[3]));
        matrices.push_back(2.0 * (quat[1] * quat[3] + quat[0] * quat[2]));

        matrices.push_back(2.0 * (quat[1] * quat[2] + quat[0] * quat[3]));
        matrices.push_back(qbar + 2.0 * quat[2] * quat[2]);
        matrices.push_back(2.0 * (quat[2] * quat[3] - quat[0] * quat[1]));

        matrices.push_back(2.0 * (quat[1] * quat[3] - quat[0] * quat[2]));
        matrices.push_back(2.0 * (quat[2] * quat[3] + quat[0] * quat[1]));
        matrices.push_back(qbar + 2.0 * quat[3] * quat[3]);
    }
    
    return matrices;
}

SimulationState::OrientationConfig SimulationState::PrepareOrientationForRegion(const MaterialOptions& material) {
    OrientationConfig config;
    
    if (!material.grain_info.has_value() || !m_shared_orientation_data.is_loaded) {
        return config; // Return invalid config
    }
    
    const auto& grain_info = material.grain_info.value();
    
    // Verify grain count consistency
    if (m_shared_orientation_data.num_grains != grain_info.num_grains) {
        if (my_id == 0) {
            std::cerr << "Error: Grain count mismatch for material " << material.material_name 
                      << ". Expected " << grain_info.num_grains 
                      << " but shared data has " << m_shared_orientation_data.num_grains << std::endl;
        }
        return config;
    }
    
    // Convert shared quaternions to the format required by this material
    if (grain_info.ori_type == OriType::QUAT) {
        // Material needs quaternions - use shared data directly
        config.data = m_shared_orientation_data.quaternions;
        config.stride = 4;
        if (my_id == 0) {
            std::cout << "Using quaternion format for material " << material.material_name << std::endl;
        }
    } else if (grain_info.ori_type == OriType::EULER) {
        // Material needs Euler angles - convert from quaternions
        config.data = ConvertQuaternionsToEuler(m_shared_orientation_data.quaternions, grain_info.num_grains);
        config.stride = 3;
        if (my_id == 0) {
            std::cout << "Converted quaternions to Euler angles for material " << material.material_name << std::endl;
        }
    } else if (grain_info.ori_type == OriType::CUSTOM) {
        // Handle custom formats
        if (grain_info.ori_stride == 9) {
            // Assume custom format wants rotation matrices
            config.data = ConvertQuaternionsToMatrix(m_shared_orientation_data.quaternions, grain_info.num_grains);
            config.stride = 9;
            if (my_id == 0) {
                std::cout << "Converted quaternions to rotation matrices for material " << material.material_name << std::endl;
            }
        } else if (grain_info.ori_stride == 4) {
            // Custom format wants quaternions
            config.data = m_shared_orientation_data.quaternions;
            config.stride = 4;
            if (my_id == 0) {
                std::cout << "Using quaternion format for custom material " << material.material_name << std::endl;
            }
        } else {
            // Unsupported custom stride
            if (my_id == 0) {
                std::cerr << "Error: Unsupported custom orientation stride (" << grain_info.ori_stride 
                          << ") for material " << material.material_name << std::endl;
            }
            return config;
        }
    }
    
    // Calculate placement offsets
    auto offsets = CalculateOrientationOffsets(material, config.stride);
    config.offset_start = offsets.first;
    config.offset_end = offsets.second;
    config.is_valid = true;
    
    return config;
}

std::pair<int, int> SimulationState::CalculateOrientationOffsets(const MaterialOptions& material, int orientation_stride) {
    if (!material.grain_info.has_value() || orientation_stride == 0) {
        return {-1, 0};
    }
    
    const auto& grain_info = material.grain_info.value();
    const int state_var_size = material.state_vars.num_vars;
    
    int offset_start, offset_end;
    
    if (grain_info.ori_state_var_loc < 0) {
        // Put orientation data at the end
        if (my_id == 0) {
            std::cout << "Note: Orientation data placed at end of state variable array "
                      << "for material " << material.material_name << std::endl;
        }
        offset_start = state_var_size - 1;
        offset_end = state_var_size + orientation_stride;
    } else if (grain_info.ori_state_var_loc == 0) {
        // Put orientation data at the beginning
        offset_start = -1;
        offset_end = orientation_stride;
    } else {
        // Put orientation data at specified location
        offset_start = grain_info.ori_state_var_loc - 1;
        offset_end = grain_info.ori_state_var_loc + orientation_stride;
    }
    
    return {offset_start, offset_end};
}

void SimulationState::FillOrientationData(double* qf_data, int qpt_base_index, int qf_vdim, 
                                         int grain_id, const OrientationConfig& orientation_config) {
    if (!orientation_config.is_valid || orientation_config.stride == 0) {
        return; // No orientation data to fill
    }
    
    for (int k = 0; k < qf_vdim; ++k) {
        if (k > orientation_config.offset_start && k < orientation_config.offset_end) {
            // This is orientation data
            const int grain_idx = k - orientation_config.offset_start - 1;
            const int orient_idx = orientation_config.stride * (grain_id - 1) + grain_idx;
            
            if (orient_idx < static_cast<int>(orientation_config.data.size())) {
                qf_data[qpt_base_index + k] = orientation_config.data[orient_idx];
            } else {
                qf_data[qpt_base_index + k] = 0.0; // Default value if data is missing
                if (my_id == 0) {
                    std::cerr << "Warning: Missing orientation data for grain " 
                              << grain_id << ", component " << grain_idx << std::endl;
                }
            }
        }
    }
}