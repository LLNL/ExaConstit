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

// In simulation_state.cpp - add these method implementations:

void SimulationState::InitializeStateVariables() {
    // Create the grain to region mapping
    const auto grains2region = ::create_grains_to_map(m_options, *m_grains);
    
    // Initialize state variables for each material region
    for (size_t i = 0; i < m_options.materials.size(); ++i) {
        const auto& material = m_options.materials[i];
        InitializeRegionStateVariables(material.region_id, material, grains2region);
    }
}

void SimulationState::InitializeRegionStateVariables(int region_id, 
                                                    const MaterialOptions& material,
                                                    const std::map<int, int>& grains2region) {
    // Get the state variable QuadratureFunction for this region
    auto state_var_beg_name = GetQuadratureFunctionMapName("state_var_beg", region_id);
    auto state_var_qf = m_map_qfs[state_var_beg_name];
    
    // Get the QuadratureSpace for this region
    std::string qspace_name = GetRegionName(region_id);
    auto qspace = m_map_qs[qspace_name];
    
    const int state_var_size = material.state_vars.num_vars;
    
    // Load state variable initial values
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
    if (state_var_data.size() != static_cast<size_t>(state_var_size)) {
        if (my_id == 0) {
            std::cerr << "Warning: State variable data size (" << state_var_data.size() 
                      << ") doesn't match expected size (" << state_var_size 
                      << ") for material " << material.material_name << std::endl;
        }
    }
    
    // Load orientation data if grain information is provided
    std::vector<double> orientation_data;
    int orientation_stride = 0;
    int orientation_offset = -1;
    
    if (material.grain_info.has_value()) {
        const auto& grain_info = material.grain_info.value();
        orientation_offset = grain_info.ori_state_var_loc;
        orientation_stride = grain_info.ori_stride;
        
        // Load orientation data from file
        if (grain_info.orientation_file.has_value()) {
            std::ifstream orient_file(grain_info.orientation_file.value());
            if (!orient_file.is_open()) {
                if (my_id == 0) {
                    std::cerr << "Error: Cannot open orientation file: " 
                              << grain_info.orientation_file.value() << std::endl;
                }
                return;
            }
            
            const int expected_size = orientation_stride * grain_info.num_grains;
            double value;
            while (orient_file >> value && orientation_data.size() < static_cast<size_t>(expected_size)) {
                orientation_data.push_back(value);
            }
            orient_file.close();
            
            if (orientation_data.size() != static_cast<size_t>(expected_size)) {
                if (my_id == 0) {
                    std::cerr << "Warning: Orientation data size (" << orientation_data.size() 
                              << ") doesn't match expected size (" << expected_size 
                              << ") for material " << material.material_name << std::endl;
                }
            }
        }
    }
    
    // Determine where to place orientation data in the state variable array
    int offset1, offset2;
    if (orientation_stride == 0) {
        // No orientation data
        offset1 = -1;
        offset2 = 0;
    } else if (orientation_offset < 0) {
        // Put orientation data at the end
        if (my_id == 0) {
            std::cout << "Warning: Orientation data placed at end of state variable array "
                      << "for material " << material.material_name << std::endl;
        }
        offset1 = state_var_size - 1;
        offset2 = state_var_size + orientation_stride;
    } else if (orientation_offset == 0) {
        // Put orientation data at the beginning
        offset1 = -1;
        offset2 = orientation_stride;
    } else {
        // Put orientation data at specified location
        offset1 = orientation_offset - 1;
        offset2 = orientation_offset + orientation_stride;
    }
    
    // Get the data pointer for the QuadratureFunction
    double* qf_data = state_var_qf->HostReadWrite();
    const int qf_vdim = state_var_qf->GetVDim();
    
    // Validate that our total size matches
    const int expected_total_size = state_var_size;
    if (qf_vdim != expected_total_size) {
        if (my_id == 0) {
            std::cerr << "Error: QuadratureFunction vdim (" << qf_vdim 
                      << ") doesn't match expected total size (" << expected_total_size 
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
        
        // Calculate the element offset in the QuadratureFunction data
        // Note: QuadratureFunction data is organized as [vdim components for qpt0, vdim components for qpt1, ...]
        //       for each element sequentially
        
        // Loop over quadrature points in this element
        for (int qpt = 0; qpt < num_qpts; ++qpt) {
            // Calculate the base index for this quadrature point's data
            // For partial QuadratureFunctions, elements are stored sequentially by local element index
            const int qpt_base_index = (local_elem * num_qpts + qpt) * qf_vdim;
            
            // Fill state variables and orientation data
            int grain_idx = 0;
            int state_var_idx = 0;
            
            for (int k = 0; k < qf_vdim; ++k) {
                double var_data;
                
                // Determine if this component is orientation or state variable data
                if (orientation_stride > 0 && k > offset1 && k < offset2) {
                    // This is orientation data
                    const int orient_idx = orientation_stride * (grain_id - 1) + grain_idx;
                    if (orient_idx < static_cast<int>(orientation_data.size())) {
                        var_data = orientation_data[orient_idx];
                    } else {
                        var_data = 0.0; // Default value if data is missing
                        if (my_id == 0) {
                            std::cerr << "Warning: Missing orientation data for grain " 
                                      << grain_id << ", component " << grain_idx << std::endl;
                        }
                    }
                    grain_idx++;
                } else {
                    // This is state variable data
                    if (state_var_idx < static_cast<int>(state_var_data.size())) {
                        var_data = state_var_data[state_var_idx];
                    } else {
                        var_data = 0.0; // Default value if data is missing
                        if (my_id == 0) {
                            std::cerr << "Warning: Missing state variable data, component " 
                                      << state_var_idx << std::endl;
                        }
                    }
                    state_var_idx++;
                }
                
                qf_data[qpt_base_index + k] = var_data;
            }
        }
    }
    
    if (my_id == 0) {
        std::cout << "Initialized state variables for material " << material.material_name 
                  << " (region " << region_id << ")" << std::endl;
        if (material.grain_info.has_value()) {
            std::cout << "  - Included orientation data with stride " << orientation_stride << std::endl;
        }
    }
}