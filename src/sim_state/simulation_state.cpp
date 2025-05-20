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
        for (const auto item: grains) {
            grain2regions[item] = 1;
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
                std::cerr << "Error reading data on line " << lineNumber << std::endl;
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

} // end namespace

SimulationState::SimulationState(ExaOptions& options) : m_time_manager(options), class_device(options.solvers.rtmodel) 
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

        m_model_update_qf_pairs.push_back(std::make_pair("cauchy_stress_beg", "cauchy_stress_end"));

    }

    // Material state variable and qspace setup
    {
        // create our region map now
        const int loc_nelems = m_mesh->GetNE();
        mfem::Array2D<bool> region_map(options.materials.size(), loc_nelems);
        region_map = false;
        m_grains = std::make_shared<mfem::Array<int>>(loc_nelems);
        // region numbers go from 0..N and are linearly increasing with no jumps in them
        std::copy(m_mesh->attributes.begin(), m_mesh->attributes.end(), m_grains->begin());

        const auto grains2region = ::create_grains_to_map(options, (*m_grains));

        for (int i = 0; i < loc_nelems; i++) {
            const int grain_id = (*m_grains)[i];
            const int region_id = grains2region.at(grain_id);
            m_mesh->attributes[i] = region_id;
            region_map(region_id, i) = true;
        }

        // update all of our attributes
        m_mesh->SetAttributes();

        for (auto matl : options.materials) {
            const int region_id = matl.region_id;
            m_material_properties[matl.material_name] = matl.properties.properties;
            m_material_name_region.push_back(std::make_pair(matl.material_name, region_id));
            mfem::Array<bool> loc_index(region_map.GetRow(region_id), loc_nelems, false);
            std::string qspace_name = GetRegionName(region_id);

            m_map_qs[qspace_name] = std::make_shared<mfem::expt::PartialQuadratureSpace>(m_mesh, int_order, loc_index);

            auto state_var_beg_name = GetQuadratureFunctionMapName("state_var_beg", region_id);
            auto state_var_end_name = GetQuadratureFunctionMapName("state_var_beg", region_id);

            m_map_qfs[state_var_beg_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], matl.state_vars.num_vars, 0.0);

            m_map_qfs[state_var_end_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], matl.state_vars.num_vars, 0.0);

            m_model_update_qf_pairs.push_back(std::make_pair(state_var_beg_name, state_var_end_name));
        }
    }
}