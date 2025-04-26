#pragma once

#include "mfem.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "mfem_expt/partial_qfunc.hpp"

#include <string>
#include <map>
#include <memory>
#include <functional>
#include <vector>

class SimulationState
{
private:
    // All the various quantities related to our simulations
    // aka the mesh, quadrature functions, finite element spaces,
    // mesh nodes, and various things related to our material systems
    
    // We might eventually need to make this a map or have a LOR version
    // if we decide to map our quadrature function data from a HOR set to a
    // LOR version to make visualizations easier...
    std::shared_ptr<mfem::ParMesh> m_mesh;
    // Get the PFES associated with the mesh
    // The same as below goes for the above as well
    std::shared_ptr<mfem::ParFiniteElementSpace> m_mesh_fes; 
    // Map of the QuadratureSpaceBase associated with a given name
    // These QuadratureSpaceBase might also be the PartialQuadratureSpace objects
    std::map<std::string, std::shared_ptr<mfem::expt::PartialQuadratureSpace>> m_map_qs;
    // Map of the QuadratureFunction associated with a given name
    // These QuadratureFunctions might also be a PartialQuadratureFunction class
    // for when we have have multiple materials in a simulation
    std::map<std::string, std::shared_ptr<mfem::expt::PartialQuadratureFunction>> m_map_qfs;
    // Map of the ParallelFiniteElementSpace associated with a given vector dimension
    std::map<int, std::shared_ptr<mfem::ParFiniteElementSpace>> m_map_pfes;
    // Map of the FiniteElementCollection associated with the typical FEC name
    // Typically would be something like L2_3D_P2 (FECTYPE _ #SPACEDIM D_P #MESHORDER)
    // The name is based on the name that MFEM prints out for along with any GridFunction that
    // tells us what FiniteElementCollection it belongs to
    std::map<std::string, std::shared_ptr<mfem::FiniteElementCollection>> m_map_fec;
    // Map of the mesh nodes associated with a given phase maybe?
    std::map<std::string, std::shared_ptr<mfem::ParGridFunction>> m_mesh_nodes;
    // Map of the material properties associated with a given phase name
    std::map<std::string, std::shared_ptr<mfem::Vector>> m_material_properties;
    // Vector of the material phase name and the phase index associated with it
    std::vector<std::pair<std::string, int>> m_material_name_phase;
    // Map of the quadrature function name to the potential offset in the quadrature function and
    // the vector dimension associated with that quadrature function name.
    // This variable is useful to obtain sub-mappings within a quadrature function used for all history variables
    // such as how it's done with ECMech's models.
    std::map<std::string, std::pair<int, int>> m_map_qf_mappings;

    // Vector of the names of the quadrature function pairs that have their data ptrs
    // swapped when UpdateModel() is called.
    std::vector<std::pair<std::string, std::string>> m_model_update_qf_pairs;

#if defined(EXACONSTIT_USE_AXOM)
    // We want this to be something akin to a axom::sidre::MFEMSidreDataCollection
    // However, we need it flexible enough to handle multiple different mesh topologies in it that
    // we might due to different mfem::SubMesh objects that correspond to each PartialQuadraturePoint
    std::unique_ptr<axom::sidre::DataStore> m_simulation_restart;
#endif
public:
    RTModel class_device;
public:
    SimulationState(ExaOptions& options);
    virtual ~SimulationState() = default;

    // This updates function does a simple pointer swap between the beginning and end time step values
    // of those variables that have been added by AddUpdateVariablePairNames
    void UpdateModel();
    // Mesh end coordinates need to be updated from her and not some other module
    void UpdateNodalEndCoords(const mfem::Vector& velocity, const double delta_time);
    // Returns the number of phases in the simulation
    int GetNumberOfPhases() const { return m_material_name_phase.size(); }

    // Add to the internal QF state pair mapping
    // While this is ideally material model specific info, it's quite useful for other
    // Models would largely be responsible for setting this all up
    void AddQuadratureFunctionStatePair(std::string_view state_name, std::pair<int, int> state_pair, const int phase);
    // A way to tell the class which beginning and end time step variables need to have internal
    // pointer values swapped when a call to UpdateModel is made.  
    void AddUpdateVariablePairNames(std::pair<std::string_view, std::string_view> update_var_pair);

    // This returns the correct mapping name for a quadrature function
    // If a phase is provided than the mapped name will have the material name associated with
    // the phase attached to it.
    // Phases start at 0, and a negative phase signals that a material name is not associated with things. 
    std::string GetQuadratureFunctionMapName(std::string_view& qf_name, const int phase = -1) const;
    // Must provide phase number of material we're dealing with in-order to output
    // correct quadrature function.
    // This will raise an error if a quadrature function name does not exist for a given phase
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> GetQuadratureFunction(std::string_view& qf_name, const int phase = -1);
    // Given the state variable name and phase ID we care about this returns the specific offset and vdim
    // associated with that name. Typically, you might use this when the state variable might live in a
    // larger QuadratureFunction
    std::pair<int, int> GetQuadratureFunctionStatePair(std::string_view& state_name, const int phase = -1) const;
    // Returns a pointer to a ParFiniteElementSpace (PFES) that's ordered according to VDIMs
    // and makes use of an L2 FiniteElementCollection
    // If the vdim is not in the internal mapping than a new PFES will be created
    std::shared_ptr<mfem::ParFiniteElementSpace> GetParFiniteElementSpace(const int vdim);
    // Gets the PFES associated with the mesh
    std::shared_ptr<mfem::ParFiniteElementSpace> GetMeshParFiniteElementSpace();

private:
};