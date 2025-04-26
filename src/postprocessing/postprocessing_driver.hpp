#pragma once

#include "mfem.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "mfem_expt/partial_qfunc.hpp"

#include <string>
#include <map>
#include <memory>
#include <functional>
#include <vector>

// None of this we should have to worry about for restart functions
class PostProcessingDriver
{
    private:
        // Need a few mapping functions to hold the stuff relevant
        // to our datacollections / post-processing routines
        std::map<std::string, std::unique_ptr<mfem::ParGridFunction>> m_map_gfs;
        std::map<int, std::shared_ptr<mfem::ParFiniteElementSpace>>  m_map_gfs_pfes;
        std::map<std::string, std::function<void(const int phase)>> m_map_gfs_fcns;
        std::map<std::string, std::unique_ptr<mfem::DataCollection>> m_map_dcs;
        // Does this need to be a map? probably?
        std::unique_ptr<mfem::QuadratureFunction> m_evec;
        // Our volume average / integrated quantity files just require a base file name
        // The other portions of the name are generated on the fly and include information such
        // as the material name and its phase number if its a material specific quantity 
        std::filesystem::file_path m_avg_filepath_base;
        std::map<std::string, std::function<void(const int phase, const double time)>> m_map_avg_fcns;
        // We only need a const reference to our material model aspect of things
        // as we shouldn't ever be changing things related to it
        // Only need this for the GetQFMapping aspect of things
        // Probably want to do something like a
        // std::map<std::string, std::shared_ptr<std::unordered_map<std::string, std::pair<int, int>>> >
        // m_map_qf_mappings;
        const SimulationState& m_sim_state;
        const int m_mpi_rank;

    public:

        PostProcessingDriver(SimulationState& sim_state, ExaOptions &options);
        ~PostProcessingDriver();

        // Primary function used to update our data collections
        // and print out any volume integrated/average
        void Update(const int step, const double time);
        void PrintVolValues(const double time);
        void UpdateDataCollections(const int step, const double time);

    private:

        // Our various volume average/integrated quantities that users can output
        // These quantities are all none material specific and so would generate
        // one file per type
        void VolumeAvgStress(const int /* phase */, const double /* time */);
        void VolumeAvgEulerStrain(const int /* phase */, const double /* time */);
        void VolumeAvgDefGrad(const int /* phase */,const double /* time */);

        // The below are all material specific and will have their own material specific file outputted
        // So, they will really be material specific volume calculations.
        //
        // We should have this print out the volume as well for the plastic work
        // From there we would be able to derive any other volume average/etc type calculations if need be
        void VolumePlWork(const int phase, const double /* time */);
        // This one will require us to rotate  strains from crystal to sample frame
        void VolumeAvgElasticStrain(const int phase, const double /* time */);

        void CalcElementAvg(mfem::Vector *elemVal, const mfem::QuadratureFunction *qf);
        // Our various projection methods that we need to update our models with
        // First group are projection models available to every model
        void ProjectCentroid(const int /* phase */);
        void ProjectVolume(const int /* phase */);
        void ProjectModelStress(const int /* phase */);
        void ProjectVonMisesStress(const int /* phase */);
        void ProjectHydroStress(const int /* phase */);

        // All of these would output phase specific quantities
        // These next group of Project* functions are only available with ExaCMech type models
        void ProjectDpEff(const int phase);
        void ProjectEffPlasticStrain(const int phase);
        void ProjectShearRate(const int phase);
        // This one requires that the orientations be made unit normals afterwards
        void ProjectOrientation(const int phase);
        // Here this can be either the CRSS for a voce model or relative dislocation density
        // value for the MTS model.
        void ProjectH(const int phase);
        // This one requires that the deviatoric strain be converted from 5d rep to 6d
        // and have vol. contribution added.
        void ProjectElasticStrains(const int phase);
};

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

