#pragma once

#include "options_parser.hpp"

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
        std::string m_avg_filepath_base;
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
        std::map<std::string, std::shared_ptr<mfem::QuadratureSpaceBase>> m_map_qs;
        // Map of the QuadratureFunction associated with a given name
        std::map<std::string, std::shared_ptr<mfem::QuadratureFunction>> m_map_qfs;
        // Map of the ParallelFiniteElementSpace associated with a given vector dimension
        std::map<int, std::shared_ptr<mfem::ParFiniteElementSpace>> m_map_pfes;
        // Map of the FiniteElementCollection associated with the typical FEC name
        // Typically would be something like L2_3D_P2 (FECTYPE _ #SPACEDIM D_P #MESHORDER)
        // We'll only ever use 
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
        std::unique_ptr<axom::sidre::MFEMSidreDataCollection> m_simulation_restart;
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
        void AddQuadratureFunctionStatePair(std::string state_name, std::pair<int, int> state_pair, const int phase);
        // A way to tell the class which beginning and end time step variables need to have internal
        // pointer values swapped when a call to UpdateModel is made.  
        void AddUpdateVariablePairNames(std::pair<std::string, std::string> update_var_pair);

        // This returns the correct mapping name for a quadrature function
        // If a phase is provided than the mapped name will have the material name associated with
        // the phase attached to it.
        // Phases start at 0, and a negative phase signals that a material name is not associated with things. 
        std::string GetQuadratureFunctionMapName(std::string& qf_name, const int phase = -1) const;
        // Must provide phase number of material we're dealing with in-order to output
        // correct quadrature function.
        // This will raise an error if a quadrature function name does not exist for a given phase
        std::shared_ptr<mfem::QuadratureFunction> GetQuadratureFunction(std::string& qf_name, const int phase = -1);
        // Given the state variable name and phase ID we care about this returns the specific offset and vdim
        // associated with that name. Typically, you might use this when the state variable might live in a
        // larger QuadratureFunction
        std::pair<int, int> GetQuadratureFunctionStatePair(std::string& state_name, const int phase = -1) const;
        // Returns a pointer to a ParFiniteElementSpace (PFES) that's ordered according to VDIMs
        // and makes use of an L2 FiniteElementCollection
        // If the vdim is not in the internal mapping than a new PFES will be created
        std::shared_ptr<mfem::ParFiniteElementSpace> GetParFiniteElementSpace(const int vdim);
        // Gets the PFES associated with the mesh
        std::shared_ptr<mfem::ParFiniteElementSpace> GetMeshParFiniteElementSpace();

    private:
};

class PartialQuadratureSpace : public mfem::QuadratureSpaceBase {
    private:
        // Size of partial values and maps the array index number to
        // the global processor local array value. This mapping is necessary to
        // map back to a QuadratureSpace / FaceQuadratureSpace.
        mfem::Array<int> local2global;
        // Size of global processor local indices which allows us to map
        // from the global processor local indices to the local indices
        // related to the partial QuadratureSpace.
        // We utilize negative values here such as -1 to correspond to a value
        // which are not utilized in our partial mapping.
        // Note, we could probably use a hash map here as well for the mapping if we
        // really wanted to but the linear mapping used does allow us to easily use this
        // on the GPU.
        mfem::Array2D<int> global2local;
protected:
   void ConstructOffsets();
   void Construct();

public:
    /// Create a PartialQuadratureSpace based on the global rules from #IntRules.
    PartialQuadratureSpace(Mesh *mesh_, int order_, mfem::Array<bool> partial_index);

    /// @brief Create a PartialQuadratureSpace with an IntegrationRule, valid only when
    /// the mesh has one element type.
    PartialQuadratureSpace(Mesh &mesh_, const IntegrationRule &ir, mfem::Array<bool> partial_index);

    /// Read a PartialQuadratureSpace from the stream @a in.
    /// Might just get rid of this version...
    PartialQuadratureSpace(Mesh *mesh_, std::istream &in);
public:

   /// Get the (element or face) transformation of entity @a idx.
   virtual ElementTransformation *GetTransformation(int idx) = 0;

   /// Return the geometry type of entity (element or face) @a idx.
   virtual Geometry::Type GetGeometry(int idx) const = 0;

   /// @brief Returns the permuted index of the @a iq quadrature point in entity
   /// @a idx.
   ///
   /// For tensor-product faces, returns the lexicographic index of the
   /// quadrature point, oriented relative to "element 1". For QuadratureSpace%s
   /// defined on elements (not faces), the permutation is trivial, and this
   /// returns @a iq.
   virtual int GetPermutedIndex(int idx, int iq) const = 0;

   /// Write the QuadratureSpace to the stream @a out.
   virtual void Save(std::ostream &out) const = 0;

   virtual ~QuadratureSpaceBase() { }

}

//
class PartialQuadratureFunction : public mfem::QuadratureFunction {
    private:
        // MFEM's QuadratureFunction utilizes a pointer here.
        // However, we don't need to follow MFEM's standard here and can be a bit smarter here
        // and utilize smart pointers for this sorta thing...
        std::shared_ptr<PartialQuadratureSpace> part_quad_space;

    public:

        /// Constructor function which takes in a shared ptr to our PartialQuadratureSpace, the vector dimension
        /// for this QF, and the default value we want to set for our projections to either GFs or QFs.
        PartialQuadratureSpace(std::shared_ptr<PartialQuadratureSpace>, int vdim, double default = -1.0);

        // Want to be safe here and return the QuadratureSpace in a sane manner
        // although I might need to revert this back eventuall....
        std::shared_ptr<PartialQuadratureSpace> GetSpace() const { return part_quad_space; }

        /// Set this equal to a constant value.
        PartialQuadratureFunction &operator=(double value);

        /// Copy the data from @a vec.
        /** The size of @a vec must be equal to the size of the associated
            QuadratureSpaceBase #qspace times the PartialQuadratureFunction vector
            dimension i.e. PartialQuadratureFunction::Size(). */
        PartialQuadratureFunction &operator=(const mfem::Vector &vec);

        /// Copy the data from @a qf.
        /** The VDIM and integration order of @a qf must be the same as
            PartialQuadratureFucntion and if not then this function will fail. */
        PartialQuadratureFunction &operator=(const mfem::QuadratureFunction &qf);

        /// Takes in a quadrature function and fill with either the values contained in this
        /// class or the default value provided by users.
        void FillQuadratureFunction(mfem::QuadratureFunction &qf);

        /// Return all values associated with mesh element @a idx in a Vector.
        /** The result is stored in the Vector @a values as a reference to the
            global values.

            Inside the Vector @a values, the index `i+vdim*j` corresponds to the
            `i`-th vector component at the `j`-th quadrature point.
        */
        inline void GetValues(int idx, Vector &values);

        /// Return all values associated with mesh element @a idx in a Vector.
        /** The result is stored in the Vector @a values as a copy of the
            global values.

            Inside the Vector @a values, the index `i+vdim*j` corresponds to the
            `i`-th vector component at the `j`-th quadrature point.
        */
        inline void GetValues(int idx, Vector &values) const;

        /// Return the quadrature function values at an integration point.
        /** The result is stored in the Vector @a values as a reference to the
            global values. */
        inline void GetValues(int idx, const int ip_num, Vector &values);

        /// Return the quadrature function values at an integration point.
        /** The result is stored in the Vector @a values as a copy to the
            global values. */
        inline void GetValues(int idx, const int ip_num, Vector &values) const;

        /// Return all values associated with mesh element @a idx in a DenseMatrix.
        /** The result is stored in the DenseMatrix @a values as a reference to the
            global values.

            Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
            `i`-th vector component at the `j`-th quadrature point.
        */
        inline void GetValues(int idx, DenseMatrix &values);

        /// Return all values associated with mesh element @a idx in a const DenseMatrix.
        /** The result is stored in the DenseMatrix @a values as a copy of the
            global values.

            Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
            `i`-th vector component at the `j`-th quadrature point.
        */
        inline void GetValues(int idx, DenseMatrix &values) const;

        /// Get the IntegrationRule associated with entity (element or face) @a idx.
        const IntegrationRule &GetIntRule(int idx) const
        { return GetSpace()->GetIntRule(idx); }

};

