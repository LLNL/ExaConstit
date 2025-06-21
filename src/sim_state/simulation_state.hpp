#pragma once

#include "options/option_parser_v2.hpp"
#include "BCManager.hpp"

#include "mfem.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "mfem_expt/partial_qfunc.hpp"

#include <algorithm>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <vector>

enum class TimeStep {NORMAL, RETRIAL, SUBSTEP, FAILED, FINAL, FINISHED};

class TimeManagement {
private:
    double time = 0.0;
    double time_final = 0.0;
    double dt = 1.0;
    double dt_orig = 1.0;
    double prev_dt = 1.0;
    double dt_min = 1.0;
    double dt_max = 1.0;
    double dt_scale = 0.25;
    double dt_fixed = 1.0;
    TimeStepType time_type = TimeStepType::NOTYPE;
    std::vector<double> custom_dt = {};
    size_t simulation_cycle = 0;
    size_t max_nr_steps = 25;
    size_t max_failures = 4;
    size_t num_failures = 0;
    size_t required_num_sub_steps = 0;
    size_t num_sub_steps = 0;
    std::string auto_dt_file;
    TimeStep internal_tracker = TimeStep::NORMAL;
public:

    TimeManagement(ExaOptions& options) : time_type(options.time.time_type){
        if (time_type == TimeStepType::FIXED) {
            dt = options.time.fixed_time->dt;
            dt_fixed = dt;
            dt_min = std::pow(dt_scale, max_failures) * dt;
            time_final = options.time.fixed_time->t_final;
        }
        else if (time_type == TimeStepType::AUTO) {
            dt = options.time.auto_time->dt_start;
            dt_min = options.time.auto_time->dt_min;
            dt_max = options.time.auto_time->dt_max;
            dt_scale = options.time.auto_time->dt_scale;
            time_final = options.time.auto_time->t_final;
            max_nr_steps = options.solvers.nonlinear_solver.iter;
            auto_dt_file = options.time.auto_time->auto_dt_file;
            // insert logic to write out the first time step maybe?
        }
        else if (time_type == TimeStepType::CUSTOM) {
            // const auto dt_beg = options.time.custom_time->dt_values.begin();
            // const auto dt_end = options.time.custom_time->dt_values.end();
            custom_dt = options.time.custom_time->dt_values;
            dt = custom_dt[0];
            dt_min = std::pow(dt_scale, max_failures) * (double)(*std::min_element(custom_dt.begin(), custom_dt.end()));
            time_final = std::accumulate(custom_dt.begin(), custom_dt.end(), 0.0);
        }
   
        prev_dt = dt;
        // Set our first cycle to the initial dt value;
        time = dt;

        const double tf_dt = std::abs(time_final - dt);
        if (tf_dt <= std::abs(1e-3 * dt))
        {
            internal_tracker = TimeStep::FINAL;
        }
    }

    double getTime() const { return time; }
    double getDeltaTime() const { return dt; }
    TimeStep
    updateDeltaTime(const int nr_steps, const bool success = true) {
        // If simulation failed we want to scale down our dt by some factor
        if (!success) {
            // If we were already sub-stepping through a simulation and encouter this just fail out
            if (internal_tracker == TimeStep::SUBSTEP) {
                return TimeStep::FAILED;
            }
            // For the very first failure we want to save off the initial guessed time step
            if (num_failures == 0) {
                dt_orig = dt;
            }
            // reset the time, update dt, and then update the time to correct time
            resetTime();
            dt *= dt_scale;
            if (dt < dt_min) { dt = dt_min; }
            updateTime();
            num_failures++;
            num_sub_steps = 1;
            // If we've failed too many times just give up at this point
            if (num_failures > max_failures) {
                return TimeStep::FAILED;
            }
            // else we need to let the simulation now it's retrying it's time step again
            else {
                return TimeStep::RETRIAL;
            }
        }

        if (internal_tracker == TimeStep::FINAL) {
            internal_tracker = TimeStep::FINISHED;
            return TimeStep::FINISHED;
        }
        // This means we had a successful time step but previously we failed
        // Since we were using a fixed / custom dt here that means we need to substep
        // to get our desired dt that the user was asking for
        if (num_failures > 0) {
            required_num_sub_steps = (time_type != TimeStepType::AUTO) ? 
                                     ((size_t) 1.0 / std::pow(dt_scale, num_failures)) :
                                     0;
            num_failures = 0;
        }
        // If sub-stepping through our original dt then need to update the time while we go along
        if ((num_sub_steps < required_num_sub_steps) and (time_type != TimeStepType::AUTO)) {
            num_sub_steps += 1;
            updateTime();
            internal_tracker = TimeStep::SUBSTEP;
            return TimeStep::SUBSTEP;
        }

        prev_dt = dt;
        simulation_cycle++;
        // update our time based on the following logic
        if (time_type == TimeStepType::AUTO) {
            // update the dt
            const double niter_scale = ((double) max_nr_steps) * dt_scale;
            const double nr_iter = (double) nr_steps;
            // Will approach dt_scale as nr_iter -> newton_iter
            // dt increases as long as nr_iter > niter_scale
            const double factor = niter_scale / nr_iter;
            dt *= factor;
            if (dt < dt_min) { dt = dt_min; }
            if (dt > dt_max) { dt = dt_max; }
        } else if (time_type == TimeStepType::CUSTOM) {
            dt = custom_dt[simulation_cycle];
        } else {
            dt = dt_fixed;
        }
        const double tnew = time + dt;
        const double tf_dt = std::abs(tnew - time_final);
        if (tf_dt <= std::abs(1e-3 * dt)) 
        {
            internal_tracker = TimeStep::FINAL;
            time = tnew;
            return TimeStep::FINAL;
        } else if ((tnew - time_final) > 0)
        {
            internal_tracker = TimeStep::FINAL;
            dt = time_final - time;
            time = time_final;
            return TimeStep::FINAL;
        }
        time = tnew;
        // We're back on a normal time stepping procedure
        internal_tracker = TimeStep::NORMAL;
        return TimeStep::NORMAL;
    }

    // returns false if our time step isn't close to the boundary
    // returns true if the step will land us on the desired boundary.
    // It's up to the user to then check and see if they're past the point already or
    // if there's more time steps left.
    bool BCTime(const double desired_bc_time) {
        // if time is already past the desired_bc_time before updating this then we're not going to
        // update things to nail it
        if (time > desired_bc_time) { return false; }
        const double tnew = time + dt;
        const double tf_dt = desired_bc_time - tnew;
        // First check if we're when the radius when the next time step would be don't care about sign yet
        if (std::abs(tf_dt) < std::abs(dt)) {
            // Now only update the dt value if we're past the original value 
            if (tf_dt < 0.0) {
                resetTime();
                dt += tf_dt;
                updateTime();
                return true;
            }
        }
        return false;
    }

    void updateTime() { time += dt; }
    void resetTime() { time -= dt; }

    void restartTimeState(const double time_restart, const double dt_restart, const size_t cycle)
    {
        simulation_cycle = cycle;
        time = time_restart;
        dt = dt_restart;
    }

    void saveDeltaTime() const {
        std::ofstream file;
        file.open(auto_dt_file, std::ios_base::app);
        file << std::setprecision(12) << dt << std::endl;
    }

    void printSubStepStats() const {
        std::cout << "Previous attempts to converge failed but now starting sub-stepping of our desired time step: desired dt old was " << dt_orig << " sub-stepping dt is " << dt << " and number of sub-steps required is " << required_num_sub_steps << std::endl;
    }

    void printTimeStats() const {
        const double factor = dt / prev_dt;
        std::cout << "Time "<< time << " dt old was " << prev_dt << " dt has been updated to " << dt << " and changed by a factor of " << factor << std::endl;
    }

    bool isLastStep() const { return internal_tracker == TimeStep::FINAL; }
    bool isFinished() const { return internal_tracker == TimeStep::FINISHED; }

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
    // Map of the mesh nodes associated with a given region maybe?
    std::map<std::string, std::shared_ptr<mfem::ParGridFunction>> m_mesh_nodes;
    // Map of the mesh nodes associated with a QoI aka x_nodes-> time_{0}, time_{i}, time_{i+1}, velocity, displacement
    std::map<std::string, std::shared_ptr<mfem::ParGridFunction>> m_mesh_qoi_nodes;

    // Our velocity field
    std::shared_ptr<mfem::Vector> m_primal_field;
    std::shared_ptr<mfem::Vector> m_primal_field_prev;
    std::shared_ptr<mfem::Array<int>> m_grains;

    // Map of the material properties associated with a given region name
    std::map<std::string, std::vector<double>> m_material_properties;
    // Vector of the material region name and the region index associated with it
    std::vector<std::pair<std::string, int>> m_material_name_region;
    std::vector<MechType> m_region_material_type;
    // Map of the quadrature function name to the potential offset in the quadrature function and
    // the vector dimension associated with that quadrature function name.
    // This variable is useful to obtain sub-mappings within a quadrature function used for all history variables
    // such as how it's done with ECMech's models.
    std::map<std::string, std::pair<int, int>> m_map_qf_mappings;
    // Class devoted to updating our time based on various logic we might have.
    TimeManagement m_time_manager;
    // Only need 1 instance of our boundary condition manager
    // BCManager m_bc_manager;

    // Vector of the names of the quadrature function pairs that have their data ptrs
    // swapped when UpdateModel() is called.
    std::vector<std::pair<std::string, std::string>> m_model_update_qf_pairs;

    ExaOptions& m_options;

#if defined(EXACONSTIT_USE_AXOM)
    // We want this to be something akin to a axom::sidre::MFEMSidreDataCollection
    // However, we need it flexible enough to handle multiple different mesh topologies in it that
    // we might due to different mfem::SubMesh objects that correspond to each PartialQuadraturePoint
    std::unique_ptr<axom::sidre::DataStore> m_simulation_restart;
#endif
    int my_id;
public:
    RTModel class_device;
public:
    SimulationState(ExaOptions& options);
    virtual ~SimulationState() = default;

    /**
     * @brief Initialize state variables and grain orientation data for all material regions
     * This replaces the global setStateVarData function with a per-region approach
     */
    void InitializeStateVariables();

    // A way to tell the class which beginning and end time step variables need to have internal
    // pointer values swapped when a call to UpdateModel is made.  
    void AddUpdateVariablePairNames(std::pair<std::string_view, std::string_view> update_var_pair) {
        std::string view1(update_var_pair.first);
        std::string view2(update_var_pair.second);
        m_model_update_qf_pairs.push_back({view1, view2});
    }

    // If the QuadratureFunction name already exists for a given region
    // this will return false
    // else this will return true
    // A region number of -1 tells us that
    // we're dealing with a global space
    bool AddQuadratureFunction(const std::string_view& qf_name, const int vdim = 1, const int region = -1) {
        std::string qf_name_mat = GetQuadratureFunctionMapName(qf_name, region);
        if (m_map_qfs.find(qf_name_mat) == m_map_qfs.end())
        {
            std::string qspace_name = GetRegionName(region);
            m_map_qfs.emplace(qf_name_mat, std::make_shared<mfem::expt::PartialQuadratureFunction>(m_map_qs[qspace_name], vdim, 0.0));
            return true;
        }
        return false;
    }

    // Add to the internal QF state pair mapping
    // While this is ideally material model specific info, it's quite useful for other
    // Models would largely be responsible for setting this all up
    bool AddQuadratureFunctionStatePair(const std::string_view state_name, std::pair<int, int> state_pair, const int region)
    {
        std::string mat_name = GetQuadratureFunctionMapName(state_name, region);
        if (m_map_qf_mappings.find(mat_name) == m_map_qf_mappings.end())
        {
            m_map_qf_mappings.emplace(mat_name, state_pair);
            return true;
        }
        return false;
    }

    // This updates function does a simple pointer swap between the beginning and end time step values
    // of those variables that have been added by AddUpdateVariablePairNames
    void UpdateModel()
    {
        for (auto [name_prev, name_cur] : m_model_update_qf_pairs) {
            m_map_qfs[name_prev]->Swap(*m_map_qfs[name_cur]);
        }
    }

    // Mesh end coordinates need to be updated from here and not some other module
    void UpdateNodalEndCoords()
    {
        m_mesh_qoi_nodes["velocity"]->Distribute(*m_primal_field);
        (*m_mesh_nodes["mesh_current"]) = *m_mesh_qoi_nodes["velocity"];
        (*m_mesh_nodes["mesh_current"]) *= getDeltaTime();
        (*m_mesh_nodes["mesh_current"]) += *m_mesh_nodes["mesh_t_beg"];
    }

    // When the delta time step was bad we need to restart our mesh nodes to the prev state and then move to the right one
    void restartCycle()
    {
        m_mesh_qoi_nodes["velocity"]->Distribute(*m_primal_field_prev);
        (*m_primal_field) = *m_primal_field_prev;
        (*m_mesh_nodes["mesh_current"]) = (*m_mesh_nodes["mesh_t_beg"]);
    }

    // When our solver converged this makes sure our mesh nodes our correctly update as well as our state variables
    void finishCycle() {
        (*m_primal_field_prev) = *m_primal_field;
        (*m_mesh_qoi_nodes["displacement"]) = *m_mesh_nodes["mesh_current"];
        (*m_mesh_qoi_nodes["displacement"]) -= *m_mesh_nodes["mesh_ref"];
        m_mesh_qoi_nodes["velocity"]->Distribute(*m_primal_field);
        // Code previously had beg time coords updated after the update model aspect of things
        // UpdateModel();
        (*m_mesh_nodes["mesh_t_beg"]) = *m_mesh_nodes["mesh_current"];
    }

    std::shared_ptr<mfem::Vector> getPrimalField() { return m_primal_field; }
    std::shared_ptr<mfem::Vector> getPrimalFieldPrev() { return m_primal_field_prev; }
    std::shared_ptr<mfem::Array<int>> getGrains() { return m_grains; }
    std::shared_ptr<mfem::ParMesh> getMesh() { return m_mesh; }
    std::shared_ptr<mfem::ParGridFunction> getCurrentCoords() { return m_mesh_nodes["mesh_current"]; }
    std::shared_ptr<mfem::ParGridFunction> getTimeStartCoords() { return m_mesh_nodes["mesh_t_beg"]; }
    std::shared_ptr<mfem::ParGridFunction> getRefCoords() { return m_mesh_nodes["mesh_ref"]; }
    std::shared_ptr<mfem::ParGridFunction> getDisplacement() { return m_mesh_qoi_nodes["displacement"]; }
    std::shared_ptr<mfem::ParGridFunction> getVelocity() { return m_mesh_qoi_nodes["velocity"]; }
    std::shared_ptr<mfem::expt::PartialQuadratureSpace> getGlobalVizQuadSpace() { return m_map_qs["global_ord_0"]; }


    // Returns the number of regions in the simulation
    int GetNumberOfRegions() const { return m_material_name_region.size(); }

    MechType GetRegionModelType(const int idx) const { return m_region_material_type[idx]; }

    std::string GetRegionName(const int region) const {
        if (region < 0) { return "global"; }
        return m_material_name_region[region].first + "_" + std::to_string(m_material_name_region[region].second);
    }

    /**
     * @brief Get material properties for a specific region
     * 
     * @param region Region index
     * @return const reference to material properties vector
     */
    const std::vector<double>& GetMaterialProperties(const int region) const {
        const auto region_name = GetRegionName(region);
        return GetMaterialProperties(region_name);
    }

    /**
     * @brief Get material properties by region name
     * 
     * @param region_name Name of the region
     * @return const reference to material properties vector
     */
    const std::vector<double>& GetMaterialProperties(const std::string& region_name) const {
        return m_material_properties.at(region_name);
    }

    // This returns the correct mapping name for a quadrature function
    // If a region is provided than the mapped name will have the material name associated with
    // the region attached to it.
    // regions start at 0, and a negative region signals that a material name is not associated with things. 
    std::string GetQuadratureFunctionMapName(const std::string_view& qf_name, const int region = -1) const
    {
        if (region < 0) { return std::string(qf_name); }
        std::string mat_name = GetRegionName(region);
        std::string qf_name_mat = std::string(qf_name) + "_" + mat_name;
        return qf_name_mat;
    }

    // Must provide region number of material we're dealing with in-order to output
    // correct quadrature function.
    // This will raise an error if a quadrature function name does not exist for a given region
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> GetQuadratureFunction(const std::string_view& qf_name, const int region = -1)
    {
        return m_map_qfs[GetQuadratureFunctionMapName(qf_name, region)];
    }

    // Given the state variable name and region ID we care about this returns the specific offset and vdim
    // associated with that name. Typically, you might use this when the state variable might live in a
    // larger QuadratureFunction
    std::pair<int, int> GetQuadratureFunctionStatePair(const std::string_view& state_name, const int region = -1) const
    {
        std::string mat_name = GetQuadratureFunctionMapName(state_name, region);
        if (m_map_qf_mappings.find(mat_name) == m_map_qf_mappings.end()) { return {-1, -1}; }
        const std::pair<int, int> output = m_map_qf_mappings.at(mat_name);
        return output;
    }

    // Returns a pointer to a ParFiniteElementSpace (PFES) that's ordered according to VDIMs
    // and makes use of an L2 FiniteElementCollection
    // If the vdim is not in the internal mapping than a new PFES will be created
    std::shared_ptr<mfem::ParFiniteElementSpace> GetParFiniteElementSpace(const int vdim)
    {
        if (m_map_pfes.find(vdim) == m_map_pfes.end())
        {
            const int space_dim = m_mesh->SpaceDimension();
            std::string l2_fec_str = "L2_" + std::to_string(space_dim) + "D_P" + std::to_string(0);
            auto l2_fec = m_map_fec[l2_fec_str];
            auto key = std::make_shared<mfem::ParFiniteElementSpace>(m_mesh.get(), l2_fec.get(), vdim, mfem::Ordering::byVDIM);
            m_map_pfes.emplace(vdim, std::move(key));
        }
        return m_map_pfes[vdim];
    }
    
    // Gets the PFES associated with the mesh
    std::shared_ptr<mfem::ParFiniteElementSpace> GetMeshParFiniteElementSpace() { return m_mesh_fes; }

    const ExaOptions& getOptions() const { return m_options; }

    double getTime() const { return m_time_manager.getTime(); }
    double getDeltaTime() const { return m_time_manager.getDeltaTime(); }

    TimeStep
    updateDeltaTime(const int nr_steps, const bool failure = false) { return m_time_manager.updateDeltaTime(nr_steps, failure); }

    bool isLastStep() const { return m_time_manager.isLastStep(); }
    bool isFinished() const { return m_time_manager.isFinished(); }
    void printTimeStats() const { m_time_manager.printTimeStats(); }

private:
    /**
     * @brief Initialize state variables for a specific material region
     * @param region_id The material region to initialize
     * @param material The material configuration
     * @param grains2region Mapping from grain IDs to region IDs
     */
    void InitializeRegionStateVariables(int region_id, 
        const MaterialOptions& material,
        const std::map<int, int>& grains2region);

    /**
     * @brief Utility function to update the number of state variables count in our options if a model uses orientations
     */
    void UpdateExaOptionsWithOrientationCounts();

    // Shared orientation data (loaded once, used by all regions)
    struct SharedOrientationData {
        std::vector<double> quaternions;  // Always unit quaternions (passive rotations)
        int num_grains;
        bool is_loaded;
        
        SharedOrientationData() : num_grains(0), is_loaded(false) {}
    };
    
    // Per-region orientation configuration
    struct OrientationConfig {
        std::vector<double> data;  // Converted to format required by this region
        int stride;
        int offset_start;
        int offset_end;
        bool is_valid;
        
        OrientationConfig() : stride(0), offset_start(-1), offset_end(0), is_valid(false) {}
    };
    
    // Shared orientation data for all regions
    SharedOrientationData m_shared_orientation_data;
    
    /**
     * @brief Load unit quaternion orientation data from file (called once for all regions)
     * @param orientation_file Path to orientation file containing unit quaternions
     * @param num_grains Number of grains expected
     * @return True if successfully loaded
     */
    bool LoadSharedOrientationData(const std::string& orientation_file, int num_grains);
    
    /**
     * @brief Convert unit quaternions to Euler angles (Bunge convention)
     * @param quaternions Vector containing unit quaternions (stride 4)
     * @param num_grains Number of grains
     * @return Vector of Euler angles (stride 3)
     */
    std::vector<double> ConvertQuaternionsToEuler(const std::vector<double>& quaternions, int num_grains);
    
    /**
     * @brief Convert unit quaternions to rotation matrices  
     * @param quaternions Vector containing unit quaternions (stride 4)
     * @param num_grains Number of grains
     * @return Vector of 3x3 rotation matrices (stride 9)
     */
    std::vector<double> ConvertQuaternionsToMatrix(const std::vector<double>& quaternions, int num_grains);
    
    /**
     * @brief Prepare orientation data for a specific region/material
     * @param material Material options containing grain info and orientation requirements
     * @return OrientationConfig with data converted to the format required by this material
     */
    OrientationConfig PrepareOrientationForRegion(const MaterialOptions& material);
    
    /**
     * @brief Calculate the effective state variable count including orientations
     * @param material Material options
     * @return Total count including orientation variables if present
     */
    int CalculateEffectiveStateVarCount(const MaterialOptions& material);
    
    /**
     * @brief Determine placement offsets for orientation data in state variable array
     * @param material Material options
     * @param orientation_stride Number of orientation components per grain
     * @return Pair of (offset_start, offset_end) indices
     */
    std::pair<int, int> CalculateOrientationOffsets(const MaterialOptions& material, int orientation_stride);
    
    /**
     * @brief Fill orientation data into the state variable array at a specific quadrature point
     * @param qf_data Pointer to QuadratureFunction data
     * @param qpt_base_index Base index for current quadrature point
     * @param qf_vdim Vector dimension of QuadratureFunction
     * @param grain_id Grain ID for current element
     * @param orientation_config Orientation configuration with data and offsets
     */
    void FillOrientationData(double* qf_data, int qpt_base_index, int qf_vdim, 
                           int grain_id, const OrientationConfig& orientation_config);
    
    /**
     * @brief Clean up shared orientation data after all regions are initialized
     * This frees memory used by the shared orientation data
     */
    void CleanupSharedOrientationData();

};