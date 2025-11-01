
#ifndef BCMANAGER
#define BCMANAGER

#include "boundary_conditions/BCData.hpp"
#include "options/option_parser_v2.hpp"

// C/C++ includes
#include <algorithm>
#include <mutex>
#include <unordered_map> // for std::unordered_map
#include <vector>

/**
 * @brief Singleton manager for all boundary conditions in the simulation
 *
 * @details This class implements the Singleton pattern to provide centralized management
 * of boundary conditions throughout the simulation. It coordinates time-dependent boundary
 * conditions, manages multiple BCData instances, and provides the interface between
 * the options system and the finite element assembly process.
 *
 * Key responsibilities:
 * - Manage time-dependent boundary condition changes
 * - Store and organize essential velocity and velocity gradient data
 * - Create and maintain BCData instances for each boundary
 * - Coordinate between different boundary condition types (velocity vs velocity gradient)
 * - Provide thread-safe initialization and access
 *
 * The class supports complex boundary condition scenarios including:
 * - Multi-step boundary condition evolution
 * - Mixed velocity and velocity gradient constraints
 * - Component-wise boundary condition application
 * - Time-dependent boundary condition updates
 */
class BCManager {
public:
    /**
     * @brief Get the singleton instance of BCManager
     *
     * @return Reference to the singleton BCManager instance
     *
     * @details Implements the Meyer's singleton pattern for thread-safe initialization.
     * The instance is created on first call and persists for the lifetime of the program.
     */
    static BCManager& GetInstance() {
        static BCManager bc_manager;
        return bc_manager;
    }

    /**
     * @brief Initialize the BCManager with time-dependent boundary condition data
     *
     * @param u_step Vector of time steps when boundary conditions should be updated
     * @param ess_vel Map from time step to essential velocity values
     * @param ess_vgrad Map from time step to essential velocity gradient values
     * @param ess_comp Map from BC type and time step to component IDs
     * @param ess_id Map from BC type and time step to boundary IDs
     *
     * @details Thread-safe initialization using std::call_once. This method should be called
     * once during simulation setup to configure all time-dependent boundary condition data.
     * The data structures support complex time-dependent scenarios where different boundaries
     * can have different constraint patterns that change over time.
     *
     * The map_of_imap type represents nested maps: map<string, map<int, vector<int>>>
     * where the outer key is the BC type ("ess_vel", "ess_vgrad", "total") and the inner
     * key is the time step number.
     */
    void Init(const std::vector<int>& u_step,
              const std::unordered_map<int, std::vector<double>>& ess_vel,
              const std::unordered_map<int, std::vector<double>>& ess_vgrad,
              const map_of_imap& ess_comp,
              const map_of_imap& ess_id) {
        std::call_once(init_flag, [&]() {
            update_step = u_step;
            map_ess_vel = ess_vel;
            map_ess_vgrad = ess_vgrad;
            map_ess_comp = ess_comp;
            map_ess_id = ess_id;
        });
    }

    /**
     * @brief Get a boundary condition instance by ID
     *
     * @param bcID Boundary condition identifier
     * @return Reference to the BCData instance for the specified boundary
     *
     * @details Provides access to a specific boundary condition instance. The bcID
     * corresponds to mesh boundary attributes. Used during assembly to access
     * boundary condition data for specific mesh boundaries.
     */
    BCData& GetBCInstance(int bcID) {
        return m_bc_instances.find(bcID)->second;
    }

    /**
     * @brief Get a boundary condition instance by ID (const version)
     *
     * @param bcID Boundary condition identifier
     * @return Const reference to the BCData instance for the specified boundary
     *
     * @details Const version of GetBCInstance for read-only access to boundary condition data.
     */
    const BCData& GetBCInstance(int bcID) const {
        return m_bc_instances.find(bcID)->second;
    }

    /**
     * @brief Create or access a boundary condition instance
     *
     * @param bcID Boundary condition identifier
     * @return Reference to the BCData instance (created if it doesn't exist)
     *
     * @details Creates a new BCData instance if one doesn't exist for the given bcID,
     * or returns a reference to the existing instance. This is used during boundary
     * condition setup to ensure all required BCData objects are available.
     */
    BCData& CreateBCs(int bcID) {
        return m_bc_instances[bcID];
    }

    /**
     * @brief Get all boundary condition instances
     *
     * @return Reference to the map containing all BCData instances
     *
     * @details Provides access to the complete collection of boundary condition instances.
     * Useful for iteration or bulk operations on all boundary conditions.
     */
    std::unordered_map<int, BCData>& GetBCInstances() {
        return m_bc_instances;
    }

    /**
     * @brief Update boundary condition data for the current time step
     *
     * @param ess_bdr Map of essential boundary arrays by BC type
     * @param scale 2D array of scaling factors for boundary conditions
     * @param vgrad Vector of velocity gradient values
     * @param component Map of component activation arrays by BC type
     *
     * @details Main coordination method that updates all boundary condition data structures
     * for the current simulation time step. This method:
     * 1. Clears previous boundary condition data
     * 2. Sets up combined boundary condition information
     * 3. Calls specialized update methods for velocity and velocity gradient BCs
     * 4. Coordinates between different boundary condition types
     *
     * This is called at the beginning of each time step where boundary conditions change.
     */
    void UpdateBCData(std::unordered_map<std::string, mfem::Array<int>>& ess_bdr,
                      mfem::Array2D<double>& scale,
                      mfem::Vector& vgrad,
                      std::unordered_map<std::string, mfem::Array2D<bool>>& component);

    /**
     * @brief Check if the current step requires boundary condition updates
     *
     * @param step_ Time step number to check
     * @return True if boundary conditions should be updated at this step
     *
     * @details Determines whether boundary conditions need to be updated at the specified
     * time step by checking against the list of update steps provided during initialization.
     * If an update is needed, the internal step counter is also updated.
     */
    bool GetUpdateStep(int step_) {
        if (std::find(update_step.begin(), update_step.end(), step_) != update_step.end()) {
            step = step_;
            return true;
        } else {
            return false;
        }
    }

private:
    /**
     * @brief Private constructor for singleton pattern
     *
     * @details Default constructor is private to enforce singleton pattern.
     */
    BCManager() {}

    /**
     * @brief Deleted copy constructor for singleton pattern
     */
    BCManager(const BCManager&) = delete;

    /**
     * @brief Deleted copy assignment operator for singleton pattern
     */
    BCManager& operator=(const BCManager&) = delete;

    /**
     * @brief Deleted move constructor for singleton pattern
     */
    BCManager(BCManager&&) = delete;

    /**
     * @brief Deleted move assignment operator for singleton pattern
     */
    BCManager& operator=(BCManager&&) = delete;

    /**
     * @brief Update velocity gradient boundary condition data
     *
     * @param ess_bdr Essential boundary array for velocity gradient BCs
     * @param vgrad Velocity gradient vector to populate
     * @param component Component activation array for velocity gradient BCs
     *
     * @details Specialized update method for velocity gradient boundary conditions.
     * Processes the velocity gradient data for the current time step and sets up
     * the appropriate data structures for finite element assembly.
     */
    void
    UpdateBCData(mfem::Array<int>& ess_bdr, mfem::Vector& vgrad, mfem::Array2D<bool>& component);

    /**
     * @brief Update velocity boundary condition data
     *
     * @param ess_bdr Essential boundary array for velocity BCs
     * @param scale Scaling factors for velocity BCs
     * @param component Component activation array for velocity BCs
     *
     * @details Specialized update method for velocity boundary conditions. Creates BCData
     * instances for each active boundary, sets up scaling factors, and prepares data
     * structures for finite element assembly. This method:
     * 1. Clears existing BCData instances
     * 2. Processes essential velocity data for the current time step
     * 3. Creates BCData objects with appropriate velocity and component settings
     * 4. Sets up scaling and boundary activation arrays
     */
    void UpdateBCData(mfem::Array<int>& ess_bdr,
                      mfem::Array2D<double>& scale,
                      mfem::Array2D<bool>& component);

    /** @brief Thread-safe initialization flag */
    std::once_flag init_flag;

    /** @brief Current simulation time step */
    int step = 0;

    /** @brief Collection of boundary condition data instances */
    std::unordered_map<int, BCData> m_bc_instances;

    /** @brief Time steps when boundary conditions should be updated */
    std::vector<int> update_step;

    /** @brief Essential velocity values by time step */
    std::unordered_map<int, std::vector<double>> map_ess_vel;

    /** @brief Essential velocity gradient values by time step */
    std::unordered_map<int, std::vector<double>> map_ess_vgrad;

    /** @brief Component IDs by BC type and time step */
    map_of_imap map_ess_comp;

    /** @brief Boundary IDs by BC type and time step */
    map_of_imap map_ess_id;
};

#endif
