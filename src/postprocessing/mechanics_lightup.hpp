#pragma once

#include "mfem_expt/partial_qfunc.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "options/option_parser_v2.hpp"
#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

/**
 * @brief General crystal lattice structure and symmetry operations
 *
 * Provides crystal lattice parameters, reciprocal lattice vectors,
 * and symmetry operations for all eight supported lattice types and their
 * corresponding Laue groups (cubic to triclinic). Used by LightUp
 * for crystal-structure-specific calculations.
 *
 * The class computes reciprocal lattice vectors from direct lattice
 * parameters and generates symmetry-equivalent directions for HKL families
 * based on the appropriate point group symmetries.
 *
 * Supported crystal systems:
 * - Cubic (24 symmetry operations)
 * - Hexagonal (12 symmetry operations)
 * - Trigonal (6 symmetry operations)
 * - Rhombohedral (6 symmetry operations)
 * - Tetragonal (8 symmetry operations)
 * - Orthorhombic (4 symmetry operations)
 * - Monoclinic (2 symmetry operations)
 * - Triclinic (1 symmetry operation)
 *
 * @ingroup ExaConstit_postprocessing_lightup
 */
class LatticeTypeGeneral {
public:
    /**
     * @brief Number of symmetry operations for the crystal lattice
     *
     * Point group symmetry operations (rotations and inversions) for the
     * specified crystal system. The number varies by lattice type:
     * cubic (24), hexagonal (12), trigonal (6), rhombohedral (6), tetragonal (8),
     * orthorhombic (4), monoclinic (2), triclinic (1).
     * Used for generating symmetrically equivalent crystallographic directions.
     */
    const size_t NSYM = 1;

    /**
     * @brief Constructor for general crystal lattice structure
     *
     * @param lattice_param_a Vector of lattice parameters specific to crystal system
     * @param lattice_type Crystallographic lattice type enum specifying crystal system
     *
     * Initializes crystal lattice structure by computing reciprocal lattice vectors
     * and generating point group symmetry operations for the specified crystal system.
     *
     * The constructor:
     * 1. Determines the number of symmetry operations for the crystal system
     * 2. Generates quaternion representations of all symmetry operations
     * 3. Computes reciprocal lattice parameter matrix from direct lattice parameters
     * 4. Stores lattice geometry for HKL direction transformations
     *
     * Lattice parameter requirements by crystal system:
     * - **Cubic**: a (lattice parameter)
     * - **Hexagonal**: a, c (basal and c-axis parameters)
     * - **Trigonal**: a, c (basal and c-axis parameters)
     * - **Rhombohedral**: a, α (lattice parameter and angle in radians)
     * - **Tetragonal**: a, c (basal and c-axis parameters)
     * - **Orthorhombic**: a, b, c (three distinct lattice parameters)
     * - **Monoclinic**: a, b, c, β (three lattice parameters and monoclinic angle in radians)
     * - **Triclinic**: a, b, c, α, β, γ (three lattice parameters and three angles in radians)
     *
     * The reciprocal lattice matrix enables transformation of Miller indices (HKL)
     * to crystallographic direction vectors, while symmetry operations generate
     * equivalent directions for powder diffraction calculations in LightUp analysis.
     *
     * @note All angular parameters must be provided in radians
     * @see SymmetricQuaternions() for details on symmetry operation generation
     * @see ComputeLatticeBParam() for reciprocal lattice computation
     */
    LatticeTypeGeneral(const std::vector<double>& lattice_param_a, const LatticeType& lattice_type);

    ~LatticeTypeGeneral() = default;

    /**
     * @brief Compute reciprocal lattice parameter matrix
     *
     * @param lparam_a Direct lattice parameters
     * @param lattice_type Crystal system type
     *
     * Computes the reciprocal lattice vectors (lattice_b matrix) from
     * direct lattice parameters for any crystal system. The method handles
     * the varying number of parameters required for each system:
     * cubic (a), hexagonal/trigonal (a,c), tetragonal (a,c),
     * orthorhombic (a,b,c), monoclinic (a,b,c,β), triclinic (a,b,c,α,β,γ).
     * The reciprocal lattice is used to transform HKL indices to direction
     * vectors in reciprocal space.
     */
    void ComputeLatticeBParam(const std::vector<double>& lparam_a, const LatticeType& lattice_type);

    /**
     * @brief Generate and store symmetry operation quaternions for crystal system
     *
     * @param lattice_type Crystal system type specifying the point group
     *
     * Generates the complete set of point group symmetry operations for the
     * specified crystal system and stores them in the quat_symm member variable.
     * Each symmetry operation is represented as a quaternion in the form
     * [angle, x, y, z] where angle is the rotation angle in radians and
     * [x, y, z] is the normalized rotation axis.
     *
     * The method:
     * 1. Calls GetSymmetryGroups() to obtain symmetry operations for the crystal system
     * 2. Flattens the quaternion array into the quat_symm storage vector
     * 3. Stores quaternions sequentially for efficient access during calculations
     *
     * The number and type of symmetry operations generated depend on the crystal system:
     * - Cubic: 24 quaternions (full octahedral symmetry)
     * - Hexagonal: 12 quaternions (hexagonal point group)
     * - Trigonal: 6 quaternions (trigonal point group)
     * - Rhombohedral: 6 quaternions (rhombohedral point group)
     * - Tetragonal: 8 quaternions (tetragonal point group)
     * - Orthorhombic: 4 quaternions (orthogonal symmetries)
     * - Monoclinic: 2 quaternions (monoclinic symmetry)
     * - Triclinic: 1 quaternion (identity only)
     *
     * These stored quaternions are subsequently used to generate symmetrically
     * equivalent HKL directions during lattice strain calculations in the
     * LightUp analysis framework.
     *
     * @note Called automatically during LatticeTypeGeneral construction
     * @see GetSymmetryGroups() for symmetry operation generation
     * @see quat_symm member variable for quaternion storage
     */
    void SymmetricQuaternions(const LatticeType& lattice_type);

public:
    /**
     * @brief Reciprocal lattice parameter matrix
     *
     * 3x3 matrix containing reciprocal lattice vectors as columns.
     * Used to transform HKL indices to direction vectors in reciprocal space.
     * Computed from direct lattice parameters in constructor.
     */
    double lattice_b[3 * 3];
    std::vector<double> quat_symm;
};

/**
 * @brief Lattice strain analysis class for powder diffraction simulation
 *
 * The LightUp class performs in-situ lattice strain calculations that simulate
 * powder diffraction experiments on polycrystalline materials. It computes
 * lattice strains for specified crystallographic directions (HKL) based on
 * crystal orientation evolution and stress state from ExaCMech simulations.
 *
 * Supports all eight crystal systems (cubic, hexagonal, trigonal, rhombohedral,
 * tetragonal, orthorhombic, monoclinic, triclinic) through the generalized
 * LatticeTypeGeneral class which provides appropriate symmetry operations for each system.
 *
 * Key capabilities:
 * - Lattice strain calculation for multiple HKL directions
 * - Taylor factor and plastic strain rate analysis
 * - Directional stiffness computation
 * - Volume-weighted averaging over grains/orientations
 * - Real-time output for experimental comparison
 *
 * The class interfaces with ExaCMech state variables including:
 * - Crystal orientations (quaternions)
 * - Elastic strain tensors
 * - Relative volume changes
 * - Plastic strain rates and slip system activities
 *
 * Applications:
 * - Validation against in-situ diffraction experiments
 * - Prediction of lattice strain evolution during deformation
 * - Analysis of load partitioning between crystallographic directions
 * - Study of texture effects on mechanical response
 *
 * @ingroup ExaConstit_postprocessing_lightup
 */
class LightUp {
public:
    /**
     * @brief Constructor for LightUp analysis
     *
     * @param hkls Vector of HKL directions for lattice strain calculation
     * @param distance_tolerance Angular tolerance for fiber direction matching
     * @param s_dir Sample direction vector for reference frame
     * @param qspace Partial quadrature space for region-specific operations
     * @param sim_state Reference to simulation state for data access
     * @param region Region index for analysis
     * @param rtmodel Runtime model for device execution policy
     * @param lattice_basename Base filename for output files
     * @param lattice_params Crystal lattice parameters [a, b, c]
     *
     * Initializes LightUp analysis with specified crystallographic directions
     * and computational parameters. The constructor:
     * 1. Normalizes the sample direction vector
     * 2. Computes reciprocal lattice vectors for each HKL direction
     * 3. Applies crystal symmetry operations to create equivalent directions
     * 4. Initializes in-fiber boolean arrays for each HKL direction
     * 5. Sets up output files with HKL direction headers
     *
     * The distance_tolerance parameter controls the angular tolerance for
     * determining which crystal orientations are "in-fiber" for each HKL direction.
     * Uses the crystal system's symmetry operations to find equivalent directions.
     */
    LightUp(const std::vector<std::array<double, 3>>& hkls,
            const double distance_tolerance,
            const std::array<double, 3> s_dir,
            std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
            const std::shared_ptr<SimulationState> sim_state,
            const int region,
            const RTModel& rtmodel,
            const fs::path& lattice_basename,
            const std::vector<double>& lattice_params,
            const LatticeType& lattice_type);

    ~LightUp() = default;

    /**
     * @brief Main entry point for LightUp data calculation
     *
     * @param history State variable quadrature function containing crystal data
     * @param stress Stress quadrature function for current state
     *
     * Orchestrates the complete LightUp analysis pipeline:
     * 1. Retrieves state variable offsets for orientations, strains, and rates
     * 2. Sets up in-fiber calculations for all HKL directions
     * 3. Computes lattice strains, Taylor factors, and directional stiffness
     * 4. Outputs results to region-specific files with MPI rank 0 handling I/O
     *
     * This method is called at each output timestep to maintain continuous
     * lattice strain evolution tracking throughout the simulation.
     */
    void CalculateLightUpData(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                              const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress);

    /**
     * @brief Determine in-fiber orientations for a specific HKL direction
     *
     * @param history State variable data containing crystal orientations
     * @param quats_offset Offset to quaternion data in state variable array
     * @param hkl_index Index of HKL direction for calculation
     *
     * Determines which crystal orientations are "in-fiber" (aligned within
     * the distance tolerance) for the specified HKL direction. Uses the
     * appropriate crystal system's symmetry operations to find the maximum
     * dot product between the sample direction and all symmetrically
     * equivalent HKL directions.
     *
     * The algorithm:
     * 1. Extracts quaternion orientations for each quadrature point
     * 2. Converts quaternions to rotation matrices
     * 3. Applies crystal system's symmetry operations to HKL directions
     * 4. Computes alignment with sample direction using all equivalent directions
     * 5. Sets boolean flags for orientations within angular tolerance
     */
    void CalculateInFibers(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                           const size_t quats_offset,
                           const size_t hkl_index);

    /**
     * @brief Calculate lattice strains with volume weighting
     *
     * @param history State variable data
     * @param strain_offset Offset to elastic strain data
     * @param quats_offset Offset to quaternion orientation data
     * @param rel_vol_offset Offset to relative volume data
     * @param lattice_strains_output Output vector for lattice strain results
     * @param lattice_volumes_output Output vector for volume weighting data
     *
     * Computes lattice strains by projecting elastic strain tensors onto the
     * sample direction vector. The calculation accounts for crystal rotations
     * and volume changes through the deformation history.
     *
     * Key steps:
     * 1. Constructs projection vector from normalized sample direction
     * 2. Rotates elastic strain from lattice to sample coordinates
     * 3. Computes strain projection along sample direction
     * 4. Applies volume-weighted averaging using in-fiber filters
     *
     * The method outputs both strain values and corresponding volumes for
     * each HKL direction and overall average.
     */
    void CalcLatticeStrains(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                            const size_t strain_offset,
                            const size_t quats_offset,
                            const size_t rel_vol_offset,
                            std::vector<double>& lattice_strains_output,
                            std::vector<double>& lattice_volumes_output);

    /**
     * @brief Calculate Taylor factors and effective plastic strain rates
     *
     * @param history State variable data
     * @param dpeff_offset Offset to effective plastic strain rate data
     * @param gdot_offset Offset to slip system rate data
     * @param gdot_length Number of slip systems
     * @param lattice_tay_facs Output vector for Taylor factors
     * @param lattice_dpeff Output vector for effective plastic strain rates
     *
     * Computes Taylor factors as the ratio of total slip system activity to
     * effective plastic strain rate. Taylor factors indicate the efficiency
     * of plastic deformation for different crystal orientations.
     *
     * The calculation:
     * 1. Sums absolute values of all slip system shear rates
     * 2. Divides by effective plastic strain rate (with zero-division protection)
     * 3. Applies volume-weighted averaging using in-fiber filters
     *
     * Results provide insight into plastic anisotropy and orientation effects
     * on deformation resistance in textured polycrystalline materials.
     */
    void CalcLatticeTaylorFactorDpeff(
        const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
        const size_t dpeff_offset,
        const size_t gdot_offset,
        const size_t gdot_length,
        std::vector<double>& lattice_tay_facs,
        std::vector<double>& lattice_dpeff);

    /**
     * @brief Calculate directional elastic stiffness properties
     *
     * @param history State variable data
     * @param stress Stress quadrature function data
     * @param strain_offset Offset to elastic strain data
     * @param quats_offset Offset to quaternion orientation data
     * @param rel_vol_offset Offset to relative volume data
     * @param lattice_dir_stiff Output vector for directional stiffness values
     *
     * Computes directional elastic stiffness by analyzing the stress-strain
     * relationship along crystal directions. The method rotates both stress
     * and strain tensors to crystal coordinates and computes the ratio.
     *
     * The algorithm:
     * 1. Projects stress and strain tensors onto sample direction
     * 2. Accounts for crystal orientation through rotation matrices
     * 3. Computes stiffness as stress/strain ratio (with zero-strain protection)
     * 4. Applies volume-weighted averaging for each HKL direction
     *
     * Results provide directional elastic moduli for validation against
     * experimental measurements and constitutive model verification.
     */
    void CalcLatticeDirectionalStiffness(
        const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
        const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress,
        const size_t strain_offset,
        const size_t quats_offset,
        const size_t rel_vol_offset,
        std::vector<std::array<double, 3>>& lattice_dir_stiff);
    /**
     * @brief Get the region ID for this LightUp instance
     *
     * @return Region identifier
     *
     * Returns the material region index associated with this LightUp analysis.
     * Used for accessing region-specific data and organizing multi-region output.
     */
    int GetRegionID() const {
        return m_region;
    }

private:
    /**
     * @brief Vector of HKL crystallographic directions for analysis
     *
     * Contains the original HKL direction vectors specified by the user.
     * A [0,0,0] entry is added at the beginning during construction to
     * represent the overall average (all orientations). Each direction
     * represents a family of crystallographic planes for diffraction analysis.
     */
    std::vector<std::array<double, 3>> m_hkls;
    /**
     * @brief Angular tolerance for in-fiber determination
     *
     * Maximum angular deviation (in radians) for crystal orientations
     * to be considered "in-fiber" for each HKL direction. Controls the
     * selectivity of orientation filtering in lattice strain calculations.
     */
    const double m_distance_tolerance;
    /**
     * @brief Normalized sample direction vector
     *
     * Three-component array defining the reference direction in sample
     * coordinates. Normalized during construction and used for computing
     * directional projections of stress and strain tensors.
     */
    double m_s_dir[3];
    /**
     * @brief Number of quadrature points in the region
     *
     * Total number of quadrature points for the partial quadrature space.
     * Used for array sizing and loop bounds in device kernels.
     */
    const size_t m_npts;
    /**
     * @brief Runtime execution model for device portability
     *
     * Specifies execution policy (CPU, OpenMP, GPU) for computational kernels.
     * Enables device-portable execution across different hardware architectures.
     */
    const RTModel m_class_device;
    /**
     * @brief Reference to simulation state database
     *
     * Provides access to state variable mappings, quadrature functions,
     * and material properties for the analysis region.
     */
    const std::shared_ptr<SimulationState> m_sim_state;
    /**
     * @brief Material region identifier
     *
     * Index of the material region being analyzed. Used to access
     * region-specific state variables and organize output files.
     */
    const int m_region;
    /**
     * @brief Region-specific output file basename
     *
     * Base filename for all LightUp output files including region identifier.
     * Constructed using get_lattice_basename() to ensure unique naming
     * across multiple regions.
     */
    const fs::path m_lattice_basename;
    /**
     * @brief Crystal lattice structure and symmetry operations
     *
     * Instance of LatticeTypeGeneral containing lattice parameters,
     * reciprocal lattice vectors, and point group symmetry operations
     * for the specified crystal system. Provides crystal structure
     * information for all supported Laue groups from cubic to triclinic.
     */
    const LatticeTypeGeneral m_lattice;
    /**
     * @brief Workspace for temporary calculations
     *
     * Partial quadrature function used as temporary storage for intermediate
     * calculations. Avoids repeated memory allocations and enables efficient
     * device-portable computations.
     */
    mfem::expt::PartialQuadratureFunction m_workspace;
    /**
     * @brief In-fiber boolean arrays for each HKL direction
     *
     * Vector of boolean arrays indicating which quadrature points have
     * crystal orientations aligned with each HKL direction (within tolerance).
     * First entry [0] is always true (overall average), subsequent entries
     * correspond to specific HKL directions.
     */
    std::vector<mfem::Array<bool>> m_in_fibers;
    /**
     * @brief Rotation matrices for crystal symmetry operations
     *
     * Vector of MFEM vectors containing rotation matrices that transform
     * HKL directions through all crystal symmetry operations of the
     * lattice's point group. Each vector contains NSYM*3 values representing
     * the transformed direction vectors for one HKL direction, where NSYM
     * is determined by the crystal system.
     */
    std::vector<mfem::Vector> m_rmat_fr_qsym_c_dir;
};