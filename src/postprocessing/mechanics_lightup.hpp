#pragma once

#include "options/option_parser_v2.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "mfem_expt/partial_qfunc.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/rotations.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "SNLS_linalg.h"
#include "ECMech_const.h"
#include "ECMech_gpu_portability.h"

#include <utility>
#include <unordered_map>
#include <string>
#include <array>
#include <vector>

#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <type_traits>

/**
 * @brief Lattice strain analysis class for powder diffraction simulation
 * 
 * @tparam LatticeType Crystal lattice type (e.g., LatticeTypeCubic)
 * 
 * The LightUp class performs in-situ lattice strain calculations that simulate
 * powder diffraction experiments on polycrystalline materials. It computes
 * lattice strains for specified crystallographic directions (HKL) based on
 * crystal orientation evolution and stress state from ExaCMech simulations.
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
template<class LatticeType>
class LightUp {
public:

/**
 * @brief Constructor for LightUp analysis
 * 
 * @param hkls Vector of HKL directions for lattice strain calculation
 * @param distance_tolerance Angular tolerance for fiber direction matching
 * @param s_dir Sample direction vector for reference frame
 * @param pfes Parallel finite element space for mesh information
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
 */
LightUp(const std::vector<std::array<double, 3>> &hkls,
        const double distance_tolerance,
        const std::array<double, 3> s_dir,
        const mfem::ParFiniteElementSpace* pfes,
        std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
        const SimulationState &sim_state,
        const int region,
        const RTModel &rtmodel,
        const std::string &lattice_basename,
        const std::array<double, 3> lattice_params);

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
void calculate_lightup_data(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                            const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress);

/**
 * @brief Determine in-fiber orientations for a specific HKL direction
 * 
 * @param history State variable data containing crystal orientations
 * @param quats_offset Offset to quaternion data in state variable array
 * @param hkl_index Index of HKL direction for calculation
 * 
 * Determines which crystal orientations are "in-fiber" (aligned within
 * the distance tolerance) for the specified HKL direction. Uses crystal
 * symmetry operations to find the maximum dot product between the sample
 * direction and all symmetrically equivalent HKL directions.
 * 
 * The algorithm:
 * 1. Extracts quaternion orientations for each quadrature point
 * 2. Converts quaternions to rotation matrices
 * 3. Applies crystal symmetry operations to HKL directions
 * 4. Computes alignment with sample direction
 * 5. Sets boolean flags for orientations within angular tolerance
 */
void calculate_in_fibers(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
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
void calc_lattice_strains(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
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
void calc_lattice_taylor_factor_dpeff(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                                      const size_t dpeff_offset,
                                      const size_t gdot_offset,
                                      const size_t gdot_length,
                                      std::vector<double> &lattice_tay_facs,
                                      std::vector<double> &lattice_dpeff);

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
void calc_lattice_directional_stiffness(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                                        const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress,
                                        const size_t strain_offset,
                                        const size_t quats_offset,
                                        const size_t rel_vol_offset,
                                        std::vector<std::array<double, 3>> &lattice_dir_stiff);
/**
 * @brief Get the region ID for this LightUp instance
 * 
 * @return Region identifier
 * 
 * Returns the material region index associated with this LightUp analysis.
 * Used for accessing region-specific data and organizing multi-region output.
 */
int get_region_id() const { return m_region; }

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
     * @brief Pointer to parallel finite element space
     * 
     * Provides access to mesh and finite element information for the
     * analysis region. Used for geometric calculations and data layout.
     */
    const mfem::ParFiniteElementSpace* m_pfes;
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
    const SimulationState& m_sim_state;
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
    const std::string m_lattice_basename;
    /**
     * @brief Crystal lattice structure and symmetry operations
     * 
     * Instance of the lattice type (e.g., LatticeTypeCubic) containing
     * lattice parameters, reciprocal lattice vectors, and symmetry operations.
     * Provides crystal structure information for calculations.
     */
    const LatticeType m_lattice;
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
     * HKL directions through all crystal symmetry operations. Each vector
     * contains NSYM*3 values representing the transformed direction vectors
     * for one HKL direction.
     */
    std::vector<mfem::Vector> m_rmat_fr_qsym_c_dir;
};

/**
 * @brief Cubic crystal lattice structure and symmetry operations
 * 
 * Provides cubic crystal lattice parameters, reciprocal lattice vectors,
 * and the 24 symmetry operations of the cubic point group. Used by
 * LightUp for crystal-structure-specific calculations.
 * 
 * The class computes reciprocal lattice vectors from direct lattice
 * parameters and generates symmetry-equivalent directions for HKL families.
 * 
 * @ingroup ExaConstit_postprocessing_lightup
 */
class LatticeTypeCubic {
public:
/**
 * @brief Number of symmetry operations for cubic crystals
 * 
 * Cubic point group has 24 symmetry operations (rotations and inversions).
 * Used for generating symmetrically equivalent crystallographic directions.
 */
static constexpr size_t NSYM = 24;

/**
 * @brief Constructor for cubic lattice
 * 
 * @param lattice_param_a Array of lattice parameters [a, b, c]
 * 
 * Initializes cubic lattice structure by computing reciprocal lattice
 * vectors and generating the 24 cubic symmetry quaternions.
 */
LatticeTypeCubic(const std::array<double, 3> lattice_param_a)
{
    symmetric_cubic_quaternions();
    compute_lattice_b_param(lattice_param_a);
}

~LatticeTypeCubic() = default;

/**
 * @brief Compute reciprocal lattice parameter matrix
 * 
 * @param lparam_a Direct lattice parameters [a, b, c]
 * 
 * Computes the reciprocal lattice vectors (lattice_b matrix) from
 * direct lattice parameters. For cubic crystals, assumes 90-degree
 * angles between axes. The reciprocal lattice is used to transform
 * HKL indices to direction vectors in reciprocal space.
 */
void
compute_lattice_b_param(const std::array<double, 3> lparam_a)
{
    constexpr double FRAC_PI_2 = 1.57079632679489661923132169163975144;
    const double cellparms[6] = {lparam_a[0], lparam_a[1], lparam_a[2], FRAC_PI_2, FRAC_PI_2, FRAC_PI_2};

    const double alfa = cellparms[3];
    const double beta = cellparms[4];
    const double gamma = cellparms[5];

    const double cosalfar = (cos(beta) * cos(gamma) - cos(alfa)) / (sin(beta) * sin(gamma));
    const double sinalfar = sqrtf(1.0 - cosalfar * cosalfar);

    const double a[3] = {cellparms[0], 0.0, 0.0};
    const double b[3] = {cellparms[1] * cos(gamma), cellparms[1] * sin(gamma), 0.0};
    const double c[3] = {cellparms[2] * cos(beta), -cellparms[2] * cosalfar * sin(beta), cellparms[2] * sinalfar * sin(beta)};

    // Cell volume
    double vol[3] = {};
    auto cross_prod = [&](const double* const vec1, 
                          const double* const vec2,
                          double* const prod) {
        prod[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        prod[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        prod[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];  
    };

    cross_prod(b, c, vol);
    const double inv_vol = 1.0 / snls::linalg::dotProd<3>(a, vol);

    // Reciprocal lattice vectors
    auto cross_prod_inv_v = [&](const double* const vec1, const double* const vec2, double* cross_prod_v) {
        cross_prod(vec1, vec2, cross_prod_v);
        cross_prod_v[0] *= inv_vol;
        cross_prod_v[1] *= inv_vol;
        cross_prod_v[2] *= inv_vol;
    };

    double * latb[3] = {&lattice_b[0], &lattice_b[3], &lattice_b[6]}; 
    // B takes components in the reciprocal lattice to X
    cross_prod_inv_v(b, c, latb[0]);
    cross_prod_inv_v(c, a, latb[1]);
    cross_prod_inv_v(a, b, latb[2]);
} 

/**
 * @brief Generate cubic crystal symmetry quaternions
 * 
 * Computes the 24 quaternions representing all symmetry operations
 * of the cubic point group. These quaternions are used to generate
 * symmetrically equivalent HKL directions for lattice strain analysis.
 * 
 * The symmetry operations include rotations about 4-fold, 3-fold,
 * and 2-fold axes, plus inversion operations.
 */
void 
symmetric_cubic_quaternions() 
{
    constexpr double PI = 3.14159265358979323846264338327950288;
    constexpr double FRAC_PI_2 = 1.57079632679489661923132169163975144;
    constexpr double FRAC_PI_3 = 1.04719755119659774615421446109316763;

    constexpr double angle_axis_symm [NSYM][4] = {
            {0.0, 1.0, 0.0, 0.0},  // identity
            {FRAC_PI_2, 1.0, 0.0, 0.0},  // fourfold about   1  0  0 (x1)
            {PI, 1.0, 0.0, 0.0},  //
            {FRAC_PI_2 * 3.0, 1.0, 0.0, 0.0},  //
            {FRAC_PI_2, 0.0, 1.0, 0.0},  // fourfold about   0  1  0 (x2)
            {PI, 0.0, 1.0, 0.0},  //
            {FRAC_PI_2 * 3.0, 0.0, 1.0, 0.0},  //
            {FRAC_PI_2, 0.0, 0.0, 1.0},  // fourfold about   0  0  1 (x3)
            {PI, 0.0, 0.0, 1.0},  //
            {FRAC_PI_2 * 3.0, 0.0, 0.0, 1.0},  //
            {FRAC_PI_3 * 2.0, 1.0, 1.0, 1.0},  // threefold about  1  1  1
            {FRAC_PI_3 * 4.0, 1.0, 1.0, 1.0},  //
            {FRAC_PI_3 * 2.0, -1.0, 1.0, 1.0},  // threefold about -1  1  1
            {FRAC_PI_3 * 4.0, -1.0, 1.0, 1.0},  //
            {FRAC_PI_3 * 2.0, -1.0, -1.0, 1.0},  // threefold about -1 -1  1
            {FRAC_PI_3 * 4.0, -1.0, -1.0, 1.0},  //
            {FRAC_PI_3 * 2.0, 1.0, -1.0, 1.0},  // threefold about  1 -1  1
            {FRAC_PI_3 * 4.0, 1.0, -1.0, 1.0},  //
            {PI, 1.0, 1.0, 0.0},  // twofold about    1  1  0
            {PI, -1.0, 1.0, 0.0},  // twofold about   -1  1  0
            {PI, 1.0, 0.0, 1.0},  // twofold about    1  0  1
            {PI, 0.0, 1.0, 1.0},  // twofold about    0  1  1
            {PI, -1.0, 0.0, 1.0},  // twofold about   -1  0  1
            {PI, 0.0, -1.0, 1.0},  // twofold about    0 -1  1
    };

    constexpr double inv2 = 1.0 / 2.0;

    for (size_t isym = 0; isym < NSYM; isym++) {
        double *symm_quat = &quat_symm[isym * 4];
        const double s = sin(inv2 * angle_axis_symm[isym][0]); 
        symm_quat[0] = cos(inv2 * angle_axis_symm[isym][0]);
        double inv_norm_axis = 1.0 / snls::linalg::norm<3>(&angle_axis_symm[isym][1]);
        symm_quat[1] = s * angle_axis_symm[isym][1] * inv_norm_axis;
        symm_quat[2] = s * angle_axis_symm[isym][2] * inv_norm_axis;
        symm_quat[3] = s * angle_axis_symm[isym][3] * inv_norm_axis;

        inv_norm_axis = 1.0;
        if (symm_quat[0] < 0.0) {
            inv_norm_axis *= -1.0;
        }

        symm_quat[0] *= inv_norm_axis;
        symm_quat[1] *= inv_norm_axis;
        symm_quat[2] *= inv_norm_axis;
        symm_quat[3] *= inv_norm_axis;
    }
}

public:
    /**
     * @brief Reciprocal lattice parameter matrix
     * 
     * 3x3 matrix containing reciprocal lattice vectors as columns.
     * Used to transform HKL indices to direction vectors in reciprocal space.
     * Computed from direct lattice parameters in constructor.
     */
    double lattice_b[3 * 3];
    /**
     * @brief Cubic symmetry quaternions
     * 
     * Array of 24 quaternions (96 double values) representing all
     * symmetry operations of the cubic point group. Each quaternion
     * is stored as [q0, q1, q2, q3] where q0 is the scalar component.
     */
    double quat_symm[24 * 4];
};

/**
 * @brief Type trait for detecting std::array types
 * 
 * Helper template for template metaprogramming to distinguish
 * std::array types from other types in generic printing functions.
 */
namespace no_std {
template<typename T>
struct IsStdArray : std::false_type {};
template<typename T, std::size_t N>
struct IsStdArray<std::array<T, N>> : std::true_type {};
}

/**
 * @brief Print std::array to output stream with formatting
 * 
 * @tparam T Array element type
 * @tparam N Array size
 * @param stream Output stream for writing
 * @param array Array to print
 * 
 * Formats std::array output as "[ val1, val2, val3 ]" with scientific
 * notation and 6-digit precision. Used for consistent formatting of
 * HKL directions and other array data in output files.
 */
template<typename T, std::size_t N>
void printArray(std::ostream &stream, std::array<T,N> &array) {
    stream << "\"[ ";
    for (size_t i = 0; i < N - 1; i++) {
        stream << std::scientific << std::setprecision(6) << array[i] << ",";
    }
     stream << array[N - 1] << " ]\"\t";
}

/**
 * @brief Generic value printing with type-specific formatting
 * 
 * @tparam T Value type
 * @param stream Output stream for writing
 * @param t Value to print
 * 
 * Prints values with appropriate formatting based on type:
 * - std::array types use printArray() for structured output
 * - Other types use scientific notation with 6-digit precision
 * 
 * Enables generic output formatting for different data types
 * in LightUp file output operations.
 */
template <typename T>
void printValues(std::ostream &stream, T& t) {
    if constexpr (no_std::IsStdArray<T>::value) {
        printArray(stream, t);
    }
    else {
        stream << std::scientific << std::setprecision(6) << t << "\t";
    }
}

/**
 * @brief Generate region-specific lattice output basename
 * 
 * @param lattice_basename Base filename from configuration
 * @param region_id Region identifier
 * @return Region-specific filename prefix
 * 
 * Constructs unique output file basename by appending region identifier.
 * Format: "basename_region_N_" where N is the region ID. Ensures
 * separate output files for each material region in multi-region simulations.
 */
std::string get_lattice_basename(const std::string& lattice_basename, const int region_id) {
    return lattice_basename + "region_" + std::to_string(region_id) + 
"_";
}

template<class LatticeType>
LightUp<LatticeType>::LightUp(const std::vector<std::array<double, 3>> &hkls,
                 const double distance_tolerance,
                 const std::array<double, 3> s_dir,
                 const mfem::ParFiniteElementSpace* pfes,
                 std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
                 const SimulationState &sim_state,
                 const int region,
                 const RTModel &rtmodel,
                 const std::string &lattice_basename,
                 const std::array<double, 3> lattice_params) : 
    m_hkls(hkls),
    m_distance_tolerance(distance_tolerance),
    m_pfes(pfes),
    m_npts(qspace->GetSize()),
    m_class_device(rtmodel),
    m_sim_state(sim_state),
    m_region(region),
    m_lattice_basename(get_lattice_basename(lattice_basename, region)),
    m_lattice(lattice_params),
    m_workspace(qspace, 3)
{
    m_s_dir[0] = s_dir[0];
    m_s_dir[1] = s_dir[1];
    m_s_dir[2] = s_dir[2];

    const double inv_s_norm = 1.0 / snls::linalg::norm<3>(m_s_dir);
    m_s_dir[0] *= inv_s_norm;
    m_s_dir[1] *= inv_s_norm;
    m_s_dir[2] *= inv_s_norm;

    auto lat_vec_ops_b = m_lattice.lattice_b;
    // First one we'll always set to be all the values
    m_in_fibers.push_back(mfem::Array<bool>(m_npts));
    for (auto &hkl: hkls) {
        m_in_fibers.push_back(mfem::Array<bool>(m_npts));
        // Computes reciprocal lattice B but different from HEXRD we return as row matrix as that's the easiest way of doing things
        double c_dir[3];
        // compute crystal direction from planeData
        snls::linalg::matTVecMult<3,3>(lat_vec_ops_b, hkl.data(), c_dir);

        const double inv_c_norm = 1.0 / snls::linalg::norm<3>(c_dir);
        c_dir[0] *= inv_c_norm;
        c_dir[1] *= inv_c_norm;
        c_dir[2] *= inv_c_norm;

        // Could maybe move this over to a vec if we want this to be easily generic over a ton of symmetry conditions...
        double rmat_fr_qsym_c_dir[LatticeType::NSYM][3] = {};
        mfem::Vector tmp(LatticeType::NSYM * 3);
        for (size_t isym=0; isym < LatticeType::NSYM; isym++) {
            double rmat[3 * 3] = {};
            quat2rmat(&m_lattice.quat_symm[isym * 4], rmat);
            snls::linalg::matTVecMult<3,3>(rmat, c_dir, rmat_fr_qsym_c_dir[isym]);
            tmp(isym * 3 + 0) = rmat_fr_qsym_c_dir[isym][0];
            tmp(isym * 3 + 1) = rmat_fr_qsym_c_dir[isym][1];
            tmp(isym * 3 + 2) = rmat_fr_qsym_c_dir[isym][2];
        }
        tmp.UseDevice(true);
        m_rmat_fr_qsym_c_dir.push_back(tmp);
    }

    m_hkls.insert(m_hkls.begin(), {0.0, 0.0, 0.0});
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    // Now we're going to save off the lattice values to a file
    if (my_id == 0) {

        auto file_line_print = [&](auto& basename, auto& name, auto &m_hkls) {
            std::string filename = basename + name;
            std::ofstream file;
            file.open(filename, std::ios_base::out);

            file << "#" << "\t";

            for (auto& item : m_hkls) {
                file << std::setprecision(1) << "\"[ " <<item[0] << ", " << item[1] << ", " << item[2] << " ]\"" << "\t";
            }
            file << std::endl;

            file.close();
        };

        file_line_print(m_lattice_basename, "strains.txt", m_hkls);
        file_line_print(m_lattice_basename, "volumes.txt", m_hkls);
        file_line_print(m_lattice_basename, "dpeff.txt", m_hkls);
        file_line_print(m_lattice_basename, "taylor_factor.txt", m_hkls);
        file_line_print(m_lattice_basename, "directional_stiffness.txt", m_hkls);
    }

    /* add a working array for the QF and in_fiber arrays */
    // If we really wanted to we could lower try and calculate the elements
    // that aren't unique here but that's not worth the effort at all given
    // how fast things are
    // let c_syms: Vec<[f64; 3]> = find_unique_tolerance::<SYM_LEN>(&rmat_fr_qsym_c_dir, f64::sqrt(f64::EPSILON));

    // Move all of the above to the object constructor
    // rmat_fr_qsym_c_dir move to an mfem vector and then use it's data down here
    // same with s_dir and c_dir
    // Here iterate on which HKL we're using maybe have a map for these rmat_fr_qsym_c_dir and c_dir
}

template<class LatticeType>
void
LightUp<LatticeType>::calculate_lightup_data(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                                             const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress)
{
    std::string s_estrain = "elastic_strain";
    std::string s_rvol = "relative_volume";
    std::string s_quats = "quats";
    std::string s_gdot = "shear_rate";
    std::string s_shrateEff = "eq_pl_strain_rate";

    const size_t quats_offset = m_sim_state.GetQuadratureFunctionStatePair(s_quats, m_region).first;
    const size_t strain_offset = m_sim_state.GetQuadratureFunctionStatePair(s_estrain, m_region).first;
    const size_t rel_vol_offset = m_sim_state.GetQuadratureFunctionStatePair(s_rvol, m_region).first;
    const size_t dpeff_offset = m_sim_state.GetQuadratureFunctionStatePair(s_shrateEff, m_region).first;
    const size_t gdot_offset = m_sim_state.GetQuadratureFunctionStatePair(s_gdot, m_region).first;
    const size_t gdot_length = m_sim_state.GetQuadratureFunctionStatePair(s_gdot, m_region).second;

    m_in_fibers[0] = true;
    for (size_t ihkl = 0; ihkl < m_rmat_fr_qsym_c_dir.size(); ihkl++) {
        calculate_in_fibers(history, quats_offset, ihkl);
    }

    std::vector<double> lattice_strains_output;
    std::vector<double> lattice_volumes_output;

    calc_lattice_strains(history, strain_offset, quats_offset, rel_vol_offset, lattice_strains_output, lattice_volumes_output);

    std::vector<double> lattice_dpeff_output;
    std::vector<double> lattice_tayfac_output;

    calc_lattice_taylor_factor_dpeff(history, dpeff_offset, gdot_offset, gdot_length, lattice_tayfac_output, lattice_dpeff_output);

    std::vector<std::array<double, 3>> lattice_dir_stiff_output;

    calc_lattice_directional_stiffness(history, stress, strain_offset, quats_offset, rel_vol_offset, lattice_dir_stiff_output);

    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    // Now we're going to save off the lattice values to a file
    if (my_id == 0) {

        auto file_line_print = [&](auto& basename, auto& name, auto &vec) {
            std::string filename = basename + name;
            std::ofstream file;
            file.open(filename, std::ios_base::app);

            for (auto& item : vec) {
                printValues(file, item);
            }
            file << std::endl;

            file.close();
        };

        file_line_print(m_lattice_basename, "strains.txt", lattice_strains_output);
        file_line_print(m_lattice_basename, "volumes.txt", lattice_volumes_output);
        file_line_print(m_lattice_basename, "dpeff.txt", lattice_dpeff_output);
        file_line_print(m_lattice_basename, "taylor_factor.txt", lattice_tayfac_output);
        file_line_print(m_lattice_basename, "directional_stiffness.txt", lattice_dir_stiff_output);
    }

}

template<class LatticeType>
void
LightUp<LatticeType>::calculate_in_fibers(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                             const size_t quats_offset,
                             const size_t hkl_index)
{
    // Same could be said for in_fiber down here
    // that way we just need to know which hkl and quats we're running with
    const size_t vdim = history->GetVDim();
    const auto history_data = history->Read();

    // First hkl_index is always completely true so we can easily
    // compute the total volume average values
    auto in_fiber_view = m_in_fibers[hkl_index + 1].Write();
    auto rmat_fr_qsym_c_dir = m_rmat_fr_qsym_c_dir[hkl_index].Read();

    mfem::Vector s_dir(3);
    s_dir[0] = m_s_dir[0]; s_dir[1] = m_s_dir[1]; s_dir[2] = m_s_dir[2];
    auto s_dir_data = s_dir.Read();
    auto distance_tolerance = m_distance_tolerance;

    mfem::MFEM_FORALL(iquats, m_npts, {
    // for(size_t iquats = 0; iquats < m_npts; iquats++) {

        const auto quats = &history_data[iquats * vdim + quats_offset];
        double rmat[3 * 3] = {};
        quat2rmat(quats, rmat);

        double sine = -10;
        for (size_t isym = 0; isym < LatticeType::NSYM; isym++) {
            double prod[3] = {};
            snls::linalg::matVecMult<3,3>(rmat, &rmat_fr_qsym_c_dir[isym * 3], prod);
            double tmp = snls::linalg::dotProd<3>(s_dir_data, prod);
            sine = (tmp > sine) ? tmp : sine;
        }
        if (fabs(sine) > 1.00000001) {
            sine = (sine >= 0) ? 1.0 : -1.0;
        }
        in_fiber_view[iquats] = acos(sine) <= distance_tolerance;
    });
}

template<class LatticeType>
void
LightUp<LatticeType>::calc_lattice_strains(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                              const size_t strain_offset,
                              const size_t quats_offset,
                              const size_t rel_vol_offset,
                              std::vector<double>& lattice_strains_output,
                              std::vector<double>& lattice_volumes_output)
{
    const double project_vec[6] = {m_s_dir[0] * m_s_dir[0],
                                   m_s_dir[1] * m_s_dir[1],
                                   m_s_dir[2] * m_s_dir[2],
                                   2.0 * m_s_dir[1] * m_s_dir[2],
                                   2.0 * m_s_dir[0] * m_s_dir[2],
                                   2.0 * m_s_dir[0] * m_s_dir[1]};

    const size_t vdim = history->GetVDim();
    const auto history_data = history->Read();
    m_workspace = 0.0;
    auto lattice_strains = m_workspace.Write();

    // Only need to compute this once
    mfem::MFEM_FORALL(iqpts, m_npts, {
    // for(size_t iqpts = 0; iqpts < m_npts; iqpts++) {
        const auto strain_lat = &history_data[iqpts * vdim + strain_offset];
        const auto quats = &history_data[iqpts * vdim + quats_offset];
        const auto rel_vol = history_data[iqpts * vdim + rel_vol_offset];

        double strain[6] = {};
        {
            double strainm[3 * 3] = {};
            double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
            const double t1 = ecmech::sqr2i * strain_lat[0];
            const double t2 = ecmech::sqr6i * strain_lat[1];
            //
            // Volume strain is ln(V^e_mean) term aka ln(relative volume)
            // Our plastic deformation has a det(1) aka no change in volume change
            const double elas_vol_strain = log(rel_vol);
            // We output elastic strain formulation such that the relationship
            // between V^e and \varepsilon is just V^e = I + \varepsilon
            strain_m[0][0] = (t1 - t2) + elas_vol_strain; // 11
            strain_m[1][1] = (-t1 - t2) + elas_vol_strain ; // 22
            strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
            strain_m[1][2] = ecmech::sqr2i * strain_lat[4]; // 23
            strain_m[2][0] = ecmech::sqr2i * strain_lat[3]; // 31
            strain_m[0][1] = ecmech::sqr2i * strain_lat[2]; // 12

            strain_m[2][1] = strain_m[1][2];
            strain_m[0][2] = strain_m[2][0];
            strain_m[1][0] = strain_m[0][1];

            double rmat[3 * 3] = {};
            double strain_samp[3 * 3] = {};

            quat2rmat(quats, rmat);
            snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);

            strain_m[0] = &strain_samp[0];
            strain_m[1] = &strain_samp[3];
            strain_m[2] = &strain_samp[6];
            strain[0] = strain_m[0][0];
            strain[1] = strain_m[1][1];
            strain[2] = strain_m[2][2];
            strain[3] = strain_m[1][2];
            strain[4] = strain_m[0][2];
            strain[5] = strain_m[0][1];

        }
        const double proj_strain = snls::linalg::dotProd<6>(project_vec, strain);
        lattice_strains[iqpts] = proj_strain;

    });

    for (const auto& in_fiber_hkl : m_in_fibers){
        mfem::Vector lattice_strain_hkl(1);
        auto region_comm = m_sim_state.GetRegionCommunicator(m_region);
        const double lat_vol = exaconstit::kernel::ComputeVolAvgTensorFilterFromPartial<true>(&m_workspace, &in_fiber_hkl, lattice_strain_hkl, 1, m_class_device, region_comm);

        lattice_volumes_output.push_back(lat_vol);
        lattice_strains_output.push_back(lattice_strain_hkl(0));
    }
}

template<class LatticeType>
void
LightUp<LatticeType>::calc_lattice_taylor_factor_dpeff(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                                          const size_t dpeff_offset,
                                          const size_t gdot_offset,
                                          const size_t gdot_length,
                                          std::vector<double> &lattice_tay_facs,
                                          std::vector<double> &lattice_dpeff)
{

    const size_t vdim = history->GetVDim();
    const auto history_data = history->Read();
    m_workspace = 0.0;
    auto lattice_tayfac_dpeffs = m_workspace.Write();

    // Only need to compute this once
    mfem::MFEM_FORALL(iqpts, m_npts, {
    // for(size_t iqpts = 0; iqpts < m_npts; iqpts++) {
        const auto dpeff = &history_data[iqpts * vdim + dpeff_offset];
        const auto gdots = &history_data[iqpts * vdim + gdot_offset];
        auto lattice_tayfac_dpeff = &lattice_tayfac_dpeffs[iqpts * 2];
        double abs_gdot = 0.0;
        for (size_t islip = 0; islip < gdot_length; islip++) {
            abs_gdot += fabs(gdots[islip]);
        }
        lattice_tayfac_dpeff[0] = (fabs(*dpeff) <= 1.0e-14) ? 0.0 : (abs_gdot / *dpeff);
        lattice_tayfac_dpeff[1] = *dpeff;
    });

    for (const auto& in_fiber_hkl : m_in_fibers){
        mfem::Vector lattice_tayfac_dpeff_hkl(2);
        auto region_comm = m_sim_state.GetRegionCommunicator(m_region);
        [[maybe_unused]] double _ = exaconstit::kernel::ComputeVolAvgTensorFilterFromPartial<true>(&m_workspace, &in_fiber_hkl, lattice_tayfac_dpeff_hkl, 2, m_class_device, region_comm);
        lattice_tay_facs.push_back(lattice_tayfac_dpeff_hkl(0));
        lattice_dpeff.push_back(lattice_tayfac_dpeff_hkl(1));
    }
}


template<class LatticeType>
void
LightUp<LatticeType>::calc_lattice_directional_stiffness(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
                                            const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress,
                                            const size_t strain_offset,
                                            const size_t quats_offset,
                                            const size_t rel_vol_offset,
                                            std::vector<std::array<double, 3>> &lattice_dir_stiff)
{

    const size_t vdim = history->GetVDim();
    const auto history_data = history->Read();
    const auto stress_data  = stress->Read();
    m_workspace = 0.0;
    auto lattice_directional_stiffness = m_workspace.Write();

    // Only need to compute this once
    mfem::MFEM_FORALL(iqpts, m_npts, {
    // for(size_t iqpts = 0; iqpts < m_npts; iqpts++) {
        const auto strain_lat = &history_data[iqpts * vdim + strain_offset];
        const auto quats = &history_data[iqpts * vdim + quats_offset];
        const auto rel_vol = history_data[iqpts * vdim + rel_vol_offset];
        const auto stress_l = &stress_data[iqpts * 6];
        auto lds = &lattice_directional_stiffness[iqpts * 3];

        double strain[6] = {};
        {
            double strainm[3 * 3] = {};
            double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
            const double t1 = ecmech::sqr2i * strain_lat[0];
            const double t2 = ecmech::sqr6i * strain_lat[1];
            //
            // Volume strain is ln(V^e_mean) term aka ln(relative volume)
            // Our plastic deformation has a det(1) aka no change in volume change
            const double elas_vol_strain = log(rel_vol);
            // We output elastic strain formulation such that the relationship
            // between V^e and \varepsilon is just V^e = I + \varepsilon
            strain_m[0][0] = (t1 - t2) + elas_vol_strain; // 11
            strain_m[1][1] = (-t1 - t2) + elas_vol_strain ; // 22
            strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
            strain_m[1][2] = ecmech::sqr2i * strain_lat[4]; // 23
            strain_m[2][0] = ecmech::sqr2i * strain_lat[3]; // 31
            strain_m[0][1] = ecmech::sqr2i * strain_lat[2]; // 12

            strain_m[2][1] = strain_m[1][2];
            strain_m[0][2] = strain_m[2][0];
            strain_m[1][0] = strain_m[0][1];

            double rmat[3 * 3] = {};
            double strain_samp[3 * 3] = {};

            quat2rmat(quats, rmat);

            snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);

            strain_m[0] = &strain_samp[0];
            strain_m[1] = &strain_samp[3];
            strain_m[2] = &strain_samp[6];

            strain[0] = strain_m[0][0];
            strain[1] = strain_m[1][1];
            strain[2] = strain_m[2][2];
            strain[3] = strain_m[1][2];
            strain[4] = strain_m[0][2];
            strain[5] = strain_m[0][1];
        }

        for (size_t ipt = 0; ipt < 3; ipt++) {
            lds[ipt] = (fabs(strain[ipt]) < 1e-12) ? 0.0 : (stress_l[ipt] / strain[ipt]);
        }
    });

    for (const auto& in_fiber_hkl : m_in_fibers){
        mfem::Vector lattice_direct_stiff(3);
        auto region_comm = m_sim_state.GetRegionCommunicator(m_region);
        [[maybe_unused]] double _ = exaconstit::kernel::ComputeVolAvgTensorFilterFromPartial<true>(&m_workspace, &in_fiber_hkl, lattice_direct_stiff, 3, m_class_device, region_comm);
        std::array<double, 3> stiff_tmp;
        for (size_t ipt = 0; ipt < 3; ipt++) {
            stiff_tmp[ipt] = lattice_direct_stiff(ipt);
        }
        lattice_dir_stiff.push_back(stiff_tmp);
    }
}

using LightUpCubic = LightUp<LatticeTypeCubic>;