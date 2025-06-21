#pragma once

#include "options/option_parser_v2.hpp"
#include "mechanics_kernels.hpp"

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

template<class LatticeType>
class LightUp {
public:

LightUp(const std::vector<std::array<double, 3>> &hkls,
        const double distance_tolerance,
        const std::array<double, 3> s_dir,
        const mfem::ParFiniteElementSpace* pfes,
        mfem::QuadratureSpaceBase* qspace,
        const std::unordered_map<std::string, std::pair<int, int> > &qf_mapping,
        const RTModel &rtmodel,
        const std::string &lattice_basename,
        const std::array<double, 3> lattice_params);

~LightUp() = default;

void calculate_lightup_data(const mfem::QuadratureFunction& history,
                            const mfem::QuadratureFunction& stress);

void calculate_in_fibers(const mfem::QuadratureFunction& history,
                         const size_t quats_offset,
                         const size_t hkl_index);


void calc_lattice_strains(const mfem::QuadratureFunction& history,
                          const size_t strain_offset,
                          const size_t quats_offset,
                          const size_t rel_vol_offset,
                          std::vector<double>& lattice_strains_output,
                          std::vector<double>& lattice_volumes_output);

void calc_lattice_taylor_factor_dpeff(const mfem::QuadratureFunction& history,
                                      const size_t dpeff_offset,
                                      const size_t gdot_offset,
                                      const size_t gdot_length,
                                      std::vector<double> &lattice_tay_facs,
                                      std::vector<double> &lattice_dpeff);

void calc_lattice_directional_stiffness(const mfem::QuadratureFunction& history,
                                        const mfem::QuadratureFunction& stress,
                                        const size_t strain_offset,
                                        const size_t quats_offset,
                                        const size_t rel_vol_offset,
                                        std::vector<std::array<double, 3>> &lattice_dir_stiff);

private:
    std::vector<std::array<double, 3>> m_hkls;
    const double m_distance_tolerance;
    double m_s_dir[3];
    const mfem::ParFiniteElementSpace* m_pfes;
    const size_t m_npts;
    const RTModel m_class_device;
    const std::unordered_map<std::string, std::pair<int, int> > m_qf_mapping;
    const std::string m_lattice_basename;
    const LatticeType m_lattice;
    mfem::QuadratureFunction m_workspace;
    std::vector<mfem::Array<bool>> m_in_fibers;
    std::vector<mfem::Vector> m_rmat_fr_qsym_c_dir;
};

class LatticeTypeCubic {
public:
static constexpr size_t NSYM = 24;

LatticeTypeCubic(const std::array<double, 3> lattice_param_a)
{
    symmetric_cubic_quaternions();
    compute_lattice_b_param(lattice_param_a);
}

~LatticeTypeCubic() = default;

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
    double lattice_b[3 * 3];
    double quat_symm[24 * 4];
};

namespace no_std {
template<typename T>
struct IsStdArray : std::false_type {};
template<typename T, std::size_t N>
struct IsStdArray<std::array<T, N>> : std::true_type {};
}

template<typename T, std::size_t N>
void printArray(std::ostream &stream, std::array<T,N> &array) {
    stream << "\"[ ";
    for (size_t i = 0; i < N - 1; i++) {
        stream << std::scientific << std::setprecision(6) << array[i] << ",";
    }
     stream << array[N - 1] << " ]\"\t";
}

template <typename T>
void printValues(std::ostream &stream, T& t) {
    if constexpr (no_std::IsStdArray<T>::value) {
        printArray(stream, t);
    }
    else {
        stream << std::scientific << std::setprecision(6) << t << "\t";
    }
}

__ecmech_hdev__
inline
void 
quat2rmat(const double* const quat,
          double* const rmats) 
{
    double qbar =  quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

    double* rmat[3] = {&rmats[0], &rmats[3], &rmats[6]};

    rmat[0][0] = qbar + 2.0 * quat[1] * quat[1];
    rmat[1][0] = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    rmat[2][0] = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);

    rmat[0][1] = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    rmat[1][1] = qbar + 2.0 * quat[2] * quat[2];
    rmat[2][1] = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);

    rmat[0][2] = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
    rmat[1][2] = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
    rmat[2][2] = qbar + 2.0 * quat[3] * quat[3];
}

template<class LatticeType>
LightUp<LatticeType>::LightUp(const std::vector<std::array<double, 3>> &hkls,
                 const double distance_tolerance,
                 const std::array<double, 3> s_dir,
                 const mfem::ParFiniteElementSpace* pfes,
                 mfem::QuadratureSpaceBase* qspace,
                 const std::unordered_map<std::string, std::pair<int, int> > &qf_mapping,
                 const RTModel &rtmodel,
                 const std::string &lattice_basename,
                 const std::array<double, 3> lattice_params) : 
    m_hkls(hkls),
    m_distance_tolerance(distance_tolerance),
    m_pfes(pfes),
    m_npts(qspace->GetSize()),
    m_class_device(rtmodel),
    m_qf_mapping(qf_mapping),
    m_lattice_basename(lattice_basename),
    m_lattice(lattice_params)
{
    m_s_dir[0] = s_dir[0];
    m_s_dir[1] = s_dir[1];
    m_s_dir[2] = s_dir[2];

    m_workspace.SetSpace(qspace, 3);

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
LightUp<LatticeType>::calculate_lightup_data(const mfem::QuadratureFunction& history,
                                const mfem::QuadratureFunction& stress)
{
    std::string s_estrain = "elastic_strain";
    std::string s_rvol = "relative_volume";
    std::string s_quats = "quats";
    std::string s_gdot = "shear_rate";
    std::string s_shrateEff = "eq_pl_strain_rate";

    const size_t quats_offset = m_qf_mapping.find(s_quats)->second.first;
    const size_t strain_offset = m_qf_mapping.find(s_estrain)->second.first;
    const size_t rel_vol_offset = m_qf_mapping.find(s_rvol)->second.first;
    const size_t dpeff_offset = m_qf_mapping.find(s_shrateEff)->second.first;
    const size_t gdot_offset = m_qf_mapping.find(s_gdot)->second.first;
    const size_t gdot_length = m_qf_mapping.find(s_gdot)->second.second;

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
LightUp<LatticeType>::calculate_in_fibers(const mfem::QuadratureFunction& history,
                             const size_t quats_offset,
                             const size_t hkl_index)
{
    // Same could be said for in_fiber down here
    // that way we just need to know which hkl and quats we're running with
    const size_t vdim = history.GetVDim();
    const auto history_data = history.Read();

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
LightUp<LatticeType>::calc_lattice_strains(const mfem::QuadratureFunction& history,
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

    const size_t vdim = history.GetVDim();
    const auto history_data = history.Read();
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
        const double lat_vol = exaconstit::kernel::ComputeVolAvgTensorFilter<true>(m_pfes, &m_workspace, &in_fiber_hkl, lattice_strain_hkl, 1, m_class_device);

        lattice_volumes_output.push_back(lat_vol);
        lattice_strains_output.push_back(lattice_strain_hkl(0));
    }
}

template<class LatticeType>
void
LightUp<LatticeType>::calc_lattice_taylor_factor_dpeff(const mfem::QuadratureFunction& history,
                                          const size_t dpeff_offset,
                                          const size_t gdot_offset,
                                          const size_t gdot_length,
                                          std::vector<double> &lattice_tay_facs,
                                          std::vector<double> &lattice_dpeff)
{

    const size_t vdim = history.GetVDim();
    const auto history_data = history.Read();
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
        [[maybe_unused]] double _ = exaconstit::kernel::ComputeVolAvgTensorFilter<true>(m_pfes, &m_workspace, &in_fiber_hkl, lattice_tayfac_dpeff_hkl, 2, m_class_device);
        lattice_tay_facs.push_back(lattice_tayfac_dpeff_hkl(0));
        lattice_dpeff.push_back(lattice_tayfac_dpeff_hkl(1));
    }
}


template<class LatticeType>
void
LightUp<LatticeType>::calc_lattice_directional_stiffness(const mfem::QuadratureFunction& history,
                                            const mfem::QuadratureFunction& stress,
                                            const size_t strain_offset,
                                            const size_t quats_offset,
                                            const size_t rel_vol_offset,
                                            std::vector<std::array<double, 3>> &lattice_dir_stiff)
{

    const size_t vdim = history.GetVDim();
    const auto history_data = history.Read();
    const auto stress_data  = stress.Read();
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
        [[maybe_unused]] double _ = exaconstit::kernel::ComputeVolAvgTensorFilter<true>(m_pfes, &m_workspace, &in_fiber_hkl, lattice_direct_stiff, 3, m_class_device);
        std::array<double, 3> stiff_tmp;
        for (size_t ipt = 0; ipt < 3; ipt++) {
            stiff_tmp[ipt] = lattice_direct_stiff(ipt);
        }
        lattice_dir_stiff.push_back(stiff_tmp);
    }
}

using LightUpCubic = LightUp<LatticeTypeCubic>;