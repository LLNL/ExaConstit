#pragma once

#include "option_types.hpp"
#include "mfem.hpp"

#include <utility>
#include <unordered_map>
#include <string>

template<class LatticeType>
class LightUp {
public:

LightUp(const std::vector<double[3]> hkls,
        const double distance_tolerance,
        const double s_dir[3],
        const mfem::ParFiniteElementSpace* pfes,
        const mfem::QuadratureSpaceBase* qspace,
        const std::unordered_map<std::string, std::pair<int, int> > qf_mapping,
        const RTModel rtmodel,
        const std::string lattice_basename,
        const double lattice_params[3]);

~LightUp() = default;

void calculate_lightup_data(const mfem::QuadratureFunction& history,
                            const mfem::QuadratureFunction& stress);

void calculate_in_fibers(const mfem::QuadratureFunction& history,
                         const double quats_offset,
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
                                      std::vector<double> &lattice_tay_fac,
                                      std::vector<double> &lattice_dpeff);

void calc_lattice_directional_stiffness(const mfem::QuadratureFunction& history,
                                        const mfem::QuadratureFunction& stress,
                                        const size_t strain_offset,
                                        const size_t quats_offset,
                                        const size_t rel_vol_offset,
                                        std::vector<double[3]> &lattice_dir_stiff);

private:
    const std::vector<double[3]> m_hkls;
    const double m_distance_tolerance;
    const double m_s_dir[3];
    const mfem::ParFiniteElementSpace* m_pfes;
    const size_t m_npts;
    const RTModel m_class_device;
    const std::unordered_map<std::string, std::pair<int, int> > qf_mapping;
    const std::string m_lattice_basename;
    const LatticeType m_lattice;
    mfem::QuadratureFunction m_workspace;
    std::vector<mfem::Array<bool>> m_in_fibers;
    std::vector<double[LatticeType::NSYM][3]> m_rmat_fr_qsym_c_dir;
};

class LatticeTypeCubic {
public:
constexpr size_t NSYM = 24;

LatticeTypeCubic(const double lattice_param_a[3]);
~LatticeTypeCubic() = default;

void compute_lattice_b_param(const double lparam_a[3]);
void symmetric_cubic_quaternions();

public:
    double lattice_b[3][3];
    double quat_symm[24][4];
};

using LightUpCubic = LightUp<LatticeTypeCubic>;