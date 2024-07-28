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

/*
// Should move this to the strain calculation we do elsewhere...
// Maybe the actual fiber strain calculation
#[inline(always)]
fn rotate_strain_to_sample(quat: &[f64], strain_vec: &mut[f64]) {
    assert!(quat.len() >= 4);
    assert!(strain_vec.len() >= 6);
    let rmat = quat2rmat(quat);
    let strain = 
    {
        let mut strain = [[0.0; 3]; 3];

        strain[0][0] = strain_vec[0];
        strain[1][1] = strain_vec[1];
        strain[2][2] = strain_vec[2];
        strain[1][2] = strain_vec[3]; strain[2][1] = strain[1][2];
        strain[0][2] = strain_vec[4]; strain[2][0] = strain[0][2];
        strain[0][1] = strain_vec[5]; strain[1][0] = strain[0][1];

        strain
    };

    {
        let mut strain_samp = [[0.0; 3]; 3];
        rotate_matrix::<3, false, f64>(&rmat, &strain, &mut strain_samp);

        strain_vec[0] = strain_samp[0][0];
        strain_vec[1] = strain_samp[1][1];
        strain_vec[2] = strain_samp[2][2];
        strain_vec[3] = strain_samp[1][2];
        strain_vec[4] = strain_samp[0][2];
        strain_vec[5] = strain_samp[0][1];
    }
}

#[inline(always)]
fn calc_volume_terms_fiber(evs_step: &[f64],
                           in_fiber_hkl_step: &[bool],
                           per_rank_update: bool
                          ) -> (f64, f64) {
    let total_lat_vol = evs_step.into_iter()
                        .zip(in_fiber_hkl_step)
                        .filter(|ev_in_fiber| *ev_in_fiber.1)
                        .map(|(ev, _)| *ev)
                        .reduce(|ev_total, ev| {
                            ev_total + ev
                        }).unwrap_or(0.0);

    let inv_total_lat_vol = 
    if per_rank_update {
        1.0
    } else {
        if total_lat_vol > f64::EPSILON {
            1.0 / total_lat_vol
        } else {
            0.0
        }
    };
    (total_lat_vol, inv_total_lat_vol)
}

#[inline(always)]
fn calculate_in_fibers(lparam_a: f64,
                       hkl: &[f64],
                       s_dir: &[f64],
                       quats: &[f64],
                       distance_tolerance: f64,
                       in_fiber: &mut [bool]
) {
    // Computes reciprocal lattice B but different from HEXRD we return as row matrix as that's the easiest way of doing things
    let lat_vec_ops_b = compute_lattice_b_param_cubic(lparam_a);

    // compute crystal direction from planeData
    let c_dir = {
        let mut tmp_cdir = [0.0; 3];
        matTVecMult::<3, 3, f64>(&lat_vec_ops_b, &hkl, &mut tmp_cdir);
        tmp_cdir
    };

    let symm_quat = symmetric_cubic_quaternions();

    within_fiber::<24>(&c_dir, s_dir, quats, &symm_quat, distance_tolerance, in_fiber);
}

#[inline(always)]
fn quat2rmat(quat: &[f64]) -> [[f64; 3]; 3] {
    assert!(quat.len() >= 4);
    let qbar =  quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

    let mut rmat = [[0.0; 3]; 3];

    rmat[0][0] = qbar + 2.0 * quat[1] * quat[1];
    rmat[1][0] = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    rmat[2][0] = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);

    rmat[0][1] = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    rmat[1][1] = qbar + 2.0 * quat[2] * quat[2];
    rmat[2][1] = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);

    rmat[0][2] = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
    rmat[1][2] = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
    rmat[2][2] = qbar + 2.0 * quat[3] * quat[3];

    rmat
}

#[inline(always)]
fn rotate_strain_to_sample(quat: &[f64], strain_vec: &mut[f64]) {
    assert!(quat.len() >= 4);
    assert!(strain_vec.len() >= 6);
    let rmat = quat2rmat(quat);
    let strain = 
    {
        let mut strain = [[0.0; 3]; 3];

        strain[0][0] = strain_vec[0];
        strain[1][1] = strain_vec[1];
        strain[2][2] = strain_vec[2];
        strain[1][2] = strain_vec[3]; strain[2][1] = strain[1][2];
        strain[0][2] = strain_vec[4]; strain[2][0] = strain[0][2];
        strain[0][1] = strain_vec[5]; strain[1][0] = strain[0][1];

        strain
    };

    {
        let mut strain_samp = [[0.0; 3]; 3];
        rotate_matrix::<3, false, f64>(&rmat, &strain, &mut strain_samp);

        strain_vec[0] = strain_samp[0][0];
        strain_vec[1] = strain_samp[1][1];
        strain_vec[2] = strain_samp[2][2];
        strain_vec[3] = strain_samp[1][2];
        strain_vec[4] = strain_samp[0][2];
        strain_vec[5] = strain_samp[0][1];
    }
}

#[inline(always)]
fn calc_taylor_factor(gdots: &[f64]) -> (f64, f64) {
    assert!(gdots.len() >= 12);
    let symm_schmid = calculate_fcc_symm_schmid_tensor();
    let mut plastic_def_rate_vec = [0.0; 6];
    matTVecMult::<12, 6, f64>(&symm_schmid, gdots, &mut plastic_def_rate_vec);

    let eff_plastic_def_rate = { 
        let norm_vec = norm::<6, f64>(&plastic_def_rate_vec);
        norm_vec * f64::sqrt(2.0 / 3.0) 
    };

    if eff_plastic_def_rate <= f64::EPSILON {
        return (0.0, 0.0)
    }

    let abs_sum_shear_rate = {
        let mut sum = 0.0;
        for gdot in gdots.iter() {
            sum += f64::fabs(*gdot);
        }
        sum
    };

    (abs_sum_shear_rate / eff_plastic_def_rate, eff_plastic_def_rate)
}

/// Computes reciprocal lattice B but different from HEXRD we return as row matrix as that's the easiest way of doing things
#[inline(always)]
fn compute_lattice_b_param_cubic(lparam_a: f64) -> [[f64; 3]; 3] {
    let deg90 = PI / 2.0;
    let cellparms = [lparam_a, lparam_a, lparam_a, deg90, deg90, deg90];

    let alfa = cellparms[3];
    let beta = cellparms[4];
    let gamma = cellparms[5];

    let cosalfar = (f64::cos(beta) * f64::cos(gamma) - f64::cos(alfa)) / (f64::sin(beta) * f64::sin(gamma));
    let sinalfar = f64::sqrt(1.0 - cosalfar * cosalfar);

    let a = [cellparms[0], 0.0, 0.0];
    let b = [cellparms[1] * f64::cos(gamma), cellparms[1] * f64::sin(gamma), 0.0];
    let c = [cellparms[2] * f64::cos(beta), -cellparms[2] * cosalfar * f64::sin(beta), cellparms[2] * sinalfar * f64::sin(beta)];

    // Cell volume
    let inv_vol = {
        let v_temp = cross_prod(&b, &c);
        1.0 / dot_prod::<3, f64>(&a, &v_temp)
    };

    // Reciprocal lattice vectors
    let cross_prod_inv_v = |vec1: &[f64], vec2: &[f64], inv_vol: f64| -> [f64; 3] 
    {
        let mut tmp = cross_prod(vec1, vec2);
        tmp[0] *= inv_vol;
        tmp[1] *= inv_vol;
        tmp[2] *= inv_vol;
        tmp
    };

    let astar = cross_prod_inv_v(&b, &c, inv_vol);
    let bstar = cross_prod_inv_v(&c, &a, inv_vol);
    let cstar = cross_prod_inv_v(&a, &b, inv_vol);

    // B takes components in the reciprocal lattice to X
    [astar, bstar, cstar]
}

#[inline(always)]
fn symmetric_cubic_quaternions() -> [[f64; 4]; 24] {
    let angle_axis_symm = [
            [0.0,       1.0,    0.0,    0.0],  // identity
            [FRAC_PI_2,     1.0,    0.0,    0.0],  // fourfold about   1  0  0 (x1)
            [PI,        1.0,    0.0,    0.0],  //
            [FRAC_PI_2 * 3.0,   1.0,    0.0,    0.0],  //
            [FRAC_PI_2,     0.0,    1.0,    0.0],  // fourfold about   0  1  0 (x2)
            [PI,        0.0,    1.0,    0.0],  //
            [FRAC_PI_2 * 3.0,   0.0,    1.0,    0.0],  //
            [FRAC_PI_2,     0.0,    0.0,    1.0],  // fourfold about   0  0  1 (x3)
            [PI,        0.0,    0.0,    1.0],  //
            [FRAC_PI_2 * 3.0,   0.0,    0.0,    1.0],  //
            [FRAC_PI_3 * 2.0,   1.0,    1.0,    1.0],  // threefold about  1  1  1
            [FRAC_PI_3 * 4.0,   1.0,    1.0,    1.0],  //
            [FRAC_PI_3 * 2.0,  -1.0,    1.0,    1.0],  // threefold about -1  1  1
            [FRAC_PI_3 * 4.0,  -1.0,    1.0,    1.0],  //
            [FRAC_PI_3 * 2.0,  -1.0,   -1.0,    1.0],  // threefold about -1 -1  1
            [FRAC_PI_3 * 4.0,  -1.0,   -1.0,    1.0],  //
            [FRAC_PI_3 * 2.0,   1.0,   -1.0,    1.0],  // threefold about  1 -1  1
            [FRAC_PI_3 * 4.0,   1.0,   -1.0,    1.0],  //
            [PI,        1.0,    1.0,    0.0],  // twofold about    1  1  0
            [PI,       -1.0,    1.0,    0.0],  // twofold about   -1  1  0
            [PI,        1.0,    0.0,    1.0],  // twofold about    1  0  1
            [PI,        0.0,    1.0,    1.0],  // twofold about    0  1  1
            [PI,       -1.0,    0.0,    1.0],  // twofold about   -1  0  1
            [PI,        0.0,   -1.0,    1.0],  // twofold about    0 -1  1
    ];

    let inv2 = 1.0 / 2.0;
    let mut quat_symm = [[0.0; 4]; 24];
    quat_symm.iter_mut()
    .zip(angle_axis_symm)
    .for_each(|(quat, ang_axis)| {
        let s = f64::sin(inv2 * ang_axis[0]); 
        quat[0] = f64::cos(inv2 * ang_axis[0]);
        let mut inv_norm_axis = 1.0 / norm::<3, f64>(&ang_axis[1..4]);
        quat[1] = s * ang_axis[1] * inv_norm_axis;
        quat[2] = s * ang_axis[2] * inv_norm_axis;
        quat[3] = s * ang_axis[3] * inv_norm_axis;

        inv_norm_axis = 1.0;
        if quat[0] < 0.0 {
            inv_norm_axis *= -1.0;
        }

        quat[0] *= inv_norm_axis;
        quat[1] *= inv_norm_axis;
        quat[2] *= inv_norm_axis;
        quat[3] *= inv_norm_axis;
    });

    quat_symm
}

/// Returns all that aren't 
// #[inline(always)]
// fn find_unique_tolerance<const SYM_LEN: usize>(rmat_fr_qsym_c_dir: &[[f64; 3]], tolerance: f64) -> Vec<[f64; 3]> {
//     rmat_fr_qsym_c_dir.to_vec()
// }

#[inline(always)]
fn within_fiber<const SYM_LEN: usize>(c_dir: &[f64],
                                      s_dir: &[f64],
                                      quats: &[f64],
                                      symm_quat: &[[f64; 4]],
                                      distance_tolerance: f64,
                                      in_fibers: &mut [bool]) {

    assert!(c_dir.len() >= 3);
    assert!(s_dir.len() >= 3);
    assert!(quats.len() >= 4);
    assert!(symm_quat.len() >= SYM_LEN);

    assert!(in_fibers.len() == (quats.len() / 4));

    let c = {
        let inv_c_norm = 1.0 / norm::<3, f64>(c_dir);
        [c_dir[0] * inv_c_norm, c_dir[1] * inv_c_norm, c_dir[2] * inv_c_norm]
    };

    let s = {
        let inv_s_norm = 1.0 / norm::<3, f64>(s_dir);
        [s_dir[0] * inv_s_norm, s_dir[1] * inv_s_norm, s_dir[2] * inv_s_norm]
    };

    // Could maybe move this over to a vec if we want this to be easily generic over a ton of symmetry conditions...
    let mut rmat_fr_qsym_c_dir = [[0.0; 3]; SYM_LEN];
    rmat_fr_qsym_c_dir.iter_mut()
    .zip(symm_quat)
    .for_each(|(prod, quat)| {
        let rmat = quat2rmat(quat);
        // Might need to make this the transpose...
        matTVecMult::<3, 3, f64>(&rmat, &c, prod);
    });

    // If we really wanted to we could lower try and calculate the elements
    // that aren't unique here but that's not worth the effort at all given
    // how fast things are
    // let c_syms: Vec<[f64; 3]> = find_unique_tolerance::<SYM_LEN>(&rmat_fr_qsym_c_dir, f64::sqrt(f64::EPSILON));

    let s_rmat_csym_prod: Vec<f64> = quats.chunks_exact(4).map(|quat| {
        let sine = rmat_fr_qsym_c_dir.iter().map(|c_sym| {
            let rmat = quat2rmat(quat);
            let mut prod = [0.0; 3];
            // Might need to make this the transpose...
            matVecMult::<3, 3, f64>(&rmat, c_sym, &mut prod);
            dot_prod::<3, f64>(&s, &prod)
        })
        .fold(std::f64::MIN, |a, b| a.max(b));
        sine
    }).collect();

    in_fibers.iter_mut()
    .zip(s_rmat_csym_prod)
    .for_each(|(in_fiber, sine)| {
        let sine_safe = 
        {
            if f64::fabs(sine) > 1.00000001 {
                sine.signum()
            }
            else {
                sine
            }
        };
        
        let distance = f64::acos(sine_safe);
        *in_fiber = distance <= distance_tolerance;
    })
}
*/