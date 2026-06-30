// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — shared driver for the 3D mortar-PBC patch tests.
//
// Three patch test variants share 95% of their orchestration code:
//
//   * Homogeneous            (`patch_test_3d_pbc.py` — single material)
//   * Heterogeneous strip    (`patch_test_3d_heterogeneous.py` — left/right
//                             halves, x = L/2 vertical interface)
//   * Heterogeneous checker  (`patch_test_3d_checkerboard.py` — 2x2x2
//                             octant XOR, alternating attrs)
//
// They differ only in:
//   1. How element attributes are assigned to the mesh.
//   2. Which Lamé parameters are used (one set vs two distinct sets).
//   3. The PASS criteria for ||du||_∞:
//        - homogeneous: fluctuation should be ~0 (du = 0 exact)
//        - heterogeneous: fluctuation must be NON-zero (genuine periodic
//          response of the heterogeneous RVE)
//
// The Method-D RHS construction has a critical subtlety for the
// heterogeneous case: r1 must be K_full * u_lin (un-eliminated K),
// NOT K_eliminated * u_lin. See the cpp file for details.
//
// Phase 5.5.B.2.A — `ConstraintStorage` enum, `constraint_storage`
// field, `ab_compare` / `ab_compare_tol` fields all removed. The
// HypreParMatrix-C path was retired (see Phase 5.5.B.2.A README);
// only the EA path (MortarConstraintOperator) remains, so there is
// no second path to A/B-compare against.
//
// References
// ----------
//   * `mortar_pbc/multistep_driver.py::_solve_independently` — the
//     RHS-construction method whose docstring explains the K_full
//     vs K_eliminated subtlety.
//   * `examples/patch_test_3d_heterogeneous.py` — the strip-split
//     Python driver.
//   * `examples/patch_test_3d_checkerboard.py` — the octant-XOR
//     Python driver.

#pragma once

#include "mfem.hpp"

#include <functional>
#include <string>

namespace mortar_pbc {

/**
 * @brief Element-attribute assignment pattern for the patch test mesh.
 */
enum class PatchTestPattern
{
    /// All elements get attribute 1; PWConstCoefficient with a single
    /// Lamé pair. Mathematically equivalent to
    /// `AssembleLinearElasticKHypre`, but goes through the same
    /// PWConstCoefficient codepath as the heterogeneous variants for
    /// consistency. The fluctuation `du` should be ~0 for any F.
    Homogeneous,
    /// Strip split: attribute 1 if `x_centroid < L/2`, else attribute 2.
    /// The material discontinuity is the y-z plane at x = L/2; this
    /// puts the interface PARALLEL to one of the periodic face pairs,
    /// stressing within-material periodicity (y, z) AND across-material
    /// periodicity (x) simultaneously.
    Strip,
    /// 2x2x2 octant XOR: `attr = 1` if even number of `centroid_d > L/2`,
    /// else `attr = 2`. Adjacent octants always carry opposite
    /// attributes. Maximum stress on the constraint machinery: every
    /// matched pair of periodic boundary elements crosses a material
    /// interface.
    Checkerboard,
};

/**
 * @brief Configuration for a single patch test run.
 */
struct PatchTestConfig
{
    PatchTestPattern pattern = PatchTestPattern::Homogeneous;

    /// Cells per direction. Default 4 (small enough to be fast,
    /// large enough that face-mortar DOFs are non-trivial).
    int n = 4;
    /// Cube side length.
    double L = 1.0;
    /// Macroscopic deformation gradient name. One of:
    /// "mild", "uniaxial", "shear", "biaxial", "mild-shear".
    std::string F_choice = "mild";

    /// Material 1 Young's modulus. For Homogeneous, E2 is ignored
    /// (or set equal to E1).
    double E1 = 70.0e3;
    /// Material 2 Young's modulus. Only used for Strip / Checkerboard.
    /// 5x contrast by default for strip / checker; matches the Python.
    double E2 = 350.0e3;
    /// Poisson's ratio (uniform across materials in this prototype).
    double nu = 0.3;

    /// If true, write a ParaView `.pvd` collection to `paraview_dir`.
    bool paraview = false;
    /// Output directory for ParaView output. Created if missing.
    std::string paraview_dir = "./paraview_3d_patch";
    /// Optional collection name override; default derived from pattern + F.
    std::string paraview_name;

    /// Override the PASS bound on `||du||_∞` for the homogeneous test.
    /// Default 1e-7. Heterogeneous tests use a different criterion
    /// (`du_min`, see below) — this is only used for `Pattern::Homogeneous`.
    double du_max_homogeneous = 1.0e-7;
    /// Lower bound on `||du||_∞` for heterogeneous tests — fluctuation
    /// must be present, otherwise the test is meaningless. Default 1e-12.
    double du_min_heterogeneous = 1.0e-12;
    /// Tolerance on the constraint residual `||C·u_total - C·u_lin||_∞`.
    double constraint_residual_tol = 1.0e-9;
    /// Tolerance on the volume-averaged-F homogenization check.
    double F_average_tol = 1.0e-9;

    /// Phase 4.4 / Batch 4.4-E Part 2 — optional in-place mesh
    /// perturbation, applied to the **serial** mesh after
    /// `MakeCartesian3D` and `ApplyAttributePattern`, before
    /// `ParMesh` construction. Used by the non-conforming patch
    /// test driver to introduce an in-plane node shift on one
    /// periodic face so the centroid-based conforming match fails
    /// and the clipped fallback fires.
    ///
    /// Contract:
    ///   * Must preserve corner positions (so corner Dirichlet BCs
    ///     stay aligned with `u_lin = (F - I) X`).
    ///   * Must keep the faces on each periodic axis FLAT (constant
    ///     perpendicular coordinate per face) so axis-aligned face-
    ///     element assumption in the clipped path still holds.
    ///   * Must not produce degenerate or self-intersecting hex
    ///     elements.
    ///
    /// Default `nullptr` means "no perturbation" — conforming mesh
    /// as before.
    std::function<void(mfem::Mesh&)> mesh_perturbation = nullptr;
};

/**
 * @brief Run a 3D mortar-PBC patch test end to end.
 *
 * @param cfg   Configuration controlling pattern, mesh size, F choice,
 *              materials, and PASS thresholds.
 *
 * @return 0 on PASS, 1 on FAIL. The function does NOT call
 *         `MPI_Init` / `MPI_Finalize` — caller (the thin `main()`
 *         in each test driver) is responsible for that.
 *
 * @details Mirrors the 11-step pipeline of
 * `examples/patch_test_3d_pbc.py` (and its heterogeneous /
 * checkerboard cousins): mesh → attributes → classifier → C →
 * K (K_full + K_eliminated for heterogeneous) → u_lin → Method-D
 * RHS → saddle-point solve → recovery → ⟨F⟩ check → PASS/FAIL
 * summary on rank 0.
 *
 * On `cfg.paraview = true`, writes a two-cycle `.pvd` collection
 * suitable for cross-validation against the Python reference.
 *
 * @par MPI scope
 * Collective on `MPI_COMM_WORLD`. Does not enter / finalize MPI.
 */
int RunPatchTest3D(const PatchTestConfig& cfg);

}  // namespace mortar_pbc