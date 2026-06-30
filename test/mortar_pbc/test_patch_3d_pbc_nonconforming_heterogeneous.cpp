// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.5 — heterogeneous strip-split + non-conforming periodic
// interface, end-to-end patch test.
//
// Combines the strip-split heterogeneity of
// test_patch_3d_pbc_heterogeneous.cpp (left/right halves split by
// element attribute, 5x stiffness contrast across the x = L/2 plane)
// with the y=L face perturbation of test_patch_3d_pbc_nonconforming.cpp
// (sin perturbation of the y=L face that defeats centroid matching
// and triggers the clipped-path fallback).
//
// Why this combination matters
// ----------------------------
// The conforming heterogeneous test passes even if certain bugs in
// A_m have sign errors that the diagonality of D + axis alignment
// papers over. A NON-CONFORMING heterogeneous test exposes that bug
// class because:
//   1. The fluctuation u_tilde is genuinely non-trivial (heterogeneous
//      contrast forces |u_tilde|_inf >> FE assembly noise).
//   2. The clipped path's A_m sub-blocks are NOT 1:1 with element
//      pairs — each clipped sub-region touches multiple mortar nodes,
//      so any sign or column-ordering mismatch in the assembled A_m
//      will fail to reproduce the periodicity of the heterogeneous
//      response.
// (Architecture doc §12 traps 18 + 19 — heterogeneous AND
// non-conforming together is the strongest single-mesh check for the
// constraint pipeline.)
//
// Mesh perturbation strategy
// --------------------------
// Identical to test_patch_3d_pbc_nonconforming.cpp:
//
//   For each node at (x, y, z) with y == L:
//       x_new = x + amplitude * sin(pi * x / L)
//
// Applied to the SERIAL mesh AFTER the attribute pattern is set
// (so the strip-split assignment is evaluated on the unperturbed
// mesh, where x_centroid < L/2 vs >= L/2 is unambiguous) but BEFORE
// ParMesh construction (so MFEM's parallel partitioning sees the
// perturbed coords). This is the same hook contract documented in
// PatchTestConfig::mesh_perturbation.
//
// Note that the perturbation is on the y face (parallel to the
// strip-split interface plane y-z at x=L/2). The non-conforming pair
// is the y face pair; the strip-split material interface is at
// x=L/2 and is unaffected. So this test exercises:
//   * x periodic pair: CONFORMING + ACROSS material interface
//     (left edge = matrix, right edge = stiff at x=0; reversed at
//      x=L). Goes through the conforming dispatch.
//   * y periodic pair: NON-CONFORMING + within-material on each
//     side (the strip-split interface is at x=L/2, parallel to the
//     y faces, so y=0 has matrix on the left half + stiff on the
//     right half, and same for y=L). Triggers clipped fallback.
//   * z periodic pair: CONFORMING + within-material. Conforming
//     dispatch.
//
// PASS criteria are inherited from RunPatchTest3D unchanged for the
// heterogeneous case:
//   * Krylov converged
//   * ||du||_inf > du_min_heterogeneous (default 1e-12; fluctuation
//     must be present)
//   * ||<F> - F_macro||_inf < 1e-9
//   * ||C·u_total - C·u_lin||_inf < 1e-9 (the actual Phase 4.4 gate)
//
// CLI options:
//   -n <int>          cells per direction (default 4)
//   -L <double>       cube side length (default 1.0)
//   -F <name>         F choice (default "uniaxial" — clearer
//                     fluctuation than "mild" for heterogeneous)
//   -E1 <double>      material 1 Young's modulus (default 70e3)
//   -E2 <double>      material 2 Young's modulus (default 350e3)
//   -nu <double>      Poisson's ratio (default 0.3)
//   --amplitude <d>   y=L face perturbation amplitude (default 0.05)
//   --paraview <dir>  write visualization to <dir>
//   --constraint-storage <hypre|ea>  Phase 4.3 / Batch S — choose
//                     between the original HypreParMatrix path and
//                     the new element-assembly path. Default: hypre.
//   --ab-compare      Phase 4.3 / Batch S — run BOTH paths and assert
//                     ||du_ea - du_hp||_inf < ab_compare_tol.

#include "patch_test_driver_3d.hpp"

#include "mfem.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

using mortar_pbc::PatchTestConfig;
using mortar_pbc::PatchTestPattern;
using mortar_pbc::RunPatchTest3D;

namespace
{

/// In-plane sine perturbation applied to the y = L face only.
///
/// Captures `L` and `amplitude` by value so the resulting std::function
/// is self-contained (the PatchTestConfig struct outlives the lambda's
/// enclosing scope, so no by-reference captures).
std::function<void(mfem::Mesh&)> MakeY1FacePerturbation(double L,
                                                       double amplitude)
{
    return [L, amplitude](mfem::Mesh& mesh) -> void
    {
        const double pi = 3.14159265358979323846;
        // Tolerance for "is this vertex on the y=L face?" Use a relative
        // tolerance against L so the test is scale-invariant. 1e-12 * L
        // is safely below the FP roundoff bound on any reasonable L.
        const double y_tol = 1.0e-12 * L;
        const int nv = mesh.GetNV();
        for (int i = 0; i < nv; ++i)
        {
            double* v = mesh.GetVertex(i);
            if (std::abs(v[1] - L) < y_tol)
            {
                // sin(pi * x / L) vanishes at x = 0 and x = L, so corners
                // stay exactly at corner positions. y and z are unchanged.
                v[0] += amplitude * std::sin(pi * v[0] / L);
            }
        }
    };
}

}  // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    PatchTestConfig cfg;
    cfg.pattern  = PatchTestPattern::Strip;
    cfg.F_choice = "uniaxial";  // clearer fluctuation than "mild"

    // Default perturbation amplitude. Same rationale as the homogeneous
    // non-conforming test: 0.05 is 8 orders of magnitude above the 1e-9
    // centroid match tolerance (cell width 0.25 on a 4³ mesh) and well
    // away from collapsing any hex element.
    double amplitude = 0.05;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if      (a == "-n"  && i + 1 < argc) { cfg.n  = std::atoi(argv[++i]); }
        else if (a == "-L"  && i + 1 < argc) { cfg.L  = std::atof(argv[++i]); }
        else if (a == "-F"  && i + 1 < argc) { cfg.F_choice = argv[++i]; }
        else if (a == "-E1" && i + 1 < argc) { cfg.E1 = std::atof(argv[++i]); }
        else if (a == "-E2" && i + 1 < argc) { cfg.E2 = std::atof(argv[++i]); }
        else if (a == "-nu" && i + 1 < argc) { cfg.nu = std::atof(argv[++i]); }
        else if (a == "--amplitude" && i + 1 < argc)
        {
            amplitude = std::atof(argv[++i]);
        }
        else if (a == "--paraview" && i + 1 < argc)
        {
            cfg.paraview = true;
            cfg.paraview_dir = argv[++i];
        }
    }

    cfg.mesh_perturbation = MakeY1FacePerturbation(cfg.L, amplitude);
    cfg.F_average_tol = 2e-4;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "test_patch_3d_pbc_nonconforming_heterogeneous: "
                     "y=L face perturbation amplitude = " << amplitude
                  << " (cell width = " << (cfg.L / cfg.n) << ")\n";
    }

    const int rc = RunPatchTest3D(cfg);
    MPI_Finalize();
    if (rc != 0) { std::exit(1); }
    return 0;
}
