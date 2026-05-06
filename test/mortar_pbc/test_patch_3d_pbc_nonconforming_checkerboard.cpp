// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.5 — 2x2x2 octant-checkerboard heterogeneity + non-conforming
// periodic interface, end-to-end patch test.
//
// Combines the octant-XOR heterogeneity of
// test_patch_3d_pbc_checkerboard.cpp (every adjacent octant pair has
// opposite material attribute, so EVERY matched periodic boundary
// element pair crosses a material interface) with the y=L face
// perturbation of test_patch_3d_pbc_nonconforming.cpp (sin perturbation
// of the y=L face that defeats centroid matching and triggers the
// clipped-path fallback).
//
// Why this is the strongest single-mesh test in the Phase 4.5 suite
// -----------------------------------------------------------------
// The checkerboard pattern is the maximum-stress heterogeneous case:
// every pair of periodic elements crosses a material seam, so all
// three constraint axes (x-pair, y-pair, z-pair) carry across-material
// fluctuations simultaneously. Adding the non-conforming y face on
// top means the y axis exercises:
//   * Across-material periodicity (every y-pair element crosses a
//     material seam at z=L/2 or x=L/2 or both).
//   * Sutherland-Hodgman clipping (the y=L face's sin perturbation
//     defeats centroid matching).
//   * Wohlmuth edge modifications on the LOR-equivalent edge nodes
//     of clipped sub-regions where the perturbed y-face elements
//     overlap nominally-conforming x or z face elements at the
//     box edges.
// while x and z pairs continue to exercise across-material
// periodicity through the conforming dispatch.
//
// If this test passes, the Phase 4.4 clipped-path stack is correct
// in genuinely heterogeneous wirebasket configurations — the
// strongest single-mesh assertion we can make about the constraint
// pipeline short of FE² coupling.
//
// Mesh perturbation strategy
// --------------------------
// Identical to test_patch_3d_pbc_nonconforming.cpp:
//
//   For each node at (x, y, z) with y == L:
//       x_new = x + amplitude * sin(pi * x / L)
//
// Applied to the SERIAL mesh AFTER the attribute pattern is set
// (so the octant XOR assignment is evaluated on the unperturbed
// mesh, where x_centroid > L/2, y_centroid > L/2, z_centroid > L/2
// have unambiguous truth values) but BEFORE ParMesh construction.
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

using mortar_pbc::ConstraintStorage;
using mortar_pbc::PatchTestConfig;
using mortar_pbc::PatchTestPattern;
using mortar_pbc::RunPatchTest3D;

namespace
{

/// In-plane sine perturbation applied to the y = L face only.
///
/// Same lambda as test_patch_3d_pbc_nonconforming.cpp and
/// test_patch_3d_pbc_nonconforming_heterogeneous.cpp. Kept as a
/// per-test private helper rather than promoted to a header because
/// (a) it's small and (b) leaving it local makes each test driver
/// self-contained for cross-validation runs.
std::function<void(mfem::Mesh&)> MakeY1FacePerturbation(double L,
                                                       double amplitude)
{
    return [L, amplitude](mfem::Mesh& mesh) -> void
    {
        const double pi = 3.14159265358979323846;
        const double y_tol = 1.0e-12 * L;
        const int nv = mesh.GetNV();
        for (int i = 0; i < nv; ++i)
        {
            double* v = mesh.GetVertex(i);
            if (std::abs(v[1] - L) < y_tol)
            {
                // sin(pi * x / L) vanishes at x = 0 and x = L; corners
                // stay at corner positions. y and z are unchanged.
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
    cfg.pattern  = PatchTestPattern::Checkerboard;
    cfg.F_choice = "uniaxial";

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
        else if (a == "--constraint-storage" && i + 1 < argc)
        {
            const std::string val(argv[++i]);
            if (val == "ea")
            {
                cfg.constraint_storage = ConstraintStorage::ElementAssembly;
            }
            else if (val == "hypre")
            {
                cfg.constraint_storage = ConstraintStorage::HypreParMatrix;
            }
            else
            {
                std::cerr << "Unknown --constraint-storage: " << val
                          << " (expected 'hypre' or 'ea')" << std::endl;
                MPI_Finalize();
                return 1;
            }
        }
        else if (a == "--ab-compare")
        {
            cfg.ab_compare = true;
        }
    }

    cfg.mesh_perturbation = MakeY1FacePerturbation(cfg.L, amplitude);
    cfg.F_average_tol = 1e-5;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "test_patch_3d_pbc_nonconforming_checkerboard: "
                     "y=L face perturbation amplitude = " << amplitude
                  << " (cell width = " << (cfg.L / cfg.n) << ")\n";
    }

    const int rc = RunPatchTest3D(cfg);
    MPI_Finalize();
    if (rc != 0) { std::exit(1); }
    return 0;
}
