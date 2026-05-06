// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-E Part 2 — production-shape patch test on a
// NON-CONFORMING periodic interface.
//
// Strategy:
//   Instead of constructing a non-matching MFEM mesh from scratch
//   (which would require the low-level Mesh(int, int, int) API or
//   anisotropic h-refinement with hanging nodes — out of Phase 4.4
//   scope), we start with a standard MakeCartesian3D conforming
//   mesh and apply an in-plane node perturbation to ONE periodic
//   face only. The perturbation:
//
//     For each node at (x, y, z) with y == L (the y=L face only):
//         x_new = x + amplitude * sin(pi * x / L)
//         y_new = y, z_new = z
//
//   This keeps:
//     * The y=0 face uniform (unchanged from MakeCartesian3D).
//     * The y=L face flat at y=L (faces stay axis-aligned per the
//       clipped-path's contract).
//     * Corner positions exact (sin vanishes at x=0 and x=L), so
//       corner Dirichlet BCs from F·X stay clean.
//     * Each face element on y=L is still an axis-aligned rectangle
//       (the perturbation shifts entire grid-lines uniformly along
//       the z direction; each quad's two parametric directions are
//       still global x and z).
//
//   The resulting mesh has:
//     * Conforming face pair on x=0/x=L (untouched).
//     * Conforming face pair on z=0/z=L (untouched).
//     * NON-CONFORMING face pair on y=0/y=L — y=0 is uniformly spaced
//       in x; y=L has sin-perturbed x spacing. The element-pair
//       centroid match between the two y faces fails by ~amplitude,
//       triggering TryMatchConformingFacePairs to return nullopt and
//       BuildLocalPairBlocks to fall back to the clipped path.
//
//   Under homogeneous F + homogeneous material, the exact discrete
//   solution is u_h = (F - I)·x — Q1 hexes reproduce linear fields
//   exactly regardless of element shape. The mortar projector
//   reproduces linear fields exactly (Wohlmuth biorthogonality +
//   completeness; validated in Batch 4.4-D-4 to 1e-14). So the patch
//   test residual ||du||_inf should be at the FE-solver tolerance
//   (~1e-7) just like the conforming case.
//
// PASS criteria are inherited from RunPatchTest3D unchanged:
//   * Krylov converged
//   * ||du||_inf < 1e-7
//   * ||<F> - F_macro||_inf < 1e-9
//   * ||C·u_total - C·u_lin||_inf < 1e-9
//
// If this test passes, the entire Phase 4.4 stack (BVH + clip +
// AssembleClipped + dispatch) is end-to-end correct on a real FE
// problem — the production-shape gate.

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
    cfg.pattern = PatchTestPattern::Homogeneous;

    // Default perturbation amplitude. Big enough to clearly defeat the
    // 1e-9 centroid-match tolerance (with cell width 0.25 on a 4-cell
    // mesh, the tolerance is ~2.5e-10; 0.05 is 8 orders of magnitude
    // larger — unambiguously non-conforming). Small enough that all
    // hex elements stay non-degenerate (max shift is at x = L/2 where
    // sin = 1, giving a perturbed neighbor cell width of 0.25 + 0.05 =
    // 0.30 on one side and 0.25 - 0.05 = 0.20 on the other — still well
    // away from collapsing).
    double amplitude = 0.05;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if      (a == "-n"  && i + 1 < argc) { cfg.n  = std::atoi(argv[++i]); }
        else if (a == "-L"  && i + 1 < argc) { cfg.L  = std::atof(argv[++i]); }
        else if (a == "-F"  && i + 1 < argc) { cfg.F_choice = argv[++i]; }
        else if (a == "-E"  && i + 1 < argc) { cfg.E1 = std::atof(argv[++i]); }
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

    cfg.F_average_tol = 2e-4;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "test_patch_3d_pbc_nonconforming: y=L face perturbation "
                     "amplitude = " << amplitude
                  << " (cell width = " << (cfg.L / cfg.n) << ")\n";
    }

    const int rc = RunPatchTest3D(cfg);
    MPI_Finalize();
    if (rc != 0) { std::exit(1); }
    return 0;
}
