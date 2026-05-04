// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — heterogeneous strip-split mortar-PBC patch test.
//
// Direct C++ analog of `examples/patch_test_3d_heterogeneous.py`.
// Element attribute is 1 for `x_centroid < L/2` (left half, soft
// material) and 2 for `x_centroid >= L/2` (right half, stiff
// material). The material discontinuity is parallel to the y-z
// nonmortar/mortar face pair, so the constraint machinery is
// exercised both within material (y, z pairings) AND across
// material (x pairing) on the same run.
//
// Unlike the homogeneous case (where du = 0 by construction), the
// fluctuation `u_tilde = u_total - u_lin` is genuinely non-trivial
// here. The PASS criteria therefore require ||du||_∞ > 1e-12 (a
// LOWER bound — fluctuation must be present) instead of an upper
// bound.
//
// PASS criteria:
//   * Krylov converged
//   * ||du||_inf > 1e-12  (heterogeneous response; lower bound)
//   * ||<F> - F_macro||_inf < 1e-9  (Hill-Mandel volume average)
//   * ||C · u_total - C · u_lin||_inf < 1e-9  (periodicity exact)
//
// CLI options:
//   -n <int>          cells per direction (default 4)
//   -L <double>       cube side length (default 1.0)
//   -F <name>         F choice (default "uniaxial" for clearer fluctuation)
//   -E1 <double>      material 1 (left) Young's modulus (default 70e3)
//   -E2 <double>      material 2 (right) Young's modulus (default 350e3)
//   -nu <double>      Poisson's ratio (default 0.3, both materials)
//   --paraview <dir>  write visualization to <dir>
//   --constraint-storage <hypre|ea>  Phase 4.3 / Batch S — choose
//                     between the original HypreParMatrix path and
//                     the new element-assembly path. Default: hypre.
//   --ab-compare      Phase 4.3 / Batch S — run BOTH paths and assert
//                     ||du_ea - du_hp||_inf < ab_compare_tol.

#include "patch_test_driver_3d.hpp"

#include "mfem.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

using mortar_pbc::ConstraintStorage;
using mortar_pbc::PatchTestConfig;
using mortar_pbc::PatchTestPattern;
using mortar_pbc::RunPatchTest3D;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    PatchTestConfig cfg;
    cfg.pattern = PatchTestPattern::Strip;
    cfg.F_choice = "uniaxial";  // clearer fluctuation than "mild"

    for (int i = 1; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if      (a == "-n"   && i + 1 < argc) { cfg.n  = std::atoi(argv[++i]); }
        else if (a == "-L"   && i + 1 < argc) { cfg.L  = std::atof(argv[++i]); }
        else if (a == "-F"   && i + 1 < argc) { cfg.F_choice = argv[++i]; }
        else if (a == "-E1"  && i + 1 < argc) { cfg.E1 = std::atof(argv[++i]); }
        else if (a == "-E2"  && i + 1 < argc) { cfg.E2 = std::atof(argv[++i]); }
        else if (a == "-nu"  && i + 1 < argc) { cfg.nu = std::atof(argv[++i]); }
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

    const int rc = RunPatchTest3D(cfg);
    MPI_Finalize();
    if (rc != 0) { std::exit(1); }
    return 0;
}
