// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A capstone — homogeneous mortar-PBC patch test driver.
//
// Thin wrapper over `RunPatchTest3D` with `Pattern::Homogeneous`.
// All algorithm and PASS-criterion logic lives in
// `patch_test_driver_3d.{hpp,cpp}` so the homogeneous, strip, and
// checkerboard variants share the same code path.
//
// Mirrors `examples/patch_test_3d_pbc.py`. PASS criteria:
//   * Krylov converged
//   * ||du||_inf < 1e-7 (homogeneous-elastic exactness)
//   * ||<F> - F_macro||_inf < 1e-9
//   * ||C · u_total - C · u_lin||_inf < 1e-9
//
// CLI options:
//   -n <int>          cells per direction (default 4)
//   -L <double>       cube side length (default 1.0)
//   -F <name>         F choice (default "mild")
//   -E <double>       Young's modulus (default 70e3)
//   -nu <double>      Poisson's ratio (default 0.3)
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
    cfg.pattern = PatchTestPattern::Homogeneous;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if      (a == "-n"  && i + 1 < argc) { cfg.n  = std::atoi(argv[++i]); }
        else if (a == "-L"  && i + 1 < argc) { cfg.L  = std::atof(argv[++i]); }
        else if (a == "-F"  && i + 1 < argc) { cfg.F_choice = argv[++i]; }
        else if (a == "-E"  && i + 1 < argc) { cfg.E1 = std::atof(argv[++i]); }
        else if (a == "-nu" && i + 1 < argc) { cfg.nu = std::atof(argv[++i]); }
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
