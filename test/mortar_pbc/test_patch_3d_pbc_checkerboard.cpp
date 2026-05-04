// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — checkerboard mortar-PBC patch test.
//
// Direct C++ analog of `examples/patch_test_3d_checkerboard.py`.
// Element attribute is determined by 2x2x2 octant XOR:
// `attr = 1` if even number of `centroid_d > L/2`, else `attr = 2`.
// Adjacent octants always carry opposite attributes, so EVERY
// matched pair of periodic boundary elements crosses a material
// interface — maximum stress on the constraint machinery for a
// given mesh size and material contrast.
//
// Like the strip-split variant, the fluctuation `u_tilde` is
// non-trivial; the PASS criterion is a lower bound on ||du||_∞.
//
// PASS criteria:
//   * Krylov converged
//   * ||du||_inf > 1e-12  (checkerboard response; lower bound)
//   * ||<F> - F_macro||_inf < 1e-9
//   * ||C · u_total - C · u_lin||_inf < 1e-9
//
// CLI options:
//   -n <int>          cells per direction (default 4)
//   -L <double>       cube side length (default 1.0)
//   -F <name>         F choice (default "uniaxial")
//   -E1 <double>      material 1 Young's modulus (default 70e3)
//   -E2 <double>      material 2 Young's modulus (default 350e3)
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
    cfg.pattern = PatchTestPattern::Checkerboard;
    cfg.F_choice = "uniaxial";

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
