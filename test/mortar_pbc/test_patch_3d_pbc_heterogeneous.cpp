// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — heterogeneous strip-split patch test.
//
// Two-material strip-split at x = L/2: attribute 1 on the left half,
// attribute 2 on the right half. The interface is parallel to one of
// the periodic face pairs, stressing within-material periodicity (y, z)
// AND across-material periodicity (x) simultaneously. Fluctuation
// `du` must be NON-zero — the heterogeneous elastic response of the
// RVE produces a real periodic perturbation around the affine field.
//
// CLI flags
// ---------
//   -n N              Cells per direction (default 4).
//   -L L              Cube side length (default 1.0).
//   -F NAME           Macroscopic F choice; one of "mild",
//                     "uniaxial", "biaxial", "shear", "mild-shear".
//                     Default "uniaxial" — produces a clearer
//                     fluctuation than "mild".
//   -E1 E             Material 1 Young's modulus (default 70e3).
//   -E2 E             Material 2 Young's modulus (default 350e3 —
//                     5x contrast).
//   -nu NU            Shared Poisson's ratio (default 0.3).
//   --paraview DIR    Write ParaView output to DIR (default OFF).
//
// Phase 5.5.B.2.A — `--constraint-storage` and `--ab-compare` flags
// removed. EA path is the only option.

#include "patch_test_driver_3d.hpp"

#include "mfem.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

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
    }

    const int rc = RunPatchTest3D(cfg);
    MPI_Finalize();
    if (rc != 0) { std::exit(1); }
    return 0;
}