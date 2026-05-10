// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — homogeneous patch test (single-material baseline).
//
// Validates the complete mortar-PBC pipeline on a cube with a single
// material. The fluctuation `du` should be ~0 for any F since the
// homogeneous-elastic affine field is the equilibrium solution
// exactly.
//
// CLI flags
// ---------
//   -n N              Cells per direction (default 4).
//   -L L              Cube side length (default 1.0).
//   -F NAME           Macroscopic F choice; one of "mild",
//                     "uniaxial", "biaxial", "shear", "mild-shear".
//                     Default "mild".
//   -E E              Young's modulus (default 70e3 — typical of
//                     Al alloys).
//   -nu NU            Poisson's ratio (default 0.3).
//   --paraview DIR    Write ParaView output to DIR (default OFF).
//
// Phase 5.5.B.2.A — `--constraint-storage` and `--ab-compare` flags
// removed. The HypreParMatrix-C path was retired and the EA path is
// now the only option.

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
    }

    const int rc = RunPatchTest3D(cfg);
    MPI_Finalize();
    if (rc != 0) { std::exit(1); }
    return 0;
}