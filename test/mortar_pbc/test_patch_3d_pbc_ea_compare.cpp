// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.3 / Batch S — dedicated A/B comparison driver for the
// element-assembly constraint path.
//
// This test runs all three patch-test patterns (homogeneous, strip,
// checkerboard) twice each — once via the HypreParMatrix path, once
// via the EA path — and asserts that the resulting displacement
// fluctuation `du` agrees between paths to a tight tolerance. The
// agreement is measured as `||du_ea - du_hp||_inf` with a global
// MPI_MAX reduction, and the test fails if any of the three patterns
// produces a divergence above `ab_compare_tol`.
//
// Why this test is the cross-rank firewall:
//
// The unit-test-level A/B harness in `test_mortar_constraint_operator`
// (Batch Q) validates the EA `Mult` and `MultTranspose` against the
// HypreParMatrix path at np=1. At np=1 every gtdof is FES-owned
// locally, so the EA path's off-rank import / export Alltoallv calls
// are degenerate — they execute but exchange zero data. That batch
// catches algorithmic bugs in the per-pair scatter loop but cannot
// catch cross-rank communication bugs.
//
// This test, when run at np>1 (e.g. np=4, np=7), exercises the
// Alltoallv import (during Mult) and Alltoallv export with element-
// wise add (during MultTranspose) on real off-rank data. A bug in the
// topology construction (e.g. a wrong destination rank in the
// gtdof-to-slot lookup) shows up here as a `||du_ea - du_hp||_inf`
// spike well above tolerance, often by orders of magnitude.
//
// Tolerance:
//   The two paths' Krylov solves diverge in FP-summation order
//   (each path's matvec sums in a different order, leading to slightly
//   different per-iteration residuals which compound). Empirical
//   observation on the 4³ test problem at np=1 is ~1e-9. We use
//   `ab_compare_tol = 1e-7` as the default, leaving 2 orders of
//   magnitude of headroom for cross-rank summation order variance.
//
// CLI options:
//   -n <int>          cells per direction (default 4)
//   --tol <double>    ab_compare_tol override (default 1e-7)
//   --pattern <name>  run only one pattern: 'homogeneous', 'strip',
//                     'checkerboard'. Default: run all three.
//   --F <name>        F_macro choice for non-homogeneous patterns.
//                     Default: 'uniaxial'. (Homogeneous always uses
//                     'mild' since du = 0 analytically — F choice
//                     doesn't meaningfully exercise the constraint.)
//   --f-sweep         For each non-homogeneous pattern, run with all
//                     five F choices: mild, uniaxial, biaxial,
//                     shear, mild-shear. Each F produces a
//                     qualitatively different stress field, so
//                     sweeping them stresses the constraint
//                     machinery across deformation modes.
//                     Implies --pattern is ignored for the sweep
//                     side; if --pattern is set, only that pattern
//                     gets the F sweep applied.

#include "patch_test_driver_3d.hpp"

#include "mfem.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using mortar_pbc::ConstraintStorage;
using mortar_pbc::PatchTestConfig;
using mortar_pbc::PatchTestPattern;
using mortar_pbc::RunPatchTest3D;

namespace {

const char* PatternName(PatchTestPattern p)
{
    switch (p)
    {
        case PatchTestPattern::Homogeneous:  return "homogeneous";
        case PatchTestPattern::Strip:        return "strip";
        case PatchTestPattern::Checkerboard: return "checkerboard";
    }
    return "unknown";
}

int RunOnePattern(PatchTestPattern pat,
                  const std::string& F_choice,
                  int n_per_side,
                  double tol,
                  bool& any_failed)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << std::endl
                  << "================================================="
                  << std::endl
                  << "  EA A/B compare: pattern = " << PatternName(pat)
                  << ", F = " << F_choice
                  << ", n = " << n_per_side
                  << ", tol = " << tol
                  << std::endl
                  << "================================================="
                  << std::endl;
    }

    PatchTestConfig cfg;
    cfg.pattern  = pat;
    cfg.n        = n_per_side;
    cfg.F_choice = F_choice;
    cfg.ab_compare    = true;
    cfg.ab_compare_tol = tol;
    // Primary path is EA — that is what production will use, so
    // we want the patch-test PASS criteria to be evaluated against
    // the EA-path du / dlam. The A/B comparison runs in addition.
    cfg.constraint_storage = ConstraintStorage::ElementAssembly;

    const int rc = RunPatchTest3D(cfg);
    if (rc != 0)
    {
        any_failed = true;
        if (rank == 0)
        {
            std::cerr << "[FAIL] EA A/B for pattern '" << PatternName(pat)
                      << "', F='" << F_choice
                      << "' returned rc=" << rc << std::endl;
        }
    }
    return rc;
}

}  // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int n_per_side = 4;
    double tol     = 1.0e-7;
    std::string single_pattern;     // empty = run all three
    std::string F_override;         // empty = use default per pattern
    bool f_sweep = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if      (a == "-n"        && i + 1 < argc) { n_per_side = std::atoi(argv[++i]); }
        else if (a == "--tol"     && i + 1 < argc) { tol        = std::atof(argv[++i]); }
        else if (a == "--pattern" && i + 1 < argc) { single_pattern = argv[++i]; }
        else if (a == "--F"       && i + 1 < argc) { F_override = argv[++i]; }
        else if (a == "--f-sweep")                 { f_sweep = true; }
    }

    bool any_failed = false;

    // F choices to use for non-homogeneous patterns.
    // - Default (no flags): single "uniaxial" (matches pre-existing
    //   coverage; the heterogeneous patch tests historically used
    //   uniaxial as their default).
    // - --F <name>: user-specified single F.
    // - --f-sweep: all five choices.
    //
    // Homogeneous pattern: always uses "mild" (du = 0 analytically
    // for any F, so F choice does not exercise the constraint
    // operator's implementation differences). Listed for
    // completeness but does not vary across the F sweep.
    std::vector<std::string> hetero_F_list;
    if (f_sweep)
    {
        hetero_F_list = {"mild", "uniaxial", "biaxial", "shear", "mild-shear"};
    }
    else if (!F_override.empty())
    {
        hetero_F_list = {F_override};
    }
    else
    {
        hetero_F_list = {"uniaxial"};
    }

    auto pattern_matches = [&](PatchTestPattern p)
    {
        return single_pattern.empty()
               || single_pattern == PatternName(p);
    };

    // Homogeneous: one run with "mild".
    if (pattern_matches(PatchTestPattern::Homogeneous))
    {
        const std::string F_for_homog =
            (!F_override.empty()) ? F_override : "mild";
        RunOnePattern(PatchTestPattern::Homogeneous,
                      F_for_homog, n_per_side, tol, any_failed);
    }

    // Heterogeneous patterns: sweep over hetero_F_list.
    for (PatchTestPattern p : {PatchTestPattern::Strip,
                                PatchTestPattern::Checkerboard})
    {
        if (!pattern_matches(p)) { continue; }
        for (const std::string& F : hetero_F_list)
        {
            RunOnePattern(p, F, n_per_side, tol, any_failed);
        }
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << std::endl
                  << "================================================="
                  << std::endl;
        if (any_failed)
        {
            std::cout << "  EA A/B compare: ONE OR MORE COMBINATIONS FAILED"
                      << std::endl;
        }
        else
        {
            std::cout << "  EA A/B compare: all combinations passed."
                      << std::endl;
        }
        std::cout << "================================================="
                  << std::endl;
    }

    MPI_Finalize();
    return any_failed ? 1 : 0;
}
