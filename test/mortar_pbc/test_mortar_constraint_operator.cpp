// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.3 / Batches O, P, Q — A/B validation harness for
// MortarConstraintOperator vs the HypreParMatrix path.
//
// Coverage progression:
//   - Batch O: construction + dimension match.
//   - Batch P: single-size (4³) Mult / MultTranspose match.
//   - Batch Q (this batch): multiple mesh sizes (4³, 6³, 8³),
//                            tightened tolerance, a negative test
//                            that confirms the harness catches a
//                            deliberately-perturbed result.
//
// Scope decision:
// All tests here run at np=1, matching the rest of the unit-test
// suite. Cross-rank A/B validation (the Alltoallv import/export
// path actually exchanging data) is exercised by the end-to-end
// patch tests at np=4 / np=7 with the --constraint-storage=ea
// flag (Phase 4.3 / Batch S). This file's purpose is the matvec-
// level contract: at fixed np, EA and HypreParMatrix paths
// produce identical y to FP-rearrangement precision.
//
// Tolerance contract (per §P4.4.6.3): the difference must be
// below 1e-12 * (||C||_F * ||u||_2) — for the small meshes here
//
// Phase 4.3.B / Batch X — GPU port note:
// Although this file runs serially on host, after the GPU port
// the matvec hot path goes through mfem::forall with full
// Read/Write memory-manager annotations. To exercise the
// memory-manager invariants in CI, build MFEM with DEVICE_DEBUG
// enabled and re-run this test — any host-stale or device-stale
// access pattern will trigger an MFEM_ASSERT failure rather than
// silently corrupting. (DEVICE_DEBUG works on host-only builds
// too; it's a memory-manager validation mode, not a device
// requirement.)
// (||C||_F ~ O(1), ||u||_2 ~ O(1)) this is 1e-12 absolute. Tests
// use 1e-12 with a max(1, ||y_hp||_2) safety floor.
//
// Each test function exits via std::exit(1) on failure (with a
// diagnostic to stderr) or returns normally on success.

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "mortar_constraint_operator.hpp"
#include "surface_projector.hpp"
#include "diagonal_scaler.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ConstraintBuilder3D;
using mortar_pbc::MortarConstraintOperator;
using mortar_pbc::SurfaceProjector;
using mortar_pbc::DiagonalScaler;

namespace {

void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

struct FesBundle
{
    std::unique_ptr<mfem::ParMesh> pmesh;
    std::unique_ptr<mfem::H1_FECollection> fec;
    std::unique_ptr<mfem::ParFiniteElementSpace> fes;
};

struct SharedFesBundle
{
    std::shared_ptr<mfem::ParMesh> pmesh;
    std::shared_ptr<mfem::H1_FECollection> fec;
    std::shared_ptr<mfem::ParFiniteElementSpace> fes;
};

FesBundle BuildHexFesBundle(MPI_Comm comm, int n_per_side, int order = 1)
{
    FesBundle b;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n_per_side, n_per_side, n_per_side,
        mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
    b.fec = std::make_unique<mfem::H1_FECollection>(order, /*dim=*/3);
    b.fes = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3, mfem::Ordering::byNODES);
    return b;
}

SharedFesBundle BuildSharedHexFesBundle(MPI_Comm comm, int n_per_side)
{
    SharedFesBundle b;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n_per_side, n_per_side, n_per_side,
        mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.pmesh = std::make_shared<mfem::ParMesh>(comm, serial);
    b.fec = std::make_shared<mfem::H1_FECollection>(/*order=*/1, /*dim=*/3);
    b.fes = std::make_shared<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3, mfem::Ordering::byNODES);
    return b;
}

SharedFesBundle BuildSharedTetFesBundle(MPI_Comm comm, int n_per_side,
                                        int order)
{
    SharedFesBundle b;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n_per_side, n_per_side, n_per_side,
        mfem::Element::TETRAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.pmesh = std::make_shared<mfem::ParMesh>(comm, serial);
    if (order > 1)
    {
        b.pmesh->SetCurvature(order, /*discontinuous=*/false,
                              /*space_dim=*/3,
                              mfem::Ordering::byNODES);
    }
    b.fec = std::make_shared<mfem::H1_FECollection>(order, /*dim=*/3);
    b.fes = std::make_shared<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3,
        mfem::Ordering::byNODES);
    return b;
}

void FillParentAffineTrace(const mfem::ParFiniteElementSpace& fes,
                           const double L[3][3],
                           mfem::Vector& x_true)
{
    x_true.SetSize(fes.GetTrueVSize());
    x_true = 0.0;

    const HYPRE_BigInt first = fes.GetTrueDofOffsets()[0];
    const HYPRE_BigInt last = fes.GetTrueDofOffsets()[1];
    mfem::ParMesh* mesh = fes.GetParMesh();

    mfem::Array<int> scalar_dofs;
    mfem::Vector x_phys(3);
    for (int be = 0; be < mesh->GetNBE(); ++be)
    {
        fes.GetBdrElementDofs(be, scalar_dofs);
        const mfem::FiniteElement* fe = fes.GetBE(be);
        const mfem::IntegrationRule& nodes = fe->GetNodes();
        mfem::ElementTransformation* tr =
            mesh->GetBdrElementTransformation(be);
        AssertOrDie(nodes.GetNPoints() == scalar_dofs.Size(),
                    "P2 tet affine trace fill",
                    "boundary FE node count does not match scalar DOFs");

        for (int i = 0; i < scalar_dofs.Size(); ++i)
        {
            tr->Transform(nodes.IntPoint(i), x_phys);
            for (int c = 0; c < 3; ++c)
            {
                const double value = L[c][0] * x_phys[0]
                                   + L[c][1] * x_phys[1]
                                   + L[c][2] * x_phys[2];
                const int vdof = fes.DofToVDof(scalar_dofs[i], c);
                const int gtdof = fes.GetGlobalTDofNumber(vdof);
                if (static_cast<HYPRE_BigInt>(gtdof) >= first
                    && static_cast<HYPRE_BigInt>(gtdof) < last)
                {
                    x_true[static_cast<int>(
                        static_cast<HYPRE_BigInt>(gtdof) - first)] =
                        value;
                }
            }
        }
    }
}

void BuildAffineRhsFromRowFactors(
    const ConstraintBuilder3D& builder,
    const double L[3][3],
    mfem::Vector& rhs)
{
    mfem::Vector period_signed;
    mfem::Array<int> comp_idx;
    mfem::Vector ell_hat;
    builder.EmitRowFactors(period_signed, comp_idx, ell_hat);

    rhs.SetSize(ell_hat.Size());
    for (int i = 0; i < rhs.Size(); ++i)
    {
        const int c = comp_idx[i];
        rhs[i] = ell_hat[i]
               * (L[c][0] * period_signed[3*i + 0]
                  + L[c][1] * period_signed[3*i + 1]
                  + L[c][2] * period_signed[3*i + 2]);
    }
}

void BuildAffineRhsFromFilteredRowFactors(
    const ConstraintBuilder3D& builder,
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask,
    const double L[3][3],
    mfem::Vector& rhs,
    mfem::Array<int>& comp_idx)
{
    mfem::Vector period_signed;
    mfem::Vector ell_hat;
    builder.EmitRowFactors(active_pair_labels, comp_mask,
                           period_signed, comp_idx, ell_hat);

    rhs.SetSize(ell_hat.Size());
    for (int i = 0; i < rhs.Size(); ++i)
    {
        const int c = comp_idx[i];
        rhs[i] = ell_hat[i]
               * (L[c][0] * period_signed[3*i + 0]
                  + L[c][1] * period_signed[3*i + 1]
                  + L[c][2] * period_signed[3*i + 2]);
    }
}

std::vector<std::string> ActiveMortarLabelsForAxis(
    const BoundaryClassifier3D& classifier,
    const std::string& axis)
{
    std::vector<std::string> labels;
    for (const auto& tup : classifier.FacePairs())
    {
        if (std::get<0>(tup) == axis)
        {
            labels.push_back(std::get<1>(tup));
        }
    }
    return labels;
}

std::vector<HYPRE_BigInt> ReferenceNonzeroColumns(
    const mfem::HypreParMatrix& H)
{
    std::set<HYPRE_BigInt> cols;
    const HYPRE_BigInt row_first = H.GetRowStarts()[0];

    mfem::SparseMatrix diag;
    H.GetDiag(diag);
    const int* diag_i = diag.GetI();
    const int* diag_j = diag.GetJ();
    const double* diag_a = diag.GetData();
    for (int r = 0; r < diag.Height(); ++r)
    {
        for (int k = diag_i[r]; k < diag_i[r + 1]; ++k)
        {
            if (diag_a[k] != 0.0)
            {
                cols.insert(row_first + static_cast<HYPRE_BigInt>(diag_j[k]));
            }
        }
    }

    mfem::SparseMatrix offd;
    HYPRE_BigInt* cmap = nullptr;
    H.GetOffd(offd, cmap);
    const int* offd_i = offd.GetI();
    const int* offd_j = offd.GetJ();
    const double* offd_a = offd.GetData();
    for (int r = 0; r < offd.Height(); ++r)
    {
        for (int k = offd_i[r]; k < offd_i[r + 1]; ++k)
        {
            if (offd_a[k] != 0.0)
            {
                cols.insert(cmap[offd_j[k]]);
            }
        }
    }

    return std::vector<HYPRE_BigInt>(cols.begin(), cols.end());
}

// ===========================================================================
// Test 1: Operator constructs successfully on the smallest non-trivial mesh.
// ===========================================================================
void test_constructs_on_2x2x2()
{
    std::cout << "Test 1: MortarConstraintOperator constructs on 2x2x2 hex"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);
    int global_height = 0;
    int global_width = 0;
    int local_height = op.Height();
    int local_width = op.Width();
    MPI_Allreduce(&local_height, &global_height, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_width, &global_width, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    AssertOrDie(global_height > 0,
                "MortarConstraintOperator::Height()",
                "rank-summed height is 0, expected positive");
    AssertOrDie(global_width > 0,
                "MortarConstraintOperator::Width()",
                "rank-summed width is 0, expected positive");
    std::cout << "  PASS  Height=" << op.Height()
              << ", Width=" << op.Width() << std::endl;
}

// ===========================================================================
// Test 2: Height / Width match the HypreParMatrix path on np=1.
//
// At np=1 every constraint row is local (FES-aligned and fair-split
// degenerate to the same partition), so the HypreParMatrix's
// (Height, Width) and the EA operator's (Height, Width) must be
// identical. At np>1 they would also be identical because both paths
// use the same FES-aligned row partition (Batch N) and FES TDOF
// column partition (§P4.8.9), but this test runs at np=1 to keep
// it within the unit-test harness.
// ===========================================================================
void test_dimensions_match_hypre_path()
{
    std::cout << "Test 2: dimensions match HypreParMatrix path" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);

    ConstraintBuilder3D builder(cl);
    std::unique_ptr<mfem::HypreParMatrix> H(builder.BuildHypreParMatrix());

    // At np=1 the HypreParMatrix's local Height equals its global
    // Height; ditto for Width. We compare the EA operator's local
    // dimensions to those.
    AssertOrDie(op.Height() == H->Height(),
                "Height matches HypreParMatrix",
                "EA=" + std::to_string(op.Height())
                + ", Hypre=" + std::to_string(H->Height()));
    AssertOrDie(op.Width() == H->Width(),
                "Width matches HypreParMatrix",
                "EA=" + std::to_string(op.Width())
                + ", Hypre=" + std::to_string(H->Width()));
    std::cout << "  PASS  EA(Height,Width) = ("
              << op.Height() << ", " << op.Width()
              << ") matches HypreParMatrix" << std::endl;
}

// ===========================================================================
// Test 2b: AMGF coupled-DOF index set follows active mortar constraints.
//
// The AMGF accessor should return exactly the nonzero column set of the
// active constraint matrix C. Compare against ConstraintBuilder3D's
// assembled HypreParMatrix path instead of a geometric hand count: the
// Wohlmuth corner modifications and row sentinels intentionally mean
// "all boundary nodes" is too broad.
// ===========================================================================
void test_constraint_coupled_dof_indices_q1()
{
    std::cout << "Test 2b: AMGF coupled DOF index set on Q1 2x2x2 hex"
              << std::endl;

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);
    MortarConstraintOperator op(cl);

    const auto& full = op.GetConstraintCoupledDofIndices();
    std::unique_ptr<mfem::HypreParMatrix> H_full(
        builder.BuildHypreParMatrix());
    const auto full_ref = ReferenceNonzeroColumns(*H_full);
    AssertOrDie(std::is_sorted(full.begin(), full.end()),
                "constraint-coupled full set sorted",
                "full set is not sorted");
    AssertOrDie(std::adjacent_find(full.begin(), full.end()) == full.end(),
                "constraint-coupled full set unique",
                "full set has duplicate entries");
    AssertOrDie(full == full_ref,
                "constraint-coupled full set matches HypreParMatrix columns",
                "accessor size " + std::to_string(full.size())
                + " != reference size " + std::to_string(full_ref.size()));
    const std::size_t full_size = full.size();

    const auto active_x_labels = ActiveMortarLabelsForAxis(cl, "x");
    AssertOrDie(!active_x_labels.empty(),
                "constraint-coupled x labels",
                "classifier did not expose an x-axis face pair");

    const std::array<bool, 3> x_only = {{true, false, false}};
    op.Reset(active_x_labels, x_only);

    const auto& filtered = op.GetConstraintCoupledDofIndices();
    std::unique_ptr<mfem::HypreParMatrix> H_filtered(
        builder.BuildHypreParMatrix(active_x_labels, x_only));
    const auto filtered_ref = ReferenceNonzeroColumns(*H_filtered);
    AssertOrDie(std::is_sorted(filtered.begin(), filtered.end()),
                "constraint-coupled filtered set sorted",
                "filtered set is not sorted");
    AssertOrDie(std::adjacent_find(filtered.begin(), filtered.end())
                    == filtered.end(),
                "constraint-coupled filtered set unique",
                "filtered set has duplicate entries");
    AssertOrDie(filtered == filtered_ref,
                "constraint-coupled filtered set matches HypreParMatrix columns",
                "accessor size " + std::to_string(filtered.size())
                + " != reference size " + std::to_string(filtered_ref.size()));
    AssertOrDie(filtered.size() < full_size,
                "constraint-coupled filter shrinks set",
                "filtered size did not shrink");

    std::cout << "  PASS  |I_K| full=" << full_size
              << ", x-only=" << filtered.size() << std::endl;
}

// ===========================================================================
// A/B harness helper: at a given mesh size, builds both EA operator and
// HypreParMatrix, applies both to the same random u (and lambda for
// transpose), verifies the difference is below tolerance.
//
// Returns the absolute and relative error for diagnostic logging by
// the caller. Aborts on failure.
//
// `tag` shows up in PASS/FAIL diagnostics so multi-size runs can
// identify which size failed.
// ===========================================================================
struct AbDiff
{
    double mult_err_abs;
    double mult_norm;
    double mult_T_err_abs;
    double mult_T_norm;
};

AbDiff RunAbHarness(int n_per_side, double tol, const std::string& tag)
{
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, n_per_side);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);
    ConstraintBuilder3D builder(cl);
    std::unique_ptr<mfem::HypreParMatrix> H(builder.BuildHypreParMatrix());

    AssertOrDie(op.Width()  == H->Width(),
                tag + ": Width matches",
                "EA=" + std::to_string(op.Width())
                + ", H=" + std::to_string(H->Width()));
    AssertOrDie(op.Height() == H->Height(),
                tag + ": Height matches",
                "EA=" + std::to_string(op.Height())
                + ", H=" + std::to_string(H->Height()));

    // Deterministic LCG-generated u and lambda. Different seeds for
    // the two vectors so MultTranspose isn't accidentally exercising
    // the same data layout as Mult.
    auto fill_lcg = [](mfem::Vector& v, unsigned seed)
    {
        for (int i = 0; i < v.Size(); ++i)
        {
            seed = seed * 1103515245u + 12345u;
            v[i] = (static_cast<int>(seed) % 1000) / 1000.0 - 0.5;
        }
    };

    mfem::Vector u(op.Width());
    mfem::Vector lambda(op.Height());
    fill_lcg(u, 12345);
    fill_lcg(lambda, 67890);

    AbDiff result;

    // ----- Mult -----
    {
        mfem::Vector y_ea(op.Height());
        mfem::Vector y_hp(op.Height());
        op.Mult(u, y_ea);
        H->Mult(u, y_hp);

        mfem::Vector diff(op.Height());
        diff = y_ea;
        diff -= y_hp;
        result.mult_err_abs = diff.Norml2();
        result.mult_norm    = y_hp.Norml2();

        const double tol_abs = tol * std::max(1.0, result.mult_norm);
        if (result.mult_err_abs > tol_abs)
        {
            std::cerr << "  FAIL  " << tag
                      << ": ||C_ea u - C_hp u||_2 = "
                      << result.mult_err_abs
                      << " > tol*max(1, ||y_hp||) = " << tol_abs
                      << " (||y_hp||_2 = " << result.mult_norm << ")"
                      << std::endl;
            std::exit(1);
        }
    }

    // ----- MultTranspose -----
    {
        mfem::Vector y_ea(op.Width());
        mfem::Vector y_hp(op.Width());
        op.MultTranspose(lambda, y_ea);
        H->MultTranspose(lambda, y_hp);

        mfem::Vector diff(op.Width());
        diff = y_ea;
        diff -= y_hp;
        result.mult_T_err_abs = diff.Norml2();
        result.mult_T_norm    = y_hp.Norml2();

        const double tol_abs = tol * std::max(1.0, result.mult_T_norm);
        if (result.mult_T_err_abs > tol_abs)
        {
            std::cerr << "  FAIL  " << tag
                      << ": ||C^T_ea lambda - C^T_hp lambda||_2 = "
                      << result.mult_T_err_abs
                      << " > tol*max(1, ||y_hp||) = " << tol_abs
                      << " (||y_hp||_2 = " << result.mult_T_norm << ")"
                      << std::endl;
            std::exit(1);
        }
    }

    return result;
}

// ===========================================================================
// Test 3: A/B at multiple mesh sizes. Catches size-dependent bugs that
// might pass at one size but fail at another (e.g. an off-by-one in
// the per-pair scatter that only triggers when n_n > 1, or sparsity-
// pattern bugs that only show up when A_m has multiple nnz per row).
// ===========================================================================
void test_ab_multi_size()
{
    std::cout << "Test 3: A/B at multiple mesh sizes" << std::endl;
    // Phase 4.3 / Batch Q tolerance contract: 1e-12 abs (per
    // §P4.4.6.3). Headroom: typical FP-rearrangement error at these
    // sizes is ~1e-14, so 1e-12 catches real bugs while leaving 2
    // orders of magnitude for FP drift.
    constexpr double kTol = 1.0e-12;

    for (int n : {2, 4, 6, 8})
    {
        const std::string tag = "n=" + std::to_string(n);
        AbDiff d = RunAbHarness(n, kTol, tag);
        std::cout << "  PASS  " << tag
                  << ":  Mult err=" << d.mult_err_abs
                  << " (rel " << d.mult_err_abs / std::max(1.0, d.mult_norm)
                  << "),  MultT err=" << d.mult_T_err_abs
                  << " (rel " << d.mult_T_err_abs
                                / std::max(1.0, d.mult_T_norm)
                  << ")" << std::endl;
    }
}

// ===========================================================================
// Test 4: zero-input invariant. Both Mult(0, _) and MultTranspose(0, _)
// must produce zero output (Cu = 0 when u = 0; same for transpose).
// This is a basic linearity sanity check; if either path's
// initialization or accumulation is buggy it can leave residual
// noise in the output even on zero input.
// ===========================================================================
void test_zero_input()
{
    std::cout << "Test 4: zero-input produces zero output" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    MortarConstraintOperator op(cl);

    mfem::Vector u(op.Width());
    mfem::Vector lambda(op.Height());
    u = 0.0;
    lambda = 0.0;

    mfem::Vector y(op.Height());
    op.Mult(u, y);
    AssertOrDie(y.Norml2() < 1.0e-14,
                "Mult(0)",
                "||y||_2 = " + std::to_string(y.Norml2()));

    mfem::Vector z(op.Width());
    op.MultTranspose(lambda, z);
    AssertOrDie(z.Norml2() < 1.0e-14,
                "MultTranspose(0)",
                "||z||_2 = " + std::to_string(z.Norml2()));

    std::cout << "  PASS  Mult(0)=0 and MultTranspose(0)=0" << std::endl;
}

// ===========================================================================
// Test 5: harness self-check (negative test). Build the EA output,
// perturb one entry, and verify our A/B-comparison logic catches the
// difference. This guards against the harness being too lenient — if
// future tightening of tol breaks this check, the harness will alert
// us before silently accepting a real EA bug.
// ===========================================================================
void test_negative_harness_self_check()
{
    std::cout << "Test 5: harness catches a deliberately perturbed result"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);
    ConstraintBuilder3D builder(cl);
    std::unique_ptr<mfem::HypreParMatrix> H(builder.BuildHypreParMatrix());

    mfem::Vector u(op.Width());
    {
        unsigned seed = 12345;
        for (int i = 0; i < op.Width(); ++i)
        {
            seed = seed * 1103515245u + 12345u;
            u[i] = (static_cast<int>(seed) % 1000) / 1000.0 - 0.5;
        }
    }

    mfem::Vector y_ea(op.Height());
    mfem::Vector y_hp(op.Height());
    op.Mult(u, y_ea);
    H->Mult(u, y_hp);

    // Inject a 1e-3 perturbation — well above any tolerance we'd ever
    // realistically use. The harness comparison MUST flag this.
    constexpr double kPerturbation = 1.0e-3;
    if (y_ea.Size() > 0) { y_ea[0] += kPerturbation; }

    mfem::Vector diff(op.Height());
    diff = y_ea;
    diff -= y_hp;
    const double err  = diff.Norml2();
    const double norm = y_hp.Norml2();
    constexpr double kHarnessTol = 1.0e-12;
    const double tol_abs = kHarnessTol * std::max(1.0, norm);

    AssertOrDie(err > tol_abs,
                "harness catches perturbation",
                "perturbation " + std::to_string(kPerturbation)
                + " yielded ||diff||_2 = " + std::to_string(err)
                + " <= tol_abs " + std::to_string(tol_abs)
                + " (harness is too loose to catch real bugs)");
    std::cout << "  PASS  harness flags " << kPerturbation
              << "-magnitude perturbation: ||diff||_2 = " << err
              << " > " << tol_abs << std::endl;
}

// ===========================================================================
// Test 6 (Phase 4.3 / Batch R): ComputeInvDiagSchur agrees with the
// HypreParMatrix-path formula.
//
// The formula:
//   schur_diag[i] = sum_j C[i,j]^2 * inv_diag_K[j]
//
// We pick inv_diag_K = ones(global_size) so the formula simplifies to
//   schur_diag[i] = sum_j C[i,j]^2 = ||C[i,:]||_2^2.
//
// Then both:
//   - op.ComputeInvDiagSchur(ones).inv -> schur_diag (after element
//                                                     -wise reciprocal)
//   - HypreParMatrix C: walk CSR, sum squares per row -> schur_diag
//
// must match to FP precision. We compare the un-inverted Schur diagonals
// (not the inverses) to avoid 1/0 issues on Dirichlet-zeroed rows; the
// reciprocal logic is the same in both paths so we don't need to test
// it separately.
// ===========================================================================
void test_compute_inv_diag_schur_matches_hypre()
{
    std::cout << "Test 6: ComputeInvDiagSchur agrees with HypreParMatrix path"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);
    ConstraintBuilder3D builder(cl);
    std::unique_ptr<mfem::HypreParMatrix> H(builder.BuildHypreParMatrix());

    // inv_diag_K = ones(local_size). At np=1 local_size = global_size.
    // Phase 5.5 — ComputeInvDiagSchur now takes a `const mfem::Solver&`;
    // wrap inv_diag_K in a DiagonalScaler whose Mult(ones, _) returns
    // the same values back.
    mfem::Vector inv_diag_K(op.Width());
    inv_diag_K = 1.0;
    DiagonalScaler K_jacobi_prec(inv_diag_K.Size(), inv_diag_K);

    // EA path: returns inv_schur. Invert back to schur for comparison.
    mfem::Vector inv_schur_ea = op.ComputeInvDiagSchur(K_jacobi_prec);
    mfem::Vector schur_ea(op.Height());
    for (int i = 0; i < op.Height(); ++i)
    {
        const double v = inv_schur_ea[i];
        schur_ea[i] = (std::abs(v) > 1.0e-300) ? (1.0 / v) : 0.0;
    }

    // HypreParMatrix path: sum-of-squares per row from the local CSR
    // blocks. Under MPI, Hypre splits columns into diag and offd
    // blocks; both contribute to C_i * diag(K)^{-1} * C_i^T. The
    // offd column map is irrelevant here because inv_diag_K is ones.
    mfem::Vector schur_hp(op.Height());
    schur_hp = 0.0;
    {
        mfem::SparseMatrix C_diag;
        H->GetDiag(C_diag);
        const int* I    = C_diag.GetI();
        const double* A = C_diag.GetData();
        for (int i = 0; i < op.Height(); ++i)
        {
            double s = 0.0;
            for (int k = I[i]; k < I[i + 1]; ++k)
            {
                s += A[k] * A[k];
            }
            schur_hp[i] = s;
        }

        mfem::SparseMatrix C_offd;
        HYPRE_BigInt* cmap = nullptr;
        H->GetOffd(C_offd, cmap);
        const int* OI    = C_offd.GetI();
        const double* OA = C_offd.GetData();
        for (int i = 0; i < op.Height(); ++i)
        {
            double s = schur_hp[i];
            for (int k = OI[i]; k < OI[i + 1]; ++k)
            {
                s += OA[k] * OA[k];
            }
            schur_hp[i] = s;
        }
    }

    mfem::Vector diff(op.Height());
    diff = schur_ea;
    diff -= schur_hp;
    const double err  = diff.Norml2();
    const double norm = schur_hp.Norml2();
    constexpr double kTol = 1.0e-12;
    const double tol_abs = kTol * std::max(1.0, norm);

    if (err > tol_abs)
    {
        std::cerr << "  FAIL  ||schur_ea - schur_hp||_2 = " << err
                  << " > " << tol_abs
                  << " (||schur_hp||_2 = " << norm << ")" << std::endl;
        // Diagnostic: print a few entries.
        std::cerr << "  First 5 entries (ea, hp, diff):" << std::endl;
        for (int i = 0; i < std::min(5, op.Height()); ++i)
        {
            std::cerr << "    [" << i << "] " << schur_ea[i] << ", "
                      << schur_hp[i] << ", "
                      << (schur_ea[i] - schur_hp[i]) << std::endl;
        }
        std::exit(1);
    }
    std::cout << "  PASS  ||schur_ea - schur_hp||_2 = " << err
              << " (rel " << err / std::max(1.0, norm) << ")" << std::endl;
}

// ===========================================================================
// Test 7: BuildCTransposeC agrees with the operator composition C^T(C u).
//
// This is the Phase D setup contract. The augmented-Lagrangian K block uses
// K_gamma = K + gamma C^T C, so the explicitly assembled C^T C matrix must
// act exactly like applying the current EA constraint operator and then its
// transpose. The second check applies an X-only Reset before assembly to make
// sure the setup path follows the same active-pair/component filter as Mult.
// ===========================================================================
void test_build_c_transpose_c_action_matches_operator_composition()
{
    std::cout << "Test 7: BuildCTransposeC matches C^T(Cu), including filter"
              << std::endl;

    auto run_case = [](bool x_only_filter)
    {
        auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
        BoundaryClassifier3D cl(*b.pmesh, *b.fes);
        MortarConstraintOperator op(cl);

        std::string tag = "BuildCTransposeC full XYZ";
        if (x_only_filter)
        {
            const std::vector<std::string> active_x_labels =
                ActiveMortarLabelsForAxis(cl, "x");
            AssertOrDie(!active_x_labels.empty(),
                        "BuildCTransposeC X-only active labels",
                        "classifier did not expose an x-axis face pair");
            op.Reset(active_x_labels, {{true, false, false}});
            tag = "BuildCTransposeC X-only";
        }

        std::unique_ptr<mfem::HypreParMatrix> CtC =
            op.BuildCTransposeC();

        AssertOrDie(CtC != nullptr, tag,
                    "BuildCTransposeC returned null");
        AssertOrDie(CtC->Height() == op.Width()
                    && CtC->Width() == op.Width(),
                    tag,
                    "local matrix dimensions ("
                    + std::to_string(CtC->Height()) + ", "
                    + std::to_string(CtC->Width())
                    + ") do not match operator Width() "
                    + std::to_string(op.Width()));

        mfem::Vector u(op.Width());
        unsigned seed = x_only_filter ? 98765u : 24680u;
        for (int i = 0; i < u.Size(); ++i)
        {
            seed = seed * 1103515245u + 12345u;
            u[i] = (static_cast<int>(seed) % 2000) / 1000.0 - 1.0;
        }

        mfem::Vector Cu(op.Height());
        mfem::Vector y_ref(op.Width());
        mfem::Vector y_mat(op.Width());
        op.Mult(u, Cu);
        op.MultTranspose(Cu, y_ref);
        CtC->Mult(u, y_mat);

        mfem::Vector diff(op.Width());
        diff = y_mat;
        diff -= y_ref;

        const double err = diff.Norml2();
        const double scale = std::max(1.0, y_ref.Norml2());
        const double tol = 2.0e-12 * scale;
        AssertOrDie(err <= tol, tag,
                    "||C^T C u - C^T(Cu)||_2 = "
                    + std::to_string(err)
                    + " > " + std::to_string(tol));

        std::cout << "  PASS  " << tag << ": rows=" << op.Height()
                  << ", ||diff||_2=" << err << std::endl;
    };

    run_case(false);
    run_case(true);
}

// ===========================================================================
// Test 8 (Phase 6.0.E): projector-mediated direct path matches the
// legacy parent-FES operator at lor_depth=1.
//
// This is the first operator-level Phase 6 gate. The new constructor
// consumes a classifier built on an unrefined boundary ParSubMesh and
// a SurfaceProjector that maps that submesh FES back to the parent
// volume FES. At p=1 that projector is a permutation, so the resulting
// constraint values must match the legacy classifier-built-on-parent
// path up to row permutation. The submesh classifier is allowed to
// enumerate rows in submesh-local order; this test therefore compares
// the sorted Mult output and uses an all-ones lambda for MultTranspose,
// which is invariant under row permutation.
// ===========================================================================
void test_projector_direct_path_matches_legacy_operator()
{
    std::cout << "Test 7: projector direct path matches legacy operator"
              << std::endl;

    auto b = BuildSharedHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D legacy_classifier(*b.pmesh, *b.fes);
    MortarConstraintOperator legacy_op(legacy_classifier);

    mfem::Array<int> bdr_attrs(b.pmesh->bdr_attributes);
    auto bdr_submesh = std::make_shared<mfem::ParSubMesh>(
        mfem::ParSubMesh::CreateFromBoundary(*b.pmesh, bdr_attrs));
    auto bdr_fec = std::make_shared<mfem::H1_FECollection>(
        /*order=*/1, bdr_submesh->SpaceDimension());
    auto bdr_fes = std::make_shared<mfem::ParFiniteElementSpace>(
        bdr_submesh.get(), bdr_fec.get(), /*vdim=*/3,
        mfem::Ordering::byNODES);

    auto projected_classifier = std::make_shared<BoundaryClassifier3D>(
        bdr_submesh, bdr_fes);
    auto projector = std::make_shared<SurfaceProjector>(
        b.fes, bdr_fes, bdr_submesh, /*snap_tol=*/1.0e-10);
    MortarConstraintOperator projected_op(projected_classifier, projector,
                                          b.fes);

    AssertOrDie(projected_op.Width() == legacy_op.Width(),
                "projector direct Width",
                "projected=" + std::to_string(projected_op.Width())
                + ", legacy=" + std::to_string(legacy_op.Width()));
    AssertOrDie(projected_op.Height() == legacy_op.Height(),
                "projector direct Height",
                "projected=" + std::to_string(projected_op.Height())
                + ", legacy=" + std::to_string(legacy_op.Height()));

    auto fill_lcg = [](mfem::Vector& v, unsigned seed)
    {
        for (int i = 0; i < v.Size(); ++i)
        {
            seed = seed * 1103515245u + 12345u;
            v[i] = (static_cast<int>(seed) % 1000) / 1000.0 - 0.5;
        }
    };

    mfem::Vector u(legacy_op.Width());
    mfem::Vector lambda(legacy_op.Height());
    fill_lcg(u, 24680);
    lambda = 1.0;

    mfem::Vector y_legacy(legacy_op.Height());
    mfem::Vector y_projected(projected_op.Height());
    legacy_op.Mult(u, y_legacy);
    projected_op.Mult(u, y_projected);

    std::vector<double> y_legacy_sorted(y_legacy.Size());
    std::vector<double> y_projected_sorted(y_projected.Size());
    for (int i = 0; i < y_legacy.Size(); ++i)
    {
        y_legacy_sorted[i] = y_legacy[i];
        y_projected_sorted[i] = y_projected[i];
    }
    std::sort(y_legacy_sorted.begin(), y_legacy_sorted.end());
    std::sort(y_projected_sorted.begin(), y_projected_sorted.end());

    double mult_err_sq = 0.0;
    for (int i = 0; i < y_legacy.Size(); ++i)
    {
        const double d = y_projected_sorted[i] - y_legacy_sorted[i];
        mult_err_sq += d * d;
    }
    const double mult_err = std::sqrt(mult_err_sq);
    const double mult_tol = 1.0e-12 * std::max(1.0, y_legacy.Norml2());
    AssertOrDie(mult_err <= mult_tol,
                "projector direct Mult row multiset",
                "||sort(projected) - sort(legacy)||_2 = "
                + std::to_string(mult_err)
                + " > " + std::to_string(mult_tol));

    mfem::Vector z_legacy(legacy_op.Width());
    mfem::Vector z_projected(projected_op.Width());
    legacy_op.MultTranspose(lambda, z_legacy);
    projected_op.MultTranspose(lambda, z_projected);

    mfem::Vector z_diff(z_legacy.Size());
    z_diff = z_projected;
    z_diff -= z_legacy;
    const double mult_t_err = z_diff.Norml2();
    const double mult_t_tol = 1.0e-12 * std::max(1.0, z_legacy.Norml2());
    AssertOrDie(mult_t_err <= mult_t_tol,
                "projector direct MultTranspose",
                "||projected - legacy||_2 = "
                + std::to_string(mult_t_err)
                + " > " + std::to_string(mult_t_tol));

    std::cout << "  PASS  projector direct path matches legacy up to row "
              << "permutation: sorted Mult err=" << mult_err
              << ", ones-lambda MultT err=" << mult_t_err << std::endl;
}

// ===========================================================================
// Test 8 (Phase 6.1.B): P2 tetrahedral parent space with a once-refined
// linear LOR boundary.
//
// This is the operator-level smoke test for `mesh.order = 2` /
// `lor_depth = 2`. The classifier and row metadata live on the refined
// boundary submesh, while the operator domain is the parent P2 volume
// FE space. For an affine field u(x) = L x, the projected constraint
// output must equal the reference RHS assembled from
// ConstraintBuilder3D::EmitRowFactors. That checks the LOR row walk,
// parent-column projection, and signed-period convention together.
// ===========================================================================
void test_p2_tet_lor_affine_constraint_rhs()
{
    std::cout << "Test 8: P2 tet LOR affine constraint RHS" << std::endl;

    auto b = BuildSharedTetFesBundle(MPI_COMM_WORLD,
                                     /*n_per_side=*/4,
                                     /*order=*/2);

    mfem::Array<int> bdr_attrs(b.pmesh->bdr_attributes);
    auto bdr_submesh = std::make_shared<mfem::ParSubMesh>(
        mfem::ParSubMesh::CreateFromBoundary(*b.pmesh, bdr_attrs));
    bdr_submesh->UniformRefinement();

    auto bdr_fec = std::make_shared<mfem::H1_FECollection>(
        /*order=*/1, bdr_submesh->SpaceDimension());
    auto bdr_fes = std::make_shared<mfem::ParFiniteElementSpace>(
        bdr_submesh.get(), bdr_fec.get(), /*vdim=*/3,
        mfem::Ordering::byNODES);

    auto classifier = std::make_shared<BoundaryClassifier3D>(
        bdr_submesh, bdr_fes);
    auto projector = std::make_shared<SurfaceProjector>(
        b.fes, bdr_fes, bdr_submesh, /*snap_tol=*/1.0e-10);
    ConstraintBuilder3D builder(classifier, projector, b.fes);
    MortarConstraintOperator op(classifier, projector, b.fes);

    AssertOrDie(op.Width() == b.fes->GetTrueVSize(),
                "P2 tet LOR operator Width",
                "operator width does not match parent P2 FES true size");
    AssertOrDie(op.Height() == builder.NumLocalRows(),
                "P2 tet LOR operator Height",
                "operator height does not match projected builder rows");
    int global_height = 0;
    int local_height = op.Height();
    MPI_Allreduce(&local_height, &global_height, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    AssertOrDie(global_height > 0,
                "P2 tet LOR operator nonempty",
                "expected positive rank-summed constraint rows");

    const double L[3][3] = {
        { 0.20, -0.05,  0.03},
        { 0.07,  0.11, -0.02},
        {-0.04,  0.06,  0.13}
    };

    mfem::Vector u_parent;
    FillParentAffineTrace(*b.fes, L, u_parent);

    mfem::Vector y(op.Height());
    op.Mult(u_parent, y);

    mfem::Vector rhs;
    BuildAffineRhsFromRowFactors(builder, L, rhs);
    AssertOrDie(rhs.Size() == y.Size(),
                "P2 tet LOR RHS size",
                "row-factor RHS size does not match operator output");

    mfem::Vector diff(y.Size());
    diff = y;
    diff -= rhs;
    const double err = diff.Norml2();
    const double scale = std::max(1.0, rhs.Norml2());
    const double tol = 2.0e-12 * scale;
    AssertOrDie(err <= tol,
                "P2 tet LOR affine constraint RHS",
                "||C*u_affine - g_affine||_2 = "
                + std::to_string(err)
                + " > " + std::to_string(tol));

    std::cout << "  PASS  P2 tet LOR affine RHS: rows=" << op.Height()
              << ", parent_width=" << op.Width()
              << ", ||C*u-g||_2=" << err << std::endl;
}

// ===========================================================================
// Test 9 (Phase 6.1.D): component-restricted P2 tetrahedral LOR path.
//
// Phase 5.9 lets SystemDriver rebuild the active PBC spec at runtime:
// active face-pair labels select periodic directions and comp_mask
// selects which vector components are constrained. This test exercises
// that same reset cascade on the Phase 6 projected P2 tet geometry:
//   1. build the default all-pair/all-component P2 LOR operator,
//   2. reset it to the x-axis face pair with x-component rows only,
//   3. verify builder/operator row counts and row-factor RHS sizing
//      agree after the reset,
//   4. verify the affine solution satisfies the filtered constraint.
// ===========================================================================
void test_p2_tet_lor_x_only_filter_affine_rhs()
{
    std::cout << "Test 9: P2 tet LOR X-only filter affine RHS"
              << std::endl;

    auto b = BuildSharedTetFesBundle(MPI_COMM_WORLD,
                                     /*n_per_side=*/4,
                                     /*order=*/2);

    mfem::Array<int> bdr_attrs(b.pmesh->bdr_attributes);
    auto bdr_submesh = std::make_shared<mfem::ParSubMesh>(
        mfem::ParSubMesh::CreateFromBoundary(*b.pmesh, bdr_attrs));
    bdr_submesh->UniformRefinement();

    auto bdr_fec = std::make_shared<mfem::H1_FECollection>(
        /*order=*/1, bdr_submesh->SpaceDimension());
    auto bdr_fes = std::make_shared<mfem::ParFiniteElementSpace>(
        bdr_submesh.get(), bdr_fec.get(), /*vdim=*/3,
        mfem::Ordering::byNODES);

    auto classifier = std::make_shared<BoundaryClassifier3D>(
        bdr_submesh, bdr_fes);
    auto projector = std::make_shared<SurfaceProjector>(
        b.fes, bdr_fes, bdr_submesh, /*snap_tol=*/1.0e-10);
    ConstraintBuilder3D builder(classifier, projector, b.fes);
    MortarConstraintOperator op(classifier, projector, b.fes);

    const int full_height = op.Height();
    const auto active_x_labels = ActiveMortarLabelsForAxis(*classifier, "x");
    AssertOrDie(!active_x_labels.empty(),
                "P2 tet LOR X-only active labels",
                "classifier did not expose an x-axis face pair");

    const std::array<bool, 3> x_only = {{true, false, false}};
    op.Reset(active_x_labels, x_only);

    const int filtered_height = op.Height();
    const int builder_height =
        builder.NumLocalRows(active_x_labels, x_only);
    AssertOrDie(filtered_height == builder_height,
                "P2 tet LOR filtered Height",
                "operator height " + std::to_string(filtered_height)
                + " != builder NumLocalRows "
                + std::to_string(builder_height));
    AssertOrDie(filtered_height <= full_height,
                "P2 tet LOR filtered Height <= full Height",
                "filtered height " + std::to_string(filtered_height)
                + " > full height " + std::to_string(full_height));

    int global_filtered_height = 0;
    int local_filtered_height = filtered_height;
    MPI_Allreduce(&local_filtered_height, &global_filtered_height,
                  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    AssertOrDie(global_filtered_height > 0,
                "P2 tet LOR filtered nonempty",
                "expected positive rank-summed filtered rows");

    const double L[3][3] = {
        { 0.20, -0.05,  0.03},
        { 0.07,  0.11, -0.02},
        {-0.04,  0.06,  0.13}
    };

    mfem::Vector u_parent;
    FillParentAffineTrace(*b.fes, L, u_parent);

    mfem::Vector y(op.Height());
    op.Mult(u_parent, y);

    mfem::Vector rhs;
    mfem::Array<int> comp_idx;
    BuildAffineRhsFromFilteredRowFactors(
        builder, active_x_labels, x_only, L, rhs, comp_idx);

    AssertOrDie(rhs.Size() == y.Size(),
                "P2 tet LOR filtered RHS size",
                "row-factor RHS size " + std::to_string(rhs.Size())
                + " != operator output size " + std::to_string(y.Size()));
    AssertOrDie(comp_idx.Size() == y.Size(),
                "P2 tet LOR filtered component-index size",
                "component-index size " + std::to_string(comp_idx.Size())
                + " != operator output size " + std::to_string(y.Size()));
    for (int i = 0; i < comp_idx.Size(); ++i)
    {
        AssertOrDie(comp_idx[i] == 0,
                    "P2 tet LOR filtered component index",
                    "expected x-component row, got component "
                    + std::to_string(comp_idx[i]));
    }

    mfem::Vector diff(y.Size());
    diff = y;
    diff -= rhs;
    const double err = diff.Norml2();
    const double scale = std::max(1.0, rhs.Norml2());
    const double tol = 2.0e-12 * scale;
    AssertOrDie(err <= tol,
                "P2 tet LOR X-only affine constraint RHS",
                "||C_x*u_affine - g_x||_2 = "
                + std::to_string(err)
                + " > " + std::to_string(tol));

    std::cout << "  PASS  P2 tet LOR X-only filter: rows="
              << filtered_height << " (global "
              << global_filtered_height << "), ||C*u-g||_2="
              << err << std::endl;
}

}  // anonymous namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "==============================================="
                  << std::endl;
        std::cout << "test_mortar_constraint_operator (Phase 4.3/R)"
                  << std::endl;
        std::cout << "==============================================="
                  << std::endl;
    }

    if (argc > 1 && std::string(argv[1]) == "--ctc-only")
    {
        test_build_c_transpose_c_action_matches_operator_composition();
        if (rank == 0)
        {
            std::cout << "==============================================="
                      << std::endl;
            std::cout << "MortarConstraintOperator C^T C tests passed."
                      << std::endl;
            std::cout << "==============================================="
                      << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    test_constructs_on_2x2x2();
    test_dimensions_match_hypre_path();
    test_constraint_coupled_dof_indices_q1();
    test_ab_multi_size();
    test_zero_input();
    test_negative_harness_self_check();
    test_compute_inv_diag_schur_matches_hypre();
    test_build_c_transpose_c_action_matches_operator_composition();
    test_projector_direct_path_matches_legacy_operator();
    test_p2_tet_lor_affine_constraint_rhs();
    test_p2_tet_lor_x_only_filter_affine_rhs();

    if (rank == 0)
    {
        std::cout << "==============================================="
                  << std::endl;
        std::cout << "All MortarConstraintOperator tests passed."
                  << std::endl;
        std::cout << "==============================================="
                  << std::endl;
    }
    MPI_Finalize();
    return 0;
}
