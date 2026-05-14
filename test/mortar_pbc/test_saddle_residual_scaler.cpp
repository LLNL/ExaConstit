// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.C — unit tests for SaddleResidualScaler.
//
// Most tests construct the scaler with a small hand-crafted partition
// (via SetPartitionDirect) — n_u = 2 or 4, n_lambda = 6, 2 sub-blocks
// — so the math can be verified without building an MFEM mesh.
//
// One integration test (test_rebuild_partition_from_builder) does
// build a 2x2x2 hex mesh + BoundaryClassifier3D + ConstraintBuilder3D
// to exercise RebuildPartition's delegation to GetRowSubblockIds
// (Phase 5.11.B).
//
// Each test function exits via std::exit(1) on failure (with a
// diagnostic to stderr) or returns normally on success.

#include "saddle_residual_scaler.hpp"
#include "constraint_builder_3d.hpp"
#include "boundary_classifier_3d.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using mortar_pbc::SaddleResidualScaler;
using mortar_pbc::SaddleResidualScalerConfig;
using mortar_pbc::SubblockPartition;
using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ConstraintBuilder3D;

namespace
{

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

void AssertNear(double a, double b, double tol,
                const std::string& test_name,
                const std::string& detail)
{
    if (std::abs(a - b) > tol)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail
                  << "  (got " << a << ", expected " << b
                  << ", diff " << std::abs(a - b)
                  << ", tol " << tol << ")" << std::endl;
        std::exit(1);
    }
}

struct FesBundle
{
    std::unique_ptr<mfem::ParMesh> pmesh;
    std::unique_ptr<mfem::H1_FECollection> fec;
    std::unique_ptr<mfem::ParFiniteElementSpace> fes;
};

FesBundle BuildHexFesBundle(MPI_Comm comm, int n_per_side)
{
    FesBundle b;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n_per_side, n_per_side, n_per_side,
        mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
    b.pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
    b.fec = std::make_unique<mfem::H1_FECollection>(1, 3);
    b.fes = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), 3, mfem::Ordering::byNODES);
    return b;
}

// Hand-crafted partition: 6 lambda rows, 2 sub-blocks (rows 0-2 in
// sub-block 0 "edge", rows 3-5 in sub-block 1 "face").
void SetupTestPartition(SaddleResidualScaler& scaler)
{
    std::vector<std::string> labels = {"edge", "face"};
    mfem::Array<int> sb_of_row(6);
    sb_of_row[0] = 0; sb_of_row[1] = 0; sb_of_row[2] = 0;
    sb_of_row[3] = 1; sb_of_row[4] = 1; sb_of_row[5] = 1;
    scaler.SetPartitionDirect(labels, sb_of_row);
}

// Build a 3-entry block offsets array for layout (n_u | n_lam).
//
// Returns by value: `mfem::Array<int>` owns its own data, so RVO /
// move / copy all produce a caller-owned array safe to use as the
// backing for a BlockVector in the caller's scope.
mfem::Array<int> MakeOffsets(int n_u, int n_lam)
{
    mfem::Array<int> offs(3);
    offs[0] = 0;
    offs[1] = n_u;
    offs[2] = n_u + n_lam;
    return offs;
}

// Fill a pre-constructed BlockVector with block values.
//
// IMPORTANT (MFEM gotcha): we deliberately do NOT provide a
// `MakeBlockVector(...)` helper that returns a BlockVector by value.
// `mfem::BlockVector` stores a `const Array<int>*` pointer (not a
// copy) to its offsets array; if the offsets array goes out of scope
// while the BlockVector is still alive, that pointer dangles. Each
// test owns its own `mfem::Array<int> offs` (via `MakeOffsets`) and
// constructs `mfem::BlockVector r(offs)` directly so the offsets'
// lifetime brackets the BlockVector's.
void FillBlockVector(mfem::BlockVector& r,
                      std::initializer_list<double> u_vals,
                      std::initializer_list<double> lam_vals)
{
    int i = 0;
    for (double v : u_vals)   { r.GetBlock(0)[i++] = v; }
    i = 0;
    for (double v : lam_vals) { r.GetBlock(1)[i++] = v; }
}

//==============================================================================
// Test 1: constructor leaves scaler in identity / empty-partition state
//==============================================================================
void test_constructor_defaults()
{
    std::cout << "Test 1: constructor defaults" << std::endl;
    SaddleResidualScalerConfig cfg;
    SaddleResidualScaler scaler(cfg);

    AssertOrDie(!scaler.IsEnabled(), "default enabled",
                "expected disabled by default");
    AssertOrDie(scaler.NumSubblocks() == 0,
                "default NumSubblocks",
                "expected 0 (no partition set yet)");
    AssertOrDie(scaler.GetDu() == 1.0,
                "default d_u",
                "expected 1.0 (identity)");
    AssertOrDie(scaler.GetDLambda().Size() == 0,
                "default d_lambda size",
                "expected 0");
    AssertOrDie(scaler.SubblockLabels().empty(),
                "default labels",
                "expected empty");

    std::cout << "  PASS  default: disabled, 0 sub-blocks, identity scaling"
              << std::endl;
}

//==============================================================================
// Test 2: SetPartitionDirect populates state + resets to identity
//==============================================================================
void test_set_partition_direct()
{
    std::cout << "Test 2: SetPartitionDirect populates state" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    SaddleResidualScaler scaler(cfg);

    SetupTestPartition(scaler);

    AssertOrDie(scaler.NumSubblocks() == 2,
                "n_subblocks", "expected 2");
    AssertOrDie(scaler.SubblockLabels().size() == 2,
                "labels size", "expected 2");
    AssertOrDie(scaler.SubblockLabels()[0] == "edge",
                "labels[0]", "expected 'edge'");
    AssertOrDie(scaler.SubblockLabels()[1] == "face",
                "labels[1]", "expected 'face'");
    AssertOrDie(scaler.SubblockOfRow().Size() == 6,
                "subblock_of_row size", "expected 6");
    AssertOrDie(scaler.GetDLambda().Size() == 6,
                "d_lambda size", "expected 6 (matches n_lambda)");

    // All scaling factors initialized to identity (1.0).
    AssertOrDie(scaler.GetDu() == 1.0,
                "d_u after partition", "expected 1");
    for (int i = 0; i < 6; ++i)
    {
        AssertOrDie(scaler.GetDLambda()[i] == 1.0,
                    "d_lambda[" + std::to_string(i) + "] after partition",
                    "expected 1");
    }

    std::cout << "  PASS  partition set; scaling factors identity (1.0)"
              << std::endl;
}

//==============================================================================
// Test 3: Choose with per_subblock = false (joint scaling)
//==============================================================================
void test_choose_per_subblock_off()
{
    std::cout << "Test 3: Choose per_subblock = false" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = false;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    // r_u_norm = 7; per-sub-block lambda norms = {3, 4}.
    // joint lambda norm = sqrt(9 + 16) = 5.
    const double r_u = 7.0;
    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 3.0;
    r_lam_sb[1] = 4.0;
    scaler.Choose(r_u, r_lam_sb);

    AssertNear(scaler.GetDu(), 7.0, 1e-14, "d_u", "expected 7");

    // All 6 lambda rows get joint d_lambda = 5.
    for (int i = 0; i < 6; ++i)
    {
        AssertNear(scaler.GetDLambda()[i], 5.0, 1e-14,
                   "d_lambda[" + std::to_string(i) + "]",
                   "expected 5 (joint)");
    }

    std::cout << "  PASS  joint d_lambda = sqrt(3^2 + 4^2) = 5 broadcast to "
              << "all rows" << std::endl;
}

//==============================================================================
// Test 4: Choose with per_subblock = true
//==============================================================================
void test_choose_per_subblock_on()
{
    std::cout << "Test 4: Choose per_subblock = true" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    const double r_u = 11.0;
    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 3.0;     // edge sub-block norm
    r_lam_sb[1] = 100.0;   // face sub-block norm
    scaler.Choose(r_u, r_lam_sb);

    AssertNear(scaler.GetDu(), 11.0, 1e-14, "d_u", "expected 11");

    // Rows 0-2 (sub-block 0): d_lambda = 3.
    for (int i = 0; i < 3; ++i)
    {
        AssertNear(scaler.GetDLambda()[i], 3.0, 1e-14,
                   "d_lambda[" + std::to_string(i) + "] sb0",
                   "expected 3 (edge)");
    }
    // Rows 3-5 (sub-block 1): d_lambda = 100.
    for (int i = 3; i < 6; ++i)
    {
        AssertNear(scaler.GetDLambda()[i], 100.0, 1e-14,
                   "d_lambda[" + std::to_string(i) + "] sb1",
                   "expected 100 (face)");
    }

    std::cout << "  PASS  per-sub-block d_lambda: 3 (edge), 100 (face)"
              << std::endl;
}

//==============================================================================
// Test 5: floor guard — sub-block norms below floor → d = 1.0
//==============================================================================
void test_choose_floor_guard()
{
    std::cout << "Test 5: floor guard" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    cfg.floor = 1.0e-12;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    const double r_u = 1.0e-15;   // below floor
    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 1.0e-16;        // below floor
    r_lam_sb[1] = 100.0;          // above floor
    scaler.Choose(r_u, r_lam_sb);

    // r_u < floor → d_u = 1 (NOT d_u = floor — the floor guard sets
    // d = 1 explicitly so tiny residuals don't get amplified by 1/floor).
    AssertNear(scaler.GetDu(), 1.0, 1e-14,
               "d_u floor guard", "expected 1 (norm below floor)");

    for (int i = 0; i < 3; ++i)
    {
        AssertNear(scaler.GetDLambda()[i], 1.0, 1e-14,
                   "d_lambda[" + std::to_string(i) + "] sb0 floor guard",
                   "expected 1");
    }
    for (int i = 3; i < 6; ++i)
    {
        AssertNear(scaler.GetDLambda()[i], 100.0, 1e-14,
                   "d_lambda[" + std::to_string(i) + "] sb1 normal",
                   "expected 100");
    }

    std::cout << "  PASS  floor guard: sub-norms < floor → d = 1; "
              << "above-floor norms use their value" << std::endl;
}

//==============================================================================
// Test 6: range cap — huge norms clipped at cap
//==============================================================================
void test_choose_range_cap()
{
    std::cout << "Test 6: range cap" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    cfg.range_cap = 1.0e4;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    const double r_u = 1.0e10;    // above cap
    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 5.0e3;          // below cap (within range)
    r_lam_sb[1] = 1.0e15;         // above cap
    scaler.Choose(r_u, r_lam_sb);

    AssertNear(scaler.GetDu(), 1.0e4, 1e-8,
               "d_u range cap", "expected 1e4 (clipped)");
    for (int i = 0; i < 3; ++i)
    {
        AssertNear(scaler.GetDLambda()[i], 5.0e3, 1e-8,
                   "d_lambda[" + std::to_string(i) + "] within cap",
                   "expected 5e3");
    }
    for (int i = 3; i < 6; ++i)
    {
        AssertNear(scaler.GetDLambda()[i], 1.0e4, 1e-8,
                   "d_lambda[" + std::to_string(i) + "] above cap",
                   "expected 1e4 (clipped)");
    }

    std::cout << "  PASS  range cap: above-cap norms clipped to cap value"
              << std::endl;
}

//==============================================================================
// Test 7: Apply / Unapply roundtrip is identity
//==============================================================================
void test_apply_unapply_inverse()
{
    std::cout << "Test 7: Apply then Unapply restores original" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    // Non-trivial scaling via Choose: d_u = 3, d_lambda = (2,2,2,7,7,7).
    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 2.0;
    r_lam_sb[1] = 7.0;
    scaler.Choose(3.0, r_lam_sb);

    auto offs = MakeOffsets(4, 6);
    mfem::BlockVector r(offs);
    FillBlockVector(r,
                    {1.0, 2.0, 3.0, 4.0},
                    {10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
    mfem::BlockVector r_orig(r);

    // r → D^-1 r → D D^-1 r = r
    scaler.ApplyToResidual(r);
    scaler.UnapplyToIncrement(r);

    for (int i = 0; i < 4; ++i)
    {
        AssertNear(r.GetBlock(0)[i], r_orig.GetBlock(0)[i], 1e-13,
                   "u[" + std::to_string(i) + "] roundtrip",
                   "Apply-then-Unapply not identity");
    }
    for (int i = 0; i < 6; ++i)
    {
        AssertNear(r.GetBlock(1)[i], r_orig.GetBlock(1)[i], 1e-13,
                   "lambda[" + std::to_string(i) + "] roundtrip",
                   "Apply-then-Unapply not identity");
    }

    std::cout << "  PASS  Apply then Unapply restores original to FP "
              << "precision" << std::endl;
}

//==============================================================================
// Test 8: ApplyToResidual produces D^-1 r with expected values
//==============================================================================
void test_apply_to_residual_values()
{
    std::cout << "Test 8: ApplyToResidual = D^-1 r" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    // d_u = 10; d_lambda = (2, 2, 2, 5, 5, 5).
    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 2.0;
    r_lam_sb[1] = 5.0;
    scaler.Choose(10.0, r_lam_sb);

    auto offs = MakeOffsets(2, 6);
    mfem::BlockVector r(offs);
    FillBlockVector(r,
                    {30.0, 40.0},
                    {6.0, 8.0, 10.0, 25.0, 50.0, 100.0});
    scaler.ApplyToResidual(r);

    // u: each /= 10
    AssertNear(r.GetBlock(0)[0],  3.0, 1e-13, "r_u[0]", "30/10 = 3");
    AssertNear(r.GetBlock(0)[1],  4.0, 1e-13, "r_u[1]", "40/10 = 4");

    // lambda rows 0-2: /= 2; rows 3-5: /= 5
    AssertNear(r.GetBlock(1)[0],  3.0, 1e-13, "r_lam[0]", "6/2 = 3");
    AssertNear(r.GetBlock(1)[1],  4.0, 1e-13, "r_lam[1]", "8/2 = 4");
    AssertNear(r.GetBlock(1)[2],  5.0, 1e-13, "r_lam[2]", "10/2 = 5");
    AssertNear(r.GetBlock(1)[3],  5.0, 1e-13, "r_lam[3]", "25/5 = 5");
    AssertNear(r.GetBlock(1)[4], 10.0, 1e-13, "r_lam[4]", "50/5 = 10");
    AssertNear(r.GetBlock(1)[5], 20.0, 1e-13, "r_lam[5]", "100/5 = 20");

    std::cout << "  PASS  block-wise division produces D^-1 r exactly"
              << std::endl;
}

//==============================================================================
// Test 9: ApplyToIncrement is inverse of UnapplyToIncrement
//==============================================================================
void test_apply_increment_inverse()
{
    std::cout << "Test 9: ApplyToIncrement is inverse of UnapplyToIncrement"
              << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 2.0;
    r_lam_sb[1] = 5.0;
    scaler.Choose(3.0, r_lam_sb);

    auto offs = MakeOffsets(4, 6);
    mfem::BlockVector dx(offs);
    FillBlockVector(dx,
                    {1.0, 2.0, 3.0, 4.0},
                    {10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
    mfem::BlockVector dx_orig(dx);

    // dx → D^-1 dx (apply) → D D^-1 dx = dx (unapply)
    scaler.ApplyToIncrement(dx);
    scaler.UnapplyToIncrement(dx);

    for (int i = 0; i < 4; ++i)
    {
        AssertNear(dx.GetBlock(0)[i], dx_orig.GetBlock(0)[i], 1e-13,
                   "u[" + std::to_string(i) + "] roundtrip",
                   "ApplyToIncrement-then-Unapply not identity");
    }
    for (int i = 0; i < 6; ++i)
    {
        AssertNear(dx.GetBlock(1)[i], dx_orig.GetBlock(1)[i], 1e-13,
                   "lambda[" + std::to_string(i) + "] roundtrip",
                   "ApplyToIncrement-then-Unapply not identity");
    }

    std::cout << "  PASS  ApplyToIncrement followed by Unapply restores "
              << "original" << std::endl;
}

//==============================================================================
// Test 10: ScaledNorm computes ||D^-1 r||_2
//==============================================================================
void test_scaled_norm()
{
    std::cout << "Test 10: ScaledNorm = ||D^-1 r||_2" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 2.0;
    r_lam_sb[1] = 5.0;
    scaler.Choose(10.0, r_lam_sb);

    auto offs = MakeOffsets(2, 6);
    mfem::BlockVector r(offs);
    FillBlockVector(r,
                    {30.0, 40.0},
                    {6.0, 8.0, 10.0, 25.0, 50.0, 100.0});

    // Scaled u   : (3, 4)         → 9 + 16 = 25
    // Scaled lam : (3, 4, 5, 5, 10, 20)  → 9 + 16 + 25 + 25 + 100 + 400 = 575
    // total sum_sq = 600, ScaledNorm = sqrt(600)
    const double sn = scaler.ScaledNorm(r);
    AssertNear(sn, std::sqrt(600.0), 1e-12,
               "ScaledNorm", "expected sqrt(600)");

    std::cout << "  PASS  ScaledNorm = sqrt(600) = "
              << std::sqrt(600.0) << std::endl;
}

//==============================================================================
// Test 11: ScaledBlockNorms decomposes by sub-block
//==============================================================================
void test_scaled_block_norms()
{
    std::cout << "Test 11: ScaledBlockNorms" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 2.0;
    r_lam_sb[1] = 5.0;
    scaler.Choose(10.0, r_lam_sb);

    auto offs = MakeOffsets(2, 6);
    mfem::BlockVector r(offs);
    FillBlockVector(r,
                    {30.0, 40.0},
                    {6.0, 8.0, 10.0, 25.0, 50.0, 100.0});

    double r_u_sc;
    mfem::Vector r_lam_sc;
    scaler.ScaledBlockNorms(r, r_u_sc, r_lam_sc);

    // u scaled: (3, 4), norm = 5
    AssertNear(r_u_sc, 5.0, 1e-12, "r_u_scaled", "expected 5");
    AssertOrDie(r_lam_sc.Size() == 2,
                "r_lam_scaled size", "expected 2");

    // sub-block 0 scaled: (3, 4, 5) → norm = sqrt(9+16+25) = sqrt(50)
    AssertNear(r_lam_sc[0], std::sqrt(50.0), 1e-12,
               "r_lambda_sb0_scaled", "expected sqrt(50)");
    // sub-block 1 scaled: (5, 10, 20) → norm = sqrt(25+100+400) = sqrt(525)
    AssertNear(r_lam_sc[1], std::sqrt(525.0), 1e-12,
               "r_lambda_sb1_scaled", "expected sqrt(525)");

    std::cout << "  PASS  ScaledBlockNorms: r_u_sc = 5; r_lam_sc = "
              << "(sqrt(50), sqrt(525))" << std::endl;
}

//==============================================================================
// Test 12: UnscaledLambdaSubblockNormsSqLocal
//==============================================================================
void test_unscaled_lambda_subblock_norms_sq()
{
    std::cout << "Test 12: UnscaledLambdaSubblockNormsSqLocal" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    mfem::Vector r_lam(6);
    r_lam[0] = 3.0; r_lam[1] = 4.0; r_lam[2] = 0.0;
    r_lam[3] = 5.0; r_lam[4] = 12.0; r_lam[5] = 0.0;

    mfem::Vector norms_sq;
    scaler.UnscaledLambdaSubblockNormsSqLocal(r_lam, norms_sq);

    AssertOrDie(norms_sq.Size() == 2, "norms_sq size", "expected 2");
    // sub-block 0 (rows 0-2): 9 + 16 + 0 = 25
    AssertNear(norms_sq[0], 25.0, 1e-13,
               "norms_sq[0]", "expected 25");
    // sub-block 1 (rows 3-5): 25 + 144 + 0 = 169
    AssertNear(norms_sq[1], 169.0, 1e-13,
               "norms_sq[1]", "expected 169");

    std::cout << "  PASS  per-sub-block sums of squares: 25, 169" << std::endl;
}

//==============================================================================
// Test 13: Reset restores identity scaling, preserves partition
//==============================================================================
void test_reset()
{
    std::cout << "Test 13: Reset" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);

    mfem::Vector r_lam_sb(2);
    r_lam_sb[0] = 3.0;
    r_lam_sb[1] = 5.0;
    scaler.Choose(7.0, r_lam_sb);

    AssertOrDie(scaler.GetDu() == 7.0, "before reset d_u", "expected 7");
    AssertOrDie(scaler.GetDLambda()[0] == 3.0,
                "before reset d_lam[0]", "expected 3");

    scaler.Reset();

    AssertOrDie(scaler.GetDu() == 1.0,
                "after reset d_u", "expected 1");
    for (int i = 0; i < 6; ++i)
    {
        AssertOrDie(scaler.GetDLambda()[i] == 1.0,
                    "after reset d_lambda[" + std::to_string(i) + "]",
                    "expected 1");
    }
    // Partition preserved.
    AssertOrDie(scaler.NumSubblocks() == 2,
                "after reset n_subblocks",
                "expected 2 (partition preserved)");
    AssertOrDie(scaler.GetDLambda().Size() == 6,
                "after reset d_lambda size",
                "expected 6 (partition preserved)");

    std::cout << "  PASS  Reset: factors → 1; partition preserved"
              << std::endl;
}

//==============================================================================
// Test 14: Identity scaling (d_u=1, all d_lambda=1) leaves vectors unchanged
//==============================================================================
void test_identity_scaling_is_noop()
{
    std::cout << "Test 14: identity scaling is no-op" << std::endl;
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    SaddleResidualScaler scaler(cfg);
    SetupTestPartition(scaler);
    // No Choose call — d_u = 1, all d_lambda = 1 from SetPartitionDirect.

    auto offs = MakeOffsets(4, 6);
    mfem::BlockVector r(offs);
    FillBlockVector(r,
                    {1.5, 2.5, 3.5, 4.5},
                    {10.5, 20.5, 30.5, 40.5, 50.5, 60.5});
    mfem::BlockVector r_orig(r);

    scaler.ApplyToResidual(r);

    for (int i = 0; i < 4; ++i)
    {
        AssertNear(r.GetBlock(0)[i], r_orig.GetBlock(0)[i], 1e-14,
                   "u[" + std::to_string(i) + "] under identity",
                   "expected unchanged");
    }
    for (int i = 0; i < 6; ++i)
    {
        AssertNear(r.GetBlock(1)[i], r_orig.GetBlock(1)[i], 1e-14,
                   "lambda[" + std::to_string(i) + "] under identity",
                   "expected unchanged");
    }

    std::cout << "  PASS  identity scaling preserves vector to FP precision"
              << std::endl;
}

//==============================================================================
// Test 15: RebuildPartition from ConstraintBuilder3D (integration test)
//==============================================================================
void test_rebuild_partition_from_builder()
{
    std::cout << "Test 15: RebuildPartition from ConstraintBuilder3D"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.partition = SubblockPartition::FaceEdge;
    SaddleResidualScaler scaler(cfg);

    // --- Full XYZ filter ---
    std::vector<std::string> all_pairs = {"top", "right", "back"};
    std::array<bool, 3> all_comps = {true, true, true};
    scaler.RebuildPartition(builder, all_pairs, all_comps);

    // FaceEdge always emits 2 sub-blocks.
    AssertOrDie(scaler.NumSubblocks() == 2,
                "n_subblocks full XYZ",
                "expected 2 (FaceEdge always emits 2)");
    AssertOrDie(scaler.SubblockLabels()[0] == "edge",
                "labels[0] full XYZ", "expected 'edge'");
    AssertOrDie(scaler.SubblockLabels()[1] == "face",
                "labels[1] full XYZ", "expected 'face'");
    // 2x2x2 mesh unfiltered: 36 lambda rows.
    AssertOrDie(scaler.GetDLambda().Size() == 36,
                "d_lambda size full XYZ",
                "expected 36 (2x2x2 unfiltered row count)");

    // --- Switch to x-only filter ---
    std::vector<std::string> x_only = {"right"};
    scaler.RebuildPartition(builder, x_only, all_comps);

    AssertOrDie(scaler.NumSubblocks() == 2,
                "n_subblocks x-only",
                "FaceEdge always emits 2 labels even when one sub-block "
                "has 0 rows");
    // x-only: 1 face pair × 1 interior × 3 comps = 3 rows.
    AssertOrDie(scaler.GetDLambda().Size() == 3,
                "d_lambda size x-only",
                "expected 3 (1 face pair, 3 comps)");

    std::cout << "  PASS  RebuildPartition handles full and filtered specs"
              << std::endl;
}

}   // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running SaddleResidualScaler unit tests" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    test_constructor_defaults();
    test_set_partition_direct();
    test_choose_per_subblock_off();
    test_choose_per_subblock_on();
    test_choose_floor_guard();
    test_choose_range_cap();
    test_apply_unapply_inverse();
    test_apply_to_residual_values();
    test_apply_increment_inverse();
    test_scaled_norm();
    test_scaled_block_norms();
    test_unscaled_lambda_subblock_norms_sq();
    test_reset();
    test_identity_scaling_is_noop();
    test_rebuild_partition_from_builder();

    if (rank == 0)
    {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "All SaddleResidualScaler tests passed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}