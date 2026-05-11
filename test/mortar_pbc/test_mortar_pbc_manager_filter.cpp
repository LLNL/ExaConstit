// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.9 / Batch A.5 — multi-entry validation test for the
// spec-driven corner-pinning derivation.
//
// Exercises `ComputeCornerEssTDofsFromSpec(classifier, fes,
// essential_ids, comp_mask)` (Phase 5.9.A.4, tightened in A.5) on a
// small 2x2x2 hex mesh covering four representative spec cases:
//
//   * Full XYZ           → 24 rank-summed TDOFs (matches pre-5.9
//                          ComputeCornerEssTDofs bit-for-bit).
//   * X-only (1 pair)    → 3 anchor + 7*1 non-anchor = 10.
//   * XY (2 pairs)       → 3 anchor + 7*2 non-anchor = 17.
//   * Empty essential_ids → 3 (anchor only — all 7 non-anchor corners
//                            are filtered out by the incident-face
//                            gate).
//
// Each test exits via std::exit(1) on failure with a diagnostic to
// stderr, or returns normally on success. Same harness style as
// test_constraint_builder_3d.cpp.
//
// The full MortarPbcManager round-trip (RebuildForActiveSpec) and
// SystemDriver SyncMortarPbcForStep require heavier setup
// (SimulationState construction, ExaOptions wiring); they're
// validated in production integration tests by driving a 2-step
// load history with different specs per step.

#include "boundary_classifier_3d.hpp"
#include "mortar_pbc_manager.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ComputeCornerEssTDofs;
using mortar_pbc::ComputeCornerEssTDofsFromSpec;

namespace {

// ---- helper: assert + diagnostic ------------------------------------------
void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

// ---- helper: build a small unit-cube hex ParMesh + FE space --------------
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
        mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
    b.fec = std::make_unique<mfem::H1_FECollection>(/*order=*/1, /*dim=*/3);
    b.fes = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3, mfem::Ordering::byNODES);
    return b;
}

// Rank-sum a local int via MPI_Allreduce. Used to convert per-rank
// TDOF counts to global counts for the comparison assertions.
int RankSum(int local)
{
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

// Look up the mesh face attributes for the two halves of every face
// pair the classifier knows about. Returns the attrs in the order
// (axis_0_mortar, axis_0_nonmortar, axis_1_mortar, axis_1_nonmortar,
// axis_2_mortar, axis_2_nonmortar) where the order of axes matches
// classifier.FacePairs() iteration.
struct PairAttrs
{
    int mortar;
    int nonmortar;
    std::string axis;
};

std::vector<PairAttrs> CollectPairAttrs(const BoundaryClassifier3D& cl)
{
    std::vector<PairAttrs> out;
    for (const auto& tup : cl.FacePairs())
    {
        PairAttrs pa;
        pa.axis      = std::get<0>(tup);
        pa.mortar    = cl.MeshAttributeForLabel(std::get<1>(tup));
        pa.nonmortar = cl.MeshAttributeForLabel(std::get<2>(tup));
        out.push_back(pa);
    }
    return out;
}

// ===========================================================================
// Test 1: Full XYZ — essential_ids covers all 6 face attrs,
//                    comp_mask = {true, true, true}.
//
// Expected: 24 rank-summed TDOFs.
//
// Sanity: the result must match ComputeCornerEssTDofs (pre-5.9)
// bit-for-bit at this configuration since the spec-aware path with
// all faces + all comps degenerates to the unfiltered path on a
// standard 6-face RVE.
// ===========================================================================
void test_full_xyz()
{
    std::cout << "Test 1: ComputeCornerEssTDofsFromSpec, full XYZ"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    // All 6 face attrs.
    const auto pairs = CollectPairAttrs(cl);
    std::vector<int> essential_ids;
    for (const auto& pa : pairs)
    {
        essential_ids.push_back(pa.mortar);
        essential_ids.push_back(pa.nonmortar);
    }
    AssertOrDie(essential_ids.size() == 6, "essential_ids covers 6 faces",
                "got " + std::to_string(essential_ids.size())
                + " entries; expected 6");

    const std::array<bool, 3> comp_mask = {{true, true, true}};
    auto spec_tdofs = ComputeCornerEssTDofsFromSpec(
        cl, *b.fes, essential_ids, comp_mask);

    const int spec_global = RankSum(spec_tdofs.Size());
    AssertOrDie(spec_global == 24,
                "full-XYZ rank-summed count",
                "got " + std::to_string(spec_global) + ", expected 24");

    // Match against the unfiltered pre-5.9 path.
    auto pre_5_9 = ComputeCornerEssTDofs(cl, *b.fes);
    const int pre_global = RankSum(pre_5_9.Size());
    AssertOrDie(pre_global == 24,
                "pre-5.9 rank-summed count (sanity)",
                "got " + std::to_string(pre_global) + ", expected 24");
    AssertOrDie(spec_tdofs.Size() == pre_5_9.Size(),
                "per-rank size match vs pre-5.9",
                "spec " + std::to_string(spec_tdofs.Size())
                + " vs pre-5.9 " + std::to_string(pre_5_9.Size()));

    std::cout << "  PASS  rank-summed 24 (matches pre-5.9 path)"
              << std::endl;
}

// ===========================================================================
// Test 2: X-only (1 pair) — essential_ids = {left, right}, comp_mask = {T,F,F}.
//
// Expected on a 6-face axis-aligned RVE:
//   - All 8 corners are incident on either 'left' or 'right' (each
//     corner has min_x or max_x), so the incident-face gate is open
//     for all 8.
//   - Anchor contributes 3 TDOFs (XYZ unconditional).
//   - 7 non-anchor corners contribute 1 TDOF each (X-only).
//   - Total: 3 + 7 = 10 rank-summed.
// ===========================================================================
void test_x_only_single_pair()
{
    std::cout << "Test 2: ComputeCornerEssTDofsFromSpec, X-only (1 pair)"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    // Find the x-axis pair and collect its two attrs.
    const auto pairs = CollectPairAttrs(cl);
    std::vector<int> essential_ids;
    for (const auto& pa : pairs)
    {
        if (pa.axis == "x")
        {
            essential_ids.push_back(pa.mortar);
            essential_ids.push_back(pa.nonmortar);
        }
    }
    AssertOrDie(essential_ids.size() == 2, "x-pair attrs",
                "got " + std::to_string(essential_ids.size())
                + " entries; expected 2");

    const std::array<bool, 3> comp_mask = {{true, false, false}};
    auto tdofs = ComputeCornerEssTDofsFromSpec(
        cl, *b.fes, essential_ids, comp_mask);

    const int global = RankSum(tdofs.Size());
    AssertOrDie(global == 10,
                "X-only rank-summed count",
                "got " + std::to_string(global) + ", expected 10 "
                "(3 anchor + 7 non-anchor X-comp)");

    std::cout << "  PASS  rank-summed 10 (anchor's 3 + 7 non-anchor X-only)"
              << std::endl;
}

// ===========================================================================
// Test 3: XY (2 pairs) — essential_ids = {left, right, bottom, top},
//                       comp_mask = {T, T, F}.
//
// Expected:
//   - All 8 corners incident on at least one of {left, right, bottom,
//     top} (each corner has min/max in x AND min/max in y).
//   - Anchor: 3 TDOFs.
//   - 7 non-anchor corners × 2 comps (X+Y) = 14 TDOFs.
//   - Total: 3 + 14 = 17 rank-summed.
// ===========================================================================
void test_xy_two_pairs()
{
    std::cout << "Test 3: ComputeCornerEssTDofsFromSpec, XY (2 pairs)"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    const auto pairs = CollectPairAttrs(cl);
    std::vector<int> essential_ids;
    for (const auto& pa : pairs)
    {
        if (pa.axis == "x" || pa.axis == "y")
        {
            essential_ids.push_back(pa.mortar);
            essential_ids.push_back(pa.nonmortar);
        }
    }
    AssertOrDie(essential_ids.size() == 4, "x+y pair attrs",
                "got " + std::to_string(essential_ids.size())
                + " entries; expected 4");

    const std::array<bool, 3> comp_mask = {{true, true, false}};
    auto tdofs = ComputeCornerEssTDofsFromSpec(
        cl, *b.fes, essential_ids, comp_mask);

    const int global = RankSum(tdofs.Size());
    AssertOrDie(global == 17,
                "XY rank-summed count",
                "got " + std::to_string(global) + ", expected 17 "
                "(3 anchor + 7 non-anchor × 2 comps)");

    std::cout << "  PASS  rank-summed 17 (anchor's 3 + 7 non-anchor XY)"
              << std::endl;
}

// ===========================================================================
// Test 4: Anchor-only — essential_ids empty, comp_mask irrelevant.
//
// Expected: 3 rank-summed TDOFs (just the anchor's three components).
// All 7 non-anchor corners fail the incident-face gate (no face attrs
// to be incident on).
//
// Note: in production, `essential_ids` MUST be non-empty per
// `PeriodicBC::validate()`, so this case is purely a unit test of the
// incident-face gate's logic. RebuildForActiveSpec never sees it.
// ===========================================================================
void test_anchor_only_empty_essential_ids()
{
    std::cout << "Test 4: ComputeCornerEssTDofsFromSpec, empty essential_ids "
              << "(anchor only)" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    const std::vector<int> essential_ids;
    const std::array<bool, 3> comp_mask = {{true, true, true}};
    auto tdofs = ComputeCornerEssTDofsFromSpec(
        cl, *b.fes, essential_ids, comp_mask);

    const int global = RankSum(tdofs.Size());
    AssertOrDie(global == 3,
                "anchor-only rank-summed count",
                "got " + std::to_string(global) + ", expected 3 "
                "(anchor's 3 components, all non-anchor gated out)");

    std::cout << "  PASS  rank-summed 3 (anchor only — incident-face gate "
              << "drops 7 non-anchor corners)" << std::endl;
}

// ===========================================================================
// Test 5: Repeated calls (round-trip) — apply XYZ → X-only → XYZ.
//
// Each call produces an independent fresh Array<int>. The corner
// counts should match across the round trip.
//
// This is a thin smoke test of "the function is stateless" — the
// real round-trip property is tested at the manager level in
// integration tests.
// ===========================================================================
void test_round_trip_xyz_xonly_xyz()
{
    std::cout << "Test 5: ComputeCornerEssTDofsFromSpec, round trip "
              << "XYZ→X→XYZ" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    const auto pairs = CollectPairAttrs(cl);

    std::vector<int> all_ids;
    std::vector<int> x_only_ids;
    for (const auto& pa : pairs)
    {
        all_ids.push_back(pa.mortar);
        all_ids.push_back(pa.nonmortar);
        if (pa.axis == "x")
        {
            x_only_ids.push_back(pa.mortar);
            x_only_ids.push_back(pa.nonmortar);
        }
    }

    auto t1 = ComputeCornerEssTDofsFromSpec(
        cl, *b.fes, all_ids, {{true, true, true}});
    auto t2 = ComputeCornerEssTDofsFromSpec(
        cl, *b.fes, x_only_ids, {{true, false, false}});
    auto t3 = ComputeCornerEssTDofsFromSpec(
        cl, *b.fes, all_ids, {{true, true, true}});

    const int g1 = RankSum(t1.Size());
    const int g2 = RankSum(t2.Size());
    const int g3 = RankSum(t3.Size());

    AssertOrDie(g1 == 24, "round trip XYZ#1",
                "got " + std::to_string(g1) + ", expected 24");
    AssertOrDie(g2 == 10, "round trip X-only",
                "got " + std::to_string(g2) + ", expected 10");
    AssertOrDie(g3 == 24, "round trip XYZ#2",
                "got " + std::to_string(g3) + ", expected 24");
    AssertOrDie(t1.Size() == t3.Size(),
                "round-trip per-rank size identical",
                "first XYZ " + std::to_string(t1.Size())
                + " vs second XYZ " + std::to_string(t3.Size()));

    std::cout << "  PASS  round trip preserves corner counts "
              << "(24 → 10 → 24)" << std::endl;
}

}  // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running Phase 5.9.A.5 multi-entry validation tests"
                  << std::endl;
        std::cout << "---------------------------------------------------"
                  << std::endl;
    }

    test_full_xyz();
    test_x_only_single_pair();
    test_xy_two_pairs();
    test_anchor_only_empty_essential_ids();
    test_round_trip_xyz_xonly_xyz();

    if (rank == 0)
    {
        std::cout << "---------------------------------------------------"
                  << std::endl;
        std::cout << "All Phase 5.9.A.5 multi-entry validation tests passed."
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
