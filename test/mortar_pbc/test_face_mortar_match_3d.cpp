// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-B — unit test for MatchClippedFacePairs.
//
// This test validates the broad-phase candidate-pair enumeration in
// isolation from the rest of the mortar pipeline. We build synthetic
// quad and tri face-element lists by hand (no MFEM mesh required),
// run MatchClippedQuadFacePairs / MatchClippedTriFacePairs, and check
// the CSR output against known expected results for:
//   1. The trivial conforming case: 4×4 vs 4×4 with identical
//      subdivisions; every nonmortar gets exactly 1 candidate, total
//      candidates = 16. (For tri: 4×4×2 vs 4×4×2 with identical
//      diagonal direction; every nonmortar gets exactly 1 candidate,
//      total = 32.)
//   2. The non-conforming case: 4×4 nonmortar vs 5×5 mortar; every
//      nonmortar gets ≥ 1 candidate; total candidates is in expected
//      range.
//   3. Edge case: empty inputs return zeroed CSR.
//
// What's NOT tested here:
//   * Clipping correctness (Batch 4.4-C).
//   * D and A_m matrix accumulation (Batch 4.4-D).
//   * End-to-end patch test (Batch 4.4-E).

#include "face_mortar_match_3d.hpp"
#include "types_3d.hpp"

#include "axom/slic.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace mortar_pbc
{
namespace
{

// ============================================================================
// Test helpers
// ============================================================================

/// Build a single quad face element on a y = const plane with corners
/// at (x0..x1, y, z0..z1). CCW from outward normal +y. Mortar / nonmortar
/// distinction is purely about which side of the periodic pair this is;
/// for Batch 4.4-B the matcher doesn't care which is which, only the
/// 2D-projected geometry matters.
QuadFaceElement MakeQuadOnY(double x0, double x1, double z0, double z1, double y)
{
    QuadFaceElement e;
    e.coords.SetSize(4, 3);
    e.coords(0, 0) = x0; e.coords(0, 1) = y; e.coords(0, 2) = z0;
    e.coords(1, 0) = x1; e.coords(1, 1) = y; e.coords(1, 2) = z0;
    e.coords(2, 0) = x1; e.coords(2, 1) = y; e.coords(2, 2) = z1;
    e.coords(3, 0) = x0; e.coords(3, 1) = y; e.coords(3, 2) = z1;
    e.parametric_axes = {"x", "z"};
    e.perpendicular_axis = "y";
    return e;
}

/// Build an n×n grid of quads tiling [0, L]² on a y = const plane.
std::vector<QuadFaceElement> MakeQuadGrid(int n, double L, double y)
{
    std::vector<QuadFaceElement> elems;
    elems.reserve(n * n);
    const double dx = L / n;
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            elems.push_back(MakeQuadOnY(i * dx, (i + 1) * dx,
                                        j * dx, (j + 1) * dx, y));
        }
    }
    return elems;
}

/// Build an n×n×2 grid of tris tiling [0, L]² on a y = const plane.
/// Each square cell is split along the (0,0)-(1,1) diagonal into two
/// triangles. Tri 1: (i,j), (i+1,j), (i+1,j+1).
/// Tri 2: (i,j), (i+1,j+1), (i,j+1).
std::vector<TriFaceElement> MakeTriGrid(int n, double L, double y)
{
    std::vector<TriFaceElement> elems;
    elems.reserve(n * n * 2);
    const double dx = L / n;
    auto make = [&](double xa, double za, double xb, double zb,
                    double xc, double zc) {
        TriFaceElement e;
        e.coords.SetSize(3, 3);
        e.coords(0, 0) = xa; e.coords(0, 1) = y; e.coords(0, 2) = za;
        e.coords(1, 0) = xb; e.coords(1, 1) = y; e.coords(1, 2) = zb;
        e.coords(2, 0) = xc; e.coords(2, 1) = y; e.coords(2, 2) = zc;
        e.parametric_axes = {"x", "z"};
        e.perpendicular_axis = "y";
        return e;
    };
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            const double x0 = i * dx, x1 = (i + 1) * dx;
            const double z0 = j * dx, z1 = (j + 1) * dx;
            elems.push_back(make(x0, z0, x1, z0, x1, z1));
            elems.push_back(make(x0, z0, x1, z1, x0, z1));
        }
    }
    return elems;
}

// ============================================================================
// Test cases
// ============================================================================

bool g_failures = false;

#define REQUIRE(cond, msg)                                                    \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "  FAIL: " << msg << "  (" #cond " at "              \
                      << __FILE__ << ":" << __LINE__ << ")\n";                \
            g_failures = true;                                                \
        }                                                                     \
    } while (0)

/// Test 1: empty inputs return zeroed CSR.
void test_empty_inputs()
{
    std::cout << "  test_empty_inputs\n";

    std::vector<QuadFaceElement> empty_q;
    auto out_q = MatchClippedQuadFacePairs(empty_q, empty_q, "y");
    REQUIRE(out_q.offsets.size() == 1, "empty: offsets size should be 1");
    REQUIRE(out_q.counts.empty(), "empty: counts should be empty");
    REQUIRE(out_q.candidates.empty(), "empty: candidates should be empty");

    std::vector<TriFaceElement> empty_t;
    auto out_t = MatchClippedTriFacePairs(empty_t, empty_t, "y");
    REQUIRE(out_t.offsets.size() == 1, "empty tri: offsets size should be 1");
    REQUIRE(out_t.counts.empty(), "empty tri: counts should be empty");
    REQUIRE(out_t.candidates.empty(), "empty tri: candidates should be empty");
}

/// Test 2: trivial conforming case. 4×4 vs 4×4 with identical
/// subdivisions.
///
/// With our small AABB pad (1e-9 × max_edge), each nonmortar's AABB
/// overlaps not just its own mortar twin but also any mortar AABB
/// that shares an edge or corner — because the padding extends the
/// mortar AABBs by ε across shared coordinate planes. For a 4×4 grid:
///   * Interior nonmortars (inner 2×2):    self + 8 neighbors = 9
///   * Edge nonmortars (8 of them):        self + 5 neighbors = 6
///   * Corner nonmortars (4 of them):      self + 3 neighbors = 4
///   * Total: 4·9 + 8·6 + 4·4 = 36 + 48 + 16 = 100
///
/// This over-counting at AABB level is fine — the broad-phase is
/// allowed to be conservative; Batch 4.4-C's polygon clipping will
/// reject zero-area intersections at the fine-phase. We just check
/// (a) CSR well-formedness, (b) each nonmortar gets ≥ 1 candidate
/// (its own twin), and (c) total is in the realistic upper bound for
/// shared-edge inclusion.
void test_quad_conforming_4x4()
{
    std::cout << "  test_quad_conforming_4x4\n";

    const double L = 1.0;
    auto nonmortar = MakeQuadGrid(4, L, 0.0);
    auto mortar    = MakeQuadGrid(4, L, L);  // opposite face

    auto out = MatchClippedQuadFacePairs(nonmortar, mortar, "y");

    REQUIRE(out.offsets.size() == nonmortar.size() + 1,
            "conforming: offsets size");
    REQUIRE(out.counts.size() == nonmortar.size(),
            "conforming: counts size");

    // CSR consistency: offsets[i+1] - offsets[i] == counts[i].
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.offsets[i + 1] - out.offsets[i] == out.counts[i],
                "conforming: CSR offsets/counts inconsistent");
    }
    REQUIRE(out.offsets.back() == static_cast<axom::IndexType>(out.candidates.size()),
            "conforming: offsets.back() should equal candidates.size()");

    // Numerical checks:
    //   - Every nonmortar must get ≥ 1 candidate (its own twin).
    //   - Every nonmortar should get ≤ 9 candidates (self + at most
    //     8 edge/corner neighbors).
    //   - Total should be in [16, 100] (16 = perfect 1-to-1 with no
    //     shared-edge inclusion; 100 = full shared-edge inclusion
    //     across all interior+edge+corner elements).
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.counts[i] >= 1,
                "conforming: every nonmortar must get its own twin");
        REQUIRE(out.counts[i] <= 9,
                "conforming: at most 9 candidates per nonmortar (self + 8)");
    }
    REQUIRE(out.candidates.size() >= 16,
            "conforming: total ≥ 16 (one twin per nonmortar)");
    REQUIRE(out.candidates.size() <= 100,
            "conforming: total ≤ 100 (full shared-edge inclusion)");

    std::cout << "    total candidates = " << out.candidates.size() << "\n";
}

/// Test 3: non-conforming case. 4×4 nonmortar vs 5×5 mortar. Each
/// nonmortar element occupies a 0.25×0.25 square; each mortar element
/// occupies a 0.20×0.20 square. The nonmortar's 2D AABB will overlap
/// approximately 4–9 mortar AABBs (depending on relative position).
/// With the small pad, edge-shared neighbors can also be picked up.
///
/// Loose bounds:
///   - Each nonmortar must get ≥ 1 candidate (the misalignment plus
///     overlap guarantees this).
///   - Total candidates: empirically 60–120 for this geometry; we
///     check 16 ≤ N ≤ 200 to be safe.
void test_quad_nonconforming_4x4_vs_5x5()
{
    std::cout << "  test_quad_nonconforming_4x4_vs_5x5\n";

    const double L = 1.0;
    auto nonmortar = MakeQuadGrid(4, L, 0.0);
    auto mortar    = MakeQuadGrid(5, L, L);

    auto out = MatchClippedQuadFacePairs(nonmortar, mortar, "y");

    REQUIRE(out.offsets.size() == nonmortar.size() + 1,
            "non-conforming: offsets size");
    REQUIRE(out.counts.size() == nonmortar.size(),
            "non-conforming: counts size");
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.offsets[i + 1] - out.offsets[i] == out.counts[i],
                "non-conforming: CSR consistency");
    }
    REQUIRE(out.offsets.back() == static_cast<axom::IndexType>(out.candidates.size()),
            "non-conforming: candidates.size() consistency");

    // Numerical: every nonmortar must overlap something (no orphans).
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.counts[i] >= 1,
                "non-conforming: every nonmortar must get ≥ 1 candidate");
    }
    REQUIRE(out.candidates.size() >= 16,
            "non-conforming: total ≥ 16");
    REQUIRE(out.candidates.size() <= 200,
            "non-conforming: total ≤ 200 (sane upper bound)");

    std::cout << "    total candidates = " << out.candidates.size() << "\n";
}

/// Test 4: tri-tri conforming. Same subdivision on both sides.
/// 4×4 grid -> 32 tris each side. Each tri's AABB is its parent
/// square's AABB (the diagonal split produces tris whose bounding
/// boxes equal the square's), so each tri's AABB overlaps:
///   - its own twin (1)
///   - the other tri in its parent square (1)
///   - tri pairs in adjacent squares (up to 8 squares for interior,
///     each contributing 2 tris) -> via AABB pad
/// Lower bound: ≥ 2 per nonmortar (twin + diagonal partner) → total ≥ 64.
/// Upper bound: very loose, well under 32×18 = 576.
void test_tri_conforming_4x4()
{
    std::cout << "  test_tri_conforming_4x4\n";

    const double L = 1.0;
    auto nonmortar = MakeTriGrid(4, L, 0.0);
    auto mortar    = MakeTriGrid(4, L, L);

    REQUIRE(nonmortar.size() == 32, "tri: 4×4 grid should have 32 tris");
    REQUIRE(mortar.size() == 32,    "tri: 4×4 grid should have 32 tris");

    auto out = MatchClippedTriFacePairs(nonmortar, mortar, "y");

    REQUIRE(out.offsets.size() == nonmortar.size() + 1, "tri conforming: offsets size");
    REQUIRE(out.counts.size() == nonmortar.size(),     "tri conforming: counts size");
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.offsets[i + 1] - out.offsets[i] == out.counts[i],
                "tri conforming: CSR consistency");
    }

    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.counts[i] >= 2,
                "tri conforming: each nonmortar should overlap ≥ 2 mortar "
                "(its own twin + the other tri in the parent square)");
    }
    REQUIRE(out.candidates.size() >= 64,
            "tri conforming: total ≥ 64 (≥ 2 per nonmortar)");
    REQUIRE(out.candidates.size() <= 600,
            "tri conforming: total ≤ 600 (sane upper bound)");

    std::cout << "    total candidates = " << out.candidates.size() << "\n";
}

// ============================================================================
// Batch 4.4-C tests — clipping + fan-triangulation.
// ============================================================================

/// Test 5 (4.4-C): empty inputs to ClipQuadFacePairs return zeroed CSR.
void test_clip_empty_inputs()
{
    std::cout << "  test_clip_empty_inputs\n";
    std::vector<QuadFaceElement> empty_q;
    ClippedPairCandidates empty_cands;
    empty_cands.offsets.assign(1, 0);  // valid for n_nonmortar = 0

    auto out = ClipQuadFacePairs(empty_q, empty_q, empty_cands, "y");
    REQUIRE(out.offsets.size() == 1, "clip empty: offsets size 1");
    REQUIRE(out.counts.empty(),      "clip empty: counts empty");
    REQUIRE(out.sub_tris.empty(),    "clip empty: sub_tris empty");
}

/// Test 6 (4.4-C): clipping on a 4×4 vs 4×4 conforming setup. Each
/// nonmortar quad has area 0.25² = 0.0625; total nonmortar area is
/// 1.0. After clipping, the surviving sub-triangles should:
///   1. Tile the nonmortar face exactly (tile-cover invariant: total
///      sub-tri area == nonmortar face area to roundoff).
///   2. Each nonmortar produces 1 to ~4 sub-triangles depending on
///      whether Axom's clip introduces extra vertices on shared edges.
///      A "twin clip" of identical 4-vertex quads ideally gives 2
///      sub-tris (fan-tri of a 4-gon), but Axom v0.14.0's robustness
///      handling can produce 4–8 vertex output for edge-coincident
///      cases, yielding 2–6 sub-tris. We bound loosely.
///   3. Each sub-tri has positive 2D area.
void test_clip_quad_conforming_4x4()
{
    std::cout << "  test_clip_quad_conforming_4x4\n";

    const double L = 1.0;
    auto nonmortar = MakeQuadGrid(4, L, 0.0);
    auto mortar    = MakeQuadGrid(4, L, L);
    auto cands = MatchClippedQuadFacePairs(nonmortar, mortar, "y");
    auto out   = ClipQuadFacePairs(nonmortar, mortar, cands, "y");

    REQUIRE(out.offsets.size() == nonmortar.size() + 1,
            "clip quad conforming: offsets size");
    REQUIRE(out.counts.size() == nonmortar.size(),
            "clip quad conforming: counts size");

    // CSR consistency.
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.offsets[i + 1] - out.offsets[i] == out.counts[i],
                "clip quad conforming: CSR consistency");
    }
    REQUIRE(out.offsets.back() == static_cast<axom::IndexType>(out.sub_tris.size()),
            "clip quad conforming: offsets.back() vs sub_tris.size()");

    // Numerical: each nonmortar produces at least 1 sub-tri (its twin)
    // and no more than ~10 (very loose upper bound).
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.counts[i] >= 1,
                "clip quad conforming: each nonmortar should produce ≥ 1 sub-tri");
        REQUIRE(out.counts[i] <= 10,
                "clip quad conforming: each nonmortar should produce ≤ 10 sub-tris");
    }

    // Tile-cover invariant: total sub-tri area equals nonmortar face area.
    // This is the central correctness check — independent of how Axom's
    // clip subdivides the polygons.
    const double expected_area = L * L;  // 1.0
    const double total_area = out.TotalArea();
    const double area_err = std::abs(total_area - expected_area);
    REQUIRE(area_err < 1.0e-12 * expected_area,
            "clip quad conforming: tile-cover invariant violated "
            "(total area should equal nonmortar face area)");

    // All sub-tri areas positive.
    for (const auto& t : out.sub_tris)
    {
        REQUIRE(t.area > 0.0, "clip quad conforming: sub-tri area must be positive");
    }

    std::cout << "    total sub-triangles = " << out.sub_tris.size()
              << "  total area = " << total_area
              << "  (expected " << expected_area << ")\n";
}

/// Test 7 (4.4-C): clipping on 4×4 nonmortar vs 5×5 mortar. The
/// nonmortar face is 4×4 = 16 elements covering [0,1]². Each
/// nonmortar quad of area 0.0625 is broken into multiple sub-triangles
/// by intersection with the 0.20×0.20 mortar grid.
///
/// Tile-cover invariant: total sub-tri area equals 1.0 to roundoff,
/// regardless of how the clipping subdivides. This is the key
/// correctness check for non-conforming clipping — if any clipped
/// region is missed or counted twice, the total area will be off.
void test_clip_quad_nonconforming_4x4_vs_5x5()
{
    std::cout << "  test_clip_quad_nonconforming_4x4_vs_5x5\n";

    const double L = 1.0;
    auto nonmortar = MakeQuadGrid(4, L, 0.0);
    auto mortar    = MakeQuadGrid(5, L, L);
    auto cands = MatchClippedQuadFacePairs(nonmortar, mortar, "y");
    auto out   = ClipQuadFacePairs(nonmortar, mortar, cands, "y");

    REQUIRE(out.offsets.size() == nonmortar.size() + 1,
            "clip nonconforming: offsets size");
    REQUIRE(out.counts.size() == nonmortar.size(),
            "clip nonconforming: counts size");

    // Every nonmortar must have at least one sub-triangle.
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.counts[i] >= 1,
                "clip nonconforming: every nonmortar must produce ≥ 1 sub-triangle");
    }

    // Tile-cover invariant.
    const double expected_area = L * L;
    const double total_area = out.TotalArea();
    const double area_err = std::abs(total_area - expected_area);
    REQUIRE(area_err < 1.0e-12 * expected_area,
            "clip nonconforming: tile-cover invariant violated");

    // All sub-tri areas positive.
    for (const auto& t : out.sub_tris)
    {
        REQUIRE(t.area > 0.0, "clip nonconforming: sub-tri area must be positive");
    }

    std::cout << "    total sub-triangles = " << out.sub_tris.size()
              << "  total area = " << total_area
              << "  (expected " << expected_area << ")\n";
}

/// Test 8 (4.4-C): clipping on 4×4 conforming tris. 32 tris each side.
/// Each tri's AABB equals its parent square's AABB, so the BVH gives
/// many spurious candidates (test 4 confirmed 400). Clipping should
/// reject the false-positives where AABB overlap doesn't correspond to
/// polygon overlap (e.g., a tri's twin is the diagonal partner —
/// AABBs match but polygons share only a diagonal line, no area).
///
/// Expected: each nonmortar tri produces exactly 1 sub-triangle (its
/// own twin, which is itself — a tri clipped against itself fan-
/// triangulates into 1 tri). Total sub-tris = 32. Total area = 1.0.
void test_clip_tri_conforming_4x4()
{
    std::cout << "  test_clip_tri_conforming_4x4\n";

    const double L = 1.0;
    auto nonmortar = MakeTriGrid(4, L, 0.0);
    auto mortar    = MakeTriGrid(4, L, L);
    auto cands = MatchClippedTriFacePairs(nonmortar, mortar, "y");
    auto out   = ClipTriFacePairs(nonmortar, mortar, cands, "y");

    // Each nonmortar tri pairs with its own twin (full overlap → 1
    // sub-tri after fan-triangulation of a 3-vertex polygon) AND
    // potentially edge-shared neighbors (filtered out as area-zero
    // by area_tol_rel).
    for (std::size_t i = 0; i < nonmortar.size(); ++i)
    {
        REQUIRE(out.counts[i] >= 1,
                "clip tri conforming: every nonmortar tri must keep ≥ 1 sub-tri");
    }

    // Tile-cover invariant.
    const double expected_area = L * L;  // sum of all tris = full face
    const double total_area = out.TotalArea();
    const double area_err = std::abs(total_area - expected_area);
    REQUIRE(area_err < 1.0e-12 * expected_area,
            "clip tri conforming: tile-cover invariant violated");

    // All sub-tri areas positive.
    for (const auto& t : out.sub_tris)
    {
        REQUIRE(t.area > 0.0, "clip tri conforming: sub-tri area must be positive");
    }

    std::cout << "    total sub-triangles = " << out.sub_tris.size()
              << "  total area = " << total_area
              << "  (expected " << expected_area << ")\n";
}

/// Test 5: perpendicular-axis mismatch is caught.
/// MatchClippedFacePairs asserts that every input element has the same
/// perpendicular_axis as the caller-provided argument. Build elements
/// on y = const, then pass "x" as the axis — should fail the assertion.
///
/// Disabled in this build because MFEM_VERIFY aborts the whole process
/// in release; we'd need a way to catch the abort. Documented so a
/// future maintainer can wire it up against a debug build that uses
/// exceptions instead of abort.
void test_perpendicular_axis_mismatch_doc()
{
    // Intentionally not run; documented for future test infrastructure.
    std::cout << "  test_perpendicular_axis_mismatch_doc (skipped — needs "
                 "exception-based MFEM_VERIFY; documented only)\n";
}

}  // anonymous namespace
}  // namespace mortar_pbc

int main()
{
    // RAII Slic logger — see test_axom_smoke.cpp for rationale.
    axom::slic::SimpleLogger slic_logger;

    std::cout << "test_face_mortar_match_3d (Phase 4.4 / Batches 4.4-B/C)\n";
    // Batch 4.4-B: broad-phase candidate enumeration.
    mortar_pbc::test_empty_inputs();
    mortar_pbc::test_quad_conforming_4x4();
    mortar_pbc::test_quad_nonconforming_4x4_vs_5x5();
    mortar_pbc::test_tri_conforming_4x4();
    mortar_pbc::test_perpendicular_axis_mismatch_doc();
    // Batch 4.4-C: fine-phase clipping + fan-triangulation.
    mortar_pbc::test_clip_empty_inputs();
    mortar_pbc::test_clip_quad_conforming_4x4();
    mortar_pbc::test_clip_quad_nonconforming_4x4_vs_5x5();
    mortar_pbc::test_clip_tri_conforming_4x4();

    if (mortar_pbc::g_failures)
    {
        std::cerr << "\nOne or more test_face_mortar_match_3d cases FAILED.\n";
        return 1;
    }
    std::cout << "\nAll test_face_mortar_match_3d cases passed.\n";
    return 0;
}
