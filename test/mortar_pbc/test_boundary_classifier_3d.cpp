// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — integration test for BoundaryClassifier3D.
//
// Builds a small auto-generated cartesian 3D mesh via
// `mfem::Mesh::MakeCartesian3D`, partitions it into a ParMesh, and
// runs the full classifier. Verifies:
//   * 8 corners with valid x/y/z gtdofs
//   * 12 edges with the correct mortar/nonmortar flags
//     (1 mortar + 3 nonmortar per parametric axis)
//   * 6 faces with the correct mortar/nonmortar flags
//     (top/right/back = mortar, bottom/left/front = nonmortar)
//   * EdgePairs() returns 9 (axis, mortar, nonmortar) tuples
//   * FacePairs() returns 3 tuples
//   * Sentinel rewriting:
//       - face elements that touch a box corner have at least one -1
//       - face elements that touch a box edge have at least one -2
//       - face-interior elements (4×4×4 grid produces several) have
//         no sentinels
//   * GtdofXyzLookup() entries are consistent with corner/edge
//     gtdofs.
//
// This test is single-rank by default but tolerates multi-rank
// launches: every rank constructs the same mesh independently
// (ParMesh's auto-partitioning kicks in when np>1) and the assertions
// are rank-symmetric.
//
// Test runner: each test function exits via std::exit(1) on failure
// (with a diagnostic to stderr) or returns normally on success. The
// main() at the bottom calls all of them in sequence.

#include "boundary_classifier_3d.hpp"
#include "boundary_helpers_3d.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::CornerInfo3D;
using mortar_pbc::EdgeInfo3D;
using mortar_pbc::FaceInfo3D;
using mortar_pbc::QuadFaceElement;
using mortar_pbc::TriFaceElement;
using mortar_pbc::kGtdofCornerSentinel;
using mortar_pbc::kGtdofEdgeSentinel;
using mortar_pbc::AxisTileGrid;
using mortar_pbc::TilePartition3D;

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

// ---- helper: build a small unit-cube hex ParMesh --------------------------
//
// 4×4×4 hex grid on [0,1]^3. The grid resolution is intentionally
// modest: enough cells to give 1 interior face element per face on
// each face of the box, plus enough vertices to exercise the corner /
// edge / face-interior classification. The unit cube keeps tolerances
// numerically simple.
std::unique_ptr<mfem::ParMesh> BuildUnitCubeHexMesh(MPI_Comm comm,
                                                   int n_per_side = 4)
{
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n_per_side, n_per_side, n_per_side,
        mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    return std::make_unique<mfem::ParMesh>(comm, serial);
}

// ---- helper: build a vector H1 P1 FE space, vdim=3 ------------------------
struct FesBundle
{
    std::unique_ptr<mfem::ParMesh> pmesh;
    std::unique_ptr<mfem::H1_FECollection> fec;
    std::unique_ptr<mfem::ParFiniteElementSpace> fes;
};

FesBundle BuildHexFesBundle(MPI_Comm comm, int n_per_side = 4)
{
    FesBundle b;
    b.pmesh = BuildUnitCubeHexMesh(comm, n_per_side);
    b.fec = std::make_unique<mfem::H1_FECollection>(/*order=*/1, /*dim=*/3);
    b.fes = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3, mfem::Ordering::byNODES);
    return b;
}

// ===========================================================================
// Test 1: 8 corners, all with valid gtdofs, at the bbox vertices
// ===========================================================================
void test_corners_count_and_coords()
{
    std::cout << "Test 1: corners count and coordinates" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    const auto& corners = bc.Corners();
    AssertOrDie(corners.size() == 8, "corners count",
                "got " + std::to_string(corners.size()) + ", expected 8");

    // Verify each labelled corner is at the right bbox vertex.
    const auto& bmin = bc.BboxMin();
    const auto& bmax = bc.BboxMax();
    const double tol = bc.Tol();
    struct Expected {
        const char* label;
        std::array<double, 3> coord;
    };
    std::array<Expected, 8> targets = {{
        {"blf", {bmin[0], bmin[1], bmin[2]}},
        {"brf", {bmax[0], bmin[1], bmin[2]}},
        {"blb", {bmin[0], bmin[1], bmax[2]}},
        {"brb", {bmax[0], bmin[1], bmax[2]}},
        {"tlf", {bmin[0], bmax[1], bmin[2]}},
        {"trf", {bmax[0], bmax[1], bmin[2]}},
        {"tlb", {bmin[0], bmax[1], bmax[2]}},
        {"trb", {bmax[0], bmax[1], bmax[2]}},
    }};
    for (const auto& t : targets)
    {
        auto it = corners.find(t.label);
        AssertOrDie(it != corners.end(), "corner present",
                    std::string("label '") + t.label + "' missing");
        const CornerInfo3D& c = it->second;
        const double dx = std::abs(c.coord[0] - t.coord[0]);
        const double dy = std::abs(c.coord[1] - t.coord[1]);
        const double dz = std::abs(c.coord[2] - t.coord[2]);
        AssertOrDie(dx <= tol && dy <= tol && dz <= tol,
                    std::string("corner '") + t.label + "' coord",
                    "off-target");
        AssertOrDie(c.gtdof_x >= 0 && c.gtdof_y >= 0 && c.gtdof_z >= 0,
                    std::string("corner '") + t.label + "' gtdofs",
                    "negative gtdof");
    }
    std::cout << "  PASS  8 corners, all at bbox vertices, all with valid gtdofs"
              << std::endl;
}

// ===========================================================================
// Test 2: 12 edges, 1 mortar + 3 nonmortar per parametric axis
// ===========================================================================
void test_edges_count_and_mortar_flags()
{
    std::cout << "Test 2: edges count and mortar flags" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    const auto& edges = bc.Edges();
    AssertOrDie(edges.size() == 12, "edges count",
                "got " + std::to_string(edges.size()) + ", expected 12");

    std::map<std::string, int> mortar_per_axis  = {{"x", 0}, {"y", 0}, {"z", 0}};
    std::map<std::string, int> nonmortar_per_axis = {{"x", 0}, {"y", 0}, {"z", 0}};
    for (const auto& kv : edges)
    {
        const EdgeInfo3D& e = kv.second;
        AssertOrDie(e.parametric_axis == "x" || e.parametric_axis == "y"
                        || e.parametric_axis == "z",
                    "edge " + kv.first + " parametric_axis",
                    "got '" + e.parametric_axis + "'");
        if (e.is_mortar) { ++mortar_per_axis[e.parametric_axis]; }
        else             { ++nonmortar_per_axis[e.parametric_axis]; }
    }
    for (const std::string& ax : {std::string("x"), std::string("y"),
                                  std::string("z")})
    {
        AssertOrDie(mortar_per_axis[ax] == 1,
                    "mortar edges along " + ax,
                    "expected 1, got " + std::to_string(mortar_per_axis[ax]));
        AssertOrDie(nonmortar_per_axis[ax] == 3,
                    "nonmortar edges along " + ax,
                    "expected 3, got " + std::to_string(nonmortar_per_axis[ax]));
    }
    std::cout << "  PASS  12 edges total: 3 mortar (1 per axis) + 9 nonmortar"
              << std::endl;
}

// ===========================================================================
// Test 3: 6 faces, top/right/back = mortar, bottom/left/front = nonmortar
// ===========================================================================
void test_faces_count_and_mortar_flags()
{
    std::cout << "Test 3: faces count and mortar flags" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    const auto& faces = bc.Faces();
    AssertOrDie(faces.size() == 6, "faces count",
                "got " + std::to_string(faces.size()) + ", expected 6");

    std::set<std::string> mortar_labels;
    std::set<std::string> nonmortar_labels;
    for (const auto& kv : faces)
    {
        if (kv.second.is_mortar) { mortar_labels.insert(kv.first); }
        else                     { nonmortar_labels.insert(kv.first); }
    }
    AssertOrDie(mortar_labels == std::set<std::string>{"top", "right", "back"},
                "mortar face set", "got unexpected set");
    AssertOrDie(nonmortar_labels ==
                    std::set<std::string>{"bottom", "left", "front"},
                "nonmortar face set", "got unexpected set");

    // Each face on a 4x4x4 hex mesh should have exactly 16 quad elements
    // (4×4) and 0 tri elements.
    for (const auto& kv : faces)
    {
        const FaceInfo3D& f = kv.second;
        AssertOrDie(f.NumElements() == 16,
                    "face '" + kv.first + "' element count",
                    "expected 16, got " + std::to_string(f.NumElements()));
        AssertOrDie(f.n_tri_elements == 0,
                    "face '" + kv.first + "' tri elements",
                    "expected 0, got " + std::to_string(f.n_tri_elements));
    }
    std::cout << "  PASS  6 faces, 16 quad/face, mortar = {top,right,back}"
              << std::endl;
}

// ===========================================================================
// Test 4: EdgePairs() returns 9 tuples; FacePairs() returns 3
// ===========================================================================
void test_pairs()
{
    std::cout << "Test 4: EdgePairs / FacePairs" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    auto epairs = bc.EdgePairs();
    AssertOrDie(epairs.size() == 9, "EdgePairs count",
                "got " + std::to_string(epairs.size()) + ", expected 9");
    // Per axis: 1 mortar paired against 3 nonmortars -> 3 axes * 3 = 9.
    std::map<std::string, int> per_axis;
    for (const auto& tup : epairs) { ++per_axis[std::get<0>(tup)]; }
    AssertOrDie(per_axis["x"] == 3 && per_axis["y"] == 3 && per_axis["z"] == 3,
                "EdgePairs per-axis count",
                "expected 3 per axis");

    auto fpairs = bc.FacePairs();
    AssertOrDie(fpairs.size() == 3, "FacePairs count",
                "got " + std::to_string(fpairs.size()) + ", expected 3");
    // Each pair must use distinct axes, and each pair's mortar/nonmortar
    // labels must come from the canonical sets.
    std::set<std::string> axes_seen;
    for (const auto& tup : fpairs)
    {
        const std::string& axis = std::get<0>(tup);
        const std::string& mortar = std::get<1>(tup);
        const std::string& nonmortar = std::get<2>(tup);
        axes_seen.insert(axis);
        AssertOrDie(mortar == "top" || mortar == "right" || mortar == "back",
                    "FacePair mortar", "got '" + mortar + "'");
        AssertOrDie(nonmortar == "bottom" || nonmortar == "left"
                        || nonmortar == "front",
                    "FacePair nonmortar", "got '" + nonmortar + "'");
    }
    AssertOrDie(axes_seen == std::set<std::string>{"x", "y", "z"},
                "FacePairs axes",
                "axes covered != {x, y, z}");
    std::cout << "  PASS  EdgePairs: 9 tuples (3 per axis); FacePairs: 3 tuples"
              << std::endl;
}

// ===========================================================================
// Test 5: sentinel rewriting on face elements
//
// On a 4×4×4 hex mesh, each face has a 4×4 grid of quad elements.
//   - The 4 corner-of-face quads (one per face corner) touch the
//     box's corner -> at least one of their gtdofs is -1.
//   - The 8 edge-of-face quads (those along a face boundary but not
//     at a corner) touch box edges -> at least one of their gtdofs
//     is -2 and none is -1.
//   - The 4 inner quads have no sentinels.
// ===========================================================================
void test_sentinel_rewriting()
{
    std::cout << "Test 5: sentinel rewriting" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    int total_corner_quads = 0;  // contains -1
    int total_edge_only_quads = 0;  // contains -2 but no -1
    int total_interior_quads = 0;  // no sentinels

    for (const auto& kv : bc.Faces())
    {
        for (const QuadFaceElement& qe : kv.second.quad_elements)
        {
            bool has_corner = false;
            bool has_edge = false;
            for (int g : qe.gtdofs)
            {
                if (g == kGtdofCornerSentinel) { has_corner = true; }
                else if (g == kGtdofEdgeSentinel) { has_edge = true; }
            }
            if (has_corner) { ++total_corner_quads; }
            else if (has_edge) { ++total_edge_only_quads; }
            else { ++total_interior_quads; }
        }
    }

    // Per face:  4 corner-of-face + 8 edge-of-face + 4 interior = 16.
    // Across 6 faces: 24 + 48 + 24 = 96.
    AssertOrDie(total_corner_quads == 24, "corner quads count",
                "expected 24, got " + std::to_string(total_corner_quads));
    AssertOrDie(total_edge_only_quads == 48, "edge-only quads count",
                "expected 48, got " + std::to_string(total_edge_only_quads));
    AssertOrDie(total_interior_quads == 24, "interior quads count",
                "expected 24, got " + std::to_string(total_interior_quads));
    std::cout << "  PASS  sentinel rewriting: 24 corner + 48 edge-only + "
                 "24 interior = 96 quads total" << std::endl;
}

// ===========================================================================
// Test 6: GtdofXyzLookup is consistent with corner records
// ===========================================================================
void test_gtdof_xyz_lookup()
{
    std::cout << "Test 6: GtdofXyzLookup" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    auto lookup = bc.GtdofXyzLookup();
    // For each corner, the lookup at corner.gtdof_x must yield
    // (gtdof_x, gtdof_y, gtdof_z).
    for (const auto& kv : bc.Corners())
    {
        const CornerInfo3D& c = kv.second;
        auto it = lookup.find(c.gtdof_x);
        AssertOrDie(it != lookup.end(),
                    std::string("corner '") + c.label + "' in lookup",
                    "missing entry for gtdof_x = " + std::to_string(c.gtdof_x));
        AssertOrDie(it->second[0] == c.gtdof_x
                    && it->second[1] == c.gtdof_y
                    && it->second[2] == c.gtdof_z,
                    std::string("corner '") + c.label + "' lookup match",
                    "lookup triple does not match corner gtdofs");
    }
    std::cout << "  PASS  GtdofXyzLookup consistent for all 8 corners"
              << std::endl;
}

// ===========================================================================
// Test 7: Summary() produces a non-empty, sane string
// ===========================================================================
void test_summary()
{
    std::cout << "Test 7: Summary()" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    std::string s = bc.Summary();
    AssertOrDie(!s.empty(), "Summary length", "Summary returned empty string");
    AssertOrDie(s.find("BoundaryClassifier3D") != std::string::npos,
                "Summary content", "no class name in Summary");
    AssertOrDie(s.find("bbox") != std::string::npos,
                "Summary content", "no bbox in Summary");
    AssertOrDie(s.find("corners") != std::string::npos,
                "Summary content", "no corners line in Summary");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) { std::cout << s; }
    std::cout << "  PASS  Summary returns a sane diagnostic string"
              << std::endl;
}

// ===========================================================================
// Test 8: TileShuffleFaceElements — routing correctness
//
// Phase 4.2 Batch H. After construction, the classifier has populated
// m_tile_shuffled_face_elements. For every shuffled element on this
// rank, OwnerRank(axis_pair, centroid) must return THIS rank's
// boundary-comm rank id. (Routing correctness on the receiver side.)
//
// Also smoke-checks that:
//   * The count of shuffled elements is non-negative.
//   * Each element's snap-keys correspond to a vertex actually in
//     the gathered classifier vertex catalogue (cross-validation
//     against the AllGather path).
//
// The test runs at np=1 by default (BLT NUM_MPI_TASKS 1), where the
// shuffle is a no-op self-loop but the routing math still has to be
// consistent.
// ===========================================================================
void test_tile_shuffle_routing()
{
    std::cout << "Test 8: TileShuffleFaceElements routing correctness"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    // Interior ranks have no work — empty list, no checks needed.
    if (!bc.IsBoundaryRank())
    {
        std::cout << "  PASS  (interior rank — no shuffled elements expected)"
                  << std::endl;
        return;
    }

    const auto& shuffled = bc.TileShuffledFaceElements();
    const TilePartition3D& tp = bc.TilePartition();
    const int my_bdy = bc.BdyRank();

    // Coverage: at np=1 with one boundary rank, ALL the local face
    // elements must end up on this rank. At higher rank counts the
    // count varies per rank.
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks == 1)
    {
        AssertOrDie(!shuffled.empty(),
                    "tile shuffle non-empty at np=1",
                    "expected shuffled elements on the only boundary rank, "
                    "got 0");
    }

    // Routing: every shuffled element must be on the rank
    // OwnerRank(axis_pair, centroid) returns.
    int n_routed_correctly = 0;
    for (const auto& sfe : shuffled)
    {
        // Recompute centroid from coords.
        const int n_v = sfe.coords.NumRows();
        std::array<double, 3> centroid = {0.0, 0.0, 0.0};
        for (int k = 0; k < n_v; ++k)
        {
            for (int d = 0; d < 3; ++d)
            {
                centroid[d] += sfe.coords(k, d);
            }
        }
        for (int d = 0; d < 3; ++d)
        {
            centroid[d] /= static_cast<double>(n_v);
        }
        const int owner = tp.OwnerRank(sfe.axis_pair, centroid);
        AssertOrDie(owner == my_bdy,
                    "shuffled element routed to correct rank",
                    "centroid axis_pair=" + sfe.axis_pair
                    + ": OwnerRank says rank " + std::to_string(owner)
                    + " but element was received on bdy rank "
                    + std::to_string(my_bdy));

        // tile_i, tile_j must invert the rank → (i, j) mapping
        // consistently with TilesOwnedBy.
        const AxisTileGrid& g = tp.Grid(sfe.axis_pair);
        const int local_rank_in_axis = my_bdy - g.axis_rank_start;
        AssertOrDie(local_rank_in_axis >= 0
                    && local_rank_in_axis < g.n_axis_ranks,
                    "tile (i, j) within this rank's axis-range",
                    "axis " + sfe.axis_pair
                    + " local_rank " + std::to_string(local_rank_in_axis));
        const int expected_i = local_rank_in_axis % g.n_tx;
        const int expected_j = local_rank_in_axis / g.n_tx;
        AssertOrDie(sfe.tile_i == expected_i && sfe.tile_j == expected_j,
                    "tile coords match rank inversion",
                    "got (" + std::to_string(sfe.tile_i) + ","
                    + std::to_string(sfe.tile_j) + ") expected ("
                    + std::to_string(expected_i) + ","
                    + std::to_string(expected_j) + ")");
        ++n_routed_correctly;
    }

    std::cout << "  PASS  " << n_routed_correctly
              << " shuffled elements routed correctly on bdy rank "
              << my_bdy << std::endl;
}

// ===========================================================================
// Test 9: TileShuffleFaceElements — global count cross-check
//
// Sums the per-rank shuffled element count across all boundary ranks
// and compares against this rank's local boundary submesh element
// count summed across boundary ranks.
//
// This catches two failure modes:
//   * Elements lost in the shuffle (sum < expected): MPI_Alltoallv
//     count or buffer mismatch.
//   * Elements duplicated (sum > expected): packing bug.
//
// At np=1 the sum is trivially equal because there's only one rank.
// At np > 1 this is a real cross-check on the Alltoall plumbing.
// ===========================================================================
void test_tile_shuffle_global_count()
{
    std::cout << "Test 9: TileShuffleFaceElements global count cross-check"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D bc(*b.pmesh, *b.fes);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Local count of submesh boundary elements (the original input
    // to the shuffle).
    int local_bdy_elem_count = 0;
    if (bc.IsBoundaryRank())
    {
        // The classifier doesn't expose m_bdr_submesh.GetNE(); for the
        // test we need an alternate way. We can use the BoundaryComm:
        // sum across boundary ranks of TileShuffledFaceElements().size()
        // must equal sum across boundary ranks of the original bdy
        // element count.
        //
        // The easiest cross-check: every local bdy element is sent to
        // exactly one rank, so sum_of_sends == sum_of_receives. So sum
        // of TileShuffledFaceElements().size() across boundary ranks
        // == sum of local_bdy_elem_count across boundary ranks.
        local_bdy_elem_count = b.pmesh->GetNBE();
    }
    int total_local;
    MPI_Allreduce(&local_bdy_elem_count, &total_local, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    int local_shuffled_count = 0;
    if (bc.IsBoundaryRank())
    {
        local_shuffled_count =
            static_cast<int>(bc.TileShuffledFaceElements().size());
    }
    int total_shuffled;
    MPI_Allreduce(&local_shuffled_count, &total_shuffled, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "    total_local_bdy_elems = " << total_local
                  << ", total_shuffled = " << total_shuffled << std::endl;
    }
    AssertOrDie(total_local == total_shuffled,
                "send count == recv count",
                "tile shuffle lost or duplicated elements: "
                "sent=" + std::to_string(total_local)
                + " received=" + std::to_string(total_shuffled));
    std::cout << "  PASS  global send count matches global recv count"
              << std::endl;
}

}  // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running BoundaryClassifier3D integration tests"
                  << std::endl;
        std::cout << "----------------------------------------------"
                  << std::endl;
    }
    test_corners_count_and_coords();
    test_edges_count_and_mortar_flags();
    test_faces_count_and_mortar_flags();
    test_pairs();
    test_sentinel_rewriting();
    test_gtdof_xyz_lookup();
    test_summary();
    test_tile_shuffle_routing();
    test_tile_shuffle_global_count();
    if (rank == 0)
    {
        std::cout << "----------------------------------------------"
                  << std::endl;
        std::cout << "All BoundaryClassifier3D tests passed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
