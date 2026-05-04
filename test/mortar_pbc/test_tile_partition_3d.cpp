// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.2 — unit test for TilePartition3D.
//
// All tests are pure arithmetic — no MPI collectives, no mesh, no FES.
// The map is constructed from (bbox, n_bdy_ranks) and tested against
// expected values for several rank counts.
//
// Coverage:
//   1. Axis-rank allocation across the 3 axis-pairs.
//   2. Tile-grid factorisation for various rank counts (perfect
//      squares, primes, composites).
//   3. OwnerRank / OwnerRankFast — point-to-tile dispatch.
//   4. TilesOwnedBy — inversion of the rank → tile map; every tile
//      claimed by exactly one rank.
//   5. Round-trip consistency: pick a random parametric centroid,
//      look up the owner, query that owner's tile list, verify the
//      tile contains the centroid.
//   6. Determinism: building the same partition on two distinct
//      instances yields identical maps (every accessor agrees).

#include "tile_partition_3d.hpp"

#include "mfem.hpp"  // for MFEM_VERIFY (used internally) + main MPI

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using mortar_pbc::AxisTileGrid;
using mortar_pbc::TilePartition3D;

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

const std::array<double, 3> kBboxMin = {0.0, 0.0, 0.0};
const std::array<double, 3> kBboxMax = {1.0, 1.0, 1.0};

// ===========================================================================
// Test 1: axis-rank allocation
//
// n_bdy_ranks  →  expected (n_x, n_y, n_z)
//      1       →  every axis gets 1 (degenerate; rank 0 covers all)
//      2       →  every axis gets 1 (degenerate; ranks share)
//      3       →  (1, 1, 1)
//      4       →  (2, 1, 1)
//      5       →  (2, 2, 1)
//      6       →  (2, 2, 2)
//      7       →  (3, 2, 2)
//     12       →  (4, 4, 4)
//     30       →  (10, 10, 10)
// ===========================================================================
void test_axis_rank_allocation()
{
    std::cout << "Test 1: axis-rank allocation across 3 axes" << std::endl;
    struct Case { int n; std::array<int, 3> expected; };
    const std::vector<Case> cases = {
        {1,  {1, 1, 1}}, {2,  {1, 1, 1}}, {3,  {1, 1, 1}},
        {4,  {2, 1, 1}}, {5,  {2, 2, 1}}, {6,  {2, 2, 2}},
        {7,  {3, 2, 2}}, {12, {4, 4, 4}}, {30, {10, 10, 10}},
    };
    for (const auto& c : cases)
    {
        TilePartition3D tp(kBboxMin, kBboxMax, c.n);
        const int got_x = tp.Grid("x").n_axis_ranks;
        const int got_y = tp.Grid("y").n_axis_ranks;
        const int got_z = tp.Grid("z").n_axis_ranks;
        std::stringstream s;
        s << "n_bdy=" << c.n << ", expected ("
          << c.expected[0] << "," << c.expected[1] << "," << c.expected[2]
          << "), got (" << got_x << "," << got_y << "," << got_z << ")";
        AssertOrDie(got_x == c.expected[0] && got_y == c.expected[1]
                    && got_z == c.expected[2],
                    "axis allocation", s.str());
    }
    std::cout << "  PASS  9 allocation cases match expected" << std::endl;
}

// ===========================================================================
// Test 2: tile-grid factorisation
//
// For each axis, n_tx * n_ty must equal n_axis_ranks, and n_tx must be
// as close to √N as possible (i.e., the largest divisor ≤ √N).
//
// n_axis_ranks  →  (n_tx, n_ty)
//        1      →  (1, 1)
//        2      →  (1, 2)        (prime)
//        4      →  (2, 2)        (perfect square)
//        6      →  (2, 3)        (composite, sqrt(6)≈2.45 → 2 is largest divisor ≤ 2.45)
//        9      →  (3, 3)
//       16      →  (4, 4)
//       25      →  (5, 5)
//       12      →  (3, 4)        (sqrt(12)≈3.46 → 3 is largest divisor ≤ 3.46)
//        7      →  (1, 7)        (prime)
// ===========================================================================
void test_tile_grid_factorisation()
{
    std::cout << "Test 2: tile-grid factorisation" << std::endl;
    // We can't directly access FactorTileGrid (private static); we
    // validate via the resulting AxisTileGrid for n_bdy values that
    // produce known per-axis rank counts.
    struct Case { int n_bdy; int axis; std::pair<int, int> expected; };
    const std::vector<Case> cases = {
        // n_bdy=3 → (1,1,1) per axis. Each axis gets 1 rank → 1×1.
        { 3, 0, {1, 1}}, { 3, 1, {1, 1}}, { 3, 2, {1, 1}},
        // n_bdy=12 → (4,4,4). Each axis gets 4 ranks → 2×2.
        {12, 0, {2, 2}}, {12, 1, {2, 2}}, {12, 2, {2, 2}},
        // n_bdy=27 → (9,9,9). 3×3.
        {27, 0, {3, 3}}, {27, 1, {3, 3}}, {27, 2, {3, 3}},
        // n_bdy=21 → (7,7,7). 1×7 (prime).
        {21, 0, {1, 7}}, {21, 1, {1, 7}}, {21, 2, {1, 7}},
        // n_bdy=18 → (6,6,6). 2×3 (sqrt(6)≈2.45, 2 is largest divisor).
        {18, 0, {2, 3}}, {18, 1, {2, 3}}, {18, 2, {2, 3}},
        // n_bdy=4 → (2,1,1). x-axis 2 ranks → 1×2; others 1×1.
        { 4, 0, {1, 2}}, { 4, 1, {1, 1}}, { 4, 2, {1, 1}},
    };
    const std::array<const char*, 3> axis_names = {"x", "y", "z"};
    for (const auto& c : cases)
    {
        TilePartition3D tp(kBboxMin, kBboxMax, c.n_bdy);
        const AxisTileGrid& g = tp.Grid(axis_names[c.axis]);
        std::stringstream s;
        s << "n_bdy=" << c.n_bdy << " axis=" << axis_names[c.axis]
          << " expected (" << c.expected.first << "x" << c.expected.second
          << "), got (" << g.n_tx << "x" << g.n_ty << ")";
        AssertOrDie(g.n_tx == c.expected.first && g.n_ty == c.expected.second,
                    "tile grid factorisation", s.str());
        // Sanity: product matches n_axis_ranks.
        AssertOrDie(g.n_tx * g.n_ty == g.n_axis_ranks,
                    "n_tx * n_ty == n_axis_ranks",
                    "violated for n_bdy=" + std::to_string(c.n_bdy)
                    + " axis=" + axis_names[c.axis]);
    }
    std::cout << "  PASS  18 factorisation cases match expected" << std::endl;
}

// ===========================================================================
// Test 3: OwnerRank — point-to-tile dispatch
// ===========================================================================
void test_owner_rank()
{
    std::cout << "Test 3: OwnerRank dispatch" << std::endl;
    // Use n_bdy=12 → each axis 2×2 grid, axis_rank_start = (0, 4, 8).
    TilePartition3D tp(kBboxMin, kBboxMax, 12);

    // For axis "x", parametric plane is (y, z). Tile (i, j) at
    // (y in [i/2, (i+1)/2), z in [j/2, (j+1)/2)) → rank 0 + j*2 + i.
    {
        // Centroid (0.25, 0.25) on x-axis: y=0.25 → i=0, z=0.25 → j=0
        // → tile (0, 0) → rank 0.
        const int rank = tp.OwnerRank("x", {0.5, 0.25, 0.25});
        AssertOrDie(rank == 0, "OwnerRank x (0.25,0.25)",
                    "expected 0, got " + std::to_string(rank));
    }
    {
        // (0.75, 0.75) on x-axis: y=0.75 → i=1, z=0.75 → j=1
        // → tile (1, 1) → rank 0 + 1*2 + 1 = 3.
        const int rank = tp.OwnerRank("x", {0.5, 0.75, 0.75});
        AssertOrDie(rank == 3, "OwnerRank x (0.75,0.75)",
                    "expected 3, got " + std::to_string(rank));
    }
    {
        // y-axis: parametric plane is (x, z). (0.25, 0.75)
        // → i=0, j=1 → tile (0, 1) → rank 4 + 1*2 + 0 = 6.
        const int rank = tp.OwnerRank("y", {0.25, 0.5, 0.75});
        AssertOrDie(rank == 6, "OwnerRank y (0.25,0.75)",
                    "expected 6, got " + std::to_string(rank));
    }
    {
        // z-axis: parametric plane is (x, y). (0.75, 0.75)
        // → i=1, j=1 → tile (1, 1) → rank 8 + 1*2 + 1 = 11.
        const int rank = tp.OwnerRank("z", {0.75, 0.75, 0.5});
        AssertOrDie(rank == 11, "OwnerRank z (0.75,0.75)",
                    "expected 11, got " + std::to_string(rank));
    }
    // Boundary snap: a coord exactly at bbox_max should fall in the
    // last tile, not outside.
    {
        const int rank = tp.OwnerRank("x", {0.5, 1.0, 1.0});
        AssertOrDie(rank == 3, "OwnerRank x boundary",
                    "expected 3 (last tile), got " + std::to_string(rank));
    }
    std::cout << "  PASS  5 OwnerRank dispatches match expected" << std::endl;
}

// ===========================================================================
// Test 4: TilesOwnedBy — every tile claimed by exactly one rank
// ===========================================================================
void test_tiles_owned_by()
{
    std::cout << "Test 4: TilesOwnedBy partition coverage" << std::endl;
    for (int n_bdy : {3, 4, 6, 12, 27}) {
        TilePartition3D tp(kBboxMin, kBboxMax, n_bdy);
        // Aggregate (axis, i, j) tuples claimed across all ranks.
        std::set<std::tuple<std::string, int, int>> claimed;
        for (int r = 0; r < n_bdy; ++r)
        {
            const auto tiles = tp.TilesOwnedBy(r);
            for (const auto& t : tiles)
            {
                AssertOrDie(claimed.insert(t).second,
                            "no double-claim",
                            "tile claimed twice at n_bdy="
                            + std::to_string(n_bdy));
            }
        }
        // Total expected tiles: sum over axes of (n_tx * n_ty).
        const int expected_total =
            tp.Grid("x").n_tx * tp.Grid("x").n_ty
          + tp.Grid("y").n_tx * tp.Grid("y").n_ty
          + tp.Grid("z").n_tx * tp.Grid("z").n_ty;
        AssertOrDie(static_cast<int>(claimed.size()) == expected_total,
                    "all tiles claimed",
                    "n_bdy=" + std::to_string(n_bdy)
                    + ": expected " + std::to_string(expected_total)
                    + " claimed " + std::to_string(claimed.size()));
    }
    std::cout << "  PASS  every tile claimed by exactly one rank "
                 "across 5 rank counts" << std::endl;
}

// ===========================================================================
// Test 5: round-trip consistency
//
// For random parametric centroids: OwnerRank → TilesOwnedBy → check
// the centroid falls inside that rank's claimed tile bounds.
// ===========================================================================
void test_round_trip()
{
    std::cout << "Test 5: round-trip parametric → owner → tile bounds"
              << std::endl;
    TilePartition3D tp(kBboxMin, kBboxMax, 12);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int n_checked = 0;
    for (int trial = 0; trial < 200; ++trial)
    {
        const double a = dist(rng);
        const double b = dist(rng);
        for (const std::string axis : {"x", "y", "z"})
        {
            std::array<double, 3> par = {0.5, 0.5, 0.5};
            const AxisTileGrid& g = tp.Grid(axis);
            par[g.a_idx] = a;
            par[g.b_idx] = b;
            const int owner = tp.OwnerRank(axis, par);
            const auto tiles = tp.TilesOwnedBy(owner);
            // Find the tile on the matching axis.
            bool found = false;
            for (const auto& [ax_name, i, j] : tiles)
            {
                if (ax_name != axis) { continue; }
                const double a_lo = g.a_min + i * g.dx;
                const double a_hi = g.a_min + (i + 1) * g.dx;
                const double b_lo = g.b_min + j * g.dy;
                const double b_hi = g.b_min + (j + 1) * g.dy;
                if (a >= a_lo && a < a_hi + 1e-12
                 && b >= b_lo && b < b_hi + 1e-12)
                {
                    found = true;
                    break;
                }
            }
            AssertOrDie(found, "centroid in owner's tile",
                        "axis=" + axis + " a=" + std::to_string(a)
                        + " b=" + std::to_string(b)
                        + " owner=" + std::to_string(owner));
            ++n_checked;
        }
    }
    std::cout << "  PASS  " << n_checked
              << " random round-trips (no centroid escapes its claimed tile)"
              << std::endl;
}

// ===========================================================================
// Test 6: determinism — same inputs give same output across instances
// ===========================================================================
void test_determinism()
{
    std::cout << "Test 6: determinism across two instances" << std::endl;
    TilePartition3D a(kBboxMin, kBboxMax, 12);
    TilePartition3D b(kBboxMin, kBboxMax, 12);
    for (const std::string axis : {"x", "y", "z"})
    {
        const AxisTileGrid& ga = a.Grid(axis);
        const AxisTileGrid& gb = b.Grid(axis);
        AssertOrDie(ga.n_tx == gb.n_tx && ga.n_ty == gb.n_ty
                    && ga.axis_rank_start == gb.axis_rank_start
                    && ga.n_axis_ranks == gb.n_axis_ranks,
                    "grid match", "axis=" + axis);
    }
    // Spot-check a few owner lookups.
    for (int trial = 0; trial < 50; ++trial)
    {
        const std::array<double, 3> par = {0.1 * (trial % 9), 0.1 * (trial % 7),
                                           0.1 * (trial % 5)};
        AssertOrDie(a.OwnerRank("x", par) == b.OwnerRank("x", par),
                    "OwnerRank match", "trial " + std::to_string(trial));
    }
    std::cout << "  PASS  two TilePartition3D instances agree on grids "
                 "and 50 lookups" << std::endl;
}

}  // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running TilePartition3D unit tests" << std::endl;
        std::cout << "----------------------------------------------"
                  << std::endl;
    }

    // The tile partition is pure arithmetic — every rank runs every
    // test independently. No collectives needed.
    test_axis_rank_allocation();
    test_tile_grid_factorisation();
    test_owner_rank();
    test_tiles_owned_by();
    test_round_trip();
    test_determinism();

    if (rank == 0)
    {
        std::cout << "----------------------------------------------"
                  << std::endl;
        std::cout << "All TilePartition3D tests passed." << std::endl;
    }
    MPI_Finalize();
    return 0;
}
