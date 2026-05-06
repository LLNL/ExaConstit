// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.2 — implementation of TilePartition3D.

#include "tile_partition_3d.hpp"

#include "mfem.hpp"  // for MFEM_VERIFY / MFEM_ABORT

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mortar_pbc {

namespace {

//==============================================================================
// Index of an axis-pair name in {"x", "y", "z"} → 0, 1, 2.
//==============================================================================
int AxisIdxFromName(const std::string& axis)
{
    if (axis == "x") { return 0; }
    if (axis == "y") { return 1; }
    if (axis == "z") { return 2; }
    MFEM_ABORT("TilePartition3D: unknown axis '" << axis << "'");
    return -1;
}

//==============================================================================
// Perpendicular axes for a given axis-pair.
//
// For axis-pair x (x=const planes), the parametric plane is (y, z).
// For axis-pair y, the plane is (x, z). For axis-pair z, the plane is
// (x, y). This is the convention used throughout the boundary helpers.
//==============================================================================
std::pair<int, int> PerpAxes(int axis_idx)
{
    switch (axis_idx)
    {
        case 0: return {1, 2};  // x-pair → (y, z)
        case 1: return {0, 2};  // y-pair → (x, z)
        case 2: return {0, 1};  // z-pair → (x, y)
        default:
            MFEM_ABORT("TilePartition3D: invalid axis_idx " << axis_idx);
    }
    return {-1, -1};
}

}  // anonymous namespace

//==============================================================================
// AllocateAxisRanks — distribute n_bdy_ranks across 3 axis-pairs
//
// floor(N/3) ranks per axis-pair, plus one extra each to the first
// (N % 3) axes. So:
//   n_bdy = 1  → (1, 1, 1)  (degenerate; every axis shares the rank)
//   n_bdy = 2  → (1, 1, 1)  (degenerate; ranks 0 and 1 each cover all 3 axes)
//   n_bdy = 3  → (1, 1, 1)
//   n_bdy = 4  → (2, 1, 1)
//   n_bdy = 6  → (2, 2, 2)
//   n_bdy = 12 → (4, 4, 4)
//
// SPECIAL CASE: when n_bdy < 3, we replicate axis assignment across all
// available ranks. In that regime there's no scaling concern anyway.
//==============================================================================
std::array<int, 3> TilePartition3D::AllocateAxisRanks(int n_bdy_ranks)
{
    MFEM_VERIFY(n_bdy_ranks >= 1,
                "TilePartition3D: n_bdy_ranks must be >= 1, got "
                << n_bdy_ranks);

    if (n_bdy_ranks < 3)
    {
        // All axes use the same rank pool; report 1 rank per axis as
        // the "fair" allocation (the actual rank-list assignment in
        // the ctor handles the wrap-around so no rank is overloaded).
        // For 1 rank it's truly degenerate; for 2 ranks the axis-rank
        // ranges overlap.
        return {1, 1, 1};
    }

    const int base = n_bdy_ranks / 3;
    const int rem  = n_bdy_ranks % 3;

    std::array<int, 3> out;
    out[0] = base + (rem > 0 ? 1 : 0);
    out[1] = base + (rem > 1 ? 1 : 0);
    out[2] = base;
    return out;
}

//==============================================================================
// FactorTileGrid — find (n_tx, n_ty) with n_tx * n_ty == N
//
// Strategy: walk down from floor(sqrt(N)) to find the largest divisor.
// That gives us n_tx; then n_ty = N / n_tx. For prime N this falls back
// to (1, N).
//==============================================================================
std::pair<int, int> TilePartition3D::FactorTileGrid(int n_axis_ranks)
{
    MFEM_VERIFY(n_axis_ranks >= 1,
                "TilePartition3D: n_axis_ranks must be >= 1, got "
                << n_axis_ranks);

    const int sqrt_floor = static_cast<int>(std::floor(std::sqrt(
        static_cast<double>(n_axis_ranks))));
    // sqrt_floor is at least 1 for n_axis_ranks >= 1.
    for (int n_tx = sqrt_floor; n_tx >= 1; --n_tx)
    {
        if (n_axis_ranks % n_tx == 0)
        {
            return {n_tx, n_axis_ranks / n_tx};
        }
    }
    // Unreachable: n_tx=1 always divides.
    return {1, n_axis_ranks};
}

//==============================================================================
// Constructor — build the three axis grids deterministically
//==============================================================================
TilePartition3D::TilePartition3D(const std::array<double, 3>& bbox_min,
                                 const std::array<double, 3>& bbox_max,
                                 int n_bdy_ranks)
    : m_n_bdy_ranks(n_bdy_ranks)
{
    MFEM_VERIFY(n_bdy_ranks >= 1,
                "TilePartition3D: n_bdy_ranks must be >= 1, got "
                << n_bdy_ranks);
    for (int d = 0; d < 3; ++d)
    {
        MFEM_VERIFY(bbox_max[d] > bbox_min[d],
                    "TilePartition3D: bbox extent on axis " << d
                    << " is non-positive: ["
                    << bbox_min[d] << ", " << bbox_max[d] << ")");
    }

    const std::array<int, 3> n_axis_ranks = AllocateAxisRanks(n_bdy_ranks);

    // axis_rank_start: cumulative sum of allocations. Special-cased
    // for the degenerate small-n_bdy regime (n_bdy < 3): every axis
    // starts at rank 0 and shares the pool.
    std::array<int, 3> axis_rank_start;
    if (n_bdy_ranks < 3)
    {
        axis_rank_start = {0, 0, 0};
    }
    else
    {
        axis_rank_start[0] = 0;
        axis_rank_start[1] = n_axis_ranks[0];
        axis_rank_start[2] = n_axis_ranks[0] + n_axis_ranks[1];
    }

    // Build each axis grid.
    auto build_grid = [&](int axis_idx, AxisTileGrid& g)
    {
        const auto [a_idx, b_idx] = PerpAxes(axis_idx);
        const auto [n_tx, n_ty] = FactorTileGrid(n_axis_ranks[axis_idx]);
        g.n_tx = n_tx;
        g.n_ty = n_ty;
        g.axis_rank_start = axis_rank_start[axis_idx];
        g.n_axis_ranks = n_axis_ranks[axis_idx];
        g.a_idx = a_idx;
        g.b_idx = b_idx;
        g.a_min = bbox_min[a_idx];
        g.b_min = bbox_min[b_idx];
        g.dx = (bbox_max[a_idx] - bbox_min[a_idx]) / n_tx;
        g.dy = (bbox_max[b_idx] - bbox_min[b_idx]) / n_ty;
    };
    build_grid(0, m_grid_x);
    build_grid(1, m_grid_y);
    build_grid(2, m_grid_z);
}

//==============================================================================
// Grid — accessor by axis name
//==============================================================================
const AxisTileGrid& TilePartition3D::Grid(const std::string& axis) const
{
    const int idx = AxisIdxFromName(axis);
    switch (idx)
    {
        case 0: return m_grid_x;
        case 1: return m_grid_y;
        case 2: return m_grid_z;
    }
    MFEM_ABORT("unreachable");
    return m_grid_x;
}

//==============================================================================
// OwnerRankFast — translate (pa, pb) to a tile-owning rank
//
// Tile (i, j) for i ∈ [0, n_tx), j ∈ [0, n_ty) maps to rank
//   axis_rank_start + j * n_tx + i.
// Coords on the upper boundary (== bbox_max) are snapped to the last
// interior tile so the partition covers the closed bbox.
//==============================================================================
int TilePartition3D::OwnerRankFast(double pa, double pb,
                                   const AxisTileGrid& grid)
{
    int i = static_cast<int>(std::floor((pa - grid.a_min) / grid.dx));
    int j = static_cast<int>(std::floor((pb - grid.b_min) / grid.dy));
    if (i < 0) { i = 0; }
    if (i >= grid.n_tx) { i = grid.n_tx - 1; }
    if (j < 0) { j = 0; }
    if (j >= grid.n_ty) { j = grid.n_ty - 1; }
    return grid.axis_rank_start + j * grid.n_tx + i;
}

//==============================================================================
// OwnerRank — axis-string dispatch wrapper
//==============================================================================
int TilePartition3D::OwnerRank(const std::string& axis,
                               const std::array<double, 3>& parametric) const
{
    const AxisTileGrid& g = Grid(axis);
    return OwnerRankFast(parametric[g.a_idx], parametric[g.b_idx], g);
}

//==============================================================================
// TilesOwnedBy — invert the rank → tile mapping for a given rank
//==============================================================================
std::vector<std::tuple<std::string, int, int>>
TilePartition3D::TilesOwnedBy(int my_bdy_rank) const
{
    std::vector<std::tuple<std::string, int, int>> out;
    const std::array<const AxisTileGrid*, 3> grids = {
        &m_grid_x, &m_grid_y, &m_grid_z
    };
    const std::array<const char*, 3> names = {"x", "y", "z"};
    for (int axis_idx = 0; axis_idx < 3; ++axis_idx)
    {
        const AxisTileGrid& g = *grids[axis_idx];
        const int local_rank = my_bdy_rank - g.axis_rank_start;
        if (local_rank < 0 || local_rank >= g.n_axis_ranks)
        {
            continue;  // this rank doesn't own a tile on this axis
        }
        const int i = local_rank % g.n_tx;
        const int j = local_rank / g.n_tx;
        out.emplace_back(std::string(names[axis_idx]), i, j);
    }
    return out;
}

}  // namespace mortar_pbc
