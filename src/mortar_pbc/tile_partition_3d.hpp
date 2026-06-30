// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.2 — deterministic tile-rank map for distributed mortar
// pair matching.
//
// What this is
// ------------
// Phase 4.1's `BoundaryClassifier3D` AllGathers all per-rank boundary
// face-element records, so every boundary rank ends up with a full
// global view. This is O(boundary_size) per rank and saturates around
// p ~ 13 (n_bdy_ranks ~ 1000–2000).
//
// Phase 4.2 replaces that AllGather with a tile-partitioned shuffle:
// for each periodic-pair axis, the parametric (a, b) plane is tiled
// into a regular grid; each tile is owned by a deterministic rank in
// `boundary_comm`. Face elements are routed to the rank owning the
// tile their parametric centroid falls into. Mortar/nonmortar partners
// route identically (their parametric coords match modulo period), so
// matching becomes tile-local.
//
// `TilePartition3D` is the deterministic tile-to-rank map. It's a
// pure-function helper:
//   * Inputs:  global bbox; n_bdy_ranks (size of boundary subcomm).
//   * Outputs: per-axis (n_tx, n_ty) tile grid; per-axis tile-to-rank
//              array; per-axis (a, b) parametric perpendicular axes;
//              method to translate a parametric centroid to its
//              tile-owning rank.
//
// The map is constructed identically on every rank (no MPI), so any
// inconsistency would be a deterministic bug, not a synchronization
// issue. The header is small and unit-tested in isolation
// (see `test_tile_partition_3d.cpp`).
//
// Design notes
// ------------
// * **Axis-rank assignment.** Each of the 3 axis-pairs (x, y, z) gets
//   `floor(n_bdy / 3)` ranks; the remainder (`n_bdy % 3`) is
//   distributed one extra rank per axis-pair starting at x. So for
//   `n_bdy = 4` we get axis ranks (2, 1, 1); for `n_bdy = 7` we get
//   (3, 2, 2); for `n_bdy = 1` we get (1, 1, 1) (every axis-pair
//   shares the single rank — duplicating is fine because the matching
//   is per-axis anyway).
//
// * **Tile-grid factorisation.** For an axis with `N` ranks, we pick
//   `(n_tx, n_ty)` such that `n_tx * n_ty == N` and `n_tx` is as close
//   to `√N` as possible. Find the largest divisor of `N` not exceeding
//   `floor(√N)`, set `n_tx` to that and `n_ty = N / n_tx`. For prime
//   `N`, this falls back to `1 × N` (a stripe). The aspect-ratio
//   penalty is mild and only material at small `N`.
//
// * **Tile-to-rank ordering.** Tile `(i, j)` in `[0, n_tx) × [0, n_ty)`
//   maps to the `j * n_tx + i`'th rank in the axis-pair's rank list.
//   The rank list itself is the contiguous slice of `boundary_comm`
//   ranks `[axis_rank_start, axis_rank_start + N)` where
//   `axis_rank_start = sum_{prior_axes}(N_prior)`. With the rank-
//   count distribution above, this gives:
//     - `n_bdy=4`:  x ranks [0, 1] (2x1), y ranks [2] (1), z ranks [3] (1).
//     - `n_bdy=12`: x ranks [0..3] (2x2), y ranks [4..7] (2x2), z ranks [8..11] (2x2).
//     - `n_bdy=1`:  every axis owns rank 0 (degenerate, single tile).
//
// * **Parametric perpendicular axes.** For axis `x` (x-axis pair), the
//   parametric plane is (y, z); for `y` it's (x, z); for `z` it's (x, y).
//   Each axis's tile grid spans `[bbox_min[a], bbox_max[a]) × [bbox_min[b], bbox_max[b])`.
//
// References
// ----------
//   * §P4.4.4 Strategy B in PHASE4_CPP_PORT_PLAN.md.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mortar_pbc {

/**
 * @brief Per-axis tile grid description.
 */
struct AxisTileGrid
{
    /// Number of tiles along the "a" perpendicular axis.
    int n_tx = 0;
    /// Number of tiles along the "b" perpendicular axis.
    int n_ty = 0;
    /// First rank in `boundary_comm` owning a tile of this axis.
    /// Tiles `(i, j)` for `i ∈ [0, n_tx)`, `j ∈ [0, n_ty)` map to
    /// rank `axis_rank_start + j * n_tx + i`.
    int axis_rank_start = 0;
    /// Total number of ranks owning tiles on this axis-pair.
    /// Equals `n_tx * n_ty`.
    int n_axis_ranks = 0;
    /// Tile size along the "a" perpendicular axis.
    /// `(bbox_max[a_idx] - bbox_min[a_idx]) / n_tx`.
    double dx = 0.0;
    /// Tile size along the "b" perpendicular axis.
    double dy = 0.0;
    /// Lower bound of the tile grid on the "a" perpendicular axis.
    /// Equals `bbox_min[a_idx]`.
    double a_min = 0.0;
    /// Lower bound of the tile grid on the "b" perpendicular axis.
    double b_min = 0.0;
    /// Index of the "a" perpendicular axis (0=x, 1=y, 2=z).
    int a_idx = -1;
    /// Index of the "b" perpendicular axis.
    int b_idx = -1;
};

/**
 * @brief Deterministic tile-to-rank partition for the three axis-pairs.
 *
 * @details Built identically on every rank from `(bbox, n_bdy_ranks)`.
 * No MPI calls; pure local arithmetic.
 */
class TilePartition3D
{
public:
    /**
     * @brief Build the partition.
     *
     * @param bbox_min      Lower-corner of the global bounding box.
     * @param bbox_max      Upper-corner of the global bounding box.
     * @param n_bdy_ranks   Size of the boundary subcommunicator. Must
     *                      be >= 1.
     */
    TilePartition3D(const std::array<double, 3>& bbox_min,
                    const std::array<double, 3>& bbox_max,
                    int n_bdy_ranks);

    /// Per-axis-pair tile grid. Index by `axis` ∈ {"x", "y", "z"}.
    const AxisTileGrid& Grid(const std::string& axis) const;

    /// Number of boundary-comm ranks the partition was built for.
    int NBdyRanks() const { return m_n_bdy_ranks; }

    /**
     * @brief Map a parametric (a, b) coordinate on a given axis-pair
     *        to the boundary-comm rank that owns the containing tile.
     *
     * @param axis        Axis-pair identifier ("x", "y", or "z").
     * @param parametric  3D coordinate; only the (a, b) components
     *                    perpendicular to `axis` are used.
     *
     * @return Boundary-comm rank index in `[0, n_bdy_ranks)`.
     *
     * @details Coordinate components on the boundary of the bbox are
     * snapped to the last interior tile so a centroid exactly at
     * `bbox_max[a]` does not fall outside the grid.
     */
    int OwnerRank(const std::string& axis,
                  const std::array<double, 3>& parametric) const;

    /**
     * @brief Same, but pass already-extracted (a, b) parametric coords
     *        and the axis grid directly. Avoids the axis-string
     *        dispatch in tight loops.
     */
    static int OwnerRankFast(double pa, double pb, const AxisTileGrid& grid);

    /**
     * @brief List of (axis, tile_i, tile_j) tuples this rank owns.
     *
     * @param my_bdy_rank  This rank's index in `boundary_comm`.
     *
     * @return Possibly empty vector. Empty for ranks not assigned to
     *         any axis (which can happen at very small `n_bdy_ranks`,
     *         or when an axis grid has fewer tiles than its allocated
     *         rank count — but our factorisation guarantees
     *         `n_tx * n_ty == n_axis_ranks` so this can't happen with
     *         the current scheme).
     */
    std::vector<std::tuple<std::string, int, int>> TilesOwnedBy(
        int my_bdy_rank) const;

private:
    /// Allocate ranks across the 3 axis pairs.
    /// Returns `(n_x_ranks, n_y_ranks, n_z_ranks)`. Sums to `n_bdy_ranks`.
    static std::array<int, 3> AllocateAxisRanks(int n_bdy_ranks);

    /// Given a rank count, find `(n_tx, n_ty)` with `n_tx * n_ty == N`
    /// and `n_tx` as close to `√N` as possible (but never larger).
    static std::pair<int, int> FactorTileGrid(int n_axis_ranks);

    int m_n_bdy_ranks = 0;
    AxisTileGrid m_grid_x;
    AxisTileGrid m_grid_y;
    AxisTileGrid m_grid_z;
};

}  // namespace mortar_pbc
