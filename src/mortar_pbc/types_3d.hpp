// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `mortar_pbc/types_3d.py`
//
// Pure data containers for the 3D mortar PBC machinery, mirroring the
// Python prototype's `types_3d.py`. These are the data contracts between
// `BoundaryClassifier3D` (producer) and `ConstraintBuilder3D` (consumer);
// keeping them in a header-only module with minimal dependencies means
// they can be constructed in unit tests without invoking the full
// classifier.
//
// References:
//   * MORTAR_PBC_ARCHITECTURE.md §5.4 (3D wirebasket hierarchy)
//   * MORTAR_PBC_ARCHITECTURE.md §11.7 (BoundaryClassifier3D design)
//   * PHASE4_CPP_PORT_PLAN.md §P4.4.2 (this directory layout)

#pragma once
#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mortar_pbc {

// ============================================================================
// Sentinel values for the wirebasket hierarchy
// ============================================================================
//
// Each face/edge element node carries a global TDOF index (per spatial
// component). When the node has been classified as belonging to a higher
// level of the wirebasket hierarchy (corner or edge), the gtdof is replaced
// by a sentinel:
//
//   gtdof >= 0     : face-interior DOF — kept in D and A^m row/col.
//   gtdof == -1    : corner DOF — Dirichlet-pinned at u_lin per Method-D
//                    (architecture §2.2). Row dropped (nonmortar side); col
//                    dropped (mortar side); the corresponding constraint
//                    contribution is NOT added to the RHS because the corner
//                    pin is enforced at the primal level via EliminateRowsCols.
//   gtdof == -2    : edge DOF — constrained by 1D edge mortar (§11.5). Row
//                    dropped (nonmortar); col dropped (mortar); the edge
//                    mortar block handles this DOF's periodicity.
//
// This mirrors the Python prototype's MortarAssembler2D._integrate_overlap_segment
// (mortar_2d.py:396-414) and the §5.4 wirebasket hierarchy.

constexpr int kGtdofCornerSentinel = -1;
constexpr int kGtdofEdgeSentinel   = -2;

inline bool IsKeptGtdof(int gtdof) noexcept {
    return gtdof >= 0;
}

inline bool IsCornerSentinel(int gtdof) noexcept {
    return gtdof == kGtdofCornerSentinel;
}

inline bool IsEdgeSentinel(int gtdof) noexcept {
    return gtdof == kGtdofEdgeSentinel;
}

// Edge connectivity sentinels — used in `EdgeInfo3D::elements` to indicate
// that one or both endpoints of a line-2 boundary element coincide with
// a box corner (so its row should be dropped after assembly).
constexpr int kEdgeNodeLeftCornerSentinel  = -1;  // = edge_min along param axis
constexpr int kEdgeNodeRightCornerSentinel = -2;  // = edge_max along param axis

/**
 * @brief One of the 8 corner nodes of a box-shaped RVE.
 *
 * @details A 3D box RVE has exactly 8 corners. Under Method-D PBC
 * (architecture §2), each corner is essentially Dirichlet-prescribed
 * at \f$u_{\rm lin}[\mathrm{corner}] = (F_{\rm macro} - I)\,
 * X[\mathrm{corner}]\f$, where \f$X[\mathrm{corner}]\f$ is the
 * reference-frame corner coordinate. The 8 corners pin rigid-body
 * modes (3 translations + 3 rotations) plus the linear-affine
 * macroscopic part of the deformation. The LM rows for these DOFs
 * are dropped by the Wohlmuth modification (architecture §5.1 /
 * §5.2 / §5.3).
 *
 * @details `label` is one of the 8 strings:
 *   "blf" (bottom-left-front), "brf", "tlf", "trf",
 *   "blb" (bottom-left-back),  "brb", "tlb", "trb"
 * where:
 *   - first letter:  b = bottom (y_min) / t = top   (y_max)
 *   - second letter: l = left   (x_min) / r = right (x_max)
 *   - third letter:  f = front  (z_min) / b = back  (z_max)
 */
struct CornerInfo3D
{
    std::string label;
    std::array<double, 3> coord = {0.0, 0.0, 0.0};
    // Global TDOF indices of the x, y, z displacement components.
    // Set to -1 if not owned on this rank (after AllGather merging this
    // should never be -1 if the corner is in the global mesh).
    int gtdof_x = -1;
    int gtdof_y = -1;
    int gtdof_z = -1;

    /// Convenience accessor returning all three component TDOFs.
    std::array<int, 3> GTDofs() const noexcept {
        return {gtdof_x, gtdof_y, gtdof_z};
    }
};

/**
 * @brief One of the 12 boundary edges of a box-shaped RVE.
 *
 * @details A 3D box RVE has exactly 12 edges. The edge mortar
 * (architecture §11.5) couples parallel edges in periodic groups of 4
 * (one mortar + 3 nonmortars per spatial direction). Each edge
 * carries line-2 boundary elements with Wohlmuth corner modification
 * at its two corner endpoints.
 *
 * The `elements` vector encodes the 1D line-2 connectivity along the
 * edge. Each entry is a `(node_a_idx, node_b_idx)` pair where:
 *   - non-negative indices point into the `coords` row index (the
 *     i-th interior node)
 *   - `kEdgeNodeLeftCornerSentinel`  (= -1) marks the corner at edge_min
 *   - `kEdgeNodeRightCornerSentinel` (= -2) marks the corner at edge_max
 *
 * For an edge with N interior nodes, the connectivity is:
 * `{(-1, 0), (0, 1), ..., (N-2, N-1), (N-1, -2)}` — i.e. N+1 elements
 * total, two of which touch a corner.
 */
struct EdgeInfo3D
{
    std::string label;        ///< e.g. "x-bottom-front" — see classifier
    /// True iff this is the mortar edge (the side that does NOT carry
    /// the LM rows) in its periodic 4-group. The other 3 are nonmortar.
    bool is_mortar = false;
    std::string parametric_axis;  ///< "x", "y", or "z"
    double edge_min = 0.0;
    double edge_max = 1.0;

    // Reference-frame coordinates of N interior edge nodes, sorted ascending
    // along the parametric axis.
    //   Stored as (N, 3) using `mfem::DenseMatrix` for natural integration
    //   with the rest of the C++ codebase (vs. Python's (N, 3) np.ndarray).
    mfem::DenseMatrix coords;     // (N, 3); column-major, indexed (i, j) for node i, axis j

    // Global TDOF indices for each component at each interior node.
    //   gtdofs_x[i] is the global TDOF for the x-component at node i.
    mfem::Array<int> gtdofs_x;
    mfem::Array<int> gtdofs_y;
    mfem::Array<int> gtdofs_z;

    // Line-2 element connectivity (see comment block above).
    std::vector<std::pair<int, int>> elements;

    // Labels of the two CornerInfo3D instances bounding this edge — used
    // for crosspoint-modification look-ups during constraint assembly.
    std::string corner_min_label;
    std::string corner_max_label;

    /// Number of interior nodes on this edge (excluding corners).
    int NumNodes() const { return coords.NumRows(); }

    /// Coordinate of the i-th interior node along this edge's parametric axis.
    /// Convenience accessor used by MortarAssembler2D.
    double NodeParam(int i) const {
        const int axis_idx = ParamAxisColumn();
        return coords(i, axis_idx);
    }

    /// Mapping from parametric_axis label to coords-column index. Used by the
    /// mortar assembler to extract the parametric coord from a 3D vertex.
    /// Throws on invalid input.
    int ParamAxisColumn() const {
        if (parametric_axis == "x") { return 0; }
        if (parametric_axis == "y") { return 1; }
        if (parametric_axis == "z") { return 2; }
        MFEM_ABORT("EdgeInfo3D: unknown parametric_axis '" << parametric_axis
                      << "'; expected one of {x, y, z}.");
        return -1;  // unreachable
    }
};

// ============================================================================
// Face elements — per-element data consumed by FaceMortarAssembler3D
// ============================================================================

/// A single 4-node face element on a periodic boundary face.
///
/// Local node numbering follows the standard quad-4 convention:
///
///     node 3 ---- node 2     local axes:  xi  ∈ [-1, +1] (axis 0 of parametric_axes)
///       |           |                     eta ∈ [-1, +1] (axis 1 of parametric_axes)
///       |           |
///     node 0 ---- node 1
///                              ordering: ccw viewed from outward normal of
///                              the nonmortar face (so that the Jacobian is
///                              positive)
///
/// `boundary_tag` is a Wohlmuth dual-basis selector. Possible values
/// (mirror of types_3d.py):
///   "none"          : interior face element, standard dual.
///   "edge-xi-low"   : eta-low/-high or xi-low/-high — one element edge
///   "edge-xi-high"    coincides with a face-boundary edge.
///   "edge-eta-low"
///   "edge-eta-high"
///   "corner-LL"     : a corner of this element coincides with a face corner.
///   "corner-LR"       (LL = local node 0; LR = node 1; UR = node 2; UL = node 3.)
///   "corner-UR"
///   "corner-UL"
struct QuadFaceElement
{
    mfem::DenseMatrix coords;        ///< (4, 3): physical coords of corners 0..3
    std::array<int, 4> gtdofs = {-1, -1, -1, -1};
    std::array<std::string, 2> parametric_axes = {"", ""};
    std::string perpendicular_axis;
    std::string boundary_tag = "none";

    static constexpr int NumNodes() { return 4; }

    /// True if any of the 4 nodes is a corner sentinel (=-1).
    bool HasCornerNode() const {
        for (int v : gtdofs) { if (v == kGtdofCornerSentinel) { return true; } }
        return false;
    }
    /// True if any of the 4 nodes is an edge sentinel (=-2).
    bool HasEdgeNode() const {
        for (int v : gtdofs) { if (v == kGtdofEdgeSentinel) { return true; } }
        return false;
    }
};

/// A single 3-node face element on a periodic boundary face.
///
/// Local node numbering: barycentric coordinates λ_1, λ_2, λ_3 with
/// λ_1 at vertex 0, λ_2 at vertex 1, λ_3 at vertex 2. Vertices are
/// listed in CCW order viewed from the outward normal of the nonmortar
/// face (so the Jacobian is positive).
///
/// `boundary_tag` for tri-3:
///   "none"            : no vertex on face boundary, standard dual.
///   "v0" / "v1" / "v2": one vertex at a face corner; that vertex's
///                       row is dropped (it's a CornerInfo3D dof).
///   "v0-v1" / "v0-v2" / "v1-v2": two vertices on a face edge;
///                       two rows dropped.
struct TriFaceElement
{
    mfem::DenseMatrix coords;        ///< (3, 3): physical coords of vertices
    std::array<int, 3> gtdofs = {-1, -1, -1};
    std::array<std::string, 2> parametric_axes = {"", ""};
    std::string perpendicular_axis;
    std::string boundary_tag = "none";

    static constexpr int NumNodes() { return 3; }

    bool HasCornerNode() const {
        for (int v : gtdofs) { if (v == kGtdofCornerSentinel) { return true; } }
        return false;
    }
    bool HasEdgeNode() const {
        for (int v : gtdofs) { if (v == kGtdofEdgeSentinel) { return true; } }
        return false;
    }
};

/**
 * @brief One of the 6 boundary faces of a box-shaped RVE.
 *
 * @details A 3D box RVE has exactly 6 faces. The face mortar
 * (architecture §11.6) couples opposite faces in 3 periodic pairs
 * (one direction each).
 *
 * For mixed hex-tet RVEs (architecture §11.4), a single face may
 * contain both quad-4 and tri-3 face elements; the constraint builder
 * filters and dispatches per-element-type.
 */
struct FaceInfo3D
{
    std::string label;            ///< "bottom" (y_min), "top" (y_max), "left" (x_min),
                                            ///< "right" (x_max), "front" (z_min), "back" (z_max)
    /// True iff this is the mortar face (the side that does NOT carry
    /// the LM rows) in its periodic pair.
    bool is_mortar = false;
    std::string perpendicular_axis;
    double plane_value = 0.0;
    std::array<std::string, 2> parametric_axes = {"", ""};

    int n_quad_elements = 0;
    int n_tri_elements  = 0;

    // Heterogeneous list of face elements. We store quads and tris in
    // separate vectors (vs. Python's heterogeneous list) so the constraint
    // builder can iterate type-homogeneously without runtime polymorphism.
    std::vector<QuadFaceElement> quad_elements;
    std::vector<TriFaceElement>  tri_elements;

    // Face-interior global TDOFs (excluding edges and corners). The
    // face-mortar LM rows correspond to these.
    mfem::Array<int> interior_gtdofs_x;
    mfem::Array<int> interior_gtdofs_y;
    mfem::Array<int> interior_gtdofs_z;

    // Labels of the four EdgeInfo3D instances bounding this face — used to
    // look up edge DOFs for the §5.2 / §5.3 Wohlmuth modifications dropping
    // edge LM rows.
    std::vector<std::string> bounding_edge_labels;

    /// Total face-element count (quads + tris).
    int NumElements() const {
        return n_quad_elements + n_tri_elements;
    }

    /// Mapping from perpendicular_axis label to the 0/1/2 column index.
    int PerpAxisColumn() const {
        if (perpendicular_axis == "x") { return 0; }
        if (perpendicular_axis == "y") { return 1; }
        if (perpendicular_axis == "z") { return 2; }
        MFEM_ABORT("FaceInfo3D: unknown perpendicular_axis '"
                      << perpendicular_axis << "'");
        return -1;
    }
};

/**
 * @brief Assembled mortar quantities for one nonmortar/mortar face pair.
 *
 * @details 3D analog of MortarBlock2D (in mortar_assembler_2d.hpp).
 * The pair-level result has rows indexed by *kept* nonmortar gtdofs
 * and columns indexed by *kept* mortar gtdofs (sentinel rows/cols
 * dropped during assembly).
 *
 * Naming convention follows the Lopes paper and the Wohlmuth-mortar
 * literature: the **nonmortar** side carries the Lagrange-multiplier
 * rows (the "+" / "n" superscript on \f$D^{nm}\f$); the **mortar**
 * side provides the values that feed into the constraint (the "−" /
 * "m" superscript on \f$A^m\f$).
 */
struct FaceMortarPairBlock
{
    /// Mortar coupling matrix: A_m[k, l] = ∫_Γ M_k(ξ) N^mortar_l(Π(ξ)) dA.
    ///
    /// Phase 4.2 / Batch L: stored as `mfem::SparseMatrix` rather
    /// than `mfem::DenseMatrix`. For conforming-mesh face mortars,
    /// each nonmortar node connects to a small number of mortar
    /// nodes (at most 16 for hex8 — the union of mortar nodes from
    /// all matched element pairs touching that nonmortar node).
    /// Dense storage is therefore a factor of O(n_m) too large; at
    /// production scale (n_m ≈ 10⁴) this is the dominant memory
    /// term.
    ///
    /// Lifecycle: producers (`AssemblePairConforming`) construct
    /// `A_m` in build mode (`mfem::SparseMatrix(n_rows, n_cols)`),
    /// `Add()` entries during integration, and call `Finalize()`
    /// before returning. Consumers may use `operator()(i, j)` (slow)
    /// or walk the CSR arrays via `GetI()`, `GetJ()`, `GetData()`
    /// (fast). `Finalize` is idempotent — calling it on an already-
    /// finalized matrix is a no-op.
    mfem::SparseMatrix A_m;
    /// Diagonal lumping vector: D[k] = ∫_Γ N^nonmortar_k dA.
    /// Stored as 1D since D is diagonal in the dual basis.
    mfem::Vector D;

    std::string nonmortar_face_name;
    std::string mortar_face_name;

    /// Global TDOFs (primary component) of the kept nonmortar rows.
    mfem::Array<int> nonmortar_gtdofs;
    /// Global TDOFs (primary component) of the kept mortar cols.
    mfem::Array<int> mortar_gtdofs;

    /// Number of kept nonmortar rows in this block.
    int NumNonmortarKept() const { return nonmortar_gtdofs.Size(); }
    /// Number of kept mortar cols in this block.
    int NumMortarKept() const { return mortar_gtdofs.Size(); }
};

}  // namespace mortar_pbc
