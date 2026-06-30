"""Pure-Python data containers for the 3D mortar PBC machinery.

WHAT
----
Three dataclasses, mirroring the 2D types in ``types_2d.py`` but for the
3D wirebasket hierarchy (§5.4 of MORTAR_PBC_ARCHITECTURE.md):

    * ``CornerInfo3D`` : one of the 8 corner nodes of a 3D box-shaped RVE.
                         Used in Phase 3.1+.
    * ``EdgeInfo3D``   : one of the 12 boundary edges of a 3D RVE, with
                         its interior-node coords, global true-DOF
                         indices, and 1D element connectivity (with
                         corner sentinels). Used in Phase 3.3+.
    * ``FaceInfo3D``   : one of the 6 boundary faces of a 3D RVE. Carries
                         either quad-4 or tri-3 face elements (or a mix
                         for hex+tet meshes). Used in Phase 3.3+.

WHY
---
Same rationale as ``types_2d.py``: isolate the data contracts in an
MFEM-/MPI-free module so the mortar machinery (mortar matrix assembly,
constraint construction) can be unit-tested without pyMFEM installed.

Phase 3.1 only uses ``CornerInfo3D``; ``EdgeInfo3D`` and ``FaceInfo3D``
are stubbed here for forward compatibility but consumed only by
``boundary_3d.py`` and ``constraint_builder_3d.py`` in Phase 3.3.

WHO PRODUCES THEM
-----------------
``BoundaryClassifier3D`` (Phase 3.3, MFEM-dependent) builds these from a
``ParMesh`` + ``ParFiniteElementSpace``. Test code can construct them
directly with synthetic data.

REFERENCES
----------
* MORTAR_PBC_ARCHITECTURE.md §5.4 (3D wirebasket hierarchy).
* MORTAR_PBC_ARCHITECTURE.md §11.7 (BoundaryClassifier3D design).
* ExaConstit boundary-attribute convention (3D layout from
  ``setBdrConditions`` in ``src/sim_state/simulation_state.cpp``):
    1 = bottom (y = y_min)
    2 = front  (z = z_min)
    3 = right  (x = x_max)
    4 = back   (z = z_max)
    5 = left   (x = x_min)
    6 = top    (y = y_max)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np


# =============================================================================
# Corner: a 0-dim feature, used in Phase 3.1+
# =============================================================================

@dataclass
class CornerInfo3D:
    """A single corner node of a 3D box-shaped RVE.

    A 3D box RVE has exactly 8 corners. Under Method D PBC (§2 of the
    architecture doc), each corner is essentially Dirichlet-prescribed
    at u_lin[corner] = (F_macro - I) X[corner], where X[corner] is the
    reference-frame corner coordinate. The 8 corners pin the rigid-body
    modes (3 translations + 3 rotations) plus the linear-affine
    macroscopic part of the deformation — the LM rows for these DOFs
    are dropped by the Wohlmuth modification (§5.1 / §5.2 / §5.3).

    Attributes
    ----------
    label : str
        One of "blf" (bottom-left-front), "brf", "tlf", "trf",
                "blb" (bottom-left-back),  "brb", "tlb", "trb".
        First letter:  b = bottom  (y = y_min)  / t = top   (y = y_max)
        Second letter: l = left    (x = x_min)  / r = right (x = x_max)
        Third letter:  f = front   (z = z_min)  / b = back  (z = z_max)
    coord : (3,) float64 ndarray
        Physical reference-frame coordinates of the corner.
    gtdof_x, gtdof_y, gtdof_z : int
        Global true-DOF indices of the x, y, z displacement components.
        Set to -1 if not owned on this rank (after AllGather merging
        this should never be -1 if the corner is in the global mesh).
    """
    label: str
    coord: np.ndarray
    gtdof_x: int
    gtdof_y: int
    gtdof_z: int

    @property
    def gtdofs(self) -> Tuple[int, int, int]:
        """All three component TDOFs as a tuple (convenience)."""
        return (self.gtdof_x, self.gtdof_y, self.gtdof_z)


# =============================================================================
# Edge: a 1D feature, used in Phase 3.3+
# =============================================================================

@dataclass
class EdgeInfo3D:
    """A single boundary edge of a 3D box-shaped RVE, corners excluded.

    A 3D box RVE has exactly 12 edges. The edge mortar (§11.5) couples
    parallel edges in periodic groups of 4 (one mortar + 3 nonmortars per
    spatial direction). Each edge carries line-2 boundary elements with
    Wohlmuth corner modification at its two corner endpoints.

    Phase 3.3 will populate these from ``BoundaryClassifier3D``; Phase
    3.1 ignores them entirely (Phase 3.1 has no mortar coupling).

    Attributes
    ----------
    label : str
        Identifier, e.g. "bl-y" (bottom-left edge, parallel to y).
        Twelve possible labels; convention: "{face1}{face2}-{axis}"
        where the two faces meet at this edge and `axis` ∈ {x, y, z}
        is the direction along the edge.
    is_mortar : bool
        True iff this edge is the mortar in its periodic group of 4.
        Each direction has exactly one mortar and three nonmortars.
    parametric_axis : str
        "x", "y", or "z" — the spatial direction of the edge.
    edge_min, edge_max : float
        Extent of the edge along ``parametric_axis``.
    coords : (N, 3) float64 ndarray
        Reference-frame coordinates of the N interior edge nodes
        (corners excluded), sorted ascending along ``parametric_axis``.
    gtdofs_x, gtdofs_y, gtdofs_z : (N,) int64 ndarrays
        Global true-DOF indices for each component at each interior
        node. -1 = not owned on this rank.
    elements : list[(int, int)]
        1D line-2 connectivity along the edge with corner sentinels:
            -1 = "left  corner" (= edge_min along parametric_axis)
            -2 = "right corner" (= edge_max along parametric_axis)
        For an edge with N interior nodes, the connectivity is:
            (-1, 0), (0, 1), ..., (N-2, N-1), (N-1, -2)
        i.e. N+1 elements total, two of which touch a corner.
    corner_min_label, corner_max_label : str
        Labels of the two ``CornerInfo3D`` instances that bound this
        edge. Used to look up the corner DOFs for crosspoint
        modifications.
    """
    label: str
    is_mortar: bool
    parametric_axis: str
    edge_min: float
    edge_max: float
    coords: np.ndarray
    gtdofs_x: np.ndarray
    gtdofs_y: np.ndarray
    gtdofs_z: np.ndarray
    elements: List[Tuple[int, int]] = field(default_factory=list)
    corner_min_label: str = ""
    corner_max_label: str = ""

    @property
    def n_nodes(self) -> int:
        """Number of *interior* nodes on this edge (corners excluded)."""
        return self.coords.shape[0]


# =============================================================================
# Face: a 2D feature, used in Phase 3.3+
# =============================================================================

@dataclass
class FaceInfo3D:
    """A single boundary face of a 3D box-shaped RVE, edges excluded.

    A 3D box RVE has exactly 6 faces. The face mortar (§11.6) couples
    opposite faces in 3 periodic pairs (one direction each).

    For mixed hex-tet RVEs (§11.4), a single face may contain both
    quad-4 elements (from hex volumes) and tri-3 elements (from tet
    volumes). The face element groupings are stored separately so the
    polymorphic ``MortarFaceAssembler`` (§11.4) can dispatch per-element
    on ``GetGeometryType()``.

    Phase 3.3 architecture revision (§11.7 of architecture doc): expose
    each face as a ``mfem.ParSubMesh`` extracted via
    ``ParSubMesh.CreateFromBoundary``. The submesh handles MPI
    distribution natively and pre-groups face elements by geometry
    type. The fields below are kept for downstream consumers that
    prefer raw arrays; both the submesh and the arrays are populated
    by ``BoundaryClassifier3D``.

    Phase 3.1 ignores this entirely.

    Attributes
    ----------
    label : str
        One of "bottom" (y_min), "top" (y_max), "left" (x_min),
        "right" (x_max), "front" (z_min), "back" (z_max).
    is_mortar : bool
        True iff this face is the mortar in its periodic pair.
        Convention: bottom, left, front are mortars; top, right, back
        are nonmortars.
    perpendicular_axis : str
        "x", "y", or "z" — the axis perpendicular to the face. Periodic
        translation Π acts along this axis.
    plane_value : float
        The constant value of the perpendicular coordinate on this
        face (e.g. y_min for "bottom").
    parametric_axes : tuple[str, str]
        Two-letter pair giving the in-face coordinate axes.
        E.g. ("x", "z") for "bottom" and "top".
    n_quad_elements : int
        Number of quad-4 face elements on this face (from hex volumes).
    n_tri_elements : int
        Number of tri-3 face elements on this face (from tet volumes).
    submesh : Optional[object]
        ``mfem.ParSubMesh`` of this face's boundary attribute. None
        until populated by ``BoundaryClassifier3D``. Marked optional
        because the dataclass must remain importable in pyMFEM-free
        environments (unit tests).
    interior_gtdofs_x, interior_gtdofs_y, interior_gtdofs_z : np.ndarray
        Face-interior global TDOFs (excluding edges and corners). The
        face-mortar LM rows correspond to these.
    bounding_edge_labels : list[str]
        Labels of the four ``EdgeInfo3D`` instances that bound this
        face. Used to look up edge DOFs for the §5.2 / §5.3 Wohlmuth
        modifications dropping edge LM rows.
    """
    label: str
    is_mortar: bool
    perpendicular_axis: str
    plane_value: float
    parametric_axes: Tuple[str, str]
    n_quad_elements: int = 0
    n_tri_elements: int = 0
    # ``submesh``: optional reference to the parent ParSubMesh used to
    # build this face. Held only when downstream code (e.g. transfer
    # of grid functions) needs it; for pure-Python constraint
    # assembly the ``face_elements`` list is sufficient and ``submesh``
    # may be left None.
    submesh: Optional[object] = None
    # ``face_elements``: list of per-element face data consumed by the
    # Phase 3.2.B face-mortar assemblers. Mixed-element faces (hex+tet,
    # §11.4) carry a heterogeneous list of QuadFaceElement and
    # TriFaceElement; the constraint builder filters by element type
    # and dispatches to the appropriate concrete assembler.
    face_elements: List[object] = field(default_factory=list)
    interior_gtdofs_x: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    interior_gtdofs_y: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    interior_gtdofs_z: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    bounding_edge_labels: List[str] = field(default_factory=list)


# =============================================================================
# Face elements: per-element data consumed by MortarFaceAssembler (Phase 3.2.B+)
# =============================================================================
#
# These are the unit on which face-mortar integration operates. One
# QuadFaceElement / TriFaceElement per face element on the nonmortar or mortar
# side of a periodic face pair. The MFEM-free design means tests can build
# them from synthetic data without pyMFEM.
#
# Sentinel convention for boundary-feature row/column dropping
# ------------------------------------------------------------
# Each face-element node carries a global TDOF index (per spatial component).
# When the node has been classified as belonging to a *higher* level of the
# wirebasket hierarchy (corner or edge), the gtdof is replaced by a sentinel:
#
#     gtdof >= 0  : face-interior DOF — kept in D and A^m row/col.
#     gtdof == -1 : corner DOF — Dirichlet-pinned at u_lin per Method-D §2.2.
#                    Row dropped (nonmortar side); col dropped (mortar side); the
#                    corresponding constraint contribution is NOT added to
#                    the RHS because the corner pin is enforced at the primal
#                    level via EliminateRowsCols, not at the constraint level.
#     gtdof == -2 : edge DOF — constrained by 1D edge mortar (§11.5).
#                    Row dropped (nonmortar); col dropped (mortar); the edge
#                    mortar block handles this DOF's periodicity.
#
# This mirrors `MortarAssembler2D._integrate_overlap_segment`
# (mortar_2d.py:396-414) and the §5.4 wirebasket hierarchy: corners pin
# rigid-body + affine modes, edges handle 1D periodicity, faces handle the
# remaining 2D periodicity on face-interior nodes only.
#
# Boundary tag for Wohlmuth-modified dual basis selection
# -------------------------------------------------------
# The `boundary_tag` field tells the assembler which Wohlmuth modification
# of the nonmortar-side dual basis to use. Possible values:
#
#     "none"          : interior face element, standard dual.
#     "edge-{loc}"    : one edge of this element coincides with a face-
#                        boundary edge. {loc} ∈ {"xi-low", "xi-high",
#                        "eta-low", "eta-high"} for quad-4, or {"v0", "v1",
#                        "v2"} for tri-3 to identify which local-frame
#                        feature is the boundary.
#     "corner-{loc}"  : a corner of this element coincides with a face
#                        corner. {loc} encodes the corner index.
#
# These tags translate directly to the `side_xi`/`side_eta` arguments of
# `M_quad4_dual_modified` and the `boundary_nodes` argument of
# `M_tri3_dual_modified`. The translation is done inside the concrete
# `QuadFaceMortarAssembler` / `TriFaceMortarAssembler` subclasses.

@dataclass
class QuadFaceElement:
    """A single 4-node face element on a periodic boundary face.

    Local node numbering follows the standard quad-4 convention:

        node 3 ---- node 2     local axes:  xi  ∈ [-1, +1] (axis 0 of parametric_axes)
          |           |                     eta ∈ [-1, +1] (axis 1 of parametric_axes)
          |           |
        node 0 ---- node 1
                                ordering: ccw viewed from outward normal of nonmortar face
                                (so that the Jacobian is positive)

    For a face on x = 0 with parametric_axes = ("y", "z"), the outward
    normal is -x, and the CCW ordering is taken viewed from -x (i.e.
    looking at the face from outside the RVE).

    Attributes
    ----------
    coords : (4, 3) float64 ndarray
        Physical reference-frame coordinates of the 4 corner nodes in
        local-node order (0 -> 1 -> 2 -> 3).
    gtdofs : (4,) tuple of int
        Global TDOFs of the *primary* spatial component for each local
        node. Sentinels: -1 = corner DOF, -2 = edge DOF (see header).
        The constraint builder expands these to per-component TDOFs at
        global-C-assembly time.
    parametric_axes : (str, str)
        Pair of axis labels giving the two parametric dimensions of the
        face. E.g. ("x", "z") for a y-perpendicular face.
    perpendicular_axis : str
        Axis label of the face normal. E.g. "y" for the bottom/top pair.
    boundary_tag : str
        Wohlmuth dual-basis selector. One of {"none", "edge-xi-low",
        "edge-xi-high", "edge-eta-low", "edge-eta-high", "corner-{0..3}",
        ...}. See module header.
    """
    coords: np.ndarray
    gtdofs: Tuple[int, int, int, int]
    parametric_axes: Tuple[str, str]
    perpendicular_axis: str
    boundary_tag: str = "none"

    @property
    def n_nodes(self) -> int:
        return 4

    @property
    def jacobian_axis_aligned(self) -> float:
        """Constant Jacobian for an axis-aligned rectangular face element.

        For an axis-aligned rectangular quad-4 with reference [-1,+1]^2
        and physical extents (Δa, Δb) along its two parametric axes,
        the Jacobian determinant is constant: |J| = (Δa/2) · (Δb/2).
        Useful for the Phase 3.2.B conforming-pair tests where
        MakeCartesian3D produces axis-aligned face elements.

        Returns NaN if the element is not axis-aligned (a non-trivial
        bilinear-quad Jacobian must be computed point-by-point in
        general; subclass `_nonmortar_jacobian` handles this case).
        """
        # Identify the two parametric axes' indices.
        axis_idx = {"x": 0, "y": 1, "z": 2}
        a_idx = axis_idx[self.parametric_axes[0]]
        b_idx = axis_idx[self.parametric_axes[1]]
        # Extents along each parametric axis.
        a_lo = float(self.coords[:, a_idx].min())
        a_hi = float(self.coords[:, a_idx].max())
        b_lo = float(self.coords[:, b_idx].min())
        b_hi = float(self.coords[:, b_idx].max())
        # Check axis-aligned: 2 distinct values per parametric axis.
        a_vals = np.unique(np.round(self.coords[:, a_idx], 12))
        b_vals = np.unique(np.round(self.coords[:, b_idx], 12))
        if len(a_vals) != 2 or len(b_vals) != 2:
            return float("nan")
        return 0.25 * (a_hi - a_lo) * (b_hi - b_lo)


@dataclass
class TriFaceElement:
    """A single 3-node face element on a periodic boundary face.

    Local node numbering: barycentric coordinates λ_1, λ_2, λ_3 with
    λ_1 at vertex 0, λ_2 at vertex 1, λ_3 at vertex 2. Vertices are
    listed in CCW order viewed from the outward normal of the nonmortar
    face (so the Jacobian is positive).

    Attributes
    ----------
    coords : (3, 3) float64 ndarray
        Physical reference-frame coordinates of the 3 vertex nodes.
    gtdofs : (3,) tuple of int
        Global TDOFs of the primary spatial component. Sentinels:
        -1 = corner DOF, -2 = edge DOF. (See module header.)
    parametric_axes : (str, str)
        In-face axis labels.
    perpendicular_axis : str
        Face-normal axis label.
    boundary_tag : str
        Wohlmuth selector. For tri-3:
            "none"            : no vertex on face boundary, standard dual.
            "v0" / "v1" / "v2": one vertex at a face corner; that vertex's
                                row is dropped (it's a CornerInfo3D dof).
            "v0-v1" / "v0-v2" / "v1-v2": two vertices on a face edge;
                                two rows dropped.
        These tags route to `M_tri3_dual_modified` with the matching
        `boundary_nodes` set.
    """
    coords: np.ndarray
    gtdofs: Tuple[int, int, int]
    parametric_axes: Tuple[str, str]
    perpendicular_axis: str
    boundary_tag: str = "none"

    @property
    def n_nodes(self) -> int:
        return 3

    @property
    def physical_area(self) -> float:
        """|T| = ½ |(P1 - P0) × (P2 - P0)| projected onto the face plane.

        For an axis-aligned tri-3 face element on a face perpendicular
        to one cardinal axis, this is the in-plane triangle area.
        """
        v01 = self.coords[1] - self.coords[0]
        v02 = self.coords[2] - self.coords[0]
        cross = np.cross(v01, v02)
        return 0.5 * float(np.linalg.norm(cross))


# =============================================================================
# Face mortar pair block: result of one nonmortar-mortar face pair assembly
# =============================================================================

@dataclass
class FaceMortarPairBlock:
    """Assembled mortar quantities for one (nonmortar, mortar) face pair.

    The 3D analog of ``MortarBlock2D`` — see the 2D version for the
    semantics of ``D`` and ``A_m``. The pair-level result is stored
    with row indexing by *kept* nonmortar gtdofs and column indexing by
    *kept* mortar gtdofs (sentinel rows/cols are dropped during
    assembly).

    Attributes
    ----------
    A_m : (n_nonmortar_kept, n_mortar_kept) float64 ndarray
        Mortar coupling matrix, ``A_m[k, l] = ∫_Γ⁻ M_k(ξ) N^mortar_l(Π(ξ)) dA``.
    D : (n_nonmortar_kept,) float64 ndarray
        Diagonal lumping vector, ``D[k] = ∫_Γ⁻ N^nonmortar_k dA``.
        Stored as 1D (D is diagonal in the dual basis).
    nonmortar_face_name : str
        Name of the nonmortar face (e.g. "bottom").
    mortar_face_name : str
        Name of the mortar face (e.g. "top").
    nonmortar_gtdofs : (n_nonmortar_kept,) int64 ndarray
        Global TDOFs (primary component) of the kept nonmortar rows.
    mortar_gtdofs : (n_mortar_kept,) int64 ndarray
        Global TDOFs (primary component) of the kept mortar cols.
    """
    A_m: np.ndarray
    D: np.ndarray
    nonmortar_face_name: str
    mortar_face_name: str
    nonmortar_gtdofs: np.ndarray
    mortar_gtdofs: np.ndarray
