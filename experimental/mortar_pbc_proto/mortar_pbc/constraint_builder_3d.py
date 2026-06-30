"""3D mortar-PBC constraint matrix builder — Phase 3.3.C.

WHAT
----
``ConstraintBuilder3D`` consumes a ``BoundaryClassifier3D`` (Phase
3.3.B) plus the three element-type-specific assemblers (Phases 3.2.B
and 3.3.A) and produces the global mortar-periodic constraint matrix
``C`` as a SciPy CSR sparse matrix.

The constraint matrix has shape ``(n_constraint_rows, n_global_tdofs)``
and encodes Eq. (1.1) of MORTAR_PBC_ARCHITECTURE.md: for each "kept"
nonmortar-side DOF index ``k`` and each spatial component ``c``,

    C[(k, c), :] · u  =  D[k] u_nonmortar_c[k]  -  Σ_l A_m[k, l] u_mortar_c[l]
                       =  0   (nonmortar/mortar coupling)

WHY
---
This is the orchestration layer that ties together:

  * The 3D edge mortar (9 pairs: 3 axes × 3 nonmortar edges per axis,
    paired against 1 mortar edge per axis) — uses
    ``MortarAssembler2D.assemble_pair`` with the Phase 3.3.A axis-
    generic dispatch on ``EdgeInfo3D``.
  * The 3D face mortar (3 pairs: 1 per axis) — uses the polymorphic
    ``QuadFaceMortarAssembler`` and ``TriFaceMortarAssembler`` from
    Phase 3.2.B. Mixed hex+tet faces dispatch by element type and
    accumulate row-stacked.

Stacking these into one global C lets the saddle-point solve (already
in place from the 2D Phase 1B work) pick up the 3D periodicity without
any further structural change.

DESIGN NOTES
------------
* **Pure-Python.** No MFEM dependency. Same separation of concerns as
  Phase 3.2.B: the classifier (Phase 3.3.B) holds the MFEM-touching
  bits; this builder works off the classifier's pure-Python output.

* **vdim=3 expansion is explicit.** The mortar blocks (both edge and
  face) operate on scalar gtdofs (one entry per node). Each scalar
  constraint expands to 3 vector-component constraints by replicating
  the row across the (x, y, z) gtdofs of the same node. The
  classifier's ``gtdof_xyz_lookup()`` provides the
  ``primary_gtdof → (gx, gy, gz)`` map needed for this expansion.

* **Sentinel handling is already done by the classifier.** Per Phase
  3.3.B, the per-face-element gtdofs and the per-edge-interior gtdofs
  arrive with corner DOFs (-1) and edge DOFs (-2) already stripped
  (faces) or already excluded (edges, by construction since edge
  records hold only edge-interior nodes). The Phase 3.2.B face
  assembler returns ``FaceMortarPairBlock`` with sentinel rows/cols
  ALREADY DROPPED. So this builder treats every gtdof as a real,
  positive global TDOF index.

* **CSR replicated on every rank.** Same convention as
  ``ConstraintBuilder2D``: every rank has the same global C, sized
  ``(n_constraints, n_global_tdofs)``. The downstream saddle-point
  solver (``SaddlePointSolver`` from Phase 1B) picks up the
  appropriate rows by row-ownership splits.

* **Empty-block tolerance.** A face mortar/nonmortar pair may have only
  quad elements (hex mesh) or only tri elements (tet mesh). The
  builder dispatches based on the actual element types present on
  each face — it doesn't blindly call both assemblers. For mixed
  meshes (Phase 3.5+) both assemblers run and their blocks are
  row-stacked.

REFERENCES
----------
* MORTAR_PBC_ARCHITECTURE.md §11.8 Phase 3.3.C (this layer).
* MORTAR_PBC_ARCHITECTURE.md §11.5 (3D edge mortar).
* MORTAR_PBC_ARCHITECTURE.md §11.6 (face mortar geometric matching).
* mortar_pbc/constraint_builder.py — ``ConstraintBuilder2D``, the
  pattern this layer generalises.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .face_mortar_3d import (
    QuadFaceMortarAssembler,
    TriFaceMortarAssembler,
    match_conforming_face_pairs,
)
from .mortar_2d import MortarAssembler2D, MortarBlock2D
from .types_3d import (
    FaceInfo3D,
    FaceMortarPairBlock,
    QuadFaceElement,
    TriFaceElement,
)


__all__ = ["ConstraintBuilder3D"]


class ConstraintBuilder3D:
    """Assemble the global mortar-periodic constraint matrix C in CSR form.

    Parameters
    ----------
    classifier : BoundaryClassifier3D
        Output of Phase 3.3.B. Must expose ``edges``, ``faces``,
        ``corners``, ``n_global_tdofs``, ``gtdof_xyz_lookup``,
        ``edge_pairs``, ``face_pairs``.
    edge_assembler : MortarAssembler2D, optional
        2D mortar assembler reused for 3D edges (Phase 3.3.A). If
        omitted, a fresh ``MortarAssembler2D(_DummyClassifier())`` is
        instantiated — the 2D classifier reference is unused by
        ``assemble_pair``, only by the legacy ``assemble_all`` path.
    quad_face_assembler : QuadFaceMortarAssembler, optional
        Phase 3.2.B; instantiated by default if omitted.
    tri_face_assembler : TriFaceMortarAssembler, optional
        Phase 3.2.B; instantiated by default if omitted.
    period : (3,) array-like, optional
        Periodic translation vector for face matching. Defaults to
        ``[L_x, L_y, L_z]`` derived from the classifier's bbox.
    pair_match_tol_rel : float
        Tolerance for ``match_conforming_face_pairs``; default 1e-9.
    """

    VDIM = 3   # 3D vector elasticity

    def __init__(
        self,
        classifier,
        *,
        edge_assembler: Optional[MortarAssembler2D] = None,
        quad_face_assembler: Optional[QuadFaceMortarAssembler] = None,
        tri_face_assembler: Optional[TriFaceMortarAssembler] = None,
        period: Optional[np.ndarray] = None,
        pair_match_tol_rel: float = 1e-9,
    ) -> None:
        self.cl = classifier
        # Lazy default-construct each assembler if not supplied.
        if edge_assembler is None:
            edge_assembler = MortarAssembler2D(_DummyEdgeClassifier())
        self.edge_assembler = edge_assembler
        if quad_face_assembler is None:
            quad_face_assembler = QuadFaceMortarAssembler()
        self.quad_face_assembler = quad_face_assembler
        if tri_face_assembler is None:
            tri_face_assembler = TriFaceMortarAssembler()
        self.tri_face_assembler = tri_face_assembler
        # Period vector for face matching.
        if period is None:
            period = classifier.bbox_max - classifier.bbox_min
        self.period = np.asarray(period, dtype=np.float64)
        self.pair_match_tol_rel = pair_match_tol_rel

        # Cached gtdof lookup: primary x-component gtdof -> (gx, gy, gz).
        self._gtdof_lookup: Dict[int, Tuple[int, int, int]] = (
            classifier.gtdof_xyz_lookup()
        )

    # -------------------------------------------------------------- API ---
    def build(self) -> sp.csr_matrix:
        """Build and return the global constraint matrix C as CSR sparse.

        Layout: edge constraints first (9 pairs), face constraints
        second (3 pairs). Within each pair, rows are vdim-replicated
        per kept nonmortar node.
        """
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        row_offset = 0

        # ===== Edge mortar blocks (9 pairs) =====
        for axis, mortar_label, nonmortar_label in self.cl.edge_pairs():
            mortar_edge = self.cl.edges[mortar_label]
            nonmortar_edge  = self.cl.edges[nonmortar_label]
            block = self.edge_assembler.assemble_pair(nonmortar_edge, mortar_edge)
            row_offset = self._scatter_edge_block(
                block, nonmortar_edge, mortar_edge,
                rows, cols, vals, row_offset,
            )

        # ===== Face mortar blocks (3 pairs) =====
        for axis, mortar_label, nonmortar_label in self.cl.face_pairs():
            mortar_face: FaceInfo3D = self.cl.faces[mortar_label]
            nonmortar_face:  FaceInfo3D = self.cl.faces[nonmortar_label]
            row_offset = self._scatter_face_pair(
                nonmortar_face, mortar_face, axis,
                rows, cols, vals, row_offset,
            )

        n_rows = row_offset
        n_cols = self.cl.n_global_tdofs
        if n_rows == 0:
            return sp.csr_matrix((0, n_cols))
        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_rows, n_cols)
        ).tocsr()

    # ------------------------------------------------------------- counts ---
    def n_constraints(self) -> int:
        """Number of constraint rows the build will emit.

        edges:   sum over 9 pairs of vdim * n_interior_nonmortar_nodes
        faces:   sum over 3 pairs of vdim * n_kept_nonmortar_face_dofs

        For face pairs, the kept-nonmortar count requires running the
        Phase-3.2.B assembler dedup (or pre-counting via the
        classifier's per-face interior_gtdofs_x) — we use the latter
        since it's already computed.
        """
        n = 0
        for axis, mortar_label, nonmortar_label in self.cl.edge_pairs():
            nonmortar_edge = self.cl.edges[nonmortar_label]
            n += self.VDIM * nonmortar_edge.n_nodes
        for axis, mortar_label, nonmortar_label in self.cl.face_pairs():
            nonmortar_face = self.cl.faces[nonmortar_label]
            n += self.VDIM * len(nonmortar_face.interior_gtdofs_x)
        return n

    # ------------------------------------------------------------- internals -
    def _scatter_edge_block(
        self,
        block: MortarBlock2D,
        nonmortar_edge,
        mortar_edge,
        rows: List[int],
        cols: List[int],
        vals: List[float],
        row_offset: int,
    ) -> int:
        """Append rows for one edge mortar block.

        For 3D edges, ``nonmortar_edge`` is a nonmortar EdgeInfo3D in the
        classifier's convention (is_mortar=False, plus_edge in the
        2D mortar's "plus_edge" naming). The mortar assembler returns
        ``D_nm`` indexed by nonmortar-edge interior nodes and ``A_m``
        indexed by (nonmortar, mortar) interior nodes. We replicate per
        spatial component.

        Note: ``MortarAssembler2D.assemble_pair(plus_edge, minus_edge)``
        treats plus_edge as the NONMORTAR side (the edge whose nodes are
        the constraint-row owners). We pass nonmortar_edge as plus and
        mortar_edge as minus to match this convention.
        """
        n_nonmortar  = nonmortar_edge.n_nodes
        n_mortar = mortar_edge.n_nodes

        for k in range(n_nonmortar):
            D_kk = float(block.D_nm[k])
            nonmortar_g_xyz = (
                int(nonmortar_edge.gtdofs_x[k]),
                int(nonmortar_edge.gtdofs_y[k]),
                int(nonmortar_edge.gtdofs_z[k]),
            )
            if D_kk == 0.0:
                # Degenerate row (could happen if a nonmortar node is
                # entirely covered by a corner-modified element).
                # Skip but still consume row indices to keep the
                # vdim-aligned layout.
                row_offset += self.VDIM
                continue

            # Diagonal D entry per component.
            for c in range(self.VDIM):
                gd = nonmortar_g_xyz[c]
                if gd < 0:
                    continue
                rows.append(row_offset + c)
                cols.append(gd)
                vals.append(D_kk)

            # Off-diagonal -A_m entries over mortar interior nodes.
            for l in range(n_mortar):
                A_kl = float(block.A_m[k, l])
                if A_kl == 0.0:
                    continue
                mortar_g_xyz = (
                    int(mortar_edge.gtdofs_x[l]),
                    int(mortar_edge.gtdofs_y[l]),
                    int(mortar_edge.gtdofs_z[l]),
                )
                for c in range(self.VDIM):
                    gd = mortar_g_xyz[c]
                    if gd < 0:
                        continue
                    rows.append(row_offset + c)
                    cols.append(gd)
                    vals.append(-A_kl)

            row_offset += self.VDIM
        return row_offset

    def _scatter_face_pair(
        self,
        nonmortar_face: FaceInfo3D,
        mortar_face: FaceInfo3D,
        axis: str,
        rows: List[int],
        cols: List[int],
        vals: List[float],
        row_offset: int,
    ) -> int:
        """Run the appropriate face-mortar assembler(s) on this pair
        and append rows.

        Mixed-element faces (hex+tet) run both assemblers; their
        blocks are row-stacked (the kept-nonmortar gtdofs may overlap if
        a nonmortar node is shared by quads and tris, in which case both
        assemblers will emit a row for it — they integrate over their
        own element subset and the row-stacking gives the right
        union-of-supports constraint).
        """
        # Period vector signed for nonmortar→mortar direction.
        ax_idx = {"x": 0, "y": 1, "z": 2}[axis]
        period_signed = float(
            mortar_face.plane_value - nonmortar_face.plane_value
        )

        # Partition each face's elements by geometry type.
        nonmortar_quads = [e for e in nonmortar_face.face_elements
                       if isinstance(e, QuadFaceElement)]
        nonmortar_tris  = [e for e in nonmortar_face.face_elements
                       if isinstance(e, TriFaceElement)]
        mortar_quads = [e for e in mortar_face.face_elements
                        if isinstance(e, QuadFaceElement)]
        mortar_tris  = [e for e in mortar_face.face_elements
                        if isinstance(e, TriFaceElement)]

        # Quad sub-pair (if both sides have quads).
        if nonmortar_quads and mortar_quads:
            pair_matches = match_conforming_face_pairs(
                nonmortar_quads, mortar_quads,
                perpendicular_axis=axis,
                period=period_signed,
                tol_rel=self.pair_match_tol_rel,
            )
            block = self.quad_face_assembler.assemble_pair_conforming(
                nonmortar_elems=nonmortar_quads,
                mortar_elems=mortar_quads,
                pair_matches=pair_matches,
                nonmortar_face_name=nonmortar_face.label,
                mortar_face_name=mortar_face.label,
            )
            row_offset = self._scatter_face_block(
                block, rows, cols, vals, row_offset,
            )

        # Tri sub-pair (if both sides have tris).
        if nonmortar_tris and mortar_tris:
            pair_matches = match_conforming_face_pairs(
                nonmortar_tris, mortar_tris,
                perpendicular_axis=axis,
                period=period_signed,
                tol_rel=self.pair_match_tol_rel,
            )
            block = self.tri_face_assembler.assemble_pair_conforming(
                nonmortar_elems=nonmortar_tris,
                mortar_elems=mortar_tris,
                pair_matches=pair_matches,
                nonmortar_face_name=nonmortar_face.label,
                mortar_face_name=mortar_face.label,
            )
            row_offset = self._scatter_face_block(
                block, rows, cols, vals, row_offset,
            )

        # Mixed cases (nonmortar_quads & mortar_tris, or nonmortar_tris &
        # mortar_quads): only arise on Phase 3.5+ non-conforming
        # mixed meshes where the nonmortar/mortar faces have DIFFERENT
        # element types. For Phase 3.3.C we error out clearly.
        nonmortar_has_both = bool(nonmortar_quads) and bool(nonmortar_tris)
        mortar_has_both = bool(mortar_quads) and bool(mortar_tris)
        nonmortar_quads_mortar_tris = bool(nonmortar_quads) and not mortar_quads
        nonmortar_tris_mortar_quads = bool(nonmortar_tris) and not mortar_tris
        if (nonmortar_quads_mortar_tris and mortar_tris) or \
           (nonmortar_tris_mortar_quads and mortar_quads):
            raise NotImplementedError(
                f"ConstraintBuilder3D: face pair "
                f"{nonmortar_face.label!r} <-> {mortar_face.label!r} has "
                f"asymmetric element types (nonmortar: {len(nonmortar_quads)} "
                f"quads + {len(nonmortar_tris)} tris; mortar: "
                f"{len(mortar_quads)} quads + {len(mortar_tris)} tris). "
                f"Phase 3.3.C handles same-type quad-quad and tri-tri "
                f"pairings; mixed-type is Phase 3.5+."
            )

        return row_offset

    def _scatter_face_block(
        self,
        block: FaceMortarPairBlock,
        rows: List[int],
        cols: List[int],
        vals: List[float],
        row_offset: int,
    ) -> int:
        """Append rows for one face mortar block (already sentinel-stripped
        by the Phase 3.2.B assembler).

        ``block.nonmortar_gtdofs[k]`` is the primary-component (x) gtdof
        of nonmortar node k; we look up the per-component triple via
        ``self._gtdof_lookup``.
        """
        n_nonmortar_kept = block.D.shape[0]
        n_mortar_kept = block.A_m.shape[1]

        for k in range(n_nonmortar_kept):
            D_kk = float(block.D[k])
            nonmortar_gx = int(block.nonmortar_gtdofs[k])
            nonmortar_g_xyz = self._gtdof_lookup.get(nonmortar_gx)
            if nonmortar_g_xyz is None:
                raise RuntimeError(
                    f"ConstraintBuilder3D: nonmortar gtdof {nonmortar_gx} "
                    f"(face block) has no entry in classifier's "
                    f"gtdof_xyz_lookup. The face assembler emitted a "
                    f"nonmortar gtdof not seen by the boundary classifier."
                )

            if D_kk == 0.0:
                row_offset += self.VDIM
                continue

            # Diagonal D entries.
            for c in range(self.VDIM):
                gd = nonmortar_g_xyz[c]
                if gd < 0:
                    continue
                rows.append(row_offset + c)
                cols.append(gd)
                vals.append(D_kk)

            # Off-diagonal -A_m entries.
            for l in range(n_mortar_kept):
                A_kl = float(block.A_m[k, l])
                if A_kl == 0.0:
                    continue
                mortar_gx = int(block.mortar_gtdofs[l])
                mortar_g_xyz = self._gtdof_lookup.get(mortar_gx)
                if mortar_g_xyz is None:
                    raise RuntimeError(
                        f"ConstraintBuilder3D: mortar gtdof {mortar_gx} "
                        f"has no entry in classifier's gtdof_xyz_lookup."
                    )
                for c in range(self.VDIM):
                    gd = mortar_g_xyz[c]
                    if gd < 0:
                        continue
                    rows.append(row_offset + c)
                    cols.append(gd)
                    vals.append(-A_kl)

            row_offset += self.VDIM
        return row_offset


# =============================================================================
# Internal: dummy classifier for MortarAssembler2D.assemble_pair-only path
# =============================================================================

class _DummyEdgeClassifier:
    """Minimal stand-in for MortarAssembler2D when only assemble_pair
    is used (i.e., the legacy assemble_all path needs ``cl.edges``,
    but assemble_pair takes the edges directly).
    """
    edges = {}
