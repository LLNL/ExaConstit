"""Boundary classification for 2D rectangular RVE meshes.

WHAT
----
For a 2D rectangular RVE we need to identify, from a parallel MFEM mesh:
    * 4 corner nodes (Dirichlet u=0 to remove rigid-body modes)
    * 4 edge groups (bottom / top / left / right), each EXCLUDING corners,
      with their global true-DOF indices
    * The mortar/non-mortar designation (per Lopes et al. Fig. 5a):
          bottom = non-mortar (+),  top   = mortar (-)
          left   = non-mortar (+),  right = mortar (-)
    * The interior-DOF list (everything that is NOT on the boundary)

WHY (MPI structure)
-------------------
Each rank of a ``ParMesh`` knows only its locally-owned boundary nodes.
The mortar machinery, however, needs the FULL boundary picture to perform
non-conforming integration along an entire edge.  Phase 1 design:
    AllGather every boundary-node record (coords + global TDOF IDs) so
    every rank ends up with the same global edge classification.

For typical RVE sizes the boundary has O(N^((d-1)/d)) DOFs versus N total,
so this AllGather is cheap.  The architecture is set up so a future
distributed boundary assembly can swap in via the same dataclass interface
(``EdgeNodes2D``) without touching downstream consumers
(``MortarAssembler2D``, ``ConstraintBuilder2D``).

BOUNDARY-ATTRIBUTE CONVENTION (matches ExaConstit)
--------------------------------------------------
ExaConstit (``src/sim_state/simulation_state.cpp``, ``setBdrConditions``)
uses the following attribute layout for 2D:
    1 = bottom (y = y_min)
    2 = left   (x = x_min)
    3 = top    (y = y_max)        [in 3D, attribute 3 is "front" z=z_min]
    4 = right  (x = x_max)        [in 3D, attribute 4 is "top"   y=y_max]
This module assumes the 2D layout above; callers must set boundary
attributes on the mesh accordingly before constructing the classifier.

WHAT THE CLASSIFIER PRODUCES
----------------------------
After construction:
    * ``self.corners``  : dict  {label -> ``CornerInfo``}
                          labels are "bl", "br", "tl", "tr"
    * ``self.edges``    : dict  {edge_name -> ``EdgeNodes2D``}
                          edge_name in {"bottom", "top", "left", "right"}
    * ``self.interior_gtdofs`` : (Ni,) int64 ndarray of global TDOFs that
      are NOT on any boundary.  Sorted ascending.
    * ``self.boundary_gtdofs`` : (Nb,) int64 ndarray of all boundary TDOFs.
    * ``self.n_global_tdofs``  : total number of global TDOFs (FE space).

REFERENCES
----------
Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.
ExaConstit boundary convention: ``setBdrConditions`` in
``src/sim_state/simulation_state.cpp``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# These imports are eager (this module IS the MFEM-dependent half of the
# package).  The package's ``__init__.py`` imports ``BoundaryClassifier2D``
# lazily so unit tests of the pure-NumPy mortar machinery can run without
# pyMFEM / mpi4py installed.
from mpi4py import MPI
import mfem.par as mfem

from .types_2d import EdgeNodes2D, CornerInfo


# =============================================================================
# Main classifier
# =============================================================================

class BoundaryClassifier2D:
    """Classify boundary DOFs of a rectangular 2D RVE into mortar groups.

    Parameters
    ----------
    pmesh : mfem.par.ParMesh
        Parallel mesh.  Boundary attributes 1..4 must encode bottom / left
        / top / right (see module docstring).
    fes : mfem.par.ParFiniteElementSpace
        Vector H1 space of dimension 2.  Linear (order 1) is supported in
        Phase 1; higher order requires extending the edge-element extraction
        and the mortar shape-function basis.
    tol_rel : float, default 1e-9
        Relative tolerance (vs. bbox diagonal) for determining corner
        identity and on-edge classification.

    Notes
    -----
    Mortar designation (Lopes Fig. 5a):
        bottom (y=y_min) = non-mortar (+)    top   (y=y_max) = mortar (-)
        left   (x=x_min) = non-mortar (+)    right (x=x_max) = mortar (-)
    """

    # Boundary attribute -> edge name (ExaConstit 2D convention)
    BDR_ATTR_MAP = {1: "bottom", 2: "left", 3: "top", 4: "right"}
    # Mortar designation: True = non-mortar (+, carries multipliers)
    NON_MORTAR_EDGES = {"bottom", "left"}
    # Parametric axis along each edge (the OTHER coord is constant)
    PARAM_AXIS = {"bottom": "x", "top": "x", "left": "y", "right": "y"}

    def __init__(
        self,
        pmesh: mfem.ParMesh,
        fes: mfem.ParFiniteElementSpace,
        tol_rel: float = 1e-9,
    ) -> None:
        if pmesh.Dimension() != 2:
            raise ValueError("BoundaryClassifier2D requires a 2D mesh")
        if fes.GetVDim() != 2:
            raise ValueError("Expected a 2D vector FE space (vdim=2)")

        self.pmesh = pmesh
        self.fes = fes
        # ParMesh always uses MPI_COMM_WORLD per pyMFEM convention
        self.comm: MPI.Intracomm = MPI.COMM_WORLD
        self.rank   = self.comm.Get_rank()
        self.nranks = self.comm.Get_size()

        # ----- Bounding box (Allreduce min/max across ranks) -----
        self._compute_bbox()
        bbox_diagonal = np.linalg.norm(self.bbox_max - self.bbox_min)
        self.tol = tol_rel * bbox_diagonal

        # ----- Gather every boundary node globally -----
        self._gather_boundary_nodes()

        # ----- Classify into corners and edges -----
        self.corners: dict[str, CornerInfo] = {}
        self.edges:   dict[str, EdgeNodes2D] = {}
        self._build_corners_and_edges()

        # ----- Compute the interior-DOF list -----
        self._compute_interior_tdofs()

    # ---------------------------------------------------------------- bbox ---
    def _compute_bbox(self) -> None:
        """Compute the global RVE bounding box across all ranks.

        Uses vertex coordinates (linear-mesh assumption in Phase 1; for
        higher-order curved boundaries we would need to walk
        ``GetNodes()`` instead).
        """
        local_min = np.full(2, np.inf)
        local_max = np.full(2, -np.inf)
        for v in range(self.pmesh.GetNV()):
            xy = np.array([self.pmesh.GetVertexArray(v)[d] for d in range(2)])
            local_min = np.minimum(local_min, xy)
            local_max = np.maximum(local_max, xy)

        self.bbox_min = np.zeros(2)
        self.bbox_max = np.zeros(2)
        self.comm.Allreduce(local_min, self.bbox_min, op=MPI.MIN)
        self.comm.Allreduce(local_max, self.bbox_max, op=MPI.MAX)

    # -------------------------------------------------------------- gather ---
    def _gather_boundary_nodes(self) -> None:
        """Walk local boundary elements, collect (vertex, edge-name) pairs,
        AllGather a deduplicated global list keyed by snapped coordinate.

        Output (stored on self):
            self.global_nodes  : (N, 2) ndarray of unique boundary node coords
            self.global_attrs  : list[set[str]] of edge names per node
                                 (a corner belongs to two edges, so its
                                 set has size 2)
            self.gtdof_x       : (N,) int64; global TDOF for x-component,
                                 -1 if no rank reported it (would be a bug
                                 after the merge step below).
            self.gtdof_y       : (N,) int64; same for y-component.

        Coordinate snapping
        -------------------
        Floating-point coordinates from different ranks for the same
        physical vertex can differ by ULPs.  We snap to a tolerance grid
        (``round(x / tol)``) so set-keying is stable.
        """
        # Step 1: local pass -- collect (x, y, edge_name) for every boundary
        # vertex on this rank.
        local_records: list[tuple[float, float, str]] = []
        for be in range(self.pmesh.GetNBE()):
            attr = self.pmesh.GetBdrAttribute(be)
            if attr not in self.BDR_ATTR_MAP:
                continue
            edge_name = self.BDR_ATTR_MAP[attr]
            # pyMFEM convention: GetBdrElementVertices returns the vertex
            # array directly (no C++ out-parameter).  Coerce to plain ints
            # for safe handling regardless of whether the return type is
            # an mfem.intArray proxy, a list, or a numpy array.
            verts = [int(v) for v in self.pmesh.GetBdrElementVertices(be)]
            for v in verts:
                xy = self.pmesh.GetVertexArray(v)
                local_records.append((float(xy[0]), float(xy[1]), edge_name))

        # Step 2: build a local map (snapped_coord -> (gtdof_x, gtdof_y))
        # so we can merge TDOF indices across ranks.
        snap = self.tol
        def snap_key(x: float, y: float) -> tuple[int, int]:
            return (round(x / snap), round(y / snap))

        local_coord_to_gtdof: dict[tuple[int, int], tuple[int, int]] = {}
        for be in range(self.pmesh.GetNBE()):
            attr = self.pmesh.GetBdrAttribute(be)
            if attr not in self.BDR_ATTR_MAP:
                continue
            verts = [int(v) for v in self.pmesh.GetBdrElementVertices(be)]
            for v in verts:
                xy = self.pmesh.GetVertexArray(v)
                # Vector-linear H1 vertex DOFs: ``GetVertexDofs`` returns
                # the local-DOF (LDOF) indices for both components.  Like
                # GetBdrElementVertices, pyMFEM exposes this as a return
                # value, not a C++-style out-parameter.
                ldofs = [int(d) for d in self.fes.GetVertexDofs(v)]
                # For a vector FE space, ``GetVertexDofs(v)`` returns
                # the SCALAR DOF indices on vertex v (one per scalar
                # vertex DOF -- so length 1 for P1).  The vector-
                # component LDOFs are obtained by ``DofToVDof(scalar_ldof,
                # vd)`` where vd in {0, 1} indexes spatial component.
                # This mapping respects the FE space's Ordering (byNODES
                # vs byVDIM), so it works regardless of layout.
                if len(ldofs) >= 1:
                    scalar_ldof = ldofs[0]
                    ldof_x = self.fes.DofToVDof(scalar_ldof, 0)
                    ldof_y = self.fes.DofToVDof(scalar_ldof, 1)
                    gtdof_x = self.fes.GetGlobalTDofNumber(ldof_x) if ldof_x >= 0 else -1
                    gtdof_y = self.fes.GetGlobalTDofNumber(ldof_y) if ldof_y >= 0 else -1
                else:
                    gtdof_x = -1
                    gtdof_y = -1
                local_coord_to_gtdof[snap_key(xy[0], xy[1])] = (gtdof_x, gtdof_y)

        # Step 3: AllGather records and TDOF maps.
        all_records   = self.comm.allgather(local_records)
        all_tdof_maps = self.comm.allgather(local_coord_to_gtdof)

        # Step 4: merge records -- one entry per snapped coord, with the
        # SET of edge names this node belongs to (a corner is on 2 edges).
        merged: dict[tuple[int, int], dict] = {}
        for rec_list in all_records:
            for x, y, edge_name in rec_list:
                key = snap_key(x, y)
                if key not in merged:
                    merged[key] = {"x": x, "y": y, "attrs": set()}
                merged[key]["attrs"].add(edge_name)

        # Step 5: merge TDOF maps -- a node's gtdof is whichever rank
        # reported a non-negative value (in practice all ranks owning the
        # node should agree, since true-DOF numbering is global).
        merged_tdofs: dict[tuple[int, int], tuple[int, int]] = {}
        for tdof_map in all_tdof_maps:
            for key, (gx, gy) in tdof_map.items():
                if key not in merged_tdofs:
                    merged_tdofs[key] = (gx, gy)
                else:
                    existing_gx, existing_gy = merged_tdofs[key]
                    merged_tdofs[key] = (
                        gx if existing_gx < 0 else existing_gx,
                        gy if existing_gy < 0 else existing_gy,
                    )

        # Step 6: deterministic global ordering (sorted by physical x then y).
        keys_sorted = sorted(
            merged.keys(),
            key=lambda k: (merged[k]["x"], merged[k]["y"]),
        )
        N = len(keys_sorted)
        self.global_nodes  = np.zeros((N, 2))
        self.global_attrs: list[set[str]] = []
        self.gtdof_x = np.full(N, -1, dtype=np.int64)
        self.gtdof_y = np.full(N, -1, dtype=np.int64)
        self._key_to_gid: dict[tuple[int, int], int] = {}
        for i, key in enumerate(keys_sorted):
            data = merged[key]
            self.global_nodes[i] = [data["x"], data["y"]]
            self.global_attrs.append(data["attrs"])
            tdof_x, tdof_y = merged_tdofs.get(key, (-1, -1))
            self.gtdof_x[i] = tdof_x
            self.gtdof_y[i] = tdof_y
            self._key_to_gid[key] = i

    # ----------------------------------------------------- corners/edges ---
    def _is_at(self, val: float, target: float) -> bool:
        """Coordinate-equality test using the absolute tolerance."""
        return abs(val - target) <= self.tol

    def _build_corners_and_edges(self) -> None:
        """Identify the 4 corners by coord match, then build the 4
        edge-node groups (corners excluded, sorted by parametric axis)."""
        x_min, y_min = self.bbox_min
        x_max, y_max = self.bbox_max

        corner_targets = {
            "bl": (x_min, y_min),
            "br": (x_max, y_min),
            "tl": (x_min, y_max),
            "tr": (x_max, y_max),
        }
        corner_gids: dict[str, int] = {}
        for label, (cx, cy) in corner_targets.items():
            for i in range(self.global_nodes.shape[0]):
                xi, yi = self.global_nodes[i]
                if self._is_at(xi, cx) and self._is_at(yi, cy):
                    corner_gids[label] = i
                    self.corners[label] = CornerInfo(
                        label=label,
                        coord=self.global_nodes[i].copy(),
                        gtdof_x=int(self.gtdof_x[i]),
                        gtdof_y=int(self.gtdof_y[i]),
                    )
                    break
        if len(self.corners) != 4:
            raise RuntimeError(
                f"Expected 4 corners, found {len(self.corners)}: "
                f"{list(self.corners)}"
            )

        # Build the four interior-edge node lists.
        for edge_name in ("bottom", "top", "left", "right"):
            self.edges[edge_name] = self._extract_edge(edge_name, corner_gids)

    def _extract_edge(
        self, edge_name: str, corner_gids: dict[str, int]
    ) -> EdgeNodes2D:
        """Build the ``EdgeNodes2D`` for one edge: collect interior nodes,
        sort by parametric axis, and stitch them into a 1D element list with
        corner sentinels at the ends.

        The corner sentinels (-1 = left-along-param, -2 = right-along-param)
        are the convention shared with ``mortar_2d.MortarAssembler2D``.
        """
        x_min, y_min = self.bbox_min
        x_max, y_max = self.bbox_max
        if edge_name == "bottom":
            on_edge   = lambda xy: self._is_at(xy[1], y_min)
            param_axis = "x"
            edge_min, edge_max = x_min, x_max
        elif edge_name == "top":
            on_edge   = lambda xy: self._is_at(xy[1], y_max)
            param_axis = "x"
            edge_min, edge_max = x_min, x_max
        elif edge_name == "left":
            on_edge   = lambda xy: self._is_at(xy[0], x_min)
            param_axis = "y"
            edge_min, edge_max = y_min, y_max
        elif edge_name == "right":
            on_edge   = lambda xy: self._is_at(xy[0], x_max)
            param_axis = "y"
            edge_min, edge_max = y_min, y_max
        else:
            raise ValueError(edge_name)

        # Collect global IDs of interior nodes (skip corners).  Use the
        # ``global_attrs`` set membership as a sanity filter so we only
        # include nodes whose boundary records actually carried this
        # edge name (handles mesh decompositions where a node sits on
        # the interior face between two ranks but not actually on the edge).
        corner_set = set(corner_gids.values())
        interior_node_gids: list[int] = []
        for i in range(self.global_nodes.shape[0]):
            if i in corner_set:
                continue
            if on_edge(self.global_nodes[i]) and (edge_name in self.global_attrs[i]):
                interior_node_gids.append(i)

        # Sort interior nodes by the parametric axis coord.
        param_axis_idx = 0 if param_axis == "x" else 1
        interior_node_gids.sort(
            key=lambda g: self.global_nodes[g, param_axis_idx]
        )

        # Pack into local (per-edge) arrays.
        N = len(interior_node_gids)
        coords = np.zeros((N, 2))
        gtdofs_x = np.zeros(N, dtype=np.int64)
        gtdofs_y = np.zeros(N, dtype=np.int64)
        for k, gid in enumerate(interior_node_gids):
            coords[k]   = self.global_nodes[gid]
            gtdofs_x[k] = self.gtdof_x[gid]
            gtdofs_y[k] = self.gtdof_y[gid]

        # Stitch edge connectivity:
        #   left_corner -> node_0 -> node_1 -> ... -> node_{N-1} -> right_corner
        # Sentinels: -1 = left-along-param, -2 = right-along-param.
        # (Corner labels for sanity in case future debug prints want them.)
        if param_axis == "x":
            left_corner_label  = "bl" if edge_name == "bottom" else "tl"
            right_corner_label = "br" if edge_name == "bottom" else "tr"
        else:
            left_corner_label  = "bl" if edge_name == "left" else "br"
            right_corner_label = "tl" if edge_name == "left" else "tr"
        # Sequence of (node_idx_or_sentinel, label_for_diag).  Each consecutive
        # pair becomes one 1D element.
        seq = (
            [(-1, left_corner_label)]
            + [(k, None) for k in range(N)]
            + [(-2, right_corner_label)]
        )
        elements: list[tuple[int, int]] = []
        for (a_idx, _a_lbl), (b_idx, _b_lbl) in zip(seq[:-1], seq[1:]):
            elements.append((a_idx, b_idx))

        return EdgeNodes2D(
            name=edge_name,
            is_nonmortar=(edge_name in self.NON_MORTAR_EDGES),
            coords=coords,
            gtdofs_x=gtdofs_x,
            gtdofs_y=gtdofs_y,
            elements=elements,
            parametric_axis=param_axis,
            edge_min=edge_min,
            edge_max=edge_max,
        )

    # ------------------------------------------------------------- interior ---
    def _compute_interior_tdofs(self) -> None:
        """Compute the global TDOF list for nodes NOT on any boundary.

        Stored on self as:
            self.interior_gtdofs : (Ni,) int64 ndarray, sorted ascending
            self.boundary_gtdofs : (Nb,) int64 ndarray, sorted ascending
            self.n_global_tdofs  : int, total global TDOFs in the FE space
        """
        boundary_gtdofs: set[int] = set()
        for c in self.corners.values():
            if c.gtdof_x >= 0:
                boundary_gtdofs.add(int(c.gtdof_x))
            if c.gtdof_y >= 0:
                boundary_gtdofs.add(int(c.gtdof_y))
        for e in self.edges.values():
            for v in e.gtdofs_x:
                if v >= 0:
                    boundary_gtdofs.add(int(v))
            for v in e.gtdofs_y:
                if v >= 0:
                    boundary_gtdofs.add(int(v))

        # AllGather the per-rank boundary sets so every rank has the same
        # global classification.
        all_boundary_sets = self.comm.allgather(boundary_gtdofs)
        global_boundary: set[int] = set()
        for s in all_boundary_sets:
            global_boundary |= s

        n_tdof_global = self.fes.GlobalTrueVSize()
        all_tdofs = set(range(n_tdof_global))
        self.interior_gtdofs = np.array(
            sorted(all_tdofs - global_boundary), dtype=np.int64
        )
        self.boundary_gtdofs = np.array(sorted(global_boundary), dtype=np.int64)
        self.n_global_tdofs  = n_tdof_global

    # --------------------------------------------------------------- helpers ---
    def corner_dirichlet_gtdofs(self) -> np.ndarray:
        """Return the global TDOFs that should be prescribed to zero
        (rigid-body-mode removal at the four corners).
        """
        out: list[int] = []
        for c in self.corners.values():
            if c.gtdof_x >= 0:
                out.append(c.gtdof_x)
            if c.gtdof_y >= 0:
                out.append(c.gtdof_y)
        # Allgather + dedup (corner DOFs may be reported by multiple ranks).
        all_lists = self.comm.allgather(out)
        merged = sorted({v for lst in all_lists for v in lst})
        return np.array(merged, dtype=np.int64)

    def summary(self) -> str:
        """Human-readable summary; useful in driver scripts for sanity checks."""
        lines = [f"BoundaryClassifier2D (rank {self.rank}/{self.nranks})"]
        lines.append(f"  bbox: {self.bbox_min} -> {self.bbox_max}")
        lines.append(f"  total global TDOFs:    {self.n_global_tdofs}")
        lines.append(f"  boundary global TDOFs: {len(self.boundary_gtdofs)}")
        for label, c in self.corners.items():
            lines.append(
                f"  corner {label}: {c.coord}  tdofs=({c.gtdof_x},{c.gtdof_y})"
            )
        for edge_name, e in self.edges.items():
            kind = "(+)" if e.is_nonmortar else "(-)"
            lines.append(
                f"  edge {edge_name}{kind}: {e.n_nodes} nodes, "
                f"{len(e.elements)} elements along {e.parametric_axis}"
            )
        return "\n".join(lines)
