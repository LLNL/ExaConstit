"""Pure-Python data containers shared across the mortar PBC modules.

WHAT
----
Two dataclasses:
    * ``EdgeNodes2D`` : one boundary edge (bottom / top / left / right) with
      its interior-node coords, global true-DOF indices, and 1D element
      connectivity (with corner sentinels).
    * ``CornerInfo``  : one of the four corner nodes of a 2D rectangular RVE.

WHY
---
These are the structs the mortar matrix assembler operates on.  Isolating
them in this MFEM-/MPI-free module means ``mortar_2d.py``,
``constraint_builder.py``, and the unit tests can be imported and run
without pyMFEM or mpi4py installed -- which is critical because the
mathematical correctness of the mortar machinery should be testable without
the full parallel FE infrastructure.

WHO PRODUCES THEM
-----------------
``BoundaryClassifier2D`` (in ``boundary_2d.py``, MFEM-dependent) builds these
from a ``ParMesh`` + ``ParFiniteElementSpace``.  Test code can construct
them directly with synthetic data -- see ``tests/test_mortar_2d_unit.py``.

REFERENCES
----------
Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.
ExaConstit boundary-attribute convention: ``src/sim_state/simulation_state.cpp``
in the ExaConstit codebase (1=bottom, 2=left, 3=top, 4=right for 2D).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class EdgeNodes2D:
    """A single edge of a 2D rectangular RVE boundary, corners excluded.

    The four edges (bottom / top / left / right) are each represented by an
    ``EdgeNodes2D`` instance.  Corner nodes are NOT included here -- they
    are tracked separately as ``CornerInfo`` instances because they are
    Dirichlet-prescribed (set to zero, to remove rigid-body modes) rather
    than coupled by the mortar constraint.

    Attributes
    ----------
    name : str
        One of "bottom", "top", "left", "right".
    is_nonmortar : bool
        True iff this edge carries Lagrange multipliers (the "+" side in
        Lopes et al. Fig. 5a).  Convention: bottom and left are
        non-mortar; top and right are mortar.
    coords : (N, 2) ndarray
        Coordinates of the N interior edge nodes (corners excluded),
        sorted ascending along ``parametric_axis``.
    gtdofs_x : (N,) int64 ndarray
        Global true-DOF index for the x-component at each interior node.
        Set to -1 if the DOF is not owned on this rank (in the AllGathered
        merged list, it should be filled in by some rank; -1 indicates an
        unfilled entry, which would be a bug).
    gtdofs_y : (N,) int64 ndarray
        Same as gtdofs_x for the y-component.
    elements : list[(int, int)]
        1D line-2 boundary elements as ordered ``(node_a_idx, node_b_idx)``
        pairs.  Sentinels:
            -1 = "left  corner" along the parametric axis (= edge_min)
            -2 = "right corner" along the parametric axis (= edge_max)
        For an edge with N interior nodes, the connectivity is:
            (-1, 0), (0, 1), ..., (N-2, N-1), (N-1, -2)
        i.e. N+1 elements total, two of which touch a corner.
    parametric_axis : str
        "x" for horizontal edges (bottom / top) -- the parametric coord is
        x and y is constant along the edge.  "y" for vertical edges
        (left / right).
    edge_min : float
        Minimum value of the parametric coord on this edge (= the
        coordinate of the "left" corner along the parametric axis).
    edge_max : float
        Maximum value of the parametric coord on this edge.
    """
    name: str
    is_nonmortar: bool
    coords: np.ndarray
    gtdofs_x: np.ndarray
    gtdofs_y: np.ndarray
    elements: List[Tuple[int, int]] = field(default_factory=list)
    parametric_axis: str = "x"
    edge_min: float = 0.0
    edge_max: float = 1.0

    @property
    def n_nodes(self) -> int:
        """Number of *interior* nodes on this edge (corners excluded)."""
        return self.coords.shape[0]


@dataclass
class CornerInfo:
    """A single corner node of a 2D rectangular RVE.

    A 2D RVE has exactly four corners, prescribed to ``u_tilde = 0`` to
    remove rigid-body modes.  These are handled OUTSIDE the mortar coupling
    (the corner DOFs do not appear as rows of the constraint matrix).

    Attributes
    ----------
    label : str
        One of "bl", "br", "tl", "tr"
        (bottom-left, bottom-right, top-left, top-right).
    coord : (2,) ndarray
        Physical coordinates of the corner.
    gtdof_x : int
        Global true-DOF index of the x-component, or -1 if not owned on
        this rank (after AllGather merging this should never be -1 if the
        corner is in the global mesh).
    gtdof_y : int
        Same for the y-component.
    """
    label: str
    coord: np.ndarray
    gtdof_x: int
    gtdof_y: int
