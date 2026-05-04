"""Build the global constraint matrix C from per-edge mortar blocks.

WHAT
----
Given the per-edge-pair mortar blocks ``(D^{nm}, A^m)`` produced by
``MortarAssembler2D``, assemble the global constraint matrix C such that

    C · v_global  =  0                                                    (*)

is the discrete periodicity condition on the global true-DOF vector
``v_global``.  ``v_global`` is the *fluctuation* (or its Newton increment),
since ExaConstit's velocity-based updated-Lagrangian formulation expresses
periodicity on the velocity update at each step:

    F = F_macro + grad(u_tilde),     u_tilde periodic on opposite faces.

In the saddle-point Newton system (see ``saddle_point.py``)

    [ K   C^T ] [ Δv     ]   [ ... ]
    [ C   0   ] [ Δλ     ] = [ ... ]

C is the constraint block built here.

WHY (algorithmic structure)
---------------------------
For each non-mortar (+) edge node k and each spatial component c ∈ {x, y}
we get one constraint row of the form

    D^{nm}_{kk}  v^+_{k, c}   -   Σ_l A^m_{kl}  v^-_{l, c}   =   0.        (**)

The coupling matrices ``D^{nm}`` and ``A^m`` are scalar (per-edge-node);
each spatial component is constrained independently with the same
coefficients.  This reflects the fact that periodicity is a *kinematic*
constraint, not a stress one -- each component of the displacement
fluctuation is periodic on its own.

Global true-DOF indexing comes from MFEM via the boundary classifier:
each edge node carries (gtdof_x, gtdof_y) and the constraint row reaches
into the global vector by those indices.

WHO CALLS WHOM
--------------
    BoundaryClassifier2D  -->  edges (with gtdofs)
    MortarAssembler2D     -->  D^{nm}, A^m  (one per edge pair)
    ConstraintBuilder2D   -->  C  (this module)
    SaddlePointSolver     -->  consumes (K, C, ...)

EXTENSION POINT FOR UNIFORM TRACTION (DEFERRED)
-----------------------------------------------
ExaConstit currently has no traction BC, so uniform traction (UT) is
deferred to a later phase (Lopes et al. §3.2).  When added, UT will be
its OWN constraint assembler producing its OWN small constraint block
(a few rows: one per component of the macroscopic-deformation-gradient
constraint ``∫ (u_tilde ⊗ N) dA = 0``).  The saddle-point solver should
take a *list* of constraint matrices (or one assembled by stacking) so
that adding UT does not require touching mortar code -- this module's
output is one C; UT will produce another C; both are stacked vertically
into the saddle-point system.  See the ``ConstraintAssembler`` ABC
sketch in the next phase of this prototype.

REFERENCES
----------
Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.
    * Eq. (59)   : saddle-point Newton system
    * §3.3, §C  : dual-basis mortar formulation
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .types_2d import EdgeNodes2D
from .mortar_2d import MortarBlock2D


class ConstraintBuilder2D:
    """Assemble the global mortar-periodic constraint matrix C in CSR form.

    Phase 1 assumption: vdim = 2 (planar).  Each non-mortar node produces
    *vdim* constraint rows; the mortar block matrices are scalar and
    applied identically to each spatial component.

    Parameters
    ----------
    classifier : duck-typed object
        Must expose:
            * ``.edges`` : dict of edge name -> ``EdgeNodes2D``
            * ``.n_global_tdofs`` : total number of global true DOFs
    blocks : dict[(str, str), MortarBlock2D]
        The per-pair mortar matrices, keyed by ``(plus_name, minus_name)``,
        as produced by ``MortarAssembler2D.assemble_all()``.

    Output of ``build()``
    ---------------------
    ``C`` : (n_constraints, n_global_tdofs) scipy CSR sparse matrix
        where ``n_constraints = vdim * sum(n_plus over edge pairs)``.
        Each row encodes one scalar component of Eq. (**) for one
        non-mortar node.  Corner DOFs do NOT appear as constraint rows
        (corners are Dirichlet); they MAY appear as columns iff a -
        edge node next to a corner contributes there -- but in our
        construction the - corner sentinels are dropped from A^m so
        those columns are zero too.
    """

    VDIM = 2  # 2D planar; planar elasticity has 2 components per node

    def __init__(
        self,
        classifier,
        blocks: dict,
    ) -> None:
        self.cl = classifier
        self.blocks = blocks

    # -------------------------------------------------------------- API ---
    def build(self) -> sp.csr_matrix:
        """Build and return the global constraint matrix C as a CSR sparse.

        Algorithm
        ---------
        Walk every (+, -) edge pair, every interior + node k, every
        spatial component c.  For each (k, c):
            1. Emit a +D_kk entry at column ``gtdof_+[k, c]``.
            2. Emit a -A_kl entry at column ``gtdof_-[l, c]`` for every
               interior - node l with nonzero ``A^m_{kl}``.
        Skip rows where ``D_kk == 0`` (would happen if a corner-mod-only
        + element wiped the row; degenerate but possible for
        odd-edge-count meshes).
        """
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        constraint_row_offset = 0

        for (plus_name, minus_name), block in self.blocks.items():
            plus_edge:  EdgeNodes2D = self.cl.edges[plus_name]
            minus_edge: EdgeNodes2D = self.cl.edges[minus_name]
            n_plus  = plus_edge.n_nodes
            n_minus = minus_edge.n_nodes

            for k in range(n_plus):
                gtdofs_at_plus_node = (
                    plus_edge.gtdofs_x[k],
                    plus_edge.gtdofs_y[k],
                )
                D_kk = block.D_nm[k]
                if D_kk == 0.0:
                    # Could happen if a node sits between two "both-corner"
                    # elements (the dual basis modification kills the row
                    # entirely).  Skip: no meaningful constraint to enforce.
                    constraint_row_offset += self.VDIM
                    continue

                # ----- Diagonal D^{nm} entry, one per spatial component -----
                for component_idx in range(self.VDIM):
                    gtdof_plus = int(gtdofs_at_plus_node[component_idx])
                    if gtdof_plus < 0:
                        continue
                    rows.append(constraint_row_offset + component_idx)
                    cols.append(gtdof_plus)
                    vals.append(D_kk)

                # ----- Off-diagonal -A^m entries over all - nodes -----
                for l in range(n_minus):
                    A_kl = block.A_m[k, l]
                    if A_kl == 0.0:
                        continue
                    gtdofs_at_minus_node = (
                        minus_edge.gtdofs_x[l],
                        minus_edge.gtdofs_y[l],
                    )
                    for component_idx in range(self.VDIM):
                        gtdof_minus = int(gtdofs_at_minus_node[component_idx])
                        if gtdof_minus < 0:
                            continue
                        rows.append(constraint_row_offset + component_idx)
                        cols.append(gtdof_minus)
                        vals.append(-A_kl)

                constraint_row_offset += self.VDIM

        n_rows = constraint_row_offset
        n_cols = self.cl.n_global_tdofs
        if n_rows == 0:
            return sp.csr_matrix((0, n_cols))
        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_rows, n_cols)
        ).tocsr()

    # ------------------------------------------------------------ helpers ---
    def n_constraints(self) -> int:
        """Return the number of constraint rows (= vdim * total + nodes).

        Use this to size the multiplier vector in the saddle-point system.
        """
        n = 0
        for (plus_name, _), _block in self.blocks.items():
            plus_edge = self.cl.edges[plus_name]
            n += self.VDIM * plus_edge.n_nodes
        return n
