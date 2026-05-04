"""Abstract interface for constraint assemblers.

WHAT
----
A small ABC + composition helper that lets the saddle-point solver consume
a *list* of constraint contributions, each producing its own slice of the
global C matrix.  Phase 1 has only one concrete implementation (the
mortar-PBC contribution from ``ConstraintBuilder2D``); the design exists
to make adding uniform-traction (UT) constraints later a drop-in.

WHY (architectural rationale)
-----------------------------
ExaConstit currently has no traction BC, so the uniform-traction (UT)
formulation from Lopes et al. §3.2 is deferred.  However, when UT IS
added, it will produce its OWN constraint block:

    Mortar PBC :  C_mortar  =  one row per (interior + node, component)
                              -- this can be a few hundred to thousands of
                              rows for a typical RVE
    Uniform tx :  C_ut      =  4 rows in 2D (or 9 in 3D), one per
                              component of the macroscopic-deformation-
                              gradient compatibility statement
                              ∫ (u_tilde ⊗ N) dA = 0

Without this ABC, adding UT would mean either:
    (a) coupling UT logic into ``ConstraintBuilder2D`` (bad: mixing
        mathematically distinct constraints in one class), or
    (b) editing every consumer (the saddle-point solver, the example
        scripts) to know about both kinds (bad: changes ripple).

With this ABC, adding UT means: write a new ``UniformTractionAssembler2D``
that subclasses ``ConstraintAssembler``, returns its own (small) C block
from ``assemble()``, and pass a list ``[mortar_asm, ut_asm]`` to the
solver.  The solver vstacks the C blocks and treats them uniformly.

EXTENSION-POINT NOTES FOR THE FUTURE UT IMPLEMENTATION
------------------------------------------------------
The UT assembler will need:
    * The boundary classifier (or just a list of all boundary edges)
      so it can integrate ``∫ u_tilde ⊗ N dA`` over the full
      ∂Ω_micro.
    * The macroscopic deformation gradient F_macro, possibly to set
      a corresponding RHS.  In Lopes' formulation the homogeneous-
      kinematics insertion is u_lin = (F-I)X, applied as the linear
      part of the displacement; the UT constraint then enforces that
      the *fluctuation* u_tilde produces zero average ⊗ N, which is
      a homogeneous constraint regardless of F.
    * No mortar matrices (UT doesn't pair edges; it integrates over
      the whole boundary).

The 2D version of the UT constraint produces 4 rows
(2 components × 2 directions of N for a rectangular RVE):
    ∫_∂Ω u_tilde_x N_x dA = 0
    ∫_∂Ω u_tilde_x N_y dA = 0
    ∫_∂Ω u_tilde_y N_x dA = 0
    ∫_∂Ω u_tilde_y N_y dA = 0
where N is the outward boundary normal.  These integrals reduce to
trapezoidal sums over corner/edge-node displacements weighted by edge
geometry.

REFERENCES
----------
Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.
    * §3.2     : uniform traction (UT) formulation
    * §3.3, §C : mortar PBC formulation
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from .constraint_builder import ConstraintBuilder2D
from .mortar_2d import MortarBlock2D


# =============================================================================
# Abstract interface
# =============================================================================

class ConstraintAssembler(ABC):
    """Produce the constraint contribution C_block (and optional RHS g_block).

    Subclasses
    ----------
    Each concrete subclass corresponds to one mathematically distinct
    constraint family.  Examples (current and planned):
        MortarPbcConstraintAssembler  -- mortar periodic BCs (Phase 1)
        UniformTractionConstraintAssembler -- UT (deferred, future)

    Sign convention
    ---------------
    The saddle-point system is

        [ K   C^T ] [Δv]   [ -r + C^T λ ]
        [ C   0   ] [Δλ] = [ -C v + g    ]

    so an assembler with non-zero ``g`` is asserting ``C v = g``.  For
    homogeneous constraints (the only kind we use in Phase 1) ``g == 0``.
    The default ``rhs()`` returns zeros for that reason.
    """

    @abstractmethod
    def name(self) -> str:
        """Short name for diagnostics (e.g. ``"mortar_pbc"``)."""
        raise NotImplementedError

    @abstractmethod
    def n_rows(self) -> int:
        """Number of constraint rows this assembler will contribute."""
        raise NotImplementedError

    @abstractmethod
    def assemble(self) -> sp.csr_matrix:
        """Return the (n_rows, n_global_tdofs) CSR contribution to C."""
        raise NotImplementedError

    def rhs(self) -> np.ndarray:
        """Return the (n_rows,) RHS vector g for ``C v = g``.

        Default: zeros (homogeneous constraint).  Override for
        inhomogeneous constraints if you need them.
        """
        return np.zeros(self.n_rows())


# =============================================================================
# Concrete: mortar PBC (wraps the existing ConstraintBuilder2D)
# =============================================================================

class MortarPbcConstraintAssembler(ConstraintAssembler):
    """Produce the mortar PBC contribution to the global C matrix.

    This is a thin adapter around ``ConstraintBuilder2D`` that conforms
    to the ``ConstraintAssembler`` interface.  Existing call sites that
    use ``ConstraintBuilder2D`` directly continue to work unchanged;
    new call sites that want the uniform multi-constraint interface
    construct a list of ``ConstraintAssembler`` instances and use
    :func:`stack_constraints` (below).

    Parameters
    ----------
    classifier : duck-typed
        Must expose ``.edges`` (dict) and ``.n_global_tdofs`` (int).
    blocks : dict[(str, str), MortarBlock2D]
        Per-pair mortar blocks from ``MortarAssembler2D.assemble_all()``.
    """

    def __init__(self, classifier, blocks: dict) -> None:
        self._builder = ConstraintBuilder2D(classifier, blocks)
        self._n_rows  = self._builder.n_constraints()
        self._cached_C: sp.csr_matrix | None = None

    def name(self) -> str:
        return "mortar_pbc"

    def n_rows(self) -> int:
        return self._n_rows

    def assemble(self) -> sp.csr_matrix:
        # Cache: ConstraintBuilder2D.build() is idempotent but not free;
        # callers may invoke ``assemble()`` more than once (e.g. for
        # diagnostics + the actual solve), so we memoize.
        if self._cached_C is None:
            self._cached_C = self._builder.build()
        return self._cached_C


# =============================================================================
# Composition helper
# =============================================================================

def stack_constraints(
    assemblers: list[ConstraintAssembler],
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Vertically stack the contributions of multiple constraint assemblers.

    Parameters
    ----------
    assemblers : list[ConstraintAssembler]
        One per constraint family.  Order matters only for diagnostics
        (which constraint rows are which); the saddle-point system is
        invariant to row permutations.

    Returns
    -------
    C : (sum_i n_rows_i, n_global_tdofs) scipy CSR
        Full constraint matrix to feed the saddle-point solver.
    g : (sum_i n_rows_i,) ndarray
        RHS vector for ``C v = g`` (zeros for homogeneous constraints).

    Notes
    -----
    All assemblers must produce blocks with the same number of columns
    (= n_global_tdofs).  This is enforced by sharing the boundary
    classifier across them.
    """
    if not assemblers:
        raise ValueError("stack_constraints requires at least one assembler")

    blocks   = [a.assemble() for a in assemblers]
    rhs_vecs = [a.rhs()      for a in assemblers]

    # Sanity: all blocks share the same column count.
    n_cols = blocks[0].shape[1]
    for asm, blk in zip(assemblers, blocks):
        if blk.shape[1] != n_cols:
            raise ValueError(
                f"Constraint assembler '{asm.name()}' produced a block "
                f"with {blk.shape[1]} columns, expected {n_cols}"
            )

    C = sp.vstack(blocks, format="csr")
    g = np.concatenate(rhs_vecs)
    return C, g
