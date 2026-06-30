"""2D mortar matrix assembly for non-conforming periodic boundary conditions.

WHAT
----
Build the mortar coupling matrices A^m and D^{nm} for a single (+, -) edge
pair of a 2D rectangular RVE.  The output of this module feeds the global
constraint matrix C built by ``constraint_builder.py``, which in turn enters
the saddle-point Newton system in ``saddle_point.py``.

WHY (quick primer for ExaConstit-familiar readers)
--------------------------------------------------
The weak statement of periodicity is

    ∫_Γ  λ · (u^+ - u^-) dA  =  0     ∀ λ ∈ M_h,                     (*)

where Γ is the non-mortar ("+") edge, u^+ is the FE trace on the + edge,
u^- is the *projection onto Γ* of the opposite-edge ("-") solution, and
M_h is the discrete multiplier space.

Standard mortar methods pick λ ∈ span(N^+_k); that yields a *non-diagonal*
A^{nm} matrix and the constraint elimination requires inverting A^{nm}.

The DUAL-BASIS approach (Lopes et al. §3.3, §C) instead picks λ in the
dual basis M_k bi-orthogonal to N^+_k:

    ∫_{ref elem}  M_k(ξ) N_l(ξ) dξ  =  δ_{kl}.                        (Eq. C.1)

With this choice, after element-wise integration over Γ,

    A^{nm}_{kl}  =  ∫_Γ  M_k N^+_l dA  =  δ_{kl} ∫_Γ N^+_l dA  =  δ_{kl} D^{nm}_{kk},

so A^{nm} reduces to a *diagonal* D^{nm}.  The constraint becomes one
scalar equation per non-mortar node:

    D^{nm}_{kk} u^+_k  -  Σ_l A^m_{kl} u^-_l  =  0,    A^m_{kl} = ∫_Γ M_k N^-_l dA.

Diagonal D^{nm} means eliminating multipliers in the saddle-point system
costs nothing -- this is the algorithmic payoff of the dual basis.

WHAT THIS MODULE COMPUTES
-------------------------
For a given (+, -) edge pair of a 2D RVE this module assembles
    * A^m       : (n_plus, n_minus) ndarray, the off-diagonal coupling
    * D^{nm}    : (n_plus,)        ndarray, the diagonal non-mortar mass
in *physical-edge-node* indexing.  ``ConstraintBuilder2D`` then maps these
indices to global true-DOF indices (vector components handled there).

NOTES ON THE TRICKY PARTS
-------------------------
1. The line-2 dual basis (Eq. C.1) is ASYMMETRIC on [-1, 1]: M_1(ξ) is
   negative for ξ > 1/3.  This is essential for bi-orthogonality, but it
   means individual entries (and even row sums) of A^m can be NEGATIVE.
   That's fine; only the *moment* statements (constant and linear field
   reproduction) need to hold globally.

2. The Wohlmuth corner modification (Eq. C.2: M_1 = 0, M_2 = 1, or vice
   versa) is applied on every + element that touches a Dirichlet corner.
   This DELIBERATELY breaks bi-orthogonality on those segments; it is
   the price paid to avoid over-constraining the corner DOF (which is
   already prescribed = 0 by the rigid-body-mode removal) and to avoid
   spurious oscillations.  Linear-field reproduction therefore CANNOT
   hold on corner segments by design; it is the FE patch test (the
   homogeneous RVE recovering u_tilde = 0, Lopes §5.1.1) that validates
   the corner-modified machinery end-to-end.

3. D^{nm}_{kk} = ∫_Γ N_k dA uses the *standard* shape function N_k on the
   nonmortar (NOT the modified dual M_k).  D^{nm} is the *measure* node k
   carries along Γ; it does not depend on the multiplier basis.

4. We DROP rows and columns corresponding to corner sentinels in A^m
   and D^{nm}.  Corner DOFs are essential (set to zero for rigid-body
   mode removal) and are handled outside the mortar constraint.

REFERENCES
----------
Lopes, Ferreira, Andrade Pires, "On the efficient enforcement of uniform
traction and mortar periodic boundary conditions in computational
homogenisation", CMAME 384 (2021) 113930.
    * Eqs. (56)-(57): mortar matrix integrals
    * Eq. (C.1)    : line-2 dual basis
    * Eq. (C.2)    : Wohlmuth corner modifications
    * Fig. 5(a)    : non-mortar / mortar designation for 2D RVE
    * §5.1.1       : homogeneous RVE patch test
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types_2d import EdgeNodes2D


# =============================================================================
# Reference shape functions and dual basis (line-2 element, ξ ∈ [-1, 1])
# =============================================================================

def N_line2(xi: float) -> tuple[float, float]:
    """Standard line-2 (linear Lagrange) shape functions on the reference
    element ξ ∈ [-1, 1].

    Returns
    -------
    (N_1, N_2) : tuple[float, float]
        N_1(ξ) = (1 - ξ)/2,  N_2(ξ) = (1 + ξ)/2.

    Properties
    ----------
    Partition of unity: N_1 + N_2 = 1.
    Both N_k are non-negative on [-1, 1] (this is what makes the standard
    basis well-suited as a *trial* basis for displacement, not as a test
    basis for the multiplier).
    """
    return 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)


def M_line2_dual(xi: float) -> tuple[float, float]:
    """Line-2 dual basis (Lopes et al. Eq. C.1).

    Returns
    -------
    (M_1, M_2) : tuple[float, float]
        M_1(ξ) = (1 - 3ξ)/2,  M_2(ξ) = (1 + 3ξ)/2.

    Properties
    ----------
    Bi-orthogonal to the standard line-2 basis on the reference element:
        ∫_{-1}^{+1} M_k(ξ) N_l(ξ) dξ  =  δ_{kl}.
    Note M_1 is *negative* for ξ > 1/3 and M_2 is negative for ξ < -1/3.
    This sign change is essential for bi-orthogonality.
    """
    return 0.5 * (1.0 - 3.0 * xi), 0.5 * (1.0 + 3.0 * xi)


def M_line2_dual_modified(xi: float, side: str) -> tuple[float, float]:
    """Wohlmuth-modified dual basis when one endpoint of the + element is
    a Dirichlet corner (Lopes et al. Eq. C.2).

    Parameters
    ----------
    xi : float
        Reference coord on the + parent element.  Ignored: the modified
        basis is constant per-side.  (Argument kept in the signature for
        symmetry with ``M_line2_dual`` so callers can swap.)
    side : {"left", "right", "both"}
        Identifies WHICH local endpoint of the + element is the corner:
            "left"  : node 1 (ξ=-1 in local coords) is the corner ->
                      M_1 = 0, M_2 = 1   (transfer everything to node 2)
            "right" : node 2 (ξ=+1) is the corner ->
                      M_1 = 1, M_2 = 0
            "both"  : both endpoints are corners (the entire edge has
                      no interior node).  Constraint is empty;
                      M_1 = M_2 = 0.

    Returns
    -------
    (M_1, M_2) : tuple[float, float]
        Modified dual values at this Gauss point.

    Notes
    -----
    These modifications BREAK bi-orthogonality on the corner element:
    e.g. for ``side="left"``, ∫ M_2 N_1 dξ = ∫ 1 · (1-ξ)/2 dξ = 1, which
    is non-zero (vs. zero in the standard dual case).  This is intentional
    and accepted; see the module docstring "tricky parts" §2.
    """
    if side == "left":
        return 0.0, 1.0
    elif side == "right":
        return 1.0, 0.0
    elif side == "both":
        return 0.0, 0.0
    raise ValueError(
        f"Unknown corner side {side!r}; expected 'left', 'right', or 'both'"
    )


# 3-point Gauss-Legendre quadrature on the reference interval [-1, 1].
# Integrates polynomials of degree <= 5 exactly.  The integrand here is
# a product of two linears (degree 2) per Gauss-point loop, so 2-point
# would suffice; 3-point is used for robustness on the *segment* (which
# subdivides the parent + element) where the effective polynomial degree
# can rise slightly due to compositions.
_GL3_PTS = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GL3_WTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


# =============================================================================
# Block container
# =============================================================================

@dataclass
class MortarBlock2D:
    """Assembled mortar quantities for one (+, -) edge pair.

    Indexing of A_m and D_nm is by *position along the edge among interior
    (non-corner) nodes*, ordered in increasing parametric coord.  Corner
    sentinels (-1, -2) are NOT present as indices: they were dropped during
    assembly because corner DOFs are essential / Dirichlet = 0 elsewhere.

    Attributes
    ----------
    A_m : (n_plus, n_minus) ndarray
        Mortar coupling matrix.  ``A_m[k, l] = ∫_Γ M_k(ξ) N^-_l(ζ(ξ)) dA``.
        Stored dense for the prototype (boundary is small).
    D_nm : (n_plus,) ndarray
        Diagonal non-mortar matrix.  ``D_nm[k] = ∫_Γ N^+_k dA``.
    plus_edge_name : str
        Name of the non-mortar edge ("bottom", "left").
    minus_edge_name : str
        Name of the mortar edge ("top", "right").
    """
    A_m: np.ndarray
    D_nm: np.ndarray
    plus_edge_name: str
    minus_edge_name: str


# =============================================================================
# Assembler
# =============================================================================

class MortarAssembler2D:
    """Build mortar block matrices for the (+, -) edge pairs of a 2D RVE.

    Pairing convention (matches Lopes et al. Fig. 5a):
        bottom (+)  <->  top    (-)
        left   (+)  <->  right  (-)

    Usage
    -----
    >>> classifier = BoundaryClassifier2D(pmesh, fes)
    >>> assembler  = MortarAssembler2D(classifier)
    >>> blocks     = assembler.assemble_all()
    >>> bottom_top_block = blocks[("bottom", "top")]

    Algorithm (per pair)
    --------------------
    1. Loop over + elements (1D line-2 segments along the + edge).
    2. For each + element, accumulate D^{nm} contributions: the standard
       N^+_k integrates to the segment's Jacobian, distributed equally to
       both endpoints.
    3. Find each - element overlapping this + element's parametric range
       (interval intersection on the parametric axis).
    4. Integrate M_k(ξ_+) N^-_l(ξ_-) over each overlap segment using
       3-point Gauss quadrature; accumulate into A^m.
    5. Drop entries corresponding to corner sentinels (rows from + side,
       cols from - side).

    The classifier is duck-typed: it must expose ``.edges`` (a dict of
    edge name -> ``EdgeNodes2D``).
    """

    PAIRS = [("bottom", "top"), ("left", "right")]

    def __init__(self, classifier) -> None:
        self.cl = classifier

    # ----------------------------------------------------------------- API ---
    def assemble_all(self) -> dict[tuple[str, str], MortarBlock2D]:
        """Assemble both (+, -) pairs and return a dict keyed by pair name."""
        out: dict[tuple[str, str], MortarBlock2D] = {}
        for plus_name, minus_name in self.PAIRS:
            out[(plus_name, minus_name)] = self._assemble_pair(
                self.cl.edges[plus_name], self.cl.edges[minus_name]
            )
        return out

    def assemble_pair(self, plus_edge, minus_edge) -> MortarBlock2D:
        """Public-facing wrapper around `_assemble_pair`.

        Identical to `_assemble_pair`; exists so 3D code paths
        (`ConstraintBuilder3D` in Phase 3.3.C, processing 9 edge pairs
        at once) can reuse this assembler on `EdgeInfo3D` objects
        without reaching for a single-underscore private method.

        Both `EdgeNodes2D` and `EdgeInfo3D` are duck-type compatible:
        each provides ``parametric_axis`` (the axis label, validated
        against `_AXIS_TO_COLUMN`), ``edge_min``/``edge_max``,
        ``coords`` (2D array), ``elements`` (list of (n1, n2) tuples
        with corner sentinels), and ``n_nodes``. The assembler does
        not touch ``gtdofs_*`` — that's the caller's concern.
        """
        return self._assemble_pair(plus_edge, minus_edge)

    # ----------------------------------------------------------- internals ---
    def _assemble_pair(
        self, plus_edge, minus_edge,
    ) -> MortarBlock2D:
        """Assemble A^m and D^{nm} for one pair of opposite edges.

        Duck-typed on the edge arguments; see `assemble_pair` for the
        contract. See class docstring "Algorithm (per pair)" for the
        high-level steps.
        """
        n_plus = plus_edge.n_nodes
        n_minus = minus_edge.n_nodes
        A_m  = np.zeros((n_plus, n_minus))
        D_nm = np.zeros(n_plus)

        # -------------------------------------------- loop over + elements ---
        for plus_node1_idx, plus_node2_idx in plus_edge.elements:
            # Physical-edge-coord endpoints of this + element.
            # Sentinel handling: -1 -> edge_min, -2 -> edge_max (see helper).
            plus_phys_lo, plus_phys_hi = self._param_endpoints(
                plus_edge, plus_node1_idx, plus_node2_idx,
            )
            if plus_phys_hi <= plus_phys_lo:
                continue
            # dphys / dxi on the + parent element (xi in [-1, 1]).
            plus_jacobian = 0.5 * (plus_phys_hi - plus_phys_lo)

            # Identify which side(s) (if any) of this element touch a Dirichlet
            # corner; selects the dual basis variant used on this element.
            corner_side = self._corner_side(plus_node1_idx, plus_node2_idx)

            # ----- (1) D^{nm} contribution from this + element -----
            # D_kk = ∫ N^+_k dA, using STANDARD N (not modified M);
            # this is the *measure* the nonmortar node carries.  For a line-2
            # element with constant Jacobian J, ∫_-1^1 N_k(ξ) J dξ = J,
            # i.e. each endpoint receives J = (phys_hi - phys_lo)/2.
            for plus_node_idx in (plus_node1_idx, plus_node2_idx):
                if plus_node_idx < 0:
                    continue  # corner sentinel: row dropped
                D_nm[plus_node_idx] += plus_jacobian

            # ----- (2) A^m contribution: integrate over each - element overlap -----
            for minus_node1_idx, minus_node2_idx in minus_edge.elements:
                minus_phys_lo, minus_phys_hi = self._param_endpoints(
                    minus_edge, minus_node1_idx, minus_node2_idx,
                )
                if minus_phys_hi <= minus_phys_lo:
                    continue
                # Interval intersection in physical edge coords.
                overlap_phys_lo = max(plus_phys_lo, minus_phys_lo)
                overlap_phys_hi = min(plus_phys_hi, minus_phys_hi)
                if overlap_phys_hi - overlap_phys_lo <= 1e-14 * max(
                    abs(plus_phys_hi - plus_phys_lo), 1.0
                ):
                    continue
                self._integrate_overlap_segment(
                    A_m,
                    plus_local_nodes=(plus_node1_idx, plus_node2_idx),
                    minus_local_nodes=(minus_node1_idx, minus_node2_idx),
                    plus_parent_phys=(plus_phys_lo, plus_phys_hi),
                    minus_parent_phys=(minus_phys_lo, minus_phys_hi),
                    overlap_phys=(overlap_phys_lo, overlap_phys_hi),
                    corner_side=corner_side,
                )

        return MortarBlock2D(
            A_m=A_m,
            D_nm=D_nm,
            # `EdgeNodes2D` has `.name`; `EdgeInfo3D` has `.label`.
            # Accept either so the assembler is dim-agnostic.
            plus_edge_name=getattr(plus_edge, "name", None) or getattr(plus_edge, "label", ""),
            minus_edge_name=getattr(minus_edge, "name", None) or getattr(minus_edge, "label", ""),
        )

    # ---------------------------------------- segment-level integration ---
    def _integrate_overlap_segment(
        self,
        A_m: np.ndarray,
        plus_local_nodes: tuple[int, int],
        minus_local_nodes: tuple[int, int],
        plus_parent_phys: tuple[float, float],
        minus_parent_phys: tuple[float, float],
        overlap_phys: tuple[float, float],
        corner_side: str,
    ) -> None:
        """Integrate M_k(ξ_+) · N^-_l(ξ_-) over one overlap segment using
        3-point Gauss-Legendre quadrature, accumulating into A_m.

        Parametric maps (linear in physical edge coord):
            ξ_+ = (phys - plus_parent_mid)  / plus_parent_half_length
            ξ_- = (phys - minus_parent_mid) / minus_parent_half_length

        The Gauss points themselves are placed on the OVERLAP, parameterized
        by η ∈ [-1, 1]; the overlap Jacobian dphys / dη maps reference
        weight to physical weight.
        """
        overlap_phys_lo, overlap_phys_hi = overlap_phys
        # dphys / d(eta) on the overlap, where eta is the GL reference coord.
        overlap_jacobian = 0.5 * (overlap_phys_hi - overlap_phys_lo)
        overlap_phys_mid = 0.5 * (overlap_phys_hi + overlap_phys_lo)

        plus_phys_lo, plus_phys_hi = plus_parent_phys
        plus_parent_mid         = 0.5 * (plus_phys_hi + plus_phys_lo)
        plus_parent_half_length = 0.5 * (plus_phys_hi - plus_phys_lo)

        minus_phys_lo, minus_phys_hi = minus_parent_phys
        minus_parent_mid         = 0.5 * (minus_phys_hi + minus_phys_lo)
        minus_parent_half_length = 0.5 * (minus_phys_hi - minus_phys_lo)

        plus_node1_idx, plus_node2_idx = plus_local_nodes
        minus_node1_idx, minus_node2_idx = minus_local_nodes

        for gp_eta, gp_weight in zip(_GL3_PTS, _GL3_WTS):
            # Physical edge coord at this Gauss point.
            phys_at_gp = overlap_phys_mid + overlap_jacobian * gp_eta
            # Reference coord on each parent element.
            xi_on_plus  = (phys_at_gp - plus_parent_mid)  / plus_parent_half_length
            xi_on_minus = (phys_at_gp - minus_parent_mid) / minus_parent_half_length

            # Dual basis on + element (with corner modification if applicable).
            if corner_side == "none":
                M_at_n1, M_at_n2 = M_line2_dual(xi_on_plus)
            else:
                M_at_n1, M_at_n2 = M_line2_dual_modified(xi_on_plus, corner_side)
            # Standard line-2 shape on - element.
            N_minus_at_n1, N_minus_at_n2 = N_line2(xi_on_minus)

            # Physical-coord weight: w_eta * (dphys / d eta).
            phys_weight = gp_weight * overlap_jacobian

            # Accumulate into A^m.  Drop rows for + corner sentinels
            # (those DOFs are Dirichlet) and cols for - corner sentinels
            # (those values are also prescribed = 0, so they don't need
            # constraint columns).
            for plus_node_idx, M_value in (
                (plus_node1_idx, M_at_n1),
                (plus_node2_idx, M_at_n2),
            ):
                if plus_node_idx < 0:
                    continue
                for minus_node_idx, N_value in (
                    (minus_node1_idx, N_minus_at_n1),
                    (minus_node2_idx, N_minus_at_n2),
                ):
                    if minus_node_idx < 0:
                        continue
                    A_m[plus_node_idx, minus_node_idx] += (
                        phys_weight * M_value * N_value
                    )

    # ------------------- parametric endpoint resolution (corner-aware) ---

    # Axis label → coords-column index. Maps both 2D edges (parametric
    # axis ∈ {"x", "y"}) and 3D edges (parametric axis ∈ {"x", "y",
    # "z"}); the assembler core math is fully dim-generic, so the same
    # _assemble_pair / _integrate_overlap_segment / _corner_side
    # machinery works for 3D edge pairs from EdgeInfo3D too. See
    # §11.8 Phase 3.3.A.
    _AXIS_TO_COLUMN: dict[str, int] = {"x": 0, "y": 1, "z": 2}

    def _param_endpoints(
        self, edge, node_a_idx: int, node_b_idx: int,
    ) -> tuple[float, float]:
        """Return (phys_lo, phys_hi) along the edge's parametric axis.

        Sentinels:
            -1 -> ``edge.edge_min`` (left along the parametric axis)
            -2 -> ``edge.edge_max`` (right along the parametric axis)
        Otherwise, look up the node's coordinate.

        Duck-typed on ``edge``: requires ``parametric_axis`` (str in
        {"x", "y", "z"}), ``edge_min``, ``edge_max``, and ``coords``
        as a 2D array with at least the parametric-axis column. Both
        ``EdgeNodes2D`` and ``EdgeInfo3D`` satisfy this contract.
        """
        axis = self._AXIS_TO_COLUMN[edge.parametric_axis]

        def coord_or_sentinel(node_idx: int) -> float:
            if node_idx == -1:
                return edge.edge_min
            if node_idx == -2:
                return edge.edge_max
            return edge.coords[node_idx, axis]

        a_phys = coord_or_sentinel(node_a_idx)
        b_phys = coord_or_sentinel(node_b_idx)
        if a_phys <= b_phys:
            return a_phys, b_phys
        return b_phys, a_phys

    @staticmethod
    def _corner_side(node1_idx: int, node2_idx: int) -> str:
        """Classify a + element by which local endpoint(s) are corner sentinels.

        Note on naming: "left"/"right" here refer to the LOCAL node
        ordering of the element (node 1 corresponds to local ξ=-1, node 2
        to local ξ=+1).  This is the convention the dual basis modifications
        in Eq. (C.2) are stated in (M_1 = 0 means "node 1 is corner").

        Because of how ``BoundaryClassifier2D`` builds element connectivity
        along an edge, in practice ``-1`` always sits at ``node1_idx`` and
        ``-2`` always sits at ``node2_idx``, so the sentinel-value test is
        not strictly necessary; we keep both branches for defensive symmetry.

        Returns
        -------
        str : one of {"left", "right", "both", "none"}
        """
        node1_is_corner = node1_idx in (-1, -2)
        node2_is_corner = node2_idx in (-1, -2)
        if node1_is_corner and node2_is_corner:
            return "both"
        if node1_is_corner:
            return "left"     # node 1 (local ξ=-1) is the corner
        if node2_is_corner:
            return "right"    # node 2 (local ξ=+1) is the corner
        return "none"
