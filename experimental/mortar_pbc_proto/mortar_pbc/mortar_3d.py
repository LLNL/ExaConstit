"""3D mortar machinery: shape functions, dual bases, Wohlmuth modifications.

WHAT
----
Pure-NumPy / Python implementations of the building blocks needed for 3D
mortar PBC face and edge coupling:

    Shape functions (standard FE Lagrange basis):
      - N_line2(xi)                            line-2: 1D, p=1
      - N_line3(xi)                            line-3: 1D, p=2 (lumped-positivity test only)
      - N_tri3(lam)                            tri-3: 2D simplex, p=1
      - N_tri6(lam)                            tri-6: 2D simplex, p=2 (lumped-positivity test only)
      - N_quad4(xi, eta)                       quad-4: 2D tensor, p=1
      - N_quad8(xi, eta)                       quad-8 serendipity (lumped-positivity test only)
      - N_quad9(xi, eta)                       quad-9 full Lagrangian (lumped-positivity test only)
      - N_tet4(lam)                            tet-4: 3D simplex, p=1
      - N_tet10(lam)                           tet-10 (lumped-positivity test only)

    Dual bases (closed-form per §4 of MORTAR_PBC_ARCHITECTURE.md):
      - M_tri3_dual(lam)                       tri-3 dual: M_i = 4 lam_i - 1     (eq. 4.19)
      - M_quad4_dual(xi, eta)                  quad-4 dual: tensor product       (eq. 4.16)
      - M_tet4_dual(lam)                       tet-4 dual: M_i = 5 lam_i - 1     (eq. 4.21)

    Wohlmuth modifications (§5.2, §5.3):
      - M_tri3_dual_modified(lam, boundary_nodes)    eqs. 5.5, 5.6
      - M_quad4_dual_modified(xi, eta, side_xi, side_eta)   eqs. 5.8, 5.10

    Quadrature (reference-element):
      - GAUSS_LINE_3PT       1D Gauss-Legendre 3-point (degree 5 exact)
      - GAUSS_QUAD_3X3       2D tensor 3x3 Gauss (degree 5 each direction)
      - GAUSS_TRI_3PT        2D triangle 3-point (degree 2 exact)
      - GAUSS_TET_4PT        3D tetrahedron 4-point (degree 2 exact)

    Lumped-positivity check:
      - lumped_positivity(N_func, quad_pts, quad_wts) -> ndarray of s_j

WHY
---
This module is the pure-Python (no MFEM, no MPI) layer that the
constraint builder consumes. Same architectural choice as ``mortar_2d.py``:
isolating the math from the FE infrastructure means we can unit-test
bi-orthogonality, partition-of-unity, and the lumped-positivity criterion
(§4.9.1 of MORTAR_PBC_ARCHITECTURE.md) without pyMFEM installed.

The line-3 / tri-6 / quad-8 / tet-10 shape functions are included **only
for the lumped-positivity precondition tests** (per the §4.9 obstruction
analysis). They are NOT used in mortar assembly because:
    - line-3, quad-9, hex-27: their dual bases (eqs. 4.25-4.27) are
      not implemented in Phase 3.2; deferred to Phase 6+ (higher-order
      primal field; see §4.12 recommendation for ExaConstit).
    - tri-6, tet-10, quad-8: strict bi-orthogonality fails (§4.9.2);
      requires basis-transformation (§4.10) or LOR (§4.11), again
      deferred to Phase 6+.

The lumped-positivity tests EXIST as guards against silently shipping
a broken dual when a new element type is added later. If a future
contributor adds ``M_quad8_dual`` and the quad-8 lumped diagonal is
negative (which it is), the test will refuse to PASS until they
implement the basis transformation properly.

REFERENCES
----------
* MORTAR_PBC_ARCHITECTURE.md §4 (dual basis derivations)
* MORTAR_PBC_ARCHITECTURE.md §4.9 (the obstruction at p>=2)
* MORTAR_PBC_ARCHITECTURE.md §5.2, §5.3 (Wohlmuth modifications)
* Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.
* Lamichhane & Wohlmuth (2002), Calcolo 39 (line-3 dual).
* Popp, Wohlmuth, Gee, Wall (2012), SIAM J Sci Comput 34 (basis transformation).
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


# =============================================================================
# Reference shape functions
# =============================================================================

# ----- 1D: line-2 (linear), line-3 (quadratic) --------------------------------

def N_line2(xi: float) -> Tuple[float, float]:
    """Line-2 (1D, p=1) standard shape functions on xi in [-1, +1].

    Returns (N_1, N_2) with N_1(xi) = (1-xi)/2, N_2(xi) = (1+xi)/2.
    """
    return 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)


def N_line3(xi: float) -> Tuple[float, float, float]:
    """Line-3 (1D, p=2) standard Lagrange shape functions on xi in [-1,+1].

    Node ordering: (left corner xi=-1, right corner xi=+1, mid-node xi=0).

    Returns (N_1, N_2, N_3) where:
        N_1(xi) = xi (xi - 1) / 2     [left corner, peak at xi=-1]
        N_2(xi) = xi (xi + 1) / 2     [right corner, peak at xi=+1]
        N_3(xi) = 1 - xi^2            [mid-node, peak at xi=0]
    """
    return (
        0.5 * xi * (xi - 1.0),
        0.5 * xi * (xi + 1.0),
        1.0 - xi * xi,
    )


# ----- 2D simplex: tri-3 (linear), tri-6 (quadratic) --------------------------

def N_tri3(lam: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Tri-3 (2D simplex, p=1) shape functions in barycentric coordinates.

    Node ordering: vertices (lam = (1,0,0), (0,1,0), (0,0,1)).

    Returns (N_1, N_2, N_3) = (lam_1, lam_2, lam_3).
    """
    return float(lam[0]), float(lam[1]), float(lam[2])


def N_tri6(lam: Tuple[float, float, float]) -> Tuple[
    float, float, float, float, float, float
]:
    """Tri-6 (2D simplex, p=2) shape functions in barycentric coordinates.

    Node ordering: 3 corners (vertices), then 3 mid-edge nodes:
        N_1, N_2, N_3 : corners at lam = (1,0,0), (0,1,0), (0,0,1)
        N_4 : mid-edge between vertices 1-2 (lam = (1/2, 1/2, 0))
        N_5 : mid-edge between vertices 2-3 (lam = (0, 1/2, 1/2))
        N_6 : mid-edge between vertices 3-1 (lam = (1/2, 0, 1/2))

    Formulas (standard quadratic Lagrange on simplex):
        N_corner_i = lam_i (2 lam_i - 1)
        N_midedge_ij = 4 lam_i lam_j

    Per §4.9.2 of MORTAR_PBC_ARCHITECTURE.md, the corner integrals
    integrate to ZERO on the reference triangle, which is the
    obstruction to strict bi-orthogonality.
    """
    l1, l2, l3 = float(lam[0]), float(lam[1]), float(lam[2])
    return (
        l1 * (2.0 * l1 - 1.0),    # corner 1
        l2 * (2.0 * l2 - 1.0),    # corner 2
        l3 * (2.0 * l3 - 1.0),    # corner 3
        4.0 * l1 * l2,            # mid-edge 1-2
        4.0 * l2 * l3,            # mid-edge 2-3
        4.0 * l3 * l1,            # mid-edge 3-1
    )


# ----- 2D tensor: quad-4, quad-8 (serendipity), quad-9 (full Lagrangian) -----

def N_quad4(xi: float, eta: float) -> Tuple[float, float, float, float]:
    """Quad-4 (bilinear) standard shape functions on (xi, eta) in [-1,+1]^2.

    Node ordering (standard counter-clockwise from (-1,-1)):
        N_1 at (-1, -1)
        N_2 at (+1, -1)
        N_3 at (+1, +1)
        N_4 at (-1, +1)
    """
    return (
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    )


def N_quad8(xi: float, eta: float) -> Tuple[
    float, float, float, float, float, float, float, float
]:
    """Quad-8 serendipity standard shape functions on (xi, eta) in [-1,+1]^2.

    Node ordering: 4 corners, then 4 mid-edge nodes (no central bubble):
        N_1..N_4 : corners (-1,-1), (+1,-1), (+1,+1), (-1,+1)
        N_5..N_8 : mid-edges (0,-1), (+1,0), (0,+1), (-1,0)

    Formulas (standard serendipity, e.g. Zienkiewicz & Taylor):
        N_corner_i = (1/4)(1+xi*xi_i)(1+eta*eta_i)(xi*xi_i + eta*eta_i - 1)
        N_midedge in xi-direction (xi_i=0):
            (1/2)(1 - xi^2)(1 + eta*eta_i)
        N_midedge in eta-direction (eta_i=0):
            (1/2)(1 + xi*xi_i)(1 - eta^2)

    Per §4.9.2: corner lumped integrals are NEGATIVE (s_corner = -2/3 * |E|/8
    per Lamichhane-Wohlmuth 2004 calculation), which breaks the strict
    bi-orthogonality construction.
    """
    # Corner shape functions: encode the corner sign vectors.
    xi_signs = (-1.0, +1.0, +1.0, -1.0)
    eta_signs = (-1.0, -1.0, +1.0, +1.0)
    Ns_corner = tuple(
        0.25 * (1.0 + xi * xi_signs[i]) * (1.0 + eta * eta_signs[i])
        * (xi * xi_signs[i] + eta * eta_signs[i] - 1.0)
        for i in range(4)
    )
    # Mid-edge shape functions.
    N5 = 0.5 * (1.0 - xi * xi) * (1.0 - eta)   # bottom edge midnode (0,-1)
    N6 = 0.5 * (1.0 + xi) * (1.0 - eta * eta)  # right edge midnode (+1,0)
    N7 = 0.5 * (1.0 - xi * xi) * (1.0 + eta)   # top edge midnode (0,+1)
    N8 = 0.5 * (1.0 - xi) * (1.0 - eta * eta)  # left edge midnode (-1,0)
    return Ns_corner + (N5, N6, N7, N8)


def N_quad9(xi: float, eta: float) -> Tuple[
    float, float, float, float, float, float, float, float, float
]:
    """Quad-9 full-Lagrangian biquadratic shape functions on [-1,+1]^2.

    Tensor product of line-3 in xi and line-3 in eta.

    Node ordering: 4 corners, 4 mid-edges, 1 centroid.
        N_1..N_4 : corners (-1,-1), (+1,-1), (+1,+1), (-1,+1)
        N_5..N_8 : mid-edges (0,-1), (+1,0), (0,+1), (-1,0)
        N_9      : centroid (0, 0)

    Per §4.9.3: all 9 lumped integrals are positive (the central bubble
    absorbs the redistribution that would otherwise zero out corner
    integrals), so strict bi-orthogonality EXISTS via tensor product
    of the line-3 dual.
    """
    Nx_left, Nx_right, Nx_mid = N_line3(xi)
    Ny_left, Ny_right, Ny_mid = N_line3(eta)
    return (
        Nx_left * Ny_left,        # corner 1: (-1,-1)
        Nx_right * Ny_left,       # corner 2: (+1,-1)
        Nx_right * Ny_right,      # corner 3: (+1,+1)
        Nx_left * Ny_right,       # corner 4: (-1,+1)
        Nx_mid * Ny_left,         # mid-edge 5: (0,-1)
        Nx_right * Ny_mid,        # mid-edge 6: (+1,0)
        Nx_mid * Ny_right,        # mid-edge 7: (0,+1)
        Nx_left * Ny_mid,         # mid-edge 8: (-1,0)
        Nx_mid * Ny_mid,          # centroid 9
    )


# ----- 3D simplex: tet-4 (linear), tet-10 (quadratic) ------------------------

def N_tet4(
    lam: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Tet-4 (3D simplex, p=1) shape functions in barycentric coordinates.

    Node ordering: vertices (lam = e_1, e_2, e_3, e_4).
    Returns (N_1, N_2, N_3, N_4) = (lam_1, lam_2, lam_3, lam_4).
    """
    return float(lam[0]), float(lam[1]), float(lam[2]), float(lam[3])


def N_tet10(
    lam: Tuple[float, float, float, float],
) -> Tuple[
    float, float, float, float, float, float, float, float, float, float
]:
    """Tet-10 (3D simplex, p=2) shape functions in barycentric coordinates.

    Node ordering: 4 corners, then 6 mid-edges:
        N_1..N_4 : corners at lam = e_1, e_2, e_3, e_4
        N_5..N_10 : mid-edges (1-2), (2-3), (3-1), (1-4), (2-4), (3-4)

    Per §4.9.3: corner lumped integrals integrate to ZERO on the
    reference tetrahedron (same mechanism as tri-6).
    """
    l1, l2, l3, l4 = (float(lam[i]) for i in range(4))
    return (
        l1 * (2.0 * l1 - 1.0),    # corner 1
        l2 * (2.0 * l2 - 1.0),    # corner 2
        l3 * (2.0 * l3 - 1.0),    # corner 3
        l4 * (2.0 * l4 - 1.0),    # corner 4
        4.0 * l1 * l2,            # mid-edge 1-2
        4.0 * l2 * l3,            # mid-edge 2-3
        4.0 * l3 * l1,            # mid-edge 3-1
        4.0 * l1 * l4,            # mid-edge 1-4
        4.0 * l2 * l4,            # mid-edge 2-4
        4.0 * l3 * l4,            # mid-edge 3-4
    )


# =============================================================================
# Dual bases (Phase 3.2 actively-used; Phase 6+ for higher orders)
# =============================================================================

def M_line2_dual(xi: float) -> Tuple[float, float]:
    """Line-2 dual basis (eq. 4.10 simplified, d=1).

    M_i(xi) = (d+2) N_i - 1 with d=1 gives M_i = 3 N_i - 1.
    Equivalent forms:
        M_1(xi) = (1 - 3 xi) / 2
        M_2(xi) = (1 + 3 xi) / 2
    """
    return 0.5 * (1.0 - 3.0 * xi), 0.5 * (1.0 + 3.0 * xi)


def M_tri3_dual(
    lam: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Tri-3 dual basis (eq. 4.19 of MORTAR_PBC_ARCHITECTURE.md).

    Closed form via the unified simplex formula M_i = (d+2) N_i - 1 with
    d=2:
        M_i(lam) = 4 lam_i - 1

    Bi-orthogonality on the reference triangle T (|T| = 1/2):
        int_T M_i N_j dA = delta_ij * (|T|/3)

    Partition of unity:
        sum_i M_i = 4 (lam_1 + lam_2 + lam_3) - 3 = 4 - 3 = 1
    """
    l1, l2, l3 = float(lam[0]), float(lam[1]), float(lam[2])
    return 4.0 * l1 - 1.0, 4.0 * l2 - 1.0, 4.0 * l3 - 1.0


def M_quad4_dual(xi: float, eta: float) -> Tuple[float, float, float, float]:
    """Quad-4 dual basis (eq. 4.16 of MORTAR_PBC_ARCHITECTURE.md).

    Tensor product of the line-2 dual:
        M_i(xi, eta) = M_line2_dual(xi)_i_xi * M_line2_dual(eta)_i_eta

    Node ordering matches N_quad4: (-1,-1), (+1,-1), (+1,+1), (-1,+1).

    Bi-orthogonality on [-1,+1]^2 (|E| = 4):
        int_E M_i N_j dA = delta_ij * (|E|/4) = delta_ij * 1

    Partition of unity:
        sum_i M_i = (M_xi_l + M_xi_r) (M_eta_l + M_eta_r)
                  = 1 * 1 = 1   (since line-2 dual's PoU is 1)
    """
    M_xi_l, M_xi_r = M_line2_dual(xi)
    M_eta_l, M_eta_r = M_line2_dual(eta)
    return (
        M_xi_l * M_eta_l,    # node 1: (-1, -1)
        M_xi_r * M_eta_l,    # node 2: (+1, -1)
        M_xi_r * M_eta_r,    # node 3: (+1, +1)
        M_xi_l * M_eta_r,    # node 4: (-1, +1)
    )


def M_tet4_dual(
    lam: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Tet-4 dual basis (eq. 4.21 of MORTAR_PBC_ARCHITECTURE.md).

    Closed form via the unified simplex formula M_i = (d+2) N_i - 1 with
    d=3:
        M_i(lam) = 5 lam_i - 1

    Bi-orthogonality on the reference tet (|T| = 1/6):
        int_T M_i N_j dV = delta_ij * (|T|/4)

    Note: tet-4 dual is used for VOLUME mortar (e.g. mortared
    multi-domain problems with tet meshes); face mortar on tet meshes
    uses tri-3 face elements with M_tri3_dual. This function is
    documented for completeness and future use.
    """
    return tuple(5.0 * float(lam[i]) - 1.0 for i in range(4))  # type: ignore[return-value]


# =============================================================================
# Wohlmuth corner/edge modifications (eqs. 5.5, 5.6, 5.8, 5.10)
# =============================================================================

def M_line2_dual_modified(
    xi: float, side: str,
) -> Tuple[float, float]:
    """Wohlmuth-modified line-2 dual basis (Lopes 2021 Eq. C.2).

    Parameters
    ----------
    xi : float
        Reference coord (passthrough; ignored when modification active).
    side : {"none", "left", "right", "both"}
        Identifies which endpoint is a Dirichlet corner:
            "none"  : no corner; standard dual M_line2_dual(xi).
            "left"  : node 1 (xi=-1) is corner -> M_1 = 0, M_2 = 1.
            "right" : node 2 (xi=+1) is corner -> M_1 = 1, M_2 = 0.
            "both"  : both endpoints corners -> M_1 = M_2 = 0.

    Returns
    -------
    (M_1, M_2) : tuple[float, float]

    Notes
    -----
    The "none" case is added in Phase 3.2 (vs. the 2D ``mortar_2d``
    module's same-named function which only accepts {left, right, both})
    so that the quad-4 modification can use a single tensor-product call
    even when only one parametric direction is modified.
    """
    if side == "none":
        return M_line2_dual(xi)
    if side == "left":
        return 0.0, 1.0
    if side == "right":
        return 1.0, 0.0
    if side == "both":
        return 0.0, 0.0
    raise ValueError(
        f"Unknown corner side {side!r}; expected 'none', 'left', 'right', or 'both'"
    )


def M_tri3_dual_modified(
    lam: Tuple[float, float, float],
    boundary_nodes: Tuple[bool, bool, bool],
) -> Tuple[float, float, float]:
    """Wohlmuth-modified tri-3 dual basis (eqs. 5.5, 5.6 of architecture doc).

    Parameters
    ----------
    lam : (lam_1, lam_2, lam_3)
        Barycentric coords on the reference triangle.
    boundary_nodes : (b_1, b_2, b_3)
        b_i = True iff vertex i is on a face-boundary feature (edge or
        corner of the parent face) and therefore the corresponding LM
        row should be dropped (M_i^mod = 0).

    Cases:
      0 boundary nodes: standard tri-3 dual (M_i = 4 lam_i - 1).
      1 boundary node: edge-adjacent modification (eq. 5.5):
                       For dropped vertex i, kept vertices j, k:
                           M_i = 0
                           M_j = 1/2 + 2 lam_j - 2 lam_k
                           M_k = 1/2 - 2 lam_j + 2 lam_k
      2 boundary nodes: corner-adjacent modification (eq. 5.6):
                       For non-dropped vertex i:
                           M_i = 1   (constant)
                           M_j = M_k = 0
      3 boundary nodes: all dropped:  M_i = M_j = M_k = 0.

    Notes
    -----
    The 1-boundary case is the most subtle: the formula above assumes
    we permute (lam, M) so that the dropped vertex is "vertex 1". In
    code we identify the dropped vertex's index and apply the formula
    over the appropriate triple of (kept_a_lam, kept_b_lam) pairs.

    Verification of (5.5) for the case where vertex 1 is dropped:
      M_2(lam) = 1/2 + 2 lam_2 - 2 lam_3
      M_3(lam) = 1/2 - 2 lam_2 + 2 lam_3
      M_2 + M_3 = 1   ✓ (partition of unity in the kept rows)
      int_T M_2 lam_2 dA = (1/2)(|T|/3) + 2(|T|/6) - 2(|T|/12)
                        = |T|/6 + |T|/3 - |T|/6 = |T|/3   ✓ (target met)
      int_T M_2 lam_3 dA = (1/2)(|T|/3) + 2(|T|/12) - 2(|T|/6)
                        = |T|/6 + |T|/6 - |T|/3 = 0       ✓ (off-diag = 0)
      int_T M_2 lam_1 dA = "leak" (intentional, harmless after corner
                        column zeroing of C).
    """
    n_dropped = sum(boundary_nodes)

    if n_dropped == 0:
        return M_tri3_dual(lam)

    if n_dropped == 3:
        return 0.0, 0.0, 0.0

    if n_dropped == 2:
        # Two corners dropped, one kept. The kept vertex's M is
        # identically 1 (eq. 5.6).
        result = [0.0, 0.0, 0.0]
        for i, b in enumerate(boundary_nodes):
            if not b:
                result[i] = 1.0
                break
        return tuple(result)  # type: ignore[return-value]

    # n_dropped == 1: edge-adjacent, eq. (5.5).
    # Identify dropped index and the two kept indices (in cyclic order).
    idx_dropped = boundary_nodes.index(True)
    # Kept indices: the other two, in cyclic order. For the (5.5)
    # formula we need to label them as "j" (the +2 lam_j coefficient
    # vertex) and "k" (the -2 lam_k coefficient vertex). The choice of
    # labeling is symmetric (swapping j<->k just swaps M_j <-> M_k),
    # so we go in (idx_dropped+1, idx_dropped+2) cyclic order.
    idx_j = (idx_dropped + 1) % 3
    idx_k = (idx_dropped + 2) % 3

    lam_j = float(lam[idx_j])
    lam_k = float(lam[idx_k])

    M_j = 0.5 + 2.0 * lam_j - 2.0 * lam_k
    M_k = 0.5 - 2.0 * lam_j + 2.0 * lam_k

    result = [0.0, 0.0, 0.0]
    result[idx_j] = M_j
    result[idx_k] = M_k
    # result[idx_dropped] stays 0.0
    return tuple(result)  # type: ignore[return-value]


def M_quad4_dual_modified(
    xi: float, eta: float,
    side_xi: str = "none",
    side_eta: str = "none",
) -> Tuple[float, float, float, float]:
    """Wohlmuth-modified quad-4 dual basis (eqs. 5.8, 5.10 of architecture doc).

    Parameters
    ----------
    xi, eta : float
        Reference coords on [-1, +1]^2.
    side_xi : {"none", "left", "right", "both"}
        Modification along the xi direction. "left" drops the xi=-1
        side (nodes 1 and 4); "right" drops the xi=+1 side (nodes 2
        and 3); "both" drops all four nodes; "none" = no xi modification.
    side_eta : {"none", "bottom", "top", "both"}
        Modification along the eta direction. "bottom" drops the eta=-1
        side (nodes 1 and 2); "top" drops the eta=+1 side (nodes 3 and
        4); "both" drops all four nodes; "none" = no eta modification.

    Returns
    -------
    (M_1, M_2, M_3, M_4) : tuple[float, float, float, float]
        Modified dual values at this Gauss point. Node ordering matches
        ``N_quad4``: 1 at (-1,-1), 2 at (+1,-1), 3 at (+1,+1), 4 at
        (-1,+1).

    Notes
    -----
    Tensor product structure (eq. 5.8, 5.10): we map ``side_eta`` from
    ("bottom"/"top") into the line-2 left/right convention and call
    ``M_line2_dual_modified`` twice; the quad-4 modified dual is then
    the outer product. This works because the line-2 modification is
    a per-direction operation and the quad-4 dual itself is built as
    a tensor product (eq. 4.16 / function ``M_quad4_dual``).
    """
    # Map side_eta to line-2 left/right semantics.
    side_eta_mapped = {
        "none": "none",
        "bottom": "left",
        "top": "right",
        "both": "both",
    }.get(side_eta)
    if side_eta_mapped is None:
        raise ValueError(
            f"Unknown side_eta {side_eta!r}; expected 'none', 'bottom', 'top', or 'both'"
        )

    M_xi_l, M_xi_r = M_line2_dual_modified(xi, side_xi)
    M_eta_l, M_eta_r = M_line2_dual_modified(eta, side_eta_mapped)

    return (
        M_xi_l * M_eta_l,    # node 1: (-1, -1)
        M_xi_r * M_eta_l,    # node 2: (+1, -1)
        M_xi_r * M_eta_r,    # node 3: (+1, +1)
        M_xi_l * M_eta_r,    # node 4: (-1, +1)
    )


# =============================================================================
# Reference-element quadrature rules
# =============================================================================

# 1D Gauss-Legendre, 3-point on [-1, +1] (degree-5 exact).
_GL3_PTS_1D: np.ndarray = np.array(
    [-np.sqrt(3.0 / 5.0), 0.0, +np.sqrt(3.0 / 5.0)], dtype=np.float64,
)
_GL3_WTS_1D: np.ndarray = np.array(
    [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64,
)


def gauss_line_3pt() -> Tuple[np.ndarray, np.ndarray]:
    """Return (pts, wts) for 3-point Gauss-Legendre on [-1, +1] (degree 5)."""
    return _GL3_PTS_1D.copy(), _GL3_WTS_1D.copy()


def gauss_quad_3x3() -> Tuple[np.ndarray, np.ndarray]:
    """Return (pts, wts) for 3x3 Gauss on [-1,+1]^2 (degree 5 each direction).

    pts has shape (9, 2); wts has shape (9,).
    """
    px, wx = gauss_line_3pt()
    pts = np.empty((9, 2), dtype=np.float64)
    wts = np.empty(9, dtype=np.float64)
    k = 0
    for i in range(3):
        for j in range(3):
            pts[k, 0] = px[i]
            pts[k, 1] = px[j]
            wts[k] = wx[i] * wx[j]
            k += 1
    return pts, wts


def gauss_tri_3pt() -> Tuple[np.ndarray, np.ndarray]:
    """Return (pts_bary, wts) for 3-point degree-2 rule on the reference
    triangle T with |T| = 1/2.

    Reference triangle: T = {lam in R^3 : lam_i >= 0, sum lam_i = 1}.

    Returns
    -------
    pts_bary : (3, 3) ndarray
        Barycentric coordinates of each Gauss point.
    wts : (3,) ndarray
        Quadrature weights, summing to |T| = 1/2.

    Reference: e.g. Strang & Fix (1973). Exact for polynomials of
    total degree <= 2 on the simplex.
    """
    pts = np.array([
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ], dtype=np.float64)
    # Each weight = |T|/3 with |T| = 1/2 ; sum = |T| = 1/2.
    wts = np.full(3, 1.0 / 6.0, dtype=np.float64)
    return pts, wts


def gauss_tet_4pt() -> Tuple[np.ndarray, np.ndarray]:
    """Return (pts_bary, wts) for 4-point degree-2 rule on the reference
    tetrahedron T with |T| = 1/6.

    Reference tet: T = {lam in R^4 : lam_i >= 0, sum lam_i = 1}.

    Returns
    -------
    pts_bary : (4, 4) ndarray
        Barycentric coordinates.
    wts : (4,) ndarray
        Quadrature weights, summing to |T| = 1/6.

    Standard symmetric rule, exact for polynomials of total degree <= 2:
        a = (5 + 3 sqrt(5)) / 20  ≈ 0.5854...
        b = (5 -   sqrt(5)) / 20  ≈ 0.1382...
        Each Gauss pt is a permutation of (a, b, b, b).
    """
    a = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    b = (5.0 - np.sqrt(5.0)) / 20.0
    pts = np.array([
        [a, b, b, b],
        [b, a, b, b],
        [b, b, a, b],
        [b, b, b, a],
    ], dtype=np.float64)
    # Each weight = |T|/4 with |T| = 1/6 ; sum = 1/6.
    wts = np.full(4, 1.0 / 24.0, dtype=np.float64)
    return pts, wts


# =============================================================================
# Lumped-positivity check (the §4.9.1 criterion)
# =============================================================================

def lumped_positivity(
    N_func: Callable,
    quad_pts: np.ndarray,
    quad_wts: np.ndarray,
    n_basis: int,
    *,
    use_tuple_input: bool = True,
) -> np.ndarray:
    """Compute the lumped diagonal s_j = int_E N_j dE for every shape function.

    Per §4.9.1 of MORTAR_PBC_ARCHITECTURE.md, strict bi-orthogonal
    locally-supported dual basis exists iff every s_j is nonzero (and
    ideally positive). This function is the O(1) precondition test for
    new element types.

    Parameters
    ----------
    N_func : callable
        Shape function evaluator. Either takes a barycentric tuple
        (lam_1, ..., lam_d+1) — for simplices — or a reference coord
        tuple (xi, eta, ...) — for tensor-product elements. The
        ``use_tuple_input`` flag controls which calling convention.
    quad_pts : (Nq, dim) or (Nq, d+1) ndarray
        Quadrature points: barycentric for simplices, reference coords
        for tensor-product. The function unpacks and passes via *args
        if ``use_tuple_input=False``, or wraps in a tuple otherwise.
    quad_wts : (Nq,) ndarray
        Quadrature weights.
    n_basis : int
        Number of shape functions returned by N_func.
    use_tuple_input : bool, default True
        If True, N_func is called as N_func(quad_pts[q]) (good for
        barycentric simplex shape functions which take a tuple of
        lam's). If False, N_func is called as N_func(*quad_pts[q])
        (good for tensor-product shape functions which take xi, eta
        as separate args).

    Returns
    -------
    s : (n_basis,) ndarray
        s[j] = int_E N_j dE, computed by the supplied quadrature.

    Notes
    -----
    Expected outcomes per the §4.9 obstruction analysis:
        line-2:  s = (1, 1)                         all positive
        line-3:  s = (1/3, 1/3, 4/3)                all positive
        tri-3:   s = (1/6, 1/6, 1/6) = |T|/3 each   all positive
        tri-6:   s_corner = 0,  s_midedge = |T|/3   FAILURE: corners zero
        quad-4:  s = (1, 1, 1, 1) = |E|/4 each      all positive
        quad-8:  s_corner = -1/3, s_midedge = +4/3  FAILURE: corners negative
        quad-9:  s_corner=1/9,s_midedge=4/9,s_centroid=16/9  all positive
        tet-4:   s = (1/24, 1/24, 1/24, 1/24) = |T|/4 each   all positive
        tet-10:  s_corner = 0, s_midedge = positive       FAILURE: corners zero

    Tests in tests/test_mortar_3d_unit.py verify these expected values.
    """
    s = np.zeros(n_basis, dtype=np.float64)
    for q, w in zip(quad_pts, quad_wts):
        if use_tuple_input:
            N_vals = N_func(tuple(q))
        else:
            N_vals = N_func(*q)
        for j in range(n_basis):
            s[j] += w * float(N_vals[j])
    return s
