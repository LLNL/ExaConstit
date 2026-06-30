"""Unit tests for the 3D mortar machinery (Phase 3.2).

These verify the building blocks that don't require pyMFEM:

  * Lumped-positivity precondition (s_j > 0 per §4.9.1) for ALL element
    types currently in the prototype roadmap, including the failing
    cases (tri-6, quad-8, tet-10) which serve as guards.
  * Bi-orthogonality of the implemented dual bases (tri-3, quad-4,
    tet-4) on their reference elements.
  * Partition of unity of both the standard FE bases and the dual
    bases (sum_i N_i = sum_i M_i = 1).
  * Wohlmuth modifications (tri-3 edge-/corner-adjacent, quad-4
    edge-/corner-adjacent) preserve PoU in the kept rows and break
    bi-orthogonality only as predicted.
  * Pure-Python parts of types_3d.CornerInfo3D (no MFEM).

Run with:
    python tests/test_mortar_3d_unit.py
"""
from __future__ import annotations

import os
import sys

# ----------------------------------------------------------------------
# Defensive path setup — see test_face_mortar_3d.py for full rationale.
# Briefly: prefer the local `mortar_pbc/` over any stale `pip install -e`
# of an older prototype, and diagnose loudly if Python still resolves
# elsewhere.
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_LOCAL_PKG = os.path.join(_PARENT, "mortar_pbc")
if not os.path.isdir(_LOCAL_PKG):
    raise RuntimeError(
        f"Cannot find mortar_pbc package at {_LOCAL_PKG!r}."
    )
sys.path.insert(0, _PARENT)
for _mod_name in list(sys.modules.keys()):
    if _mod_name == "mortar_pbc" or _mod_name.startswith("mortar_pbc."):
        del sys.modules[_mod_name]

import mortar_pbc                                                    # noqa: E402
_actual_pkg_dir = os.path.realpath(os.path.dirname(mortar_pbc.__file__))
_expected_pkg_dir = os.path.realpath(_LOCAL_PKG)
if _actual_pkg_dir != _expected_pkg_dir:
    raise RuntimeError(
        f"\n  mortar_pbc resolved to a DIFFERENT location than expected:\n"
        f"      resolved : {_actual_pkg_dir}\n"
        f"      expected : {_expected_pkg_dir}\n\n"
        f"  This usually means your Python environment has a stale\n"
        f"  `pip install -e <some-older-prototype>/`. Likely fixes:\n\n"
        f"      pip uninstall mortar-pbc          # remove the stale install\n"
        f"      pip show mortar-pbc               # see what's currently installed\n"
        f"      unset PYTHONPATH                  # clear any env override\n"
    )

import numpy as np                                                    # noqa: E402

from mortar_pbc.mortar_3d import (                                    # noqa: E402
    # shape functions
    N_line2, N_line3,
    N_tri3, N_tri6,
    N_quad4, N_quad8, N_quad9,
    N_tet4, N_tet10,
    # dual bases
    M_line2_dual, M_tri3_dual, M_quad4_dual, M_tet4_dual,
    # Wohlmuth modifications
    M_line2_dual_modified,
    M_tri3_dual_modified, M_quad4_dual_modified,
    # quadrature
    gauss_line_3pt, gauss_quad_3x3, gauss_tri_3pt, gauss_tet_4pt,
    # the §4.9.1 criterion
    lumped_positivity,
)
from mortar_pbc.types_3d import CornerInfo3D                          # noqa: E402


# =============================================================================
# §4.9.1 LUMPED-POSITIVITY PRECONDITION TESTS
# =============================================================================
#
# These compute s_j = int_E N_j dE for each element type and assert the
# expected sign pattern. The "PASS-list" elements (line-2, line-3, tri-3,
# quad-4, quad-9, tet-4) have all-positive s; the "FAIL-list" elements
# (tri-6, quad-8, tet-10) have some s_j zero or negative, which is the
# §4.9 obstruction. The FAIL-list tests are EXPECTED FAILURES of the
# strict construction; we test that they fail in the documented way to
# guard against silent breakage when a new element type is added later.
# =============================================================================

def test_lumped_positivity_line2():
    """Line-2: s = (1, 1), both positive. Standard PASS case."""
    pts, wts = gauss_line_3pt()
    # N_line2(xi) takes single arg; wrap to match signature.
    s = lumped_positivity(
        lambda x: N_line2(x[0]),
        pts.reshape(-1, 1), wts, n_basis=2, use_tuple_input=True,
    )
    expected = np.array([1.0, 1.0])  # |E|/2 each on |E|=2
    err = np.max(np.abs(s - expected))
    assert err < 1e-12, f"line-2 lumped: s = {s}, expected {expected}"
    assert (s > 0).all()
    print(f"  PASS  line-2 lumped positivity: s = ({s[0]:.4f}, {s[1]:.4f})  "
          f"all > 0, err vs expected = {err:.2e}")


def test_lumped_positivity_line3():
    """Line-3 (1D, p=2): s = (1/3, 1/3, 4/3), all positive (§4.8 verifies).

    This is the SUFFICIENT condition that the strict line-3 dual
    (eq. 4.25) exists.
    """
    pts, wts = gauss_line_3pt()
    s = lumped_positivity(
        lambda x: N_line3(x[0]),
        pts.reshape(-1, 1), wts, n_basis=3, use_tuple_input=True,
    )
    expected = np.array([1.0 / 3.0, 1.0 / 3.0, 4.0 / 3.0])
    err = np.max(np.abs(s - expected))
    assert err < 1e-12, f"line-3 lumped: s = {s}, expected {expected}"
    assert (s > 0).all()
    print(f"  PASS  line-3 lumped positivity: s = ({s[0]:.4f}, {s[1]:.4f}, "
          f"{s[2]:.4f})  all > 0, err = {err:.2e}")


def test_lumped_positivity_tri3():
    """Tri-3: s = (|T|/3, |T|/3, |T|/3) = (1/6, 1/6, 1/6) all positive."""
    pts, wts = gauss_tri_3pt()
    s = lumped_positivity(N_tri3, pts, wts, n_basis=3, use_tuple_input=True)
    expected = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    err = np.max(np.abs(s - expected))
    assert err < 1e-12, f"tri-3 lumped: s = {s}, expected {expected}"
    assert (s > 0).all()
    print(f"  PASS  tri-3 lumped positivity: s = ({s[0]:.4f}, {s[1]:.4f}, "
          f"{s[2]:.4f})  all > 0, err = {err:.2e}")


def test_lumped_positivity_tri6_failure():
    """Tri-6: corner s vanishes (§4.9.2). FAIL-list precondition guard.

    Per eq. (4.28): s_corner = 2 * int lam^2 - int lam = 2(|T|/6) - |T|/3
    = |T|/3 - |T|/3 = 0.

    This test asserts the FAILURE: we EXPECT s_corner = 0 to within
    quadrature noise; if a future contributor changes the shape
    functions or the rule misbehaves, this catches it.
    """
    pts, wts = gauss_tri_3pt()
    s = lumped_positivity(N_tri6, pts, wts, n_basis=6, use_tuple_input=True)
    # Corners 1, 2, 3 should integrate to 0.
    s_corners = s[:3]
    s_midedges = s[3:]
    err_corners = np.max(np.abs(s_corners))
    expected_midedge = 1.0 / 6.0  # = |T|/3 with |T|=1/2; 4 lam_i lam_j integrates to 2|T|/12 * 4 = 2|T|/3 = 1/3 -- wait, check this.
    # Actually for tri-6 mid-edge: N_4 = 4 lam_1 lam_2.
    # int N_4 dA = 4 int lam_1 lam_2 dA = 4 * (|T|/12) = |T|/3 = 1/6.
    err_midedges = np.max(np.abs(s_midedges - expected_midedge))
    assert err_corners < 1e-12, f"tri-6 corner s should be 0; got {s_corners}"
    assert err_midedges < 1e-12, f"tri-6 mid-edge s = |T|/3; got {s_midedges}"
    assert (s_corners == 0).all() | np.isclose(s_corners, 0, atol=1e-13).all()
    assert (s_midedges > 0).all()
    print(f"  PASS  tri-6 lumped positivity (FAIL-list): "
          f"s_corner = {s_corners.tolist()} (== 0, obstruction confirmed); "
          f"s_midedge = {s_midedges[0]:.4f} > 0")


def test_lumped_positivity_quad4():
    """Quad-4: s = (1, 1, 1, 1) all positive. PASS case."""
    pts, wts = gauss_quad_3x3()
    s = lumped_positivity(
        lambda xy: N_quad4(xy[0], xy[1]),
        pts, wts, n_basis=4, use_tuple_input=True,
    )
    expected = np.array([1.0, 1.0, 1.0, 1.0])  # |E|/4 each on |E|=4
    err = np.max(np.abs(s - expected))
    assert err < 1e-12, f"quad-4 lumped: s = {s}, expected {expected}"
    assert (s > 0).all()
    print(f"  PASS  quad-4 lumped positivity: s = {tuple(round(si, 4) for si in s)} "
          f" all > 0, err = {err:.2e}")


def test_lumped_positivity_quad8_failure():
    """Quad-8 (serendipity): corner s NEGATIVE (§4.9.2). FAIL-list guard.

    Per Lamichhane & Wohlmuth (2004): the lack of central bubble in
    serendipity elements leaves corner integrals negative. Specifically
    for the 8-node quad on [-1,+1]^2 (|E| = 4):
        s_corner = -|E|/12 = -1/3
        s_midedge = +|E|/3 =  4/3
    """
    pts, wts = gauss_quad_3x3()
    s = lumped_positivity(
        lambda xy: N_quad8(xy[0], xy[1]),
        pts, wts, n_basis=8, use_tuple_input=True,
    )
    s_corners = s[:4]
    s_midedges = s[4:]
    err_corners = np.max(np.abs(s_corners - (-1.0 / 3.0)))
    err_midedges = np.max(np.abs(s_midedges - (4.0 / 3.0)))
    assert err_corners < 1e-10, f"quad-8 corner s should be -1/3; got {s_corners}"
    assert err_midedges < 1e-10, f"quad-8 mid-edge s should be 4/3; got {s_midedges}"
    assert (s_corners < 0).all()
    assert (s_midedges > 0).all()
    print(f"  PASS  quad-8 lumped positivity (FAIL-list): "
          f"s_corner = {s_corners[0]:.4f} (< 0, obstruction confirmed); "
          f"s_midedge = {s_midedges[0]:.4f}")


def test_lumped_positivity_quad9():
    """Quad-9 (full Lagrangian): all s positive (§4.9.3). PASS case.

    Tensor product of line-3 lumped weights:
      Corner:   (1/3) * (1/3) = 1/9
      Mid-edge: (1/3) * (4/3) = 4/9   (or (4/3)*(1/3) symmetrically)
      Centroid: (4/3) * (4/3) = 16/9
    Sum: 4*(1/9) + 4*(4/9) + 16/9 = 4/9 + 16/9 + 16/9 = 36/9 = 4 = |E|. ✓
    """
    pts, wts = gauss_quad_3x3()
    s = lumped_positivity(
        lambda xy: N_quad9(xy[0], xy[1]),
        pts, wts, n_basis=9, use_tuple_input=True,
    )
    s_corners = s[:4]
    s_midedges = s[4:8]
    s_center = s[8]
    expected_corner = 1.0 / 9.0
    expected_midedge = 4.0 / 9.0
    expected_center = 16.0 / 9.0
    err = max(
        np.max(np.abs(s_corners - expected_corner)),
        np.max(np.abs(s_midedges - expected_midedge)),
        abs(s_center - expected_center),
    )
    assert err < 1e-12, f"quad-9 lumped: s = {s}; mismatch from analytics"
    assert (s > 0).all(), f"quad-9 expected all positive but got {s}"
    print(f"  PASS  quad-9 lumped positivity: s_corner = {s_corners[0]:.4f}, "
          f"s_midedge = {s_midedges[0]:.4f}, s_center = {s_center:.4f}  "
          f"all > 0 (tensor of line-3)")


def test_lumped_positivity_tet4():
    """Tet-4: s = (|T|/4, ...) = (1/24, 1/24, 1/24, 1/24) all positive."""
    pts, wts = gauss_tet_4pt()
    s = lumped_positivity(N_tet4, pts, wts, n_basis=4, use_tuple_input=True)
    expected = np.full(4, 1.0 / 24.0)
    err = np.max(np.abs(s - expected))
    assert err < 1e-12, f"tet-4 lumped: s = {s}, expected {expected}"
    assert (s > 0).all()
    print(f"  PASS  tet-4 lumped positivity: s = ({s[0]:.5f},) x 4  "
          f"all > 0, err = {err:.2e}")


def test_lumped_positivity_tet10_failure():
    """Tet-10: corner s NEGATIVE (-|T|/20 = -1/120). FAIL-list guard.

    UPDATED Phase 3.2 finding: the architecture doc §4.9.2 originally
    claimed tet-10 corner integrates to zero (by analogy with tri-6),
    but the actual arithmetic gives a *negative* value:

        s_corner_P2 = (2 - d) / ((d+1)(d+2)) * |T|

    For d=3 (tet), |T| = 1/6:
        s_corner = (2-3) / (4*5) * (1/6) = -1/(20*6) = -1/120

    This is qualitatively DIFFERENT from tri-6 (where s_corner = 0
    exactly). In 3D the tet-10 corner is structurally similar to the
    serendipity-element case rather than to its 2D analog tri-6 — the
    sign of the obstruction is dimension-dependent.

    Mid-edge value:
        s_midedge = ∫ 4 lam_i lam_j dV = 4 * (1/120) = 1/30

    Note: gauss_tet_4pt is degree-2 exact, which is sufficient because
    N_corner has degree 2.
    """
    pts, wts = gauss_tet_4pt()
    s = lumped_positivity(N_tet10, pts, wts, n_basis=10, use_tuple_input=True)
    s_corners = s[:4]
    s_midedges = s[4:]
    expected_corner = -1.0 / 120.0    # = -|T|/20
    expected_midedge = 1.0 / 30.0     # = 4 * |T|/20
    err_corners = np.max(np.abs(s_corners - expected_corner))
    err_midedges = np.max(np.abs(s_midedges - expected_midedge))
    assert err_corners < 1e-12, (
        f"tet-10 corner s should be -1/120 = {expected_corner}; got {s_corners}"
    )
    assert err_midedges < 1e-12, (
        f"tet-10 mid-edge s should be 1/30 = {expected_midedge}; got {s_midedges}"
    )
    assert (s_corners < 0).all()
    assert (s_midedges > 0).all()
    print(f"  PASS  tet-10 lumped positivity (FAIL-list): "
          f"s_corner = {s_corners[0]:.5f} (= -|T|/20 < 0, obstruction confirmed); "
          f"s_midedge = {s_midedges[0]:.5f}")


# =============================================================================
# BI-ORTHOGONALITY OF THE IMPLEMENTED DUAL BASES
# =============================================================================

def test_biorthogonality_line2():
    """int_{-1}^{+1} M_i N_j dxi = delta_ij * s_j  with s_j = 1."""
    pts, wts = gauss_line_3pt()
    M_NN = np.zeros((2, 2))
    for x, w in zip(pts, wts):
        M = M_line2_dual(x)
        N = N_line2(x)
        for i in range(2):
            for j in range(2):
                M_NN[i, j] += w * M[i] * N[j]
    err = np.max(np.abs(M_NN - np.eye(2)))
    assert err < 1e-12, f"line-2 biorth: M @ N = {M_NN}"
    print(f"  PASS  line-2 dual biorthogonality (max err = {err:.2e})")


def test_biorthogonality_tri3():
    """int_T M_i N_j dA = delta_ij * (|T|/3)   with M_tri3_dual."""
    pts, wts = gauss_tri_3pt()
    M_NN = np.zeros((3, 3))
    for q, w in zip(pts, wts):
        lam = tuple(q)
        M = M_tri3_dual(lam)
        N = N_tri3(lam)
        for i in range(3):
            for j in range(3):
                M_NN[i, j] += w * M[i] * N[j]
    expected = (1.0 / 6.0) * np.eye(3)  # |T|/3 = 1/6 per row
    err = np.max(np.abs(M_NN - expected))
    assert err < 1e-12, f"tri-3 biorth: M @ N = {M_NN}, expected diag(1/6) * 3"
    print(f"  PASS  tri-3 dual biorthogonality "
          f"(diag = ({M_NN[0,0]:.4f}, ...), max off-diag = "
          f"{np.max(np.abs(M_NN - np.diag(np.diag(M_NN)))):.2e})")


def test_biorthogonality_quad4():
    """int_E M_i N_j dA = delta_ij * (|E|/4) = delta_ij * 1   on quad-4."""
    pts, wts = gauss_quad_3x3()
    M_NN = np.zeros((4, 4))
    for q, w in zip(pts, wts):
        xi, eta = q
        M = M_quad4_dual(xi, eta)
        N = N_quad4(xi, eta)
        for i in range(4):
            for j in range(4):
                M_NN[i, j] += w * M[i] * N[j]
    err = np.max(np.abs(M_NN - np.eye(4)))
    assert err < 1e-12, f"quad-4 biorth: M @ N = {M_NN}"
    print(f"  PASS  quad-4 dual biorthogonality (max err = {err:.2e})")


def test_biorthogonality_tet4():
    """int_T M_i N_j dV = delta_ij * (|T|/4) = delta_ij * 1/24   on tet-4."""
    pts, wts = gauss_tet_4pt()
    M_NN = np.zeros((4, 4))
    for q, w in zip(pts, wts):
        lam = tuple(q)
        M = M_tet4_dual(lam)
        N = N_tet4(lam)
        for i in range(4):
            for j in range(4):
                M_NN[i, j] += w * M[i] * N[j]
    expected = (1.0 / 24.0) * np.eye(4)
    err = np.max(np.abs(M_NN - expected))
    assert err < 1e-12, f"tet-4 biorth: M @ N = {M_NN}, expected diag(1/24)"
    print(f"  PASS  tet-4 dual biorthogonality "
          f"(diag = ({M_NN[0,0]:.5f},) x 4, max off-diag = "
          f"{np.max(np.abs(M_NN - np.diag(np.diag(M_NN)))):.2e})")


# =============================================================================
# PARTITION OF UNITY (BOTH N AND M)
# =============================================================================

def test_partition_of_unity_dual_bases():
    """sum_i M_i = 1 for line-2, tri-3, quad-4, tet-4 dual bases."""
    # Line-2 at a few points.
    for xi in [-0.7, 0.0, 0.3, 0.9]:
        s = sum(M_line2_dual(xi))
        assert abs(s - 1.0) < 1e-14, f"line-2 dual PoU fail at xi={xi}: {s}"
    # Tri-3 at sample barycentric points.
    for lam in [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (1.0/3, 1.0/3, 1.0/3)]:
        s = sum(M_tri3_dual(lam))
        assert abs(s - 1.0) < 1e-14, f"tri-3 dual PoU fail at lam={lam}: {s}"
    # Quad-4 at sample (xi, eta).
    for xi, eta in [(-0.7, 0.3), (0.0, 0.0), (0.5, -0.4), (0.9, 0.9)]:
        s = sum(M_quad4_dual(xi, eta))
        assert abs(s - 1.0) < 1e-14, (
            f"quad-4 dual PoU fail at ({xi}, {eta}): {s}"
        )
    # Tet-4 at sample barycentric points.
    for lam in [(1.0, 0.0, 0.0, 0.0), (0.25, 0.25, 0.25, 0.25),
                (0.4, 0.3, 0.2, 0.1)]:
        s = sum(M_tet4_dual(lam))
        assert abs(s - 1.0) < 1e-14, f"tet-4 dual PoU fail at {lam}: {s}"
    print(f"  PASS  partition of unity for line-2, tri-3, quad-4, tet-4 dual bases")


def test_partition_of_unity_N_bases():
    """sum_i N_i = 1 for line-2, line-3, tri-3, tri-6, quad-4, quad-8,
    quad-9, tet-4, tet-10."""
    # Line-2, line-3.
    for xi in [-0.7, 0.0, 0.3, 0.9]:
        assert abs(sum(N_line2(xi)) - 1.0) < 1e-14
        assert abs(sum(N_line3(xi)) - 1.0) < 1e-14
    # Tri-3, tri-6.
    for lam in [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (1.0/3, 1.0/3, 1.0/3),
                (0.2, 0.3, 0.5)]:
        assert abs(sum(N_tri3(lam)) - 1.0) < 1e-14
        assert abs(sum(N_tri6(lam)) - 1.0) < 1e-14
    # Quad-4, quad-8, quad-9.
    for xi, eta in [(-0.7, 0.3), (0.0, 0.0), (0.5, -0.4), (0.9, 0.9),
                    (-1.0, -1.0), (1.0, 1.0)]:
        assert abs(sum(N_quad4(xi, eta)) - 1.0) < 1e-14
        assert abs(sum(N_quad8(xi, eta)) - 1.0) < 1e-13, (
            f"quad-8 PoU fail at ({xi}, {eta}): {sum(N_quad8(xi, eta))}"
        )
        assert abs(sum(N_quad9(xi, eta)) - 1.0) < 1e-13, (
            f"quad-9 PoU fail at ({xi}, {eta}): {sum(N_quad9(xi, eta))}"
        )
    # Tet-4, tet-10.
    for lam in [(1.0, 0.0, 0.0, 0.0), (0.25, 0.25, 0.25, 0.25),
                (0.4, 0.3, 0.2, 0.1)]:
        assert abs(sum(N_tet4(lam)) - 1.0) < 1e-14
        assert abs(sum(N_tet10(lam)) - 1.0) < 1e-14, (
            f"tet-10 PoU fail at {lam}: {sum(N_tet10(lam))}"
        )
    print(f"  PASS  partition of unity for all standard FE shape functions "
          f"(line-2, line-3, tri-3, tri-6, quad-4, quad-8, quad-9, tet-4, tet-10)")


# =============================================================================
# WOHLMUTH MODIFICATIONS
# =============================================================================

def test_wohlmuth_line2_modification_extended():
    """The 3D mortar_3d's M_line2_dual_modified now also accepts 'none'.
    Verify the 'none' case passes through to the standard dual."""
    for xi in [-0.7, 0.0, 0.5]:
        std = M_line2_dual(xi)
        mod = M_line2_dual_modified(xi, "none")
        assert mod[0] == std[0] and mod[1] == std[1], (
            f"line-2 'none' case should equal standard dual: "
            f"std = {std}, mod = {mod}"
        )
    # Sanity-check the existing left/right/both cases still work.
    assert M_line2_dual_modified(0.5, "left") == (0.0, 1.0)
    assert M_line2_dual_modified(0.5, "right") == (1.0, 0.0)
    assert M_line2_dual_modified(0.5, "both") == (0.0, 0.0)
    print(f"  PASS  line-2 dual modified: 'none' passthrough + left/right/both")


def test_wohlmuth_tri3_no_boundary():
    """0 boundary nodes: should equal standard tri-3 dual."""
    test_pts = [(0.5, 0.3, 0.2), (1.0/3, 1.0/3, 1.0/3), (0.7, 0.2, 0.1)]
    for lam in test_pts:
        std = M_tri3_dual(lam)
        mod = M_tri3_dual_modified(lam, (False, False, False))
        for i in range(3):
            assert abs(std[i] - mod[i]) < 1e-14, (
                f"tri-3 0-bdry case at {lam}: std={std}, mod={mod}"
            )
    print(f"  PASS  tri-3 modified (0 boundary nodes) = standard dual")


def test_wohlmuth_tri3_one_vertex_dropped():
    """1 boundary node: edge-adjacent (eq. 5.5).

    Verifies:
    - Dropped vertex's M = 0 identically.
    - Sum of kept M's = 1 identically (PoU on kept rows).
    - int M_kept_i N_kept_i = |T|/3 (target diagonal).
    - int M_kept_i N_kept_j (i!=j) = 0 (off-diag in kept block).
    """
    pts, wts = gauss_tri_3pt()
    # Try each of the 3 single-vertex-dropped configs.
    for idx_dropped in range(3):
        boundary_nodes = tuple(i == idx_dropped for i in range(3))
        idx_j = (idx_dropped + 1) % 3
        idx_k = (idx_dropped + 2) % 3

        # Check at sample points: dropped is 0, kept sum to 1.
        for q in pts:
            lam = tuple(q)
            M = M_tri3_dual_modified(lam, boundary_nodes)
            assert abs(M[idx_dropped]) < 1e-14, (
                f"tri-3 1-bdry: dropped vertex {idx_dropped} has M = "
                f"{M[idx_dropped]} != 0 at lam={lam}"
            )
            kept_sum = M[idx_j] + M[idx_k]
            assert abs(kept_sum - 1.0) < 1e-13, (
                f"tri-3 1-bdry: kept sum = {kept_sum} != 1 at lam={lam}"
            )

        # Quadrature check: int M_kept_i N_kept_j on the kept block.
        kept_block = np.zeros((2, 2))  # rows: kept M; cols: kept N
        kept_indices = [idx_j, idx_k]
        for q, w in zip(pts, wts):
            lam = tuple(q)
            M = M_tri3_dual_modified(lam, boundary_nodes)
            N = N_tri3(lam)
            for ii, ki in enumerate(kept_indices):
                for jj, kj in enumerate(kept_indices):
                    kept_block[ii, jj] += w * M[ki] * N[kj]

        expected = (1.0 / 6.0) * np.eye(2)  # |T|/3 = 1/6
        err = np.max(np.abs(kept_block - expected))
        assert err < 1e-12, (
            f"tri-3 1-bdry biorth on kept block (dropped={idx_dropped}): "
            f"got\n{kept_block}\nexpected\n{expected}"
        )
    print(f"  PASS  tri-3 modified (1 vertex dropped) for all 3 configs: "
          f"dropped row M=0, kept-block diag = |T|/3, off-diag = 0")


def test_wohlmuth_tri3_two_vertices_dropped():
    """2 boundary nodes: corner-adjacent (eq. 5.6) — kept vertex M = 1."""
    pts, wts = gauss_tri_3pt()
    for idx_kept in range(3):
        boundary_nodes = tuple(i != idx_kept for i in range(3))
        for q in pts:
            lam = tuple(q)
            M = M_tri3_dual_modified(lam, boundary_nodes)
            for i in range(3):
                if i == idx_kept:
                    assert abs(M[i] - 1.0) < 1e-14
                else:
                    assert abs(M[i]) < 1e-14
        # Bi-orthogonality on the kept (1x1) block:
        # int M_kept N_kept = int 1 * lam_kept dA = |T|/3.
        accum = 0.0
        for q, w in zip(pts, wts):
            lam = tuple(q)
            M = M_tri3_dual_modified(lam, boundary_nodes)
            N = N_tri3(lam)
            accum += w * M[idx_kept] * N[idx_kept]
        assert abs(accum - 1.0 / 6.0) < 1e-12, (
            f"tri-3 2-bdry biorth: int M N = {accum}, expected 1/6"
        )
    print(f"  PASS  tri-3 modified (2 vertices dropped) for all 3 configs: "
          f"kept M = 1 (constant), int M N = |T|/3")


def test_wohlmuth_tri3_three_vertices_dropped():
    """3 boundary nodes: degenerate, all M = 0."""
    for q in gauss_tri_3pt()[0]:
        lam = tuple(q)
        M = M_tri3_dual_modified(lam, (True, True, True))
        for i in range(3):
            assert M[i] == 0.0
    print(f"  PASS  tri-3 modified (3 vertices dropped): all M = 0")


def test_wohlmuth_quad4_edge_adjacent():
    """Quad-4 edge-adjacent (eq. 5.8).

    Configuration: bottom edge (eta = -1, nodes 1 & 2) is on the
    face-boundary edge. side_eta = 'bottom'. Expected:
        M_1 = M_2 = 0
        M_3 = (1 + 3 xi)/2     (line-2 dual at xi, with eta-side = 1)
        M_4 = (1 - 3 xi)/2
        sum M = 1 (PoU)
    """
    pts, wts = gauss_quad_3x3()
    sample_xi = [-0.5, 0.0, 0.5]
    for xi_val in sample_xi:
        eta_val = 0.3
        M = M_quad4_dual_modified(xi_val, eta_val,
                                  side_xi="none", side_eta="bottom")
        assert abs(M[0]) < 1e-14, f"quad-4 edge-adj: M_1 should be 0, got {M[0]}"
        assert abs(M[1]) < 1e-14, f"quad-4 edge-adj: M_2 should be 0, got {M[1]}"
        expected_M3 = 0.5 * (1.0 + 3.0 * xi_val)
        expected_M4 = 0.5 * (1.0 - 3.0 * xi_val)
        assert abs(M[2] - expected_M3) < 1e-14
        assert abs(M[3] - expected_M4) < 1e-14
        assert abs(sum(M) - 1.0) < 1e-14

    # Check the kept (2x2) bi-orthogonality block:
    # int M_i N_j over the kept indices {3, 4}; node 3 at (+1,+1), node 4 at (-1,+1).
    kept = [2, 3]
    block = np.zeros((2, 2))
    for q, w in zip(pts, wts):
        xi_val, eta_val = q
        M = M_quad4_dual_modified(xi_val, eta_val, "none", "bottom")
        N = N_quad4(xi_val, eta_val)
        for ii, ki in enumerate(kept):
            for jj, kj in enumerate(kept):
                block[ii, jj] += w * M[ki] * N[kj]
    # Expected (kept block): integrating M_3(xi)·1·N_3(xi)·N_eta=(1+eta)/2
    # over [-1,1]^2. The eta integration of (1+eta)/2 gives 1; the xi
    # integration is the line-2 bi-orthogonality which gives identity
    # (with s_j = 1). So the kept block should be the 2x2 identity.
    expected = np.eye(2)
    err = np.max(np.abs(block - expected))
    assert err < 1e-12, (
        f"quad-4 edge-adj biorth on kept block: got\n{block}\nexpected\n{expected}"
    )
    print(f"  PASS  quad-4 modified edge-adjacent (bottom): kept block = I_2, "
          f"err = {err:.2e}")


def test_wohlmuth_quad4_corner_adjacent():
    """Quad-4 corner-adjacent (eq. 5.10).

    Configuration: side_xi='left' AND side_eta='bottom' — node 1 is on
    a face corner, nodes 2 and 4 are on adjacent face-boundary edges,
    only node 3 (diagonally opposite) is interior.
        M_1 = M_2 = M_4 = 0   (all the boundary-touching nodes)
        M_3 = 1               (constant, identically 1)
    """
    pts, wts = gauss_quad_3x3()
    for q in pts:
        xi_val, eta_val = q
        M = M_quad4_dual_modified(xi_val, eta_val, "left", "bottom")
        assert abs(M[0]) < 1e-14
        assert abs(M[1]) < 1e-14
        assert abs(M[2] - 1.0) < 1e-14, (
            f"quad-4 corner-adj: M_3 (diagonal) should be 1, got {M[2]} "
            f"at ({xi_val}, {eta_val})"
        )
        assert abs(M[3]) < 1e-14
        assert abs(sum(M) - 1.0) < 1e-14

    # The 1x1 kept block: int M_3 N_3 dA = int 1 * (1+xi)(1+eta)/4 dxi deta
    # = (1/4) (∫(1+xi) dxi) (∫(1+eta) deta) = (1/4)(2)(2) = 1.
    accum = 0.0
    for q, w in zip(pts, wts):
        xi_val, eta_val = q
        M = M_quad4_dual_modified(xi_val, eta_val, "left", "bottom")
        N = N_quad4(xi_val, eta_val)
        accum += w * M[2] * N[2]
    assert abs(accum - 1.0) < 1e-12, (
        f"quad-4 corner-adj biorth: int M_3 N_3 = {accum}, expected 1"
    )
    print(f"  PASS  quad-4 modified corner-adjacent: M_diagonal = 1 (constant), "
          f"int M N = 1 = |E|/4")


# =============================================================================
# CONFORMING-PAIR LUMPING RECOVERY (sanity check, follows Phase 2 pattern)
# =============================================================================

def test_conforming_pair_recovers_lumping_quad4():
    """For matching quad-4 elements on opposite faces, the face mortar
    matrix should reduce to a signed identity (eq. 3.8 of architecture
    doc).

    We test this by computing int_E M_i N_j on a SINGLE quad-4 element
    and verifying it equals diag(s_j) = diag(1, 1, 1, 1) — the lumped
    mass. Bi-orthogonality already gives diag = identity (after
    division by s_j), and on conforming pairs A^m and D^nm both reduce
    to this same lumping.

    This is the building block of the Phase 3.4 conforming-mesh sanity
    test (which will integrate across two opposite faces).
    """
    pts, wts = gauss_quad_3x3()
    block = np.zeros((4, 4))
    for q, w in zip(pts, wts):
        xi_val, eta_val = q
        M = M_quad4_dual(xi_val, eta_val)
        N = N_quad4(xi_val, eta_val)
        for i in range(4):
            for j in range(4):
                block[i, j] += w * M[i] * N[j]
    expected = np.diag([1.0, 1.0, 1.0, 1.0])
    err = np.max(np.abs(block - expected))
    assert err < 1e-12, f"quad-4 conforming-pair lumping: {block}"
    print(f"  PASS  conforming-pair lumping on single quad-4: "
          f"diag = (1,1,1,1) = s_j, off-diag err = {err:.2e}")


def test_conforming_pair_recovers_lumping_tri3():
    """Same as above for tri-3: int M_i N_j = diag(|T|/3) on a single
    tri-3 element."""
    pts, wts = gauss_tri_3pt()
    block = np.zeros((3, 3))
    for q, w in zip(pts, wts):
        lam = tuple(q)
        M = M_tri3_dual(lam)
        N = N_tri3(lam)
        for i in range(3):
            for j in range(3):
                block[i, j] += w * M[i] * N[j]
    expected = (1.0 / 6.0) * np.eye(3)
    err = np.max(np.abs(block - expected))
    assert err < 1e-12, f"tri-3 conforming-pair lumping: {block}"
    print(f"  PASS  conforming-pair lumping on single tri-3: "
          f"diag = (|T|/3,)*3, off-diag err = {err:.2e}")


# =============================================================================
# PHASE 3.1 PURE-PYTHON TYPE TESTS
# =============================================================================

def test_corner_info_3d_construction_and_gtdofs():
    """CornerInfo3D round-trip: construction, .gtdofs property."""
    c = CornerInfo3D(
        label="blf",
        coord=np.array([0.0, 0.0, 0.0]),
        gtdof_x=10, gtdof_y=11, gtdof_z=12,
    )
    assert c.label == "blf"
    assert c.coord.shape == (3,)
    assert c.gtdof_x == 10 and c.gtdof_y == 11 and c.gtdof_z == 12
    assert c.gtdofs == (10, 11, 12)
    # Top-right-back corner with realistic coords.
    c2 = CornerInfo3D(
        label="trb", coord=np.array([1.0, 1.0, 1.0]),
        gtdof_x=100, gtdof_y=200, gtdof_z=300,
    )
    assert c2.gtdofs == (100, 200, 300)
    print(f"  PASS  CornerInfo3D round-trip + .gtdofs property")


def test_corner_info_3d_label_convention():
    """Verify the 8-corner label convention is internally consistent.

    Labels: first letter b/t -> y_min/y_max,
            second letter l/r -> x_min/x_max,
            third letter f/b -> z_min/z_max.
    """
    expected_labels = {"blf", "brf", "tlf", "trf",
                       "blb", "brb", "tlb", "trb"}
    # Decode: build from decomposed letters and verify all 8 unique.
    decoded = set()
    for y_letter in "bt":
        for x_letter in "lr":
            for z_letter in "fb":
                decoded.add(y_letter + x_letter + z_letter)
    assert decoded == expected_labels, (
        f"label convention mismatch: decoded {decoded} vs {expected_labels}"
    )
    print(f"  PASS  CornerInfo3D label convention: 8 unique labels span all "
          f"corner combinations")


# =============================================================================
# Driver
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Phase 3.2 unit tests — 3D dual basis machinery")
    print(" + Phase 3.1 type tests for CornerInfo3D")
    print("=" * 60)

    print("\n[Lumped-positivity precondition (§4.9.1)]")
    test_lumped_positivity_line2()
    test_lumped_positivity_line3()
    test_lumped_positivity_tri3()
    test_lumped_positivity_tri6_failure()
    test_lumped_positivity_quad4()
    test_lumped_positivity_quad8_failure()
    test_lumped_positivity_quad9()
    test_lumped_positivity_tet4()
    test_lumped_positivity_tet10_failure()

    print("\n[Bi-orthogonality of implemented dual bases]")
    test_biorthogonality_line2()
    test_biorthogonality_tri3()
    test_biorthogonality_quad4()
    test_biorthogonality_tet4()

    print("\n[Partition of unity]")
    test_partition_of_unity_dual_bases()
    test_partition_of_unity_N_bases()

    print("\n[Wohlmuth modifications]")
    test_wohlmuth_line2_modification_extended()
    test_wohlmuth_tri3_no_boundary()
    test_wohlmuth_tri3_one_vertex_dropped()
    test_wohlmuth_tri3_two_vertices_dropped()
    test_wohlmuth_tri3_three_vertices_dropped()
    test_wohlmuth_quad4_edge_adjacent()
    test_wohlmuth_quad4_corner_adjacent()

    print("\n[Conforming-pair lumping recovery]")
    test_conforming_pair_recovers_lumping_quad4()
    test_conforming_pair_recovers_lumping_tri3()

    print("\n[Phase 3.1: pure-Python types]")
    test_corner_info_3d_construction_and_gtdofs()
    test_corner_info_3d_label_convention()

    print("\n" + "=" * 60)
    print(" All Phase 3.2 unit tests passed.")
    print("=" * 60)
