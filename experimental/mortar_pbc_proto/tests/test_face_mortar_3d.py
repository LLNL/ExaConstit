"""Unit tests for the Phase 3.2.B face-mortar assembler.

Pure-Python tests, no MFEM dependency. Construct synthetic face-element
data, run the assembler, verify against analytic expectations.

References
----------
* MORTAR_PBC_ARCHITECTURE.md §3.6 (conforming free-pass case, eq. 3.8).
* MORTAR_PBC_ARCHITECTURE.md §4.9.1 (lumped-positivity criterion).
* MORTAR_PBC_ARCHITECTURE.md §11.6 / §11.8 Phase 3.2.B.
"""
from __future__ import annotations

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_LOCAL_PKG = os.path.join(_PARENT, "mortar_pbc")

# Sanity check: the local mortar_pbc/ must exist where we expect.
if not os.path.isdir(_LOCAL_PKG):
    raise RuntimeError(
        f"Cannot find mortar_pbc package at {_LOCAL_PKG!r}. "
        f"This script expected to live in <mortar_pbc_proto>/tests/."
    )

# Insert the local prototype directory at the front of sys.path so the
# co-located `mortar_pbc/` is preferred over any stale install.
sys.path.insert(0, _PARENT)

# Defensive eviction: if any earlier import (e.g. via a conftest, a .pth
# file from `pip install -e <other-prototype>/`, or a stale entry in
# PYTHONPATH) cached a different mortar_pbc in sys.modules, evict it so
# our import below resolves through the freshly-prepended sys.path[0].
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
        f"  `pip install -e <some-older-prototype>/` of an earlier\n"
        f"  mortar_pbc_proto. Likely fixes:\n\n"
        f"      pip uninstall mortar-pbc          # remove the stale install\n"
        f"      pip show mortar-pbc               # see what's currently installed\n"
        f"      unset PYTHONPATH                  # clear any env override\n\n"
        f"  Once the stale install is gone, this and the other tests will\n"
        f"  consistently use the local prototype directory.\n"
    )

# Use the canonical package-level re-exports (same pattern as
# test_mortar_3d_unit.py). The defensive block above guarantees we're
# pulling them from the local prototype, not a stale install.
from mortar_pbc import (                                              # noqa: E402
    QuadFaceElement, TriFaceElement,
    QuadFaceMortarAssembler, TriFaceMortarAssembler,
    MortarFaceAssembler,
    match_conforming_face_pairs,
    N_tri6, N_tri3, M_tri3_dual,
    M_quad4_dual_modified, gauss_quad_3x3, gauss_tri_3pt,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_quad_y(*, x_lo, x_hi, z_lo, z_hi, y, gtdofs, boundary_tag="none"):
    """Build a y-perpendicular axis-aligned QuadFaceElement.

    Local node ordering, CCW viewed from +y (matches N_quad4):
        node 0: (x_lo, y, z_lo)   xi=-1, eta=-1
        node 1: (x_hi, y, z_lo)   xi=+1, eta=-1
        node 2: (x_hi, y, z_hi)   xi=+1, eta=+1
        node 3: (x_lo, y, z_hi)   xi=-1, eta=+1
    """
    coords = np.asarray([
        [x_lo, y, z_lo],
        [x_hi, y, z_lo],
        [x_hi, y, z_hi],
        [x_lo, y, z_hi],
    ], dtype=np.float64)
    return QuadFaceElement(
        coords=coords, gtdofs=gtdofs,
        parametric_axes=("x", "z"), perpendicular_axis="y",
        boundary_tag=boundary_tag,
    )


# =============================================================================
# Test 1: lumped-positivity guard PASSES for quad-4 / tri-3 assemblers
# =============================================================================

def test_lumped_positivity_guard_passes():
    QuadFaceMortarAssembler()
    TriFaceMortarAssembler()
    print("  PASS  lumped-positivity guard: quad-4 and tri-3 assemblers instantiate")


# =============================================================================
# Test 2: lumped-positivity guard CATCHES a hypothetical broken basis
# =============================================================================

def test_lumped_positivity_guard_catches_broken_basis():
    """Subclass with tri-6 corner shape (s_corner = 0) must raise."""
    class BrokenTri6Assembler(MortarFaceAssembler):
        def _eval_nonmortar_dual(self, q_pt, tag):       return np.zeros(6)
        def _eval_nonmortar_shape(self, q_pt):           return np.zeros(6)
        def _eval_mortar_shape(self, q_pt):          return np.zeros(6)
        def _build_quadrature(self, order):          return gauss_tri_3pt()
        def _nonmortar_jacobian(self, e):                return lambda q: 1.0
        def _n_nodes_per_elem(self):                 return 6
        def _n_basis_for_lumped_check(self):         return 6
        def _shape_for_lumped_check(self):           return N_tri6
        def _ref_quad_for_lumped_check(self):        return gauss_tri_3pt()
        def _lumped_uses_tuple_input(self):          return True
        def _mortar_node_permutation_apply(self, p, q): return q

    raised = False
    try:
        BrokenTri6Assembler()
    except RuntimeError as e:
        raised = True
        assert "lumped-positivity check failed" in str(e)
    assert raised, "BrokenTri6Assembler should have raised"
    print("  PASS  lumped-positivity guard catches tri-6-like broken basis")


# =============================================================================
# Test 3: single quad-4 conforming pair — D = A_m = (face_area / 4) * I_4
# =============================================================================

def test_face_mortar_quad_single_elem_conforming():
    """Bi-orthogonality => D and A_m both diagonal, equal to (Δx·Δz)/4 each."""
    Lx, Lz = 2.0, 3.0   # non-unit dims to catch axis confusion
    nonmortar = _make_quad_y(x_lo=0, x_hi=Lx, z_lo=0, z_hi=Lz, y=0.0,
                         gtdofs=(0, 1, 2, 3))
    mortar = _make_quad_y(x_lo=0, x_hi=Lx, z_lo=0, z_hi=Lz, y=1.0,
                          gtdofs=(10, 11, 12, 13))
    asm = QuadFaceMortarAssembler()
    block = asm.assemble_pair_conforming(
        nonmortar_elems=[nonmortar], mortar_elems=[mortar],
        pair_matches=[(0, 0, (0, 1, 2, 3))],
        nonmortar_face_name="bottom", mortar_face_name="top",
    )
    expected = (Lx * Lz) / 4.0   # = 1.5
    assert np.allclose(block.D, expected * np.ones(4), atol=1e-13), (
        f"D = {block.D}, expected {expected}")
    assert np.allclose(block.A_m, expected * np.eye(4), atol=1e-13), (
        f"A_m = {block.A_m}")
    assert np.array_equal(block.nonmortar_gtdofs, [0, 1, 2, 3])
    assert np.array_equal(block.mortar_gtdofs, [10, 11, 12, 13])
    print(f"  PASS  single quad-4 conforming pair: D = {expected:.4f} * 1_4, "
          f"A_m = D * I_4 (face area = {Lx*Lz})")


# =============================================================================
# Test 4: 2x2 grid of quads conforming pair
# =============================================================================

def test_face_mortar_quad_2x2_grid_conforming():
    """2x2 sub-element grid: D pattern reflects per-node sub-element count."""
    L = 2.0
    n = 2
    xs = np.linspace(0.0, L, n + 1)
    zs = np.linspace(0.0, L, n + 1)
    nonmortar_elems = []
    mortar_elems = []

    def nonmortar_tdof(i, j):  return i * (n + 1) + j
    def mortar_tdof(i, j): return 100 + i * (n + 1) + j

    for i in range(n):
        for j in range(n):
            x_lo, x_hi = xs[i], xs[i + 1]
            z_lo, z_hi = zs[j], zs[j + 1]
            nonmortar_elems.append(_make_quad_y(
                x_lo=x_lo, x_hi=x_hi, z_lo=z_lo, z_hi=z_hi, y=0.0,
                gtdofs=(nonmortar_tdof(i, j), nonmortar_tdof(i + 1, j),
                        nonmortar_tdof(i + 1, j + 1), nonmortar_tdof(i, j + 1)),
            ))
            mortar_elems.append(_make_quad_y(
                x_lo=x_lo, x_hi=x_hi, z_lo=z_lo, z_hi=z_hi, y=1.0,
                gtdofs=(mortar_tdof(i, j), mortar_tdof(i + 1, j),
                        mortar_tdof(i + 1, j + 1), mortar_tdof(i, j + 1)),
            ))

    asm = QuadFaceMortarAssembler()
    pair_matches = match_conforming_face_pairs(
        nonmortar_elems, mortar_elems, perpendicular_axis="y", period=1.0,
    )
    assert len(pair_matches) == 4
    for s_idx, m_idx, perm in pair_matches:
        assert perm == (0, 1, 2, 3)

    block = asm.assemble_pair_conforming(
        nonmortar_elems=nonmortar_elems, mortar_elems=mortar_elems,
        pair_matches=pair_matches,
    )
    # 9 unique nodes; sorted gtdofs = (0..8) in lex (i, j) order.
    # Sub-element count per node (3x3 grid): corners 1, edge-mids 2, center 4.
    n_per_node = np.asarray([
        1, 2, 1,    # i=0 row
        2, 4, 2,    # i=1 row
        1, 2, 1,    # i=2 row
    ])
    sub_area = 1.0
    expected_D = (sub_area / 4.0) * n_per_node
    assert np.allclose(block.D, expected_D, atol=1e-13), (
        f"D = {block.D}, expected {expected_D}")
    diff = np.linalg.norm(block.A_m - np.diag(block.D))
    assert diff < 1e-12, f"||A_m - diag(D)||_F = {diff}"
    print(f"  PASS  2x2 quad-4 grid: D pattern = {n_per_node.tolist()} * 0.25, "
          f"A_m = diag(D), err = {diff:.2e}")


# =============================================================================
# Test 5: single tri-3 conforming pair — D = A_m = (|T|/3) * I_3
# =============================================================================

def test_face_mortar_tri_single_elem_conforming():
    """Bi-orthogonality on tri-3 => A_m = D = (|T|/3) * I_3."""
    coords_s = np.asarray([[0., 0., 0.], [2., 0., 0.], [0., 0., 3.]])
    coords_m = coords_s + np.asarray([0., 1., 0.])
    nonmortar = TriFaceElement(coords=coords_s, gtdofs=(0, 1, 2),
                           parametric_axes=("x", "z"), perpendicular_axis="y")
    mortar = TriFaceElement(coords=coords_m, gtdofs=(10, 11, 12),
                            parametric_axes=("x", "z"), perpendicular_axis="y")
    asm = TriFaceMortarAssembler()
    block = asm.assemble_pair_conforming(
        nonmortar_elems=[nonmortar], mortar_elems=[mortar],
        pair_matches=[(0, 0, (0, 1, 2))],
    )
    # |T| = 0.5 * |2 * 3| = 3.0; |T|/3 = 1.0.
    expected = 1.0
    assert np.allclose(block.D, expected * np.ones(3), atol=1e-13), (
        f"D = {block.D}")
    assert np.allclose(block.A_m, expected * np.eye(3), atol=1e-13), (
        f"A_m = {block.A_m}")
    print(f"  PASS  single tri-3 conforming pair: D = {expected:.4f} * 1_3, "
          f"A_m = D * I_3 (|T| = 3.0)")


# =============================================================================
# Test 6: sentinel-row drop on quad-4 (no Wohlmuth modification)
# =============================================================================

def test_face_mortar_quad_sentinel_drop():
    """Nonmortar with gtdofs (0, -1, 1, 2): row at local-node 1 is absent."""
    Lx, Lz = 2.0, 2.0
    nonmortar = _make_quad_y(x_lo=0, x_hi=Lx, z_lo=0, z_hi=Lz, y=0.0,
                         gtdofs=(0, -1, 1, 2))
    mortar = _make_quad_y(x_lo=0, x_hi=Lx, z_lo=0, z_hi=Lz, y=1.0,
                          gtdofs=(10, 11, 12, 13))
    asm = QuadFaceMortarAssembler()
    block = asm.assemble_pair_conforming(
        nonmortar_elems=[nonmortar], mortar_elems=[mortar],
        pair_matches=[(0, 0, (0, 1, 2, 3))],
    )
    assert block.D.shape == (3,)
    assert block.A_m.shape == (3, 4)
    assert np.array_equal(block.nonmortar_gtdofs, [0, 1, 2])
    expected_Am = (Lx * Lz / 4.0) * np.asarray([
        [1.0, 0.0, 0.0, 0.0],   # nonmortar-local 0 -> mortar-local 0
        [0.0, 0.0, 1.0, 0.0],   # nonmortar-local 2 -> mortar-local 2
        [0.0, 0.0, 0.0, 1.0],   # nonmortar-local 3 -> mortar-local 3
    ])
    assert np.allclose(block.A_m, expected_Am, atol=1e-13), (
        f"A_m = {block.A_m}\nexpected = {expected_Am}")
    print(f"  PASS  sentinel drop on quad-4: kept (3, 4) block as expected")


# =============================================================================
# Test 7: Wohlmuth corner-LL modification on quad-4
# =============================================================================

def test_face_mortar_quad_with_corner_modification():
    """Corner-adjacent nonmortar with corner-LL Wohlmuth dual.

    Verify:
      (a) corner row dropped via sentinel mechanism;
      (b) D rows unchanged from unmodified case (D uses standard N, not M);
      (c) A_m row sums DIFFER from unmodified case (modification active);
      (d) modified dual still partition-of-unity at every Gauss point.
    """
    Lx, Lz = 2.0, 2.0
    nonmortar_mod = _make_quad_y(x_lo=0, x_hi=Lx, z_lo=0, z_hi=Lz, y=0.0,
                             gtdofs=(-1, 0, 1, 2),
                             boundary_tag="corner-LL")
    nonmortar_unmod = _make_quad_y(x_lo=0, x_hi=Lx, z_lo=0, z_hi=Lz, y=0.0,
                               gtdofs=(-1, 0, 1, 2),
                               boundary_tag="none")
    mortar = _make_quad_y(x_lo=0, x_hi=Lx, z_lo=0, z_hi=Lz, y=1.0,
                          gtdofs=(10, 11, 12, 13))
    asm = QuadFaceMortarAssembler()
    blk_mod = asm.assemble_pair_conforming(
        [nonmortar_mod], [mortar], [(0, 0, (0, 1, 2, 3))])
    blk_unmod = asm.assemble_pair_conforming(
        [nonmortar_unmod], [mortar], [(0, 0, (0, 1, 2, 3))])

    # (a) corner row dropped
    assert blk_mod.D.shape == (3,) and blk_mod.A_m.shape == (3, 4)
    assert np.array_equal(blk_mod.nonmortar_gtdofs, [0, 1, 2])

    # (b) D should be the same in both modified and unmodified
    assert np.allclose(blk_mod.D, blk_unmod.D, atol=1e-13), (
        f"D mod = {blk_mod.D}, D unmod = {blk_unmod.D}")

    # (c) row-sum of A_m differs between mod and unmod
    rs_mod = blk_mod.A_m.sum(axis=1)
    rs_unmod = blk_unmod.A_m.sum(axis=1)
    diff = np.max(np.abs(rs_mod - rs_unmod))
    assert diff > 1e-3, (
        f"Wohlmuth modification did not change A_m row sums: diff = {diff}")

    # (d) PoU of the modified dual at every Gauss point
    pts, wts = gauss_quad_3x3()
    for q in pts:
        M = M_quad4_dual_modified(float(q[0]), float(q[1]),
                                   side_xi="left", side_eta="bottom")
        assert abs(sum(M) - 1.0) < 1e-13, f"PoU broken at {q}: sum = {sum(M)}"

    print(f"  PASS  Wohlmuth corner-LL on quad-4: corner row dropped, "
          f"row-sum diff vs unmod = {diff:.4f}, PoU preserved")


# =============================================================================
# Test 8: tri-3 with one vertex dropped (edge-adjacent Wohlmuth)
# =============================================================================

def test_face_mortar_tri_with_one_vertex_dropped():
    """Tri-3 nonmortar with vertex 0 = sentinel + Wohlmuth boundary_tag='v0'.

    With vertex 0 dropped, M_2_modified = 0.5 + 2 lam_2 - 2 lam_3 and
    M_3_modified = 0.5 - 2 lam_2 + 2 lam_3 per eq. 5.5. Bi-orthogonality
    targets verified in the architecture doc:
      ∫ M_2_mod * lam_1 dA = "leak" (non-zero, harmless after corner-col zero)
      ∫ M_2_mod * lam_2 dA = |T|/3
      ∫ M_2_mod * lam_3 dA = 0
    Symmetric for M_3_mod.

    Test: kept nonmortar rows = (1, 2); A_m kept block on mortar cols (1, 2)
    matches diag(|T|/3); leak col 0 is non-zero but unconstrained.
    """
    coords_s = np.asarray([[0., 0., 0.], [2., 0., 0.], [0., 0., 3.]])
    coords_m = coords_s + np.asarray([0., 1., 0.])
    nonmortar = TriFaceElement(
        coords=coords_s, gtdofs=(-1, 0, 1),
        parametric_axes=("x", "z"), perpendicular_axis="y",
        boundary_tag="v0",
    )
    mortar = TriFaceElement(
        coords=coords_m, gtdofs=(10, 11, 12),
        parametric_axes=("x", "z"), perpendicular_axis="y",
    )
    asm = TriFaceMortarAssembler()
    block = asm.assemble_pair_conforming(
        [nonmortar], [mortar], [(0, 0, (0, 1, 2))])

    assert block.D.shape == (2,)
    assert block.A_m.shape == (2, 3)
    assert np.array_equal(block.nonmortar_gtdofs, [0, 1])

    # Kept block on cols (1, 2): expected diag(|T|/3) = diag(1.0)
    kept_block = block.A_m[:, 1:]   # cols 1 and 2
    expected_kept = np.eye(2)        # |T|/3 = 1
    assert np.allclose(kept_block, expected_kept, atol=1e-12), (
        f"A_m kept block (cols 1-2) = {kept_block}, expected I_2")
    # Leak col (col 0) should be NON-zero (per the doc's eq. 5.5
    # verification: ∫ M_2 lam_1 dA = leak).
    leak = block.A_m[:, 0]
    assert np.max(np.abs(leak)) > 1e-3, (
        f"Wohlmuth tri-3 should leak into corner col, leak = {leak}")
    print(f"  PASS  tri-3 v0 Wohlmuth: kept (2, 3); cols (1,2) = I_2, "
          f"col 0 leak = ({leak[0]:.4f}, {leak[1]:.4f})")


# =============================================================================
# Test 9: match_conforming_face_pairs - identity perm on aligned mesh
# =============================================================================

def test_match_conforming_face_pairs_axis_aligned():
    """A 3x3 face-element grid pairs 1:1 with identity perm."""
    L = 3.0
    n = 3
    xs = np.linspace(0.0, L, n + 1)
    zs = np.linspace(0.0, L, n + 1)
    nonmortar_elems = []
    mortar_elems = []
    for i in range(n):
        for j in range(n):
            nonmortar_elems.append(_make_quad_y(
                x_lo=xs[i], x_hi=xs[i+1], z_lo=zs[j], z_hi=zs[j+1], y=0.0,
                gtdofs=(0, 1, 2, 3),  # not testing gtdof here
            ))
            mortar_elems.append(_make_quad_y(
                x_lo=xs[i], x_hi=xs[i+1], z_lo=zs[j], z_hi=zs[j+1], y=1.0,
                gtdofs=(10, 11, 12, 13),
            ))
    pair_matches = match_conforming_face_pairs(
        nonmortar_elems, mortar_elems, perpendicular_axis="y", period=1.0)
    assert len(pair_matches) == 9
    # Each nonmortar should pair with its identical-centroid mortar
    for s_idx, m_idx, perm in pair_matches:
        # In our build order, nonmortar_idx == mortar_idx
        assert s_idx == m_idx, f"s={s_idx}, m={m_idx}"
        assert perm == (0, 1, 2, 3), f"perm = {perm}"
    print(f"  PASS  match_conforming_face_pairs: 9-element grid, identity perm")


# =============================================================================
# Test 10: match_conforming_face_pairs - permuted mortar order recovered
# =============================================================================

def test_match_conforming_face_pairs_shuffled_mortar_order():
    """Shuffling mortar_elems list is recovered by the matcher."""
    L = 2.0
    n = 2
    xs = np.linspace(0.0, L, n + 1)
    zs = np.linspace(0.0, L, n + 1)
    nonmortar_elems = []
    mortar_elems = []
    for i in range(n):
        for j in range(n):
            nonmortar_elems.append(_make_quad_y(
                x_lo=xs[i], x_hi=xs[i+1], z_lo=zs[j], z_hi=zs[j+1], y=0.0,
                gtdofs=(0, 1, 2, 3)))
            mortar_elems.append(_make_quad_y(
                x_lo=xs[i], x_hi=xs[i+1], z_lo=zs[j], z_hi=zs[j+1], y=1.0,
                gtdofs=(10, 11, 12, 13)))
    # Reverse mortar order
    mortar_shuffled = list(reversed(mortar_elems))
    pair_matches = match_conforming_face_pairs(
        nonmortar_elems, mortar_shuffled, perpendicular_axis="y", period=1.0)
    assert len(pair_matches) == 4
    # Nonmortar i should pair with mortar_shuffled index that has same centroid.
    for s_idx, m_idx, perm in pair_matches:
        s_centroid = nonmortar_elems[s_idx].coords.mean(axis=0)[[0, 2]]
        m_centroid = mortar_shuffled[m_idx].coords.mean(axis=0)[[0, 2]]
        assert np.allclose(s_centroid, m_centroid, atol=1e-12), (
            f"Mismatch: nonmortar {s_idx} {s_centroid} vs mortar {m_idx} {m_centroid}")
        assert perm == (0, 1, 2, 3)
    print(f"  PASS  match_conforming_face_pairs: shuffled-mortar order recovered")


# =============================================================================
# Test 11: match_conforming_face_pairs - non-conforming case raises
# =============================================================================

def test_match_conforming_face_pairs_nonconforming_raises():
    """A 2x2 nonmortar grid against a 3x3 mortar grid is non-conforming."""
    L = 2.0
    nonmortar_elems = []
    for i in range(2):
        for j in range(2):
            nonmortar_elems.append(_make_quad_y(
                x_lo=L*i/2, x_hi=L*(i+1)/2, z_lo=L*j/2, z_hi=L*(j+1)/2, y=0.0,
                gtdofs=(0, 1, 2, 3)))
    mortar_elems = []
    for i in range(3):
        for j in range(3):
            mortar_elems.append(_make_quad_y(
                x_lo=L*i/3, x_hi=L*(i+1)/3, z_lo=L*j/3, z_hi=L*(j+1)/3, y=1.0,
                gtdofs=(10, 11, 12, 13)))
    raised = False
    try:
        match_conforming_face_pairs(
            nonmortar_elems, mortar_elems, perpendicular_axis="y", period=1.0)
    except RuntimeError:
        raised = True
    assert raised, "Non-conforming grids should fail to match"
    print(f"  PASS  match_conforming_face_pairs: non-conforming case raises")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Phase 3.2.B face-mortar assembler unit tests")
    print("=" * 60)

    print("\n[Construction guards]")
    test_lumped_positivity_guard_passes()
    test_lumped_positivity_guard_catches_broken_basis()

    print("\n[Conforming-pair lumping recovery (eq. 3.8)]")
    test_face_mortar_quad_single_elem_conforming()
    test_face_mortar_quad_2x2_grid_conforming()
    test_face_mortar_tri_single_elem_conforming()

    print("\n[Sentinel-row drop]")
    test_face_mortar_quad_sentinel_drop()

    print("\n[Wohlmuth modifications via boundary_tag]")
    test_face_mortar_quad_with_corner_modification()
    test_face_mortar_tri_with_one_vertex_dropped()

    print("\n[Conforming-pair matching helper]")
    test_match_conforming_face_pairs_axis_aligned()
    test_match_conforming_face_pairs_shuffled_mortar_order()
    test_match_conforming_face_pairs_nonconforming_raises()

    print()
    print("=" * 60)
    print(" All Phase 3.2.B tests passed.")
    print("=" * 60)
