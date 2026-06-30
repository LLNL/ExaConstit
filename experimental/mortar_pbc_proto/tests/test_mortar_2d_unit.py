"""Unit tests for the mortar machinery that don't require pyMFEM.

These verify the building blocks (dual basis bi-orthogonality, segment
intersection, mortar matrix consistency on a *conforming* edge pair where
A^m and D^nm should both reduce to the lumped-mass matrix) before any
finite element coupling is involved.

Run with:
    python tests/test_mortar_2d_unit.py
"""
import sys, os

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

from mortar_pbc.mortar_2d import (                                    # noqa: E402
    N_line2, M_line2_dual, _GL3_PTS, _GL3_WTS,
    MortarAssembler2D,
)
from mortar_pbc.types_2d import EdgeNodes2D                           # noqa: E402


# ---------------------------------------------------------------------------
def test_dual_basis_biorthogonality():
    """∫_-1^1 M_i(ξ) N_j(ξ) dξ = δ_ij."""
    pts, wts = _GL3_PTS, _GL3_WTS
    M_NN = np.zeros((2, 2))
    for x, w in zip(pts, wts):
        M = M_line2_dual(x)
        N = N_line2(x)
        for i in range(2):
            for j in range(2):
                M_NN[i, j] += w * M[i] * N[j]
    expected = np.eye(2)
    err = np.max(np.abs(M_NN - expected))
    assert err < 1e-12, f"dual bi-orthogonality failed: M*N = {M_NN}"
    print(f"  PASS  dual basis bi-orthogonality (max err {err:.2e})")


def test_dual_basis_partition_of_unity():
    """∫_-1^1 N_i(ξ) dξ = 1 for line-2 shape functions."""
    pts, wts = _GL3_PTS, _GL3_WTS
    integrals = np.zeros(2)
    for x, w in zip(pts, wts):
        N = N_line2(x)
        for i in range(2):
            integrals[i] += w * N[i]
    err = np.max(np.abs(integrals - 1.0))
    assert err < 1e-12, f"N integrals = {integrals}"
    print(f"  PASS  N partition of unity (max err {err:.2e})")


# ---------------------------------------------------------------------------
def test_wohlmuth_crosspoint_modification():
    """Verify Lopes 2021 Appendix C eq. (C.2): the Wohlmuth corner
    modification of the line-2 dual basis preserves partition-of-unity
    and breaks bi-orthogonality in the predicted way.

    Standard dual basis (Eq. C.1): M_1=(1-3ξ)/2, M_2=(1+3ξ)/2
    Modified at corner (Eq. C.2):  M_1=0, M_2=1   (left node = corner)
                                or M_1=1, M_2=0   (right node = corner)

    Three properties checked:
      (a) Partition of unity:  M_1 + M_2 ≡ 1 on [-1, 1].  Both standard
          and modified bases satisfy this trivially -- the modified
          basis MORE strongly (constant 1 vs sum-of-two-linear-pieces).
      (b) The corner-side basis function is identically zero, so
          ∫ M_corner * (anything) = 0.  This is what implements
          "corner LM dropped from the constraint."
      (c) The neighbor-side basis function INTEGRATES against the
          standard FE shape function correctly.  For side='left'
          (node 1 = corner), M_2 ≡ 1 and ∫ M_2 * N_1 dξ = ∫ N_1 dξ = 1
          (the boundary mass at the corner under linear interpolation).
          ∫ M_2 * N_2 dξ = ∫ N_2 dξ = 1 (by symmetry of N_1 + N_2 = 1).
          So the row-sum is 2 (the full segment length on [-1, 1]).
    """
    from mortar_pbc.mortar_2d import M_line2_dual_modified
    pts, wts = _GL3_PTS, _GL3_WTS

    # ----- Property (a): partition of unity for both modifications -----
    for side in ("left", "right"):
        M_sum_max_dev = 0.0
        for x in pts:
            M = M_line2_dual_modified(x, side)
            M_sum_max_dev = max(M_sum_max_dev, abs(M[0] + M[1] - 1.0))
        assert M_sum_max_dev < 1e-15, (
            f"side={side}: M_1 + M_2 deviates from 1 by {M_sum_max_dev:.2e}"
        )

    # ----- Property (b): corner-side function is identically zero -----
    for x in pts:
        M_left = M_line2_dual_modified(x, "left")    # left node is corner
        assert M_left[0] == 0.0, f"side='left': M_1({x}) = {M_left[0]} != 0"
        M_right = M_line2_dual_modified(x, "right")  # right node is corner
        assert M_right[1] == 0.0, f"side='right': M_2({x}) = {M_right[1]} != 0"

    # ----- Property (c): neighbor-side function integrates as constant 1 -----
    # side='left' -> M_2 = 1 on [-1, 1]
    #   ∫ M_2 N_1 dξ = ∫ (1-ξ)/2 dξ from -1 to 1 = 1
    #   ∫ M_2 N_2 dξ = ∫ (1+ξ)/2 dξ from -1 to 1 = 1
    integrals_left = np.zeros(2)
    for x, w in zip(pts, wts):
        M = M_line2_dual_modified(x, "left")
        N = N_line2(x)
        for j in range(2):
            integrals_left[1] += w * M[1] * N[j] / 2.0   # avg over both Ns
        # Also gather individual integrals for the assertion:
    # Recompute directly:
    int_M2_N1 = sum(w * M_line2_dual_modified(x, "left")[1] * N_line2(x)[0]
                    for x, w in zip(pts, wts))
    int_M2_N2 = sum(w * M_line2_dual_modified(x, "left")[1] * N_line2(x)[1]
                    for x, w in zip(pts, wts))
    err_M2_N1 = abs(int_M2_N1 - 1.0)
    err_M2_N2 = abs(int_M2_N2 - 1.0)
    assert err_M2_N1 < 1e-12, f"∫ M_2 N_1 (side=left) = {int_M2_N1}, expected 1"
    assert err_M2_N2 < 1e-12, f"∫ M_2 N_2 (side=left) = {int_M2_N2}, expected 1"

    # Symmetric check for side='right' -> M_1 = 1 on [-1, 1].
    int_M1_N1 = sum(w * M_line2_dual_modified(x, "right")[0] * N_line2(x)[0]
                    for x, w in zip(pts, wts))
    int_M1_N2 = sum(w * M_line2_dual_modified(x, "right")[0] * N_line2(x)[1]
                    for x, w in zip(pts, wts))
    assert abs(int_M1_N1 - 1.0) < 1e-12
    assert abs(int_M1_N2 - 1.0) < 1e-12

    print(f"  PASS  Wohlmuth crosspoint mod (Lopes 2021 Eq. C.2)")
    print(f"        partition-of-unity preserved, corner func = 0,")
    print(f"        neighbor-func integrals = 1 (constant 1 reproduces "
          f"unit boundary mass)")


def test_conforming_pair_recovers_lumping():
    """For two opposite edges with IDENTICAL node spacing, the mortar
    coupling matrix A^m equals the lumped boundary mass D^nm (so the
    dependency matrix α = D^-1 A = I, recovering standard PBC).

    Build a + edge along y=0 and a - edge along y=1 with the same x-spacing,
    and verify A^m == diag(D^nm).
    """
    L = 1.0
    n_nodes = 5  # 4 elements + 4 corner sentinels in our scheme
    xs = np.linspace(0.0, L, n_nodes)

    def make_edge(name: str, y_const: float, is_plus: bool) -> EdgeNodes2D:
        # corners excluded from coords/elements per our scheme:
        # interior = nodes 1..n-2; nodes 0 and n-1 are corners (sentinels)
        interior_xs = xs[1:-1]
        N = len(interior_xs)
        coords = np.column_stack([interior_xs, np.full(N, y_const)])
        gtx = np.arange(N, dtype=np.int64)        # mock TDOFs
        gty = np.arange(N, dtype=np.int64) + 100
        # Elements: corner -> 0, 0->1, 1->2, ..., N-1 -> corner
        elements = [(-1, 0)]
        for k in range(N - 1):
            elements.append((k, k + 1))
        elements.append((N - 1, -2))
        return EdgeNodes2D(
            name=name,
            is_nonmortar=is_plus,
            coords=coords,
            gtdofs_x=gtx,
            gtdofs_y=gty,
            elements=elements,
            parametric_axis="x",
            edge_min=0.0,
            edge_max=L,
        )

    bottom = make_edge("bottom", 0.0, True)
    top    = make_edge("top",    L,   False)

    # Mock classifier
    class MockCl:
        edges = {"bottom": bottom, "top": top}

    asm = MortarAssembler2D(MockCl())
    block = asm._assemble_pair(bottom, top)

    # For a CONFORMING pair, A^m should be diag(D^nm) for interior nodes.
    diff = np.linalg.norm(block.A_m - np.diag(block.D_nm))
    print(f"  D^nm = {block.D_nm}")
    print(f"  diag(A^m) = {np.diag(block.A_m)}")
    print(f"  ||A^m - diag(D^nm)||_F = {diff:.3e}")
    # On a conforming aligned pair the off-diagonals must vanish and
    # diagonals match.
    assert diff < 1e-12, "A^m should equal diag(D^nm) on conforming aligned pair"
    print(f"  PASS  conforming pair recovers lumped mass")


def test_nonconforming_pair_consistency():
    """Linear-field reproduction on a non-conforming pair.

    For + and - edges with NO corner segments (corners excluded from the
    element list), the standard dual basis is bi-orthogonal to N^+ and
    the standard linear shape functions on the - side reproduce linear
    fields exactly.  Therefore for a linear field u(Y) = a + bY sampled
    at all + and - nodes:

        D^nm * u^+  -  A^m * u^-  =  0   (exactly, to round-off).

    Note on corner-modified segments: the Wohlmuth corner modifications
    (M_1=0, M_2=1) intentionally break bi-orthogonality on segments
    touching Dirichlet corners.  That's the trade-off the paper accepts
    to avoid over-constraint at corner nodes.  Linear-field reproduction
    on corner segments therefore CANNOT hold by design; it's the FE
    patch test (homogeneous RVE under macroscopic F, recovering
    u_tilde = 0 -- Section 5.1.1) that validates the corner-modified
    machinery end-to-end, not a unit-level mortar-matrix test.

    This unit test isolates the CORE assembly machinery (segmentation,
    parametric mapping, GL3 quadrature, dual-basis bi-orthogonality)
    by removing the corner-modification path entirely.
    """
    # Use only the interior of [0, L] so corners aren't in any element.
    Y0, Y1 = 0.1, 0.9

    def make_edge(name, y_const, xs, is_plus):
        N = len(xs)
        coords = np.column_stack([xs, np.full(N, y_const)])
        gtx = np.arange(N, dtype=np.int64)
        gty = np.arange(N, dtype=np.int64) + 100
        # Elements connect adjacent interior nodes ONLY -- no corner sentinels.
        elements = [(k, k + 1) for k in range(N - 1)]
        return EdgeNodes2D(
            name=name, is_nonmortar=is_plus,
            coords=coords, gtdofs_x=gtx, gtdofs_y=gty,
            elements=elements, parametric_axis="x",
            edge_min=Y0, edge_max=Y1,
        )

    plus_xs = np.array([0.10, 0.27, 0.41, 0.58, 0.73, 0.90])  # 6 nodes, 5 elems
    minus_xs = np.array([0.10, 0.35, 0.62, 0.90])              # 4 nodes, 3 elems
    bot = make_edge("bottom", 0.0, plus_xs,  is_plus=True)
    top = make_edge("top",    1.0, minus_xs, is_plus=False)

    class MockCl:
        edges = {"bottom": bot, "top": top}

    asm = MortarAssembler2D(MockCl())
    block = asm._assemble_pair(bot, top)

    print(f"  + nodes ({len(plus_xs)}): {plus_xs}")
    print(f"  - nodes ({len(minus_xs)}): {minus_xs}")
    print(f"  D^nm shape = {block.D_nm.shape}, A^m shape = {block.A_m.shape}")

    # Sanity: D^nm should be ∫ N^+_k dA = (h_left + h_right)/2 for interior k.
    # For node k with neighbors at x_{k-1}, x_{k+1}: D^nm[k] = (x_{k+1}-x_{k-1})/2.
    expected_Dnm = np.array([
        (plus_xs[1] - plus_xs[0]) / 2.0,                              # endpoint
        (plus_xs[2] - plus_xs[0]) / 2.0,
        (plus_xs[3] - plus_xs[1]) / 2.0,
        (plus_xs[4] - plus_xs[2]) / 2.0,
        (plus_xs[5] - plus_xs[3]) / 2.0,
        (plus_xs[5] - plus_xs[4]) / 2.0,                              # endpoint
    ])
    diff_D = np.linalg.norm(block.D_nm - expected_Dnm, ord=np.inf)
    assert diff_D < 1e-14, f"D^nm wrong: got {block.D_nm}, expected {expected_Dnm}"
    print(f"  D^nm matches analytic formula (||err||_inf = {diff_D:.2e})")

    # Linear-field patch test.
    a, b = -0.5, 2.0
    u_plus  = a + b * plus_xs
    u_minus = a + b * minus_xs
    residual = block.D_nm * u_plus - block.A_m @ u_minus
    err = np.linalg.norm(residual, ord=np.inf)
    print(f"  ||D^nm u^+ - A^m u^-||_inf = {err:.3e}  (linear field a+bY)")
    assert err < 1e-12, \
        f"Linear-field patch test FAILED: residual = {residual}"

    # Constant-field check for good measure (a=c, b=0 => row sums of A^m
    # should equal D^nm exactly).
    row_sum = block.A_m.sum(axis=1)
    diff_const = np.linalg.norm(row_sum - block.D_nm, ord=np.inf)
    assert diff_const < 1e-13, \
        f"Constant field FAILED: row_sum(A^m) = {row_sum}, D^nm = {block.D_nm}"
    print(f"  Row sums of A^m match D^nm (||err||_inf = {diff_const:.2e})")
    print(f"  PASS  non-conforming pair reproduces constant + linear fields")


def test_constraint_assembler_abc():
    """ConstraintAssembler ABC + stack_constraints helper.

    Builds a tiny mortar block by hand, wraps it in a
    ``MortarPbcConstraintAssembler``, and verifies that:
        * ``assemble()`` produces a CSR matrix with the correct shape
          and the same nonzeros that ``ConstraintBuilder2D.build()``
          would have produced directly,
        * ``stack_constraints([assembler])`` round-trips through to
          the same C and a zero RHS,
        * Stacking the same assembler twice gives a 2x-tall block --
          a sanity check that the vstack code path is correct (this
          mirrors what the future-UT case will look like: one mortar
          assembler + one UT assembler stacked).
    """
    from mortar_pbc.constraint_builder import ConstraintBuilder2D
    from mortar_pbc.constraint_assembler import (
        MortarPbcConstraintAssembler, stack_constraints,
    )
    from mortar_pbc.mortar_2d import MortarBlock2D

    # Hand-rolled tiny scenario: 2 + nodes, 3 - nodes, vdim=2.
    # gtdofs are arbitrary indices in some pretend global space.
    plus_edge = EdgeNodes2D(
        name="bottom", is_nonmortar=True,
        coords=np.array([[0.3, 0.0], [0.7, 0.0]]),
        gtdofs_x=np.array([10, 12], dtype=np.int64),
        gtdofs_y=np.array([11, 13], dtype=np.int64),
        elements=[(0, 1)],
        parametric_axis="x", edge_min=0.0, edge_max=1.0,
    )
    minus_edge = EdgeNodes2D(
        name="top", is_nonmortar=False,
        coords=np.array([[0.2, 1.0], [0.5, 1.0], [0.8, 1.0]]),
        gtdofs_x=np.array([20, 22, 24], dtype=np.int64),
        gtdofs_y=np.array([21, 23, 25], dtype=np.int64),
        elements=[(0, 1), (1, 2)],
        parametric_axis="x", edge_min=0.0, edge_max=1.0,
    )

    # Synthetic D^nm and A^m -- numerical content doesn't matter, only
    # that the builder routes them to the right (row, col) entries.
    block = MortarBlock2D(
        A_m=np.array([[0.1, 0.2, 0.0], [0.0, 0.3, 0.4]]),
        D_nm=np.array([0.5, 0.6]),
        plus_edge_name="bottom", minus_edge_name="top",
    )
    blocks = {("bottom", "top"): block}

    class MockClassifier:
        edges = {"bottom": plus_edge, "top": minus_edge,
                 "left": plus_edge, "right": minus_edge}
        n_global_tdofs = 30  # any number bigger than the largest gtdof

    cl = MockClassifier()

    # Reference path: direct ConstraintBuilder2D.
    # Override PAIRS so the assembler doesn't try to walk left/right too.
    from mortar_pbc.mortar_2d import MortarAssembler2D as MA
    direct_blocks = {("bottom", "top"): block}
    ref_C = ConstraintBuilder2D(cl, direct_blocks).build()

    # New path: via the ABC.
    asm = MortarPbcConstraintAssembler(cl, direct_blocks)
    assert asm.name() == "mortar_pbc"
    assert asm.n_rows() == ref_C.shape[0]
    abc_C = asm.assemble()
    assert abc_C.shape == ref_C.shape
    diff = (abc_C - ref_C).toarray()
    assert np.allclose(diff, 0.0), f"ABC produced different C: max abs diff = {np.abs(diff).max()}"
    print(f"  Single-assembler path: shape={abc_C.shape}, nnz={abc_C.nnz}")

    # Caching: second call should return the same object.
    abc_C2 = asm.assemble()
    assert abc_C2 is abc_C, "assemble() should cache"
    print(f"  assemble() correctly caches across calls")

    # stack_constraints with one assembler.
    C_stacked, g_stacked = stack_constraints([asm])
    assert C_stacked.shape == abc_C.shape
    assert np.allclose((C_stacked - abc_C).toarray(), 0.0)
    assert g_stacked.shape == (abc_C.shape[0],)
    assert np.allclose(g_stacked, 0.0)
    print(f"  stack_constraints([asm]) round-trip OK")

    # stack_constraints with two assemblers (mock the future UT case).
    asm2 = MortarPbcConstraintAssembler(cl, direct_blocks)  # second instance
    C_two, g_two = stack_constraints([asm, asm2])
    assert C_two.shape == (2 * abc_C.shape[0], abc_C.shape[1])
    # Both halves should equal abc_C
    top_half = C_two[:abc_C.shape[0]].toarray()
    bot_half = C_two[abc_C.shape[0]:].toarray()
    assert np.allclose(top_half, abc_C.toarray())
    assert np.allclose(bot_half, abc_C.toarray())
    assert g_two.shape == (2 * abc_C.shape[0],) and np.allclose(g_two, 0.0)
    print(f"  stack_constraints([asm, asm]) gives 2x-tall block correctly")

    print(f"  PASS  ConstraintAssembler ABC + stack_constraints")


if __name__ == "__main__":
    print("Running mortar 2D unit tests")
    print("-" * 60)
    print("Test 1: dual basis bi-orthogonality")
    test_dual_basis_biorthogonality()
    print("Test 2: shape function partition of unity")
    test_dual_basis_partition_of_unity()
    print("Test 3: Wohlmuth crosspoint modification (Lopes Eq. C.2)")
    test_wohlmuth_crosspoint_modification()
    print("Test 4: conforming pair recovers lumped mass")
    test_conforming_pair_recovers_lumping()
    print("Test 5: non-conforming pair row-sum consistency")
    test_nonconforming_pair_consistency()
    print("Test 6: ConstraintAssembler ABC + stack_constraints")
    test_constraint_assembler_abc()
    print("-" * 60)
    print("All unit tests passed.")
