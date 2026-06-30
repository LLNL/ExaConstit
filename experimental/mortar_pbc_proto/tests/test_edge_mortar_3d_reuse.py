"""Phase 3.3.A unit tests — `MortarAssembler2D` reuse on 3D edges.

The 2D edge-mortar machinery is dim-generic in its math (purely 1D
parametric integration with the line-2 dual basis). Only the axis
lookup in `_param_endpoints` was 2D-specific; Phase 3.3.A made it
support `"z"` too. These tests verify that:

  1. `MortarAssembler2D` instantiated with a duck-typed mock classifier
     of `EdgeInfo3D` objects produces correct mortar blocks for 3D
     edge pairs.
  2. The "z"-axis path returns the same lumping recovery (D = A_m =
     diag(per-segment Jacobian) on a conforming pair) as the existing
     "x"/"y"-axis paths in the 2D suite.
  3. All three axes behave identically up to coordinate relabelling
     (sanity check that the axis dispatch is symmetric).

References
----------
* MORTAR_PBC_ARCHITECTURE.md §11.8 Phase 3.3.A.
* `tests/test_mortar_2d_unit.py` — the 2D analog these tests parallel.
"""
from __future__ import annotations

import os
import sys

# ----------------------------------------------------------------------
# Defensive path setup — see test_face_mortar_3d.py for full rationale.
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_LOCAL_PKG = os.path.join(_PARENT, "mortar_pbc")
if not os.path.isdir(_LOCAL_PKG):
    raise RuntimeError(f"Cannot find mortar_pbc package at {_LOCAL_PKG!r}.")
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
        f"  Run `pip uninstall mortar-pbc` to remove a stale editable install.\n"
    )

import numpy as np                                                    # noqa: E402

from mortar_pbc import MortarAssembler2D                              # noqa: E402
from mortar_pbc.types_3d import EdgeInfo3D                            # noqa: E402


# =============================================================================
# Helper: build a synthetic conforming edge pair along an axis-aligned 3D edge
# =============================================================================

def _make_conforming_edge_pair(
    parametric_axis: str,
    edge_lo: float,
    edge_hi: float,
    n_nodes: int,
    *,
    perp_coords: tuple[float, float],
    mortar_perp_coords: tuple[float, float] | None = None,
):
    """Build a conforming (matching-element) 3D EdgeInfo3D pair.

    The `parametric_axis` defines the direction the edge runs in; the
    other two axes are held at the constant `perp_coords`. For the
    mortar edge, `mortar_perp_coords` (if given) places it offset
    along the perpendicular plane; otherwise the mortar is at the
    same perpendicular position as the nonmortar (only relevant for tests
    that don't actually distinguish mortar vs nonmortar geometrically —
    the mortar block depends only on parametric matching).

    The "elements" connectivity is the line-2 chain along the edge
    with corner sentinels at both ends:
        (-1, 0), (0, 1), (1, 2), ..., (n-1, -2)

    Returns (nonmortar_edge, mortar_edge), both `EdgeInfo3D` instances
    with `n_nodes` interior nodes (excluding corners).
    """
    if parametric_axis not in ("x", "y", "z"):
        raise ValueError(f"parametric_axis must be x/y/z, got {parametric_axis!r}")
    axis_idx = {"x": 0, "y": 1, "z": 2}[parametric_axis]

    if mortar_perp_coords is None:
        mortar_perp_coords = perp_coords

    # Interior node positions along the parametric axis (no corners).
    param_xs = np.linspace(edge_lo, edge_hi, n_nodes + 2)[1:-1]

    def build(perp: tuple[float, float], gtdof_offset: int) -> EdgeInfo3D:
        coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for i, t in enumerate(param_xs):
            xyz = [0.0, 0.0, 0.0]
            xyz[axis_idx] = float(t)
            other_axes = [a for a in (0, 1, 2) if a != axis_idx]
            xyz[other_axes[0]] = perp[0]
            xyz[other_axes[1]] = perp[1]
            coords[i] = xyz
        # Mock TDOFs (each component); the assembler doesn't read them.
        gtx = np.arange(n_nodes, dtype=np.int64) + gtdof_offset
        gty = np.arange(n_nodes, dtype=np.int64) + gtdof_offset + 1000
        gtz = np.arange(n_nodes, dtype=np.int64) + gtdof_offset + 2000
        # line-2 connectivity with corner sentinels at endpoints
        elements = [(-1, 0)]
        for k in range(n_nodes - 1):
            elements.append((k, k + 1))
        elements.append((n_nodes - 1, -2))
        return EdgeInfo3D(
            label=f"edge-{parametric_axis}",
            is_mortar=(gtdof_offset == 100),
            parametric_axis=parametric_axis,
            edge_min=edge_lo,
            edge_max=edge_hi,
            coords=coords,
            gtdofs_x=gtx, gtdofs_y=gty, gtdofs_z=gtz,
            elements=elements,
        )

    nonmortar = build(perp_coords, gtdof_offset=0)
    mortar = build(mortar_perp_coords, gtdof_offset=100)
    return nonmortar, mortar


class _MockClassifier:
    """Minimum mock that `MortarAssembler2D.__init__` accepts.

    The assembler only uses `cl.edges[name]` in `assemble_all`, but
    `assemble_pair` (the 3D entry point) doesn't go through that
    indirection — it takes the edges directly. We never use this
    mock's `edges` dict in the 3D tests.
    """
    edges = {}


# =============================================================================
# Test 1: x-axis 3D edge pair — conforming lumping recovery
# =============================================================================

def test_3d_edge_mortar_x_axis_conforming():
    """A conforming line-2 pair along the x-axis recovers signed-identity lumping."""
    nonmortar, mortar = _make_conforming_edge_pair(
        parametric_axis="x",
        edge_lo=0.0, edge_hi=2.0,
        n_nodes=4,                             # 4 interior nodes => 5 segments
        perp_coords=(0.0, 0.0),                # nonmortar at (y=0, z=0)
        mortar_perp_coords=(1.0, 1.0),         # mortar at (y=1, z=1) — offset OK
    )

    asm = MortarAssembler2D(_MockClassifier())
    block = asm.assemble_pair(nonmortar, mortar)

    # On a conforming aligned pair, A^m should equal diag(D^nm).
    diff = np.linalg.norm(block.A_m - np.diag(block.D_nm))
    assert diff < 1e-12, (
        f"x-axis 3D edge: ||A^m - diag(D^nm)||_F = {diff}, expected ~0"
    )
    # Each interior node carries Jacobian = (segment_length / 2) per
    # adjacent line-2 element; with two adjacent segments per interior
    # node and uniform spacing 2/5 = 0.4, D[k] = 2 * (0.4/2) = 0.4.
    expected = 0.4
    assert np.allclose(block.D_nm, expected, atol=1e-13), (
        f"x-axis 3D edge: D = {block.D_nm}, expected uniform {expected}"
    )
    print(f"  PASS  x-axis 3D edge: D = {expected:.4f} * 1_4, "
          f"A^m = diag(D), err = {diff:.2e}")


# =============================================================================
# Test 2: z-axis 3D edge pair — the new 3D-specific axis path
# =============================================================================

def test_3d_edge_mortar_z_axis_conforming():
    """A conforming line-2 pair along the z-axis (the new 3D axis path)."""
    nonmortar, mortar = _make_conforming_edge_pair(
        parametric_axis="z",
        edge_lo=0.0, edge_hi=3.0,              # different length to catch axis confusion
        n_nodes=5,                             # 5 interior nodes => 6 segments
        perp_coords=(0.0, 0.0),                # nonmortar at (x=0, y=0)
        mortar_perp_coords=(2.0, 2.0),         # mortar offset
    )
    asm = MortarAssembler2D(_MockClassifier())
    block = asm.assemble_pair(nonmortar, mortar)

    diff = np.linalg.norm(block.A_m - np.diag(block.D_nm))
    assert diff < 1e-12, f"z-axis 3D edge: ||A^m - diag(D^nm)||_F = {diff}"
    # Segment length = 3.0 / 6 = 0.5; per interior node = 2 * 0.5 / 2 = 0.5.
    expected = 0.5
    assert np.allclose(block.D_nm, expected, atol=1e-13), (
        f"z-axis 3D edge: D = {block.D_nm}, expected uniform {expected}"
    )
    print(f"  PASS  z-axis 3D edge: D = {expected:.4f} * 1_5, "
          f"A^m = diag(D), err = {diff:.2e}")


# =============================================================================
# Test 3: axis symmetry — same answer regardless of which axis the edge runs along
# =============================================================================

def test_3d_edge_mortar_axis_symmetry():
    """All three axes should give bit-identical mortar blocks for the same
    parametric 1D geometry. This sanity-checks the axis dispatch is
    symmetric — swapping x ↔ y ↔ z while keeping the parametric range
    fixed should produce the same D^nm and A^m up to numerical noise.
    """
    asm = MortarAssembler2D(_MockClassifier())

    blocks = {}
    for axis in ("x", "y", "z"):
        nonmortar, mortar = _make_conforming_edge_pair(
            parametric_axis=axis,
            edge_lo=0.0, edge_hi=1.0,
            n_nodes=3,
            perp_coords=(0.0, 0.0),
            mortar_perp_coords=(0.5, 0.5),
        )
        blocks[axis] = asm.assemble_pair(nonmortar, mortar)

    # All three should produce identical D^nm and A^m.
    D_x = blocks["x"].D_nm
    A_x = blocks["x"].A_m
    for axis in ("y", "z"):
        D_diff = np.max(np.abs(blocks[axis].D_nm - D_x))
        A_diff = np.max(np.abs(blocks[axis].A_m - A_x))
        assert D_diff < 1e-15, (
            f"axis symmetry: D^nm differs between x and {axis} by {D_diff}"
        )
        assert A_diff < 1e-15, (
            f"axis symmetry: A^m differs between x and {axis} by {A_diff}"
        )
    print(f"  PASS  axis symmetry: D^nm and A^m identical for x, y, z "
          f"(max diff {max(D_diff, A_diff):.2e})")


# =============================================================================
# Test 4: mixed-axis pairing (NEGATIVE test) — different axes must NOT pair
# =============================================================================

def test_3d_edge_mortar_axis_mismatch_misuse():
    """Edges on different parametric axes share no parametric overlap.

    This isn't a feature of the assembler itself — `MortarAssembler2D`
    will dutifully integrate whatever it's given — but it exercises
    the axis-dispatch path in `_param_endpoints` to confirm no
    cross-axis coordinate confusion happens. Specifically: if we
    mismatch a y-axis edge with a z-axis edge, the parametric
    coordinates compared are y on one side and z on the other; with
    edges on disjoint parametric ranges, the overlap should be zero
    and A^m should come back all-zero.
    """
    # Nonmortar on y-axis, range y ∈ [10, 20]. Mortar on z-axis, range z ∈ [0, 1].
    # No overlap in either parametric axis taken on its own; A^m = 0.
    nonmortar, _ = _make_conforming_edge_pair(
        parametric_axis="y",
        edge_lo=10.0, edge_hi=20.0,
        n_nodes=3,
        perp_coords=(0.0, 0.0),
    )
    mortar, _ = _make_conforming_edge_pair(
        parametric_axis="z",
        edge_lo=0.0, edge_hi=1.0,
        n_nodes=3,
        perp_coords=(0.0, 0.0),
    )
    asm = MortarAssembler2D(_MockClassifier())
    block = asm.assemble_pair(nonmortar, mortar)
    # D^nm uses only the nonmortar-side parametric range, so it's nonzero
    # (mortar_2d.py:_assemble_pair lines 304-307); A^m involves overlap
    # between nonmortar and mortar, and the nonmortar's y range vs mortar's z
    # range do NOT overlap geometrically — but the assembler compares
    # parametric coords directly. Since y ∈ [10, 20] never intersects
    # z ∈ [0, 1] (treated as scalars on the same number line), the
    # interval-intersection check rejects all overlaps.
    A_max = float(np.max(np.abs(block.A_m)))
    assert A_max == 0.0, (
        f"mismatch axes: expected A^m all zeros, got max |A^m| = {A_max}"
    )
    # D^nm is independent of mortar and should still be nonzero.
    assert float(np.min(block.D_nm)) > 0, (
        f"D^nm should be positive (nonmortar-side only), got {block.D_nm}"
    )
    print(f"  PASS  axis-mismatch sanity: A^m = 0 (no overlap), "
          f"D^nm = {block.D_nm[0]:.4f} * 1_3 (nonmortar-only)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Phase 3.3.A unit tests — MortarAssembler2D reuse on 3D edges")
    print("=" * 60)

    print()
    print("[3D edge-mortar reuse]")
    test_3d_edge_mortar_x_axis_conforming()
    test_3d_edge_mortar_z_axis_conforming()
    test_3d_edge_mortar_axis_symmetry()
    test_3d_edge_mortar_axis_mismatch_misuse()

    print()
    print("=" * 60)
    print(" All Phase 3.3.A tests passed.")
    print("=" * 60)
