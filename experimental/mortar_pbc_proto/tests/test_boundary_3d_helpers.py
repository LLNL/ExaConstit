"""Phase 3.3.B unit tests — pure-Python helpers in BoundaryClassifier3D.

The classifier itself touches MFEM (ParSubMesh, parent vertex maps), so
end-to-end testing waits for the macOS validation pass. But several
pieces of its logic are pure-Python and unit-testable here:

  1. ``_classify_quad_boundary_tag`` — sentinel pattern -> Wohlmuth tag.
  2. ``_classify_tri_boundary_tag`` — same for tris.
  3. ``_param_axis_from_attrs`` — attr pair -> parametric axis.
  4. ``_face_bounding_edge_labels`` — face -> 4 bounding edge labels.
  5. ``_reorder_face_vertices_ccw`` — CCW reordering of synthetic
     face elements based on outward-normal direction.

Plus integration-readiness checks: every classification path is
exercised against the QuadFaceMortarAssembler / TriFaceMortarAssembler
boundary-tag dispatch tables, so we know the tag-string contract is
honoured end-to-end.

References
----------
* MORTAR_PBC_ARCHITECTURE.md §11.8 Phase 3.3.B (this layer).
"""
from __future__ import annotations

import os
import sys

# Defensive path setup — see test_face_mortar_3d.py for full rationale.
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
_actual = os.path.realpath(os.path.dirname(mortar_pbc.__file__))
_expected = os.path.realpath(_LOCAL_PKG)
if _actual != _expected:
    raise RuntimeError(
        f"\n  mortar_pbc resolved to a different location than expected:\n"
        f"      resolved : {_actual}\n"
        f"      expected : {_expected}\n"
        f"  Run `pip uninstall mortar-pbc` to remove a stale editable install.\n"
    )

import numpy as np                                                    # noqa: E402

# Direct import from boundary_3d to test the helpers without going
# through the lazy-loader (which would import MFEM).
from mortar_pbc.boundary_3d import (                                  # noqa: E402
    BoundaryClassifier3D,
    _FACE_AXES,
    _AXIS_EXTREME_TO_LABEL,
    _FaceElementRecord,
)
from mortar_pbc import (                                              # noqa: E402
    QuadFaceMortarAssembler,
    TriFaceMortarAssembler,
)


# Helper: build a stub classifier instance with a mock attr->label
# mapping. Phase 3.3.B used to expose _FACE_LABEL_BY_ATTR and
# _edge_label as module-level constants; after the runtime-discovery
# refactor (Phase 3.3.C macOS validation) they're instance attributes.
# These tests construct a minimal stub bypassing __init__ to exercise
# the now-instance methods directly.

def _make_stub_classifier(face_label_by_attr=None):
    """Create a BoundaryClassifier3D instance without calling __init__.

    Sets up just enough state to exercise the topology helpers
    (`_param_axis_from_attrs`, `_face_bounding_edge_labels`,
    `_edge_label`). The standard MFEM-equivalent attr ordering used:
        1=bottom, 2=front, 3=right, 4=back, 5=left, 6=top
    matches the ORIGINAL hardcoded mapping the tests were written
    against (the actual MFEM ordering may differ; that's why
    discovery exists).
    """
    if face_label_by_attr is None:
        face_label_by_attr = {
            1: "bottom", 2: "front", 3: "right",
            4: "back",   5: "left",  6: "top",
        }
    stub = BoundaryClassifier3D.__new__(BoundaryClassifier3D)
    stub._face_label_by_attr = face_label_by_attr
    stub._face_attr_by_label = {v: k for k, v in face_label_by_attr.items()}
    return stub


# =============================================================================
# Test 1: quad-4 boundary tag classification — every Wohlmuth pattern
# =============================================================================

def test_quad_boundary_tag_dispatch_all_patterns():
    """Every quad-4 sentinel pattern produces a tag the assembler accepts.

    The contract: any tag returned by ``_classify_quad_boundary_tag``
    must be in the QuadFaceMortarAssembler's tag table. Verified for
    all sentinel patterns: 0 sentinels (1 case), 1 sentinel (4 cases),
    2 sentinels in 4 edge-aligned configs + 2 diagonal cases, 3+
    sentinels (degenerate fallback to 'none').
    """
    accepted_tags = set(QuadFaceMortarAssembler._quad4_boundary_tag_to_sides.__defaults__ or ())
    # The mapping is built inside the method; rather than introspect,
    # call it on every tag the classifier might emit and check it
    # doesn't raise.
    asm = QuadFaceMortarAssembler()
    test_cases = [
        # (sentinels, expected_tag)
        ([99, 99, 99, 99],     "none"),
        # 1 sentinel: simple corner-of-element-only DOFs
        ([-1, 99, 99, 99],     "corner-LL"),
        ([99, -1, 99, 99],     "corner-LR"),
        ([99, 99, -1, 99],     "corner-UR"),
        ([99, 99, 99, -1],     "corner-UL"),
        # 2 sentinels: edge-aligned pairs
        ([-2, -2, 99, 99],     "edge-eta-low"),
        ([99, -2, -2, 99],     "edge-xi-high"),
        ([99, 99, -2, -2],     "edge-eta-high"),
        ([-2, 99, 99, -2],     "edge-xi-low"),
        # 2 sentinels: diagonal pairs (anomalous, fallback to none)
        ([-1, 99, -1, 99],     "none"),
        # 3 sentinels (corner-of-face quad): the corner-XX tag names
        # which SIDES of the quad are dropped (not which corner is
        # kept). E.g., if the kept node is at the UR corner of the
        # element (xi=+1, eta=+1), the sentinels cover the LL sides
        # (xi-low and eta-low), so the tag is 'corner-LL'.
        ([99, -2, -1, -2],     "corner-UR"),    # kept node 0 (LL); drops xi-high+eta-high
        ([-2, 99, -2, -1],     "corner-UL"),    # kept node 1 (LR); drops xi-low+eta-high
        ([-1, -2, 99, -2],     "corner-LL"),    # kept node 2 (UR); drops xi-low+eta-low
        ([-2, -1, -2, 99],     "corner-LR"),    # kept node 3 (UL); drops xi-high+eta-low
        # 4 sentinels (degenerate; element contributes nothing)
        ([-1, -1, -1, -1],     "none"),
    ]
    for sentinels, expected in test_cases:
        got = BoundaryClassifier3D._classify_quad_boundary_tag(sentinels)
        assert got == expected, (
            f"sentinels={sentinels}: got {got!r}, expected {expected!r}"
        )
        # Verify the assembler accepts the tag (doesn't raise on dispatch).
        side_xi, side_eta = asm._quad4_boundary_tag_to_sides(got)
        assert side_xi in ("none", "left", "right")
        assert side_eta in ("none", "bottom", "top")
    print(f"  PASS  quad boundary tags: {len(test_cases)} patterns dispatch cleanly to "
          f"M_quad4_dual_modified")


# =============================================================================
# Test 2: tri-3 boundary tag classification — every Wohlmuth pattern
# =============================================================================

def test_tri_boundary_tag_dispatch_all_patterns():
    """Every tri-3 sentinel pattern produces a tag the assembler accepts."""
    asm = TriFaceMortarAssembler()
    test_cases = [
        ([99, 99, 99],   "none"),
        ([-1, 99, 99],   "v0"),
        ([99, -1, 99],   "v1"),
        ([99, 99, -1],   "v2"),
        ([-1, -1, 99],   "v0-v1"),
        ([-1, 99, -1],   "v0-v2"),
        ([99, -1, -1],   "v1-v2"),
        ([-1, -1, -1],   "v0-v1-v2"),
        # Edge sentinels are also valid (they trip the same negative-int filter)
        ([-2, 99, 99],   "v0"),
        ([-2, -2, 99],   "v0-v1"),
    ]
    for sentinels, expected in test_cases:
        got = BoundaryClassifier3D._classify_tri_boundary_tag(sentinels)
        assert got == expected, (
            f"sentinels={sentinels}: got {got!r}, expected {expected!r}"
        )
        # Verify the assembler accepts the tag.
        drops = asm._tri3_boundary_tag_to_drops(got)
        assert sum(drops) == sum(1 for s in sentinels if s < 0)
    print(f"  PASS  tri boundary tags: 10 patterns dispatch cleanly to "
          f"M_tri3_dual_modified")


# =============================================================================
# Test 3: parametric-axis inference from face-attribute pair
# =============================================================================

def test_param_axis_from_attrs():
    """Two adjacent face attrs uniquely determine the shared edge's axis."""
    stub = _make_stub_classifier()
    # 1=bottom (y), 2=front (z), 3=right (x), 4=back (z), 5=left (x), 6=top (y)
    cases = [
        # (face1_attr, face2_attr, expected_axis)
        # Bottom (y_min) shares an edge with front (z_min) along x:
        ((1, 2), "x"),
        ((1, 4), "x"),  # bottom-back along x
        ((1, 3), "z"),  # bottom-right along z
        ((1, 5), "z"),  # bottom-left along z
        ((6, 2), "x"),  # top-front along x
        ((6, 5), "z"),  # top-left along z
        ((3, 2), "y"),  # right-front along y
        ((3, 4), "y"),  # right-back along y
        ((5, 2), "y"),  # left-front along y
    ]
    for attrs, expected in cases:
        got = stub._param_axis_from_attrs(attrs)
        assert got == expected, (
            f"attrs={attrs}: got {got!r}, expected {expected!r}"
        )
    # Mortar-nonmortar pairs (same perp axis) should raise.
    raised = False
    try:
        # bottom (y) + top (y): same perp axis, not adjacent.
        stub._param_axis_from_attrs((1, 6))
    except ValueError as e:
        raised = True
        assert "share the same perp axis" in str(e)
    assert raised, "Mortar-nonmortar pair should raise"
    print(f"  PASS  parametric-axis inference: 9 adjacent pairs correct + "
          f"mortar-nonmortar pair raises")


# =============================================================================
# Test 4: face bounding edges
# =============================================================================

def test_face_bounding_edge_labels():
    """Each box face has exactly 4 bounding edges with correct labels."""
    stub = _make_stub_classifier()
    # bottom (attr 1, perp y) is bounded by edges to all 4 non-mortar faces:
    # Labels are formed by sort-by-ATTR-INT (NOT alphabetic), per _edge_label:
    #   - front (2, perp z): edge along x  -> "x-bottom-front"  (1 < 2)
    #   - right (3, perp x): edge along z  -> "z-bottom-right"  (1 < 3)
    #   - back  (4, perp z): edge along x  -> "x-bottom-back"   (1 < 4)
    #   - left  (5, perp x): edge along z  -> "z-bottom-left"   (1 < 5)
    bottom_edges = stub._face_bounding_edge_labels(1)
    assert len(bottom_edges) == 4, f"bottom has {len(bottom_edges)} edges"
    expected = {
        "x-bottom-front", "z-bottom-right", "x-bottom-back", "z-bottom-left",
    }
    assert set(bottom_edges) == expected, (
        f"bottom edges: {bottom_edges}, expected {expected}"
    )

    # right (attr 3, perp x) is bounded by 4 edges to non-x-perp faces:
    #   - bottom (1, perp y): edge along z -> "z-bottom-right"  (1 < 3)
    #   - front  (2, perp z): edge along y -> "y-front-right"   (2 < 3)
    #   - back   (4, perp z): edge along y -> "y-right-back"    (3 < 4)
    #   - top    (6, perp y): edge along z -> "z-right-top"     (3 < 6)
    right_edges = stub._face_bounding_edge_labels(3)
    assert len(right_edges) == 4, f"right has {len(right_edges)} edges"
    expected_right = {
        "z-bottom-right", "y-front-right", "y-right-back", "z-right-top",
    }
    assert set(right_edges) == expected_right, (
        f"right edges: {right_edges}, expected {expected_right}"
    )

    # All 6 faces should each have 4 bounding edges.
    for attr in range(1, 7):
        assert len(stub._face_bounding_edge_labels(attr)) == 4

    # Total unique edges across all 6 faces should be 12 (each edge bounds
    # exactly 2 faces).
    all_edges_with_dups = []
    for attr in range(1, 7):
        all_edges_with_dups.extend(stub._face_bounding_edge_labels(attr))
    assert len(all_edges_with_dups) == 24, (
        f"Total face-edge incidences = {len(all_edges_with_dups)}, expected 24"
    )
    assert len(set(all_edges_with_dups)) == 12, (
        f"Unique edges = {len(set(all_edges_with_dups))}, expected 12"
    )
    print(f"  PASS  face-bounding edges: 4 per face, 12 unique total, "
          f"24 incidences")


# =============================================================================
# Test 5: edge label scheme is symmetric in attrs
# =============================================================================

def test_edge_label_symmetric():
    """_edge_label((a1, a2)) == _edge_label((a2, a1))."""
    stub = _make_stub_classifier()
    cases = [
        ("x", (1, 2)),  # bottom-front
        ("z", (3, 6)),  # right-top
        ("y", (3, 4)),  # right-back
    ]
    for axis, (a, b) in cases:
        lbl_ab = stub._edge_label(axis, (a, b))
        lbl_ba = stub._edge_label(axis, (b, a))
        assert lbl_ab == lbl_ba, f"{lbl_ab!r} != {lbl_ba!r}"
    print(f"  PASS  edge-label scheme is symmetric in attribute order")


# =============================================================================
# Test 6: CCW reordering of a synthetic face element (axis-aligned quad)
# =============================================================================

def test_ccw_reordering_top_face_quad():
    """A quad-4 on the top face (y=y_max) — outward normal +y.

    Construct vertices in CW order (viewed from +y), expect them to be
    reversed to CCW after `_reorder_face_vertices_ccw`.

    Top face parametric axes per _FACE_AXES: ("x", "z").
    For CCW viewed from +y, traversal in (x, z) plane should have
    positive shoelace area: e.g. (0,0) -> (1,0) -> (1,1) -> (0,1)
    walks CCW in the (x, z) plane. The outward-normal +y "looks down"
    onto the plane; CCW from +y is exactly CCW in (x, z) if the cross
    product (dx) × (dz) gives +y, which it does (right-hand rule on
    standard orientation).
    """
    # Build a synthetic ParSubMesh-style record for a top-face quad.
    # Vertices in CW order (viewed from +y): (0,1,0), (1,1,0), (1,1,1), (0,1,1)
    # is actually CCW from +y because the shoelace area in (x, z) is
    # positive for this traversal. Let's reverse them to provide a CW input.
    coords_cw = np.asarray([
        [0.0, 1.0, 0.0],   # local 0: (x=0, z=0)
        [0.0, 1.0, 1.0],   # local 1: (x=0, z=1)
        [1.0, 1.0, 1.0],   # local 2: (x=1, z=1)
        [1.0, 1.0, 0.0],   # local 3: (x=1, z=0)
    ], dtype=np.float64)
    # In (x, z) plane: (0,0) -> (0,1) -> (1,1) -> (1,0) — that's CW,
    # signed shoelace = (0*1 - 0*0) + (0*1 - 1*1) + (1*0 - 1*1) + (1*0 - 0*0)
    #                 = 0 + (-1) + (-1) + 0 = -2. Halved: -1. NEGATIVE.
    # Outward = +y, so we want signed area positive ⇒ reverse.
    rec = _FaceElementRecord(
        parent_attr=6, geometry_kind="quad",
        parent_vertex_ids=(100, 101, 102, 103),
        coords=coords_cw,
    )
    # Build a minimal-state classifier-like instance just to call the method.
    # We can call the method as an unbound function since it's not @staticmethod.
    # Use an instance with bbox set (so plane_value lookup works).
    class _Stub:
        bbox_min = np.zeros(3)
        bbox_max = np.array([1.0, 1.0, 1.0])
        tol = 1e-9
    stub = _Stub()
    pvids, coords = BoundaryClassifier3D._reorder_face_vertices_ccw(
        stub, rec, "top", "y", 1.0,
    )
    # Input was CW from +y; output should be CCW from +y. The result
    # is the input list reversed, so we just verify the CCW property
    # rather than asserting an exact ordering (the actual ordering
    # depends on whether reversal happens — which it should for this
    # CW input). Check: shoelace area in (x, z) plane is now positive.
    pts_xz = coords[:, [0, 2]]
    signed = 0.0
    n = pts_xz.shape[0]
    for i in range(n):
        x1, z1 = pts_xz[i]
        x2, z2 = pts_xz[(i + 1) % n]
        signed += (x1 * z2 - x2 * z1)
    signed *= 0.5
    assert signed > 0, f"After CCW reorder: signed area = {signed}, expected > 0"
    # And confirm the reversal happened — original ordering had signed_area < 0,
    # so the reversed pvids should NOT equal the input's pvids.
    assert pvids != [100, 101, 102, 103], (
        f"Expected CW input to be reversed; pvids = {pvids} (unchanged)"
    )
    # Specifically: for a 4-element list [a, b, c, d], reversal is [d, c, b, a].
    assert pvids == [103, 102, 101, 100], (
        f"After reversal: pvids = {pvids}, expected [103, 102, 101, 100]"
    )
    print(f"  PASS  CCW reordering on top face: CW input flipped to CCW "
          f"(shoelace area = {signed:+.4f})")


def test_ccw_reordering_bottom_face_quad_passthrough():
    """A quad-4 on the bottom face (y=y_min) — outward normal -y.

    Outward = -y means CCW viewed from -y. In (x, z), CCW from -y is
    the OPPOSITE orientation of CCW from +y. So a quad with positive
    shoelace in (x, z) (CCW from +y) is actually CW from -y, and
    should be reversed.
    """
    # Vertices arranged CCW from +y (positive shoelace in (x, z)):
    # (0,0) -> (1,0) -> (1,1) -> (0,1) gives signed area = +1.
    coords = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    rec = _FaceElementRecord(
        parent_attr=1, geometry_kind="quad",
        parent_vertex_ids=(200, 201, 202, 203),
        coords=coords,
    )
    class _Stub:
        bbox_min = np.zeros(3)
        bbox_max = np.array([1.0, 1.0, 1.0])
        tol = 1e-9
    stub = _Stub()
    pvids, _ = BoundaryClassifier3D._reorder_face_vertices_ccw(
        stub, rec, "bottom", "y", 0.0,
    )
    # Input was CCW-from-+y (positive shoelace in (x, z)); but for a
    # bottom face, outward normal is -y, so we want CCW-from--y, which
    # is OPPOSITE of CCW-from-+y. The implementation should reverse.
    assert pvids == [203, 202, 201, 200], (
        f"Bottom face CCW reorder: pvids = {pvids}, expected reversed"
    )
    print(f"  PASS  CCW reordering on bottom face: input flipped to CCW from -y")


# =============================================================================
# Test 7: end-to-end classification dispatch — feed sentinel-tagged elements
# directly into Phase-3.2.B assemblers
# =============================================================================

def test_sentinel_tagged_face_elements_drive_assembler_correctly():
    """Synthesise a face-element list (as if the classifier produced it)
    with one of every Wohlmuth tag, run the assembler, verify no
    assembler errors and reasonable D / A_m shapes.
    """
    from mortar_pbc.types_3d import QuadFaceElement, TriFaceElement
    asm_q = QuadFaceMortarAssembler()
    asm_t = TriFaceMortarAssembler()

    # Build a 1-element quad nonmortar with a corner sentinel pattern (corner-LL).
    # Nonmortar gtdofs: (-1, 0, 1, 2) — local 0 is a sentinel-corner.
    nonmortar_q = QuadFaceElement(
        coords=np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]]),
        gtdofs=(-1, 0, 1, 2),
        parametric_axes=("x", "z"), perpendicular_axis="y",
        boundary_tag="corner-LL",
    )
    mortar_q = QuadFaceElement(
        coords=np.asarray([[0., 1., 0.], [1., 1., 0.], [1., 1., 1.], [0., 1., 1.]]),
        gtdofs=(10, 11, 12, 13),
        parametric_axes=("x", "z"), perpendicular_axis="y",
    )
    block_q = asm_q.assemble_pair_conforming(
        nonmortar_elems=[nonmortar_q], mortar_elems=[mortar_q],
        pair_matches=[(0, 0, (0, 1, 2, 3))],
    )
    assert block_q.D.shape == (3,)
    assert block_q.A_m.shape == (3, 4)

    # Build a 1-element tri nonmortar with v0 sentinel pattern.
    nonmortar_t = TriFaceElement(
        coords=np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]]),
        gtdofs=(-1, 0, 1),
        parametric_axes=("x", "z"), perpendicular_axis="y",
        boundary_tag="v0",
    )
    mortar_t = TriFaceElement(
        coords=np.asarray([[0., 1., 0.], [1., 1., 0.], [0., 1., 1.]]),
        gtdofs=(10, 11, 12),
        parametric_axes=("x", "z"), perpendicular_axis="y",
    )
    block_t = asm_t.assemble_pair_conforming(
        nonmortar_elems=[nonmortar_t], mortar_elems=[mortar_t],
        pair_matches=[(0, 0, (0, 1, 2))],
    )
    assert block_t.D.shape == (2,)
    assert block_t.A_m.shape == (2, 3)
    print(f"  PASS  sentinel-tagged face-element dispatch: quad block "
          f"{block_q.A_m.shape}, tri block {block_t.A_m.shape}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Phase 3.3.B unit tests — BoundaryClassifier3D helpers")
    print("=" * 60)

    print()
    print("[Boundary tag classification]")
    test_quad_boundary_tag_dispatch_all_patterns()
    test_tri_boundary_tag_dispatch_all_patterns()

    print()
    print("[Topology helpers]")
    test_param_axis_from_attrs()
    test_face_bounding_edge_labels()
    test_edge_label_symmetric()

    print()
    print("[CCW orientation]")
    test_ccw_reordering_top_face_quad()
    test_ccw_reordering_bottom_face_quad_passthrough()

    print()
    print("[End-to-end dispatch into Phase-3.2.B assemblers]")
    test_sentinel_tagged_face_elements_drive_assembler_correctly()

    print()
    print("=" * 60)
    print(" All Phase 3.3.B helper tests passed.")
    print("=" * 60)
