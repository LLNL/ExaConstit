// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — unit tests for boundary_helpers_3d.{hpp,cpp},
// mirroring tests/test_boundary_3d_helpers.py. These tests cover the
// pure (no MFEM mesh, no MPI) helpers; the full-classifier integration
// tests come with Batch B / the patch-test driver.
//
// Each test function exits via std::exit(1) on failure (with a
// diagnostic to stderr) or returns normally on success. The main()
// at the bottom calls all of them in sequence and prints a summary.

#include "boundary_helpers_3d.hpp"
#include "face_mortar_assembler_3d.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using mortar_pbc::AxisExtremeToLabel;
using mortar_pbc::ClassifyQuadBoundaryTag;
using mortar_pbc::ClassifyTriBoundaryTag;
using mortar_pbc::EdgeLabel;
using mortar_pbc::FaceAxes;
using mortar_pbc::FaceBoundingEdgeLabels;
using mortar_pbc::FacePairs;
using mortar_pbc::MortarLabels;
using mortar_pbc::ParamAxisFromAttrs;
using mortar_pbc::ReorderFaceVerticesCcw;

namespace {

// ---- helper: standard 1=bottom, 2=front, 3=right, 4=back, 5=left, 6=top
//
// This matches the ordering used in test_boundary_3d_helpers.py's
// _make_stub_classifier helper.
const std::map<int, std::string>& StandardFaceLabelByAttr()
{
    static const std::map<int, std::string> kMap = {
        {1, "bottom"}, {2, "front"}, {3, "right"},
        {4, "back"},   {5, "left"},  {6, "top"},
    };
    return kMap;
}

// ---- helper: assert + diagnostic ------------------------------------------
void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

// ===========================================================================
// Test 1: AxisExtremeToLabel mapping is well-formed
// ===========================================================================
void test_axis_extreme_to_label()
{
    std::cout << "Test 1: AxisExtremeToLabel" << std::endl;
    AssertOrDie(AxisExtremeToLabel("y", "min") == "bottom", "AxisExtremeToLabel",
                "(y,min) != bottom");
    AssertOrDie(AxisExtremeToLabel("y", "max") == "top", "AxisExtremeToLabel",
                "(y,max) != top");
    AssertOrDie(AxisExtremeToLabel("z", "min") == "front", "AxisExtremeToLabel",
                "(z,min) != front");
    AssertOrDie(AxisExtremeToLabel("z", "max") == "back", "AxisExtremeToLabel",
                "(z,max) != back");
    AssertOrDie(AxisExtremeToLabel("x", "min") == "left", "AxisExtremeToLabel",
                "(x,min) != left");
    AssertOrDie(AxisExtremeToLabel("x", "max") == "right", "AxisExtremeToLabel",
                "(x,max) != right");
    std::cout << "  PASS  AxisExtremeToLabel: 6 canonical mappings correct"
              << std::endl;
}

// ===========================================================================
// Test 2: FacePairs and MortarLabels are consistent
// ===========================================================================
void test_face_pairs_mortar_labels()
{
    std::cout << "Test 2: FacePairs / MortarLabels" << std::endl;
    const auto& pairs = FacePairs();
    AssertOrDie(pairs.size() == 3, "FacePairs", "size != 3");
    const auto& mortars = MortarLabels();
    AssertOrDie(mortars.size() == 3, "MortarLabels", "size != 3");

    // Mortar labels should be exactly the first elements of each pair.
    std::set<std::string> first_of_pairs;
    for (const auto& p : pairs) { first_of_pairs.insert(p.first); }
    AssertOrDie(first_of_pairs == mortars, "consistency",
                "MortarLabels != first-of-FacePairs");

    // Specifically, the locked convention.
    AssertOrDie(mortars == std::set<std::string>{"top", "right", "back"},
                "convention",
                "Mortar labels not {top, right, back}");
    std::cout << "  PASS  FacePairs/MortarLabels: 3 pairs, mortar = "
                 "{top, right, back}" << std::endl;
}

// ===========================================================================
// Test 3: FaceAxes consistency for all 6 faces
// ===========================================================================
void test_face_axes()
{
    std::cout << "Test 3: FaceAxes" << std::endl;
    for (const std::string& f :
         {std::string("bottom"), std::string("top"), std::string("front"),
          std::string("back"), std::string("left"), std::string("right")})
    {
        auto pa = FaceAxes(f);
        const std::string& perp = pa.first;
        const auto& params = pa.second;
        // Perp must be one of x/y/z, params must be the other two,
        // and the two params must be distinct.
        std::set<std::string> all{perp, params[0], params[1]};
        AssertOrDie(all == std::set<std::string>{"x", "y", "z"},
                    "FaceAxes(" + f + ")",
                    "axes don't form {x, y, z}");
    }
    // Specific relationships matter for CCW reordering: top/bottom should
    // share (perp=y, params=(x,z)), etc.
    AssertOrDie(FaceAxes("top").first == "y",
                "FaceAxes top", "perp != y");
    AssertOrDie(FaceAxes("bottom").first == "y",
                "FaceAxes bottom", "perp != y");
    AssertOrDie(FaceAxes("right").first == "x",
                "FaceAxes right", "perp != x");
    AssertOrDie(FaceAxes("back").first == "z",
                "FaceAxes back", "perp != z");
    std::cout << "  PASS  FaceAxes: 6 faces all consistent (perp/param "
                 "axes form xyz partition)" << std::endl;
}

// ===========================================================================
// Test 4: ParamAxisFromAttrs — the unique perp-perp axis
// ===========================================================================
void test_param_axis_from_attrs()
{
    std::cout << "Test 4: ParamAxisFromAttrs" << std::endl;
    const auto& m = StandardFaceLabelByAttr();

    // (face1_attr, face2_attr, expected_axis)
    struct Case { int a; int b; std::string expected; };
    std::vector<Case> cases = {
        // bottom (y_min) shares an edge with front (z_min) along x:
        {1, 2, "x"},
        {1, 4, "x"},  // bottom-back along x
        {1, 3, "z"},  // bottom-right along z
        {1, 5, "z"},  // bottom-left along z
        {6, 2, "x"},  // top-front along x
        {6, 5, "z"},  // top-left along z
        {3, 2, "y"},  // right-front along y
        {3, 4, "y"},  // right-back along y
        {5, 2, "y"},  // left-front along y
    };
    for (const auto& c : cases)
    {
        std::string got = ParamAxisFromAttrs({c.a, c.b}, m);
        AssertOrDie(got == c.expected,
                    "ParamAxisFromAttrs",
                    "attrs=(" + std::to_string(c.a) + "," + std::to_string(c.b)
                    + "): got '" + got + "', expected '" + c.expected + "'");
    }
    std::cout << "  PASS  ParamAxisFromAttrs: 9 adjacent pairs correct"
              << std::endl;
}

// ===========================================================================
// Test 5: EdgeLabel is symmetric in attrs (sorted by integer)
// ===========================================================================
void test_edge_label_symmetric()
{
    std::cout << "Test 5: EdgeLabel symmetry" << std::endl;
    const auto& m = StandardFaceLabelByAttr();
    struct Case { std::string axis; int a; int b; };
    std::vector<Case> cases = {
        {"x", 1, 2},  // bottom-front
        {"z", 3, 6},  // right-top
        {"y", 3, 4},  // right-back
    };
    for (const auto& c : cases)
    {
        std::string ab = EdgeLabel(c.axis, {c.a, c.b}, m);
        std::string ba = EdgeLabel(c.axis, {c.b, c.a}, m);
        AssertOrDie(ab == ba, "EdgeLabel symmetry",
                    "EdgeLabel('" + c.axis + "',"
                    + std::to_string(c.a) + "," + std::to_string(c.b)
                    + ") = '" + ab + "' != EdgeLabel(reversed) = '" + ba + "'");
    }
    std::cout << "  PASS  EdgeLabel: symmetric in attribute order" << std::endl;
}

// ===========================================================================
// Test 6: FaceBoundingEdgeLabels — 4 edges per face, 12 unique total
// ===========================================================================
void test_face_bounding_edge_labels()
{
    std::cout << "Test 6: FaceBoundingEdgeLabels" << std::endl;
    const auto& m = StandardFaceLabelByAttr();

    // bottom (attr 1, perp y) is bounded by edges to all 4 non-mortar
    // axis faces. Labels follow EdgeLabel(axis, sorted(attrs)):
    //   - front (2, perp z): edge along x  -> "x-bottom-front"
    //   - right (3, perp x): edge along z  -> "z-bottom-right"
    //   - back  (4, perp z): edge along x  -> "x-bottom-back"
    //   - left  (5, perp x): edge along z  -> "z-bottom-left"
    std::vector<std::string> bottom_edges = FaceBoundingEdgeLabels(1, m);
    AssertOrDie(bottom_edges.size() == 4, "bottom edges count",
                "got " + std::to_string(bottom_edges.size()));
    std::set<std::string> bottom_set(bottom_edges.begin(), bottom_edges.end());
    std::set<std::string> expected_bottom = {
        "x-bottom-front", "z-bottom-right", "x-bottom-back", "z-bottom-left",
    };
    AssertOrDie(bottom_set == expected_bottom,
                "bottom edges set",
                "FaceBoundingEdgeLabels(1) does not match expected");

    // right (attr 3, perp x) is bounded by 4 edges to non-x-perp faces:
    //   - bottom (1, perp y): edge along z -> "z-bottom-right"  (1<3)
    //   - front  (2, perp z): edge along y -> "y-front-right"   (2<3)
    //   - back   (4, perp z): edge along y -> "y-right-back"    (3<4)
    //   - top    (6, perp y): edge along z -> "z-right-top"     (3<6)
    std::vector<std::string> right_edges = FaceBoundingEdgeLabels(3, m);
    AssertOrDie(right_edges.size() == 4, "right edges count",
                "got " + std::to_string(right_edges.size()));
    std::set<std::string> right_set(right_edges.begin(), right_edges.end());
    std::set<std::string> expected_right = {
        "z-bottom-right", "y-front-right", "y-right-back", "z-right-top",
    };
    AssertOrDie(right_set == expected_right,
                "right edges set",
                "FaceBoundingEdgeLabels(3) does not match expected");

    // All 6 faces should each have 4 bounding edges.
    int total_incidences = 0;
    std::set<std::string> all_unique_edges;
    for (int attr = 1; attr <= 6; ++attr)
    {
        std::vector<std::string> edges = FaceBoundingEdgeLabels(attr, m);
        AssertOrDie(edges.size() == 4, "edges per face",
                    "face attr " + std::to_string(attr) + " has "
                    + std::to_string(edges.size()) + " edges, expected 4");
        total_incidences += static_cast<int>(edges.size());
        for (const auto& e : edges) { all_unique_edges.insert(e); }
    }
    AssertOrDie(total_incidences == 24, "total incidences",
                "got " + std::to_string(total_incidences) + ", expected 24");
    AssertOrDie(all_unique_edges.size() == 12, "unique edges",
                "got " + std::to_string(all_unique_edges.size())
                + ", expected 12");

    std::cout << "  PASS  FaceBoundingEdgeLabels: 4 per face, 12 unique total, "
                 "24 incidences" << std::endl;
}

// ===========================================================================
// Test 7: ClassifyQuadBoundaryTag — every Wohlmuth pattern
// ===========================================================================
void test_classify_quad_boundary_tag()
{
    std::cout << "Test 7: ClassifyQuadBoundaryTag" << std::endl;
    struct Case { std::array<int, 4> sentinels; std::string expected; };
    std::vector<Case> cases = {
        // 0 sentinels: face-interior quad
        {{99, 99, 99, 99},     "none"},
        // 1 sentinel: simple corner-of-element-only DOFs
        {{-1, 99, 99, 99},     "corner-LL"},
        {{99, -1, 99, 99},     "corner-LR"},
        {{99, 99, -1, 99},     "corner-UR"},
        {{99, 99, 99, -1},     "corner-UL"},
        // 2 sentinels: edge-aligned pairs
        {{-2, -2, 99, 99},     "edge-eta-low"},
        {{99, -2, -2, 99},     "edge-xi-high"},
        {{99, 99, -2, -2},     "edge-eta-high"},
        {{-2, 99, 99, -2},     "edge-xi-low"},
        // 2 sentinels: diagonal pairs (anomalous, fallback to none)
        {{-1, 99, -1, 99},     "none"},
        // 3 sentinels (corner-of-face quad): the corner-XX tag names
        // which SIDES of the quad are dropped (not which corner is
        // kept). E.g., kept node 0 (LL) -> drops xi-high+eta-high -> UR.
        {{99, -2, -1, -2},     "corner-UR"},  // kept node 0
        {{-2, 99, -2, -1},     "corner-UL"},  // kept node 1
        {{-1, -2, 99, -2},     "corner-LL"},  // kept node 2
        {{-2, -1, -2, 99},     "corner-LR"},  // kept node 3
        // 4 sentinels (degenerate; element contributes nothing)
        {{-1, -1, -1, -1},     "none"},
    };
    for (const auto& c : cases)
    {
        std::string got = ClassifyQuadBoundaryTag(c.sentinels);
        std::ostringstream detail;
        detail << "sentinels=[" << c.sentinels[0] << "," << c.sentinels[1]
               << "," << c.sentinels[2] << "," << c.sentinels[3]
               << "]: got '" << got << "', expected '" << c.expected << "'";
        AssertOrDie(got == c.expected, "ClassifyQuadBoundaryTag", detail.str());
    }
    std::cout << "  PASS  ClassifyQuadBoundaryTag: " << cases.size()
              << " patterns dispatch correctly" << std::endl;
}

// ===========================================================================
// Test 8: ClassifyTriBoundaryTag — every Wohlmuth tri pattern
// ===========================================================================
void test_classify_tri_boundary_tag()
{
    std::cout << "Test 8: ClassifyTriBoundaryTag" << std::endl;
    struct Case { std::array<int, 3> sentinels; std::string expected; };
    std::vector<Case> cases = {
        {{99, 99, 99},  "none"},
        {{-1, 99, 99},  "v0"},
        {{99, -1, 99},  "v1"},
        {{99, 99, -1},  "v2"},
        {{-1, -1, 99},  "v0-v1"},
        {{-1, 99, -1},  "v0-v2"},
        {{99, -1, -1},  "v1-v2"},
        {{-1, -1, -1},  "v0-v1-v2"},
    };
    for (const auto& c : cases)
    {
        std::string got = ClassifyTriBoundaryTag(c.sentinels);
        std::ostringstream detail;
        detail << "sentinels=[" << c.sentinels[0] << "," << c.sentinels[1]
               << "," << c.sentinels[2] << "]: got '" << got
               << "', expected '" << c.expected << "'";
        AssertOrDie(got == c.expected, "ClassifyTriBoundaryTag", detail.str());
    }
    std::cout << "  PASS  ClassifyTriBoundaryTag: " << cases.size()
              << " patterns dispatch correctly" << std::endl;
}

// ===========================================================================
// Test 9: ReorderFaceVerticesCcw — top-face quad with CW input
// ===========================================================================
void test_reorder_top_face_quad()
{
    std::cout << "Test 9: ReorderFaceVerticesCcw on top face" << std::endl;
    // Input: vertices arranged CW (viewed from +y, the outward normal).
    // In (x, z) plane: (0,0) -> (0,1) -> (1,1) -> (1,0) is CW
    // (signed shoelace = -1, NEGATIVE). Outward normal = +y, so
    // CCW-from-outward needs signed_area > 0 — reorder should reverse.
    mfem::DenseMatrix coords(4, 3);
    // Format: (x, y, z) with y = 1.0 fixed (top face)
    double cw_data[4][3] = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 0.0},
    };
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j) { coords(i, j) = cw_data[i][j]; }
    }
    std::vector<int> pvids = {100, 101, 102, 103};
    ReorderFaceVerticesCcw(coords, pvids, "top");

    // After reordering, signed shoelace area in (x, z) must be > 0.
    double signed_area = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        const int ip1 = (i + 1) % 4;
        const double x1 = coords(i, 0), z1 = coords(i, 2);
        const double x2 = coords(ip1, 0), z2 = coords(ip1, 2);
        signed_area += (x1 * z2 - x2 * z1);
    }
    signed_area *= 0.5;
    AssertOrDie(signed_area > 0.0, "top face CCW",
                "signed area = " + std::to_string(signed_area)
                + ", expected > 0");

    // Specifically, reversal of [100, 101, 102, 103] is [103, 102, 101, 100].
    AssertOrDie(pvids == std::vector<int>{103, 102, 101, 100},
                "top face vertex_ids reversal",
                "pvids did not reverse as expected");
    std::cout << "  PASS  ReorderFaceVerticesCcw on top face: CW input flipped "
                 "to CCW (signed area = " << signed_area << ")" << std::endl;
}

// ===========================================================================
// Test 10: ReorderFaceVerticesCcw — bottom-face quad with input that's
// CCW-from-+y (which is CW-from--y, i.e. wrong for the bottom outward normal)
// ===========================================================================
void test_reorder_bottom_face_quad()
{
    std::cout << "Test 10: ReorderFaceVerticesCcw on bottom face" << std::endl;
    mfem::DenseMatrix coords(4, 3);
    // CCW-from-+y in (x, z): (0,0) -> (1,0) -> (1,1) -> (0,1)
    //   shoelace = (0*0 - 1*0) + (1*1 - 1*0) + (1*1 - 0*1) + (0*0 - 0*1)
    //            = 0 + 1 + 1 + 0 = +2 -> halved = +1 (positive)
    // Outward = -y, so we want signed_area < 0; thus reorder should reverse.
    double data[4][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 0.0, 1.0},
    };
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j) { coords(i, j) = data[i][j]; }
    }
    std::vector<int> pvids = {200, 201, 202, 203};
    ReorderFaceVerticesCcw(coords, pvids, "bottom");

    AssertOrDie(pvids == std::vector<int>{203, 202, 201, 200},
                "bottom face vertex_ids reversal",
                "pvids did not reverse for bottom face (outward = -y)");
    std::cout << "  PASS  ReorderFaceVerticesCcw on bottom face: input "
                 "flipped for outward normal -y" << std::endl;
}

// ===========================================================================
// Test 11: integration smoke — every quad tag is accepted by the assembler
// ===========================================================================
//
// This test mirrors test_sentinel_tagged_face_elements_drive_assembler_correctly
// from the Python prototype: it confirms that every tag the classifier might
// emit is one that QuadFaceMortarAssembler / TriFaceMortarAssembler can
// dispatch via their internal boundary_tag tables.
//
// We do this by constructing a dummy QuadFacePairMatch / TriFacePairMatch
// and calling AssemblePairConforming on a single-element pair with each
// tag. The assembler should not throw. We don't check numerical results
// here — that's covered by test_face_mortar_assembler_3d.cpp.
void test_assembler_accepts_all_tags()
{
    std::cout << "Test 11: integration smoke — assemblers accept all tags"
              << std::endl;

    using mortar_pbc::QuadFaceElement;
    using mortar_pbc::QuadFaceMortarAssembler;
    using mortar_pbc::QuadFacePairMatch;
    using mortar_pbc::TriFaceElement;
    using mortar_pbc::TriFaceMortarAssembler;
    using mortar_pbc::TriFacePairMatch;

    // The full set of quad tags the classifier emits. This must agree
    // with QuadFaceMortarAssembler's internal dispatch table.
    std::vector<std::string> quad_tags = {
        "none",
        "edge-xi-low", "edge-xi-high",
        "edge-eta-low", "edge-eta-high",
        "corner-LL", "corner-LR", "corner-UR", "corner-UL",
    };
    QuadFaceMortarAssembler quad_asm;
    for (const std::string& tag : quad_tags)
    {
        // Build a single conforming nonmortar/mortar pair on the y=0 / y=1
        // faces. Geometry: unit-square quad in (x, z), y-perp.
        QuadFaceElement nm;
        nm.coords.SetSize(4, 3);
        double nm_data[4][3] = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {1.0, 0.0, 1.0}, {0.0, 0.0, 1.0},
        };
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j) { nm.coords(i, j) = nm_data[i][j]; }
        }
        nm.gtdofs = {0, 1, 2, 3};
        nm.parametric_axes = {"x", "z"};
        nm.perpendicular_axis = "y";
        nm.boundary_tag = tag;

        QuadFaceElement m;
        m.coords.SetSize(4, 3);
        double m_data[4][3] = {
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0},
            {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0},
        };
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j) { m.coords(i, j) = m_data[i][j]; }
        }
        m.gtdofs = {10, 11, 12, 13};
        m.parametric_axes = {"x", "z"};
        m.perpendicular_axis = "y";
        m.boundary_tag = "none";  // mortar side never has a Wohlmuth tag

        QuadFacePairMatch match;
        match.nonmortar_idx = 0;
        match.mortar_idx = 0;
        match.mortar_node_perm = {0, 1, 2, 3};

        // Should not throw.
        try
        {
            (void)quad_asm.AssemblePairConforming(
                {nm}, {m}, {match}, "nonmortar", "mortar");
        }
        catch (const std::exception& e)
        {
            std::cerr << "  FAIL  quad tag '" << tag
                      << "': assembler threw: " << e.what() << std::endl;
            std::exit(1);
        }
    }

    // Tri tags
    std::vector<std::string> tri_tags = {
        "none", "v0", "v1", "v2", "v0-v1", "v0-v2", "v1-v2",
    };
    TriFaceMortarAssembler tri_asm;
    for (const std::string& tag : tri_tags)
    {
        TriFaceElement nm;
        nm.coords.SetSize(3, 3);
        double nm_data[3][3] = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
        };
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j) { nm.coords(i, j) = nm_data[i][j]; }
        }
        nm.gtdofs = {0, 1, 2};
        nm.parametric_axes = {"x", "z"};
        nm.perpendicular_axis = "y";
        nm.boundary_tag = tag;

        TriFaceElement m;
        m.coords.SetSize(3, 3);
        double m_data[3][3] = {
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0},
        };
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j) { m.coords(i, j) = m_data[i][j]; }
        }
        m.gtdofs = {10, 11, 12};
        m.parametric_axes = {"x", "z"};
        m.perpendicular_axis = "y";
        m.boundary_tag = "none";

        TriFacePairMatch match;
        match.nonmortar_idx = 0;
        match.mortar_idx = 0;
        match.mortar_node_perm = {0, 1, 2};

        try
        {
            (void)tri_asm.AssemblePairConforming(
                {nm}, {m}, {match}, "nonmortar", "mortar");
        }
        catch (const std::exception& e)
        {
            std::cerr << "  FAIL  tri tag '" << tag
                      << "': assembler threw: " << e.what() << std::endl;
            std::exit(1);
        }
    }

    std::cout << "  PASS  every quad tag (" << quad_tags.size() << ") and tri "
                 "tag (" << tri_tags.size()
              << ") is accepted by its assembler" << std::endl;
}

}  // anonymous namespace

int main(int /*argc*/, char** /*argv*/)
{
    std::cout << "Running boundary helpers (3D) unit tests" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    test_axis_extreme_to_label();
    test_face_pairs_mortar_labels();
    test_face_axes();
    test_param_axis_from_attrs();
    test_edge_label_symmetric();
    test_face_bounding_edge_labels();
    test_classify_quad_boundary_tag();
    test_classify_tri_boundary_tag();
    test_reorder_top_face_quad();
    test_reorder_bottom_face_quad();
    test_assembler_accepts_all_tags();
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "All unit tests passed." << std::endl;
    return 0;
}
