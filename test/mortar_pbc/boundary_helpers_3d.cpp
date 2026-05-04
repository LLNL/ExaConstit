// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of pure helpers for boundary
// classification, ported from Python `mortar_pbc/boundary_3d.py`.

#include "boundary_helpers_3d.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mortar_pbc {

namespace {

//==============================================================================
// Module-level lookup tables (file-scope, not exported)
//==============================================================================

// Canonical (axis, extreme) -> face-label mapping.
const std::map<std::pair<std::string, std::string>, std::string>&
GetAxisExtremeToLabel()
{
    static const std::map<std::pair<std::string, std::string>, std::string> kTable = {
        {{"y", "min"}, "bottom"},
        {{"y", "max"}, "top"},
        {{"z", "min"}, "front"},
        {{"z", "max"}, "back"},
        {{"x", "min"}, "left"},
        {{"x", "max"}, "right"},
    };
    return kTable;
}

// 3 mortar/nonmortar pairs: (mortar, nonmortar) per axis.
const std::array<std::pair<std::string, std::string>, 3>& GetFacePairs()
{
    static const std::array<std::pair<std::string, std::string>, 3> kPairs = {{
        {"top",   "bottom"},   // y-pair
        {"right", "left"},     // x-pair
        {"back",  "front"},    // z-pair
    }};
    return kPairs;
}

const std::set<std::string>& GetMortarLabels()
{
    static const std::set<std::string> kLabels = {"top", "right", "back"};
    return kLabels;
}

// Each face's perpendicular axis and parametric axes.
//   "bottom" / "top"   : perp = y, params = (x, z)
//   "front"  / "back"  : perp = z, params = (x, y)
//   "left"   / "right" : perp = x, params = (y, z)
const std::map<std::string, std::pair<std::string, std::array<std::string, 2>>>&
GetFaceAxes()
{
    static const std::map<std::string,
                          std::pair<std::string, std::array<std::string, 2>>>
        kTable = {
            {"bottom", {"y", {"x", "z"}}},
            {"top",    {"y", {"x", "z"}}},
            {"front",  {"z", {"x", "y"}}},
            {"back",   {"z", {"x", "y"}}},
            {"left",   {"x", {"y", "z"}}},
            {"right",  {"x", {"y", "z"}}},
        };
    return kTable;
}

// "x" -> 0, "y" -> 1, "z" -> 2. Aborts on unknown axis.
int AxisToIndex(const std::string& axis)
{
    if (axis == "x") { return 0; }
    if (axis == "y") { return 1; }
    if (axis == "z") { return 2; }
    MFEM_ABORT("AxisToIndex: unknown axis '" << axis << "'");
    return -1;  // unreachable
}

}  // anonymous namespace

//==============================================================================
// Public accessors for module-level conventions
//==============================================================================

const std::string& AxisExtremeToLabel(const std::string& axis,
                                      const std::string& extreme)
{
    const auto& table = GetAxisExtremeToLabel();
    auto it = table.find({axis, extreme});
    MFEM_VERIFY(it != table.end(),
                "AxisExtremeToLabel: unknown (axis, extreme) = ('"
                << axis << "', '" << extreme << "')");
    return it->second;
}

const std::array<std::pair<std::string, std::string>, 3>& FacePairs()
{
    return GetFacePairs();
}

const std::set<std::string>& MortarLabels()
{
    return GetMortarLabels();
}

std::pair<std::string, std::array<std::string, 2>>
FaceAxes(const std::string& face_label)
{
    const auto& table = GetFaceAxes();
    auto it = table.find(face_label);
    MFEM_VERIFY(it != table.end(),
                "FaceAxes: unknown face label '" << face_label << "'");
    return it->second;
}

//==============================================================================
// EdgeLabel — composes "{axis}-{face1}-{face2}" with attrs sorted
//==============================================================================

std::string EdgeLabel(const std::string& parametric_axis,
                      const std::pair<int, int>& attrs,
                      const std::map<int, std::string>& face_label_by_attr)
{
    int f1 = std::min(attrs.first, attrs.second);
    int f2 = std::max(attrs.first, attrs.second);
    auto it1 = face_label_by_attr.find(f1);
    auto it2 = face_label_by_attr.find(f2);
    MFEM_VERIFY(it1 != face_label_by_attr.end(),
                "EdgeLabel: attr " << f1 << " not in face_label_by_attr map");
    MFEM_VERIFY(it2 != face_label_by_attr.end(),
                "EdgeLabel: attr " << f2 << " not in face_label_by_attr map");
    std::ostringstream oss;
    oss << parametric_axis << "-" << it1->second << "-" << it2->second;
    return oss.str();
}

//==============================================================================
// ParamAxisFromAttrs — the unique axis perpendicular to both face normals
//==============================================================================

std::string ParamAxisFromAttrs(
    const std::pair<int, int>& attrs,
    const std::map<int, std::string>& face_label_by_attr)
{
    auto it1 = face_label_by_attr.find(attrs.first);
    auto it2 = face_label_by_attr.find(attrs.second);
    MFEM_VERIFY(it1 != face_label_by_attr.end(),
                "ParamAxisFromAttrs: attr " << attrs.first
                << " not in face_label_by_attr map");
    MFEM_VERIFY(it2 != face_label_by_attr.end(),
                "ParamAxisFromAttrs: attr " << attrs.second
                << " not in face_label_by_attr map");
    const std::string& f1_name = it1->second;
    const std::string& f2_name = it2->second;
    const auto& axes_table = GetFaceAxes();
    const std::string& perp1 = axes_table.at(f1_name).first;
    const std::string& perp2 = axes_table.at(f2_name).first;
    MFEM_VERIFY(perp1 != perp2,
                "ParamAxisFromAttrs: faces '" << f1_name << "' and '"
                << f2_name << "' share the same perp axis '" << perp1
                << "'; they're a mortar/nonmortar pair, not adjacent — "
                "they don't share an edge.");
    for (const std::string& ax : {std::string("x"), std::string("y"),
                                  std::string("z")})
    {
        if (ax != perp1 && ax != perp2) { return ax; }
    }
    MFEM_ABORT("ParamAxisFromAttrs: unreachable");
    return {};
}

//==============================================================================
// FaceBoundingEdgeLabels — the 4 edges bounding the given face
//==============================================================================

std::vector<std::string> FaceBoundingEdgeLabels(
    int face_attr,
    const std::map<int, std::string>& face_label_by_attr)
{
    auto it = face_label_by_attr.find(face_attr);
    MFEM_VERIFY(it != face_label_by_attr.end(),
                "FaceBoundingEdgeLabels: attr " << face_attr
                << " not in face_label_by_attr map");
    const std::string& face_label = it->second;
    const auto& axes_table = GetFaceAxes();
    const std::string& perp_face = axes_table.at(face_label).first;

    // Adjacent attributes: those with a different perpendicular axis.
    // Iterate in sorted attribute order for determinism.
    std::vector<int> adjacent;
    for (const auto& kv : face_label_by_attr)
    {
        int other_attr = kv.first;
        if (other_attr == face_attr) { continue; }
        const std::string& other_label = kv.second;
        const std::string& perp_other = axes_table.at(other_label).first;
        if (perp_other != perp_face) { adjacent.push_back(other_attr); }
    }

    std::vector<std::string> out;
    out.reserve(adjacent.size());
    for (int other_attr : adjacent)
    {
        const std::string& other_label = face_label_by_attr.at(other_attr);
        const std::string& perp_other = axes_table.at(other_label).first;
        // Parametric axis of the shared edge: perpendicular to both face
        // normals.
        for (const std::string& ax : {std::string("x"), std::string("y"),
                                      std::string("z")})
        {
            if (ax != perp_face && ax != perp_other)
            {
                out.push_back(EdgeLabel(ax, {face_attr, other_attr},
                                        face_label_by_attr));
                break;
            }
        }
    }
    return out;
}

//==============================================================================
// ClassifyQuadBoundaryTag — sentinel pattern -> Wohlmuth tag
//==============================================================================

std::string ClassifyQuadBoundaryTag(const std::array<int, 4>& sentinels)
{
    // Collect the local-node positions of any sentinel-marked vertices
    // (negative gtdof values).
    std::vector<int> sentinel_locs;
    sentinel_locs.reserve(4);
    for (int i = 0; i < 4; ++i)
    {
        if (sentinels[i] < 0) { sentinel_locs.push_back(i); }
    }
    const int n = static_cast<int>(sentinel_locs.size());

    if (n == 0) { return "none"; }

    if (n == 1)
    {
        // 1 sentinel = corner DOF only at the named local node.
        static const std::array<std::string, 4> kTags = {
            "corner-LL", "corner-LR", "corner-UR", "corner-UL"};
        return kTags[sentinel_locs[0]];
    }

    if (n == 2)
    {
        std::set<int> s(sentinel_locs.begin(), sentinel_locs.end());
        if (s == std::set<int>{0, 3}) { return "edge-xi-low"; }
        if (s == std::set<int>{1, 2}) { return "edge-xi-high"; }
        if (s == std::set<int>{0, 1}) { return "edge-eta-low"; }
        if (s == std::set<int>{2, 3}) { return "edge-eta-high"; }
        // Diagonal-pair sentinels ({0,2} or {1,3}): anomalous on
        // MakeCartesian3D meshes; fall through to "none" — the lumped-
        // positivity guard catches any actual integrity issue.
        return "none";
    }

    if (n == 3)
    {
        // The 4 cases name the kept node:
        //   kept node 0 -> sentinels {1, 2, 3} -> drops xi-high & eta-high
        //                  -> "corner-UR" (the kept node sits at LL)
        //   kept node 1 -> sentinels {0, 2, 3} -> "corner-UL"
        //   kept node 2 -> sentinels {0, 1, 3} -> "corner-LL"
        //   kept node 3 -> sentinels {0, 1, 2} -> "corner-LR"
        std::set<int> ss(sentinel_locs.begin(), sentinel_locs.end());
        int kept = -1;
        for (int i = 0; i < 4; ++i)
        {
            if (ss.find(i) == ss.end()) { kept = i; break; }
        }
        MFEM_ASSERT(kept >= 0, "ClassifyQuadBoundaryTag: kept node not found");
        static const std::array<std::string, 4> kTags = {
            "corner-UR", "corner-UL", "corner-LL", "corner-LR"};
        return kTags[kept];
    }

    // n == 4: every row dropped, element contributes nothing — "none"
    // is harmless.
    return "none";
}

//==============================================================================
// ClassifyTriBoundaryTag — sentinel pattern -> Wohlmuth tag
//==============================================================================

std::string ClassifyTriBoundaryTag(const std::array<int, 3>& sentinels)
{
    std::vector<int> sentinel_locs;
    sentinel_locs.reserve(3);
    for (int i = 0; i < 3; ++i)
    {
        if (sentinels[i] < 0) { sentinel_locs.push_back(i); }
    }
    if (sentinel_locs.empty()) { return "none"; }

    // Build "v{i}-v{j}-v{k}" with i < j < k.
    std::sort(sentinel_locs.begin(), sentinel_locs.end());
    std::ostringstream oss;
    oss << "v" << sentinel_locs[0];
    for (std::size_t k = 1; k < sentinel_locs.size(); ++k)
    {
        oss << "-v" << sentinel_locs[k];
    }
    return oss.str();
}

//==============================================================================
// ReorderFaceVerticesCcw — flip CW -> CCW from outward normal
//==============================================================================

void ReorderFaceVerticesCcw(mfem::DenseMatrix& coords,
                            std::vector<int>& vertex_ids,
                            const std::string& face_label)
{
    const int n = coords.NumRows();
    MFEM_VERIFY(coords.NumCols() == 3,
                "ReorderFaceVerticesCcw: coords must be (n, 3)");
    MFEM_VERIFY(static_cast<int>(vertex_ids.size()) == n,
                "ReorderFaceVerticesCcw: vertex_ids size (" << vertex_ids.size()
                << ") does not match coords rows (" << n << ")");

    // The two parametric axes for this face.
    const auto axes = FaceAxes(face_label);
    const int a_idx = AxisToIndex(axes.second[0]);
    const int b_idx = AxisToIndex(axes.second[1]);

    // Outward-normal sign: positive (along +perp) for top/right/back;
    // negative (along -perp) for bottom/left/front.
    const auto& mortar_labels = GetMortarLabels();
    const bool outward_pos = (mortar_labels.find(face_label) != mortar_labels.end());

    // Shoelace area in the (a, b) plane.
    double signed_area = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double a1 = coords(i, a_idx);
        const double b1 = coords(i, b_idx);
        const int ip1 = (i + 1) % n;
        const double a2 = coords(ip1, a_idx);
        const double b2 = coords(ip1, b_idx);
        signed_area += (a1 * b2 - a2 * b1);
    }
    signed_area *= 0.5;

    // The (a, b) ordering in FaceAxes is chosen so that
    // a × b = +perp. So `signed_area > 0` corresponds to CCW viewed
    // from +perp. We want CCW viewed from the OUTWARD normal:
    //   - outward = +perp (mortar side) -> want signed_area > 0
    //   - outward = -perp (nonmortar side) -> want signed_area < 0
    const bool want_positive = outward_pos;
    const bool need_reverse =
        (want_positive && signed_area < 0.0) ||
        (!want_positive && signed_area > 0.0);

    if (need_reverse)
    {
        // Reverse vertex_ids and coords rows in place.
        std::reverse(vertex_ids.begin(), vertex_ids.end());

        mfem::DenseMatrix tmp(n, 3);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < 3; ++j) { tmp(i, j) = coords(n - 1 - i, j); }
        }
        coords = tmp;
    }
}

}  // namespace mortar_pbc
