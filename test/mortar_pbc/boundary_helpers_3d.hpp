// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of the pure (no-MFEM-mesh, no-MPI) helpers from
// Python `mortar_pbc/boundary_3d.py`. These functions are the
// topology-only logic: face-label conventions, edge/corner naming,
// boundary-tag dispatch for sentinel-flagged face elements, and
// face-vertex CCW reordering.
//
// The full BoundaryClassifier3D class (which wraps an MFEM ParMesh,
// performs the runtime attribute discovery, and gathers boundary
// records via MPI) is delivered separately in
// boundary_classifier_3d.{hpp,cpp} (Phase 4.1.A Batch B). It calls the
// helpers here for its internal logic.
//
// Why split this off
// ------------------
// In the Python prototype these helpers sit on the classifier class
// but most are exercised in tests via __new__-bypass tricks because
// they don't actually need a mesh. C++ doesn't allow that pattern
// cleanly, so the helpers move to free functions in the mortar_pbc
// namespace, taking the runtime-discovered `face_label_by_attr`
// mapping as an explicit argument when needed. This also clarifies
// the dependency: helpers depend on the lookup table, classifier
// owns the table.

#pragma once

#include "types_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mortar_pbc {

//==============================================================================
// Module-level conventions (locked here, mirror Python boundary_3d.py)
//==============================================================================

/**
 * @brief Canonical (axis, extreme) -> face-label mapping.
 *
 * @details The 6 box faces of a 3D RVE are named per:
 *   - "bottom" : at y_min, perp = y
 *   - "top"    : at y_max, perp = y
 *   - "front"  : at z_min, perp = z
 *   - "back"   : at z_max, perp = z
 *   - "left"   : at x_min, perp = x
 *   - "right"  : at x_max, perp = x
 *
 * @param axis     One of {"x", "y", "z"}.
 * @param extreme  One of {"min", "max"}.
 * @return The canonical label string. Aborts via MFEM_ABORT if
 *         (axis, extreme) is not a valid combination.
 */
const std::string& AxisExtremeToLabel(const std::string& axis,
                                      const std::string& extreme);

/**
 * @brief Returns the 3 mortar/nonmortar face-label pairs.
 *
 * @details Convention (locked here): mortar = top, right, back (the
 * "high" side along each axis); nonmortar = bottom, left, front (the
 * "low" side). Each pair is (mortar_label, nonmortar_label).
 *
 * @return A const reference to the 3-element pair list.
 */
const std::array<std::pair<std::string, std::string>, 3>& FacePairs();

/**
 * @brief Returns the set of mortar face labels {"top", "right", "back"}.
 */
const std::set<std::string>& MortarLabels();

/**
 * @brief For a given face label, return its perpendicular axis and its
 *        two parametric axes.
 *
 * @param face_label  One of {"bottom", "top", "front", "back", "left", "right"}.
 * @return A pair `(perp_axis, {param_axis_a, param_axis_b})` where each
 *         axis is "x", "y", or "z". Aborts via MFEM_ABORT if the label
 *         is unknown.
 *
 * @details The (param_axis_a, param_axis_b) ordering is chosen so that
 * the right-hand-rule cross product `a × b = +perp` for the
 * mortar-side faces (top/right/back). For the nonmortar-side faces
 * (bottom/left/front) this convention means the resulting (a, b)
 * traversal is CCW when viewed from `+perp`, which is the OPPOSITE of
 * outward-normal CCW. ReorderFaceVerticesCcw flips orientation
 * accordingly.
 */
std::pair<std::string, std::array<std::string, 2>>
FaceAxes(const std::string& face_label);

//==============================================================================
// Free helper functions
//==============================================================================

/**
 * @brief Build an edge label from the parametric axis and the two
 *        adjacent face attributes.
 *
 * @param parametric_axis    One of "x", "y", "z".
 * @param attrs              Two adjacent face attributes (any order).
 * @param face_label_by_attr Runtime-discovered mapping (built by
 *                           BoundaryClassifier3D from the actual mesh).
 * @return Label of the form `"{axis}-{face1_label}-{face2_label}"`
 *         where face1 < face2 by attribute integer.
 *
 * @details The two attributes are sorted by integer value, then mapped
 * to face labels via `face_label_by_attr`. This makes the labelling
 * symmetric in the input attribute order — `EdgeLabel("x", {a, b}, m)
 * == EdgeLabel("x", {b, a}, m)`.
 *
 * Aborts via MFEM_VERIFY if either attribute is missing from the map.
 */
std::string EdgeLabel(const std::string& parametric_axis,
                      const std::pair<int, int>& attrs,
                      const std::map<int, std::string>& face_label_by_attr);

/**
 * @brief Derive the parametric axis of the edge shared by two adjacent
 *        faces.
 *
 * @param attrs              Two adjacent face attributes.
 * @param face_label_by_attr Runtime-discovered mapping.
 * @return The unique axis perpendicular to both face normals (i.e. the
 *         axis along which the shared edge runs).
 *
 * @details Aborts via MFEM_VERIFY if the two faces share the same
 * perpendicular axis (i.e. they're a mortar/nonmortar pair, not
 * adjacent — they don't share an edge).
 */
std::string ParamAxisFromAttrs(
    const std::pair<int, int>& attrs,
    const std::map<int, std::string>& face_label_by_attr);

/**
 * @brief Return the 4 edge labels bounding the face with given attribute.
 *
 * @param face_attr          Attribute integer of the face.
 * @param face_label_by_attr Runtime-discovered mapping. Must contain
 *                           all 6 face attributes.
 * @return Vector of 4 edge labels.
 *
 * @details Each box face has exactly 4 bounding edges; each is shared
 * with one adjacent face (those with a different perpendicular axis).
 */
std::vector<std::string> FaceBoundingEdgeLabels(
    int face_attr,
    const std::map<int, std::string>& face_label_by_attr);

/**
 * @brief Map sentinel pattern of a quad-4 face element to a Wohlmuth
 *        boundary tag.
 *
 * @param sentinels  4-element array of per-vertex sentinel values.
 *                   A negative value (e.g. `kGtdofCornerSentinel` = -1
 *                   or `kGtdofEdgeSentinel` = -2) marks the vertex as
 *                   sitting on a face-boundary feature; a non-negative
 *                   value is a regular face-interior DOF.
 *
 * @return One of: "none", "edge-xi-low", "edge-xi-high",
 *         "edge-eta-low", "edge-eta-high", "corner-LL", "corner-LR",
 *         "corner-UR", "corner-UL". The tag selects which rows of the
 *         dual basis to drop in MQuad4DualModified.
 *
 * @details Quad-4 local-node convention (CCW from outward normal):
 * @code
 *     node 3 -- node 2     eta=+1
 *       |          |
 *     node 0 -- node 1     eta=-1
 *     xi=-1     xi=+1
 * @endcode
 *
 * Sentinel patterns and their geometric meanings are documented in
 * MORTAR_PBC_ARCHITECTURE.md §11.7 / §4.4.2 (Wohlmuth modification).
 *
 * @note This function is pure — no lookup table needed.
 */
std::string ClassifyQuadBoundaryTag(const std::array<int, 4>& sentinels);

/**
 * @brief Map sentinel pattern of a tri-3 face element to a Wohlmuth
 *        boundary tag.
 *
 * @param sentinels  3-element array of per-vertex sentinel values.
 * @return One of: "none", "v0", "v1", "v2", "v0-v1", "v0-v2", "v1-v2",
 *         "v0-v1-v2".
 *
 * @note This function is pure — no lookup table needed.
 */
std::string ClassifyTriBoundaryTag(const std::array<int, 3>& sentinels);

/**
 * @brief Reorder a face element's vertices so they are CCW viewed from
 *        the OUTWARD normal of the face.
 *
 * @param[in,out] coords      `(n, 3)` matrix of vertex coordinates.
 *                            Reordered in place if reversal is needed.
 * @param[in,out] vertex_ids  Vector of `n` vertex IDs (parent or
 *                            global). Reordered in place to track
 *                            `coords`.
 * @param         face_label  One of {"bottom","top","front","back","left","right"}.
 *
 * @details Outward normal direction:
 *   - face = "top"     -> +y
 *   - face = "bottom"  -> -y
 *   - face = "right"   -> +x
 *   - face = "left"    -> -x
 *   - face = "back"    -> +z
 *   - face = "front"   -> -z
 *
 * Algorithm: project to 2D in the face's parametric plane, compute the
 * signed shoelace area; reverse the vertex list if the sign is wrong
 * for the desired outward normal.
 *
 * @note This function is pure — no lookup table needed beyond the
 * canonical FaceAxes() table.
 */
void ReorderFaceVerticesCcw(mfem::DenseMatrix& coords,
                            std::vector<int>& vertex_ids,
                            const std::string& face_label);

}  // namespace mortar_pbc
