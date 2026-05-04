// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-A — Axom smoke test.
//
// This file's only purpose is to verify that Axom is discoverable
// at build time and that the headers we depend on for the
// non-conforming face mortar work compile cleanly. It is
// intentionally a no-op: it constructs the types we need, exercises
// their basic APIs, and exits.
//
// If this file fails to compile, the rest of Phase 4.4 cannot
// proceed. Treat any failure here as a build-system issue (missing
// find_package, missing AXOM_DIR / axom_DIR hint, version skew) and
// fix it before moving on.
//
// References:
//   * Phase 4 plan §P4.4.6.10 — Phase 4.4 architectural plan.
//   * Axom docs: https://axom.readthedocs.io/

#include "axom/core.hpp"
#include "axom/primal.hpp"
#include "axom/spin.hpp"
#include "axom/slic.hpp"

#include <iostream>

namespace
{

using Point2D = axom::primal::Point<double, 2>;
using BBox2D  = axom::primal::BoundingBox<double, 2>;
using Poly2D  = axom::primal::Polygon<double, 2>;
using BVH2D   = axom::spin::BVH<2>;

/// Construct a unit-square BBox and a unit-square Polygon, query
/// containment, and clip the polygon against itself. Verifies that
/// the API surface we plan to use in Batches 4.4-B/C/D is present
/// and links.
void smoke_test_axom_primitives()
{
    // ----- primal::Point and primal::BoundingBox -----
    const Point2D pmin{0.0, 0.0};
    const Point2D pmax{1.0, 1.0};
    BBox2D bb(pmin, pmax);
    bb.addPoint(Point2D{0.5, 0.5});
    const bool contains_origin = bb.contains(pmin);
    if (!contains_origin)
    {
        // The BBox must contain its own min corner. Real Axom returns
        // true here; the stub also returns true. If a future Axom
        // version changes this, we'd want to know.
        std::cerr << "axom smoke: BBox::contains(min) returned false\n";
    }

    // ----- primal::Polygon -----
    Poly2D unit_square;
    unit_square.addVertex(Point2D{0.0, 0.0});
    unit_square.addVertex(Point2D{1.0, 0.0});
    unit_square.addVertex(Point2D{1.0, 1.0});
    unit_square.addVertex(Point2D{0.0, 1.0});

    // ----- primal::clip — self-clip should produce the same polygon -----
    Poly2D self_clip = axom::primal::clip(unit_square, unit_square);
    (void)self_clip;  // sandbox stub returns empty; real Axom returns the input

    // ----- spin::BVH<2> -----
    BVH2D bvh;
    BBox2D bboxes[1] = {bb};
    int status = bvh.initialize(bboxes, 1);
    (void)status;
}

}  // anonymous namespace

int main()
{
    // RAII Slic logger: initializes Slic on construction, finalizes on
    // destruction at end of main. Without this, Axom prints a runtime
    // warning that slic::initialize() was not called before SLIC was
    // exercised internally (e.g., by spin::BVH::findBoundingBoxes).
    axom::slic::SimpleLogger slic_logger;

    std::cout << "Axom smoke test (Phase 4.4 / Batch 4.4-A)\n";
    smoke_test_axom_primitives();
    std::cout << "  OK  axom primitives compile and link\n";
    return 0;
}
