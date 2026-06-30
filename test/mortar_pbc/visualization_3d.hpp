// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of `mortar_pbc/visualization.py` (single-step
// path only). Writes a two-cycle ParaView `.pvd` collection:
//
//   * cycle 0 (time = 0.0): undeformed reference configuration with
//     all displacement fields zero.
//   * cycle 1 (time = 1.0): deformed configuration — mesh nodes
//     warped by `u_total` so ParaView shows the actual deformed RVE
//     without any "Warp by Vector" filter.
//
// Open `<name>.pvd` in ParaView and use the time slider.
//
// Scope (deliberate)
// ------------------
// The Python provided BOTH a single-step convenience function and a
// stateful `PbcVisualizationWriter` class for multi-step runs. Only
// the single-step path is ported here because the Phase 4.1.A
// patch-test driver is a one-shot solve. The multi-step class is a
// straightforward extension (snapshot reference nodes once in the
// ctor, repeat reset+warp+save+reset on each `WriteStep`) and will
// be added in Phase 4.2 if/when a multi-step driver lands.
//
// Mesh-node-update mechanics (shared with Python)
// -----------------------------------------------
// MFEM meshes built from `MakeCartesian3D` store geometry as a
// vertex array, not a nodal grid function. `GetNodes()` returns
// nullptr in that case. To attach a nodal grid function, this helper
// calls `pmesh.SetCurvature(1, /*discontinuous=*/false, /*space_dim=*/-1,
// fes.GetOrdering())`. After that, `GetNodes()` returns a
// GridFunction whose values ARE the nodal coordinates and whose
// component ordering matches the displacement FE space.
//
// CRITICAL: the helper ALWAYS restores the mesh to its reference
// configuration before returning. Leaving the mesh deformed would
// corrupt subsequent `ApplyLinearPart` projections (which evaluate
// `(F-I) X` using the mesh's current nodal coordinates as `X`),
// `compute_volume_averaged_F` integrations, and any nonlinear
// integrator's `GetGradient` assembly. This is the SMALL-STRAIN /
// TOTAL-LAGRANGIAN convention: assembly/integration always happens
// on the reference mesh; the deformed mesh is purely a visualization
// artifact.

#pragma once

#include "mfem.hpp"

#include <string>

namespace mortar_pbc {

/**
 * @brief Write a two-cycle ParaView visualization of a mortar-PBC
 *        solution: undeformed reference (cycle 0) + deformed (cycle 1).
 *
 * @param[in,out] pmesh       Parallel mesh; will be temporarily warped
 *                            during the call but is RESTORED to the
 *                            reference configuration before return.
 * @param         fes         Vector H1 displacement FE space, vdim=3.
 *                            Mesh-node ordering is forced to match this
 *                            FES's ordering on first call.
 * @param         u_total     Total displacement TDOFs (u_lin + du).
 * @param         u_lin       Affine part of the displacement, projected
 *                            onto the FES.
 * @param         du          Fluctuation part (`u_tilde = u_total - u_lin`).
 * @param         output_dir  Directory to write the `<name>.pvd` and
 *                            per-rank `.vtu` files into. Created on
 *                            rank 0 if it doesn't exist.
 * @param         name        Collection name (default `"solution"`).
 *
 * @details The file `<output_dir>/<name>.pvd` and a sibling
 * `<output_dir>/<name>/` directory containing per-rank, per-cycle
 * `.vtu` files will be created. The collection contains four
 * registered fields: `u_total`, `u_lin`, `u_tilde`, and `material`
 * (a per-element constant grid function with the value of each
 * element's attribute, useful for color-coding heterogeneous RVEs).
 *
 * @par MPI scope
 * Collective on `pmesh.GetComm()`: a barrier after the rank-0
 * `MPI_File` directory creation, plus the `ParaViewDataCollection::Save`
 * collectives.
 *
 * @par Cross-validation against the Python prototype
 * The output is structurally identical to the Python's
 * `write_pbc_visualization` (same field names, same cycle layout,
 * same mesh-warp convention), so a side-by-side ParaView comparison
 * of the C++ and Python `.pvd` outputs on the same input is the
 * intended cross-validation path.
 */
void WriteVisualization(mfem::ParMesh& pmesh,
                        mfem::ParFiniteElementSpace& fes,
                        const mfem::Vector& u_total,
                        const mfem::Vector& u_lin,
                        const mfem::Vector& du,
                        const std::string& output_dir,
                        const std::string& name = "solution");

}  // namespace mortar_pbc
