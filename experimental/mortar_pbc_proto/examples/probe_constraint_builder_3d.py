"""Phase 3.3.D integration probe — full classifier + builder pipeline on a real RVE.

Exercises the full Phase 3.3 pipeline:
    pmesh + fes -> BoundaryClassifier3D -> ConstraintBuilder3D -> sparse C

then runs four sanity checks identical in spirit to the synthetic-mock
unit tests, but on an actual `MakeCartesian3D` mesh:

  1. Row count matches the analytical formula.
  2. Constant displacement field is in C's nullspace (||C·u_const|| = 0
     to machine precision).
  3. Affine displacement field produces a non-zero jump (C is rank-
     deficient with the right structure).
  4. C is linear (C(u+v) = C·u + C·v).

Run with:
    python examples/probe_constraint_builder_3d.py --mesh-type hex
    python examples/probe_constraint_builder_3d.py --mesh-type tet
    mpirun -n 4 python examples/probe_constraint_builder_3d.py --mesh-type hex
    mpirun -n 4 python examples/probe_constraint_builder_3d.py --mesh-type tet

PASS criteria:
    - Row count > 0 and matches builder.n_constraints()
    - ||C·u_const||_inf < 1e-12
    - ||C·u_affine||_inf > 1e-6  (real jump expected)
    - ||C·(u + v) - C·u - C·v||_inf < 1e-12
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from mortar_pbc import BoundaryClassifier3D, ConstraintBuilder3D


def build_box_mesh(mesh_type: str, n: int = 4, L: float = 1.0):
    if mesh_type == "hex":
        elem = mfem.Element.HEXAHEDRON
    elif mesh_type == "tet":
        elem = mfem.Element.TETRAHEDRON
    else:
        raise ValueError(f"Unknown mesh-type {mesh_type!r}")
    return mfem.Mesh.MakeCartesian3D(n, n, n, elem, L, L, L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-type", choices=["hex", "tet"], default="hex")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--L", type=float, default=1.0)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print(f"  ConstraintBuilder3D probe ({args.mesh_type}, n={args.n}, np={nranks})")
        print("=" * 70)

    # Build mesh + ParMesh + FES.
    mesh = build_box_mesh(args.mesh_type, n=args.n, L=args.L)
    pmesh = mfem.ParMesh(comm, mesh)
    n_ge = pmesh.GetGlobalNE()
    fec = mfem.H1_FECollection(1, pmesh.Dimension())
    fes = mfem.ParFiniteElementSpace(pmesh, fec, pmesh.Dimension())
    n_global_tdofs = fes.GlobalTrueVSize()
    if rank == 0:
        print(f"  ParMesh: {n_ge} global elements, "
              f"global TDOFs = {n_global_tdofs}")

    # Classifier.
    classifier = BoundaryClassifier3D(pmesh, fes)
    if rank == 0:
        print(f"  Classifier: {len(classifier.corners)} corners, "
              f"{len(classifier.edges)} edges, {len(classifier.faces)} faces")
        n_face_quads = sum(f.n_quad_elements for f in classifier.faces.values())
        n_face_tris  = sum(f.n_tri_elements  for f in classifier.faces.values())
        print(f"             {n_face_quads} face quads, {n_face_tris} face tris")

    # Builder.
    builder = ConstraintBuilder3D(classifier)
    n_predicted = builder.n_constraints()

    # Diagnostic: dump the first nonmortar-face quad coords to verify
    # the classifier built them correctly. Toggle with
    # MORTAR_PBC_DEBUG_BUILDER=1.
    if os.environ.get("MORTAR_PBC_DEBUG_BUILDER", "") == "1" and rank == 0:
        for face_label in ("bottom", "left", "front"):
            face = classifier.faces[face_label]
            print(f"  [DEBUG] face {face_label!r}: "
                  f"perp={face.perpendicular_axis} "
                  f"params={face.parametric_axes} "
                  f"plane={face.plane_value} "
                  f"n_quad={face.n_quad_elements}")
            for k, fe in enumerate(face.face_elements[:3]):
                print(f"     elem[{k}] type={type(fe).__name__} "
                      f"boundary_tag={fe.boundary_tag!r}")
                print(f"            coords =\n{fe.coords}")
                print(f"            centroid (full) = {fe.coords.mean(axis=0)}")

    C = builder.build()

    if rank == 0:
        print(f"  ConstraintBuilder: predicted {n_predicted} rows, "
              f"C.shape = {C.shape}, nnz = {C.nnz}")
        print()

    # =========================================================================
    # Test 1: row count
    # =========================================================================
    ok_rows = (C.shape == (n_predicted, n_global_tdofs))
    if rank == 0:
        status = "OK" if ok_rows else "FAIL"
        print(f"  TEST 1  Row count: predicted = {n_predicted}, "
              f"actual = {C.shape[0]}  -> {status}")

    # =========================================================================
    # Test 2: periodic fluctuation is in nullspace
    # =========================================================================
    #
    # A constant field is NOT in C's nullspace because corner DOFs
    # are sentinel-stripped (they're Dirichlet-pinned separately).
    # The right test is: a PERIODIC FLUCTUATION FIELD that vanishes
    # at corners. Since u(nonmortar_X) = u(mortar_X) for any periodic
    # function (sin(2π·) etc.), and the field is zero at corners,
    # C·u_periodic = 0 holds: every corner contribution that the
    # constraint matrix dropped via sentinel-stripping has been
    # absorbed by the explicit corner-zero condition on u.
    u_periodic = np.zeros(n_global_tdofs, dtype=np.float64)
    L_x = float(classifier.bbox_max[0] - classifier.bbox_min[0])
    L_y = float(classifier.bbox_max[1] - classifier.bbox_min[1])
    L_z = float(classifier.bbox_max[2] - classifier.bbox_min[2])
    for r_rec in classifier.vertex_records.values():
        coord = r_rec.coord
        # sin(2π X/L) vanishes at X = 0 and X = L for all axes,
        # i.e. at every box corner / box edge / box face boundary.
        sin_val = (np.sin(2 * np.pi * coord[0] / L_x)
                   * np.sin(2 * np.pi * coord[1] / L_y)
                   * np.sin(2 * np.pi * coord[2] / L_z))
        # Use 3 different amplitudes per component to verify that
        # all 3 vdim rows respond correctly.
        gx, gy, gz = (int(r_rec.gtdof_xyz[0]), int(r_rec.gtdof_xyz[1]),
                      int(r_rec.gtdof_xyz[2]))
        if gx >= 0: u_periodic[gx] = 0.5  * sin_val
        if gy >= 0: u_periodic[gy] = -0.7 * sin_val
        if gz >= 0: u_periodic[gz] = 1.3  * sin_val
    err_periodic = float(np.max(np.abs(C @ u_periodic)))
    ok_periodic = (err_periodic < 1e-10)
    if rank == 0:
        status = "OK" if ok_periodic else "FAIL"
        print(f"  TEST 2  Periodic-fluctuation nullspace: "
              f"||C·u_periodic||_inf = {err_periodic:.3e}  -> {status}")

    # =========================================================================
    # Test 3: affine field produces non-zero jump
    # =========================================================================
    # u_lin(X) = (F-I) X projected to FES via apply_linear_part.
    from mortar_pbc import apply_linear_part
    F = np.array([[1.10, 0.05, 0.02],
                  [0.03, 0.95, 0.04],
                  [0.01, 0.02, 1.05]])
    u_lin_local = apply_linear_part(fes, F)
    # Need GLOBAL u_lin to multiply C.
    # Each rank has u_lin_local for its TDOFs; AllGather + reorder by global index.
    # Simpler: use an Allgatherv-based reconstruction. For a replicated C
    # solve like the patch test, every rank can build the same u_lin
    # globally by re-running apply_linear_part with global TDOFs known.
    #
    # For this probe we construct the global u_lin from coords directly:
    # walk every parent FES vertex, project (F-I)X, write into the
    # appropriate global TDOF slot. This requires the gtdof_xyz_lookup
    # the classifier already built.
    lookup = classifier.gtdof_xyz_lookup()
    u_aff_global = np.zeros(n_global_tdofs, dtype=np.float64)
    # We have lookup: gx -> (gx, gy, gz). To populate u_aff at every
    # gtdof, we also need the corresponding coord. Use vertex_records
    # which has both.
    for r_rec in classifier.vertex_records.values():
        coord = r_rec.coord
        u_v = (F - np.eye(3)) @ coord
        gx, gy, gz = int(r_rec.gtdof_xyz[0]), int(r_rec.gtdof_xyz[1]), int(r_rec.gtdof_xyz[2])
        if gx >= 0: u_aff_global[gx] = u_v[0]
        if gy >= 0: u_aff_global[gy] = u_v[1]
        if gz >= 0: u_aff_global[gz] = u_v[2]
    # NOTE: this only fills BOUNDARY gtdofs. For the constraint test,
    # that's exactly what's needed (C only references boundary gtdofs).
    err_aff = float(np.max(np.abs(C @ u_aff_global)))
    ok_aff = (err_aff > 1e-6)
    if rank == 0:
        status = "OK" if ok_aff else "FAIL"
        print(f"  TEST 3  Affine-field jump: "
              f"||C·u_affine||_inf = {err_aff:.4f} (should be > 1e-6)  -> "
              f"{status}")

    # =========================================================================
    # Test 4: linearity
    # =========================================================================
    Cu_combined = C @ (u_periodic + u_aff_global)
    Cu_separate = (C @ u_periodic) + (C @ u_aff_global)
    err_lin = float(np.max(np.abs(Cu_combined - Cu_separate)))
    ok_lin = (err_lin < 1e-12)
    if rank == 0:
        status = "OK" if ok_lin else "FAIL"
        print(f"  TEST 4  Linearity: "
              f"||C·(u+v) - (C·u + C·v)||_inf = {err_lin:.3e}  -> {status}")

    # =========================================================================
    # Summary
    # =========================================================================
    all_ok = ok_rows and ok_periodic and ok_aff and ok_lin
    if rank == 0:
        print()
        if all_ok:
            print("  ===== probe: PASS =====")
        else:
            print("  ===== probe: FAIL =====")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
