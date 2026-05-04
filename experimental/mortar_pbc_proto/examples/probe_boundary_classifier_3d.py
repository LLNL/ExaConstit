"""Phase 3.3.B integration probe — instantiate BoundaryClassifier3D on
a small RVE mesh and print a summary.

This isn't a PASS/FAIL test (we don't check exact numerical values
against expectations); it's a smoke-test for the MFEM-touching pieces
of the classifier — ParSubMesh, parent vertex/element maps,
GetVertexDofs, GetGlobalTDofNumber. Run on macOS where pyMFEM is
available; sandbox testing covered the pure-Python helpers separately
(see tests/test_boundary_3d_helpers.py).

What we expect to see, validating the §10.4 invariants:
  * 8 corners with all 8 standard label strings.
  * 12 edges, 4 per parametric axis, mortar/nonmortar assignment correct
    (1 mortar + 3 nonmortars per direction).
  * 6 faces with element counts:
      - hex: 16 quads per face (for 4x4x4 mesh)
      - tet: 32 tris per face (each hex face split into 2 tris;
        actually MFEM splits each hex into 6 tets which gives ~32
        tris on each face for a 4x4x4 mesh — exact count depends on
        the splitting pattern).
  * No deadlocks at np > 1 (per §10.4); summary print order is
    rank-0-only.

Run with:
    python examples/probe_boundary_classifier_3d.py --mesh-type hex
    python examples/probe_boundary_classifier_3d.py --mesh-type tet
    mpirun -n 4 python examples/probe_boundary_classifier_3d.py --mesh-type hex
    mpirun -n 4 python examples/probe_boundary_classifier_3d.py --mesh-type tet
"""
from __future__ import annotations

import argparse
import os
import sys

# Make 'mortar_pbc' importable when running from project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from mortar_pbc import BoundaryClassifier3D


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
    parser.add_argument("--n", type=int, default=4,
                        help="Cells per direction (default 4)")
    parser.add_argument("--L", type=float, default=1.0,
                        help="Cube side length (default 1.0)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print(f"  BoundaryClassifier3D probe ({args.mesh_type}, n={args.n}, np={nranks})")
        print("=" * 70)

    # Build mesh + ParMesh
    mesh = build_box_mesh(args.mesh_type, n=args.n, L=args.L)
    pmesh = mfem.ParMesh(comm, mesh)

    # GetGlobalNE() is COLLECTIVE — call on all ranks (per §10.4).
    n_ge = pmesh.GetGlobalNE()
    if rank == 0:
        print(f"  ParMesh: {n_ge} global elements ({args.mesh_type})")

    # Build vector H1 FES
    fec = mfem.H1_FECollection(1, pmesh.Dimension())
    fes = mfem.ParFiniteElementSpace(pmesh, fec, pmesh.Dimension())

    n_tdofs = fes.GlobalTrueVSize()
    if rank == 0:
        print(f"  FES: vdim={fes.GetVDim()} order=1 global TDOFs={n_tdofs}")
        print()

    # Run the classifier (lots of collectives inside; see §10.4)
    classifier = BoundaryClassifier3D(pmesh, fes)

    if rank == 0:
        print(classifier.summary())
        print()

        # Sanity checks visible at rank-0.
        n_corners = len(classifier.corners)
        n_edges = len(classifier.edges)
        n_faces = len(classifier.faces)
        ok_topology = (n_corners == 8 and n_edges == 12 and n_faces == 6)
        n_mortar_edges = sum(
            1 for e in classifier.edges.values() if e.is_mortar
        )
        n_mortar_faces = sum(
            1 for f in classifier.faces.values() if f.is_mortar
        )
        ok_mortars = (n_mortar_edges == 3 and n_mortar_faces == 3)
        n_total_face_quads = sum(f.n_quad_elements for f in classifier.faces.values())
        n_total_face_tris = sum(f.n_tri_elements for f in classifier.faces.values())

        print(f"  TOPOLOGY:    {n_corners} corners, {n_edges} edges, "
              f"{n_faces} faces  -> {'OK' if ok_topology else 'FAIL'}")
        print(f"  MORTARS:     {n_mortar_edges} mortar edges (expect 3), "
              f"{n_mortar_faces} mortar faces (expect 3)  -> "
              f"{'OK' if ok_mortars else 'FAIL'}")
        print(f"  FACE ELEMS:  {n_total_face_quads} quads + {n_total_face_tris} tris")
        print()

        # Show one face's elements as a spot-check.
        print(f"  Spot-check: first 3 face_elements on 'top':")
        top = classifier.faces["top"]
        for k, fe in enumerate(top.face_elements[:3]):
            tag = fe.boundary_tag
            cls = type(fe).__name__
            print(f"    [{k}] {cls} boundary_tag={tag!r}  gtdofs={fe.gtdofs}")

        print()
        if ok_topology and ok_mortars:
            print("  ===== probe: PASS =====")
        else:
            print("  ===== probe: FAIL =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
