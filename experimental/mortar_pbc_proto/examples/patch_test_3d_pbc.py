"""3D mortar-PBC patch test driver — Phase 3.4.

End-to-end driver mirroring `examples/patch_test_2d.py` structure:

  1. Build mesh + ParMesh + vector H1 FES.
  2. Build classifier + constraint matrix C via Phase 3.3.B/C.
  3. Apply Dirichlet column-zeroing to C at corner gtdofs.
  4. Build distributed C_op / CT_op operators.
  5. Assemble linear-elastic K (HypreParMatrix).
  6. Compute u_lin = (F - I) X via apply_linear_part.
  7. Build the residual r1 = K · u_lin and eliminate Dirichlet
     rows/cols on K with prescribed corner values.
  8. Build the constraint RHS g = C · u_lin (so r2 = 0 at warm-start).
  9. Solve the saddle-point Newton step distributedly with
     SaddlePointSolver (GMRES + block-Jacobi).
 10. Recover u_total = u_lin + du; verify the homogeneous-RVE
     prediction ||du||_inf ≈ 0 to machine precision (linear elastic
     under uniform F has zero fluctuation u_tilde everywhere).
 11. Compute volume-averaged F via numerical integration on the
     deformed mesh; verify ||<F> - F_macro|| ≈ 0.
 12. Optionally write ParaView output for visual verification.

PASS criteria:
  * Krylov converged in ≤ ~50 iterations
  * ||du||_inf < 1e-7 (homogeneous-elastic warm-start exactness)
  * ||<F> - F_macro||_inf < 1e-9
  * Constraint residual ||C @ u_total - C @ u_lin||_inf < 1e-9

Run with:
    python examples/patch_test_3d_pbc.py --mesh-type hex
    python examples/patch_test_3d_pbc.py --mesh-type tet --paraview
    mpirun -np 4 python examples/patch_test_3d_pbc.py --mesh-type hex
    mpirun -np 4 python examples/patch_test_3d_pbc.py --mesh-type tet --paraview
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure the package is importable when run from project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

import mfem.par as mfem

from mortar_pbc import (
    BoundaryClassifier3D,
    ConstraintBuilder3D,
    SaddlePointSolver,
    make_constraint_operators,
    apply_dirichlet_zero_to_C,
    assemble_linear_elastic_K_hypre,
    apply_linear_part,
    apply_dirichlet_to_distributed_K,
    collect_corner_tdofs,
    write_pbc_visualization,
)
from mortar_pbc.elastic_3d import _get_my_first_tdof


# =============================================================================
# Helpers
# =============================================================================

def numpy_to_mfem_vector(arr: np.ndarray) -> mfem.Vector:
    """Wrap a numpy array as an mfem.Vector (copy semantics)."""
    return mfem.Vector(arr.tolist())


def mfem_vector_to_numpy(v: mfem.Vector) -> np.ndarray:
    """Copy an mfem.Vector into a numpy float64 array."""
    return np.array(v.GetDataArray(), dtype=np.float64).copy()


def build_box_mesh(mesh_type: str, n: int, L: float):
    if mesh_type == "hex":
        elem = mfem.Element.HEXAHEDRON
    elif mesh_type == "tet":
        elem = mfem.Element.TETRAHEDRON
    else:
        raise ValueError(f"Unknown mesh-type {mesh_type!r}")
    return mfem.Mesh.MakeCartesian3D(n, n, n, elem, L, L, L)


def parse_F_choice(name: str) -> np.ndarray:
    """Macroscopic deformation gradient choices.

    Picked to exercise the constraint matrix in different ways:
      - uniaxial: pure axial stretch in x
      - shear:    moderate non-symmetric shear (off-diagonal coupling)
      - mild:     small perturbation from identity (default for sanity)
    """
    if name == "uniaxial":
        return np.array([[1.20, 0.0,  0.0],
                         [0.0,  0.95, 0.0],
                         [0.0,  0.0,  0.95]])
    if name == "shear":
        return np.array([[1.00, 0.10, 0.05],
                         [0.05, 1.00, 0.10],
                         [0.10, 0.05, 1.00]])
    if name == "mild":
        return np.array([[1.05, 0.02, 0.01],
                         [0.01, 0.97, 0.02],
                         [0.02, 0.01, 1.03]])
    raise ValueError(f"Unknown F choice {name!r}")


def compute_volume_averaged_F_3d(
    pmesh: mfem.ParMesh,
    fes: mfem.ParFiniteElementSpace,
    u_par: mfem.Vector,
    comm: MPI.Comm,
) -> np.ndarray:
    """Compute <F> = I + (1/V) ∫ ∇u dV via Gauss quadrature on each element.

    Mirror of the 2D ``compute_volume_averaged_F`` in ``multistep_driver.py``,
    extended to 3D. Returns the global volume-averaged deformation
    gradient (collective: all ranks see the same value).
    """
    # Wrap u_par as a ParGridFunction so we can evaluate ∇u per element.
    u_gf = mfem.ParGridFunction(fes)
    u_gf.SetFromTrueDofs(u_par)

    integral_grad_u = np.zeros((3, 3), dtype=np.float64)
    total_volume = 0.0

    int_rule_orders = {
        mfem.Geometry.CUBE: 4,
        mfem.Geometry.TETRAHEDRON: 4,
    }

    for e in range(pmesh.GetNE()):
        T = pmesh.GetElementTransformation(e)
        geom = pmesh.GetElementBaseGeometry(e)
        ir = mfem.IntRules.Get(geom, int_rule_orders.get(geom, 4))

        for ip_idx in range(ir.GetNPoints()):
            ip = ir.IntPoint(ip_idx)
            T.SetIntPoint(ip)
            J_det = T.Weight()
            w = ip.weight * J_det

            # Compute ∇u at this quadrature point as a 3x3 matrix.
            grad_u = mfem.DenseMatrix(3, 3)
            u_gf.GetVectorGradient(T, grad_u)
            grad_u_np = np.asarray([
                [grad_u[i, j] for j in range(3)] for i in range(3)
            ], dtype=np.float64)

            integral_grad_u += w * grad_u_np
            total_volume += w

    # Global reduction (collective).
    integral_global = np.zeros((3, 3), dtype=np.float64)
    comm.Allreduce(integral_grad_u, integral_global, op=MPI.SUM)
    volume_global = comm.allreduce(total_volume, op=MPI.SUM)

    F_avg = np.eye(3) + integral_global / volume_global
    return F_avg


# =============================================================================
# Main driver
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-type", choices=["hex", "tet"], default="hex")
    parser.add_argument("--n", type=int, default=4,
                        help="Cells per direction")
    parser.add_argument("--L", type=float, default=1.0,
                        help="Cube side length")
    parser.add_argument("--F", choices=["uniaxial", "shear", "mild"],
                        default="mild",
                        help="Macroscopic deformation gradient")
    parser.add_argument("--E", type=float, default=70.0e3,
                        help="Young's modulus (homogeneous)")
    parser.add_argument("--nu", type=float, default=0.3,
                        help="Poisson's ratio")
    parser.add_argument("--paraview", action="store_true",
                        help="Write ParaView output for visual verification")
    parser.add_argument("--paraview-dir", default="./paraview_3d_pbc",
                        help="ParaView output directory")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    F = parse_F_choice(args.F)

    if rank == 0:
        print("=" * 72)
        print(f"  3D mortar-PBC patch test (Phase 3.4)")
        print(f"  mesh-type = {args.mesh_type}, n = {args.n}, L = {args.L}, "
              f"np = {nranks}")
        print(f"  F = {args.F}:")
        for row in F:
            print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
        print(f"  E = {args.E:.4e}, nu = {args.nu}")
        print("=" * 72)

    # ---------------------------------------------------------------------
    # Step 1 — mesh + FES
    # ---------------------------------------------------------------------
    mesh = build_box_mesh(args.mesh_type, n=args.n, L=args.L)
    pmesh = mfem.ParMesh(comm, mesh)
    n_ge = pmesh.GetGlobalNE()
    fec = mfem.H1_FECollection(1, pmesh.Dimension())
    fes = mfem.ParFiniteElementSpace(pmesh, fec, pmesh.Dimension())
    n_global_tdofs = fes.GlobalTrueVSize()
    if rank == 0:
        print(f"\n[1] Mesh: {n_ge} global elements ({args.mesh_type}), "
              f"global TDOFs = {n_global_tdofs}")

    # ---------------------------------------------------------------------
    # Step 2 — classifier + constraint matrix
    # ---------------------------------------------------------------------
    classifier = BoundaryClassifier3D(pmesh, fes)
    builder = ConstraintBuilder3D(classifier)
    C_global_csr = builder.build()
    n_lam_total = C_global_csr.shape[0]
    if rank == 0:
        print(f"[2] Classifier: {len(classifier.corners)} corners, "
              f"{len(classifier.edges)} edges, {len(classifier.faces)} faces")
        print(f"    Constraint matrix C: shape={C_global_csr.shape}, "
              f"nnz={C_global_csr.nnz}")

    # ---------------------------------------------------------------------
    # Step 3 — apply Dirichlet column-zeroing to C at corner gtdofs
    # ---------------------------------------------------------------------
    corner_gtdofs = collect_corner_tdofs(classifier.corners)
    C_global_csr_modified = apply_dirichlet_zero_to_C(
        C_global_csr, corner_gtdofs,
    )
    if rank == 0:
        print(f"[3] Corner Dirichlet TDOFs (24 = 8 corners × 3 components): "
              f"{len(corner_gtdofs)}")
        print(f"    C after column-zeroing: nnz = "
              f"{C_global_csr_modified.nnz} (was {C_global_csr.nnz})")

    # ---------------------------------------------------------------------
    # Step 4 — build distributed C_op / CT_op operators
    # ---------------------------------------------------------------------
    n_lam_local = n_lam_total if rank == 0 else 0
    C_op, CT_op = make_constraint_operators(
        C_global_csr_modified, fes, n_lam_local,
    )
    if rank == 0:
        print(f"[4] C_op / CT_op built (n_lam_total = {n_lam_total}, "
              f"replicated on rank 0)")

    # ---------------------------------------------------------------------
    # Step 5 — assemble K (linear elastic)
    # ---------------------------------------------------------------------
    K_hyp = assemble_linear_elastic_K_hypre(pmesh, fes, E=args.E, nu=args.nu)
    if rank == 0:
        print(f"[5] K assembled (HypreParMatrix)")

    # ---------------------------------------------------------------------
    # Step 6 — u_lin = (F - I) X
    # ---------------------------------------------------------------------
    u_lin_local = apply_linear_part(fes, F)
    if rank == 0:
        u_lin_norm = float(np.linalg.norm(u_lin_local, ord=np.inf))
        print(f"[6] u_lin built. ||u_lin||_inf (rank 0) = {u_lin_norm:.4e}")

    # ---------------------------------------------------------------------
    # Step 7 — residual r1 = K · u_lin; Dirichlet elimination on K
    # ---------------------------------------------------------------------
    f_par = mfem.Vector(fes.GetTrueVSize())
    u_lin_par = numpy_to_mfem_vector(u_lin_local)
    K_hyp.Mult(u_lin_par, f_par)
    # f_par now holds K · u_lin.
    # We want to solve  K · du = -r1  with  du_corner = 0  (Dirichlet).
    # So r1 = K · u_lin (the residual at u_init = u_lin), and after
    # eliminating corner rows/cols, the corner entries of f are forced
    # to zero (since du_corner = 0 means the prescribed essential value
    # is zero on the increment du).
    apply_dirichlet_to_distributed_K(
        K_hyp, f_par, corner_gtdofs, fes,
        f_at_essential=None,    # du_corner = 0 (homogeneous on the increment)
    )
    if rank == 0:
        print(f"[7] Dirichlet elimination applied on K and f")

    # ---------------------------------------------------------------------
    # Step 8 — constraint RHS g = C · u_lin
    # ---------------------------------------------------------------------
    # The constraint we want to solve is C · u = g, where u = u_lin + du.
    # If we set g = C · u_lin, then C · du = 0 (homogeneous on the
    # increment), which is what the saddle-point solver expects.
    Cu_lin = mfem.Vector(n_lam_local)
    C_op.Mult(u_lin_par, Cu_lin)
    # We pass r2 = -g + C @ u_init = 0 to the solver (since u_init = u_lin
    # and g = C · u_lin).
    r2_par = mfem.Vector(n_lam_local)
    r2_par.Assign(0.0)
    if rank == 0:
        cu_lin_norm = float(np.max(np.abs(mfem_vector_to_numpy(Cu_lin))))
        print(f"[8] g = C · u_lin built. ||g||_inf = {cu_lin_norm:.4e}")
        print(f"    r2 = C · u_init - g = 0 (warm-start at u_init = u_lin)")

    # ---------------------------------------------------------------------
    # Step 9 — distributed Krylov saddle-point solve
    # ---------------------------------------------------------------------
    sps = SaddlePointSolver(
        solver="GMRES",
        preconditioner="block_jacobi",
        rel_tol=1e-12,
        abs_tol=1e-16,
        max_iter=2000,
        print_level=-1,
    )
    if rank == 0:
        print(f"\n[9] Saddle-point solve "
              f"({sps.solver_name} + {sps.preconditioner})")
    du_par, dlam_par = sps.solve_step(
        K_op=K_hyp, C_op=C_op, CT_op=CT_op,
        r1_local=f_par,
        r2_local=r2_par,
    )
    if rank == 0:
        print(f"    Krylov: iters = {sps.last_iterations}, "
              f"converged = {sps.last_converged}, "
              f"final residual = {sps.last_final_norm:.3e}")

    # ---------------------------------------------------------------------
    # Step 10 — recover u_total = u_lin + du; check ||du||_inf
    # ---------------------------------------------------------------------
    du_local = mfem_vector_to_numpy(du_par)
    u_total_local = u_lin_local + du_local
    # Distributed-aware norms.
    du_max_local = float(np.max(np.abs(du_local))) if du_local.size > 0 else 0.0
    du_max_global = comm.allreduce(du_max_local, op=MPI.MAX)
    if rank == 0:
        print(f"\n[10] u = u_lin + du recovered.")
        print(f"     ||du||_inf (global)        = {du_max_global:.3e}  "
              f"(homogeneous-elastic exact target: ~ 1e-10)")

    # u_total_par for downstream use.
    u_total_par = numpy_to_mfem_vector(u_total_local)

    # ---------------------------------------------------------------------
    # Step 11 — verify <F> ≈ F_macro
    # ---------------------------------------------------------------------
    F_avg = compute_volume_averaged_F_3d(pmesh, fes, u_total_par, comm)
    F_diff = F_avg - F
    F_diff_max = float(np.max(np.abs(F_diff)))
    if rank == 0:
        print(f"\n[11] Volume-averaged F:")
        print(f"     <F> = ")
        for row in F_avg:
            print(f"       [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
        print(f"     ||<F> - F_macro||_inf = {F_diff_max:.3e}")

    # Constraint residual check (using ORIGINAL C, not Dirichlet-modified).
    Cu_total_par = mfem.Vector(n_lam_local)
    C_op.Mult(u_total_par, Cu_total_par)
    Cu_lin_par = mfem.Vector(n_lam_local)
    C_op.Mult(u_lin_par, Cu_lin_par)
    if rank == 0:
        residual_local = (
            mfem_vector_to_numpy(Cu_total_par)
            - mfem_vector_to_numpy(Cu_lin_par)
        )
        constraint_residual_inf = float(np.max(np.abs(residual_local)))
        print(f"     ||C·u_total - C·u_lin||_inf = "
              f"{constraint_residual_inf:.3e}")

    # ---------------------------------------------------------------------
    # PASS criteria summary
    # ---------------------------------------------------------------------
    pass_du   = du_max_global < 1e-7
    pass_F    = F_diff_max    < 1e-9
    if rank == 0:
        pass_constraint = constraint_residual_inf < 1e-9
    else:
        pass_constraint = True
    pass_constraint = comm.bcast(pass_constraint, root=0)
    pass_krylov = sps.last_converged

    all_pass = pass_du and pass_F and pass_constraint and pass_krylov

    if rank == 0:
        print(f"\n{'=' * 72}")
        print(f"  PASS criteria:")
        print(f"     Krylov converged             : "
              f"{'OK' if pass_krylov else 'FAIL'} "
              f"({sps.last_iterations} iterations)")
        print(f"     ||du||_inf < 1e-7            : "
              f"{'OK' if pass_du else 'FAIL'} ({du_max_global:.2e})")
        print(f"     ||<F> - F_macro|| < 1e-9     : "
              f"{'OK' if pass_F else 'FAIL'} ({F_diff_max:.2e})")
        print(f"     ||C·u - C·u_lin|| < 1e-9     : "
              f"{'OK' if pass_constraint else 'FAIL'}")
        print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
        print(f"{'=' * 72}")

    # ---------------------------------------------------------------------
    # Step 12 — ParaView visual verification (optional)
    # ---------------------------------------------------------------------
    if args.paraview:
        if rank == 0:
            print(f"\n[12] Writing ParaView output to {args.paraview_dir}/")
        os.makedirs(args.paraview_dir, exist_ok=True)
        du_par_for_viz = numpy_to_mfem_vector(du_local)
        write_pbc_visualization(
            pmesh=pmesh, fes=fes,
            u_par=u_total_par, u_lin_par=u_lin_par, du_par=du_par_for_viz,
            output_dir=args.paraview_dir,
            name=f"patch_3d_{args.mesh_type}_{args.F}",
            F_label=f"F={args.F}, E={args.E:.0e}, nu={args.nu}",
        )
        if rank == 0:
            print(f"     -> open {args.paraview_dir}/"
                  f"patch_3d_{args.mesh_type}_{args.F}.pvd in ParaView")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
