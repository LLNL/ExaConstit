"""3D mortar PBC patch test — linear elastic, 2x2x2 OCTANT CHECKERBOARD.

Direct 3D analog of `examples/patch_test_2d_checkerboard.py` (which uses
4-quadrant XOR), extended to a 2x2x2 octant XOR pattern. This is the
**most stressful** Phase 3.5 test for the constraint machinery because
material seams now coincide with **three orthogonal interior planes**
(x=L/2, y=L/2, z=L/2) — the closest analog in a unit cube of a real
3D wirebasket configuration where material discontinuities cross the
corner / edge / face periodic constraints simultaneously.

Material setup
--------------
Octant-XOR by sign of (x - L/2, y - L/2, z - L/2):
  * Count = number of "high" signs (x>L/2, y>L/2, z>L/2 each contribute 1).
  * count even (0 or 2 highs)  -> attribute 1 (matrix material)
  * count odd  (1 or 3 highs)  -> attribute 2 (stiff material)

This produces an alternating black/white 3D pattern: every shared face
between two adjacent octants joins materials of opposite type, so:

  * Periodic BC in x  : ALL four x=0 ↔ x=L nonmortar/mortar pairings
                        cross a material interface (front-bottom is
                        matrix, back-bottom is stiff at x=0; reversed
                        at x=L). Forces non-trivial fluctuation in x.
  * Periodic BC in y  : same — every y-pairing crosses an interface.
  * Periodic BC in z  : same.

So all THREE periodic-axis constraint blocks see across-material
coupling on every matched element pair. By contrast, the strip-split
test (`patch_test_3d_heterogeneous.py`) only crosses the interface on
the x-pairing; y and z pairings stay within material. The checkerboard
exercises the full constraint apparatus: face-center face-mortar
coupling, edge-center edge-mortar coupling, AND corner-Dirichlet
prescription must all coordinate to produce a consistent fluctuation.

Method-D + multi-step warm-start
---------------------------------
Identical to the strip-split test. PASS criteria are identical:
  * Krylov converged
  * ||C·u_tilde||_2 < 1e-8 (constraint residual after solve)
  * ||u_tilde||_inf > 1e-12 (heterogeneous fluctuation must be present)
  * |<F> - F_macro|_max < 1e-9 (Hill-Mandel homogenization consistency)

Macroscopic F selectable via --F flag (same options as het):
  --F=uniaxial  (default) : axial stretch in x, Poisson contraction in y/z
  --F=biaxial             : stretch in x, y; contract in z
  --F=shear               : full off-diagonal coupling
  --F=mild-shear          : small perturbation (sanity check)

Run with:
    python examples/patch_test_3d_checkerboard.py
    python examples/patch_test_3d_checkerboard.py --F=shear --paraview
    mpirun -np 4 python examples/patch_test_3d_checkerboard.py --steps=3
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
import scipy.sparse as sp
from mpi4py import MPI

import mfem.par as mfem

from mortar_pbc import (
    BoundaryClassifier3D,
    ConstraintBuilder3D,
    SaddlePointSolver,
    make_constraint_operators,
    apply_dirichlet_zero_to_C,
    apply_linear_part,
    apply_dirichlet_to_distributed_K,
    collect_corner_tdofs,
    PbcVisualizationWriter,
    MortarPbcDriver2D,    # name is historical; class is dim-generic
)
from mortar_pbc.elastic_3d import _get_my_first_tdof


# =============================================================================
# Helpers (same as patch_test_3d_pbc.py)
# =============================================================================

def numpy_to_mfem_vector(arr: np.ndarray) -> mfem.Vector:
    return mfem.Vector(arr.tolist())


def mfem_vector_to_numpy(v: mfem.Vector) -> np.ndarray:
    return np.array(v.GetDataArray(), dtype=np.float64).copy()


# =============================================================================
# Checkerboard mesh: 2x2x2 octant XOR (3D analog of 4-quadrant 2D test)
# =============================================================================

def build_checkerboard_mesh_3d(
    mesh_type: str, n: int, L: float,
) -> mfem.Mesh:
    """3D RVE on [0, L]^3 with 2x2x2 octant-XOR element attributes.

    For each element with centroid (x_c, y_c, z_c), let
        bx = (x_c >= L/2),  by = (y_c >= L/2),  bz = (z_c >= L/2)
    and count = bx + by + bz (in {0, 1, 2, 3}). Then
        attribute = 1 if count is even (0 or 2 highs)
        attribute = 2 if count is odd  (1 or 3 highs)

    This produces a 3D black/white checkerboard:
        BLF (000) -> attr 1     BRF (100) -> attr 2
        TLF (010) -> attr 2     TRF (110) -> attr 1
        BLB (001) -> attr 2     BRB (101) -> attr 1
        TLB (011) -> attr 1     TRB (111) -> attr 2

    Adjacent octants always carry opposite attributes, so every pair of
    matched periodic-boundary elements (nonmortar on one side, mortar on
    the opposite face) crosses a material interface. Maximum stress on
    the constraint machinery for a given mesh size and contrast.
    """
    if mesh_type == "hex":
        elem = mfem.Element.HEXAHEDRON
    elif mesh_type == "tet":
        elem = mfem.Element.TETRAHEDRON
    else:
        raise ValueError(f"Unknown mesh-type {mesh_type!r}")
    mesh = mfem.Mesh.MakeCartesian3D(n, n, n, elem, L, L, L)

    L_half = 0.5 * L
    for e in range(mesh.GetNE()):
        verts = [int(v) for v in mesh.GetElementVertices(e)]
        xs = [mesh.GetVertexArray(v)[0] for v in verts]
        ys = [mesh.GetVertexArray(v)[1] for v in verts]
        zs = [mesh.GetVertexArray(v)[2] for v in verts]
        x_centroid = sum(xs) / len(xs)
        y_centroid = sum(ys) / len(ys)
        z_centroid = sum(zs) / len(zs)
        bx = 1 if x_centroid >= L_half else 0
        by = 1 if y_centroid >= L_half else 0
        bz = 1 if z_centroid >= L_half else 0
        count = bx + by + bz
        # XOR pattern: even count -> mat 1, odd count -> mat 2.
        if count % 2 == 0:
            mesh.SetAttribute(e, 1)
        else:
            mesh.SetAttribute(e, 2)
    # Force MFEM to refresh the cached attribute set so PWConstCoefficient
    # sees both 1 and 2.
    mesh.SetAttributes()
    return mesh


# =============================================================================
# Heterogeneous K assembly (PWConstCoefficient on Lame parameters)
# =============================================================================

def assemble_heterogeneous_K_hypre(
    pmesh: mfem.ParMesh,
    fes: mfem.ParFiniteElementSpace,
    *,
    E_1: float, nu_1: float,
    E_2: float, nu_2: float,
):
    """Assemble two HypreParMatrices (full and to-be-eliminated)
    with per-element-attribute Lame parameters.

    Returns (K_full, K_eliminated). The reason for two: per MFEM #793,
    `ParBilinearForm.ParallelAssemble` may share underlying SparseMatrix
    data between the form and the matrix; calling it twice on the same
    form gives two HypreParMatrices that may alias. We build TWO
    independent bilinear forms so each is independently safe to mutate.
    """
    mu_1  = 0.5 * E_1 / (1.0 + nu_1)
    lam_1 = E_1 * nu_1 / ((1.0 + nu_1) * (1.0 - 2.0 * nu_1))
    mu_2  = 0.5 * E_2 / (1.0 + nu_2)
    lam_2 = E_2 * nu_2 / ((1.0 + nu_2) * (1.0 - 2.0 * nu_2))

    mu_vec  = mfem.Vector([mu_1,  mu_2 ])
    lam_vec = mfem.Vector([lam_1, lam_2])
    mu_coef  = mfem.PWConstCoefficient(mu_vec)
    lam_coef = mfem.PWConstCoefficient(lam_vec)

    a_full = mfem.ParBilinearForm(fes)
    a_full.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coef, mu_coef))
    a_full.Assemble()
    a_full.Finalize()
    K_full = a_full.ParallelAssemble()

    a_elim = mfem.ParBilinearForm(fes)
    a_elim.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coef, mu_coef))
    a_elim.Assemble()
    a_elim.Finalize()
    K_elim = a_elim.ParallelAssemble()

    return K_full, K_elim


# =============================================================================
# F_macro choices for 3D
# =============================================================================

def parse_F_choice(name: str) -> np.ndarray:
    if name == "uniaxial":
        # Axial stretch in x, Poisson contraction in y/z.
        return np.array([[1.20, 0.0,  0.0],
                         [0.0,  0.95, 0.0],
                         [0.0,  0.0,  0.95]])
    if name == "biaxial":
        return np.array([[1.15, 0.0,  0.0],
                         [0.0,  1.10, 0.0],
                         [0.0,  0.0,  0.90]])
    if name == "shear":
        return np.array([[1.10, 0.10, 0.05],
                         [0.05, 1.00, 0.10],
                         [0.10, 0.05, 1.05]])
    if name == "mild-shear":
        return np.array([[1.05, 0.05, 0.02],
                         [0.02, 1.02, 0.05],
                         [0.05, 0.02, 1.03]])
    raise ValueError(f"Unknown F choice: {name!r}")


def build_F_ramp(F_target: np.ndarray, n_steps: int) -> list:
    """Linear ramp from F=I (no load) to F_target in n_steps."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    F_minus_I = F_target - np.eye(3)
    return [
        np.eye(3) + ((k + 1) / n_steps) * F_minus_I
        for k in range(n_steps)
    ]


# =============================================================================
# Pretty-print step result
# =============================================================================

def _print_step_result(r) -> None:
    print(f"      Krylov: {r.krylov_iters} iters, "
          f"converged={r.krylov_converged}, "
          f"final_norm={r.krylov_final_norm:.3e}")
    print(f"      ||u||_inf       = {r.u_inf:.3e}")
    print(f"      ||u_tilde||_inf = {r.u_tilde_inf:.3e}  "
          f"(<- non-zero for heterogeneous material)")
    print(f"      ||C·u_tilde||_2 = {r.constraint_residual:.3e}")
    print(f"      |<F> - F_macro|_max = {r.F_average_error:.3e}")


def _indent(s: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in s.splitlines())


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-type", choices=["hex", "tet"], default="hex")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--F", default="uniaxial",
                        choices=["uniaxial", "biaxial", "shear", "mild-shear"])
    parser.add_argument("--steps", type=int, default=3,
                        help="Number of ramp steps from F=I to F=F_target")
    parser.add_argument("--E1", type=float, default=70.0e3,
                        help="Material 1 Young's modulus (even-octant attr=1)")
    parser.add_argument("--E2", type=float, default=350.0e3,
                        help="Material 2 Young's modulus (odd-octant attr=2, stiff)")
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--paraview", action="store_true")
    parser.add_argument("--paraview-dir",
                        default="./paraview_3d_checkerboard")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    F_target = parse_F_choice(args.F)
    F_ramp   = build_F_ramp(F_target, args.steps)

    if rank == 0:
        print("=" * 72)
        print(f"  3D checkerboard (octant-XOR) mortar-PBC patch test "
              f"(Phase 3.5 extension)")
        print(f"  mesh-type = {args.mesh_type}, n = {args.n}, L = {args.L}, "
              f"np = {nranks}")
        print(f"  F = {args.F}, ramp steps = {args.steps}")
        print(f"  Target F_macro:")
        for row in F_target:
            print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
        print(f"  Material 1 (even-octant, attr=1): "
              f"E={args.E1:.3e}, nu={args.nu}")
        print(f"  Material 2 (odd-octant,  attr=2): "
              f"E={args.E2:.3e}, nu={args.nu}  "
              f"(contrast = {args.E2/args.E1:.1f}x)")
        print("=" * 72)

    # ---------------------------------------------------------------------
    # Step 1 — heterogeneous mesh + FES
    # ---------------------------------------------------------------------
    mesh = build_checkerboard_mesh_3d(args.mesh_type, n=args.n, L=args.L)
    pmesh = mfem.ParMesh(comm, mesh)
    n_ge = pmesh.GetGlobalNE()
    fec = mfem.H1_FECollection(1, pmesh.Dimension())
    fes = mfem.ParFiniteElementSpace(pmesh, fec, pmesh.Dimension())
    n_global_tdofs = fes.GlobalTrueVSize()
    if rank == 0:
        attrs_list = []
        for e in range(pmesh.GetNE()):
            attrs_list.append(int(pmesh.GetAttribute(e)))
        from collections import Counter
        attr_cnt = Counter(attrs_list)
        print(f"\n[1] Mesh: {n_ge} global elements ({args.mesh_type}), "
              f"global TDOFs = {n_global_tdofs}")
        print(f"    Element-attribute distribution (rank 0): {dict(attr_cnt)}")

    # ---------------------------------------------------------------------
    # Step 2 — classifier + constraint matrix
    # ---------------------------------------------------------------------
    classifier = BoundaryClassifier3D(pmesh, fes)
    builder = ConstraintBuilder3D(classifier)
    C_global_csr = builder.build()
    n_lam_total = C_global_csr.shape[0]
    if rank == 0:
        print(f"[2] Classifier + ConstraintBuilder3D: "
              f"C shape={C_global_csr.shape}, nnz={C_global_csr.nnz}")

    # ---------------------------------------------------------------------
    # Step 3 — corner Dirichlet, build C_op / CT_op
    # ---------------------------------------------------------------------
    corner_gtdofs = collect_corner_tdofs(classifier.corners)
    C_global_csr_modified = apply_dirichlet_zero_to_C(
        C_global_csr, corner_gtdofs,
    )
    n_lam_local = n_lam_total if rank == 0 else 0
    C_op, CT_op = make_constraint_operators(
        C_global_csr_modified, fes, n_lam_local,
    )
    if rank == 0:
        print(f"[3] 24 corner TDOFs identified; C column-zeroed")
        print(f"    Distributed C_op / CT_op built")

    # ---------------------------------------------------------------------
    # Step 4 — heterogeneous K (full + eliminated)
    # ---------------------------------------------------------------------
    K_full, K_hyp = assemble_heterogeneous_K_hypre(
        pmesh, fes,
        E_1=args.E1, nu_1=args.nu,
        E_2=args.E2, nu_2=args.nu,
    )
    # Apply Dirichlet to K_hyp (the eliminated copy). Pass a zero RHS;
    # the multi-step driver constructs its own RHS per step.
    f_dummy = mfem.Vector(fes.GetTrueVSize())
    f_dummy.Assign(0.0)
    apply_dirichlet_to_distributed_K(
        K_hyp, f_dummy, corner_gtdofs, fes, f_at_essential=None,
    )
    if rank == 0:
        print(f"[4] K assembled with PWConstCoefficient (E_1, E_2 distinct); "
              f"corner rows/cols eliminated")

    # ---------------------------------------------------------------------
    # Step 5 — saddle-point solver + multi-step driver
    # ---------------------------------------------------------------------
    sps = SaddlePointSolver(
        solver="GMRES",
        preconditioner="block_jacobi",
        rel_tol=1e-12,
        abs_tol=1e-16,
        max_iter=5000,
        print_level=-1,
    )

    # Build the local-corner-TDOF index list (per-rank slices into vectors).
    my_first_tdof = _get_my_first_tdof(fes, rank)
    my_n_tdof = fes.GetTrueVSize()
    local_corner_tdofs = [
        gt - my_first_tdof for gt in corner_gtdofs
        if my_first_tdof <= gt < my_first_tdof + my_n_tdof
    ]

    driver = MortarPbcDriver2D(
        pmesh=pmesh, fes=fes,
        K_op=K_hyp, K_op_full=K_full,
        C_op=C_op, CT_op=CT_op,
        corner_tdofs=corner_gtdofs,
        apply_linear_part_fn=apply_linear_part,
        numpy_to_mfem_vector_fn=numpy_to_mfem_vector,
        sps=sps,
        n_lam_local=n_lam_local,
        local_corner_tdofs=local_corner_tdofs,
    )
    if rank == 0:
        print(f"[5] SaddlePointSolver + MortarPbcDriver constructed "
              f"(used dim-generically in 3D)")

    # ---------------------------------------------------------------------
    # Step 6 — ramp through F (multi-step warm-start)
    # ---------------------------------------------------------------------
    pv_writer = None
    if args.paraview:
        os.makedirs(args.paraview_dir, exist_ok=True)
        pv_writer = PbcVisualizationWriter(
            pmesh, fes,
            output_dir=args.paraview_dir,
            name=f"checker_{args.mesh_type}_{args.F}",
        )

    if rank == 0:
        print(f"\n{'=' * 72}")
        print(f"Ramping F: {args.steps} step{'s' if args.steps != 1 else ''}")
        print(f"{'=' * 72}")

    for step_idx, F_k in enumerate(F_ramp):
        if rank == 0:
            print(f"\n  --- Step {step_idx+1}/{args.steps}  ({args.F}) ---")
            print(f"      F_k =\n{_indent(repr(F_k), 12)}")
        if step_idx == 0:
            result = driver.solve_first_step(F_k)
        else:
            result = driver.solve_next_step(F_k)
        if rank == 0:
            _print_step_result(result)
        if pv_writer is not None:
            u_lin_k_local = apply_linear_part(fes, F_k)
            u_lin_k_par   = numpy_to_mfem_vector(u_lin_k_local)
            du_k_par      = mfem.Vector(my_n_tdof)
            for i in range(my_n_tdof):
                du_k_par[i] = float(driver.u_par[i]) - float(u_lin_k_par[i])
            pv_writer.write_step(
                driver.u_par, u_lin_k_par, du_k_par,
                time=float(step_idx + 1),
                F_label=f"{args.F}/step{step_idx+1}",
                write_undeformed_first=(step_idx == 0),
            )

    # ---------------------------------------------------------------------
    # Step 7 — final-step PASS / FAIL summary
    # ---------------------------------------------------------------------
    final = driver.history[-1]
    if rank == 0:
        print(f"\n{'=' * 72}")
        print("Final-step PASS / FAIL")
        print(f"{'=' * 72}")
        pass_constraint_atol = 1.0e-8
        pass_fluct_lower_bnd = 1.0e-12
        pass_F_avg_atol      = 1.0e-9

        passed = (
            final.krylov_converged
            and final.constraint_residual < pass_constraint_atol
            and final.u_tilde_inf         > pass_fluct_lower_bnd
            and final.F_average_error     < pass_F_avg_atol
        )

        print(f"  Krylov converged    : "
              f"{'OK' if final.krylov_converged else 'FAIL'} "
              f"({final.krylov_iters} iters, final={final.krylov_final_norm:.3e})")
        print(f"  Constraint residual : "
              f"{'OK' if final.constraint_residual < pass_constraint_atol else 'FAIL'} "
              f"(||C·u_tilde||_2 = {final.constraint_residual:.3e}, "
              f"tol = {pass_constraint_atol:.0e})")
        print(f"  Fluctuation present : "
              f"{'OK' if final.u_tilde_inf > pass_fluct_lower_bnd else 'FAIL'} "
              f"(||u_tilde||_inf = {final.u_tilde_inf:.3e}, "
              f"lower bound = {pass_fluct_lower_bnd:.0e})")
        print(f"  Volume-averaged F   : "
              f"{'OK' if final.F_average_error < pass_F_avg_atol else 'FAIL'} "
              f"(|<F> - F_macro|_max = {final.F_average_error:.3e}, "
              f"tol = {pass_F_avg_atol:.0e})")
        print()
        print(f"  Overall: {'PASS' if passed else 'FAIL'}")
        if pv_writer is not None:
            print(f"\n  ParaView output: {args.paraview_dir}/"
                  f"checker_{args.mesh_type}_{args.F}.pvd")

    # Broadcast pass status for the return code.
    pass_bool = comm.bcast(
        bool(
            final.krylov_converged
            and final.constraint_residual < 1.0e-8
            and final.u_tilde_inf > 1.0e-12
            and final.F_average_error < 1.0e-9
        ) if rank == 0 else False,
        root=0,
    )
    return 0 if pass_bool else 1


if __name__ == "__main__":
    sys.exit(main())
