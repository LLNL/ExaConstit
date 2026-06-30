"""Phase 3.1 patch test: 3D linear-elastic homogeneous RVE, NO mortar.

Per MORTAR_PBC_ARCHITECTURE.md §11.8 Phase 3.1 (revised):

    Hex mesh built via ``mfem.Mesh.MakeCartesian3D`` OR tet mesh built
    via ``MakeCartesian3D`` with ``Element.TETRAHEDRON``. **Full
    Dirichlet** on all 6 boundary faces at u_lin = (F - I) X. NO
    periodic constraint, NO traction. Solve linear elastic K · u = 0
    with the prescribed Dirichlet boundary. For homogeneous material,
    the unique solution is u = u_lin everywhere.

Why full-boundary Dirichlet, not corner-only
--------------------------------------------
The original Phase 3.1 design (8 corner Dirichlets, free Neumann
elsewhere) does NOT have u_lin as its solution. For homogeneous linear
elasticity with affine u_lin:
    div σ(u_lin) = 0 in Ω      (constant stress ⇒ zero divergence)
    σ · n ≠ 0    on ∂Ω         (constant stress hits surface normal)

Pinning corners only leaves ∂Ω\corners with the "natural" BC σ · n = 0,
which is incompatible with the constant-stress field. The minimum-
energy field then relaxes outward and is NOT u_lin. The corner-only
mismatch shows up in practice as ‖K · u_lin‖_inf ≫ assembly noise on
boundary DOFs, and ‖du‖_inf at the percent level.

Full-boundary Dirichlet at u_lin makes the BVP well-posed: only
interior DOFs are free, and ∫ ∇N_i dV = 0 for compactly-supported
interior basis functions, so (K · u_lin)_i = 0 for all interior i. The
solver then drives du = 0 to machine precision.

In the production phasing, the missing "boundary tractions" on the
free-Neumann boundary are supplied by the *mortar PBC* (= periodic
nonmortar-mortar coupling, no traction freedom across periodic faces) +
*8 corner Dirichlets* (the affine-mode pin). That's Phase 3.4. Phase
3.1 here is only validating K + Dirichlet + CG-AMG infrastructure.

PASS criteria
-------------
    * |u - u_lin|_inf < 1e-10   (machine precision)
    * |⟨F⟩ - F_macro|_max < 1e-12   (homogenization consistency)

Solve structure
---------------
Newton-step from u_init = u_lin (on ALL DOFs):

    Step 1: u_init = u_lin everywhere (boundary AND interior).
    Step 2: r1 = K · u_init = K · u_lin (full operator action).
    Step 3: Eliminate K's boundary rows/cols, set r1[boundary] = 0
            (since du[boundary] = 0 — u_init already at u_lin on bdry).
    Step 4: Solve K_eliminated · du = -r1, with du[boundary] = 0
            absorbed by the identity rows on the eliminated DOFs.
    Step 5: u = u_init + du.

For a homogeneous medium under uniform F, K · u_lin = 0 in the
interior (linear-elastic operator on an affine field has zero
divergence), so r1[interior] ≈ 0 to assembly noise. After eliminating
boundary, the free-DOF system K_ii · du_i = 0 has unique solution
du_i = 0 (K_ii is SPD). So u ≈ u_lin to the linear-solver noise floor.

Phase 3.1 establishes (with NO mortar):
    * 3D mesh handling on hex AND tet meshes (one --mesh-type flag)
    * 3D vector FES (vdim = 3)
    * Linear-elastic K assembly (dim-generic, inherits from 2D)
    * 3D corner identification (find_corners_3d)
    * 3D Dirichlet on the distributed K (dim-generic helper)
    * 3D ⟨F⟩ diagnostic (compute_volume_averaged_F is dim-generic)

Run with:
    python examples/patch_test_3d_homogeneous.py --mesh-type hex
    python examples/patch_test_3d_homogeneous.py --mesh-type tet
    mpirun -n 2 python examples/patch_test_3d_homogeneous.py --mesh-type hex
    mpirun -n 4 python examples/patch_test_3d_homogeneous.py --mesh-type tet
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from mortar_pbc import (
    assemble_linear_elastic_K_hypre,
    apply_linear_part,
    find_corners_3d,
    apply_dirichlet_to_distributed_K,
    newton_residual_at_u_lin,
    collect_corner_tdofs,
    find_all_boundary_tdofs,
    compute_volume_averaged_F,
)


# =============================================================================
# Mesh construction
# =============================================================================

def build_3d_box_mesh(mesh_type: str, nx: int = 4, ny: int = 4, nz: int = 4,
                      L: float = 1.0) -> mfem.Mesh:
    """Build a 3D box RVE of side L with nx × ny × nz cells.

    Parameters
    ----------
    mesh_type : {"hex", "tet"}
        "hex" → MakeCartesian3D with hex-8 elements.
        "tet" → MakeCartesian3D with tet-4 elements (MFEM subdivides each
        hex cell into 6 tets internally when given Element.TETRAHEDRON).
    nx, ny, nz : int
        Cells per direction.
    L : float
        Cube side length.

    Returns
    -------
    mesh : mfem.Mesh
        Serial mesh, ready for ParMesh construction. Boundary attributes
        are set by MakeCartesian3D following the convention:
            1 = bottom (y=0)   2 = front (z=0)   3 = right (x=L)
            4 = back   (z=L)   5 = left  (x=0)   6 = top   (y=L)
    """
    if mesh_type == "hex":
        elem_type = mfem.Element.HEXAHEDRON
    elif mesh_type == "tet":
        elem_type = mfem.Element.TETRAHEDRON
    else:
        raise ValueError(f"Unknown mesh_type {mesh_type!r}; expected 'hex' or 'tet'")

    # MakeCartesian3D signature (per pyMFEM/mfem-cpp):
    #   MakeCartesian3D(nx, ny, nz, type, sx=1.0, sy=1.0, sz=1.0,
    #                   sfc_ordering=True)
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, elem_type, L, L, L)
    return mesh


# =============================================================================
# Driver
# =============================================================================

def run_phase31(args) -> int:
    """Run Phase 3.1; return 0 on PASS, 1 on FAIL."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    # ----- Choose F_macro -----
    if args.F_mode == "uniaxial":
        # Volume-preserving uniaxial: stretch x by 5%, compress y & z accordingly.
        s = 1.05
        F_macro = np.diag([s, 1.0 / np.sqrt(s), 1.0 / np.sqrt(s)])
    elif args.F_mode == "shear":
        # Pure simple shear in xy plane.
        F_macro = np.array([[1.0, 0.05, 0.0],
                            [0.0, 1.0,  0.0],
                            [0.0, 0.0,  1.0]])
    else:  # general
        # General F with all 9 entries non-trivial.
        F_macro = np.array([[1.10, 0.05, 0.02],
                            [0.03, 0.95, 0.04],
                            [0.01, 0.02, 1.05]])

    if rank == 0:
        print("=" * 76)
        print(f"  Phase 3.1 patch test - 3D linear-elastic homogeneous RVE")
        print(f"  (NO mortar, just corner Dirichlet u_lin = (F-I) X)")
        print("=" * 76)
        print(f"  mesh-type: {args.mesh_type}")
        print(f"  cells:     {args.nx} x {args.ny} x {args.nz}  on cube of side {args.L}")
        print(f"  F-mode:    {args.F_mode}")
        print(f"  F_macro =")
        for row in F_macro:
            print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
        print(f"  E = {args.E:.3e}, nu = {args.nu}")
        print(f"  np = {nranks}")
        print()

    # ----- Mesh + ParMesh -----
    # Each rank builds the same serial mesh (cheap; the partitioner does the
    # work). For very large RVEs, we'd switch to MFEM's distributed mesh
    # readers; for the prototype, the serial-mesh-then-partition pattern
    # mirrors the established 2D approach.
    mesh_serial = build_3d_box_mesh(
        args.mesh_type, args.nx, args.ny, args.nz, args.L,
    )
    pmesh = mfem.ParMesh(comm, mesh_serial)

    # CRITICAL: ``ParMesh::GetGlobalNE()`` does an internal MPI_Allreduce
    # over the ParMesh communicator (it sums the per-rank element count
    # across ranks). Calling it inside ``if rank == 0:`` strands rank 0
    # in the Allreduce while ranks 1..N-1 fly past and enter the next
    # collective (``ParFiniteElementSpace`` below) alone — classic
    # rank-asymmetric-collective deadlock at np > 1. Same warning as the
    # 2D driver's lines 649-654: rank-0-only I/O can be sandwiched between
    # collectives, but the COLLECTIVE itself must run on all ranks.
    n_global_elements = pmesh.GetGlobalNE()   # COLLECTIVE — all ranks
    if rank == 0:
        print(f"  ParMesh:  global elements = {n_global_elements} ({args.mesh_type})")

    # ----- FE space (vector H1, vdim=3) -----
    # Use Ordering::byNODES to match the 2D prototype convention.
    fec = mfem.H1_FECollection(1, pmesh.Dimension())
    fes = mfem.ParFiniteElementSpace(pmesh, fec, pmesh.Dimension())
    n_global_tdofs = fes.GlobalTrueVSize()
    n_local_tdofs = fes.GetTrueVSize()
    if rank == 0:
        print(f"  FES:      global TDOFs = {n_global_tdofs}, "
              f"vdim = {fes.GetVDim()}, ordering = {fes.GetOrdering()}")
        print()

    # ----- Identify the 8 corners (for diagnostic; not used as Dirichlet set) -----
    # Phase 3.4 will use these as the essential set; here we only check
    # that find_corners_3d works on hex AND tet meshes — Phase 3.1's
    # Dirichlet set is the FULL boundary.
    corners = find_corners_3d(pmesh, fes)
    if rank == 0:
        print(f"  Corners:  found 8 corners at the 8 box vertices  "
              f"(for diagnostic; Phase 3.1 pins ALL of ∂Ω)")

    # ----- u_lin = (F-I) X projected onto FES -----
    u_lin_local = apply_linear_part(fes, F_macro)

    # ----- Assemble K (linear elastic, distributed HypreParMatrix) -----
    K_hyp = assemble_linear_elastic_K_hypre(pmesh, fes, E=args.E, nu=args.nu)

    # ----- Newton-step: r1 = K . u_lin (full operator, before elimination) -----
    # For homogeneous material with affine u_lin:
    #   * Interior basis functions N_i (compactly supported, ∫∇N_i dV = 0):
    #       (K · u_lin)_i = σ_const : ∫∇N_i dV = 0  ⇒ assembly noise.
    #   * Boundary basis functions:
    #       (K · u_lin)_i = σ_const : ∫_∂(supp N_i) N_i n dS  ≠ 0
    #       (this is the integrated boundary traction σ·n).
    # So we EXPECT ‖r1‖_inf to be O(σ_const) ~ O(E·|F-I|) on the boundary.
    # That's correct and harmless: those rows are about to be Dirichlet-
    # eliminated anyway. The interior rows of r1 are the only ones that
    # matter, and they should be at the noise floor.
    r1_par = newton_residual_at_u_lin(K_hyp, u_lin_local)

    # ----- Apply FULL-boundary Dirichlet -----
    # Get every boundary TDOF (all vector components, all 6 faces) on
    # this rank, in global indices. Each rank passes its own subset;
    # apply_dirichlet_to_distributed_K filters by ownership internally.
    boundary_global_tdofs = find_all_boundary_tdofs(pmesh, fes)

    # Allreduce on all ranks (NOT inside if rank == 0) to get a global
    # count for the diagnostic print. Calling Allreduce only on rank 0
    # would deadlock — see the GetGlobalNE() comment earlier.
    n_bdr_global = comm.allreduce(len(boundary_global_tdofs), op=MPI.SUM)
    if rank == 0:
        print(f"  Dirichlet: {n_bdr_global} boundary TDOFs (global; full-∂Ω at u_lin)")

    # f_at_essential=None  =>  homogeneous Dirichlet on du
    # (i.e. du[boundary] = 0). This is correct because u_init = u_lin
    # already on the boundary, and we want u_new[boundary] = u_lin
    # (no movement).
    apply_dirichlet_to_distributed_K(
        K_hyp, r1_par, boundary_global_tdofs, fes,
        f_at_essential=None,
    )

    # ----- Solve K_eliminated . du = -r1 -----
    # After full-boundary elimination, the free-DOF system is
    # K_ii · du_i = -(K · u_lin)_i. For homogeneous material the RHS
    # is zero to assembly noise, and du_i = 0 is the unique solution.
    r1_par *= -1.0

    # CG + AMG: K is SPD after corner elimination.
    amg = mfem.HypreBoomerAMG(K_hyp)
    amg.SetSystemsOptions(pmesh.Dimension())
    amg.SetPrintLevel(0)

    cg = mfem.CGSolver(comm)
    cg.SetRelTol(1e-12)
    cg.SetAbsTol(0.0)
    cg.SetMaxIter(2000)
    cg.SetPrintLevel(0)
    cg.SetPreconditioner(amg)
    cg.SetOperator(K_hyp)

    du_par = mfem.Vector(n_local_tdofs)
    du_par.Assign(0.0)
    cg.Mult(r1_par, du_par)

    converged = bool(cg.GetConverged())
    iters = int(cg.GetNumIterations())
    final_norm = float(cg.GetFinalNorm())

    if rank == 0:
        print(f"  Solve:    CG+AMG iters = {iters}, converged = {converged}, "
              f"||r||_2 = {final_norm:.3e}")

    # ----- Update: u = u_lin + du -----
    du_local = np.array(du_par.GetDataArray(), dtype=np.float64)
    u_local = u_lin_local + du_local

    # ----- PASS CHECK 1: ||du||_inf ~ 0 (i.e. u ~ u_lin) -----
    du_inf_global = comm.allreduce(float(np.max(np.abs(du_local))), op=MPI.MAX)

    if rank == 0:
        print()
        print(f"  ||du||_inf =  {du_inf_global:.3e}  "
              f"(target < 1e-10; equivalent to ||u - u_lin||_inf)")

    pass_du = du_inf_global < 1e-10

    # ----- PASS CHECK 2: <F> = F_macro to machine precision -----
    u_par = mfem.Vector(u_local.tolist())
    F_avg = compute_volume_averaged_F(pmesh, fes, u_par)
    F_err = float(np.max(np.abs(F_avg - F_macro)))

    if rank == 0:
        print(f"  |<F> - F_macro|_max  = {F_err:.3e}  (target < 1e-12)")

    pass_F = F_err < 1e-12

    # ----- Optional ParaView output -----
    if args.paraview:
        from mortar_pbc import write_pbc_visualization
        u_lin_par = mfem.Vector(u_lin_local.tolist())
        # u_par built above for compute_volume_averaged_F; reuse it.
        # du_par was built earlier and consumed by cg.Mult; rebuild from
        # du_local for clean lifetime.
        du_par_for_viz = mfem.Vector(du_local.tolist())
        out_dir = args.paraview_dir
        if rank == 0 and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        comm.Barrier()
        F_label = (
            f"F=[[{F_macro[0,0]:.3f},{F_macro[0,1]:.3f},{F_macro[0,2]:.3f}],"
            f"[{F_macro[1,0]:.3f},{F_macro[1,1]:.3f},{F_macro[1,2]:.3f}],"
            f"[{F_macro[2,0]:.3f},{F_macro[2,1]:.3f},{F_macro[2,2]:.3f}]]"
        )
        write_pbc_visualization(
            pmesh, fes, u_par, u_lin_par, du_par_for_viz,
            output_dir=out_dir,
            name=f"phase31_{args.mesh_type}",
            F_label=F_label,
        )
        if rank == 0:
            print(f"  ParaView: wrote phase31_{args.mesh_type}.pvd in {out_dir}/")
            print(f"            (cycle 0 = reference; cycle 1 = deformed by u)")

    # ----- Summary -----
    if rank == 0:
        print()
        all_pass = pass_du and pass_F and converged
        status = "PASS" if all_pass else "FAIL"
        print(f"  ===== Phase 3.1 patch test ({args.mesh_type}): {status} =====")
        print()

    return 0 if (pass_du and pass_F and converged) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-type", choices=["hex", "tet"], default="hex",
                        help="3D mesh element type (default: hex)")
    parser.add_argument("--nx", type=int, default=4, help="Cells in x")
    parser.add_argument("--ny", type=int, default=4, help="Cells in y")
    parser.add_argument("--nz", type=int, default=4, help="Cells in z")
    parser.add_argument("--L", type=float, default=1.0, help="Cube side length")
    parser.add_argument("--F-mode", choices=["uniaxial", "shear", "general"],
                        default="general",
                        help="Macroscopic deformation gradient pattern")
    parser.add_argument("--E", type=float, default=70.0e3, help="Young's modulus")
    parser.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio")
    parser.add_argument(
        "--paraview", action="store_true",
        help="Write a ParaView .pvd collection (reference + deformed cycles) "
             "with u, u_lin, du fields for visual verification.",
    )
    parser.add_argument(
        "--paraview-dir", type=str, default="phase31_paraview",
        help="Output directory for ParaView files (default: phase31_paraview)",
    )
    args = parser.parse_args()
    return run_phase31(args)


if __name__ == "__main__":
    sys.exit(main())
