"""Diagnostic utilities for mortar PBC patch tests.

Currently exposes ``volume_averaged_F``, which computes the
volume-averaged deformation gradient

    bar_F = (1/|Omega|) * integral_Omega(grad u + I) dV
          = I + (1/|Omega|) * integral_Omega(grad u) dV

over the RVE.  By the homogenization theorem (Hill-Mandel / divergence
theorem), this should equal the prescribed macroscopic F to roughly
machine precision when the periodic boundary conditions are correctly
enforced -- it's a clean integral check that the mortar machinery is
delivering the macroscopic deformation faithfully.

Why this is a good check
------------------------
Equivalent surface form:
    bar_F = I + (1/|Omega|) * integral_dOmega(u (x) n) dS
With strict periodicity, the boundary integral picks up exactly the
prescribed corner displacements multiplied by their associated edge
lengths and the outward normals, giving F_macro identically.  With
mortar (weak periodicity), the result is no longer identically equal
but should differ by O(machine precision) on a properly assembled
problem -- significantly larger errors indicate a bug in the
constraint, not a discretization artifact.

We use the volume form because it doesn't depend on having the
boundary parameterization right and works the same whether the mesh
is conforming or not.
"""
from __future__ import annotations

import numpy as np
import mfem.par as mfem
from mpi4py import MPI


def volume_averaged_F(
    pmesh: mfem.ParMesh,
    fes:   mfem.ParFiniteElementSpace,
    u_par: mfem.Vector,
) -> np.ndarray:
    """Compute the volume-averaged deformation gradient over the RVE.

    Parameters
    ----------
    pmesh
        Parallel mesh.
    fes
        H1 vdim=d displacement FE space corresponding to ``u_par``.
    u_par
        True-DOF vector of the total displacement field.

    Returns
    -------
    bar_F : np.ndarray, shape (d, d)
        bar_F = I + (1/|Omega|) * integral_Omega(grad u) dV.
        Identical on every rank (Allreduce'd).

    Notes
    -----
    Quadrature: each element is integrated using its native FE order
    plus 1 for safety.  For our linear H1 quad meshes that's order 2
    Gauss product (4 points per quad), more than enough for an
    integral of ``grad u`` (which is constant per quadrilateral element
    -- but we use an honest quadrature loop so the routine works
    unchanged on higher-order meshes too).
    """
    comm = MPI.COMM_WORLD
    dim  = pmesh.Dimension()

    # Build a ParGridFunction wrapper around u_par so we can evaluate
    # its gradient at quadrature points using native MFEM machinery.
    gf_u = mfem.ParGridFunction(fes)
    gf_u.SetFromTrueDofs(u_par)

    # Local accumulators on this rank.
    local_grad_u_int = np.zeros((dim, dim), dtype=np.float64)
    local_volume     = 0.0

    # Loop over local elements.  For each element we get the
    # ElementTransformation and a quadrature rule of sufficient order,
    # evaluate grad u at each quadrature point, and accumulate
    # weight * |J| * grad u  into local_grad_u_int.  Volume picks up
    # weight * |J| at the same quadrature points.
    grad_u_pt = mfem.DenseMatrix(dim, dim)
    for e in range(pmesh.GetNE()):
        Tr = pmesh.GetElementTransformation(e)
        fe = fes.GetFE(e)
        # Integration rule order: shape function gradient is order p-1
        # times Jacobian of order at most p-1 (linear quads => constants);
        # to integrate it safely take order = 2*p (overkill for linear,
        # exact for higher).
        order = 2 * fe.GetOrder()
        ir = mfem.IntRules.Get(fe.GetGeomType(), order)
        for q in range(ir.GetNPoints()):
            ip = ir.IntPoint(q)
            Tr.SetIntPoint(ip)
            # Evaluate grad u at this quadrature point.  GetVectorGradient
            # writes into a DenseMatrix of shape (vdim, dim).
            gf_u.GetVectorGradient(Tr, grad_u_pt)
            w_jac = ip.weight * Tr.Weight()
            for i in range(dim):
                for j in range(dim):
                    local_grad_u_int[i, j] += w_jac * grad_u_pt[i, j]
            local_volume += w_jac

    # Allreduce both quantities to rank 0 (and to all ranks, via
    # ``comm.allreduce`` so the return value is consistent on every
    # process).
    global_grad_u_int = np.zeros_like(local_grad_u_int)
    comm.Allreduce(local_grad_u_int, global_grad_u_int, op=MPI.SUM)
    global_volume = comm.allreduce(local_volume, op=MPI.SUM)

    if global_volume <= 0.0:
        raise RuntimeError(
            f"volume_averaged_F: total RVE volume is non-positive "
            f"({global_volume}); something is very wrong with the mesh."
        )

    bar_F = np.eye(dim) + global_grad_u_int / global_volume
    return bar_F


def report_F_diagnostic(
    bar_F: np.ndarray,
    F_macro: np.ndarray,
    rtol: float = 1.0e-10,
    label: str = "",
) -> bool:
    """Pretty-print ``bar_F`` against the prescribed ``F_macro`` and
    return True if the agreement is within ``rtol`` (relative).

    Designed for use at the end of a load step in patch-test drivers.
    """
    abs_err = np.max(np.abs(bar_F - F_macro))
    macro_norm = float(np.max(np.abs(F_macro)))
    rel_err = abs_err / macro_norm if macro_norm > 0.0 else abs_err

    title = f"Volume-averaged F diagnostic{(' (' + label + ')') if label else ''}"
    print()
    print(title)
    print("-" * len(title))
    print("  prescribed F_macro:")
    for row in F_macro:
        print(f"    [ {row[0]:+.6e}  {row[1]:+.6e} ]")
    print("  computed bar_F = I + (1/|Omega|) integral grad u dV:")
    for row in bar_F:
        print(f"    [ {row[0]:+.6e}  {row[1]:+.6e} ]")
    print(f"  ||bar_F - F_macro||_inf = {abs_err:.3e}  "
          f"(rel = {rel_err:.3e})")
    if rel_err < rtol:
        print(f"  PASS  matches to relative tolerance {rtol:.0e}")
        return True
    else:
        print(f"  FAIL  exceeds relative tolerance {rtol:.0e}")
        return False
