"""Minimal NeoHookean integrator diagnostic on a 2x2 mesh.

Strips away PBC, constraints, parallelism, heterogeneity -- just calls
``HyperelasticNLFIntegrator(NeoHookeanModel(...))`` on a 2x2 unit-square
mesh with both materials, then with each material individually, and
prints the full stiffness matrix and Mult output at u=0.

We compare four configurations:
    1. NeoHookean(mu_const, K_const)              -- scalar constants
    2. NeoHookean(mu_pwc_uniform, K_pwc_uniform)  -- PWConstCoefficient
                                                     with same value on
                                                     both attributes
    3. NeoHookean(mu_pwc_5x, K_pwc_5x)            -- PWConstCoefficient
                                                     with 5x contrast
    4. NeoHookean(mu_const, K_const) on a single-attribute mesh
                                                  -- baseline sanity check

If config 1 works and config 2 fails, the bug is in PWConstCoefficient
plumbing.  If config 4 works and config 1 fails, the bug is in
multi-attribute mesh handling regardless of coefficient type.

Run:
    python examples/diag_neohookean_2x2.py
"""

import sys
import numpy as np
import mfem.par as mfem
from mpi4py import MPI


def build_2x2_mesh(L: float = 1.0, two_attributes: bool = True) -> mfem.Mesh:
    """Build a 2x2 quad mesh on [0, L]^2 with optional left/right
    attribute split.  Uses the same factory as the production drivers:
    ``Mesh.MakeCartesian2D(nx, ny, type, generate_edges, sx, sy)``."""
    mesh = mfem.Mesh.MakeCartesian2D(
        2, 2, mfem.Element.QUADRILATERAL, True, L, L,
    )
    if two_attributes:
        L_half = 0.5 * L
        for e in range(mesh.GetNE()):
            verts = [int(v) for v in mesh.GetElementVertices(e)]
            xs = [mesh.GetVertexArray(v)[0] for v in verts]
            x_centroid = sum(xs) / len(xs)
            mesh.SetAttribute(e, 1 if x_centroid < L_half else 2)
    mesh.SetAttributes()
    return mesh


def stats(arr_np: np.ndarray, label: str) -> None:
    n_nan    = int(np.sum(np.isnan(arr_np)))
    n_inf    = int(np.sum(np.isinf(arr_np)))
    n_finite = int(arr_np.size) - n_nan - n_inf
    if n_finite > 0:
        ff = arr_np[np.isfinite(arr_np)]
        amax = float(np.max(np.abs(ff)))
        amin = float(np.min(ff))
        amax_signed = float(np.max(ff))
    else:
        amax = amin = amax_signed = float("nan")
    print(f"    {label:48s}  n={int(arr_np.size):3d}  "
          f"finite={n_finite:3d}  nan={n_nan:3d}  inf={n_inf:3d}  "
          f"min={amin:+.3e}  max={amax_signed:+.3e}  |max|={amax:.3e}")


def build_nlf(fes: mfem.ParFiniteElementSpace,
              mu_coef, K_coef) -> mfem.ParNonlinearForm:
    nh = mfem.NeoHookeanModel(mu_coef, K_coef)
    nlf = mfem.ParNonlinearForm(fes)
    nlf.AddDomainIntegrator(mfem.HyperelasticNLFIntegrator(nh))
    return nlf, nh


def build_nlf_scalar(fes: mfem.ParFiniteElementSpace,
                     mu_value: float, K_value: float):
    """Build NLF using the SCALAR NeoHookeanModel(double, double)
    constructor -- mirroring ex10p's pattern exactly."""
    nh = mfem.NeoHookeanModel(mu_value, K_value)
    nlf = mfem.ParNonlinearForm(fes)
    nlf.AddDomainIntegrator(mfem.HyperelasticNLFIntegrator(nh))
    return nlf, nh


def run_config(name: str, fes: mfem.ParFiniteElementSpace,
               mu_coef, K_coef, n_tdof: int, comm) -> None:
    rank = comm.Get_rank()
    nlf, nh = build_nlf(fes, mu_coef, K_coef)
    _run_one(name, nlf, n_tdof, comm)


def run_config_scalar(name: str, fes: mfem.ParFiniteElementSpace,
                      mu_value: float, K_value: float, n_tdof: int,
                      comm) -> None:
    rank = comm.Get_rank()
    nlf, nh = build_nlf_scalar(fes, mu_value, K_value)
    _run_one(name, nlf, n_tdof, comm)


def _run_one(name: str, nlf: mfem.ParNonlinearForm, n_tdof: int, comm) -> None:
    rank = comm.Get_rank()

    # Test at u = 0 (undeformed reference state)
    u  = mfem.Vector(n_tdof); u.Assign(0.0)
    r  = mfem.Vector(n_tdof); r.Assign(float("nan"))
    if rank == 0:
        print(f"\n  --- Config: {name} ---")

    try:
        nlf.Mult(u, r)
        r_np = np.array(r.GetDataArray(), dtype=np.float64).copy()
        if rank == 0:
            stats(r_np, "Mult(u=0) residual")
    except Exception as e:
        if rank == 0:
            print(f"    Mult(u=0) RAISED: {type(e).__name__}: {e}")
        return

    # Test gradient at u = 0 (initial stiffness K0).
    try:
        K_op = nlf.GetGradient(u)
        if rank == 0:
            print(f"    GetGradient(u=0) returned: {type(K_op).__name__}")
    except Exception as e:
        if rank == 0:
            print(f"    GetGradient(u=0) RAISED: {type(e).__name__}: {e}")
        return

    # Try to extract K's diagonal.
    diag = mfem.Vector(n_tdof); diag.Assign(0.0)
    try:
        K_op.AssembleDiagonal(diag)
        d_np = np.array(diag.GetDataArray(), dtype=np.float64).copy()
        if rank == 0:
            stats(d_np, "diag(K0) via AssembleDiagonal")
    except Exception as e:
        if rank == 0:
            print(f"    AssembleDiagonal RAISED: {type(e).__name__}: {e}")
            try:
                K_op.GetDiag(diag)
                d_np = np.array(diag.GetDataArray(), dtype=np.float64).copy()
                stats(d_np, "diag(K0) via GetDiag")
            except Exception as e2:
                print(f"    GetDiag RAISED: {type(e2).__name__}: {e2}")

    # Print K_op @ e_0  ... K_op @ e_{N-1}  to dump the whole matrix.
    if rank == 0 and n_tdof <= 18:        # only for small meshes
        print(f"    K0 dump (each col = K0 @ e_i):")
        ej = mfem.Vector(n_tdof); ej.Assign(0.0)
        Kj = mfem.Vector(n_tdof)
        for j in range(n_tdof):
            ej.Assign(0.0)
            ej[j] = 1.0
            try:
                K_op.Mult(ej, Kj)
                col = np.array(Kj.GetDataArray(), dtype=np.float64).copy()
                col_str = " ".join(f"{c:+.2e}" for c in col)
                n_nan = int(np.sum(np.isnan(col)))
                tag = "NAN" if n_nan > 0 else "ok "
                print(f"      [{tag}] col {j:2d}:  {col_str}")
            except Exception as e:
                print(f"      col {j:2d}: RAISED {type(e).__name__}: {e}")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print(f"=== Minimal NeoHookean integrator diagnostic (rank {rank}) ===")

    # ---- Build a 2x2 mesh with two attributes (left/right strip) ----
    L = 1.0
    smesh = build_2x2_mesh(L=L, two_attributes=True)
    pmesh = mfem.ParMesh(comm, smesh)

    fec = mfem.H1_FECollection(1, 2)
    fes = mfem.ParFiniteElementSpace(pmesh, fec, 2)        # vdim=2
    n_tdof = fes.GetTrueVSize()
    if rank == 0:
        print(f"\n  Mesh: 2x2 quads, {pmesh.GetNE()} elements, "
              f"vdim=2, n_tdof={n_tdof}")
        attrs = sorted(set(pmesh.GetAttribute(e) for e in range(pmesh.GetNE())))
        print(f"  Attributes: {attrs}")

    # ---- Compute material parameters for E=70e3, nu=0.3 ----
    E_baseline   = 70.0e3
    nu_baseline  = 0.3
    mu_value     = E_baseline / (2.0 * (1.0 + nu_baseline))
    K_value      = E_baseline / (3.0 * (1.0 - 2.0 * nu_baseline))
    if rank == 0:
        print(f"  Reference material: mu={mu_value:.3e}, K={K_value:.3e}")

    # ---- Config 1: scalar ConstantCoefficient ----
    mu_const = mfem.ConstantCoefficient(mu_value)
    K_const  = mfem.ConstantCoefficient(K_value)
    run_config("1. NeoHookean(mu_const, K_const)",
               fes, mu_const, K_const, n_tdof, comm)

    # ---- Config 2: PWConstCoefficient with same value on both attrs ----
    mu_vec_unif = mfem.Vector([mu_value, mu_value])
    K_vec_unif  = mfem.Vector([K_value,  K_value])
    mu_pwc_unif = mfem.PWConstCoefficient(mu_vec_unif)
    K_pwc_unif  = mfem.PWConstCoefficient(K_vec_unif)
    run_config("2. NeoHookean(PWC_uniform)  -- same val on both attrs",
               fes, mu_pwc_unif, K_pwc_unif, n_tdof, comm)

    # ---- Config 3: PWConstCoefficient with 5x contrast ----
    mu_vec_5x = mfem.Vector([mu_value,       5.0 * mu_value])
    K_vec_5x  = mfem.Vector([K_value,        5.0 * K_value])
    mu_pwc_5x = mfem.PWConstCoefficient(mu_vec_5x)
    K_pwc_5x  = mfem.PWConstCoefficient(K_vec_5x)
    run_config("3. NeoHookean(PWC_5x)       -- 5x contrast",
               fes, mu_pwc_5x, K_pwc_5x, n_tdof, comm)

    # ---- Config 4: scalar coefficient, single-attribute mesh ----
    smesh4 = build_2x2_mesh(L=L, two_attributes=False)
    pmesh4 = mfem.ParMesh(comm, smesh4)
    fes4   = mfem.ParFiniteElementSpace(pmesh4, fec, 2)
    n_tdof4 = fes4.GetTrueVSize()
    if rank == 0:
        print(f"\n  Single-attribute mesh: n_tdof={n_tdof4}")
    mu_const4 = mfem.ConstantCoefficient(mu_value)
    K_const4  = mfem.ConstantCoefficient(K_value)
    run_config("4. NeoHookean(mu_const, K_const)  on single-attr mesh",
               fes4, mu_const4, K_const4, n_tdof4, comm)

    # ---- Config 5: SCALAR floats (mirroring ex10p exactly) ----
    # ex10p builds ``mfem.NeoHookeanModel(mu, K)`` with PYTHON FLOATS,
    # not Coefficient objects.  This tests whether the SWIG-wrapped
    # ``NeoHookeanModel(double, double)`` constructor works while the
    # ``NeoHookeanModel(Coefficient&, Coefficient&)`` overload is broken.
    run_config_scalar(
        "5. NeoHookean(mu_VALUE, K_VALUE)  scalar-float ctor (ex10p pattern)",
        fes4, mu_value, K_value, n_tdof4, comm)


if __name__ == "__main__":
    main()
