"""ParaView visualization helpers for mortar PBC drivers.

Wraps ``mfem.ParaViewDataCollection`` to dump two cycles per solve:
    * cycle 0 (time=0.0) : undeformed reference configuration with the
      affine field ``u_lin``, fluctuation ``u_tilde``, total displacement
      ``u_total``, and the per-element material attribute.
    * cycle 1 (time=1.0) : DEFORMED configuration -- mesh node
      coordinates updated by adding ``u_total`` so ParaView shows the
      actual deformed RVE without needing the user to apply a "Warp by
      Vector" filter post-hoc.

Open the ``solution.pvd`` file in ParaView and use the time slider to
flip between undeformed and deformed states.

API
---
Single entry point::

    write_pbc_visualization(
        pmesh, fes, u_par, u_lin_par, du_par,
        output_dir, name="solution", F_label=None,
    )

The caller is responsible for choosing the output directory; the
function creates it on rank 0 if it doesn't exist and synchronizes
across ranks before writing.

Notes on mesh-node update mechanics
-----------------------------------
By default an MFEM mesh built from ``Mesh.MakeCartesian2D`` stores
geometry as a vertex array (no nodal grid function).  ``GetNodes()``
returns ``nullptr`` in that case.  To attach a nodal grid function we
call ``SetCurvature(order=1, ordering=fes.GetOrdering())``.  After
that, ``GetNodes()`` returns a ``GridFunction`` whose values ARE the
node coordinates and whose component ordering matches the displacement
FE space; adding ``u_total`` to it (in TDOF space) shifts the mesh
correctly, and ``NodesUpdated()`` makes MFEM invalidate any cached
geometric factors.

**Ordering matters.**  By default ``ParFiniteElementSpace`` uses
``Ordering::byNODES`` while ``Mesh::SetCurvature`` uses ``byVDIM``.
Adding the displacement TDOF vector elementwise to the mesh-node
TDOF vector under a mismatch silently swaps x/y components and
produces a geometrically wrong deformed mesh.  The helper
``_ensure_nodal_with_matching_ordering`` reads the displacement FES's
ordering and passes it to ``SetCurvature`` to enforce parity.

For the visualization-only purpose we don't actually need to invalidate
geometric factors (we're not computing anything more on the deformed
mesh -- we're just dumping it), but calling ``NodesUpdated()`` keeps
the mesh in a consistent internal state.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import mfem.par as mfem
from mpi4py import MPI


def _ensure_nodal_with_matching_ordering(
    pmesh: mfem.ParMesh,
    fes: mfem.ParFiniteElementSpace,
) -> None:
    """Promote ``pmesh`` to nodal form with the SAME ordering convention
    as ``fes`` (the displacement FE space) so that adding a displacement
    TDOF vector to the mesh-node TDOF vector is component-aligned.

    Why this matters
    ----------------
    By default:
      * ``ParFiniteElementSpace(pmesh, fec, vdim)`` defaults to
        ``Ordering::byNODES`` (per FiniteElementSpace.hpp).
      * ``Mesh::SetCurvature(order)``               defaults to
        ``Ordering::byVDIM``  (per Mesh.cpp).
    If the displacement FES and the mesh-node FES disagree on ordering,
    adding a byNODES displacement vector elementwise to a byVDIM mesh-
    node vector silently swaps x/y components and produces a deformed
    mesh that is geometrically wrong.

    Strategy
    --------
    Read ``fes.GetOrdering()`` and pass it explicitly to
    ``SetCurvature(order=1, discont=False, space_dim=-1, ordering=...)``.
    For linear meshes (which is our case for the patch tests) order=1
    means one nodal DOF per FE-vertex; values equal vertex coordinates
    initially.  After this call, ``pmesh.GetNodes()`` returns a
    ParGridFunction whose FE space's ordering matches ``fes``.

    No-op if the mesh is already nodal AND its ordering matches.
    """
    fes_ordering = fes.GetOrdering()

    nodes = pmesh.GetNodes()
    if nodes is not None:
        # Already nodal -- check ordering compatibility.
        nodes_fes = nodes.FESpace()
        if nodes_fes.GetOrdering() == fes_ordering:
            return  # already aligned, nothing to do
        # Mismatched ordering on an already-promoted mesh; rebuild.

    # Promote (or re-promote) to nodal form with matching ordering.
    # SetCurvature signature (per MFEM 4.x):
    #     SetCurvature(int order, bool discont=false, int space_dim=-1,
    #                  int ordering=Ordering::byVDIM)
    pmesh.SetCurvature(1, False, -1, fes_ordering)


def _resolve_vtk_binary_format(mfem_module):
    """Return the BINARY VTKFormat enum value for this pyMFEM build.

    pyMFEM exposes nested enums under different names depending on the
    SWIG build: some builds use the C++-style ``mfem.VTKFormat.BINARY``,
    others flatten it as ``mfem.VTKFormat_BINARY``.  Try both; return
    None if neither is found (caller falls back to default BINARY).
    """
    for attr in ("VTKFormat_BINARY",):
        if hasattr(mfem_module, attr):
            return getattr(mfem_module, attr)
    if hasattr(mfem_module, "VTKFormat"):
        fmt_class = getattr(mfem_module, "VTKFormat")
        if hasattr(fmt_class, "BINARY"):
            return fmt_class.BINARY
    return None


def _build_material_gridfunction(pmesh: mfem.ParMesh) -> mfem.ParGridFunction:
    """Return an L2-order-0 grid function whose value on each element
    equals the element attribute (1, 2, ...)."""
    fec_l2 = mfem.L2_FECollection(0, pmesh.Dimension())
    fes_l2 = mfem.ParFiniteElementSpace(pmesh, fec_l2, 1)
    gf_mat = mfem.ParGridFunction(fes_l2)
    gf_mat.Assign(0.0)
    for e in range(pmesh.GetNE()):
        gf_mat[e] = float(pmesh.GetAttribute(e))
    # Keep the FE space alive by attaching it to the GridFunction;
    # otherwise it can be garbage-collected before Save() runs.
    gf_mat._keep_alive_fes  = fes_l2
    gf_mat._keep_alive_fec  = fec_l2
    return gf_mat


def write_pbc_visualization(
    pmesh: mfem.ParMesh,
    fes:   mfem.ParFiniteElementSpace,
    u_par:     mfem.Vector,
    u_lin_par: mfem.Vector,
    du_par:    mfem.Vector,
    output_dir: str,
    name: str = "solution",
    F_label: Optional[str] = None,
) -> None:
    """Single-step convenience wrapper around ``PbcVisualizationWriter``.

    Writes a two-cycle ParaView collection: cycle 0 = undeformed
    reference; cycle 1 = deformed (mesh nodes warped by ``u_total``).
    Equivalent to::

        writer = PbcVisualizationWriter(pmesh, fes, output_dir, name=name)
        writer.write_step(u_par, u_lin_par, du_par,
                          F_label=F_label, write_undeformed_first=True)
    """
    writer = PbcVisualizationWriter(pmesh, fes, output_dir, name=name)
    writer.write_step(u_par, u_lin_par, du_par,
                      F_label=F_label, write_undeformed_first=True)


class PbcVisualizationWriter:
    """Stateful ParaView writer for multi-step mortar-PBC simulations.

    Each call to :meth:`write_step` saves a new cycle (deformed
    configuration at the current step) to the same ``.pvd`` collection.
    Open the resulting collection in ParaView and use the time slider
    to step through the load increments.

    Mesh-node update mechanics
    --------------------------
    The mesh is promoted to a nodal form whose ordering matches the
    displacement FE space's ordering on the first call (no-op if
    already nodal-with-matching-ordering).  Each :meth:`write_step`
    call:

      1. Resets node coordinates to the captured reference snapshot.
      2. Warps by the supplied ``u_total`` and saves the cycle.
      3. RESTORES node coordinates to the reference snapshot before
         returning.

    Step 3 is critical: leaving the mesh in a deformed state would
    corrupt subsequent ``apply_linear_part`` projections (which
    evaluate ``(F-I) X`` using the mesh's current nodal coordinates as
    ``X``) and any assembly / integration that depends on element
    transformations.  By restoring the reference state, the writer
    becomes side-effect-free with respect to the mesh.

    Parameters
    ----------
    pmesh
        The parallel mesh.  Will be mutated by mesh-node updates.
    fes
        The H1 vector displacement FE space (vdim = 2 for 2D, vdim = 3
        for 3D).  Must have the same ordering as the mesh's nodal FE
        space (the helper enforces this on first call).
    output_dir
        Directory to write the ``<name>.pvd`` and per-rank ``.vtu``
        files into.  Created if it doesn't exist.
    name
        Collection name.  Default ``"solution"``.
    """

    def __init__(
        self,
        pmesh: mfem.ParMesh,
        fes:   mfem.ParFiniteElementSpace,
        output_dir: str,
        name: str = "solution",
    ) -> None:
        comm = pmesh.GetComm() if hasattr(pmesh, "GetComm") else MPI.COMM_WORLD
        rank = comm.Get_rank()

        _ensure_nodal_with_matching_ordering(pmesh, fes)

        # Snapshot the reference (undeformed) node coordinates so we
        # can RESET on each write_step call.  Without this, successive
        # warp-then-save calls would accumulate the displacement
        # additively, producing nonsense for any step beyond step 1.
        nodes_gf = pmesh.GetNodes()
        ref_nodes_tdofs = mfem.Vector()
        nodes_gf.GetTrueDofs(ref_nodes_tdofs)
        # Save a copy so subsequent operations don't alias.
        self._ref_nodes_np = np.array(
            ref_nodes_tdofs.GetDataArray(), dtype=np.float64, copy=True
        )

        # Set up output directory.
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        comm.Barrier()

        # Build the data collection ONCE; write_step appends cycles.
        pv_dc = mfem.ParaViewDataCollection(name, pmesh)
        pv_dc.SetPrefixPath(output_dir)
        pv_dc.SetLevelsOfDetail(1)
        fmt = _resolve_vtk_binary_format(mfem)
        if fmt is not None:
            try:
                pv_dc.SetDataFormat(fmt)
            except Exception:
                pass
        pv_dc.SetHighOrderOutput(False)

        # Pre-allocate the GridFunctions we'll register; we'll
        # SetFromTrueDofs into them on each call instead of rebuilding.
        self._gf_u       = mfem.ParGridFunction(fes)
        self._gf_u_lin   = mfem.ParGridFunction(fes)
        self._gf_u_tilde = mfem.ParGridFunction(fes)
        self._gf_mat     = _build_material_gridfunction(pmesh)

        pv_dc.RegisterField("u_total",  self._gf_u)
        pv_dc.RegisterField("u_lin",    self._gf_u_lin)
        pv_dc.RegisterField("u_tilde",  self._gf_u_tilde)
        pv_dc.RegisterField("material", self._gf_mat)

        self.pmesh = pmesh
        self.fes   = fes
        self.pv_dc = pv_dc
        self.output_dir = output_dir
        self.name = name
        self.next_cycle = 0
        self.comm = comm
        self.rank = rank

    def write_step(
        self,
        u_par:     mfem.Vector,
        u_lin_par: mfem.Vector,
        du_par:    mfem.Vector,
        time: Optional[float] = None,
        F_label: Optional[str] = None,
        write_undeformed_first: bool = False,
    ) -> None:
        """Write a deformed-configuration cycle for the current step.

        Parameters
        ----------
        u_par, u_lin_par, du_par
            Total / affine / fluctuation displacement true-DOF vectors.
        time
            ParaView "time" stamp for this cycle.  Defaults to the
            cycle number (0, 1, 2, ...).
        F_label
            Optional human-readable load case identifier
            (printed to rank-0 stdout).
        write_undeformed_first
            If True AND this is the very first write call, prepend
            cycle 0 = undeformed reference (with zero displacement
            fields).  Useful for replicating the single-step helper's
            two-cycle output.
        """
        if write_undeformed_first and self.next_cycle == 0:
            # Cycle 0 = undeformed reference.  Reset mesh nodes (no-op
            # on first call but defensive), zero the displacement
            # fields, write.
            self._reset_mesh_to_reference()
            zero_par = mfem.Vector(u_par.Size())
            zero_par.Assign(0.0)
            self._gf_u.SetFromTrueDofs(zero_par)
            self._gf_u_lin.SetFromTrueDofs(zero_par)
            self._gf_u_tilde.SetFromTrueDofs(zero_par)
            self.pv_dc.SetCycle(self.next_cycle)
            self.pv_dc.SetTime(0.0)
            self.pv_dc.Save()
            self.next_cycle += 1

        # Reset mesh to reference, then warp by the new u_total.
        self._reset_mesh_to_reference()
        self._gf_u.SetFromTrueDofs(u_par)
        self._gf_u_lin.SetFromTrueDofs(u_lin_par)
        self._gf_u_tilde.SetFromTrueDofs(du_par)
        self._warp_mesh_by(u_par)

        cycle = self.next_cycle
        t = float(time) if time is not None else float(cycle)
        self.pv_dc.SetCycle(cycle)
        self.pv_dc.SetTime(t)
        self.pv_dc.Save()
        self.next_cycle += 1

        # CRITICAL: restore the mesh to its REFERENCE configuration
        # before returning.  The writer must not leave the mesh in a
        # deformed state because:
        #   * ``apply_linear_part`` projects (F-I) X using the mesh's
        #     CURRENT nodal coordinates as X.  If the mesh is deformed
        #     when the next step calls ``apply_linear_part``, X is no
        #     longer the reference position and u_lin gets evaluated
        #     against deformed coordinates -- producing a u_lin that
        #     looks "more stretched" than it should be.
        #   * ``compute_volume_averaged_F`` evaluates ∫ ∇u dx using
        #     the current mesh's element transformations.  A deformed
        #     mesh changes the integration domain and the gradient
        #     reference frame, giving a numerically different (and
        #     physically wrong) <F>.
        #   * For nonlinear materials, K = nlf.GetGradient(u) gets
        #     re-assembled on every Newton iterate, and the assembly
        #     uses the current mesh's geometric factors.  A deformed
        #     mesh would make K correspond to a different reference
        #     configuration than the one the integrator expects.
        # This is the SMALL-STRAIN / TOTAL-LAGRANGIAN convention: all
        # FE operations (assembly, projection, integration, gradient
        # evaluation) are done on the REFERENCE mesh, and the deformed
        # mesh is purely a visualization artifact.
        self._reset_mesh_to_reference()

        if self.rank == 0:
            rel = os.path.relpath(self.output_dir, os.getcwd())
            tag = f" (F={F_label})" if F_label else ""
            print(f"    ParaView{tag}: cycle {cycle} (t={t:.3g}) -> {rel}")

    # ---------------------------------------------------------- private --

    def _reset_mesh_to_reference(self) -> None:
        nodes_gf = self.pmesh.GetNodes()
        ref_vec = mfem.Vector()
        nodes_gf.GetTrueDofs(ref_vec)        # allocate to right size
        for i in range(ref_vec.Size()):
            ref_vec[i] = float(self._ref_nodes_np[i])
        nodes_gf.SetFromTrueDofs(ref_vec)
        self.pmesh.NodesUpdated()

    def _warp_mesh_by(self, u_par: mfem.Vector) -> None:
        """Add u_par to the (already-reset) reference mesh nodes."""
        nodes_gf = self.pmesh.GetNodes()
        nodes_fes = nodes_gf.FESpace()
        assert nodes_fes.GetOrdering() == self.fes.GetOrdering(), (
            f"Mesh-node ordering ({nodes_fes.GetOrdering()}) != "
            f"displacement-FES ordering ({self.fes.GetOrdering()})."
        )
        nodes_tdofs = mfem.Vector()
        nodes_gf.GetTrueDofs(nodes_tdofs)
        n = nodes_tdofs.Size()
        if n != u_par.Size():
            raise RuntimeError(
                f"Mesh node TDOF count ({n}) != displacement TDOF "
                f"count ({u_par.Size()})."
            )
        for i in range(n):
            nodes_tdofs[i] = float(nodes_tdofs[i]) + float(u_par[i])
        nodes_gf.SetFromTrueDofs(nodes_tdofs)
        self.pmesh.NodesUpdated()
