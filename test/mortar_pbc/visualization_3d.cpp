// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of WriteVisualization. See header for
// design doc. Mirrors `mortar_pbc/visualization.py`'s single-step
// `write_pbc_visualization` path.

#include "visualization_3d.hpp"

#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <filesystem>
#include <string>

namespace mortar_pbc {

namespace {

//==============================================================================
// Build a per-element constant grid function (one DOF per element)
// holding each element's attribute as a double. Used for colour-
// coding material regions in ParaView, mirroring the Python helper
// `_build_material_gridfunction`.
//==============================================================================
//
// The returned GridFunction owns nothing of the FE collection / FE
// space; the caller passes those in by reference and owns their
// lifetime. We allocate the GridFunction on the heap and let the
// caller manage it via unique_ptr in the call site.
mfem::ParGridFunction* MakeMaterialGridFunction(
    mfem::ParMesh& pmesh,
    mfem::L2_FECollection& l2_fec,
    mfem::ParFiniteElementSpace& l2_fes)
{
    auto* gf = new mfem::ParGridFunction(&l2_fes);
    *gf = 0.0;
    // L2 order-0 has exactly one DOF per element; the DOF index
    // matches the element index for byNODES ordering.
    const int n_loc_elems = pmesh.GetNE();
    for (int e = 0; e < n_loc_elems; ++e)
    {
        mfem::Array<int> dofs;
        l2_fes.GetElementDofs(e, dofs);
        // Should be exactly one DOF; defensive in case of refinement.
        const double attr = static_cast<double>(pmesh.GetAttribute(e));
        for (int i = 0; i < dofs.Size(); ++i)
        {
            (*gf)[dofs[i]] = attr;
        }
    }
    (void)l2_fec;  // silence unused-arg in case the L2 type isn't queried
    return gf;
}

//==============================================================================
// Snapshot the mesh's nodal TDOFs so we can restore at end of call.
//==============================================================================
void SnapshotNodes(mfem::ParMesh& pmesh, mfem::Vector& out_ref_tdofs)
{
    mfem::GridFunction* nodes_gf = pmesh.GetNodes();
    MFEM_VERIFY(nodes_gf != nullptr,
                "WriteVisualization: pmesh.GetNodes() returned null after "
                "SetCurvature; the mesh has no nodal grid function.");
    nodes_gf->GetTrueDofs(out_ref_tdofs);
}

//==============================================================================
// Restore the mesh to its reference configuration from a snapshot.
//==============================================================================
void RestoreNodes(mfem::ParMesh& pmesh, const mfem::Vector& ref_tdofs)
{
    mfem::GridFunction* nodes_gf = pmesh.GetNodes();
    MFEM_VERIFY(nodes_gf != nullptr,
                "WriteVisualization: pmesh.GetNodes() returned null during "
                "restore step.");
    // SetFromTrueDofs takes a non-const Vector& by API; copy into a
    // local non-const vector to satisfy the signature without
    // const_cast.
    mfem::Vector tmp(ref_tdofs.Size());
    for (int i = 0; i < ref_tdofs.Size(); ++i) { tmp(i) = ref_tdofs(i); }
    nodes_gf->SetFromTrueDofs(tmp);
    pmesh.NodesUpdated();
}

//==============================================================================
// Warp the mesh: nodes_tdofs += u_tdofs; SetFromTrueDofs; NodesUpdated.
//==============================================================================
void WarpMeshBy(mfem::ParMesh& pmesh,
                mfem::ParFiniteElementSpace& fes,
                const mfem::Vector& u_tdofs)
{
    mfem::GridFunction* nodes_gf = pmesh.GetNodes();
    MFEM_VERIFY(nodes_gf != nullptr,
                "WriteVisualization: pmesh.GetNodes() returned null during "
                "warp step.");
    mfem::FiniteElementSpace* nodes_fes = nodes_gf->FESpace();
    MFEM_VERIFY(nodes_fes->GetOrdering() == fes.GetOrdering(),
                "WriteVisualization: mesh-node ordering ("
                << static_cast<int>(nodes_fes->GetOrdering())
                << ") does not match displacement-FES ordering ("
                << static_cast<int>(fes.GetOrdering()) << "). "
                "SetCurvature should have been called with the FES's "
                "ordering — this is a logic error in the visualization "
                "helper.");

    mfem::Vector nodes_tdofs;
    nodes_gf->GetTrueDofs(nodes_tdofs);
    MFEM_VERIFY(nodes_tdofs.Size() == u_tdofs.Size(),
                "WriteVisualization: mesh-node TDOF count ("
                << nodes_tdofs.Size() << ") != displacement TDOF count ("
                << u_tdofs.Size() << "). The displacement FES and the "
                "mesh's nodal FES must have the same vdim and the same "
                "global TDOF count.");

    for (int i = 0; i < nodes_tdofs.Size(); ++i)
    {
        nodes_tdofs(i) += u_tdofs(i);
    }
    nodes_gf->SetFromTrueDofs(nodes_tdofs);
    pmesh.NodesUpdated();
}

}  // anonymous namespace

//==============================================================================
// WriteVisualization (single-step convenience)
//==============================================================================

void WriteVisualization(mfem::ParMesh& pmesh,
                        mfem::ParFiniteElementSpace& fes,
                        const mfem::Vector& u_total,
                        const mfem::Vector& u_lin,
                        const mfem::Vector& du,
                        const std::string& output_dir,
                        const std::string& name)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::visualization::write");

    MPI_Comm comm = pmesh.GetComm();
    int rank;
    MPI_Comm_rank(comm, &rank);

    //---- Promote mesh to nodal form (no-op if already nodal) ----
    // SetCurvature(order, discontinuous, space_dim, ordering):
    //   * order = 1 -> linear nodal field (matches H1_FECollection(1))
    //   * discontinuous = false (continuous H1)
    //   * space_dim = -1 -> default to mesh dim
    //   * ordering = match the displacement FES so per-component DOF
    //     indices line up between the node GF and u_total.
    pmesh.SetCurvature(/*order=*/1, /*discontinuous=*/false,
                       /*space_dim=*/-1,
                       /*ordering=*/static_cast<int>(fes.GetOrdering()));

    //---- Snapshot the reference (undeformed) node coordinates ----
    mfem::Vector ref_node_tdofs;
    SnapshotNodes(pmesh, ref_node_tdofs);

    //---- Create output directory on rank 0; barrier ----
    if (rank == 0)
    {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        // create_directories does not error if the dir already exists;
        // ec is set only on actual filesystem errors. Tolerate the
        // already-exists case silently.
    }
    MPI_Barrier(comm);

    //---- Build pre-allocated grid functions for the four fields ----
    mfem::ParGridFunction gf_u(&fes);
    mfem::ParGridFunction gf_u_lin(&fes);
    mfem::ParGridFunction gf_u_tilde(&fes);

    mfem::L2_FECollection l2_fec(/*order=*/0, pmesh.Dimension());
    mfem::ParFiniteElementSpace l2_fes(&pmesh, &l2_fec);
    std::unique_ptr<mfem::ParGridFunction> gf_mat(
        MakeMaterialGridFunction(pmesh, l2_fec, l2_fes));

    //---- Build the ParaView collection ----
    mfem::ParaViewDataCollection pv_dc(name, &pmesh);
    pv_dc.SetPrefixPath(output_dir);
    pv_dc.SetLevelsOfDetail(1);
    pv_dc.SetHighOrderOutput(false);
    pv_dc.RegisterField("u_total", &gf_u);
    pv_dc.RegisterField("u_lin",   &gf_u_lin);
    pv_dc.RegisterField("u_tilde", &gf_u_tilde);
    pv_dc.RegisterField("material", gf_mat.get());

    //---- Cycle 0: undeformed reference, all displacement fields zero ----
    {
        mfem::Vector zero(u_total.Size());
        zero = 0.0;
        gf_u.SetFromTrueDofs(zero);
        gf_u_lin.SetFromTrueDofs(zero);
        gf_u_tilde.SetFromTrueDofs(zero);
        // Mesh is already at the reference (we just snapshotted it).
        pv_dc.SetCycle(0);
        pv_dc.SetTime(0.0);
        pv_dc.Save();
    }

    //---- Cycle 1: deformed; warp mesh by u_total ----
    {
        // Need non-const views because SetFromTrueDofs takes Vector& by
        // API. Make local copies — these are TDOF vectors so the size
        // is local-rank-bounded, not large.
        mfem::Vector u_local(u_total.Size());
        for (int i = 0; i < u_total.Size(); ++i) { u_local(i) = u_total(i); }
        mfem::Vector u_lin_local(u_lin.Size());
        for (int i = 0; i < u_lin.Size(); ++i) { u_lin_local(i) = u_lin(i); }
        mfem::Vector du_local(du.Size());
        for (int i = 0; i < du.Size(); ++i) { du_local(i) = du(i); }

        gf_u.SetFromTrueDofs(u_local);
        gf_u_lin.SetFromTrueDofs(u_lin_local);
        gf_u_tilde.SetFromTrueDofs(du_local);

        WarpMeshBy(pmesh, fes, u_total);

        pv_dc.SetCycle(1);
        pv_dc.SetTime(1.0);
        pv_dc.Save();
    }

    //---- CRITICAL: restore mesh to reference before returning ----
    RestoreNodes(pmesh, ref_node_tdofs);
}

}  // namespace mortar_pbc
