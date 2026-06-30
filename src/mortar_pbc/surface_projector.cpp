// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors

#include "surface_projector.hpp"

#include "utilities/mechanics_log.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace mortar_pbc {

namespace {

using SnapCoordKey = std::array<long long, 3>;
using ComponentGtdofs = std::array<int, 3>;

constexpr int kVDim = 3;
constexpr int kParentPackStride = 6;  // snap key (3) + parent gtdofs (3)
constexpr int kSubmeshPackStride = 2; // submesh gtdof + parent gtdof

bool SameComponentGtdofs(const ComponentGtdofs& a, const ComponentGtdofs& b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

} // namespace

std::array<long long, 3> SurfaceProjector::SnapKey(const mfem::Vector& x,
                                                   double snap_tol)
{
    // MFEM may evaluate the same geometric node through two different
    // paths: parent boundary element transformation vs submesh vertex
    // coordinate storage. Quantizing each component avoids fragile
    // floating-point tuple comparison while keeping the lookup exact.
    return {static_cast<long long>(std::llround(x[0] / snap_tol)),
            static_cast<long long>(std::llround(x[1] / snap_tol)),
            static_cast<long long>(std::llround(x[2] / snap_tol))};
}

SurfaceProjector::SurfaceProjector(
    std::shared_ptr<const mfem::ParFiniteElementSpace> parent_fes,
    std::shared_ptr<const mfem::ParFiniteElementSpace> submesh_fes,
    std::shared_ptr<const mfem::ParMesh> submesh,
    double snap_tol)
    : mfem::Operator(submesh_fes->GetTrueVSize(),
                     parent_fes->GetTrueVSize())
    , m_parent_fes(std::move(parent_fes))
    , m_submesh_fes(std::move(submesh_fes))
    , m_submesh(std::move(submesh))
    , m_snap_tol(snap_tol)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::surface_projector::ctor");

    MFEM_VERIFY(m_parent_fes != nullptr,
                "SurfaceProjector: parent FES must be non-null.");
    MFEM_VERIFY(m_submesh_fes != nullptr,
                "SurfaceProjector: submesh FES must be non-null.");
    MFEM_VERIFY(m_submesh != nullptr,
                "SurfaceProjector: submesh must be non-null.");
    MFEM_VERIFY(m_parent_fes->GetVDim() == kVDim,
                "SurfaceProjector: parent FES must have vdim=3.");
    MFEM_VERIFY(m_submesh_fes->GetVDim() == kVDim,
                "SurfaceProjector: submesh FES must have vdim=3.");
    MFEM_VERIFY(m_parent_fes->GetOrdering() == mfem::Ordering::byNODES,
                "SurfaceProjector: parent FES must use byNODES ordering.");
    MFEM_VERIFY(m_submesh_fes->GetOrdering() == mfem::Ordering::byNODES,
                "SurfaceProjector: submesh FES must use byNODES ordering.");
    MFEM_VERIFY(m_submesh_fes->GetParMesh() == m_submesh.get(),
                "SurfaceProjector: submesh FES is not defined on the "
                "supplied submesh.");
    MFEM_VERIFY(m_snap_tol > 0.0,
                "SurfaceProjector: snap tolerance must be positive.");

    m_comm = m_parent_fes->GetComm();
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_nranks);

    {
        const int local_order =
            (m_submesh->GetNE() > 0) ? m_submesh_fes->GetOrder(0) : -1;
        const int local_bad =
            (local_order >= 0 && local_order != 1) ? local_order : -1;
        int global_bad = -1;
        MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, m_comm);
        MFEM_VERIFY(global_bad == -1,
                    "SurfaceProjector: submesh FES must be order 1.");
    }

    // Parent global true DOFs are block-partitioned by rank. Keep the
    // offsets locally so later setup code can ask who owns a mapped
    // parent true DOF without consulting Hypre.
    m_parent_tdof_offsets_all.assign(static_cast<std::size_t>(m_nranks + 1), 0);
    const HYPRE_BigInt my_start = m_parent_fes->GetTrueDofOffsets()[0];
    MPI_Allgather(&my_start, 1, HYPRE_MPI_BIG_INT,
                  m_parent_tdof_offsets_all.data(), 1,
                  HYPRE_MPI_BIG_INT, m_comm);
    m_parent_tdof_offsets_all[m_nranks] =
        static_cast<HYPRE_BigInt>(m_parent_fes->GlobalTrueVSize());

    BuildMap();
}

int SurfaceProjector::ParentOwnerRank(int parent_gtdof) const
{
    MFEM_ASSERT(parent_gtdof >= 0
                    && parent_gtdof < m_parent_fes->GlobalTrueVSize(),
                "SurfaceProjector::ParentOwnerRank: parent gtdof out of range.");
    const auto it = std::upper_bound(m_parent_tdof_offsets_all.begin(),
                                     m_parent_tdof_offsets_all.end(),
                                     static_cast<HYPRE_BigInt>(parent_gtdof));
    const int owner = static_cast<int>(
        (it - m_parent_tdof_offsets_all.begin()) - 1);
    MFEM_ASSERT(owner >= 0 && owner < m_nranks,
                "SurfaceProjector::ParentOwnerRank: invalid owner rank.");
    return owner;
}

int SurfaceProjector::ParentGtdof(int submesh_gtdof) const
{
    const auto it = m_map.submesh_to_parent_gtdof.find(submesh_gtdof);
    MFEM_VERIFY(it != m_map.submesh_to_parent_gtdof.end(),
                "SurfaceProjector: no parent TDOF for submesh TDOF "
                    << submesh_gtdof);
    return it->second;
}

int SurfaceProjector::ParentLocalOrMinus1(int submesh_gtdof) const
{
    const int parent_gtdof = ParentGtdof(submesh_gtdof);
    const HYPRE_BigInt first = m_parent_fes->GetTrueDofOffsets()[0];
    const HYPRE_BigInt end = m_parent_fes->GetTrueDofOffsets()[1];
    if (static_cast<HYPRE_BigInt>(parent_gtdof) >= first
        && static_cast<HYPRE_BigInt>(parent_gtdof) < end)
    {
        return static_cast<int>(
            static_cast<HYPRE_BigInt>(parent_gtdof) - first);
    }
    return -1;
}

void SurfaceProjector::BuildMap()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::surface_projector::build_map");

    // ------------------------------------------------------------------
    // Step 1: enumerate this rank's parent boundary Lagrange nodes.
    //
    // Primary path: walk the parent boundary elements' nodal points.
    // This is the Phase 6 higher-order/LOR path because it sees the
    // full boundary trace node set, including mid-edge nodes for P2.
    //
    // Supplemental path: also walk the parent boundary mesh vertices.
    // On imported P1 tet meshes we have seen legitimate boundary
    // vertices present on the boundary ParSubMesh but absent from the
    // FE-node enumeration above. Seeding the hash with explicit parent
    // boundary vertices preserves the higher-order path while making
    // the direct P1 boundary trace robust on unstructured imported
    // surfaces.
    //
    // Each snapped coordinate stores all three vector-component true
    // DOFs because the mortar stack always uses byNODES ordering with
    // vdim=3.
    // ------------------------------------------------------------------
    std::map<SnapCoordKey, ComponentGtdofs> local_parent_nodes;
    mfem::ParMesh* parent_mesh = m_parent_fes->GetParMesh();
    mfem::Array<int> scalar_dofs;
    mfem::Vector x_phys;

    const int nbe = parent_mesh->GetNBE();
    for (int be = 0; be < nbe; ++be)
    {
        m_parent_fes->GetBdrElementDofs(be, scalar_dofs);
        const mfem::FiniteElement* fe = m_parent_fes->GetBE(be);
        const mfem::IntegrationRule& nodes = fe->GetNodes();
        mfem::ElementTransformation* tr =
            parent_mesh->GetBdrElementTransformation(be);
        MFEM_VERIFY(nodes.GetNPoints() == scalar_dofs.Size(),
                    "SurfaceProjector: boundary FE node count does not "
                    "match scalar dof count.");

        for (int i = 0; i < scalar_dofs.Size(); ++i)
        {
            tr->Transform(nodes.IntPoint(i), x_phys);
            ComponentGtdofs gtdofs = {-1, -1, -1};
            for (int c = 0; c < kVDim; ++c)
            {
                const int vdof = m_parent_fes->DofToVDof(scalar_dofs[i], c);
                gtdofs[c] = m_parent_fes->GetGlobalTDofNumber(vdof);
            }

            const SnapCoordKey key = SnapKey(x_phys, m_snap_tol);
            const auto inserted = local_parent_nodes.emplace(key, gtdofs);
            if (!inserted.second)
            {
                // Adjacent boundary elements share Lagrange nodes. The
                // duplicate is valid only when it resolves to the same
                // parent true DOFs for all vector components.
                MFEM_VERIFY(SameComponentGtdofs(inserted.first->second, gtdofs),
                            "SurfaceProjector: duplicate snapped parent "
                            "boundary node has inconsistent TDOFs.");
            }
        }
    }

    // Boundary-vertex backfill for the direct P1 path. When the FE-node
    // walk above already found the vertex, this simply checks that the
    // vertex-based DOF lookup is consistent.
    mfem::Array<int> bdr_verts;
    for (int be = 0; be < nbe; ++be)
    {
        parent_mesh->GetBdrElementVertices(be, bdr_verts);
        for (int i = 0; i < bdr_verts.Size(); ++i)
        {
            const int parent_vertex = bdr_verts[i];
            const double* xyz = parent_mesh->GetVertex(parent_vertex);
            for (int d = 0; d < kVDim; ++d) { x_phys[d] = xyz[d]; }

            MFEM_VERIFY(parent_vertex >= 0
                            && parent_vertex < parent_mesh->GetNV(),
                        "SurfaceProjector: boundary element references "
                        "invalid parent vertex " << parent_vertex << ".");

            m_parent_fes->GetVertexDofs(parent_vertex, scalar_dofs);
            MFEM_VERIFY(scalar_dofs.Size() > 0,
                        "SurfaceProjector: parent boundary vertex "
                            << parent_vertex
                            << " has no H1 vertex dof.");

            const int scalar_dof = scalar_dofs[0];
            ComponentGtdofs gtdofs = {-1, -1, -1};
            for (int c = 0; c < kVDim; ++c)
            {
                const int vdof = m_parent_fes->DofToVDof(scalar_dof, c);
                gtdofs[c] = m_parent_fes->GetGlobalTDofNumber(vdof);
            }

            const SnapCoordKey key = SnapKey(x_phys, m_snap_tol);
            const auto inserted = local_parent_nodes.emplace(key, gtdofs);
            if (!inserted.second)
            {
                MFEM_VERIFY(SameComponentGtdofs(inserted.first->second, gtdofs),
                            "SurfaceProjector: boundary vertex and FE-node "
                            "enumerations disagree on parent TDOFs.");
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 2: allgather parent node records so every rank can match
    // its local submesh vertices even when parent/submesh partitions
    // do not coincide.
    //
    // This allgather is intentionally simple. It is a setup-time path,
    // and using one global coordinate table avoids subtle assumptions
    // about how MFEM partitions the parent volume mesh vs the boundary
    // ParSubMesh, especially after the boundary mesh is LOR-refined.
    // ------------------------------------------------------------------
    std::vector<long long> parent_send(
        local_parent_nodes.size() * kParentPackStride);
    int parent_write = 0;
    for (const auto& kv : local_parent_nodes)
    {
        long long* slot =
            parent_send.data() + parent_write * kParentPackStride;
        slot[0] = kv.first[0];
        slot[1] = kv.first[1];
        slot[2] = kv.first[2];
        slot[3] = kv.second[0];
        slot[4] = kv.second[1];
        slot[5] = kv.second[2];
        ++parent_write;
    }

    std::vector<int> parent_counts(m_nranks, 0);
    const int parent_local_count =
        static_cast<int>(local_parent_nodes.size());
    MPI_Allgather(&parent_local_count, 1, MPI_INT,
                  parent_counts.data(), 1, MPI_INT, m_comm);

    int parent_total = 0;
    std::vector<int> parent_recv_counts(m_nranks, 0);
    std::vector<int> parent_displs(m_nranks, 0);
    for (int r = 0; r < m_nranks; ++r)
    {
        parent_displs[r] = parent_total * kParentPackStride;
        parent_recv_counts[r] = parent_counts[r] * kParentPackStride;
        parent_total += parent_counts[r];
    }

    std::vector<long long> parent_recv(
        static_cast<std::size_t>(parent_total) * kParentPackStride);
    MPI_Allgatherv(parent_send.data(),
                   parent_local_count * kParentPackStride,
                   MPI_LONG_LONG,
                   parent_recv.data(),
                   parent_recv_counts.data(),
                   parent_displs.data(),
                   MPI_LONG_LONG,
                   m_comm);

    std::map<SnapCoordKey, ComponentGtdofs> parent_nodes;
    for (int i = 0; i < parent_total; ++i)
    {
        const long long* slot =
            parent_recv.data() + i * kParentPackStride;
        const SnapCoordKey key = {slot[0], slot[1], slot[2]};
        const ComponentGtdofs gtdofs = {
            static_cast<int>(slot[3]),
            static_cast<int>(slot[4]),
            static_cast<int>(slot[5])};
        const auto inserted = parent_nodes.emplace(key, gtdofs);
        if (!inserted.second)
        {
            // The same physical boundary node can be visible on more
            // than one rank. It must still refer to the same global
            // parent true DOFs; otherwise the snap tolerance is too
            // large or the input mesh/FES pair is inconsistent.
            MFEM_VERIFY(SameComponentGtdofs(inserted.first->second, gtdofs),
                        "SurfaceProjector: global duplicate snapped "
                        "parent node has inconsistent TDOFs.");
        }
    }

    // ------------------------------------------------------------------
    // Step 3: enumerate local submesh vertices and resolve each
    // component true DOF to the matching parent component true DOF.
    //
    // The submesh FES is order 1, so every scalar vertex dof is a
    // Lagrange node. The resulting global submesh->parent pairs are
    // allgathered in Step 4 so setup code can translate classifier
    // records that were gathered from other ranks.
    // ------------------------------------------------------------------
    std::map<int, int> local_submesh_to_parent;
    mfem::Array<int> submesh_scalar_dofs;
    mfem::Vector vertex_coord(kVDim);

    const int nv = m_submesh->GetNV();
    for (int v = 0; v < nv; ++v)
    {
        const double* xyz = m_submesh->GetVertex(v);
        for (int d = 0; d < kVDim; ++d) { vertex_coord[d] = xyz[d]; }
        const SnapCoordKey key = SnapKey(vertex_coord, m_snap_tol);

        // The LOR-refined submesh vertex and the parent P2 node are the
        // same geometric point evaluated through two different FP paths
        // (UniformRefinement vs boundary-element Transform); on a P2 mesh
        // they can disagree by ~1e-10 and straddle a snap-grid bin edge.
        // snap_tol << node spacing, so at most one real parent node is
        // within +/-1 bin: scan the 27-cell neighborhood.
        const ComponentGtdofs* parent_gtdofs = nullptr;
        {
            auto it = parent_nodes.find(key);
            if (it != parent_nodes.end()) { parent_gtdofs = &it->second; }
            else
            {
                for (long long dz = -1; dz <= 1 && !parent_gtdofs; ++dz)
                for (long long dy = -1; dy <= 1 && !parent_gtdofs; ++dy)
                for (long long dx = -1; dx <= 1 && !parent_gtdofs; ++dx)
                {
                    if (dx == 0 && dy == 0 && dz == 0) { continue; }
                    const SnapCoordKey nk{key[0] + dx, key[1] + dy, key[2] + dz};
                    auto nit = parent_nodes.find(nk);
                    if (nit != parent_nodes.end()) { parent_gtdofs = &nit->second; }
                }
            }
        }
        MFEM_VERIFY(parent_gtdofs != nullptr,
                    "SurfaceProjector: submesh vertex at ("
                    << xyz[0] << ", " << xyz[1] << ", " << xyz[2]
                    << ") has no matching parent boundary Lagrange node "
                    "within one snap cell.");

        m_submesh_fes->GetVertexDofs(v, submesh_scalar_dofs);
        MFEM_VERIFY(submesh_scalar_dofs.Size() > 0,
                    "SurfaceProjector: submesh vertex has no H1 dof.");
        const int scalar_dof = submesh_scalar_dofs[0];
        for (int c = 0; c < kVDim; ++c)
        {
            const int sub_vdof = m_submesh_fes->DofToVDof(scalar_dof, c);
            const int sub_gtdof = m_submesh_fes->GetGlobalTDofNumber(sub_vdof);
            local_submesh_to_parent[sub_gtdof] = (*parent_gtdofs)[c];
        }
    }

    // ------------------------------------------------------------------
    // Step 4: allgather the submesh-global -> parent-global pairs.
    //
    // Constraint-builder data can contain classifier-side submesh true
    // DOFs originating from other ranks. Keeping this communicator-wide
    // map locally makes later Phase 6 setup code a pure table lookup.
    // ------------------------------------------------------------------
    std::vector<long long> sub_send(
        local_submesh_to_parent.size() * kSubmeshPackStride);
    int sub_write = 0;
    for (const auto& kv : local_submesh_to_parent)
    {
        long long* slot = sub_send.data()
                        + sub_write * kSubmeshPackStride;
        slot[0] = kv.first;
        slot[1] = kv.second;
        ++sub_write;
    }

    std::vector<int> sub_counts(m_nranks, 0);
    const int sub_local_count =
        static_cast<int>(local_submesh_to_parent.size());
    MPI_Allgather(&sub_local_count, 1, MPI_INT,
                  sub_counts.data(), 1, MPI_INT, m_comm);

    int sub_total = 0;
    std::vector<int> sub_recv_counts(m_nranks, 0);
    std::vector<int> sub_displs(m_nranks, 0);
    for (int r = 0; r < m_nranks; ++r)
    {
        sub_displs[r] = sub_total * kSubmeshPackStride;
        sub_recv_counts[r] = sub_counts[r] * kSubmeshPackStride;
        sub_total += sub_counts[r];
    }

    std::vector<long long> sub_recv(
        static_cast<std::size_t>(sub_total) * kSubmeshPackStride);
    MPI_Allgatherv(sub_send.data(),
                   sub_local_count * kSubmeshPackStride,
                   MPI_LONG_LONG,
                   sub_recv.data(),
                   sub_recv_counts.data(),
                   sub_displs.data(),
                   MPI_LONG_LONG,
                   m_comm);

    for (int i = 0; i < sub_total; ++i)
    {
        const long long* slot = sub_recv.data() + i * kSubmeshPackStride;
        m_map.submesh_to_parent_gtdof[static_cast<int>(slot[0])] =
            static_cast<int>(slot[1]);
    }

    // ------------------------------------------------------------------
    // Step 5: materialize local arrays for fast local translation.
    //
    // These arrays cover exactly the local submesh true-vector range.
    // The second one precomputes the parent local index when this rank
    // owns the mapped parent true DOF; callers can use -1 as the remote
    // sentinel when constructing import/export tables.
    // ------------------------------------------------------------------
    const HYPRE_BigInt sub_first = m_submesh_fes->GetTrueDofOffsets()[0];
    m_map.local_submesh_to_parent_gtdof.SetSize(Height());
    m_map.local_submesh_to_parent_local_or_minus1.SetSize(Height());
    for (int i = 0; i < Height(); ++i)
    {
        const int sub_gtdof = static_cast<int>(sub_first) + i;
        const int parent_gtdof = ParentGtdof(sub_gtdof);
        m_map.local_submesh_to_parent_gtdof[i] = parent_gtdof;
        m_map.local_submesh_to_parent_local_or_minus1[i] =
            ParentLocalOrMinus1(sub_gtdof);
    }
}

void SurfaceProjector::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::surface_projector::mult");

    MFEM_VERIFY(x.Size() == Width(),
                "SurfaceProjector::Mult: input size " << x.Size()
                << " != Width() " << Width());
    MFEM_VERIFY(y.Size() == Height(),
                "SurfaceProjector::Mult: output size " << y.Size()
                << " != Height() " << Height());

    // Runtime use is setup/diagnostic only, so gather the full parent
    // true vector and apply the permutation locally. The constraint
    // operator will use Map() directly for hot-path construction; no
    // Krylov matvec should pay this allgather cost.
    const int n_local = x.Size();
    std::vector<int> counts(m_nranks, 0);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, m_comm);

    int n_global = 0;
    std::vector<int> recv_counts(m_nranks, 0);
    std::vector<int> displs(m_nranks, 0);
    for (int r = 0; r < m_nranks; ++r)
    {
        displs[r] = n_global;
        recv_counts[r] = counts[r];
        n_global += counts[r];
    }

    std::vector<double> x_global(static_cast<std::size_t>(n_global), 0.0);
    MPI_Allgatherv(x.HostRead(), n_local, MPI_DOUBLE,
                   x_global.data(), recv_counts.data(), displs.data(),
                   MPI_DOUBLE, m_comm);

    double* y_data = y.HostWrite();
    for (int i = 0; i < Height(); ++i)
    {
        y_data[i] = x_global[m_map.local_submesh_to_parent_gtdof[i]];
    }
}

void SurfaceProjector::MultTranspose(const mfem::Vector& x,
                                     mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::surface_projector::mult_transpose");

    MFEM_VERIFY(x.Size() == Height(),
                "SurfaceProjector::MultTranspose: input size "
                << x.Size() << " != Height() " << Height());
    MFEM_VERIFY(y.Size() == Width(),
                "SurfaceProjector::MultTranspose: output size "
                << y.Size() << " != Width() " << Width());

    // R^T is a scatter-add from submesh true DOFs back to parent true
    // DOFs. Different ranks can hold submesh entries that map to the
    // same parent global true DOF, so the local dense accumulation is
    // reduced globally before extracting this rank's parent partition.
    const int n_global_parent = m_parent_fes->GlobalTrueVSize();
    std::vector<double> local_accum(
        static_cast<std::size_t>(n_global_parent), 0.0);
    const double* x_data = x.HostRead();
    for (int i = 0; i < Height(); ++i)
    {
        local_accum[m_map.local_submesh_to_parent_gtdof[i]] += x_data[i];
    }

    std::vector<double> global_accum(
        static_cast<std::size_t>(n_global_parent), 0.0);
    MPI_Allreduce(local_accum.data(), global_accum.data(), n_global_parent,
                  MPI_DOUBLE, MPI_SUM, m_comm);

    const HYPRE_BigInt parent_first = m_parent_fes->GetTrueDofOffsets()[0];
    double* y_data = y.HostWrite();
    for (int i = 0; i < Width(); ++i)
    {
        y_data[i] = global_accum[static_cast<int>(parent_first) + i];
    }
}

}  // namespace mortar_pbc
