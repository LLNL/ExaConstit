// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 6.0.D — SurfaceProjector.
//
// SurfaceProjector is the construction-time bridge between the parent
// volume FE space and the boundary-submesh FE space used by the mortar
// classifier. At order 1 it is the boundary trace permutation. At
// order 2 with a once-refined LOR boundary submesh, it maps refined
// surface vertices to the coincident parent boundary Lagrange nodes.
//
// This file intentionally documents both the mathematical contract and
// the MPI ownership contract because later Phase 6 components consume
// the TDofMap directly when replacing classifier-side surface TDOFs by
// parent-volume TDOFs.

#pragma once

#include "mfem.hpp"

#include <array>
#include <map>
#include <memory>
#include <vector>

namespace mortar_pbc {

/**
 * @brief Construction-time trace map from parent true DOFs to
 *        boundary-submesh true DOFs.
 *
 * @details The Phase 6 LOR formulation relies on the Pazner-Kolev
 * coincidence property: the order-p parent boundary Lagrange nodes
 * coincide with vertices of the low-order-refined boundary submesh.
 * For nodal H1 spaces this means the trace map is a permutation, not
 * an interpolation. `SurfaceProjector` builds that permutation once by
 * snap-coordinate matching and exposes both:
 *   - a lookup table for setup code that needs parent global/local
 *     TDOF indices for a classifier-side submesh TDOF; and
 *   - an `mfem::Operator` interface for diagnostics and setup-time
 *     vector projection.
 *
 * If `R` denotes the Boolean trace/permutation matrix, the operator
 * interface implements:
 *
 *     y_submesh = R x_parent
 *     y_parent  = R^T x_submesh
 *
 * The primary production use is not the operator interface, however.
 * `MortarConstraintOperator` and `ConstraintBuilder3D` need to rewrite
 * classifier-side surface true DOFs into parent-volume true DOFs while
 * building their row data. The `TDofMap` tables expose exactly that
 * setup-time translation.
 *
 * This class is intentionally host/setup oriented. The mortar
 * constraint operator's Krylov matvec path should consume the map at
 * construction time and bake parent-local indices into its flat arrays;
 * it should not call `SurfaceProjector::Mult` per iteration.
 *
 * @par Space Requirements
 * Both FE spaces must be vector H1 spaces with `vdim = 3` and
 * `mfem::Ordering::byNODES`. The submesh FE space must be order 1
 * because the classifier works on the low-order boundary mesh. The
 * parent FE space may be order 1 or 2 in the current Phase 6 Day-1
 * scope. Higher parent orders are not rejected here if MFEM can expose
 * compatible boundary nodal rules, but the Phase 6 validation target is
 * p=1 direct and p=2 once-refined LOR.
 *
 * @par Mesh Relationship
 * `submesh_fes` must be defined on `submesh`. `submesh` is expected to
 * be either the unrefined boundary `ParSubMesh` extracted from the
 * parent mesh or its once-uniformly-refined LOR descendant. Every
 * local submesh vertex must snap to exactly one parent boundary
 * Lagrange node; construction aborts if a vertex cannot be matched or
 * if duplicate snapped parent coordinates point to inconsistent parent
 * true DOFs.
 *
 * @par MPI Scope
 * Construction and `Mult`/`MultTranspose` are collective on the parent
 * FES communicator. The current implementation uses allgather-style
 * setup/runtime exchanges because this class is not in the Krylov hot
 * path; a later optimization can replace those with the Alltoallv
 * topology described in the Phase 6 plan without changing callers.
 *
 * @par Global vs local numbering
 * MFEM/Hypre true-DOF global numbers are represented as `int` in these
 * tables to match the existing mortar-PBC data structures. This is
 * consistent with the current ExaConstit/MFEM builds where
 * `HYPRE_BigInt` is 32-bit. If ExaConstit moves to 64-bit Hypre global
 * IDs, this class and the surrounding mortar-PBC gtdof storage should
 * be widened together.
 */
class SurfaceProjector : public mfem::Operator
{
public:
    /**
     * @brief Per-submesh-TDOF translation to parent-FES TDOFs.
     *
     * @details `submesh_to_parent_gtdof` is a communicator-wide table
     * keyed by submesh-FES global true DOF. It exists because
     * classifier rows may be gathered from ranks other than the rank
     * that owns the corresponding submesh true DOF.
     *
     * The two `local_*` arrays cover only this rank's local submesh true
     * vector, i.e. entries `[0, Height())` correspond to global submesh
     * true DOFs
     * `[submesh_fes.GetTrueDofOffsets()[0],
     *   submesh_fes.GetTrueDofOffsets()[1])`.
     *
     * `local_submesh_to_parent_local_or_minus1[i]` stores the parent
     * local true DOF when the mapped parent true DOF is owned by this
     * rank. It is `-1` when the parent owner is remote, allowing caller
     * setup code to decide whether a communication import/export entry
     * is needed.
     */
    struct TDofMap
    {
        /// Global submesh true DOF -> global parent true DOF.
        std::map<int, int> submesh_to_parent_gtdof;

        /// Local submesh true DOF -> global parent true DOF.
        mfem::Array<int> local_submesh_to_parent_gtdof;

        /// Local submesh true DOF -> local parent true DOF, or -1 if remote.
        mfem::Array<int> local_submesh_to_parent_local_or_minus1;
    };

    /**
     * @brief Build the surface projection map by snapped physical
     *        coordinate matching.
     *
     * @param parent_fes  Parent volume FES. `Width()` equals
     *                    `parent_fes->GetTrueVSize()` on this rank.
     * @param submesh_fes Boundary/LOR submesh FES. `Height()` equals
     *                    `submesh_fes->GetTrueVSize()` on this rank.
     * @param submesh     Boundary submesh on which `submesh_fes` is
     *                    defined. The object must outlive the projector
     *                    through shared ownership.
     * @param snap_tol    Absolute snap-coordinate tolerance used to
     *                    quantize physical coordinates. Choose this
     *                    small relative to the minimum edge length but
     *                    large enough to absorb roundoff from parent and
     *                    submesh coordinate evaluation.
     *
     * @pre `parent_fes`, `submesh_fes`, and `submesh` are non-null.
     * @pre Both spaces use byNODES vector ordering with `vdim = 3`.
     * @pre `submesh_fes` is an order-1 H1 space on `submesh`.
     *
     * @post `Map()` contains one entry for every local submesh true DOF
     *       and a communicator-wide lookup for all gathered submesh
     *       true DOFs.
     */
    SurfaceProjector(
        std::shared_ptr<const mfem::ParFiniteElementSpace> parent_fes,
        std::shared_ptr<const mfem::ParFiniteElementSpace> submesh_fes,
        std::shared_ptr<const mfem::ParMesh> submesh,
        double snap_tol);

    /**
     * @brief Return the constructed submesh-to-parent true-DOF map.
     *
     * @details The returned reference remains valid for the lifetime of
     * the projector. Callers must treat it as read-only setup data; the
     * projector does not rebuild the map after construction.
     */
    const TDofMap& Map() const { return m_map; }

    /**
     * @brief Return the parent-FES global true DOF corresponding to a
     *        submesh-FES global true DOF.
     *
     * @param submesh_gtdof Global true DOF in `submesh_fes`.
     *
     * @return Global true DOF in `parent_fes`.
     *
     * @throws via `MFEM_VERIFY` if `submesh_gtdof` is not present in
     *         the communicator-wide map.
     */
    int ParentGtdof(int submesh_gtdof) const;

    /**
     * @brief Return the parent-FES local true DOF for a submesh true
     *        DOF when this rank owns it, or -1 otherwise.
     *
     * @param submesh_gtdof Global true DOF in `submesh_fes`.
     *
     * @return Local true DOF in `parent_fes` when the mapped parent DOF
     *         belongs to this rank's true-vector partition; otherwise
     *         `-1`.
     */
    int ParentLocalOrMinus1(int submesh_gtdof) const;

    /**
     * @brief Return the owner rank of a parent-FES global true DOF.
     *
     * @details Ownership is computed from all gathered parent true-DOF
     * offsets. This helper is primarily for setup code that must decide
     * whether a mapped parent true DOF should be handled locally or via
     * an MPI import/export path.
     */
    int ParentOwnerRank(int parent_gtdof) const;

    /**
     * @brief Apply `y_submesh = R x_parent`.
     *
     * @details `x` is this rank's local parent true vector and `y` is
     * this rank's local submesh true vector. The implementation gathers
     * the full parent true vector across the communicator and performs
     * local table lookup. This is intended for diagnostics and
     * setup-time projection checks, not Krylov matvecs.
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Apply `y_parent = R^T x_submesh`.
     *
     * @details Contributions from duplicated/shared submesh-side
     * entries are summed by a communicator-wide reduction into the
     * parent true-vector partition. `y` is overwritten with this rank's
     * local parent true-vector contribution.
     */
    void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

private:
    /// Parent volume FE space that owns the primal solution true vector.
    std::shared_ptr<const mfem::ParFiniteElementSpace> m_parent_fes;

    /// Boundary/LOR FE space used by the classifier and multiplier rows.
    std::shared_ptr<const mfem::ParFiniteElementSpace> m_submesh_fes;

    /// Boundary/LOR mesh whose vertices are matched against parent nodes.
    std::shared_ptr<const mfem::ParMesh> m_submesh;

    /// Communicator shared by the parent and submesh FE spaces.
    MPI_Comm m_comm = MPI_COMM_NULL;

    /// Rank id within `m_comm`.
    int m_rank = -1;

    /// Number of ranks in `m_comm`.
    int m_nranks = -1;

    /// Absolute coordinate snapping tolerance.
    double m_snap_tol = 0.0;

    /// Constructed global and local true-DOF translation tables.
    TDofMap m_map;

    /// Gathered parent true-DOF partition offsets, length `m_nranks + 1`.
    std::vector<HYPRE_BigInt> m_parent_tdof_offsets_all;

    /**
     * @brief Quantize a physical coordinate to an integer snap key.
     *
     * @details Snapping replaces tolerance-based floating-point map
     * lookup with exact integer-key lookup. Coordinates whose
     * component-wise distance is less than roughly half `snap_tol` map
     * to the same key.
     */
    static std::array<long long, 3> SnapKey(const mfem::Vector& x,
                                            double snap_tol);

    /**
     * @brief Build all global and local submesh-to-parent true-DOF maps.
     *
     * @details Collective on `m_comm`. The implementation first gathers
     * snapped parent boundary Lagrange-node records from every rank, then
     * matches this rank's local submesh vertices against that global
     * coordinate table, and finally gathers the resulting submesh-parent
     * true-DOF pairs so every rank can translate classifier records.
     */
    void BuildMap();
};

}  // namespace mortar_pbc
