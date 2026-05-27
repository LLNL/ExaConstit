// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// AMGF utilities for mortar-periodic saddle preconditioning.
//
// The helpers in this file are setup-time infrastructure for MFEM's
// AMG-with-filtering preconditioner. They deliberately live outside
// MortarConstraintOperator: the constraint operator identifies the
// active coupled displacement true DOFs, while these utilities convert
// that algebraic index set into Hypre/MFEM objects suitable for AMGF.

#pragma once

#include "mfem.hpp"
#include "mpi.h"

#include <vector>

namespace exaconstit::amgf {

/**
 * @brief Build a Boolean prolongation matrix for an AMGF filtered subspace.
 *
 * @details The returned matrix \f$P \in \mathbb{R}^{n_u \times n_c}\f$
 * has one nonzero in each column: \f$P[I_j, j] = 1\f$, where
 * \f$I\f$ is the communicator-wide sorted union of `idx_global`.
 * `idx_global` is allowed to be a rank-local view of the active
 * nonzero columns of the mortar constraint matrix, including off-rank
 * columns. This routine performs the required collective union before
 * assigning compact column numbers, so each global displacement true
 * DOF contributes exactly one AMGF subspace column.
 *
 * The row partition of \f$P\f$ is exactly `k_row_starts`, matching the
 * tangent stiffness block \f$K\f$. The column partition is compact and
 * derived from row ownership: rank r owns the AMGF columns whose parent
 * displacement DOFs lie in rank r's K row range. This is the layout
 * needed for \f$P^T K P\f$ to form the principal boundary-coupled
 * submatrix used by `mfem::AMGFSolver`.
 *
 * @param n_global_rows Total displacement true-DOF count.
 * @param idx_global Rank-local sorted/unique or unsorted/nonunique
 *                   global displacement true DOFs touched by active
 *                   mortar constraints.
 * @param k_row_starts Two-entry local row partition for K:
 *                     `[first_row_on_rank, end_row_on_rank)`.
 * @param comm MPI communicator shared with K and the mortar operator.
 *
 * @return Heap-allocated `mfem::HypreParMatrix`; caller owns it.
 *
 * @par Cost
 * Collective setup only. The index union is an `MPI_Allgatherv` over
 * boundary-coupled DOF ids; this is acceptable because AMGF setup is
 * performed once per operator rebuild, not per Krylov matvec.
 */
mfem::HypreParMatrix* BuildBooleanRestrictionProlongation(
    HYPRE_BigInt n_global_rows,
    const std::vector<HYPRE_BigInt>& idx_global,
    const HYPRE_BigInt* k_row_starts,
    MPI_Comm comm);

}  // namespace exaconstit::amgf
