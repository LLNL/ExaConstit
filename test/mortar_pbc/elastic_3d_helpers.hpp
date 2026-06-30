// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `mortar_pbc/elastic_3d.py` (helpers
// only). Provides the linear-elastic stiffness assembly, the
// (F-I)X projection, and the distributed Dirichlet elimination —
// the three building blocks the saddle-point solver and patch-test
// driver consume.
//
// Scope (deliberate)
// ------------------
// The Python module also contained `find_corners_3d` and
// `collect_corner_tdofs`. Those are NOT ported here because
// `BoundaryClassifier3D::Corners()` already returns the 8 corner
// records — drivers walk the classifier's catalogue directly. This
// keeps elastic helpers focused on linear-elasticity machinery and
// avoids duplicating boundary-classification logic.
//
// References
// ----------
//   * MORTAR_PBC_ARCHITECTURE.md §6.4 (Dirichlet elimination gotcha).
//   * MORTAR_PBC_ARCHITECTURE.md §7.4 (Newton warm-start at u_lin).

#pragma once

#include "mfem.hpp"

#include <vector>

namespace mortar_pbc {

/**
 * @brief Assemble the small-strain linear-elastic tangent K as a
 *        distributed `HypreParMatrix`.
 *
 * @param pmesh  Parallel mesh (2D or 3D — dimension generic).
 * @param fes    Vector H1 space with `vdim == pmesh.Dimension()`.
 * @param E      Young's modulus.
 * @param nu     Poisson's ratio.
 *
 * @return A heap-allocated `HypreParMatrix*` owning the assembled
 *         stiffness. Caller owns; must `delete`.
 *
 * @details Uses `mfem::ElasticityIntegrator(lambda, mu)` on a
 * `ParBilinearForm`, then `ParallelAssemble()`. Both the integrator
 * and the form pick up the spatial dimension from `fes`, so this
 * function works in 2D or 3D unchanged.
 *
 * For heterogeneous RVEs, the stable refactor is to take per-region
 * Lamé parameters as `mfem::PWConstCoefficient` instead of `(E, nu)`
 * scalars; that's a Phase 4.2+ change tracked separately.
 *
 * @par MPI scope
 * Collective on `pmesh.GetComm()` (one `ParallelAssemble` collective
 * call internal to MFEM).
 *
 * @par GPU
 * Host-only. The integrator's PA path is not used here since the
 * linear-elastic K has no need for a partial-assembled tangent at
 * the same level of detail as ExaConstit's nonlinear ICExaNLFIntegrator.
 *
 * @par Linearity
 * @code
 *     mu  = 0.5 * E / (1 + nu)
 *     lam = E * nu / ((1 + nu) * (1 - 2 nu))
 * @endcode
 */
mfem::HypreParMatrix* AssembleLinearElasticKHypre(
    mfem::ParMesh& pmesh,
    mfem::ParFiniteElementSpace& fes,
    double E,
    double nu);

/**
 * @brief Project `u_lin(X) = (F - I) X` onto the FE space and return
 *        the local-rank true-DOF vector.
 *
 * @param fes      Vector H1 space; `vdim` must equal `F_macro` order.
 * @param F_macro  Macroscopic deformation gradient as a
 *                 `mfem::DenseMatrix` of shape `(vdim, vdim)`.
 *
 * @return `mfem::Vector` of size `fes.GetTrueVSize()` containing this
 *         rank's portion of the projected `u_lin`.
 *
 * @details Builds an `mfem::VectorFunctionCoefficient` that evaluates
 * `(F - I) X` at the supplied physical-space point, projects via
 * `ParGridFunction::ProjectCoefficient`, and converts to a true-DOF
 * vector via `GetTrueDofs`.
 *
 * @par MPI scope
 * Collective on `fes.GetComm()` — `ProjectCoefficient` itself is
 * local but `GetTrueDofs` triggers communication for shared vertices.
 *
 * @par Use cases
 *   - **Method-D PBC**: extract the corner entries of `u_lin` for
 *     `f_at_essential` in `ApplyDirichletToDistributedK`.
 *   - **Patch test**: warm-start the Newton solve at `u_init = u_lin`
 *     so `r1 = K · u_lin = 0` to numerical roundoff for a
 *     homogeneous material.
 */
mfem::Vector ApplyLinearPart(mfem::ParFiniteElementSpace& fes,
                             const mfem::DenseMatrix& F_macro);

/**
 * @brief Eliminate essential-DOF rows/cols on the distributed K and
 *        write prescribed values into the corresponding entries of f.
 *
 * @param[in,out] K_hyp              Distributed stiffness; modified
 *                                   in place via `EliminateRowsCols`.
 * @param[in,out] f_par              Distributed RHS; entries at
 *                                   essential TDOFs set to
 *                                   `f_at_essential` (or 0 if empty).
 * @param         ess_global_tdofs   Global TDOF indices of essential
 *                                   DOFs. Each rank passes the same
 *                                   list (or its own subset — the
 *                                   helper filters by ownership).
 * @param         fes                FE space; provides the rank's
 *                                   TDOF range.
 * @param         f_at_essential     Prescribed values at the essential
 *                                   TDOFs in the SAME ORDER as
 *                                   `ess_global_tdofs`. If empty
 *                                   (default), entries are zeroed
 *                                   (homogeneous Dirichlet).
 *
 * @par Crucial gotcha (architecture §6.4)
 * `EliminateRowsCols` zeros the *full* corner row of K, including the
 * off-diagonal coupling K_uc into free DOFs. To preserve consistency
 * of the RHS for non-zero Dirichlet, the caller must add
 * `K_uc · u_corner` to f BEFORE calling this function. The pattern is:
 *
 * @code
 *     b_lhs = K.Mult(u_lin);           // action on u_corner-extended u
 *     f -= b_lhs;                       // subtract K_uc · u_c
 *     ApplyDirichletToDistributedK(K, f, ess_tdofs, fes, u_corner_vals);
 * @endcode
 *
 * @par MPI scope
 * Collective on `fes.GetComm()` — `EliminateRowsCols` is collective.
 */
void ApplyDirichletToDistributedK(mfem::HypreParMatrix& K_hyp,
                                  mfem::Vector& f_par,
                                  const std::vector<int>& ess_global_tdofs,
                                  mfem::ParFiniteElementSpace& fes,
                                  const std::vector<double>& f_at_essential);

/// Convenience overload: homogeneous Dirichlet (`f_at_essential = 0`).
void ApplyDirichletToDistributedK(mfem::HypreParMatrix& K_hyp,
                                  mfem::Vector& f_par,
                                  const std::vector<int>& ess_global_tdofs,
                                  mfem::ParFiniteElementSpace& fes);

/**
 * @brief Compute the Newton-step residual `r1 = K · u_lin` at the
 *        warm-start initial iterate.
 *
 * @param K_hyp         Distributed stiffness (NOT yet eliminated).
 * @param u_lin_local   Local-rank true-DOF view of u_lin = (F-I) X.
 *
 * @return Distributed `mfem::Vector` containing `r1 = K · u_lin`.
 *
 * @details For a homogeneous patch test, `K · u_lin = 0` to roundoff
 * (the linear-elastic operator on an affine field is zero). For
 * heterogeneous RVEs, `r1` is non-zero in the interior because the
 * spatially-varying stiffness produces non-zero stress under uniform
 * F; mortar PBC fixes the result by adding the constraint coupling.
 *
 * @par MPI scope
 * Collective on `K_hyp`'s communicator (one parallel matvec).
 */
mfem::Vector NewtonResidualAtULin(const mfem::HypreParMatrix& K_hyp,
                                  const mfem::Vector& u_lin_local);

/**
 * @brief Return the global TDOFs of every boundary node, all
 *        spatial components, that this rank owns.
 *
 * @param pmesh  Parallel mesh.
 * @param fes    Vector H1 space; `vdim` sets components per node.
 *
 * @return Global TDOF indices owned by this rank that lie on the
 *         boundary. Each value is in
 *         `[my_first_tdof, my_first_tdof + my_n_tdof)`.
 *
 * @details Used by the patch test (homogeneous full-Dirichlet
 * validation): the affine field `u_lin = (F-I) X` is the unique
 * minimum-energy solution iff Dirichlet is imposed on the ENTIRE
 * boundary. Pinning only the 8 corners leaves the rest of `∂Ω` with
 * natural (zero-traction) Neumann, which is incompatible with the
 * constant stress under uniform F; the solver then finds a non-affine
 * field that satisfies `σ · n = 0` on the free boundary.
 *
 * Implementation: marks all boundary attributes essential, calls
 * `ParFiniteElementSpace::GetEssentialTrueDofs` (which is vdim-aware
 * — all spatial components included), then converts local TDOFs to
 * globals by adding this rank's TDOF offset.
 *
 * @par MPI scope
 * Local — no collective communication.
 */
std::vector<int> FindAllBoundaryTdofs(mfem::ParMesh& pmesh,
                                      mfem::ParFiniteElementSpace& fes);

/**
 * @brief For each global TDOF in `boundary_global_tdofs`, return its
 *        `u_lin` value from this rank's local TDOF array (or 0 if
 *        not owned on this rank).
 *
 * @param boundary_global_tdofs  Global TDOF indices.
 * @param u_lin_local            Local-rank true-DOF view of u_lin.
 * @param fes                    FE space; provides this rank's TDOF
 *                               range.
 *
 * @return Vector aligned with `boundary_global_tdofs`; entries for
 *         non-owned TDOFs are 0.0 (the Dirichlet helper filters by
 *         ownership anyway).
 *
 * @details Used to build the `f_at_essential` argument for
 * `ApplyDirichletToDistributedK` when Dirichlet values are
 * `u_lin = (F-I) X` (full-boundary patch test) or `u_lin[corner]`
 * (Method-D PBC at the 8 corners).
 *
 * @par MPI scope
 * Local — no collective communication.
 */
std::vector<double> CollectBoundaryTdofValues(
    const std::vector<int>& boundary_global_tdofs,
    const mfem::Vector& u_lin_local,
    mfem::ParFiniteElementSpace& fes);

}  // namespace mortar_pbc
