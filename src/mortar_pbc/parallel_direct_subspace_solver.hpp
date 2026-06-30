// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// MFEM parallel-sparse-direct adapter for AMGF mortar-PBC subspace solves.
//
// The AMGF implementation for mortar periodic boundary conditions uses MFEM's
// `AMGFSolver` as the displacement-block preconditioner. MFEM owns the
// filtering algorithm and forms the filtered subspace operator P^T A P as a
// genuinely distributed `mfem::HypreParMatrix`; it then applies the subspace
// correction through a caller-supplied `mfem::Solver`. This header provides the
// adapter that satisfies that interface with an *exact distributed* sparse
// direct factorization (SuperLU_DIST by default, MUMPS or STRUMPACK if those
// are the backends MFEM was built with).
//
// This file replaces the previous Ginkgo-based adapter. The Ginkgo adapter
// factored only `HypreParMatrix::GetDiag()` — the rank-local diagonal block of
// P^T A P — which silently degraded the subspace solve into a block-Jacobi
// approximation across the MPI rank partition. AMGF's convergence theory
// requires an *exact* solve on the filtered subspace, so on more than one rank
// the inexact subspace solve weakened the preconditioner and, under a finite
// inner Krylov tolerance / iteration cap, produced rank-count-dependent
// (under-converged) Newton trajectories. A true distributed direct solve on
// P^T A P restores the exact subspace correction and the rank-independent
// converged solution (to the inner linear tolerance).

#pragma once

#include "mfem.hpp"

#include <memory>
#include <string>

namespace exaconstit::amgf {

/**
 * @brief Parallel sparse-direct backend used for the AMGF subspace solve.
 *
 * @details Mirrors the backends wrapped by MFEM's contact-miniapp
 * `mfem::ParallelDirectSolver`, but the enum lives in the ExaConstit tree so
 * the TOML option maps directly onto it and so we can set fill-reducing
 * ordering options rather than the bare backend defaults.
 *
 * `AUTO` resolves at construction to the first backend MFEM was actually built
 * with, in the priority order SuperLU_DIST → MUMPS → STRUMPACK. SuperLU_DIST is
 * preferred because it is the backend the ExaConstit install scripts provision
 * for the AMGF path.
 */
enum class DirectBackend
{
    AUTO,
    SUPERLU,
    MUMPS,
    STRUMPACK
};

/**
 * @brief Parse a user-facing backend selector string.
 *
 * @param spec One of "auto", "superlu", "mumps", "strumpack"
 *             (case-insensitive). Aborts on any other value.
 */
DirectBackend ParseDirectBackend(const std::string& spec);

/**
 * @brief Whether MFEM was built with at least one supported parallel direct
 *        solver backend.
 *
 * @details Resolved entirely at compile time from MFEM's configuration macros
 * (`MFEM_USE_SUPERLU`, `MFEM_USE_MUMPS`, `MFEM_USE_STRUMPACK`). Option
 * validation uses this to reject AMGF early, before any heavy setup, when no
 * backend is available.
 */
bool ParallelDirectSolverAvailable();

/**
 * @brief Human-readable name of the backend that `AUTO` would resolve to in
 *        this build (or "none" if no backend is available).
 *
 * @details Intended for diagnostics and clear error messages.
 */
const char* DefaultDirectBackendName();

/**
 * @brief `mfem::Solver` adapter around an exact distributed sparse direct
 *        factorization of the AMGF filtered-subspace operator.
 *
 * @details `mfem::AMGFSolver` forms the filtered subspace operator
 * \f$P^T A P\f$ as a distributed `mfem::HypreParMatrix` over the same
 * communicator as the displacement block \f$K\f$, then applies the subspace
 * correction by calling this solver's `Mult`. Because the AMGF Boolean
 * prolongation \f$P\f$ partitions its columns by displacement-DOF ownership,
 * \f$P^T A P\f$ couples boundary DOFs owned by different ranks through its
 * off-diagonal blocks. A correct subspace solve must therefore factor the full
 * distributed operator, not a rank-local block.
 *
 * This adapter wraps MFEM's parallel direct solvers, which factor exactly the
 * full distributed matrix:
 *  - `SUPERLU`   : `mfem::SuperLUSolver` over a `mfem::SuperLURowLocMatrix`.
 *  - `STRUMPACK` : `mfem::STRUMPACKSolver` over a `mfem::STRUMPACKRowLocMatrix`.
 *  - `MUMPS`     : `mfem::MUMPSSolver` directly on the `HypreParMatrix`.
 *
 * @par Symmetry
 * The `symmetric` flag selects ordering/pivoting suited to the symmetric
 * positive-definite tangent (and the Path-D augmented block
 * \f$K_\gamma = K + \gamma C^T C\f$, also SPD when \f$K\f$ is SPD) versus the
 * mildly non-symmetric tangent arising from non-associated plastic flow. For
 * SuperLU it controls `SetSymmetricPattern` and the row-permutation choice.
 *
 * @par Reproducibility note
 * Parallel sparse direct solvers do not in general produce bit-identical
 * results across different MPI rank counts (reduction order in the parallel
 * factorization/triangular solve is not fixed). That is acceptable here: this
 * object supplies a *preconditioner* subspace correction that is accurate to
 * round-off, so the outer Krylov method converges to the same solution to its
 * tolerance regardless of rank count. The fix this class delivers is
 * rank-independence of the converged answer *to the inner linear tolerance*,
 * not bitwise reproducibility of the preconditioner application.
 *
 * @par Cost
 * `SetOperator()` rebuilds the backend solver and performs the (expensive)
 * symbolic+numeric factorization. It is called once per Newton operator
 * rebuild. `Mult()` performs only the forward/back substitution.
 *
 * @par Host residency
 * The current AMGF path is FULL-assembly CPU/OpenMP only (enforced by option
 * validation), so the subspace matrices are host-resident. SuperLU_DIST built
 * with CUDA/HIP can still offload its own factorization internally, but no
 * host/device staging is performed here. The future hybrid GPU mode is out of
 * scope for this adapter.
 */
class ParallelDirectSubspaceSolver : public mfem::Solver
{
public:
    /**
     * @brief Construct an unfactored subspace solver.
     *
     * @param comm        Communicator of the AMGF subspace operator (same as
     *                    the displacement block K). Must not be MPI_COMM_NULL.
     * @param backend     Direct-solver backend; `AUTO` resolves to the first
     *                    available backend in this build.
     * @param symmetric   True for the SPD tangent / SPD augmented block;
     *                    false for the non-symmetric non-associated-flow case.
     * @param print_level Backend factorization/solve statistics verbosity.
     */
    ParallelDirectSubspaceSolver(MPI_Comm comm,
                                 DirectBackend backend = DirectBackend::AUTO,
                                 bool symmetric = true,
                                 int print_level = 0);

    /**
     * @brief Construct from a backend selector string ("auto", "superlu",
     *        "mumps", "strumpack").
     */
    ParallelDirectSubspaceSolver(MPI_Comm comm,
                                 const std::string& backend_spec,
                                 bool symmetric = true,
                                 int print_level = 0);

    ~ParallelDirectSubspaceSolver() override;

    ParallelDirectSubspaceSolver(const ParallelDirectSubspaceSolver&) = delete;
    ParallelDirectSubspaceSolver& operator=(
        const ParallelDirectSubspaceSolver&) = delete;

    /**
     * @brief Factor the supplied AMGF subspace operator.
     *
     * @details `op` must be an `mfem::HypreParMatrix` (the type produced by
     * MFEM's Galerkin path for \f$P^T A P\f$). The backend solver and any
     * row-local matrix representation are rebuilt from scratch on each call so
     * no stale factorization state survives a Newton operator change.
     */
    void SetOperator(const mfem::Operator& op) override;

    /**
     * @brief Apply the cached direct solve to one right-hand side.
     */
    void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

    /// Resolved backend (after `AUTO` resolution).
    DirectBackend Backend() const { return backend_; }

private:
    /// Build (or rebuild) the backend solver and apply ordering options.
    void InitBackend();

    MPI_Comm comm_;
    DirectBackend backend_;
    bool symmetric_;
    int print_level_;

    /// Owning pointer to the underlying MFEM backend solver.
    std::unique_ptr<mfem::Solver> solver_;

#ifdef MFEM_USE_SUPERLU
    /// Row-local representation required by SuperLUSolver; must outlive solves.
    std::unique_ptr<mfem::SuperLURowLocMatrix> superlu_mat_;
#endif
#ifdef MFEM_USE_STRUMPACK
    /// Row-local representation required by STRUMPACKSolver.
    std::unique_ptr<mfem::STRUMPACKRowLocMatrix> strumpack_mat_;
#endif
};

}  // namespace exaconstit::amgf
