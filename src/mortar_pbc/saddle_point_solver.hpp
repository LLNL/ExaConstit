// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of `mortar_pbc/saddle_point.py` (the
// SaddlePointSolver class). Solves one Newton step of the
// constrained problem
//
//      [ K   C^T ] [ du ]   [ -r1 ]
//      [ C   0   ] [ dλ ] = [ -r2 ]                                  (*)
//
// per Lopes et al. (2021), Eq. (59).
//
// What this layer does
// --------------------
// Given a tangent stiffness `K` (HypreParMatrix), a constraint
// matrix `C` (HypreParMatrix), and the two halves `r1`, `r2` of the
// Newton residual, the solver:
//
//   1. Constructs an `mfem::BlockOperator` representing the LHS of (*).
//   2. Optionally builds a block-diagonal preconditioner (Jacobi).
//   3. Runs the chosen Krylov method (MINRES, GMRES, or BiCGStab) on
//      the distributed block system.
//   4. Returns the solution split into `du` and `dλ` halves.
//
// CG is rejected up front: the (2, 2) zero block guarantees the
// system is symmetric indefinite, and CG diverges on indefinite
// systems.
//
// Scope reductions vs. the Python prototype
// -----------------------------------------
//   * The Python wrapped a SciPy CSR `C` as a "PyOperator" with
//     custom Mult / MultTranspose / WeightedRowSqSum that gathered
//     and locally CSR-multiplied. NOT NEEDED in C++: our
//     ConstraintBuilder3D::BuildHypreParMatrix already produces a
//     real distributed HypreParMatrix.
//   * The Python had elaborate PyOperator dispatch sanity checks
//     and SWIG-director caveats. NOT NEEDED in C++: there's no
//     dispatch boundary.
//   * The Python's "diagnostic_mode" dump path is omitted; if a
//     C++ driver wants min/max/NaN-count diagnostics it can call
//     `mfem::Vector::Print` directly on the block residual vector.
//
// References
// ----------
//   * Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.
//     Eq. (59), Table 5.
//   * MFEM example 28 / ex28p (BuildNormalConstraints + saddle-point).
//   * MORTAR_PBC_ARCHITECTURE.md §6.5 (SPS method choice).

#pragma once

#include "mfem.hpp"

#include <memory>

namespace mortar_pbc {

class MortarConstraintOperator;  // forward decl — defined in
                                  // mortar_constraint_operator.hpp.
                                  // Not included to keep the saddle-
                                  // point solver header lightweight.

/**
 * @brief Krylov solver type for `SaddlePointSolver`.
 *
 * @details CG is intentionally absent — see class docstring.
 */
enum class KrylovType
{
    /// MINRES — the canonical choice for symmetric indefinite systems.
    /// Use when K is symmetric (which holds for linear elasticity and
    /// for any tangent stiffness derived from a symmetric integrator).
    MINRES,
    /// GMRES — for non-symmetric K (e.g. some plasticity formulations
    /// where the consistent tangent loses symmetry). More expensive
    /// per iteration than MINRES.
    GMRES,
    /// BiCGStab — alternative for non-symmetric systems. Sometimes
    /// converges faster than GMRES on saddle-point problems but is
    /// less robust.
    BiCGSTAB,
};

/**
 * @brief Preconditioner choice for the saddle-point Krylov solve.
 */
enum class SaddlePrecType
{
    /// Identity preconditioner. Useful for tiny problems and tests
    /// where Krylov converges quickly without acceleration. Not for
    /// production at any meaningful scale.
    None,
    /// Block-diagonal Jacobi:
    /// \f$P^{-1} = \mathrm{diag}(\mathrm{diag}(K)^{-1},
    /// \mathrm{diag}(C\,\mathrm{diag}(K)^{-1}\,C^T)^{-1})\f$.
    /// Cheap to build, GPU-friendly. Recommended default.
    BlockJacobi,
};

/**
 * @brief Configuration for `SaddlePointSolver`.
 */
struct SaddlePointSolverConfig
{
    KrylovType solver_type   = KrylovType::MINRES;
    SaddlePrecType prec_type = SaddlePrecType::BlockJacobi;
    double rel_tol           = 1.0e-10;
    double abs_tol           = 1.0e-12;
    int max_iter             = 500;
    /// MFEM Krylov print level: 0 silent, 1 first+last, 2 every iter.
    int print_level          = 0;
    /// GMRES restart parameter (k-dim). Defaults to 50 in MFEM; for
    /// small problems where the n-step finite-termination property
    /// matters, set this to a value larger than the global system
    /// size to disable restarting. Ignored for non-GMRES solvers.
    int gmres_kdim           = 50;
};

/**
 * @brief Distributed Krylov solver for one Newton step of the
 *        mortar-PBC saddle-point system.
 *
 * @details The solver is **stateless across calls** — every `Solve()`
 * builds its own `BlockOperator` and Krylov instance. Callers can
 * reuse the same `SaddlePointSolver` across Newton steps and across
 * load increments; the `K` and `C` arguments to `Solve()` are
 * non-owning references and may change between calls (which they
 * will, in a Newton outer loop where K is reassembled at each step).
 *
 * Convergence diagnostics from the most recent `Solve()` call are
 * available via `LastIterations()`, `LastConverged()`, and
 * `LastFinalNorm()`.
 *
 * @par MPI scope
 * `Solve()` is collective on `K.GetComm()` (which must equal
 * `C.GetComm()` and the multiplier-vector's communicator).
 *
 * @par GPU
 * The Krylov solver and `BlockOperator::Mult` dispatch correctly
 * regardless of whether K is HypreParMatrix or an MFEM Operator-only
 * PA / EA wrapper, because they only use the Mult interface. The
 * preconditioner currently uses K's diagonal via
 * `HypreParMatrix::GetDiag` — that's host-bound; switch to
 * `Operator::AssembleDiagonal` when adding PA-K support.
 */
class SaddlePointSolver
{
public:
    /**
     * @brief Construct with the given configuration.
     *
     * @param cfg  Solver configuration. Defaults are MINRES + block
     *             Jacobi + tight tolerances + 500 max iterations.
     *
     * @throws Aborts via MFEM_ABORT if `cfg.solver_type` is missing
     *         from the enum (defensive; the enum has no CG entry).
     */
    explicit SaddlePointSolver(
        const SaddlePointSolverConfig& cfg = SaddlePointSolverConfig{});

    // Non-copyable / non-movable: holds Krylov-solver scratch state.
    SaddlePointSolver(const SaddlePointSolver&) = delete;
    SaddlePointSolver& operator=(const SaddlePointSolver&) = delete;

    /**
     * @brief Solve one Newton step of the constrained system.
     *
     * @param[in]  K          Tangent stiffness as HypreParMatrix.
     *                        Caller owns; lifetime must exceed this
     *                        call.
     * @param[in]  C          Constraint matrix as HypreParMatrix
     *                        (typically from
     *                        `ConstraintBuilder3D::BuildHypreParMatrix`).
     * @param[in]  r1         Top Newton residual; size must equal
     *                        `K`'s local row count.
     * @param[in]  r2         Bottom Newton residual; size must equal
     *                        `C`'s local row count.
     * @param[out] du         Local TDOF slice of the velocity-block
     *                        increment. Will be sized to `K.Height()`.
     * @param[out] dlam       Local slice of the multiplier-block
     *                        increment. Will be sized to `C.Height()`.
     *
     * @par Newton step solved
     * For the constrained equilibrium
     * \f$F_{\mathrm{int}}(u) + C^T \lambda = 0\f$ with \f$C u = 0\f$,
     * the linearization at iterate \f$(u_k, \lambda_k)\f$ gives
     * @code
     *      [ K    C^T ] [ du ]   [ -r1 ]
     *      [ C    0   ] [ dλ ] = [ -r2 ]
     * @endcode
     * where the caller supplies
     * @code
     *      r1 = F_int(u_lin + u_k) + C^T λ_k    (force imbalance)
     *      r2 = C u_k                            (constraint violation)
     * @endcode
     *
     * @par Sign convention
     * The right-hand side is simply the negation of `(r1, r2)`.
     * Caller is responsible for forming the FULL Newton residual
     * including the `C^T λ_k` contribution; this matches what would
     * be required anyway to compute the Newton convergence check
     * \f$\|F_{\mathrm{int}} + C^T \lambda\|\f$.
     *
     * @par MPI scope
     * Collective on `K.GetComm()`. Issues one Krylov solve plus any
     * preconditioner-setup collectives.
     */
    void Solve(const mfem::HypreParMatrix& K,
               const mfem::HypreParMatrix& C,
               const mfem::Vector& r1,
               const mfem::Vector& r2,
               mfem::Vector& du,
               mfem::Vector& dlam);

    /**
     * @brief Phase 4.3 / Batch S — element-assembly path overload.
     *
     * @details Same Krylov solve as the HypreParMatrix overload, but
     * with the constraint matrix supplied as a
     * `MortarConstraintOperator` (the EA path) instead of a
     * `HypreParMatrix`. K stays as `HypreParMatrix` because that is
     * what the current patch-test driver assembles; switching K to
     * a matrix-free representation is a separate concern (Phase 5
     * for nonlinear K via `BlockNonlinearForm` + adapter).
     *
     * The block-Jacobi preconditioner uses
     * `MortarConstraintOperator::ComputeInvDiagSchur` (Batch R) for
     * the Schur-complement diagonal. The result is bit-equivalent
     * (modulo FP-summation order) to what `BuildInvDiagSchur` would
     * compute from the HypreParMatrix form of `C`.
     *
     * @param[in]  K          Tangent stiffness as `HypreParMatrix`.
     * @param[in]  C_op       Constraint operator as
     *                        `MortarConstraintOperator`.
     * @param[in]  r1         Top Newton residual.
     * @param[in]  r2         Bottom Newton residual.
     * @param[out] du         Velocity-block increment (sized
     *                        internally to `K.Height()`).
     * @param[out] dlam       Multiplier-block increment (sized
     *                        internally to `C_op.Height()`).
     *
     * @par MPI scope
     * Collective on `K.GetComm()`. Same collective profile as the
     * HypreParMatrix overload, plus one Allgather and one Allgatherv
     * for `inv_diag_K` inside `ComputeInvDiagSchur`. Each Krylov
     * iteration adds one `MPI_Alltoallv` (off-rank u-import for
     * `Mult`) and one `MPI_Alltoallv` (off-rank residual-export for
     * `MultTranspose`) — the EA matvec cost.
     */
    void Solve(const mfem::HypreParMatrix& K,
               const MortarConstraintOperator& C_op,
               const mfem::Vector& r1,
               const mfem::Vector& r2,
               mfem::Vector& du,
               mfem::Vector& dlam);

    /// Iterations used in the last `Solve()` call. -1 if no solve yet.
    int LastIterations() const { return m_last_iterations; }
    /// Did the last `Solve()` converge?
    bool LastConverged() const { return m_last_converged; }
    /// Final residual norm from the last `Solve()`.
    double LastFinalNorm() const { return m_last_final_norm; }

private:
    SaddlePointSolverConfig m_cfg;
    int m_last_iterations  = -1;
    bool m_last_converged  = false;
    double m_last_final_norm = -1.0;

    // Phase 4.3 / Batch S — shared inner-loop helper used by both
    // Solve overloads. Takes K and C as `mfem::Operator&` (caller
    // supplies the right type-safety casts) plus already-computed
    // `inv_diag_K` and `inv_diag_S` for the block-Jacobi
    // preconditioner. Builds the BlockOperator + BlockDiagonal
    // preconditioner + Krylov solver and runs one solve.
    //
    // Both `inv_diag_K` and `inv_diag_S` are passed by non-const
    // reference because the helper moves them into `DiagonalScaler`
    // instances (avoiding a per-iteration copy). After this call
    // returns, both vectors are in moved-from state.
    void SolveImplInternal(mfem::Operator& K_op,
                           mfem::Operator& C_op,
                           MPI_Comm comm,
                           mfem::Vector& inv_diag_K,
                           mfem::Vector& inv_diag_S,
                           int n_v_local,
                           int n_lam_local,
                           const mfem::Vector& r1,
                           const mfem::Vector& r2,
                           mfem::Vector& du,
                           mfem::Vector& dlam);
};

}  // namespace mortar_pbc
