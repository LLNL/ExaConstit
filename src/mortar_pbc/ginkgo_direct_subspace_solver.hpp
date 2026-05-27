// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// MFEM-to-Ginkgo adapter for AMGF mortar-PBC subspace solves.
//
// The AMGF implementation for mortar periodic boundary conditions uses MFEM's
// `AMGFSolver` as the displacement-block preconditioner. MFEM owns the
// filtering algorithm and the filtered coarse/subspace operator, but the
// subspace correction itself is exposed through the generic `mfem::Solver`
// interface. This header provides the small adapter layer that lets that
// interface call a Ginkgo sparse direct factorization.

#pragma once

#include "mfem.hpp"

#include <memory>
#include <string>

namespace gko {
class Executor;
class LinOp;
}  // namespace gko

namespace exaconstit::amgf {

/**
 * @brief Create the Ginkgo executor used by the AMGF filtered-subspace solve.
 *
 * @details The initial AMGF path is intentionally host-resident: option
 * validation rejects AMGF unless ExaConstit is running with FULL assembly on
 * CPU/OpenMP. Consequently, `"auto"` currently resolves to Ginkgo's OpenMP
 * executor, as does `"omp"`. GPU executor strings are rejected here rather than
 * silently selecting a different backend; the future hybrid GPU path will need
 * explicit ownership of matrix residency and transfer costs.
 *
 * @param spec User-facing executor selector from `amgf_subspace_executor`.
 *
 * @return Shared Ginkgo executor suitable for constructing matrices and direct
 * solvers in `GinkgoDirectSubspaceSolver`.
 */
std::shared_ptr<const gko::Executor> MakeGinkgoExecutor(
    const std::string& spec);

/**
 * @brief `mfem::Solver` adapter around a Ginkgo sparse direct solve.
 *
 * @details `mfem::AMGFSolver` forms the filtered subspace operator
 * \f$P^T A P\f$ and calls a user-provided `mfem::Solver` to apply the
 * subspace correction. This class satisfies that interface by converting the
 * local diagonal block of the supplied `mfem::HypreParMatrix` into a Ginkgo CSR
 * matrix and generating a Ginkgo `experimental::solver::Direct` solver.
 *
 * If `symmetric` is true, setup uses a sparse Cholesky factorization. If false,
 * setup uses sparse LU. The latter is intended for mildly nonsymmetric tangent
 * blocks from non-associated flow; the former is the expected path for the
 * symmetric K-block and Path-D augmented K-block cases.
 *
 * @warning This first adapter factors `HypreParMatrix::GetDiag()` only. That is
 * the local diagonal block of the distributed filtered operator, not a
 * distributed sparse-direct solve. It is the smallest useful bridge needed for
 * AMGF wiring and single-rank tests. Production multi-rank use should revisit
 * this assumption once MFEM's exact AMGF call shape is exercised on a genuinely
 * distributed filtered subspace.
 *
 * @par Setup cost
 * `SetOperator()` owns the expensive factorization and is expected to run once
 * per Newton operator rebuild. `Mult()` only performs dense-vector marshaling
 * plus the Ginkgo triangular/direct solve.
 */
class GinkgoDirectSubspaceSolver : public mfem::Solver
{
public:
    /**
     * @brief Construct an unfactored Ginkgo-backed subspace solver.
     *
     * @param exec Ginkgo executor selected from the AMGF options.
     * @param symmetric True for Cholesky, false for LU.
     */
    GinkgoDirectSubspaceSolver(std::shared_ptr<const gko::Executor> exec,
                               bool symmetric);

    /**
     * @brief Convert and factor the supplied AMGF subspace operator.
     *
     * @details The operator must be an `mfem::HypreParMatrix`, which is the
     * type produced by MFEM's Galerkin path for the AMGF filtered subspace.
     */
    void SetOperator(const mfem::Operator& op) override;

    /**
     * @brief Apply the cached direct solve to one right-hand side.
     */
    void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

private:
    std::shared_ptr<const gko::Executor> exec_;
    bool symmetric_ = true;
    std::unique_ptr<gko::LinOp> solver_;
};

}  // namespace exaconstit::amgf
