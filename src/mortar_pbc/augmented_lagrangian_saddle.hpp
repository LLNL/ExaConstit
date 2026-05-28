// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase D — augmented-Lagrangian saddle wrappers.

#pragma once

#include "mortar_constraint_operator.hpp"
#include "saddle_residual_scaler.hpp"

#include "mfem.hpp"

#include <memory>

namespace mortar_pbc {

/**
 * @brief Jacobian wrapper for the augmented-Lagrangian saddle method.
 *
 * @details The physical mortar residual remains
 *
 * \f[
 *     R(u,\lambda) =
 *     \begin{bmatrix}
 *       r_u(u) + C^T\lambda \\
 *       C u - g
 *     \end{bmatrix}.
 * \f]
 *
 * The augmented-Lagrangian linearization used for the Newton correction is
 *
 * \f[
 *     J_\gamma =
 *     \begin{bmatrix}
 *       K + \gamma C^T C & C^T \\
 *       C                & 0
 *     \end{bmatrix}.
 * \f]
 *
 * This class wraps the unaugmented saddle Jacobian returned by
 * `MortarSaddlePointSystem::GetGradient` and adds the
 * \f$\gamma C^T C\f$ contribution in `Mult`/`MultTranspose`. The wrapped
 * unaugmented Jacobian is still available through
 * `GetUnaugmentedGradient()` because the augmented preconditioner must build
 * \f$K_\gamma\f$ from the original mechanics K block, not from an already
 * augmented K block.
 */
class AugmentedLagrangianSaddleJacobian : public mfem::Operator
{
public:
    AugmentedLagrangianSaddleJacobian(
        mfem::Operator& unaugmented_gradient,
        std::shared_ptr<const MortarConstraintOperator> C_op,
        double gamma,
        const mfem::Array<int>& block_offsets);

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
    void MultTranspose(const mfem::Vector& x,
                       mfem::Vector& y) const override;

    void Refresh(mfem::Operator& unaugmented_gradient,
                 const mfem::Array<int>& block_offsets);

    mfem::Operator& GetUnaugmentedGradient() const
    {
        return *m_unaugmented_gradient;
    }

    double Gamma() const { return m_gamma; }

private:
    void AddAugmentedBlockContribution(const mfem::Vector& x,
                                       mfem::Vector& y) const;
    void EnsureScratch() const;

    mfem::Operator* m_unaugmented_gradient = nullptr;
    std::shared_ptr<const MortarConstraintOperator> m_C_op;
    double m_gamma = 0.0;
    mfem::Array<int> m_block_offsets;

    mutable mfem::Vector m_C_x_u;
    mutable mfem::Vector m_Ct_C_x_u;
};

/**
 * @brief Residual-preserving operator wrapper for the augmented saddle method.
 *
 * @details `Mult()` delegates directly to the wrapped physical saddle residual
 * operator. This is intentional: nonlinear convergence tests, line-search
 * acceptance, trust-region acceptance, and diagnostics must continue to measure
 * the true residual \f$R(u,\lambda)\f$, not the algebraically augmented linear
 * residual.
 *
 * `GetGradient()` returns an `AugmentedLagrangianSaddleJacobian`, so the linear
 * correction is computed with \f$K + \gamma C^T C\f$ in the displacement block.
 * The matching RHS augmentation is handled by
 * `AugmentedLagrangianRhsSolver`.
 */
class AugmentedLagrangianSaddleOperator : public mfem::Operator
{
public:
    AugmentedLagrangianSaddleOperator(
        std::shared_ptr<mfem::Operator> unaugmented_operator,
        std::shared_ptr<const MortarConstraintOperator> C_op,
        double gamma,
        const mfem::Array<int>& block_offsets);

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
    mfem::Operator& GetGradient(const mfem::Vector& x) const override;

    void Refresh(std::shared_ptr<mfem::Operator> unaugmented_operator,
                 const mfem::Array<int>& block_offsets);

    mfem::Operator& GetUnaugmentedOperator() const
    {
        return *m_unaugmented_operator;
    }

private:
    std::shared_ptr<mfem::Operator> m_unaugmented_operator;
    std::shared_ptr<const MortarConstraintOperator> m_C_op;
    double m_gamma = 0.0;
    mfem::Array<int> m_block_offsets;
    mutable std::unique_ptr<AugmentedLagrangianSaddleJacobian> m_gradient;
};

/**
 * @brief Linear-solver wrapper that applies the augmented RHS shift.
 *
 * @details ExaNewtonSolver supplies the current physical residual (or, when
 * saddle scaling is enabled, the scaled residual) to the configured linear
 * solver. For the augmented-Lagrangian saddle method the linear solve must use
 *
 * \f[
 *     b_{u,\gamma} = b_u + \gamma C^T b_\lambda,
 *     \qquad b_{\lambda,\gamma} = b_\lambda,
 * \f]
 *
 * where `b` is the residual-side right-hand side used by the existing Newton
 * convention. This wrapper performs that algebraic shift immediately before
 * delegating to the wrapped solver. When a saddle residual scaler is active,
 * the wrapper temporarily maps `b_solver` back to physical units, applies the
 * shift, then maps the shifted RHS back to solver units. The output increment
 * remains whatever coordinate system the wrapped solver promises; for the
 * existing scaled stack that is already `dx_phys`.
 */
class AugmentedLagrangianRhsSolver : public mfem::Solver
{
public:
    AugmentedLagrangianRhsSolver(
        std::shared_ptr<mfem::Solver> inner_solver,
        std::shared_ptr<const MortarConstraintOperator> C_op,
        double gamma,
        const mfem::Array<int>& block_offsets,
        std::shared_ptr<const SaddleResidualScaler> scaler = nullptr);

    void Mult(const mfem::Vector& b, mfem::Vector& x) const override;
    void SetOperator(const mfem::Operator& op) override;

    void Refresh(std::shared_ptr<mfem::Solver> inner_solver,
                 const mfem::Array<int>& block_offsets,
                 std::shared_ptr<const SaddleResidualScaler> scaler);

    mfem::Solver& GetInner() const { return *m_inner_solver; }

private:
    void EnsureScratch() const;
    void AddRhsShiftInPlace(mfem::BlockVector& rhs_phys) const;

    std::shared_ptr<mfem::Solver> m_inner_solver;
    std::shared_ptr<const MortarConstraintOperator> m_C_op;
    std::shared_ptr<const SaddleResidualScaler> m_scaler;
    double m_gamma = 0.0;
    mfem::Array<int> m_block_offsets;

    mutable mfem::Vector m_rhs_storage;
    mutable mfem::BlockVector m_rhs_view;
    mutable mfem::Vector m_C_rhs_lambda;
};

}  // namespace mortar_pbc
