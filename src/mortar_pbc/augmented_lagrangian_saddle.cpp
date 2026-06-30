// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase D — augmented-Lagrangian saddle wrappers implementation.

#include "augmented_lagrangian_saddle.hpp"

#include "utilities/mechanics_log.hpp"

#include "mfem/general/forall.hpp"

#include <utility>

namespace mortar_pbc {

namespace {

void CopyVector(const mfem::Vector& src, mfem::Vector& dst)
{
    MFEM_VERIFY(src.Size() == dst.Size(),
                "CopyVector: size mismatch");
    const int n = src.Size();
    const double* s = src.Read();
    double* d = dst.Write();
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int i) { d[i] = s[i]; });
}

void EnsureBlockScratch(mfem::Vector& storage,
                        mfem::BlockVector& view,
                        const mfem::Array<int>& offsets)
{
    const int total = offsets.Last();
    if (storage.Size() != total) {
        storage.SetSize(total, mfem::Device::GetMemoryType());
        storage.UseDevice(true);
    }
    view.Update(storage, offsets);
}

}  // namespace

AugmentedLagrangianSaddleJacobian::AugmentedLagrangianSaddleJacobian(
    mfem::Operator& unaugmented_gradient,
    std::shared_ptr<const MortarConstraintOperator> C_op,
    double gamma,
    const mfem::Array<int>& block_offsets)
    : mfem::Operator(unaugmented_gradient.Height(),
                     unaugmented_gradient.Width())
    , m_unaugmented_gradient(&unaugmented_gradient)
    , m_C_op(std::move(C_op))
    , m_gamma(gamma)
    , m_block_offsets(block_offsets)
{
    MFEM_VERIFY(m_C_op,
                "AugmentedLagrangianSaddleJacobian: C_op is null");
    MFEM_VERIFY(m_block_offsets.Size() == 3,
                "AugmentedLagrangianSaddleJacobian: expected two saddle "
                "blocks");
    MFEM_VERIFY(m_block_offsets.Last() == Height(),
                "AugmentedLagrangianSaddleJacobian: block offsets do not "
                "match operator height");
}

void AugmentedLagrangianSaddleJacobian::EnsureScratch() const
{
    const int n_lam = m_block_offsets[2] - m_block_offsets[1];
    const int n_u = m_block_offsets[1] - m_block_offsets[0];
    if (m_C_x_u.Size() != n_lam) {
        m_C_x_u.SetSize(n_lam, mfem::Device::GetMemoryType());
        m_C_x_u.UseDevice(true);
    }
    if (m_Ct_C_x_u.Size() != n_u) {
        m_Ct_C_x_u.SetSize(n_u, mfem::Device::GetMemoryType());
        m_Ct_C_x_u.UseDevice(true);
    }
}

void AugmentedLagrangianSaddleJacobian::AddAugmentedBlockContribution(
    const mfem::Vector& x,
    mfem::Vector& y) const
{
    EnsureScratch();

    const double* x_data = x.HostRead();
    double* y_data = y.HostReadWrite();
    mfem::Vector x_u(const_cast<double*>(x_data),
                     m_block_offsets[1] - m_block_offsets[0]);
    mfem::Vector y_u(y_data, m_block_offsets[1] - m_block_offsets[0]);

    m_C_op->Mult(x_u, m_C_x_u);
    m_C_op->MultTranspose(m_C_x_u, m_Ct_C_x_u);
    y_u.Add(m_gamma, m_Ct_C_x_u);
}

void AugmentedLagrangianSaddleJacobian::Mult(const mfem::Vector& x,
                                             mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::augmented_saddle_jacobian::mult");
    m_unaugmented_gradient->Mult(x, y);
    AddAugmentedBlockContribution(x, y);
}

void AugmentedLagrangianSaddleJacobian::MultTranspose(
    const mfem::Vector& x,
    mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::augmented_saddle_jacobian::mult_transpose");
    m_unaugmented_gradient->MultTranspose(x, y);
    AddAugmentedBlockContribution(x, y);
}

void AugmentedLagrangianSaddleJacobian::Refresh(
    mfem::Operator& unaugmented_gradient,
    const mfem::Array<int>& block_offsets)
{
    m_unaugmented_gradient = &unaugmented_gradient;
    m_block_offsets = block_offsets;
    height = unaugmented_gradient.Height();
    width = unaugmented_gradient.Width();
    EnsureScratch();
}

AugmentedLagrangianSaddleOperator::AugmentedLagrangianSaddleOperator(
    std::shared_ptr<mfem::Operator> unaugmented_operator,
    std::shared_ptr<const MortarConstraintOperator> C_op,
    double gamma,
    const mfem::Array<int>& block_offsets)
    : mfem::Operator(unaugmented_operator->Height(),
                     unaugmented_operator->Width())
    , m_unaugmented_operator(std::move(unaugmented_operator))
    , m_C_op(std::move(C_op))
    , m_gamma(gamma)
    , m_block_offsets(block_offsets)
{
    MFEM_VERIFY(m_unaugmented_operator,
                "AugmentedLagrangianSaddleOperator: inner operator is null");
    MFEM_VERIFY(m_C_op,
                "AugmentedLagrangianSaddleOperator: C_op is null");
}

void AugmentedLagrangianSaddleOperator::Mult(const mfem::Vector& x,
                                             mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::augmented_saddle_operator::mult");
    m_unaugmented_operator->Mult(x, y);
}

mfem::Operator& AugmentedLagrangianSaddleOperator::GetGradient(
    const mfem::Vector& x) const
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::augmented_saddle_operator::get_gradient");
    mfem::Operator& unaugmented_gradient =
        m_unaugmented_operator->GetGradient(x);

    if (!m_gradient) {
        m_gradient = std::make_unique<AugmentedLagrangianSaddleJacobian>(
            unaugmented_gradient, m_C_op, m_gamma, m_block_offsets);
    }
    else {
        m_gradient->Refresh(unaugmented_gradient, m_block_offsets);
    }
    return *m_gradient;
}

void AugmentedLagrangianSaddleOperator::Refresh(
    std::shared_ptr<mfem::Operator> unaugmented_operator,
    const mfem::Array<int>& block_offsets)
{
    MFEM_VERIFY(unaugmented_operator,
                "AugmentedLagrangianSaddleOperator::Refresh: inner "
                "operator is null");
    m_unaugmented_operator = std::move(unaugmented_operator);
    m_block_offsets = block_offsets;
    height = m_unaugmented_operator->Height();
    width = m_unaugmented_operator->Width();
    m_gradient.reset();
}

AugmentedLagrangianRhsSolver::AugmentedLagrangianRhsSolver(
    std::shared_ptr<mfem::Solver> inner_solver,
    std::shared_ptr<const MortarConstraintOperator> C_op,
    double gamma,
    const mfem::Array<int>& block_offsets,
    std::shared_ptr<const SaddleResidualScaler> scaler)
    : mfem::Solver(inner_solver->Height(), inner_solver->Width())
    , m_inner_solver(std::move(inner_solver))
    , m_C_op(std::move(C_op))
    , m_scaler(std::move(scaler))
    , m_gamma(gamma)
    , m_block_offsets(block_offsets)
{
    MFEM_VERIFY(m_inner_solver,
                "AugmentedLagrangianRhsSolver: inner solver is null");
    MFEM_VERIFY(m_C_op, "AugmentedLagrangianRhsSolver: C_op is null");
    EnsureScratch();
}

void AugmentedLagrangianRhsSolver::EnsureScratch() const
{
    EnsureBlockScratch(m_rhs_storage, m_rhs_view, m_block_offsets);
    const int n_u = m_block_offsets[1] - m_block_offsets[0];
    if (m_C_rhs_lambda.Size() != n_u) {
        m_C_rhs_lambda.SetSize(n_u, mfem::Device::GetMemoryType());
        m_C_rhs_lambda.UseDevice(true);
    }
}

void AugmentedLagrangianRhsSolver::AddRhsShiftInPlace(
    mfem::BlockVector& rhs_phys) const
{
    mfem::Vector& rhs_u = rhs_phys.GetBlock(0);
    mfem::Vector& rhs_lam = rhs_phys.GetBlock(1);
    m_C_op->MultTranspose(rhs_lam, m_C_rhs_lambda);
    rhs_u.Add(m_gamma, m_C_rhs_lambda);
}

void AugmentedLagrangianRhsSolver::Mult(const mfem::Vector& b,
                                        mfem::Vector& x) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::augmented_rhs_solver::mult");
    MFEM_VERIFY(b.Size() == Height(),
                "AugmentedLagrangianRhsSolver::Mult: RHS size mismatch");
    MFEM_VERIFY(x.Size() == Width(),
                "AugmentedLagrangianRhsSolver::Mult: solution size "
                "mismatch");

    EnsureScratch();
    CopyVector(b, static_cast<mfem::Vector&>(m_rhs_view));

    if (m_scaler && m_scaler->IsEnabled()) {
        // b_solver -> b_phys, apply the algebraic augmented RHS shift in
        // physical units, then map back to the solver-scaled RHS expected by
        // the wrapped ScaledSaddleSolver.
        m_scaler->UnapplyToIncrement(m_rhs_view);
        AddRhsShiftInPlace(m_rhs_view);
        m_scaler->ApplyToResidual(m_rhs_view);
    }
    else {
        AddRhsShiftInPlace(m_rhs_view);
    }

    m_inner_solver->iterative_mode = iterative_mode;
    m_inner_solver->Mult(m_rhs_view, x);
}

void AugmentedLagrangianRhsSolver::SetOperator(const mfem::Operator& op)
{
    m_inner_solver->SetOperator(op);
    height = op.Height();
    width = op.Width();
}

void AugmentedLagrangianRhsSolver::Refresh(
    std::shared_ptr<mfem::Solver> inner_solver,
    const mfem::Array<int>& block_offsets,
    std::shared_ptr<const SaddleResidualScaler> scaler)
{
    MFEM_VERIFY(inner_solver,
                "AugmentedLagrangianRhsSolver::Refresh: inner solver is null");
    m_inner_solver = std::move(inner_solver);
    m_block_offsets = block_offsets;
    m_scaler = std::move(scaler);
    height = m_inner_solver->Height();
    width = m_inner_solver->Width();
    EnsureScratch();
}

}  // namespace mortar_pbc
