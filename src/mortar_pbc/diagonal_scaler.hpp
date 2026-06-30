#ifndef EXACONSTIT_MORTAR_PBC_DIAGONAL_SCALER_HPP
#define EXACONSTIT_MORTAR_PBC_DIAGONAL_SCALER_HPP

// Phase 5.5.B.2 — diagonal scaling solver, lifted out of
// saddle_point_solver.cpp's anonymous namespace into a shared header
// so MortarSaddlePreconditioner can reuse it without duplication.

#include "mfem.hpp"

#include <utility>

namespace mortar_pbc {

/**
 * @brief Diagonal-scaling solver: applies `y[i] = inv_diag[i] * x[i]`.
 *
 * @details Used for both the K block and the Schur block of the
 * block-Jacobi saddle-point preconditioner. Stateless beyond the
 * stored `inv_diag` vector — `SetOperator` is a no-op since the
 * scaling factors are baked in at construction time.
 *
 * @par Use as a Jacobi-prec probe target
 * Because `Mult(ones, y)` produces `y[i] = inv_diag[i]`, this class
 * doubles as a stand-in K-Jacobi preconditioner whose `Mult(ones)`
 * action exposes `diag(K)^{-1}` directly. This is the contract that
 * `MortarConstraintOperator::ComputeInvDiagSchur` relies on.
 *
 * @par Memory model
 * Phase 4.3.B / Batch X — host-only access via typed memory-manager
 * accessors (`HostRead` / `HostWrite`) so the class works under
 * MFEM's `DEVICE_DEBUG` mode. The block-Jacobi preconditioner that
 * uses this builds sub-vector views on its outputs; those views are
 * in "no valid copy" memory state on first use, and the unsafe
 * `GetData()` call would fail the
 *   `(Empty() || (flags & VALID_HOST))`
 * assertion. The typed accessors declare access intent to the
 * memory manager and avoid that.
 */
class DiagonalScaler : public mfem::Solver
{
public:
    /**
     * @brief Construct with explicit inverse-diagonal values.
     *
     * @param size      Operator size (height == width).
     * @param inv_diag  Vector of `1/diag(K)` values; size must equal
     *                  `size`. Moved into the solver.
     */
    DiagonalScaler(int size, mfem::Vector inv_diag)
        : mfem::Solver(size, size),
          m_inv_diag(std::move(inv_diag))
    {
        MFEM_VERIFY(m_inv_diag.Size() == size,
                    "DiagonalScaler: inv_diag size (" << m_inv_diag.Size()
                    << ") does not match operator size (" << size << ")");
    }

    /**
     * @brief Apply the inverse-diagonal scaling: `y[i] = inv_diag[i] * x[i]`.
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override
    {
        const int n = m_inv_diag.Size();
        MFEM_ASSERT(x.Size() == n && y.Size() == n,
                    "DiagonalScaler::Mult: size mismatch");
        const double* xd  = x.HostRead();
        const double* idd = m_inv_diag.HostRead();
        double*       yd  = y.HostWrite();
        for (int i = 0; i < n; ++i) { yd[i] = idd[i] * xd[i]; }
    }

    /**
     * @brief No-op. The inverse-diagonal is fixed at construction;
     *        the outer Jacobian/operator is not needed because the
     *        diagonal scaling acts purely on the input vector.
     */
    void SetOperator(const mfem::Operator& /*op*/) override {}

    /// Read-only access to the stored inverse diagonal.
    const mfem::Vector& InvDiag() const { return m_inv_diag; }

private:
    mfem::Vector m_inv_diag;
};

}  // namespace mortar_pbc

#endif  // EXACONSTIT_MORTAR_PBC_DIAGONAL_SCALER_HPP
