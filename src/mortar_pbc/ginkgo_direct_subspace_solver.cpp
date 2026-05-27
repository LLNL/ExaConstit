// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors

// Include the guarded Ginkgo umbrella before mfem.hpp. Some Ginkgo public
// subheaders in the installed package are intentionally not standalone guarded,
// and MFEM's Ginkgo support also includes this umbrella.
#include <ginkgo/ginkgo.hpp>

#include "mortar_pbc/ginkgo_direct_subspace_solver.hpp"

#include "utilities/mechanics_log.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace exaconstit::amgf {

namespace {

using GinkgoIndex = int;
using GinkgoValue = double;
using GinkgoCsr = gko::matrix::Csr<GinkgoValue, GinkgoIndex>;
using GinkgoDense = gko::matrix::Dense<GinkgoValue>;
using GinkgoDirect =
    gko::experimental::solver::Direct<GinkgoValue, GinkgoIndex>;

std::unique_ptr<GinkgoCsr> ConvertLocalDiagBlockToGinkgoCsr(
    const mfem::HypreParMatrix& A,
    const std::shared_ptr<const gko::Executor>& exec)
{
    mfem::SparseMatrix local_diag;
    A.GetDiag(local_diag);

    MFEM_VERIFY(local_diag.Height() == local_diag.Width(),
                "GinkgoDirectSubspaceSolver requires a square local "
                "subspace block; got " << local_diag.Height() << " x "
                << local_diag.Width());

    const int nrows = local_diag.Height();
    const int nnz = local_diag.NumNonZeroElems();

    const int* mfem_i = local_diag.ReadI(false);
    const int* mfem_j = local_diag.ReadJ(false);
    const double* mfem_data = local_diag.ReadData(false);

    std::vector<GinkgoIndex> row_ptrs(
        mfem_i, mfem_i + static_cast<std::ptrdiff_t>(nrows + 1));
    std::vector<GinkgoIndex> col_idxs(
        mfem_j, mfem_j + static_cast<std::ptrdiff_t>(nnz));
    std::vector<GinkgoValue> values(
        mfem_data, mfem_data + static_cast<std::ptrdiff_t>(nnz));

    return GinkgoCsr::create(
        exec,
        gko::dim<2>{static_cast<gko::size_type>(nrows),
                    static_cast<gko::size_type>(nrows)},
        gko::array<GinkgoValue>(exec, values.begin(), values.end()),
        gko::array<GinkgoIndex>(exec, col_idxs.begin(), col_idxs.end()),
        gko::array<GinkgoIndex>(exec, row_ptrs.begin(), row_ptrs.end()));
}

std::unique_ptr<gko::LinOp> BuildDirectSolver(
    std::shared_ptr<const GinkgoCsr> matrix,
    const std::shared_ptr<const gko::Executor>& exec,
    bool symmetric)
{
    if (symmetric)
    {
        auto factory =
            GinkgoDirect::build()
                .with_factorization(
                    gko::experimental::factorization::Cholesky<
                        GinkgoValue, GinkgoIndex>::build())
                .on(exec);
        return factory->generate(matrix);
    }

    auto factory =
        GinkgoDirect::build()
            .with_factorization(
                gko::experimental::factorization::Lu<
                    GinkgoValue, GinkgoIndex>::build())
            .on(exec);
    return factory->generate(matrix);
}

}  // namespace

std::shared_ptr<const gko::Executor> MakeGinkgoExecutor(
    const std::string& spec)
{
    if (spec == "auto")
    {
#ifdef EXACONSTIT_GINKGO_HAS_OMP
        try
        {
            return gko::OmpExecutor::create();
        }
        catch (const std::exception& exc)
        {
            MFEM_WARNING(
                std::string("AMGF requested Ginkgo executor 'auto'; OMP executor is not "
                "available in this Ginkgo build, falling back to the "
                "reference executor. Ginkgo reported: ") + exc.what());
            return gko::ReferenceExecutor::create();
        }
#else
        return gko::ReferenceExecutor::create();
#endif
    }

    if (spec == "omp" || spec == "openmp")
    {
#ifdef EXACONSTIT_GINKGO_HAS_OMP
        try
        {
            return gko::OmpExecutor::create();
        }
        catch (const std::exception& exc)
        {
            MFEM_ABORT("AMGF requested Ginkgo executor '" << spec
                       << "', but the OMP executor is not available in this "
                       << "Ginkgo build. Use 'auto' for reference fallback or "
                       << "rebuild Ginkgo with OMP enabled. Ginkgo reported: "
                       << exc.what());
        }
#else
        MFEM_ABORT("AMGF requested Ginkgo executor '" << spec
                   << "', but the configured Ginkgo package was built without "
                   << "the OMP backend. Use 'auto' for reference fallback or "
                   << "rebuild Ginkgo with OMP enabled.");
#endif
    }
    if (spec == "reference" || spec == "ref")
    {
        return gko::ReferenceExecutor::create();
    }

    if (spec == "cuda" || spec == "hip" || spec == "dpcpp")
    {
        MFEM_ABORT("AMGF Ginkgo executor '" << spec
                   << "' is reserved for a future hybrid GPU path. "
                   << "The current AMGF implementation requires FULL "
                   << "assembly on CPU/OpenMP; use 'auto' or 'omp'.");
    }

    MFEM_ABORT("Unknown AMGF Ginkgo executor '" << spec
               << "'. Supported values are 'auto', 'omp', and 'reference'.");
    return nullptr;
}

GinkgoDirectSubspaceSolver::GinkgoDirectSubspaceSolver(
    std::shared_ptr<const gko::Executor> exec,
    bool symmetric)
    : exec_(std::move(exec)),
      symmetric_(symmetric)
{
    MFEM_VERIFY(exec_, "GinkgoDirectSubspaceSolver requires a non-null "
                       "Ginkgo executor");
}

void GinkgoDirectSubspaceSolver::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE(
        "exaconstit::amgf::ginkgo_direct_subspace_solver::set_operator");

    const auto* hypre_op = dynamic_cast<const mfem::HypreParMatrix*>(&op);
    MFEM_VERIFY(hypre_op != nullptr,
                "GinkgoDirectSubspaceSolver expects an mfem::HypreParMatrix "
                "from AMGFSolver's filtered subspace operator");

    MFEM_VERIFY(hypre_op->Height() == hypre_op->Width(),
                "GinkgoDirectSubspaceSolver requires a square subspace "
                "operator; got " << hypre_op->Height() << " x "
                << hypre_op->Width());

    height = hypre_op->Height();
    width = hypre_op->Width();

    auto matrix = ConvertLocalDiagBlockToGinkgoCsr(*hypre_op, exec_);
    solver_ = BuildDirectSolver(std::move(matrix), exec_, symmetric_);
}

void GinkgoDirectSubspaceSolver::Mult(
    const mfem::Vector& b,
    mfem::Vector& x) const
{
    CALI_CXX_MARK_SCOPE(
        "exaconstit::amgf::ginkgo_direct_subspace_solver::mult");

    MFEM_VERIFY(solver_,
                "GinkgoDirectSubspaceSolver::SetOperator must be called "
                "before Mult");
    MFEM_VERIFY(b.Size() == height,
                "GinkgoDirectSubspaceSolver RHS has size " << b.Size()
                << " but expected " << height);

    x.SetSize(width);

    const double* b_host = b.Read(false);
    std::vector<GinkgoValue> rhs(
        b_host, b_host + static_cast<std::ptrdiff_t>(b.Size()));
    auto gko_b = GinkgoDense::create(
        exec_,
        gko::dim<2>{static_cast<gko::size_type>(height), 1},
        gko::array<GinkgoValue>(exec_, rhs.begin(), rhs.end()),
        1);
    auto gko_x = GinkgoDense::create(
        exec_,
        gko::dim<2>{static_cast<gko::size_type>(width), 1});

    solver_->apply(gko_b.get(), gko_x.get());

    auto gko_x_host = gko::clone(exec_->get_master(), gko_x);
    const GinkgoValue* values = gko_x_host->get_const_values();
    for (int i = 0; i < width; ++i)
    {
        x[i] = values[i];
    }
}

}  // namespace exaconstit::amgf
