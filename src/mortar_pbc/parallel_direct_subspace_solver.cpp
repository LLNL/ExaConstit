// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors

#include "mortar_pbc/parallel_direct_subspace_solver.hpp"

#include "utilities/mechanics_log.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

// The backend classes (mfem::SuperLUSolver / SuperLURowLocMatrix,
// mfem::MUMPSSolver, mfem::STRUMPACKSolver / STRUMPACKRowLocMatrix, and the
// mfem::superlu / strumpack option enums) are declared by mfem.hpp — which the
// header already includes — whenever the matching MFEM_USE_* feature is on.
// We deliberately do NOT include the internal "linalg/superlu.hpp" headers
// directly: those resolve inside the MFEM source tree but not always for an
// external consumer's include path.
//
// CALI_CXX_MARK_SCOPE is provided (with a no-op fallback when Caliper is off)
// by utilities/mechanics_log.hpp, matching the rest of the AMGF sources.

namespace exaconstit::amgf {

DirectBackend ParseDirectBackend(const std::string& spec_in)
{
    std::string spec = spec_in;
    std::transform(spec.begin(), spec.end(), spec.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (spec == "auto") { return DirectBackend::AUTO; }
    if (spec == "superlu" || spec == "superlu_dist") { return DirectBackend::SUPERLU; }
    if (spec == "mumps") { return DirectBackend::MUMPS; }
    if (spec == "strumpack") { return DirectBackend::STRUMPACK; }

    MFEM_ABORT("Unknown AMGF subspace direct-solver backend '"
               << spec_in << "'. Supported values are 'auto', 'superlu', "
               << "'mumps', and 'strumpack'.");
    return DirectBackend::AUTO;  // unreachable
}

bool ParallelDirectSolverAvailable()
{
#if defined(MFEM_USE_SUPERLU) || defined(MFEM_USE_MUMPS) || \
    defined(MFEM_USE_STRUMPACK)
    return true;
#else
    return false;
#endif
}

const char* DefaultDirectBackendName()
{
#if defined(MFEM_USE_SUPERLU)
    return "superlu";
#elif defined(MFEM_USE_MUMPS)
    return "mumps";
#elif defined(MFEM_USE_STRUMPACK)
    return "strumpack";
#else
    return "none";
#endif
}

namespace {

/// Resolve AUTO to the first backend MFEM was built with, preferring
/// SuperLU_DIST (the backend the install scripts provision for AMGF).
DirectBackend ResolveBackend(DirectBackend requested)
{
    if (requested != DirectBackend::AUTO) { return requested; }
#if defined(MFEM_USE_SUPERLU)
    return DirectBackend::SUPERLU;
#elif defined(MFEM_USE_MUMPS)
    return DirectBackend::MUMPS;
#elif defined(MFEM_USE_STRUMPACK)
    return DirectBackend::STRUMPACK;
#else
    MFEM_ABORT("AMGF subspace solver requested but MFEM was built without any "
               "parallel sparse direct solver (SuperLU_DIST, MUMPS, or "
               "STRUMPACK). Rebuild MFEM with one of these backends (see "
               "scripts/install) or disable AMGF.");
    return DirectBackend::AUTO;  // unreachable
#endif
}

}  // namespace

ParallelDirectSubspaceSolver::ParallelDirectSubspaceSolver(
    MPI_Comm comm,
    DirectBackend backend,
    bool symmetric,
    int print_level)
    : mfem::Solver(0, 0),
      comm_(comm),
      backend_(ResolveBackend(backend)),
      symmetric_(symmetric),
      print_level_(print_level)
{
    MFEM_VERIFY(comm_ != MPI_COMM_NULL,
                "ParallelDirectSubspaceSolver requires a valid MPI "
                "communicator");
}

ParallelDirectSubspaceSolver::ParallelDirectSubspaceSolver(
    MPI_Comm comm,
    const std::string& backend_spec,
    bool symmetric,
    int print_level)
    : ParallelDirectSubspaceSolver(comm,
                                   ParseDirectBackend(backend_spec),
                                   symmetric,
                                   print_level)
{
}

ParallelDirectSubspaceSolver::~ParallelDirectSubspaceSolver() = default;

void ParallelDirectSubspaceSolver::InitBackend()
{
    switch (backend_)
    {
        case DirectBackend::SUPERLU:
        {
#ifdef MFEM_USE_SUPERLU
            auto slu = std::make_unique<mfem::SuperLUSolver>(comm_);

            // Fill-reducing ordering. SuperLU_DIST is built here WITHOUT
            // (Par)METIS (METIS-only ordering in SuperLU is reachable only
            // through ParMETIS, which is not a project dependency), so use the
            // built-in minimum-degree ordering on the symmetrized pattern
            // A^T + A. It needs no external ordering library and is a fine
            // choice for the small boundary-coupled subspace block.
            slu->SetColumnPermutation(mfem::superlu::MMD_AT_PLUS_A);

            // Symmetric-pattern hint improves the symbolic factorization for
            // the SPD subspace block. For the non-symmetric case fall back to
            // a static-pivoting row permutation (MC64) for numerical
            // stability; the SPD case needs no row permutation.
            slu->SetSymmetricPattern(symmetric_);
            slu->SetRowPermutation(symmetric_
                                       ? mfem::superlu::NOROWPERM
                                       : mfem::superlu::LargeDiag_MC64);

            // The boundary-coupled subspace block can be close to singular
            // (constraint-induced near-null modes are precisely what AMGF is
            // designed to absorb). Replacing tiny pivots keeps the
            // factorization well defined.
            slu->SetReplaceTinyPivot(true);

            slu->SetPrintStatistics(print_level_ != 0);
            solver_ = std::move(slu);
#else
            MFEM_ABORT("AMGF subspace backend SUPERLU selected but "
                       "MFEM_USE_SUPERLU is not defined.");
#endif
            break;
        }

        case DirectBackend::MUMPS:
        {
#ifdef MFEM_USE_MUMPS
            auto mumps = std::make_unique<mfem::MUMPSSolver>(comm_);
            mumps->SetMatrixSymType(
                symmetric_
                    ? mfem::MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE
                    : mfem::MUMPSSolver::MatType::UNSYMMETRIC);
            mumps->SetPrintLevel(print_level_);
            solver_ = std::move(mumps);
#else
            MFEM_ABORT("AMGF subspace backend MUMPS selected but "
                       "MFEM_USE_MUMPS is not defined.");
#endif
            break;
        }

        case DirectBackend::STRUMPACK:
        {
#ifdef MFEM_USE_STRUMPACK
            auto sp = std::make_unique<mfem::STRUMPACKSolver>(comm_);
            sp->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
            sp->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
            sp->SetMatching(strumpack::MatchingJob::NONE);
            sp->SetCompression(strumpack::CompressionType::NONE);
            sp->SetPrintFactorStatistics(print_level_ != 0);
            sp->SetPrintSolveStatistics(print_level_ != 0);
            solver_ = std::move(sp);
#else
            MFEM_ABORT("AMGF subspace backend STRUMPACK selected but "
                       "MFEM_USE_STRUMPACK is not defined.");
#endif
            break;
        }

        default:
            MFEM_ABORT("ParallelDirectSubspaceSolver: unresolved backend.");
    }
}

void ParallelDirectSubspaceSolver::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE(
        "exaconstit::amgf::parallel_direct_subspace_solver::set_operator");

    const auto* hypre_op = dynamic_cast<const mfem::HypreParMatrix*>(&op);
    MFEM_VERIFY(hypre_op != nullptr,
                "ParallelDirectSubspaceSolver expects an mfem::HypreParMatrix "
                "from AMGFSolver's filtered subspace operator");
    MFEM_VERIFY(hypre_op->Height() == hypre_op->Width(),
                "ParallelDirectSubspaceSolver requires a square subspace "
                "operator; got " << hypre_op->Height() << " x "
                << hypre_op->Width());

    height = hypre_op->Height();
    width = hypre_op->Width();

    // Rebuild the backend solver from scratch each Newton operator change so
    // that no stale symbolic/numeric factorization state can leak between
    // iterates. The boundary subspace is small, so this is cheap relative to
    // the BoomerAMG re-setup that happens in the same SetOperator pass.
    InitBackend();

    switch (backend_)
    {
        case DirectBackend::SUPERLU:
#ifdef MFEM_USE_SUPERLU
            // SuperLUSolver keeps a reference to its row-local matrix, so the
            // matrix is owned here for the lifetime of the factorization.
            superlu_mat_.reset(new mfem::SuperLURowLocMatrix(*hypre_op));
            solver_->SetOperator(*superlu_mat_);
#endif
            break;

        case DirectBackend::STRUMPACK:
#ifdef MFEM_USE_STRUMPACK
            strumpack_mat_.reset(new mfem::STRUMPACKRowLocMatrix(*hypre_op));
            solver_->SetOperator(*strumpack_mat_);
#endif
            break;

        case DirectBackend::MUMPS:
#ifdef MFEM_USE_MUMPS
            // MUMPS accepts the HypreParMatrix directly.
            solver_->SetOperator(*hypre_op);
#endif
            break;

        default:
            MFEM_ABORT("ParallelDirectSubspaceSolver::SetOperator: "
                       "unresolved backend.");
    }
}

void ParallelDirectSubspaceSolver::Mult(const mfem::Vector& b,
                                        mfem::Vector& x) const
{
    CALI_CXX_MARK_SCOPE(
        "exaconstit::amgf::parallel_direct_subspace_solver::mult");

    MFEM_VERIFY(solver_,
                "ParallelDirectSubspaceSolver::SetOperator must be called "
                "before Mult");
    MFEM_VERIFY(b.Size() == height,
                "ParallelDirectSubspaceSolver RHS has size " << b.Size()
                << " but expected " << height);

    x.SetSize(width);
    solver_->Mult(b, x);
}

}  // namespace exaconstit::amgf
