// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Unit tests for the AMGF-backed mortar saddle preconditioner.

#include "boundary_classifier_3d.hpp"
#include "diagonal_scaler.hpp"
#include "mortar_pbc/amgf_utils.hpp"
#include "mortar_pbc/ginkgo_direct_subspace_solver.hpp"
#include "mortar_pbc/mortar_saddle_preconditioner_amgf.hpp"
#include "mortar_constraint_operator.hpp"

#include "mfem.hpp"
#include "mpi.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::DiagonalScaler;
using mortar_pbc::MortarConstraintOperator;
using mortar_pbc::MortarSaddlePreconditionerAMGF;

namespace {

void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

struct FesBundle
{
    std::unique_ptr<mfem::ParMesh> pmesh;
    std::unique_ptr<mfem::H1_FECollection> fec;
    std::unique_ptr<mfem::ParFiniteElementSpace> fes;
};

FesBundle BuildHexFesBundle(MPI_Comm comm, int n_per_side)
{
    FesBundle b;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n_per_side, n_per_side, n_per_side,
        mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
    b.fec = std::make_unique<mfem::H1_FECollection>(/*order=*/1, /*dim=*/3);
    b.fes = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3, mfem::Ordering::byNODES);
    return b;
}

std::unique_ptr<mfem::HypreParMatrix> BuildPinnedElasticityHypre(
    mfem::ParMesh& pmesh, mfem::ParFiniteElementSpace& fes)
{
    mfem::Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 1;
    mfem::Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    const double E = 100.0;
    const double nu = 0.3;
    const double mu = 0.5 * E / (1.0 + nu);
    const double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    mfem::ConstantCoefficient lam_coef(lam);
    mfem::ConstantCoefficient mu_coef(mu);

    mfem::ParBilinearForm a(&fes);
    a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lam_coef, mu_coef));
    a.Assemble();
    a.Finalize();

    std::unique_ptr<mfem::HypreParMatrix> hypre_A(a.ParallelAssemble());
    std::unique_ptr<mfem::HypreParMatrix> eliminated(
        hypre_A->EliminateRowsCols(ess_tdofs));
    hypre_A->EliminateZeroRows();
    return hypre_A;
}

void TestConstructsAndSetOperator()
{
    const std::string name =
        "MortarSaddlePreconditionerAMGF construction and SetOperator";

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D classifier(*b.pmesh, *b.fes);
    auto C_op = std::make_shared<MortarConstraintOperator>(classifier);

    const int n_K = C_op->Width();
    const int n_lam = C_op->Height();
    AssertOrDie(n_K > 0 && n_lam > 0, name,
                "expected non-empty displacement and constraint spaces");

    std::unique_ptr<mfem::HypreParMatrix> K(
        BuildPinnedElasticityHypre(*b.pmesh, *b.fes));

    std::unique_ptr<mfem::HypreParMatrix> P(
        exaconstit::amgf::BuildBooleanRestrictionProlongation(
            K->GetGlobalNumRows(),
            C_op->GetConstraintCoupledDofIndices(),
            K->GetRowStarts(),
            MPI_COMM_WORLD));
    AssertOrDie(P->Width() > 0, name,
                "AMGF transfer P should have at least one filtered column");
    const int n_filter = P->Width();

    mfem::Vector inv_diag_K(n_K);
    inv_diag_K = 0.01;
    auto K_jacobi_prec =
        std::make_shared<DiagonalScaler>(n_K, inv_diag_K);
    auto subspace_solver =
        std::make_shared<exaconstit::amgf::GinkgoDirectSubspaceSolver>(
            exaconstit::amgf::MakeGinkgoExecutor("auto"),
            /*symmetric=*/true);

    MortarSaddlePreconditionerAMGF prec(
        K_jacobi_prec, C_op, std::move(P), subspace_solver,
        /*use_path_d=*/false, /*gamma_override=*/-1.0,
        /*vector_dim=*/1, /*order_bynodes=*/true, /*print_level=*/0);

    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = n_K;
    offsets[2] = n_K + n_lam;

    mfem::BlockOperator saddle(offsets);
    saddle.SetBlock(0, 0, K.get());

    prec.SetOperator(saddle);

    AssertOrDie(prec.Height() == n_K + n_lam, name,
                "unexpected preconditioner height");
    AssertOrDie(prec.Width() == n_K + n_lam, name,
                "unexpected preconditioner width");
    AssertOrDie(std::abs(prec.gamma()) < 1.0e-14, name,
                "Path-A gamma should remain zero");

    std::cout << "  PASS  " << name << " (n_K = " << n_K
              << ", n_lam = " << n_lam
              << ", n_filter = " << n_filter << ")"
              << std::endl;
}

void TestPathASchurBlockMatchesExistingDiagonalProbe()
{
    const std::string name =
        "MortarSaddlePreconditionerAMGF Path-A Schur diagonal";

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D classifier(*b.pmesh, *b.fes);
    auto C_op = std::make_shared<MortarConstraintOperator>(classifier);

    const int n_K = C_op->Width();
    const int n_lam = C_op->Height();
    std::unique_ptr<mfem::HypreParMatrix> K(
        BuildPinnedElasticityHypre(*b.pmesh, *b.fes));

    std::unique_ptr<mfem::HypreParMatrix> P(
        exaconstit::amgf::BuildBooleanRestrictionProlongation(
            K->GetGlobalNumRows(),
            C_op->GetConstraintCoupledDofIndices(),
            K->GetRowStarts(),
            MPI_COMM_WORLD));

    mfem::Vector inv_diag_K(n_K);
    inv_diag_K = 0.01;
    auto K_jacobi_prec =
        std::make_shared<DiagonalScaler>(n_K, inv_diag_K);
    auto subspace_solver =
        std::make_shared<exaconstit::amgf::GinkgoDirectSubspaceSolver>(
            exaconstit::amgf::MakeGinkgoExecutor("auto"),
            /*symmetric=*/true);

    MortarSaddlePreconditionerAMGF prec(
        K_jacobi_prec, C_op, std::move(P), subspace_solver,
        /*use_path_d=*/false, /*gamma_override=*/-1.0,
        /*vector_dim=*/1, /*order_bynodes=*/true, /*print_level=*/0);

    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = n_K;
    offsets[2] = n_K + n_lam;

    mfem::BlockOperator saddle(offsets);
    saddle.SetBlock(0, 0, K.get());
    prec.SetOperator(saddle);

    DiagonalScaler reference_probe(n_K, inv_diag_K);
    mfem::Vector expected_inv_diag_S =
        C_op->ComputeInvDiagSchur(reference_probe);
    AssertOrDie(expected_inv_diag_S.Size() == n_lam, name,
                "unexpected reference Schur diagonal size");

    constexpr double tol = 1.0e-11;
    double max_err = 0.0;
    const mfem::Vector& actual_inv_diag_S =
        prec.GetPathAInverseSchurDiagonal();
    AssertOrDie(actual_inv_diag_S.Size() == n_lam, name,
                "unexpected AMGF preconditioner Schur diagonal size");
    for (int i = 0; i < n_lam; ++i)
    {
        max_err = std::max(max_err,
                           std::abs(actual_inv_diag_S[i]
                                    - expected_inv_diag_S[i]));
    }
    AssertOrDie(max_err < tol, name,
                "inverse Schur diagonal differs from existing path by "
                + std::to_string(max_err));

    std::cout << "  PASS  " << name << " (max lower-block error = "
              << max_err << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "Running MortarSaddlePreconditionerAMGF tests"
                  << std::endl;
        std::cout << "------------------------------------------------"
                  << std::endl;
    }

    TestConstructsAndSetOperator();
    TestPathASchurBlockMatchesExistingDiagonalProbe();

    if (rank == 0)
    {
        std::cout << "------------------------------------------------"
                  << std::endl;
        std::cout << "All MortarSaddlePreconditionerAMGF tests passed."
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
