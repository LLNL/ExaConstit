// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Unit tests for the Ginkgo-backed AMGF filtered-subspace solver adapter.

#include "mortar_pbc/ginkgo_direct_subspace_solver.hpp"

#include "mfem.hpp"
#include "mpi.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

std::unique_ptr<mfem::HypreParMatrix> BuildHypreMatrix(
    MPI_Comm comm,
    const std::vector<std::vector<double>>& dense_rows)
{
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    AssertOrDie(nranks == 1, "BuildHypreMatrix",
                "this focused adapter test is registered for one MPI rank");

    const int n = static_cast<int>(dense_rows.size());
    const HYPRE_BigInt row_starts[2] = {0, n};

    mfem::SparseMatrix local(n, n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (std::abs(dense_rows[i][j]) > 0.0)
            {
                local.Add(i, j, dense_rows[i][j]);
            }
        }
    }
    local.Finalize();

    return std::unique_ptr<mfem::HypreParMatrix>(
        new mfem::HypreParMatrix(comm, n, n, n,
                                 local.ReadI(false), local.ReadJ(false),
                                 local.ReadData(false), row_starts,
                                 row_starts));
}

mfem::Vector MatVec(const std::vector<std::vector<double>>& A,
                    const mfem::Vector& x)
{
    mfem::Vector b(x.Size());
    b = 0.0;
    for (int i = 0; i < x.Size(); ++i)
    {
        for (int j = 0; j < x.Size(); ++j)
        {
            b[i] += A[i][j] * x[j];
        }
    }
    return b;
}

void AssertVectorNear(const mfem::Vector& actual,
                      const mfem::Vector& expected,
                      const std::string& test_name)
{
    AssertOrDie(actual.Size() == expected.Size(), test_name,
                "unexpected vector size");
    for (int i = 0; i < actual.Size(); ++i)
    {
        AssertOrDie(std::abs(actual[i] - expected[i]) < 1.0e-11,
                    test_name,
                    "entry " + std::to_string(i) + " differs: got " +
                    std::to_string(actual[i]) + ", expected " +
                    std::to_string(expected[i]));
    }
}

void TestCholeskySolve()
{
    const std::string name = "Ginkgo direct subspace Cholesky solve";

    const std::vector<std::vector<double>> A = {
        {4.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 2.0}};

    mfem::Vector expected(3);
    expected[0] = 1.0;
    expected[1] = 2.0;
    expected[2] = 3.0;

    std::unique_ptr<mfem::HypreParMatrix> hypre_A =
        BuildHypreMatrix(MPI_COMM_WORLD, A);
    exaconstit::amgf::GinkgoDirectSubspaceSolver solver(
        exaconstit::amgf::MakeGinkgoExecutor("auto"), true);
    solver.SetOperator(*hypre_A);

    mfem::Vector x;
    solver.Mult(MatVec(A, expected), x);
    AssertVectorNear(x, expected, name);
}

void TestLuSolve()
{
    const std::string name = "Ginkgo direct subspace LU solve";

    // Structurally symmetric but numerically nonsymmetric. This mirrors the
    // non-associated-flow path without forcing Ginkgo to build a fully general
    // nonsymmetric symbolic factorization in this focused unit test.
    const std::vector<std::vector<double>> A = {
        {2.0, 1.0, 0.0},
        {4.0, 3.0, 1.0},
        {0.0, 5.0, 4.0}};

    mfem::Vector expected(3);
    expected[0] = 1.0;
    expected[1] = 2.0;
    expected[2] = -1.0;

    std::unique_ptr<mfem::HypreParMatrix> hypre_A =
        BuildHypreMatrix(MPI_COMM_WORLD, A);
    exaconstit::amgf::GinkgoDirectSubspaceSolver solver(
        exaconstit::amgf::MakeGinkgoExecutor("auto"), false);
    solver.SetOperator(*hypre_A);

    mfem::Vector x;
    solver.Mult(MatVec(A, expected), x);
    AssertVectorNear(x, expected, name);
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    TestCholeskySolve();
    TestLuSolve();
    MPI_Finalize();
    return 0;
}
