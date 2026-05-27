// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Unit tests for mortar-PBC AMGF setup utilities.

#include "mortar_pbc/amgf_utils.hpp"

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

std::unique_ptr<mfem::HypreParMatrix> BuildDiagonalHypreMatrix(
    MPI_Comm comm, HYPRE_BigInt n_global, const HYPRE_BigInt* row_starts)
{
    const int n_local = static_cast<int>(row_starts[1] - row_starts[0]);
    mfem::SparseMatrix local(n_local, static_cast<int>(n_global));
    for (int i = 0; i < n_local; ++i)
    {
        const HYPRE_BigInt global_row = row_starts[0] + i;
        local.Add(i, static_cast<int>(global_row),
                  static_cast<double>(global_row + 1));
    }
    local.Finalize();

    return std::unique_ptr<mfem::HypreParMatrix>(
        new mfem::HypreParMatrix(comm, n_local, n_global, n_global,
                                 local.ReadI(false), local.ReadJ(false),
                                 local.ReadData(false), row_starts,
                                 row_starts));
}

void TestBooleanProlongationMult()
{
    const std::string name = "Boolean AMGF prolongation Mult";
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const HYPRE_BigInt n_global = 10;
    const HYPRE_BigInt row_starts[2] = {0, n_global};

    // Include duplicates and rely on the utility to sort/unique the union.
    const std::vector<HYPRE_BigInt> idx =
        (rank == 0) ? std::vector<HYPRE_BigInt>{7, 0, 3, 3}
                    : std::vector<HYPRE_BigInt>{};

    std::unique_ptr<mfem::HypreParMatrix> P(
        exaconstit::amgf::BuildBooleanRestrictionProlongation(
            n_global, idx, row_starts, comm));

    AssertOrDie(P->GetGlobalNumRows() == n_global, name,
                "unexpected global row count");
    AssertOrDie(P->GetGlobalNumCols() == 3, name,
                "unexpected compact column count");

    mfem::Vector x(P->Width());
    x[0] = 10.0;
    x[1] = 20.0;
    x[2] = 30.0;

    mfem::Vector y(P->Height());
    P->Mult(x, y);

    for (int i = 0; i < y.Size(); ++i)
    {
        double expected = 0.0;
        if (i == 0) { expected = 10.0; }
        if (i == 3) { expected = 20.0; }
        if (i == 7) { expected = 30.0; }
        AssertOrDie(std::abs(y[i] - expected) < 1.0e-14,
                    name, "unexpected y[" + std::to_string(i) + "]");
    }

    mfem::Vector z(P->Width());
    P->MultTranspose(y, z);
    AssertOrDie(std::abs(z[0] - 10.0) < 1.0e-14, name,
                "unexpected transpose entry 0");
    AssertOrDie(std::abs(z[1] - 20.0) < 1.0e-14, name,
                "unexpected transpose entry 1");
    AssertOrDie(std::abs(z[2] - 30.0) < 1.0e-14, name,
                "unexpected transpose entry 2");
}

void TestBooleanProlongationRAP()
{
    const std::string name = "Boolean AMGF prolongation RAP";
    MPI_Comm comm = MPI_COMM_WORLD;
    const HYPRE_BigInt n_global = 10;
    const HYPRE_BigInt row_starts[2] = {0, n_global};
    const std::vector<HYPRE_BigInt> idx = {0, 3, 7};

    std::unique_ptr<mfem::HypreParMatrix> A(
        BuildDiagonalHypreMatrix(comm, n_global, row_starts));
    std::unique_ptr<mfem::HypreParMatrix> P(
        exaconstit::amgf::BuildBooleanRestrictionProlongation(
            n_global, idx, row_starts, comm));
    std::unique_ptr<mfem::HypreParMatrix> Ac(mfem::RAP(A.get(), P.get()));

    mfem::Vector x(Ac->Width());
    x[0] = 1.0;
    x[1] = 1.0;
    x[2] = 1.0;
    mfem::Vector y(Ac->Height());
    Ac->Mult(x, y);

    AssertOrDie(y.Size() == 3, name, "unexpected coarse local height");
    AssertOrDie(std::abs(y[0] - 1.0) < 1.0e-14, name,
                "coarse diag entry for row 0 should be A(0,0)");
    AssertOrDie(std::abs(y[1] - 4.0) < 1.0e-14, name,
                "coarse diag entry for row 3 should be A(3,3)");
    AssertOrDie(std::abs(y[2] - 8.0) < 1.0e-14, name,
                "coarse diag entry for row 7 should be A(7,7)");
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    TestBooleanProlongationMult();
    TestBooleanProlongationRAP();
    MPI_Finalize();
    return 0;
}
