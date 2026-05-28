// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase D — tests for augmented-Lagrangian saddle wrappers.

#include "augmented_lagrangian_saddle.hpp"
#include "boundary_classifier_3d.hpp"
#include "mortar_constraint_operator.hpp"

#include "mfem.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using mortar_pbc::AugmentedLagrangianRhsSolver;
using mortar_pbc::AugmentedLagrangianSaddleJacobian;
using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::MortarConstraintOperator;

namespace {

void AssertOrDie(bool cond,
                 const std::string& test_name,
                 const std::string& detail)
{
    if (!cond) {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

void AssertVectorNear(const mfem::Vector& actual,
                      const mfem::Vector& expected,
                      double tol,
                      const std::string& test_name)
{
    AssertOrDie(actual.Size() == expected.Size(), test_name, "size mismatch");
    const double* a = actual.HostRead();
    const double* e = expected.HostRead();
    for (int i = 0; i < actual.Size(); ++i) {
        if (std::abs(a[i] - e[i]) > tol) {
            AssertOrDie(false,
                        test_name,
                        "entry " + std::to_string(i)
                        + " got " + std::to_string(a[i])
                        + " expected " + std::to_string(e[i]));
        }
    }
}

struct FesBundle
{
    std::unique_ptr<mfem::ParMesh> pmesh;
    std::unique_ptr<mfem::H1_FECollection> fec;
    std::unique_ptr<mfem::ParFiniteElementSpace> fes;
};

FesBundle BuildHexFesBundle(MPI_Comm comm)
{
    FesBundle b;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        2, 2, 2,
        mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
    b.fec = std::make_unique<mfem::H1_FECollection>(1, 3);
    b.fes = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), 3, mfem::Ordering::byNODES);
    return b;
}

void FillDeterministic(mfem::Vector& v, double scale)
{
    double* data = v.HostWrite();
    for (int i = 0; i < v.Size(); ++i) {
        data[i] = scale * static_cast<double>((i % 7) - 3);
    }
}

class RecordingSolver : public mfem::Solver
{
public:
    explicit RecordingSolver(int n)
        : mfem::Solver(n, n)
        , last_rhs(n)
    {
        last_rhs = 0.0;
    }

    void SetOperator(const mfem::Operator& op) override
    {
        height = op.Height();
        width = op.Width();
        last_rhs.SetSize(height);
    }

    void Mult(const mfem::Vector& b, mfem::Vector& x) const override
    {
        last_rhs = b;
        x = b;
    }

    mutable mfem::Vector last_rhs;
};

void TestAugmentedJacobianAddsCtC()
{
    const std::string name = "augmented Jacobian adds gamma C^T C";
    auto b = BuildHexFesBundle(MPI_COMM_WORLD);
    BoundaryClassifier3D classifier(*b.pmesh, *b.fes);
    auto C_op = std::make_shared<MortarConstraintOperator>(classifier);
    const int n_u = C_op->Width();
    const int n_lam = C_op->Height();
    const double gamma = 2.25;

    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = n_u;
    offsets[2] = n_u + n_lam;

    mfem::IdentityOperator K(n_u);
    mfem::TransposeOperator Ct(C_op.get());
    mfem::BlockOperator unaugmented(offsets);
    unaugmented.SetBlock(0, 0, &K);
    unaugmented.SetBlock(0, 1, &Ct);
    unaugmented.SetBlock(1, 0, C_op.get());

    AugmentedLagrangianSaddleJacobian jac(
        unaugmented, C_op, gamma, offsets);

    mfem::BlockVector x(offsets);
    FillDeterministic(x.GetBlock(0), 0.125);
    FillDeterministic(x.GetBlock(1), -0.25);

    mfem::Vector actual(offsets.Last());
    mfem::Vector expected(offsets.Last());
    jac.Mult(x, actual);
    unaugmented.Mult(x, expected);

    mfem::Vector Cxu(n_lam);
    mfem::Vector CtCxu(n_u);
    C_op->Mult(x.GetBlock(0), Cxu);
    C_op->MultTranspose(Cxu, CtCxu);

    mfem::BlockVector expected_blocks;
    expected_blocks.Update(expected, offsets);
    expected_blocks.GetBlock(0).Add(gamma, CtCxu);

    AssertVectorNear(actual, expected, 1.0e-12, name);
}

void TestRhsSolverAddsCtLambdaBlock()
{
    const std::string name = "augmented RHS solver adds gamma C^T b_lambda";
    auto b = BuildHexFesBundle(MPI_COMM_WORLD);
    BoundaryClassifier3D classifier(*b.pmesh, *b.fes);
    auto C_op = std::make_shared<MortarConstraintOperator>(classifier);
    const int n_u = C_op->Width();
    const int n_lam = C_op->Height();
    const double gamma = 1.75;

    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = n_u;
    offsets[2] = n_u + n_lam;

    auto inner = std::make_shared<RecordingSolver>(offsets.Last());
    AugmentedLagrangianRhsSolver solver(
        inner, C_op, gamma, offsets);

    mfem::BlockVector rhs(offsets);
    FillDeterministic(rhs.GetBlock(0), 0.2);
    FillDeterministic(rhs.GetBlock(1), -0.1);

    mfem::Vector solution(offsets.Last());
    solver.Mult(rhs, solution);

    mfem::Vector expected(offsets.Last());
    expected = rhs;
    mfem::BlockVector expected_blocks;
    expected_blocks.Update(expected, offsets);
    mfem::Vector Ct_rhs_lam(n_u);
    C_op->MultTranspose(rhs.GetBlock(1), Ct_rhs_lam);
    expected_blocks.GetBlock(0).Add(gamma, Ct_rhs_lam);

    AssertVectorNear(inner->last_rhs, expected, 1.0e-12, name);
    AssertVectorNear(solution, expected, 1.0e-12, name + " solution");
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        TestAugmentedJacobianAddsCtC();
        TestRhsSolverAddsCtLambdaBlock();
        std::cout << "All augmented-Lagrangian saddle wrapper tests passed."
                  << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
