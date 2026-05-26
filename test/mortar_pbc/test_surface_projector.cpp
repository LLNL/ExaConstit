// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 6.0.D — SurfaceProjector unit tests.
//
// The projector is the construction-time bridge between parent-volume
// true DOFs and the boundary-submesh true DOFs used by the classifier.
// These tests keep the scope deliberately narrow:
//   * p=1 direct-path projection is a boundary trace permutation.
//   * p=2 with one LOR boundary refinement reproduces linear fields at
//     the refined surface vertices.
//   * local map arrays are populated and agree with parent ownership.
//   * MultTranspose performs the matching scatter-add back to parent
//     true DOFs.

#include "surface_projector.hpp"

#include "mfem.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {

constexpr int kVDim = 3;
constexpr double kSnapTol = 1.0e-10;
constexpr double kCheckTol = 1.0e-12;

struct SurfaceBundle
{
    std::shared_ptr<mfem::ParMesh> parent_mesh;
    std::shared_ptr<mfem::H1_FECollection> parent_fec;
    std::shared_ptr<mfem::ParFiniteElementSpace> parent_fes;
    std::shared_ptr<mfem::ParSubMesh> submesh;
    std::shared_ptr<mfem::H1_FECollection> submesh_fec;
    std::shared_ptr<mfem::ParFiniteElementSpace> submesh_fes;
};

void Require(bool cond, const std::string& test_name,
             const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

std::array<double, kVDim> LinearField(const mfem::Vector& x)
{
    return {1.0 + x[0] + 2.0 * x[1] - 0.5 * x[2],
            -0.25 + 0.75 * x[0] - x[1] + 1.5 * x[2],
            0.5 - 2.0 * x[0] + 0.25 * x[1] + x[2]};
}

SurfaceBundle BuildHexSurfaceBundle(MPI_Comm comm, int order,
                                    bool refine_boundary)
{
    SurfaceBundle b;

    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        /*nx=*/1, /*ny=*/1, /*nz=*/1, mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.parent_mesh = std::make_shared<mfem::ParMesh>(comm, serial);
    if (order > 1)
    {
        b.parent_mesh->SetCurvature(order, /*discontinuous=*/false,
                                    /*space_dim=*/kVDim,
                                    mfem::Ordering::byNODES);
    }

    b.parent_fec = std::make_shared<mfem::H1_FECollection>(
        order, b.parent_mesh->Dimension());
    b.parent_fes = std::make_shared<mfem::ParFiniteElementSpace>(
        b.parent_mesh.get(), b.parent_fec.get(), kVDim,
        mfem::Ordering::byNODES);

    mfem::Array<int> bdr_attrs(b.parent_mesh->bdr_attributes);
    b.submesh = std::make_shared<mfem::ParSubMesh>(
        mfem::ParSubMesh::CreateFromBoundary(*b.parent_mesh, bdr_attrs));
    if (refine_boundary) { b.submesh->UniformRefinement(); }

    b.submesh_fec = std::make_shared<mfem::H1_FECollection>(
        /*order=*/1, b.submesh->SpaceDimension());
    b.submesh_fes = std::make_shared<mfem::ParFiniteElementSpace>(
        b.submesh.get(), b.submesh_fec.get(), kVDim,
        mfem::Ordering::byNODES);

    return b;
}

void FillParentBoundaryLinearTrace(const mfem::ParFiniteElementSpace& fes,
                                   mfem::Vector& x_true)
{
    x_true.SetSize(fes.GetTrueVSize());
    x_true = 0.0;

    const HYPRE_BigInt first = fes.GetTrueDofOffsets()[0];
    const HYPRE_BigInt last = fes.GetTrueDofOffsets()[1];
    mfem::ParMesh* mesh = fes.GetParMesh();

    mfem::Array<int> scalar_dofs;
    mfem::Vector x_phys;
    for (int be = 0; be < mesh->GetNBE(); ++be)
    {
        fes.GetBdrElementDofs(be, scalar_dofs);
        const mfem::FiniteElement* fe = fes.GetBE(be);
        const mfem::IntegrationRule& nodes = fe->GetNodes();
        mfem::ElementTransformation* tr =
            mesh->GetBdrElementTransformation(be);
        Require(nodes.GetNPoints() == scalar_dofs.Size(),
                "parent trace fill",
                "boundary node count does not match scalar DOF count");

        for (int i = 0; i < scalar_dofs.Size(); ++i)
        {
            tr->Transform(nodes.IntPoint(i), x_phys);
            const auto values = LinearField(x_phys);
            for (int c = 0; c < kVDim; ++c)
            {
                const int vdof = fes.DofToVDof(scalar_dofs[i], c);
                const int gtdof = fes.GetGlobalTDofNumber(vdof);
                if (static_cast<HYPRE_BigInt>(gtdof) >= first
                    && static_cast<HYPRE_BigInt>(gtdof) < last)
                {
                    x_true[static_cast<int>(
                        static_cast<HYPRE_BigInt>(gtdof) - first)] =
                        values[c];
                }
            }
        }
    }
}

void CheckProjectedLinearTrace(const mfem::ParFiniteElementSpace& submesh_fes,
                               const mfem::ParMesh& submesh,
                               const mfem::Vector& y_true,
                               const std::string& test_name)
{
    const HYPRE_BigInt first = submesh_fes.GetTrueDofOffsets()[0];
    mfem::Array<int> scalar_dofs;
    mfem::Vector vertex_coord(kVDim);

    for (int v = 0; v < submesh.GetNV(); ++v)
    {
        const double* raw = submesh.GetVertex(v);
        for (int d = 0; d < kVDim; ++d) { vertex_coord[d] = raw[d]; }
        const auto expected = LinearField(vertex_coord);

        submesh_fes.GetVertexDofs(v, scalar_dofs);
        Require(scalar_dofs.Size() > 0, test_name,
                "submesh vertex has no scalar H1 DOF");
        const int scalar_dof = scalar_dofs[0];
        for (int c = 0; c < kVDim; ++c)
        {
            const int vdof = submesh_fes.DofToVDof(scalar_dof, c);
            const int gtdof = submesh_fes.GetGlobalTDofNumber(vdof);
            const int ltdof = static_cast<int>(
                static_cast<HYPRE_BigInt>(gtdof) - first);
            Require(ltdof >= 0 && ltdof < y_true.Size(), test_name,
                    "submesh vertex true DOF is not rank-local");
            const double err = std::abs(y_true[ltdof] - expected[c]);
            Require(err <= kCheckTol, test_name,
                    "linear trace mismatch at vertex "
                    + std::to_string(v) + ", component "
                    + std::to_string(c));
        }
    }
}

void CheckLocalMapConsistency(
    const mortar_pbc::SurfaceProjector& projector,
    const mfem::ParFiniteElementSpace& parent_fes,
    const mfem::ParFiniteElementSpace& submesh_fes,
    const std::string& test_name)
{
    const auto& map = projector.Map();
    Require(map.local_submesh_to_parent_gtdof.Size() == projector.Height(),
            test_name, "local parent-gtdof map has wrong size");
    Require(map.local_submesh_to_parent_local_or_minus1.Size()
                == projector.Height(),
            test_name, "local parent-local map has wrong size");

    const HYPRE_BigInt sub_first = submesh_fes.GetTrueDofOffsets()[0];
    const HYPRE_BigInt parent_first = parent_fes.GetTrueDofOffsets()[0];
    const HYPRE_BigInt parent_last = parent_fes.GetTrueDofOffsets()[1];
    for (int i = 0; i < projector.Height(); ++i)
    {
        const int sub_gtdof = static_cast<int>(sub_first) + i;
        const int parent_gtdof = projector.ParentGtdof(sub_gtdof);
        Require(parent_gtdof == map.local_submesh_to_parent_gtdof[i],
                test_name, "ParentGtdof disagrees with local map");
        Require(parent_gtdof >= 0
                    && parent_gtdof < parent_fes.GlobalTrueVSize(),
                test_name, "mapped parent true DOF is out of range");

        const bool parent_owned =
            static_cast<HYPRE_BigInt>(parent_gtdof) >= parent_first
            && static_cast<HYPRE_BigInt>(parent_gtdof) < parent_last;
        const int expected_local =
            parent_owned
                ? static_cast<int>(static_cast<HYPRE_BigInt>(parent_gtdof)
                                   - parent_first)
                : -1;
        Require(projector.ParentLocalOrMinus1(sub_gtdof) == expected_local,
                test_name, "ParentLocalOrMinus1 returned wrong local index");
        Require(map.local_submesh_to_parent_local_or_minus1[i]
                    == expected_local,
                test_name, "local parent-local map returned wrong index");
    }
}

void CheckConstantProjection(const mortar_pbc::SurfaceProjector& projector,
                             const std::string& test_name)
{
    constexpr double value = 2.75;
    mfem::Vector x_parent(projector.Width());
    mfem::Vector y_submesh(projector.Height());
    x_parent = value;
    y_submesh = 0.0;

    projector.Mult(x_parent, y_submesh);
    for (int i = 0; i < y_submesh.Size(); ++i)
    {
        Require(std::abs(y_submesh[i] - value) <= kCheckTol, test_name,
                "constant projection mismatch at local submesh TDOF "
                + std::to_string(i));
    }
}

void CheckTransposeScatterAdd(
    const mortar_pbc::SurfaceProjector& projector,
    const mfem::ParFiniteElementSpace& parent_fes,
    const std::string& test_name)
{
    mfem::Vector x_submesh(projector.Height());
    mfem::Vector y_parent(projector.Width());
    x_submesh = 1.0;
    y_parent = 0.0;

    projector.MultTranspose(x_submesh, y_parent);

    mfem::Vector expected(projector.Width());
    expected = 0.0;
    const HYPRE_BigInt parent_first = parent_fes.GetTrueDofOffsets()[0];
    const HYPRE_BigInt parent_last = parent_fes.GetTrueDofOffsets()[1];

    // With unit submesh input, R^T counts how many submesh true DOFs map
    // to each parent true DOF. Building the expectation from the public
    // global map verifies the scatter-add semantics independently of the
    // dense reduction used inside MultTranspose.
    for (const auto& kv : projector.Map().submesh_to_parent_gtdof)
    {
        const int parent_gtdof = kv.second;
        if (static_cast<HYPRE_BigInt>(parent_gtdof) >= parent_first
            && static_cast<HYPRE_BigInt>(parent_gtdof) < parent_last)
        {
            expected[static_cast<int>(
                static_cast<HYPRE_BigInt>(parent_gtdof) - parent_first)]
                += 1.0;
        }
    }

    for (int i = 0; i < y_parent.Size(); ++i)
    {
        Require(std::abs(y_parent[i] - expected[i]) <= kCheckTol,
                test_name,
                "transpose scatter mismatch at local parent TDOF "
                + std::to_string(i));
    }
}

void RunProjectionCase(const std::string& test_name, int parent_order,
                       bool refine_boundary)
{
    std::cout << test_name << std::endl;
    auto b = BuildHexSurfaceBundle(MPI_COMM_WORLD, parent_order,
                                   refine_boundary);
    mortar_pbc::SurfaceProjector projector(b.parent_fes, b.submesh_fes,
                                           b.submesh, kSnapTol);

    CheckConstantProjection(projector, test_name);

    mfem::Vector x_parent(projector.Width());
    FillParentBoundaryLinearTrace(*b.parent_fes, x_parent);

    mfem::Vector y_submesh(projector.Height());
    projector.Mult(x_parent, y_submesh);

    CheckProjectedLinearTrace(*b.submesh_fes, *b.submesh, y_submesh,
                              test_name);
    CheckLocalMapConsistency(projector, *b.parent_fes, *b.submesh_fes,
                             test_name);
    CheckTransposeScatterAdd(projector, *b.parent_fes, test_name);

    std::cout << "  PASS  " << test_name << std::endl;
}

}  // namespace

int main(int argc, char* argv[])
{
    mfem::Mpi::Init(argc, argv);
    mfem::Device device("cpu");

    RunProjectionCase("p=1 direct boundary projection",
                      /*parent_order=*/1, /*refine_boundary=*/false);
    RunProjectionCase("p=2 once-refined LOR boundary projection",
                      /*parent_order=*/2, /*refine_boundary=*/true);

    return 0;
}
