// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.3.B — tests for `MortarPbcManager`'s corner-essential
// TDOF builder.
//
// Constructing a full `MortarPbcManager` requires a `SimulationState`
// (parsed options, materials, etc.), which is heavier than what a
// unit test should carry. Instead we exercise the algorithm directly
// via `mortar_pbc::ComputeCornerEssTDofs(classifier, fes)`, which is
// the same free function `MortarPbcManager::BuildCornerEssTDofs`
// calls internally. Both the manager method and this test go through
// the same code path, so the test catches drift and the assertions
// here mirror the runtime sanity check the manager does after
// calling it (`MPI_Allreduce(local count) == 24`).
//
// Phase 6 adds projector-aware overloads for the same corner-pinning
// helpers. Those overloads accept a classifier built on a boundary/LOR
// submesh and return parent-volume TDOFs by translating through
// `SurfaceProjector`. The direct-path tests below compare those
// projected results against the legacy parent-classifier results so
// manager wiring can safely switch to the LOR path without changing
// the mechanics essential-BC index space.
//
// Coverage:
//   1. Algorithm runs cleanly on a 2x2x2 hex mesh; the rank-summed
//      TDOF count equals 24 (8 corners x 3 components).
//   2. Same on a larger 4x4x4 hex mesh — count is invariant under
//      mesh refinement (a property of the corners themselves, not
//      of the bulk discretization).
//   3. All rank-local TDOFs returned fall in the valid local range
//      `[0, fes.GetTrueVSize())`.
//   4. Within a rank, no duplicate TDOFs appear (each corner
//      component is owned by exactly one rank, and at most once).
//
// Each test function exits via std::exit(1) on failure (with a
// diagnostic to stderr) or returns normally on success. Registered
// at NUM_MPI_TASKS = 1 by convention; running by hand with np>1
// exercises the rank-split path.

#include "mortar_pbc_manager.hpp"

#include "boundary_classifier_3d.hpp"
#include "surface_projector.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ComputeCornerEssTDofs;
using mortar_pbc::ComputeCornerEssTDofsFromSpec;
using mortar_pbc::MortarPbcManager;
using mortar_pbc::SurfaceProjector;

namespace {

constexpr double kSnapTol = 1.0e-10;

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
    b.fec   = std::make_unique<mfem::H1_FECollection>(/*order=*/1, /*dim=*/3);
    b.fes   = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3, mfem::Ordering::byNODES);
    return b;
}

struct SharedSurfaceBundle
{
    std::shared_ptr<mfem::ParMesh> parent_mesh;
    std::shared_ptr<mfem::H1_FECollection> parent_fec;
    std::shared_ptr<mfem::ParFiniteElementSpace> parent_fes;
    std::shared_ptr<mfem::ParSubMesh> submesh;
    std::shared_ptr<mfem::H1_FECollection> submesh_fec;
    std::shared_ptr<mfem::ParFiniteElementSpace> submesh_fes;
};

SharedSurfaceBundle BuildSharedSurfaceBundle(MPI_Comm comm, int n_per_side)
{
    SharedSurfaceBundle b;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n_per_side, n_per_side, n_per_side,
        mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.parent_mesh = std::make_shared<mfem::ParMesh>(comm, serial);
    b.parent_fec = std::make_shared<mfem::H1_FECollection>(
        /*order=*/1, /*dim=*/3);
    b.parent_fes = std::make_shared<mfem::ParFiniteElementSpace>(
        b.parent_mesh.get(), b.parent_fec.get(), /*vdim=*/3,
        mfem::Ordering::byNODES);

    mfem::Array<int> bdr_attrs(b.parent_mesh->bdr_attributes);
    b.submesh = std::make_shared<mfem::ParSubMesh>(
        mfem::ParSubMesh::CreateFromBoundary(*b.parent_mesh, bdr_attrs));
    b.submesh_fec = std::make_shared<mfem::H1_FECollection>(
        /*order=*/1, b.submesh->SpaceDimension());
    b.submesh_fes = std::make_shared<mfem::ParFiniteElementSpace>(
        b.submesh.get(), b.submesh_fec.get(), /*vdim=*/3,
        mfem::Ordering::byNODES);
    return b;
}

std::set<int> ToSet(const mfem::Array<int>& a)
{
    return std::set<int>(a.begin(), a.end());
}

void AssertSameTdofSet(const mfem::Array<int>& got,
                       const mfem::Array<int>& expected,
                       const std::string& tag)
{
    const std::set<int> got_set = ToSet(got);
    const std::set<int> expected_set = ToSet(expected);
    AssertOrDie(got_set == expected_set,
                tag + ": projected/local TDOF set equality",
                "projected set has size "
                + std::to_string(got_set.size())
                + ", expected size "
                + std::to_string(expected_set.size()));
}

// Helper: run the corner-TDOF algorithm against a freshly-built
// classifier and FES, then run the rank-summed-count + range-+-
// uniqueness checks. Used by both mesh-size tests below.
void RunCornerTdofChecks(int n_per_side, const std::string& tag)
{
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, n_per_side);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    AssertOrDie(cl.Corners().size() == 8,
                tag + ": classifier corner count",
                "got " + std::to_string(cl.Corners().size())
                + ", expected 8");

    mfem::Array<int> corner_tdofs = ComputeCornerEssTDofs(cl, *b.fes);

    // (1) Rank-summed count.
    int local_count = corner_tdofs.Size();
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    AssertOrDie(global_count == 24,
                tag + ": rank-summed corner TDOF count",
                "got " + std::to_string(global_count) + ", expected 24");

    // (2) Range check — every entry is a valid rank-local TDOF.
    const int n_local_tdofs = b.fes->GetTrueVSize();
    for (int i = 0; i < corner_tdofs.Size(); ++i)
    {
        const int t = corner_tdofs[i];
        AssertOrDie(t >= 0 && t < n_local_tdofs,
                    tag + ": local TDOF in range",
                    "got " + std::to_string(t)
                    + ", expected within [0, "
                    + std::to_string(n_local_tdofs) + ")");
    }

    // (3) No duplicates within a rank.
    std::set<int> uniq(corner_tdofs.begin(), corner_tdofs.end());
    AssertOrDie(static_cast<int>(uniq.size()) == corner_tdofs.Size(),
                tag + ": rank-local TDOFs unique",
                "got " + std::to_string(corner_tdofs.Size())
                + " entries with " + std::to_string(uniq.size())
                + " unique values");

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "  PASS  " << tag << ": global=" << global_count
                  << " (=24), local=" << local_count
                  << ", n_local_tdofs=" << n_local_tdofs << std::endl;
    }
}

// ===========================================================================
// Test 1: 2x2x2 hex mesh — smallest case with all 8 corners present.
// ===========================================================================
void test_corner_tdofs_2x2x2()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "Test 1: corner TDOFs on 2x2x2 hex mesh" << std::endl;
    }
    RunCornerTdofChecks(2, "2x2x2");
}

// ===========================================================================
// Test 2: 4x4x4 hex mesh — verifies the count is invariant under
// refinement (the 8 corners are topologically fixed; the bulk DOFs
// grow but the corner-pinning set does not).
// ===========================================================================
void test_corner_tdofs_4x4x4()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "Test 2: corner TDOFs on 4x4x4 hex mesh" << std::endl;
    }
    RunCornerTdofChecks(4, "4x4x4");
}

// ===========================================================================
// Test 3: Phase 6 direct path — projected full-corner pinning matches
// the legacy parent-classifier result exactly on a linear parent FES.
// ===========================================================================
void test_projected_corner_tdofs_direct_path()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "Test 3: projected corner TDOFs match direct path"
                  << std::endl;
    }

    auto b = BuildSharedSurfaceBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D legacy_cl(*b.parent_mesh, *b.parent_fes);
    auto submesh_cl = std::make_shared<BoundaryClassifier3D>(
        b.submesh, b.submesh_fes, kSnapTol);
    SurfaceProjector projector(
        b.parent_fes, b.submesh_fes, b.submesh, kSnapTol);

    const auto legacy = ComputeCornerEssTDofs(legacy_cl, *b.parent_fes);
    const auto projected = ComputeCornerEssTDofs(
        *submesh_cl, projector, *b.parent_fes);

    AssertSameTdofSet(projected, legacy, "projected full corners");

    int local_count = projected.Size();
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    AssertOrDie(global_count == 24,
                "projected full corners: rank-summed count",
                "got " + std::to_string(global_count) + ", expected 24");

    if (rank == 0)
    {
        std::cout << "  PASS  projected full-corner set matches legacy"
                  << std::endl;
    }
}

// ===========================================================================
// Test 4: Phase 6 direct path — projected filtered pinning preserves
// the Phase 5.9 spec semantics while returning parent-FES local TDOFs.
// ===========================================================================
void test_projected_corner_tdofs_from_spec_direct_path()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "Test 4: projected spec-filtered corner TDOFs"
                  << std::endl;
    }

    auto b = BuildSharedSurfaceBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D legacy_cl(*b.parent_mesh, *b.parent_fes);
    auto submesh_cl = std::make_shared<BoundaryClassifier3D>(
        b.submesh, b.submesh_fes, kSnapTol);
    SurfaceProjector projector(
        b.parent_fes, b.submesh_fes, b.submesh, kSnapTol);

    const auto full_spec =
        MortarPbcManager::SynthesizeDefaultPbcSpec(*submesh_cl);
    const auto projected_full = ComputeCornerEssTDofsFromSpec(
        *submesh_cl, projector, *b.parent_fes,
        full_spec.first, {{true, true, true}});
    const auto legacy_full = ComputeCornerEssTDofs(legacy_cl, *b.parent_fes);
    AssertSameTdofSet(projected_full, legacy_full,
                      "projected full-spec corners");

    const std::vector<int> x_pair_ids = {
        submesh_cl->MeshAttributeForLabel("left"),
        submesh_cl->MeshAttributeForLabel("right")};
    const std::array<bool, 3> x_only = {{true, false, false}};
    const auto projected_x = ComputeCornerEssTDofsFromSpec(
        *submesh_cl, projector, *b.parent_fes, x_pair_ids, x_only);

    int local_count = projected_x.Size();
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    AssertOrDie(global_count == 10,
                "projected X-only corners: rank-summed count",
                "got " + std::to_string(global_count) + ", expected 10");

    const int n_local_tdofs = b.parent_fes->GetTrueVSize();
    for (int i = 0; i < projected_x.Size(); ++i)
    {
        AssertOrDie(projected_x[i] >= 0 && projected_x[i] < n_local_tdofs,
                    "projected X-only corners: parent local TDOF range",
                    "got " + std::to_string(projected_x[i])
                    + ", valid range is [0, "
                    + std::to_string(n_local_tdofs) + ")");
    }

    AssertOrDie(static_cast<int>(ToSet(projected_x).size())
                    == projected_x.Size(),
                "projected X-only corners: uniqueness",
                "duplicate local parent TDOFs returned");

    if (rank == 0)
    {
        std::cout << "  PASS  projected filtered pinning returns "
                  << global_count << " parent TDOFs" << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running MortarPbcManager corner-TDOF tests" << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    }

    test_corner_tdofs_2x2x2();
    test_corner_tdofs_4x4x4();
    test_projected_corner_tdofs_direct_path();
    test_projected_corner_tdofs_from_spec_direct_path();

    if (rank == 0)
    {
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "All MortarPbcManager corner-TDOF tests passed."
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
