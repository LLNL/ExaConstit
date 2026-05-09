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

#include "mfem.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ComputeCornerEssTDofs;

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
    b.fec   = std::make_unique<mfem::H1_FECollection>(/*order=*/1, /*dim=*/3);
    b.fes   = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3, mfem::Ordering::byNODES);
    return b;
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

    if (rank == 0)
    {
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "All MortarPbcManager corner-TDOF tests passed."
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
