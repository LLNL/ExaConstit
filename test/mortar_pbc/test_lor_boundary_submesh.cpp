// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 6.0.A/B — SimulationState LOR boundary infrastructure.
//
// These tests deliberately stop at the shared mesh/FES layer. The
// classifier and projector are validated in later Phase 6 slices, after
// their code is introduced. The goal here is narrower:
//   * `lor_depth = 1` returns aliased direct-path boundary handles.
//   * `lor_depth = 2` builds a distinct, uniformly refined boundary
//     submesh without mutating the unrefined cached submesh.
//   * the surface FE spaces are vector H1(P1), vdim=3, byNODES.
//   * mesh-option validation accepts only the Phase 6-supported
//     higher-order LOR combinations.

#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {

void Require(bool cond, const std::string& test_name, const std::string& detail)
{
    if (!cond) {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

ExaOptions MakeMinimalOptions(int order, int lor_depth)
{
    ExaOptions opts;
    opts.mesh.mesh_type = MeshType::AUTO;
    opts.mesh.nxyz = {1, 1, 1};
    opts.mesh.mxyz = {1.0, 1.0, 1.0};
    opts.mesh.order = order;
    opts.mesh.periodicity = true;
    opts.mesh.lor_depth = lor_depth;

    TimeOptions::FixedTimeOptions fixed;
    fixed.dt = 1.0;
    fixed.t_final = 1.0;
    opts.time.time_type = TimeStepType::FIXED;
    opts.time.fixed_time = fixed;

    MaterialOptions mat;
    mat.material_name = "test_material";
    mat.region_id = 1;
    mat.state_vars.num_vars = 1;
    mat.state_vars.initial_values = {0.0};
    opts.materials = {mat};

    return opts;
}

void CheckSurfaceFes(const mfem::ParFiniteElementSpace& fes,
                     const std::string& test_name)
{
    Require(fes.GetVDim() == 3, test_name, "surface FES vdim must be 3");
    Require(fes.GetOrdering() == mfem::Ordering::byNODES,
            test_name, "surface FES must use byNODES ordering");
    Require(fes.GetOrder(0) == 1, test_name, "surface FES must be order 1");
}

void test_lor_depth_one_aliases_direct_boundary()
{
    const std::string name = "lor_depth=1 aliases direct boundary";
    auto opts = MakeMinimalOptions(/*order=*/1, /*lor_depth=*/1);
    SimulationState state(opts);

    auto bdr = state.GetBoundarySubMesh();
    auto lor = state.GetLorBoundarySubMesh();
    auto bdr_fes = state.GetBoundarySubMeshFes();
    auto lor_fes = state.GetLorBoundarySubMeshFes();

    Require(bdr.get() == lor.get(), name,
            "LOR boundary submesh should alias direct boundary at depth 1");
    Require(bdr_fes.get() == lor_fes.get(), name,
            "LOR boundary FES should alias direct boundary FES at depth 1");
    CheckSurfaceFes(*bdr_fes, name);
    std::cout << "  PASS  " << name << std::endl;
}

void test_lor_depth_two_refines_without_mutating_direct_boundary()
{
    const std::string name = "lor_depth=2 refines separate boundary";
    auto opts = MakeMinimalOptions(/*order=*/2, /*lor_depth=*/2);
    SimulationState state(opts);

    auto bdr = state.GetBoundarySubMesh();
    const int direct_ne = bdr->GetNE();
    auto bdr_fes = state.GetBoundarySubMeshFes();

    auto lor = state.GetLorBoundarySubMesh();
    auto lor_fes = state.GetLorBoundarySubMeshFes();

    Require(bdr.get() != lor.get(), name,
            "LOR boundary submesh must be distinct at depth 2");
    Require(bdr->GetNE() == direct_ne, name,
            "building the LOR submesh mutated the direct boundary submesh");
    Require(lor->GetNE() == 4 * direct_ne, name,
            "one uniform surface refinement should quadruple quad boundary elements");
    Require(bdr_fes.get() != lor_fes.get(), name,
            "LOR boundary FES must be distinct at depth 2");
    CheckSurfaceFes(*lor_fes, name);
    std::cout << "  PASS  " << name << std::endl;
}

void test_mesh_option_lor_validation()
{
    const std::string name = "MeshOptions LOR validation";

    MeshOptions ok_p1;
    ok_p1.periodicity = true;
    ok_p1.order = 1;
    ok_p1.lor_depth = 1;
    Require(ok_p1.validate(), name, "order=1, lor_depth=1 should validate");

    MeshOptions ok_p2;
    ok_p2.periodicity = true;
    ok_p2.order = 2;
    ok_p2.lor_depth = 2;
    Require(ok_p2.validate(), name, "order=2, lor_depth=2 should validate");

    MeshOptions stale_when_not_periodic;
    stale_when_not_periodic.periodicity = false;
    stale_when_not_periodic.order = 2;
    stale_when_not_periodic.lor_depth = 1;
    Require(stale_when_not_periodic.validate(), name,
            "periodicity=false should ignore mortar-only LOR validation");

    MeshOptions mismatch_low;
    mismatch_low.periodicity = true;
    mismatch_low.order = 1;
    mismatch_low.lor_depth = 2;
    Require(!mismatch_low.validate(), name,
            "order=1, lor_depth=2 should be rejected");

    MeshOptions mismatch_high;
    mismatch_high.periodicity = true;
    mismatch_high.order = 2;
    mismatch_high.lor_depth = 1;
    Require(!mismatch_high.validate(), name,
            "order=2, lor_depth=1 should be rejected");

    MeshOptions unsupported_depth;
    unsupported_depth.periodicity = true;
    unsupported_depth.order = 3;
    unsupported_depth.lor_depth = 3;
    Require(!unsupported_depth.validate(), name,
            "lor_depth=3 should be rejected in Phase 6 Day-1 scope");

    std::cout << "  PASS  " << name << std::endl;
}

} // namespace

int main(int argc, char* argv[])
{
    mfem::Mpi::Init(argc, argv);
    mfem::Device device("cpu");

    test_mesh_option_lor_validation();
    test_lor_depth_one_aliases_direct_boundary();
    test_lor_depth_two_refines_without_mutating_direct_boundary();

    return 0;
}
