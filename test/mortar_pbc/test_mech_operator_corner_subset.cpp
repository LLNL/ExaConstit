// Phase 5.4.B smoke test
//
// Verifies that `mfem::ParNonlinearForm::SetEssentialTrueDofs` correctly
// handles essential TDOFs supplied directly as a list (the path
// `NonlinearMechOperator::UpdateEssTDofsCornerSubset` uses for mortar
// PBC corner pinning).
//
// Scope per Phase 5 v4 plan §5.4.B: confirm that
// `ParNonlinearForm::SetEssentialTrueDofs` accepts and remembers a
// 24-entry TDOF list, that subsequent `Mult` zero-eliminates those
// rows, and that `GetGradient` builds a Jacobian whose row/col
// elimination at those positions matches MFEM's standard Dirichlet
// elimination convention (row = identity row).
//
// `NonlinearMechOperator` itself is intentionally NOT exercised here:
// constructing it requires a full `SimulationState` (options +
// materials + sim state plumbing). End-to-end coverage of the
// wrapper lands with the Phase 5.5 / 5.6 patch tests; the wrapper
// is a 2-line passthrough so the meaningful smoke test is on the
// underlying MFEM behavior.

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace {

void AssertOrDie(bool cond, const std::string &msg)
{
   if (!cond) {
      std::cerr << "FAILED: " << msg << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
   }
}

}  // anonymous namespace

int main(int argc, char *argv[])
{
   mfem::Mpi::Init(argc, argv);
   const int rank    = mfem::Mpi::WorldRank();
   const int n_ranks = mfem::Mpi::WorldSize();

   // Small 4x4x4 hex mesh — a few hundred DOFs, plenty for a
   // 24-element ess subset to be a meaningful fraction.
   constexpr int n_per_side = 4;
   mfem::Mesh smesh = mfem::Mesh::MakeCartesian3D(
      n_per_side, n_per_side, n_per_side, mfem::Element::HEXAHEDRON,
      1.0, 1.0, 1.0);
   mfem::ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();

   constexpr int vdim  = 3;
   constexpr int order = 1;
   mfem::H1_FECollection fec(order, pmesh.Dimension());
   mfem::ParFiniteElementSpace fes(&pmesh, &fec, vdim, mfem::Ordering::byNODES);

   if (rank == 0) {
      std::cout << "test_mech_operator_corner_subset: nranks=" << n_ranks
                << "  global TrueVSize=" << fes.GlobalTrueVSize()
                << std::endl;
   }

   // Pick up to 24 rank-local TDOFs (the first 24 if available;
   // otherwise the rank contributes fewer and the rank-summed total
   // is still ≤ 24 — exercises the small/empty-partition boundary
   // case under MPI).
   const int local_true_size = fes.GetTrueVSize();
   const int local_n_target  = std::min(24, local_true_size);
   mfem::Array<int> ess_tdofs(local_n_target);
   for (int i = 0; i < local_n_target; ++i) { ess_tdofs[i] = i; }

   // Build a ParNonlinearForm with a NeoHookean integrator. The
   // integrator is just for making the form non-trivial — what we're
   // testing is the essential-TDOF mechanics, not the constitutive
   // model. mu=0.5, K=1.0 are arbitrary positive values.
   mfem::NeoHookeanModel hyperelastic_model(/*mu=*/0.5, /*K=*/1.0);
   mfem::ParNonlinearForm nlf(&fes);
   nlf.AddDomainIntegrator(
      new mfem::HyperelasticNLFIntegrator(&hyperelastic_model));

   // The path under test — install the ess TDOF list directly.
   nlf.SetEssentialTrueDofs(ess_tdofs);

   // Round-trip: GetEssentialTrueDofs should return exactly what we
   // set, in the same order.
   {
      const mfem::Array<int> &got = nlf.GetEssentialTrueDofs();
      AssertOrDie(got.Size() == ess_tdofs.Size(),
                  "GetEssentialTrueDofs() size round-trip");
      for (int i = 0; i < ess_tdofs.Size(); ++i) {
         AssertOrDie(got[i] == ess_tdofs[i],
                     "GetEssentialTrueDofs() entry "
                     + std::to_string(i) + " round-trip");
      }
   }

   // Build a non-trivial input: project the linear field v(x) = x
   // onto the FES TDOFs. Gives a non-zero NeoHookean residual.
   mfem::Vector v(fes.GetTrueVSize());
   v.UseDevice(true);
   {
      mfem::ParGridFunction gf(&fes);
      gf = 0.0;
      const auto *nodes = pmesh.GetNodes();
      const bool have_nodes = (nodes != nullptr);
      for (int v_i = 0; v_i < pmesh.GetNV(); ++v_i) {
         double coords[3] = {0.0, 0.0, 0.0};
         if (have_nodes) {
            // Higher-order or moved meshes route through GetNodes.
            mfem::Vector vc;
            nodes->GetVectorValue(v_i, mfem::IntegrationPoint(), vc);
            for (int c = 0; c < vdim; ++c) { coords[c] = vc(c); }
         }
         else {
            const double *raw = pmesh.GetVertex(v_i);
            for (int c = 0; c < vdim; ++c) { coords[c] = raw[c]; }
         }
         for (int c = 0; c < vdim; ++c) {
            const int dof = fes.DofToVDof(v_i, c);
            gf[dof] = coords[c];
         }
      }
      gf.GetTrueDofs(v);
   }

   // Mult: residual at essential TDOFs should be zero.
   mfem::Vector r(fes.GetTrueVSize());
   r.UseDevice(true);
   nlf.Mult(v, r);
   {
      const double *r_data = r.HostRead();
      for (int i = 0; i < ess_tdofs.Size(); ++i) {
         const int row = ess_tdofs[i];
         AssertOrDie(std::abs(r_data[row]) < 1e-14,
                     "Mult(v, r) zero-eliminates essential row "
                     + std::to_string(row)
                     + " (got " + std::to_string(r_data[row]) + ")");
      }
   }

   // GetGradient: rows i in ess_tdofs become identity rows. So
   // K * e_i has a 1 at row i and zeros elsewhere (assuming the
   // column elimination has also occurred — MFEM does both for
   // ParNonlinearForm::GetGradient). Check the first, middle, last
   // ess entries.
   if (ess_tdofs.Size() > 0) {
      mfem::Operator &K = nlf.GetGradient(v);

      const int trueV = fes.GetTrueVSize();
      mfem::Vector e_i(trueV);
      e_i.UseDevice(true);
      mfem::Vector r2(trueV);
      r2.UseDevice(true);

      const int probes[3] = {0,
                             ess_tdofs.Size() / 2,
                             ess_tdofs.Size() - 1};
      for (int p = 0; p < 3; ++p) {
         const int idx = probes[p];
         if (idx < 0 || idx >= ess_tdofs.Size()) { continue; }
         const int row = ess_tdofs[idx];

         e_i = 0.0;
         e_i.HostWrite()[row] = 1.0;
         K.Mult(e_i, r2);

         const double *r2_d = r2.HostRead();
         AssertOrDie(std::abs(r2_d[row] - 1.0) < 1e-12,
                     "Gradient[" + std::to_string(row) + ", "
                     + std::to_string(row) + "] = 1 on identity row "
                     "(got " + std::to_string(r2_d[row]) + ")");

         // Off-diagonal entries in the same row should also be zero
         // — but Mult on K touches rows of K, not specific entries,
         // so we can't directly probe K[row, j]. Instead, probe by
         // multiplying e_j (j != row, j NOT in ess set) and asking
         // whether r3[row] is zero — which checks K[row, j] = 0
         // (column elimination at the ess row).
      }

      // Column elimination check: pick a non-essential column j,
      // multiply K * e_j, verify rows in ess_tdofs are zero.
      {
         int j_non_ess = -1;
         // Find a TDOF not in ess_tdofs. Simple O(n*ess) scan.
         for (int j = 0; j < trueV; ++j) {
            bool in_ess = false;
            for (int k = 0; k < ess_tdofs.Size(); ++k) {
               if (ess_tdofs[k] == j) { in_ess = true; break; }
            }
            if (!in_ess) { j_non_ess = j; break; }
         }
         if (j_non_ess >= 0) {
            e_i = 0.0;
            e_i.HostWrite()[j_non_ess] = 1.0;
            K.Mult(e_i, r2);
            const double *r2_d = r2.HostRead();
            for (int i = 0; i < ess_tdofs.Size(); ++i) {
               const int row = ess_tdofs[i];
               AssertOrDie(std::abs(r2_d[row]) < 1e-12,
                           "Gradient column-eliminates ess row "
                           + std::to_string(row)
                           + " when probed by non-ess col "
                           + std::to_string(j_non_ess)
                           + " (got " + std::to_string(r2_d[row]) + ")");
            }
         }
      }
   }

   if (rank == 0) {
      std::cout << "PASS  test_mech_operator_corner_subset"
                << std::endl;
   }

   return 0;
}