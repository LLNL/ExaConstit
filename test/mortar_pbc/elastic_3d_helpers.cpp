// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of elastic_3d_helpers.{hpp,cpp},
// ported from `mortar_pbc/elastic_3d.py`. See header for design doc.

#include "elastic_3d_helpers.hpp"

#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mortar_pbc {

//==============================================================================
// AssembleLinearElasticKHypre
//==============================================================================

mfem::HypreParMatrix* AssembleLinearElasticKHypre(
    mfem::ParMesh& pmesh,
    mfem::ParFiniteElementSpace& fes,
    double E,
    double nu)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::elastic::assemble_K_hypre");

    MFEM_VERIFY(fes.GetVDim() == pmesh.Dimension(),
                "AssembleLinearElasticKHypre: vdim (" << fes.GetVDim()
                << ") must match mesh dim (" << pmesh.Dimension() << ")");
    MFEM_VERIFY(nu < 0.5 && nu > -1.0,
                "AssembleLinearElasticKHypre: Poisson's ratio nu="
                << nu << " out of physical range (-1, 0.5)");
    MFEM_VERIFY(E > 0.0,
                "AssembleLinearElasticKHypre: Young's modulus E="
                << E << " must be positive");

    const double mu  = 0.5 * E / (1.0 + nu);
    const double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    mfem::ConstantCoefficient lam_coef(lam);
    mfem::ConstantCoefficient mu_coef(mu);

    mfem::ParBilinearForm a(&fes);
    a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lam_coef, mu_coef));
    a.Assemble();
    a.Finalize();

    // ParallelAssemble returns a freshly-allocated HypreParMatrix that
    // copies the data into HYPRE arrays, so returning it after `a`
    // goes out of scope is safe in current MFEM (>= 4.0). See
    // mfem/mfem#793 for the underlying lifetime rationale.
    return a.ParallelAssemble();
}

//==============================================================================
// ApplyLinearPart — project u_lin = (F - I) X onto the FE space
//==============================================================================

mfem::Vector ApplyLinearPart(mfem::ParFiniteElementSpace& fes,
                             const mfem::DenseMatrix& F_macro)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::elastic::apply_linear_part");

    const int vdim = fes.GetVDim();
    MFEM_VERIFY(F_macro.NumRows() == vdim && F_macro.NumCols() == vdim,
                "ApplyLinearPart: F_macro must be (" << vdim << ", " << vdim
                << "); got (" << F_macro.NumRows() << ", "
                << F_macro.NumCols() << ")");

    // F - I: copy and subtract the identity in place.
    mfem::DenseMatrix F_minus_I(F_macro);
    for (int i = 0; i < vdim; ++i) { F_minus_I(i, i) -= 1.0; }

    // VectorFunctionCoefficient takes a (Vector x_in, Vector& y_out)
    // callable; we capture F_minus_I by value for thread-safety
    // (the lambda is invoked at every quadrature/nodal point).
    mfem::VectorFunctionCoefficient coef(
        vdim,
        [F_minus_I, vdim](const mfem::Vector& x, mfem::Vector& y) -> void
        {
            for (int i = 0; i < vdim; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < vdim; ++j)
                {
                    sum += F_minus_I(i, j) * x(j);
                }
                y(i) = sum;
            }
        });

    mfem::ParGridFunction gf(&fes);
    gf.ProjectCoefficient(coef);

    mfem::Vector u_lin_local(fes.GetTrueVSize());
    gf.GetTrueDofs(u_lin_local);
    return u_lin_local;
}

//==============================================================================
// ApplyDirichletToDistributedK — eliminate corner rows/cols, set f
//==============================================================================

void ApplyDirichletToDistributedK(mfem::HypreParMatrix& K_hyp,
                                  mfem::Vector& f_par,
                                  const std::vector<int>& ess_global_tdofs,
                                  mfem::ParFiniteElementSpace& fes,
                                  const std::vector<double>& f_at_essential)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::elastic::apply_dirichlet");

    const bool have_values = !f_at_essential.empty();
    if (have_values)
    {
        MFEM_VERIFY(f_at_essential.size() == ess_global_tdofs.size(),
                    "ApplyDirichletToDistributedK: f_at_essential size ("
                    << f_at_essential.size() << ") does not match "
                    "ess_global_tdofs size (" << ess_global_tdofs.size()
                    << ")");
    }

    const int my_first_tdof = fes.GetMyTDofOffset();
    const int my_n_tdof = fes.GetTrueVSize();

    // Filter to TDOFs owned by this rank and translate to local indices.
    std::vector<int> local_indices;
    std::vector<double> local_vals;
    local_indices.reserve(ess_global_tdofs.size());
    local_vals.reserve(ess_global_tdofs.size());
    const std::size_t n = ess_global_tdofs.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const int gd = ess_global_tdofs[i];
        if (gd >= my_first_tdof && gd < my_first_tdof + my_n_tdof)
        {
            local_indices.push_back(gd - my_first_tdof);
            local_vals.push_back(have_values ? f_at_essential[i] : 0.0);
        }
    }

    // EliminateRowsCols expects an mfem::Array<int>.
    mfem::Array<int> ess_tdof_arr(static_cast<int>(local_indices.size()));
    for (std::size_t i = 0; i < local_indices.size(); ++i)
    {
        ess_tdof_arr[static_cast<int>(i)] = local_indices[i];
    }
    K_hyp.EliminateRowsCols(ess_tdof_arr);

    // Write the prescribed (or 0) values at the eliminated rows.
    for (std::size_t i = 0; i < local_indices.size(); ++i)
    {
        f_par(local_indices[i]) = local_vals[i];
    }
}

void ApplyDirichletToDistributedK(mfem::HypreParMatrix& K_hyp,
                                  mfem::Vector& f_par,
                                  const std::vector<int>& ess_global_tdofs,
                                  mfem::ParFiniteElementSpace& fes)
{
    ApplyDirichletToDistributedK(K_hyp, f_par, ess_global_tdofs, fes,
                                 std::vector<double>{});
}

//==============================================================================
// NewtonResidualAtULin — r1 = K · u_lin
//==============================================================================

mfem::Vector NewtonResidualAtULin(const mfem::HypreParMatrix& K_hyp,
                                  const mfem::Vector& u_lin_local)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::elastic::newton_residual_at_u_lin");
    mfem::Vector r1(u_lin_local.Size());
    K_hyp.Mult(u_lin_local, r1);
    return r1;
}

//==============================================================================
// FindAllBoundaryTdofs
//==============================================================================

std::vector<int> FindAllBoundaryTdofs(mfem::ParMesh& pmesh,
                                      mfem::ParFiniteElementSpace& fes)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::elastic::find_all_boundary_tdofs");

    MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0,
                "FindAllBoundaryTdofs: parent ParMesh has no boundary "
                "attributes.");
    const int n_bdr_attrs = pmesh.bdr_attributes.Max();

    // Mark all boundary attributes essential.
    mfem::Array<int> ess_bdr(n_bdr_attrs);
    ess_bdr = 1;

    // GetEssentialTrueDofs is vdim-aware: it returns local TDOFs for
    // ALL vector components on the marked boundary.
    mfem::Array<int> ess_tdof_list;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    const int offset = fes.GetMyTDofOffset();
    std::vector<int> out;
    out.reserve(ess_tdof_list.Size());
    for (int i = 0; i < ess_tdof_list.Size(); ++i)
    {
        out.push_back(ess_tdof_list[i] + offset);
    }
    return out;
}

//==============================================================================
// CollectBoundaryTdofValues
//==============================================================================

std::vector<double> CollectBoundaryTdofValues(
    const std::vector<int>& boundary_global_tdofs,
    const mfem::Vector& u_lin_local,
    mfem::ParFiniteElementSpace& fes)
{
    const int my_first = fes.GetMyTDofOffset();
    const int my_n = fes.GetTrueVSize();

    std::vector<double> vals;
    vals.reserve(boundary_global_tdofs.size());
    for (int gd : boundary_global_tdofs)
    {
        if (gd >= my_first && gd < my_first + my_n)
        {
            vals.push_back(u_lin_local(gd - my_first));
        }
        else
        {
            vals.push_back(0.0);
        }
    }
    return vals;
}

}  // namespace mortar_pbc
