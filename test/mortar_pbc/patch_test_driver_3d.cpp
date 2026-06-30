// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of the shared 3D mortar-PBC patch test
// driver. See header for design doc.

#include "patch_test_driver_3d.hpp"

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "elastic_3d_helpers.hpp"
#include "mortar_constraint_operator.hpp"
#include "saddle_point_solver.hpp"
#include "visualization_3d.hpp"

#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mortar_pbc {

namespace {

//==============================================================================
// F-choice parser — superset of all three driver's choices.
//==============================================================================
mfem::DenseMatrix ParseFChoice(const std::string& name)
{
    mfem::DenseMatrix F(3, 3);
    F = 0.0;
    if (name == "uniaxial")
    {
        F(0,0) = 1.20; F(1,1) = 0.95; F(2,2) = 0.95;
    }
    else if (name == "biaxial")
    {
        F(0,0) = 1.15; F(1,1) = 1.10; F(2,2) = 0.90;
    }
    else if (name == "shear")
    {
        F(0,0) = 1.00; F(0,1) = 0.10; F(0,2) = 0.05;
        F(1,0) = 0.05; F(1,1) = 1.00; F(1,2) = 0.10;
        F(2,0) = 0.10; F(2,1) = 0.05; F(2,2) = 1.00;
    }
    else if (name == "mild")
    {
        F(0,0) = 1.05; F(0,1) = 0.02; F(0,2) = 0.01;
        F(1,0) = 0.01; F(1,1) = 0.97; F(1,2) = 0.02;
        F(2,0) = 0.02; F(2,1) = 0.01; F(2,2) = 1.03;
    }
    else if (name == "mild-shear")
    {
        F(0,0) = 1.05; F(0,1) = 0.05; F(0,2) = 0.02;
        F(1,0) = 0.02; F(1,1) = 1.02; F(1,2) = 0.05;
        F(2,0) = 0.05; F(2,1) = 0.02; F(2,2) = 1.03;
    }
    else
    {
        MFEM_ABORT("ParseFChoice: unknown F choice '" << name << "'");
    }
    return F;
}

//==============================================================================
// Pattern label and PASS-criterion helpers
//==============================================================================
const char* PatternName(PatchTestPattern p)
{
    switch (p)
    {
        case PatchTestPattern::Homogeneous:  return "homogeneous";
        case PatchTestPattern::Strip:        return "strip";
        case PatchTestPattern::Checkerboard: return "checkerboard";
    }
    return "unknown";
}

bool PatternIsHeterogeneous(PatchTestPattern p)
{
    return p != PatchTestPattern::Homogeneous;
}

//==============================================================================
// Element-attribute assignment per pattern.
//
// Mirrors the Python `build_*_mesh_3d` helpers exactly. Acts on a
// SERIAL `mfem::Mesh` BEFORE it gets wrapped into a `ParMesh`, so
// every rank applies the same attribute pattern (then METIS
// partitions; attributes follow elements through the partition).
//==============================================================================
void ApplyAttributePattern(mfem::Mesh& mesh,
                           PatchTestPattern pattern,
                           double L)
{
    if (pattern == PatchTestPattern::Homogeneous)
    {
        for (int e = 0; e < mesh.GetNE(); ++e) { mesh.SetAttribute(e, 1); }
        mesh.SetAttributes();
        return;
    }

    const double L_half = 0.5 * L;
    for (int e = 0; e < mesh.GetNE(); ++e)
    {
        mfem::Array<int> verts;
        mesh.GetElementVertices(e, verts);
        double xc = 0.0, yc = 0.0, zc = 0.0;
        for (int k = 0; k < verts.Size(); ++k)
        {
            const double* xyz = mesh.GetVertex(verts[k]);
            xc += xyz[0]; yc += xyz[1]; zc += xyz[2];
        }
        const double inv_n = 1.0 / static_cast<double>(verts.Size());
        xc *= inv_n; yc *= inv_n; zc *= inv_n;

        int attr = 1;
        if (pattern == PatchTestPattern::Strip)
        {
            attr = (xc < L_half) ? 1 : 2;
        }
        else  // Checkerboard
        {
            const int bx = (xc >= L_half) ? 1 : 0;
            const int by = (yc >= L_half) ? 1 : 0;
            const int bz = (zc >= L_half) ? 1 : 0;
            attr = ((bx + by + bz) % 2 == 0) ? 1 : 2;
        }
        mesh.SetAttribute(e, attr);
    }
    mesh.SetAttributes();
}

//==============================================================================
// PWConstCoefficient-based linear-elastic K assembly.
//
// Returns the freshly-allocated HypreParMatrix; caller owns and
// must `delete`. Per MFEM #793 (and the Python's
// `assemble_heterogeneous_K_hypre` docstring), we build a fresh
// ParBilinearForm each call so the returned HypreParMatrix does not
// alias any other instance — important because the heterogeneous
// path needs TWO independent K's (full + eliminated).
//==============================================================================
mfem::HypreParMatrix* AssemblePWConstK(mfem::ParFiniteElementSpace& fes,
                                       double E1, double E2, double nu)
{
    const double mu_1  = 0.5 * E1 / (1.0 + nu);
    const double lam_1 = E1 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu_2  = 0.5 * E2 / (1.0 + nu);
    const double lam_2 = E2 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    mfem::Vector mu_vec(2);  mu_vec(0)  = mu_1;  mu_vec(1)  = mu_2;
    mfem::Vector lam_vec(2); lam_vec(0) = lam_1; lam_vec(1) = lam_2;

    mfem::PWConstCoefficient mu_coef(mu_vec);
    mfem::PWConstCoefficient lam_coef(lam_vec);

    mfem::ParBilinearForm a(&fes);
    a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lam_coef, mu_coef));
    a.Assemble();
    a.Finalize();
    return a.ParallelAssemble();
}

//==============================================================================
// Volume-averaged F via Gauss quadrature.
//
// <F> = I + (1/V) ∫ ∇u dV. Mirrors `compute_volume_averaged_F_3d`
// in the Python multi-step driver.
//==============================================================================
mfem::DenseMatrix ComputeVolumeAveragedF(mfem::ParMesh& pmesh,
                                         mfem::ParFiniteElementSpace& fes,
                                         const mfem::Vector& u_total)
{
    MPI_Comm comm = pmesh.GetComm();
    mfem::ParGridFunction u_gf(&fes);
    {
        mfem::Vector u_local(u_total.Size());
        // DEVICE_DEBUG-clean copy from u_total to u_local. SetFromTrueDofs
        // takes a const reference and reads it through the memory manager.
        const double* src = u_total.HostRead();
        double*       dst = u_local.HostWrite();
        for (int i = 0; i < u_total.Size(); ++i) { dst[i] = src[i]; }
        u_gf.SetFromTrueDofs(u_local);
    }

    double integral_grad_u_local[9] = {0.0};
    double total_volume_local = 0.0;

    const int n_loc_elems = pmesh.GetNE();
    for (int e = 0; e < n_loc_elems; ++e)
    {
        mfem::ElementTransformation* T = pmesh.GetElementTransformation(e);
        const int geom = pmesh.GetElementBaseGeometry(e);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(geom, 4);

        const int n_q = ir.GetNPoints();
        for (int qp = 0; qp < n_q; ++qp)
        {
            const mfem::IntegrationPoint& ip = ir.IntPoint(qp);
            T->SetIntPoint(&ip);
            const double w = ip.weight * T->Weight();

            mfem::DenseMatrix grad_u(3, 3);
            grad_u = 0.0;
            u_gf.GetVectorGradient(*T, grad_u);
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    integral_grad_u_local[i*3 + j] += w * grad_u(i, j);
                }
            }
            total_volume_local += w;
        }
    }

    double integral_global[9] = {0.0};
    double total_volume_global = 0.0;
    MPI_Allreduce(integral_grad_u_local, integral_global, 9, MPI_DOUBLE,
                  MPI_SUM, comm);
    MPI_Allreduce(&total_volume_local, &total_volume_global, 1, MPI_DOUBLE,
                  MPI_SUM, comm);

    mfem::DenseMatrix F_avg(3, 3);
    F_avg = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            F_avg(i, j) = integral_global[i*3 + j] / total_volume_global
                         + (i == j ? 1.0 : 0.0);
        }
    }
    return F_avg;
}

//==============================================================================
// Pretty-print helpers for rank-0 output.
//==============================================================================
void PrintMatrix(const mfem::DenseMatrix& M, const std::string& label)
{
    std::cout << "  " << label << " =" << std::endl;
    for (int i = 0; i < M.NumRows(); ++i)
    {
        std::cout << "    [";
        for (int j = 0; j < M.NumCols(); ++j)
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "% .6f", M(i, j));
            std::cout << buf;
            if (j + 1 < M.NumCols()) { std::cout << ", "; }
        }
        std::cout << "]" << std::endl;
    }
}

double MaxAbs(const mfem::DenseMatrix& M)
{
    double m = 0.0;
    for (int i = 0; i < M.NumRows(); ++i)
    {
        for (int j = 0; j < M.NumCols(); ++j)
        {
            m = std::max(m, std::abs(M(i, j)));
        }
    }
    return m;
}

}  // anonymous namespace

//==============================================================================
// RunPatchTest3D — main driver entry point
//==============================================================================

int RunPatchTest3D(const PatchTestConfig& cfg)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::patch_test::run");

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const mfem::DenseMatrix F = ParseFChoice(cfg.F_choice);
    const bool heterogeneous = PatternIsHeterogeneous(cfg.pattern);

    if (rank == 0)
    {
        std::cout << "========================================================="
                  << std::endl;
        std::cout << "  3D mortar-PBC patch test (Phase 4.1.A C++ port)"
                  << std::endl;
        std::cout << "  pattern = " << PatternName(cfg.pattern)
                  << ", n = " << cfg.n
                  << ", L = " << cfg.L
                  << ", np = " << nranks << std::endl;
        std::cout << "  F = " << cfg.F_choice << ":" << std::endl;
        PrintMatrix(F, "F_macro");
        if (heterogeneous)
        {
            std::cout << "  Material 1 (attr=1): E = " << cfg.E1
                      << ", nu = " << cfg.nu << std::endl;
            std::cout << "  Material 2 (attr=2): E = " << cfg.E2
                      << ", nu = " << cfg.nu
                      << "  (contrast = " << (cfg.E2 / cfg.E1) << "x)"
                      << std::endl;
        }
        else
        {
            std::cout << "  E = " << cfg.E1 << ", nu = " << cfg.nu << std::endl;
        }
        std::cout << "========================================================="
                  << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 1 — mesh + attribute pattern + FES
    //--------------------------------------------------------------------------
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        cfg.n, cfg.n, cfg.n,
        mfem::Element::HEXAHEDRON,
        cfg.L, cfg.L, cfg.L, /*sfc_ordering=*/false);
    ApplyAttributePattern(serial, cfg.pattern, cfg.L);

    // Phase 4.4 / Batch 4.4-E Part 2 — optional in-place mesh perturbation.
    // Applied AFTER attribute pattern (so element grouping is set on the
    // unperturbed mesh, where the strip/checkerboard split is unambiguous)
    // but BEFORE ParMesh construction (so MFEM's parallel partitioning
    // sees the perturbed coords). The hook contract is documented in
    // PatchTestConfig::mesh_perturbation.
    if (cfg.mesh_perturbation)
    {
        cfg.mesh_perturbation(serial);
    }

    mfem::ParMesh pmesh(MPI_COMM_WORLD, serial);
    mfem::H1_FECollection fec(/*order=*/1, /*dim=*/3);
    mfem::ParFiniteElementSpace fes(&pmesh, &fec, /*vdim=*/3,
                                    mfem::Ordering::byNODES);

    // Lessons learned §P4.8.8: collective MFEM ops must be called on
    // every rank; capture before printing.
    const int n_global_elems = pmesh.GetGlobalNE();
    const int n_global_tdofs = fes.GlobalTrueVSize();
    if (rank == 0)
    {
        std::cout << std::endl
                  << "[1] Mesh: " << n_global_elems
                  << " global elements (hex), global TDOFs = "
                  << n_global_tdofs << std::endl;
        if (heterogeneous)
        {
            // Element-attribute distribution on rank 0 (informational
            // only; not used for correctness).
            int n_attr1 = 0, n_attr2 = 0;
            for (int e = 0; e < pmesh.GetNE(); ++e)
            {
                if (pmesh.GetAttribute(e) == 1) { ++n_attr1; }
                else if (pmesh.GetAttribute(e) == 2) { ++n_attr2; }
            }
            std::cout << "    Element-attribute distribution (rank 0): "
                      << "{1: " << n_attr1 << ", 2: " << n_attr2 << "}"
                      << std::endl;
        }
    }

    //--------------------------------------------------------------------------
    // Step 2 — classifier + constraint matrix
    //--------------------------------------------------------------------------
    BoundaryClassifier3D classifier(pmesh, fes);
    ConstraintBuilder3D builder(classifier);
    const int n_lam_total = builder.NumConstraints();
    if (rank == 0)
    {
        std::cout << "[2] Classifier: " << classifier.Corners().size()
                  << " corners, " << classifier.Edges().size()
                  << " edges, " << classifier.Faces().size() << " faces"
                  << std::endl;
        std::cout << "    Constraint matrix C: " << n_lam_total << " rows"
                  << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 3 — collect corner gtdofs (for both K-Dirichlet and corner
    //          column zeroing — the latter is implicit in the C++
    //          builder; see test_patch_3d_pbc.cpp comment).
    //--------------------------------------------------------------------------
    std::vector<int> corner_gtdofs;
    corner_gtdofs.reserve(24);
    for (const auto& kv : classifier.Corners())
    {
        const auto& c = kv.second;
        corner_gtdofs.push_back(c.gtdof_x);
        corner_gtdofs.push_back(c.gtdof_y);
        corner_gtdofs.push_back(c.gtdof_z);
    }
    if (rank == 0)
    {
        std::cout << "[3] Corner Dirichlet TDOFs: " << corner_gtdofs.size()
                  << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 4 — build distributed C as HypreParMatrix and/or as the EA
    // operator (Phase 4.3 / Batch S).
    //
    // Phase 4.2 / Batch N: row partition is FES-aligned; the builder
    // derives n_lam_local internally from routed-block content. Use
    // NumLocalRows() to query the value for diagnostics.
    //
    // Phase 4.3 / Batch S: with the EA path now available, the
    // construction depends on cfg.constraint_storage:
    //   - HypreParMatrix path: build `C` (HypreParMatrix). Used by
    //     step 9's saddle-point solve and by step 11's constraint
    //     residual check.
    //   - ElementAssembly path: build `C_op` (MortarConstraintOperator).
    //     Used analogously.
    //   - cfg.ab_compare = true: build BOTH; the saddle-point solve
    //     runs once per path; step 11 uses whichever path is chosen
    //     as the primary (driven by cfg.constraint_storage).
    //--------------------------------------------------------------------------

    std::unique_ptr<MortarConstraintOperator> C_op = std::make_unique<MortarConstraintOperator>(classifier);

    const int n_lam_local = builder.NumLocalRows();
    if (rank == 0)
    {
        std::cout << "[4] C built ("
                  << ("HypreParMatrix + EA")
                  << "); this rank owns "
                  << n_lam_local << " of " << n_lam_total << " rows"
                  << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 5 — assemble K via PWConstCoefficient.
    //
    // For HOMOGENEOUS: one K matrix; r1 = K · u_lin then Dirichlet-
    //   eliminate K and r1 in one shot.
    //
    // For HETEROGENEOUS: TWO K matrices. K_full stays untouched and
    //   is used for r1 = K_full · u_lin. K_eliminated has Dirichlet
    //   applied and is the saddle-point top block.
    //
    // CRITICAL — do NOT compute r1 = K_eliminated · u_lin: with
    //   heterogeneous material under affine BC, the affine field is
    //   NOT the equilibrium, so K_full · u_lin ≠ 0 at free rows
    //   (specifically, the K_uc · u_lin[corner] coupling). Eliminating
    //   K first zeros out K_uc, which would falsify r1 to look like
    //   equilibrium and force the solver to invent a wrong fluctuation
    //   du to "correct" a residual that physically isn't there. The
    //   sign of the resulting du would be wrong.
    //
    //   This is a bug we WILL hit if r1's K is eliminated before the
    //   matvec — there's no automatic "wrong K" detection. The Python
    //   `multistep_driver._solve_independently` docstring (lines
    //   333-358) is the canonical write-up of this trap.
    //--------------------------------------------------------------------------
    std::unique_ptr<mfem::HypreParMatrix> K_full;
    std::unique_ptr<mfem::HypreParMatrix> K_eliminated;
    if (heterogeneous)
    {
        K_full.reset(AssemblePWConstK(fes, cfg.E1, cfg.E2, cfg.nu));
        K_eliminated.reset(AssemblePWConstK(fes, cfg.E1, cfg.E2, cfg.nu));
    }
    else
    {
        // Homogeneous: PWConstCoefficient with E1=E2 is identical to
        // a single ConstantCoefficient. We still go through the same
        // path so the codepath is exercised.
        const double E_uniform = cfg.E1;
        K_eliminated.reset(AssemblePWConstK(fes, E_uniform, E_uniform, cfg.nu));
        // K_full not needed for homogeneous (the homogeneous
        // single-K-with-elimination path is mathematically equivalent
        // because K_full · u_lin = 0 anyway).
    }
    if (rank == 0)
    {
        std::cout << "[5] K (HypreParMatrix) assembled "
                  << (heterogeneous ? "(K_full + K_eliminated)"
                                    : "(single K)") << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 6 — u_lin = (F - I) X
    //--------------------------------------------------------------------------
    mfem::Vector u_lin = ApplyLinearPart(fes, F);
    if (rank == 0)
    {
        std::cout << "[6] u_lin built. ||u_lin||_inf (rank 0) = "
                  << u_lin.Normlinf() << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 7 — residual r1, then Dirichlet on K_eliminated + r1 corners
    //--------------------------------------------------------------------------
    mfem::Vector r1(K_eliminated->Height());
    if (heterogeneous)
    {
        // r1 = K_full · u_lin (un-eliminated K — see Step 5 comment).
        K_full->Mult(u_lin, r1);
        // Zero corner entries of r1 directly. The saddle-point top
        // block uses K_eliminated which has identity rows at corners,
        // so r1[corner] = 0 enforces du[corner] = 0 (i.e. the
        // increment respects the corner BC).
        ApplyDirichletToDistributedK(*K_eliminated, r1, corner_gtdofs, fes);
    }
    else
    {
        // Homogeneous: r1 = K · u_lin then ApplyDirichlet zeroes both
        // the corner rows/cols of K and r1[corner].
        K_eliminated->Mult(u_lin, r1);
        ApplyDirichletToDistributedK(*K_eliminated, r1, corner_gtdofs, fes);
    }
    if (rank == 0)
    {
        std::cout << "[7] r1 = K"
                  << (heterogeneous ? "_full" : "")
                  << " · u_lin computed; Dirichlet applied to "
                  << "K_eliminated and r1 corners" << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 8 — constraint RHS r2 = 0
    //--------------------------------------------------------------------------
    mfem::Vector r2(n_lam_local);
    r2 = 0.0;
    if (rank == 0)
    {
        std::cout << "[8] r2 = 0 (warm-start at u_init = u_lin)" << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 9 — distributed Krylov saddle-point solve.
    //
    // Phase 4.3 / Batch S: branches on cfg.constraint_storage.
    //--------------------------------------------------------------------------
    SaddlePointSolverConfig sps_cfg;
    sps_cfg.solver_type = KrylovType::GMRES;
    sps_cfg.prec_type   = SaddlePrecType::BlockJacobi;
    sps_cfg.rel_tol     = 1.0e-12;
    sps_cfg.abs_tol     = 1.0e-16;
    sps_cfg.max_iter    = 5000;
    sps_cfg.gmres_kdim  = std::min(2000, n_global_tdofs + n_lam_total);
    sps_cfg.print_level = 0;

    mfem::Vector du, dlam;          // primary path's results (used downstream)
    bool primary_converged = false; // primary path's Krylov convergence,
                                    // checked by PASS criteria below.
    int  primary_iters     = -1;    // iteration count for diagnostic.

    // Phase 5.5.B.2.A — single EA path; K_eliminated viewed as an
    // Operator, K_jacobi_prec as a HypreSmoother(K, Jacobi).
    mfem::HypreSmoother K_jacobi_prec(*K_eliminated,
                                       mfem::HypreSmoother::Jacobi);

    SaddlePointSolver sps(sps_cfg);
    if (rank == 0)
    {
        std::cout << std::endl
                  << "[9] Saddle-point solve (Element-Assembly path, "
                  << "Krylov + block-Jacobi)" << std::endl;
    }
    sps.Solve(*K_eliminated, *C_op, K_jacobi_prec,
              r1, r2, du, dlam);
    primary_converged = sps.LastConverged();
    primary_iters     = sps.LastIterations();
    if (rank == 0)
    {
        std::cout << "    Krylov: iters = " << primary_iters
                  << ", converged = "
                  << (primary_converged ? "yes" : "NO")
                  << ", final residual = "
                  << sps.LastFinalNorm() << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 10 — recover u_total = u_lin + du; ||du||_∞
    //--------------------------------------------------------------------------
    mfem::Vector u_total(u_lin.Size());
    {
        // DEVICE_DEBUG-clean: u_lin and du come from elsewhere with
        // unknown memory state; declare host access intent here.
        const double* ul = u_lin.HostRead();
        const double* dd = du.HostRead();
        double*       ut = u_total.HostWrite();
        for (int i = 0; i < u_lin.Size(); ++i)
        {
            ut[i] = ul[i] + dd[i];
        }
    }
    const double du_max_local = du.Normlinf();
    double du_max_global = 0.0;
    MPI_Allreduce(&du_max_local, &du_max_global, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << std::endl
                  << "[10] u_total = u_lin + du recovered." << std::endl;
        std::cout << "     ||du||_inf (global)    = " << du_max_global;
        if (heterogeneous)
        {
            std::cout << "  (heterogeneous: must be > "
                      << cfg.du_min_heterogeneous
                      << " — fluctuation must be present)";
        }
        else
        {
            std::cout << "  (homogeneous: must be < "
                      << cfg.du_max_homogeneous
                      << " — fluctuation should be ~0)";
        }
        std::cout << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 11 — verify <F> ≈ F_macro and constraint residual
    //--------------------------------------------------------------------------
    mfem::DenseMatrix F_avg = ComputeVolumeAveragedF(pmesh, fes, u_total);
    mfem::DenseMatrix F_diff(F_avg);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j) { F_diff(i, j) -= F(i, j); }
    }
    const double F_diff_max = MaxAbs(F_diff);
    if (rank == 0)
    {
        std::cout << std::endl << "[11] Volume-averaged F:" << std::endl;
        PrintMatrix(F_avg, "<F>");
        std::cout << "     ||<F> - F_macro||_inf = " << F_diff_max << std::endl;
    }

    // Constraint residual check. In EA-only mode, `C` (HypreParMatrix)
    // is null; we route through C_op. In all other cases, `C` is
    // non-null and we keep the original HypreParMatrix path. Both paths
    // produce the same answer to FP-rearrangement precision (Batch Q
    // tightened this to 1e-12), so the constraint_residual_tol of
    // 1e-9 has plenty of headroom either way.
    mfem::Vector Cu_total(n_lam_local);
    mfem::Vector Cu_lin(n_lam_local);

    MFEM_ASSERT(C_op != nullptr,
                "patch driver: neither C nor C_op is built — "
                "constraint_storage logic error");
    C_op->Mult(u_total, Cu_total);
    C_op->Mult(u_lin,   Cu_lin);

    mfem::Vector residual(n_lam_local);
    {
        const double* ct = Cu_total.HostRead();
        const double* cl = Cu_lin.HostRead();
        double*       rd = residual.HostWrite();
        for (int i = 0; i < n_lam_local; ++i)
        {
            rd[i] = ct[i] - cl[i];
        }
    }
    const double constraint_residual_local = residual.Normlinf();
    double constraint_residual_global = 0.0;
    MPI_Allreduce(&constraint_residual_local, &constraint_residual_global, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << "     ||C·u_total - C·u_lin||_inf = "
                  << constraint_residual_global << std::endl;
    }

    //--------------------------------------------------------------------------
    // PASS criteria
    //--------------------------------------------------------------------------
    const bool pass_krylov     = primary_converged;
    bool pass_du;
    if (heterogeneous)
    {
        // For heterogeneous, the fluctuation MUST be non-trivial. A
        // ~0 du indicates a porting bug — most likely r1 was computed
        // with K_eliminated instead of K_full (see Step 5 comment).
        pass_du = du_max_global > cfg.du_min_heterogeneous;
    }
    else
    {
        // For homogeneous, du is the analytical zero up to roundoff.
        pass_du = du_max_global < cfg.du_max_homogeneous;
    }
    const bool pass_F          = F_diff_max < cfg.F_average_tol;
    const bool pass_constraint =
        constraint_residual_global < cfg.constraint_residual_tol;
    const bool all_pass = pass_krylov && pass_du && pass_F && pass_constraint;

    if (rank == 0)
    {
        const char* sep =
            "=========================================================";
        std::cout << std::endl << sep << std::endl;
        std::cout << "  PASS criteria (" << PatternName(cfg.pattern) << "):"
                  << std::endl;
        std::cout << "     Krylov converged             : "
                  << (pass_krylov ? "OK" : "FAIL") << " ("
                  << primary_iters << " iters)" << std::endl;
        if (heterogeneous)
        {
            std::cout << "     ||du||_inf > "
                      << cfg.du_min_heterogeneous
                      << "        : "
                      << (pass_du ? "OK" : "FAIL") << " ("
                      << du_max_global << ")" << std::endl;
        }
        else
        {
            std::cout << "     ||du||_inf < "
                      << cfg.du_max_homogeneous
                      << "        : "
                      << (pass_du ? "OK" : "FAIL") << " ("
                      << du_max_global << ")" << std::endl;
        }
        std::cout << "     ||<F> - F_macro|| < " << cfg.F_average_tol
                  << "    : "
                  << (pass_F ? "OK" : "FAIL") << " ("
                  << F_diff_max << ")" << std::endl;
        std::cout << "     ||C·u - C·u_lin|| < "
                  << cfg.constraint_residual_tol
                  << "    : "
                  << (pass_constraint ? "OK" : "FAIL") << " ("
                  << constraint_residual_global << ")" << std::endl;
        std::cout << "  Overall: " << (all_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << sep << std::endl;
    }

    //--------------------------------------------------------------------------
    // Step 12 — ParaView visualization (optional)
    //--------------------------------------------------------------------------
    if (cfg.paraview)
    {
        std::string viz_name = cfg.paraview_name;
        if (viz_name.empty())
        {
            viz_name = std::string("patch_3d_") + PatternName(cfg.pattern)
                     + "_" + cfg.F_choice;
        }
        if (rank == 0)
        {
            std::cout << std::endl
                      << "[12] Writing ParaView output to "
                      << cfg.paraview_dir << "/ as " << viz_name
                      << ".pvd" << std::endl;
        }
        WriteVisualization(pmesh, fes, u_total, u_lin, du,
                           cfg.paraview_dir, viz_name);
    }

    return all_pass ? 0 : 1;
}

}  // namespace mortar_pbc
