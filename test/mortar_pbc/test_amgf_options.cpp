#include "options/option_parser_v2.hpp"

#include "TOML_Reader/toml.hpp"
#include "mpi.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& details)
{
    if (!cond)
    {
        std::cerr << "[FAIL] " << test_name << ": " << details << std::endl;
        std::exit(1);
    }
}

void TestAmgfStringParsing()
{
    const std::string name = "AMGF preconditioner string parsing";
    AssertOrDie(string_to_preconditioner_type("AMGF") == PreconditionerType::AMGF,
                name, "uppercase AMGF did not parse");
    AssertOrDie(string_to_preconditioner_type("amgf") == PreconditionerType::AMGF,
                name, "lowercase amgf did not parse");
    AssertOrDie(string_to_preconditioner_type("AMGF_AUG_LAGRANGIAN") ==
                    PreconditionerType::AMGF_AUG_LAGRANGIAN,
                name, "uppercase AMGF_AUG_LAGRANGIAN did not parse");
    AssertOrDie(string_to_preconditioner_type("amgf_aug_lagrangian") ==
                    PreconditionerType::AMGF_AUG_LAGRANGIAN,
                name, "lowercase amgf_aug_lagrangian did not parse");
    AssertOrDie(string_to_saddle_point_method("AUGMENTED_LAGRANGIAN") ==
                    SaddlePointMethod::AUGMENTED_LAGRANGIAN,
                name, "uppercase AUGMENTED_LAGRANGIAN method did not parse");
    AssertOrDie(string_to_saddle_point_method("augmented_lagrangian") ==
                    SaddlePointMethod::AUGMENTED_LAGRANGIAN,
                name, "lowercase augmented_lagrangian method did not parse");
    AssertOrDie(string_to_saddle_point_method("STANDARD") ==
                    SaddlePointMethod::STANDARD,
                name, "uppercase STANDARD method did not parse");
}

void TestAmgfTomlParsing()
{
    const std::string name = "AMGF LinearSolver TOML parsing";
    std::istringstream input(R"(
        preconditioner = "amgf_aug_lagrangian"
        amgf_gamma = 2.5
        amgf_subspace_executor = "auto"
    )");
    const toml::value table = toml::parse(input, "amgf option test");

    const auto opts = LinearSolverOptions::from_toml(table);
    AssertOrDie(opts.preconditioner == PreconditionerType::AMGF_AUG_LAGRANGIAN,
                name, "preconditioner did not parse");
    AssertOrDie(opts.amgf_gamma == 2.5, name, "amgf_gamma did not parse");
    AssertOrDie(opts.amgf_subspace_executor == "auto",
                name, "amgf_subspace_executor did not parse");
}

void TestAugmentedLagrangianSaddleTomlParsing()
{
    const std::string name = "Augmented-Lagrangian SaddlePoint TOML parsing";
    std::istringstream input(R"(
        method = "augmented_lagrangian"
        linear_solver = "GMRES"
        preconditioner = "BLOCK_JACOBI"
        augmented_lagrangian_gamma = 12.25
    )");
    const toml::value table = toml::parse(input, "saddle option test");

    const auto opts = SaddlePointSolverOptions::from_toml(table);
    AssertOrDie(opts.method == SaddlePointMethod::AUGMENTED_LAGRANGIAN,
                name, "method did not parse");
    AssertOrDie(opts.linear_solver == SaddlePointSolverType::GMRES,
                name, "linear_solver did not parse");
    AssertOrDie(opts.preconditioner == SaddlePointPreconditioner::BLOCK_JACOBI,
                name, "preconditioner did not parse");
    AssertOrDie(opts.augmented_lagrangian_gamma == 12.25,
                name, "augmented_lagrangian_gamma did not parse");
}

SolverOptions MakeSolverOptions(AssemblyType assembly, RTModel rtmodel,
                                PreconditionerType preconditioner)
{
    SolverOptions opts;
    opts.assembly = assembly;
    opts.rtmodel = rtmodel;
    opts.integ_model = IntegrationModel::DEFAULT;
    opts.linear_solver.solver_type = LinearSolverType::GMRES;
    opts.linear_solver.preconditioner = preconditioner;
    opts.linear_solver.max_iter = 10;
    opts.nonlinear_solver.nl_solver = NonlinearSolverType::TRDOG;
    opts.nonlinear_solver.iter = 10;
    return opts;
}

void TestAmgfValidation()
{
    const std::string name = "AMGF solver validation";

    auto ok = MakeSolverOptions(AssemblyType::FULL, RTModel::CPU,
                                PreconditionerType::AMGF);
    AssertOrDie(ok.validate(), name, "FULL + CPU + AMGF should validate");

    auto ok_omp = MakeSolverOptions(AssemblyType::FULL, RTModel::OPENMP,
                                    PreconditionerType::AMGF_AUG_LAGRANGIAN);
    AssertOrDie(ok_omp.validate(), name,
                "FULL + OPENMP + AMGF_AUG_LAGRANGIAN should validate");

    auto ea = MakeSolverOptions(AssemblyType::EA, RTModel::CPU,
                                PreconditionerType::AMGF);
    AssertOrDie(!ea.validate(), name, "EA + AMGF must be rejected");
    AssertOrDie(ea.linear_solver.preconditioner == PreconditionerType::AMGF,
                name, "EA + AMGF must not be silently rewritten to Jacobi");

    auto pa = MakeSolverOptions(AssemblyType::PA, RTModel::CPU,
                                PreconditionerType::AMGF_AUG_LAGRANGIAN);
    AssertOrDie(!pa.validate(), name,
                "PA + AMGF_AUG_LAGRANGIAN must be rejected");
    AssertOrDie(pa.linear_solver.preconditioner ==
                    PreconditionerType::AMGF_AUG_LAGRANGIAN,
                name, "PA + AMGF_AUG_LAGRANGIAN must not be silently rewritten");

    auto gpu_ea = MakeSolverOptions(AssemblyType::EA, RTModel::GPU,
                                    PreconditionerType::AMGF);
    AssertOrDie(!gpu_ea.validate(), name, "GPU + EA + AMGF must be rejected");
    AssertOrDie(gpu_ea.linear_solver.preconditioner == PreconditionerType::AMGF,
                name, "GPU + EA + AMGF must not be silently rewritten to Jacobi");

    auto invalid_exec = MakeSolverOptions(AssemblyType::FULL, RTModel::CPU,
                                          PreconditionerType::AMGF);
    invalid_exec.linear_solver.amgf_subspace_executor = "serial";
    AssertOrDie(!invalid_exec.validate(), name,
                "invalid AMGF subspace executor must be rejected");

    auto minres = MakeSolverOptions(AssemblyType::FULL, RTModel::CPU,
                                    PreconditionerType::AMGF);
    minres.linear_solver.solver_type = LinearSolverType::MINRES;
    AssertOrDie(!minres.validate(), name,
                "AMGF + MINRES must be rejected because AMGFSolver is not "
                "guaranteed to satisfy MINRES' symmetric preconditioner "
                "contract");

    auto saddle_minres = SaddlePointSolverOptions{};
    saddle_minres.linear_solver = SaddlePointSolverType::MINRES;
    AssertOrDie(!saddle_minres.validate_for_mortar_preconditioner(
                    PreconditionerType::AMGF),
                name,
                "AMGF + mortar SaddlePoint MINRES must be rejected because "
                "AMGFSolver is not guaranteed to satisfy MINRES' symmetric "
                "preconditioner contract");

    auto saddle_gmres = SaddlePointSolverOptions{};
    saddle_gmres.linear_solver = SaddlePointSolverType::GMRES;
    AssertOrDie(saddle_gmres.validate_for_mortar_preconditioner(
                    PreconditionerType::AMGF),
                name, "AMGF + mortar SaddlePoint GMRES should validate");

    auto saddle_augmented_amg_minres = SaddlePointSolverOptions{};
    saddle_augmented_amg_minres.method =
        SaddlePointMethod::AUGMENTED_LAGRANGIAN;
    saddle_augmented_amg_minres.linear_solver = SaddlePointSolverType::MINRES;
    AssertOrDie(saddle_augmented_amg_minres.validate_for_mortar_preconditioner(
                    PreconditionerType::AMG),
                name,
                "augmented-Lagrangian mortar method without AMGF should not "
                "inherit the AMGF + MINRES rejection");
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    TestAmgfStringParsing();
    TestAmgfTomlParsing();
    TestAugmentedLagrangianSaddleTomlParsing();
    TestAmgfValidation();
    MPI_Finalize();
    return 0;
}
