#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

#include <iostream>

LinearSolverOptions LinearSolverOptions::from_toml(const toml::value& toml_input) {
    LinearSolverOptions options;

    if (toml_input.contains("solver") || toml_input.contains("solver_type")) {
        // Support both naming conventions
        const auto& solver_key = toml_input.contains("solver") ? "solver" : "solver_type";
        options.solver_type = string_to_linear_solver_type(
            toml::find<std::string>(toml_input, solver_key));
    }

    if (toml_input.contains("preconditioner")) {
        options.preconditioner = string_to_preconditioner_type(
            toml::find<std::string>(toml_input, "preconditioner"));
    }

    if (toml_input.contains("abs_tol")) {
        options.abs_tol = toml::find<double>(toml_input, "abs_tol");
    }

    if (toml_input.contains("rel_tol")) {
        options.rel_tol = toml::find<double>(toml_input, "rel_tol");
    }

    if (toml_input.contains("max_iter") || toml_input.contains("iter")) {
        // Support both naming conventions
        const auto& iter_key = toml_input.contains("max_iter") ? "max_iter" : "iter";
        options.max_iter = toml::find<int>(toml_input, iter_key);
    }

    if (toml_input.contains("print_level")) {
        options.print_level = toml::find<int>(toml_input, "print_level");
    }

    return options;
}

NonlinearSolverOptions NonlinearSolverOptions::from_toml(const toml::value& toml_input) {
    NonlinearSolverOptions options;

    if (toml_input.contains("iter")) {
        options.iter = toml::find<int>(toml_input, "iter");
    }

    if (toml_input.contains("rel_tol")) {
        options.rel_tol = toml::find<double>(toml_input, "rel_tol");
    }

    if (toml_input.contains("abs_tol")) {
        options.abs_tol = toml::find<double>(toml_input, "abs_tol");
    }

    if (toml_input.contains("nl_solver")) {
        options.nl_solver = string_to_nonlinear_solver_type(
            toml::find<std::string>(toml_input, "nl_solver"));
    }

    return options;
}

SolverOptions SolverOptions::from_toml(const toml::value& toml_input) {
    SolverOptions options;

    if (toml_input.contains("assembly")) {
        options.assembly = string_to_assembly_type(toml::find<std::string>(toml_input, "assembly"));
    }

    if (toml_input.contains("rtmodel")) {
        options.rtmodel = string_to_rt_model(toml::find<std::string>(toml_input, "rtmodel"));
    }

    if (toml_input.contains("integ_model")) {
        options.integ_model = string_to_integration_model(
            toml::find<std::string>(toml_input, "integ_model"));
    }

    // Parse linear solver section
    if (toml_input.contains("Krylov")) {
        options.linear_solver = LinearSolverOptions::from_toml(toml::find(toml_input, "Krylov"));
    }

    // Parse nonlinear solver section (NR = Newton-Raphson)
    if (toml_input.contains("NR")) {
        options.nonlinear_solver = NonlinearSolverOptions::from_toml(toml::find(toml_input, "NR"));
    }

    return options;
}

bool LinearSolverOptions::validate() const {
    if (max_iter < 1) {
        WARNING_0_OPT("Error: LinearSolver table did not provide a positive iteration count");
        return false;
    }

    if (abs_tol < 0) {
        WARNING_0_OPT("Error: LinearSolver table provided a negative absolute tolerance");
        return false;
    }

    if (rel_tol < 0) {
        WARNING_0_OPT("Error: LinearSolver table provided a negative relative tolerance");
        return false;
    }

    if (solver_type == LinearSolverType::NOTYPE) {
        WARNING_0_OPT("Error: LinearSolver table did not provide a valid solver type (CG, GMRES, "
                      "MINRES, or BICGSTAB)");
        return false;
    }

    if (preconditioner == PreconditionerType::NOTYPE) {
        WARNING_0_OPT("Error: LinearSolver table did not provide a valid preconditioner type "
                      "(JACOBI, AMG, ILU, L1GS, CHEBYSHEV)");
        return false;
    }

    // Implement validation logic
    return true;
}

bool NonlinearSolverOptions::validate() const {
    if (iter < 1) {
        WARNING_0_OPT("Error: NonLinearSolver table did not provide a positive iteration count");
        return false;
    }

    if (abs_tol < 0) {
        WARNING_0_OPT("Error: NonLinearSolver table provided a negative absolute tolerance");
        return false;
    }

    if (rel_tol < 0) {
        WARNING_0_OPT("Error: NonLinearSolver table provided a negative relative tolerance");
        return false;
    }

    if (nl_solver != NonlinearSolverType::NR && nl_solver != NonlinearSolverType::NRLS) {
        WARNING_0_OPT("Error: NonLinearSolver table did not provide a valid nl_solver option (`NR` "
                      "or `NRLS`)");
        return false;
    }

    // Implement validation logic
    return true;
}

bool SolverOptions::validate() {
    if (!nonlinear_solver.validate())
        return false;
    if (!linear_solver.validate())
        return false;

    if (assembly == AssemblyType::NOTYPE) {
        WARNING_0_OPT(
            "Error: Solver table did not provide a valid assembly option (`FULL`, `PA`, or `EA`)");
        return false;
    }

    if (rtmodel == RTModel::NOTYPE) {
        WARNING_0_OPT("Error: Solver table did not provide a valid rtmodel option (`CPU`, "
                      "`OPENMP`, or `GPU`)");
        return false;
    }

    if (integ_model == IntegrationModel::NOTYPE) {
        WARNING_0_OPT(
            "Error: Solver table did not provide a valid integ_model option (`FULL` or `BBAR`)");
        return false;
    }

    if (rtmodel == RTModel::GPU && assembly == AssemblyType::FULL) {
        WARNING_0_OPT("Error: Solver table did not provide a valid assembly option when using GPU "
                      "rtmodel: `FULL` assembly can not be used with `GPU` rtmodels");
        return false;
    }

    if (rtmodel == RTModel::GPU && linear_solver.preconditioner != PreconditionerType::JACOBI) {
        WARNING_0_OPT("Warning: Solver table did not provide a valid preconditioner option when "
                      "using GPU rtmodel: `JACOBI` preconditioner is the only one that can be used "
                      "with `GPU` rtmodels");
        WARNING_0_OPT("Warning: Updating the preconditioner value for you to `JACOBI`");
        linear_solver.preconditioner = PreconditionerType::JACOBI;
    }

    if (assembly != AssemblyType::FULL &&
        linear_solver.preconditioner != PreconditionerType::JACOBI) {
        WARNING_0_OPT("Warning: Solver table did not provide a valid preconditioner option when "
                      "using either `EA` or `PA` assembly: `JACOBI` preconditioner is the only one "
                      "that can be used with those assembly options");
        WARNING_0_OPT("Warning: This can be a result of using legacy decks which did not have this "
                      "field and if so just ignore this warning.");
        WARNING_0_OPT("Warning: Updating the preconditioner value for you to `JACOBI`");
        linear_solver.preconditioner = PreconditionerType::JACOBI;
    }

    // Implement validation logic
    return true;
}
