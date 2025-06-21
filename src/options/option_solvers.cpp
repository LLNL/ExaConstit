#include "options/option_parser_v2.hpp"

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
        options.nl_solver = string_to_nonlinear_solver_type(toml::find<std::string>(toml_input, "nl_solver"));
    }
    
    return options;
}

SolverOptions SolverOptions::from_toml(const toml::value& toml_input) {
    SolverOptions options;
    
    if (toml_input.contains("assembly")) {
        options.assembly = string_to_assembly_type(
            toml::find<std::string>(toml_input, "assembly"));
    }
    
    if (toml_input.contains("rtmodel")) {
        options.rtmodel = string_to_rt_model(
            toml::find<std::string>(toml_input, "rtmodel"));
    }
    
    if (toml_input.contains("integ_model")) {
        options.integ_model = string_to_integration_model(
            toml::find<std::string>(toml_input, "integ_model"));
    }
    
    // Parse linear solver section
    if (toml_input.contains("Krylov")) {
        options.linear_solver = LinearSolverOptions::from_toml(
            toml::find(toml_input, "Krylov"));
    }
    
    // Parse nonlinear solver section (NR = Newton-Raphson)
    if (toml_input.contains("NR")) {
        options.nonlinear_solver = NonlinearSolverOptions::from_toml(
            toml::find(toml_input, "NR"));
    }
    
    return options;
}

bool LinearSolverOptions::validate() const {

    if (max_iter < 1) {
        std::cerr << "Error: LinearSolver table did not provide a positive iteration count" << std::endl;
        return false;
    }

    if (abs_tol < 0) {
        std::cerr << "Error: LinearSolver table provided a negative absolute tolerance" << std::endl;
        return false;
    }

    if (rel_tol < 0) {
        std::cerr << "Error: LinearSolver table provided a negative relative tolerance" << std::endl;
        return false;
    }

    if (solver_type == LinearSolverType::NOTYPE) {
        std::cerr << "Error: LinearSolver table did not provide a valid solver type (CG, GMRES, or MINRES)" << std::endl;
        return false;
    }

    if (preconditioner == PreconditionerType::NOTYPE) {
        std::cerr << "Error: LinearSolver table did not provide a valid preconditioner type (JACOBI or AMG)" << std::endl;
        return false;
    }

    // Implement validation logic
    return true;
}

bool NonlinearSolverOptions::validate() const {
    int iter = 25;
    double rel_tol = 1e-5;
    double abs_tol = 1e-10;
    std::string nl_solver = "NR";

    if (iter < 1) {
        std::cerr << "Error: NonLinearSolver table did not provide a positive iteration count" << std::endl;
        return false;
    }

    if (abs_tol < 0) {
        std::cerr << "Error: NonLinearSolver table provided a negative absolute tolerance" << std::endl;
        return false;
    }

    if (rel_tol < 0) {
        std::cerr << "Error: NonLinearSolver table provided a negative relative tolerance" << std::endl;
        return false;
    }

    if (nl_solver != "NR" && nl_solver != "NRLS") {
        std::cerr << "Error: NonLinearSolver table did not provide a valid nl_solver option (`NR` or `NRLS`)" << std::endl;
        return false;
    }

    // Implement validation logic
    return true;
}

bool SolverOptions::validate() const {

    nonlinear_solver.validate();
    linear_solver.validate();

    if (assembly == AssemblyType::NOTYPE) {
        std::cerr << "Error: Solver table did not provide a valid assembly option (`FULL`, `PA`, or `EA`)" << std::endl;
        return false;
    }

    if (rtmodel == RTModel::NOTYPE) {
        std::cerr << "Error: Solver table did not provide a valid rtmodel option (`CPU`, `OPENMP`, or `GPU`)" << std::endl;
        return false;
    }

    if (integ_model == IntegrationModel::NOTYPE) {
        std::cerr << "Error: Solver table did not provide a valid integ_model option (`FULL` or `BBAR`)" << std::endl;
        return false;
    }

    if (rtmodel == RTModel::GPU && assembly == AssemblyType::FULL) {
        std::cerr << "Error: Solver table did not provide a valid assembly option when using GPU rtmodel: `FULL` assembly can not be used with `GPU` rtmodels" << std::endl;
        return false;
    }

    // Implement validation logic
    return true;
}
