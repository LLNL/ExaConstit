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

    if (toml_input.contains("amgf_gamma")) {
        options.amgf_gamma = toml::find<double>(toml_input, "amgf_gamma");
    }

    return options;
}

/**
 * @brief Parse trust-region options from a TOML sub-table.
 *
 * Each field is optional — if not present in the TOML, the struct's default
 * value is preserved. This lets users override only the parameters they need
 * to tune.
 */
TrustRegionOptions TrustRegionOptions::from_toml(const toml::value& toml_input) {
    TrustRegionOptions options;

    if (toml_input.contains("delta_init")) {
        options.delta_init = toml::find<double>(toml_input, "delta_init");
    }

    if (toml_input.contains("delta_min")) {
        options.delta_min = toml::find<double>(toml_input, "delta_min");
    }

    if (toml_input.contains("delta_max")) {
        options.delta_max = toml::find<double>(toml_input, "delta_max");
    }

    if (toml_input.contains("xi_lg")) {
        options.xi_lg = toml::find<double>(toml_input, "xi_lg");
    }

    if (toml_input.contains("xi_ug")) {
        options.xi_ug = toml::find<double>(toml_input, "xi_ug");
    }

    if (toml_input.contains("xi_lo")) {
        options.xi_lo = toml::find<double>(toml_input, "xi_lo");
    }

    if (toml_input.contains("xi_uo")) {
        options.xi_uo = toml::find<double>(toml_input, "xi_uo");
    }

    if (toml_input.contains("xi_inc")) {
        options.xi_inc = toml::find<double>(toml_input, "xi_inc");
    }

    if (toml_input.contains("xi_dec")) {
        options.xi_dec = toml::find<double>(toml_input, "xi_dec");
    }

    if (toml_input.contains("xi_forced_inc")) {
        options.xi_forced_inc = toml::find<double>(toml_input, "xi_forced_inc");
    }

    if (toml_input.contains("reject_increase")) {
        options.reject_increase = toml::find<bool>(toml_input, "reject_increase");
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

    // Parse the optional trust-region sub-table when using the dogleg solver.
    // We always parse the table if present (regardless of nl_solver) so that
    // options validation can flag inconsistent configurations later.
    if (toml_input.contains("trust_region")) {
        options.trust_region = TrustRegionOptions::from_toml(
            toml::find(toml_input, "trust_region"));
    }

    return options;
}

/**
 * @brief Parse the saddle-system residual scaling options (Phase 5.11).
 *
 * Each field is optional — missing fields preserve the struct
 * defaults defined in option_parser_v2.hpp (enabled=false,
 * per_subblock=false, partition=FACE_EDGE, floor=1e-12,
 * range_cap=1e12). Accepted TOML keys: `enabled` (bool),
 * `per_subblock` (bool), `partition` (string), `floor` (double),
 * `range_cap` (double).
 */
SaddleScalingOptions SaddleScalingOptions::from_toml(const toml::value& toml_input) {
    SaddleScalingOptions options;

    if (toml_input.contains("enabled")) {
        options.enabled = toml::find<bool>(toml_input, "enabled");
    }

    if (toml_input.contains("per_subblock")) {
        options.per_subblock = toml::find<bool>(toml_input, "per_subblock");
    }

    if (toml_input.contains("partition")) {
        options.partition = string_to_subblock_partition(
            toml::find<std::string>(toml_input, "partition"));
    }

    if (toml_input.contains("floor")) {
        options.floor = toml::find<double>(toml_input, "floor");
    }

    if (toml_input.contains("range_cap")) {
        options.range_cap = toml::find<double>(toml_input, "range_cap");
    }

    return options;
}

/**
 * @brief Parse the mortar-PBC saddle-point solver options (Phase 5).
 *
 * Each field is optional — missing fields preserve the struct defaults
 * defined in option_parser_v2.hpp (MINRES, rel_tol=1e-10, abs_tol=1e-12,
 * max_iter=500, BLOCK_JACOBI, print_level=0). The accepted TOML keys
 * mirror the existing `[Solvers.Krylov]` table for consistency:
 * `linear_solver` (string), `rel_tol`, `abs_tol`, `max_iter`,
 * `preconditioner` (string), `print_level`.
 */
SaddlePointSolverOptions SaddlePointSolverOptions::from_toml(const toml::value& toml_input) {
    SaddlePointSolverOptions options;

    if (toml_input.contains("method")) {
        options.method = string_to_saddle_point_method(
            toml::find<std::string>(toml_input, "method"));
    }
    
    if (toml_input.contains("linear_solver") || toml_input.contains("solver")) {
        // Support both naming conventions for parity with [Solvers.Krylov].
        const auto& key = toml_input.contains("linear_solver") ? "linear_solver" : "solver";
        options.linear_solver = string_to_saddle_point_solver_type(
            toml::find<std::string>(toml_input, key));
    }
    
    if (toml_input.contains("preconditioner")) {
        options.preconditioner = string_to_saddle_point_preconditioner(
            toml::find<std::string>(toml_input, "preconditioner"));
    }
    
    if (toml_input.contains("rel_tol")) {
        options.rel_tol = toml::find<double>(toml_input, "rel_tol");
    }
    
    if (toml_input.contains("abs_tol")) {
        options.abs_tol = toml::find<double>(toml_input, "abs_tol");
    }
    
    if (toml_input.contains("max_iter") || toml_input.contains("iter")) {
        const auto& key = toml_input.contains("max_iter") ? "max_iter" : "iter";
        options.max_iter = toml::find<int>(toml_input, key);
    }
    
    if (toml_input.contains("print_level")) {
        options.print_level = toml::find<int>(toml_input, "print_level");
    }

    if (toml_input.contains("augmented_lagrangian_gamma")) {
        options.augmented_lagrangian_gamma =
            toml::find<double>(toml_input, "augmented_lagrangian_gamma");
    }

    // Phase 5.11 — saddle-system residual scaling sub-table.
    // Optional; when absent, options.scaling stays as nullopt and
    // the Newton solver runs the unscaled path.
    if (toml_input.contains("Scaling")) {
        options.scaling = SaddleScalingOptions::from_toml(
            toml::find(toml_input, "Scaling"));
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

    // Parse mortar-PBC saddle-point solver section (Phase 5).
    // The table is optional — when not present, the SaddlePointSolverOptions
    // defaults apply, which is the right behavior for non-mortar runs
    // (the saddle_point options are simply unused).
    if (toml_input.contains("SaddlePoint")) {
        options.saddle_point = SaddlePointSolverOptions::from_toml(
            toml::find(toml_input, "SaddlePoint"));
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
                      "(JACOBI, AMG, ILU, L1GS, CHEBYSHEV, AMGF, or "
                      "AMGF_AUG_LAGRANGIAN)");
        return false;
    }

    // Implement validation logic
    return true;
}

/**
 * @brief Validate trust-region option ranges and consistency.
 *
 * Step-by-step verification:
 *   1. Trust-region radius bounds: delta_min must be positive and delta_max
 *      must exceed delta_min
 *   2. Initial radius must lie within [delta_min, delta_max]
 *   3. The "good" rho band [xi_lg, xi_ug] must lie inside the "ok" band
 *      [xi_lo, xi_uo] — otherwise the radius update logic is inconsistent
 *   4. Increase factors must be > 1 and decrease factor must be in (0, 1)
 *
 * Each failure is reported with WARNING_0_OPT pointing to the offending field.
 */
bool TrustRegionOptions::validate() const {
    if (delta_min <= 0.0) {
        WARNING_0_OPT("Error: TrustRegion table provided a non-positive delta_min");
        return false;
    }

    if (delta_max <= delta_min) {
        WARNING_0_OPT("Error: TrustRegion table provided delta_max <= delta_min");
        return false;
    }

    if (delta_init < delta_min || delta_init > delta_max) {
        WARNING_0_OPT("Error: TrustRegion table provided delta_init outside [delta_min, delta_max]");
        return false;
    }

    if (xi_lg <= xi_lo) {
        WARNING_0_OPT("Error: TrustRegion table requires xi_lg > xi_lo "
                      "(good band must lie inside ok band)");
        return false;
    }

    if (xi_ug >= xi_uo) {
        WARNING_0_OPT("Error: TrustRegion table requires xi_ug < xi_uo "
                      "(good band must lie inside ok band)");
        return false;
    }

    if (xi_lg >= xi_ug) {
        WARNING_0_OPT("Error: TrustRegion table requires xi_lg < xi_ug");
        return false;
    }

    if (xi_lo >= xi_uo) {
        WARNING_0_OPT("Error: TrustRegion table requires xi_lo < xi_uo");
        return false;
    }

    if (xi_inc <= 1.0) {
        WARNING_0_OPT("Error: TrustRegion table requires xi_inc > 1.0");
        return false;
    }

    if (xi_dec <= 0.0 || xi_dec >= 1.0) {
        WARNING_0_OPT("Error: TrustRegion table requires xi_dec in (0, 1)");
        return false;
    }

    if (xi_forced_inc <= 1.0) {
        WARNING_0_OPT("Error: TrustRegion table requires xi_forced_inc > 1.0");
        return false;
    }

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

    if (nl_solver != NonlinearSolverType::NR &&
        nl_solver != NonlinearSolverType::NRLS &&
        nl_solver != NonlinearSolverType::TRDOG) {
        WARNING_0_OPT("Error: NonLinearSolver table did not provide a valid nl_solver option "
                      "(`NR`, `NRLS`, or `TRDOG`)");
        return false;
    }

    // If trust-region parameters were supplied, verify they are self-consistent.
    // We allow a TRDOG solver without a [trust_region] sub-table — the defaults
    // are applied in that case.
    if (trust_region.has_value()) {
        if (!trust_region->validate()) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Validate the saddle-system residual scaling options (Phase 5.11).
 *
 * Step-by-step verification:
 *   1. `partition` must be a recognized enum value (not NOTYPE).
 *   2. `floor` must be strictly positive — guards against division
 *      by zero in the scaling rule.
 *   3. `range_cap` must exceed 1.0 — clamping below unity would
 *      mean even commensurate residuals get rescaled, which is
 *      not useful.
 *   4. `range_cap` must exceed `floor` — the clip interval
 *      $[\mathrm{floor},\, \mathrm{range\_cap}]$ must be valid.
 *
 * Per-field validation failures emit `WARNING_0_OPT` pointing at
 * the offending key. Validation auto-passes when the master
 * `enabled` flag is false (defaults are valid; we don't bother
 * range-checking a disabled scaling configuration).
 */
bool SaddleScalingOptions::validate() const {
    if (!enabled) {
        // Disabled scaling: don't bother range-checking. Defaults
        // and any user values are fine because they're unused.
        return true;
    }

    if (partition == SubblockPartition::NOTYPE) {
        WARNING_0_OPT("Error: SaddlePoint.Scaling table did not provide a valid "
                      "`partition` (FACE_EDGE or PER_PAIR)");
        return false;
    }

    if (floor <= 0.0) {
        WARNING_0_OPT("Error: SaddlePoint.Scaling table provided a non-positive `floor` "
                      "(must be strictly positive)");
        return false;
    }

    if (range_cap <= 1.0) {
        WARNING_0_OPT("Error: SaddlePoint.Scaling table provided `range_cap` <= 1.0 "
                      "(must be > 1 for meaningful clamping)");
        return false;
    }

    if (range_cap <= floor) {
        WARNING_0_OPT("Error: SaddlePoint.Scaling table provided `range_cap` <= `floor` "
                      "(clip interval must be non-degenerate)");
        return false;
    }

    return true;
}

/**
 * @brief Validate the mortar-PBC saddle-point solver options (Phase 5).
 *
 * The defaults set in option_parser_v2.hpp are valid, so missing
 * `[Solvers.SaddlePoint]` tables auto-pass. Only explicit user
 * configuration can fail here — invalid solver type, invalid
 * preconditioner, non-positive iteration count, or negative
 * tolerances.
 */
bool SaddlePointSolverOptions::validate() const {
    if (method == SaddlePointMethod::NOTYPE) {
        WARNING_0_OPT("Error: SaddlePoint table did not provide a valid `method` "
                      "(STANDARD or AUGMENTED_LAGRANGIAN)");
        return false;
    }
    if (linear_solver == SaddlePointSolverType::NOTYPE) {
        WARNING_0_OPT("Error: SaddlePoint table did not provide a valid `linear_solver` "
                      "(MINRES, GMRES, or BICGSTAB)");
        return false;
    }
    if (preconditioner == SaddlePointPreconditioner::NOTYPE) {
        WARNING_0_OPT("Error: SaddlePoint table did not provide a valid `preconditioner` "
                      "(BLOCK_JACOBI or NONE)");
        return false;
    }
    if (max_iter < 1) {
        WARNING_0_OPT("Error: SaddlePoint table did not provide a positive `max_iter`");
        return false;
    }
    if (rel_tol < 0.0) {
        WARNING_0_OPT("Error: SaddlePoint table provided a negative `rel_tol`");
        return false;
    }
    if (abs_tol < 0.0) {
        WARNING_0_OPT("Error: SaddlePoint table provided a negative `abs_tol`");
        return false;
    }
    // Phase 5.11 — validate the scaling sub-table if present.
    // When absent (nullopt), nothing to check; when present, the
    // scaling struct's own validate() runs its range checks.
    if (scaling.has_value() && !scaling->validate()) {
        return false;
    }
    return true;
}

bool SaddlePointSolverOptions::validate_for_mortar_preconditioner(
    PreconditionerType k_preconditioner) const {
    if (!validate()) {
        return false;
    }

    const bool amgf_prec =
        k_preconditioner == PreconditionerType::AMGF ||
        k_preconditioner == PreconditionerType::AMGF_AUG_LAGRANGIAN;
    if (amgf_prec && linear_solver == SaddlePointSolverType::MINRES) {
        WARNING_0_OPT("Error: AMGF preconditioners cannot be used with the mortar "
                      "SaddlePoint MINRES solver. MFEM's AMGFSolver is a "
                      "filtered/multiplicative preconditioner and is not guaranteed "
                      "to satisfy MINRES' symmetric preconditioner contract. Use "
                      "`[Solvers.SaddlePoint] linear_solver = \"GMRES\"` for AMGF.");
        return false;
    }

    if (amgf_prec) {
#ifndef EXACONSTIT_HAVE_PARALLEL_DIRECT_SOLVER
        WARNING_0_OPT("Error: AMGF preconditioner requires MFEM to be built with a "
                    "parallel sparse direct solver (SuperLU_DIST) for the exact "
                    "filtered-subspace solve, but this build has none. Rebuild "
                    "MFEM with MFEM_USE_SUPERLU=YES (see scripts/install) or "
                    "choose a different preconditioner.");
        return false;
#endif
    }

    return true;
}

bool SolverOptions::validate() {
    if (!nonlinear_solver.validate())
        return false;
    if (!linear_solver.validate())
        return false;

    // Phase 5+ — `saddle_point.validate()` is invoked from
    // ExaOptions::validate() under a `mesh.periodicity` gate (see
    // option_parser_v2.cpp). It's skipped here because SolverOptions
    // has no visibility into mesh.periodicity, and we don't want
    // stale [Solvers.SaddlePoint] tables to fail validation on
    // non-mortar runs.

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

    const bool amgf_prec =
        linear_solver.preconditioner == PreconditionerType::AMGF ||
        linear_solver.preconditioner == PreconditionerType::AMGF_AUG_LAGRANGIAN;

    // AMGF needs assembled HypreParMatrix operators for BoomerAMG and the
    // filtered-subspace solve. Reject unsupported configurations explicitly
    // before the legacy GPU/EA/PA path silently rewrites the preconditioner.
    if (amgf_prec) {
        if (linear_solver.solver_type == LinearSolverType::MINRES) {
            WARNING_0_OPT("Error: AMGF preconditioners cannot be used with MINRES. "
                          "MFEM's AMGFSolver is a filtered/multiplicative "
                          "preconditioner and is not guaranteed to satisfy "
                          "MINRES' symmetric-positive-definite preconditioner "
                          "contract. Use `solver = \"GMRES\"` for AMGF.");
            return false;
        }
        if (assembly != AssemblyType::FULL) {
            WARNING_0_OPT("Error: AMGF preconditioner requires FULL assembly. Element Assembly "
                          "(EA) is preferred on GPU for performance but the AMGF "
                          "preconditioner branch requires a fully assembled HypreParMatrix "
                          "for BoomerAMG and the filtered subspace solver. Either set "
                          "`assembly = \"FULL\"` and switch rtmodel to CPU or OPENMP, or "
                          "disable AMGF.");
            return false;
        }
        if (rtmodel == RTModel::GPU) {
            WARNING_0_OPT("Error: AMGF requires FULL assembly, which is not supported on GPU "
                          "runtime. Switch rtmodel to CPU or OPENMP.");
            return false;
        }
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
