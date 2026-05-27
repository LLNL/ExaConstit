
#ifndef MECHANICS_SOLVER
#define MECHANICS_SOLVER

#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"

#include <functional>
#include <memory>

//==============================================================================
// Phase 5.11.F — Newton diagnostic sink.
//
// Optional per-iteration callback for the ExaNewton* family. Invoked
// at the top of each Newton iteration AFTER the new residual norm is
// computed and BEFORE the convergence-check break decides whether
// this iteration is the last. Lets external code (SystemDriver +
// MortarPbcManager when saddle-residual scaling is active, future
// diagnostic post-processors) record norm progression and convergence
// status in a structured way independent of `print_level`-gated
// stdout logging.
//
// When the sink is unset (default), no overhead beyond a null-check
// per iteration. Bit-for-bit pre-5.11.F behavior is preserved.
//
// Note that with the ScaledSaddleOperator from Phase 5.11.D installed
// as the Newton solver's operator, the `norm` field below is in
// scaled coordinates (||D^-1 r||); without the wrapper installed it's
// in physical coordinates. The sink itself doesn't know which —
// that's the caller's responsibility to track.
//==============================================================================
struct NewtonIterDiagnostic
{
    int    iter;            ///< 0-based Newton iteration index
    double norm;             ///< current ||r||
    double norm0;            ///< initial ||r|| (captured at iter 0)
    double norm_max;         ///< convergence threshold
                             ///<   = max(rel_tol*norm0, abs_tol)
    bool   converged_now;    ///< true if (norm <= norm_max) and this
                             ///<   iter's check will break the loop
    // Phase 5.11.J — pointers to the Newton solver's current
    // residual and solution iterate at the moment the sink is
    // invoked. Both are NON-OWNING — the Newton solver owns the
    // underlying storage and may mutate it after the sink returns.
    // Sinks must not retain these pointers; copy data out if
    // persistence is needed.
    //
    // Both default to nullptr to preserve API compatibility with
    // existing sinks (the Phase 5.11.I sink, the test_newton_
    // diagnostic_sink.cpp unit test). New sinks can opt into
    // residual access when these are non-null.
    const mfem::Vector* residual = nullptr;
    const mfem::Vector* solution = nullptr;
};

using NewtonDiagnosticSink =
    std::function<void(const NewtonIterDiagnostic&)>;

struct LinearSolveDiagnostic
{
    int iterations = -1;        ///< Krylov iterations, or -1 if unavailable
    double final_norm = -1.0;   ///< Krylov final residual norm, or -1
    bool converged = false;     ///< Krylov solver convergence flag
};

using LinearSolveDiagnosticSink =
    std::function<void(const LinearSolveDiagnostic&)>;

/**
 * @brief Newton-Raphson solver for nonlinear solid mechanics problems
 *
 * @details This class implements Newton's method for solving nonlinear systems of the form F(x) = b
 * where F is a nonlinear operator. It extends MFEM's IterativeSolver to provide specialized
 * functionality for ExaConstit's solid mechanics applications.
 *
 * The solver uses the Newton-Raphson iteration:
 * x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i) - b]
 *
 * Key features:
 * - Device-compatible implementation for CPU/GPU execution
 * - Integration with MFEM's operator and linear solver framework
 * - Specialized handling for NonlinearForm operators in solid mechanics
 * - Automatic scaling factor adjustment for convergence improvement
 * - Caliper performance profiling integration
 *
 * The method GetGradient() must be implemented for the operator F.
 * The preconditioner is used (in non-iterative mode) to evaluate
 * the action of the inverse gradient of the operator.
 */
class ExaNewtonSolver : public mfem::IterativeSolver {
protected:
    /** @brief Residual vector for Newton iterations */
    mutable mfem::Vector r;

    /** @brief Correction vector for Newton iterations */
    mutable mfem::Vector c;

    /** @brief Pointer to the mechanics nonlinear form operator */
    std::shared_ptr<mfem::Operator> oper_mech;

    /** @brief Pointer to the preconditioner */
    std::shared_ptr<mfem::Solver> prec_mech;

    /// Phase 5.11.F — per-iter callback; null if unset.
    NewtonDiagnosticSink m_diagnostic_sink;

    /// Optional post-linear-solve callback; null if unset.
    LinearSolveDiagnosticSink m_linear_diagnostic_sink;

public:
    /**
     * @brief Default constructor
     *
     * @details Creates an ExaNewtonSolver instance for single-processor execution.
     * The operator and linear solver must be set separately using SetOperator() and SetSolver().
     */
    ExaNewtonSolver() {}

#ifdef MFEM_USE_MPI
    /**
     * @brief MPI constructor
     *
     * @param _comm MPI communicator for parallel execution
     *
     * @details Creates an ExaNewtonSolver instance for parallel execution using the specified
     * MPI communicator. This enables the solver to work with distributed finite element spaces
     * and parallel linear solvers.
     */
    ExaNewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) {}
#endif
    /**
     * @brief Set the nonlinear operator to be solved
     *
     * @param op The nonlinear operator representing F in F(x) = b
     *
     * @details Sets up the solver to work with the given operator. The operator must be square
     * (height == width) and must implement the GetGradient() method for computing Jacobians.
     * This method also initializes the internal residual and correction vectors with appropriate
     * device memory settings.
     *
     * @pre The operator must be square (height == width)
     * @post Internal vectors r and c are sized and configured for device execution
     */
    virtual void SetOperator(const mfem::Operator& op);

    /**
     * @brief Set the operator to be solved (shared-ownership variant).
     *
     * @param op  Shared-pointer to the operator. The operator must
     *            be square (`height == width`) and must implement
     *            `GetGradient` for Jacobian computation.
     *
     * @details Phase 5.5 — accepts any `mfem::Operator` so the same
     * Newton solver can iterate on either a `NonlinearMechOperator`
     * (standard production path) or a `MortarSaddlePointSystem`
     * (mortar PBC path) without a separate solver class.
     *
     * Stores the shared pointer in `oper_mech` so the solver retains
     * ownership across calls, and forwards the raw pointer into the
     * inherited `mfem::IterativeSolver::oper` so the base class's
     * size / preconditioner machinery sees the right operator.
     *
     * @pre The operator must be square (`height == width`).
     * @post `oper`, `oper_mech`, `r`, and `c` are all initialized.
     *
     * @note `shared_ptr<Derived>` to `shared_ptr<Operator>` is an
     *       implicit conversion when `Derived` publicly inherits
     *       from `mfem::Operator`, so existing call sites that
     *       pass a `shared_ptr<NonlinearMechOperator>` continue to
     *       work without source changes.
     */
    virtual void SetOperator(std::shared_ptr<mfem::Operator> op);

    /**
     * @brief Set the linear solver for inverting the Jacobian
     *
     * @param solver Linear solver for the Newton correction equation
     *
     * @details This method is equivalent to calling SetPreconditioner(). The linear solver
     * is used to solve the linearized system [DF(x_i)] c = [F(x_i) - b] at each Newton iteration.
     * Common choices include:
     * - CGSolver for symmetric positive definite systems
     * - GMRESSolver for general nonsymmetric systems
     * - MINRESSolver for symmetric indefinite systems
     */
    virtual void SetSolver(mfem::Solver& solver) {
        prec = &solver;
    }

    /**
     * @brief Set the linear solver for inverting the Jacobian
     *
     * @param solver Linear solver for the Newton correction equation
     *
     * @details This method is equivalent to calling SetPreconditioner(). The linear solver
     * is used to solve the linearized system [DF(x_i)] c = [F(x_i) - b] at each Newton iteration.
     * Common choices include:
     * - CGSolver for symmetric positive definite systems
     * - GMRESSolver for general nonsymmetric systems
     * - MINRESSolver for symmetric indefinite systems
     */
    virtual void SetSolver(std::shared_ptr<mfem::Solver> solver) {
        prec_mech = solver;
    }

    /**
     * @brief Solve the linearized Newton correction equation
     *
     * @param oper Linear operator (typically the Jacobian)
     * @param b Right-hand side vector
     * @param x Solution vector (output)
     *
     * @details This method solves the linearized Newton system using the configured linear solver.
     * It sets up the preconditioner/solver with the given operator and applies it to compute
     * the Newton correction. The method is marked with Caliper profiling for performance analysis.
     *
     * The operation performed is: x = [oper]^{-1} b
     *
     * @note This method may use different Krylov solvers (CG, GMRES, MINRES) depending on
     * the configuration provided during solver setup.
     */
    virtual void CGSolver(mfem::Operator& oper, const mfem::Vector& b, mfem::Vector& x) const;

    /**
     * @brief Solve the nonlinear system F(x) = b using Newton-Raphson method
     *
     * @param b Right-hand side vector (if b.Size() != Height(), assumes b = 0)
     * @param x Solution vector (input: initial guess, output: converged solution)
     *
     * @details Main solution method that implements the Newton-Raphson algorithm:
     *
     * 1. **Initialization**: Set up initial residual r = F(x) - b
     * 2. **Newton Iteration Loop**:
     *    - Check convergence: ||r|| <= max(rel_tol * ||r_0||, abs_tol)
     *    - Compute Jacobian: J = DF(x_i)
     *    - Solve linear system: J * c = r
     *    - Apply scaling factor: x_{i+1} = x_i - scale * c
     *    - Update residual: r = F(x_{i+1}) - b
     *    - Adjust scaling factor if convergence stalls
     * 3. **Convergence Check**: Exit when tolerance is met or max iterations reached
     *
     * **Adaptive Scaling**: The solver automatically reduces the scaling factor to 0.5
     * when the residual ratio exceeds 0.5, helping to stabilize convergence for
     * difficult nonlinear problems.
     *
     * **Performance Profiling**: Includes Caliper markers for detailed performance analysis
     * of Newton iterations and linear solver calls.
     *
     * @pre SetOperator() and SetSolver() must be called before Mult()
     * @pre The operator must implement GetGradient() for Jacobian computation
     *
     * @post final_iter contains the number of Newton iterations performed
     * @post final_norm contains the final residual norm
     * @post converged flag indicates whether the solver converged
     */
    virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const;

    // We're going to comment this out for now.
    /** @brief This method can be overloaded in derived classes to implement line
        search algorithms. */
    /** The base class implementation (NewtonSolver) simply returns 1. A return
        value of 0 indicates a failure, interrupting the Newton iteration. */
    // virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const
    // { return 1.0; }

    /**
     * @brief Phase 5.11.F — install a per-iter diagnostic callback.
     *
     * @param sink  Callable to invoke once per Newton iter at the
     *              top of the loop, after norm computation and
     *              before the convergence-check break. Pass a
     *              default-constructed `NewtonDiagnosticSink{}` (or
     *              `nullptr` to the implicit conversion) to disable.
     *
     * @details Inherited as-is by `ExaNewtonLSSolver` and (post-
     * 5.11.G) `ExaTrustRegionSolver` — both invoke the same sink
     * from their own `Mult` bodies.
     *
     * The sink is invoked AFTER each iter's residual norm has been
     * computed (so `norm` is the up-to-date value) and BEFORE the
     * `if (norm <= norm_max) break` check, with
     * `converged_now = (norm <= norm_max)`. The sink thus knows
     * whether this iter is the loop's last.
     *
     * The sink runs on ALL ranks (it's called from inside `Mult`
     * which is per-rank Newton machinery). If the sink performs I/O,
     * the implementer is responsible for rank-gating
     * (e.g. only printing on rank 0).
     */
    void SetDiagnosticSink(NewtonDiagnosticSink sink)
    {
        m_diagnostic_sink = std::move(sink);
    }

    void SetLinearDiagnosticSink(LinearSolveDiagnosticSink sink)
    {
        m_linear_diagnostic_sink = std::move(sink);
    }
};

/**
 * @brief Newton-Raphson solver with line search for enhanced convergence
 *
 * @details This class extends ExaNewtonSolver to include a line search algorithm that
 * improves convergence robustness for highly nonlinear problems. The line search method
 * uses a quadratic variation approach to find an optimal scaling factor for each Newton step.
 *
 * The line search algorithm:
 * 1. Evaluates the residual at three points: x, x - 0.5*c, x - c
 * 2. Fits a quadratic polynomial to these residual norms
 * 3. Finds the minimum of the quadratic to determine optimal step size
 * 4. Falls back to heuristic rules if the quadratic fit is invalid
 *
 * This approach is particularly useful for:
 * - Large deformation problems with geometric nonlinearities
 * - Material models with strong nonlinearities (e.g., plasticity, damage)
 * - Problems where standard Newton-Raphson exhibits oscillatory behavior
 *
 * The method GetGradient() must be implemented for the operator F.
 * The preconditioner is used (in non-iterative mode) to evaluate
 * the action of the inverse gradient of the operator.
 *
 * Reference: Based on quadratic variation line search described in
 * "Numerical Methods for Large Eigenvalue Problems" (https://doi.org/10.1007/978-3-642-01970-8_46)
 */
class ExaNewtonLSSolver : public ExaNewtonSolver {
public:
    /**
     * @brief Default constructor
     *
     * @details Creates an ExaNewtonLSSolver instance for single-processor execution.
     * Inherits all functionality from ExaNewtonSolver and adds line search capabilities.
     */
    ExaNewtonLSSolver() {}

#ifdef MFEM_USE_MPI
    /**
     * @brief MPI constructor
     *
     * @param _comm MPI communicator for parallel execution
     *
     * @details Creates an ExaNewtonLSSolver instance for parallel execution using the specified
     * MPI communicator. The line search algorithm works correctly in parallel environments.
     */
    ExaNewtonLSSolver(MPI_Comm _comm) : ExaNewtonSolver(_comm) {}
#endif
    /** @brief Use parent class SetOperator methods */
    using ExaNewtonSolver::SetOperator;

    /** @brief Use parent class SetSolver methods */
    using ExaNewtonSolver::SetSolver;

    /** @brief Use parent class CGSolver method */
    using ExaNewtonSolver::CGSolver;

    /**
     * @brief Solve the nonlinear system F(x) = b using Newton-Raphson with line search
     *
     * @param b Right-hand side vector (if b.Size() != Height(), assumes b = 0)
     * @param x Solution vector (input: initial guess, output: converged solution)
     *
     * @details Enhanced Newton-Raphson method with quadratic line search for improved robustness:
     *
     * 1. **Standard Newton Setup**: Compute residual and Jacobian as in standard Newton
     * 2. **Line Search Algorithm**:
     *    - Store current state: x_prev = x
     *    - Evaluate residual at x - c: q3 = ||F(x - c) - b||
     *    - Evaluate residual at x - 0.5*c: q2 = ||F(x - 0.5*c) - b||
     *    - Current residual: q1 = ||F(x) - b||
     *    - Fit quadratic: ε = (3*q1 - 4*q2 + q3) / (4*(q1 - 2*q2 + q3))
     *    - Apply optimal step: x = x_prev - ε*c
     * 3. **Fallback Strategy**:
     *    - If quadratic fit is invalid: ε = 1.0 (full Newton step)
     *    - If full step increases residual: ε = 0.05 (heavily damped step)
     *    - If algorithm fails completely: terminate with convergence failure
     *
     * **Line Search Benefits**:
     * - Prevents divergence in highly nonlinear problems
     * - Reduces oscillatory behavior near solution
     * - Maintains quadratic convergence when possible
     * - Provides automatic step size control
     *
     * **Performance Considerations**:
     * - Requires 2 additional function evaluations per iteration
     * - Includes Caliper profiling for line search performance analysis
     * - May increase computational cost but improves robustness
     *
     * @pre SetOperator() and SetSolver() must be called before Mult()
     * @pre The operator must implement GetGradient() for Jacobian computation
     *
     * @post final_iter contains the number of Newton iterations performed
     * @post final_norm contains the final residual norm
     * @post converged flag indicates whether the solver converged
     *
     * @note The line search algorithm prints the relaxation factor when print_level >= 0
     */
    virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const;
};

#endif
