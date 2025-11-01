#ifndef mechanics_operator_ext_hpp
#define mechanics_operator_ext_hpp

#include "fem_operators/mechanics_integrators.hpp"

#include "mfem.hpp"

/**
 * @brief L1-Jacobi smoothing preconditioner for mechanics finite element operators.
 *
 * MechOperatorJacobiSmoother implements an efficient Jacobi-type preconditioner specifically
 * designed for mechanics problems with essential boundary conditions. The preconditioner
 * uses diagonal scaling with damping to provide effective preconditioning for iterative
 * linear solvers in Newton-Raphson frameworks.
 *
 * Key features for mechanics applications:
 * - L1-Jacobi scaling for improved convergence on mechanics problems
 * - Proper essential boundary condition handling with identity scaling
 * - Damping parameter for stability control and convergence tuning
 * - Device-compatible implementation for GPU acceleration
 * - Integration with partial and element assembly operators
 *
 * The L1-Jacobi approach:
 * - Uses L1 norm of matrix rows for diagonal approximation when full diagonal unavailable
 * - Provides more robust scaling than simple Jacobi for some problem types
 * - Incorporates damping for stability in challenging nonlinear problems
 * - Handles essential boundary conditions through identity preconditioning
 *
 * Essential boundary condition treatment:
 * - Essential DOFs receive identity preconditioning (scaling factor = damping)
 * - Maintains consistency with constrained operator structure
 * - Preserves constraint satisfaction during iterative solution
 * - Prevents ill-conditioning from constraint enforcement
 *
 * Performance characteristics:
 * - Setup cost: O(ndof) diagonal inverse computation
 * - Application cost: O(ndof) scaled vector addition
 * - Memory usage: O(ndof) for diagonal storage
 * - Device execution: Full GPU compatibility for large-scale problems
 *
 * @ingroup ExaConstit_fem_operators
 */
class MechOperatorJacobiSmoother : public mfem::Solver {
public:
    /**
     * @brief Construct Jacobi smoother with diagonal vector and essential boundary conditions.
     *
     * @param d Diagonal vector (or approximation) for preconditioning scaling
     * @param ess_tdofs Array of essential true DOF indices
     * @param damping Damping parameter for stability control (default: 1.0)
     *
     * Initializes the Jacobi smoother by computing damped diagonal inverse and
     * setting up essential boundary condition handling. The damping parameter
     * provides stability control and can improve convergence for difficult problems.
     *
     * Initialization process:
     * 1. Sets up solver with system size from diagonal vector
     * 2. Allocates device-compatible vectors for diagonal inverse and residual
     * 3. Calls Setup() to compute damped diagonal inverse
     * 4. Configures essential boundary condition treatment
     *
     * Damping parameter effects:
     * - damping < 1.0: Under-relaxation for stability in difficult problems
     * - damping = 1.0: Standard Jacobi scaling (default)
     * - damping > 1.0: Over-relaxation (use with caution)
     *
     * Essential boundary condition setup:
     * - Essential DOFs receive identity scaling (dinv[i] = damping)
     * - Maintains consistency with constrained system structure
     * - Prevents numerical issues from constraint enforcement
     *
     * @note Diagonal vector ownership not transferred to smoother
     * @note Essential DOF array reference must remain valid for smoother lifetime
     * @note Damping parameter affects both regular and essential DOFs
     */
    MechOperatorJacobiSmoother(const mfem::Vector& d,
                               const mfem::Array<int>& ess_tdofs,
                               const double damping = 1.0);
    ~MechOperatorJacobiSmoother() {}

    /**
     * @brief Apply Jacobi preconditioning to input vector.
     *
     * @param x Input vector (right-hand side or residual)
     * @param y Output vector (preconditioned result)
     *
     * Applies damped Jacobi preconditioning to the input vector, providing
     * diagonal scaling with proper essential boundary condition handling.
     * The method supports both direct and iterative application modes.
     *
     * Application modes:
     * - Direct mode (iterative_mode=false): y = dinv .* x
     * - Iterative mode (iterative_mode=true): y += dinv .* (x - A*y)
     *
     * Direct mode application:
     * - Simple diagonal scaling of input vector
     * - Efficient for basic preconditioning in Krylov solvers
     * - Cost: O(ndof) vector operations
     *
     * Iterative mode application:
     * - Computes residual r = x - A*y using provided operator
     * - Updates solution y += dinv .* r
     * - Suitable for stationary iteration and smoothing applications
     *
     * Implementation features:
     * - Device-compatible vector operations for GPU execution
     * - Vectorized scaling operations for performance
     * - Proper handling of essential boundary conditions
     * - Integration with MFEM's solver framework
     *
     * Error checking:
     * - Validates input and output vector sizes
     * - Ensures dimensional consistency for safe operation
     *
     * @note Iterative mode requires valid operator pointer from SetOperator()
     * @note All vector operations performed on device when available
     * @note Essential boundary conditions handled automatically through diagonal setup
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const;

    /**
     * @brief Set operator for iterative mode residual computation.
     *
     * @param op Reference to operator for residual computation in iterative mode
     *
     * Configures the smoother for iterative mode operation by storing a reference
     * to the linear operator. This enables residual-based smoothing operations
     * commonly used in multigrid and stationary iteration methods.
     *
     * The operator is used for:
     * - Residual computation: r = b - A*x in iterative mode
     * - Stationary iteration: x_new = x_old + dinv .* r
     * - Smoothing operations in multigrid hierarchies
     *
     * @note Operator reference must remain valid for smoother lifetime
     * @note Required for iterative_mode=true in Mult() operations
     */
    void SetOperator(const mfem::Operator& op) {
        oper = &op;
    }

    /**
     * @brief Setup diagonal inverse with damping and boundary condition handling.
     *
     * @param diag Diagonal vector for inverse computation and scaling setup
     *
     * Computes the damped diagonal inverse required for Jacobi preconditioning,
     * including proper treatment of essential boundary conditions. This method
     * can be called multiple times to update the preconditioner with new diagonal
     * information during Newton-Raphson iterations.
     *
     * The setup algorithm:
     * 1. Configures vectors for device execution
     * 2. Computes damped diagonal inverse: dinv[i] = damping / diag[i]
     * 3. Applies essential boundary condition treatment: dinv[ess_dof] = damping
     * 4. Ensures all operations are device-compatible for GPU execution
     *
     * Diagonal inverse computation:
     * - Standard DOFs: Uses damped inverse of provided diagonal entries
     * - Essential DOFs: Uses damping parameter directly for identity scaling
     * - Device execution: Vectorized operations for GPU performance
     *
     * Essential boundary condition handling:
     * - Overwrites diagonal inverse for essential DOFs with damping value
     * - Provides identity preconditioning for constrained degrees of freedom
     * - Maintains numerical stability and constraint satisfaction
     *
     * @note Can be called multiple times to update diagonal information
     * @note All operations performed on device when GPU execution enabled
     * @note Essential DOF treatment ensures stable constraint handling
     */
    void Setup(const mfem::Vector& diag);

private:
    /** @brief Total number of degrees of freedom in the system */
    const int ndofs;

    /** @brief Diagonal inverse with damping for preconditioning application */
    mfem::Vector dinv;

    /** @brief Damping parameter for stability and convergence control */
    const double damping;

    /** @brief Reference to essential true DOF indices for boundary condition handling */
    const mfem::Array<int>& ess_tdof_list;

    /** @brief Working vector for residual computation in iterative mode */
    mutable mfem::Vector residual;

    /** @brief Pointer to operator for iterative mode residual computation */
    const mfem::Operator* oper;
};

#endif /* mechanics_operator_hpp */
