#ifndef mechanics_operator_ext_hpp
#define mechanics_operator_ext_hpp

#include "fem_operators/mechanics_integrators.hpp"

#include "mfem.hpp"

/**
 * @brief Abstract base class for extended nonlinear mechanics operators with advanced assembly strategies.
 * 
 * NonlinearMechOperatorExt provides a unified interface for high-performance mechanics operators
 * that implement specialized assembly strategies beyond standard MFEM capabilities. This class
 * serves as the foundation for partial assembly (PA) and element assembly (EA) implementations
 * optimized for large-scale nonlinear mechanics simulations.
 * 
 * The extension framework enables:
 * - Matrix-free operator implementations for memory efficiency
 * - Specialized assembly strategies for different hardware architectures
 * - Custom preconditioning approaches for mechanics problems
 * - Device-portable implementations for CPU/GPU execution
 * 
 * Key design principles:
 * - Pure virtual interface ensuring consistent derived class implementation
 * - Memory class abstraction for automatic device memory management
 * - Integration with MFEM's operator framework for solver compatibility
 * - Separation of assembly and application phases for optimization
 * 
 * Derived classes implement specific strategies:
 * - PANonlinearMechOperatorGradExt: Partial assembly for moderate memory use
 * - EANonlinearMechOperatorGradExt: Element assembly for minimal memory use
 * - Future extensions for other assembly approaches
 * 
 * @ingroup ExaConstit_fem_operators
 */
class NonlinearMechOperatorExt : public mfem::Operator
{
   protected:
      /** @brief Reference to underlying MFEM nonlinear form (not owned by this class) */
      mfem::NonlinearForm *oper_mech; // Not owned
   public:
      /**
       * @brief Construct extended operator from existing MFEM nonlinear form.
       * 
       * @param _mech_operator Pointer to MFEM nonlinear form containing integrators and finite element space
       * 
       * Initializes the extended operator wrapper around an existing MFEM nonlinear form.
       * The constructor establishes the operator size based on the finite element space
       * true vector size, ensuring compatibility with MFEM's linear solver interfaces.
       * 
       * The base constructor:
       * - Sets operator dimensions from finite element space
       * - Stores reference to nonlinear form for integrator access
       * - Prepares foundation for derived class assembly implementations
       * 
       * @note Does not take ownership of the nonlinear form pointer
       * @note Derived classes must implement assembly and diagonal assembly methods
       */
      NonlinearMechOperatorExt(mfem::NonlinearForm *_mech_operator);
      /**
       * @brief Get memory class for device-aware memory management.
       * 
       * @return Memory class enum specifying device or host memory requirements
       * 
       * Returns the appropriate memory class for the current device configuration,
       * enabling automatic selection between host and device memory allocation.
       * This ensures optimal memory placement for CPU or GPU execution.
       * 
       * Memory class selection affects:
       * - Vector allocation strategies in derived classes
       * - Data transfer optimization between host and device
       * - Performance optimization for target hardware architecture
       * 
       * @note Implementation delegates to MFEM's device memory management
       * @note Derived classes should use this for consistent memory allocation
       */
      virtual mfem::MemoryClass GetMemoryClass() const
      { return mfem::Device::GetMemoryClass(); }

      /**
       * @brief Assemble operator data structures for subsequent operations.
       * 
       * Pure virtual method that derived classes must implement to perform
       * assembly-specific setup operations. This includes precomputing and
       * storing data structures needed for efficient operator application.
       * 
       * Assembly responsibilities vary by strategy:
       * - Partial assembly: Precompute element matrices and store compactly
       * - Element assembly: Precompute full element matrices
       * - Custom strategies: Problem-specific optimizations
       * 
       * The assembly phase is separated from operator application to:
       * - Amortize setup costs over multiple operator applications
       * - Enable assembly reuse across Newton-Raphson iterations
       * - Optimize memory layout for specific hardware architectures
       * 
       * @note Must be called before operator application methods
       * @note Assembly cost is amortized over multiple Mult() calls
       */
      virtual void Assemble() = 0;

      /**
       * @brief Assemble diagonal entries for preconditioning operations.
       * 
       * @param diag Output vector for assembled diagonal entries
       * 
       * Pure virtual method for computing diagonal entries of the operator,
       * which are essential for Jacobi preconditioning in iterative linear solvers.
       * The diagonal provides a simple but effective preconditioner for many
       * mechanics problems, particularly with appropriate damping strategies.
       * 
       * Diagonal assembly considerations:
       * - Must handle essential boundary conditions appropriately
       * - Should provide reasonable approximation to full matrix diagonal
       * - Memory efficient computation without full matrix assembly
       * - Device-compatible implementation for GPU execution
       * 
       * The diagonal is used for:
       * - Jacobi preconditioning in Krylov solvers
       * - Scaling operations in multi-physics coupling
       * - Condition number estimation and monitoring
       * - Adaptive solution strategies
       * 
       * @note Output vector must be properly sized for true DOF space
       * @note Essential boundary condition handling varies by implementation
       */
      virtual void AssembleDiagonal(mfem::Vector &diag) const = 0;
};

/**
 * @brief Partial assembly implementation for nonlinear mechanics gradient operators.
 * 
 * PANonlinearMechOperatorGradExt implements a memory-efficient partial assembly strategy
 * for nonlinear mechanics Jacobian operators. This approach provides a balance between
 * computational efficiency and memory usage by precomputing and storing element-level
 * data structures while avoiding full matrix assembly.
 * 
 * The partial assembly strategy:
 * - Precomputes element matrices in compressed format
 * - Applies operators through element-level matrix-vector products
 * - Maintains compatibility with MFEM's finite element framework
 * - Supports efficient device execution for CPU and GPU platforms
 * 
 * Key advantages:
 * - Significantly reduced memory requirements compared to full assembly
 * - Faster setup time compared to element assembly approaches
 * - Good computational efficiency for moderate-sized problems
 * - Natural support for adaptive mesh refinement and contact
 * 
 * Memory efficiency features:
 * - Element data stored in compressed format
 * - Temporary vectors reused across operations
 * - Device-aware memory allocation and management
 * - Minimal overhead for essential boundary condition handling
 * 
 * The implementation supports:
 * - Standard and local (unconstrained) operator applications
 * - Efficient diagonal assembly for Jacobi preconditioning
 * - Template-based optimization for different operation types
 * - Integration with ExaConstit's material model framework
 * 
 * @ingroup ExaConstit_fem_operators
 */
class PANonlinearMechOperatorGradExt : public NonlinearMechOperatorExt
{
   protected:
      /** @brief Finite element space for DOF management and element operations */
      const mfem::FiniteElementSpace *fes; // Not owned
      
      /** @brief Local element vector in element DOF ordering */
      mutable mfem::Vector localX;
      
      /** @brief Local element result vector in element DOF ordering */
      mutable mfem::Vector localY;
      
      /** @brief Working vector initialized to ones for certain operations */
      mutable mfem::Vector ones;
      
      /** @brief Prolongation operation result vector */
      mutable mfem::Vector px;
      
      /** @brief Element restriction operator for local-to-global DOF mapping */
      const mfem::Operator *elem_restrict_lex; // Not owned
      
      /** @brief Prolongation operator for conforming finite element spaces */
      const mfem::Operator *P;
      
      /** @brief Reference to essential true DOF list for boundary condition enforcement */
      const mfem::Array<int> &ess_tdof_list;
   public:
      /**
       * @brief Construct partial assembly operator with essential boundary conditions.
       * 
       * @param _mech_operator Pointer to MFEM nonlinear form containing integrators
       * @param ess_tdofs Reference to array of essential true DOF indices
       * 
       * Initializes the partial assembly operator by setting up element restriction
       * and prolongation operators, allocating working vectors, and preparing data
       * structures for efficient element-level operations.
       * 
       * The constructor:
       * 1. Calls base class constructor for operator size setup
       * 2. Extracts finite element space from nonlinear form
       * 3. Sets up element restriction operator for local-global mapping
       * 4. Allocates device-compatible working vectors
       * 5. Stores reference to essential DOF list for constraint handling
       * 
       * Element restriction setup:
       * - Uses native DOF ordering for optimal memory access patterns
       * - Configures prolongation operator for conforming spaces
       * - Allocates working vectors with appropriate device memory type
       * - Initializes vectors for device execution compatibility
       * 
       * @note Essential DOF reference must remain valid for operator lifetime
       * @note Working vectors are configured for device execution when available
       */
      PANonlinearMechOperatorGradExt(mfem::NonlinearForm *_mech_operator,
                                     const mfem::Array<int> &ess_tdofs);

      /**
       * @brief Assemble partial assembly data for all integrators.
       * 
       * Performs partial assembly for all nonlinear form integrators, precomputing
       * and storing element-level data structures needed for efficient operator
       * application. This includes both residual and Jacobian assembly setup.
       * 
       * The assembly process:
       * 1. Iterates through all domain integrators in the nonlinear form
       * 2. Calls partial assembly setup for residual evaluation (AssemblePA)
       * 3. Calls gradient partial assembly setup for Jacobian operations (AssembleGradPA)
       * 4. Stores precomputed data in integrator-specific formats
       * 
       * Performance optimization:
       * - Element-level data precomputation amortizes setup costs
       * - Compressed storage format reduces memory requirements
       * - Device-compatible data layout for GPU execution
       * - Caliper profiling for performance analysis
       * 
       * After assembly, the operator is ready for:
       * - Multiple Mult() operations with minimal overhead
       * - Diagonal assembly for preconditioning
       * - Mixed local and global operations
       * 
       * @note Must be called before operator application methods
       * @note Assembly cost is amortized over multiple applications
       */
      virtual void Assemble() override;

      /**
       * @brief Assemble diagonal entries using partial assembly approach.
       * 
       * @param diag Output vector for diagonal entries
       * 
       * Computes diagonal entries of the Jacobian operator using the partial assembly
       * approach, providing efficient diagonal extraction without full matrix assembly.
       * The diagonal is essential for Jacobi preconditioning in iterative solvers.
       * 
       * The diagonal assembly process:
       * 1. Initializes local result vector to zero
       * 2. Calls diagonal assembly for each integrator
       * 3. Applies element restriction transpose to map to global DOFs
       * 4. Enforces essential boundary conditions with identity entries
       * 
       * Element restriction handling:
       * - Uses element restriction transpose for efficient global assembly
       * - Applies prolongation transpose for conforming finite element spaces
       * - Handles both restricted and unrestricted finite element spaces
       * 
       * Boundary condition treatment:
       * - Sets diagonal entries to 1.0 for essential DOFs
       * - Maintains positive definiteness for constrained systems
       * - Ensures preconditioner stability and effectiveness
       * 
       * @note Output vector must be properly sized for true DOF space
       * @note Essential boundary conditions receive unit diagonal entries
       * @note Caliper profiling enabled for performance monitoring
       */
      virtual void AssembleDiagonal(mfem::Vector &diag) const override;

      /**
       * @brief Template method for standard and local operator applications.
       * 
       * @tparam local_action Boolean controlling boundary condition application
       * @param x Input vector for operator application
       * @param y Output vector for result
       * 
       * Template implementation of partial assembly operator application that can
       * operate in two modes: standard (with boundary condition enforcement) and
       * local (unconstrained for specialized algorithms).
       * 
       * Operation modes:
       * - local_action=false: Standard application with essential BC enforcement
       * - local_action=true: Local application without boundary condition constraints
       * 
       * The algorithm:
       * 1. Applies essential boundary conditions to input vector (if !local_action)
       * 2. Performs element restriction to local DOF ordering
       * 3. Applies integrator gradient operations element-wise
       * 4. Maps results back to global DOF space via restriction transpose
       * 5. Enforces essential boundary conditions on output (if !local_action)
       * 
       * Memory efficiency:
       * - Reuses working vectors across operations
       * - Minimizes data movement between host and device
       * - Efficient element-wise operations with vectorized loops
       * 
       * @note Template enables compile-time optimization for different operation types
       * @note Essential boundary condition handling varies by template parameter
       * @note Caliper profiling scope for performance analysis
       */
      template<bool local_action>
      void TMult(const mfem::Vector &x, mfem::Vector &y) const;

      /**
       * @brief Standard operator application with boundary condition enforcement.
       * 
       * @param x Input vector for operator application
       * @param y Output vector for result
       * 
       * Applies the partial assembly operator with full essential boundary condition
       * enforcement, suitable for use in standard Newton-Raphson iterations and
       * linear solver applications.
       * 
       * This method calls the template implementation with local_action=false,
       * ensuring that essential boundary conditions are properly enforced in
       * both input preprocessing and output postprocessing.
       * 
       * @note Essential boundary conditions are enforced on both input and output
       * @note Suitable for standard iterative linear solver applications
       */
      virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

      /**
       * @brief Local operator application without boundary condition constraints.
       * 
       * @param x Input vector for operator application
       * @param y Output vector for result
       * 
       * Applies the partial assembly operator without essential boundary condition
       * enforcement on the output, useful for specialized algorithms that need
       * access to unconstrained operator actions.
       * 
       * This method calls the template implementation with local_action=true,
       * allowing access to the unconstrained operator for applications such as:
       * - Algebraic multigrid setup algorithms
       * - Domain decomposition methods
       * - Specialized preconditioning strategies
       * 
       * @note Input boundary conditions are still enforced for consistency
       * @note Output retains unconstrained degrees of freedom for specialized use
       */
      virtual void LocalMult(const mfem::Vector &x, mfem::Vector &y) const;

      /**
       * @brief Vector-valued operator application for multi-component problems.
       * 
       * @param x Input vector for operator application
       * @param y Output vector for result
       * 
       * Specialized operator application for vector-valued finite element problems,
       * providing optimized handling for displacement, velocity, or other vector
       * field problems common in mechanics applications.
       * 
       * The method uses the same partial assembly infrastructure but with
       * vector-specific optimizations:
       * - Component-wise element operations
       * - Efficient vector DOF mapping
       * - Optimized memory access patterns for vector data
       * 
       * This implementation is particularly effective for:
       * - Multi-dimensional mechanics problems
       * - Coupled physics applications with vector fields
       * - Problems with block structure in the finite element space
       * 
       * @note Input and output vectors must match finite element space dimensions
       * @note Essential boundary conditions applied according to component specification
       */
      virtual void MultVec(const mfem::Vector &x, mfem::Vector &y) const;
};

/**
 * @brief Element assembly implementation for nonlinear mechanics gradient operators.
 * 
 * EANonlinearMechOperatorGradExt implements an element assembly strategy that provides
 * maximum memory efficiency by storing full element matrices and applying them through
 * element-wise matrix-vector products. This approach minimizes memory usage at the cost
 * of increased computational work, making it ideal for very large problems or memory-
 * constrained environments.
 * 
 * The element assembly strategy:
 * - Stores complete element matrices for each mesh element
 * - Applies operators through explicit element matrix-vector products
 * - Provides minimal memory footprint for large-scale problems
 * - Enables fine-grained computational control and optimization
 * 
 * Key advantages:
 * - Minimal memory requirements (stores only element matrices)
 * - Predictable memory access patterns for cache optimization
 * - Natural parallelization over elements for GPU execution
 * - Exact operator representation without approximation
 * 
 * Memory characteristics:
 * - Element matrix storage: O(nelem × ndof²) memory usage
 * - No global matrix assembly or storage required
 * - Efficient for problems where nelem × ndof² < ntotal_dof²
 * - Optimal for high-order finite elements with many DOFs per element
 * 
 * The implementation extends partial assembly with:
 * - Full element matrix storage in device-compatible format
 * - Optimized element-wise matrix-vector product kernels
 * - Enhanced diagonal assembly for preconditioning efficiency
 * - Template-based operator application for performance
 * 
 * @ingroup ExaConstit_fem_operators
 */
class EANonlinearMechOperatorGradExt : public PANonlinearMechOperatorGradExt
{
   protected:
      /** @brief Number of elements in the finite element mesh */
      int NE;
      
      /** @brief Total degrees of freedom per element (ndof × ncomponents) */
      int elemDofs;
      
      /** @brief Element matrix data storage in device-compatible format */
      mfem::Vector ea_data;
      
      /** @brief Number of interior face integrators */
      int nf_int;
      
      /** @brief Number of boundary face integrators */
      int nf_bdr;
      
      /** @brief Degrees of freedom per face for face integrators */
      int faceDofs;
   public:
      /**
       * @brief Construct element assembly operator with essential boundary conditions.
       * 
       * @param _mech_operator Pointer to MFEM nonlinear form containing integrators
       * @param ess_tdofs Reference to array of essential true DOF indices
       * 
       * Initializes the element assembly operator by computing element-level dimensions,
       * allocating storage for element matrices, and preparing data structures for
       * efficient element-wise operations.
       * 
       * The constructor:
       * 1. Calls partial assembly base constructor for common setup
       * 2. Computes number of elements and DOFs per element from finite element space
       * 3. Allocates device-compatible storage for all element matrices
       * 4. Prepares element matrix data structure for subsequent assembly
       * 
       * Memory allocation:
       * - Total size: NE × elemDofs × elemDofs for dense element matrices
       * - Device-compatible allocation for GPU execution
       * - Contiguous storage for optimal memory access patterns
       * 
       * Element DOF calculation:
       * - Accounts for vector components in the finite element space
       * - Uses spatial dimension to compute total DOFs per element
       * - Ensures consistent sizing across all mesh elements
       * 
       * @note Element matrix storage allocated immediately for predictable memory usage
       * @note All elements must have the same finite element type and DOF count
       */
      EANonlinearMechOperatorGradExt(mfem::NonlinearForm *_mech_operator,
                                     const mfem::Array<int> &ess_tdofs);

      /**
       * @brief Assemble element matrices for all domain integrators.
       * 
       * Performs element assembly for all nonlinear form integrators, computing and
       * storing complete element matrices needed for operator application. This
       * includes both partial assembly setup and element matrix computation.
       * 
       * The assembly process:
       * 1. Initializes element matrix storage to zero
       * 2. Performs partial assembly setup for all integrators
       * 3. Computes full element matrices via AssembleEA calls
       * 4. Stores matrices in device-compatible contiguous format
       * 
       * Element matrix format:
       * - Dense matrices stored element-by-element
       * - Row-major ordering within each element matrix
       * - Contiguous storage for all elements for memory efficiency
       * 
       * Performance characteristics:
       * - Assembly cost higher than partial assembly due to full matrix computation
       * - Assembly cost amortized over many operator applications
       * - Element matrices enable exact operator representation
       * - Caliper profiling for performance monitoring and optimization
       * 
       * After assembly, the operator provides:
       * - Exact matrix-vector products through element matrices
       * - Efficient diagonal extraction for preconditioning
       * - Memory-efficient storage compared to global matrix assembly
       * 
       * @note Assembly must complete before operator application methods
       * @note Element matrices provide exact representation without approximation
       */
      void Assemble() override;

      /**
       * @brief Assemble diagonal entries using element assembly approach.
       * 
       * @param diag Output vector for diagonal entries
       * 
       * Computes diagonal entries by extracting diagonal elements from the stored
       * element matrices, providing exact diagonal values for preconditioning.
       * This approach gives the most accurate diagonal representation possible.
       * 
       * The diagonal extraction algorithm:
       * 1. Initializes output vector appropriately for element restriction
       * 2. Extracts diagonal entries from each element matrix
       * 3. Assembles global diagonal through element restriction transpose
       * 4. Applies essential boundary condition treatment
       * 
       * Element matrix diagonal extraction:
       * - Direct access to stored element matrix diagonal entries
       * - Vectorized operations over all elements simultaneously
       * - Device-compatible implementation for GPU execution
       * 
       * Assembly process:
       * - Element restriction transpose maps local to global DOFs
       * - Prolongation transpose handles conforming finite element spaces
       * - Essential boundary conditions enforced with unit diagonal entries
       * 
       * Advantages of element assembly approach:
       * - Exact diagonal entries (no approximation)
       * - Consistent with operator application
       * - Efficient extraction from precomputed element matrices
       * 
       * @note Provides exact diagonal entries from stored element matrices
       * @note Essential boundary conditions receive unit diagonal entries
       * @note Caliper profiling scope for performance analysis
       */
      virtual void AssembleDiagonal(mfem::Vector &diag) const override;

      /**
       * @brief Template method for element-wise operator application.
       * 
       * @tparam local_action Boolean controlling boundary condition application
       * @param x Input vector for operator application
       * @param y Output vector for result
       * 
       * Template implementation of element assembly operator application using
       * explicit element matrix-vector products. Provides both standard and
       * local operation modes for different algorithmic requirements.
       * 
       * Operation modes:
       * - local_action=false: Standard application with essential BC enforcement
       * - local_action=true: Local application without output boundary constraints
       * 
       * The element assembly algorithm:
       * 1. Applies essential boundary conditions to input (if !local_action)
       * 2. Maps input to element-local DOF ordering via restriction
       * 3. Performs element matrix-vector products for all elements
       * 4. Assembles results to global DOF space via restriction transpose
       * 5. Enforces essential boundary conditions on output (if !local_action)
       * 
       * Element matrix-vector product:
       * - Dense matrix-vector products for each element
       * - Vectorized implementation over all elements
       * - Memory access optimized for element matrix storage layout
       * - Device execution for GPU performance
       * 
       * Performance characteristics:
       * - Computational cost: O(nelem × ndof²) per application
       * - Memory access: Predictable patterns with good cache locality
       * - Parallelization: Natural element-wise parallelism
       * 
       * @note Template enables compile-time optimization for boundary condition handling
       * @note Element matrices provide exact operator application
       */
      template<bool local_action>
      void TMult(const mfem::Vector &x, mfem::Vector &y) const;

      /**
       * @brief Standard operator application with boundary condition enforcement.
       * 
       * @param x Input vector for operator application
       * @param y Output vector for result
       * 
       * Applies the element assembly operator with full essential boundary condition
       * enforcement through explicit element matrix-vector products, providing exact
       * operator application suitable for Newton-Raphson and linear solver use.
       * 
       * This method calls the template implementation with local_action=false,
       * ensuring proper boundary condition handling for standard applications.
       * 
       * @note Uses precomputed element matrices for exact operator application
       * @note Essential boundary conditions enforced on both input and output
       */
      void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

      /**
       * @brief Local operator application without output boundary constraints.
       * 
       * @param x Input vector for operator application
       * @param y Output vector for result
       * 
       * Applies the element assembly operator without essential boundary condition
       * enforcement on the output, enabling specialized algorithms that require
       * access to unconstrained operator actions.
       * 
       * This method calls the template implementation with local_action=true,
       * providing unconstrained operator application for specialized uses.
       * 
       * @note Input boundary conditions still enforced for consistency
       * @note Output preserves unconstrained DOF values for specialized algorithms
       */
      void LocalMult(const mfem::Vector &x, mfem::Vector &y) const override;

      using PANonlinearMechOperatorGradExt::MultVec;
      // void MultVec(const mfem::Vector &x, mfem::Vector &y) const;
};

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
class MechOperatorJacobiSmoother  : public mfem::Solver
{
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
      MechOperatorJacobiSmoother(const mfem::Vector &d,
                                 const mfem::Array<int> &ess_tdofs,
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
      void Mult(const mfem::Vector &x, mfem::Vector &y) const;

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
      void SetOperator(const mfem::Operator &op) { oper = &op; }

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
      void Setup(const mfem::Vector &diag);

   private:
      /** @brief Total number of degrees of freedom in the system */
      const int N;
      
      /** @brief Diagonal inverse with damping for preconditioning application */
      mfem::Vector dinv;
      
      /** @brief Damping parameter for stability and convergence control */
      const double damping;
      
      /** @brief Reference to essential true DOF indices for boundary condition handling */
      const mfem::Array<int> &ess_tdof_list;
      
      /** @brief Working vector for residual computation in iterative mode */
      mutable mfem::Vector residual;

      /** @brief Pointer to operator for iterative mode residual computation */
      const mfem::Operator *oper;
};


#endif /* mechanics_operator_hpp */
