/**
 * @file mechanics_log.hpp
 * @brief Conditional compilation macros for Caliper performance profiling.
 *
 * This header provides a unified interface for performance profiling using
 * the Caliper performance analysis toolkit. When HAVE_CALIPER is defined,
 * the macros expand to actual Caliper profiling calls. When not defined,
 * they expand to empty statements, allowing code to be compiled without
 * the Caliper dependency.
 *
 * Caliper provides low-overhead performance measurement and analysis for
 * HPC applications, enabling detailed profiling of ExaConstit simulations.
 *
 * @ingroup ExaConstit_utilities
 */
#ifndef MECHANICS_LOG
#define MECHANICS_LOG

#ifdef HAVE_CALIPER
#include "caliper/cali-mpi.h"
#include "caliper/cali.h"
/**
 * @brief Initialize Caliper for MPI applications.
 *
 * This macro initializes both the MPI-aware Caliper profiling system and
 * the standard Caliper profiling system. It should be called once at the
 * beginning of the main function, after MPI_Init().
 *
 * When HAVE_CALIPER is not defined, this expands to nothing.
 */
#define CALI_INIT    \
    cali_mpi_init(); \
    cali_init();
#else
#define CALI_INIT
/**
 * @brief Mark a C++ function for profiling (disabled when Caliper unavailable).
 *
 * When HAVE_CALIPER is defined, this macro marks the current function for
 * automatic profiling. When Caliper is not available, this expands to nothing.
 */
#define CALI_CXX_MARK_FUNCTION
/**
 * @brief Begin a named profiling region (disabled when Caliper unavailable).
 *
 * @param name String literal name for the profiling region
 *
 * When HAVE_CALIPER is defined, this macro begins a named profiling region.
 * Must be paired with CALI_MARK_END. When Caliper is not available, this
 * expands to nothing.
 *
 * Example usage:
 * @code
 * CALI_MARK_BEGIN("matrix_assembly");
 * // ... matrix assembly code ...
 * CALI_MARK_END("matrix_assembly");
 * @endcode
 */
#define CALI_MARK_BEGIN(name)
/**
 * @brief End a named profiling region (disabled when Caliper unavailable).
 *
 * @param name String literal name for the profiling region (must match CALI_MARK_BEGIN)
 *
 * When HAVE_CALIPER is defined, this macro ends a named profiling region.
 * Must be paired with CALI_MARK_BEGIN. When Caliper is not available, this
 * expands to nothing.
 */
#define CALI_MARK_END(name)
/**
 * @brief Mark a C++ scope for profiling (disabled when Caliper unavailable).
 *
 * @param name String literal name for the profiling scope
 *
 * When HAVE_CALIPER is defined, this macro marks the current scope for
 * profiling using RAII (the profiling region ends when the scope exits).
 * When Caliper is not available, this expands to nothing.
 *
 * Example usage:
 * @code
 * void myFunction() {
 *     CALI_CXX_MARK_SCOPE("myFunction");
 *     // ... function body profiled automatically ...
 * } // profiling region ends here
 * @endcode
 */
#define CALI_CXX_MARK_SCOPE(name)
/**
 * @brief Initialize Caliper (disabled when Caliper unavailable).
 *
 * When HAVE_CALIPER is not defined, this expands to nothing, allowing
 * code to compile without the Caliper dependency.
 */
#endif

#endif