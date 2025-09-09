#pragma once

#include "umats/userumat.h"

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>

/**
 * @brief Platform-specific library handle type.
 * 
 * On Windows platforms, this is an HMODULE. On Unix-like platforms (Linux/macOS),
 * this is a void pointer used by dlopen/dlsym/dlclose functions.
 */
#ifdef _WIN32
    #include <windows.h>
    using LibraryHandle = HMODULE;
#else
    #include <dlfcn.h>
    using LibraryHandle = void*;
#endif

/**
 * @brief Manages dynamic loading and unloading of UMAT shared libraries.
 * 
 * This class provides thread-safe loading of UMAT shared libraries with
 * support for multiple concurrent UMATs across different regions in
 * multi-material finite element simulations. It implements various loading
 * strategies to optimize performance and memory usage.
 * 
 * The class handles platform-specific differences between Windows (.dll),
 * Linux (.so), and macOS (.dylib) shared libraries, providing a unified
 * interface for UMAT library management.
 * 
 * Key features:
 * - Thread-safe library loading and unloading using mutex protection
 * - Multiple loading strategies (persistent, load-on-setup, lazy loading)
 * - Automatic symbol resolution with fallback to common UMAT symbol names
 * - Reference counting for safe shared library management
 * - Platform-specific error handling and reporting
 * 
 * @ingroup ExaConstit_utilities
 */
class DynamicUmatLoader {
public:
     /**
     * @brief Library management strategy for controlling when libraries are loaded/unloaded.
     * 
     * Different strategies provide trade-offs between memory usage, performance,
     * and library management complexity based on simulation requirements.
     */
    enum class LoadStrategy {
        /**
         * @brief Load during ModelSetup, unload after each step.
         * 
         * This strategy minimizes memory usage by loading the library only
         * when needed and immediately unloading it after use. Best for
         * simulations with limited memory or infrequent UMAT calls.
         */
        LOAD_ON_SETUP,
        
        /**
         * @brief Load once and keep loaded for entire simulation lifetime.
         * 
         * This strategy maximizes performance by avoiding repeated loading/unloading
         * overhead. The library remains in memory throughout the simulation.
         * Best for simulations with frequent UMAT calls.
         */
        PERSISTENT,
        
        /**
         * @brief Load on first use, keep until explicitly unloaded.
         * 
         * This strategy provides a balance between memory usage and performance.
         * Libraries are loaded when first needed and can be explicitly unloaded
         * when no longer required. Best for simulations with varying UMAT usage.
         */
        LAZY_LOAD
    };

    /**
     * @brief Information about a loaded UMAT library.
     * 
     * This structure contains all the metadata and handles associated with
     * a dynamically loaded UMAT library, including the library path,
     * platform-specific handle, function pointer, loading strategy,
     * and reference counting information.
     */
    struct UmatLibraryInfo {
        /**
         * @brief Path to the shared library file.
         * 
         * Full path to the UMAT shared library (.so, .dll, .dylib).
         * Used as the unique identifier for the library in the cache.
         */
        std::string library_path;

        /**
         * @brief Found function symbol that we're using.
         * 
         * Used to identify which function was found and loaded up from this library.
         */
        std::string found_symbol; 
        
        /**
         * @brief Platform-specific handle to the loaded library.
         * 
         * On Windows, this is an HMODULE. On Unix-like systems, this is
         * a void pointer returned by dlopen().
         */
        LibraryHandle handle;
        
        /**
         * @brief Pointer to the resolved UMAT function.
         * 
         * Function pointer to the UMAT subroutine entry point within
         * the loaded shared library. nullptr if symbol resolution failed.
         */
        UmatFunction umat_function;
        
        /**
         * @brief Loading strategy used for this library.
         * 
         * Determines when the library should be loaded and unloaded
         * based on the specified LoadStrategy.
         */
        LoadStrategy strategy;
        
        /**
         * @brief Reference count for shared library usage.
         * 
         * Tracks how many objects are currently using this library.
         * Used to determine when it's safe to unload the library.
         */
        int reference_count;
        
        /**
         * @brief Flag indicating if the library is currently loaded.
         * 
         * True if the library handle is valid and the function pointer
         * has been successfully resolved.
         */
        bool is_loaded;
        
        /**
         * @brief Default constructor initializing all members to safe defaults.
         */
        UmatLibraryInfo() : handle(nullptr), umat_function(nullptr), 
                           strategy(LoadStrategy::PERSISTENT), 
                           reference_count(0), is_loaded(false) {}
    };

private:
    /**
     * @brief Static cache of loaded UMAT libraries.
     * 
     * Maps library paths to their corresponding UmatLibraryInfo structures.
     * This cache enables sharing of loaded libraries across multiple material
     * models and regions, reducing memory usage and loading overhead.
     * 
     * The key is the library path string, which serves as a unique identifier.
     * The value is a unique_ptr to UmatLibraryInfo for automatic memory management.
     */
    static std::unordered_map<std::string, std::unique_ptr<UmatLibraryInfo>> loaded_libraries_;
    /**
     * @brief Mutex for thread-safe access to the library cache.
     * 
     * Protects the loaded_libraries_ map from concurrent access in multi-threaded
     * environments. All public methods acquire this mutex before modifying
     * the library cache.
     */
    static std::mutex library_mutex_;

public:
    /**
     * @brief Load a UMAT shared library and return the function pointer.
     * 
     * @param library_path Path to the shared library (.so, .dll, .dylib)
     * @param strategy Loading strategy to use for this library
     * @param function_name User-supplied function name
     * @return Pointer to UMAT function if successful, nullptr otherwise
     * 
     * This method handles the complete process of loading a UMAT shared library:
     * 1. Checks if the library is already loaded (reference counting)
     * 2. Loads the library using platform-specific mechanisms
     * 3. Resolves the UMAT function symbol with fallback names
     * 4. Stores the library information in the cache
     * 5. Returns the function pointer for immediate use
     * 
     * The method is thread-safe and supports multiple loading strategies.
     * If the library is already loaded, it increments the reference count
     * and returns the existing function pointer.
     * 
     * @note This method may print error messages to std::cerr if loading fails
     */
    static UmatFunction LoadUmat(const std::string& library_path, 
                                LoadStrategy strategy = LoadStrategy::PERSISTENT,
                                const std::string& function_name = "umat_call");

    /**
     * @brief Unload a UMAT shared library and free its resources.
     * 
     * @param library_path Path to the shared library to unload
     * @return true if unloaded successfully, false otherwise
     * 
     * This method decrements the reference count for the specified library
     * and unloads it if the reference count reaches zero. The unloading
     * process includes:
     * 1. Checking if the library is currently loaded
     * 2. Decrementing the reference count
     * 3. Unloading the library if reference count reaches zero
     * 4. Removing the library from the cache
     * 5. Releasing platform-specific resources
     * 
     * The method is thread-safe and respects reference counting to ensure
     * libraries are not unloaded while still in use.
     */
    static bool UnloadUmat(const std::string& library_path);

    /**
     * @brief Get loaded UMAT function pointer without loading.
     * 
     * @param library_path Path to the shared library
     * @return Pointer to UMAT function if already loaded, nullptr otherwise
     * 
     * This method provides access to an already-loaded UMAT function without
     * attempting to load the library. It's useful for checking if a library
     * is available or for accessing functions when loading is handled elsewhere.
     * 
     * The method is thread-safe and does not modify the library cache or
     * reference counts.
     */
    static UmatFunction GetUmat(const std::string& library_path);

    /**
     * @brief Check if a UMAT library is currently loaded.
     * 
     * @param library_path Path to the shared library
     * @return true if loaded, false otherwise
     * 
     * This method queries the library cache to determine if the specified
     * library is currently loaded and available. It's useful for conditional
     * loading logic and debugging.
     * 
     * The method is thread-safe and does not modify the library cache.
     */
    static bool IsLoaded(const std::string& library_path);

    /**
     * @brief Unload all loaded UMAT libraries and free resources.
     * 
     * This method forcibly unloads all libraries in the cache, regardless
     * of reference counts. It's typically called during program shutdown
     * or when a complete reset of the library cache is needed.
     * 
     * The unloading process includes:
     * 1. Iterating through all cached libraries
     * 2. Unloading each library using platform-specific mechanisms
     * 3. Clearing the entire cache
     * 4. Releasing all associated memory
     * 
     * The method is thread-safe and provides a clean shutdown mechanism.
     * 
     * @warning This method ignores reference counts and forcibly unloads
     *          all libraries, which may cause issues if libraries are still in use.
     */
    static void UnloadAll();

    /**
     * @brief Get list of currently loaded library paths.
     * 
     * @return Vector of library paths for all currently loaded libraries
     * 
     * This method returns a list of all library paths that are currently
     * loaded and available in the cache. It's useful for debugging,
     * monitoring, and providing status information about loaded libraries.
     * 
     * The method is thread-safe and returns a copy of the library paths
     * to avoid issues with concurrent modifications to the cache.
     */
    static std::vector<std::string> GetLoadedLibraries();

private:
    /**
     * @brief Platform-specific library loading implementation.
     * 
     * @param path Path to the shared library to load
     * @return Platform-specific library handle, or nullptr on failure
     * 
     * This method encapsulates platform-specific library loading:
     * - Windows: Uses LoadLibraryA() to load .dll files
     * - Unix/Linux: Uses dlopen() with RTLD_LAZY | RTLD_LOCAL flags
     * 
     * The method handles platform differences transparently and provides
     * a unified interface for library loading across operating systems.
     */
    static LibraryHandle LoadLibrary(const std::string& path);
    
    /**
     * @brief Platform-specific function symbol resolution.
     * 
     * @param handle Valid library handle from LoadLibrary()
     * @param function_name User-supplied function name
     * @return Pointer to UMAT function, or nullptr if symbol not found
     * 
     * This method attempts to resolve the UMAT function symbol from the
     * loaded library using user supplied function name and multiple common symbol names:
     * - "umat_call" (primary symbol name)
     * - "umat" (alternative symbol name)
     * - "umat_" (Fortran-style symbol name with trailing underscore)
     * 
     * The method handles platform-specific symbol resolution:
     * - Windows: Uses GetProcAddress()
     * - Unix/Linux: Uses dlsym() with error checking
     * 
     * This approach provides compatibility with various UMAT implementations
     * and compiler conventions.
     */
    static UmatFunction GetUmatSymbol(LibraryHandle handle, 
                                     const std::string& requested_function);
    
    /**
     * @brief Platform-specific library unloading implementation.
     * 
     * @param handle Valid library handle to unload
     * @return true if unloaded successfully, false otherwise
     * 
     * This method encapsulates platform-specific library unloading:
     * - Windows: Uses FreeLibrary()
     * - Unix/Linux: Uses dlclose()
     * 
     * The method properly releases platform-specific resources and
     * provides error checking for the unloading operation.
     */
    static bool UnloadLibrary(LibraryHandle handle);
    
    /**
     * @brief Get platform-specific error message for library operations.
     * 
     * @return String describing the last library operation error
     * 
     * This method retrieves detailed error information from platform-specific
     * library loading systems:
     * - Windows: Uses GetLastError() and FormatMessage() for detailed error strings
     * - Unix/Linux: Uses dlerror() to get dlopen/dlsym error messages
     * 
     * The method provides consistent error reporting across platforms,
     * enabling better debugging and error handling in library operations.
     */
    static std::string GetLastError();
};