#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

#ifdef _WIN32
    #include <windows.h>
    using LibraryHandle = HMODULE;
#else
    #include <dlfcn.h>
    using LibraryHandle = void*;
#endif

// Forward declare the UMAT function signature
using UmatFunction = void(*)(
        double *stress, double *statev, double *ddsdde,
        double *sse, double *spd, double *scd, double *rpl,
        double *ddsdt, double *drplde, double *drpldt,
        double *stran, double *dstran, double *time,
        double *deltaTime, double *tempk, double *dtemp, double *predef,
        double *dpred, double *cmname, int *ndi, int *nshr, int *ntens,
        int *nstatv, double *props, int *nprops, double *coords,
        double *drot, double *pnewdt, double *celent,
        double *dfgrd0, double *dfgrd1, int *noel, int *npt,
        int *layer, int *kspt, int *kstep, int *kinc
    );

/**
 * @brief Manages dynamic loading and unloading of UMAT shared libraries
 * 
 * This class provides thread-safe loading of UMAT shared libraries with
 * support for multiple concurrent UMATs across different regions.
 */
class DynamicUmatLoader {
public:
    /**
     * @brief Library management strategy
     */
    enum class LoadStrategy {
        LOAD_ON_SETUP,     ///< Load during ModelSetup, unload after
        PERSISTENT,        ///< Load once and keep loaded for simulation lifetime
        LAZY_LOAD         ///< Load on first use, keep until explicitly unloaded
    };

    /**
     * @brief Information about a loaded UMAT library
     */
    struct UmatLibraryInfo {
        std::string library_path;
        LibraryHandle handle;
        UmatFunction umat_function;
        LoadStrategy strategy;
        int reference_count;
        bool is_loaded;
        
        UmatLibraryInfo() : handle(nullptr), umat_function(nullptr), 
                           strategy(LoadStrategy::PERSISTENT), 
                           reference_count(0), is_loaded(false) {}
    };

private:
    static std::unordered_map<std::string, std::unique_ptr<UmatLibraryInfo>> loaded_libraries_;
    static std::mutex library_mutex_;

public:
    /**
     * @brief Load a UMAT shared library
     * 
     * @param library_path Path to the shared library (.so, .dll, .dylib)
     * @param strategy Loading strategy to use
     * @return Pointer to UMAT function if successful, nullptr otherwise
     */
    static UmatFunction LoadUmat(const std::string& library_path, 
                                LoadStrategy strategy = LoadStrategy::PERSISTENT);

    /**
     * @brief Unload a UMAT shared library
     * 
     * @param library_path Path to the shared library to unload
     * @return true if unloaded successfully, false otherwise
     */
    static bool UnloadUmat(const std::string& library_path);

    /**
     * @brief Get loaded UMAT function without loading
     * 
     * @param library_path Path to the shared library
     * @return Pointer to UMAT function if already loaded, nullptr otherwise
     */
    static UmatFunction GetUmat(const std::string& library_path);

    /**
     * @brief Check if a UMAT library is currently loaded
     * 
     * @param library_path Path to the shared library
     * @return true if loaded, false otherwise
     */
    static bool IsLoaded(const std::string& library_path);

    /**
     * @brief Unload all loaded UMAT libraries
     */
    static void UnloadAll();

    /**
     * @brief Get list of currently loaded libraries
     * 
     * @return Vector of library paths
     */
    static std::vector<std::string> GetLoadedLibraries();

private:
    /**
     * @brief Platform-specific library loading
     */
    static LibraryHandle LoadLibrary(const std::string& path);
    
    /**
     * @brief Platform-specific function symbol loading
     */
    static UmatFunction GetUmatSymbol(LibraryHandle handle);
    
    /**
     * @brief Platform-specific library unloading
     */
    static bool UnloadLibrary(LibraryHandle handle);
    
    /**
     * @brief Get platform-specific error message
     */
    static std::string GetLastError();
};