#pragma once

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace exaconstit {

/**
 * @brief Platform-specific library handle abstraction
 */
class LibraryHandle {
public:
    LibraryHandle() = default;
    explicit LibraryHandle(void* handle_) : handle(handle_) {}

    // Move-only semantics
    LibraryHandle(const LibraryHandle&) = delete;
    LibraryHandle& operator=(const LibraryHandle&) = delete;
    LibraryHandle(LibraryHandle&& other) noexcept : handle(std::exchange(other.handle, nullptr)) {}
    LibraryHandle& operator=(LibraryHandle&& other) noexcept {
        if (this != &other) {
            unload();
            handle = std::exchange(other.handle, nullptr);
        }
        return *this;
    }

    ~LibraryHandle() {
        unload();
    }

    void* get() const {
        return handle;
    }
    explicit operator bool() const {
        return handle != nullptr;
    }

    void* release() {
        return std::exchange(handle, nullptr);
    }

private:
    void unload() {
        if (handle) {
#ifdef _WIN32
            ::FreeLibrary(static_cast<HMODULE>(handle));
#else
            ::dlclose(handle);
#endif
            handle = nullptr;
        }
    }

    void* handle = nullptr;
};

/**
 * @brief Loading strategies for dynamic libraries
 */
enum class LoadStrategy {
    PERSISTENT,    ///< Keep loaded for entire application lifetime
    LOAD_ON_SETUP, ///< Load during setup, unload after each use
    LAZY_LOAD      ///< Load on first use, unload when refs drop to zero
};

/**
 * @brief Symbol resolution configuration
 */
struct SymbolConfig {
    std::vector<std::string> search_names; ///< Symbol names to search for
    bool enable_fortran_mangling = true;   ///< Generate Fortran name variants
    bool enable_builtin_search = true;     ///< Search in main executable
    bool case_sensitive = true;            ///< Case-sensitive symbol search
};

/**
 * @brief Information about a loaded function
 */
template <typename FuncType>
struct LoadedFunction {
    std::string library_path;
    std::string resolved_symbol;
    FuncType function = nullptr; // Changed from FuncType* to FuncType
    LoadStrategy strategy = LoadStrategy::PERSISTENT;
    std::atomic<int> reference_count{0};

    // Default constructor
    LoadedFunction() = default;

    // Move constructor (needed because std::atomic is not moveable)
    LoadedFunction(LoadedFunction&& other) noexcept
        : library_path(std::move(other.library_path)),
          resolved_symbol(std::move(other.resolved_symbol)), function(other.function),
          strategy(other.strategy), reference_count(other.reference_count.load()) {
        other.function = nullptr;
    }

    // Deleted copy constructor and assignment
    LoadedFunction(const LoadedFunction&) = delete;
    LoadedFunction& operator=(const LoadedFunction&) = delete;
    LoadedFunction& operator=(LoadedFunction&&) = delete;
};

/**
 * @brief Generic dynamic function loader with symbol resolution
 *
 * @tparam FuncType Function pointer type (e.g., UmatFunction)
 */
template <typename FuncType>
class DynamicFunctionLoader {
    static_assert(std::is_function_v<std::remove_pointer_t<FuncType>>,
                  "FuncType must be a function pointer type");

public:
    /**
     * @brief Result type for load operations
     */
    struct LoadResult {
        FuncType function = nullptr; // Changed from FuncType* to FuncType
        std::string resolved_symbol;
        std::string error_message;
        bool success = false;

        operator bool() const {
            return success;
        }
    };

    /**
     * @brief Load a function from a library
     *
     * @param library_path Path to the library (empty for built-in search)
     * @param config Symbol resolution configuration
     * @param strategy Loading strategy
     * @return LoadResult containing function pointer and status
     */
    static LoadResult load(const std::string& library_path,
                           const SymbolConfig& config,
                           LoadStrategy strategy = LoadStrategy::PERSISTENT) {
        std::lock_guard<std::mutex> lock(mutex_lock);

        // Check cache first
        auto cache_key = make_cache_key(library_path, config);
        auto it = loaded_libraries.find(cache_key);
        if (it != loaded_libraries.end()) {
            it->second.reference_count++;
            return {it->second.function, it->second.resolved_symbol, "", true};
        }

        // Perform the load
        LoadResult result;

        if (library_path.empty() && config.enable_builtin_search) {
            result = load_builtin(config);
        } else if (!library_path.empty()) {
            result = load_from_library(library_path, config);
        } else {
            result.error_message = "No library path specified and built-in search disabled";
        }

        // Cache successful loads
        if (result.success) {
            LoadedFunction<FuncType> info;
            info.library_path = library_path;
            info.resolved_symbol = result.resolved_symbol;
            info.function = result.function;
            info.strategy = strategy;
            info.reference_count = 1;

            loaded_libraries.emplace(cache_key, std::move(info));
        }

        return result;
    }

    /**
     * @brief Unload a previously loaded function
     */
    static bool unload(const std::string& library_path, const SymbolConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_lock);

        auto cache_key = make_cache_key(library_path, config);
        auto it = loaded_libraries.find(cache_key);
        if (it == loaded_libraries.end()) {
            return false;
        }

        if (--it->second.reference_count <= 0) {
            if (it->second.strategy != LoadStrategy::PERSISTENT) {
                auto lib_it = library_handles.find(library_path);
                if (lib_it != library_handles.end()) {
                    library_handles.erase(lib_it);
                }
                loaded_libraries.erase(it);
            }
        }

        return true;
    }

    /**
     * @brief Check if a library provides a valid function
     */
    static bool validate(const std::string& library_path, const SymbolConfig& config) {
        auto result = load(library_path, config, LoadStrategy::LAZY_LOAD);
        if (result.success) {
            unload(library_path, config);
        }
        return result.success;
    }

    /**
     * @brief Get the last error message for the current thread
     */
    static std::string get_last_error() {
        return last_error;
    }

    /**
     * @brief Clear all cached libraries and force unload
     */
    static void clear_cache() {
        std::lock_guard<std::mutex> lock(mutex_lock);
        loaded_libraries.clear();
        library_handles.clear();
    }

private:
    // Static members
    static std::unordered_map<std::string, LoadedFunction<FuncType>> loaded_libraries;
    static std::unordered_map<std::string, LibraryHandle> library_handles;
    static std::mutex mutex_lock;
    static thread_local std::string last_error;

    /**
     * @brief Generate all possible symbol variants based on config
     */
    static std::vector<std::string> generate_symbol_variants(const SymbolConfig& config) {
        std::vector<std::string> variants;

        for (const auto& base_name : config.search_names) {
            // Original name
            variants.push_back(base_name);

            if (config.enable_fortran_mangling) {
                // Common Fortran manglings
                variants.push_back(base_name + "_");       // gfortran/flang default
                variants.push_back(base_name + "__");      // g77 with underscores
                variants.push_back("_" + base_name);       // Leading underscore
                variants.push_back("_" + base_name + "_"); // Both

                if (!config.case_sensitive) {
                    // Uppercase variants
                    std::string upper = base_name;
                    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
                    variants.push_back(upper);
                    variants.push_back(upper + "_");

                    // Lowercase variants
                    std::string lower = base_name;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    if (lower != base_name) {
                        variants.push_back(lower);
                        variants.push_back(lower + "_");
                    }
                }
            }
        }

        // Remove duplicates while preserving order
        std::vector<std::string> unique_variants;
        std::unordered_set<std::string> seen;
        for (const auto& variant : variants) {
            if (seen.insert(variant).second) {
                unique_variants.push_back(variant);
            }
        }
        variants = std::move(unique_variants);

        return variants;
    }

    /**
     * @brief Try to find a symbol in a handle
     */
    static void* find_symbol(void* handle, const std::string& symbol) {
#ifdef _WIN32
        return ::GetProcAddress(static_cast<HMODULE>(handle), symbol.c_str());
#else
        // Clear any existing error
        ::dlerror();
        return ::dlsym(handle, symbol.c_str());
#endif
    }

    /**
     * @brief Load from the main executable (built-in)
     */
    static LoadResult load_builtin(const SymbolConfig& config) {
        LoadResult result;

#ifdef _WIN32
        void* handle = ::GetModuleHandle(nullptr);
#else
        void* handle = RTLD_DEFAULT;
#endif

        auto variants = generate_symbol_variants(config);
        for (const auto& symbol : variants) {
            if (void* func = find_symbol(handle, symbol)) {
                result.function = reinterpret_cast<FuncType>(func);
                result.resolved_symbol = symbol;
                result.success = true;
                last_error = "Found built-in function: " + symbol;
                return result;
            }
        }

        result.error_message = "No built-in symbol found. Searched: ";
        for (const auto& sym : variants) {
            result.error_message += sym + " ";
        }
        last_error = result.error_message;
        return result;
    }

    /**
     * @brief Load from a specific library file
     */
    static LoadResult load_from_library(const std::string& library_path,
                                        const SymbolConfig& config) {
        LoadResult result;

        // Get or create library handle
        auto& handle = library_handles[library_path];
        if (!handle) {
#ifdef _WIN32
            void* h = ::LoadLibraryA(library_path.c_str());
            if (!h) {
                DWORD error = ::GetLastError();
                char error_buf[256];
                ::FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL,
                                 error,
                                 MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                 error_buf,
                                 sizeof(error_buf),
                                 NULL);
                result.error_message = "Failed to load library: " + std::string(error_buf);
            }
#else
            void* h = ::dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!h) {
                const char* error = ::dlerror();
                result.error_message = "Failed to load library: " +
                                       std::string(error ? error : "Unknown error");
            }
#endif
            if (!h) {
                last_error = result.error_message;
                return result;
            }
            handle = LibraryHandle(h);
        }

        // Search for symbols
        auto variants = generate_symbol_variants(config);
        for (const auto& symbol : variants) {
            if (void* func = find_symbol(handle.get(), symbol)) {
                result.function = reinterpret_cast<FuncType>(func);
                result.resolved_symbol = symbol;
                result.success = true;
                last_error = "Found function '" + symbol + "' in library: " + library_path;
                return result;
            }
        }

        // Symbol not found
        result.error_message = "No symbol found in '" + library_path + "'. Searched: ";
        for (const auto& sym : variants) {
            result.error_message += sym + " ";
        }
        last_error = result.error_message;

        // Remove handle if we couldn't find the symbol
        library_handles.erase(library_path);

        return result;
    }

    /**
     * @brief Create a cache key from library path and config
     */
    static std::string make_cache_key(const std::string& library_path, const SymbolConfig& config) {
        std::string key = library_path + "|";
        for (const auto& name : config.search_names) {
            key += name + ",";
        }
        return key;
    }
};

// Static member definitions
template <typename FuncType>
std::unordered_map<std::string, LoadedFunction<FuncType>>
    DynamicFunctionLoader<FuncType>::loaded_libraries;

template <typename FuncType>
std::unordered_map<std::string, LibraryHandle> DynamicFunctionLoader<FuncType>::library_handles;

template <typename FuncType>
std::mutex DynamicFunctionLoader<FuncType>::mutex_lock;

template <typename FuncType>
thread_local std::string DynamicFunctionLoader<FuncType>::last_error;

} // namespace exaconstit