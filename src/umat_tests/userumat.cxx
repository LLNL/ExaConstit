#include "userumat.h"
#include <atomic>
#include <mutex>
#include <cstring>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace exaconstit {

thread_local std::string UmatResolver::last_error_;

namespace {

// Common UMAT symbol variants to try
const std::vector<std::string> umat_symbol_variants = {
    "umat",       // No mangling
    "umat_",      // Single underscore (gfortran/flang default)
    "UMAT",       // Uppercase
    "UMAT_",      // Uppercase with underscore
    "_umat",      // Leading underscore
    "_umat_",     // Leading and trailing
    "umat__",     // Double underscore
    // Also check for common wrapper names
    "umat_call",  // Common C wrapper name
    "userumat",   // Another common name
    "userumat_",
    "USERUMAT",
    "USERUMAT_"
};
    
// Platform-specific symbol lookup in a specific library/module
void* find_symbol_in_handle(void* handle, const std::string& symbol) {
#ifdef _WIN32
    return GetProcAddress(static_cast<HMODULE>(handle), symbol.c_str());
#else
    dlerror(); // Clear any existing error
    return dlsym(handle, symbol.c_str());
#endif
}
    
// Search for UMAT function in a loaded library handle
UmatFunction find_umat_in_handle(void* handle, std::string& found_symbol) {
    for (const auto& symbol : umat_symbol_variants) {
        void* func = find_symbol_in_handle(handle, symbol);
        if (func) {
            found_symbol = symbol;
            return reinterpret_cast<UmatFunction>(func);
        }
    }
    return nullptr;
}
    
// Get handle to main executable for built-in symbols
void* get_main_executable_handle() {
#ifdef _WIN32
    return GetModuleHandle(nullptr);
#else
    return RTLD_DEFAULT; // Special handle that searches all loaded symbols
#endif
}
    
// Load a library and get its handle
void* load_library(const std::string& path) {
#ifdef _WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}
    
// Unload a library
void unload_library(void* handle) {
    if (!handle) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}
    
// Get platform-specific error message
std::string get_load_error() {
#ifdef _WIN32
    DWORD error = GetLastError();
    if (error == 0) return "No error";
    
    char error_buf[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                    NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    error_buf, sizeof(error_buf), NULL);
    return std::string(error_buf);
#else
    const char* error = dlerror();
    return error ? std::string(error) : "No error";
#endif
}
    
// Cache for built-in UMAT resolution
struct BuiltInUmatCache {
    std::atomic<UmatFunction> resolved_func{nullptr};
    std::once_flag init_flag;
    std::string found_symbol;
    
    UmatFunction resolve() {
        std::call_once(init_flag, [this]() {
            void* handle = get_main_executable_handle();
            resolved_func = find_umat_in_handle(handle, found_symbol);
        });
        
        return resolved_func.load();
    }
};

static BuiltInUmatCache built_in_cache;
}

UmatFunction
UmatResolver::GetUmat(const std::string& library_path,
                      const std::string& function_name) {
    std::string found_symbol;
    
    // Case 1: Built-in UMAT (empty library path)
    if (library_path.empty()) {
        UmatFunction func = built_in_cache.resolve();
        if (func) {
            last_error_ = "Found built-in UMAT as symbol: " + built_in_cache.found_symbol;
        } else {
            last_error_ = "No built-in UMAT found. Searched symbols: ";
            for (const auto& sym : umat_symbol_variants) {
                last_error_ += sym + " ";
            }
        }
        return func;
    }
    
    // Case 2: Dynamic loading with automatic symbol resolution
    void* handle = load_library(library_path);
    if (!handle) {
        last_error_ = "Failed to load library '" + library_path + "': " + get_load_error();
        return nullptr;
    }
    
    UmatFunction func = nullptr;
    
    // If function_name is provided and not "auto", try it first
    if (!function_name.empty() && function_name != "auto") {
        void* sym = find_symbol_in_handle(handle, function_name);
        if (sym) {
            func = reinterpret_cast<UmatFunction>(sym);
            found_symbol = function_name;
        }
    }
    
    // If not found, or if "auto" was specified, search common variants
    if (!func) {
        func = find_umat_in_handle(handle, found_symbol);
    }
    
    if (func) {
        last_error_ = "Successfully loaded UMAT from '" + library_path + 
                     "' using symbol: " + found_symbol;
        // Note: We're NOT unloading here - DynamicUmatLoader handles lifetime
    } else {
        last_error_ = "No UMAT symbol found in '" + library_path + 
                     "'. Searched: " + function_name + " ";
        for (const auto& sym : umat_symbol_variants) {
            last_error_ += sym + " ";
        }
        unload_library(handle); // Clean up on failure
    }
    
    return func;
}

std::string UmatResolver::GetLastError() {
    return last_error_;
}

bool UmatResolver::ValidateLibrary(const std::string& library_path,
                                   const std::string& function_name) {
    // Load library temporarily just for validation
    void* handle = load_library(library_path);
    if (!handle) {
        last_error_ = "Failed to load library for validation: " + get_load_error();
        return false;
    }
    
    std::string found_symbol;
    bool valid = false;
    
    // Check specific function name if provided
    if (!function_name.empty() && function_name != "auto") {
        void* sym = find_symbol_in_handle(handle, function_name);
        if (sym) {
            valid = true;
            found_symbol = function_name;
        }
    }
    
    // Otherwise search for any UMAT symbol
    if (!valid) {
        UmatFunction func = find_umat_in_handle(handle, found_symbol);
        valid = (func != nullptr);
    }
    
    if (valid) {
        last_error_ = "Library is valid. Found UMAT symbol: " + found_symbol;
    } else {
        last_error_ = "No valid UMAT symbol found in library";
    }
    
    // Clean up - just for validation
    unload_library(handle);
    
    return valid;
}

} // namespace ExaConstit