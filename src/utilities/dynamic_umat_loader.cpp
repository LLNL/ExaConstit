#include "dynamic_umat_loader.hpp"
#include "unified_logger.hpp"

#include <iostream>
#include <filesystem>

// Static member definitions
std::unordered_map<std::string, std::unique_ptr<DynamicUmatLoader::UmatLibraryInfo>> 
    DynamicUmatLoader::loaded_libraries_;
std::mutex DynamicUmatLoader::library_mutex_;

// Implementation

UmatFunction DynamicUmatLoader::LoadUmat(const std::string& library_path, LoadStrategy strategy, const std::string& function_name) {
    std::lock_guard<std::mutex> lock(library_mutex_);
    
    // Check if already loaded
    auto it = loaded_libraries_.find(library_path);
    if (it != loaded_libraries_.end() && it->second->is_loaded) {
        it->second->reference_count++;
        
        // Warn if requesting different function name
        if (it->second->found_symbol != function_name) {
            std::cerr << "Warning: Library already loaded with symbol '" 
                     << it->second->found_symbol << "', ignoring request for '" 
                     << function_name << "'" << std::endl;
        }
        
        return it->second->umat_function;
    }
    
    // Load new library
    LibraryHandle handle = LoadLibrary(library_path);
    if (!handle) {
        std::cerr << "Failed to load UMAT library: " << library_path 
                  << "\nError: " << GetLastError() << std::endl;
        return nullptr;
    }
    
    // Get UMAT function symbol
    UmatFunction umat_func = GetUmatSymbol(handle, function_name);
    if (!umat_func) {
        std::cerr << "Failed to find 'umat_call' symbol in library: " << library_path 
                  << "\nError: " << GetLastError() << std::endl;
        UnloadLibrary(handle);
        return nullptr;
    }
    
    // Store library info
    auto lib_info = std::make_unique<UmatLibraryInfo>();
    lib_info->library_path = library_path;
    lib_info->handle = handle;
    lib_info->umat_function = umat_func;
    lib_info->strategy = strategy;
    lib_info->reference_count = 1;
    lib_info->is_loaded = true;
    lib_info->found_symbol = function_name;
    
    UmatFunction result = umat_func;
    loaded_libraries_[library_path] = std::move(lib_info);
    
    return result;
}

bool DynamicUmatLoader::UnloadUmat(const std::string& library_path) {
    std::lock_guard<std::mutex> lock(library_mutex_);
    
    auto it = loaded_libraries_.find(library_path);
    if (it == loaded_libraries_.end() || !it->second->is_loaded) {
        return false;
    }
    
    it->second->reference_count--;
    
    // Only unload if no more references (except for PERSISTENT strategy)
    if (it->second->reference_count <= 0 && it->second->strategy != LoadStrategy::PERSISTENT) {
        bool success = UnloadLibrary(it->second->handle);
        if (success) {
            loaded_libraries_.erase(it);
        }
        return success;
    }
    
    return true;
}

UmatFunction DynamicUmatLoader::GetUmat(const std::string& library_path) {
    std::lock_guard<std::mutex> lock(library_mutex_);
    
    auto it = loaded_libraries_.find(library_path);
    if (it != loaded_libraries_.end() && it->second->is_loaded) {
        return it->second->umat_function;
    }
    return nullptr;
}

bool DynamicUmatLoader::IsLoaded(const std::string& library_path) {
    std::lock_guard<std::mutex> lock(library_mutex_);
    
    auto it = loaded_libraries_.find(library_path);
    return (it != loaded_libraries_.end() && it->second->is_loaded);
}

void DynamicUmatLoader::UnloadAll() {
    std::lock_guard<std::mutex> lock(library_mutex_);
    
    for (auto& pair : loaded_libraries_) {
        if (pair.second->is_loaded) {
            UnloadLibrary(pair.second->handle);
        }
    }
    loaded_libraries_.clear();
    std::cout << "Unloaded all UMAT libraries" << std::endl;
}

std::vector<std::string> DynamicUmatLoader::GetLoadedLibraries() {
    std::lock_guard<std::mutex> lock(library_mutex_);
    
    std::vector<std::string> loaded_paths;
    for (const auto& pair : loaded_libraries_) {
        if (pair.second->is_loaded) {
            loaded_paths.push_back(pair.first);
        }
    }
    return loaded_paths;
}

// Platform-specific implementations
#ifdef _WIN32
LibraryHandle DynamicUmatLoader::LoadLibrary(const std::string& path) {
    return ::LoadLibraryA(path.c_str());
}

UmatFunction DynamicUmatLoader::GetUmatSymbol(LibraryHandle handle, const std::string& function_name) {
    return reinterpret_cast<UmatFunction>(::GetProcAddress(handle, function_name.c_str()));
}

bool DynamicUmatLoader::UnloadLibrary(LibraryHandle handle) {
    return ::FreeLibrary(handle) != 0;
}

std::string DynamicUmatLoader::GetLastError() {
    DWORD error = ::GetLastError();
    if (error == 0) return "No error";
    
    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), 
        (LPSTR)&messageBuffer, 0, NULL);
    
    std::string message(messageBuffer, size);
    LocalFree(messageBuffer);
    return message;
}

#else // Unix/Linux/macOS
LibraryHandle DynamicUmatLoader::LoadLibrary(const std::string& path) {
    return dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
}

UmatFunction DynamicUmatLoader::GetUmatSymbol(LibraryHandle handle, 
                                              const std::string& requested_function) {

    // Helper to generate variants of a base name
    auto generate_variants = [](const std::string& base) -> std::vector<std::string> {
        std::vector<std::string> variants;
        
        // Original name
        variants.push_back(base);
        
        // Common Fortran manglings
        variants.push_back(base + "_");      // gfortran/flang default
        variants.push_back(base + "__");     // g77 with underscores in name
        
        // Uppercase variants
        std::string upper = base;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        variants.push_back(upper);
        variants.push_back(upper + "_");
        
        // Leading underscore variants
        variants.push_back("_" + base);
        variants.push_back("_" + base + "_");
        
        return variants;
    };

#ifdef _WIN32
    auto try_symbol = [&](const std::string& symbol) -> void* {
        return ::GetProcAddress(handle, symbol.c_str());
    };
#else
    dlerror(); // Clear any existing error
    
    // For debugging: show what library we're actually searching
    if (std::getenv("EXACONSTIT_DEBUG_UMAT")) {
        Dl_info handle_info;
        // Get any symbol from the library to find its path
        void* any_sym = dlsym(handle, "_DYNAMIC"); // Common symbol in shared libraries
        if (any_sym && dladdr(any_sym, &handle_info)) {
            std::cout << "Searching for symbols in library: " 
                        << (handle_info.dli_fname ? handle_info.dli_fname : "unknown") << std::endl;
        }
    }
    
    auto try_symbol = [&](const std::string& symbol) -> void* {
        void* sym = dlsym(handle, symbol.c_str());

        // Debug: show where symbol was found
        if (sym && std::getenv("EXACONSTIT_DEBUG_UMAT")) {
            Dl_info info;
            if (dladdr(sym, &info)) {
                std::cout << "  Symbol '" << symbol << "' found in: " 
                            << (info.dli_fname ? info.dli_fname : "unknown") << std::endl;
            }
        }

        return sym;
    };
#endif

    // First, try the user-requested function and its variants
    auto requested_variants = generate_variants(requested_function);
    for (const auto& symbol : requested_variants) {
        if (void* func = try_symbol(symbol)) {
            if (symbol != requested_function) {
                std::ostringstream out;
                out << "Warning: Requested function '" << requested_function 
                         << "' not found, using '" << symbol << "' instead" << std::endl;
                MFEM_WARNING_0(out.str());
            } else {
                std::cout << "Found requested UMAT function: " << symbol << std::endl;
            }
            return reinterpret_cast<UmatFunction>(func);
        }
    }

    // If user specified something other than default, warn before falling back
    if (requested_function != "umat_call") {
        std::ostringstream out;
        out << "Warning: Could not find requested function '" << requested_function
                  << "' or its variants. Trying default UMAT symbols..." << std::endl;
        MFEM_WARNING_0(out.str());
    }

    // Common UMAT symbol variants to try
    const std::vector<std::string> default_symbols = {
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

    for (const auto& symbol : default_symbols) {
        if (symbol == requested_function) {
            continue; // Already tried this
        }
        if (void* func = try_symbol(symbol)) {
            std::ostringstream out;
            out << "Warning: Using fallback UMAT symbol: " << symbol 
                << " (requested: " << requested_function << ")" << std::endl;
        MFEM_WARNING_0(out.str());
            return reinterpret_cast<UmatFunction>(func);
        }
    }
    
    // Report what we tried
    std::ostringstream err;
    err << "Could not find UMAT symbol. Tried:" << std::endl;
    err << "  Requested function variants:";
    for (const auto& s : requested_variants) {
        err << " " << s;
    }
    err << std::endl << "  Default symbols:";
    for (const auto& symbol : default_symbols) {
        err << " " << symbol;
    }
    err << std::endl;

    MFEM_ABORT_0(err.str());
    
    return nullptr;
}

bool DynamicUmatLoader::UnloadLibrary(LibraryHandle handle) {
    return dlclose(handle) == 0;
}

std::string DynamicUmatLoader::GetLastError() {
    const char* error = dlerror();
    return error ? std::string(error) : "No error";
}
#endif