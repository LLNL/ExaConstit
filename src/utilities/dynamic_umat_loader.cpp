#include "dynamic_umat_loader.hpp"

#include <iostream>
#include <filesystem>

// Static member definitions
std::unordered_map<std::string, std::unique_ptr<DynamicUmatLoader::UmatLibraryInfo>> 
    DynamicUmatLoader::loaded_libraries_;
std::mutex DynamicUmatLoader::library_mutex_;

// Implementation

UmatFunction DynamicUmatLoader::LoadUmat(const std::string& library_path, LoadStrategy strategy) {
    std::lock_guard<std::mutex> lock(library_mutex_);
    
    // Check if already loaded
    auto it = loaded_libraries_.find(library_path);
    if (it != loaded_libraries_.end() && it->second->is_loaded) {
        it->second->reference_count++;
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
    UmatFunction umat_func = GetUmatSymbol(handle);
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

UmatFunction DynamicUmatLoader::GetUmatSymbol(LibraryHandle handle) {
    return reinterpret_cast<UmatFunction>(::GetProcAddress(handle, "umat_call"));
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

UmatFunction DynamicUmatLoader::GetUmatSymbol(LibraryHandle handle) {
    // Clear any existing error
    dlerror();
    
    // Try common UMAT symbol names
    UmatFunction func = reinterpret_cast<UmatFunction>(dlsym(handle, "umat_call"));
    if (!func) {
        func = reinterpret_cast<UmatFunction>(dlsym(handle, "umat"));
    }
    if (!func) {
        func = reinterpret_cast<UmatFunction>(dlsym(handle, "umat_"));
    }
    
    return func;
}

bool DynamicUmatLoader::UnloadLibrary(LibraryHandle handle) {
    return dlclose(handle) == 0;
}

std::string DynamicUmatLoader::GetLastError() {
    const char* error = dlerror();
    return error ? std::string(error) : "No error";
}
#endif