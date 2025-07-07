#pragma once

#include "options/option_parser_v2.hpp"

#include <filesystem>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <memory>

namespace fs = std::filesystem;

/**
 * @brief Utility class for managing file paths and directories in PostProcessingDriver
 * 
 * This class handles:
 * 1. Proper file naming according to ExaOptions conventions
 * 2. Directory creation when needed
 * 3. Path resolution (relative vs absolute)
 * 4. Error handling for file operations
 */
class PostProcessingFileManager {
public:
    /**
     * @brief Initialize file manager with ExaOptions
     */
    PostProcessingFileManager(const ExaOptions& options, int mpi_rank = 0);
    
    /**
     * @brief Get the full file path for a volume average output
     * 
     * @param calc_type Type of calculation (e.g., "stress", "def_grad")
     * @param region Region index (-1 for global)
     * @param region_name Optional region name (if available)
     * @return Full file path
     */
    std::string GetVolumeAverageFilePath(const std::string& calc_type, 
                                        int region = -1, 
                                        const std::string& region_name = "") const;
    
    /**
     * @brief Get the output directory path
     */
    std::string GetOutputDirectory() const { return m_output_directory; }
    /**
     * @brief Get the visualization directory path
     */
    std::string GetVizDirectory() const { return m_output_viz; }
    
    /**
     * @brief Get the base filename (without extension)
     */
    std::string GetBaseFilename() const { return m_base_filename; }
    
    /**
     * @brief Create output directory if it doesn't exist
     * 
     * @return true if directory exists or was created successfully
     */
    bool EnsureOutputDirectoryExists();

    /**
     * @brief Create directory if it doesn't exist
     * 
     * @return true if directory exists or was created successfully
     */
    bool EnsureDirectoryExists(std::string& output_dir);
    
    /**
     * @brief Create and open an output file with proper error handling
     * 
     * @param filepath Full path to the file
     * @param append Whether to append to existing file
     * @return Unique pointer to opened file stream
     */
    std::unique_ptr<std::ofstream> CreateOutputFile(const std::string& filepath, 
                                                   bool append = true);
    
    /**
     * @brief Get the header string for volume average files
     * 
     * @param calc_type Type of calculation
     * @return Header string
     */
    std::string GetVolumeAverageHeader(const std::string& calc_type) const;
    
    /**
     * @brief Check if we should output at this frequency
     * 
     * @param step Current step
     * @return true if should output
     */
    bool ShouldOutputAtStep(int step) const;
    
private:
    /**
     * @brief Get the specific filename from ExaOptions if available
     * 
     * @param calc_type Type of calculation
     * @return Filename from options, or default based on calc_type
     */
    std::string GetSpecificFilename(const std::string& calc_type) const;
    
    /**
     * @brief Construct region-specific filename
     * 
     * @param base_filename Base filename without extension
     * @param extension File extension
     * @param region Region index
     * @param region_name Region name
     * @return Region-specific filename
     */
    std::string ConstructRegionFilename(const std::string& base_filename,
                                       const std::string& extension,
                                       int region,
                                       const std::string& region_name) const;
    
private:
    const ExaOptions& m_options;
    int m_mpi_rank;
    std::string m_output_directory;
    std::string m_output_viz;
    std::string m_base_filename;
    int m_output_frequency;
    
    // Cache of opened files to avoid reopening
    mutable std::map<std::string, std::weak_ptr<std::ofstream>> m_file_cache;
};

// Implementation

inline PostProcessingFileManager::PostProcessingFileManager(const ExaOptions& options, int mpi_rank)
    : m_options(options), m_mpi_rank(mpi_rank) {
    
    // Use the basename from ExaOptions
    m_base_filename = options.basename;
    
    // Get output directory from volume averages options
    m_output_directory = options.post_processing.volume_averages.output_directory;
    
    // Get output frequency
    m_output_frequency = options.post_processing.volume_averages.output_frequency;
    
    // Ensure output directory has trailing slash
    if (!m_output_directory.empty() && m_output_directory.back() != '/') {
        m_output_directory += '/';
    }
    
    // If output directory is empty, use current directory
    if (m_output_directory.empty()) {
        m_output_directory = "./";
    }
    m_output_directory += m_base_filename + '/';
    if (options.visualization.visit || options.visualization.paraview || options.visualization.adios2) {
        m_output_viz = m_output_directory + std::string("visualizations/");
    }
}

inline std::string PostProcessingFileManager::GetVolumeAverageFilePath(
    const std::string& calc_type, int region, const std::string& region_name) const {
    
    // Get base filename with extension for this calculation type
    std::string specific_filename = GetSpecificFilename(calc_type);
    
    // Split into base and extension
    std::string base_name, extension;
    size_t dot_pos = specific_filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        base_name = specific_filename.substr(0, dot_pos);
        extension = specific_filename.substr(dot_pos);
    } else {
        base_name = specific_filename;
        extension = ".txt";
    }
    
    std::string filename;
    if (region == -1) {
        // Global file
        filename =  base_name + "_global" + extension;
    } else {
        // Region-specific file
        filename = ConstructRegionFilename(base_name, extension, region, region_name);
    }
    
    return m_output_directory + filename;
}

inline std::string PostProcessingFileManager::GetSpecificFilename(const std::string& calc_type) const {
    const auto& vol_opts = m_options.post_processing.volume_averages;
    // Map calculation types to specific filenames from ExaOptions
    if (calc_type == "stress") {
        return vol_opts.avg_stress_fname;
    } else if (calc_type == "def_grad") {
        return vol_opts.avg_def_grad_fname;
    } else if (calc_type == "plastic_work" || calc_type == "pl_work") {
        return vol_opts.avg_pl_work_fname;
    } else if (calc_type == "euler_strain") {
        return vol_opts.avg_euler_strain_fname;
    } else if (calc_type == "eps" || calc_type == "eq_pl_strain") {
        return vol_opts.avg_eq_pl_strain_fname;
    } 
    else if (calc_type == "elastic_strain" || calc_type == "estrain") {
        return vol_opts.avg_elastic_strain_fname;
    } else {
        // Default naming for custom calculation types
        return "avg_" + calc_type + ".txt";
    }
}

inline std::string PostProcessingFileManager::ConstructRegionFilename(
    const std::string& base_filename, const std::string& extension,
    int region, const std::string& region_name) const {
    
    if (!region_name.empty()) {
        // Use region name if available
        return  base_filename + "_region_" + region_name + extension;
    } else {
        // Use region index
        return base_filename + "_region_" + std::to_string(region) + extension;
    }
}

inline bool PostProcessingFileManager::EnsureDirectoryExists(std::string& output_dir) {
    bool success = false;
    if (m_mpi_rank == 0) {
        try {
                if (!fs::exists(output_dir)) {
                        std::cout << "Creating output directory: " << output_dir << std::endl;
                }
                success = fs::create_directories(output_dir);
                if (!success) {
                        std::cerr << "Warning: Failed to create output directory: " 
                                << output_dir << std::endl;
                }
                // Check if directory is writable
                fs::path test_file = fs::path(output_dir) / "test_write.tmp";
                std::ofstream test_stream(test_file);
                if (!test_stream.is_open()) {
                    success = false;
                    std::cerr << "Warning: Output directory is not writable: " 
                                << output_dir << std::endl;
                }
                test_stream.close();
                fs::remove(test_file);
        } catch (const fs::filesystem_error& ex) {
            success = false;
            std::cerr << "Filesystem error when creating directory " 
                    << output_dir << ": " << ex.what() << std::endl;
        } catch (const std::exception& ex) {
            success = false;
            std::cerr << "Error when creating directory " 
                    << output_dir << ": " << ex.what() << std::endl;
        }
    }
    bool success_t = false;
    MPI_Allreduce(&success, &success_t, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    return success_t;
}

inline bool PostProcessingFileManager::EnsureOutputDirectoryExists() {

    bool success = EnsureDirectoryExists(m_output_directory);
    if (m_output_viz.size() > 0) {
        bool viz_success = EnsureDirectoryExists(m_output_viz);
        success &= viz_success;
    }

    return success;
}

inline std::unique_ptr<std::ofstream> PostProcessingFileManager::CreateOutputFile(
    const std::string& filepath, bool append) {
    
    // Ensure directory exists
    fs::path file_path(filepath);
    fs::path dir_path = file_path.parent_path();
    
    try {
        if (!dir_path.empty() && !fs::exists(dir_path)) {
            if (m_mpi_rank == 0) {
                std::cout << "Creating directory: " << dir_path << std::endl;
            }
            fs::create_directories(dir_path);
        }
        
        // Open file
        std::ios_base::openmode mode = std::ios_base::out;
        if (append) {
            mode |= std::ios_base::app;
        }
        
        auto file = std::make_unique<std::ofstream>(filepath, mode);
        
        if (!file->is_open()) {
            if (m_mpi_rank == 0) {
                std::cerr << "Warning: Failed to open output file: " << filepath << std::endl;
            }
            return nullptr;
        }
        
        return file;
        
    } catch (const fs::filesystem_error& ex) {
        if (m_mpi_rank == 0) {
            std::cerr << "Filesystem error when creating file " 
                      << filepath << ": " << ex.what() << std::endl;
        }
        return nullptr;
    } catch (const std::exception& ex) {
        if (m_mpi_rank == 0) {
            std::cerr << "Error when creating file " 
                      << filepath << ": " << ex.what() << std::endl;
        }
        return nullptr;
    }
}

inline std::string PostProcessingFileManager::GetVolumeAverageHeader(const std::string& calc_type) const {
    if (calc_type == "stress") {
        return "# Time, Volume, Sxx, Syy, Szz, Sxy, Sxz, Syz\n";
    } else if (calc_type == "def_grad") {
        return "# Time, Volume, F11, F12, F13, F21, F22, F23, F31, F32, F33\n";
    } else if (calc_type == "euler_strain") {
        return "# Time, Volume, E11, E22, E33, E23, E13, E12\n";
    } else if (calc_type == "plastic_work" || calc_type == "pl_work") {
        return "# Time, Volume, Plastic_Work\n";
    } else if (calc_type == "elastic_strain") {
        return "# Time, Volume, Ee11, Ee22, Ee33, Ee23, Ee13, Ee12\n";
    } else if (calc_type == "eps" || calc_type == "eq_pl_strain") {
        return "# Time, Volume, Equivalent_Plastic_Strain\n";
    }
    else {
        return "# Time, Volume, " + calc_type + "\n";
    }
}

inline bool PostProcessingFileManager::ShouldOutputAtStep(int step) const {
    return (step % m_output_frequency == 0);
}