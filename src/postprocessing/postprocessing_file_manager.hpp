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
     * 
     * Ensures the main output directory exists before file operations.
     * Creates the directory structure using filesystem operations with
     * proper error handling. Only MPI rank 0 performs directory creation
     * to avoid race conditions in parallel execution.
     */
    bool EnsureOutputDirectoryExists();

    /**
     * @brief Create directory if it doesn't exist
     * 
     * @param output_dir Directory path to create
     * @return true if directory exists or was created successfully
     * 
     * Generic directory creation utility with filesystem error handling.
     * Used for both main output directory and subdirectory creation
     * such as visualization output folders.
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
     * @brief Get column header string for volume average output files
     * 
     * @param calc_type Type of calculation
     * @return Header string with column descriptions
     * 
     * Provides standardized column headers for volume average output files.
     * Headers include time, volume, and appropriate component labels for
     * each calculation type (tensor components, scalar values, etc.).
     * 
     * Ensures consistent output format for post-processing tools and
     * provides clear documentation of data organization in output files.
     */
    std::string GetVolumeAverageHeader(const std::string& calc_type) const;
    
    /**
     * @brief Check if output should occur at the current step
     * 
     * @param step Current time step number
     * @return true if output should occur, false otherwise
     * 
     * Implements output frequency control based on ExaOptions configuration.
     * Uses modulo operation to determine if current step matches the
     * configured output frequency for volume averaging operations.
     */
    bool ShouldOutputAtStep(int step) const;
    
private:
    /**
     * @brief Get specific filename for a calculation type
     * 
     * @param calc_type Type of calculation (e.g., "stress", "def_grad")
     * @return Filename with extension from ExaOptions configuration
     * 
     * Maps calculation type strings to configured filenames from ExaOptions.
     * Supports standard calculation types (stress, deformation gradient,
     * plastic work, strains) with fallback to default naming for custom types.
     * 
     * Enables user customization of output filenames through configuration
     * while maintaining consistent internal calculation type naming.
     */
    std::string GetSpecificFilename(const std::string& calc_type) const;
    
    /**
     * @brief Construct region-specific filename with proper formatting
     * 
     * @param base_name Base filename without extension
     * @param extension File extension (including dot)
     * @param region Region index
     * @param region_name Optional region name for descriptive filenames
     * @return Formatted filename with region identifier
     * 
     * Creates region-specific filenames using either region index or
     * descriptive region name when available. Handles special formatting
     * requirements and ensures consistent naming across all output files.
     * 
     * Format examples:
     * - "stress_region_0.txt" (index-based)
     * - "stress_grain_austenite.txt" (name-based)
     */
    std::string ConstructRegionFilename(const std::string& base_filename,
                                       const std::string& extension,
                                       int region,
                                       const std::string& region_name) const;
    
private:
    /**
     * @brief Reference to ExaOptions configuration
     * 
     * Provides access to user-specified configuration including output
     * directories, filenames, and frequency settings. Used throughout
     * the file manager for consistent configuration-driven behavior.
     */
    const ExaOptions& m_options;
    /**
     * @brief MPI rank for parallel output control
     * 
     * Used to ensure only rank 0 performs file I/O operations in parallel
     * execution. Prevents race conditions and duplicate file creation
     * while maintaining proper parallel execution semantics.
     */
    int m_mpi_rank;
    /**
     * @brief Main output directory path
     * 
     * Base directory for all postprocessing output files. Constructed
     * from ExaOptions basename and output directory settings with
     * proper path formatting and trailing slash handling.
     */
    std::string m_output_directory;
    /**
     * @brief Visualization output directory path
     * 
     * Subdirectory for visualization files (VisIt, ParaView, ADIOS2).
     * Created only when visualization output is enabled in ExaOptions.
     * Provides organized separation of data files and visualization files.
     */
    std::string m_output_viz;
    /**
     * @brief Base filename without extension
     * 
     * Core filename component used for all output files. Derived from
     * ExaOptions basename setting and used as the foundation for
     * region-specific and calculation-specific filename construction.
     */
    std::string m_base_filename;
    /**
     * @brief Output frequency for volume averaging
     * 
     * Timestep interval for volume average output. Copied from ExaOptions
     * volume averaging configuration and used by ShouldOutputAtStep()
     * for consistent output timing control.
     */
    int m_output_frequency;
    
    /**
     * @brief Cache of opened files to avoid reopening
     * 
     * Weak pointer cache that tracks opened ofstream objects to prevent
     * repeated file opening/closing operations. Uses weak_ptr to allow
     * automatic cleanup when files are no longer referenced elsewhere.
     * Improves performance for frequent output operations to the same files.
     */
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