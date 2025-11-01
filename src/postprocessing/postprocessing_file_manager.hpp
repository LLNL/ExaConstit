#pragma once

#include "options/option_parser_v2.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

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
    PostProcessingFileManager(const ExaOptions& options);

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
    fs::path GetOutputDirectory() const {
        return m_output_directory;
    }
    /**
     * @brief Get the visualization directory path
     */
    fs::path GetVizDirectory() const {
        return m_output_viz;
    }

    /**
     * @brief Get the base filename (without extension)
     */
    std::string GetBaseFilename() const {
        return m_base_filename;
    }

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
     * @param comm MPI communicator associated with a given region
     * @return true if directory exists or was created successfully
     *
     * Generic directory creation utility with filesystem error handling.
     * Used for both main output directory and subdirectory creation
     * such as visualization output folders.
     */
    bool EnsureDirectoryExists(fs::path& output_dir, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Create and open an output file with proper error handling
     *
     * @param filepath Full path to the file
     * @param append Whether to append to existing file
     * @param comm MPI communicator associated with a given region
     * @return Unique pointer to opened file stream
     */
    std::unique_ptr<std::ofstream>
    CreateOutputFile(const fs::path& filepath, bool append = true, MPI_Comm comm = MPI_COMM_WORLD);

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

    /**
     * @brief Write time and volume with consistent formatting
     *
     * @param stream Output file stream
     * @param time Current simulation time
     * @param volume Total volume
     *
     * Provides consistent formatting for the first two columns
     * that appear in all volume average output files.
     */
    void WriteTimeAndVolume(std::ofstream& stream, double time, double volume) const {
        stream << std::setw(COLUMN_WIDTH) << time << std::setw(COLUMN_WIDTH) << volume;
    }

    /**
     * @brief Write vector data with consistent column formatting
     *
     * @param stream Output file stream
     * @param data Vector or array containing the data values
     * @param size Number of elements to write
     *
     * Writes each element with proper column width alignment.
     * Template allows use with mfem::Vector, std::vector, or C arrays.
     */
    template <typename T>
    void WriteVectorData(std::ofstream& stream, const T& data, int size) const {
        for (int i = 0; i < size; ++i) {
            stream << std::setw(COLUMN_WIDTH) << data[i];
        }
    }

    /**
     * @brief Write single scalar value with consistent formatting
     *
     * @param stream Output file stream
     * @param value Scalar value to write
     *
     * Writes a single value with proper column width alignment.
     */
    template <typename T>
    void WriteScalarData(std::ofstream& stream, const T& value) const {
        stream << std::setw(COLUMN_WIDTH) << value;
    }

    /**
     * @brief Safe template version that avoids deprecated conversions
     */
    template <typename T>
    void WriteVolumeAverage(const std::string& calc_type,
                            int region,
                            const std::string& region_name,
                            double time,
                            double volume,
                            const T& data,
                            int data_size = -1,
                            MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank != 0)
            return;

        auto filepath = GetVolumeAverageFilePath(calc_type, region, region_name);

        bool file_exists = fs::exists(filepath);
        auto file = CreateOutputFile(filepath, true);

        if (file && file->is_open()) {
            if (!file_exists) {
                *file << GetVolumeAverageHeader(calc_type);
            }

            WriteTimeAndVolume(*file, time, volume);
            WriteDataSafe(*file, data, data_size);
            *file << "\n" << std::flush;
        }
    }

private:
    // Column width for scientific notation with 12 digits: "-1.234567890123e-05"
    static constexpr int COLUMN_WIDTH = 18;
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
     * @brief Configure stream for high-precision output
     *
     * @param stream Reference to output stream to configure
     * @param precision Number of digits of precision (default 15)
     *
     * Centralizes precision configuration for all output streams.
     * Uses scientific notation to ensure consistent formatting for
     * small values that might be missed with default precision.
     */
    void ConfigureStreamPrecision(std::ofstream& stream, int precision = 8) const {
        stream.precision(precision);
        stream.setf(std::ios::scientific, std::ios::floatfield);
        // Optional: Set width for consistent column alignment
        // stream.width(22); // Adjust based on your needs
    }

    /**
     * @brief Safe data writing that avoids deprecated conversions
     */
    void WriteDataSafe(std::ofstream& stream, const mfem::Vector& data, int size) const {
        int actual_size = (size > 0) ? size : data.Size();
        for (int i = 0; i < actual_size; ++i) {
            stream << std::setw(COLUMN_WIDTH) << data[i]; // Use operator[] instead of conversion
        }
    }

    void WriteDataSafe(std::ofstream& stream, double data, int /*size*/) const {
        stream << std::setw(COLUMN_WIDTH) << data; // Direct scalar value
    }

    void WriteDataSafe(std::ofstream& stream, const double* data, int size) const {
        for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
            stream << std::setw(COLUMN_WIDTH) << data[i]; // Array access, no pointer dereferencing
        }
    }

    void WriteDataSafe(std::ofstream& stream, const std::vector<double>& data, int size) const {
        const size_t actual_size = (size > 0) ? static_cast<size_t>(size) : data.size();
        for (size_t i = 0; i < actual_size; ++i) {
            stream << std::setw(COLUMN_WIDTH) << data[i];
        }
    }

    /**
     * @brief Create a centered string within a fixed column width
     *
     * @param text Text to center
     * @param width Total column width
     * @return Centered string with padding
     *
     * Centers text within the specified width using spaces for padding.
     * If the text is longer than the width, it will be truncated.
     */
    std::string CenterText(const std::string& text, int width) const {
        const size_t width_z = static_cast<size_t>(width);
        if (text.length() >= width_z) {
            return text.substr(0, width_z); // Truncate if too long
        }

        const size_t padding = width_z - text.length();
        const size_t left_pad = padding / 2;
        const size_t right_pad = padding - left_pad; // Handle odd padding

        return std::string(left_pad, ' ') + text + std::string(right_pad, ' ');
    }

    /**
     * @brief Reference to ExaOptions configuration
     *
     * Provides access to user-specified configuration including output
     * directories, filenames, and frequency settings. Used throughout
     * the file manager for consistent configuration-driven behavior.
     */
    const ExaOptions& m_options;
    /**
     * @brief Main output directory path
     *
     * Base directory for all postprocessing output files. Constructed
     * from ExaOptions basename and output directory settings with
     * proper path formatting and trailing slash handling.
     */
    fs::path m_output_directory;
    /**
     * @brief Visualization output directory path
     *
     * Subdirectory for visualization files (VisIt, ParaView, ADIOS2).
     * Created only when visualization output is enabled in ExaOptions.
     * Provides organized separation of data files and visualization files.
     */
    fs::path m_output_viz;
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

inline PostProcessingFileManager::PostProcessingFileManager(const ExaOptions& options)
    : m_options(options) {
    // Use the basename from ExaOptions
    m_base_filename = options.basename;

    // Get output directory from volume averages options
    m_output_directory = options.post_processing.volume_averages.output_directory;

    // Get output frequency
    m_output_frequency = options.post_processing.volume_averages.output_frequency;

    // If output directory is empty, use current directory
    if (m_output_directory.empty()) {
        m_output_directory = ".";
    }
    // Resolve symlinks and build path
    m_output_directory = fs::weakly_canonical(m_output_directory) / m_base_filename;

    if (options.visualization.visit || options.visualization.paraview ||
        options.visualization.adios2) {
        m_output_viz = m_output_directory / "visualizations";
    }
}

inline std::string PostProcessingFileManager::GetVolumeAverageFilePath(
    const std::string& calc_type, int region, const std::string& region_name) const {
    // Get base filename with extension for this calculation type
    fs::path specific_filename = GetSpecificFilename(calc_type);

    // Split into base and extension
    fs::path extension = specific_filename.extension();
    fs::path base_name = specific_filename;
    if (extension.string().empty()) {
        extension = ".txt";
    } else {
        base_name = base_name.stem();
    }

    fs::path filename;
    if (region == -1) {
        // Global file
        filename = base_name.string() + "_global" + extension.string();
    } else {
        // Region-specific file
        filename = ConstructRegionFilename(
            base_name.string(), extension.string(), region, region_name);
    }

    return m_output_directory / filename;
}

inline std::string
PostProcessingFileManager::GetSpecificFilename(const std::string& calc_type) const {
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
    } else if (calc_type == "elastic_strain" || calc_type == "estrain") {
        return vol_opts.avg_elastic_strain_fname;
    } else {
        // Default naming for custom calculation types
        return "avg_" + calc_type + ".txt";
    }
}

inline std::string
PostProcessingFileManager::ConstructRegionFilename(const std::string& base_filename,
                                                   const std::string& extension,
                                                   int region,
                                                   const std::string& region_name) const {
    if (!region_name.empty()) {
        // Use region name if available
        return base_filename + "_region_" + region_name + extension;
    } else {
        // Use region index
        return base_filename + "_region_" + std::to_string(region) + extension;
    }
}

inline bool PostProcessingFileManager::EnsureDirectoryExists(fs::path& output_dir, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    bool success = false;
    if (rank == 0) {
        try {
            // Use weakly_canonical to resolve as much as possible
            // This handles symlinks and normalizes the path

            // Example: output_dir = "./results/test_case1/visualizations"
            // where ./results is a symlink to /storage/results
            // but test_case1/visualizations doesn't exist yet

            // weakly_canonical will resolve to:
            // /storage/results/test_case1/visualizations
            fs::path canonical_path = fs::weakly_canonical(output_dir);

            // Now check if this canonical path exists
            if (fs::exists(canonical_path)) {
                if (!fs::is_directory(canonical_path)) {
                    std::cerr << "Error: Path exists but is not a directory: " << canonical_path
                              << std::endl;
                    success = false;
                } else {
                    std::cout << "Using existing directory: " << canonical_path << std::endl;
                    output_dir = canonical_path;
                    success = true;
                }
            } else {
                // Directory doesn't exist, create it
                std::cout << "Creating output directory: " << canonical_path << std::endl;
                success = fs::create_directories(canonical_path);
                if (success) {
                    output_dir = canonical_path;
                } else {
                    std::cerr << "Warning: Failed to create output directory: " << canonical_path
                              << std::endl;
                }
            }

            // Write test remains the same...
            if (success) {
                fs::path test_file = canonical_path / "test_write.tmp";
                std::ofstream test_stream(test_file);
                if (!test_stream.is_open()) {
                    success = false;
                    std::cerr << "Warning: Output directory is not writable: " << canonical_path
                              << std::endl;
                } else {
                    test_stream.close();
                    fs::remove(test_file);
                }
            }
        } catch (const fs::filesystem_error& ex) {
            success = false;
            std::cerr << "Filesystem error when creating directory " << output_dir << ": "
                      << ex.what() << std::endl;
        } catch (const std::exception& ex) {
            success = false;
            std::cerr << "Error when creating directory " << output_dir << ": " << ex.what()
                      << std::endl;
        }
    }

    // Broadcast the potentially updated output_dir to all ranks
    std::string path_str = output_dir.string();
    int dir_length = static_cast<int>(path_str.length());
    MPI_Bcast(&dir_length, 1, MPI_INT, 0, comm);
    path_str.resize(static_cast<size_t>(dir_length));
    MPI_Bcast(&path_str[0], dir_length, MPI_CHAR, 0, comm);
    output_dir = path_str;

    bool success_t = false;
    MPI_Allreduce(&success, &success_t, 1, MPI_C_BOOL, MPI_LOR, comm);
    return success_t;
}

inline bool PostProcessingFileManager::EnsureOutputDirectoryExists() {
    bool success = EnsureDirectoryExists(m_output_directory);
    if (!m_output_viz.empty()) {
        bool viz_success = EnsureDirectoryExists(m_output_viz);
        success &= viz_success;
    }

    return success;
}

inline std::unique_ptr<std::ofstream>
PostProcessingFileManager::CreateOutputFile(const fs::path& filepath, bool append, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    try {
        // Use weakly_canonical to resolve symlinks in the path
        fs::path resolved_path = fs::weakly_canonical(filepath);
        fs::path dir_path = resolved_path.parent_path();

        // Ensure directory exists (only check and create if needed)
        if (!dir_path.empty() && !fs::exists(dir_path)) {
            if (rank == 0) {
                std::cout << "Creating directory: " << dir_path << std::endl;
                fs::create_directories(dir_path);
            }
            // Synchronize to ensure directory is created before all ranks proceed
            MPI_Barrier(comm);
        }

        // Open file
        std::ios_base::openmode mode = std::ios_base::out;
        if (append) {
            mode |= std::ios_base::app;
        }

        auto file = std::make_unique<std::ofstream>(resolved_path, mode);

        if (!file->is_open()) {
            if (rank == 0) {
                std::cerr << "Warning: Failed to open output file: " << resolved_path << std::endl;
            }
            return nullptr;
        }
        // Apply precision configuration
        ConfigureStreamPrecision(*file);
        return file;

    } catch (const fs::filesystem_error& ex) {
        if (rank == 0) {
            std::cerr << "Filesystem error when creating file " << filepath << ": " << ex.what()
                      << std::endl;
        }
        return nullptr;
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Error when creating file " << filepath << ": " << ex.what() << std::endl;
        }
        return nullptr;
    }
}

// Updated GetVolumeAverageHeader method with proper alignment:
inline std::string
PostProcessingFileManager::GetVolumeAverageHeader(const std::string& calc_type) const {
    std::ostringstream header;

    // Set formatting for header to match data columns
    header << CenterText("# Time", COLUMN_WIDTH);
    header << CenterText("Volume", COLUMN_WIDTH);

    if (calc_type == "stress") {
        header << CenterText("Sxx", COLUMN_WIDTH);
        header << CenterText("Syy", COLUMN_WIDTH);
        header << CenterText("Szz", COLUMN_WIDTH);
        header << CenterText("Sxy", COLUMN_WIDTH);
        header << CenterText("Sxz", COLUMN_WIDTH);
        header << CenterText("Syz", COLUMN_WIDTH);
    } else if (calc_type == "def_grad") {
        header << CenterText("F11", COLUMN_WIDTH);
        header << CenterText("F12", COLUMN_WIDTH);
        header << CenterText("F13", COLUMN_WIDTH);
        header << CenterText("F21", COLUMN_WIDTH);
        header << CenterText("F22", COLUMN_WIDTH);
        header << CenterText("F23", COLUMN_WIDTH);
        header << CenterText("F31", COLUMN_WIDTH);
        header << CenterText("F32", COLUMN_WIDTH);
        header << CenterText("F33", COLUMN_WIDTH);
    } else if (calc_type == "euler_strain") {
        header << CenterText("E11", COLUMN_WIDTH);
        header << CenterText("E22", COLUMN_WIDTH);
        header << CenterText("E33", COLUMN_WIDTH);
        header << CenterText("E23", COLUMN_WIDTH);
        header << CenterText("E13", COLUMN_WIDTH);
        header << CenterText("E12", COLUMN_WIDTH);
    } else if (calc_type == "plastic_work" || calc_type == "pl_work") {
        header << CenterText("Plastic_Work", COLUMN_WIDTH);
    } else if (calc_type == "elastic_strain") {
        header << CenterText("Ee11", COLUMN_WIDTH);
        header << CenterText("Ee22", COLUMN_WIDTH);
        header << CenterText("Ee33", COLUMN_WIDTH);
        header << CenterText("Ee23", COLUMN_WIDTH);
        header << CenterText("Ee13", COLUMN_WIDTH);
        header << CenterText("Ee12", COLUMN_WIDTH);
    } else if (calc_type == "eps" || calc_type == "eq_pl_strain") {
        header << CenterText("Equiv_Plastic_Strain", COLUMN_WIDTH); // Shortened to fit better
    } else {
        header << CenterText(calc_type, COLUMN_WIDTH);
    }

    header << "\n";
    return header.str();
}

inline bool PostProcessingFileManager::ShouldOutputAtStep(int step) const {
    return (step % m_output_frequency == 0);
}