#pragma once

#include "mfem.hpp"

#include <atomic>
#include <chrono>
#include <ctime>
#include <fcntl.h> // For open(), pipe(), dup2() - POSIX file operations
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <mutex>
#include <optional>
#include <sstream>
#include <stack>
#include <system_error>
#include <thread>
#include <unistd.h> // For read(), write(), close() - POSIX I/O

#ifdef RAJA_ENABLE_CUDA
#include <cuda_runtime.h>
#elif defined(RAJA_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

class PostProcessingFileManager;
class ExaOptions;

namespace exaconstit {

// ============================================================================
// RAII WRAPPERS FOR SAFER POSIX OPERATIONS
// ============================================================================

/**
 * @brief RAII wrapper for file descriptors
 *
 * @details Provides automatic cleanup of file descriptors and prevents
 * resource leaks. File descriptors are move-only (can't be copied) to
 * maintain single ownership semantics.
 *
 * Why we need this:
 * - Raw file descriptors (int) don't clean themselves up
 * - Easy to forget close() calls, especially with exceptions
 * - Prevents accidental copying of file descriptors
 *
 * Usage:
 * ```cpp
 * FileDescriptor fd(open("file.txt", O_RDONLY));
 * if (!fd) throw std::runtime_error("Failed to open");
 * // fd automatically closed when out of scope
 * ```
 */
class FileDescriptor {
private:
    int fd; // The underlying POSIX file descriptor (-1 = invalid)

public:
    // Default constructor creates invalid descriptor
    FileDescriptor() : fd(-1) {}

    // Wrap an existing file descriptor
    explicit FileDescriptor(int fd_) : fd(fd_) {}

    // Move constructor - transfers ownership
    FileDescriptor(FileDescriptor&& other) noexcept : fd(other.fd) {
        other.fd = -1; // Other no longer owns the descriptor
    }

    // Move assignment - close current and transfer ownership
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            close(); // Close our current descriptor if valid
            fd = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    // Prevent copying - file descriptors should have single owner
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    // Destructor ensures descriptor is closed
    ~FileDescriptor() {
        close();
    }

    // Explicitly close the descriptor
    void close() {
        if (fd >= 0) {
            ::close(fd); // :: means global namespace (POSIX close)
            fd = -1;
        }
    }

    // Getters
    int get() const {
        return fd;
    }
    bool is_valid() const {
        return fd >= 0;
    }

    // Release ownership without closing
    int release() {
        int temp = fd;
        fd = -1;
        return temp;
    }

    // Convenience operators
    operator int() const {
        return fd;
    } // Allow implicit conversion to int
    explicit operator bool() const {
        return is_valid();
    } // if (fd) {...}
};

/**
 * @brief RAII wrapper for pipe creation
 *
 * @details Creates a pipe and manages both ends with automatic cleanup.
 * A pipe is a unidirectional communication channel:
 * - Data written to write_end can be read from read_end
 * - Used for capturing stdout/stderr by redirecting them to write_end
 *
 * Why we need pipes:
 * - C++ streams only capture C++ iostream output
 * - Pipes capture ALL output: printf, fprintf, Fortran WRITE, etc.
 * - Works at the OS level, not language level
 */
class Pipe {
private:
    FileDescriptor read_end_var;  // Where we read captured data from
    FileDescriptor write_end_var; // Where stdout/stderr write to

public:
    // Constructor creates the pipe
    Pipe() {
        int pipe_fds[2];
        if (::pipe(pipe_fds) == -1) {
            // Convert errno to C++ exception with descriptive message
            throw std::system_error(errno, std::system_category(), "Failed to create pipe");
        }
        // pipe() fills array: [0] = read end, [1] = write end
        read_end_var = FileDescriptor(pipe_fds[0]);
        write_end_var = FileDescriptor(pipe_fds[1]);
    }

    // Access to pipe ends
    FileDescriptor& read_end() {
        return read_end_var;
    }
    FileDescriptor& write_end() {
        return write_end_var;
    }
    const FileDescriptor& read_end() const {
        return read_end_var;
    }
    const FileDescriptor& write_end() const {
        return write_end_var;
    }

    /**
     * @brief Make pipe non-blocking for efficient reading
     *
     * @param read Make read end non-blocking
     * @param write Make write end non-blocking
     *
     * @details Non-blocking is important for read end so our reader
     * thread doesn't hang waiting for data. It can check for data
     * and do other work if none is available.
     */
    void set_non_blocking(bool read = true, bool write = false) {
        if (read && read_end_var) {
            int flags = fcntl(read_end_var.get(), F_GETFL, 0);
            fcntl(read_end_var.get(), F_SETFL, flags | O_NONBLOCK);
        }
        if (write && write_end_var) {
            int flags = fcntl(write_end_var.get(), F_GETFL, 0);
            fcntl(write_end_var.get(), F_SETFL, flags | O_NONBLOCK);
        }
    }
};

/**
 * @brief RAII wrapper for saving and restoring file descriptors
 *
 * @details When we redirect stdout/stderr, we need to:
 * 1. Save the current descriptor so we can restore it later
 * 2. Redirect to our pipe
 * 3. Later restore the original descriptor
 *
 * This class handles that pattern safely with automatic restoration.
 */
class FileDescriptorDuplicator {
private:
    FileDescriptor saved_fd; // Copy of original descriptor
    int target_fd;           // Which descriptor we're managing (1=stdout, 2=stderr)

public:
    // Constructor saves a copy of the current descriptor
    FileDescriptorDuplicator(int fd_to_save) : saved_fd(::dup(fd_to_save)), target_fd(fd_to_save) {
        if (!saved_fd) {
            throw std::system_error(
                errno, std::system_category(), "Failed to duplicate file descriptor");
        }
    }

    // Destructor automatically restores original
    ~FileDescriptorDuplicator() {
        restore();
    }

    // Manually restore original descriptor
    void restore() {
        if (saved_fd) {
            ::dup2(saved_fd.get(), target_fd); // Restore original
            saved_fd.close();                  // Close our saved copy
        }
    }

    // Redirect target to a new descriptor
    void redirect_to(int new_fd) {
        ::dup2(new_fd, target_fd); // target_fd now points to new_fd
    }

    // Prevent copying
    FileDescriptorDuplicator(const FileDescriptorDuplicator&) = delete;
    FileDescriptorDuplicator& operator=(const FileDescriptorDuplicator&) = delete;
};

/**
 * @brief RAII wrapper for C++ stream buffer management
 *
 * @details C++ streams (cout/cerr) use streambuf objects for actual I/O.
 * By replacing the streambuf, we can redirect where the stream writes.
 * This class safely saves and restores the original streambuf.
 *
 * Used for tee functionality where we want cout to write to both
 * terminal and log file.
 */
class StreamBufferGuard {
private:
    std::ostream* stream;            // The stream we're managing (cout or cerr)
    std::streambuf* original_buffer; // Original buffer to restore
    bool active;                     // Whether we still need to restore

public:
    StreamBufferGuard(std::ostream& stream_)
        : stream(&stream_), original_buffer(stream_.rdbuf()), // Save current buffer
          active(true) {}

    ~StreamBufferGuard() {
        restore(); // Ensure buffer is restored
    }

    // Replace stream's buffer with a new one
    void set_buffer(std::streambuf* new_buffer) {
        if (active && stream) {
            stream->rdbuf(new_buffer);
        }
    }

    // Restore original buffer
    void restore() {
        if (active && stream && original_buffer) {
            stream->rdbuf(original_buffer);
            active = false; // Don't restore twice
        }
    }

    std::streambuf* get_original() const {
        return original_buffer;
    }

    // Move-only semantics
    StreamBufferGuard(StreamBufferGuard&& other) noexcept
        : stream(other.stream), original_buffer(other.original_buffer), active(other.active) {
        other.active = false; // Other shouldn't restore
    }

    // No copy
    StreamBufferGuard(const StreamBufferGuard&) = delete;
    StreamBufferGuard& operator=(const StreamBufferGuard&) = delete;
};

// ============================================================================
// UNIFIED LOGGER CLASS
// ============================================================================

/**
 * @brief Unified logging system for ExaConstit
 *
 * @details Provides two main modes of operation:
 *
 * 1. **Tee Mode (Main Logging)**:
 *    - Active throughout simulation
 *    - All stdout/stderr goes to BOTH terminal AND log file
 *    - Uses custom streambuf to duplicate output
 *
 * 2. **Capture Mode (Material Logging)**:
 *    - Temporarily activated for specific code sections
 *    - All stdout/stderr goes ONLY to a specific file
 *    - Terminal sees nothing during capture
 *    - Files only created if content is actually captured
 *
 * Architecture:
 * - Singleton pattern ensures single instance
 * - RAII wrappers ensure exception safety
 * - Thread-safe for parallel MPI execution
 * - Captures all output types (C++, C, Fortran, GPU)
 */
class UnifiedLogger {
private:
    // ========================================================================
    // SINGLETON PATTERN
    // ========================================================================
    static std::unique_ptr<UnifiedLogger> instance;
    static std::mutex instance_mutex;

    // ========================================================================
    // MPI INFORMATION
    // ========================================================================
    int mpi_rank; // This process's rank
    int mpi_size; // Total number of processes
    bool debugging_logging = false;

    // ========================================================================
    // MAIN LOG FILE MANAGEMENT
    // ========================================================================
    std::filesystem::path log_directory;          // Where all logs are stored
    std::filesystem::path main_log_filename;      // Main simulation log
    std::unique_ptr<std::ofstream> main_log_file; // File stream for main log

    // ========================================================================
    // TEE STREAMBUF FOR DUAL OUTPUT
    // ========================================================================
    /**
     * @brief Custom stream buffer that writes to two destinations
     *
     * @details Inherits from std::streambuf to intercept stream operations.
     * When data is written to cout/cerr, this buffer writes it to both
     * the terminal and a log file simultaneously.
     *
     * How it works:
     * - overflow(): Called for single character output
     * - xsputn(): Called for string output (more efficient)
     * - Both methods write to terminal AND file
     * - Thread-safe via mutex
     */
    class TeeStreambuf : public std::streambuf {
    protected:
        std::streambuf* original_buf; // The ORIGINAL buffer (terminal)
        std::ostream* file_stream;    // The log file stream
        std::mutex mutex_lock;

    protected:
        virtual int overflow(int c) override {
            int result = c;
            if (c != EOF) {
                char cchar = static_cast<char>(c);
                std::lock_guard<std::mutex> lock(mutex_lock);

                // Write to original buffer first
                if (original_buf && original_buf->sputc(cchar) == EOF) {
                    result = EOF;
                }

                // Then write to file
                if (file_stream) {
                    file_stream->put(cchar);
                }
            }
            return result;
        }

        virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
            std::lock_guard<std::mutex> lock(mutex_lock);

            // Write to original buffer
            std::streamsize result = n;
            if (original_buf) {
                result = original_buf->sputn(s, n);
            }

            // Also write to file
            if (file_stream && result > 0) {
                file_stream->write(s, result);
            }

            return result;
        }

        // Critical: sync() must flush both destinations
        virtual int sync() override {
            std::lock_guard<std::mutex> lock(mutex_lock);
            int result = 0;

            if (original_buf) {
                result = original_buf->pubsync();
            }

            if (file_stream) {
                file_stream->flush();
            }

            return result;
        }

    public:
        TeeStreambuf(std::streambuf* terminal, std::ostream* file)
            : original_buf(terminal), file_stream(file) {}
    };

    // ========================================================================
    // STREAM MANAGEMENT FOR TEE MODE
    // ========================================================================
    std::optional<StreamBufferGuard> cout_guard; // Manages cout buffer
    std::optional<StreamBufferGuard> cerr_guard; // Manages cerr buffer
    std::unique_ptr<TeeStreambuf> cout_tee;      // Tee buffer for cout
    std::unique_ptr<TeeStreambuf> cerr_tee;      // Tee buffer for cerr

    // Add a flag to track if we're in capture mode
    bool in_capture_mode = false;

    // Store original streambufs when disabling tee temporarily
    std::optional<StreamBufferGuard> temp_cout_guard;
    std::optional<StreamBufferGuard> temp_cerr_guard;
    std::optional<StreamBufferGuard> temp_mfem_out_guard;
    std::optional<StreamBufferGuard> temp_mfem_err_guard;

    std::optional<StreamBufferGuard> mfem_out_guard;
    std::optional<StreamBufferGuard> mfem_err_guard;
    std::unique_ptr<TeeStreambuf> mfem_out_tee;
    std::unique_ptr<TeeStreambuf> mfem_err_tee;

    // ========================================================================
    // CAPTURE CONTEXT FOR REDIRECTED OUTPUT
    // ========================================================================
    /**
     * @brief State for active output capture
     *
     * @details When capturing output to a file, we need to:
     * 1. Save current stdout/stderr state
     * 2. Create a pipe for capturing
     * 3. Redirect stdout/stderr to the pipe
     * 4. Read from pipe in separate thread
     * 5. Restore everything when done
     *
     * This structure holds all that state with RAII for safety.
     */
    struct CaptureContext {
        // RAII wrappers for file descriptor management
        std::optional<FileDescriptorDuplicator> stdout_dup; // Saves/restores stdout
        std::optional<FileDescriptorDuplicator> stderr_dup; // Saves/restores stderr
        Pipe capture_pipe;                                  // Pipe for capturing output

        // Thread that reads from pipe
        std::thread reader_thread;

        // Output accumulation
        std::stringstream captured_output;     // Where we store captured text
        std::atomic<bool> stop_reading{false}; // Signal to stop reader thread

        // Metadata
        std::filesystem::path output_filename; // Where to write captured output
        bool suppress_non_zero_ranks;          // Only capture on rank 0?
        bool has_content{false};               // Did we capture anything?

        // Statistics
        std::chrono::steady_clock::time_point start_time; // When capture started
        size_t bytes_captured{0};                         // Total bytes captured

        // Add these for C++ stream handling during suppression
        std::unique_ptr<std::ofstream> null_stream;
        std::optional<StreamBufferGuard> cout_null_guard;
        std::optional<StreamBufferGuard> cerr_null_guard;
        std::optional<StreamBufferGuard> mfem_out_null_guard;
        std::optional<StreamBufferGuard> mfem_err_null_guard;
    };

    // Stack allows nested captures (capture within capture)
    std::stack<std::unique_ptr<CaptureContext>> capture_stack;
    std::mutex capture_mutex; // Thread safety for capture operations

    // ========================================================================
    // STATISTICS TRACKING
    // ========================================================================
    struct LogStatistics {
        int total_captures = 0;                 // Total begin_capture calls
        int files_created = 0;                  // Files actually written
        std::vector<std::string> created_files; // List of created files
    } stats_;

    // ========================================================================
    // PRIVATE CONSTRUCTOR (SINGLETON)
    // ========================================================================
    UnifiedLogger() {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }

    // ========================================================================
    // PRIVATE HELPER METHODS
    // ========================================================================

    /**
     * @brief Thread function that reads captured output from pipe
     *
     * @param ctx Capture context containing pipe and output buffer
     *
     * @details Runs in separate thread during capture. Continuously reads
     * from the pipe where stdout/stderr are redirected and accumulates
     * in a stringstream. Uses select() for efficient non-blocking I/O.
     */
    void reader_thread_func(CaptureContext* ctx);

    /**
     * @brief Ensure GPU printf buffers are flushed
     *
     * @details GPU kernels use device-side printf buffers that aren't
     * automatically flushed. This forces synchronization to ensure
     * all GPU output is captured before ending a capture session.
     */
    void flush_gpu_output();

    /**
     * @brief Restores the tee after capture to ensure proper logging
     */
    void restore_tee_after_capture();

public:
    // Delete copy operations (singleton must not be copied)
    UnifiedLogger(const UnifiedLogger&) = delete;
    UnifiedLogger& operator=(const UnifiedLogger&) = delete;

    /**
     * @brief Get the singleton instance
     *
     * @return Reference to the single UnifiedLogger instance
     *
     * @details Thread-safe lazy initialization. First call creates
     * the instance, subsequent calls return the same instance.
     */
    static UnifiedLogger& get_instance();

    /**
     * @brief Initialize the logging system
     *
     * @param options ExaOptions containing configuration
     *
     * @details Must be called once after MPI initialization. Sets up:
     * - Log directory structure
     * - Main simulation log file
     * - Tee functionality for cout/cerr
     * - Initial log headers with timestamp
     */
    void initialize(const ExaOptions& options);

    /**
     * @brief Enable tee output for main logging
     *
     * @details Activates custom streambuf that duplicates all output
     * to both terminal and log file. Called automatically by initialize().
     *
     * Implementation:
     * - Saves original cout/cerr buffers
     * - Creates TeeStreambuf instances
     * - Redirects cout/cerr to use TeeStreambuf
     */
    void enable_main_logging();

    /**
     * @brief Disable main logging tee
     *
     * @details Restores original cout/cerr buffers. Called by shutdown().
     * RAII guards ensure proper cleanup even if exceptions occur.
     */
    void disable_main_logging();

    /**
     * @brief Start capturing output to a specific file
     *
     * @param filename Output filename relative to log directory
     * @param suppress_non_zero_ranks If true, only rank 0 captures
     *
     * @details Redirects ALL output (stdout/stderr) to go ONLY to file.
     * Terminal sees nothing until end_capture() is called.
     *
     * Process:
     * 1. Create pipe for communication
     * 2. Save current stdout/stderr descriptors
     * 3. Redirect stdout/stderr to pipe's write end
     * 4. Start thread to read from pipe's read end
     *
     * Supports nesting - previous capture is paused and resumes later.
     */
    void begin_capture(const std::string& filename, bool suppress_non_zero_ranks = false);

    /**
     * @brief End current capture and optionally write to file
     *
     * @return Filename if file was created, empty string otherwise
     *
     * @details Restores stdout/stderr and writes captured content to file
     * ONLY if content exists (prevents empty log files).
     *
     * Process:
     * 1. Flush GPU and C streams
     * 2. Restore original stdout/stderr
     * 3. Stop reader thread and collect output
     * 4. Write to file if content exists
     * 5. Update statistics
     */
    std::string end_capture();

    /**
     * @brief RAII helper for automatic capture management
     *
     * @details Ensures capture is properly ended even if exceptions occur.
     *
     * Usage:
     * ```cpp
     * {
     *     UnifiedLogger::ScopedCapture capture("output.log");
     *     // All output here goes only to output.log
     * } // Automatically ends capture
     * ```
     */
    class ScopedCapture {
    private:
        UnifiedLogger& logger;
        std::string filename_created;

    public:
        ScopedCapture(const std::string& filename, bool suppress_non_zero = false);
        ~ScopedCapture();

        bool file_was_created() const {
            return !filename_created.empty();
        }
        const std::string& get_created_filename() const {
            return filename_created;
        }
    };

    /**
     * @brief Generate standardized material log filename
     *
     * @param model_type Type of model (e.g., "ExaCMech", "UMAT")
     * @param region_id Material region index
     * @param context Optional context (e.g., "step_50")
     * @return Formatted filename like "material_ExaCMech_region_0_step_50_rank_3.log"
     */
    std::string get_material_log_filename(const std::string& model_type,
                                          int region_id,
                                          const std::string& context = "");

    /**
     * @brief Print summary of capture statistics
     *
     * @details Shows how many captures were performed and which files
     * were created. Only rank 0 prints to avoid duplication.
     */
    void print_capture_statistics();

    /**
     * @brief Execute code with output suppressed on non-zero ranks
     *
     * @param func Function/lambda to execute
     *
     * @details On rank 0, executes normally. On other ranks, redirects
     * all output to /dev/null during execution.
     */
    template <typename Func>
    void execute_on_rank_zero_only(Func&& func) {
        if (mpi_rank == 0) {
            // Rank 0: execute normally without any capture
            func();
        } else {
            // Non-zero ranks: suppress output
            std::stringstream ss;
            ss << "mfem_logging" << "_rank_" << mpi_rank << ".log";

            ScopedCapture suppress(ss.str());
            func();
        }
    }

    /**
     * @brief Clean shutdown of logging system
     *
     * @details Call before MPI_Finalize(). Performs:
     * - Ends any active captures
     * - Prints statistics
     * - Writes footer with timestamp
     * - Disables tee functionality
     * - Closes all files
     */
    void shutdown();
};

// ============================================================================
// CONVENIENCE MACROS
// ============================================================================

/**
 * @brief Create scoped capture with automatic cleanup
 *
 * Usage: SCOPED_CAPTURE("filename.log");
 * The capture lasts until end of current scope.
 */
#define SCOPED_CAPTURE(filename, ...) \
    exaconstit::UnifiedLogger::ScopedCapture _capture(filename, ##__VA_ARGS__)
} // namespace exaconstit

/**
 * @brief MFEM macros that only print on rank 0
 *
 * These suppress output on non-zero ranks while still executing
 * the underlying macro (preserving side effects like MPI_Abort).
 */
#define MFEM_WARNING_0(...)                                                     \
    exaconstit::UnifiedLogger::get_instance().execute_on_rank_zero_only([&]() { \
        MFEM_WARNING(__VA_ARGS__);                                              \
    })

#define MFEM_ABORT_0(...)                                                       \
    exaconstit::UnifiedLogger::get_instance().execute_on_rank_zero_only([&]() { \
        MFEM_ABORT(__VA_ARGS__);                                                \
    })

#define MFEM_VERIFY_0(condition, ...)                                                   \
    do {                                                                                \
        if (!(condition)) {                                                             \
            exaconstit::UnifiedLogger::get_instance().execute_on_rank_zero_only([&]() { \
                MFEM_VERIFY(false, __VA_ARGS__);                                        \
            });                                                                         \
        }                                                                               \
    } while (0)

#define MFEM_ASSERT_0(condition, ...)                                                   \
    do {                                                                                \
        if (!(condition)) {                                                             \
            exaconstit::UnifiedLogger::get_instance().execute_on_rank_zero_only([&]() { \
                MFEM_ASSERT(false, __VA_ARGS__);                                        \
            });                                                                         \
        }                                                                               \
    } while (0)
