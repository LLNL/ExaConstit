#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <optional>
#include <memory>
#include <functional>
#include <array>
#include <filesystem>
#include <numeric>

#include "TOML_Reader/toml.hpp"

// Enumeration types
enum class MeshType { AUTO, FILE, NOTYPE };
enum class TimeStepType { FIXED, AUTO, CUSTOM, NOTYPE };
enum class OriType { EULER, QUAT, CUSTOM, NOTYPE };
enum class MechType { UMAT, EXACMECH, NOTYPE };
enum class RTModel { CPU, OPENMP, GPU, NOTYPE };
enum class AssemblyType { FULL, PA, EA, NOTYPE };
enum class IntegrationModel { DEFAULT, BBAR, NOTYPE };
enum class LinearSolverType { CG, GMRES, MINRES, NOTYPE };
enum class NonlinearSolverType { NR, NRLS, NOTYPE };
enum class PreconditionerType { JACOBI, AMG, NOTYPE };

using map_of_imap = std::unordered_map<std::string, 
std::unordered_map<int, std::vector<int>>>;

// Mesh configuration options
struct MeshOptions {
    MeshType mesh_type = MeshType::FILE;
    std::string mesh_file;
    
    // Auto mesh generation
    std::array<int, 3> nxyz = {1, 1, 1};
    std::array<double, 3> mxyz = {1.0, 1.0, 1.0};
    
    // Refinement
    int ref_ser = 0;
    int ref_par = 0;
    int order = 1;
    
    // Periodicity
    bool periodicity = false;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static MeshOptions from_toml(const toml::value& toml_input);
};

// Grain information for crystal plasticity models
struct GrainInfo {
    // Optional files for grain data
    std::optional<std::string> orientation_file;
    std::optional<std::string> grain_file;

    
    // Orientation parameters
    int ori_state_var_loc = -1;
    int ori_stride = 0;
    OriType ori_type = OriType::QUAT;
    int num_grains = 0;

    // Validation
    bool validate() const;
    
    // Conversion from toml
    static GrainInfo from_toml(const toml::value& toml_input);
};

// Material properties configuration
struct MaterialProperties {
    std::string properties_file;
    int num_props = 0;
    std::vector<double> properties;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static MaterialProperties from_toml(const toml::value& toml_input);
};

// State variables configuration
struct StateVariables {
    std::string state_file;
    int num_vars = 0;
    std::vector<double> initial_values;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static StateVariables from_toml(const toml::value& toml_input);
};

// UMAT-specific options
struct UmatOptions {
    // Existing fields
    std::string library_path;
    std::string function_name = "umat_call";  // Default function name
    bool thermal = false;
    
    // New dynamic loading fields
    std::string load_strategy = "persistent";  // "persistent", "load_on_setup", "lazy_load"
    bool enable_dynamic_loading = true;        // Enable/disable dynamic loading
    std::vector<std::string> search_paths;     // Additional search paths for libraries
    
    // Validation
    bool validate() const;
    
    // Helper to convert string to LoadStrategy enum
    bool isValidLoadStrategy() const;
    
    // Conversion from toml
    static UmatOptions from_toml(const toml::value& toml_input);
};

// ExaCMech-specific options
struct ExaCMechModelOptions {
    // Modern approach - direct shortcut specification
    std::string shortcut;

    int gdot_size = 0;
    int hard_size = 0;
    
    // Legacy approach - these are used to derive the shortcut if not specified
    std::string xtal_type;   // FCC, BCC, or HCP
    std::string slip_type;   // PowerVoce, PowerVoceNL, or MTSDD
    
    // Validation
    bool validate() const;
    
    // Get the effective shortcut name (either directly specified or derived from legacy fields)
    std::string getEffectiveShortcut() const;
    
    // Static conversion from TOML
    static ExaCMechModelOptions from_toml(const toml::value& toml_input);
};

// Material model options
struct MaterialModelOptions {
    // Common model parameters
    bool crystal_plasticity = true;
    
    // Model-specific options
    std::optional<UmatOptions> umat;
    std::optional<ExaCMechModelOptions> exacmech;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static MaterialModelOptions from_toml(const toml::value& toml_input);
};

// Material options for a specific material/region
struct MaterialOptions {
    // Material identification
    std::string material_name = "default";
    int region_id = 0;
    MechType mech_type = MechType::NOTYPE;
    
    // Material data
    MaterialProperties properties;
    StateVariables state_vars;
    std::optional<GrainInfo> grain_info;
    MaterialModelOptions model;
    
    // Temperature
    double temperature = 298.0;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static MaterialOptions from_toml(const toml::value& toml_input);
    static std::vector<MaterialOptions> from_toml_array(const toml::value& toml_input);
};

// Time stepping configuration
struct TimeOptions {
    // Common options
    TimeStepType time_type = TimeStepType::FIXED;
    
    // Auto time stepping options
    struct AutoTimeOptions {
        double dt_start = 0.1;
        double dt_min = 0.05;
        double dt_max = 1e9;
        double dt_scale = 0.25;
        double t_final = 1.0;
        std::string auto_dt_file = "auto_dt_out.txt";
        
        static AutoTimeOptions from_toml(const toml::value& toml_input);
    };
    
    // Fixed time stepping options
    struct FixedTimeOptions {
        double dt = 1.0;
        double t_final = 1.0;
        
        static FixedTimeOptions from_toml(const toml::value& toml_input);
    };
    
    // Custom time stepping options
    struct CustomTimeOptions {
        int nsteps = 1;
        std::string floc = "custom_dt.txt";
        std::vector<double> dt_values;  // Populated from file during validation
        
        static CustomTimeOptions from_toml(const toml::value& toml_input);
        bool load_custom_dt_values();
    };
    
    // The actual options for each time stepping mode
    std::optional<AutoTimeOptions> auto_time;
    std::optional<FixedTimeOptions> fixed_time;
    std::optional<CustomTimeOptions> custom_time;
    
    // Restart options - common to all time stepping modes
    bool restart = false;
    double restart_time = 0.0;
    size_t restart_cycle = 0;
    
    // Determine which time stepping mode is active based on priority:
    // 1. Custom  2. Auto  3. Fixed
    void determine_time_type();
    
    // Static conversion from TOML
    static TimeOptions from_toml(const toml::value& toml_input);
    
    // Validation
    bool validate();
};

// Linear solver options
struct LinearSolverOptions {
    LinearSolverType solver_type = LinearSolverType::CG;
    PreconditionerType preconditioner = PreconditionerType::JACOBI;
    double abs_tol = 1e-30;
    double rel_tol = 1e-10;
    int max_iter = 1000;
    int print_level = 0;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static LinearSolverOptions from_toml(const toml::value& toml_input);
};

// Nonlinear solver options
struct NonlinearSolverOptions {
    int iter = 25;
    double rel_tol = 1e-5;
    double abs_tol = 1e-10;
    NonlinearSolverType nl_solver = NonlinearSolverType::NR;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static NonlinearSolverOptions from_toml(const toml::value& toml_input);
};

// Solver configuration
struct SolverOptions {
    // Assembly options
    AssemblyType assembly = AssemblyType::FULL;
    RTModel rtmodel = RTModel::CPU;
    
    // Integration model
    IntegrationModel integ_model = IntegrationModel::DEFAULT;
    
    // Solver options
    LinearSolverOptions linear_solver;
    NonlinearSolverOptions nonlinear_solver;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static SolverOptions from_toml(const toml::value& toml_input);
};

// Time-dependent boundary condition data
struct BCTimeInfo {
    bool time_dependent = false;
    bool cycle_dependent = false;

    std::vector<double> times;
    std::vector<int> cycles;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static BCTimeInfo from_toml(const toml::value& toml_input);
};

// Velocity boundary condition
struct VelocityBC {
    std::vector<int> essential_ids;
    std::vector<int> essential_comps;
    std::vector<double> essential_vals;

    // Validation
    bool validate() const;

    // Conversion from toml
    static VelocityBC from_toml(const toml::value& toml_input);
};

// Velocity gradient boundary condition
struct VelocityGradientBC {
    std::vector<double> velocity_gradient;
    std::vector<int> essential_comps;
    std::vector<int> essential_ids;
    BCTimeInfo time_info;
    std::optional<std::array<double, 3>> origin; // Origin point for velocity gradient
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static VelocityGradientBC from_toml(const toml::value& toml_input);
};

struct LegacyBC {
    bool changing_ess_bcs = false;
    std::vector<int> update_steps = {1};
    
    // These can be either flat vectors (for constant BCs) or nested vectors (for time-dependent BCs)
    std::variant<
        std::vector<int>, 
        std::vector<std::vector<int>>
    > essential_ids;
    
    std::variant<
        std::vector<int>, 
        std::vector<std::vector<int>>
    > essential_comps;
    
    std::variant<
        std::vector<double>, 
        std::vector<std::vector<double>>
    > essential_vals;
    
    std::variant<
        std::vector<std::vector<double>>, 
        std::vector<std::vector<std::vector<double>>>
    > essential_vel_grad;
    
    std::vector<double> vgrad_origin = {0.0, 0.0, 0.0};
};

// Boundary conditions configuration
struct BoundaryOptions {
    // Modern structurexd approach
    std::vector<VelocityBC> velocity_bcs;
    std::vector<VelocityGradientBC> vgrad_bcs;
    // Legacy format support for direct compatibility
    LegacyBC legacy_bcs;
    // Maps for BCManager compatibility (populated during validation)
    using map_of_imap = std::unordered_map<std::string, 
                        std::unordered_map<int, std::vector<int>>>;
    
    std::unordered_map<int, std::vector<double>> map_ess_vel;
    std::unordered_map<int, std::vector<double>> map_ess_vgrad;
    map_of_imap map_ess_comp;
    map_of_imap map_ess_id;

    std::vector<int> update_steps;
    BCTimeInfo time_info;


    // Transform raw BC data into structured format during validation
    bool validate();
    
    // Transform legacy flat arrays into structured VelocityBC objects
    void transformLegacyFormat();
    
    // Populate the map structures expected by BCManager
    void populateBCManagerMaps();

    // Helper method to create BC objects from legacy arrays
    void createBoundaryConditions(int step, 
                                  const std::vector<int>& ess_ids,
                                  const std::vector<int>& ess_comps,
                                  const std::vector<double>& essential_vals,
                                  const std::vector<std::vector<double>>& essential_vel_grad);

    // Conversion from toml
    static BoundaryOptions from_toml(const toml::value& toml_input);
};

// Visualization options for lattice orientation
struct LightUpOptions {
    bool enabled = false;

    std::string material_name = "";  // Name to match with MaterialOptions::material_name
    std::optional<int> region_id;    // Will be resolved during validation

    std::vector<std::array<double, 3>> hkl_directions;
    double distance_tolerance = 0.0873;
    std::array<double, 3> sample_direction = {0.0, 0.0, 1.0};
    std::array<double, 3> lattice_parameters = {3.6, 3.6, 3.6};
    std::string lattice_basename = "lattice_avg_";
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static LightUpOptions from_toml(const toml::value& toml_input);
    // Conversion from toml with legacy check
    static std::vector<LightUpOptions> from_toml_with_legacy(const toml::value& toml_input);

    // NEW: Method to resolve material_name to region_id
    bool resolve_region_id(const std::vector<MaterialOptions>& materials);
};

// Visualization and output options
struct VisualizationOptions {
    bool visit = false;
    bool paraview = false;
    bool conduit = false;
    bool adios2 = false;
    
    int output_frequency = 1;
    
    // Output file locations
    std::string floc = "results/exaconstit";
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static VisualizationOptions from_toml(const toml::value& toml_input);
};

// Volume average calculation options
struct VolumeAverageOptions {

    std::string avg_stress_fname = "avg_stress.txt";
    std::string avg_def_grad_fname = "avg_def_grad.txt";
    std::string avg_pl_work_fname = "avg_pl_work.txt";
    std::string avg_eq_pl_strain_fname = "avg_eq_pl_strain.txt";
    std::string avg_euler_strain_fname = "avg_euler_strain.txt";
    std::string avg_elastic_strain_fname = "avg_elastic_strain.txt";

    bool enabled = true;
    bool stress = true;
    bool def_grad = false;
    bool euler_strain = false;
    bool eq_pl_strain = false;
    bool plastic_work = false;
    // likely only ecmech based for this and not the other models
    bool elastic_strain = false;
    // Additional averages flag
    bool additional_avgs = false;
    
    std::string output_directory = "results/";
    int output_frequency = 1;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static VolumeAverageOptions from_toml(const toml::value& toml_input);

    // Conversion from toml with legacy check
    static VolumeAverageOptions from_toml_with_legacy(const toml::value& toml_input);
};

// Projection options for visualization
struct ProjectionOptions {
    std::vector<std::string> enabled_projections;
    bool auto_enable_compatible = true;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static ProjectionOptions from_toml(const toml::value& toml_input);
};

// Post-processing options
struct PostProcessingOptions {
    VolumeAverageOptions volume_averages;
    ProjectionOptions projections;
    // LightUp options
    std::vector<LightUpOptions> light_up_configs;
    
    // Validation
    bool validate() const;
    
    // Conversion from toml
    static PostProcessingOptions from_toml(const toml::value& toml_input);

    std::vector<LightUpOptions> get_enabled_light_up_configs() const;
    LightUpOptions* get_light_up_config_for_region(int region_id);
    const LightUpOptions* get_light_up_config_for_region(int region_id) const;
};

// Main options class
class ExaOptions {
public:
    // Core simulation metadata
    std::string basename = "exaconstit";
    std::string version = "0.8.0";
    
    // Structured option components
    MeshOptions mesh;
    TimeOptions time;
    SolverOptions solvers;
    VisualizationOptions visualization;
    std::vector<MaterialOptions> materials;
    BoundaryOptions boundary_conditions;
    PostProcessingOptions post_processing;
    
    // Configuration paths for modular approach
    std::vector<std::string> material_files;
    std::optional<std::string> post_processing_file;

    std::optional<std::string> orientation_file;
    std::optional<std::string> grain_file;
    std::optional<std::string> region_mapping_file;

    // Parse the main configuration file
    void parse_options(const std::string& filename, int my_id);
    
    // Core option parsing methods
    void parse_from_toml(const toml::value& toml_input);
    
    // Validation
    bool validate();

    // Print all options in a formatted way
    void print_options() const;
    
private:
    // Component parsers
    void parse_mesh_options(const toml::value& toml_input);
    void parse_time_options(const toml::value& toml_input);
    void parse_solver_options(const toml::value& toml_input);
    void parse_material_options(const toml::value& toml_input);
    void parse_boundary_options(const toml::value& toml_input);
    void parse_visualization_options(const toml::value& toml_input);
    void parse_post_processing_options(const toml::value& toml_input);
    
    // Helper to parse model options for a material
    void parse_model_options(const toml::value& toml_input, MaterialOptions& material);
    
    // Modular file handling
    void load_material_files();
    void load_post_processing_file();

    // Helper print methods for each component
    void print_mesh_options() const;
    void print_time_options() const;
    void print_solver_options() const;
    void print_material_options() const;
    void print_boundary_options() const;
    void print_visualization_options() const;
    void print_post_processing_options() const;
};

// Utility functions - string to enum conversion
MeshType string_to_mesh_type(const std::string& str);
TimeStepType string_to_time_step_type(const std::string& str);
MechType string_to_mech_type(const std::string& str);
RTModel string_to_rt_model(const std::string& str);
AssemblyType string_to_assembly_type(const std::string& str);
IntegrationModel string_to_integration_model(const std::string& str);
LinearSolverType string_to_linear_solver_type(const std::string& str);
NonlinearSolverType string_to_nonlinear_solver_type(const std::string& str);
PreconditionerType string_to_preconditioner_type(const std::string& str);
OriType string_to_ori_type(const std::string& str);