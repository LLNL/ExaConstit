#include "options/option_parser_v2.hpp"

#include "TOML_Reader/toml.hpp"
#include "mfem.hpp"
#include "ECMech_cases.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

// Utility functions for parsing TOML
namespace {

// Convert string to enum with validation
template<typename EnumType>
EnumType string_to_enum(const std::string& str, 
                        const std::map<std::string, EnumType>& mapping,
                        EnumType default_value,
                        const std::string& enum_name) {
    auto it = mapping.find(str);
    if (it != mapping.end()) {
        return it->second;
    }
    
    std::cerr << "Warning: Unknown " << enum_name << " type '" << str 
              << "', using default." << std::endl;
    return default_value;
}

// Load vector from file
std::vector<double> load_vector_from_file(const std::string& filename, int expected_size) {
    std::vector<double> result;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    double value;
    while (file >> value) {
        result.push_back(value);
    }
    
    if (expected_size > 0 && result.size() != static_cast<size_t>(expected_size)) {
        std::cerr << "Warning: File " << filename << " contains " << result.size() 
                  << " values, but " << expected_size << " were expected." << std::endl;
    }
    
    return result;
}

}  // anonymous namespace

// Mesh type conversion
MeshType string_to_mesh_type(const std::string& str) {
    static const std::map<std::string, MeshType> mapping = {
        {"file", MeshType::FILE},
        {"auto", MeshType::AUTO},
    };
    
    return string_to_enum(str, mapping, MeshType::NOTYPE, "mesh");
}

// Time step type conversion
TimeStepType string_to_time_step_type(const std::string& str) {
    static const std::map<std::string, TimeStepType> mapping = {
        {"fixed", TimeStepType::FIXED},
        {"auto", TimeStepType::AUTO},
        {"custom", TimeStepType::CUSTOM}
    };
    
    return string_to_enum(str, mapping, TimeStepType::NOTYPE, "time step");
}

// Orientation type conversion
OriType string_to_ori_type(const std::string& str) {
    static const std::map<std::string, OriType> mapping = {
        {"quat", OriType::QUAT},
        {"custom", OriType::CUSTOM},
        {"euler", OriType::EULER}
    };
    
    return string_to_enum(str, mapping, OriType::NOTYPE, "orientation type");
}

// Material model type conversion
MechType string_to_mech_type(const std::string& str) {
    static const std::map<std::string, MechType> mapping = {
        {"umat", MechType::UMAT},
        {"exacmech", MechType::EXACMECH}
    };
    
    return string_to_enum(str, mapping, MechType::NOTYPE, "material model");
}

// Runtime model conversion
RTModel string_to_rt_model(const std::string& str) {
    static const std::map<std::string, RTModel> mapping = {
        {"CPU", RTModel::CPU},
        {"OPENMP", RTModel::OPENMP},
        {"GPU", RTModel::GPU}
    };
    
    return string_to_enum(str, mapping, RTModel::NOTYPE, "runtime model");
}

// Assembly type conversion
AssemblyType string_to_assembly_type(const std::string& str) {
    static const std::map<std::string, AssemblyType> mapping = {
        {"FULL", AssemblyType::FULL},
        {"PA", AssemblyType::PA},
        {"EA", AssemblyType::EA}
    };
    
    return string_to_enum(str, mapping, AssemblyType::NOTYPE, "assembly");
}

// Integration model conversion
IntegrationModel string_to_integration_model(const std::string& str) {
    static const std::map<std::string, IntegrationModel> mapping = {
        {"FULL", IntegrationModel::DEFAULT},
        {"BBAR", IntegrationModel::BBAR}
    };
    
    return string_to_enum(str, mapping, IntegrationModel::NOTYPE, "integration model");
}

// Linear solver type conversion
LinearSolverType string_to_linear_solver_type(const std::string& str) {
    static const std::map<std::string, LinearSolverType> mapping = {
        {"CG", LinearSolverType::CG},
        {"GMRES", LinearSolverType::GMRES},
        {"MINRES", LinearSolverType::MINRES}
    };
    
    return string_to_enum(str, mapping, LinearSolverType::NOTYPE, "linear solver");
}

// Nonlinear solver type conversion
NonlinearSolverType string_to_nonlinear_solver_type(const std::string& str) {
    static const std::map<std::string, NonlinearSolverType> mapping = {
        {"NR", NonlinearSolverType::NR},
        {"NRLS", NonlinearSolverType::NRLS}
    };
    
    return string_to_enum(str, mapping, NonlinearSolverType::NOTYPE, "nonlinear solver");
}

// Preconditioner type conversion
PreconditionerType string_to_preconditioner_type(const std::string& str) {
    static const std::map<std::string, PreconditionerType> mapping = {
        {"JACOBI", PreconditionerType::JACOBI},
        {"AMG", PreconditionerType::AMG}
    };
    
    return string_to_enum(str, mapping, PreconditionerType::NOTYPE, "preconditioner");
}

// Implementation of the struct conversion methods

MeshOptions MeshOptions::from_toml(const toml::value& toml_input) {
    MeshOptions options;
    
    if (toml_input.contains("type")) {
        options.mesh_type = string_to_mesh_type(
            toml::find<std::string>(toml_input, "type"));
    }
    
    if (options.mesh_type == MeshType::FILE) {
        if (toml_input.contains("floc")) {
            options.mesh_file = toml::find<std::string>(toml_input, "floc");
        }
    }
    
    if (toml_input.contains("refine_serial") || toml_input.contains("ref_ser")) {
        const auto& key = toml_input.contains("refine_serial") ? "refine_serial" : "ref_ser";
        options.ref_ser = toml::find<int>(toml_input, key);
    }
    
    if (toml_input.contains("refine_parallel") || toml_input.contains("ref_par")) {
        const auto& key = toml_input.contains("refine_parallel") ? "refine_parallel" : "ref_par";
        options.ref_par = toml::find<int>(toml_input, key);
    }
    
    if (toml_input.contains("order") || toml_input.contains("p_refinement")) {
        // Support both "order" and "p_refinement" for backward compatibility
        const auto& key = toml_input.contains("order") ? "order" : "p_refinement";
        options.order = toml::find<int>(toml_input, key);
    }
    
    if (toml_input.contains("periodicity")) {
        options.periodicity = toml::find<bool>(toml_input, "periodicity");
    }

    // Handle Auto mesh section
    if (options.mesh_type == MeshType::AUTO) {
        auto auto_section = toml::find(toml_input, "Auto");
        if (auto_section.contains("length") || auto_section.contains("mxyz")) {
            const auto& key = toml_input.contains("length") ? "length" : "mxyz";
            auto length_array = toml::find<std::vector<double>>(auto_section, key);
            if (length_array.size() >= 3) {
                std::copy_n(length_array.begin(), 3, options.mxyz.begin());
            }
        }

        if (auto_section.contains("ncuts") || auto_section.contains("mxyz")) {
            const auto& key = toml_input.contains("ncuts") ? "ncuts" : "nxyz";
            auto ncuts_array = toml::find<std::vector<int>>(auto_section, key);
            if (ncuts_array.size() >= 3) {
                std::copy_n(ncuts_array.begin(), 3, options.nxyz.begin());
            }
        }
    }

    return options;
}

GrainInfo GrainInfo::from_toml(const toml::value& toml_input) {
    GrainInfo info;

    if (toml_input.contains("orientation_file")) {
        info.orientation_file = toml::find<std::string>(toml_input, "orientation_file");
    }

    if (toml_input.contains("ori_state_var_loc")) {
        info.ori_state_var_loc = toml::find<int>(toml_input, "ori_state_var_loc");
    }
    
    if (toml_input.contains("ori_stride")) {
        info.ori_stride = toml::find<int>(toml_input, "ori_stride");
    }
    
    if (toml_input.contains("ori_type")) {
        info.ori_type = string_to_ori_type(toml::find<std::string>(toml_input, "ori_type"));
    }
    
    if (toml_input.contains("num_grains")) {
        info.num_grains = toml::find<int>(toml_input, "num_grains");
    }
    
    return info;
}

MaterialProperties MaterialProperties::from_toml(const toml::value& toml_input) {
    MaterialProperties props;
    
    if (toml_input.contains("floc")) {
        props.properties_file = toml::find<std::string>(toml_input, "floc");
    }
    
    if (toml_input.contains("num_props")) {
        props.num_props = toml::find<int>(toml_input, "num_props");
    }
    
    if (toml_input.contains("values")) {
        props.properties = toml::find<std::vector<double>>(toml_input, "values");
    } else if (!props.properties_file.empty() && props.num_props > 0) {
        // Load properties from file if specified and not already loaded
        try {
            props.properties = load_vector_from_file(props.properties_file, props.num_props);
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
        }
    }
    
    return props;
}

StateVariables StateVariables::from_toml(const toml::value& toml_input) {
    StateVariables vars;
    
    if (toml_input.contains("floc")) {
        vars.state_file = toml::find<std::string>(toml_input, "floc");
    }
    
    if (toml_input.contains("num_vars") || toml_input.contains("num_state_vars")) {
        // Support both "num_vars" and "num_state_vars" for backward compatibility
        const auto& key = toml_input.contains("num_vars") ? "num_vars" : "num_state_vars";
        vars.num_vars = toml::find<int>(toml_input, key);
    }
    
    if (toml_input.contains("values")) {
        vars.initial_values = toml::find<std::vector<double>>(toml_input, "values");
    } else if (!vars.state_file.empty() && vars.num_vars > 0) {
        // Load state variables from file if specified and not already loaded
        try {
            vars.initial_values = load_vector_from_file(vars.state_file, vars.num_vars);
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
        }
    }
    
    return vars;
}

UmatOptions UmatOptions::from_toml(const toml::value& toml_input) {
    UmatOptions options;
    
    if (toml_input.contains("library")) {
        options.library_path = toml::find<std::string>(toml_input, "library");
    }
    
    if (toml_input.contains("function")) {
        options.function_name = toml::find<std::string>(toml_input, "function");
    }
    
    if (toml_input.contains("thermal")) {
        options.thermal = toml::find<bool>(toml_input, "thermal");
    }
    
    return options;
}

std::string ExaCMechModelOptions::getEffectiveShortcut() const {
    if (!shortcut.empty()) {
        return shortcut;
    }
    
    // Derive shortcut from legacy fields
    if (xtal_type.empty() || slip_type.empty()) {
        return "";
    }
    
    std::string derived_shortcut = "evptn_" + xtal_type;
    
    // Map slip_type to the appropriate suffix
    if (xtal_type == "FCC" || xtal_type == "BCC") {
        if (slip_type == "PowerVoce") {
            derived_shortcut += "_A";
        }
        else if (slip_type == "PowerVoceNL") {
            derived_shortcut += "_AH";
        }
        else if (slip_type == "MTSDD") {
            derived_shortcut += "_B";
        }
    }
    else if (xtal_type == "HCP") {
        if (slip_type == "MTSDD") {
            derived_shortcut += "_A";
        }
    }
    
    return derived_shortcut;
}

ExaCMechModelOptions ExaCMechModelOptions::from_toml(const toml::value& toml_input) {
    ExaCMechModelOptions options;
    
    if (toml_input.contains("shortcut")) {
        options.shortcut = toml::find<std::string>(toml_input, "shortcut");
    }
    
    if (toml_input.contains("xtal_type")) {
        options.xtal_type = toml::find<std::string>(toml_input, "xtal_type");
    }
    
    if (toml_input.contains("slip_type")) {
        options.slip_type = toml::find<std::string>(toml_input, "slip_type");
    }
    
    return options;
}

MaterialModelOptions MaterialModelOptions::from_toml(const toml::value& toml_input) {
    MaterialModelOptions model_options;
    
    if (toml_input.contains("cp")) {
        model_options.crystal_plasticity = toml::find<bool>(toml_input, "cp");
    }
    
    // Parse UMAT-specific options
    if (toml_input.contains("UMAT")) {
        model_options.umat = UmatOptions::from_toml(toml::find(toml_input, "UMAT"));
    }
    
    // Parse ExaCMech-specific options
    if (toml_input.contains("ExaCMech")) {
        model_options.exacmech = ExaCMechModelOptions::from_toml(
            toml::find(toml_input, "ExaCMech"));
    }
    
    return model_options;
}

MaterialOptions MaterialOptions::from_toml(const toml::value& toml_input) {
    MaterialOptions options;
    
    if (toml_input.contains("name")) {
        options.material_name = toml::find<std::string>(toml_input, "name");
    }
    
    if (toml_input.contains("region_id")) {
        options.region_id = toml::find<int>(toml_input, "region_id");
    }
    
    if (toml_input.contains("mech_type")) {
        options.mech_type = string_to_mech_type(
            toml::find<std::string>(toml_input, "mech_type"));
    }
    
    if (toml_input.contains("temperature")) {
        options.temperature = toml::find<double>(toml_input, "temperature");
    }
    
    // Parse material properties section
    if (toml_input.contains("Properties") || toml_input.contains("Matl_Props")) {
        // Support both naming conventions
        const auto& props_key = toml_input.contains("Properties") ? "Properties" : "Matl_Props";
        options.properties = MaterialProperties::from_toml(
            toml::find(toml_input, props_key));
    }

    // Parse state variables section
    if (toml_input.contains("State_Vars")) {
        options.state_vars = StateVariables::from_toml(
            toml::find(toml_input, "State_Vars"));
    }
    
    // Parse grain information section
    if (toml_input.contains("Grain")) {
        options.grain_info = GrainInfo::from_toml(
            toml::find(toml_input, "Grain"));
    }
    
    // Parse model-specific options
    if (toml_input.contains("Model")) {
        options.model = MaterialModelOptions::from_toml(
            toml::find(toml_input, "Model"));
    }
    
    return options;
}

std::vector<MaterialOptions> MaterialOptions::from_toml_array(const toml::value& toml_input) {
    std::vector<MaterialOptions> materials;
    
    // Check if we have an array of materials
    if (toml_input.is_array()) {
        const auto& arr = toml_input.as_array();
        for (const auto& item : arr) {
            materials.push_back(MaterialOptions::from_toml(item));
        }
    } 
    // If it's a single table, parse it as one material
    else if (toml_input.is_table()) {
        materials.push_back(MaterialOptions::from_toml(toml_input));
    }
    
    return materials;
}

// Time options nested classes implementation
TimeOptions::AutoTimeOptions TimeOptions::AutoTimeOptions::from_toml(const toml::value& toml_input) {
    AutoTimeOptions options;
    
    if (toml_input.contains("dt_start")) {
        options.dt_start = toml::find<double>(toml_input, "dt_start");
    }
    
    if (toml_input.contains("dt_min")) {
        options.dt_min = toml::find<double>(toml_input, "dt_min");
    }
    
    if (toml_input.contains("dt_max")) {
        options.dt_max = toml::find<double>(toml_input, "dt_max");
    }
    
    if (toml_input.contains("dt_scale")) {
        options.dt_scale = toml::find<double>(toml_input, "dt_scale");
    }
    
    if (toml_input.contains("t_final")) {
        options.t_final = toml::find<double>(toml_input, "t_final");
    }
    
    if (toml_input.contains("auto_dt_file")) {
        options.auto_dt_file = toml::find<std::string>(toml_input, "auto_dt_file");
    }
    
    return options;
}

TimeOptions::FixedTimeOptions TimeOptions::FixedTimeOptions::from_toml(const toml::value& toml_input) {
    FixedTimeOptions options;
    
    if (toml_input.contains("dt")) {
        options.dt = toml::find<double>(toml_input, "dt");
    }
    
    if (toml_input.contains("t_final")) {
        options.t_final = toml::find<double>(toml_input, "t_final");
    }
    
    return options;
}

TimeOptions::CustomTimeOptions TimeOptions::CustomTimeOptions::from_toml(const toml::value& toml_input) {
    CustomTimeOptions options;
    
    if (toml_input.contains("nsteps")) {
        options.nsteps = toml::find<int>(toml_input, "nsteps");
    }
    
    if (toml_input.contains("floc")) {
        options.floc = toml::find<std::string>(toml_input, "floc");
    }
    
    return options;
}

bool TimeOptions::CustomTimeOptions::load_custom_dt_values() {
    try {
        std::ifstream file(floc);
        if (!file.is_open()) {
            return false;
        }
        
        dt_values.clear();
        double value;
        while (file >> value) {
            dt_values.push_back(value);
        }
        
        return dt_values.size() >= static_cast<size_t>(nsteps);
    } catch (...) {
        return false;
    }
}

void TimeOptions::determine_time_type() {
    if (custom_time.has_value()) {
        time_type = TimeStepType::CUSTOM;
    } else if (auto_time.has_value()) {
        time_type = TimeStepType::AUTO;
    } else if (fixed_time.has_value()) {
        time_type = TimeStepType::FIXED;
    } else {
        // Default to fixed with defaults
        fixed_time = FixedTimeOptions{};
        time_type = TimeStepType::FIXED;
    }
}

TimeOptions TimeOptions::from_toml(const toml::value& toml_input) {
    TimeOptions options;
    
    // Check for restart options
    if (toml_input.contains("restart")) {
        options.restart = toml::find<bool>(toml_input, "restart");
    }
    
    if (toml_input.contains("restart_time")) {
        options.restart_time = toml::find<double>(toml_input, "restart_time");
    }
    
    if (toml_input.contains("restart_cycle")) {
        options.restart_cycle = toml::find<size_t>(toml_input, "restart_cycle");
    }
    
    // Check for nested time stepping sections
    if (toml_input.contains("Auto")) {
        options.auto_time = AutoTimeOptions::from_toml(
            toml::find(toml_input, "Auto"));
    }
    
    if (toml_input.contains("Fixed")) {
        options.fixed_time = FixedTimeOptions::from_toml(
            toml::find(toml_input, "Fixed"));
    }
    
    if (toml_input.contains("Custom")) {
        options.custom_time = CustomTimeOptions::from_toml(
            toml::find(toml_input, "Custom"));
    }
    
    // Determine which time stepping mode to use
    options.determine_time_type();
    
    return options;
}

LinearSolverOptions LinearSolverOptions::from_toml(const toml::value& toml_input) {
    LinearSolverOptions options;
    
    if (toml_input.contains("solver") || toml_input.contains("solver_type")) {
        // Support both naming conventions
        const auto& solver_key = toml_input.contains("solver") ? "solver" : "solver_type";
        options.solver_type = string_to_linear_solver_type(
            toml::find<std::string>(toml_input, solver_key));
    }
    
    if (toml_input.contains("preconditioner")) {
        options.preconditioner = string_to_preconditioner_type(
            toml::find<std::string>(toml_input, "preconditioner"));
    }
    
    if (toml_input.contains("abs_tol")) {
        options.abs_tol = toml::find<double>(toml_input, "abs_tol");
    }
    
    if (toml_input.contains("rel_tol")) {
        options.rel_tol = toml::find<double>(toml_input, "rel_tol");
    }
    
    if (toml_input.contains("max_iter") || toml_input.contains("iter")) {
        // Support both naming conventions
        const auto& iter_key = toml_input.contains("max_iter") ? "max_iter" : "iter";
        options.max_iter = toml::find<int>(toml_input, iter_key);
    }
    
    if (toml_input.contains("print_level")) {
        options.print_level = toml::find<int>(toml_input, "print_level");
    }
    
    return options;
}

NonlinearSolverOptions NonlinearSolverOptions::from_toml(const toml::value& toml_input) {
    NonlinearSolverOptions options;
    
    if (toml_input.contains("iter")) {
        options.iter = toml::find<int>(toml_input, "iter");
    }
    
    if (toml_input.contains("rel_tol")) {
        options.rel_tol = toml::find<double>(toml_input, "rel_tol");
    }
    
    if (toml_input.contains("abs_tol")) {
        options.abs_tol = toml::find<double>(toml_input, "abs_tol");
    }
    
    if (toml_input.contains("nl_solver")) {
        options.nl_solver = string_to_nonlinear_solver_type(toml::find<std::string>(toml_input, "nl_solver"));
    }
    
    return options;
}

SolverOptions SolverOptions::from_toml(const toml::value& toml_input) {
    SolverOptions options;
    
    if (toml_input.contains("assembly")) {
        options.assembly = string_to_assembly_type(
            toml::find<std::string>(toml_input, "assembly"));
    }
    
    if (toml_input.contains("rtmodel")) {
        options.rtmodel = string_to_rt_model(
            toml::find<std::string>(toml_input, "rtmodel"));
    }
    
    if (toml_input.contains("integ_model")) {
        options.integ_model = string_to_integration_model(
            toml::find<std::string>(toml_input, "integ_model"));
    }
    
    // Parse linear solver section
    if (toml_input.contains("Krylov")) {
        options.linear_solver = LinearSolverOptions::from_toml(
            toml::find(toml_input, "Krylov"));
    }
    
    // Parse nonlinear solver section (NR = Newton-Raphson)
    if (toml_input.contains("NR")) {
        options.nonlinear_solver = NonlinearSolverOptions::from_toml(
            toml::find(toml_input, "NR"));
    }
    
    return options;
}

BCTimeInfo BCTimeInfo::from_toml(const toml::value& toml_input) {
    BCTimeInfo info;
    
    if (toml_input.contains("time_dependent")) {
        info.time_dependent = toml::find<bool>(toml_input, "time_dependent");
    }

    if (toml_input.contains("cycle_dependent")) {
        info.cycle_dependent = toml::find<bool>(toml_input, "cycle_dependent");
    }
    
    if (toml_input.contains("times")) {
        info.times = toml::find<std::vector<double>>(toml_input, "times");
    }

    if (toml_input.contains("cycles")) {
        info.cycles = toml::find<std::vector<int>>(toml_input, "cycles");
    }

    return info;
}

VelocityBC VelocityBC::from_toml(const toml::value& toml_input) {
    VelocityBC bc;
    
    if (toml_input.contains("essential_ids")) {
        bc.essential_ids = toml::find<std::vector<int>>(toml_input, "essential_ids");
    }
    
    if (toml_input.contains("essential_comps")) {
        bc.essential_comps = toml::find<std::vector<int>>(toml_input, "essential_comps");
    }
    
    if (toml_input.contains("essential_vals")) {
        bc.essential_vals = toml::find<std::vector<double>>(toml_input, "essential_vals");
    }

    return bc;
}

VelocityGradientBC VelocityGradientBC::from_toml(const toml::value& toml_input) {
    VelocityGradientBC bc;
    
    if (toml_input.contains("velocity_gradient")) {
        bc.velocity_gradient = toml::find<std::vector<double>>(toml_input, "velocity_gradient");
    }
    
    if (toml_input.contains("essential_ids")) {
        bc.essential_ids = toml::find<std::vector<int>>(toml_input, "essential_ids");
    }

    if (toml_input.contains("origin")) {
        auto origin = toml::find<std::vector<double>>(toml_input, "origin");
        if (origin.size() >= 3) {
            bc.origin = std::array<double, 3>{origin[0], origin[1], origin[2]};
        }
    }

    return bc;
}

bool BoundaryOptions::validate() {
    // For simplicity, use the legacy format if velocity_bcs is empty
    auto is_empty = [](auto && arg) -> bool {
        return std::visit([](auto&& arg)->bool {
            return arg.empty();
        }, arg);
    };

    if (velocity_bcs.empty() && !is_empty(legacy_bcs.essential_ids)) {
        transformLegacyFormat();
    }
    
    // Populate BCManager-compatible maps
    populateBCManagerMaps();
    
    return true;
}

void BoundaryOptions::transformLegacyFormat() {
    // Skip if we don't have legacy data
    auto is_empty = [](auto && arg) -> bool {
        return std::visit([](auto&& arg)->bool {
            return arg.empty();
        }, arg);
    };

    if (is_empty(legacy_bcs.essential_ids) || is_empty(legacy_bcs.essential_comps)) {
        return;
    }
    
    // First, ensure update_steps includes 1 (required for initialization)
    if (legacy_bcs.update_steps.empty() || 
        std::find(legacy_bcs.update_steps.begin(), legacy_bcs.update_steps.end(), 1) == legacy_bcs.update_steps.end()) {
        legacy_bcs.update_steps.insert(legacy_bcs.update_steps.begin(), 1);
    }
    
    // Transfer update_steps to the object field
    update_steps = legacy_bcs.update_steps;
    
    // Handle time-dependent BCs case
    if (legacy_bcs.changing_ess_bcs) {
        // We need to match nested structures: 
        // For each update step, we need corresponding essential_ids, essential_comps, etc.
        
        // Validate that array sizes match number of update steps
        const size_t num_steps = legacy_bcs.update_steps.size();
        
        // We expect nested arrays for time-dependent BCs
        if (std::holds_alternative<std::vector<std::vector<int>>>(legacy_bcs.essential_ids)) {
            auto& nested_ess_ids = std::get<std::vector<std::vector<int>>>(legacy_bcs.essential_ids);
            auto& nested_ess_comps = std::get<std::vector<std::vector<int>>>(legacy_bcs.essential_comps);
            auto& nested_ess_vals = std::get<std::vector<std::vector<double>>>(legacy_bcs.essential_vals);
            auto& nested_ess_vgrads = std::get<std::vector<std::vector<std::vector<double>>>>(legacy_bcs.essential_vel_grad);
            
            // Ensure sizes match
            if (nested_ess_ids.size() != num_steps || nested_ess_comps.size() != num_steps) {
                throw std::runtime_error("Mismatch in sizes of BC arrays vs. update_steps");
            }
            
            // Process each time step
            for (size_t i = 0; i < num_steps; ++i) {
                const int step = legacy_bcs.update_steps[i];
                const auto& ess_ids    = nested_ess_ids[i];
                const auto& ess_comps  = nested_ess_comps[i];
                const auto& ess_vals   = nested_ess_vals[i];
                const auto& ess_vgrads = nested_ess_vgrads[i];
                
                // Create BCs for this time step
                createBoundaryConditions(step, ess_ids, ess_comps, ess_vals, ess_vgrads);
            }
        }
    }
    // Simple case: constant BCs
    else {
        // For non-changing BCs, we just have one set of values for all time steps
        createBoundaryConditions(1, 
                                 std::get<std::vector<int>>(legacy_bcs.essential_ids),
                                 std::get<std::vector<int>>(legacy_bcs.essential_comps),
                                 std::get<std::vector<double>>(legacy_bcs.essential_vals),
                                 std::get<std::vector<std::vector<double>>>(legacy_bcs.essential_vel_grad));
    }
}

// Helper method to create BC objects from legacy arrays
void BoundaryOptions::createBoundaryConditions(int step, 
                                               const std::vector<int>& ess_ids,
                                               const std::vector<int>& ess_comps,
                                               const std::vector<double>& essential_vals,
                                               const std::vector<std::vector<double>>& essential_vel_grad) {
    // Separate velocity and velocity gradient BCs
    std::vector<int> vel_ids, vel_comps, vgrad_ids, vgrad_comps;

    // Configure time dependency
    time_info.cycle_dependent = true;
    time_info.cycles.push_back(step);

    // Identify which BCs are velocity vs. velocity gradient
    for (size_t i = 0; i < ess_ids.size() && i < ess_comps.size(); ++i) {
        if (ess_comps[i] >= 0) {
            vel_ids.push_back(ess_ids[i]);
            vel_comps.push_back(ess_comps[i]);
        } else {
            vgrad_ids.push_back(ess_ids[i]);
            vgrad_comps.push_back(std::abs(ess_comps[i]));
        }
    }

    // Create velocity BC if needed
    if (!vel_ids.empty()) {
        VelocityBC vel_bc;
        vel_bc.essential_ids = vel_ids;
        vel_bc.essential_comps = vel_comps;
        
        // Find velocity values for this step
        if (essential_vals.size() >= vel_ids.size() * 3) {
            vel_bc.essential_vals = essential_vals;
        }
        velocity_bcs.push_back(vel_bc);
    }

    // Create velocity gradient BC if needed
    if (!vgrad_ids.empty()) {
        VelocityGradientBC vgrad_bc;
        vgrad_bc.essential_ids = vgrad_ids;
        
        // Find velocity gradient values for this step
        if (!essential_vel_grad.empty()) {
            // Flatten the 2D array to 1D
            for (const auto& row : essential_vel_grad) {
                vgrad_bc.velocity_gradient.insert(
                    vgrad_bc.velocity_gradient.end(), row.begin(), row.end());
            }
        }
        
        // Set origin if needed
        if (!legacy_bcs.vgrad_origin.empty() && legacy_bcs.vgrad_origin.size() >= 3) {
            vgrad_bc.origin = std::array<double, 3>{
                legacy_bcs.vgrad_origin[0],
                legacy_bcs.vgrad_origin[1],
                legacy_bcs.vgrad_origin[2]
            };
        }
        vgrad_bcs.push_back(vgrad_bc);
    }
}

void BoundaryOptions::populateBCManagerMaps() {
    // Initialize the map structures
    map_ess_comp["total"] = std::unordered_map<int, std::vector<int>>();
    map_ess_comp["ess_vel"] = std::unordered_map<int, std::vector<int>>();
    map_ess_comp["ess_vgrad"] = std::unordered_map<int, std::vector<int>>();
    
    map_ess_id["total"] = std::unordered_map<int, std::vector<int>>();
    map_ess_id["ess_vel"] = std::unordered_map<int, std::vector<int>>();
    map_ess_id["ess_vgrad"] = std::unordered_map<int, std::vector<int>>();
    
    // Default entry for step 0 (used for initialization)
    map_ess_comp["total"][0] = std::vector<int>();
    map_ess_comp["ess_vel"][0] = std::vector<int>();
    map_ess_comp["ess_vgrad"][0] = std::vector<int>();
    
    map_ess_id["total"][0] = std::vector<int>();
    map_ess_id["ess_vel"][0] = std::vector<int>();
    map_ess_id["ess_vgrad"][0] = std::vector<int>();
    
    map_ess_vel[0] = std::vector<double>();
    map_ess_vgrad[0] = std::vector<double>(9, 0.0);

    // Determine which step(s) this BC applies to
    std::vector<int> steps;
    if (time_info.cycle_dependent && !time_info.cycles.empty()) {
        update_steps = time_info.cycles;
    } else if (update_steps.empty()) {
        // Default to step 1
        update_steps = {1};
    }

    // Process velocity BCs
    for (const auto& vel_bc : velocity_bcs) {
        for (int step : update_steps) {
            // Initialize maps for this step if needed
            if (map_ess_comp["total"].find(step) == map_ess_comp["total"].end()) {
                map_ess_comp["total"][step] = std::vector<int>();
                map_ess_comp["ess_vel"][step] = std::vector<int>();
                map_ess_comp["ess_vgrad"][step] = std::vector<int>();
                
                map_ess_id["total"][step] = std::vector<int>();
                map_ess_id["ess_vel"][step] = std::vector<int>();
                map_ess_id["ess_vgrad"][step] = std::vector<int>();
                
                map_ess_vel[step] = std::vector<double>();
                map_ess_vgrad[step] = std::vector<double>(9, 0.0);
            }
            
            // Add this BC's data to the maps
            for (size_t i = 0; i < vel_bc.essential_ids.size() && i < vel_bc.essential_comps.size(); ++i) {
                // Add to total maps
                map_ess_id["total"][step].push_back(vel_bc.essential_ids[i]);
                map_ess_comp["total"][step].push_back(vel_bc.essential_comps[i]);
                
                // Add to velocity-specific maps
                map_ess_id["ess_vel"][step].push_back(vel_bc.essential_ids[i]);
                map_ess_comp["ess_vel"][step].push_back(vel_bc.essential_comps[i]);
                
                // Add default entry to vgrad maps for completeness
                map_ess_id["ess_vgrad"][step].push_back(vel_bc.essential_ids[i]);
                map_ess_comp["ess_vgrad"][step].push_back(0);
            }
            // Add the values if available
            if (!vel_bc.essential_vals.empty()) {
                // Add the values to the map
                // Note: the original code expected values organized as triplets
                // of x, y, z values for each BC
                map_ess_vel[step] = vel_bc.essential_vals;
            }
        }
    }
    
    // Process velocity gradient BCs
    for (const auto& vgrad_bc : vgrad_bcs) {
        for (int step : update_steps) {
            // Initialize maps for this step if needed
            if (map_ess_comp["total"].find(step) == map_ess_comp["total"].end()) {
                map_ess_comp["total"][step] = std::vector<int>();
                map_ess_comp["ess_vel"][step] = std::vector<int>();
                map_ess_comp["ess_vgrad"][step] = std::vector<int>();
                
                map_ess_id["total"][step] = std::vector<int>();
                map_ess_id["ess_vel"][step] = std::vector<int>();
                map_ess_id["ess_vgrad"][step] = std::vector<int>();

                map_ess_vel[step] = std::vector<double>();
                map_ess_vgrad[step] = std::vector<double>(9, 0.0);
            }
            // Add this BC's data to the maps
            for (size_t i = 0; i < vgrad_bc.essential_ids.size(); ++i) {
                int comp_val = -7; // Default to all components (-7 means all components for vgrad)

                // Add to total maps with negative component to indicate vgrad BC
                map_ess_id["total"][step].push_back(vgrad_bc.essential_ids[i]);
                map_ess_comp["total"][step].push_back(comp_val);

                // Add to vgrad-specific maps
                map_ess_id["ess_vgrad"][step].push_back(vgrad_bc.essential_ids[i]);
                map_ess_comp["ess_vgrad"][step].push_back(std::abs(comp_val));
                
                // Add default entry to velocity maps for completeness
                map_ess_id["ess_vel"][step].push_back(vgrad_bc.essential_ids[i]);
                map_ess_comp["ess_vel"][step].push_back(0);
            }
            // Add the gradient values if available
            if (!vgrad_bc.velocity_gradient.empty()) {
                map_ess_vgrad[step] = vgrad_bc.velocity_gradient;
            }
        }
    }
}

BoundaryOptions BoundaryOptions::from_toml(const toml::value& toml_input) {
    BoundaryOptions options;

    // Parse legacy format flags
    if (toml_input.contains("changing_ess_bcs")) {
        options.legacy_bcs.changing_ess_bcs = toml::find<bool>(toml_input, "changing_ess_bcs");
    }

    if (toml_input.contains("update_steps")) {
        options.legacy_bcs.update_steps = toml::find<std::vector<int>>(toml_input, "update_steps");
    }

    if (toml_input.contains("time_info")) {
        options.time_info = BCTimeInfo::from_toml(toml::find(toml_input, "time_info"));
    }

    // Parse essential IDs based on format
    if (toml_input.contains("essential_ids")) {
        const auto& ids = toml_input.at("essential_ids");
        if (ids.is_array()) {
            // Check if first element is also an array (nested arrays)
            if (!ids.as_array().empty() && ids.as_array()[0].is_array()) {
                // Nested arrays for time-dependent BCs
                options.legacy_bcs.essential_ids = 
                    toml::find<std::vector<std::vector<int>>>(toml_input, "essential_ids");
            } else {
                // Flat array for constant BCs
                options.legacy_bcs.essential_ids = 
                    toml::find<std::vector<int>>(toml_input, "essential_ids");
            }
        }
    }

    // Parse essential components based on format
    if (toml_input.contains("essential_comps")) {
        const auto& comps = toml_input.at("essential_comps");
        if (comps.is_array()) {
            // Check if first element is also an array (nested arrays)
            if (!comps.as_array().empty() && comps.as_array()[0].is_array()) {
                // Nested arrays for time-dependent BCs
                options.legacy_bcs.essential_comps = 
                    toml::find<std::vector<std::vector<int>>>(toml_input, "essential_comps");
            } else {
                // Flat array for constant BCs
                options.legacy_bcs.essential_comps = 
                    toml::find<std::vector<int>>(toml_input, "essential_comps");
            }
        }
    }

    // Parse essential values based on format
    if (toml_input.contains("essential_vals")) {
        const auto& vals = toml_input.at("essential_vals");
        if (vals.is_array()) {
            // Check if first element is also an array (nested arrays)
            if (!vals.as_array().empty() && vals.as_array()[0].is_array()) {
                // Nested arrays for time-dependent BCs
                options.legacy_bcs.essential_vals = 
                    toml::find<std::vector<std::vector<double>>>(toml_input, "essential_vals");
            } else {
                // Flat array for constant BCs
                options.legacy_bcs.essential_vals = 
                    toml::find<std::vector<double>>(toml_input, "essential_vals");
            }
        }
    }
    
    // Parse velocity gradient based on format
    if (toml_input.contains("essential_vel_grad")) {
        const auto& vgrad = toml_input.at("essential_vel_grad");
        if (vgrad.is_array()) {
            // Check if we have a triple-nested array structure
            if (!vgrad.as_array().empty() && vgrad.as_array()[0].is_array() && 
                !vgrad.as_array()[0].as_array().empty() && vgrad.as_array()[0].as_array()[0].is_array()) {
                // Triple-nested arrays for time-dependent BCs with 2D gradient matrices
                options.legacy_bcs.essential_vel_grad = 
                    toml::find<std::vector<std::vector<std::vector<double>>>>(toml_input, "essential_vel_grad");
            } else {
                // Double-nested arrays for constant BCs with 2D gradient matrix
                options.legacy_bcs.essential_vel_grad = 
                    toml::find<std::vector<std::vector<double>>>(toml_input, "essential_vel_grad");
            }
        }
    }

    if (toml_input.contains("vgrad_origin")) {
        options.legacy_bcs.vgrad_origin = toml::find<std::vector<double>>(toml_input, "vgrad_origin");
    }

    // Parse modern structured format
    if (toml_input.contains("velocity_bcs")) {
        const auto& vel_bcs = toml::find(toml_input, "velocity_bcs");
        if (vel_bcs.is_array()) {
            for (const auto& bc : vel_bcs.as_array()) {
                options.velocity_bcs.push_back(VelocityBC::from_toml(bc));
            }
        } else {
            options.velocity_bcs.push_back(VelocityBC::from_toml(vel_bcs));
        }
    }

    if (toml_input.contains("velocity_gradient_bcs")) {
        const auto& vgrad_bcs = toml::find(toml_input, "velocity_gradient_bcs");
        if (vgrad_bcs.is_array()) {
            for (const auto& bc : vgrad_bcs.as_array()) {
                options.vgrad_bcs.push_back(VelocityGradientBC::from_toml(bc));
            }
        } else {
            options.vgrad_bcs.push_back(VelocityGradientBC::from_toml(vgrad_bcs));
        }
    }

    return options;
}

LightUpOptions LightUpOptions::from_toml(const toml::value& toml_input) {
    LightUpOptions options;
    
    if (toml_input.contains("light_up")) {
        options.enabled = toml::find<bool>(toml_input, "light_up");
    } else if (toml_input.contains("enabled")) {
        options.enabled = toml::find<bool>(toml_input, "enabled");
    }
    
    if (toml_input.contains("light_up_hkl")) {
        const auto& hkl = toml::find(toml_input, "light_up_hkl");
        if (hkl.is_array()) {
            for (const auto& dir : hkl.as_array()) {
                if (dir.is_array() && dir.as_array().size() >= 3) {
                    std::array<double, 3> direction;
                    auto dir_vec = toml::get<std::vector<double>>(dir);
                    std::copy_n(dir_vec.begin(), 3, direction.begin());
                    options.hkl_directions.push_back(direction);
                }
            }
        }
    } else if (toml_input.contains("hkl_directions")) {
        const auto& hkl = toml::find(toml_input, "hkl_directions");
        if (hkl.is_array()) {
            for (const auto& dir : hkl.as_array()) {
                if (dir.is_array() && dir.as_array().size() >= 3) {
                    std::array<double, 3> direction;
                    auto dir_vec = toml::get<std::vector<double>>(dir);
                    std::copy_n(dir_vec.begin(), 3, direction.begin());
                    options.hkl_directions.push_back(direction);
                }
            }
        }
    }
    
    if (toml_input.contains("light_dist_tol")) {
        options.distance_tolerance = toml::find<double>(toml_input, "light_dist_tol");
    } else if (toml_input.contains("distance_tolerance")) {
        options.distance_tolerance = toml::find<double>(toml_input, "distance_tolerance");
    }
    
    if (toml_input.contains("light_s_dir")) {
        auto dir = toml::find<std::vector<double>>(toml_input, "light_s_dir");
        if (dir.size() >= 3) {
            std::copy_n(dir.begin(), 3, options.sample_direction.begin());
        }
    } else if (toml_input.contains("sample_direction")) {
        auto dir = toml::find<std::vector<double>>(toml_input, "sample_direction");
        if (dir.size() >= 3) {
            std::copy_n(dir.begin(), 3, options.sample_direction.begin());
        }
    }
    
    if (toml_input.contains("lattice_params")) {
        auto params = toml::find<std::vector<double>>(toml_input, "lattice_params");
        if (params.size() >= 3) {
            std::copy_n(params.begin(), 3, options.lattice_parameters.begin());
        }
    } else if (toml_input.contains("lattice_parameters")) {
        auto params = toml::find<std::vector<double>>(toml_input, "lattice_parameters");
        if (params.size() >= 3) {
            std::copy_n(params.begin(), 3, options.lattice_parameters.begin());
        }
    }
    
    if (toml_input.contains("lattice_basename")) {
        options.lattice_basename = toml::find<std::string>(toml_input, "lattice_basename");
    }
    
    return options;
}

VisualizationOptions VisualizationOptions::from_toml(const toml::value& toml_input) {
    VisualizationOptions options;
    
    if (toml_input.contains("visit")) {
        options.visit = toml::find<bool>(toml_input, "visit");
    }
    
    if (toml_input.contains("paraview")) {
        options.paraview = toml::find<bool>(toml_input, "paraview");
    }
    
    if (toml_input.contains("conduit")) {
        options.conduit = toml::find<bool>(toml_input, "conduit");
    }
    
    if (toml_input.contains("adios2")) {
        options.adios2 = toml::find<bool>(toml_input, "adios2");
    }
    
    if (toml_input.contains("steps") || toml_input.contains("output_frequency")) {
        // Support both naming conventions
        const auto& freq_key = toml_input.contains("steps") ? "steps" : "output_frequency";
        options.output_frequency = toml::find<int>(toml_input, freq_key);
    }

    if (toml_input.contains("floc")) {
        options.floc = toml::find<std::string>(toml_input, "floc");
    }

    return options;
}

VolumeAverageOptions VolumeAverageOptions::from_toml(const toml::value& toml_input) {
    VolumeAverageOptions options;
    
    if (toml_input.contains("enabled")) {
        options.enabled = toml::find<bool>(toml_input, "enabled");
    }
    
    if (toml_input.contains("stress")) {
        options.stress = toml::find<bool>(toml_input, "stress");
    }
    
    if (toml_input.contains("euler_strain")) {
        options.euler_strain = toml::find<bool>(toml_input, "euler_strain");
    }
    
    if (toml_input.contains("plastic_work")) {
        options.plastic_work = toml::find<bool>(toml_input, "plastic_work");
    }
    
    if (toml_input.contains("elastic_strain")) {
        options.elastic_strain = toml::find<bool>(toml_input, "elastic_strain");
    }
    
    if (toml_input.contains("output_directory")) {
        options.output_directory = toml::find<std::string>(toml_input, "output_directory");
    }
    
    if (toml_input.contains("output_frequency")) {
        options.output_frequency = toml::find<int>(toml_input, "output_frequency");
    }
    
    return options;
}

ProjectionOptions ProjectionOptions::from_toml(const toml::value& toml_input) {
    ProjectionOptions options;
    
    if (toml_input.contains("enabled_projections")) {
        options.enabled_projections = 
            toml::find<std::vector<std::string>>(toml_input, "enabled_projections");
    }
    
    if (toml_input.contains("auto_enable_compatible")) {
        options.auto_enable_compatible = 
            toml::find<bool>(toml_input, "auto_enable_compatible");
    }
    
    return options;
}

PostProcessingOptions PostProcessingOptions::from_toml(const toml::value& toml_input) {
    PostProcessingOptions options;
    
    if (toml_input.contains("volume_averages")) {
        options.volume_averages = VolumeAverageOptions::from_toml(
            toml::find(toml_input, "volume_averages"));
    }
    
    if (toml_input.contains("projections")) {
        options.projections = ProjectionOptions::from_toml(
            toml::find(toml_input, "projections"));
    }
    
    return options;
}

void ExaOptions::parse_options(const std::string& filename, int my_id) {
    try {
        // Parse the main TOML file
        toml::value toml_input = toml::parse(filename);
        
        // Parse the full configuration
        parse_from_toml(toml_input);
        
        // Validate the complete configuration
        if (!validate()) {
            if (my_id == 0) {
                std::cerr << "Error: Configuration validation failed." << std::endl;
            }
            MFEM_ABORT("Configuration validation failed for option file");
        }
        
    } catch (const std::exception& e) {
        if (my_id == 0) {
            std::cerr << "Error parsing options: " << e.what() << std::endl;
        }
        MFEM_ABORT("Configuration validation failed for option file");
    }
}

void ExaOptions::parse_from_toml(const toml::value& toml_input) {
    // Parse basic metadata
    if (toml_input.contains("Version")) {
        version = toml::find<std::string>(toml_input, "Version");
    }

    if (toml_input.contains("basename")) {
        basename = toml::find<std::string>(toml_input, "basename");
    }

    // Check for modular configuration
    if (toml_input.contains("materials")) {
        material_files = toml::find<std::vector<std::string>>(toml_input, "materials");
    }

    if (toml_input.contains("post_processing")) {
        post_processing_file = toml::find<std::string>(toml_input, "post_processing");
    }

    if (toml_input.contains("grain_file")) {
        grain_file = toml::find<std::string>(toml_input, "grain_file");
    }

    // New fields for optional region mapping
    if (toml_input.contains("region_mapping_file")) {
        region_mapping_file = toml::find<std::string>(toml_input, "region_mapping_file");
    }

    // Parse component sections
    parse_mesh_options(toml_input);
    parse_time_options(toml_input);
    parse_solver_options(toml_input);
    parse_boundary_options(toml_input);
    parse_visualization_options(toml_input);
    
    // Parse materials from main file if no external files are specified
    if (material_files.empty()) {
        parse_material_options(toml_input);
    } else {
        load_material_files();
    }
    
    // Parse post-processing from main file if no external file is specified
    if (!post_processing_file) {
        parse_post_processing_options(toml_input);
    } else {
        load_post_processing_file();
    }
}

void ExaOptions::parse_mesh_options(const toml::value& toml_input) {
    if (toml_input.contains("Mesh")) {
        mesh = MeshOptions::from_toml(toml::find(toml_input, "Mesh"));
    }
}

void ExaOptions::parse_time_options(const toml::value& toml_input) {
    if (!toml_input.contains("Time")) {
        return;
    }
    
    const auto& time_section = toml::find(toml_input, "Time");
    
    // Parse restart options
    if (time_section.contains("restart")) {
        time.restart = toml::find<bool>(time_section, "restart");
    }
    
    if (time_section.contains("restart_time")) {
        time.restart_time = toml::find<double>(time_section, "restart_time");
    }
    
    if (time_section.contains("restart_cycle")) {
        time.restart_cycle = toml::find<size_t>(time_section, "restart_cycle");
    }
    
    // Parse nested time stepping sections
    if (time_section.contains("Auto")) {
        time.auto_time = TimeOptions::AutoTimeOptions::from_toml(
            toml::find(time_section, "Auto"));
    }
    
    if (time_section.contains("Fixed")) {
        time.fixed_time = TimeOptions::FixedTimeOptions::from_toml(
            toml::find(time_section, "Fixed"));
    }
    
    if (time_section.contains("Custom")) {
        time.custom_time = TimeOptions::CustomTimeOptions::from_toml(
            toml::find(time_section, "Custom"));
    }
    
    // Determine which time stepping mode to use
    time.determine_time_type();
}

void ExaOptions::parse_solver_options(const toml::value& toml_input) {
    if (toml_input.contains("Solvers")) {
        solvers = SolverOptions::from_toml(toml::find(toml_input, "Solvers"));
    }
}

void ExaOptions::parse_material_options(const toml::value& toml_input) {
    // Check for materials array under "Materials" section
    if (toml_input.contains("Materials")) {
        auto materials_section = toml::find(toml_input, "Materials");
        materials = MaterialOptions::from_toml_array(materials_section);
    }
    // Legacy format - material properties directly in Properties section
    else if (toml_input.contains("Properties")) {
        MaterialOptions single_material;
        
        // Parse properties section
        single_material.properties = MaterialProperties::from_toml(
            toml::find(toml_input, "Properties"));
        
        // Parse global temperature if present
        if (toml_input.at("Properties").contains("temperature")) {
            single_material.temperature = 
                toml::find<double>(toml_input.at("Properties"), "temperature");
        }
        
        // Parse state variables if present
        if (toml_input.at("Properties").contains("State_Vars")) {
            single_material.state_vars = StateVariables::from_toml(
                toml::find(toml_input.at("Properties"), "State_Vars"));
        }
        
        // Parse grain info if present
        if (toml_input.at("Properties").contains("Grain")) {
            single_material.grain_info = GrainInfo::from_toml(
                toml::find(toml_input.at("Properties"), "Grain"));
        }
        
        // Try to determine model type and options
        if (toml_input.contains("Model")) {
            parse_model_options(toml_input, single_material);
        }
        
        // Add the single material
        materials.push_back(single_material);
    }
}

void ExaOptions::parse_model_options(const toml::value& toml_input, MaterialOptions& material) {
    if (!toml_input.contains("Model")) {
        return;
    }
    
    const auto& model_section = toml::find(toml_input, "Model");
    
    // Parse common model properties
    if (model_section.contains("mech_type")) {
        std::string mech_type_str = toml::find<std::string>(model_section, "mech_type");
        material.mech_type = string_to_mech_type(mech_type_str);
    }
    
    if (model_section.contains("cp")) {
        material.model.crystal_plasticity = toml::find<bool>(model_section, "cp");
    }
    
    // Parse ExaCMech-specific options
    if (material.mech_type == MechType::EXACMECH && model_section.contains("ExaCMech")) {
        material.model.exacmech = ExaCMechModelOptions::from_toml(
            toml::find(model_section, "ExaCMech"));
        
        // Validate that we have a valid shortcut (either directly or derived)
        std::string effective_shortcut = material.model.exacmech->getEffectiveShortcut();
        
        if (effective_shortcut.empty()) {
            std::cerr << "Error: Invalid ExaCMech model configuration. "
                      << "Either shortcut or both xtal_type and slip_type must be provided." 
                      << std::endl;
        }
        
        // When using legacy parameters, set the derived shortcut for other code to use
        if (material.model.exacmech->shortcut.empty() && !effective_shortcut.empty()) {
            material.model.exacmech->shortcut = effective_shortcut;
        }

        auto index_map = ecmech::modelParamIndexMap(material.model.exacmech->shortcut);

        // add more checks later like
        material.model.exacmech->gdot_size = index_map["num_slip_system"];
        material.model.exacmech->hard_size = index_map["num_hardening"];

        /*
            auto num_props_check = index_map["num_params"];
            auto num_state_vars_check = index_map["num_hist"] + ecmech::ne + 1 - 4;

            if (numStateVars != (int) num_state_vars_check) {
            MFEM_ABORT("Properties.State_Vars.num_vars needs " << num_state_vars_check << " values for the given material choice"
                        "Note: the number of values for a quaternion "
                        "are not included in this count.");
            }

            if (nProps != (int) num_props_check) {
            MFEM_ABORT("Properties.Matl_Props.num_props needs " << num_props_check << " values for the given material choice"
                        "Note: the number of values for a quaternion "
                        "are not included in this count.");
            }
        */
    }
    // Parse UMAT-specific options
    else if (material.mech_type == MechType::UMAT && model_section.contains("UMAT")) {
        material.model.umat = UmatOptions::from_toml(
            toml::find(model_section, "UMAT"));
    }
}

void ExaOptions::parse_boundary_options(const toml::value& toml_input) {
    if (toml_input.contains("BCs")) {
        boundary_conditions = BoundaryOptions::from_toml(toml::find(toml_input, "BCs"));
        
        // Transform and validate
        boundary_conditions.validate();
    }
}

void ExaOptions::parse_visualization_options(const toml::value& toml_input) {
    if (toml_input.contains("Visualizations")) {
        visualization = VisualizationOptions::from_toml(
            toml::find(toml_input, "Visualizations"));
    }
}

void ExaOptions::parse_post_processing_options(const toml::value& toml_input) {
    if (toml_input.contains("PostProcessing")) {
        post_processing = PostProcessingOptions::from_toml(
            toml::find(toml_input, "PostProcessing"));
    }
}

void ExaOptions::load_material_files() {
    materials.clear();
    
    for (const auto& file_path : material_files) {
        try {
            toml::value mat_toml = toml::parse(file_path);
            
            // Parse the material
            auto material = MaterialOptions::from_toml(mat_toml);
            
            // Parse model options separately to handle the shortcut derivation
            if (mat_toml.contains("Model")) {
                parse_model_options(mat_toml, material);
            }
            
            // Add the material to our list
            materials.push_back(material);
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing material file " << file_path << ": " 
                      << e.what() << std::endl;
            throw; // Re-throw to propagate the error
        }
    }
}

void ExaOptions::load_post_processing_file() {
    if (post_processing_file.has_value()) {
        try {
            toml::value pp_toml = toml::parse(post_processing_file.value());
            post_processing = PostProcessingOptions::from_toml(pp_toml);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing post-processing file " 
                      << post_processing_file.value() << ": " 
                      << e.what() << std::endl;
            throw; // Re-throw to propagate the error
        }
    }
}

bool ExaOptions::validate() {
    // Basic validation - could be expanded with more comprehensive checks

    mesh.validate();
    time.validate();
    solvers.validate();
    visualization.validate();
    boundary_conditions.validate();
    post_processing.validate();

    // Check that we have at least one material
    if (materials.empty()) {
        std::cerr << "Error: No materials defined in configuration." << std::endl;
        return false;
    }

    if (materials.size() > 1) {
        if (!region_mapping_file) {
            std::cerr << "Error: region_mapping_file was not provided even though multiple materials were asked for." << std::endl;
            return false;
        }
        else if (mesh.mesh_type == MeshType::AUTO && !grain_file) {
            std::cerr << "Error: region_mapping_file was provided but no grain_file was provided when using auto mesh." << std::endl;
  return false;
        }
    }

    if (materials.size() > 1) {
        if (!region_mapping_file) {
            std::cerr << "Error: region_mapping_file was not provided even though multiple materials were asked for." << std::endl;
            return false;
        }
        else if (mesh.mesh_type == MeshType::AUTO && !grain_file) {
            std::cerr << "Error: region_mapping_file was provided but no grain_file was provided when using auto mesh." << std::endl;
  return false;
        }
    }

    size_t index = 0;
    for (auto& mat : materials) {
        mat.validate();
        // Update the region_id value after validating
        // everything so to make it easier for users to
        // validation errors
        mat.region_id = index++;
    }

    return true;
}

// Implementation of validation methods for component structs

bool MeshOptions::validate() const {
    if(mesh_type == MeshType::NOTYPE) {
        std::cerr << "Error: Mesh table was not provided an appropriate mesh type" << std::endl;
        return false;
    }

    // For auto mesh generation, check that nxyz and mxyz are valid
    if (mesh_type == MeshType::AUTO) {
        for (int i = 0; i < 3; ++i) {
            if (nxyz[i] <= 0) {
                std::cerr << "Error: Invalid mesh discretization: nxyz[" << i 
                          << "] = " << nxyz[i] << std::endl;
                return false;
            }
            if (mxyz[i] <= 0.0) {
                std::cerr << "Error: Invalid mesh dimensions: mxyz[" << i 
                          << "] = " << mxyz[i] << std::endl;
                return false;
            }
        }
    }

    // Check that mesh file exists for CUBIT or OTHER mesh types
    if ((mesh_type == MeshType::FILE) && 
        !mesh_file.empty()) {
        if (!fs::exists(mesh_file)) {
            std::cerr << "Error: Mesh file '" << mesh_file 
                      << "' does not exist." << std::endl;
            return false;
        }
    }

    if (ref_ser < 0) {
        std::cerr << "Error: Mesh table has ref_ser set to value less than 0." << std::endl;
        return false;
    }

    if (ref_par < 0) {
        std::cerr << "Error: Mesh table has ref_par set to value less than 0." << std::endl;
        return false;
    }

    if (order < 1) {
        std::cerr << "Error: Mesh table has order set to value less than 1." << std::endl;
        return false;
    }

    // Implement validation logic
    return true;
}

bool GrainInfo::validate() const {
    // Implement validation logic
    if (!orientation_file) {
        std::cerr << "Error: Grain table was provided without providing an orientation file this is required" << std::endl;
        return false;
    }

    if (ori_type == OriType::NOTYPE) {
        std::cerr << "Error: Orientation type within the Grain table was not provided a valid value (quats, euler, or custom)" << std::endl;
        return false;
    }

    if (num_grains < 1) {
        std::cerr << "Error: num_grains was provided a value less than 1" << std::endl;
        return false;
    }

    return true;
}

bool MaterialProperties::validate() const {
    // Implement validation logic
    return true;
}

bool StateVariables::validate() const {
    // Implement validation logic
    return true;
}

bool UmatOptions::validate() const {
    // Implement validation logic
    return true;
}

bool ExaCMechModelOptions::validate() const {
    // Implement validation logic
    return !getEffectiveShortcut().empty();
}

bool MaterialModelOptions::validate() const {
    if (!umat and !exacmech) {
        std::cerr << "Error: Model table has not provided either an ExaCMech or UMAT table within it." << std::endl;
        return false;
    }

    if (umat) {
        umat->validate();
    }

    if (exacmech) {
        if (!crystal_plasticity) {
            std::cerr << "Error: Model table is using an ExaCMech table but has not set variable crystal_plasticity as true." << std::endl;
            return false;
        }
        exacmech->validate();
    }

    return true;
}

bool MaterialOptions::validate() const {
    std::string mat_name = material_name + "_" + std::to_string(region_id);

    if (mech_type == MechType::NOTYPE) {
        std::cerr << "Error: Material table for material_name_region# " << mat_name << " the mech_type was not set a valid option" << std::endl;
        return false;
    }

    if (temperature <= 0) {
        std::cerr << "Error: Material table for material_name_region# " << mat_name << " the temperature was provided a negative value" << std::endl;
        return false;
    }

    properties.validate();
    state_vars.validate();
    model.validate();

    if (grain_info) {
        grain_info->validate();
    }

    if (model.crystal_plasticity) {
        if (!grain_info) {
            std::cerr << "Error: Material table for material_name_region# " << mat_name << " the material model was set to use crystal plasticity model but the Grain table was not set" << std::endl;
            return false;
        }
    }
    return true;
}

bool TimeOptions::validate() {
    switch (time_type) {
        case TimeStepType::CUSTOM:
            if (!custom_time.has_value()) {
                return false;
            }
            return custom_time->load_custom_dt_values();
            
        case TimeStepType::AUTO:
            if (!auto_time.has_value()) {
                return false;
            }
            return auto_time->dt_min > 0.0 && 
                   auto_time->dt_scale > 0.0 && 
                   auto_time->dt_scale < 1.0;
            
        case TimeStepType::FIXED:
            if (!fixed_time.has_value()) {
                return false;
            }
            return fixed_time->dt > 0.0;
            
        default:
            return false;
    }
    return true;
}

bool LinearSolverOptions::validate() const {

    if (max_iter < 1) {
        std::cerr << "Error: LinearSolver table did not provide a positive iteration count" << std::endl;
        return false;
    }

    if (abs_tol < 0) {
        std::cerr << "Error: LinearSolver table provided a negative absolute tolerance" << std::endl;
        return false;
    }

    if (rel_tol < 0) {
        std::cerr << "Error: LinearSolver table provided a negative relative tolerance" << std::endl;
        return false;
    }

    if (solver_type == LinearSolverType::NOTYPE) {
        std::cerr << "Error: LinearSolver table did not provide a valid solver type (CG, GMRES, or MINRES)" << std::endl;
        return false;
    }

    if (preconditioner == PreconditionerType::NOTYPE) {
        std::cerr << "Error: LinearSolver table did not provide a valid preconditioner type (JACOBI or AMG)" << std::endl;
        return false;
    }

    // Implement validation logic
    return true;
}

bool NonlinearSolverOptions::validate() const {
    int iter = 25;
    double rel_tol = 1e-5;
    double abs_tol = 1e-10;
    std::string nl_solver = "NR";

    if (iter < 1) {
        std::cerr << "Error: NonLinearSolver table did not provide a positive iteration count" << std::endl;
        return false;
    }

    if (abs_tol < 0) {
        std::cerr << "Error: NonLinearSolver table provided a negative absolute tolerance" << std::endl;
        return false;
    }

    if (rel_tol < 0) {
        std::cerr << "Error: NonLinearSolver table provided a negative relative tolerance" << std::endl;
        return false;
    }

    if (nl_solver != "NR" && nl_solver != "NRLS") {
        std::cerr << "Error: NonLinearSolver table did not provide a valid nl_solver option (`NR` or `NRLS`)" << std::endl;
        return false;
    }

    // Implement validation logic
    return true;
}

bool SolverOptions::validate() const {

    nonlinear_solver.validate();
    linear_solver.validate();

    if (assembly == AssemblyType::NOTYPE) {
        std::cerr << "Error: Solver table did not provide a valid assembly option (`FULL`, `PA`, or `EA`)" << std::endl;
        return false;
    }

    if (rtmodel == RTModel::NOTYPE) {
        std::cerr << "Error: Solver table did not provide a valid rtmodel option (`CPU`, `OPENMP`, or `GPU`)" << std::endl;
        return false;
    }

    if (integ_model == IntegrationModel::NOTYPE) {
        std::cerr << "Error: Solver table did not provide a valid integ_model option (`FULL` or `BBAR`)" << std::endl;
        return false;
    }

    if (rtmodel == RTModel::GPU && assembly == AssemblyType::FULL) {
        std::cerr << "Error: Solver table did not provide a valid assembly option when using GPU rtmodel: `FULL` assembly can not be used with `GPU` rtmodels" << std::endl;
        return false;
    }

    // Implement validation logic
    return true;
}

bool BCTimeInfo::validate() const {
    // Implement validation logic
    return true;
}

bool VelocityBC::validate() const {
    // Implement validation logic
    return !essential_ids.empty() && 
           !essential_comps.empty() && 
           !essential_vals.empty();
}

bool VelocityGradientBC::validate() const {
    // Implement validation logic
    return !velocity_gradient.empty();
}

bool LightUpOptions::validate() const {
    if (!enabled) { return true; }
    if (hkl_directions.size() < 1) {
        std::cerr << "Error: LightUp table did not provide any values in the hkl_directions" << std::endl;
        return false;
    }

    if (distance_tolerance < 0) {
        std::cerr << "Error: LightUp table did not provide a positive distance_tolerance value" << std::endl;
        return false;
    }

    if (lattice_parameters[0] < 0 || lattice_parameters[1] < 0 || lattice_parameters[2] < 0) {
        std::cerr << "Error: LightUp table did not provide a positive lattice_parameters value" << std::endl;
        return false;
    }

    // Implement validation logic
    return true;
}

bool VisualizationOptions::validate() const {
    // Implement validation logic
    return true;
}

bool VolumeAverageOptions::validate() const {
    // Implement validation logic
    if (!enabled) { return true; }
    if (output_frequency < 1) {
        std::cerr << "Error: Visualizations table did not provide a valid assembly option when using GPU rtmodel: `FULL` assembly can not be used with `GPU` rtmodels" << std::endl;
        return false;
    }
    return true;
}

bool ProjectionOptions::validate() const {
    // Implement validation logic
    return true;
}

bool PostProcessingOptions::validate() const {
    // Implement validation logic
    volume_averages.validate();
    projections.validate();
    light_up.validate();
    return true;
}