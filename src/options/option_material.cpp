#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

#include "ECMech_cases.h"

#include <iostream>


GrainInfo GrainInfo::from_toml(const toml::value& toml_input) {
    GrainInfo info;

    if (toml_input.contains("orientation_file")) {
        info.orientation_file = toml::find<std::string>(toml_input, "orientation_file");
    }
    else if (toml_input.contains("ori_floc")) {
        info.orientation_file = toml::find<std::string>(toml_input, "ori_floc");
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

    if (toml_input.contains("grain_file")) {
        info.grain_file = toml::find<std::string>(toml_input, "grain_file");
    }
    else if (toml_input.contains("grain_floc")) {
        info.grain_file = toml::find<std::string>(toml_input, "grain_floc");
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
        if (slip_type == "POWERVOCE") {
            derived_shortcut += "_A";
        }
        else if (slip_type == "POWERVOCENL") {
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
        std::transform(options.xtal_type.begin(), options.xtal_type.end(), options.xtal_type.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    }
    
    if (toml_input.contains("slip_type")) {
        options.slip_type = toml::find<std::string>(toml_input, "slip_type");
        std::transform(options.slip_type.begin(), options.slip_type.end(), options.slip_type.begin(),
                   [](unsigned char c){ return std::toupper(c); });    }

    if (options.shortcut.empty()) {
        options.shortcut = options.getEffectiveShortcut();
    }
    auto param_index = ecmech::modelParamIndexMap(options.shortcut);
    options.gdot_size = param_index["num_slip_system"];
    options.hard_size = param_index["num_hardening"];

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
        if (toml_input.at("temperature").is_integer()) {
            options.temperature = (double) toml::find<int>(toml_input, "temperature");
        } else {
            options.temperature = toml::find<double>(toml_input, "temperature");
        }
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