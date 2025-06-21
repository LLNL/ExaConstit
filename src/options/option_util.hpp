#pragma once

#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <map>

// Utility functions for parsing TOML
// Convert string to enum with validation
template<typename EnumType>
inline
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
inline
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
    
