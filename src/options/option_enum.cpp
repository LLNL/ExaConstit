#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

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
        {"PCG", LinearSolverType::CG},
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