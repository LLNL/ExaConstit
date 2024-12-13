
#include "option_parser.hpp"
#include "RAJA/RAJA.hpp"
#include "TOML_Reader/toml.hpp"
#include "mfem.hpp"
#include "ECMech_cases.h"
#include <iostream>
#include <fstream>

inline bool if_file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

// my_id corresponds to the processor id.
void ExaOptions::parse_options(int my_id)
{
   // From the toml file it finds all the values related to the mesh
   get_mesh();
   // From the toml file it finds all the values related to state and mat'l
   // properties
   get_properties();
   // From the toml file it finds all the values related to the BCs
   get_bcs();
   // From the toml file it finds all the values related to the model
   get_model();
   // From the toml file it finds all the values related to the time
   get_time_steps();
   // From the toml file it finds all the values related to the visualizations
   get_visualizations();
   // From the toml file it finds all the values related to the Solvers
   get_solvers();
   // If the processor is set 0 then the options are printed out.
   if (my_id == 0) {
      print_options();
   }
}

// From the toml file it finds all the values related to state and mat'l
// properties
void ExaOptions::get_properties()
{
   const auto data = toml::parse(floc);
   const auto& table = toml::find(data, "Properties");
   double _temp_k = toml::find_or<double>(table, "temperature", 298.0);

   if (_temp_k <= 0.0) {
      MFEM_ABORT("Properties.temperature is given in Kelvins and therefore can't be less than 0");
   }

   temp_k = _temp_k;

   // Check to see if our table exists
   if (table.contains("Matl_Props")) {
      // Material properties are obtained first
      const auto& prop_table = toml::find(table, "Matl_Props");
      std::string _props_file = toml::find_or<std::string>(prop_table, "floc", "props.txt");
      props_file = _props_file;
      if (!if_file_exists(props_file))
      {
         MFEM_ABORT("Property file does not exist");
      }
      nProps = toml::find_or<int>(prop_table, "num_props", 1);
   } 
   else {
      MFEM_ABORT("Properties.Matl_Props table was not provided in toml file");
   }

   // Check to see if our table exists
   if (table.contains("State_Vars")) {
      // State variable properties are now obtained
      const auto& state_table = toml::find(table, "State_Vars");
      numStateVars = toml::find_or<int>(state_table, "num_vars", 1);
      std::string _state_file = toml::find_or<std::string>(state_table, "floc", "state.txt");
      state_file = _state_file;
      if (!if_file_exists(state_file))
      {
         MFEM_ABORT("State file does not exist");
      }
   }
   else {
      MFEM_ABORT("Properties.State_Vars table was not provided in toml file");
   }

   // Check to see if our table exists
   if (table.contains("Grain")) {
      // Grain related properties are now obtained
      const auto& grain_table = toml::find(table, "Grain");
      grain_statevar_offset = toml::find_or<int>(grain_table, "ori_state_var_loc", -1);
      grain_custom_stride = toml::find_or<int>(grain_table, "ori_stride", 0);
      std::string _ori_type = toml::find_or<std::string>(grain_table, "ori_type", "euler");
      ngrains = toml::find_or<int>(grain_table, "num_grains", 0);
      std::string _ori_file = toml::find_or<std::string>(grain_table, "ori_floc", "ori.txt");
      ori_file = _ori_file;
      std::string _grain_map = toml::find_or<std::string>(grain_table, "grain_floc", "grain_map.txt");
      grain_map = _grain_map;

      if (grain_table.contains("ori_floc")) {
         if (!if_file_exists(ori_file))
         {
            MFEM_ABORT("Orientation file does not exist");
         }
      }

      if (grain_table.contains("grain_floc")) {
         if (grain_map.size() > 0 && !if_file_exists(grain_map) and (mesh_type == MeshType::AUTO))
         {
            MFEM_ABORT("Grain file does not exist");
         }
      }

      // I still can't believe C++ doesn't allow strings to be used in switch statements...
      if ((_ori_type == "euler") || _ori_type == "Euler" || (_ori_type == "EULER")) {
         ori_type = OriType::EULER;
      }
      else if ((_ori_type == "quat") || (_ori_type == "Quat") || (_ori_type == "quaternion") || (_ori_type == "Quaternion")) {
         ori_type = OriType::QUAT;
      }
      else if ((_ori_type == "custom") || (_ori_type == "Custom") || (_ori_type == "CUSTOM")) {
         ori_type = OriType::CUSTOM;
      }
      else {
         MFEM_ABORT("Properties.Grain.ori_type was not provided a valid type.");
         ori_type = OriType::NOTYPE;
      }
   } // end of if statement for grain data
} // End of propert parsing

// From the toml file it finds all the values related to the BCs
void ExaOptions::get_bcs()
{
   const auto data = toml::parse(floc);
   const auto& table = toml::find(data, "BCs");

   changing_bcs = toml::find_or<bool>(table, "changing_ess_bcs", false);

   vgrad_origin = toml::find_or<std::vector<double>>(table, "vgrad_origin", {});
   vgrad_origin_flag = !vgrad_origin.empty();

   if (vgrad_origin_flag && vgrad_origin.size() != 3) {
      MFEM_ABORT("BCs.vgrad_origin when provided must contain 3 components.");
   }

   if (!changing_bcs) {
      std::vector<int> _essential_ids = toml::find<std::vector<int>>(table, "essential_ids");
      if (_essential_ids.empty()) {
         MFEM_ABORT("BCs.essential_ids was not provided any values.");
      }
      map_ess_id["total"][0] = std::vector<int>();
      map_ess_id["total"][1] = _essential_ids;

      std::vector<int> _essential_comp = toml::find<std::vector<int>>(table, "essential_comps");
      if (_essential_comp.empty()) {
         MFEM_ABORT("BCs.essential_comps was not provided any values.");
      }

      map_ess_comp["total"][0] = std::vector<int>();
      map_ess_comp["total"][1] = _essential_comp;

      std::vector<int> _ess_vel_comp;
      std::vector<int> _ess_vgrad_comp;
      std::vector<int> _ess_vel_id;
      std::vector<int> _ess_vgrad_id;

      int count = 0;

      int ess_vel_conditions = 0;
      int ess_vgrad_conditions = 0;

      for (auto& item : _essential_comp) {
         if (item >= 0) {
            ess_vel_conditions++;
            _ess_vel_comp.push_back(item);
            _ess_vel_id.push_back(_essential_ids.at(count));
            _ess_vgrad_comp.push_back(0);
            _ess_vgrad_id.push_back(_essential_ids.at(count));
         } else {
            ess_vgrad_conditions++;
            _ess_vel_comp.push_back(0);
            _ess_vel_id.push_back(_essential_ids.at(count));
            _ess_vgrad_comp.push_back(std::abs(item));
            _ess_vgrad_id.push_back(_essential_ids.at(count));
         }
         count++;
      }

      map_ess_id["ess_vel"][0] = std::vector<int>();
      map_ess_id["ess_vel"][1] = _ess_vel_id;

      map_ess_comp["ess_vel"][0] = std::vector<int>();
      map_ess_comp["ess_vel"][1] = _ess_vel_comp;

      map_ess_id["ess_vgrad"][0] = std::vector<int>();
      map_ess_id["ess_vgrad"][1] = _ess_vgrad_id;

      map_ess_comp["ess_vgrad"][0] = std::vector<int>();
      map_ess_comp["ess_vgrad"][1] = _ess_vgrad_comp;

      // Getting out arrays of values isn't always the simplest thing to do using
      // this TOML libary.
      std::vector<double> _essential_vals = toml::find_or<std::vector<double>>(table, "essential_vals", {});
      if (_essential_vals.empty() && ess_vel_conditions > 0) {
         MFEM_ABORT("BCs.essential_vals was not provided any values  but a boundary requires this.");
      }

      std::vector<std::vector<double>> _essential_vgrad = toml::find_or<std::vector<std::vector<double>>>(table, "essential_vel_grad", {{}});
      if (_essential_vgrad.empty() && ess_vgrad_conditions > 0) {
         MFEM_ABORT("BCs.essential_vel_grad was not provided any values but a boundary requires this.");
      }

      map_ess_vgrad[0] = std::vector<double>(9, 0.0);
      map_ess_vgrad[1] = std::vector<double>();

      for(auto && v : _essential_vgrad) {
         map_ess_vgrad[1].insert(map_ess_vgrad[1].end(), v.begin(), v.end());
      }

      map_ess_vel[0] = std::vector<double>();
      map_ess_vel[1] = _essential_vals;
      updateStep.push_back(1);
   }
   else {
      updateStep = toml::find<std::vector<int>>(table, "update_steps");

      if (updateStep.empty()) {
         MFEM_ABORT("BCs.update_steps was not provided any values.");
      }
      if (std::find(updateStep.begin(), updateStep.end(), 1) == updateStep.end()) {
         MFEM_ABORT("BCs.update_steps must contain 1 in the array");
      }

      int size = updateStep.size();
      std::vector<std::vector<int>> nested_ess_ids = toml::find<std::vector<std::vector<int>>>(table, "essential_ids");
      int ilength = 0;
      map_ess_id["total"][0] = std::vector<int>();
      map_ess_id["ess_vel"][0] = std::vector<int>();
      map_ess_id["ess_vgrad"][0] = std::vector<int>();
      for (const auto &vec : nested_ess_ids) {
         int key = updateStep.at(ilength);
         map_ess_id["total"][key] = std::vector<int>();
         map_ess_id["ess_vel"][key] = std::vector<int>();
         map_ess_id["ess_vgrad"][key] = std::vector<int>();
         for (const auto &val : vec) {
            map_ess_id["total"][key].push_back(val);
         }
         if (map_ess_id["total"][key].empty()) {
            MFEM_ABORT("BCs.essential_ids contains empty array.");
         }
         ilength += 1;
      }

      if (ilength != size) {
         MFEM_ABORT("BCs.essential_ids did not contain the same number of arrays as number of update steps");
      }

      std::vector<std::vector<int>> nested_ess_comps = toml::find<std::vector<std::vector<int>>>(table, "essential_comps");
      ilength = 0;
      map_ess_comp["total"][0] = std::vector<int>();
      map_ess_comp["ess_vel"][0] = std::vector<int>();
      map_ess_comp["ess_vgrad"][0] = std::vector<int>();

      int ess_vel_conditions = 0;
      int ess_vgrad_conditions = 0;

      for (const auto &vec : nested_ess_comps) {
         int key = updateStep.at(ilength);
         map_ess_comp["total"][key] = std::vector<int>();
         map_ess_comp["ess_vel"][key] = std::vector<int>();
         map_ess_comp["ess_vgrad"][key] = std::vector<int>();
         int count = 0;
         for (const auto &val : vec) {
            map_ess_comp["total"][key].push_back(val);
            if (val >= 0) {
               ess_vel_conditions++;
               map_ess_comp["ess_vel"][key].push_back(val);
               map_ess_id["ess_vel"][key].push_back(map_ess_id["total"][key].at(count));
               map_ess_comp["ess_vgrad"][key].push_back(0);
               map_ess_id["ess_vgrad"][key].push_back(map_ess_id["total"][key].at(count));
            } else {
               ess_vgrad_conditions++;
               map_ess_comp["ess_vel"][key].push_back(0);
               map_ess_id["ess_vel"][key].push_back(map_ess_id["total"][key].at(count));
               map_ess_comp["ess_vgrad"][key].push_back(std::abs(val));
               map_ess_id["ess_vgrad"][key].push_back(map_ess_id["total"][key].at(count));
            }
            count++;
         }
         if (map_ess_comp["total"][key].empty()) {
            MFEM_ABORT("BCs.essential_comps contains empty array.");
         }
         ilength += 1;
      }

      if (ilength != size) {
         MFEM_ABORT("BCs.essential_comps did not contain the same number of arrays as number of update steps");
      }

      std::vector<std::vector<double>> nested_ess_vals = toml::find_or<std::vector<std::vector<double>>>(table, "essential_vals", {{}});
      ilength = 0;
      map_ess_vel[0] = std::vector<double>();
      for (const auto &vec : nested_ess_vals) {
         int key = updateStep.at(ilength);
         map_ess_vel[key] = std::vector<double>();
         for (const auto &val : vec) {
            map_ess_vel[key].push_back(val);
         }
         if (map_ess_vel[key].empty() && ess_vel_conditions > 0) {
            MFEM_ABORT("BCs.essential_vals contains empty array but a boundary requires this.");
         }
         ilength += 1;
      }

      std::vector<std::vector<std::vector<double>>> nested_ess_vgrad = toml::find_or<std::vector<std::vector<std::vector<double>>> >(table, "essential_vel_grad", {{{}}});
      ilength = 0;
      map_ess_vgrad[0] = std::vector<double>(9, 0.0);

      for (const auto &vec : nested_ess_vgrad) {
         int key = updateStep.at(ilength);
         map_ess_vgrad[key] = std::vector<double>();
         for(auto && v : vec) {
            map_ess_vgrad[key].insert(map_ess_vgrad[key].end(), v.begin(), v.end());
         }
         if (map_ess_vgrad[key].empty() && ess_vgrad_conditions > 0) {
            MFEM_ABORT("BCs.essential_vel_grad was not provided any values but a boundary requires this..");
         }
         ilength += 1;
      }
   }
} // end of parsing BCs

// From the toml file it finds all the values related to the model
void ExaOptions::get_model()
{
   const auto data = toml::parse(floc);
   const auto& table = toml::find(data, "Model");
   std::string _mech_type = toml::find_or<std::string>(table, "mech_type", "");

   // I still can't believe C++ doesn't allow strings to be used in switch statements...
   if ((_mech_type == "umat") || (_mech_type == "Umat") || (_mech_type == "UMAT") || (_mech_type == "UMat")) {
      mech_type = MechType::UMAT;
   }
   else if ((_mech_type == "exacmech") || (_mech_type == "Exacmech") || (_mech_type == "ExaCMech") || (_mech_type == "EXACMECH")) {
      mech_type = MechType::EXACMECH;
   }
   else {
      MFEM_ABORT("Model.mech_type was not provided a valid type.");
      mech_type = MechType::NOTYPE;
   }

   cp = toml::find_or<bool>(table, "cp", false);

   if (mech_type == MechType::EXACMECH) {
      if (!cp) {
         MFEM_ABORT("Model.cp needs to be set to true when using ExaCMech based models.");
      }

      if (ori_type != OriType::QUAT) {
         MFEM_ABORT("Properties.Grain.ori_type is not set to quaternion for use with an ExaCMech model.");
      }

      grain_statevar_offset = ecmech::evptn::iHistLbQ;

      if(table.contains("ExaCMech")) {
         const auto& exacmech_table = toml::find(table, "ExaCMech");

         shortcut = toml::find_or<std::string>(exacmech_table, "shortcut", "");
         if (shortcut.size() == 0) {
            auto slip_type = SlipType::NOTYPE;
            auto xtal_type = XtalType::NOTYPE;
            std::string _xtal_type = toml::find_or<std::string>(exacmech_table, "xtal_type", "");
            std::string _slip_type = toml::find_or<std::string>(exacmech_table, "slip_type", "");
            shortcut = "evptn_";
            if ((_xtal_type == "fcc") || (_xtal_type == "FCC")) {
               shortcut += "FCC_";
               xtal_type = XtalType::FCC;
            }
            else if ((_xtal_type == "bcc") || (_xtal_type == "BCC")) {
               shortcut += "BCC_";
               xtal_type = XtalType::BCC;
            }
            else if ((_xtal_type == "hcp") || (_xtal_type == "HCP")) {
               shortcut += "HCP_";
               xtal_type = XtalType::HCP;
            }
            else {
               MFEM_ABORT("Model.ExaCMech.xtal_type was not provided a valid type.");
               xtal_type = XtalType::NOTYPE;
            }
            if ((_slip_type == "mts") || (_slip_type == "MTS") || (_slip_type == "mtsdd") || (_slip_type == "MTSDD")) {
               slip_type = SlipType::MTSDD;
               if (xtal_type == XtalType::HCP) {
                  shortcut += "A";
               }
               else {
                  shortcut += "B";
               }
            }
            else if ((_slip_type == "powervoce") || (_slip_type == "PowerVoce") || (_slip_type == "POWERVOCE")) {
               slip_type = SlipType::POWERVOCE;
               if (xtal_type == XtalType::HCP) {
                  MFEM_ABORT("Model.ExaCMech.slip_type can not be PowerVoce for HCP materials.")
               }
               shortcut += "A";
            }
            else if ((_slip_type == "powervocenl") || (_slip_type == "PowerVoceNL") || (_slip_type == "POWERVOCENL")) {
               slip_type = SlipType::POWERVOCENL;
               if (xtal_type == XtalType::HCP) {
                  MFEM_ABORT("Model.ExaCMech.slip_type can not be PowerVoce for HCP materials.")
               }
               shortcut += "AH";
            }
            else {
               MFEM_ABORT("Model.ExaCMech.slip_type was not provided a valid type.");
               slip_type = SlipType::NOTYPE;
            }
         }

         try {
            [[maybe_unused]] auto unused = ecmech::makeMatModel(shortcut);
         } catch(...) {
            MFEM_ABORT("Model.ExaCMech.shortcut was not provided a valid name.");
         }

         auto index_map = ecmech::modelParamIndexMap(shortcut);
         auto num_props_check = index_map["num_params"];
         auto num_state_vars_check = index_map["num_hist"] + ecmech::ne + 1 - 4;

         gdot_size = index_map["num_slip_system"];
         hard_size = index_map["num_hardening"];


         if (numStateVars != num_state_vars_check) {
            MFEM_ABORT("Properties.State_Vars.num_vars needs " << num_state_vars_check << " values for the given material choice"
                     "Note: the number of values for a quaternion "
                     "are not included in this count.");
         }

         if (nProps != num_props_check) {
            MFEM_ABORT("Properties.Matl_Props.num_props needs " << num_props_check << " values for the given material choice"
                     "Note: the number of values for a quaternion "
                     "are not included in this count.");
         }
      } 
      else {
         MFEM_ABORT("The table Model.ExaCMech does not exist, but the model being used is ExaCMech.");
      }// End if ExaCMech Table Exists
   }
} // end of model parsing

// From the toml file it finds all the values related to the time
void ExaOptions::get_time_steps()
{
   const auto data = toml::parse(floc);
   const auto& table = toml::find(data, "Time");
   // First look at the fixed time stuff
   // check to see if our table exists
   if (table.contains("Fixed")) {
      const auto& fixed_table = toml::find(table, "Fixed");
      dt_cust = false;
      dt_auto = false;
      dt = toml::find_or<double>(fixed_table, "dt", 1.0);
      dt_min = dt;
      t_final = toml::find_or<double>(fixed_table, "t_final", 1.0);
   }
   if (table.contains("Auto")) {
      if (changing_bcs) {
         MFEM_ABORT("Automatic time stepping is currently not compatible with changing boundary conditions");
      }
      const auto& auto_table = toml::find(table, "Auto");
      dt_cust = false;
      dt_auto = true;
      dt = toml::find_or<double>(auto_table, "dt_start", 1.0);
      dt_scale = toml::find_or<double>(auto_table, "dt_scale", 0.25);
      if (dt_scale < 0.0 || dt_scale > 1.0) {
         MFEM_ABORT("dt_scale for auto time stepping needs to be between 0 and 1.");
      }
      dt_min = toml::find_or<double>(auto_table, "dt_min", 1.0);
      t_final = toml::find_or<double>(auto_table, "t_final", 1.0);
      dt_file = toml::find_or<std::string>(auto_table, "auto_dt_file", "auto_dt_out.txt");
   }
   // Time to look at our custom time table stuff
   // check to see if our table exists
   if (table.contains("Custom")) {
      const auto& cust_table = toml::find(table, "Custom");
      dt_cust = true;
      dt_auto = false;
      nsteps = toml::find_or<int>(cust_table, "nsteps", 1);
      std::string _dt_file = toml::find_or<std::string>(cust_table, "floc", "custom_dt.txt");
      dt_file = _dt_file;
   }
} // end of time step parsing

// From the toml file it finds all the values related to the visualizations
void ExaOptions::get_visualizations()
{
   const auto data = toml::parse(floc);
   const auto& table = toml::find(data, "Visualizations");
   vis_steps = toml::find_or<int>(table, "steps", 1);
   visit = toml::find_or<bool>(table, "visit", false);
   conduit = toml::find_or<bool>(table, "conduit", false);
   paraview = toml::find_or<bool>(table, "paraview", false);
   adios2 = toml::find_or<bool>(table, "adios2", false);
   if (conduit || adios2) {
      if (conduit) {
#ifndef MFEM_USE_CONDUIT
         MFEM_ABORT("MFEM was not built with conduit.");
#endif
      }
      else {
#ifndef MFEM_USE_ADIOS2
         MFEM_ABORT("MFEM was not built with ADIOS2");
#endif
      }
   }
   std::string _basename = toml::find_or<std::string>(table, "floc", "results/exaconstit");
   basename = _basename;
   std::string _avg_stress_fname = toml::find_or<std::string>(table, "avg_stress_fname", "avg_stress.txt");
   avg_stress_fname = _avg_stress_fname;
   bool _additional_avgs = toml::find_or<bool>(table, "additional_avgs", false);
   additional_avgs = _additional_avgs;
   std::string _avg_def_grad_fname = toml::find_or<std::string>(table, "avg_def_grad_fname", "avg_def_grad.txt");
   avg_def_grad_fname = _avg_def_grad_fname;
   std::string _avg_pl_work_fname = toml::find_or<std::string>(table, "avg_pl_work_fname", "avg_pl_work.txt");
   avg_pl_work_fname = _avg_pl_work_fname;
   light_up = toml::find_or<bool>(table, "light_up", false);
   if (light_up) {

      auto hkls = toml::find_or< std::vector<std::vector<double>> >(table, "light_up_hkl", {{}});

      for (auto& hkl : hkls) {
         std::array<double, 3> hkl_tmp = {hkl[0], hkl[1], hkl[2]};
         std::cout << "light-up hkls " << hkl_tmp[0] << " " <<  hkl_tmp[1] << " " << hkl_tmp[2] << std::endl;
         light_hkls.push_back(hkl_tmp);
      }

      light_dist_tol = toml::find_or<double>(table, "light_dist_tol", {0.07});
      std::cout << "light-up distance tolerance " << light_dist_tol << std::endl;
      auto s_dirs = toml::find_or<std::vector<double>>(table, "light_s_dir", {});

      light_s_dir[0] = s_dirs[0];
      light_s_dir[1] = s_dirs[1];
      light_s_dir[2] = s_dirs[2];

      std::cout << "light-up s direction " << light_s_dir[0] << " " <<  light_s_dir[1] << " " << light_s_dir[2] << std::endl;

      auto lparams = toml::find_or<std::vector<double>>(table, "lattice_params", {});

      lattice_params[0] = lparams[0];
      lattice_params[1] = lparams[1];
      lattice_params[2] = lparams[2];

      std::cout << "light-up lattice params " << lattice_params[0] << " " <<  lattice_params[1] << " " << lattice_params[2] << std::endl;

      lattice_basename = toml::find_or<std::string>(table, "lattice_basename", "lattice_avg_");

   }
} // end of visualization parsing

// From the toml file it finds all the values related to the Solvers
void ExaOptions::get_solvers()
{
   const auto data = toml::parse(floc);
   const auto& table = toml::find(data, "Solvers");
   std::string _assembly = toml::find_or<std::string>(table, "assembly", "FULL");
   if ((_assembly == "FULL") || (_assembly == "full")) {
      assembly = Assembly::FULL;
   }
   else if ((_assembly == "PA") || (_assembly == "pa")) {
      assembly = Assembly::PA;
   }
   else if ((_assembly == "EA") || (_assembly == "ea")) {
      assembly = Assembly::EA;
   }
   else {
      MFEM_ABORT("Solvers.assembly was not provided a valid type.");
      assembly = Assembly::NOTYPE;
   }

   std::string _rtmodel = toml::find_or<std::string>(table, "rtmodel", "CPU");
   if ((_rtmodel == "CPU") || (_rtmodel == "cpu")) {
      rtmodel = RTModel::CPU;
   }
#if defined(RAJA_ENABLE_OPENMP)
   else if ((_rtmodel == "OPENMP") || (_rtmodel == "OpenMP")|| (_rtmodel == "openmp")) {
      rtmodel = RTModel::OPENMP;
   }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
   else if ((_rtmodel == "GPU") || (_rtmodel == "gpu")) {
      if (assembly == Assembly::FULL) {
         MFEM_ABORT("Solvers.rtmodel can't be GPU if Solvers.rtmodel is FULL.");
      }
      rtmodel = RTModel::GPU;
   }
#endif
   else {
      MFEM_ABORT("Solvers.rtmodel was not provided a valid type.");
      rtmodel = RTModel::NOTYPE;
   }

   if (table.contains("NR")) {
      // Obtaining information related to the newton raphson solver
      const auto& nr_table = toml::find(table, "NR");
      std::string _solver = toml::find_or<std::string>(nr_table, "nl_solver", "NR");
      if ((_solver == "nr") || (_solver == "NR")) {
         nl_solver = NLSolver::NR;
      }
      else if ((_solver == "nrls") || (_solver == "NRLS")) {
         nl_solver = NLSolver::NRLS;
      }
      else {
         MFEM_ABORT("Solvers.NR.nl_solver was not provided a valid type.");
         nl_solver = NLSolver::NOTYPE;
      }
      newton_iter = toml::find_or<int>(nr_table, "iter", 25);
      newton_rel_tol = toml::find_or<double>(nr_table, "rel_tol", 1e-5);
      newton_abs_tol = toml::find_or<double>(nr_table, "abs_tol", 1e-10);
   } // end of NR info

   std::string _integ_model = toml::find_or<std::string>(table, "integ_model", "FULL");
   if ((_integ_model == "FULL") || (_integ_model == "full")) {
      integ_type = IntegrationType::FULL;
   }
   else if ((_integ_model == "BBAR") || (_integ_model == "bbar")) {
      integ_type = IntegrationType::BBAR;
      if (nl_solver == NLSolver::NR) {
         std::cout << "BBar method performs better when paired with a NR solver with line search" << std::endl;
      }
   }

   if (table.contains("Krylov")) {
      // Now getting information about the Krylov solvers used to the linearized
      // system of equations of the nonlinear problem.
      auto iter_table = toml::find(table, "Krylov");
      krylov_iter = toml::find_or<int>(iter_table, "iter", 200);
      krylov_rel_tol = toml::find_or<double>(iter_table, "rel_tol", 1e-10);
      krylov_abs_tol = toml::find_or<double>(iter_table, "abs_tol", 1e-30);
      std::string _solver = toml::find_or<std::string>(iter_table, "solver", "GMRES");
      if ((_solver == "GMRES") || (_solver == "gmres")) {
         solver = KrylovSolver::GMRES;
      }
      else if ((_solver == "PCG") || (_solver == "pcg")) {
         solver = KrylovSolver::PCG;
      }
      else if ((_solver == "MINRES") || (_solver == "minres")) {
         solver = KrylovSolver::MINRES;
      }
      else {
         MFEM_ABORT("Solvers.Krylov.solver was not provided a valid type.");
         solver = KrylovSolver::NOTYPE;
      }
   } // end of krylov solver info
} // end of solver parsing

// From the toml file it finds all the values related to the mesh
void ExaOptions::get_mesh()
{
   // Refinement of the mesh and element order
   const auto data = toml::parse(floc);
   const auto& table = toml::find(data, "Mesh");
   ser_ref_levels = toml::find_or<int>(table, "ref_ser", 0);
   par_ref_levels = toml::find_or<int>(table, "ref_par", 0);
   order = toml::find_or<int>(table, "p_refinement", 1);
   // file location of the mesh
   std::string _mesh_file = toml::find_or<std::string>(table, "floc", "../../data/cube-hex-ro.mesh");
   mesh_file = _mesh_file;
   // Type of mesh that we're reading/going to generate
   std::string mtype = toml::find_or<std::string>(table, "type", "other");
   if ((mtype == "cubit") || (mtype == "Cubit") || (mtype == "CUBIT")) {
      mesh_type = MeshType::CUBIT;
   }
   else if ((mtype == "auto") || (mtype == "Auto") || (mtype == "AUTO")) {
      mesh_type = MeshType::AUTO;
      if (table.contains("Auto")){
         auto auto_table = toml::find(table, "Auto");
         std::vector<double> _mxyz = toml::find<std::vector<double>>(auto_table, "length");
         if (_mxyz.size() != 3) {
            MFEM_ABORT("Mesh.Auto.length was not provided a valid array of size 3.");
         }
         mxyz[0] = _mxyz[0];
         mxyz[1] = _mxyz[1];
         mxyz[2] = _mxyz[2];

         std::vector<int> _nxyz = toml::find<std::vector<int>>(auto_table, "ncuts");
         if (_nxyz.size() != 3) {
            MFEM_ABORT("Mesh.Auto.ncuts was not provided a valid array of size 3.");
         }
         nxyz[0] = _nxyz[0];
         nxyz[1] = _nxyz[1];
         nxyz[2] = _nxyz[2]; 
      } 
      else {
         MFEM_ABORT("Mesh.type was set to Auto but Mesh.Auto does not exist");
      }
   }
   else if ((mtype == "other") || (mtype == "Other") || (mtype == "OTHER")) {
      mesh_type = MeshType::OTHER;
   }
   else {
      MFEM_ABORT("Mesh.type was not provided a valid type.");
      mesh_type = MeshType::NOTYPE;
   } // end of mesh type parsing

   if (mesh_type == MeshType::OTHER || mesh_type == MeshType::CUBIT) {
      if (!if_file_exists(mesh_file))
      {
         MFEM_ABORT("Mesh file does not exist");
      }
   }
} // End of mesh parsing

void ExaOptions::print_options()
{
   std::cout << "Mesh file location: " << mesh_file << std::endl;
   std::cout << "Mesh type: ";
   if (mesh_type == MeshType::OTHER) {
      std::cout << "other";
   }
   else if (mesh_type == MeshType::CUBIT) {
      std::cout << "cubit";
   }
   else {
      std::cout << "auto";
   }
   std::cout << std::endl;

   std::cout << "Edge dimensions (mx, my, mz): " << mxyz[0] << " " << mxyz[1] << " " << mxyz[2] << std::endl;
   std::cout << "Number of cells on an edge (nx, ny, nz): " << nxyz[0] << " " << nxyz[1] << " " << nxyz[2] << std::endl;

   std::cout << "Serial Refinement level: " << ser_ref_levels << std::endl;
   std::cout << "Parallel Refinement level: " << par_ref_levels << std::endl;
   std::cout << "P-refinement level: " << order << std::endl;

   std::cout << std::boolalpha;
   if (dt_cust) {
      std::cout << "Custom time stepping on" << std::endl;
      std::cout << "Number of time steps (nsteps): " << nsteps << std::endl;
      std::cout << "Custom time file loc (dt_file): " << dt_file << std::endl;
   }
   else if (dt_auto)
   {
      std::cout << "Auto time stepping on" << std::endl;
      std::cout << "Final time (t_final): " << t_final << std::endl;
      std::cout << "Initial time step (dt): " << dt << std::endl;
      std::cout << "Minimum time step (dt): " << dt_min << std::endl;
      std::cout << "Time step scale factor: " << dt_scale << std::endl;
      std::cout << "Auto time step output file: " << dt_file << std::endl;
   }
   else
   {
      std::cout << "Constant time stepping on" << std::endl;
      std::cout << "Final time (t_final): " << t_final << std::endl;
      std::cout << "Time step (dt): " << dt << std::endl;
   }

   std::cout << "Visit flag: " << visit << std::endl;
   std::cout << "Conduit flag: " << conduit << std::endl;
   std::cout << "Paraview flag: " << paraview << std::endl;
   std::cout << "ADIOS2 flag: " << adios2 << std::endl;
   std::cout << "Visualization steps: " << vis_steps << std::endl;
   std::cout << "Visualization directory: " << basename << std::endl;

   std::cout << "Average stress filename: " << avg_stress_fname << std::endl;
   if (additional_avgs)
   {
      std::cout << "Additional averages being computed" << std::endl;
      std::cout << "Average deformation gradient filename: " << avg_def_grad_fname << std::endl;
      std::cout << "Average plastic work filename: " << avg_pl_work_fname << std::endl;
   }
   else
   {
      std::cout << "No additional averages being computed" << std::endl;
   }
   std::cout << "Average stress filename: " << avg_stress_fname << std::endl;
   std::cout << "Light-up flag: " << light_up << std::endl;

   if (light_up) {
      for (auto& hkl : light_hkls) {
         std::array<double, 3> hkl_tmp = {hkl[0], hkl[1], hkl[2]};
         std::cout << "light-up: hkls " << hkl_tmp[0] << " " <<  hkl_tmp[1] << " " << hkl_tmp[2] << std::endl;
      }
      std::cout << "light-up: distance tolerance " << light_dist_tol << std::endl;
      std::cout << "light-up: s direction " << light_s_dir[0] << " " <<  light_s_dir[1] << " " << light_s_dir[2] << std::endl;
      std::cout << "light-up: lattice params " << lattice_params[0] << " " <<  lattice_params[1] << " " << lattice_params[2] << std::endl;
      std::cout << "light-up: lattice basename: " << lattice_basename << std::endl;
   }

   if (nl_solver == NLSolver::NR) {
      std::cout << "Nonlinear Solver is Newton Raphson" << std::endl;
   }
   else if (nl_solver == NLSolver::NRLS) {
      std::cout << "Nonlinear Solver is Newton Raphson with a line search" << std::endl;
   }

   std::cout << "Newton Raphson rel. tol.: " << newton_rel_tol << std::endl;
   std::cout << "Newton Raphson abs. tol.: " << newton_abs_tol << std::endl;
   std::cout << "Newton Raphson # of iter.: " << newton_iter << std::endl;
   std::cout << "Newton Raphson grad debug: " << grad_debug << std::endl;

   if (integ_type == IntegrationType::FULL) {
      std::cout << "Integration Type: Full" << std::endl;
   }
   else if (integ_type == IntegrationType::BBAR) {
      std::cout << "Integration Type: BBar" << std::endl;
   }

   std::cout << "Krylov solver: ";
   if (solver == KrylovSolver::GMRES) {
      std::cout << "GMRES";
   }
   else if (solver == KrylovSolver::PCG) {
      std::cout << "PCG";
   }
   else {
      std::cout << "MINRES";
   }
   std::cout << std::endl;

   std::cout << "Krylov solver rel. tol.: " << krylov_rel_tol << std::endl;
   std::cout << "Krylov solver abs. tol.: " << krylov_abs_tol << std::endl;
   std::cout << "Krylov solver # of iter.: " << krylov_iter << std::endl;

   std::cout << "Matrix Assembly is: ";
   if (assembly == Assembly::FULL) {
      std::cout << "Full Assembly" << std::endl;
   }
   else if (assembly == Assembly::PA) {
      std::cout << "Partial Assembly" << std::endl;
   }
   else {
      std::cout << "Element Assembly" << std::endl;
   }

   std::cout << "Runtime model is: ";
   if (rtmodel == RTModel::CPU) {
      std::cout << "CPU" << std::endl;
   }
   else if (rtmodel == RTModel::GPU) {
      std::cout << "GPU" << std::endl;
   }
   else if (rtmodel == RTModel::OPENMP) {
      std::cout << "OpenMP" << std::endl;
   }

   std::cout << "Mechanical model library being used ";

   if (mech_type == MechType::UMAT) {
      std::cout << "UMAT" << std::endl;
   }
   else if (mech_type == MechType::EXACMECH) {

      auto shortcut_delim = [](std::string & str, std::string delim) -> std::vector<std::string> {
         auto start = 0U;
         auto end = str.find(delim);
         std::vector<std::string> sdelim;
         while (end != std::string::npos)
         {
            sdelim.push_back(str.substr(start, end - start));
            start = end + delim.length();
            end = str.find(delim, start);
         }
         sdelim.push_back(str.substr(start, end - start));
         return sdelim;
      };

      auto sdelim = shortcut_delim(shortcut, "_");

      std::cout << "ExaCMech" << std::endl;
      std::cout << "ExaCMech shortcut name: " << shortcut << std::endl;
      std::cout << "Crystal symmetry group is " << sdelim[1] << std::endl;
   }

   std::cout << "Xtal Plasticity being used: " << cp << std::endl;

   std::cout << "Orientation file location: " << ori_file << std::endl;
   std::cout << "Grain map file location: " << grain_map << std::endl;
   std::cout << "Number of grains: " << ngrains << std::endl;

   std::cout << "Orientation type: ";
   if (ori_type == OriType::EULER) {
      std::cout << "euler";
   }
   else if (ori_type == OriType::QUAT) {
      std::cout << "quaternion";
   }
   else {
      std::cout << "custom";
   }
   std::cout << std::endl;

   std::cout << "Custom stride to read grain map file: " << grain_custom_stride << std::endl;
   std::cout << "Orientation offset in state variable file: " << grain_statevar_offset << std::endl;

   std::cout << "Number of properties: " << nProps << std::endl;
   std::cout << "Property file location: " << props_file << std::endl;

   std::cout << "Number of state variables: " << numStateVars << std::endl;
   std::cout << "State variable file location: " << state_file << std::endl;

   for (const auto key: updateStep)
   {
      std::cout << "Starting on step " << key << " essential BCs values are:" << std::endl;
      std::cout << "Essential ids are set as: ";
      for (const auto & val: map_ess_id["total"][key]) {
         std::cout << val << " ";
      }
      std::cout << std::endl << "Essential components are set as: ";
      for (const auto & val: map_ess_comp["total"][key]) {
         std::cout << val << " ";
      }
      if (map_ess_vel[key].size() > 0) {
         std::cout << std::endl << "Essential boundary velocity values are set as: ";
         for (const auto & val: map_ess_vel.at(key)) {
            std::cout << val << " ";
         }
      }
      if (map_ess_vgrad[key].size() > 0) {
         std::cout << std::endl << "Essential boundary velocity gradients are set as: ";
         for (const auto & val: map_ess_vgrad.at(key)) {
            std::cout << val << " ";
         }
      }
      std::cout << std::endl;
   }
} // End of printing out options
