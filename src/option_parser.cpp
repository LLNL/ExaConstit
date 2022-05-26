
#include "option_parser.hpp"
#include "RAJA/RAJA.hpp"
#include "TOML_Reader/toml.hpp"
#include "mfem.hpp"
#include "ECMech_cases.h"
#include "ECMech_evptnWrap.h"
#include "ECMech_const.h"
#include <iostream>
#include <fstream>

inline bool if_file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

namespace {
   typedef ecmech::evptn::matModel<ecmech::SlipGeom_BCC_A, ecmech::Kin_FCC_A, 
         ecmech::evptn::ThermoElastNCubic, ecmech::EosModelConst<false>>
         VoceBCCModel;
   typedef ecmech::evptn::matModel<ecmech::SlipGeom_BCC_A, ecmech::Kin_FCC_AH, 
            ecmech::evptn::ThermoElastNCubic, ecmech::EosModelConst<false>>
            VoceNLBCCModel;
}
// my_id corresponds to the processor id.
void ExaOptions::parse_options(int my_id)
{
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
   // From the toml file it finds all the values related to the mesh
   get_mesh();
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
   if (!changing_bcs) {
      std::vector<int> _essential_ids = toml::find<std::vector<int>>(table, "essential_ids");
      if (_essential_ids.empty()) {
         MFEM_ABORT("BCs.essential_ids was not provided any values.");
      }
      map_ess_id[0] = std::vector<int>();
      map_ess_id[1] = _essential_ids;

      std::vector<int> _essential_comp = toml::find<std::vector<int>>(table, "essential_comps");
      if (_essential_comp.empty()) {
         MFEM_ABORT("BCs.essential_comps was not provided any values.");
      }

      map_ess_comp[0] = std::vector<int>();
      map_ess_comp[1] = _essential_comp;

      // Getting out arrays of values isn't always the simplest thing to do using
      // this TOML libary.
      std::vector<double> _essential_vals = toml::find<std::vector<double>>(table, "essential_vals");
      if (_essential_vals.empty()) {
         MFEM_ABORT("BCs.essential_vals was not provided any values.");
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
      map_ess_id[0] = std::vector<int>();
      for (const auto &vec : nested_ess_ids) {
         int key = updateStep.at(ilength);
         map_ess_id[key] = std::vector<int>();
         for (const auto &val : vec) {
            map_ess_id[key].push_back(val);
         }
         if (map_ess_id[key].empty()) {
            MFEM_ABORT("BCs.essential_ids contains empty array.");
         }
         ilength += 1;
      }

      if (ilength != size) {
         MFEM_ABORT("BCs.essential_ids did not contain the same number of arrays as number of update steps");
      }

      std::vector<std::vector<int>> nested_ess_comps = toml::find<std::vector<std::vector<int>>>(table, "essential_comps");
      ilength = 0;
      map_ess_comp[0] = std::vector<int>();
      for (const auto &vec : nested_ess_comps) {
         int key = updateStep.at(ilength);
         map_ess_comp[key] = std::vector<int>();
         for (const auto &val : vec) {
            map_ess_comp[key].push_back(val);
         }
         if (map_ess_comp[key].empty()) {
            MFEM_ABORT("BCs.essential_comps contains empty array.");
         }
         ilength += 1;
      }

      if (ilength != size) {
         MFEM_ABORT("BCs.essential_comps did not contain the same number of arrays as number of update steps");
      }

      std::vector<std::vector<double>> nested_ess_vals = toml::find<std::vector<std::vector<double>>>(table, "essential_vals");
      ilength = 0;
      map_ess_vel[0] = std::vector<double>();
      for (const auto &vec : nested_ess_vals) {
         int key = updateStep.at(ilength);
         map_ess_vel[key] = std::vector<double>();
         for (const auto &val : vec) {
            map_ess_vel[key].push_back(val);
         }
         if (map_ess_vel[key].empty()) {
            MFEM_ABORT("BCs.essential_vals contains empty array.");
         }
         ilength += 1;
      }

      if (ilength != size) {
         MFEM_ABORT("BCs.essential_vals did not contain the same number of arrays as number of update steps");
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
         xtal_type = XtalType::NOTYPE;
      }

      grain_statevar_offset = ecmech::evptn::iHistLbQ;

      if(table.contains("ExaCMech")) {
         const auto& exacmech_table = toml::find(table, "ExaCMech");
         std::string _xtal_type = toml::find_or<std::string>(exacmech_table, "xtal_type", "");
         std::string _slip_type = toml::find_or<std::string>(exacmech_table, "slip_type", "");

         if ((_xtal_type == "fcc") || (_xtal_type == "FCC")) {
            xtal_type = XtalType::FCC;
         }
         else if ((_xtal_type == "bcc") || (_xtal_type == "BCC")) {
            xtal_type = XtalType::BCC;
         }
         else if ((_xtal_type == "hcp") || (_xtal_type == "HCP")) {
            xtal_type = XtalType::HCP;
         }
         else {
            MFEM_ABORT("Model.ExaCMech.xtal_type was not provided a valid type.");
            xtal_type = XtalType::NOTYPE;
         }

         if ((_slip_type == "mts") || (_slip_type == "MTS") || (_slip_type == "mtsdd") || (_slip_type == "MTSDD")) {
            slip_type = SlipType::MTSDD;
            if (xtal_type == XtalType::FCC) {
               if (nProps != ecmech::matModelEvptn_FCC_B::nParams) {
                  MFEM_ABORT("Properties.Matl_Props.num_props needs " << ecmech::matModelEvptn_FCC_B::nParams <<
                           " values for the MTSDD option and FCC option");
               }
            }
            else if (xtal_type == XtalType::BCC) {
               if (nProps != ecmech::matModelEvptn_BCC_A::nParams) {
                  MFEM_ABORT("Properties.Matl_Props.num_props needs " << ecmech::matModelEvptn_BCC_A::nParams <<
                           " values for the MTSDD option and BCC option");
               }
            }
            else if (xtal_type == XtalType::HCP) {
               if (nProps != ecmech::matModelEvptn_HCP_A::nParams) {
                  MFEM_ABORT("Properties.Matl_Props.num_props needs " << ecmech::matModelEvptn_HCP_A::nParams <<
                           " values for the MTSDD option and HCP option");
               }
            }
            else {
               MFEM_ABORT("Model.ExaCMech.slip_type can not be MTS for BCC materials.")
            }
         }
         else if ((_slip_type == "powervoce") || (_slip_type == "PowerVoce") || (_slip_type == "POWERVOCE")) {
            slip_type = SlipType::POWERVOCE;
            if (xtal_type == XtalType::FCC) {
               if (nProps != ecmech::matModelEvptn_FCC_A::nParams) {
                  MFEM_ABORT("Properties.Matl_Props.num_props needs " << ecmech::matModelEvptn_FCC_A::nParams <<
                           " values for the PowerVoce option and FCC option");
               }
            }
            else if (xtal_type == XtalType::BCC) {
               if (nProps != VoceBCCModel::nParams) {
                  MFEM_ABORT("Properties.Matl_Props.num_props needs " << VoceBCCModel::nParams <<
                           " values for the PowerVoce option and BCC option");
               }
            }  
            else {
               MFEM_ABORT("Model.ExaCMech.slip_type can not be PowerVoce for HCP or BCC_112 materials.")
            }
         }
         else if ((_slip_type == "powervocenl") || (_slip_type == "PowerVoceNL") || (_slip_type == "POWERVOCENL")) {
            slip_type = SlipType::POWERVOCENL;
            if (xtal_type == XtalType::FCC) {
               if (nProps != ecmech::matModelEvptn_FCC_AH::nParams) {
                  MFEM_ABORT("Properties.Matl_Props.num_props needs " << ecmech::matModelEvptn_FCC_AH::nParams <<
                           " values for the PowerVoceNL option and FCC option");
               }
            }
            else if (xtal_type == XtalType::BCC) {
               if (nProps != VoceNLBCCModel::nParams) {
                  MFEM_ABORT("Properties.Matl_Props.num_props needs " << VoceNLBCCModel::nParams <<
                           " values for the PowerVoceNL option and BCC option");
               }
            }
            else {
               MFEM_ABORT("Model.ExaCMech.slip_type can not be PowerVoceNL for HCP or BCC_112 materials.")
            }
         }
         else {
            MFEM_ABORT("Model.ExaCMech.slip_type was not provided a valid type.");
            slip_type = SlipType::NOTYPE;
         }

         if (slip_type != SlipType::NOTYPE) {
            if (xtal_type == XtalType::FCC) {
               int num_state_vars_check = ecmech::matModelEvptn_FCC_A::numHist + ecmech::ne + 1 - 4;
               if (numStateVars != num_state_vars_check) {
                  MFEM_ABORT("Properties.State_Vars.num_vars needs " << num_state_vars_check << " values for a "
                           "face cubic material when using an ExaCMech model. Note: the number of values for a quaternion "
                           "are not included in this count.");
               }
            }
            else if (xtal_type == XtalType::BCC) {
               // We'll probably need to modify this whenever we add support for the other BCC variations in
               // here due to the change in number of slip systems.
               int num_state_vars_check = ecmech::matModelEvptn_BCC_A::numHist + ecmech::ne + 1 - 4;
               if (numStateVars != num_state_vars_check) {
                  MFEM_ABORT("Properties.State_Vars.num_vars needs " << num_state_vars_check << " values for a "
                           "body center cubic material when using an ExaCMech model. Note: the number of values for a quaternion "
                           "are not included in this count.");
               }
            }
            else if (xtal_type == XtalType::HCP) {
               int num_state_vars_check = ecmech::matModelEvptn_HCP_A::numHist + ecmech::ne + 1 - 4;
               if (numStateVars != num_state_vars_check) {
                  MFEM_ABORT("Properties.State_Vars.num_vars needs " << num_state_vars_check << " values for a "
                           "hexagonal material when using an ExaCMech model. Note: the number of values for a quaternion "
                           "are not included in this count.");
               }
            }
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
   std::string _avg_dp_tensor_fname = toml::find_or<std::string>(table, "avg_dp_tensor_fname", "avg_dp_tensor.txt");
   avg_dp_tensor_fname = _avg_dp_tensor_fname;
   light_up = toml::find_or<bool>(table, "light_up", false);
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
#if defined(RAJA_ENABLE_CUDA)
   else if ((_rtmodel == "CUDA") || (_rtmodel == "cuda")) {
      if (assembly == Assembly::FULL) {
         MFEM_ABORT("Solvers.rtmodel can't be CUDA if Solvers.rtmodel is FULL.");
      }
      rtmodel = RTModel::CUDA;
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
   std::cout << "Custom dt flag (dt_cust): " << dt_cust << std::endl;

   if (dt_cust) {
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
      std::cout << "Average plastic strain rate tensor filename: " << avg_dp_tensor_fname << std::endl;
   }
   else
   {
      std::cout << "No additional averages being computed" << std::endl;
   }
   std::cout << "Average stress filename: " << avg_stress_fname << std::endl;
   std::cout << "Light-up flag: " << light_up << std::endl;

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
   else if (rtmodel == RTModel::CUDA) {
      std::cout << "CUDA" << std::endl;
   }
   else if (rtmodel == RTModel::OPENMP) {
      std::cout << "OpenMP" << std::endl;
   }

   std::cout << "Mechanical model library being used ";

   if (mech_type == MechType::UMAT) {
      std::cout << "UMAT" << std::endl;
   }
   else if (mech_type == MechType::EXACMECH) {
      std::cout << "ExaCMech" << std::endl;
      std::cout << "Crystal symmetry group is ";
      if (xtal_type == XtalType::FCC) {
         std::cout << "FCC" << std::endl;
      }
      else if (xtal_type == XtalType::BCC) {
         std::cout << "BCC" << std::endl;
      }
      else if (xtal_type == XtalType::HCP) {
         std::cout << "HCP" << std::endl;
      }

      std::cout << "Slip system and hardening model being used is ";

      if (slip_type == SlipType::MTSDD) {
         std::cout << "MTS slip like kinetics with dislocation density based hardening" << std::endl;
      }
      else if (slip_type == SlipType::POWERVOCE) {
         std::cout << "Power law slip kinetics with a linear Voce hardening law" << std::endl;
      }
      else if (slip_type == SlipType::POWERVOCENL) {
         std::cout << "Power law slip kinetics with a nonlinear Voce hardening law" << std::endl;
      }
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
      for (const auto & val: map_ess_id.at(key)) {
         std::cout << val << " ";
      }
      std::cout << std::endl << "Essential components are set as: ";
      for (const auto & val: map_ess_comp.at(key)) {
         std::cout << val << " ";
      }
      std::cout << std::endl << "Essential boundary values are set as: ";
      for (const auto & val: map_ess_vel.at(key)) {
         std::cout << val << " ";
      }
      std::cout << std::endl;
   }
} // End of printing out options
