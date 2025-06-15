// ***********************************************************************
// # ExaConstit App
// ## Author: Robert A. Carson
// carson16@llnl.gov
// Steven R. Wopschall
// wopschall1@llnl.gov
// Jamie Bramwell
// bramwell1@llnl.gov
// Date: Aug. 6, 2017
// Updated: Oct. 7, 2019
//
// # Description:
// The purpose of this code app is to determine bulk constitutive properties of metals.
// This is a nonlinear quasi-static, implicit solid mechanics code built on the MFEM library based
// on an updated Lagrangian formulation (velocity based). Currently, only Dirichlet boundary conditions
// (homogeneous and inhomogeneous by dof component) have been implemented. Neumann (traction) boundary
// conditions and a body force are not implemented. A new ExaModel class allows one to implement
// arbitrary constitutive models. The code currently successfully allows for various UMATs to be
// interfaced within the code framework. Development work is currently focused on allowing for the
// mechanical models to run on GPGPUs. The code supports either constant time steps or user supplied
// delta time steps. Boundary conditions are supplied for the velocity field applied on a surface.
// It supports a number of different preconditioned Krylov iterative solvers (PCG, GMRES, MINRES)
// for either symmetric or nonsymmetric positive-definite systems.
//
// ## Remark:
// See the included options.toml to see all of the various different options that are allowable in this
// code and their default values. A TOML parser has been included within this directory, since it has
// an MIT license. The repository for it can be found at: https://github.com/skystrife/cpptoml . Example
// UMATs maybe obtained from https://web.njit.edu/~sac3/Software.html . We have not included them due to
// a question of licensing. The ones that have been run and are known to work are the linear elasticity
// model and the neo-Hookean material. Although, we might be able to provide an example interface so
// users can base their interface/build scripts off of what's known to work.
// Note: the grain.txt, props.txt and state.txt files are expected inputs for CP problems,
// specifically ones that use the Abaqus UMAT interface class under the ExaModel.
//
// # Installing Notes:
// * git clone the LLNL BLT library into cmake directory. It can be obtained at https://github.com/LLNL/blt.git
// * MFEM will need to be built with Conduit (built with HDF5). The easiest way to install Conduit
// is to use spack install instruction provided by Conduit.
// * ExaCMech is required for ExaConstit to be built and can be obtained at https://github.com/LLNL/ExaCMech.git.
// * Create a build directory and cd into there
// * Run ```cmake .. -DENABLE_MPI=ON -DENABLE_FORTRAN=ON -DMFEM_DIR{mfem's installed cmake location}
// -DBLT_SOURCE_DIR=${BLT cloned location} -DECMECH_DIR=${ExaCMech installed cmake location}
// -DRAJA_DIR={RAJA installed location} -DSNLS_DIR={SNLS location in ExaCMech}
// -DMETIS_DIR={Metis used in mfem location} -DHYPRE_DIR={HYPRE install location}
// -DCONDUIT_DIR={Conduit install location} -DHDF5_ROOT:PATH={HDF5 install location}```
// * Run ```make -j 4```
//
// #  Future Implemenations Notes:
// * Visco-plasticity constitutive model
// * GPGPU material models
// * A more in-depth README that better covers the different options available.
// * debug ability to read different mesh formats
// * An up-to-date example options.toml file
// ***********************************************************************
#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_log.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "mfem_expt/partial_qfunc.hpp"
#include "sim_state/simulation_state.hpp"
#include "postprocessing/postprocessing_driver.hpp"
#include "system_driver.hpp"
#include "BCData.hpp"
#include "BCManager.hpp"
#include "options/option_parser_v2.hpp"
#include <string>
#include <sstream>

using namespace mfem;

// set kinematic functions and boundary condition functions
void ReferenceConfiguration(const Vector &x, Vector &y);

// This initializes some grid function
void InitGridFunction(const Vector & /*x*/, Vector &y);

// material input check routine
bool checkMaterialArgs(MechType mt, bool cp, int ngrains, int numProps,
                       int numStateVars);

// initialize a quadrature function with a single input value, val.
void initQuadFunc(QuadratureFunction *qf, double val);

// initialize a quadrature function that is really a tensor with the identity matrix.
// currently only works for 3x3 tensors.
void initQuadFuncTensorIdentity(QuadratureFunction *qf, ParFiniteElementSpace *fes);

// set the time step on the boundary condition objects
void setBCTimeStep(double dt, int nDBC);

// Projects the element attribute to GridFunction nodes
// This also assumes the GridFunction is an L2 FE space
void projectElemAttr2GridFunc(Mesh *mesh, ParGridFunction *elem_attr);

int main(int argc, char *argv[])
{
   CALI_INIT
   CALI_CXX_MARK_FUNCTION;
   CALI_MARK_BEGIN("main_driver_init");
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#if (MFEM_HYPRE_VERSION >= 21900)
   Hypre::Init();
#endif
// Used to scope the main program away from the main MPI Init and Finalize calls
{
   // Here we start a timer to time everything
   double start = MPI_Wtime();
   // Here we're going to measure the times of each solve.
   // It'll give us a good idea of strong and weak scaling in
   // comparison to the global value of things.
   // It'll make it easier to point out where some scaling issues might
   // be occurring.
   std::vector<double> times;
   double t1, t2;
   // print the version of the code being run
   if (myid == 0) {
      printf("MFEM Version: %d \n", GetVersion());
   }

   // All of our options are parsed in this file by default
   const char *toml_file = "options.toml";

   // We're going to use the below to allow us to easily swap between different option files
   OptionsParser args(argc, argv);
   args.AddOption(&toml_file, "-opt", "--option", "Option file to use.");
   args.Parse();
   if (!args.Good()) {
      if (myid == 0) {
         args.PrintUsage(std::cout);
      }
      CALI_MARK_END("main_driver_init");
      MPI_Finalize();
      return 1;
   }

   ExaOptions toml_opt;
   toml_opt.parse_options(toml_file, myid);
   toml_opt.print_options();

   // Set the device info here:
   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   // The current backend priority from highest to lowest is: 'occa-cuda',
   // 'raja-cuda', 'cuda', 'occa-omp', 'raja-omp', 'omp', 'occa-cpu', 'raja-cpu', 'cpu'.

   std::string device_config = "cpu";

   if (toml_opt.solvers.rtmodel == RTModel::CPU) {
      device_config = "cpu";
   }
   else if (toml_opt.solvers.rtmodel == RTModel::OPENMP) {
      device_config = "raja-omp";
   }
   else if (toml_opt.solvers.rtmodel == RTModel::GPU) {
#if defined(RAJA_ENABLE_CUDA) 
      device_config = "raja-cuda";
#elif defined(RAJA_ENABLE_HIP)
      device_config = "raja-hip";
#endif
   }
   Device device;

   if (toml_opt.solvers.rtmodel == RTModel::GPU)
   {
      device.SetMemoryTypes(MemoryType::HOST_64, MemoryType::DEVICE);
   }

   device.Configure(device_config.c_str());

   if (myid == 0) {
      printf("\n");
      device.Print();
      printf("\n");
   }

   SimulationState sim_state(toml_opt);

   auto pmesh = sim_state.getMesh();

   CALI_MARK_END("main_driver_init");

   if (myid == 0) {
      printf("after mesh section. \n");
   }

   const int dim = pmesh->Dimension();

   // Define the finite element spaces for displacement field
   // fix me: this eventually needs to be updated to using the postprocessing class
   auto& mat_0 = toml_opt.materials[0];

   auto fe_space = sim_state.GetMeshParFiniteElementSpace();
   auto l2_fes = sim_state.GetParFiniteElementSpace(1);
   auto l2_fes_pl = sim_state.GetParFiniteElementSpace(1);
   auto l2_fes_ori = sim_state.GetParFiniteElementSpace(4);
   auto l2_fes_cen = sim_state.GetParFiniteElementSpace(dim);
   auto l2_fes_voigt = sim_state.GetParFiniteElementSpace(6);
   auto l2_fes_tens = sim_state.GetParFiniteElementSpace(9);
   const int num_hard = (mat_0.model.exacmech) ? mat_0.model.exacmech->hard_size : 1;
   auto l2_fes_hard = sim_state.GetParFiniteElementSpace(num_hard);
   const int num_gdot = (mat_0.model.exacmech) ? mat_0.model.exacmech->gdot_size : 1;
   auto l2_fes_gdots = sim_state.GetParFiniteElementSpace(num_gdot);

   ParGridFunction vonMises(l2_fes.get());
   vonMises = 0.0;
   ParGridFunction volume(l2_fes.get());
   ParGridFunction hydroStress(l2_fes.get());
   hydroStress = 0.0;
   ParGridFunction stress(l2_fes_voigt.get());
   stress = 0.0;
   // Only used for light-up scripts at this point
   ParGridFunction *elem_centroid = nullptr;
   ParGridFunction *elastic_strain = nullptr;
#ifdef MFEM_USE_ADIOS2
   ParGridFunction *elem_attr = nullptr;
   if (toml_opt.visualization.adios2) {
      elem_attr = new ParGridFunction(l2_fes.get());
      // projectElemAttr2GridFunc(pmesh, elem_attr);
   }
#endif

   ParGridFunction dpeff(l2_fes_pl.get());
   ParGridFunction pleff(l2_fes_pl.get());
   ParGridFunction hardness(l2_fes_hard.get());
   ParGridFunction quats(l2_fes_ori.get());
   ParGridFunction gdots(l2_fes_gdots.get());

   if (mat_0.mech_type == MechType::EXACMECH) {
      if (toml_opt.post_processing.light_up.enabled) {
         elem_centroid = new ParGridFunction(l2_fes_cen.get());
         elastic_strain = new ParGridFunction(l2_fes_voigt.get());
      }
   }

   HYPRE_Int glob_size = fe_space->GlobalTrueVSize();

   pmesh->PrintInfo();

   // Print the mesh statistics
   if (myid == 0) {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << glob_size << "\n";
      std::cout << "***********************************************************\n";
   }

   // Used for post processing steps
   // QuadratureSpace qspace0(pmesh, 1);
   // QuadratureFunction elemMatVars(&qspace0, 1);
   // elemMatVars = 0.0;

   // read in material properties and state variables files for use with ALL models
   // store input data on Vector object. The material properties vector will be
   // passed into the Nonlinear mech operator constructor to initialize the material
   // properties vector on the model and the state variables vector will be used with
   // the grain data vector (if crystal plasticity) to populate the material state
   // vector quadrature function. It is assumed that the state variables input file
   // are initial values for all state variables applied to all quadrature points.
   // There is not a separate initialization file for each quadrature point

   if (myid == 0) {
      printf("before reading in matProps and stateVars. \n");
   }

   // Define a grid function for the global reference configuration, the beginning
   // step configuration, the global deformation, the current configuration/solution
   // guess, and the incremental nodal displacements
   // x_diff would be our displacement
   auto x_diff = sim_state.getDisplacement();
   auto v_cur = sim_state.getVelocity();


   // Define grid function for the velocity solution grid function
   // WITH Dirichlet BCs

   // Define a VectorFunctionCoefficient to initialize a grid function
   VectorFunctionCoefficient init_grid_func(dim, InitGridFunction);

   // initialize boundary condition, velocity, and
   // incremental nodal displacment grid functions by projection the
   // VectorFunctionCoefficient function onto them
   x_diff->ProjectCoefficient(init_grid_func);
   v_cur->ProjectCoefficient(init_grid_func);

   // Construct the nonlinear mechanics operator. Note that q_grain0 is
   // being passed as the matVars0 quadarture function. This is the only
   // history variable considered at this moment. Consider generalizing
   // this where the grain info is a possible subset only of some
   // material history variable quadrature function. Also handle the
   // case where there is no grain data.
   if (myid == 0) {
      printf("before SystemDriver constructor. \n");
   }

   SystemDriver oper(sim_state);

   /*
      if (toml_opt.visualization.visit || toml_opt.visualization.conduit || toml_opt.visualization.paraview || toml_opt.visualization.adios2) {
         oper.ProjectVolume(volume);
      }
   */
   if (myid == 0) {
      printf("after SystemDriver constructor. \n");
   }

   // get the essential true dof list. This may not be used.
   const Array<int> ess_tdof_list = oper.GetEssTDofList();

   // declare incremental nodal displacement solution vector
   // Vector v_prev(fe_space->TrueVSize()); v_prev.UseDevice(true);// this sizing is correct

   // Save data for VisIt visualization.
   // The below is used to take advantage of mfem's custom Visit plugin
   // It could also allow for restart files later on.
   // If we have large simulations although the current method of printing everything
   // as text will cause issues. The data should really be saved in some binary format.
   // If you don't then you'll often find that the printed data lags behind where
   // the simulation is currently at. This really becomes noticiable if you have
   // a lot of data that you want to output for the user. It might be nice if this
   // was either a netcdf or hdf5 type format instead.
   /*
      fix me
      All of the below needs to be updated to move into the internal PostProcessing variables
   */
   /*
   CALI_MARK_BEGIN("main_vis_init");
   VisItDataCollection visit_dc(toml_opt.basename, pmesh.get());
   ParaViewDataCollection paraview_dc(toml_opt.basename, pmesh.get());
#ifdef MFEM_USE_CONDUIT
   ConduitDataCollection conduit_dc(toml_opt.basename, pmesh.get());
#endif
#ifdef MFEM_USE_ADIOS2
   const std::string basename = toml_opt.basename + ".bp";
   ADIOS2DataCollection *adios2_dc = new ADIOS2DataCollection(MPI_COMM_WORLD, basename, pmesh.get());
#endif
   if (toml_opt.visualization.paraview) {
      paraview_dc.SetLevelsOfDetail(toml_opt.mesh.order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(false);

      paraview_dc.RegisterField("ElementVolume", &volume);

      if (mat_0.mech_type == MechType::EXACMECH) {
         if(toml_opt.post_processing.light_up.enabled) {
            oper.ProjectCentroid(*elem_centroid);
            oper.ProjectElasticStrains(*elastic_strain);
            oper.ProjectOrientation(quats);
            paraview_dc.RegisterField("ElemCentroid", elem_centroid);
            paraview_dc.RegisterField("XtalElasticStrain", elastic_strain);
            paraview_dc.RegisterField("LatticeOrientation", &quats);
         }
      }

      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.Save();

      paraview_dc.RegisterField("Displacement", x_diff.get());
      paraview_dc.RegisterField("Stress", &stress);
      paraview_dc.RegisterField("Velocity", v_cur.get());
      paraview_dc.RegisterField("VonMisesStress", &vonMises);
      paraview_dc.RegisterField("HydrostaticStress", &hydroStress);

      if (mat_0.mech_type == MechType::EXACMECH) {
         // We also want to project the values out originally
         // so our initial values are correct
         oper.ProjectDpEff(dpeff);
         oper.ProjectEffPlasticStrain(pleff);
         oper.ProjectOrientation(quats);
         oper.ProjectShearRate(gdots);
         oper.ProjectH(hardness);

         paraview_dc.RegisterField("DpEff", &dpeff);
         paraview_dc.RegisterField("EffPlasticStrain", &pleff);
         if(!toml_opt.post_processing.light_up.enabled) {
            paraview_dc.RegisterField("LatticeOrientation", &quats);
         }
         paraview_dc.RegisterField("ShearRate", &gdots);
         paraview_dc.RegisterField("Hardness", &hardness);
      }
   }

   if (toml_opt.visualization.visit) {
      visit_dc.SetPrecision(12);

      visit_dc.RegisterField("ElementVolume", &volume);

      if (mat_0.mech_type == MechType::EXACMECH) {
         if(toml_opt.post_processing.light_up.enabled) {
            oper.ProjectCentroid(*elem_centroid);
            oper.ProjectElasticStrains(*elastic_strain);
            oper.ProjectOrientation(quats);
            visit_dc.RegisterField("ElemCentroid", elem_centroid);
            visit_dc.RegisterField("XtalElasticStrain", elastic_strain);
            visit_dc.RegisterField("LatticeOrientation", &quats);
         }
      }

      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();

      visit_dc.RegisterField("Displacement", x_diff.get());
      visit_dc.RegisterField("Stress", &stress);
      visit_dc.RegisterField("Velocity", v_cur.get());
      visit_dc.RegisterField("VonMisesStress", &vonMises);
      visit_dc.RegisterField("HydrostaticStress", &hydroStress);

      if (mat_0.mech_type == MechType::EXACMECH) {
         // We also want to project the values out originally
         // so our initial values are correct

         oper.ProjectDpEff(dpeff);
         oper.ProjectEffPlasticStrain(pleff);
         oper.ProjectOrientation(quats);
         oper.ProjectShearRate(gdots);
         oper.ProjectH(hardness);

         visit_dc.RegisterField("DpEff", &dpeff);
         visit_dc.RegisterField("EffPlasticStrain", &pleff);
         if(!toml_opt.post_processing.light_up.enabled) {
            visit_dc.RegisterField("LatticeOrientation", &quats);
         }
         visit_dc.RegisterField("ShearRate", &gdots);
         visit_dc.RegisterField("Hardness", &hardness);
      }
   }

#ifdef MFEM_USE_CONDUIT
   if (toml_opt.visualization.conduit) {
      // conduit_dc.SetProtocol("json");
      conduit_dc.RegisterField("ElementVolume", &volume);

      conduit_dc.SetCycle(0);
      conduit_dc.SetTime(0.0);
      conduit_dc.Save();

      conduit_dc.RegisterField("Displacement", x_diff.get());
      conduit_dc.RegisterField("Stress", &stress);
      conduit_dc.RegisterField("Velocity", v_cur.get());
      conduit_dc.RegisterField("VonMisesStress", &vonMises);
      conduit_dc.RegisterField("HydrostaticStress", &hydroStress);

      if (mat_0.mech_type == MechType::EXACMECH) {
         // We also want to project the values out originally
         // so our initial values are correct
         oper.ProjectDpEff(dpeff);
         oper.ProjectEffPlasticStrain(pleff);
         oper.ProjectOrientation(quats);
         oper.ProjectShearRate(gdots);
         oper.ProjectH(hardness);

         conduit_dc.RegisterField("DpEff", &dpeff);
         conduit_dc.RegisterField("EffPlasticStrain", &pleff);
         conduit_dc.RegisterField("LatticeOrientation", &quats);
         conduit_dc.RegisterField("ShearRate", &gdots);
         conduit_dc.RegisterField("Hardness", &hardness);
      }
   }
#endif
#ifdef MFEM_USE_ADIOS2
   if (toml_opt.visualization.adios2) {
      adios2_dc->SetParameter("SubStreams", std::to_string(num_procs / 2) );

      adios2_dc->RegisterField("ElementAttribute", elem_attr);
      adios2_dc->RegisterField("ElementVolume", &volume);

      if (mat_0.mech_type == MechType::EXACMECH) {
         if(toml_opt.post_processing.light_up.enabled) {
            oper.ProjectCentroid(*elem_centroid);
            oper.ProjectElasticStrains(*elastic_strain);
            oper.ProjectOrientation(quats);
            adios2_dc->RegisterField("ElemCentroid", elem_centroid);
            adios2_dc->RegisterField("XtalElasticStrain", elastic_strain);
            adios2_dc->RegisterField("LatticeOrientation", &quats);
         }
      }

      adios2_dc->SetCycle(0);
      adios2_dc->SetTime(0.0);
      adios2_dc->Save();

      adios2_dc->DeregisterField("ElementAttribute");
      adios2_dc->RegisterField("Displacement", x_diff.get());
      adios2_dc->RegisterField("Stress", &stress);
      adios2_dc->RegisterField("Velocity", v_cur.get());
      adios2_dc->RegisterField("VonMisesStress", &vonMises);
      adios2_dc->RegisterField("HydrostaticStress", &hydroStress);

      if (mat_0.mech_type == MechType::EXACMECH) {
         // We also want to project the values out originally
         // so our initial values are correct
         oper.ProjectDpEff(dpeff);
         oper.ProjectEffPlasticStrain(pleff);
         oper.ProjectOrientation(quats);
         oper.ProjectShearRate(gdots);
         oper.ProjectH(hardness);

         adios2_dc->RegisterField("DpEff", &dpeff);
         adios2_dc->RegisterField("EffPlasticStrain", &pleff);
         // We should already have this registered if using the light-up script
         if(!toml_opt.post_processing.light_up.enabled) {
            adios2_dc->RegisterField("LatticeOrientation", &quats);
         }
         adios2_dc->RegisterField("ShearRate", &gdots);
         adios2_dc->RegisterField("Hardness", &hardness);
      }
   }
#endif
   if (myid == 0) {
      printf("after visualization if-block \n");
   }
   */
  PostProcessingDriver post_process(sim_state, toml_opt);

   CALI_MARK_END("main_vis_init");
   // initialize/set the time
   oper.SetTime(sim_state.getTime());

   bool last_step = false;

   int ti = 0;
   auto v_sol = sim_state.getPrimalField();
   while (!sim_state.isFinished()) {
      ti++;
      if (myid == 0) {
         std::cout << "Simulation cycle: " << ti << std::endl;
         sim_state.printTimeStats();
      }
      // Get out our current delta time step
      // compute current time
      last_step = sim_state.isLastStep();
      // set time on the simulation variables and the model through the
      // nonlinear mechanics operator class
      oper.SetTime(sim_state.getTime());
      oper.SetDt(sim_state.getDeltaTime());
      oper.solVars.SetLastStep(last_step);

      // If our boundary condition changes for a step, we need to have an initial
      // corrector step that ensures the solver has an easier time solving the PDE.
      t1 = MPI_Wtime();
      if (BCManager::getInstance().getUpdateStep(ti)) {
         if (myid == 0) {
            std::cout << "Changing boundary conditions this step: " << ti << std::endl;
         }
         // Update the BC data
         oper.UpdateEssBdr();
         oper.UpdateVelocity();
         oper.SolveInit();
      }
      oper.UpdateVelocity();
      oper.Solve();

      // Our expected dt could have changed
      last_step = sim_state.isLastStep();

      t2 = MPI_Wtime();
      times.push_back(t2 - t1);

      sim_state.finishCycle();
      /*
      fix me
      SimulationState should work for some of this
      */
      oper.UpdateModel();
      post_process.Update(ti, sim_state.getTime());

      /*
      fix me
      All of the below needs to be updated to move into the internal PostProcessing variables
      */
     /*
      if (last_step || (ti % toml_opt.visualization.output_frequency) == 0) {
         const double t = sim_state.getTime();
         CALI_MARK_BEGIN("main_vis_update");
         if (toml_opt.visualization.visit || toml_opt.visualization.conduit || toml_opt.visualization.paraview || toml_opt.visualization.adios2) {
            // mesh and stress output. Consider moving this to a separate routine
            // We might not want to update the vonMises stuff
            oper.ProjectModelStress(stress);
            oper.ProjectVolume(volume);
            oper.ProjectVonMisesStress(vonMises, stress);
            oper.ProjectHydroStress(hydroStress, stress);

            if (mat_0.mech_type == MechType::EXACMECH) {
               if(toml_opt.post_processing.light_up.enabled) {
                  oper.ProjectCentroid(*elem_centroid);
                  oper.ProjectElasticStrains(*elastic_strain);
               }
               oper.ProjectDpEff(dpeff);
               oper.ProjectEffPlasticStrain(pleff);
               oper.ProjectOrientation(quats);
               oper.ProjectShearRate(gdots);
               oper.ProjectH(hardness);
            }
         }

         if (toml_opt.visualization.visit) {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            // Our visit data is now saved off
            visit_dc.Save();
         }
         if (toml_opt.visualization.paraview) {
            paraview_dc.SetCycle(ti);
            paraview_dc.SetTime(t);
            // Our paraview data is now saved off
            paraview_dc.Save();
         }
#ifdef MFEM_USE_CONDUIT
         if (toml_opt.visualization.conduit) {
            conduit_dc.SetCycle(ti);
            conduit_dc.SetTime(t);
            // Our conduit data is now saved off
            conduit_dc.Save();
         }
#endif
#ifdef MFEM_USE_ADIOS2
         if (toml_opt.visualization.adios2) {
            adios2_dc->SetCycle(ti);
            adios2_dc->SetTime(t);
            // Our adios2 data is now saved off
            adios2_dc->Save();
         }
#endif
         CALI_MARK_END("main_vis_update");
      } // end output scope
      */
   } // end loop over time steps

   // Now find out how long everything took to run roughly
   double end = MPI_Wtime();

   double sim_time = end - start;
   double avg_sim_time;

   MPI_Allreduce(&sim_time, &avg_sim_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   {
      std::ostringstream oss;

      oss << "./time/time_solve." << myid << ".txt";
      std::string file_name = oss.str();
      std::ofstream file;
      file.open(file_name, std::ios::out | std::ios::app);

      for (size_t i = 0; i < times.size(); i++) {
         std::ostringstream strs;
         strs << std::setprecision(8) << times[i] << "\n";
         std::string str = strs.str();
         file << str;
      }

      file.close();
   }


   if (myid == 0) {
      printf("The process took %lf seconds to run\n", (avg_sim_time / world_size));
   }

   if(toml_opt.post_processing.light_up.enabled) {
      delete elem_centroid;
      delete elastic_strain;
   }

// #ifdef MFEM_USE_ADIOS2
//    if (toml_opt.visualization.adios2) {
//       delete elem_attr;
//    }
//    delete adios2_dc;
// #endif

} // Used to ensure any mpi functions are scopped to only this section
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();

   return 0;
}

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // set the reference, stress free, configuration
   y = x;
}

void InitGridFunction(const Vector & /*x*/, Vector &y)
{
   y = 0.;
}

bool checkMaterialArgs(MechType mt, bool cp, int ngrains, int numProps,
                       int numStateVars)
{
   bool err = true;

   if (cp && (ngrains < 1)) {
      std::cerr << "\nSpecify number of grains for use with cp input arg." << '\n';
      err = false;
   }

   if (mt !=  MechType::NOTYPE && (numProps < 1)) {
      std::cerr << "\nMust specify material properties for mechanical model or cp calculation." << '\n';
      err = false;
   }

   // always input a state variables file with initial values for all models
   if (numStateVars < 1) {
      std::cerr << "\nMust specifiy state variables." << '\n';
   }

   return err;
}


