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
#include "system_driver.hpp"
#include "boundary_conditions/BCData.hpp"
#include "boundary_conditions/BCManager.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "mfem_expt/partial_qfunc.hpp"
#include "options/option_parser_v2.hpp"
#include "postprocessing/postprocessing_driver.hpp"
#include "sim_state/simulation_state.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include <string>
#include <sstream>

using namespace mfem;

// This initializes some grid function
void InitGridFunction(const Vector & /*x*/, Vector &y);

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

   const int dim = pmesh->Dimension();

   // Define the finite element spaces for displacement field
   HYPRE_Int glob_size = sim_state.GetMeshParFiniteElementSpace()->GlobalTrueVSize();

   pmesh->PrintInfo();

   // Print the mesh statistics
   if (myid == 0) {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << glob_size << "\n";
      std::cout << "***********************************************************\n";
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
   SystemDriver oper(sim_state);

   // get the essential true dof list. This may not be used.
   const Array<int> ess_tdof_list = oper.GetEssTDofList();

   PostProcessingDriver post_process(sim_state, toml_opt);

   int ti = 0;
   auto v_sol = sim_state.getPrimalField();
   while (!sim_state.isFinished()) {
      ti++;
      if (myid == 0) {
         std::cout << "Simulation cycle: " << ti << std::endl;
         sim_state.printTimeStats();
      }
      // Get out our current delta time step
      // set time on the simulation variables and the model through the
      // nonlinear mechanics operator class
      const double sim_time = sim_state.getTime();

      // If our boundary condition changes for a step, we need to have an initial
      // corrector step that ensures the solver has an easier time solving the PDE.
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

      sim_state.finishCycle();
      /*
      fix me
      SimulationState should work for some of this
      */
      oper.UpdateModel();
      post_process.Update(ti, sim_time);
   } // end loop over time steps

   // Now find out how long everything took to run roughly
   double end = MPI_Wtime();

   double sim_time = end - start;
   double avg_sim_time;

   MPI_Allreduce(&sim_time, &avg_sim_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   if (myid == 0) {
      printf("The process took %lf seconds to run\n", (avg_sim_time / world_size));
   }

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


