#ifndef MECHANICS_UMAT
#define MECHANICS_UMAT

#include "models/mechanics_model.hpp"
#include "utilities/dynamic_umat_loader.hpp"
#include "userumat.h"

#include "mfem.hpp"

/**
 * @brief Enhanced Abaqus UMAT model with dynamic library loading support
 * 
 * This enhanced version supports loading UMAT implementations from shared libraries
 * at runtime, eliminating the need to recompile ExaConstit for new UMATs.
 * 
 * Key features:
 * - Dynamic loading of UMAT shared libraries
 * - Support for multiple UMATs in different regions
 * - Configurable loading strategies (persistent, on-demand, etc.)
 * - Thread-safe library management
 * - Automatic cleanup and error handling
 */
class AbaqusUmatModel : public ExaModel
{
   protected:

      // add member variables.
      double elemLength;

      // RETAINED: The initial local shape function gradients.
      // These are working space specific to UMAT models, so they remain as member variables
      mfem::QuadratureFunction loc0_sf_grad;

      // RETAINED: The incremental deformation gradients.
      // These are working space specific to UMAT models, so they remain as member variables
      mfem::QuadratureFunction incr_def_grad;

      // RETAINED: The end step deformation gradients.  
      // These are working space specific to UMAT models, so they remain as member variables
      mfem::QuadratureFunction end_def_grad;

      std::string umat_library_path_;           ///< Path to UMAT shared library
      UmatFunction umat_function_;              ///< Pointer to loaded UMAT function
      DynamicUmatLoader::LoadStrategy load_strategy_; ///< Loading strategy
      bool use_dynamic_loading_;                ///< Flag to enable/disable dynamic loading

   public:
      /**
       * @brief Constructor with dynamic UMAT loading support
       * 
       * @param region Region identifier
       * @param nStateVars Number of state variables
       * @param sim_state Reference to simulation state
       * @param umat_library_path Path to UMAT shared library (empty for static linking)
       * @param load_strategy Strategy for loading/unloading the library
       */
      AbaqusUmatModel(const int region, int nStateVars, 
                      SimulationState& sim_state,
                      const std::string& umat_library_path = "",
                      const DynamicUmatLoader::LoadStrategy& load_strategy = DynamicUmatLoader::LoadStrategy::PERSISTENT);

      virtual ~AbaqusUmatModel();

      // NEW: Helper method to get defGrad0 from SimulationState
      // This replaces the direct member variable access and enables dynamic access
      // to the correct region-specific deformation gradient data
      std::shared_ptr<mfem::expt::PartialQuadratureFunction> GetDefGrad0();

      // UNCHANGED: These methods remain the same since they work with internal data or don't access QFs directly
      virtual void UpdateModelVars() override;

      virtual void ModelSetup(const int nqpts, const int nelems, const int space_dim,
                              const int /*nnodes*/, const mfem::Vector &jacobian,
                              const mfem::Vector & /*loc_grad*/, const mfem::Vector &vel) override;

      /**
       * @brief Set the UMAT library path and loading strategy
       * 
       * @param library_path Path to the shared library
       * @param strategy Loading strategy to use
       * @return true if library can be loaded, false otherwise
       */
      bool SetUmatLibrary(const std::string& library_path, 
         DynamicUmatLoader::LoadStrategy strategy = DynamicUmatLoader::LoadStrategy::PERSISTENT);

      /**
      * @brief Get the current UMAT library path
      */
      const std::string& GetUmatLibraryPath() const { return umat_library_path_; }

      /**
       * @brief Check if using dynamic loading
      */
      bool UsingDynamicLoading() const { return use_dynamic_loading_; }

      /**
      * @brief Force reload of UMAT library (useful for development)
      */
      bool ReloadUmatLibrary();

      protected:
      /**
       * @brief Load the UMAT library if using dynamic loading
       */
      bool LoadUmatLibrary();
  
      /**
       * @brief Unload the UMAT library if using dynamic loading
       */
      void UnloadUmatLibrary();

protected:
      /**
       * @brief Call the UMAT function (either static or dynamic)
       */
      void CallUmat(double *stress, double *statev, double *ddsdde,
                    double *sse, double *spd, double *scd, double *rpl,
                    double *ddsdt, double *drplde, double *drpldt,
                    double *stran, double *dstran, double *time,
                    double *deltaTime, double *tempk, double *dtemp, double *predef,
                    double *dpred, double *cmname, int *ndi, int *nshr, int *ntens,
                    int *nstatv, double *props, int *nprops, double *coords,
                    double *drot, double *pnewdt, double *celent,
                    double *dfgrd0, double *dfgrd1, int *noel, int *npt,
                    int *layer, int *kspt, int *kstep, int *kinc);
  
      // Helper methods
      void init_loc_sf_grads(const std::shared_ptr<mfem::ParFiniteElementSpace> fes);
      void init_incr_end_def_grad();
      void calc_incr_end_def_grad(const mfem::ParGridFunction& x0);

      // Calculates the incremental versions of the strain measures that we're given
      // above
      void CalcLogStrainIncrement(mfem::DenseMatrix &dE, const mfem::DenseMatrix &Jpt);
      void CalcEulerianStrainIncr(mfem::DenseMatrix& dE, const mfem::DenseMatrix &Jpt);
      void CalcLagrangianStrainIncr(mfem::DenseMatrix& dE, const mfem::DenseMatrix &Jpt);
      void CalcElemLength(const double elemVol);
};

#endif