#pragma once

#include "mfem.hpp"
#include "ECMech_const.h"
#include "ECMech_matModelBase.h"
#include "mechanics_model.hpp"

/// Base class for all of our ExaCMechModels.
class ExaCMechModel : public ExaModel
{
   protected:

      // Current temperature in Kelvin degrees
      double temp_k;

      // A pointer to our actual material model class that ExaCMech uses.
      // The childern classes to this class will have also have another variable
      // that actually contains the real material model that is then dynamically casted
      // to this base class during the instantiation of the class.
      ecmech::matModelBase* mat_model_base;

      // Our accelartion that we are making use of.
      ecmech::ExecutionStrategy accel;

      // Temporary variables that we'll be making use of when running our
      // models.
      mfem::Vector *vel_grad_array;
      mfem::Vector *eng_int_array;
      mfem::Vector *w_vec_array;
      mfem::Vector *vol_ratio_array;
      mfem::Vector *stress_svec_p_array;
      mfem::Vector *d_svec_p_array;
      mfem::Vector *tempk_array;
      mfem::Vector *sdd_array;
      mfem::Vector *eff_def_rate;

      std::map<std::string, size_t> index_map;

   public:
      ExaCMechModel(mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
                    mfem::QuadratureFunction *_q_matGrad, mfem::QuadratureFunction *_q_matVars0,
                    mfem::QuadratureFunction *_q_matVars1,
                    mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                    mfem::Vector *_props, int _nProps, int _nStateVars, double _temp_k,
                    ecmech::ExecutionStrategy _accel, AssemblyType _assembly, std::string mat_model_name);

      ~ExaCMechModel()
      {
         delete vel_grad_array;
         delete eng_int_array;
         delete w_vec_array;
         delete vol_ratio_array;
         delete stress_svec_p_array;
         delete d_svec_p_array;
         delete tempk_array;
         delete sdd_array;
         delete eff_def_rate;
         delete mat_model_base;
      }

      void setup_data_structures();
      void setup_model(std::string mat_model_name);
      void init_state_vars(std::vector<double> hist_init);

      /** This model takes in the velocity, det(jacobian), and local_grad/jacobian.
       *  It then computes velocity gradient symm and skw tensors and passes
       *  that to our material model in order to get out our Cauchy stress and
       * the material tangent matrix (d \sigma / d Vgrad_{sym}). It also
       * updates all of the state variables that live at the quadrature pts.
       */
      void ModelSetup(const int nqpts, const int nelems, const int /*space_dim*/,
                      const int nnodes, const mfem::Vector &jacobian,
                      const mfem::Vector &loc_grad, const mfem::Vector &vel) override;

      /// If we needed to do anything to our state variables once things are solved
      /// for we do that here.
      virtual void UpdateModelVars() override {}
      void calcDpMat(mfem::QuadratureFunction &/* DpMat */) const override {}
};

