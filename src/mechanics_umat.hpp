#ifndef MECHANICS_UMAT
#define MECHANICS_UMAT

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "userumat.h"


// Abaqus Umat class.
class AbaqusUmatModel : public ExaModel
{
   protected:

      // add member variables.
      double elemLength;

      // The initial local shape function gradients.
      mfem::QuadratureFunction loc0_sf_grad;

      // The incremental deformation gradients.
      mfem::QuadratureFunction incr_def_grad;

      // The end step deformation gradients.
      mfem::QuadratureFunction end_def_grad;
      mfem::ParFiniteElementSpace* loc_fes;

      // add QuadratureVectorFunctionCoefficient to store the beginning step
      // Note you can compute the end step def grad from the incremental def
      // grad (from the solution: Jpt) and the beginning step def grad
      QuadratureVectorFunctionCoefficient defGrad0;

      // pointer to umat function
      // we really don't use this in the code
      void (*umatp)(double[6], double[], double[36],
                    double*, double*, double*, double*,
                    double[6], double[6], double*,
                    double[6], double[6], double[2],
                    double*, double*, double*, double*,
                    double*, double*, int*, int*, int*,
                    int *, double[], int*, double[3],
                    double[9], double*, double*,
                    double[9], double[9], int*, int*,
                    int*, int*, int*, int*);

      // Calculates the incremental versions of the strain measures that we're given
      // above
      void CalcLogStrainIncrement(mfem::DenseMatrix &dE, const mfem::DenseMatrix &Jpt);
      void CalcEulerianStrainIncr(mfem::DenseMatrix& dE, const mfem::DenseMatrix &Jpt);
      void CalcLagrangianStrainIncr(mfem::DenseMatrix& dE, const mfem::DenseMatrix &Jpt);

      // calculates the element length
      void CalcElemLength(const double elemVol);

      void init_loc_sf_grads(mfem::ParFiniteElementSpace *fes);
      void init_incr_end_def_grad();

      // For when the ParFinitieElementSpace is stored on the class...
      virtual void calc_incr_end_def_grad(const mfem::Vector &x0);

   public:
      AbaqusUmatModel(mfem::QuadratureFunction *_q_stress0, mfem::QuadratureFunction *_q_stress1,
                      mfem::QuadratureFunction *_q_matGrad, mfem::QuadratureFunction *_q_matVars0,
                      mfem::QuadratureFunction *_q_matVars1, mfem::QuadratureFunction *_q_defGrad0,
                      mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                      mfem::Vector *_props, int _nProps,
                      int _nStateVars, mfem::ParFiniteElementSpace* fes) : ExaModel(_q_stress0,
                                                                                    _q_stress1, _q_matGrad, _q_matVars0,
                                                                                    _q_matVars1,
                                                                                    _beg_coords, _end_coords,
                                                                                    _props, _nProps, _nStateVars), loc_fes(fes),
         defGrad0(_q_defGrad0)
      {
         init_loc_sf_grads(fes);
         init_incr_end_def_grad();
      }

      virtual ~AbaqusUmatModel() { }

      virtual void UpdateModelVars();

      virtual void ModelSetup(const int nqpts, const int nelems, const int space_dim,
                              const int /*nnodes*/, const mfem::Vector &jacobian,
                              const mfem::Vector & /*loc_grad*/, const mfem::Vector &vel);
};

#endif
