#ifndef MECHANICS_INTEG
#define MECHANICS_INTEG

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"

#include <utility>
#include <unordered_map>
#include <string>
// free function to compute the beginning step deformation gradient to store
// on a quadrature function
void computeDefGrad(mfem::QuadratureFunction *qf, mfem::ParFiniteElementSpace *fes,
                    mfem::Vector &x0);

class ExaModel
{
   public:
      int numProps;
      int numStateVars;
      bool init_step;

   protected:

      double dt, t;

      // --------------------------------------------------------------------------
      // The velocity method requires us to retain both the beggining and end time step
      // coordinates of the mesh. We need these to be able to compute the correct
      // incremental deformation gradient (using the beg. time step coords) and the
      // velocity gradient (uses the end time step coords).

      mfem::ParGridFunction* beg_coords;
      mfem::ParGridFunction* end_coords;

      // ---------------------------------------------------------------------------
      // STATE VARIABLES and PROPS common to all user defined models

      // quadrature vector function coefficient for the beginning step stress and
      // the end step (or incrementally upated) stress
      QuadratureVectorFunctionCoefficient stress0;
      QuadratureVectorFunctionCoefficient stress1;

      // quadrature vector function coefficient for the updated material tangent
      // stiffness matrix, which will need to be stored after an EvalP call and
      // used in a later AssembleH call
      QuadratureVectorFunctionCoefficient matGrad;

      // quadrature vector function coefficients for any history variables at the
      // beginning of the step and end (or incrementally updated) step.
      QuadratureVectorFunctionCoefficient matVars0;
      QuadratureVectorFunctionCoefficient matVars1;

      // add QuadratureVectorFunctionCoefficient to store von Mises
      // scalar stress measure
      QuadratureFunctionCoefficient vonMises;

      // add vector for material properties, which will be populated based on the
      // requirements of the user defined model. The properties are expected to be
      // the same at all quadrature points. That is, the material properties are
      // constant and not dependent on space
      mfem::Vector *matProps;
      bool PA;
      // Temporary fix just to make sure things work
      mfem::Vector matGradPA;

      std::unordered_map<std::string, std::pair<int, int> > qf_mapping;
   // ---------------------------------------------------------------------------

   public:
      ExaModel(mfem::QuadratureFunction *q_stress0, mfem::QuadratureFunction *q_stress1,
               mfem::QuadratureFunction *q_matGrad, mfem::QuadratureFunction *q_matVars0,
               mfem::QuadratureFunction *q_matVars1,
               mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
               mfem::Vector *props, int nProps, int nStateVars, bool _PA) :
         numProps(nProps), numStateVars(nStateVars),
         beg_coords(_beg_coords),
         end_coords(_end_coords),
         stress0(q_stress0),
         stress1(q_stress1),
         matGrad(q_matGrad),
         matVars0(q_matVars0),
         matVars1(q_matVars1),
         matProps(props),
         PA(_PA)
      {
         if (_PA) {
            int npts = q_matGrad->Size() / q_matGrad->GetVDim();
            matGradPA.SetSize(81 * npts, mfem::Device::GetMemoryType());
            matGradPA.UseDevice(true);
         }
      }

      virtual ~ExaModel() { }

      // This function is used in generating the B matrix commonly seen in the formation of
      // the material tangent stiffness matrix in mechanics [B^t][Cstiff][B]
      virtual void GenerateGradMatrix(const mfem::DenseMatrix& DS, mfem::DenseMatrix& B);

      // This function is used in generating the B matrix that's used in the formation
      // of the geometric stiffness contribution of the stiffness matrix seen in mechanics
      // as [B^t][sigma][B]
      virtual void GenerateGradGeomMatrix(const mfem::DenseMatrix& DS, mfem::DenseMatrix& Bgeom);

      // This function is responsible for running the entire model and will be the
      // external function that other classes/people can call.
      // It will consist of 3 stages/kernels:
      // 1.) A set-up kernel/stage that computes all of the needed values for the material model
      // 2.) A kernel that runs the material model (an t = 0 version of this will exist as well)
      // 3.) A post-processing kernel/stage that does everything after the kernel
      // e.g. All of the data is put back into the correct format here and re-arranged as needed
      // By having this function, we only need to ever right one integrator for everything.
      // It also allows us to run these models on the GPU even if the rest of the assembly operation
      // can't be there yet. If UMATs are used then these operations won't occur on the GPU.
      //
      // We'll need to supply the number of quadrature pts, number of elements, the dimension
      // of the space we're working with, the number of nodes for an element, the jacobian associated
      // with the transformation from the reference element to the local element, the quadrature integration wts,
      // and the velocity field at the elemental level (space_dim * nnodes * nelems).
      virtual void ModelSetup(const int nqpts, const int nelems, const int space_dim,
                              const int nnodes, const mfem::Vector &jacobian,
                              const mfem::Vector &loc_grad, const mfem::Vector &vel) = 0;

      // Fix me: Remove below
      //// routine to call constitutive update. Note that this routine takes
      //// the weight input argument to conform to the old AssembleH where the
      //// weight was used in the NeoHookean model. Consider refactoring this
      // virtual void EvalModel(const mfem::DenseMatrix &Jpt, const mfem::DenseMatrix &DS,
      // const double qptWeight, const double elemVol,
      // const int elemID, const int ipID, mfem::DenseMatrix &PMatO) = 0;

      // This function assembles the necessary stiffness matrix to be used in the
      // linearization of our nonlinear system of equations
      // virtual void AssembleH(const mfem::DenseMatrix &DS, const int elemID, const int ipID,
      // const double weight, mfem::DenseMatrix &A) = 0;
      //
      ////This function is needed in the UMAT child class to drive parts of the
      ////solution in the mechanics_operator file.
      ////It should just be set as a no-op
      ////in other children class if they aren't using it.
      ////For when the ParFinitieElementSpace is stored on the class...
      // virtual void calc_incr_end_def_grad(const mfem::Vector &x0) = 0;

      // routine to update the beginning step deformation gradient. This must
      // be written by a model class extension to update whatever else
      // may be required for that particular model
      virtual void UpdateModelVars() = 0;

      // set time on the base model class
      void SetModelTime(const double time) { t = time; }

      // set delta timestep on the base model class
      void SetModelDt(const double dtime) { dt = dtime; }

      // Get delta timestep on the base model class
      double GetModelDt() { return dt; }

      // return a pointer to beginning step stress. This is used for output visualization
      QuadratureVectorFunctionCoefficient *GetStress0() { return &stress0; }

      // return a pointer to beginning step stress. This is used for output visualization
      QuadratureVectorFunctionCoefficient *GetStress1() { return &stress1; }

      // function to set the internal von Mises QuadratureFuntion pointer to some
      // outside source
      void setVonMisesPtr(mfem::QuadratureFunction* vm_ptr) { vonMises = vm_ptr; }

      // return a pointer to von Mises stress quadrature vector function coefficient for visualization
      QuadratureFunctionCoefficient *GetVonMises() { return &vonMises; }

      // return a pointer to the matVars0 quadrature vector function coefficient
      QuadratureVectorFunctionCoefficient *GetMatVars0() { return &matVars0; }

      // fix me : remove
      // return a pointer to the end coordinates
      // this should probably only be used within the solver itself
      // if it's touched outside of that who knows whether or not the data
      // might be tampered with and thus we could end up with weird results
      // It's currently only being exposed due to the requirements UMATS place
      // on how things are solved outside of this class
      // fix_me
      // mfem::ParGridFunction *GetEndCoords(){return end_coords;}

      // return a pointer to the matProps vector
      mfem::Vector *GetMatProps() { return matProps; }

      // routine to get element stress at ip point. These are the six components of
      // the symmetric Cauchy stress where standard Voigt notation is being used
      void GetElementStress(const int elID, const int ipNum, bool beginStep,
                            double* stress, int numComps);

      // set the components of the member function end stress quadrature function with
      // the updated stress
      void SetElementStress(const int elID, const int ipNum, bool beginStep,
                            double* stress, int numComps);

      // routine to get the element statevars at ip point.
      void GetElementStateVars(const int elID, const int ipNum, bool beginStep,
                               double* stateVars, int numComps);

      // routine to set the element statevars at ip point
      void SetElementStateVars(const int elID, const int ipNum, bool beginStep,
                               double* stateVars, int numComps);

      // routine to get the material properties data from the decorated mfem vector
      void GetMatProps(double* props);

      // setter for the material properties data on the user defined model object
      void SetMatProps(double* props, int size);

      // routine to set the material Jacobian for this element and integration point.
      void SetElementMatGrad(const int elID, const int ipNum, double* grad, int numComps);

      // routine to get the material Jacobian for this element and integration point
      void GetElementMatGrad(const int elId, const int ipNum, double* grad, int numComps);

      int GetStressOffset();

      int GetMatGradOffset();

      int GetMatVarsOffset();

      // routine to update beginning step stress with end step values
      void UpdateStress();

      // routine to update beginning step state variables with end step values
      void UpdateStateVars();

      // Update the End Coordinates using a simple Forward Euler Integration scheme
      // The beggining time step coordinates should be updated outside of the model routines
      void UpdateEndCoords(const mfem::Vector& vel);

      // This method performs a fast approximate polar decomposition for 3x3 matrices
      // The deformation gradient or 3x3 matrix of interest to be decomposed is passed
      // in as the initial R matrix. The error on the solution can be set by the user.
      void CalcPolarDecompDefGrad(mfem::DenseMatrix& R, mfem::DenseMatrix& U,
                                  mfem::DenseMatrix& V, double err = 1e-12);

      // Various Strain measures we can use
      // Same as above should these be a protected function?

      // Lagrangian is simply E = 1/2(F^tF - I)
      void CalcLagrangianStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F);

      // Eulerian is simply e = 1/2(I - F^(-t)F^(-1))
      void CalcEulerianStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F);

      // Biot strain is simply B = U - I
      void CalcBiotStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F);

      // Log strain is equal to e = 1/2 * ln(C) or for UMATs its e = 1/2 * ln(B)
      void CalcLogStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F);

      // Some useful rotation functions that we can use
      // Do we want to have these exposed publically or should they
      // be protected?
      // Also, do we want to think about moving these type of orientation
      // conversions to their own class?
      void Quat2RMat(const mfem::Vector& quat, mfem::DenseMatrix& rmat);
      void RMat2Quat(const mfem::DenseMatrix& rmat, mfem::Vector& quat);

      // Computes the von Mises stress from the Cauchy stress
      void ComputeVonMises(const int elemID, const int ipID);

      const double *GetMTanData(){ return matGradPA.Read(); }

      void TransformMatGradTo4D();

      /// This method sets the end time step stress to the beginning step
      /// and then returns the internal data pointer of the end time step
      /// array.
      double* StressSetup();

      /// This methods set the end time step state variable array to the
      /// beginning time step values and then returns the internal data pointer
      /// of the end time step array.
      double* StateVarsSetup();

      /// Returns an unordered map that maps a given variable name to its
      /// its location and length within the state variable variable.
      const std::unordered_map<std::string, std::pair<int, int> > *GetQFMapping()
      {
         return &qf_mapping;
      }
};

// End the need for the ecmech namespace
class ExaNLFIntegrator : public mfem::NonlinearFormIntegrator
{
   private:
      ExaModel *model;
      // Will take a look and see what I need and don't need for this.
      mfem::Vector dmat;
      mfem::Vector grad;
      mfem::Vector *tan_mat; // Not owned
      mfem::Vector pa_dmat;
      mfem::Vector jacobian;
      const mfem::DofToQuad *maps; // Not owned
      const mfem::GeometricFactors *geom; // Not owned
      int space_dims, nelems, nqpts, nnodes;

   public:
      ExaNLFIntegrator(ExaModel *m) : model(m) { }

      virtual ~ExaNLFIntegrator() { }

      virtual double GetElementEnergy(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Ttr,
                                      const mfem::Vector &elfun);

      using mfem::NonlinearFormIntegrator::AssembleElementVector;
      virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                         mfem::ElementTransformation &Ttr,
                                         const mfem::Vector &elfun, mfem::Vector &elvect);

      virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Ttr,
                                       const mfem::Vector & /*elfun*/, mfem::DenseMatrix &elmat);

      // We should only really require the Assemble Partial Assembly Gradient
      // The diagonal terms will just build upon this.
      virtual void AssemblePAGrad(const mfem::FiniteElementSpace &fes) override;
      virtual void AddMultPAGrad(const mfem::Vector &x, mfem::Vector &y) override;

      using mfem::NonlinearFormIntegrator::AssemblePA;
      virtual void AssemblePA(const mfem::FiniteElementSpace &fes) override;
      virtual void AddMultPA(const mfem::Vector & /*x*/, mfem::Vector &y) const override;
};

// }

#endif
