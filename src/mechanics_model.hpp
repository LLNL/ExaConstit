#ifndef MECHANICS_MODEL
#define MECHANICS_MODEL

#include "option_types.hpp"

#include "mfem.hpp"

#include <utility>
#include <unordered_map>
#include <string>

/// free function to compute the beginning step deformation gradient to store
/// on a quadrature function
void computeDefGrad(mfem::QuadratureFunction *qf, mfem::ParFiniteElementSpace *fes,
                    mfem::Vector &x0);

class ExaModel
{
   public:
      int numProps;
      int numStateVars;
      bool init_step = false;

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

      // The beginning step stress and the end step (or incrementally upated) stress
      mfem::QuadratureFunction *stress0;
      mfem::QuadratureFunction *stress1;

      // The updated material tangent stiffness matrix, which will need to be
      // stored after an EvalP call and used in a later AssembleH call
      mfem::QuadratureFunction *matGrad;

      // quadrature vector function coefficients for any history variables at the
      // beginning of the step and end (or incrementally updated) step.
      mfem::QuadratureFunction *matVars0;
      mfem::QuadratureFunction *matVars1;

      // Stores the von Mises / hydrostatic scalar stress measure
      // we use this array to compute both the hydro and von Mises stress quantities
      mfem::QuadratureFunction *vonMises;

      // add vector for material properties, which will be populated based on the
      // requirements of the user defined model. The properties are expected to be
      // the same at all quadrature points. That is, the material properties are
      // constant and not dependent on space
      mfem::Vector *matProps;
      Assembly assembly;
      // Temporary fix just to make sure things work
      mfem::Vector matGradPA;

      std::unordered_map<std::string, std::pair<int, int> > qf_mapping;
   // ---------------------------------------------------------------------------

   public:
      ExaModel(mfem::QuadratureFunction *q_stress0, mfem::QuadratureFunction *q_stress1,
               mfem::QuadratureFunction *q_matGrad, mfem::QuadratureFunction *q_matVars0,
               mfem::QuadratureFunction *q_matVars1,
               mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
               mfem::Vector *props, int nProps, int nStateVars, Assembly _assembly);

      virtual ~ExaModel() { }

      /// This function is used in generating the B matrix commonly seen in the formation of
      /// the material tangent stiffness matrix in mechanics [B^t][Cstiff][B]
      virtual void GenerateGradMatrix(const mfem::DenseMatrix& DS, mfem::DenseMatrix& B);

      /// This function is used in generating the Bbar matrix seen in the formation of
      /// the material tangent stiffness matrix in mechanics [B^t][Cstiff][B] for
      /// incompressible materials
      virtual void GenerateGradBarMatrix(const mfem::DenseMatrix& DS, const mfem::DenseMatrix& eDS, mfem::DenseMatrix& B);

      /// This function is used in generating the B matrix that's used in the formation
      /// of the geometric stiffness contribution of the stiffness matrix seen in mechanics
      /// as [B^t][sigma][B]
      virtual void GenerateGradGeomMatrix(const mfem::DenseMatrix& DS, mfem::DenseMatrix& Bgeom);

      /** @brief This function is responsible for running the entire model and will be the
      *   external function that other classes/people can call.
      *
      *   It will consist of 3 stages/kernels:
      *   1.) A set-up kernel/stage that computes all of the needed values for the material model
      *   2.) A kernel that runs the material model (an t = 0 version of this will exist as well)
      *   3.) A post-processing kernel/stage that does everything after the kernel
      *   e.g. All of the data is put back into the correct format here and re-arranged as needed
      *   By having this function, we only need to ever right one integrator for everything.
      *   It also allows us to run these models on the GPU even if the rest of the assembly operation
      *   can't be there yet. If UMATs are used then these operations won't occur on the GPU.
      *
      *   We'll need to supply the number of quadrature pts, number of elements, the dimension
      *   of the space we're working with, the number of nodes for an element, the jacobian associated
      *   with the transformation from the reference element to the local element, the quadrature integration wts,
      *   and the velocity field at the elemental level (space_dim * nnodes * nelems).
      */
      virtual void ModelSetup(const int nqpts, const int nelems, const int space_dim,
                              const int nnodes, const mfem::Vector &jacobian,
                              const mfem::Vector &loc_grad, const mfem::Vector &vel) = 0;

      /// routine to update the beginning step deformation gradient. This must
      /// be written by a model class extension to update whatever else
      /// may be required for that particular model
      virtual void UpdateModelVars() = 0;

      /// set time on the base model class
      void SetModelTime(const double time) { t = time; }

      /// set delta timestep on the base model class
      void SetModelDt(const double dtime) { dt = dtime; }

      /// Get delta timestep on the base model class
      double GetModelDt() { return dt; }

      /// return a pointer to beginning step stress. This is used for output visualization
      mfem::QuadratureFunction *GetStress0() { return stress0; }

      /// return a pointer to beginning step stress. This is used for output visualization
      mfem::QuadratureFunction *GetStress1() { return stress1; }

      /// function to set the internal von Mises QuadratureFuntion pointer to some
      /// outside source
      void setVonMisesPtr(mfem::QuadratureFunction* vm_ptr) { vonMises = vm_ptr; }

      /// return a pointer to von Mises stress quadrature function for visualization
      mfem::QuadratureFunction *GetVonMises() { return vonMises; }

      /// return a pointer to the matVars0 quadrature function
      mfem::QuadratureFunction *GetMatVars0() { return matVars0; }

      /// return a pointer to the matGrad quadrature function
      mfem::QuadratureFunction *GetMatGrad() { return matGrad; }

      /// return a pointer to the matProps vector
      mfem::Vector *GetMatProps() { return matProps; }

      /// routine to get element stress at ip point. These are the six components of
      /// the symmetric Cauchy stress where standard Voigt notation is being used
      void GetElementStress(const int elID, const int ipNum, bool beginStep,
                            double* stress, int numComps);

      /// set the components of the member function end stress quadrature function with
      /// the updated stress
      void SetElementStress(const int elID, const int ipNum, bool beginStep,
                            double* stress, int numComps);

      /// routine to get the element statevars at ip point.
      void GetElementStateVars(const int elID, const int ipNum, bool beginStep,
                               double* stateVars, int numComps);

      /// routine to set the element statevars at ip point
      void SetElementStateVars(const int elID, const int ipNum, bool beginStep,
                               double* stateVars, int numComps);

      /// routine to get the material properties data from the decorated mfem vector
      void GetMatProps(double* props);

      /// setter for the material properties data on the user defined model object
      void SetMatProps(double* props, int size);

      /// routine to set the material Jacobian for this element and integration point.
      void SetElementMatGrad(const int elID, const int ipNum, double* grad, int numComps);

      /// routine to get the material Jacobian for this element and integration point
      void GetElementMatGrad(const int elId, const int ipNum, double* grad, int numComps);

      /// routine to update beginning step stress with end step values
      void UpdateStress();

      /// routine to update beginning step state variables with end step values
      void UpdateStateVars();

      /// Update the End Coordinates using a simple Forward Euler Integration scheme
      /// The beggining time step coordinates should be updated outside of the model routines
      void UpdateEndCoords(const mfem::Vector& vel);

      /// This method performs a fast approximate polar decomposition for 3x3 matrices
      /// The deformation gradient or 3x3 matrix of interest to be decomposed is passed
      /// in as the initial R matrix. The error on the solution can be set by the user.
      void CalcPolarDecompDefGrad(mfem::DenseMatrix& R, mfem::DenseMatrix& U,
                                  mfem::DenseMatrix& V, double err = 1e-12);

      /// Lagrangian is simply E = 1/2(F^tF - I)
      void CalcLagrangianStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F);

      /// Eulerian is simply e = 1/2(I - F^(-t)F^(-1))
      void CalcEulerianStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F);

      /// Biot strain is simply B = U - I
      void CalcBiotStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F);

      /// Log strain is equal to e = 1/2 * ln(C) or for UMATs its e = 1/2 * ln(B)
      void CalcLogStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F);

      /// Converts a unit quaternion over to rotation matrix
      void Quat2RMat(const mfem::Vector& quat, mfem::DenseMatrix& rmat);

      /// Converts a rotation matrix over to a unit quaternion
      void RMat2Quat(const mfem::DenseMatrix& rmat, mfem::Vector& quat);

      /// Returns a pointer to our 4D material tangent stiffness tensor
      const double *GetMTanData(){ return matGradPA.Read(); }

      /// Converts a normal 2D stiffness tensor into it's equivalent 4D stiffness
      /// tensor
      void TransformMatGradTo4D();

      /// This method sets the end time step stress to the beginning step
      /// and then returns the internal data pointer of the end time step
      /// array.
      double* StressSetup();

      /// This methods set the end time step state variable array to the
      /// beginning time step values and then returns the internal data pointer
      /// of the end time step array.
      double* StateVarsSetup();

      /// This function calculates the plastic strain rate tensor (D^p) with
      /// a DpMat that's a full 3x3 matrix rather than a 6-dim vector just so
      /// we can re-use storage from the deformation gradient tensor.
      virtual void calcDpMat(mfem::QuadratureFunction &DpMat) const = 0;

      /// Returns an unordered map that maps a given variable name to its
      /// its location and length within the state variable variable.
      const std::unordered_map<std::string, std::pair<int, int> > *GetQFMapping()
      {
         return &qf_mapping;
      }
};

#endif