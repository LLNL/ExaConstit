
#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_umat.hpp"
#include <string>
#include <sstream>
#include "RAJA/RAJA.hpp"

#include <gtest/gtest.h>

using namespace std;
using namespace mfem;

static int outputLevel = 0;

class test_model : public ExaModel
{
   public:

      test_model(mfem::QuadratureFunction *q_stress0, mfem::QuadratureFunction *q_stress1,
                 mfem::QuadratureFunction *q_matGrad, mfem::QuadratureFunction *q_matVars0,
                 mfem::QuadratureFunction *q_matVars1,
                 mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                 mfem::Vector *props, int nProps, int nStateVars, bool _PA) :
         ExaModel(q_stress0,
                  q_stress1, q_matGrad, q_matVars0,
                  q_matVars1,
                  beg_coords, end_coords,
                  props, nProps, nStateVars, _PA)
      {
         _beg_coords = _beg_coords;
         _end_coords = _end_coords;
      }

      virtual ~test_model() {}

      void UpdateModelVars() {}

      void ModelSetup(const int, const int, const int,
                      const int, const mfem::Vector &,
                      const mfem::Vector &, const mfem::Vector &) {}
};

// This function will either set our CMat array to all ones or something resembling a cubic symmetry like system.
template<bool cmat_ones>
void setCMat(QuadratureFunction &cmat_data);

// This function compares the difference in the formation of the GetGradient operator and then multiplying it
// by the necessary vector, and the matrix-free partial assembly formulation which avoids forming the matrix.
// It's been tested on higher order elements and multiple elements. The difference in these two methods
// should be 0.0.
template<bool cmat_ones>
double ExaNLFIntegratorPATest()
{
   int dim = 3;
   Mesh *mesh;
   // Making this mesh and test real simple with 8 elements and then a cubic element
   mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 0, 1.0, 1.0, 1.0, false);
   int order = 3;
   H1_FECollection fec(order, dim);

   ParMesh *pmesh = NULL;
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   ParFiniteElementSpace fes(pmesh, &fec, dim);

   delete mesh;

   // All of these Quadrature function variables are needed to instantiate our material model
   // We can just ignore this marked section
   /////////////////////////////////////////////////////////////////////////////////////////
   // Define a quadrature space and material history variable QuadratureFunction.
   int intOrder = 2 * order + 1;
   QuadratureSpace qspace(pmesh, intOrder); // 3rd order polynomial for 2x2x2 quadrature
   // for first order finite elements.
   QuadratureFunction q_matVars0(&qspace, 1);
   QuadratureFunction q_matVars1(&qspace, 1);
   QuadratureFunction q_sigma0(&qspace, 1);
   QuadratureFunction q_sigma1(&qspace, 1);
   // We'll modify this before doing the partial assembly
   // This is our stiffness matrix and is a 6x6 due to major and minor symmetry
   // of the 4th order tensor which has dimensions 3x3x3x3.
   QuadratureFunction q_matGrad(&qspace, 36);
   QuadratureFunction q_kinVars0(&qspace, 9);
   QuadratureFunction q_vonMises(&qspace, 1);
   ParGridFunction beg_crds(&fes);
   ParGridFunction end_crds(&fes);
   // We'll want to update this later in case we do anything more complicated.
   Vector matProps(1);

   end_crds = 1.0;

   ExaModel *model;
   // This doesn't really matter and is just needed for the integrator class.
   model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1, &q_kinVars0,
                               &beg_crds, &end_crds, &matProps, 1, 1, &fes, true);
   // Model time needs to be set.
   model->SetModelDt(1.0);
   /////////////////////////////////////////////////////////////////////////////
   ExaNLFIntegrator* nlf_int;

   nlf_int = new ExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model));

   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *Ttr;

   // So, we're going to originally support non tensor-product type elements originally.
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *elem_restrict_lex;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   // Set our field variable to a linear spacing so 1 ... ndofs in field
   Vector xtrue(end_crds.Size());
   for (int i = 0; i < xtrue.Size(); i++) {
      xtrue(i) = i + 1;
   }

   // For multiple elements xtrue and local_x are differently sized
   Vector local_x(elem_restrict_lex->Height());
   // All of our local global solution variables
   Vector y_fa(end_crds.Size());
   Vector local_y_fa(elem_restrict_lex->Height());
   Vector local_y_pa(elem_restrict_lex->Height());
   Vector y_pa(end_crds.Size());
   // Initializing them all to 1.
   y_fa = 0.0;
   y_pa = 0.0;
   local_y_pa = 0.0;
   local_y_fa = 0.0;
   // Get our local x values (element values) from the global vector
   elem_restrict_lex->Mult(xtrue, local_x);
   // Variables used to kinda mimic what the NonlinearForm::GetGradient does.
   int ndofs = el.GetDof() * el.GetDim();
   Vector elfun(ndofs), elresults(ndofs);
   DenseMatrix elmat;

   // Set our CMat array for the non-PA case
   q_matGrad = 0.0;
   setCMat<cmat_ones>(q_matGrad);
   elfun.HostReadWrite();
   elresults.HostReadWrite();
   local_x.HostReadWrite();
   local_y_fa.HostReadWrite();
   for (int i = 0; i < fes.GetNE(); i++) {
      Ttr = fes.GetElementTransformation(i);
      for (int j = 0; j < ndofs; j++) {
         elfun(j) = local_x((i * ndofs) + j);
      }

      nlf_int->AssembleElementGrad(el, *Ttr, elfun, elmat);
      // Getting out the local action of our gradient operator on
      // the local x values and then saving the results off to the
      // global variable.
      elresults = 0.0;
      elmat.AddMult(elfun, elresults);
      for (int j = 0; j < ndofs; j++) {
         local_y_fa((i * ndofs) + j) = elresults(j);
      }
   }

   // This takes our 2d cmat and transforms it into the 4d version
   model->TransformMatGradTo4D();
   // Perform the setup and action operation of our PA operation
   nlf_int->AssemblePAGrad(fes);
   nlf_int->AddMultPAGrad(local_x, local_y_pa);

   // Take all of our multiple elements and go back to the L vector.
   elem_restrict_lex->MultTranspose(local_y_fa, y_fa);
   elem_restrict_lex->MultTranspose(local_y_pa, y_pa);
   // Find out how different our solutions were from one another.
   double mag = y_fa.Norml2();
   std::cout << "y_fa mag: " << mag << std::endl;
   y_fa -= y_pa;
   double difference = y_fa.Norml2();
   // Free up memory now.
   delete nlf_int;
   delete model;
   delete pmesh;

   return difference / mag;
}

// This function compares the difference in the formation of the Mult operator and then multiplying it
// by the necessary vector, and the matrix-free partial assembly formulation.
// It's been tested on higher order elements and multiple elements. The difference in these two methods
// should be 0.0.
double ExaNLFIntegratorPAVecTest()
{
   int dim = 3;
   Mesh *mesh;
   // Making this mesh and test real simple with 8 elements and then a cubic element
   mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 0, 1.0, 1.0, 1.0, false);
   int order = 6;
   H1_FECollection fec(order, dim);

   ParMesh *pmesh = NULL;
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   ParFiniteElementSpace fes(pmesh, &fec, dim);

   delete mesh;

   // All of these Quadrature function variables are needed to instantiate our material model
   // We can just ignore this marked section
   /////////////////////////////////////////////////////////////////////////////////////////
   // Define a quadrature space and material history variable QuadratureFunction.
   int intOrder = 2 * order + 1;
   QuadratureSpace qspace(pmesh, intOrder); // 3rd order polynomial for 2x2x2 quadrature
   // for first order finite elements.
   QuadratureFunction q_matVars0(&qspace, 1);
   QuadratureFunction q_matVars1(&qspace, 1);
   QuadratureFunction q_sigma0(&qspace, 6);
   QuadratureFunction q_sigma1(&qspace, 6);
   q_sigma1 = 1.0;
   q_sigma0 = 1.0;
   // We'll modify this before doing the partial assembly
   // This is our stiffness matrix and is a 6x6 due to major and minor symmetry
   // of the 4th order tensor which has dimensions 3x3x3x3.
   QuadratureFunction q_matGrad(&qspace, 36);
   QuadratureFunction q_kinVars0(&qspace, 9);
   QuadratureFunction q_vonMises(&qspace, 1);
   ParGridFunction beg_crds(&fes);
   ParGridFunction end_crds(&fes);
   // We'll want to update this later in case we do anything more complicated.
   Vector matProps(1);

   end_crds = 1.0;

   ExaModel *model;
   // This doesn't really matter and is just needed for the integrator class.
   model = new test_model(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                          &beg_crds, &end_crds, &matProps, 1, 1, true);
   // Model time needs to be set.
   model->SetModelDt(1.0);
   /////////////////////////////////////////////////////////////////////////////
   ExaNLFIntegrator* nlf_int;

   nlf_int = new ExaNLFIntegrator(dynamic_cast<test_model*>(model));

   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *Ttr;

   // So, we're going to originally support non tensor-product type elements originally.
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *elem_restrict_lex;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   // Set our field variable to a linear spacing so 1 ... ndofs in field
   Vector xtrue(end_crds.Size());
   for (int i = 0; i < xtrue.Size(); i++) {
      xtrue(i) = i + 1;
   }

   // For multiple elements xtrue and local_x are differently sized
   Vector local_x(elem_restrict_lex->Height());
   // All of our local global solution variables
   Vector y_fa(end_crds.Size());
   Vector local_y_fa(elem_restrict_lex->Height());
   Vector local_y_pa(elem_restrict_lex->Height());
   Vector y_pa(end_crds.Size());
   // Initializing them all to 1.
   y_fa = 0.0;
   y_pa = 0.0;
   local_y_pa = 0.0;
   local_y_fa = 0.0;
   // Get our local x values (element values) from the global vector
   elem_restrict_lex->Mult(xtrue, local_x);
   // Variables used to kinda mimic what the NonlinearForm::GetGradient does.
   int ndofs = el.GetDof() * el.GetDim();
   Vector elfun(ndofs), elresults(ndofs);
   DenseMatrix elmat;
   elfun.HostReadWrite();
   elresults.HostReadWrite();
   local_x.HostReadWrite();
   local_y_fa.HostReadWrite();
   for (int i = 0; i < fes.GetNE(); i++) {
      Ttr = fes.GetElementTransformation(i);
      for (int j = 0; j < ndofs; j++) {
         elfun(j) = local_x((i * ndofs) + j);
      }

      elresults = 0.0;
      nlf_int->AssembleElementVector(el, *Ttr, elfun, elresults);
      for (int j = 0; j < ndofs; j++) {
         local_y_fa((i * ndofs) + j) = elresults(j);
      }
   }

   // Perform the setup and action operation of our PA operation
   nlf_int->AssemblePA(fes);
   nlf_int->AddMultPA(local_x, local_y_pa);

   // Take all of our multiple elements and go back to the L vector.
   elem_restrict_lex->MultTranspose(local_y_fa, y_fa);
   elem_restrict_lex->MultTranspose(local_y_pa, y_pa);
   // Find out how different our solutions were from one another.
   double mag = y_fa.Norml2();
   std::cout << "y_fa mag: " << mag << std::endl;
   y_fa -= y_pa;
   double difference = y_fa.Norml2();
   // Free up memory now.
   delete nlf_int;
   delete model;
   delete pmesh;

   return difference / mag;
}

// This function compares the difference in the formation of the GetGradient operator and then multiplying it
// by the necessary vector, and the matrix-free partial assembly formulation which avoids forming the matrix.
// It's been tested on higher order elements and multiple elements. The difference in these two methods
// should be 0.0.
template<bool cmat_ones>
double ExaNLFIntegratorEATest()
{
   int dim = 3;
   Mesh *mesh;
   // Making this mesh and test real simple with 8 elements and then a cubic element
   mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 0, 1.0, 1.0, 1.0, false);
   int order = 3;
   H1_FECollection fec(order, dim);

   ParMesh *pmesh = NULL;
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   ParFiniteElementSpace fes(pmesh, &fec, dim);

   delete mesh;

   // All of these Quadrature function variables are needed to instantiate our material model
   // We can just ignore this marked section
   /////////////////////////////////////////////////////////////////////////////////////////
   // Define a quadrature space and material history variable QuadratureFunction.
   int intOrder = 2 * order + 1;
   QuadratureSpace qspace(pmesh, intOrder); // 3rd order polynomial for 2x2x2 quadrature
   // for first order finite elements.
   QuadratureFunction q_matVars0(&qspace, 1);
   QuadratureFunction q_matVars1(&qspace, 1);
   QuadratureFunction q_sigma0(&qspace, 1);
   QuadratureFunction q_sigma1(&qspace, 1);
   // We'll modify this before doing the partial assembly
   // This is our stiffness matrix and is a 6x6 due to major and minor symmetry
   // of the 4th order tensor which has dimensions 3x3x3x3.
   QuadratureFunction q_matGrad(&qspace, 36);
   QuadratureFunction q_kinVars0(&qspace, 9);
   QuadratureFunction q_vonMises(&qspace, 1);
   ParGridFunction beg_crds(&fes);
   ParGridFunction end_crds(&fes);
   // We'll want to update this later in case we do anything more complicated.
   Vector matProps(1);

   end_crds = 1.0;

   const int NE = fes.GetMesh()->GetNE();
   const int elemDofs = fes.GetFE(0)->GetDof() * fes.GetFE(0)->GetDim();;

   Vector ea_data;

   ea_data.SetSize(NE * elemDofs * elemDofs, Device::GetMemoryType());
   ea_data.UseDevice(true);

   ExaModel *model;
   // This doesn't really matter and is just needed for the integrator class.
   model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1, &q_kinVars0,
                               &beg_crds, &end_crds, &matProps, 1, 1, &fes, true);
   // Model time needs to be set.
   model->SetModelDt(1.0);
   /////////////////////////////////////////////////////////////////////////////
   ExaNLFIntegrator* nlf_int;

   nlf_int = new ExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model));

   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *Ttr;

   // So, we're going to originally support non tensor-product type elements originally.
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *elem_restrict_lex;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   // Set our field variable to a linear spacing so 1 ... ndofs in field
   Vector xtrue(end_crds.Size());
   for (int i = 0; i < xtrue.Size(); i++) {
      xtrue(i) = i + 1;
   }

   // For multiple elements xtrue and local_x are differently sized
   Vector local_x(elem_restrict_lex->Height());
   // All of our local global solution variables
   Vector y_fa(end_crds.Size());
   Vector local_y_fa(elem_restrict_lex->Height());
   Vector local_y_ea(elem_restrict_lex->Height());
   Vector y_ea(end_crds.Size());
   // Initializing them all to 1.
   y_fa = 0.0;
   y_ea = 0.0;
   local_y_ea = 0.0;
   local_y_fa = 0.0;
   // Get our local x values (element values) from the global vector
   elem_restrict_lex->Mult(xtrue, local_x);
   // Variables used to kinda mimic what the NonlinearForm::GetGradient does.
   int ndofs = el.GetDof() * el.GetDim();
   Vector elfun(ndofs), elresults(ndofs);
   DenseMatrix elmat;

   // Set our CMat array for the non-PA case
   q_matGrad = 0.0;
   setCMat<cmat_ones>(q_matGrad);
   elfun.HostReadWrite();
   elresults.HostReadWrite();
   local_x.HostReadWrite();
   local_y_fa.HostReadWrite();
   for (int i = 0; i < fes.GetNE(); i++) {
      Ttr = fes.GetElementTransformation(i);
      for (int j = 0; j < ndofs; j++) {
         elfun(j) = local_x((i * ndofs) + j);
      }

      nlf_int->AssembleElementGrad(el, *Ttr, elfun, elmat);
      // Getting out the local action of our gradient operator on
      // the local x values and then saving the results off to the
      // global variable.
      elresults = 0.0;
      elmat.AddMult(elfun, elresults);
      for (int j = 0; j < ndofs; j++) {
         local_y_fa((i * ndofs) + j) = elresults(j);
      }
   }

   // Perform the setup and action operation of our PA operation
   nlf_int->AssembleEA(fes, ea_data);

   const bool useRestrict = true && elem_restrict_lex;
   // Apply the Element Matrices
   const int NDOFS = elemDofs;
   auto X = Reshape(useRestrict?local_x.HostRead():xtrue.HostRead(), NDOFS, NE);
   auto Y = Reshape(useRestrict?local_y_ea.HostReadWrite():y_ea.HostReadWrite(), NDOFS, NE);
   auto A = Reshape(ea_data.HostRead(), NDOFS, NDOFS, NE);
   for(int glob_j = 0; glob_j < NE*NDOFS; glob_j++)
   {
      const int e = glob_j/NDOFS;
      const int j = glob_j%NDOFS;
      double res = 0.0;
      for (int i = 0; i < NDOFS; i++)
      {
         res += A(i, j, e)*X(i, e);
      }
      Y(j, e) += res;
   }

   // Take all of our multiple elements and go back to the L vector.
   elem_restrict_lex->MultTranspose(local_y_fa, y_fa);
   elem_restrict_lex->MultTranspose(local_y_ea, y_ea);
   // Find out how different our solutions were from one another.
   double mag = y_fa.Norml2();
   std::cout << "y_fa mag: " << mag << std::endl;
   y_fa -= y_ea;
   double difference = y_fa.Norml2();
   // Free up memory now.
   delete nlf_int;
   delete model;
   delete pmesh;

   return difference / mag;
}

// This function compares the difference in the formation of the GetGradient operator and then multiplying it
// by the necessary vector, and the matrix-free partial assembly formulation which avoids forming the matrix.
// It's been tested on higher order elements and multiple elements. The difference in these two methods
// should be 0.0.
template<bool cmat_ones>
double ICExaNLFIntegratorEATest()
{
   int dim = 3;
   Mesh *mesh;
   // Making this mesh and test real simple with 8 elements and then a cubic element
   mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, 0, 1.0, 1.0, 1.0, false);
   int order = 2;
   H1_FECollection fec(order, dim);

   ParMesh *pmesh = NULL;
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   ParFiniteElementSpace fes(pmesh, &fec, dim);

   delete mesh;

   // All of these Quadrature function variables are needed to instantiate our material model
   // We can just ignore this marked section
   /////////////////////////////////////////////////////////////////////////////////////////
   // Define a quadrature space and material history variable QuadratureFunction.
   int intOrder = 2 * order + 1;
   QuadratureSpace qspace(pmesh, intOrder); // 3rd order polynomial for 2x2x2 quadrature
   // for first order finite elements.
   QuadratureFunction q_matVars0(&qspace, 1);
   QuadratureFunction q_matVars1(&qspace, 1);
   QuadratureFunction q_sigma0(&qspace, 1);
   QuadratureFunction q_sigma1(&qspace, 1);
   // We'll modify this before doing the partial assembly
   // This is our stiffness matrix and is a 6x6 due to major and minor symmetry
   // of the 4th order tensor which has dimensions 3x3x3x3.
   QuadratureFunction q_matGrad(&qspace, 36);
   QuadratureFunction q_kinVars0(&qspace, 9);
   QuadratureFunction q_vonMises(&qspace, 1);
   ParGridFunction beg_crds(&fes);
   ParGridFunction end_crds(&fes);
   // We'll want to update this later in case we do anything more complicated.
   Vector matProps(1);

   end_crds = 1.0;

   const int NE = fes.GetMesh()->GetNE();
   const int elemDofs = fes.GetFE(0)->GetDof() * fes.GetFE(0)->GetDim();;

   Vector ea_data;

   ea_data.SetSize(NE * elemDofs * elemDofs, Device::GetMemoryType());
   ea_data.UseDevice(true);

   ExaModel *model;
   // This doesn't really matter and is just needed for the integrator class.
   model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1, &q_kinVars0,
                               &beg_crds, &end_crds, &matProps, 1, 1, &fes, true);
   // Model time needs to be set.
   model->SetModelDt(1.0);
   /////////////////////////////////////////////////////////////////////////////
   ExaNLFIntegrator* nlf_int;

   nlf_int = new ICExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model));

   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *Ttr;

   // So, we're going to originally support non tensor-product type elements originally.
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *elem_restrict_lex;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   // Set our field variable to a linear spacing so 1 ... ndofs in field
   Vector xtrue(end_crds.Size());
   for (int i = 0; i < xtrue.Size(); i++) {
      xtrue(i) = i + 1;
   }

   // For multiple elements xtrue and local_x are differently sized
   Vector local_x(elem_restrict_lex->Height());
   // All of our local global solution variables
   Vector y_fa(end_crds.Size());
   Vector local_y_fa(elem_restrict_lex->Height());
   Vector local_y_ea(elem_restrict_lex->Height());
   Vector y_ea(end_crds.Size());
   // Initializing them all to 1.
   y_fa = 0.0;
   y_ea = 0.0;
   local_y_ea = 0.0;
   local_y_fa = 0.0;
   // Get our local x values (element values) from the global vector
   elem_restrict_lex->Mult(xtrue, local_x);
   // Variables used to kinda mimic what the NonlinearForm::GetGradient does.
   int ndofs = el.GetDof() * el.GetDim();
   Vector elfun(ndofs), elresults(ndofs);
   DenseMatrix elmat;

   // Set our CMat array for the non-PA case
   q_matGrad = 0.0;
   setCMat<cmat_ones>(q_matGrad);
   elfun.HostReadWrite();
   elresults.HostReadWrite();
   local_x.HostReadWrite();
   local_y_fa.HostReadWrite();
   for (int i = 0; i < fes.GetNE(); i++) {
      Ttr = fes.GetElementTransformation(i);
      for (int j = 0; j < ndofs; j++) {
         elfun(j) = local_x((i * ndofs) + j);
      }

      nlf_int->AssembleElementGrad(el, *Ttr, elfun, elmat);
      // Getting out the local action of our gradient operator on
      // the local x values and then saving the results off to the
      // global variable.
      elresults = 0.0;
      elmat.AddMult(elfun, elresults);
      for (int j = 0; j < ndofs; j++) {
         local_y_fa((i * ndofs) + j) = elresults(j);
      }
   }

   // Perform the setup and action operation of our PA operation
   nlf_int->AssemblePA(fes);
   nlf_int->AssembleEA(fes, ea_data);
   DenseMatrix elmat2(ea_data.HostReadWrite(), ndofs, ndofs);
   std::cout << std::endl;
   elmat -= elmat2;
   for (int i = 0; i < ndofs ; i++){
      for(int j = 0; j < ndofs; j++){
         if (std::abs(elmat(i,j)) > 1.0e-13) {
            std::cout << "elmat("<< i << ", " << j << ") = " << elmat(i, j) << std::endl;
         }
      }
   }

   const bool useRestrict = true && elem_restrict_lex;
   // Apply the Element Matrices
   const int NDOFS = elemDofs;
   auto X = Reshape(useRestrict?local_x.HostRead():xtrue.HostRead(), NDOFS, NE);
   auto Y = Reshape(useRestrict?local_y_ea.HostReadWrite():y_ea.HostReadWrite(), NDOFS, NE);
   auto A = Reshape(ea_data.HostRead(), NDOFS, NDOFS, NE);
   for(int glob_j = 0; glob_j < NE*NDOFS; glob_j++)
   {
      const int e = glob_j/NDOFS;
      const int j = glob_j%NDOFS;
      double res = 0.0;
      for (int i = 0; i < NDOFS; i++)
      {
         res += A(i, j, e)*X(i, e);
      }
      Y(j, e) += res;
   }

   // Take all of our multiple elements and go back to the L vector.
   elem_restrict_lex->MultTranspose(local_y_fa, y_fa);
   elem_restrict_lex->MultTranspose(local_y_ea, y_ea);
   // Find out how different our solutions were from one another.
   double mag = y_fa.Norml2();
   std::cout << "y_fa mag: " << mag << std::endl;
   y_fa -= y_ea;
   double difference = y_fa.Norml2();
   // Free up memory now.
   delete nlf_int;
   delete model;
   delete pmesh;

   return difference / mag;
}

template<bool cmat_ones>
void setCMat(QuadratureFunction &cmat_data)
{
   int npts = cmat_data.Size() / cmat_data.GetVDim();
   const int dim2 = 6;

   if (cmat_ones) {
      cmat_data = 1.0;
   }
   else {
      const int DIM3 = 3;
      std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
      // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
      RAJA::Layout<DIM3> layout_2Dtensor = RAJA::make_permuted_layout({{ dim2, dim2, npts } }, perm3);
      RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > cmat(cmat_data.HostReadWrite(), layout_2Dtensor);
      for (int i = 0; i < npts; i++) {
         cmat(0, 0, i) = 100.;
         cmat(1, 1, i) = 100.;
         cmat(2, 2, i) = 100.;
         cmat(0, 1, i) = 75.;
         cmat(1, 0, i) = 75.;
         cmat(0, 2, i) = 75.;
         cmat(2, 0, i) = 75.;
         cmat(1, 2, i) = 75.;
         cmat(2, 1, i) = 75.;
         cmat(3, 3, i) = 50.;
         cmat(4, 4, i) = 50.;
         cmat(5, 5, i) = 50.;
      } // end of pa style set ups
   } // end of cmat set 1 or as cubic material
}

TEST(exaconstit, partial_assembly)
{
   double difference = ExaNLFIntegratorPATest<false>();
   std::cout << difference << std::endl;
   EXPECT_LT(fabs(difference), 1.0e-14) << "Did not get expected value for pa false";
   difference = ExaNLFIntegratorPATest<true>();
   std::cout << difference << std::endl;
   EXPECT_LT(fabs(difference), 1.0e-14) << "Did not get expected value for pa true";
   difference = ExaNLFIntegratorPAVecTest();
   std::cout << difference << std::endl;
   EXPECT_LT(fabs(difference), 1e-15) << "Did not get expected value for pa vec";
}

TEST(exaconstit, ea_assembly)
{
   double difference = ExaNLFIntegratorEATest<false>();
   std::cout << difference << std::endl;
   EXPECT_LT(fabs(difference), 1.0e-14) << "Did not get expected value for ea false";
   difference = ExaNLFIntegratorEATest<true>();
   std::cout << difference << std::endl;
   EXPECT_LT(fabs(difference), 1.0e-14) << "Did not get expected value for ea true";
}

TEST(exaconstit, ic_ea_assembly)
{
   double difference = ICExaNLFIntegratorEATest<false>();
   std::cout << difference << std::endl;
   EXPECT_LT(fabs(difference), 1.0e-14) << "Did not get expected value for ea false";
   difference = ICExaNLFIntegratorEATest<true>();
   std::cout << difference << std::endl;
   EXPECT_LT(fabs(difference), 1.0e-14) << "Did not get expected value for ea true";
}

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   // Testing the case for a dense CMat and then a sparser version of CMat

   Device device("debug");
   printf("\n");
   device.Print();

   ::testing::InitGoogleTest(&argc, argv);
   if (argc > 1) {
      outputLevel = atoi(argv[1]);
   }
   std::cout << "got outputLevel : " << outputLevel << std::endl;

   int i = RUN_ALL_TESTS();

   MPI_Finalize();

   return i;
}