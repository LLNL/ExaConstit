

#include "mechanics_operator.hpp"
#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_umat.hpp"
#include "mechanics_ecmech.hpp"
#include "option_parser.hpp"

using namespace mfem;


NonlinearMechOperator::NonlinearMechOperator(ParFiniteElementSpace &fes,
                                             Array<int> &ess_bdr,
                                             ExaOptions &options,
                                             QuadratureFunction &q_matVars0,
                                             QuadratureFunction &q_matVars1,
                                             QuadratureFunction &q_sigma0,
                                             QuadratureFunction &q_sigma1,
                                             QuadratureFunction &q_matGrad,
                                             QuadratureFunction &q_kinVars0,
                                             QuadratureFunction &q_vonMises,
                                             ParGridFunction &beg_crds,
                                             ParGridFunction &end_crds,
                                             Vector &matProps,
                                             int nStateVars)
: NonlinearForm(&fes), fe_space(fes)
{
   Vector * rhs;
   rhs = NULL;

   mech_type = options.mech_type;
   
   // Define the parallel nonlinear form
   Hform = new ParNonlinearForm(&fes);
   
   // Set the essential boundary conditions
   Hform->SetEssentialBCPartial(ess_bdr, rhs);
   
   if (options.mech_type == MechType::UMAT) {
      //Our class will initialize our deformation gradients and
      //our local shape function gradients which are taken with respect
      //to our initial mesh when 1st created.
      model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                  &q_kinVars0, &beg_crds, &end_crds, 
                                  &matProps, options.nProps, nStateVars, &fes);
      
      // Add the user defined integrator
      Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model)));
      
   } else if (options.mech_type == MechType::EXACMECH){
      //Time to go through a nice switch field to pick out the correct model to be run...
      //Should probably figure a better way to do this in the future so this doesn't become
      //one giant switch yard. Multiphase materials will probably require a complete revamp of things...
      //First we check the xtal symmetry type
      if (options.xtal_type == XtalType::FCC){
         //Now we find out what slip kinetics and hardening law were chosen.
         if(options.slip_type == SlipType::POWERVOCE){
            //Our class will initialize our deformation gradients and
            //our local shape function gradients which are taken with respect
            //to our initial mesh when 1st created.
            model = new VoceFCCModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                  &beg_crds, &end_crds, 
                                  &matProps, options.nProps, nStateVars, options.temp_k);
      
            // Add the user defined integrator
            Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<VoceFCCModel*>(model)));
         } else if(options.slip_type == SlipType::MTSDD){
            //Our class will initialize our deformation gradients and
            //our local shape function gradients which are taken with respect
            //to our initial mesh when 1st created.
            model = new KinKMBalDDFCCModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                  &beg_crds, &end_crds, 
                                  &matProps, options.nProps, nStateVars, options.temp_k);
      
            // Add the user defined integrator
            Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<KinKMBalDDFCCModel*>(model)));
         }
      }

   }
   //We'll probably want to eventually add a print settings into our option class that tells us whether
   //or not we're going to be printing this.
   
   model->setVonMisesPtr(&q_vonMises);
}

const Array<int> &NonlinearMechOperator::GetEssTDofList()
{
   return Hform->GetEssentialTrueDofs();
}

ExaModel *NonlinearMechOperator::GetModel() const
{
   return model;
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const Vector &k, Vector &y) const
{
   //Wanted to put this in the mechanics_solver.cpp file, but I would have needed to update
   //Solver class to use the NonlinearMechOperator instead of Operator class.
   //We now update our end coordinates based on the solved for velocity.
   UpdateEndCoords(k);
   // Apply the nonlinear form
   if(mech_type == MechType::UMAT){
      //I really don't like this. It feels so hacky and
      //potentially dangerous to have these methods just
      //lying around.
      ParGridFunction* end_crds = model->GetEndCoords();
      Vector temp;
      temp.SetSize(k.Size());
      end_crds->GetTrueDofs(temp);
      //Creating a new vector that's going to be used for our
      //UMAT custom Hform->Mult
      const Vector crd(temp.GetData(), temp.Size());
      model -> calc_incr_end_def_grad(crd);

   }
   Hform->Mult(k, y);
}

//Update the end coords used in our model
void NonlinearMechOperator::UpdateEndCoords(const Vector& vel) const {
   model->UpdateEndCoords(vel);
}

// Compute the Jacobian from the nonlinear form
Operator &NonlinearMechOperator::GetGradient(const Vector &x) const
{
   Jacobian = &Hform->GetGradient(x);
   return *Jacobian;
}

NonlinearMechOperator::~NonlinearMechOperator()
{
   delete model;
   delete Hform;
}
