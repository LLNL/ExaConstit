#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_operator_ext.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_operator.hpp"
#include "RAJA/RAJA.hpp"

using namespace mfem;

MechOperatorJacobiSmoother::MechOperatorJacobiSmoother(const Vector &d,
                                                       const Array<int> &ess_tdofs,
                                                       const double dmpng)
   :
   Solver(d.Size()),
   N(d.Size()),
   dinv(N),
   damping(dmpng),
   ess_tdof_list(ess_tdofs),
   residual(N)
{
   Setup(d);
}

void MechOperatorJacobiSmoother::Setup(const Vector &diag)
{
   residual.UseDevice(true);
   dinv.UseDevice(true);
   const double delta = damping;
   auto D = diag.Read();
   auto DI = dinv.Write();
   MFEM_FORALL(i, N, DI[i] = delta / D[i]; );
   auto I = ess_tdof_list.Read();
   MFEM_FORALL(i, ess_tdof_list.Size(), DI[I[i]] = delta; );
}

void MechOperatorJacobiSmoother::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == N, "invalid input vector");
   MFEM_ASSERT(y.Size() == N, "invalid output vector");

   if (iterative_mode && oper) {
      oper->Mult(y, residual); // r = A x
      subtract(x, residual, residual); // r = b - A x
   }
   else {
      residual = x;
      y.UseDevice(true);
      y = 0.0;
   }
   auto DI = dinv.Read();
   auto R = residual.Read();
   auto Y = y.ReadWrite();
   MFEM_FORALL(i, N, Y[i] += DI[i] * R[i]; );
}

NonlinearMechOperatorExt::NonlinearMechOperatorExt(NonlinearForm *_oper_mech)
   : Operator(_oper_mech->FESpace()->GetTrueVSize()), oper_mech(_oper_mech)
{
   // empty
}

PANonlinearMechOperatorGradExt::PANonlinearMechOperatorGradExt(NonlinearForm *_oper_mech) :
   NonlinearMechOperatorExt(_oper_mech), fes(_oper_mech->FESpace())
{
   // So, we're going to originally support non tensor-product type elements originally.
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes->GetElementRestriction(ordering);
   if (elem_restrict_lex) {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      ones.SetSize(elem_restrict_lex->Width(), Device::GetMemoryType());
      ones.UseDevice(true); // ensure 'x = 1.0' is done on device
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
      localX.UseDevice(true);
      ones = 1.0;
   }
}

void PANonlinearMechOperatorGradExt::Assemble()
{
   Array<NonlinearFormIntegrator*> &integrators = *oper_mech->GetDNFI();
   const int num_int = integrators.Size();
   for (int i = 0; i < num_int; ++i) {
      integrators[i]->AssemblePAGrad(*oper_mech->FESpace());
   }
}

void PANonlinearMechOperatorGradExt::AssembleDiagonal(Vector &diag)
{
   Mult(ones, diag);
}

void PANonlinearMechOperatorGradExt::Mult(const Vector &x, Vector &y) const
{
   Array<NonlinearFormIntegrator*> &integrators = *oper_mech->GetDNFI();
   const int num_int = integrators.Size();
   if (elem_restrict_lex) {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < num_int; ++i) {
         integrators[i]->AddMultPAGrad(localX, localY);
      }

      elem_restrict_lex->MultTranspose(localY, y);
   }
   else {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < num_int; ++i) {
         integrators[i]->AddMultPAGrad(x, y);
      }
   }
}
