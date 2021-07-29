#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_operator_ext.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_log.hpp"
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

PANonlinearMechOperatorGradExt::PANonlinearMechOperatorGradExt(NonlinearForm *_oper_mech, const mfem::Array<int> &ess_tdofs) :
   NonlinearMechOperatorExt(_oper_mech), fes(_oper_mech->FESpace()), ess_tdof_list(ess_tdofs)
{
   // So, we're going to originally support non tensor-product type elements originally.
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes->GetElementRestriction(ordering);
   P = fes->GetProlongationMatrix();
   if (elem_restrict_lex) {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      px.SetSize(elem_restrict_lex->Width(), Device::GetMemoryType());
      ones.SetSize(elem_restrict_lex->Width(), Device::GetMemoryType());
      ones.UseDevice(true); // ensure 'x = 1.0' is done on device
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
      localX.UseDevice(true);
      px.UseDevice(true);
      ones = 1.0;
   }
}

void PANonlinearMechOperatorGradExt::Assemble()
{
   CALI_CXX_MARK_SCOPE("PA_Assemble");
   Array<NonlinearFormIntegrator*> &integrators = *oper_mech->GetDNFI();
   const int num_int = integrators.Size();
   for (int i = 0; i < num_int; ++i) {
      integrators[i]->AssemblePA(*oper_mech->FESpace());
      integrators[i]->AssembleGradPA(*oper_mech->FESpace());
   }
}

void PANonlinearMechOperatorGradExt::AssembleDiagonal(Vector &diag)
{
   CALI_CXX_MARK_SCOPE("AssembleDiagonal");
   Array<NonlinearFormIntegrator*> &integrators = *oper_mech->GetDNFI();
   const int num_int = integrators.Size();

   if (elem_restrict_lex) {
      localY = 0.0;
      for (int i = 0; i < num_int; ++i) {
         integrators[i]->AssembleGradDiagonalPA(localY);
      }

      elem_restrict_lex->MultTranspose(localY, px);
      P->MultTranspose(px, diag);
   }
   else {
      diag.UseDevice(true); // typically this is a large vector, so store on device
      diag = 0.0;
      for (int i = 0; i < num_int; ++i) {
         integrators[i]->AssembleGradDiagonalPA(diag);
      }
   }

   // Apply the essential boundary conditions
   auto Y = diag.ReadWrite();
   auto I = ess_tdof_list.Read();

   MFEM_FORALL(i, ess_tdof_list.Size(), Y[I[i]] = 1.0; );
}

void PANonlinearMechOperatorGradExt::Mult(const Vector &x, Vector &y) const
{
   TMult<false>(x, y);
}

void PANonlinearMechOperatorGradExt::LocalMult(const Vector &x, Vector &y) const
{
   TMult<true>(x, y);
}

template<bool local_action>
void PANonlinearMechOperatorGradExt::TMult(const Vector &x, Vector &y) const
{
   CALI_CXX_MARK_SCOPE("PA_Mult");
   Array<NonlinearFormIntegrator*> &integrators = *oper_mech->GetDNFI();
   const int num_int = integrators.Size();

   // Apply the essential boundary conditions
   ones = x;
   auto I = ess_tdof_list.Read();
   auto Y = ones.ReadWrite();
   MFEM_FORALL(i, ess_tdof_list.Size(), Y[I[i]] = 0.0; );

   if (elem_restrict_lex) {
      P->Mult(ones, px);
      elem_restrict_lex->Mult(px, localX);
      localY = 0.0;
      for (int i = 0; i < num_int; ++i) {
         integrators[i]->AddMultGradPA(localX, localY);
      }

      elem_restrict_lex->MultTranspose(localY, px);
      P->MultTranspose(px, y);
   }
   else {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < num_int; ++i) {
         integrators[i]->AddMultGradPA(x, y);
      }
   }

   // Only apply essential boundary conditions if we don't need to perform the
   // local action of the matrix
   if(!local_action) {
      // Apply the essential boundary conditions
      Y = y.ReadWrite();
      MFEM_FORALL(i, ess_tdof_list.Size(), Y[I[i]] = 0.0; );
   }
}

void PANonlinearMechOperatorGradExt::MultVec(const Vector &x, Vector &y) const
{
   Array<NonlinearFormIntegrator*> &integrators = *oper_mech->GetDNFI();
   const int num_int = integrators.Size();
   if (elem_restrict_lex) {
      P->Mult(x, px);
      elem_restrict_lex->Mult(px, localX);
      localY = 0.0;
      for (int i = 0; i < num_int; ++i) {
         integrators[i]->AddMultPA(localX, localY);
      }

      elem_restrict_lex->MultTranspose(localY, px);
      P->MultTranspose(px, y);
   }
   else {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < num_int; ++i) {
         integrators[i]->AddMultPA(x, y);
      }
   }
   // Apply the essential boundary conditions
   auto I = ess_tdof_list.Read();
   auto Y = y.ReadWrite();
   MFEM_FORALL(i, ess_tdof_list.Size(), Y[I[i]] = 0.0; );
}

// Data and methods for element-assembled bilinear forms
EANonlinearMechOperatorGradExt::EANonlinearMechOperatorGradExt(NonlinearForm *_oper_mech, const mfem::Array<int> &ess_tdofs)
   : PANonlinearMechOperatorGradExt(_oper_mech, ess_tdofs)
{
   NE = _oper_mech->FESpace()->GetMesh()->GetNE();
   elemDofs = _oper_mech->FESpace()->GetFE(0)->GetDof() * _oper_mech->FESpace()->GetFE(0)->GetDim();

   ea_data.SetSize(NE * elemDofs * elemDofs, Device::GetMemoryType());
   ea_data.UseDevice(true);
}

void EANonlinearMechOperatorGradExt::Assemble()
{
   ea_data = 0.0;

   CALI_CXX_MARK_SCOPE("EA_Assemble");
   Array<NonlinearFormIntegrator*> &integrators = *oper_mech->GetDNFI();
   const int num_int = integrators.Size();
   for (int i = 0; i < num_int; ++i) {
      integrators[i]->AssemblePA(*oper_mech->FESpace());
      integrators[i]->AssembleEA(*oper_mech->FESpace(), ea_data);
   }
}

void EANonlinearMechOperatorGradExt::AssembleDiagonal(Vector &diag)
{
   CALI_CXX_MARK_SCOPE("eaAssembleDiagonal");

   const bool useRestrict = true && elem_restrict_lex;
   if (!useRestrict) {
      diag.UseDevice(true); // typically this is a large vector, so store on device
      diag = 0.0;
   }
   else {
      localY = 0.0;
   }

   // Apply the Element Matrices
   const int NDOFS = elemDofs;
   auto Y = Reshape(useRestrict ? localY.ReadWrite() : diag.ReadWrite(), NDOFS, NE);
   auto A = Reshape(ea_data.Read(), NDOFS, NDOFS, NE);
   MFEM_FORALL(glob_j, NE * NDOFS,
   {
      const int NDOFS = elemDofs;
      const int e = glob_j / NDOFS;
      const int j = glob_j % NDOFS;
      Y(j, e) = A(j, j, e);
   });

   // Apply the Element Restriction transposed
   if (useRestrict) {
      elem_restrict_lex->MultTranspose(localY, px);
      P->MultTranspose(px, diag);
   }

   // Apply the essential boundary conditions
   auto R = diag.ReadWrite();
   auto I = ess_tdof_list.Read();

   MFEM_FORALL(i, ess_tdof_list.Size(), R[I[i]] = 1.0; );
}

void EANonlinearMechOperatorGradExt::Mult(const Vector &x, Vector &y) const
{
   TMult<false>(x, y);
}

void EANonlinearMechOperatorGradExt::LocalMult(const Vector &x, Vector &y) const
{
   TMult<true>(x, y);
}

template<bool local_action>
void EANonlinearMechOperatorGradExt::TMult(const Vector &x, Vector &y) const
{
   // Apply the Element Restriction
   // Apply the essential boundary conditions
   ones = x;
   auto I = ess_tdof_list.Read();
   auto R = ones.ReadWrite();
   MFEM_FORALL(i, ess_tdof_list.Size(), R[I[i]] = 0.0; );

   const bool useRestrict = true && elem_restrict_lex;
   if (!useRestrict) {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
   }
   else {
      P->Mult(ones, px);
      elem_restrict_lex->Mult(px, localX);
      localY = 0.0;
   }

   // Apply the Element Matrices
   const int NDOFS = elemDofs;
   auto X = Reshape(useRestrict ? localX.Read() : ones.Read(), NDOFS, NE);
   auto Y = Reshape(useRestrict ? localY.ReadWrite() : y.ReadWrite(), NDOFS, NE);
   auto A = Reshape(ea_data.Read(), NDOFS, NDOFS, NE);
   MFEM_FORALL(glob_j, NE * NDOFS,
   {
      const int e = glob_j / NDOFS;
      const int j = glob_j % NDOFS;
      double res = 0.0;
      for (int i = 0; i < NDOFS; i++) {
         res += A(i, j, e) * X(i, e);
      }

      Y(j, e) += res;
   });
   // Apply the Element Restriction transposed
   if (useRestrict) {
      elem_restrict_lex->MultTranspose(localY, px);
      P->MultTranspose(px, y);
   }

   // Only apply essential boundary conditions if we don't need to perform the
   // local action of the matrix
   if(!local_action){
      // Apply the essential boundary conditions
      R = y.ReadWrite();
      MFEM_FORALL(i, ess_tdof_list.Size(), R[I[i]] = 0.0; );
   }
}