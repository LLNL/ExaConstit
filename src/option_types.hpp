#ifndef OPTION_TYPES
#define OPTION_TYPES

// Taking advantage of C++11 to make it much clearer that we're using enums
enum class KrylovSolver { GMRES, PCG, MINRES, NOTYPE };
enum class OriType { EULER, QUAT, CUSTOM, NOTYPE };
enum class MeshType { CUBIT, AUTO, OTHER, NOTYPE };
// Later on we'll want to support multiple different types here like
// BCC and HCP at a minimum. However, we'll need to wait on that support reaching
// ExaCMech
enum class XtalType { FCC, BCC, HCP, NOTYPE };
// We currently only have support for UMATs and ExaCMech later on this might change
// to add support for more systems.
enum class MechType { UMAT, EXACMECH, NOTYPE };
// Hardening law and slip kinetics we'll be using if ExaCMech is specified
// MTSDD refers to a MTS like slip kinetics with DD hardening evolution
// POWERVOCE refers to power law slip kinetics with a linear voce hardening law
// POWERVOCENL refers to power law slip kinetics with a nonlinear voce hardening law
// We might expand upon this later on as more options are added to ExaCMech
// If ExaCMech also eventually allows for the mix and match of different slip laws with
// power laws this will also change
enum class SlipType { MTSDD, POWERVOCE, POWERVOCENL, NOTYPE };
// We're going to use this to determine what runtime model to use for our
// kernels and assembly operations.
enum class RTModel { CPU, CUDA, HIP, OPENMP, NOTYPE };
// The assembly model that we want to make use of FULL does the typical
// full assembly of all the elemental jacobian / tangent matrices, PA
// does a partial assembly type operations, and EA does an element assembly
// type operation.
// The full assembly should be faster for linear type elements and
// partial assembly should be faster for higher order elements.
// We'll have PA and EA on the GPU and the full might get on there as well at
// a later point in time.
// The PA is a matrix-free operation which means traditional preconditioners
// do not exist. Therefore, you'll be limited to Jacobi type preconditioners
// currently implemented.
enum class Assembly { PA, EA, FULL, NOTYPE };

// The nonlinear solver we're making use of to solve everything.
// The current options are Newton-Raphson or Newton-Raphson with a line search
enum class NLSolver { NR, NRLS, NOTYPE };

// Integration formulation that we want to use
enum class IntegrationType { FULL, BBAR, NOTYPE };

#endif
