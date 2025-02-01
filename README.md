## Fast Stokesian Dynamics (FSD)

Original authors: Andrew M. Fiore & James W. Swan (MIT)

>  Note: The original FSD code contains some errors in the lubrication, mobility and Brownian calculations.
>  Also, there was a bug causing memory leaks in the preconditioner.
>  This version fixes those issues and tries to improve the overall performance and clarity.
>  However, there could always be more bugs, so if you found any please do not hesitate to contact me.

A brief summary of the main files is given below. Main structure:

>	- Stokes.cc			        : C++ module to set up the method and run the integrator
>	- Stokes.cu			        : Driver function for integration (RK2, Euler-Maruyama, etc.)
>	- Precondition.cu		    : Preconditioners for the saddle point and near-field Brownian solves
>	- Integrator.cu			    : Velocity computation and integrators, including the RFD
>	- Solvers.cu			    : Methods to perform required matrix inversions
>	- Wrappers.cuh			    : C++ wrapper definitions for CUSP operations
>	- Saddle.cu			        : Saddle point multiplication and solution

Deterministic hydrodynamics:

>	- Mobility.cu			    : Far-field mobility calculations
>	- Lubrication.cu		    : Near-field resistance (lubrication) functions (RFU, RFE, RSU, RSE)
>	- Stokes_ResistanceTable.cc	: Values for pre-computed tabulation of lubrication functions

Brownian motion:

>	- Brownian_FarField.cu 		: Methods to compute the far-field Brownian displacements
>	- Brownian_NearField.cu		: Methods to compute the near-field Brownian forces

Auxiliary functions:

>	- Helper_Stokes.cu		    : Helper functions for the Stokes integrator	
>	- Helper_Integrator.cu		: Helper functions to simplify code in Integrator.cu
>	- Helper_Saddle.cu		    : Helper functions for saddle point matrix calculations
>	- Helper_Mobility.cu		: Helper functions for mobility calculations in Mobility.cu
>	- Helper_Precondition.cu	: Helper functions for preconditioning calcualtions
>	- Helper_Brownian.cu		: Helper functions used in Brownian_FarField.cu and Brownian_NearField.cu
>	- Helper_Debug.cu		    : Functions for debugging and code checking, printing output, etc.

An *active* version of the solver for simulations of suspensions of squirmers is provided in the branch `squirmer`.
The implementation is based on the *Active Stokesian Dynamics* framework 
and is backward compatible with the original FSD for passive suspensions.

> Note: Brownian motion is not fully validated. At least it should not be used with the implemented RK2 scheme.

### Reference

> Fiore, A. M., & Swan, J. W. (2019). Fast Stokesian dynamics. *Journal of Fluid Mechanics*, 878, 544-597.
> Elfring, G. J., & Brady, J. F. (2022). Active stokesian dynamics. *Journal of Fluid Mechanics*, 952, A19.
> Ge, Z., & Elfring, G. J. (2025). Hydrodynamic diffusion in apolar active suspensions of squirmers. *Journal of Fluid Mechanics*, 1003, A17.
