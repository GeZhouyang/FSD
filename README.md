## Fast Stokesian Dynamics (FSD)

Original authors: Andrew M. Fiore, James W. Swan (MIT)

>  Note: The original FSD code contains a few minor errors in the lubrication and mobility calculations.
>  Also, there was a bug causing memory leaks in the preconditioner.
>  This version fixes these issues and tries to improve the overall performance and clarity.

A brief summary of the main files is given below. Deterministic hydrodynamics:

>	- Stokes.cc			        : C++ module to set up the method and run the integrator
>	- Stokes.cu			        : Driver function for integration
>	- Integrator.cu			    : Integrator wrappers, RK2, RFD, etc.
>	- Lubrication.cu		    : Near-field resistance (lubrication) functions (RFU, RFE, RSU, RSE)
>	- Solvers.cu			    : Methods to perform required matrix inversions
>	- Saddle.cu			        : Saddle point multiplication and solution
>	- Mobility.cu			    : Far-field mobility calculations
>	- Precondition.cu		    : Build the saddle point and near-field Brownian preconditioners
>	- Wrappers.cuh			    : C++ wrapper definitions for CUSP operations
>	- Stokes_ResistanceTable.cc	: Values for pre-computed tabulation of lubrication functions
>	- Stokes_SparseMath.cc		: Initialization and setup of variables required for sparse operations

Brownian motion:

>	- Brownian_FarField.cu 		: Methods to compute the far-field Brownian displacements
>	- Brownian_NearField.cu		: Methods to compute the near-field Brownian forces

Auxiliary functions:

>	- Helper_Stokes.cu		    : Helper functions for two-step Stokes integrator	
>	- Helper_Integrator.cu		: Helper functions to simplify code in Integrator.cu
>	- Helper_Saddle.cu		    : Helper functions for saddle point matrix calculations
>	- Helper_Mobility.cu		: Helper functions for mobility calculations in Mobility.cu
>	- Helper_Precondition.cu	: Helper functions for preconditioning calcualtions
>	- Helper_Brownian.cu		: Helper functions used in Brownian_FarField.cu and Brownian_NearField.cu
>	- Helper_Debug.cu		    : Functions for debugging and code checking, printing output, etc.

Brief summary of file dependency (not listed, but all files depend on their helpers, e.g. Mobility.cu depends on Helper_Mobility.cu)

>	- Stokes.cu		            : Integrator.cu	
>	- Integrator.cu	         	: Brownian_FarField.cu	Brownian_NearField.cu	Lubrication.cu	Solvers.cu
>	- Mobility.cu	         	: Helper_Mobility.cu
>	- Brownian_FarField.cu	    : Mobility.cu
>	- Brownian_NearField.cu	    : Lubrication.cu	Preconditioner.cu
>	- Solvers.cu		        : Saddle.cu		Wrappers.cuh
>	- Saddle.cu		            : Mobility.cu		Lubrication.cu


### Reference

> Fiore, A. M., & Swan, J. W. (2019). Fast Stokesian dynamics. *Journal of Fluid Mechanics*, 878, 544-597.
