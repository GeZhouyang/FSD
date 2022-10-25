## Fast Stokesian Dynamics (FSD)

Original authors: Andrew M. Fiore, James W. Swan (MIT)

>  Note: This version fixes several bugs in the original FSD code published by the above authors (see ref below).
>  Mainly the *memory leak* (in `Precondition_IChol`), the *lubrication* calculations, 
>  and the *far-field mobility* (inconsistent sign convention with respect to the paper, to be updated).
>  Other modifications include the RK2 integrator (instead of 1st order Euler), stress post-processings, etc.
>  /*Zhouyang Ge*

Description of files within the c++/CUDA module for the Fast Stokesian Dynamics (FSD) plugin for HOOMD, called PSEv3. 
The method is based on the Positively-Split Ewald (PSE) approach for sampling the Brownian displacements of the far-field multipole expansion, 
and is an extension of PSEv1 and PSEv2, which are plugins implementing the PSE algorithm with RPY hydrodynamics and RPY hydrodynamics + stresslets, respectively. 

A brief summary of the main files for the plugin is given here:

>	- Stokes.cc			        : C++ module to set up the method and run the integrator
>	- Stokes.cu			        : Driver function for integration
>	- Integrator.cu			    : Integrator wrappers, RK2, RFD, etc.
>	- Lubrication.cu		    : Near-field resistance (lubrication) functions (RFU, RSU, RSE)
>	- Solvers.cu			    : Methods to perform required matrix inversions
>	- Saddle.cu			        : Saddle point multiplication and solution
>	- Mobility.cu			    : Far-field mobility calculations
>	- Precondition.cu		    : Build the saddle point and near-field Brownian preconditioners
>	- Wrappers.cuh			    : C++ wrapper definitions for CUSP operations
>
>	- Brownian_FarField.cu 		: Methods to compute the far-field Brownian displacements
>	- Brownian_NearField.cu		: Methods to compute the near-field Brownian forces
>	- Helper_Brownian.cu		: Helper functions used in Brownian_FarField.cu and Brownian_NearField.cu
>	
>	- Helper_Debug.cu		    : Functions for debugging and code checking, printing output, etc.
>	- Helper_Integrator.cu		: Helper functions to simplify code in Integrator.cu
>	- Helper_Mobility.cu		: Helper functions for mobility calculations in Mobility.cu
>	- Helper_Precondition.cu	: Helper functions for preconditioning calcualtions
>	- Helper_Saddle.cu		    : Helper functions for saddle point matrix calculations
>	- Helper_Stokes.cu		    : Helper functions for two-step Stokes integrator	
>
>	- Stokes_ResistanceTable.cc	: Values for pre-computed tabulation of lubrication functions
>	- Stokes_SparseMath.cc		: Initialization and setup of variables required for sparse operations

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
