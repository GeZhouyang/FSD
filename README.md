## Fast Stokesian Dynamics (FSD)

Original authors: Andrew M. Fiore & James W. Swan (MIT) [1].

>  Note: The original FSD code contains errors in the lubrication, mobility, precondition and Brownian calculations.
>  This version fixes those issues and tries to improve the overall performance and clarity.
>  Furthermore, the solver has been adapted to simulate active suspensions of squirmers based on the *Active Stokesian Dynamics* framework, 
>  see Refs. [2-3] for details.

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

Despite our effort to verify the solver and reduce the number of mistakes, there could always be more bugs. 
So, if you found any please do not hesitate to contact me.

### Acknowledgements

I would like to thank Boyuan Chen (Caltech) for extensive help in debugging the code. 
I would also like to thank William Torre (Utrecht) for discussions about the solver.

### Reference

1. Fiore, A. M., & Swan, J. W. (2019). [Fast Stokesian dynamics](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46). *Journal of Fluid Mechanics*, 878, 544-597.
2. Elfring, G. J., & Brady, J. F. (2022). [Active Stokesian dynamics](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/active-stokesian-dynamics/4FAE47B1A6F0531AE9B6C8F1EAC6D95C). *Journal of Fluid Mechanics*, 952, A19.
3. Ge, Z., & Elfring, G. J. (2025). [Hydrodynamic diffusion in apolar active suspensions of squirmers](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/hydrodynamic-diffusion-in-apolar-active-suspensions-of-squirmers/8596439F68F3E3D6B5A194EB005E992A). *Journal of Fluid Mechanics*, 1003, A17.
