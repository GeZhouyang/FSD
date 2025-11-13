## Fast Stokesian Dynamics (FSD)

Original authors: Andrew M. Fiore & James W. Swan (MIT) [1].

>  Note: The original FSD code contains errors in the lubrication, mobility, precondition and Brownian calculations.
>  This version tries to fix those issues and improve the overall performance and clarity.
>  Furthermore, the solver has been adapted to simulate active matter based on the *Active Stokesian Dynamics* framework [2]. 
>  Some recent publications using data generated from this solver is listed in [3-5].
>  Feel free to let me know if you used our code and would like your paper to be listed here.

A brief summary of the main files is given below. Main structure:

>   - DataStruct.h              : Declaration of the various data structures
>	- Stokes.cc			        : C++ module to set up the method and run the integrator
>	- Stokes.cu			        : Driver function for the velocity calculation and temporal integration
>	- Precondition.cu		    : Preconditioners for the saddle point and near-field Brownian solves
>	- Integrator.cu			    : Velocity computation and integrators, including the RFD
>	- Solvers.cu			    : Methods to perform required matrix inversions
>	- Wrappers.cuh			    : C++ wrapper definitions for CUSP operations
>	- Saddle.cu			        : Saddle point multiplication and solution

Deterministic hydrodynamics:

>	- Mobility.cu			    : Far-field mobility calculations
>	- Lubrication.cu		    : Near-field resistance (lubrication) functions (RFU, RFE, RSU, RSE)
>	- Stokes_ResistanceTable.cc	: Values for pre-computed tabulation of lubrication functions
>	- Stokes_SparseMath.cc   	: Setup of the cuSparse operations (triangular solve, Cholesky, etc)

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

### How to install

Detailed instructions can be found in the supplemental material of Andrew's JFM [1],
which still apply to the current version.
We are working on upgrading the codebase to the latest toolchains,
and will update the source code when ready.

### Disclaimer

Despite our effort to verify the solver and reduce the number of mistakes, there could always be more bugs. 
So, if you found any please do not hesitate to contact us.

### Acknowledgements

I would like to thank Boyuan Chen (Caltech) for extensive help in debugging the code. 
I would also like to thank William Torre (Utrecht) for discussions about the solver.

### Reference

1. Fiore & Swan (2019). [*Journal of Fluid Mechanics*, 878, 544-597.](https://doi.org/10.1017/jfm.2019.640)
2. Elfring & Brady (2022). [*Journal of Fluid Mechanics*, 952, A19.](https://doi.org/10.1017/jfm.2022.909)
3. Ge & Elfring (2022). [*Physical Review E*, 106, 054616 (2022)](https://doi.org/10.1103/PhysRevE.106.054616)
4. Ge & Elfring (2025). [*Journal of Fluid Mechanics*, 1003, A17.](https://doi.org/10.1017/jfm.2024.1071)
5. Ge, Brady & Elfring (2025). To appear in [*Physical Review Letters*.](https://doi.org/10.1103/54qq-1s51)
