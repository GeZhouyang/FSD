// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Integrator.cuh
    \brief Declares helper functions for integration.
*/
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

//! Define the step_one kernel
#ifndef __HELPER_INTEGRATOR_CUH__
#define __HELPER_INTEGRATOR_CUH__

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

__global__ void Integrator_RFD_RandDisp_kernel(
						float *d_psi,
						unsigned int N,
						const unsigned int seed
						);
__global__ void Integrator_ZeroVelocity_kernel( 
						float *d_b,
						unsigned int N
						);
__global__ void Integrator_AddStrainRate_kernel( 
						float *d_b,
						float shear_rate,
						unsigned int N
						);

#endif
