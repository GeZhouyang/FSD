// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Saddle.cuh
    \brief Declared helper functions for saddle point calculations
*/
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include <stdlib.h>
#include "cusparse.h"

//! Define the step_one kernel
#ifndef __HELPER_SADDLE_CUH__
#define __HELPER_SADDLE_CUH__

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

__global__ void Saddle_ZeroOutput_kernel( 
						float *d_b, 
						unsigned int N 
						);

__global__ void Saddle_AddFloat_kernel( 	float *d_a, 
						float *d_b,
						float *d_c,
						float coeff_a,
						float coeff_b,
						unsigned int N,
						int stride
					);

__global__ void Saddle_SplitGeneralizedF_kernel( 	float *d_GeneralF, 
							Scalar4 *d_net_force,
							Scalar4 *d_TorqueStress,
							unsigned int N
					);

__global__ void Saddle_MakeGeneralizedU_kernel( 	float *d_GeneralU, 
							Scalar4 *d_vel,
							Scalar4 *d_AngvelStrain,
							unsigned int N
					);


__global__ void Saddle_force2rhs_kernel(
					float *d_force, 
					float *d_rhs,
					unsigned int N
					);

__global__ void Saddle_solution2vel_kernel(
					float *d_U, 
					float *d_solution,
					unsigned int N
					);


#endif
