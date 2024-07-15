// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Stokes.cuh
    \brief Declares GPU kernel code for helper functions integration considering hydrodynamic interactions on the GPU. Used by Stokes.
*/
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

//! Define the step_one kernel
#ifndef __HELPER_STOKES_CUH__
#define __HELPER_STOKES_CUH__

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif


__global__ void Stokes_SetForce_kernel(
					Scalar4 *d_net_force,
					float   *d_AppliedForce,
					unsigned int group_size,
					unsigned int *d_group_members
					);

__global__ void Stokes_SetForce_manually_kernel(
						const Scalar4 *d_pos,
						Scalar3 *d_ori,
						float   *d_AppliedForce,
						unsigned int group_size,
						unsigned int *d_group_members,
						const unsigned int *d_nneigh, 
						unsigned int *d_nlist, 
						const unsigned int *d_headlist,
						const float ndsr,
						const float k_n,
						const float kappa,
						const float beta,
						const float epsq,
						Scalar T_ext,
						const BoxDim box
						);


__global__ void Stokes_SetVelocity_kernel(
						Scalar4 *d_vel,
						Scalar4 *d_omg,
						float   *d_Velocity,
						unsigned int group_size,
						unsigned int *d_group_members
						);


#endif
