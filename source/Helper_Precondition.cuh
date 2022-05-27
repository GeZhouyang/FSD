// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Precondition.cuh
    \brief Declares helper functions for error checking and sparse math.
*/
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include <cusparse.h>

//! Define the step_one kernel
#ifndef __HELPER_PRECONDITION_CUH__
#define __HELPER_PRECONDITION_CUH__

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

__global__ void Precondition_ZeroVector_kernel( 
						float *d_b,
						const unsigned int nnz,
						const unsigned int group_size
						);

__global__ void Precondition_ApplyRCM_Vector_kernel( 
							float *d_Scratch_Vector,
							float *d_Vector,
							const int *d_prcm,
							const int length,
							const int direction
							);

__global__ void Precondition_AddInt_kernel(
						unsigned int *d_a,
						unsigned int *d_b,
						unsigned int *d_c,
						int coeff_a,
						int coeff_b,
						unsigned int group_size 
						);

__global__ void Precondition_AddIdentity_kernel(
						float *d_L_Val,
						int   *d_L_RowPtr,
						int   *d_L_ColInd, 
						int group_size,
						float ichol_relaxer
						);

__global__ void Precondition_Inn_kernel(
						Scalar *d_y,
						Scalar *d_x,
						int *d_HasNeigh,
						int group_size
						);

__global__ void Precondition_ImInn_kernel(
						Scalar *d_y,
						Scalar *d_x,
						int *d_HasNeigh,
						int group_size
						);

__global__ void Precondition_ExpandPRCM_kernel(
						int *d_prcm,
						int *d_scratch,
						int group_size
						);

__global__ void Precondition_InitializeMap_kernel(
						int *d_map,
						int nnz
						);

__global__ void Precondition_Map_kernel(
						float *d_Scratch,
						float *d_Val,
						int *d_map,
						int nnz
						);

__global__ void Precondition_GetDiags_kernel(
						int group_size, 
						float *d_Diag,
						int   *d_L_RowPtr,
						int   *d_L_ColInd,
						float *d_L_Val
						);

__global__ void Precondition_DiagMult_kernel(
						float *d_y, // output
						float *d_x, // input
						int group_size, 
						float *d_Diag,
						int direction
						);

__global__ void Precondition_ZeroUpperTriangle_kernel( 
							int *d_RowPtr,
							int *d_ColInd,
							float *d_Val,
							int group_size
							);

__global__ void Precondition_Lmult_kernel( 
						float *d_y,
						float *d_x,
						int *d_RowPtr,
						int *d_ColInd,
						float *d_Val,
						int group_size
						);

#endif
