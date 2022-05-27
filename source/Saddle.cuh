// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Saddle.cuh
    \brief Declared functions for saddle point calculations
*/
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include "DataStruct.h"

#include <stdlib.h>

#include <cusparse.h>
#include <cusolverSp.h>

//! Define the step_one kernel
#ifndef __SADDLE_CUH__
#define __SADDLE_CUH__

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif


void Saddle_Multiply( 
                        	float *d_b, // output
				float *d_x, // input
				Scalar4 *d_pos,
				unsigned int *d_group_members,
				unsigned int group_size,
                        	const BoxDim& box,
				KernelData *ker_data,
				MobilityData *mob_data,
				ResistanceData *res_data,
				WorkData *work_data
				);

void Saddle_Preconditioner(	
				float *d_x, // Solution
				float *d_b, // RHS
				int group_size,
				void *pBuffer,
				KernelData *ker_data,
				ResistanceData *res_data
				);


#endif
