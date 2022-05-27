// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Debug.cuh
    \brief Declares helper functions for error checking and debugging.
*/
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include "DataStruct.h"

#include <cusparse.h>
#include <cusolverSp.h>

#include <time.h>

//! Define the step_one kernel
#ifndef __HELPER_DEBUG_CUH__
#define __HELPER_DEBUG_CUH__

// Error checking
#ifndef __ERRCHK_CUH__
#define __ERRCHK_CUH__

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
/*!
    \param code   returned error code
    \param file   which file the error occured in
    \param line   which line error check was tripped
    \param abort  whether to kill code upon error trigger
*/
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

void Debug_HasNaN( float *d_vec, int N );

void Debug_HasZeroDiag( float *d_Diag, int N );

void Debug_CSRzeroDiag( int *d_RowPtr, int *d_ColInd, float *d_Val, int group_size, int nnz );

void Debug_StatusCheck_cuSparse( cusparseStatus_t spStatus, const char *name );

void Debug_StatusCheck_cuSolver( cusolverStatus_t soStatus );

void Debug_PrintVector_Int( int *d_vec, int N, const char *name );

void Debug_PrintVector_Float( float *d_vec, int N, const char *name );

void Debug_PrintVector_CSR( float *d_Val, int *d_RowPtr, int *d_ColInd, int nrows, int nnz, const char *name );

void Debug_PrintVector_CSR_forMatlab( int *d_RowPtr, int *d_ColInd, float *d_Val, int nrows, int nnz );

void Debug_PrintVector_COO( float *d_Val, int *d_RowInd, int *d_ColInd, int nnz, const char *name );

void Debug_PrintVector_SpIndexing( const unsigned int *d_n_neigh, const unsigned int *d_offset, const unsigned int *d_NEPP, int N );

void Debug_PrintPos( Scalar4 *d_pos, int N );

void Debug_Lattice_SpinViscosity( 
					MobilityData *mob_data,
					ResistanceData *res_data,
					KernelData *ker_data,
					WorkData *work_data,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size,
					const BoxDim box
					);

void Debug_Lattice_ShearViscosity( 
					MobilityData *mob_data,
					ResistanceData *res_data,
					KernelData *ker_data,
					WorkData *work_data,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size,
					const BoxDim box
					);

void Debug_Random_Dss1( 
			ResistanceData *res_data,
			KernelData *ker_data,
			BrownianData *bro_data,
			MobilityData *mob_data,
			Scalar4 *d_pos,
			unsigned int *d_group_members,
			int group_size,
			int3 *d_image,
			const BoxDim box,
			float dt
			);

void Debug_Random_Dss2( 
			ResistanceData *res_data,
			KernelData *ker_data,
			BrownianData *bro_data,
			MobilityData *mob_data,
			Scalar4 *d_pos,
			unsigned int *d_group_members,
			int group_size,
			int3 *d_image,
			const BoxDim box,
			float dt
			);



#endif
