// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Helper_Debug.cuh"
#include "Helper_Brownian.cuh"
#include "Helper_Precondition.cuh"

#include "Brownian_NearField.cuh"
#include "Integrator.cuh"
#include "Lubrication.cuh"
#include "Mobility.cuh"
#include "Precondition.cuh"
#include "Solvers.cuh"

#include "hoomd/Saru.h"

#include <stdio.h>

#include <cusparse.h>
#include <cusolverSp.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! command to convert floats or doubles to integers
#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif


/*! \file Helper_Sparse.cu
    	\brief Helper functions required for error checking in sparse objects
*/

/*!
	Check the cuSPARSE returned status

	\param spStatus		status returned from cuSPARSE call

*/
void Debug_StatusCheck_cuSparse( cusparseStatus_t spStatus, const char *name ){

	if (      spStatus == CUSPARSE_STATUS_SUCCESS ){
		printf( "\tCUSPARSE Successful Execution in %s. \n", name );
	}
	else if ( spStatus == CUSPARSE_STATUS_NOT_INITIALIZED ){
		printf( "\tCUSPARSE Error in %s = Not Initialized.", name );
	}
	else if ( spStatus == CUSPARSE_STATUS_ALLOC_FAILED ){
		printf( "\tCUSPARSE Error in %s = Allocation Failed.", name );
	}
	else if ( spStatus == CUSPARSE_STATUS_INVALID_VALUE ){
		printf( "\tCUSPARSE Error in %s = Invalid Value.", name );
	}
	else if ( spStatus == CUSPARSE_STATUS_ARCH_MISMATCH ){
		printf( "\tCUSPARSE Error in %s = Architecture Mismatch.", name );
	}
	else if ( spStatus == CUSPARSE_STATUS_MAPPING_ERROR ){
		printf( "\tCUSPARSE Error in %s = Mapping Error.", name );
	}
	else if ( spStatus == CUSPARSE_STATUS_EXECUTION_FAILED ){
		printf( "\tCUSPARSE Error in %s = Execution Failed.", name );
	}
	else if ( spStatus == CUSPARSE_STATUS_INTERNAL_ERROR ){
		printf( "\tCUSPARSE Error in %s = Internal Error.", name );
	}
	else if ( spStatus == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED ){
		printf( "\tCUSPARSE Error in %s = Matrix Type Not Supported.", name );
	}
	else {
		printf( "\tCUSPARSE Undefined Status in %s.", name );
	}

	if ( spStatus != CUSPARSE_STATUS_SUCCESS ){
		printf( " Quitting.\n");
		exit(1);
	}

}

/*!
	Check the cuSOLVER returned status

	\param soStatus		status returned from cuSOLVER call

*/
void Debug_StatusCheck_cuSolver( cusolverStatus_t soStatus ){

	if ( 	  soStatus == CUSOLVER_STATUS_SUCCESS ){
		printf( "\tCUSOLVER Successful Execution. \n" );
	}
	else if ( soStatus == CUSOLVER_STATUS_NOT_INITIALIZED ){
		printf( "\tCUSOLVER Error = Not Initialized." );
	}
	else if ( soStatus == CUSOLVER_STATUS_ALLOC_FAILED ){
		printf( "\tCUSOLVER Error = Allocation Failed." );
	}
	else if ( soStatus == CUSOLVER_STATUS_INVALID_VALUE ){
		printf( "\tCUSOLVER Error = Invalid Value." );
	}
	else if ( soStatus == CUSOLVER_STATUS_ARCH_MISMATCH ){
		printf( "\tCUSOLVER Error = Architecture Mismatch." );
	}
	else if ( soStatus == CUSOLVER_STATUS_MAPPING_ERROR ){
		printf( "\tCUSOLVER Error = Mapping Error." );
	}
	else if ( soStatus == CUSOLVER_STATUS_EXECUTION_FAILED ){
		printf( "\tCUSOLVER Error = Execution Failed." );
	}
	else if ( soStatus == CUSOLVER_STATUS_INTERNAL_ERROR ){
		printf( "\tCUSOLVER Error = Internal Error." );
	}
	else if ( soStatus == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED ){
		printf( "\tCUSOLVER Error = Matrix Type Not Supported." );
	}
	else {
		printf( "\tCUSOLVER Undefined Status." );
	}
	
	if ( soStatus != CUSOLVER_STATUS_SUCCESS ){
		printf( " Quitting.\n");
		exit(1);
	}
}


/*!
	Print arrays for checks
*/
void Debug_PrintVector_Int( int *d_vec, int N, const char *name ){

	// Set up host vector
	int *h_vec;
	h_vec = (int *)malloc( N*sizeof(int) );

	// Copy data to host and print
	cudaMemcpy( h_vec, d_vec, N*sizeof(int), cudaMemcpyDeviceToHost );
	printf( "\n" );
	for ( int ii = 0; ii < N; ++ii ){
		printf( "%s(%i) = %i; \n", name, ii+1, h_vec[ii] );
	}
	printf( "\n" );

	// Add another cudaMemcpy to make sure print statement gets out
	// before any errors
	cudaMemcpy( h_vec, d_vec, N*sizeof(int), cudaMemcpyDeviceToHost );

	// Clean up
	free( h_vec );
}

/*!
	Print arrays for checks
*/
void Debug_PrintVector_Float( float *d_vec, int N, const char *name ){

	// Set up host vector
	float *h_vec;
	h_vec = (float *)malloc( N*sizeof(float) );

	// Copy data to host and print
	cudaMemcpy( h_vec, d_vec, N*sizeof(float), cudaMemcpyDeviceToHost );
	printf( "\n" );
	for ( int ii = 0; ii < N; ++ii ){
		printf( "%s(%i) = %f; \n", name, ii+1, h_vec[ii] );
	}
	printf( "\n" );
	
	// Add another cudaMemcpy to make sure print statement gets out
	// before any errors
	cudaMemcpy( h_vec, d_vec, N*sizeof(float), cudaMemcpyDeviceToHost );

	// Clean up
	free( h_vec );
}

/*!
	Print CSR matrix
*/
void Debug_PrintVector_CSR( float *d_Val, int *d_RowPtr, int *d_ColInd, int nrows, int nnz, const char *name ){

	// Set up host vectors
	int *h_RowPtr, *h_ColInd;
	float *h_Val;
	h_Val = (float *)malloc( nnz*sizeof(float) );
	h_RowPtr = (int *)malloc( (nrows+1)*sizeof(int) );
	h_ColInd = (int *)malloc( nnz*sizeof(int) );

	// Copy data to host
	cudaMemcpy( h_Val, d_Val, nnz*sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_RowPtr, d_RowPtr, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_ColInd, d_ColInd, nnz*sizeof(int), cudaMemcpyDeviceToHost );
	
	printf( "\n" );
	for ( int ii = 0; ii < nrows; ++ii ){

		int offset = h_RowPtr[ ii ];
		int ncols = h_RowPtr[ ii + 1 ] - offset;

		for ( int jj = 0; jj < ncols; ++jj ){

			int col = h_ColInd[ offset + jj ];
			float val = h_Val[ offset + jj ];

			// Print out in base-1
			printf( "%s(%5i,%5i) = %8.5f; \n", name, ii+1, col+1, val );
		}
	}
	printf( "\n" );
	
	// Add another cudaMemcpy to make sure print statement gets out
	// before any errors
	cudaMemcpy( h_Val, d_Val, nnz*sizeof(float), cudaMemcpyDeviceToHost );

	// Clean up
	free( h_Val );
	free( h_RowPtr );
	free( h_ColInd );
}

/*!
	Print Sparse matrix with indices for matlab
*/
void Debug_PrintVector_CSR_forMatlab( int *d_RowPtr, int *d_ColInd, float* d_Val, int nrows, int nnz ){

	// Set up host vectors
	int *h_RowPtr, *h_ColInd;
	float *h_Val;
	h_RowPtr = (int *)malloc( (nrows+1)*sizeof(int) );
	h_ColInd = (int *)malloc( nnz*sizeof(int) );
	h_Val = (float *)malloc( nnz*sizeof(float) );

	// Copy data to host
	cudaMemcpy( h_RowPtr, d_RowPtr, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_ColInd, d_ColInd, nnz*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_Val, d_Val, nnz*sizeof(float), cudaMemcpyDeviceToHost );
	
	printf( "\n" );
	for ( int ii = 0; ii < nrows; ++ii ){

		int offset = h_RowPtr[ ii ];
		int ncols = h_RowPtr[ ii + 1 ] - offset;

		for ( int jj = 0; jj < ncols; ++jj ){

			int col = h_ColInd[ offset + jj ];

			float val = h_Val[ offset + jj ];

			// Print out in base-1
			printf( "%i %i %f \n", ii+1, col+1, val );
		}
	}
	printf( "\n" );
	
	// Clean up
	free( h_RowPtr );
	free( h_ColInd );
	free( h_Val );
}

/*!
	Print arrays for checks
*/
void Debug_PrintVector_SpIndexing( const unsigned int *d_n_neigh, const unsigned int *d_offset, const unsigned int *d_NEPP, int N ){

	// Set up host vector
	int *h_n_neigh, *h_offset, *h_NEPP;
	h_n_neigh = (int *)malloc(   N*sizeof(int) );
	h_offset  = (int *)malloc( 3*N*sizeof(int) );
	h_NEPP    = (int *)malloc( 4*N*sizeof(int) );

	// Copy data to host
	cudaMemcpy( h_n_neigh, d_n_neigh,   N*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_offset,  d_offset,  3*N*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_NEPP,    d_NEPP,    4*N*sizeof(int), cudaMemcpyDeviceToHost );
	
	printf( "\n" );
	for ( int ii = 0; ii < N; ++ii ){
		printf( "Particle = ( %i ), Nneigh = ( %i ), NEPP = ( %3i, %3i, %3i, %3i ), Offset = ( %6i, %6i, %6i ) \n", ii, h_n_neigh[ii], h_NEPP[ii], h_NEPP[N+ii], h_NEPP[2*N+ii], h_NEPP[3*N+ii], h_offset[ii], h_offset[N+ii], h_offset[2*N+ii] );
	}
	printf( "\n" );

	// Clean up
	free( h_n_neigh );
	free( h_offset );
	free( h_NEPP );
}

/*!
	Print out particle positions
*/
void Debug_PrintPos( Scalar4 *d_pos, int N ){

	// Set up host vector
	Scalar4 *h_pos;
	h_pos = (Scalar4 *)malloc( N*sizeof(Scalar4) );

	// Memory copy
	cudaMemcpy( h_pos, d_pos, N*sizeof(Scalar4), cudaMemcpyDeviceToHost );

	// Print out
	printf( "\n" );
	for ( int ii = 0; ii < N; ++ii ){
		printf( "Particle %3i Position = ( %8.3f, %8.3f, %8.3f)\n", ii, h_pos[ii].x, h_pos[ii].y, h_pos[ii].z );
	}
	printf( "\n" );

	// Clean up
	free( h_pos );

}

/*!
	Matrix-vector multiplication

	y = L * x;

*/
__global__ void Debug_L_mult_kernel( float *d_y, float *d_x, int *d_RowPtr, int *d_ColInd, float *d_Val, int group_size, int nnz ){


	// Index for current thread 
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check that thread is within bounds, and only do work if so	
	if ( tidx < group_size ) {

		// Loop over 6 rows associated with each particle
		for ( int ii = 0; ii < 6; ++ii ){

			// Row indexing
			int row = 6 * tidx + ii;
			int start = d_RowPtr[ row ];
			int end   = d_RowPtr[ row + 1 ];

			// Initialize output
			d_y[ row ] = 0.0;
			
			// Loop over columns
			int jj = start;
			int col = d_ColInd[ jj ];
			float val = d_Val[ jj ];
			while ( ( jj < end  ) ){ // && ( row <= col  ) ){

				// Add to output
				d_y[ row ] += val * d_x[ col ];

				// Increment counter
				jj++;
				col = d_ColInd[ jj ];
				val = d_Val[ jj ];

			}

		}

	} // Check for thread in bounds

}

/*!
	Check if array has any NaNs
*/
void Debug_HasNaN( float *d_vec, int N ){

	// Set up host vector
	float *h_vec;
	h_vec = (float *)malloc( N*sizeof(float) );

	// Memory copy
	cudaMemcpy( h_vec, d_vec, N*sizeof(float), cudaMemcpyDeviceToHost );

	// Check for NaN. Comparisons involving NaNs are always false, so use
	// that to check
	int have_nan = 0;
	for ( int ii = 0; ii < N; ++ii ){

		float currval = h_vec[ ii ];

		if ( isnan( currval ) ){

			have_nan = 1;
			break;
		}

	}

	// Print out
	printf( "HasNaN: %i\n", have_nan );

	// Clean up
	free( h_vec );

}

/*!
	Check if matrix has zeros along diagonal
*/
void Debug_HasZeroDiag( float *d_Diag, int N ){

	//
	int numel = 6*N;

	// Set up host vectors
	float *h_Diag;
	h_Diag = (float *)malloc( numel*sizeof(float) );

	// Memory copy
	cudaMemcpy( h_Diag, d_Diag, numel*sizeof(float), cudaMemcpyDeviceToHost );

	// Check for zeros
	int have_zeros = 0;
	for ( int ii = 0; ii < numel; ++ii ){
		
		float currval = h_Diag[ ii ];

		if ( currval == 0 ){

			have_zeros = 1;
			break;
		}

	}

	// Print out
	printf( "HasZero: %i\n", have_zeros );

	// Clean up
	free( h_Diag );

}

/*!
	Check whether CSR matrix has zero diagonal
	(defined in Eq.(2.11) in Fiore & Swan (JFM 2019) /zhoge)
*/
void Debug_CSRzeroDiag( int *d_RowPtr, int *d_ColInd, float *d_Val, int group_size, int nnz ){

	int *h_RowPtr, *h_ColInd;
	float *h_Val;

	//h_RowPtr = (int *) malloc( 6*group_size*sizeof(int) ); 
	h_RowPtr = (int *) malloc( (6*group_size+1)*sizeof(int) );  // should have +1 entries /zhoge
	h_ColInd = (int *) malloc( nnz*sizeof(int) );
	h_Val  = (float *) malloc( nnz*sizeof(float) );

	//cudaMemcpy( h_RowPtr, d_RowPtr, 6*group_size*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_RowPtr, d_RowPtr, (6*group_size+1)*sizeof(int), cudaMemcpyDeviceToHost );  // same as above /zhoge
	cudaMemcpy( h_ColInd, d_ColInd, nnz*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_Val,    d_Val   , nnz*sizeof(float), cudaMemcpyDeviceToHost );

	//printf( "  Number of non-zeros in the CSR matrix = %7i\n", nnz );  //just to check /zhoge
	
	for ( int tidx = 0; tidx < group_size; ++tidx ){

		// Loop over 6 rows associated with each particle
		for ( int ii = 0; ii < 6; ++ii ){

			// Row indexing
			int row = 6 * tidx + ii;
			int start = h_RowPtr[ row ];
			int end   = h_RowPtr[ row + 1 ];

			// Check if row is all sparse
			if ( start == end ){
				// Print out
				printf( "  Zero Row: %i\n", row );
				exit(1);  //zhoge
			}

			// Loop over columns
			int jj = start;
			int col = h_ColInd[ jj ];
			float val = h_Val[ jj ];
			
			for( jj = start; jj < end; ++jj ){

				col = h_ColInd[ jj ];
				val = h_Val[ jj ];
				
				if ( ( row == col ) && ( val == 0.0 ) ){
					printf( "  Zero Diagonal: %i\n", row );
					exit(1);  //zhoge
				}

			}

		}

	} 

	free( h_RowPtr );
	free( h_ColInd );
	free( h_Val );

}

/*!
	Check the output of the Saddle solve for a pair of particles
*/
void Debug_Lattice_SpinViscosity( 
					MobilityData *mob_data,
					ResistanceData *res_data,
					KernelData *ker_data,
					WorkData *work_data,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size,
					const BoxDim box
					){

	//// Grid information
	//dim3 grid = ker_data->particle_grid;
	//dim3 threads = ker_data->particle_threads;
	
	//
	int numel = 6 * group_size;

	// Initialize input and output vector 
	float *h_x, *d_x;
	h_x = (float *)malloc( 17*group_size*sizeof(float) );
	cudaMalloc( (void**)&d_x, 17*group_size*sizeof(float) );
	
	float *h_y, *d_y;
	h_y = (float *)malloc( numel*sizeof(float) );
	cudaMalloc( (void**)&d_y, 17*group_size*sizeof(float) );

	for ( int ii = 0; ii < 17*group_size; ++ii ){
		h_x[ii] = 0.0;
	}
	for ( int ii = 0; ii < group_size; ++ii ){
		h_x[ 11*group_size + 6*ii + 4 ] = 1.0;
	}

	//
	cudaMemcpy( d_x, h_x, 17*group_size*sizeof(float), cudaMemcpyHostToDevice );

        void *pBuffer;
        cudaMalloc( (void**)&pBuffer, res_data->pBufferSize );

	//
	Solvers_Saddle(
			d_x,  // input 
			d_y,  // output
			d_pos,
			d_group_members,
			group_size,
			box,
			0.001,
			pBuffer,
			ker_data,
			mob_data,
			res_data,
			work_data
			);

	//
	cudaMemcpy( h_y, &d_y[11*group_size], numel*sizeof(float), cudaMemcpyDeviceToHost );

	//
	Scalar3 L = box.getL();
	float phi = 4 * 3.1415926536 / 3 * float(group_size) / (L.x*L.y*L.z);
	float spin_viscosity = 0.0;
	for ( int ii = 0; ii < group_size; ++ii ){
		spin_viscosity += ( 6.0 * phi ) * ( 3.0 / 4.0 ) * (-1.0 / h_y[ 6*ii + 4 ] );
	}
	spin_viscosity /= float( group_size );
	
	printf( "\n\nSPINVISCOSITY: %8.6f\n\n\n", spin_viscosity );

	// Clean up
	cudaFree( d_x );
	cudaFree( d_y );
	cudaFree( pBuffer );

	free( h_x );
	free( h_y );

}

/*!
	Check the output of the Saddle solve for a pair of particles
*/
void Debug_Lattice_ShearViscosity( 
					MobilityData *mob_data,
					ResistanceData *res_data,
					KernelData *ker_data,
					WorkData *work_data,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size,
					const BoxDim box
					){

	// Grid information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	////
	//int numel = 6 * group_size;

	// Initialize input and output vector 
	float *h_x, *d_x;
	h_x = (float *)malloc( 17*group_size*sizeof(float) );
	cudaMalloc( (void**)&d_x, 17*group_size*sizeof(float) );
	
	float *h_y, *d_y;
	h_y = (float *)malloc( 5*group_size*sizeof(float) );
	cudaMalloc( (void**)&d_y, 17*group_size*sizeof(float) );
        
	void *pBuffer;
        cudaMalloc( (void**)&pBuffer, res_data->pBufferSize );

	for ( int ii = 0; ii < 17*group_size; ++ii ){
		h_x[ii] = 0.0;
	}

	float alpha = 0.0;
	float beta = 0.0;
	for ( int jj = 0; jj < 2; ++jj ){

		float E[5] = {0.0};
		float P[5] = {0.0};
		E[ jj ] = 1.0;
		P[0] = 2 * E[0] + E[4];
		P[1] = 2 * E[1];
		P[2] = 2 * E[2];
		P[3] = 2 * E[3];
		P[4] = 2 * E[4] + E[0];
	
		for ( int ii = 0; ii < group_size; ++ii ){
			for ( int kk = 0; kk < 5; ++kk ){
				h_x[ 6*group_size + 5*ii + kk ] = P[kk];
			}
		}

		//
		cudaMemcpy( d_x, h_x, 17*group_size*sizeof(float), cudaMemcpyHostToDevice );
		
		//
		Solvers_Saddle(
				d_x,  // input 
				d_y,  // output
				d_pos,
				d_group_members,
				group_size,
				box,
				0.001,
				pBuffer,
				ker_data,
				mob_data,
				res_data,
				work_data
				);

		Lubrication_RSEgeneral_kernel<<< grid, threads >>>(
								&d_y[6*group_size], // output
								&d_x[6*group_size], // input
								group_size, 
								d_group_members,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist, 
								d_pos,
			      					box,
								res_data->table_dist,
								res_data->table_vals,
								res_data->table_min,
								res_data->table_dr
								);

		//
		cudaMemcpy( h_y, &d_y[6*group_size], 5*group_size*sizeof(float), cudaMemcpyDeviceToHost );

		//
		Scalar3 L = box.getL();
		float phi = 4 * 3.1415926536 / 3 * float(group_size) / (L.x*L.y*L.z);
		float shear_viscosity = 0.0;
		for ( int ii = 0; ii < group_size; ++ii ){
			shear_viscosity += ( 5.0 / 2.0 * phi ) * ( h_y[ 5*ii + jj ] );
		}
		shear_viscosity /= float( group_size );

		if ( jj == 0 ){
			alpha = 1.0 + shear_viscosity;
		}
		else if ( jj == 1 ){
			beta = 1.0 + shear_viscosity;
		}

	}	

	printf( "\n\nALPHA: %8.6f\nBETA: %8.6f\n\n", alpha, beta );

	// Clean up
	cudaFree( d_x );
	cudaFree( d_y );
	cudaFree( pBuffer );

	free( h_x );
	free( h_y );

}

//    /*!
//    	Compute short-time self-diffusion coefficient from the trace of the resistance tensor < Psi * RFU * Psi >
//    */
//    void Debug_Random_Dss1( 
//    			ResistanceData *res_data,
//    			KernelData *ker_data,
//    			BrownianData *bro_data,
//    			MobilityData *mob_data,
//    			Scalar4 *d_pos,
//    			unsigned int *d_group_members,
//    			int group_size,
//    			int3 *d_image,
//    			const BoxDim box,
//    			float dt
//    			){
//    
//    	// Number of particles
//    	int N = group_size;
//    
//    	// Kernel info
//    	dim3 grid = ker_data->particle_grid;
//    	dim3 threads = ker_data->particle_threads;
//    
//    	// Precondition stuff
//    	Precondition_Wrap(
//    				d_pos,
//    				d_group_members,
//    				group_size,
//    				box,
//    				ker_data,
//    				res_data
//    				);
//    
//    
//    	// Initialize input and output vector 
//    	float *h_x, *d_x;
//    	h_x = (float *)malloc( 17*N*sizeof(float) );
//    	cudaMalloc( (void**)&d_x, 17*N*sizeof(float) );
//    	
//    	float *h_y, *d_y;
//    	h_y = (float *)malloc( (17*N)*sizeof(float) );
//    	cudaMalloc( (void**)&d_y, (17*N)*sizeof(float) );
//    
//    	for ( int ii = 0; ii < 17*N; ++ii ){
//    		h_x[ii] = 0.0;
//    		h_y[ii] = 0.0;
//    	}
//    
//    	//
//    	int seed = bro_data->seed_nf;
//    	float T = 1.0;
//    
//    	//
//    	int nrepeats = 10;
//    	float Dss = 0.0;
//    	srand( 1 );
//    	for ( int nn = 0; nn < nrepeats; ++nn ){
//    
//    		//for ( int ii = 0; ii < N; ++ii ){
//    		//	h_x[ 11*N + 6*ii + 0 ] = sqrtf(3.0) * ( -1.0 + 2.0 * (float)( (double)rand() / (double)RAND_MAX ) );
//    		//	h_x[ 11*N + 6*ii + 1 ] = sqrtf(3.0) * ( -1.0 + 2.0 * (float)( (double)rand() / (double)RAND_MAX ) );
//    		//	h_x[ 11*N + 6*ii + 2 ] = sqrtf(3.0) * ( -1.0 + 2.0 * (float)( (double)rand() / (double)RAND_MAX ) );
//    		//}
//    		Brownian_NearField_RNG_kernel<<< grid, threads >>>(
//    									&d_x[11*N],
//    									group_size,
//    									seed,
//    									T,
//    									dt
//    									);
//    		seed++;
//    		cudaMemcpy( h_x, d_x, 17*N*sizeof(float), cudaMemcpyDeviceToHost );
//    		for ( int ii = 0; ii < N; ++ii ){
//    			h_x[ 11*N + 6*ii + 0 ] *= sqrtf( dt / 2.0 );
//    			h_x[ 11*N + 6*ii + 1 ] *= sqrtf( dt / 2.0 );
//    			h_x[ 11*N + 6*ii + 2 ] *= sqrtf( dt / 2.0 );
//    			h_x[ 11*N + 6*ii + 3 ] *= 0.0;
//    			h_x[ 11*N + 6*ii + 4 ] *= 0.0;
//    			h_x[ 11*N + 6*ii + 5 ] *= 0.0;
//    		}
//    	
//    			
//    		// Copy to device
//    		cudaMemcpy( d_x, h_x, 17*N*sizeof(float), cudaMemcpyHostToDevice );
//    		cudaMemcpy( d_y, h_y, 17*N*sizeof(float), cudaMemcpyHostToDevice );
//    
//    		// Compute the velocity		
//    		Solvers_Saddle(
//    				d_x,  // input 
//    				d_y,  // output
//    				d_pos,
//    				d_group_members,
//    				group_size,
//    				box,
//    				0.001,
//    				ker_data,
//    				mob_data,
//    				res_data
//    				);
//    		
//    		// Copy to host
//    		cudaMemcpy( h_y, d_y, 17*N*sizeof(float), cudaMemcpyDeviceToHost );
//    
//    		// Compute the dot product
//    		for ( int ii = 0; ii < N; ++ii ){
//    
//    			Dss += ( h_y[ 11*N + 6*ii + 0 ] * h_x[ 11*N + 6*ii + 0 ] ) * ( -1.0 / ( 3.0 * float(N) ) );
//    			Dss += ( h_y[ 11*N + 6*ii + 1 ] * h_x[ 11*N + 6*ii + 1 ] ) * ( -1.0 / ( 3.0 * float(N) ) );
//    			Dss += ( h_y[ 11*N + 6*ii + 2 ] * h_x[ 11*N + 6*ii + 2 ] ) * ( -1.0 / ( 3.0 * float(N) ) );
//    
//    		}
//    
//    	}
//    
//    	Dss /= float( nrepeats );
//    
//    	printf( "\n\nDss1 = %8.6f\n\n\n", Dss );
//    	
//    	// Clean up
//    	cudaFree( d_x );
//    	cudaFree( d_y );
//    	
//    	free( h_x );
//    	free( h_y );
//    
//    }
//    
//    /*!
//    	Compute short-time self-diffusion coefficient the mean of the Brownian Displacements <UB UB>
//    */
//    void Debug_Random_Dss2( 
//    			ResistanceData *res_data,
//    			KernelData *ker_data,
//    			BrownianData *bro_data,
//    			MobilityData *mob_data,
//    			Scalar4 *d_pos,
//    			unsigned int *d_group_members,
//    			int group_size,
//    			int3 *d_image,
//    			const BoxDim box,
//    			float dt
//    			){
//    
//    	// Number of particles
//    	int N = group_size;
//    
//    	// Precondition stuff
//    	Precondition_Wrap(
//    				d_pos,
//    				d_group_members,
//    				group_size,
//    				box,
//    				ker_data,
//    				res_data
//    				);
//    
//    
//    	// Initialize input and output vector 
//    	float *h_x, *d_x;
//    	h_x = (float *)malloc( 6*N*sizeof(float) );
//    	cudaMalloc( (void**)&d_x, 6*N*sizeof(float) );
//    	
//    	float *h_y, *d_y;
//    	h_y = (float *)malloc( (6*N)*sizeof(float) );
//    	cudaMalloc( (void**)&d_y, (6*N)*sizeof(float) );
//    
//    	for ( int ii = 0; ii < 6*N; ++ii ){
//    		h_x[ii] = 0.0;
//    		h_y[ii] = 0.0;
//    	}
//    
//    	// Copy to device
//    	cudaMemcpy( d_x, h_x, 6*N*sizeof(float), cudaMemcpyHostToDevice );
//    	cudaMemcpy( d_y, h_y, 6*N*sizeof(float), cudaMemcpyHostToDevice );
//    
//    	int nrepeats = 100;
//    	float Dss = 0.0;
//    	srand( 1 );
//    	for ( int nn = 0; nn < nrepeats; ++nn ){
//    
//    		// Randomize seeds for stochastic calculations
//    		srand( nn );
//    		( bro_data->seed_ff_rs ) = rand();
//    		( bro_data->seed_ff_ws ) = rand();
//    		( bro_data->seed_nf    ) = rand();
//    		( bro_data->seed_rfd   ) = rand();
//    
//    		// Compute the velocity		
//    		Integrator_ComputeVelocity(
//    						d_x, // input
//    						d_y, // output
//    						dt,
//    						d_pos,
//    						d_image,
//    						d_group_members,
//    						group_size,
//    						box,
//    						ker_data,
//    						bro_data,
//    						mob_data,
//    						res_data
//    						);
//    		
//    		// Copy to host
//    		cudaMemcpy( h_y, d_y, 6*N*sizeof(float), cudaMemcpyDeviceToHost );
//    
//    		// Compute the dot product
//    		for ( int ii = 0; ii < N; ++ii ){
//    
//    			Dss += ( h_y[ 6*ii + 0 ] * h_y[ 6*ii + 0 ] ) * ( dt / ( 3.0 * float(N) * 2.0 ) );
//    			Dss += ( h_y[ 6*ii + 1 ] * h_y[ 6*ii + 1 ] ) * ( dt / ( 3.0 * float(N) * 2.0 ) );
//    			Dss += ( h_y[ 6*ii + 2 ] * h_y[ 6*ii + 2 ] ) * ( dt / ( 3.0 * float(N) * 2.0 ) );
//    
//    		}
//    		
//    	}
//    
//    	Dss /= float( nrepeats );
//    
//    	printf( "\n\nDss2 = %8.6f\n\n\n", Dss );
//    	
//    	// Clean up
//    	cudaFree( d_x );
//    	cudaFree( d_y );
//    	
//    	free( h_x );
//    	free( h_y );
//    
//    }

