// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Helper_Precondition.cuh"

#include <stdio.h>

#include <cusparse.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! \file Helper_Precondition.cu

	Helper functions for building, permuting, and applying the preconditioners for the near-field
	Brownian iterative calculation and the saddle point solve

*/


/*!
	Zero a vector of a given length which is an integer multiple of the number
	of particles
 
	d_b	(input/output)	vector to be zeroed (zero upon output)
   	N 	(input)  	N number of particles
	stride	(input)  	Vector length multiple (length = stride * N)
*/
__global__ void Precondition_ZeroVector_kernel(
						float *d_b,
						const unsigned int nnz,
						const unsigned int group_size
						){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tid < group_size ) {
	
		// Stride for the zero-ing (how much work per thread)
		int stride = int( nnz / group_size ) + 1; 
	
		// Set up the index
		int ii = 0;
		int ind = stride * tid + ii;

		// Do the zeroing
		while ( ii < stride && ind < nnz ){
			d_b[ ind ] = 0.0;
			ii++;
			ind++;
		}
	
	}
}


/*!
        Direct addition of two int arrays

        C = a*A + b*B
        C can be A or B, so that A or B will be overwritten

        d_a		(input)  input vector, A
        d_b		(input)  input vector, B
        d_c		(output) output vector, C
        coeff_a		(input)  scaling factor for A, a
        coeff_b		(input)  scaling factor for B, b
        group_size	(input)  length of vectors
*/
__global__ void Precondition_AddInt_kernel( 	unsigned int *d_a, 
						unsigned int *d_b,
						unsigned int *d_c,
						int coeff_a,
						int coeff_b,
						unsigned int group_size 
						){

        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
	if (idx < group_size) {
       
		d_c[ idx ] = coeff_a * d_a[idx] + coeff_b * d_b[idx];
 
        }
}


/*
	Kernel Function to apply/undo reordering on a vector, given a permutation array

	After calling this kernel, call 
		cudaMemcpy( d_Vector, d_Scratch_Vector, length*sizeof(float), cudaMemcpyDeviceToDevice );
	to finish.
	
	d_Scratch_Vector	(output) Output vector (must be different from input)
	d_Vector		(input)  Input vector
	d_prcm			(input)  permutation list
	length	 		(input)  number of elements in the vector
	direction		(input)  Whether to apply or undo the RCM permutation. Must be 1 or -1

*/
__global__ void Precondition_ApplyRCM_Vector_kernel( 
							float *d_Scratch_Vector,
							float *d_Vector,
							const int *d_prcm,
							const int length,
							const int direction
							){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tid < length ) {

		// Apply the permutation
		if ( direction == 1 ){
			// Forward direction	
			d_Scratch_Vector[ 6*tid     ] = d_Vector[ d_prcm[ 6*tid     ] ];
			d_Scratch_Vector[ 6*tid + 1 ] = d_Vector[ d_prcm[ 6*tid + 1 ] ];
			d_Scratch_Vector[ 6*tid + 2 ] = d_Vector[ d_prcm[ 6*tid + 2 ] ];
			d_Scratch_Vector[ 6*tid + 3 ] = d_Vector[ d_prcm[ 6*tid + 3 ] ];
			d_Scratch_Vector[ 6*tid + 4 ] = d_Vector[ d_prcm[ 6*tid + 4 ] ];
			d_Scratch_Vector[ 6*tid + 5 ] = d_Vector[ d_prcm[ 6*tid + 5 ] ];
		}
		else if ( direction == -1 ){
			// Reverse direction
			d_Scratch_Vector[ d_prcm[ 6*tid     ] ] = d_Vector[ 6*tid     ];
			d_Scratch_Vector[ d_prcm[ 6*tid + 1 ] ] = d_Vector[ 6*tid + 1 ];
			d_Scratch_Vector[ d_prcm[ 6*tid + 2 ] ] = d_Vector[ 6*tid + 2 ];
			d_Scratch_Vector[ d_prcm[ 6*tid + 3 ] ] = d_Vector[ 6*tid + 3 ];
			d_Scratch_Vector[ d_prcm[ 6*tid + 4 ] ] = d_Vector[ 6*tid + 4 ];
			d_Scratch_Vector[ d_prcm[ 6*tid + 5 ] ] = d_Vector[ 6*tid + 5 ];
		}

	}
}
	
/*
	Kernel Function to add identity to the lubrication tensor

	d_L_Val		(input/output)  CSR values
	d_L_RowPtr	(input)  	CSR row pointers
	d_L_ColInd	(input)  	CSR column indices
	group_size	(input)  	number of particles
	ichol_relaxer	(input)  	relaxation factor for the incomplete Cholesky decomposition

*/
__global__ void Precondition_AddIdentity_kernel(
						float *d_L_Val,
						int   *d_L_RowPtr,
						int   *d_L_ColInd, 
						int   group_size,
						float ichol_relaxer
						){


	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
	if ( tid < group_size ){

		// There are six rows associated with each particles
		for ( int ii = 0; ii < 6; ++ ii ){

			// Current row
			int row = 6 * tid + ii;

			// Pointers for column indices
			int start = d_L_RowPtr[ row ];
			int end = d_L_RowPtr[ row + 1 ];

			// Loop over columns to find the diagonal
			for ( int col_ind = start; col_ind < end; col_ind++ ){
				
				// Current column
				int col = d_L_ColInd[ col_ind ];

				// If have diagonal, add identity and break
				//
				// Because all values are made dimensionless on 6*pi*eta*a,
				// the diagonal elements for FU are 1, but those for LW
				// are 4/3
				if ( col == row ){
					d_L_Val[ col_ind ] += ichol_relaxer * ( ( ii < 3 ) ? 1.0 : 1.33333333 );
					break;
				}

			}

		}

	}

}

/*
	Apply Inn to a vector

	Inn is a modified identity tensor whose diagonal entries are 1 if the particle
	has no neighbors and zero if it has neighbors

	y = y + Inn * x;

	d_y		(input/output)	output
	d_x		(input)		input
	d_HasNeigh	(input)		Which particles have neighbors
	group_size	(input)		number of particles

*/		
__global__ void Precondition_Inn_kernel(
					Scalar *d_y,
					Scalar *d_x,
					int *d_HasNeigh,
					int group_size
					){
	
	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
	if ( tid < group_size ){

		// There are 6 entries per particle
		//
		// Because all values are made dimensionless on 6*pi*eta*a,
		// the diagonal elements for FU are 1, but those for LW
		// are 4/3
		if ( d_HasNeigh[ tid ] == 0 ){
			d_y[ 6*tid     ] +=              d_x[ 6 * tid     ];
			d_y[ 6*tid + 1 ] +=              d_x[ 6 * tid + 1 ];
			d_y[ 6*tid + 2 ] +=              d_x[ 6 * tid + 2 ];
			d_y[ 6*tid + 3 ] += 1.33333333 * d_x[ 6 * tid + 3 ];
			d_y[ 6*tid + 4 ] += 1.33333333 * d_x[ 6 * tid + 4 ];
			d_y[ 6*tid + 5 ] += 1.33333333 * d_x[ 6 * tid + 5 ];	
		}

	}

}

/*

	Apply (I-Inn) to a vector 

	Inn is a modified identity tensor whose diagonal entries are 1 if the particle
	has no neighbors and zero if it has neighbors

	y = ( I - Inn ) * x;

	!!! Can work in place if needed

	d_y		(output) output
	d_x		(input)  input
	d_HasNeigh	(input)  Which particles have neighbors
	group_size	(input)  number of particles

*/		
__global__ void Precondition_ImInn_kernel(
						Scalar *d_y, // output
						Scalar *d_x, // input
						int *d_HasNeigh,
						int group_size
						){
	
	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
	if ( tid < group_size ){

		// There are 6 entries per particle
		if ( d_HasNeigh[ tid ] == 0 ){
			d_y[ 6*tid     ] = 0.0;
			d_y[ 6*tid + 1 ] = 0.0;
			d_y[ 6*tid + 2 ] = 0.0;
			d_y[ 6*tid + 3 ] = 0.0;
			d_y[ 6*tid + 4 ] = 0.0;
			d_y[ 6*tid + 5 ] = 0.0;	
		}
		else {
			d_y[ 6*tid     ] = d_x[ 6*tid     ];
			d_y[ 6*tid + 1 ] = d_x[ 6*tid + 1 ];
			d_y[ 6*tid + 2 ] = d_x[ 6*tid + 2 ];
			d_y[ 6*tid + 3 ] = d_x[ 6*tid + 3 ];
			d_y[ 6*tid + 4 ] = d_x[ 6*tid + 4 ];
			d_y[ 6*tid + 5 ] = d_x[ 6*tid + 5 ];	
		}

	}

}


/*
	Kernel to expand RCM re-ordering from particle based to full
	index-based. To improve efficiency, the RCM is computed on
	the adjacency given by the neighborlists. This function expands
	that re-ordering to match the full 6Nx6N resistance tensor.

	d_prcm		(output) reordering vector in matrix space
	d_scratch	(input)  reordering vector in particles space
	group_size	(input)  number of particles

*/		
__global__ void Precondition_ExpandPRCM_kernel(
						int *d_prcm,
						int *d_scratch,
						int group_size
						){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is in bounds
	if ( tid < group_size ){

		d_prcm[ 6*tid + 0 ] = 6 * d_scratch[ tid ] + 0;
		d_prcm[ 6*tid + 1 ] = 6 * d_scratch[ tid ] + 1;
		d_prcm[ 6*tid + 2 ] = 6 * d_scratch[ tid ] + 2;
		d_prcm[ 6*tid + 3 ] = 6 * d_scratch[ tid ] + 3;
		d_prcm[ 6*tid + 4 ] = 6 * d_scratch[ tid ] + 4;
		d_prcm[ 6*tid + 5 ] = 6 * d_scratch[ tid ] + 5;

	}

}


/*

	Kernel to apply initialize the MAP for RCM re-ordering of CSR values

	d_map	(output) list of numbers from 0:(nnz-1)
	nnz	(input)  number of non-zero elements of the matrix

*/		
__global__ void Precondition_InitializeMap_kernel(
							int *d_map,
							int nnz
							){
	
	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
	if ( tid < nnz ){

		d_map[ tid ] = tid;

	}

}


/*

	Kernel to apply the re-ordering map the CSR values

	*** After calling this kernel, have to copy d_Scratch to d_Val to finish

	d_Scratch	(output) re-ordered values
	d_Val		(input)  CSR values for RFU
	d_map		(input)  map to re-order d_Val
	nnz		(input)  number of non-zero element of the matrix

*/
__global__ void Precondition_Map_kernel(
					float *d_Scratch,
					float *d_Val,
					int *d_map,
					int nnz
					){
	
	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
	if ( tid < nnz ){

		d_Scratch[ tid ] = d_Val[ d_map[ tid ] ];
		
	}

}

/*! 
	Build the diagonal preconditioner for the Brownian calculation 

        zhoge: Note, the output is (diag)^(-1/2), where diag is the diagonal element of the reordered RFU (see Precondition_Wrap)

	group_size	(input)  number of particles
	d_Diag		(output) elements of the diagonal preconditioner  
	d_L_RowPtr	(input)  CSR row pointer for RFU
	d_L_ColInd	(input)  CSR col indices for RFU
	d_L_Val		(input)  CSR values for RFU

*/
__global__ void Precondition_GetDiags_kernel(
						int group_size, 
						float *d_Diag,
						int   *d_L_RowPtr,
						int   *d_L_ColInd,
						float *d_L_Val
						){

	// Index for current thread 
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check that thread is within bounds, and only do work if so	
	if ( tidx < group_size ){

		// Index into array for rows associated with the current
		// particle
		int offset = 6 * tidx;

		// Get the diagonal element for each row associated with the
		// current particle
		float d = 1.0;
		for ( int ii = 0; ii < 6; ++ii ){

			// Current row
			int row = offset + ii;

			// Indices into Column index array
			int rowstart = d_L_RowPtr[ row ];
			int rowend   = d_L_RowPtr[ row + 1 ];

			// Loop over columns to find the diagonal
			for ( int jj = rowstart; jj < rowend; ++jj ){
			
				// Get the current column	
				int col = d_L_ColInd[ jj ];

				// Check if found diagonal
				if ( col == row ){
					
					// Get diagonal element
					d = d_L_Val[ jj ];
					
					// Diagonal preconditioner
					if ( d >= 1.0 || d == 0.0 ){
						d = 1.0;
					}
					else {
						d = sqrtf( 1.0 / d );
					}
					
					break;

				}
			}

			// Write the output
			d_Diag[ row ] = d;

		}

	} // Check for thread in bounds

}


/*! 
	Apply the diagonal preconditioner to a vector.
	
	Support in-place computation. 

	d_y		(output) output vector
	d_x		(input)  input vector
	group_size	(input)  number of particles
	d_Diag		(input)  elements of diagonal preconditioner
	direction	(input)  direction of operation, forward or reverse (must be 1 or -1)

*/
__global__ void Precondition_DiagMult_kernel(
						float *d_y, // output
						float *d_x, // input
						int group_size, 
						float *d_Diag,  //input
						int direction
						){

	// Index for current thread 
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check that thread is within bounds, and only do work if so	
	if ( tidx < group_size ) {

		// Thread per particle, each thread does work for the 6 rows 
		// associated with that particle.
		//
		// Explicitly unroll the work. (zhoge: Since d_Diag is (diag)^(-1/2), multiply actually means divide)
		if ( direction == 1 ){
			d_y[ 6*tidx + 0 ] = d_x[ 6*tidx + 0 ] * d_Diag[ 6*tidx + 0 ];
			d_y[ 6*tidx + 1 ] = d_x[ 6*tidx + 1 ] * d_Diag[ 6*tidx + 1 ];
			d_y[ 6*tidx + 2 ] = d_x[ 6*tidx + 2 ] * d_Diag[ 6*tidx + 2 ];
			d_y[ 6*tidx + 3 ] = d_x[ 6*tidx + 3 ] * d_Diag[ 6*tidx + 3 ];
			d_y[ 6*tidx + 4 ] = d_x[ 6*tidx + 4 ] * d_Diag[ 6*tidx + 4 ];
			d_y[ 6*tidx + 5 ] = d_x[ 6*tidx + 5 ] * d_Diag[ 6*tidx + 5 ];
		}					      
		else if ( direction == -1 ){		      
			d_y[ 6*tidx + 0 ] = d_x[ 6*tidx + 0 ] / d_Diag[ 6*tidx + 0 ];
			d_y[ 6*tidx + 1 ] = d_x[ 6*tidx + 1 ] / d_Diag[ 6*tidx + 1 ];
			d_y[ 6*tidx + 2 ] = d_x[ 6*tidx + 2 ] / d_Diag[ 6*tidx + 2 ];
			d_y[ 6*tidx + 3 ] = d_x[ 6*tidx + 3 ] / d_Diag[ 6*tidx + 3 ];
			d_y[ 6*tidx + 4 ] = d_x[ 6*tidx + 4 ] / d_Diag[ 6*tidx + 4 ];
			d_y[ 6*tidx + 5 ] = d_x[ 6*tidx + 5 ] / d_Diag[ 6*tidx + 5 ];
		}

	} // Check for thread in bounds

}

/*!
	Zeroes the elements associated with the upper triangular portion
	(excluding the diagonal) of the Incomplete Cholesky factor. Have
	to do this because cuSPARSE is inconsistent in how it handles
	those parts, and doesn't seem to always ignore it properly.

	d_RowPtr	(input) 	CSR row pointer for lower Cholesky factor
	d_ColInd	(input) 	CSR col indices for lower Cholesky factor
	d_Val		(input/output)	CSR values for lower Cholesky factor
	group_size	(input)		number of particles

*/
__global__ void Precondition_ZeroUpperTriangle_kernel( 
							int   *d_RowPtr,
							int   *d_ColInd,
							float *d_Val,
							int group_size
							){


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

			// Loop over columns for the current row
			for ( int jj = start; jj < end; ++jj ){

				int col = d_ColInd[ jj ];
				
				if ( col > row ){
					d_Val[ jj ] = 0.0;
				} 

			} // loop over columns

		} // loop over 6 rows for current particle

	} // Check for thread in bounds

}

/*!
	Matrix-vector multiplication of the Lower Cholesky Factor

	y = L * x;

	Does NOT support in-place computation.

	!!! Deprecated -- was used to test, and not required for functional code

*/
__global__ void Precondition_Lmult_kernel( 
						float *d_y, // output
						float *d_x, // input
						int   *d_RowPtr,
						int   *d_ColInd,
						float *d_Val,
						int group_size
						){


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
			float output = 0.0;
			
			// Loop over columns
			int jj = start;
			int col = d_ColInd[ jj ];
			float val = d_Val[ jj ];
			while ( ( jj < end  ) && ( col <= row  ) ){

				// Add to output
				output += val * d_x[ col ];

				// Increment counter
				jj++;
				col = d_ColInd[ jj ];
				val = d_Val[ jj ];

			}

			// Write out
			d_y[ row ] = output;

		}

	} // Check for thread in bounds

}
