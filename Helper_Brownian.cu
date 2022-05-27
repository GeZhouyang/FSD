// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Helper_Brownian.cuh"

#include <stdio.h>

// LAPACK and CBLAS
#include "lapacke.h"
#include "cblas.h"

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


/*! \file Helper_Brownian.cu
    	\brief Helper functions to perform additions, dot products, etc., needed 
		in the Brownian calculations
*/

//! Shared memory array for partial sum of dot product kernel
extern __shared__ Scalar partial_sum[];

/*!
	Dot product helper function: First step
	d_a .* d_b -> d_c -> Partial sum
	BlockDim of this kernel should be 2^n, which is 512. (Based on HOOMD ComputeThermoGPU class)
	
	d_a			(input)  first vector in dot product
	d_b			(input)  second vector in dot product
	dot_sum			(output) partial dot product sum
	group_size		(input)  length of vectors a and b
        d_group_members		(input)  index into vectors
*/
__global__ void Brownian_FarField_Dot1of2_kernel(
							Scalar4 *d_a, 
							Scalar4 *d_b, 
							Scalar *dot_sum, 
							unsigned int group_size, 
							unsigned int *d_group_members
							){
	
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	Scalar temp;

	if ( idx < group_size ) {

		Scalar4 a = d_a[idx];
		Scalar4 b = d_b[idx];
		
		temp = a.x*b.x + a.y*b.y + a.z*b.z;

		a = d_a[group_size+2*idx];
		b = d_b[group_size+2*idx];

		temp += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; // Partial sum, each thread, shared memory

		a = d_a[group_size+2*idx+1];
		b = d_b[group_size+2*idx+1];

		temp += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; // Partial sum, each thread, shared memory


	}
	else {
		temp = 0;
	}

	partial_sum[threadIdx.x] = temp;

	__syncthreads();

	int offs = blockDim.x >> 1;

	while (offs > 0)
        {
        	if (threadIdx.x < offs)
            	{
            		partial_sum[threadIdx.x] += partial_sum[threadIdx.x + offs];
            	}
        	offs >>= 1;
        	__syncthreads();
        }

	if (threadIdx.x == 0){
		dot_sum[blockIdx.x] = partial_sum[0];
	}
}


/*!
	Dot product helper function: Second step
	Partial sum -> Final sum
	Only one block will be launched for this step

	dot_sum			(input/output)	partial sum from first dot product kernel
	num_partial_sums	(input) 	length of dot_sum array

*/
__global__ void Brownian_FarField_Dot2of2_kernel(
							Scalar *dot_sum, 
							unsigned int num_partial_sums
							){

	partial_sum[threadIdx.x] = 0.0;
	__syncthreads();
	for (unsigned int start = 0; start < num_partial_sums; start += blockDim.x)
       	{
        	if (start + threadIdx.x < num_partial_sums)
            	{
            		partial_sum[threadIdx.x] += dot_sum[start + threadIdx.x];
            	}
	}

	int offs = blockDim.x >> 1;
	while (offs > 0)
       	{
		__syncthreads();
            	if (threadIdx.x < offs)
                {
                	partial_sum[threadIdx.x] += partial_sum[threadIdx.x + offs];
                }
            	offs >>= 1;
            	
        }
	__syncthreads();
        if (threadIdx.x == 0)
	{
            	dot_sum[0] = partial_sum[0]; // Save the dot product to the first element of dot_sum array
	}

}

/*!

	Perform matrix-vector multiply needed for the Lanczos contribution to the Brownian velocity with FTS

		b = A * x

	d_A		(input)  matrix, N x m
	d_x		(input)  multiplying vector, m x 1
	d_b		(output) result vector, A*x, m x 1
	group_size	(input)  number of particles
	m		(input)  number of iterations ( number of columns of A, length of x )

*/

__global__ void Brownian_FarField_LanczosMatrixMultiply_kernel(
								Scalar4 *d_A, 
								Scalar *d_x, 
								Scalar4 *d_b, 
								unsigned int group_size, 
								int m
								){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < group_size) {

                Scalar3 tempprod0 = make_scalar3( 0.0, 0.0, 0.0 );
                Scalar4 tempprod1 = make_scalar4( 0.0, 0.0, 0.0, 0.0 );
                Scalar4 tempprod2 = make_scalar4( 0.0, 0.0, 0.0, 0.0 );

                // Velocity Part
                for ( int ii = 0; ii < m; ++ii ){

                    Scalar4 matidx = d_A[ idx + ii*3*group_size ];

                    Scalar xcurr = d_x[ii];

                    tempprod0.x = tempprod0.x + matidx.x * xcurr;
                    tempprod0.y = tempprod0.y + matidx.y * xcurr;
                    tempprod0.z = tempprod0.z + matidx.z * xcurr;

                }

                d_b[idx] = make_scalar4( tempprod0.x, tempprod0.y, tempprod0.z, d_A[idx].w );

                // Delu Part
                for ( int ii = 0; ii < m; ++ii ){

                    Scalar4 matidx1 = d_A[ group_size+2*idx   + ii*3*group_size ];
                    Scalar4 matidx2 = d_A[ group_size+2*idx+1 + ii*3*group_size ];

                    Scalar xcurr = d_x[ii];

                    tempprod1.x = tempprod1.x + matidx1.x * xcurr;
                    tempprod1.y = tempprod1.y + matidx1.y * xcurr;
                    tempprod1.z = tempprod1.z + matidx1.z * xcurr;
                    tempprod1.w = tempprod1.w + matidx1.w * xcurr;

                    tempprod2.x = tempprod2.x + matidx2.x * xcurr;
                    tempprod2.y = tempprod2.y + matidx2.y * xcurr;
                    tempprod2.z = tempprod2.z + matidx2.z * xcurr;
                    tempprod2.w = tempprod2.w + matidx2.w * xcurr;

                }

                d_b[group_size+2*idx]   = make_scalar4( tempprod1.x, tempprod1.y, tempprod1.z, tempprod1.w );
                d_b[group_size+2*idx+1] = make_scalar4( tempprod2.x, tempprod2.y, tempprod2.z, tempprod2.w );

        }
}

/*!

	Perform matrix-vector multiply needed for the Lanczos iteration of near-field Brownian force

		b = A * x

	d_A 		(input)  matrix, N x m
	d_x		(input)  multiplying vector, m x 1
	d_b		(output) result vector, A*x, m x 1
	group_size	(input)  number of particles
	numel		(input)  number of elements in a vector
	m		(input)  number of iterations ( number of columns of A, length of x )

*/

__global__ void Brownian_NearField_LanczosMatrixMultiply_kernel(
								Scalar *d_A,
								Scalar *d_x,
								Scalar *d_b,
								unsigned int group_size,
								int numel,
								int m
								){
        // Thread index
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
       
	// Check that thread is in bounds, and do work if so
	if (idx < group_size) {

		// Initialize output to zero
		d_b[ 6*idx + 0 ] = 0.0;
		d_b[ 6*idx + 1 ] = 0.0;
		d_b[ 6*idx + 2 ] = 0.0;
		d_b[ 6*idx + 3 ] = 0.0;
		d_b[ 6*idx + 4 ] = 0.0;
		d_b[ 6*idx + 5 ] = 0.0;
	
		// Do the multiplication
		for ( int ii = 0; ii < m; ++ii ){
		
			Scalar xcurr = d_x[ii];
			
			d_b[ 6*idx + 0 ] += d_A[ ii*numel + 6*idx + 0 ] * xcurr; 
			d_b[ 6*idx + 1 ] += d_A[ ii*numel + 6*idx + 1 ] * xcurr; 
			d_b[ 6*idx + 2 ] += d_A[ ii*numel + 6*idx + 2 ] * xcurr; 
			d_b[ 6*idx + 3 ] += d_A[ ii*numel + 6*idx + 3 ] * xcurr; 
			d_b[ 6*idx + 4 ] += d_A[ ii*numel + 6*idx + 4 ] * xcurr; 
			d_b[ 6*idx + 5 ] += d_A[ ii*numel + 6*idx + 5 ] * xcurr; 

		}

        }
}



/*!
	Add two grid vectors
	C = A + B

	d_a		(input)  input vector, A
	d_b		(input)  input vector, B
	d_c		(output) output vector, C
	N		(input)  length of vectors
*/
__global__ void Brownian_FarField_AddGrids_kernel(
							CUFFTCOMPLEX *d_a, 
							CUFFTCOMPLEX *d_b, 
							CUFFTCOMPLEX *d_c, 
							unsigned int NxNyNz
							){

	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if ( tidx < NxNyNz) {
		unsigned int idx = tidx;
		CUFFTCOMPLEX A = d_a[idx];
		CUFFTCOMPLEX B = d_b[idx];
		d_c[idx] = make_scalar2(A.x+B.x, A.y+B.y);
	}
}


/*!
	Linear combination helper function
	C = a*A + b*B
	C can be A or B, so that A or B will be overwritten
	The fource element of Scalar4 is not changed!

	d_a              (input)  input vector, A
	d_b              (input)  input vector, B
	d_c              (output) output vector, C
	coeff_a          (input)  scaling factor for A, a
	coeff_b          (input)  scaling factor for B, b
	group_size       (input)  length of vectors
	d_group_members  (input)  index into vectors
*/
__global__ void Brownian_Farfield_LinearCombinationFTS_kernel(
								Scalar4 *d_a, 
								Scalar4 *d_b, 
								Scalar4 *d_c, 
								Scalar coeff_a, 
								Scalar coeff_b, 
								unsigned int group_size, 
								unsigned int *d_group_members
								){

	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (group_idx < group_size) {
		unsigned int idx = d_group_members[group_idx];

		Scalar4 A4 = d_a[idx];
		Scalar4 B4 = d_b[idx];
		d_c[idx] = make_scalar4( coeff_a*A4.x + coeff_b*B4.x, coeff_a*A4.y + coeff_b*B4.y, coeff_a*A4.z + coeff_b*B4.z, d_c[idx].w);

		A4 = d_a[ group_size + 2*idx ];
		B4 = d_b[ group_size + 2*idx ];
		d_c[ group_size + 2*idx ] = make_scalar4( coeff_a*A4.x + coeff_b*B4.x, coeff_a*A4.y + coeff_b*B4.y, coeff_a*A4.z + coeff_b*B4.z, coeff_a*A4.w + coeff_b*B4.w );
		
		A4 = d_a[ group_size + 2*idx+1 ];
		B4 = d_b[ group_size + 2*idx+1 ];
		d_c[ group_size + 2*idx+1 ] = make_scalar4( coeff_a*A4.x + coeff_b*B4.x, coeff_a*A4.y + coeff_b*B4.y, coeff_a*A4.z + coeff_b*B4.z, coeff_a*A4.w + coeff_b*B4.w );
	}
}


/*
	Wrap all the functions to compute the square root after each
	Lanczos iteration

	m		(input)  	number of iterations
	alpha		(input/output)	diagonal elements of tridiagonal matrix, changed upon output
	beta		(input/output)	off-diagonal elements of tridiagonal matrix, changed upon output
	alpha_save	(output)	saved diagonal elements of tridiagonal matrix
	beta_save	(output)	save off-diagonal elements of tridiagonal matrix
	W		(input)		eigenvectors of eigendecomposition of tridiagonal matrix
	W1		(output)	product of sqrt(D) * W * e1 (only needed within this code)
	Tm		(output)	product of Vm * sqrt(D) * W * e1
	d_Tm		(output) 	product fo Vm * sqrt(D) * W * e1, copied to device

*/
void Brownian_Sqrt(
			int m,
			float *alpha,
			float *beta,
			float *alpha_save,
			float *beta_save,
			float *W,
			float *W1,
			float *Tm,
			float *d_Tm
			){
		
	// Save alpha, beta vectors (will be overwritten by lapack)
	for ( int ii = 0; ii < m; ++ii ){
		alpha_save[ii] = alpha[ii];
		beta_save[ii] = beta[ii];
	}
	beta_save[m] = beta[m];

	// Compute eigen-decomposition of tridiagonal matrix
	// 	alpha (input) - vector of entries on main diagonal
	//      alpha (output) - eigenvalues sorted in descending order
	//      beta (input) - vector of entries of sub-diagonal
	//      beta (output) - overwritten (zeros?)
	//      W - (output) - matrix of eigenvectors. ith column corresponds to ith eigenvalue
	// 	INFO (output) = 0 if operation was succesful
	int INFO = LAPACKE_spteqr( LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m );

	if ( INFO != 0 ){
		printf("Eigenvalue decomposition failed \n");
		printf("INFO = %i \n", INFO); 
	
		printf("\n alpha: \n");
		for( int ii = 0; ii < m; ++ii ){
			printf("%f \n", alpha_save[ii]);
		} 
		printf("\n beta: \n");
		for( int ii = 0; ii < m; ++ii ){
			printf("%f \n", beta_save[ii]);
		}
		printf("%f \n", beta_save[m]);
		exit(EXIT_FAILURE);
	}

	// Now, we have to compute Tm^(1/2) * e1
	for ( int ii = 0; ii < m; ++ii ){
	    W1[ii] = sqrtf( alpha[ii] ) * W[ii];
	}
	// Tm = W * W1 = W * Lambda^(1/2) * W^T * e1
	float tempsum;
	for ( int ii = 0; ii < m; ++ii ){
		tempsum = 0.0;
		for ( int jj = 0; jj < m; ++jj ){
			int idx = m*ii + jj;

			tempsum += W[idx] * W1[jj];
		}
		Tm[ii] = tempsum;
	}

	// Copy matrix to GPU
	cudaMemcpy( d_Tm, Tm, m*sizeof(Scalar), cudaMemcpyHostToDevice );

}
