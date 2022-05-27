// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Brownian_NearField.cuh"
#include "Precondition.cuh"
#include "Lubrication.cuh"

#include "Helper_Brownian.cuh"
#include "Helper_Debug.cuh"
#include "Helper_Precondition.cuh"

#include "hoomd/Saru.h"
using namespace hoomd;

#include <stdio.h>
#include <math.h>

// LAPACK and CBLAS
#include "lapacke.h"
#include "cblas.h"

// cuBLAS
#include "cublas_v2.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! 
	\file Brownian_NearField.cu
	\brief Defines functions to compute the near-field Brownian Forces
*/

/*!
  	Generate random numbers on particles for Near-field calculation
	
	d_Psi_nf	(output) uniform random vector
        group_size	(input)  number of particles
	seed		(input)  seed for random number generation
	T		(input)  Temperature
	dt		(input)  Time step
*/
__global__ void Brownian_NearField_RNG_kernel(
						float *d_Psi_nf,
						unsigned int group_size,
						const unsigned int seed,
						const float T,
						const float dt
						){

	// Thread index
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds, and if so do work
	if (idx < group_size) {

                // Initialize random number generator
                detail::Saru s(idx, seed);

		// Scaling factor to get the variance right
		//
		// Fluctuation dissipation says variance is ( 2 * T / dt )
		// 
		// Variance of uniform random numbers on [ -1.0, 1.0 ] is 1/3
		// so we have to multiply by 3 to get the proper variance
		//
		// Therefore the right scale is 3 * ( 2 * T / dt );
		float fac = sqrtf( 3.0 * ( 2.0 * T / dt ) );

		// Generate random numbers and assign to global output
		d_Psi_nf[ 6 * idx     ] = s.f( -fac, fac );
		d_Psi_nf[ 6 * idx + 1 ] = s.f( -fac, fac );
		d_Psi_nf[ 6 * idx + 2 ] = s.f( -fac, fac );
		d_Psi_nf[ 6 * idx + 3 ] = s.f( -fac, fac );
		d_Psi_nf[ 6 * idx + 4 ] = s.f( -fac, fac );
		d_Psi_nf[ 6 * idx + 5 ] = s.f( -fac, fac );

	} // Check if thread is in bounds

}


/*!
	Use Lanczos method to compute RFU^0.5 * psi

	This method is detailed in the publication:
	Edmond Chow and Yousef Saad, PRECONDITIONED KRYLOV SUBSPACE METHODS FOR
	SAMPLING MULTIVARIATE GAUSSIAN DISTRIBUTIONS, SIAM J. Sci. Comput., 2014

	d_FBnf			(output) near-field Brownian force
	d_psi			(input)  uniform random vector
	d_group_members		(input)  ID of particle within integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	dt			(input)  integration timestep
	pBuffer			(input)  scratch buffer space for preconditioner
	ker_data		(input)  structure containing kernel launch information
	bro_data		(input)  structure containing Brownian calculation information
	res_data		(input)  structure containing lubrication calculation information
	work_data		(input)  structure containing workspaces

*/
void Brownian_NearField_Lanczos( 	
				Scalar *d_FBnf, // output
				Scalar *d_psi,  // input
				Scalar4 *d_pos,
				unsigned int *d_group_members,
                                unsigned int group_size,
                                const BoxDim& box,
                                Scalar dt,
				void *pBuffer,
				KernelData *ker_data,
				BrownianData *bro_data,
				ResistanceData *res_data,
				WorkData *work_data
				){

	// Kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Length of vectors
	int numel = 6 * group_size;

	// Allocate storage
	// 
	int m = bro_data->m_Lanczos_nf;
	
	int m_in = m;
	int m_max = 100;

        // Storage vectors for tridiagonal factorization
	float *alpha, *beta, *alpha_save, *beta_save;
        alpha = (float *)malloc( (m_max)*sizeof(float) );
        alpha_save = (float *)malloc( (m_max)*sizeof(float) );
        beta = (float *)malloc( (m_max+1)*sizeof(float) );
        beta_save = (float *)malloc( (m_max+1)*sizeof(float) );

	// Vectors for Lapacke and square root
	float *W;
	W = (float *)malloc( (m_max*m_max)*sizeof(float) );
	float *W1; // W1 = Lambda^(1/2) * ( W^T * e1 )
	W1 = (float *)malloc( (m_max)*sizeof(float) );
	float *Tm;
	Tm = (float *)malloc( m_max*sizeof(float) );
	Scalar *d_Tm = (work_data->bro_nf_Tm);

	// Vectors for Lanczos iterations
	Scalar *d_v = (work_data->bro_nf_v);

	// Storage array for V
	Scalar *d_V = (work_data->bro_nf_V);

	// Step-norm things
	Scalar *d_FBnf_old = (work_data->bro_nf_FB_old);

	// Pointers for current and previous vector
	Scalar *d_vjm1 = d_V;
	Scalar *d_vj   = &d_V[numel];

	// Initialize cuBLAS handle
	cublasHandle_t blasHandle = (res_data->blasHandle);

	// Norm of starting vector (also psi)
        Scalar vnorm, psinorm;
	cublasSnrm2( blasHandle, numel, d_psi, 1, &vnorm );
	psinorm = vnorm;
 
        // First iteration
	// vjm1 = 0 
	// vj = psi / norm( psi )
	Scalar scale = 0.0;
	cublasSscal( blasHandle, numel, &scale, d_vjm1, 1 );

	cudaMemcpy( d_vj, d_psi, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );
	scale = 1.0 / psinorm;
	cublasSscal( blasHandle, numel, &scale, d_vj, 1 );	
	
	//
	// Do the calculation for 1 fewer iterations than requested so that we can check the step norm
	//
	m = m_in - 1;
	m = m < 1 ? 1 : m;

	Scalar tempalpha = 0.0;
	Scalar tempbeta = 0.0;
	beta[ 0 ] = 0.0;
	for ( int jj = 0; jj < m; ++jj ){
		
		// v = M*vj - betaj*vjm1
		Precondition_Brownian_RFUmultiply(	
							d_v,       // output
							d_vj,      // input
							d_pos,
							d_group_members,
							group_size, 
			      				box,
							pBuffer,
							ker_data,
							res_data
							);

		scale = -1.0 * tempbeta;
		cublasSaxpy( blasHandle, numel, &scale, d_vjm1, 1, d_v, 1 );
	
		// vj dot v
		cublasSdot( blasHandle, numel, d_v, 1, d_vj, 1, &tempalpha );
	       
		// Store updated alpha
		alpha[jj] = tempalpha;
	
		// v = v - alphaj*vj
		scale = -1.0 * tempalpha;
		cublasSaxpy( blasHandle, numel, &scale, d_vj, 1, d_v, 1 );
		
		// betajp1 = norm( v ) 
		cublasSnrm2( blasHandle, numel, d_v, 1, &tempbeta );
		beta[jj+1] = tempbeta;
		
		if ( tempbeta < 1E-8 ){
			m = jj+1;
			break;
		}

		// vjp1 = v / betajp1
		scale = 1.0 / tempbeta;
		cublasSscal( blasHandle, numel, &scale, d_v, 1 );

		// Store current basis vector
		cudaMemcpy( &d_V[(jj+2)*numel], d_v, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );

		// Point vjm1 and vj to proper locations in memory
		d_vjm1 = &d_V[(jj+1)*numel];
		d_vj   = &d_V[(jj+2)*numel];
		
	}

	//
	// Evaluate the square root to compute the near-field Force for the current number
	// of iterations.
	//

	// Compute the tridiagonal square root on the host, and copy the result to the device
	Brownian_Sqrt(
			m,
			alpha,
			beta,
			alpha_save,
			beta_save,
			W,
			W1,
			Tm,
			d_Tm
			);

	// Multiply basis vectors by Tm
	Brownian_NearField_LanczosMatrixMultiply_kernel<<<grid,threads>>>( &d_V[numel], d_Tm, d_FBnf, group_size, numel, m );

	// Copy velocity
	cudaMemcpy( d_FBnf_old, d_FBnf, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );

	// Restore alpha, beta
	for ( int ii = 0; ii < m; ++ii ){
		alpha[ii] = alpha_save[ii];
		beta[ii] = beta_save[ii];
	}
	beta[m] = beta_save[m];

	//
	// Keep adding to basis until step norm is small enough
	//
	Scalar stepnorm = 1.0;
	int jj;
	while( stepnorm > (bro_data->tol) && m < m_max ){
		m++;
		jj = m - 1;
	
		//
		// Do another Lanczos iteration
		//
		
		// v = M*vj - betaj*vjm1
		Precondition_Brownian_RFUmultiply(	
							d_v,       // output
							d_vj,      // input
							d_pos,
							d_group_members,
							group_size, 
			      				box,
							pBuffer,
							ker_data,
							res_data
							);

		scale = -1.0 * tempbeta;
		cublasSaxpy( blasHandle, numel, &scale, d_vjm1, 1, d_v, 1 );

		// vj dot v
		cublasSdot( blasHandle, numel, d_v, 1, d_vj, 1, &tempalpha );
	       
		// Store updated alpha
		alpha[jj] = tempalpha;
	
		// v = v - alphaj*vj
		scale = -1.0 * tempalpha;
		cublasSaxpy( blasHandle, numel, &scale, d_vj, 1, d_v, 1 );
		
		// betajp1 = norm( v ) 
		cublasSnrm2( blasHandle, numel, d_v, 1, &tempbeta );
		beta[jj+1] = tempbeta;
		
		if ( tempbeta < 1E-8 ){
		    m = jj+1;
		    break;
		}

		// vjp1 = v / betajp1
		scale = 1.0 / tempbeta;
		cublasSscal( blasHandle, numel, &scale, d_v, 1 );

		// Store current basis vector
		cudaMemcpy( &d_V[(jj+2)*numel], d_v, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );

		// Point vjm1 and vj to proper locations in memory
		d_vjm1 = &d_V[(jj+1)*numel];
		d_vj   = &d_V[(jj+2)*numel];

		//
		// Compute the displacement and check the step norm error
		//

		// Compute the tridiagonal square root on the host, and copy the result to the device
		Brownian_Sqrt(
				m,
				alpha,
				beta,
				alpha_save,
				beta_save,
				W,
				W1,
				Tm,
				d_Tm
				);
		
		// Multiply basis vectors by Tm -- velocity = Vm * Tm
		Brownian_NearField_LanczosMatrixMultiply_kernel<<<grid,threads>>>( &d_V[numel], d_Tm, d_FBnf, group_size, numel, m );

		//
		// Compute step norm error
		//
		scale = -1.0;
		cublasSaxpy( blasHandle, numel, &scale, d_FBnf, 1, d_FBnf_old, 1 );
		cublasSnrm2( blasHandle, numel, d_FBnf_old, 1, &stepnorm );
		
		float fbnorm = 0.0;
		cublasSnrm2( blasHandle, numel, d_FBnf, 1, &fbnorm );
		stepnorm /= fbnorm;

		// Copy velocity
		cudaMemcpy( d_FBnf_old, d_FBnf, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );

		// Restore alpha, beta
		for ( int ii = 0; ii < m; ++ii ){
			alpha[ii] = alpha_save[ii];
			beta[ii] = beta_save[ii];
		}
		beta[m] = beta_save[m];

	}

	// Save the number of required iterations
	bro_data->m_Lanczos_nf = m;
 
	// Undo the preconditioning so that the result has the proper variance
	// 
	// operates in place
	Precondition_Brownian_Undo(	
					d_FBnf,       // input/output
					group_size,
					ker_data,
					res_data
					);
	
	// Rescale by original norm of Psi
	cublasSscal( blasHandle, numel, &psinorm, d_FBnf, 1 );	
 
	// Free the memory and clear pointers
	d_v = NULL;
	d_V = NULL;
	d_Tm = NULL;
	d_FBnf_old = NULL;

	d_vj   = NULL;
	d_vjm1 = NULL;

	free(alpha);
	free(beta);
	free(alpha_save);
	free(beta_save);

	free(W);
	free(W1);
	free(Tm);
		
}



/*
	Wrap all the functions required to compute the compute the near-field
	Brownian force and compute that force.
	
	d_FBnf			(output) near-field Brownian force
	d_pos			(input)  particle positions
	d_group_members		(input)  ID of particle within integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	dt			(input)  integration timestep
	ker_data		(input)  structure containing kernel launch information
	bro_data		(input)  structure containing Brownian calculation information
	res_data		(input)  structure containing lubrication calculation information
	work_data		(input)	 structure containing workspaces

*/
void Brownian_NearField_Force(
				Scalar *d_FBnf, // output
				Scalar4 *d_pos,
				unsigned int *d_group_members,
                                unsigned int group_size,
                                const BoxDim& box,
                                Scalar dt,
				void *pBuffer,
				KernelData *ker_data,
				BrownianData *bro_data,
				ResistanceData *res_data,
				WorkData *work_data
				){

	// Kernel Information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Only do work if Temperature is positive
	if ( (bro_data->T) > 0.0 ){

		// Initialize vectors
		float *d_Psi_nf = (work_data->bro_nf_psi);

		// Generate the random vectors on each particle
		Brownian_NearField_RNG_kernel<<<grid,threads>>>( 
								d_Psi_nf,
								group_size,
								bro_data->seed_nf,
								bro_data->T,
								dt
								);
		
		// Apply the Lanczos method
		Brownian_NearField_Lanczos( 	
						d_FBnf,   // output
						d_Psi_nf, // input
						d_pos,
						d_group_members,
                        		        group_size,
                        		        box,
						dt,
						pBuffer,
						ker_data,
						bro_data,
						res_data,
						work_data
						);
		
		// Clean Up
		d_Psi_nf = NULL;

	} // Check if T > 0.0 

}


