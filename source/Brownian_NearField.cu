// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Zhouyang Ge

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

#include <curand.h>
#include <cuda_runtime.h>

// LAPACK and CBLAS
#include "lapacke.h"
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



//zhoge: Re-implemented Chow & Saad (2014) method to sample correlated noise (near-field).

void Lanczos_process( float *d_vm,  //input
		      float *d_v,   //input
		      float *d_vp,  //output
		      float *alpha, //output
		      float *beta,  //output/input
		      float tol_beta,
		      int numel,
		      const Scalar4 *d_pos,
		      unsigned int *d_group_members,
		      const int group_size, 
		      const BoxDim box,
		      void *pBuffer,
		      KernelData *ker_data,
		      ResistanceData *res_data,
		      WorkData *work_data )
{
  // cuBLAS handle
  cublasHandle_t blasHandle = work_data->blasHandle;

  // Apply the preconditioned A to d_v (d_vp = G * A * G^T * d_v, where G^T * G = A^{-1} is the preconditioner)
  Precondition_Brownian_RFUmultiply( d_vp,  // output
  				     d_v,   // input
  				     d_pos,
  				     d_group_members,
  				     group_size, 
  				     box,
  				     pBuffer,
  				     ker_data,
  				     res_data );
    
  // Project out d_vm (d_vp = d_vp - beta * d_vm)
  float scale = -1.0 * beta[0];
  cublasSaxpy( blasHandle, numel, &scale, d_vm, 1, d_vp, 1 );  //d_vp is modified in place

  // The diagonal value associated with dv (alpha = d_v \cdot d_vp)
  cublasSdot( blasHandle, numel, d_v, 1, d_vp, 1, alpha );
  
  // Project out d_v (d_vp = d_vp - alpha * d_v)
  scale = -1.0 * alpha[0];
  cublasSaxpy( blasHandle, numel, &scale, d_v, 1, d_vp, 1 );  //d_vp is modified in place

  // The norm of d_vp (betap = || d_vp ||)
  cublasSnrm2( blasHandle, numel, d_vp, 1, &beta[1] );

  // Check if the norm has become very small and if so, set d_vp = d_v
  if ( beta[1] < tol_beta )
    {
      cudaMemcpy( d_vp, d_v, numel*sizeof(float), cudaMemcpyDeviceToDevice );
    }
  else  //otherwise normalize d_vp
    {
      scale = 1.0 / beta[1];
      cublasSscal( blasHandle, numel, &scale, d_vp, 1 );  //d_vp is modified in place
    }
  
}






void Brownian_NearField_Chow_Saad( Scalar *d_y,  // output: near-field Brownian force
				   Scalar *d_x,  // input: random Gaussian variables
				   Scalar4 *d_pos,
				   unsigned int *d_group_members,
				   unsigned int group_size,
				   const BoxDim& box,
				   //Scalar dt,
				   void *pBuffer,
				   KernelData *ker_data,
				   BrownianData *bro_data,
				   ResistanceData *res_data,
				   WorkData *work_data)
{
  // cuBLAS handle
  cublasHandle_t blasHandle = work_data->blasHandle;

  // Constants
  int numel = 6 * group_size;      //size of v1,v2,...,vm_max, d_x, d_y
  int m = bro_data->m_Lanczos_nf;  //number of Lanczos iterations in step 1 (either same as last time or reset in Stokes.cc)
  int m_max = 100;                 //m_max-1 is the maximum size of Tm at the end of step 2 (set to 100 in Stokes.cc)

  //debug
  if ( m >= m_max-1 )
    {
      printf("Illegal condition: m >= m_max-1. Program aborted.");
      exit(1);
    }
  
  // Host vectors for the main and sub-diagonal values of Tm
  float *h_alpha  = (float *)malloc( (m_max)*sizeof(float) );
  float *h_beta   = (float *)malloc( (m_max)*sizeof(float) );
  float *h_alpha1 = (float *)malloc( (m_max)*sizeof(float) );  //buffer
  float *h_beta1  = (float *)malloc( (m_max)*sizeof(float) );  //buffer

  // Set the first element of beta to 0
  h_beta[0] = 0.0;

  // Set the tolerance for beta (less than 1e-6 even for single precision because ||vm|| can be << 1)
  float tol_beta = 1e-8; 

  // Buffer vector for checking convergence
  Scalar *d_y0 = work_data->bro_nf_FB_old;  

  // Lanczos basis vectors V = [v0, v1, v2, ..., vm_max], v0 is a placeholder
  Scalar *d_V = work_data->bro_nf_V;
	
  // Zero out v0
  float scale = 0.0;
  cublasSscal( blasHandle, numel, &scale, d_V, 1 );

  // Initialize v1 = d_x / ||d_x||
  float xnorm;
  cublasSnrm2( blasHandle, numel, d_x, 1, &xnorm );
  
  cudaMemcpy( &d_V[numel], d_x, numel*sizeof(float), cudaMemcpyDeviceToDevice );
  
  scale = 1.0 / xnorm;
  cublasSscal( blasHandle, numel, &scale, &d_V[numel], 1 ); 
  
  //
  // Step 1: Build Vm and Tm via the Lanczos process
  //
  for ( int j = 0; j < m; ++j )  //iterate at most m times 
    {
      // Find Vm and Tm that approximately satisfy Vm^T * A * Vm = Tm
      Lanczos_process( &d_V[  j   *numel ],  //input
		       &d_V[ (j+1)*numel ],  //input
		       &d_V[ (j+2)*numel ],  //output
		       &h_alpha[ j ],        //output
		       &h_beta[  j ],        //input [j] / output [j+1]
		       tol_beta,
		       numel,
		       d_pos,
		       d_group_members,
		       group_size, 
		       box,
		       pBuffer,
		       ker_data,
		       res_data,
		       work_data );

      // Stop if beta becomes very small
      if ( h_beta[j+1] < tol_beta )
	{
	  m = j+1;  //plus 1 because one iteration was done when j=0
	  break;
	}		
    }  

  ////debug
  //printf("m = %i\n",m);
  //for (int i=0; i<m; ++i){
  //  printf("alpha[%2i] = %f\n",i,h_alpha[i]);
  //}
  //printf("\n");
  //for (int i=0; i<m; ++i){
  //  printf("beta[ %2i] = %f\n",i,h_beta[ i]);
  //}
  //exit(1);

  
  //
  // Step 2: Iteratively compute d_y until convergence
  //
  Sqrt_multiply( &d_V[ numel ],  //input
		 h_alpha,        //input
		 h_beta,	 //input
		 h_alpha1,	 //input (buffer)
		 h_beta1,        //input (buffer)
		 m,              //input 
		 d_y0,           //output
		 numel,
		 group_size,
		 ker_data,
		 work_data );

  float error = 1.0;
  float ynorm = 1.0;

  cublasSnrm2( blasHandle, numel, d_y0, 1, &ynorm );

  ////debug
  //printf("Step 2: norm of d_y0 %10.3e\n",ynorm); 
  //exit(1);
  
  while( error > bro_data->tol and m < m_max-1 )
    {
      // Iteratively increase m
      Lanczos_process( &d_V[  m   *numel ],  //input
		       &d_V[ (m+1)*numel ],  //input
		       &d_V[ (m+2)*numel ],  //output
		       &h_alpha[ m ],        //output
		       &h_beta[  m ],        //input [m] / output [m+1]
		       tol_beta,
		       numel,
		       d_pos,
		       d_group_members,
		       group_size, 
		       box,
		       pBuffer,
		       ker_data,
		       res_data,
		       work_data );

      // Compute the new approximate solution, d_y
      Sqrt_multiply( &d_V[ numel ],  //input
		     h_alpha,        //input
		     h_beta,	     //input
		     h_alpha1,	     //input (buffer)
		     h_beta1,        //input (buffer)
		     m+1,            //input 
		     d_y,            //output
		     numel,
		     group_size,
		     ker_data,
		     work_data );

      	
      // Compute relative error = || d_y0 - d_y || / || d_y ||
      scale = -1.0;
      cublasSaxpy( blasHandle, numel, &scale, d_y, 1, d_y0, 1 );  //d_y0 is modified in place
      cublasSnrm2( blasHandle, numel, d_y0, 1, &error );
      cublasSnrm2( blasHandle, numel, d_y,  1, &ynorm );
      error /= ynorm;

      ////debug
      //printf("Chow & Saad (near-field) iteration %3i, relative error %13.6e (norm of d_y %13.6e)\n",m,error,ynorm);

      // Update solution
      cudaMemcpy( d_y0, d_y, numel*sizeof(float), cudaMemcpyDeviceToDevice );

      // Stop if beta becomes very small (even if the error is not small enough)
      if ( h_beta[m+1] < tol_beta )
	{
	  ++m;
	  break;
	}

      // Increment m
      ++m;
	
    }

  ////debug
  //printf("\n");

  // Finalize
  if ( error > bro_data->tol )
    {
      printf("\nChow & Saad (near-field) didn't converge after %i iterations.\n",m-1);
      printf("Final relative error %13.6e\n",error);
      printf("Last beta %13.6e\n",h_beta[m]);
      //printf("\nProgram aborted.\n");
      //exit(1);
    }
  
  // Save the number of required iterations (minus 1 because incremented at the end)
  bro_data->m_Lanczos_nf = m-1;

  //// Undo the preconditioning so that the result has the proper variance
  //Precondition_Brownian_Undo( d_y,       //input/output
  //			      group_size,
  //			      ker_data,
  //			      res_data );

  // Rescale by original norm of d_x
  cublasSscal( blasHandle, numel, &xnorm, d_y, 1 );	     

  
  // Clean up
  free(h_alpha);
  free(h_alpha1);
  free(h_beta);
  free(h_beta1);
		
}



/*
	Wrap all the functions required to compute the near-field Brownian force.
	
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
void Brownian_NearField_Force(Scalar *d_FBnf, // output
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
			      )
{

  //// Kernel Information
  //dim3 grid = ker_data->particle_grid;
  //dim3 threads = ker_data->particle_threads;


  // Initialize vectors
  float *d_Psi_nf = work_data->bro_nf_psi;

  //// Generate the random vectors on each particle
  //Brownian_NearField_RNG_kernel<<<grid,threads>>>( 
  //						  d_Psi_nf,  //output
  //						  group_size,
  //						  bro_data->seed_nf,
  //						  bro_data->T,
  //						  dt);
  
  //zhoge: use cuRand to generate Gaussian variables
  curandGenerator_t gen;
  unsigned int N_random = 6*group_size;                           //6 because force (3) and torque (3)
  float std_bro = sqrtf( 2.0 * bro_data->T / dt );                //standard deviation of the Brownian force
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);   //fastest generator
  curandSetPseudoRandomGeneratorSeed(gen, bro_data->seed_nf);     //set the seed (different from the ff)
  curandGenerateNormal(gen, d_Psi_nf, N_random, 0.0f, std_bro);   //mean 0, std as specified
  curandDestroyGenerator(gen);  
  
  
  // Apply the Chow & Saad method to sample the near-field force
  Brownian_NearField_Chow_Saad( d_FBnf,   //output
			        d_Psi_nf, //input
			        d_pos,
			        d_group_members,
			        group_size,
			        box,
			        //dt,
			        pBuffer,
			        ker_data,
			        bro_data,
			        res_data,
			        work_data);
		
  // Clean Up
  d_Psi_nf = NULL;

}
