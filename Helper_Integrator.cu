// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Helper_Integrator.cuh"

#include "hoomd/Saru.h"
#include "hoomd/TextureTools.h"
using namespace hoomd;

#include <stdio.h>
#include <math.h>

#include "lapacke.h"
#include "cblas.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! 
	Helper_Integrator.cu

	Helper functions for saddle point integration
*/
	
/*!
  	Generate random numbers on particles.
	
	d_psi		(output) random vector
        n		(input)  number of particles
	timestep	(input)  length of time step
	seed		(input)  seed for random number generation

*/
__global__ void Integrator_RFD_RandDisp_kernel(
								float *d_psi,
								unsigned int N,
								const unsigned int seed
								){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
	if (idx < N) {

		// Initialize random seed
                detail::Saru s(idx, seed);

		// Square root of 3
		float sqrt3 = 1.732050807568877;
		
		// Call the random number generator
		float x1 = s.f( -sqrt3, sqrt3 );
		float y1 = s.f( -sqrt3, sqrt3 );
		float z1 = s.f( -sqrt3, sqrt3 );
		float x2 = s.f( -sqrt3, sqrt3 );
		float y2 = s.f( -sqrt3, sqrt3 );
		float z2 = s.f( -sqrt3, sqrt3 );

		// Write to output
		d_psi[ 6*idx + 0 ] = x1;
		d_psi[ 6*idx + 1 ] = y1;
		d_psi[ 6*idx + 2 ] = z1;
		d_psi[ 6*idx + 3 ] = x2;
		d_psi[ 6*idx + 4 ] = y2;
		d_psi[ 6*idx + 5 ] = z2; 

	}

}

/*! 
	The output velocity

	d_b	(output) output vector
   	N 	(input)  number of particles

*/
__global__ void Integrator_ZeroVelocity_kernel( 
						float *d_b,
						unsigned int N
						){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tid < N ) {
	
		d_b[ 6*tid + 0 ] = 0.0;
		d_b[ 6*tid + 1 ] = 0.0;
		d_b[ 6*tid + 2 ] = 0.0;
		d_b[ 6*tid + 3 ] = 0.0;
		d_b[ 6*tid + 4 ] = 0.0;
		d_b[ 6*tid + 5 ] = 0.0;
	
	}
}

/*! 
	Add rate of strain from shearing to the right-hand side of the saddle point solve

	d_b		(input/output) 	right-hand side vector
	shear_rate 	(input) 	shear rate of applied deformation
   	N 		(input)  	number of particles

*/
__global__ void Integrator_AddStrainRate_kernel( 
						float *d_b,
						float shear_rate,
						unsigned int N
						){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tid < N ) {


		// Index into array
		int ind = 6*N + 5*tid;

		// Add strain rate. For each particle, stores
		// [ F1, F2, F3, L1, L2, L3, E1, E2, E3, E4, E5 ]	
		d_b[ ind + 0 ] += 0.0;
		d_b[ ind + 1 ] += shear_rate; //zhoge: because it is 2E_xy, see "Computational tricks"
		d_b[ ind + 2 ] += 0.0;
		d_b[ ind + 3 ] += 0.0;
		d_b[ ind + 4 ] += 0.0;

	}
}
