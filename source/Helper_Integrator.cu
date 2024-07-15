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
	B2              (input)         coefficient of B2 mode (spherical squirmers)
	d_ori           (input)         particle orientation (unit vector)
   	N 		(input)  	number of particles

*/
__global__ void Integrator_AddStrainRate_kernel( 
						float *d_b,
						float shear_rate,
						unsigned int *d_group_members,
						float B2,
						float *d_sqm_B2_mask,
						Scalar3 *d_ori,
						unsigned int N
						){

	// Thread index
	unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tidx < N ) {

		// Particle ID
		unsigned int tid = d_group_members[tidx];

		// Index into array
		unsigned int ind = 6*N + 5*tid;
	        
		// Add ambient strain rate Einf (E_xy = E_yx = shear_rate/2, all else 0)
		d_b[ ind + 0 ] += 0.0;        // E_xx - E_zz
		d_b[ ind + 1 ] += shear_rate; // E_xy * 2   
		d_b[ ind + 2 ] += 0.0;	      // E_xz * 2   
		d_b[ ind + 3 ] += 0.0;	      // E_yz * 2   
		d_b[ ind + 4 ] += 0.0;	      // E_yy - E_zz

		// Substract the particle strain rate from Einf
		Scalar3 pdir = d_ori[tid];
		Scalar px = pdir.x;
		Scalar py = pdir.y;
		Scalar pz = pdir.z;

		Scalar b2 = -0.6*B2*d_sqm_B2_mask[tid];  //prefactor for the active strain rate (require radius a=1)
		d_b[ ind + 0 ] -= b2*(px*px - pz*pz);   
		d_b[ ind + 1 ] -= b2*(2.*px*py);	
		d_b[ ind + 2 ] -= b2*(2.*px*pz);	
		d_b[ ind + 3 ] -= b2*(2.*py*pz);	
		d_b[ ind + 4 ] -= b2*(py*py - pz*pz);
		

	}
}
