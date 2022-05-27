// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore


#include "Helper_Saddle.cuh"

#include <cusparse.h>

#include <stdio.h>

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


/*! \file Helper_Saddle.cu
	Helper functions to perform the additions and operations required in the saddle point
	matrix calculations
*/

/*! 
	Zero the output for the saddle point multiplication
	
	d_b	(input/output) 	vector zeroed upon output
   	N	(input) 	number of particles
*/
__global__ void Saddle_ZeroOutput_kernel( 
						float *d_b,
						unsigned int N
						){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tid < N ) {
	
		// Do the zeroing
		for ( int ii = 0; ii < 17; ii++ ){ 
			d_b[ 17*tid + ii ] = 0.0;
	      	}  
	
	}
}


/*!
        Direct addition of two float arrays

        C = a*A + b*B
        C can be A or B, so that A or B will be overwritten

        d_a		(input)  input vector, A
        d_b		(input)  input vector, B
        d_c		(output) output vector, C
        coeff_a		(input)  scaling factor for A, a
        coeff_b		(input)  scaling factor for B, b
        N		(input)  length of vectors
	stride		(input)  number of repeats
*/
__global__ void Saddle_AddFloat_kernel( 	float *d_a, 
						float *d_b,
						float *d_c,
						float coeff_a,
						float coeff_b,
						unsigned int N,
						int stride
						){

	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (idx < N) {
       
		for ( int ii = 0; ii < stride; ++ii ){
			
			// Index for current striding
			int ind = stride * idx + ii;

			// Do addition
			d_c[ ind ] = coeff_a * d_a[ ind ] + coeff_b * d_b[ ind ];
 
		}

        }
}

/*!
        Split generalized force into force/torque/stresslet

	d_generalF	(input)  11N vector of generalized force (force/torque first 6N, stresslet last 5N)
	d_net_force	(output) linear force
	d_TorqueStress	(output) torque and stresslet
	N		(input)  number of particles

*/
__global__ void Saddle_SplitGeneralizedF_kernel( 	float *d_generalF, 
							Scalar4 *d_net_force,
							Scalar4 *d_TorqueStress,
							unsigned int N
){
	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (idx < N) {

		int ind1 = 6*idx;
		int ind2 = 6*N + 5*idx;      
 
		// 
		float f1 = d_generalF[ ind1 + 0 ];
		float f2 = d_generalF[ ind1 + 1 ];
		float f3 = d_generalF[ ind1 + 2 ];
		float l1 = d_generalF[ ind1 + 3 ];
		float l2 = d_generalF[ ind1 + 4 ];
		float l3 = d_generalF[ ind1 + 5 ];
		float s1 = d_generalF[ ind2 + 0 ];
		float s2 = d_generalF[ ind2 + 1 ];
		float s3 = d_generalF[ ind2 + 2 ];
		float s4 = d_generalF[ ind2 + 3 ];
		float s5 = d_generalF[ ind2 + 4 ];

		d_net_force[ idx ] = make_scalar4( f1, f2, f3, 0.0 );
		d_TorqueStress[ 2*idx + 0 ] = make_scalar4( l1, l2, l3, s1 );
		d_TorqueStress[ 2*idx + 1 ] = make_scalar4( s2, s3, s4, s5 );

        }
}

/*!
        Combine velocity/angular velocity/rate of strain into generalized velocity

	d_generalU	(output) 11N vector of generalized velocity (first 6N) and trate of strain (last 5N)
	d_vel		(input)  linear velocity
	d_AngvelStrain	(input)  angular velocity and rate of strain
	N		(input)  number of particles

*/
__global__ void Saddle_MakeGeneralizedU_kernel( 	float *d_generalU, 
							Scalar4 *d_vel,
							Scalar4 *d_AngvelStrain,
							unsigned int N
){
	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (idx < N) {

		float4 vel = d_vel[ idx ];
		float4 AS1 = d_AngvelStrain[ 2*idx + 0 ];
		float4 AS2 = d_AngvelStrain[ 2*idx + 1 ];

		int ind1 = 6*idx;
		int ind2 = 6*N + 5*idx;      
 
		d_generalU[ ind1 + 0 ] = vel.x;   // U_x - U^infty
		d_generalU[ ind1 + 1 ] = vel.y;   // U_y - U^infty
		d_generalU[ ind1 + 2 ] = vel.z;   // U_z - U^infty
		d_generalU[ ind1 + 3 ] = AS1.x;   // Omega_x - Omega^infty
		d_generalU[ ind1 + 4 ] = AS1.y;   // Omega_y - Omega^infty
		d_generalU[ ind1 + 5 ] = AS1.z;   // Omega_z - Omega^infty
		d_generalU[ ind2 + 0 ] = AS1.w;   // E_xx - E_zz
		d_generalU[ ind2 + 1 ] = AS2.x;   // E_xy * 2
		d_generalU[ ind2 + 2 ] = AS2.y;   // E_xz * 2
		d_generalU[ ind2 + 3 ] = AS2.z;   // E_yz * 2
		d_generalU[ ind2 + 4 ] = AS2.w;   // E_yy - E_zz

		////zhoge: convert the sign (debug)
		//d_generalU[ ind1 + 0 ] = - vel.x; 
		//d_generalU[ ind1 + 1 ] = - vel.y; 
		//d_generalU[ ind1 + 2 ] = - vel.z; 
		//d_generalU[ ind1 + 3 ] = - AS1.x; 
		//d_generalU[ ind1 + 4 ] = - AS1.y; 
		//d_generalU[ ind1 + 5 ] = - AS1.z; 
		//d_generalU[ ind2 + 0 ] = - AS1.w; 
		//d_generalU[ ind2 + 1 ] = - AS2.x; 
		//d_generalU[ ind2 + 2 ] = - AS2.y; 
		//d_generalU[ ind2 + 3 ] = - AS2.z; 
		//d_generalU[ ind2 + 4 ] = - AS2.w; 

        }
}

/*!
        Copy force/torque to right-hand-side vector of saddle point problem

	d_force		(input)		6*N vector of particle force/torque
	d_rhs		(input/output)	17*N vector of right-hand side vector
	N		(input)		Number of particles

*/
__global__ void Saddle_force2rhs_kernel(
					float *d_force, 
					float *d_rhs,
					unsigned int N
){
	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (idx < N) {

	  //zhoge: Here, it directly uses the GPU core index!!!
	  //       It should be d_group_members[idx] to be consistent with the rest!!!
	  //       If we use the global index array for d_group_members, idx is okay
	  //       because both index[idx] = idx (from 0 to N-1).
	  //       However, if we use tag for d_group_members, it will lead to inconsistency
	  //       because tag[idx] != inx.
	  

		d_rhs[ 11*N + 6*idx + 0 ] -= d_force[ 6*idx + 0 ];
		d_rhs[ 11*N + 6*idx + 1 ] -= d_force[ 6*idx + 1 ];
		d_rhs[ 11*N + 6*idx + 2 ] -= d_force[ 6*idx + 2 ];
		d_rhs[ 11*N + 6*idx + 3 ] -= d_force[ 6*idx + 3 ];
		d_rhs[ 11*N + 6*idx + 4 ] -= d_force[ 6*idx + 4 ];
		d_rhs[ 11*N + 6*idx + 5 ] -= d_force[ 6*idx + 5 ];

        }
}


/*!
        Copy velocity out of saddle point solution vector

	d_U		(output) 6*N vector of particle linear/angular velocities
	d_solution	(input)  17*N vector of right-hand side vector
	N		(input)  Number of particles

*/
__global__ void Saddle_solution2vel_kernel(
					float *d_U, 
					float *d_solution,
					unsigned int N
){
	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (idx < N) {
	  
		d_U[ 6*idx + 0 ] = d_solution[ 11*N + 6*idx + 0 ];
		d_U[ 6*idx + 1 ] = d_solution[ 11*N + 6*idx + 1 ];
		d_U[ 6*idx + 2 ] = d_solution[ 11*N + 6*idx + 2 ];
		d_U[ 6*idx + 3 ] = d_solution[ 11*N + 6*idx + 3 ];
		d_U[ 6*idx + 4 ] = d_solution[ 11*N + 6*idx + 4 ];
		d_U[ 6*idx + 5 ] = d_solution[ 11*N + 6*idx + 5 ];
 
        }
}



