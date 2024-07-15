// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Helper_Stokes.cuh"  //zhoge: This includes HOOMDMath.h, which includes cmath

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


/*! \file Helper_Stokes.cu
    	\brief Helper functions required for data handling in Stokes.cu
*/

/*!
	Initialize the total applied force and torque using the net_force
	vector from HOOMD which contains the contributions from external
	and interparticle potentials

	d_net_force		(input)  HOOMD force vector
	d_AppliedForce		(output) Total force experience by the particles
	group_size		(input)  length of vectors
	d_group_members		(input)  index into vectors

*/
__global__ void Stokes_SetForce_kernel(
						Scalar4 *d_net_force,
						float   *d_AppliedForce,
						unsigned int group_size,
						unsigned int *d_group_members
					){

	// Thread idx
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	// Do work if thread is in bounds
	if (tidx < group_size) {

		unsigned int idx = d_group_members[ tidx ];
		
		Scalar4 net_force = d_net_force[ idx ];

		d_AppliedForce[ 6*idx     ] = net_force.x;
		d_AppliedForce[ 6*idx + 1 ] = net_force.y;
		d_AppliedForce[ 6*idx + 2 ] = net_force.z;
		d_AppliedForce[ 6*idx + 3 ] = 0.0;
		d_AppliedForce[ 6*idx + 4 ] = 0.0;
		d_AppliedForce[ 6*idx + 5 ] = 0.0;

	}
}

__global__ void Stokes_SetForce_manually_kernel(
						const Scalar4 *d_pos,     //input
						Scalar3 *d_ori,           //input
						float   *d_AppliedForce,  //output
						unsigned int group_size,
						unsigned int *d_group_members,
						const unsigned int *d_nneigh, 
						unsigned int *d_nlist, 
						const unsigned int *d_headlist,
						const float ndsr,
						const float k_n,
						const float kappa,
						const float beta,
						const float epsq,
						Scalar T_ext,
						const BoxDim box
						){

  // Thread idx
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  // Do work if thread is in bounds
  if (tidx < group_size) {
    
    unsigned int idx = d_group_members[ tidx ];

    Scalar4 posi = d_pos[idx];  // position
    Scalar3 pdir = d_ori[idx];  // orientation

    // Interparticle force parameters
    float h_rough = sqrt(epsq);       //roughness height
    float rcol = 2.0 + 1.0*h_rough;   //collision cutoff
    float F_0 = 1.0/ndsr;             //repulsive force scale (ASSUMING sr = 1.0)
    float Hamaker = F_0*beta;         //Hamaker constant for vdW
    //float T_ext = 1.0;    //magnitude of the external torque
    
    // Interparticle force
    float F_x = 0.;
    float F_y = 0.;
    float F_z = 0.;

    // External torque (rotate particles to align with the z-dir)
    float T_x = T_ext * ( pdir.y);
    float T_y = T_ext * (-pdir.x);
    float T_z = T_ext * ( 0.    );
    
    // Neighborlist arrays
    unsigned int head_idx = d_headlist[ idx ]; // Location in head array for neighbors of current particle
    unsigned int n_neigh  = d_nneigh[ idx ];   // Number of neighbors of the nearest particle

    //float hhh = -1.0;
    //printf(" Testing the exponential function %16.8f, %16.8f\n", exp(hhh),expf(hhh));

		
    for (unsigned int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {

      // Get the current neighbor index
      unsigned int curr_neigh = d_nlist[ head_idx + neigh_idx ];
		  
      Scalar4 posj = d_pos[curr_neigh];  // position
      Scalar3 R = make_scalar3( posj.x - posi.x, posj.y - posi.y, posj.z - posi.z );  // distance vector
      R = box.minImage(R);  //periodic BC
      Scalar  distSqr = dot(R,R);          
      Scalar  dist    = sqrtf( distSqr );  // Distance magnitude

      Scalar  gap1 = dist - rcol;  //surface gap for interparticle forces
      
      float F_app_mag = 0.;  //applied force magnitude
      
      // vdW and electrostatic repulsion
      if (gap1 >= 0. && dist <= 2.0 + 10.0/kappa)
      	F_app_mag = Hamaker/(12.*(gap1*gap1 + epsq)) - F_0 * expf (-kappa * gap1);  //attraction is positive, repulsion is negative

      // Max vdW - Max electrostatic repulsion - Collision
      if (gap1 < 0.)
      	F_app_mag = Hamaker/(12.*epsq) - F_0 - k_n * abs(gap1);
      
      // Normal vector
      float normalx = R.x/dist;  //from center to neighbor
      float normaly = R.y/dist;  //from center to neighbor
      float normalz = R.z/dist;  //from center to neighbor

      // Accumulate the collision/repulsive forces
      F_x += F_app_mag * normalx;
      F_y += F_app_mag * normaly;
      F_z += F_app_mag * normalz;

    } //neighbor particle
    
    d_AppliedForce[ 6*idx     ] = F_x;
    d_AppliedForce[ 6*idx + 1 ] = F_y;
    d_AppliedForce[ 6*idx + 2 ] = F_z;
    d_AppliedForce[ 6*idx + 3 ] = T_x;//0.0;
    d_AppliedForce[ 6*idx + 4 ] = T_y;//0.0;
    d_AppliedForce[ 6*idx + 5 ] = T_z;//0.0;
    
  }
}

/*!
	Copy velocity computed from solving the hydrodynamic problem
	to the HOOMD velocity array

	d_vel			(output) HOOMD velocity vector
	d_Velocity		(input)  Velocity computed from hydrodynamics
	group_size		(input)  length of vectors
	d_group_members		(input)  index into vectors

*/
__global__ void Stokes_SetVelocity_kernel(
						Scalar4 *d_vel,
						Scalar4 *d_omg,
						float   *d_Velocity,
						unsigned int group_size,
						unsigned int *d_group_members
						){

	// Thread idx
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	// Do work if thread is in bounds
	if (tidx < group_size) {

		Scalar4 vel,omg;
		
		unsigned int idx  = d_group_members[ tidx ];
		unsigned int idx0 = 6*idx;
		//unsigned int idx1 = 6*group_size + 5*idx;
		
		vel.x = d_Velocity[ idx0     ];
		vel.y = d_Velocity[ idx0 + 1 ];
		vel.z = d_Velocity[ idx0 + 2 ];
		omg.x = d_Velocity[ idx0 + 3 ];
		omg.y = d_Velocity[ idx0 + 4 ];
		omg.z = d_Velocity[ idx0 + 5 ];

		d_vel[ idx ] = make_scalar4( vel.x, vel.y, vel.z, 0. );
		d_omg[ idx ] = make_scalar4( omg.x, omg.y, omg.z, 0. );

	}
}

