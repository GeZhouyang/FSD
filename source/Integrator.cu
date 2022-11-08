// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Zhouyang Ge

#include "Integrator.cuh"

#include "Brownian_FarField.cuh"
#include "Brownian_NearField.cuh"
#include "Lubrication.cuh"
#include "Mobility.cuh"
#include "Precondition.cuh"
#include "Solvers.cuh"
#include "Wrappers.cuh"

#include "Helper_Debug.cuh"
#include "Helper_Integrator.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Precondition.cuh"
#include "Helper_Saddle.cuh"


#include <cusparse.h>
#include <cusolverSp.h>

#include <stdio.h>
#include <math.h>

#include "lapacke.h"
#include "cblas.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file Integrator.cu
    \brief Defines integrator functions to capture Brownian drift in the
		velocity and stresslet. 
*/

/*! 
	Integrates particle position according to the Explicit Euler-Maruyama scheme
   
	d_pos_in		(input)  3Nx1 particle positions at initial point
	d_pos_out		(output) 3Nx1 new particle positions
	d_Velocity		(input)  6Nx1 generalized particle velocities
	d_image			(input)  particle periodic images
	d_group_members		(input)  indices of the mebers of the group to integrate
	group_size		(input)  Number of members in the group
	box Box			(input)  dimensions for periodic boundary condition handling
	dt			(input)  timestep
	
	This kernel must be executed with a 1D grid of any block size such that the number of threads is greater than or
	equal to the number of members in the group. The kernel's implementation simply reads one particle in each thread
	and updates that particle. 
*/
extern "C" __global__ void Integrator_ExplicitEuler_kernel(	
								Scalar4 *d_pos_in,
								Scalar4 *d_pos_out,
                             					float *d_Velocity,
                             					int3 *d_image,
                             					unsigned int *d_group_members,
                             					unsigned int group_size,
                             					BoxDim box,
                             					Scalar dt 
								){

	// Thread ID
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check that thread is in bounds
	if ( tidx < group_size ){

		// Particle ID
		unsigned int idx = d_group_members[tidx];
		
		// read the particle's posision
		Scalar4 pos4 = d_pos_in[idx];
		Scalar3 pos = make_scalar3(pos4.x, pos4.y, pos4.z);
		
		// read the particle's velocity and update position
		float ux = d_Velocity[ 6*idx ];
		float uy = d_Velocity[ 6*idx + 1 ];
		float uz = d_Velocity[ 6*idx + 2 ];
		Scalar3 vel = make_scalar3( ux, uy, uz);
	
		Scalar3 dx = vel * dt;
		pos += dx;
		
		// Read in particle's image and wrap periodic boundary
		int3 image = d_image[idx];
		box.wrap(pos, image);
		
		// write out the results
		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, pos4.w);
		d_image[idx] = image;
	}
}

/*! 
	Integrates particle position according to the Explicit Euler-Maruyama scheme, with shear
   
	d_pos_in		(input)  3Nx1 particle positions at initial point
	d_pos_out		(output) 3Nx1 new particle positions
	d_Velocity		(input)  6Nx1 generalized particle velocities
	d_image			(input)  particle periodic images
	d_group_members		(input)  indices of the mebers of the group to integrate
	group_size		(input)  Number of members in the group
	box Box			(input)  dimensions for periodic boundary condition handling
	dt			(input)  timestep
	shear_rate		(input)  shear rate for the system
	
	This kernel must be executed with a 1D grid of any block size such that the number of threads is greater than or
	equal to the number of members in the group. The kernel's implementation simply reads one particle in each thread
	and updates that particle. 
*/

extern "C" __global__ void Integrator_ExplicitEuler_Shear_kernel(Scalar4 *d_pos_in,
								 Scalar4 *d_pos_out,
								 float *d_Velocity,
								 int3 *d_image,
								 unsigned int *d_group_members,
								 unsigned int group_size,
								 BoxDim box,
								 Scalar dt,
								 Scalar shear_rate
								 ){

	// Thread ID
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check that thread is in bounds
	if ( tidx < group_size ){

		// Particle ID
		unsigned int idx = d_group_members[tidx];
		
		// read the particle's posision
		Scalar4 pos4 = d_pos_in[idx];
		Scalar3 pos = make_scalar3(pos4.x, pos4.y, pos4.z);
		  
		// read the particle's velocity and update position
		float ux = d_Velocity[ 6*idx ];
		float uy = d_Velocity[ 6*idx + 1 ];
		float uz = d_Velocity[ 6*idx + 2 ];
		Scalar3 vel = make_scalar3( ux, uy, uz);

	        // Add the shear
	        vel.x += shear_rate * pos.y;

		// Move the positions	
		Scalar3 dx = vel * dt;
		pos += dx;
		
		// Read in particle's image and wrap periodic boundary  //zhoge: Ineffective now
		//int3 image = d_image[idx]; //zhoge
		//box.wrap(pos, image);  //zhoge: Do not wrap in the Euler midstep. Allow particles to temporarily step out.
		
		// write out the results
		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, pos4.w); //last entry doesn't matter
		//d_image[idx] = image;  //zhoge
	}
}


extern "C" __global__ void Integrator_RK_Shear_kernel(Scalar coef_1, Scalar4 *d_pos_in_1,
						      Scalar coef_2, Scalar4 *d_pos_in_2,
						      Scalar4 *d_pos_out,
						      float *d_Velocity,
						      int3 *d_image,
						      unsigned int *d_group_members,
						      unsigned int group_size,
						      BoxDim box,
						      Scalar coef_3, Scalar dt,
						      Scalar shear_rate
						      ){

	// Thread ID
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check that thread is in bounds
	if ( tidx < group_size ){

		// Particle ID
		unsigned int idx = d_group_members[tidx];
		
		// read the particle's posision
		Scalar4 pos4 = d_pos_in_1[idx]; //pos4 is just a buffer
		Scalar3 pos1 = make_scalar3(pos4.x, pos4.y, pos4.z);
		pos4         = d_pos_in_2[idx]; //overwrite the buffer
		Scalar3 pos2 = make_scalar3(pos4.x, pos4.y, pos4.z);
		Scalar3 pos  = coef_1 * pos1 + coef_2 * pos2; 
		  
		// read the particle's velocity and update position
		float ux = d_Velocity[ 6*idx ];
		float uy = d_Velocity[ 6*idx + 1 ];
		float uz = d_Velocity[ 6*idx + 2 ];
		Scalar3 vel = make_scalar3( ux, uy, uz);

	        // Add the shear
	        vel.x += shear_rate * pos.y;

		// Move the positions	
		Scalar3 dx = vel * dt * coef_3;
		pos += dx;
		
		// Read in particle's image and wrap periodic boundary
		int3 image = d_image[idx];
		box.wrap(pos, image);
		
		// write out the results
		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, 0.0); //last entry doesn't matter
		d_image[idx] = image;
	}
}






extern "C" __global__ void Integrator_buffer_vel_kernel(float *d_Velocity,
							Scalar3 *vel_rk,
							unsigned int *d_group_members,
							unsigned int group_size
							){

	// Thread ID
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check that thread is in bounds
	if ( tidx < group_size ){

		// Particle ID
		unsigned int idx = d_group_members[tidx];
		  
		// read the particle's velocity 
		float ux = d_Velocity[ 6*idx ];
		float uy = d_Velocity[ 6*idx + 1 ];
		float uz = d_Velocity[ 6*idx + 2 ];
		
		// buffer the velocity 
		vel_rk[idx] = make_scalar3( ux, uy, uz);

	}
}


extern "C" __global__ void Integrator_supim_kernel(Scalar3 *vel_rk1,  
						   Scalar3 *vel_rk2,  
						   Scalar3 *vel_rk3,  
						   Scalar3 *vel_rk4,
						   float *d_Velocity,  //output
						   unsigned int *d_group_members,
						   unsigned int group_size
						   ){

	// Thread ID
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check that thread is in bounds
	if ( tidx < group_size ){

		// Particle ID
		unsigned int idx = d_group_members[tidx];

		// read the particle's velocities
		Scalar3 vel1 = vel_rk1[idx];
		Scalar3 vel2 = vel_rk2[idx];
		Scalar3 vel3 = vel_rk3[idx];
		Scalar3 vel4 = vel_rk4[idx];
		
		// superimpose the velocities
		Scalar3 vel = vel1 + 2.*vel2 + 2.*vel3 + vel4;

		d_Velocity[ 6*idx ]     = vel.x;    
		d_Velocity[ 6*idx + 1 ] = vel.y;
		d_Velocity[ 6*idx + 2 ] = vel.z;
	}
}


/*! 
	Integrates particle position according to the 2nd-order Adam-Bathforth scheme, with shear
   
	timestep                (input)  current timestep
	d_vel                   (input)  velocity at the previous time level (if timestep > 0)
	d_pos_in		(input)  3Nx1 particle positions at initial point
	d_pos_out		(output) 3Nx1 new particle positions
	d_Velocity		(input)  6Nx1 generalized particle velocities
	d_image			(input)  particle periodic images
	d_group_members		(input)  indices of the mebers of the group to integrate
	group_size		(input)  Number of members in the group
	box Box			(input)  dimensions for periodic boundary condition handling
	dt			(input)  timestep
	shear_rate		(input)  shear rate for the system
	
	This kernel must be executed with a 1D grid of any block size such that the number of threads is greater than or
	equal to the number of members in the group. The kernel's implementation simply reads one particle in each thread
	and updates that particle. 
*/

extern "C" __global__ void Integrator_AB2_Shear_kernel(unsigned int timestep, Scalar4 *d_vel,
						       Scalar4 *d_pos_in,
						       Scalar4 *d_pos_out,
                             			       float *d_Velocity,
                             			       int3 *d_image,
                             			       unsigned int *d_group_members,
                             			       unsigned int group_size,
                             			       BoxDim box,
                             			       Scalar dt,
						       Scalar shear_rate
						       ){

	// Thread ID
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check that thread is in bounds
	if ( tidx < group_size ){

		// Particle ID
		unsigned int idx = d_group_members[tidx];
		
		// read the particle's posision
		Scalar4 pos4 = d_pos_in[idx];
		Scalar3 pos = make_scalar3(pos4.x, pos4.y, pos4.z);
		
		// read the particle's current velocity 
		float ux1 = d_Velocity[ 6*idx ];
		float uy1 = d_Velocity[ 6*idx + 1 ];
		float uz1 = d_Velocity[ 6*idx + 2 ];
		
		// read the particle's previous velocity
		float ux0,uy0,uz0;
		if (timestep == 1) {
		  ux0 = ux1;
		  uy0 = uy1;
		  uz0 = uz1;
		} else {
		  ux0 = d_vel[ idx ].x;
		  uy0 = d_vel[ idx ].y;
		  uz0 = d_vel[ idx ].z;
		}

		// the velocity to update positions
		float ux = 1.5*ux1 - 0.5*ux0;  
		float uy = 1.5*uy1 - 0.5*uy0; 
		float uz = 1.5*uz1 - 0.5*uz0; 
		
		Scalar3 vel = make_scalar3( ux, uy, uz);

	        // Add the shear
	        vel.x += shear_rate * pos.y;

		// Move the positions	
		Scalar3 dx = vel * dt;
		pos += dx;
		
		// Read in particle's image and wrap periodic boundary
		int3 image = d_image[idx];
		box.wrap(pos, image);
		
		// write out the results
		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, pos4.w);
		d_image[idx] = image;
	}
}


/*! 
	Random Finite Differencing to compute the divergence of inverse(RFU)

	d_Divergence		(output) 11Nx1 divergence of RFU (first 6N) and RSU (last 5N)
	d_pos			(input)  particle positions
	d_image			(input)  particle periodic image
	d_group_members		(input)  ID of particle within integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	ker_data		(input)  structure containing kernel launch information
	bro_data		(input)  structure containing Brownian calculation information
	mob_data		(input)  structure containing mobility calculation information
	res_data		(input)  structure containing lubrication calculation information

*/
void Integrator_RFD(
			float *d_Divergence, // size=11*N, but only filled from 1:6*N
			Scalar4 *d_pos,
			int3 *d_image,
			unsigned int *d_group_members,
			unsigned int group_size,
			const BoxDim& box,
			void *pBuffer,
			KernelData *ker_data,
			BrownianData *bro_data,
			MobilityData *mob_data,
			ResistanceData *res_data,
			WorkData *work_data
			){
	
	// Displacements for central RFD are (+/-) epsilon/2.0
	float epsilon = bro_data->rfd_epsilon;

	// Get kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Random Vectors
	Scalar  *d_psi = (work_data->saddle_psi);
	Scalar4 *d_posPrime = (work_data->saddle_posPrime);

	// RHS and Solution Vectors
	Scalar *d_rhs = (work_data->saddle_rhs);
	Scalar *d_solution = (work_data->saddle_solution);

	Saddle_ZeroOutput_kernel<<<grid,threads>>>( d_rhs, group_size );
	Saddle_ZeroOutput_kernel<<<grid,threads>>>( d_solution, group_size );
	
	// Generate random variables for RFD
	Integrator_RFD_RandDisp_kernel<<<grid,threads>>>( d_psi, group_size, bro_data->seed_rfd );
	
	// Copy force to right-hand side vector
	Saddle_force2rhs_kernel<<<grid,threads>>>( d_psi, d_rhs, group_size );
		

	//
	// Solve in the positive direction along the random displacements
	//
	
	// Do the displacements
	Integrator_ExplicitEuler_kernel<<<grid,threads>>>(	
								d_pos,
								d_posPrime,
                             					d_psi,
                             					d_image,
                             					d_group_members,
                             					group_size,
                             					box,
                             					epsilon/2.0
								);

	// Solve the saddle point problem
	Solvers_Saddle(
			d_rhs, 
			d_solution,
			d_posPrime,
			d_group_members,
			group_size,
			box,
			bro_data->tol,
			pBuffer,
			ker_data,
			mob_data,
			res_data,
			work_data
			);
		
	//  Copy velocity to Divergence	
	cudaMemcpy( d_Divergence, &d_solution[11*group_size], 6*group_size*sizeof(float), cudaMemcpyDeviceToDevice );

	// Compute the near-field hydrodynamic stresslet
	Lubrication_RSU_kernel<<< grid, threads >>>(
							&d_Divergence[6*group_size], // output
							d_Divergence, // input
							d_pos,
							d_group_members,
							group_size, 
			      				box,
							res_data->nneigh, 
							res_data->nlist, 
							res_data->headlist, 
							res_data->table_dist,
							res_data->table_vals,
							res_data->table_min,
							res_data->table_dr,
							res_data->rlub
							);

	//
	// Solve in the negative direction along the random displacements
	//

	// Do the displacements
	Integrator_ExplicitEuler_kernel<<<grid,threads>>>(	
								d_pos,
								d_posPrime,
                             					d_psi,
                             					d_image,
                             					d_group_members,
                             					group_size,
                             					box,
                             					-epsilon/2.0
								);


	// Solve the saddle point problem
	Solvers_Saddle(
			d_rhs, 		// input
			d_solution,	// output
			d_posPrime,
			d_group_members,
			group_size,
			box,
			bro_data->tol,
			pBuffer,
			ker_data,
			mob_data,
			res_data,
			work_data
			);

	// Compute the near-field hydrodynamic stresslet
	Lubrication_RSU_kernel<<< grid, threads >>>(
							&d_solution[6*group_size], // output
							&d_solution[11*group_size], // input
							d_pos,
							d_group_members,
							group_size, 
			      				box,
							res_data->nneigh, 
							res_data->nlist, 
							res_data->headlist, 
							res_data->table_dist,
							res_data->table_vals,
							res_data->table_min,
							res_data->table_dr,
							res_data->rlub
							);
	
	// Take the difference and apply the appropriate scaling
	// 
	// Need the -1 because saddle point lower right gives -RFU, not RFU
	float fac = (bro_data->T) / epsilon;
	Saddle_AddFloat_kernel<<<grid,threads>>>( d_Divergence, &d_solution[11*group_size], d_Divergence, fac, -fac, group_size, 6 );	
	Saddle_AddFloat_kernel<<<grid,threads>>>( &d_Divergence[6*group_size], &d_solution[6*group_size], &d_Divergence[6*group_size], fac, -fac, group_size, 5 );	
	
	// Clean up
	d_solution = NULL;
	d_rhs = NULL;
	d_posPrime = NULL;
	d_psi = NULL;

}


/*! 
	Combine all the parts required to compute the particle displacements

	timestep                (input)  current timestep
	output_period           (input)  output per output_period steps
	d_AppliedForce		(input)  6Nx1 particle generalized forces
	d_Velocity		(output) 11Nx1 particle generalized velocities (6N) and stresslets (5N)
	dt			(input)  integration timestep
	shear_rate		(input)	 shear rate for imposed shear flow
	d_pos			(input)  particle positions
	d_image			(input)  particle periodic image
	d_group_members		(input)  ID of particle within integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	ker_data		(input)  structure containing kernel launch information
	bro_data		(input)  structure containing Brownian calculation information
	mob_data		(input)  structure containing mobility calculation information
	res_data		(input)  structure containing lubrication calculation information


*/
void Integrator_ComputeVelocity(     unsigned int timestep,
				     unsigned int output_period,
			
					float *d_AppliedForce,
					float *d_Velocity,
					float dt,
					float shear_rate,
					Scalar4 *d_pos,
					int3 *d_image,
					unsigned int *d_group_members,
					unsigned int group_size,
					const BoxDim& box,
					KernelData *ker_data,
					BrownianData *bro_data,
					MobilityData *mob_data,
					ResistanceData *res_data,
					WorkData *work_data
					){
	
	// Dereference kernel data for grid and threads
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;
	
	// Allocate the buffer space	
	void *pBuffer;
	cudaMalloc( (void**)&pBuffer, res_data->pBufferSize );//zhoge: pBufferSize computed in Precondition_IChol()

	// Zero velocity to start (in Helper_Precondition.cu)
	Precondition_ZeroVector_kernel<<< grid, threads >>>( d_Velocity, 11*group_size, group_size );

	// Get divergence from RFD, and use that to initialize the velocity
	if ( (bro_data->T) > 0.0 ){
		Integrator_RFD(
				d_Velocity,
				d_pos,
				d_image,
				d_group_members,
				group_size,
				box,
				pBuffer,
				ker_data,
				bro_data,
				mob_data,
				res_data,
				work_data
				);
	}
	
	// RHS and Solution Vectors
	Scalar *d_rhs = (work_data->saddle_rhs);
	Scalar *d_solution = (work_data->saddle_solution);
	
	Saddle_ZeroOutput_kernel<<<grid,threads>>>( d_rhs, group_size );
	Saddle_ZeroOutput_kernel<<<grid,threads>>>( d_solution, group_size );
	
	// Compute far-field stochastic slip velocity
	Brownian_FarField_SlipVelocity(	
					d_rhs, // Far-field slip is first 11*N entries of RHS vector
					d_pos,
                                	d_group_members,
                                	group_size,
                                	box,
                                	dt,
					bro_data,
					mob_data,
					ker_data,
					work_data
					);

	// Add the ambient rate of strain to the right-hand side
	//Integrator_AddStrainRate_kernel<<< grid, threads >>>( d_rhs, shear_rate, group_size );
	Integrator_AddStrainRate_kernel<<< grid, threads >>>( d_rhs,
							      -shear_rate,    //correction by madhu
							      group_size ); 

	// Compute the near-field stochastic force
	Brownian_NearField_Force(
					&d_rhs[ 11*group_size ], // Near-field Brownian force is last 6*N entries of RHS vector
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
	
	// Add imposed force to right-hand side vector (zhoge: It's actually minus force.)
	Saddle_force2rhs_kernel<<<grid,threads>>>( d_AppliedForce,
						   d_rhs,
						   group_size );
	
	// Add the near-field FE force to the right-hand side vector, RFE:Einf
	Lubrication_RFE_kernel<<< grid, threads >>>(
							&d_rhs[11*group_size],
							shear_rate,
							d_pos,
							d_group_members,
							group_size, 
			      				box,
							res_data->nneigh, 
							res_data->nlist, 
							res_data->headlist, 
							res_data->table_dist,
							res_data->table_vals,
							res_data->table_min,
							res_data->table_dr,
							res_data->rlub
							);
	
	// Do the saddle point solve
	Solvers_Saddle(
			d_rhs, 
			d_solution,
			d_pos,
			d_group_members,
			group_size,
			box,
			bro_data->tol,
			pBuffer,
			ker_data,
			mob_data,
			res_data,
			work_data
			);
		
	// Get velocity out of solution vector (in Helper_Saddle.cu)
	Saddle_AddFloat_kernel<<<grid,threads>>>( 
							d_Velocity,			// Velocity is entries 1:(6*N)
							&d_solution[11*group_size],	// Velocity is entries (11*N+1):(17*N)
							d_Velocity, 			// Add to self
							1.0, 1.0, group_size, 6 );

	
	// zhoge: Only process stresslets if they are to be written to output files
	if ( ( output_period > 0 ) && ( int(timestep) % output_period == output_period-1 ) ) //-1 because output in the next step
	  {
	    // Get the far-field stresslet out of solution vector
	    Saddle_AddFloat_kernel<<<grid,threads>>>( 
						     &d_Velocity[6*group_size], // stresslet is last 5*N entries
						     &d_solution[6*group_size], // stresslet is entries (6*N+1):(11*N)
						     &d_Velocity[6*group_size], // Add to self
						     1.0,
						     -1.0,                      // zhoge: Because shear rate was minus
						     group_size, 5 );
	
	    // Add the near-field contributions to the stresslet
	    // - RSU_nf * (U-Uinf)
	    Lubrication_RSU_kernel<<< grid, threads >>>(
							&d_Velocity[6*group_size],  // output, stresslet is last 5*N entries
							&d_solution[11*group_size], // input
							d_pos,
							d_group_members,
							group_size, 
			      				box,
							res_data->nneigh, 
							res_data->nlist, 
							res_data->headlist, 
							res_data->table_dist,
							res_data->table_vals,
							res_data->table_min,
							res_data->table_dr,
							res_data->rlub
							);
	    // + RSE_nf : Einf
	    Lubrication_RSE_kernel<<< grid, threads >>>(
							&d_Velocity[6*group_size], // output, stresslet is last 5*N entries
							shear_rate,
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
	  }


	// Clean up
	d_rhs = NULL;
	d_solution = NULL;
	cudaFree( pBuffer );
	
}
