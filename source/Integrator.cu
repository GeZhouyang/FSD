// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Modified by Zhouyang Ge

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

#include <curand.h>
#include <cuda_runtime.h>

#include <cusparse.h>
#include <cusolverSp.h>

#include <stdio.h>
#include <math.h>

#include "lapacke.h"
#include "cublas_v2.h"

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
	Integrates particle position according to the Explicit Euler scheme
   
	d_pos_in		(input)  3Nx1 particle positions at initial point
	d_pos_out		(output) 3Nx1 new particle positions
	d_Velocity		(input)  6Nx1 generalized particle velocities
	d_image			(input)  particle periodic images (zhoge: Redundent in the present code)
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
		
		// Update the positions	
		Scalar3 dx = vel * dt;
		pos += dx;
		
		// Read in particle's image and wrap periodic boundary
		int3 image = d_image[idx];
		box.wrap(pos, image);
		
		// write out the results
		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, pos4.w);
		//d_image[idx] = image;  //redundent (zhoge)

	}
}

/*! 
	Integrates particle position according to the Explicit Euler scheme, with shear
   
	d_pos_in		(input)  3Nx1 particle positions at initial point
	d_pos_out		(output) 3Nx1 new particle positions
	d_ori                   (input/output) 4Nx1 particle orientations
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
								 Scalar3 *d_ori_in,
								 Scalar4 *d_pos_out,
								 Scalar3 *d_ori_out,
								 Scalar4 *d_pos_gb,
								 float *d_Velocity,
								 float B1,
								 float *d_sqm_B1_mask,
								 Scalar3 *d_noise_ang,
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
		  
		// read the particle's velocities
		float ux = d_Velocity[ 6*idx     ];
		float uy = d_Velocity[ 6*idx + 1 ];
		float uz = d_Velocity[ 6*idx + 2 ];
		float wx = d_Velocity[ 6*idx + 3 ];
		float wy = d_Velocity[ 6*idx + 4 ];
		float wz = d_Velocity[ 6*idx + 5 ];
		
		Scalar3 vel = make_scalar3( ux, uy, uz);
		Scalar3 omg = make_scalar3( wx, wy, wz);
		
	        // Add the shear
	        vel.x += shear_rate * pos.y;
		omg.z -= shear_rate/2.;

		// Add noise
		omg += d_noise_ang[idx];
		
		// Form the unit quaternion for rotation
		Scalar  q0 = 1.0;                     //real part (default)
		Scalar3 qv = make_scalar3(0.,0.,0.);  //imag part (default)
		Scalar abs_omg = sqrtf(dot(omg,omg));
		if (abs_omg > 1e-6) {                 //if rotate
		  Scalar3 axi = omg/abs_omg;          //axis of rotation
		  Scalar angm = abs_omg * dt;         //magnitude
		  q0 = cos(angm/2.);                  //real part
		  qv = axi * sin(angm/2.);            //imag part
		}

		// Minus the active slip
		Scalar3 pv = d_ori_in[idx];
		vel -= (-2./3.*B1*d_sqm_B1_mask[idx])*pv;

		
		// Increment the global position before updating the periodic position 
		Scalar4 pos_gb = d_pos_gb[idx];
		//times dt because using Euler-Maruyama for brownian
		pos_gb.x += (vel.x - shear_rate*pos.y + shear_rate*d_pos_gb[idx].y) * dt;  //use the global y for shear
		pos_gb.y +=  vel.y * dt;
		pos_gb.z +=  vel.z * dt;
		d_pos_gb[idx] = pos_gb;  
		

		// Update the positions	
		Scalar3 dx = vel * dt;
		pos += dx;
		
		// Read in particle's image and wrap periodic boundary 
		int3 image = d_image[idx]; 
		box.wrap(pos, image); 
		
		// write out the results
		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, pos4.w); //last entry doesn't matter
		d_image[idx] = image;  //redundent (zhoge)
		
		

		// Rotate the director, pv_new = qv*pv*qv^*
		Scalar3 pv_new = (q0*q0-dot(qv,qv))*pv;
		pv_new += 2.*dot(qv,pv)*qv;
		Scalar3 qcp = make_scalar3( qv.y*pv.z-qv.z*pv.y,
					    qv.z*pv.x-qv.x*pv.z,
					    qv.x*pv.y-qv.y*pv.x);
		pv_new += 2.*q0*qcp;
		Scalar pvm = sqrtf( dot(pv_new,pv_new) );
		pv_new = pv_new/pvm;  //make it a unit vector

		// Update the orientation
		d_ori_out[idx] = pv_new;
		
		
	}
}


//zhoge: Step 1 of a second-order Runge-Kutta scheme
extern "C" __global__ void Integrator_ExplicitEuler1_Shear_kernel(Scalar4 *d_pos_in,
								  Scalar3 *d_ori_in,
								  Scalar4 *d_pos_out,
								  Scalar3 *d_ori_out,
								  Scalar4 *d_pos_gb,
								  float *d_Velocity,
								  float B1,
								  float *d_sqm_B1_mask,
								  Scalar3 *d_noise_ang,
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
		  
		// read the particle's velocities
		float ux = d_Velocity[ 6*idx     ];
		float uy = d_Velocity[ 6*idx + 1 ];
		float uz = d_Velocity[ 6*idx + 2 ];
		float wx = d_Velocity[ 6*idx + 3 ];
		float wy = d_Velocity[ 6*idx + 4 ];
		float wz = d_Velocity[ 6*idx + 5 ];
		
		Scalar3 vel = make_scalar3( ux, uy, uz);
		Scalar3 omg = make_scalar3( wx, wy, wz);
		
	        // Add the shear
	        vel.x += shear_rate * pos.y;
		omg.z -= shear_rate/2.;

		// Add noise
		omg += d_noise_ang[idx];
		
		// Form the unit quaternion for rotation
		Scalar  q0 = 1.0;                     //real part (default)
		Scalar3 qv = make_scalar3(0.,0.,0.);  //imag part (default)
		Scalar abs_omg = sqrtf(dot(omg,omg));
		if (abs_omg > 1e-6) {                 //if rotate
		  Scalar3 axi = omg/abs_omg;          //axis of rotation
		  Scalar angm = abs_omg * dt;         //magnitude
		  q0 = cos(angm/2.);                  //real part
		  qv = axi * sin(angm/2.);            //imag part
		}

		// Minus the active slip
		Scalar3 pv = d_ori_in[idx];
		vel -= (-2./3.*B1*d_sqm_B1_mask[idx])*pv;

		
		// Increment the global position before updating the periodic position (0.5*dt because it's a half step)
		Scalar4 pos_gb = d_pos_gb[idx];
		pos_gb.x += (vel.x - shear_rate*pos.y + shear_rate*d_pos_gb[idx].y) * 0.5*dt;  //use the global y for shear
		pos_gb.y +=  vel.y * 0.5*dt;
		pos_gb.z +=  vel.z * 0.5*dt;
		d_pos_gb[idx] = pos_gb;  
		

		// Update the positions	(zhoge: No wrapping back to box because of averaging later)
		Scalar3 dx = vel * dt;
		pos += dx;
		
		// write out the results
		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, pos4.w); //last entry doesn't matter
		

		// Rotate the director, pv_new = qv*pv*qv^*
		Scalar3 pv_new = (q0*q0-dot(qv,qv))*pv;
		pv_new += 2.*dot(qv,pv)*qv;
		Scalar3 qcp = make_scalar3( qv.y*pv.z-qv.z*pv.y,
					    qv.z*pv.x-qv.x*pv.z,
					    qv.x*pv.y-qv.y*pv.x);
		pv_new += 2.*q0*qcp;
		Scalar pvm = sqrtf( dot(pv_new,pv_new) );
		pv_new = pv_new/pvm;  //make it a unit vector

		// Update the orientation
		d_ori_out[idx] = pv_new;
		
		
	}
}

//zhoge: Step 2 of a second-order Runge-Kutta scheme
extern "C" __global__ void Integrator_RK_Shear_kernel(Scalar coef_1, Scalar4 *d_pos_in_1, Scalar3 *d_ori_in_1,
						      Scalar coef_2, Scalar4 *d_pos_in_2, Scalar3 *d_ori_in_2,
						      Scalar4 *d_pos_out, Scalar3 *d_ori_out,
						      Scalar4 *d_pos_gb,
						      float *d_Velocity,
						      float B1,
						      float *d_sqm_B1_mask,
						      Scalar3 *d_noise_ang,
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
		  
		// read the particle's velocities
		float ux = d_Velocity[ 6*idx     ];
		float uy = d_Velocity[ 6*idx + 1 ];
		float uz = d_Velocity[ 6*idx + 2 ];
		float wx = d_Velocity[ 6*idx + 3 ];
		float wy = d_Velocity[ 6*idx + 4 ];
		float wz = d_Velocity[ 6*idx + 5 ];
		
		Scalar3 vel = make_scalar3( ux, uy, uz);
		Scalar3 omg = make_scalar3( wx, wy, wz);
		
	        // Add the shear
	        vel.x += shear_rate * pos.y;
		omg.z -= shear_rate/2.;

		// Add noise
		omg += d_noise_ang[idx];
		
		// Form the unit quaternion for rotation
		Scalar  q0 = 1.0;                     //real part (default)
		Scalar3 qv = make_scalar3(0.,0.,0.);  //imag part (default)
		Scalar abs_omg = sqrtf(dot(omg,omg));
		if (abs_omg > 1e-6) {                 //if rotate
		  Scalar3 axi = omg/abs_omg;          //axis of rotation
		  Scalar angm = abs_omg * dt * coef_3;//magnitude
		  q0 = cos(angm/2.);                  //real part
		  qv = axi * sin(angm/2.);            //imag part
		}
		
		// Read the particle director (average of the two previous directions)
		Scalar3 pv1 = d_ori_in_1[idx];
		Scalar3 pv2 = d_ori_in_2[idx];
		Scalar3 pv  = pv1*coef_1 + pv2*coef_2;
		Scalar pvm0 = sqrtf( dot(pv,pv) );
		pv = pv/pvm0;  //make it a unit vector

		// Minus the active slip
		vel -= (-2./3.*B1*d_sqm_B1_mask[idx])*pv;

		
		// Increment the global position before updating the periodic position (0.5*dt because it's a half step)
		Scalar4 pos_gb = d_pos_gb[idx];
		pos_gb.x += (vel.x - shear_rate*pos.y + shear_rate*d_pos_gb[idx].y) * 0.5*dt;  //use the global y for shear
		pos_gb.y +=  vel.y * 0.5*dt;
		pos_gb.z +=  vel.z * 0.5*dt;
		d_pos_gb[idx] = pos_gb;  
		

		
		// Move the positions	
		Scalar3 dx = vel * dt * coef_3;
		pos += dx;
		
		// Read in particle's image and wrap periodic boundary
		int3 image = d_image[idx];
		box.wrap(pos, image);  //this does the wrapping
		
		// write out the results
		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, 0.0); //last entry doesn't matter
		d_image[idx] = image;  //redundent (zhoge)


		// Rotate the director, pv_new = qv*pv*qv^*
		Scalar3 pv_new = (q0*q0-dot(qv,qv))*pv;
		pv_new += 2.*dot(qv,pv)*qv;
		Scalar3 qcp = make_scalar3( qv.y*pv.z-qv.z*pv.y,
					    qv.z*pv.x-qv.x*pv.z,
					    qv.x*pv.y-qv.y*pv.x);
		pv_new += 2.*q0*qcp;
		Scalar pvm = sqrtf( dot(pv_new,pv_new) );
		pv_new = pv_new/pvm;  //make it a unit vector for regularity

		// Update the orientation
		d_ori_out[idx] = pv_new; 
		
	}
}



//
///*! 
//	Integrates particle position according to the 2nd-order Adam-Bathforth scheme, with shear
//   
//	timestep                (input)  current timestep
//	d_vel                   (input)  velocity at the previous time level (if timestep > 0)
//	d_pos_in		(input)  3Nx1 particle positions at initial point
//	d_pos_out		(output) 3Nx1 new particle positions
//	d_Velocity		(input)  6Nx1 generalized particle velocities
//	d_image			(input)  particle periodic images
//	d_group_members		(input)  indices of the mebers of the group to integrate
//	group_size		(input)  Number of members in the group
//	box Box			(input)  dimensions for periodic boundary condition handling
//	dt			(input)  timestep
//	shear_rate		(input)  shear rate for the system
//	
//	This kernel must be executed with a 1D grid of any block size such that the number of threads is greater than or
//	equal to the number of members in the group. The kernel's implementation simply reads one particle in each thread
//	and updates that particle. 
//*/
//
//extern "C" __global__ void Integrator_AB2_Shear_kernel(unsigned int timestep, Scalar4 *d_vel,
//						       Scalar4 *d_pos_in,
//						       Scalar4 *d_pos_out,
//                             			       float *d_Velocity,
//                             			       int3 *d_image,
//                             			       unsigned int *d_group_members,
//                             			       unsigned int group_size,
//                             			       BoxDim box,
//                             			       Scalar dt,
//						       Scalar shear_rate
//						       ){
//
//	// Thread ID
//	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
//	
//	// Check that thread is in bounds
//	if ( tidx < group_size ){
//
//		// Particle ID
//		unsigned int idx = d_group_members[tidx];
//		
//		// read the particle's posision
//		Scalar4 pos4 = d_pos_in[idx];
//		Scalar3 pos = make_scalar3(pos4.x, pos4.y, pos4.z);
//		
//		// read the particle's current velocity 
//		float ux1 = d_Velocity[ 6*idx ];
//		float uy1 = d_Velocity[ 6*idx + 1 ];
//		float uz1 = d_Velocity[ 6*idx + 2 ];
//		
//		// read the particle's previous velocity
//		float ux0,uy0,uz0;
//		if (timestep == 1) {
//		  ux0 = ux1;
//		  uy0 = uy1;
//		  uz0 = uz1;
//		} else {
//		  ux0 = d_vel[ idx ].x;
//		  uy0 = d_vel[ idx ].y;
//		  uz0 = d_vel[ idx ].z;
//		}
//
//		// the velocity to update positions
//		float ux = 1.5*ux1 - 0.5*ux0;  
//		float uy = 1.5*uy1 - 0.5*uy0; 
//		float uz = 1.5*uz1 - 0.5*uz0; 
//		
//		Scalar3 vel = make_scalar3( ux, uy, uz);
//
//	        // Add the shear
//	        vel.x += shear_rate * pos.y;
//
//		// Move the positions	
//		Scalar3 dx = vel * dt;
//		pos += dx;
//		
//		// Read in particle's image and wrap periodic boundary
//		int3 image = d_image[idx];
//		box.wrap(pos, image);
//		
//		// write out the results
//		d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, pos4.w);
//		d_image[idx] = image;
//	}
//}
//

/*! 
	Random Finite Differencing to compute the divergence of inverse(RFU)

	zhoge: Now only update the first 6N entries corresponding to the Brownian drift.
               The associated stress is computed together with the rest of velocities.
	       Note that, since we use RSU^nf later, the stress might be slightly off. (To check)

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
void Integrator_RFD( float *d_Divergence, //size 11N, but only the first 6N are modified
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
		     WorkData *work_data )
{
  // cuBLAS handle
  cublasHandle_t blasHandle = work_data->blasHandle;

  // Get kernel information
  dim3 grid    = ker_data->particle_grid;
  dim3 threads = ker_data->particle_threads;
	
  // Displacements for central RFD are (+/-) epsilon/2.0 * d_psi
  float epsilon = bro_data->rfd_epsilon;  //same as solver tolerance

  unsigned int numel = 17*group_size;

  // Random Vectors
  Scalar  *d_psi      = work_data->saddle_psi;
  Scalar4 *d_posPrime = work_data->saddle_posPrime;

  // RHS and Solution Vectors
  Scalar *d_rhs = bro_data->rfd_rhs;
  Scalar *d_sol = bro_data->rfd_sol;

  // Zero out the rhs and solution vectors
  float scale = 0.0;  
  cublasSscal( blasHandle, numel, &scale, d_rhs, 1 );
  cublasSscal( blasHandle, numel, &scale, d_sol, 1 );
	
  //// Generate random variables for RFD
  //Integrator_RFD_RandDisp_kernel<<<grid,threads>>>( d_psi, group_size, bro_data->seed_rfd );
	
  //zhoge: use cuRand to generate Gaussian variables
  curandGenerator_t gen;
  unsigned int N_random = 6*group_size;                           //6 because force (3) and torque (3)
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);   //fastest      
  curandSetPseudoRandomGeneratorSeed(gen, bro_data->seed_rfd);    //set the seed (different from the others)
  curandGenerateNormal(gen, d_psi, N_random, 0.0f, 1.0f);         //mean 0, std 1
  curandDestroyGenerator(gen);  

  // Copy force to right-hand side vector (substracted as it should be)  
  scale = -1.0;
  cublasSaxpy( blasHandle, N_random, &scale, d_psi, 1, &d_rhs[11*group_size], 1 );  //d_rhs[11N:17N] -= d_psi[0:6N]

  //
  // Solve in the positive direction along the random displacements
  //
  // Do the displacements (zhoge: no need to update quaterians since this is a fake update)
  Integrator_ExplicitEuler_kernel<<<grid,threads>>>( d_pos,
						     d_posPrime,  //output
						     d_psi,       //input: random Gaussian
						     d_image,
						     d_group_members,
						     group_size,
						     box,
						     epsilon/2.0 ); //input: forward solve
							
  // Solve the saddle point problem
  Solvers_Saddle( d_rhs, 
		  d_sol,  //output (contains U+)
		  d_posPrime,
		  d_group_members,
		  group_size,
		  box,
		  bro_data->tol,
		  pBuffer,
		  ker_data,
		  mob_data,
		  res_data,
		  work_data );
		
  //  Copy velocity to Divergence	
  cudaMemcpy( d_Divergence, &d_sol[11*group_size], 6*group_size*sizeof(float), cudaMemcpyDeviceToDevice );

  //// Compute the near-field hydrodynamic stresslet (zhoge: No need to do it separately. Moved out.)
  //Lubrication_RSU_kernel<<< grid, threads >>>(
  //						&d_Divergence[6*group_size], // output
  //						d_Divergence, // input
  //						d_pos,
  //						d_group_members,
  //						group_size, 
  //		      				box,
  //						res_data->nneigh, 
  //						res_data->nlist, 
  //						res_data->headlist, 
  //						res_data->table_dist,
  //						res_data->table_vals,
  //						res_data->table_min,
  //						res_data->table_dr,
  //						res_data->rlub
  //						);

  //
  // Solve in the negative direction along the random displacements
  //
  // Do the displacements
  Integrator_ExplicitEuler_kernel<<<grid,threads>>>( d_pos,
						     d_posPrime,   //output
						     d_psi,        //input: random Gaussian (same as before)
						     d_image,
						     d_group_members,
						     group_size,
						     box,
						     -epsilon/2.0 ); //input: backward solve
						       
  // Solve the saddle point problem
  Solvers_Saddle( d_rhs,   //input (same as before)
		  d_sol,   //output (contains U-)
		  d_posPrime,
		  d_group_members,
		  group_size,
		  box,
		  bro_data->tol,
		  pBuffer,
		  ker_data,
		  mob_data,
		  res_data,
		  work_data );

  //// Compute the near-field hydrodynamic stresslet (zhoge: No need to do it separately. Moved out.)
  //Lubrication_RSU_kernel<<< grid, threads >>>(
  //						&d_solution[6*group_size], // output
  //						&d_solution[11*group_size], // input
  //						d_pos,
  //						d_group_members,
  //						group_size, 
  //		      				box,
  //						res_data->nneigh, 
  //						res_data->nlist, 
  //						res_data->headlist, 
  //						res_data->table_dist,
  //						res_data->table_vals,
  //						res_data->table_min,
  //						res_data->table_dr,
  //						res_data->rlub
  //						);
	
  // Take the difference and apply the appropriate scaling
  float fac = (bro_data->T) / epsilon;  
  Saddle_AddFloat_kernel<<<grid,threads>>>( d_Divergence,           //input (U+)
					    &d_sol[11*group_size],  //input (U-)
					    d_Divergence,           //output
					    fac,
					    -fac,
					    group_size, 6 );
	
  //zhoge: No need to do it separately. Moved out.
  //Saddle_AddFloat_kernel<<<grid,threads>>>( &d_Divergence[6*group_size],
  //					  &d_solution[6*group_size],
  //					  &d_Divergence[6*group_size],  //output
  //					  fac,
  //					  -fac,
  //					  group_size, 5 );	
	
  // Clean up
  d_sol = NULL;  
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
	sqm_B2                  (input)  B2 mode coef (spherical squirmers)
	d_ori                   (input)  particle orientations
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
				     float sqm_B2,
				     float *d_sqm_B2_mask,
				     Scalar3 *d_ori,
				     int3 *d_image,
				     unsigned int *d_group_members,
				     unsigned int group_size,
				     const BoxDim& box,
				     KernelData *ker_data,
				     BrownianData *bro_data,
				     MobilityData *mob_data,
				     ResistanceData *res_data,
				     WorkData *work_data )
{
  // cuBLAS handle
  cublasHandle_t blasHandle = work_data->blasHandle;
	
  // Dereference kernel data for grid and threads
  dim3 grid    = ker_data->particle_grid;
  dim3 threads = ker_data->particle_threads;
	
  // Allocate the buffer space	
  void *pBuffer;
  cudaMalloc( (void**)&pBuffer, res_data->pBufferSize );//zhoge: pBufferSize computed in Precondition_IChol()
	
  // Set up RHS and solution vectors
  Scalar *d_rhs      = work_data->saddle_rhs;
  Scalar *d_solution = work_data->saddle_solution;  //keep it as the initial guess for the current saddle (boyuan)

  // Zero RHS and velocity to start
  float scale = 0.0;  
  cublasSscal( blasHandle, 11*group_size, &scale, d_Velocity, 1 );
  cublasSscal( blasHandle, 17*group_size, &scale, d_rhs,      1 );

  // Brownian calculations
  if ( bro_data->T > 0.0 )
    {
      // Compute the far-field generalized Brownian velocity (including the self part) and add to the RHS
      // d_rhs[0:11N] = \mathcal{U}_B^ff
      Brownian_FarField_SlipVelocity( d_rhs, //output
				      d_pos,
				      d_group_members,
				      group_size,
				      box,
				      dt,
				      bro_data,
				      mob_data,
				      ker_data,
				      work_data );
	    
      // Compute the near-field Brownian (lubrication) force/torque and add (or minus, doesn't matter) to the RHS
      // d_rhs[11N:17N] = F_B^nf
      Brownian_NearField_Force( &d_rhs[ 11*group_size ], //output
      			        d_pos,
      			        d_group_members,
      			        group_size,
      			        box,
      			        dt,
      			        pBuffer,
      			        ker_data,
      			        bro_data,
      			        res_data,
      			        work_data );
      	    
      // Compute the Brownian drift and add to the velocity
      // d_Velocity[0:6N] = U_drift
      Integrator_RFD( d_Velocity,  //output (zhoge: Only the first 6N entries are modified)
      		      d_pos,
      		      d_image,  //input (won't be modified)
      		      d_group_members,
      		      group_size,
      		      box,
      		      pBuffer,
      		      ker_data,
      		      bro_data,
      		      mob_data,
      		      res_data,
      		      work_data );
    }

  // Add (E_inf - E_s) to the RHS, modifying d_rhs[6N:11N]
  Integrator_AddStrainRate_kernel<<< grid, threads >>>( d_rhs,
						        shear_rate,     //zhoge: now consistent sign with FSD paper
						        d_group_members,
						        sqm_B2,         //zhoge: input B2 mode coef
						        d_sqm_B2_mask,
						        d_ori,          //zhoge: input orientation (unit vector)
						        group_size ); 
	
  // Add -F^P to the RHS, modifying d_rhs[11N:17N]
  scale = -1.0;
  cublasSaxpy( blasHandle, 6*group_size, &scale, d_AppliedForce, 1, &d_rhs[11*group_size], 1 );
	
  // Add R_FE^nf:(E_s - E_inf) to the RHS, modifying d_rhs[11N:17N]
  Lubrication_RFE_kernel<<< grid, threads >>>( &d_rhs[11*group_size],   //output
					       shear_rate,
					       d_pos,
					       sqm_B2,         //zhoge: input B2 mode coef
					       d_sqm_B2_mask,
					       d_ori,          //zhoge: input orientation (unit vector)
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
					       res_data->rlub );
	
  // Do the saddle point solve (GMRES)
  Solvers_Saddle( d_rhs, 
		  d_solution,  //output (first 6N: ff force/torque, next 5N: ff stress, last 6N: relative velocities)
		  d_pos,
		  d_group_members,
		  group_size,
		  box,
		  bro_data->tol,
		  pBuffer,
		  ker_data,
		  mob_data,
		  res_data,
		  work_data );
		
  // Get velocity out of solution vector, d_Velocity[0:6N] += d_solution[11N:17N]
  scale = 1.0;
  cublasSaxpy( blasHandle, 6*group_size, &scale, &d_solution[11*group_size], 1, d_Velocity, 1 );
	
  // Only process stresslets if they are to be written to output files
  if ( ( output_period > 0 ) && ( int(timestep+1) % output_period == 0 ) ) 
    {
      // Get the far-field stresslet out of solution vector
      //Saddle_AddFloat_kernel<<<grid,threads>>>( 
      //					       &d_Velocity[6*group_size], // stresslet is last 5N entries
      //					       &d_solution[6*group_size], // stresslet is entries 6N:11N
      //					       &d_Velocity[6*group_size], // Add to self
      //					       1.0,          
      //					       1.0,     //zhoge: corrected sign
      //					       group_size, 5 );
      scale = 1.0;
      cublasSaxpy( blasHandle, 5*group_size, &scale, &d_solution[6*group_size], 1, &d_Velocity[6*group_size], 1 );
	
      // Add the near-field contributions to the stresslet
      // - RSU_nf * (U-Uinf)
      Lubrication_RSU_kernel<<< grid, threads >>>(
						  &d_Velocity[6*group_size],  // output, stresslet is last 5*N entries
						  //&d_solution[11*group_size], // input
						  d_Velocity, //input (zhoge: use d_Velocity directly because it contains U_drift)  //to verify !!
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
						  sqm_B2,         //zhoge: input B2 mode coef
						  d_sqm_B2_mask,
						  d_ori,          //zhoge: input orientation (unit vector)
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
  cudaFree( pBuffer );
	
}
