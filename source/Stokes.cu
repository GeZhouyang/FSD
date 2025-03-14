/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander
// Modified by Gang Wang
// Modified by Andrew Fiore
// Modified by Zhouyang Ge

#include "Stokes.cuh"

#include "Integrator.cuh"
#include "Lubrication.cuh"
#include "Mobility.cuh"
#include "Precondition.cuh"
#include "Wrappers.cuh"
#include "Saddle.cuh"

#include "Helper_Debug.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Stokes.cuh"

#include <cusparse.h>
#include <cusolverSp.h>

#include <stdio.h>

#include "hoomd/Saru.h"
#include "hoomd/TextureTools.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! \file Stokes.cu
    \brief Defines GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.cc.
*/

// Texture for reading table values
scalar4_tex_t tables1_tex;

/*! 
	Step one of two-step integrator (step 2 is null) for the overdamped particle dynamics.
	Explicit Euler integration of particle positions given a velocity.

	timestep        (input)         current timestep
	output_period   (input)         output per output_period steps
	d_pos		(input/ouput)	array of particle positions
	d_ori		(input/ouput)	array of particle orientations
	d_net_force	(input)		particle forces
	d_vel		(output)	particle velocities
	d_AppliedForce	(input/output)	Array for force and torque applied on particles
	d_Velocity	(input/output)	Array for linear and angular velocity of particles and stresslets
	dt		(input)		integration time step
	m_error		(input)		calculation error tolerance
	shear_rate	(input)		shear rate in the suspension, if any
	block_size	(input)		number of threads per block for particle-based calculations
	d_image		(input)		array of particle images
	d_group_members	(input)		index of particles within the integration group
	group_size	(input)		number of particles
	box		(input)		periodic box information
	bro_data	(input)		structure containing data for Brownian calculations
	mob_data	(input)		structure containing data for Mobility calculations
	res_data	(input)		structure containing data for lubrication resistance calculations
	work_data	(input)		structure containing data for scratch arrays and workspaces
*/

cudaError_t Stokes_StepOne(     unsigned int timestep,
				unsigned int output_period,
				Scalar4 *d_pos,
				Scalar3 *d_ori,  
				Scalar4 *d_pos_gb,
				//Scalar4 *d_vel,
				//Scalar4 *d_omg,
				float *d_AppliedForce,
				float *d_Velocity,
				Scalar sqm_B1, Scalar sqm_B2,
				float *d_sqm_B1_mask,
				float *d_sqm_B2_mask,
				Scalar rot_diff,
				Scalar3 *d_noise_ang,
				Scalar T_ext,
                        	Scalar dt,
				const float m_error,
				Scalar shear_rate,
                        	unsigned int block_size,
				int3 *d_image,
                        	unsigned int *d_group_members,
                        	unsigned int group_size,
                        	const BoxDim& box,
				BrownianData *bro_data,
				MobilityData *mob_data,
				ResistanceData *res_data,
				WorkData *work_data
				){

	// *******************************************************
	// Pre-calculation setup
	// *******************************************************
	
	// Set up the blocks and threads to run the particle-based kernels
	dim3 grid( (group_size/block_size) + 1, 1, 1 );
	dim3 threads(block_size, 1, 1);

	// Set up the blocks and threads to run the FFT-grid-based kernels	
	unsigned int NxNyNz = (mob_data->Nx) * (mob_data->Ny) * (mob_data->Nz);
	int gridBlockSize = ( NxNyNz > block_size ) ? block_size : NxNyNz;
	int gridNBlock = ( NxNyNz + gridBlockSize - 1 ) / gridBlockSize ; 

	// Initialize values in the data structure for kernel information
	KernelData ker_struct = {grid,
				 threads,
				 gridNBlock,
				 gridBlockSize,
				 NxNyNz};
	KernelData *ker_data = &ker_struct;

	// Bind the real-space Ewald sum table to textured memory	
	// One dimension, Read mode: ElementType(Get what we write)
	tables1_tex.normalized = false; // Not normalized
	tables1_tex.filterMode = cudaFilterModeLinear; // Filter mode: floor of the index
	cudaBindTexture(0, tables1_tex, mob_data->ewald_table, sizeof(Scalar4) * ((mob_data->ewald_n)+1)); 

	// *******************************************************
        // Get sheared grid vectors
	// *******************************************************

        Mobility_SetGridk_kernel<<<gridNBlock,gridBlockSize>>>(mob_data->gridk,  //output
							       mob_data->Nx,	  
							       mob_data->Ny,	  
							       mob_data->Nz,	  
							       NxNyNz,		  
							       box,		  
							       mob_data->xi,	  
							       mob_data->eta);

	// *******************************************************
        // Prepare the preconditioners
	// *******************************************************
	
	// Build preconditioner (only do once, because it should still be
	// sufficiently good for RFD with small displacements)
	// zhoge: It mainly does the incomplete Cholesky factorization of P * (\tilde{R}_FU^nf + relaxer*I) * P^T
	Precondition_Wrap(d_pos,           
			  d_group_members, 
			  group_size,	   
			  box,		   
			  ker_data,	   
			  res_data,
			  work_data);

//	Debug_Lattice_SpinViscosity(mob_data,res_data,ker_data,work_data,d_pos,d_group_members,group_size,box);
//	Debug_Lattice_ShearViscosity(mob_data,res_data,ker_data,work_data,d_pos,d_group_members,group_size,box);
//	cudaUnbindTexture(tables1_tex);
//	gpuErrchk(cudaPeekAtLastError());
//	return cudaSuccess;


	// *******************************************************
        // Solve the hydrodynamic problem and do the integration
	// *******************************************************
	
	// Set applied force equal to net_force from HOOMD (pair potentials, external potentials, etc.)
        Stokes_SetForce_manually_kernel<<<grid,threads>>>(
							  d_pos,           //input
							  d_ori,           //input
							  d_AppliedForce,  //output
							  group_size,
							  d_group_members,
							  res_data->nneigh, 
							  res_data->nlist, 
							  res_data->headlist,
							  res_data->m_ndsr,
							  res_data->m_k_n, 
							  res_data->m_kappa,
							  res_data->m_beta, 
							  res_data->m_epsq,
							  T_ext,
							  box
							  );
		
	// Compute particle velocities from central RFD + Saddle point solve (in Integrator.cu)
	Integrator_ComputeVelocity(timestep, output_period,
				   d_AppliedForce,  
				   d_Velocity,      //output (FSD velocity and stresslet, 11N)
				   dt,		 
				   shear_rate,	 
				   d_pos,	    //input position
				   sqm_B2,
				   d_sqm_B2_mask,
				   d_ori,           //input: orientation
				   d_image,	 
				   d_group_members, 
				   group_size,	 
				   box,		 
				   ker_data,	 
				   bro_data,	 
				   mob_data,	 
				   res_data,	 
				   work_data);

	if ( bro_data->T > 0.0 or rot_diff > 0.0 )  //Euler-Maruyama for stochastic simulations
	  {
	    // Make the displacement
	    Integrator_ExplicitEuler_Shear_kernel<<<grid,threads>>>(d_pos,     //input
								    d_ori,     //input	
								    d_pos,     //output (overwrite)
								    d_ori,     //output (overwrite)
								    d_pos_gb,  //input/output global position (updated)
								    d_Velocity,
								    sqm_B1,
								    d_sqm_B1_mask,
								    d_noise_ang,
								    d_image,
								    d_group_members,
								    group_size,
								    box,
								    dt,
								    shear_rate
								    );
	  }
	else  //Runge-Kutta for deterministic simulations
	  {
	    // RK position storage
	    Scalar4 *pos_rk1 = work_data->pos_rk1;
	    Scalar3 *ori_rk1 = work_data->ori_rk1;

	    // Make an intermediate displacement
	    Integrator_ExplicitEuler1_Shear_kernel<<<grid,threads>>>(d_pos,     //input
								     d_ori,     //input	
								     pos_rk1,   //output
								     ori_rk1,   //output
								     d_pos_gb,  //input/output global position (updated)
								     d_Velocity,
								     sqm_B1,
								     d_sqm_B1_mask,
								     d_noise_ang,
								     d_image,
								     d_group_members,
								     group_size,
								     box,
								     dt,
								     shear_rate
								     );
	  
	
	    // second RK step
	
	    // zhoge: Probably no need to precondition again
	    Precondition_Wrap(pos_rk1,           
			      d_group_members, 
			      group_size,	   
			      box,		   
			      ker_data,	   
			      res_data,	   //input/output (pruned neighbor list)
			      work_data);
	
	    // Get the midstep interparticle force
	    Stokes_SetForce_manually_kernel<<<grid,threads>>>(
							      pos_rk1,         //input
							      ori_rk1,         //input
							      d_AppliedForce,  //output
							      group_size,
							      d_group_members,
							      res_data->nneigh, 
							      res_data->nlist, 
							      res_data->headlist,
							      res_data->m_ndsr,
							      res_data->m_k_n, 
							      res_data->m_kappa,
							      res_data->m_beta, 
							      res_data->m_epsq,
							      T_ext,
							      box
							      );
	
	    // Compute particle velocities from central RFD + Saddle point solve (in Integrator.cu)
	    Integrator_ComputeVelocity(timestep, output_period,
				       d_AppliedForce,  
				       d_Velocity,      //output (FSD velocity and stresslet, 11N)
				       dt/2.,		 
				       shear_rate,	 
				       pos_rk1,         //input position
				       sqm_B2,
				       d_sqm_B2_mask, 
				       ori_rk1,         //input orientation
				       d_image,	 
				       d_group_members, 
				       group_size,	 
				       box,		 
				       ker_data,	 
				       bro_data,	 
				       mob_data,	 
				       res_data,	 
				       work_data);
	
	    // Make the final displacement
	    Scalar coef_1 = 0.5;
	    Scalar coef_2 = 0.5;
	    Scalar coef_3 = 0.5;
	
	    Integrator_RK_Shear_kernel<<<grid,threads>>>(coef_1, d_pos,d_ori,        //input  position/orientation
							 coef_2, pos_rk1,ori_rk1,    //input  position/orientation
							 d_pos,d_ori,                //output position/orientation (overwritten)
							 d_pos_gb,                   //input/output global position (updated)
							 d_Velocity,
							 sqm_B1,
							 d_sqm_B1_mask,
							 d_noise_ang,
							 d_image,
							 d_group_members,
							 group_size,
							 box,
							 coef_3, dt,       
							 shear_rate
							 );
	    
	  }
		
	
	// Clean up
	cudaUnbindTexture(tables1_tex);

	// Error checking
	gpuErrchk(cudaPeekAtLastError());
	
	return cudaSuccess;
}