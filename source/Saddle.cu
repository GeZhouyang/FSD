// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore


#include "Saddle.cuh"
#include "Lubrication.cuh"
#include "Precondition.cuh"
#include "Mobility.cuh"
#include "Solvers.cuh"
#include "Wrappers.cuh"

#include "Helper_Debug.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Precondition.cuh"
#include "Helper_Saddle.cuh"

#include <cusparse.h>
#include <cusolverSp.h>

#include <stdio.h>

#include "hoomd/Saru.h"
#include "hoomd/TextureTools.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/version.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <stdlib.h>

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


/*
	Define the saddle point matrix describing Stokesian Dynamics,
	i.e. it describes the relationship Ax=b.
*/


//! Texture for reading table values
scalar4_tex_t tables1_tex;
//! Texture for reading particle positions
scalar4_tex_t pos_tex;

/*! 
	Matrix-vector operation associated with the saddle point matrix solve

	d_b			(output) output of matrix-vector product (a vector of size 17N)
	d_x			(input)  input of matrix-vector product
	d_pos			(input)  positions of the particles, actually they are fetched on texture memory
	d_group_members		(input)  index array to global HOOMD tag on each particle
	group_size		(input)  size of the group, i.e. number of particles
	box			(input)  array containing box dimensions
	ker_data		(input)  structure containing information for kernel launches
	mob_data		(input)  structure containing information for mobility calculations
	res_data		(input)  structure containing information for resistance calculation

*/

//zhoge// Referenced by cuspSaddle in Wrappers.cuh

void Saddle_Multiply( 
                        	float *d_b, // output
				float *d_x, // input
				Scalar4 *d_pos,
				unsigned int *d_group_members,
				unsigned int group_size,
                        	const BoxDim& box,
				KernelData *ker_data,
				MobilityData *mob_data,
				ResistanceData *res_data,
				WorkData *work_data
				){
	
	// Kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;
	
	// Set output to zero to start (size 17N)
	Saddle_ZeroOutput_kernel<<<grid,threads>>>( d_b, group_size );
	
	// Do the mobility multiplication. Afterwards, d_b[0:11N] = M*F (the generalized velocity).
	Mobility_GeneralizedMobility( 	
	    				d_b, //output
	    				d_x, //input to this function, but actually it's the unknown in FSD
					d_pos,
	    				d_group_members,
	    				group_size,
	    				box,
					ker_data,
					mob_data,
					work_data
					);
	//// Turn off far-field interactions, i.e. d_b[0:11N] = d_x[0:11N]
	//Saddle_AddFloat_kernel<<<grid,threads>>>( d_b, d_x, d_b, 0.0, 1.0, group_size, 11 );
	
	// M*F + B*U, i.e. d_b[0:6N] += U (= d_x[11N:17N])
	Saddle_AddFloat_kernel<<<grid,threads>>>( d_b, &d_x[11*group_size], d_b, 1.0, 1.0, group_size, 6 );
	
	// Do the resistance multiplication. Afterwards, d_b[11N:17N] = R_FU^nf*U (the minus sign is accounted for later).
	Lubrication_RFU_kernel<<<grid,threads>>>(
							&d_b[11*group_size], // output
							&d_x[11*group_size], // input
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
	
	// B^T * F - RFU * U, i.e. d_b[11N:17N] = F[0:6N] - d_b[11N:17N]
	Saddle_AddFloat_kernel<<<grid,threads>>>( d_x, &d_b[11*group_size], &d_b[11*group_size], 1.0, -1.0, group_size, 6 );
	
}



/*!
	Matrix-vector operation for saddle point preconditioner
		x = P \ b

	(zhoge: P \ b means P^-1 * b)

	!!! In order for this to work with cusp, the operator must be
	    able to do the linear transformation in place! (gmres.inl line 143 in CUSP)
	
	d_x			(output) Solution of preconditioner
	d_b			(input)  RHS of preconditioner solve
	group_size		(input)  size of the group, i.e. number of particles
	ker_data		(input)  structure containing information for kernel launches
	res_data		(input)  structure containing information for resistance calculation

*/
void Saddle_Preconditioner(	
				float *d_x, 		// output
				float *d_b, 		// input
				int group_size,
				void *pBuffer,
				KernelData *ker_data,
				ResistanceData *res_data
				){

	// Get kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Get pointer to scratch array (size 17N)
	float *d_Scratch = res_data->Scratch2;	

	//
	// In the preconditioner, M is approximated as identity
	// zhoge: Simply copy the first 11N entries from d_b (the previous RHS vector) to d_Scratch.
	cudaMemcpy( d_Scratch, d_b, 11*group_size*sizeof(float), cudaMemcpyDeviceToDevice );
	//cudaMemcpy( d_Scratch, d_b, 17*group_size*sizeof(float), cudaMemcpyDeviceToDevice );//night
	
	//
	// Incomplete Cholesky solves (done in place!)
	//

	// RFU^(-1) * B^T * [ U; E ]
	// zhoge: Theoretically no effect (B^T truncates up to 6N, all zeros), practically may have residuals.
	Precondition_Saddle_RFUmultiply(	
					&d_Scratch[11*group_size], // output
					d_Scratch,                 // input
					res_data->Scratch1,        // intermediate storage
					res_data->prcm,
					group_size,
					res_data->nnz,
					res_data->L_RowPtr,
					res_data->L_ColInd,
					res_data->L_Val,
					res_data->spHandle,
        				res_data->spStatus,
					res_data->descr_L,
					res_data->info_L,
					res_data->info_Lt,
					res_data->trans_L,
					res_data->trans_Lt,
					res_data->policy_L,
					res_data->policy_Lt,
					pBuffer,
					ker_data->particle_grid,
					ker_data->particle_threads
					);
	
	Saddle_AddFloat_kernel<<<grid,threads>>>( d_Scratch, &d_Scratch[11*group_size], d_Scratch, 1.0, -1.0, group_size, 6 );
	
	// RFU^(-1) * Fnf
	// actually -RFU^-1 * Fnf
	Precondition_Saddle_RFUmultiply(	
					d_b,                 // output (overwrites, but doesn't matter)
					&d_b[11*group_size], // input
					res_data->Scratch1,  // intermediate storage
					res_data->prcm,
					group_size,
					res_data->nnz,
					res_data->L_RowPtr,
					res_data->L_ColInd,
					res_data->L_Val,
					res_data->spHandle,
        				res_data->spStatus,
					res_data->descr_L,
					res_data->info_L,
					res_data->info_Lt,
					res_data->trans_L,
					res_data->trans_Lt,
					res_data->policy_L,
					res_data->policy_Lt,
					pBuffer,
					ker_data->particle_grid,
					ker_data->particle_threads
					);
	
	Saddle_AddFloat_kernel<<<grid,threads>>>( d_b, d_Scratch, d_Scratch, 1.0,  1.0, group_size, 6 );
	Saddle_AddFloat_kernel<<<grid,threads>>>( &d_Scratch[11*group_size], d_b, &d_Scratch[11*group_size], 1.0, -1.0, group_size, 6);
	
	cudaMemcpy( d_x, d_Scratch, 17*group_size*sizeof(float), cudaMemcpyDeviceToDevice );
	
	// Clean up
	d_Scratch = NULL;
		
}
