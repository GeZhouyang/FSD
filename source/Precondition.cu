// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore


#include "Precondition.cuh"
#include "Lubrication.cuh"

#include "Helper_Debug.cuh"
#include "Helper_Precondition.cuh"

#include "rcm.hpp"

#include <stdio.h>
#include <math.h>
#include "hoomd/TextureTools.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/version.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cusparse.h>
#include <cusolverSp.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! Command to convert floats or doubles to integers
#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif


/*
	This file defines functions required to build the Preconditioner
	for the lubrication and saddle point problems,
*/


/*!
	Get the Pruned neighborlist (i.e. contains only particles that are actually within
	the cutoff) and sort it by particle index.

	Function 1 -- Get the pruned number of neighbors 

	d_nneigh_pruned		(output) number of neighbors within the preconditioner cutoff
	group_size		(input)  number of particles
	d_pos			(input)  particle positions
	box			(input)  periodic box information
	d_group_members		(input)  array of particle indices within the integration group
	d_nneigh		(input)  list of number of neighbors for each particle
	d_nneigh_less		(input)  list of number of neighbors for each particle with index less than the particle's index
	d_nlist			(input)  neighborlist array
	d_headlist		(input)  indices into the neighborlist for each particle
	rp			(input)  cutoff radius for the preconditioner

*/
__global__ void Precondition_GetPrunedNneigh_kernel( 	
							unsigned int *d_nneigh_pruned, 
							const unsigned int group_size,
							const Scalar4 *d_pos,
			      				const BoxDim box,
							const unsigned int *d_group_members,
							const unsigned int *d_nneigh, 
							const unsigned int *d_nlist, 
							const unsigned int *d_headlist,
							const Scalar rp
							){

	// Current thread index	
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Value of summation variables for the current thread
	if ( tidx < group_size){

		// Cutoff radius squared
		Scalar rpsq = rp * rp;

		// Particle info for this thread
		unsigned int idx = d_group_members[ tidx ];
		
		unsigned int head_idx = d_headlist[ idx ]; // Location in head array for neighbors of current particle
		unsigned int nneigh = d_nneigh[ idx ];   // Number of neighbors of the nearest particle

		// Position of current particle
		Scalar4 posi = d_pos [ idx ];

		// Figure out how many particles are within the cutoff
		int counter = 1; // start at one to maintain self component
		for ( int ii = 0; ii < nneigh; ii++ ){
			
			// Get the current neighbor index
			unsigned int curr_neigh = d_nlist[ head_idx + ii ];

			// Position of current neighbor
			Scalar4 posj = d_pos[ curr_neigh ];
			
			// Distance between current particle and neighbor
			Scalar3 R = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
			R = box.minImage(R);
			Scalar distSqr = dot(R,R);
	
			// If particle is within cutoff, add to count	
			if ( distSqr < rpsq ){
				counter++;
			}
		}
		d_nneigh_pruned[ idx ] = counter;
	}
}

/*!
	Get the Pruned neighborlist (i.e. contains only particles that are actually within
	the cutoff) and sort it by particle index.

	Function 2 -- Get the pruned and sorted neighborlist for particles within the
			preconditioner cutoff

	Use an insertion sort -- This will be good as long as each particles don't have
		too many neighbors in the list. This will be true for the lubrication
		tensor because the cutoff radius is 4 particle radii. 
	
	d_nneigh_pruned		(input)		number of neighbors within the preconditioner cutoff
	d_nlist_pruned		(input/output) 	sorted neighborlist pruned to neighborlist cutoff
	d_headlist_pruned	(input)		headlist for each particle location in the neighborlist
	d_nneigh_less		(output)	number of particles with index less than each particle
	group_size		(input)		number of particles
	d_pos			(input)		particle positions
	box			(input)		periodic box information
	d_group_members		(input)		array of particle indices within the integration group
	d_nneigh		(input)		list of number of neighbors for each particle
	d_nneigh_less		(input)		list of number of neighbors for each particle with index less than the particle's index
	d_nlist			(input)		neighborlist array
	d_headlist		(input)		indices into the neighborlist for each particle
	rp			(input)		cutoff radius for the preconditioner
	
*/
__global__ void Precondition_GetPrunedNlist_kernel( 	
							unsigned int *d_nneigh_pruned, 
							unsigned int *d_nlist_pruned,
							unsigned int *d_headlist_pruned,
							unsigned int *d_nneigh_less, 
							const unsigned int group_size,
							const Scalar4 *d_pos,
			      				const BoxDim box,
							const unsigned int *d_group_members,
							const unsigned int *d_nneigh, 
							const unsigned int *d_nlist, 
							const unsigned int *d_headlist,
							const Scalar rp
							){

	// Current thread index	
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Value of summation variables for the current thread
	if ( tidx < group_size){

		// Cutoff radius squared
		Scalar rpsq = rp * rp;

		// Particle info for this thread
		unsigned int idx = d_group_members[ tidx ];
		
		unsigned int head_idx = d_headlist[ idx ]; // Location in head array for neighbors of current particle
		unsigned int nneigh   = d_nneigh[ idx ];   // Number of neighbors of the nearest particle
		
		// Particle info for this thread, pruned
		unsigned int phead_idx = d_headlist_pruned[ idx ]; // Location in head array for neighbors of current particle
		unsigned int pnneigh   = d_nneigh_pruned[ idx ];   // Number of neighbors of the nearest particle

		// Position of current particle
		Scalar4 posi = d_pos [ idx ];

		// Figure out how many particles are within the cutoff
		d_nlist_pruned[ phead_idx ] = idx;
		int counter = 1;
		for ( int ii = 0; ii < nneigh; ii++ ){
			
			// Get the current neighbor index
			unsigned int curr_neigh = d_nlist[ head_idx + ii ];

			// Position of current neighbor
			Scalar4 posj = d_pos[ curr_neigh ];
			
			// Distance between current particle and neighbor
			Scalar3 R = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
			R = box.minImage(R);
			Scalar distSqr = dot(R,R);
	
			// If particle is within cutoff, add to count, and put in nlist	
			if ( distSqr < rpsq ){
				d_nlist_pruned[ phead_idx + counter ] = curr_neigh;
				counter++;
			}
		}

		// Pointer to proper location within nlist
		unsigned int *A = &d_nlist_pruned[ phead_idx ];

		// Sort the neighbors using an insertion sort
		int jj, key;
		for ( int ii = 1; ii < pnneigh; ++ii ){

			// Get the current value
			key = A[ ii ];

			// Move elements that are greater than the key to one position ahead of their
			// current position
			jj = ii - 1;
			while( jj >= 0 && A[ jj ] > key ){
				A[ jj + 1 ] = A[ jj ];
				jj--;
			}
			A[ jj + 1 ] = key;
		}
	
		// Figure out how many of the neighbors have indices less than the current particle
		int nless = 0;
		for ( int ii = 0; ii < pnneigh; ++ii ){		
			if ( A[ii] < idx ){
				nless++;
			}
		}
		d_nneigh_less[ idx ] = nless;

		// Clear pointers
		A = NULL;
	}
}

/*!

	Wrap the functions to compute and sort the pruned neighborlist
	
	d_pos			(input)		particle positions
	d_group_members		(input)		array of particle indices within the integration group
	group_size		(input)		number of particles
	box			(input)		periodic box information
	res_data		(input/output)	structure containing lubrication calculation information, including neighborlist
	ker_data		(input)		structure containing kernel launch information
	
*/
void Precondition_PruneNeighborList(
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size, 
				      	BoxDim box,
					ResistanceData *res_data,
					KernelData *ker_data
					){

	// Kernel Information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Get pruned number of neighbors (rp < rlub)
	Precondition_GetPrunedNneigh_kernel<<<grid,threads>>>( 	
								res_data->nneigh_pruned, 
								group_size,
								d_pos,
			      					box,
								d_group_members,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist,
								res_data->rp
								);

	// Compute the pruned headlist with an inclusive scan (zhoge ???)
	int zero = 0;
	cudaMemcpy( res_data->headlist_pruned, &zero, sizeof(int), cudaMemcpyHostToDevice );

	thrust::device_ptr<unsigned int> i_thrustptr = thrust::device_pointer_cast( res_data->nneigh_pruned );
        thrust::device_ptr<unsigned int> o_thrustptr = thrust::device_pointer_cast( (res_data->headlist_pruned)+1 );
	thrust::inclusive_scan( i_thrustptr, i_thrustptr + group_size, o_thrustptr );
	
	// Build the (sorted) pruned neighbor list
	Precondition_GetPrunedNlist_kernel<<<grid,threads>>>( 	
								res_data->nneigh_pruned, 
								res_data->nlist_pruned,
								res_data->headlist_pruned,
								res_data->nneigh_less,
								group_size,
								d_pos,
			      					box,
								d_group_members,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist,
								res_data->rp
								);


}

/*!
	Figure out how many blocks of entries in the resistance tensor preconditioner
	that each particle has, i.e. total number of Nonzero Entries Per Particle (NEPP)

	group_size		(input)  length of vector d_nneigh
	d_group_members		(input)  array of particle indices within the integration group
	d_nneigh_pruned		(input)  list of number of neighbors for each particle in the pruned list
	d_nlist_pruned		(input)  pruned neighborlist array
	d_headlist_pruned	(input)  indices into the pruned neighborlist for each particle
	d_NEPP			(output) Number of non-zero entries per particle
	
*/
__global__ void Precondition_NEPP_kernel( 	
						unsigned int group_size,
						unsigned int *d_group_members,
						const unsigned int *d_nneigh_pruned, 
						const unsigned int *d_nlist_pruned, 
						const unsigned int *d_headlist_pruned, 
						unsigned int *d_NEPP
						){

	// Current thread index	
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Value of summation variables for the current thread
	if ( tidx < group_size) {

		// Particle info for this thread
		unsigned int idx = d_group_members[ tidx ];
	
		// Number of neighbors for current particle.	
		unsigned int nneigh = d_nneigh_pruned[ idx ]; // Number of neighbors of the nearest particle

		// Pruned neighborlist contains SELF ID as well, so if a particle has zero neighbors,
		// nneigh = 1. Also need to save space for the self block.
		int ne1 = ( nneigh > 1 ) ? ( ( 9 + 9 ) * ( nneigh ) ) : ( 3 );
		int ne2 = ( nneigh > 1 ) ? ( ( 9 + 9 ) * ( nneigh ) ) : ( 3 );

		// Write out
		d_NEPP[              idx ] = ne1;
		d_NEPP[ group_size + idx ] = ne2;

	}
}

/*!

	Figure out whether a particle has neighbors within the lubrication cutoff
	Return 1 or 0 for each particle.

	d_HasNeigh		(output) list of whether particle has neighbors in the lubrication cutoff
	group_size		(input)  number of particles in the group
	d_pos			(input)  particle positions
	box			(input)  periodic box information
	d_group_members		(input)  array of particle indices
	d_nneigh		(input)  list of number of neighbors for each particle
	d_nlist			(input)  neighborlist array
	d_headlist		(input)  indices into the neighborlist for each particle
	rlub			(input)  cutoff radius for lubrication interactions

*/
__global__ void Precondition_HasNeigh_kernel( 	
						int *d_HasNeigh,
						const unsigned int group_size,
						const Scalar4 *d_pos,
			      			const BoxDim box,
						const unsigned int *d_group_members,
						const unsigned int *d_nneigh, 
						const unsigned int *d_nlist, 
						const unsigned int *d_headlist,
						const Scalar rlub
						){

	// Current thread index	
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Value of summation variables for the current thread
	if ( tidx < group_size){

		// Cutoff radius squared
		Scalar rlubsq = rlub * rlub;

		// Particle info for this thread
		unsigned int idx = d_group_members[ tidx ];
		
		unsigned int head_idx = d_headlist[ idx ]; // Location in head array for neighbors of current particle
		unsigned int nneigh = d_nneigh[ idx ];   // Number of neighbors of the nearest particle

		// Position of current particle
		Scalar4 posi = d_pos [ idx ];

		// Figure out how many particles are within the cutoff
		int counter = 0;
		for ( int ii = 0; ii < nneigh; ii++ ){
			
			// Get the current neighbor index
			unsigned int curr_neigh = d_nlist[ head_idx + ii ];

			// Position of current neighbor
			Scalar4 posj = d_pos[ curr_neigh ];
			
			// Distance between current particle and neighbor
			Scalar3 R = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
			R = box.minImage(R);
			Scalar distSqr = dot(R,R);
	
			// If particle is within cutoff, add to count	
			if ( distSqr < rlubsq ){
				counter++;
			}
		}
		
		// Write out whether we have a neighbor or not
		d_HasNeigh[ idx ] = ( counter > 0 ) ? 1 : 0;
	}
}



/*!
	Pre-processing of data arrays in order to construct the sparse representation of the
	lubrication resistance tensor for the preconditioner

	d_group_members		(input)  array of particle indices within the group
	group_size		(input)  number of particles
	nnz			(input)  total number of non-zero elements within the preconditioner
	d_nneigh_pruned		(input)  list of pruned number of neighbors for each particle
	d_nlist_pruned		(input)  pruned neighborlist array
	d_headlist_pruned	(input)  indices into the pruned neighborlist for each particle
	d_NEPP			(output) Number of non-zero entries per particle	
	d_offset		(output) current particle's offsets into the output arrays
	grid			(input)  Grid for CUDA kernel launch
	threads			(input)  Threads for CUDA kernel launch
*/
void Precondition_PreProcess(
				unsigned int *d_group_members,
				int group_size, 
				int &nnz,
				const unsigned int *d_nneigh_pruned, 
				const unsigned int *d_nlist_pruned, 
				const unsigned int *d_headlist_pruned, 
				unsigned int *d_NEPP,
				unsigned int *d_offset,
				dim3 grid,
				dim3 threads
				){
	
	// Figure out the number of non-zero elements per particle (NEPP)
	Precondition_NEPP_kernel<<< grid, threads >>>(
							group_size,
							d_group_members,
							d_nneigh_pruned,
							d_nlist_pruned,
							d_headlist_pruned,
							d_NEPP
							);

	// First particle has offset of zero
	int zero = 0;
	cudaMemcpy( d_offset, &zero, sizeof(int), cudaMemcpyHostToDevice );
	
	// Add number of non-zero entries A,B,C for each particle 
	// ( This is needed for particle offset, but need A/B distinct 
	//   from B/C for the indexing later on )
	Precondition_AddInt_kernel<<<grid,threads>>>( &d_NEPP[0], &d_NEPP[group_size], &d_offset[1], 1, 1, group_size-1 );

	//	
	// Use THRUST to get get the cumulative sum of the numbers of entries per particle for each block
	//
	
	// Thrust device pointers for reductions
	thrust::device_ptr<unsigned int> i_thrustptr;
        thrust::device_ptr<unsigned int> o_thrustptr;
	
	// Wrap raw pointers in Thrust device pointers 
	i_thrustptr = thrust::device_pointer_cast( d_offset + 1 );	
	o_thrustptr = thrust::device_pointer_cast( d_offset + 1 );
	
	// Do the scan (cumulative sum) for the RFU mobility tensor
	thrust::inclusive_scan( i_thrustptr, i_thrustptr + (group_size-1), o_thrustptr );

	// Figure out the number of non-zero entries in each array
	int scan, end;
	cudaMemcpy( &scan, &d_offset[ group_size-1 ], sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( &end,  &d_NEPP[   group_size-1 ], sizeof(int), cudaMemcpyDeviceToHost );
	
	nnz = scan + end;

	// Have to do again for BC part of first nnz
	cudaMemcpy( &end, &d_NEPP[ 2*group_size-1 ], sizeof(int), cudaMemcpyDeviceToHost );
	nnz += end;
	
}

/*! 
	Build the preconditioner for the RFU Lubrication Tensor ( ALL PARTICLES SAME SIZE ) -- give one thread per particle 
	
	THIS VERSION OF THE FUNCTION STORES THE FULL RESISTANCE TENSOR (NOT JUST THE LOWER HALF)
		Reason: Not all cusparse operations are defined for symmetric matrices, so the 
			full, general matrices have to be used instead.
	Sparse matrix storage format is COO. 

	group_size		(input)  Number of particles
	d_group_members		(input)  array of particle indices
	d_nneigh		(input)  list of number of neighbors for each particle
	d_nneigh_less		(input)  number of neighbors with index less than current particle
	d_nlist			(input)  neighborlist array
	d_headlist		(input)  indices into the neighborlist for each particle
	d_NEPP			(input)  Number of non-zero entries per particle
	d_offset		(input)  current particle's offsets into the output arrays
	d_pos			(input)  particle positions
	box			(input)  simulation box information
	d_ResTable_dist		(input)  distances for which the resistance function has been tabulated
	d_ResTable_vals		(input)  tabulated values of the resistance tensor
	table_dr		(input)  table discretization (in log units)
	d_L_RowInd		(output) COO row indices
	d_L_ColInd		(output) COO col indices
	d_L_Val			(output) COO vals
	rp			(input)  preconditioner cutoff radius

*/
__global__ void Precondition_RFU_kernel(
					int group_size, 
					unsigned int *d_group_members,
					const unsigned int *d_nneigh, 
					unsigned int *d_nneigh_less, 
					unsigned int *d_nlist, 
					const unsigned int *d_headlist, 
					unsigned int *d_NEPP,
					unsigned int *d_offset, 
					Scalar4 *d_pos,
			      		BoxDim box,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const float table_min,
					const float table_dr,
					int   *d_L_RowInd,
					int   *d_L_ColInd,
					float *d_L_Val,
					const Scalar rp
					){

  // Index for current thread 
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
  // Check that thread is within bounds, and only do work if so	
  if ( tidx < group_size ) {

    // Square of the cutoff radius
    Scalar rpsq = rp * rp;
		
    // Particle info for this thread
    unsigned int curr_particle = d_group_members[ tidx ];
		
    Scalar4 posi = d_pos[ curr_particle ];
			
    unsigned int head_idx = d_headlist[ curr_particle ];	// Location in head array for neighbors of current particle
    unsigned int nneigh = d_nneigh[ curr_particle ];		// Number of neighbors of the nearest particle
    unsigned int nneigh_less = d_nneigh_less[ curr_particle ]; 	// Number of neighbors with index less than current particle

    // Offset information for current particle
    unsigned int offset_particle = d_offset[ curr_particle ];
    unsigned int offset_BC = d_NEPP[ curr_particle ];

    // Neighbor counters
    unsigned int neigh_idx, curr_neigh;
	
    int flag = 0; // flag to write out self elements only once
    unsigned int row; 
				
    // Loop over all the neighbors for the current particle and add those
    // pair entries to the lubrication resistance tensor
    for (neigh_idx = 0; neigh_idx < nneigh; neigh_idx++) {
	
      // Get the current neighbor index
      curr_neigh = d_nlist[ head_idx + neigh_idx ];
			
      // Position and size of neighbor particle
      Scalar4 posj = d_pos[ curr_neigh ];
	
      // Distance vector between current particle and neighbor. 
      //
      // By definition, R points from particle 1 to particle 2 (i to j), otherwise
      // the handedness and symmetry of the lubrication functions is lost
      Scalar3 R = make_scalar3( posj.x - posi.x, posj.y - posi.y, posj.z - posi.z );
			
      // Minimum image	
      R = box.minImage(R);

      // Distance magnitude
      Scalar distSqr = dot(R,R);

      // Check that particles are within the cutoff 
      if ( ( distSqr < rpsq ) && ( curr_neigh != curr_particle ) ){	
				
	// Distance 
	Scalar dist = sqrtf( distSqr );

	Scalar XA11, XA12, YA11, YA12, YB11, YB12, XC11, XC12, YC11, YC12;

	// The block below may be commented out to test stress calculation
	// for dense suspensions. (zhoge)
	///*
	if ( dist <= ( 2.0 + table_min ) ){
	  // Table is strided by 22
	  XA11 = d_ResTable_vals[ 0 ];
	  XA12 = d_ResTable_vals[ 1 ];
	  YA11 = d_ResTable_vals[ 2 ];
	  YA12 = d_ResTable_vals[ 3 ];
	  YB11 = d_ResTable_vals[ 4 ];
	  YB12 = d_ResTable_vals[ 5 ];
	  XC11 = d_ResTable_vals[ 6 ];
	  XC12 = d_ResTable_vals[ 7 ];
	  YC11 = d_ResTable_vals[ 8 ];
	  YC12 = d_ResTable_vals[ 9 ];
	}
	//*/
	//if ( dist <= 2.001 ){
	//  // In Stokes_ResistanceTable.cc, h_ResTable_dist.data[232] = 2.000997;
	//  // Table is strided by 22
	//  int i_regl = 232*22; //lubrication regularization (due to roughness) 
	//  XA11 = d_ResTable_vals[ i_regl + 0 ];
	//  XA12 = d_ResTable_vals[ i_regl + 1 ];
	//  YA11 = d_ResTable_vals[ i_regl + 2 ];
	//  YA12 = d_ResTable_vals[ i_regl + 3 ];
	//  YB11 = d_ResTable_vals[ i_regl + 4 ];
	//  YB12 = d_ResTable_vals[ i_regl + 5 ];
	//  XC11 = d_ResTable_vals[ i_regl + 6 ];
	//  XC12 = d_ResTable_vals[ i_regl + 7 ];
	//  YC11 = d_ResTable_vals[ i_regl + 8 ];
	//  YC12 = d_ResTable_vals[ i_regl + 9 ];
	//}
	//// End stress test (zhoge)
	else {

	  // Get the index of the nearest entry below the current distance in the distance array
	  // NOTE: Distances are logarithmically spaced in the tabulation
	  int ind = log10f( ( dist - 2.0 ) /  table_min ) / table_dr;
						
	  // Get the values from the distance array for interpolation
	  Scalar dist_lower = d_ResTable_dist[ ind ];
	  Scalar dist_upper = d_ResTable_dist[ ind + 1 ];
			
	  // Read the scalar resistance coefficients from the array (lower and upper values 
	  // for interpolation)
	  //
	  // Table is strided by 22
	  Scalar XA11_lower = d_ResTable_vals[ 22 * ind + 0 ];
	  Scalar XA12_lower = d_ResTable_vals[ 22 * ind + 1 ];
	  Scalar YA11_lower = d_ResTable_vals[ 22 * ind + 2 ];
	  Scalar YA12_lower = d_ResTable_vals[ 22 * ind + 3 ];
	  Scalar YB11_lower = d_ResTable_vals[ 22 * ind + 4 ];
	  Scalar YB12_lower = d_ResTable_vals[ 22 * ind + 5 ];
	  Scalar XC11_lower = d_ResTable_vals[ 22 * ind + 6 ];
	  Scalar XC12_lower = d_ResTable_vals[ 22 * ind + 7 ];
	  Scalar YC11_lower = d_ResTable_vals[ 22 * ind + 8 ];
	  Scalar YC12_lower = d_ResTable_vals[ 22 * ind + 9 ];
	
	  Scalar XA11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 0 ];
	  Scalar XA12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 1 ];
	  Scalar YA11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 2 ];
	  Scalar YA12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 3 ];
	  Scalar YB11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 4 ];
	  Scalar YB12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 5 ];
	  Scalar XC11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 6 ];
	  Scalar XC12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 7 ];
	  Scalar YC11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 8 ];
	  Scalar YC12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 9 ];
		
	  // Linear interpolation of the Table values
	  Scalar fac = ( dist - dist_lower  ) / ( dist_upper - dist_lower );
	
	  XA11 = XA11_lower + ( XA11_upper - XA11_lower ) * fac;
	  XA12 = XA12_lower + ( XA12_upper - XA12_lower ) * fac;
	  YA11 = YA11_lower + ( YA11_upper - YA11_lower ) * fac;
	  YA12 = YA12_lower + ( YA12_upper - YA12_lower ) * fac;
	  YB11 = YB11_lower + ( YB11_upper - YB11_lower ) * fac;
	  YB12 = YB12_lower + ( YB12_upper - YB12_lower ) * fac;
	  XC11 = XC11_lower + ( XC11_upper - XC11_lower ) * fac;
	  XC12 = XC12_lower + ( XC12_upper - XC12_lower ) * fac;
	  YC11 = YC11_lower + ( YC11_upper - YC11_lower ) * fac;
	  YC12 = YC12_lower + ( YC12_upper - YC12_lower ) * fac;
	}
				
	// Geometric quantities ----- zhoge: THIS PART MAY NEED CORRECTION!!!
	//
	// levi-civita is left-handed because JO system is left-handed
	Scalar rx = R.x / dist;
	Scalar ry = R.y / dist;
	Scalar rz = R.z / dist;
	Scalar r[3] = { rx, ry, rz };
	Scalar epsr[3][3] = { { 0.0, rz, -ry }, 
			      { -rz, 0.0, rx }, 
			      {  ry, -rx, 0.0} };
	//zhoge: It is not left-handed.
	//for ( int ii = 0; ii < 3; ++ii ){
	//  for ( int jj = 0; jj < 3; ++jj ){
	//    epsr[ii][jj] *= -1.0;
	//  }
	//}	

	// Indices needed to write things out
	unsigned int col_neigh, col_self;
	int oind_neigh, oind_self;
	float A_neigh, A_self;
	float B_neigh, B_self;
	float C_neigh, C_self;
	float rr, Imrr;
				
	// Delta functions needed to evaluate resistance tensors
	float delta_ij;
	
	//
	// Compute FU Block
	//
				
	for ( int ii = 0; ii < 3; ++ii ){
					
	  // Row index
	  row = 6 * curr_particle + ii;
					
	  for ( int jj = 0; jj < 3; ++jj ){
						
	    // Column index
	    col_neigh = 6 * curr_neigh + jj;
	    col_self  = 6 * curr_particle + jj;
				
	    // Tensor Quantities
	    delta_ij = ( ( ii == jj ) ? 1.0 : 0.0 );
	    rr = r[ii] * r[jj];
	    Imrr = delta_ij - rr;
		
	    // Value
	    //
	    // A_ij  = XA * ( ri * rj ) + YA * ( delta_ij - ri * rj )
	    // Bt_ij = Bji = YB * ( eps_jik * rk ) = YB * ( eps_ijk * rk )
	    // C_ij  = XC * ( ri * rj ) + YC * ( delta_ij - ri * rj )
	
	    A_neigh = XA12 * ( rr ) + YA12 * ( Imrr );
	    A_self  = XA11 * ( rr ) + YA11 * ( Imrr );
						
	    B_neigh = YB12 * ( epsr[ii][jj] );
	    B_self  = YB11 * ( epsr[ii][jj] );
						
	    C_neigh = XC12 * ( rr ) + YC12 * ( Imrr );
	    C_self  = XC11 * ( rr ) + YC11 * ( Imrr );
	
	    // Offset into output
	    oind_neigh = offset_particle + ( 6 * nneigh * ii ) + ( 6 * neigh_idx   + jj );
	    oind_self  = offset_particle + ( 6 * nneigh * ii ) + ( 6 * nneigh_less + jj );

	    // Write out the neighbor contribution			
	    d_L_RowInd[ oind_neigh ] = row; // A
	    d_L_ColInd[ oind_neigh ] = col_neigh;
	    d_L_Val[    oind_neigh ] = A_neigh; 
	
	    d_L_RowInd[ oind_neigh + 3 ] = row; // B^T  //zhoge: BT12 = B12 = BT21.transpose = B21.transpose
	    d_L_ColInd[ oind_neigh + 3 ] = col_neigh + 3;
	    d_L_Val[    oind_neigh + 3 ] = B_neigh;
	
	    d_L_RowInd[ offset_BC + oind_neigh ] = row + 3; // B
	    d_L_ColInd[ offset_BC + oind_neigh ] = col_neigh;
	    d_L_Val[    offset_BC + oind_neigh ] = B_neigh; 
	
	    d_L_RowInd[ offset_BC + oind_neigh + 3 ] = row + 3; // C
	    d_L_ColInd[ offset_BC + oind_neigh + 3 ] = col_neigh + 3;
	    d_L_Val[    offset_BC + oind_neigh + 3 ] = C_neigh; 
	
	    // Write out the self contribution (only need to record row/column
	    // for the self piece once for each ii,jj, then we can add to the value)
	    if ( flag < 9 ){
	      d_L_RowInd[ oind_self ] = row; // A
	      d_L_ColInd[ oind_self ] = col_self;
	    }
	    d_L_Val[ oind_self ] += A_self; 

	    if ( flag < 9 ){	
	      d_L_RowInd[ oind_self + 3 ] = row; // B^T  //zhoge: BT11 = B11.transpose
	      d_L_ColInd[ oind_self + 3 ] = col_self + 3;
	    }
	    d_L_Val[ oind_self + 3 ] += -B_self; 
	    // ! On preceding line, need (-) sign because levi-civita transpose is
	    //   NOT balanced by {alpha}{beta} swap (JO eqn. 1.6b) for self bit
	
	    if ( flag < 9 ){
	      d_L_RowInd[ offset_BC + oind_self ] = row + 3; // B
	      d_L_ColInd[ offset_BC + oind_self ] = col_self;
	    }
	    d_L_Val[ offset_BC + oind_self ] += B_self; 
	
	    if ( flag < 9 ){
	      d_L_RowInd[ offset_BC + oind_self + 3 ] = row + 3; // C
	      d_L_ColInd[ offset_BC + oind_self + 3 ] = col_self + 3;
	    }
	    d_L_Val[ offset_BC + oind_self + 3 ] += C_self; 
	
	    // Flag only needed when first setting
	    flag++;

	  } // loop over jj
	} // loop over ii

      } // Check on distance
	
    } // Loop over neighbors

    // Make the diagonal elements zero if there are no neighbors for this particle. Do
    // this because we need to add the identity tensor to RFU before doing the Cholesky
    // decomposition for the saddle point preconditioner, so the sparse matrix storage
    // needs to account for that
    if ( flag == 0 ){
      for ( unsigned int ii = 0; ii < 3; ++ii ){

	row = 6 * curr_particle + ii;

	d_L_RowInd[ offset_particle + ii ] = row;
	d_L_ColInd[ offset_particle + ii ] = row;
	d_L_Val[    offset_particle + ii ] = 0.0;

	d_L_RowInd[ offset_BC + offset_particle + ii ] = row + 3;
	d_L_ColInd[ offset_BC + offset_particle + ii ] = row + 3;
	d_L_Val[    offset_BC + offset_particle + ii ] = 0.0;
      }
    }

  } // Check for thread in bounds

}

/*!
	Build sparse representation for RFU preconditioner. Wrap the functions to build RFU
	in COO format then convert to CSR.

	COO  (row, column, value)
	CSR  (value, column_index, row_index), where the row_index is more of a count. 

	d_pos			(input)  particle positions
	d_group_members		(input)  array of particle indices
	group_size		(input)  Number of particles
	box			(input)  simulation box information
	d_nneigh		(input)  list of number of neighbors for each particle
	d_nneigh_less		(input)  number of neighbors with index less than current particle
	d_nlist			(input)  neighborlist array
	d_headlist		(input)  indices into the neighborlist for each particle
	d_NEPP			(input)  Number of non-zero entries per particle
	d_offset		(input)  current particle's offsets into the output arrays
	d_ResTable_dist		(input)  distances for which the resistance function has been tabulated
	d_ResTable_vals		(input)  tabulated values of the resistance tensor
	table_dr		(input)  table discretization (in log units)
	nnz			(input)  number of non-zero elements in the sparse RFU
	d_L_RowInd		(output) COO row indices
	d_L_RowPtr		(output) CSR row pointer
	d_L_ColInd		(output) COO/CSR col indices
	d_L_Val			(output) COO/CSR vals
	spHandle		(input)  opaque handle for cuSPARSE operations
	rp			(input)  preconditioner cutoff radius
	grid			(input)  Grid for CUDA kernel launch
	threads			(input)  Threads for CUDA kernel launch

*/
void Precondition_Build(
				Scalar4 *d_pos,
				unsigned int *d_group_members,
				int group_size, 
			      	BoxDim box,
				const unsigned int *d_nneigh, 
				unsigned int *d_nneigh_less, 
				unsigned int *d_nlist, 
				const unsigned int *d_headlist, 
				unsigned int *d_NEPP,
				unsigned int *d_offset, 
				const Scalar *d_ResTable_dist,
				const Scalar *d_ResTable_vals,
				const float table_min,
				const float table_dr,
				int &nnz,
				int   *d_L_RowInd,
				int   *d_L_RowPtr,
				int   *d_L_ColInd,
				float *d_L_Val,
				cusparseHandle_t spHandle,
				const Scalar rp,
				dim3 grid,
				dim3 threads
				){
	
	// Zero the value arrays (Have to zero because the diagonal terms need to be
	// added, and we need to remove any data left over from previous calculations)
	Precondition_ZeroVector_kernel<<<grid, threads>>>( d_L_Val, nnz, group_size );

	// Build the lubrication tensors
	Precondition_RFU_kernel<<<grid,threads>>>(
							group_size, 
							d_group_members,
							d_nneigh, 
							d_nneigh_less, 
							d_nlist, 
							d_headlist, 
							d_NEPP,
							d_offset, 
							d_pos,
			      				box,
							d_ResTable_dist,
							d_ResTable_vals,
							table_min,
							table_dr,
							d_L_RowInd,   //output
							d_L_ColInd,   //output
							d_L_Val,      //output
							rp
							);
		
	// Convert from COO to CSR (need constant pointers for Row Indices)
	cusparseXcoo2csr( spHandle, d_L_RowInd, nnz, 6*group_size, d_L_RowPtr, CUSPARSE_INDEX_BASE_ZERO );

}

/*!
	Do Reverse-Cuthill-Mckee Reordering of the near-field lubrication tensor preconditioner

	group_size		(input)		number of particles
	d_prcm			(output)	RCM permutation vector
	nnz			(input)		number of non-zero elements in RFU
	d_headlist_pruned	(input)		headlist into pruned neighborlist array
	d_nlist_pruned		(input)		pruned neighborlist array
	d_nneigh_pruned		(input)		pruned number of neighbors
	d_L_RowPtr		(input/output)	CSR row pointer for RFU (before/after reordering)
	d_L_ColInd		(input/output)  CSR column indices for RFU (before/after reordering)
	d_L_Val			(input/output)	CSR values for RFU (before/after reordering)
	soHandle		(input) 	opaque handle for cuSOLVER
	spHandle		(input)		opaque handle for cuSPARSE
	descr_R			(input)		cuSPARSE matrix description of RFU
	d_Scratch3		(input)		Scratch space for re-ordering
	grid			(input)		grid for CUDA kernel launch
	threads			(input)		threads for CUDA kernel launch
	d_scratch		(input)		workspace for index projection
	d_map			(input)		workspace for reorder mapping

*/
void Precondition_Reorder(
				int group_size, 
				int *d_prcm,
				int &nnz,
				unsigned int *d_headlist_pruned,
				unsigned int *d_nlist_pruned,
				unsigned int *d_nneigh_pruned,
				int   *d_L_RowPtr,
				int   *d_L_ColInd,
				float *d_L_Val,
				cusolverSpHandle_t soHandle,
				cusparseHandle_t spHandle,
				cusparseMatDescr_t descr_R,
				float *d_Scratch3,
				dim3 grid,
				dim3 threads,
				int *d_scratch,
				int *d_map
				){	
	
	// Length of nneigh
	int NeighTotal;
	cudaMemcpy( &NeighTotal, &d_headlist_pruned[group_size], sizeof(int), cudaMemcpyDeviceToHost );
		
	// Allocate Host Memory
	int *h_headlist, *h_nlist;
	h_headlist = (int *)malloc( (group_size+1)*sizeof(int) );
	h_nlist    = (int *)malloc( NeighTotal*sizeof(int) );

	int *h_L_RowPtr, *h_L_ColInd;
        h_L_RowPtr = (int *)malloc( (6*group_size+1)*sizeof(int) );
        h_L_ColInd = (int *)malloc( nnz*sizeof(int) );

	int *h_prcm;
	h_prcm = (int *)malloc( (6*group_size)*sizeof(int) );
	
	int *h_map;
	h_map = (int *)malloc( nnz * sizeof(int) );

	// Copy to host
	cudaMemcpy( h_L_RowPtr, d_L_RowPtr, (6*group_size+1)*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_L_ColInd, d_L_ColInd, nnz*sizeof(int), cudaMemcpyDeviceToHost );

	cudaMemcpy( h_headlist, d_headlist_pruned, (group_size+1)*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_nlist, d_nlist_pruned, NeighTotal*sizeof(int), cudaMemcpyDeviceToHost );
	
	// Use John Burkardt's code for RCM reordering. Alternative routines are given in
	// cuSOLVER and BOOST libraries, but both those implementations are very slow, and
	// exhibit superlinear scaling of the computational cost with number of particles.  
	//
	// Burkardt code expects 1-based lists for the neighbor adjacency.
	//
	// Burkardt code gives 1-based indexing for prcm, so fix that too.
	//
	// TODO: Put the 1-based indexing fix into ExpandPRCM_kernel
	for ( int ii = 0; ii < NeighTotal; ++ii ){
		h_nlist[ ii ] += 1;
	}
	for ( int ii = 0; ii < group_size+1; ++ii ){
		h_headlist[ ii ] += 1;
	}
	genrcm( group_size, NeighTotal, h_headlist, h_nlist, h_prcm );
	for( int ii = 0; ii < group_size; ++ii ){
		h_prcm[ ii ] -= 1;
	}

	// Expand the re-ordering from particle-based to 6N index-based
	cudaMemcpy( d_scratch, h_prcm, group_size*sizeof(int), cudaMemcpyHostToDevice );
	Precondition_ExpandPRCM_kernel<<< grid, threads >>>(
								d_prcm,
								d_scratch,
								group_size
								);
	cudaMemcpy( h_prcm, d_prcm, 6*group_size*sizeof(int), cudaMemcpyDeviceToHost );
	
	// Find the Buffer Size required to apply the permutation
	size_t pBufferSizePermute = 0;
	cusolverSpXcsrperm_bufferSizeHost(
						soHandle,
						6*group_size,
						6*group_size,
						nnz,
						descr_R,
						h_L_RowPtr,
						h_L_ColInd,
						h_prcm,
						h_prcm,
						&pBufferSizePermute
						);
	
	// Allocate buffer for reordering
	void *pBuffer;
	pBuffer = (void *)malloc( pBufferSizePermute );

	// Create a Map for the permutation
	int block_size = 256;
	dim3 val_grid( int(nnz/block_size) + 1, 1, 1 );
	dim3 val_threads(block_size, 1, 1);
	
	Precondition_InitializeMap_kernel<<< val_grid, val_threads >>>( d_map, nnz );
	cudaMemcpy( h_map, d_map, nnz*sizeof(int), cudaMemcpyDeviceToHost );
	
	// Do the permutation
	cusolverSpXcsrpermHost(
				soHandle,
				6*group_size,
				6*group_size,
				nnz,
				descr_R,
				h_L_RowPtr,
				h_L_ColInd,
				h_prcm,
				h_prcm,
				h_map,
				pBuffer
				);
	

	// Copy result to the GPU
	cudaMemcpy( d_L_RowPtr, h_L_RowPtr, (6*group_size+1)*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_L_ColInd, h_L_ColInd, nnz*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_map, h_map, nnz*sizeof(int), cudaMemcpyHostToDevice );
			
	// Apply the map to the values as well
	Precondition_Map_kernel<<< val_grid, val_threads >>>( d_Scratch3, d_L_Val, d_map, nnz );
	cudaMemcpy( d_L_Val, d_Scratch3, nnz*sizeof(float), cudaMemcpyDeviceToDevice );	

	//
	// Clean Up
	//
	free( h_headlist );
	free( h_nlist );

	free( h_L_RowPtr );
	free( h_L_ColInd );
	free( h_prcm );
	
	free( h_map );
	free( pBuffer );
	
}

/*!
	Do the incomplete Cholesky decomposition and set up the cuSPARSE
	matrix descriptions for the lubrication preconditioner

	group_size	(input)		Number of particles
	nnz		(input)		Number of nonzero elements 
	d_L_RowPtr	(input/output)	CSR row pointer to RFU / lower Cholesky factor
	d_L_ColInd	(input/output)	CSR col indices to RFU / lower Cholesky factor
	d_L_Val		(input/output)	CSR values for RFU / lower Cholesky factor
	spHandle	(input)		opaque handle for cuSPARSE operations
	spStatus	(input)		status output for cuSPARSE operations
	descr_R		(input)		cuSPARSE matrix description of RFU
	descr_L		(input)		cuSPARSE matrix description for lower Cholesky factor
	info_R		(input)		cuSPARSE info for RFU
	info_L		(input)		cuSPARSE info for lower Cholesky factor
	info_Lt		(input)		cuSPARSE info for upper Cholesky factor
	trans_L		(input)		cuSPARSE transpose operation for lower Cholesky factor
	trans_Lt	(input)		cuSPARSE transpose operation for upper Cholesky factor
	policy_R	(input)		cuSPARSE solver policy for R
	policy_L	(input)		cuSPARSE solver policy for lower Cholesky factor
	policy_Lt	(input)		cuSPARSE solver policy for upper Cholesky factor
	pBufferSize	(output)	Buffer size for cuSPARSE operations
	grid		(input)		grid for CUDA kernel launch
	threads		(input)		threads for CUDA kernel launch

*/
void Precondition_IChol(
			int group_size,
			unsigned int nnz,
			int   *d_L_RowPtr,
			int   *d_L_ColInd,
			float *d_L_Val,
			cusparseHandle_t spHandle,
        		cusparseStatus_t spStatus,
			cusparseMatDescr_t    descr_R, 
			cusparseMatDescr_t    descr_L, 
			csric02Info_t         info_R,
			csrsv2Info_t          info_L,
			csrsv2Info_t          info_Lt,
			cusparseOperation_t   trans_L,
			cusparseOperation_t   trans_Lt,
			cusparseSolvePolicy_t policy_R, 
			cusparseSolvePolicy_t policy_L,
			cusparseSolvePolicy_t policy_Lt,
			int& pBufferSize,
			dim3 grid,
			dim3 threads,
			float &ichol_relaxer,
			bool &ichol_converged
			){
		
	// 1. Incomplete Cholesky decomposition for the preconditioner needs to 
	//    be performed on ( RFUnf + relaxer*I ), so add Identity to the matrix
	Precondition_AddIdentity_kernel<<<grid,threads>>>(
								d_L_Val,
								d_L_RowPtr,
								d_L_ColInd, 
								group_size,
								ichol_relaxer
								);
		
        // 2. Get buffer memory requirements and allocate buffer
        int pBufferSize_R = 0;  // Buffer size required for calculations on R
        int pBufferSize_L = 0;  // Buffer size required for calculations on L
        int pBufferSize_Lt = 0; // Buffer size required for calculations on L^T

        cusparseScsric02_bufferSize(
        					spHandle,
        					6*group_size,
        					nnz,
                				descr_R,
        					d_L_Val,
        					d_L_RowPtr,
        					d_L_ColInd,
        					info_R,
        					&pBufferSize_R
        					);
 
	cusparseScsrsv2_bufferSize(
						spHandle,
						trans_L,
						6*group_size,
						nnz,
						descr_L,
						d_L_Val,
						d_L_RowPtr,
						d_L_ColInd,
						info_L,
						&pBufferSize_L
						);
	
	cusparseScsrsv2_bufferSize(
						spHandle,
						trans_Lt,
						6*group_size,
						nnz,
						descr_L,
						d_L_Val,
						d_L_RowPtr,
						d_L_ColInd,
						info_Lt,
						&pBufferSize_Lt
						);
	
        pBufferSize = max( pBufferSize_R, max(pBufferSize_L, pBufferSize_Lt) );
	//std::cout << "pBufferSize = " << pBufferSize << std::endl; //zhoge: GPUdebug
	void *pBuffer;
	cudaMalloc((void**)&pBuffer, pBufferSize );

	// 3. Parameters and initialization needed for cuSPARSE procedures
        int numerical_zero;  // Checks for zero pivots in matrix decomposition
        int structural_zero;

        // 4. Pre-solve analysis
        cusparseScsric02_analysis(
						spHandle,
						6*group_size,
						nnz,
						descr_R,
						d_L_Val,
						d_L_RowPtr,
						d_L_ColInd,
						info_R,
                				policy_R,
						pBuffer
						);

	cusparseScsrsv2_analysis(
						spHandle,
						trans_L,
						6*group_size,
						nnz,
						descr_L,
                				d_L_Val,
						d_L_RowPtr,
						d_L_ColInd,
                				info_L,
						policy_L,
						pBuffer
						);

        cusparseScsrsv2_analysis(
						spHandle,
						trans_Lt,
						6*group_size,
						nnz,
						descr_L,
                				d_L_Val,
						d_L_RowPtr,
						d_L_ColInd,
                				info_Lt,
						policy_Lt,
						pBuffer
						);

	// Check for zero pivot in the incomplete Cholesky decomposition
        spStatus = cusparseXcsric02_zeroPivot(spHandle, info_R, &structural_zero);
        if ( CUSPARSE_STATUS_ZERO_PIVOT == spStatus ){
        	printf("R(%d,%d) is missing \n", structural_zero, structural_zero);
		exit(1);
        }
		
        // 5. Perform incomplete Cholesky decomposition, (RFUnf + relaxer*I) = L * L'
	//
	spStatus = cusparseScsric02(
					spHandle,
					6*group_size,
					nnz,
					descr_R,
           				d_L_Val,      //input/output
					d_L_RowPtr,
					d_L_ColInd,
					info_R,
					policy_R,
					pBuffer
					);
			
	if ( spStatus != CUSPARSE_STATUS_SUCCESS) {
		printf("    Incomplete Cholesky Failed. Quitting.\n");
		Debug_StatusCheck_cuSparse( spStatus, "Ichol decomposition" );
		exit(1);
	}
	
	// Check for numerical zero (loss of positive-definite)	
	spStatus = cusparseXcsric02_zeroPivot(spHandle, info_R, &numerical_zero);
	if ( CUSPARSE_STATUS_ZERO_PIVOT == spStatus ){
	  //originally commented (zhoge: GPUdebug)
	  //printf("L(%d,%d) is zero \n", numerical_zero, numerical_zero);
	  //exit(1);

		// Set convergence flag to false and increase the relaxation
		// parameter for the next time
		ichol_converged = false;
		ichol_relaxer *= 2.0;
		
		// Return
		cudaFree(pBuffer); //zhoge: GPUdebug
		return;
	}
	
	// Set converged to true if we get this far
	ichol_converged = true;

	// Zero any entries above the main diagonal (Have to do this because,
	// at least as of the cuda-8.0 toolkit, even specifying 
	// MATRIX_FILL_MODE=LOWER doesn't prevent the CUDA functions from using
	// the full matrix in the various solves and mutiplications. 
	Precondition_ZeroUpperTriangle_kernel<<<grid,threads>>>( 
								d_L_RowPtr,
								d_L_ColInd,
								d_L_Val,
								group_size
								);

	// Clean up
	cudaFree( pBuffer );

}

/*
	Wrapper for all the functions required to build the preconditioner, in the proper order

	zhoge: It mainly does S = L * L^T, where S = P * (R_FU^nf + relaxer*I) * P^T, 
               R_FU^nf is the pruned near-field RFU,
               relaxer is in powers of 2 (starting from 1), 
               P is the RCM permutation matrix (P^T its inverse),
               and L is a lower incomplete Cholesky factor (L^T its upper factor).

	d_pos			(input)		particle positions
	d_group_members		(input)		indices for particles in the integration group
	group_size		(input)		number of particles
	box			(input)		periodic box information
	ker_data		(input)		structure containing CUDA kernel information
	res_data		(input/output)	structure containing resistance and preconditioner information

*/
void Precondition_Wrap(
			Scalar4 *d_pos,
			unsigned int *d_group_members,
			unsigned int group_size,
			const BoxDim& box,
			KernelData *ker_data,
			ResistanceData *res_data,
			WorkData *work_data
			){
	
	// Get kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;
		
	// Check whether particle has neighbors within the lubrication cutoff
	Precondition_HasNeigh_kernel<<< grid, threads >>>( 	
							  res_data->HasNeigh,  //output
							  group_size,
							  d_pos,
							  box,
							  d_group_members,
							  res_data->nneigh, 
							  res_data->nlist, 
							  res_data->headlist,
							  res_data->rlub
							  );

	// Get the pruned neighbor list (within rp instead of rlub)
	Precondition_PruneNeighborList(
					d_pos,
					d_group_members,
					group_size, 
				      	box,
					res_data,  //output
					ker_data
					);
	
	// Pre-process the arrays for the preconditioner (count certain non-zeros)
	Precondition_PreProcess(
				d_group_members,
				group_size, 
				res_data->nnz,
				res_data->nneigh_pruned, 
				res_data->nlist_pruned, 
				res_data->headlist_pruned, 
				res_data->NEPP,            //output
				res_data->offset,	   //output
				ker_data->particle_grid,
				ker_data->particle_threads
				);
		
	// Build the approximate lubrication tensor, R_FU^nf (output in COO and CSR formats)
	Precondition_Build(
				d_pos,
				d_group_members,
				group_size,
				box,
				res_data->nneigh_pruned, 
				res_data->nneigh_less, 
				res_data->nlist_pruned, 
				res_data->headlist_pruned, 
				res_data->NEPP,
				res_data->offset, 
				res_data->table_dist,
				res_data->table_vals,
				res_data->table_min,
				res_data->table_dr,
				res_data->nnz,
				res_data->L_RowInd,   //output
				res_data->L_RowPtr,   //output
				res_data->L_ColInd,   //output
				res_data->L_Val,      //output
				res_data->spHandle,
				res_data->rp,
				ker_data->particle_grid,
				ker_data->particle_threads
				);
	
	// Re-order the lubrication tensor (R_FU^nf) using RCM (using an implementation in rcm.cpp)
	// zhoge: Should result P * (R_FU^nf) * P^T
	Precondition_Reorder(
				group_size, 
				res_data->prcm,                   //output (the permutation)
				res_data->nnz,
				res_data->headlist_pruned,
				res_data->nlist_pruned,
				res_data->nneigh_pruned,
				res_data->L_RowPtr,               //input/output
				res_data->L_ColInd,		  //input/output
				res_data->L_Val,		  //input/output
				res_data->soHandle,
				res_data->spHandle,
				res_data->descr_R,
				res_data->Scratch3,
				ker_data->particle_grid,
				ker_data->particle_threads,
				work_data->precond_scratch,
				work_data->precond_map
				);
	
	// Get the inverse square root of the diagonal elements (related to near-field Brownian calculations)
	Precondition_GetDiags_kernel<<< grid, threads >>>(
								group_size, 
								res_data->Diag,     //output
								res_data->L_RowPtr, //input
								res_data->L_ColInd, //input
								res_data->L_Val	    //input
								);
	/* //zhoge: redundent (done in IChol below)
	// Add far-field contribution to the diagonal, i.e. S = R_FU^nf + ichol_relaxer*(1 or 4/3)
	Precondition_AddIdentity_kernel<<<grid,threads>>>(
							  res_data->L_Val,     //input/output
							  res_data->L_RowPtr,  //input
							  res_data->L_ColInd,  //input
							  group_size,	       //input
							  1.0                  //input: relaxation factor
							  );
	
	// Check if there are zero diagonals (in Helper_Debug.cu)
	Debug_CSRzeroDiag( res_data->L_RowPtr, res_data->L_ColInd, res_data->L_Val, group_size, res_data->nnz );
	*/

	// Backup storage for the elements
	Scalar *d_backup = (work_data->precond_backup);
	Scalar *d_values = (res_data->L_Val);
	cudaMemcpy( d_backup, d_values, (res_data->nnz)*sizeof(Scalar), cudaMemcpyDeviceToDevice );	

	// Set convergence flag false to start so that we try at least once. Then, do the IChol
	// factorization, adding along the diagonal as needed to ensure convergence. 
	(res_data->ichol_converged) = false;
	//int idebug = 0;
	while ( !(res_data->ichol_converged) ){

	  // Copy original values (res_data->L_Val) pointed to by d_values
	  cudaMemcpy( d_values, d_backup, (res_data->nnz)*sizeof(Scalar), cudaMemcpyDeviceToDevice );

	  // Do the incomplete Cholesky decomposition (L * L^T)
	  // output replaces the input
	  Precondition_IChol(
			     group_size,
			     res_data->nnz,
			     res_data->L_RowPtr,      //input/output
			     res_data->L_ColInd,      //input/output
			     res_data->L_Val,	      //input/output
			     res_data->spHandle,
			     res_data->spStatus,
			     res_data->descr_R, 
			     res_data->descr_L, 
			     res_data->info_R,
			     res_data->info_L,
			     res_data->info_Lt,
			     res_data->trans_L,
			     res_data->trans_Lt,
			     res_data->policy_R, 
			     res_data->policy_L,
			     res_data->policy_Lt,
			     res_data->pBufferSize,    //output
			     ker_data->particle_grid,
			     ker_data->particle_threads,
			     res_data->ichol_relaxer,  //relaxation factor: can be modified; reset to 1.0 periodically, see Stokes.cc
			     res_data->ichol_converged
			     );
	  //idebug++; //zhoge
	}
	//if (idebug > 1)
	//  std::cout << "IChol iterations " << idebug << std::endl; //zhoge: GPUdebug

	// Cleanup
	d_backup = NULL;
	d_values = NULL;
	
}

/*
	Preconditioned RFU multiply for the Brownian calculation. 

		y = L^(-1) * D^(-1) * P * ( R_FU^nf + I_nn ) * P^(T) * D^(-T) * L^(-T) * x

		L   = Lower Cholesky factor of ( \tilde R_FU^nf + I )
		D   = Modified diagonal elements of R_FU
		P   = RCM re-ordering
		I_nn = modified identity tensor (non-zero only if no neighbor)

	zhoge: The order of D and P is inconsistent with the FSD paper, 
               but it is correct because D was obtained from the permutated RFU.

	!!! CAN work with in-place solve (i.e. pointers d_x = d_y)

	d_y			(output) product of preconditioned matrix-vector multiply
	d_x			(input)  vector to multiply by preconditioned matrix
	d_pos			(input)  particle positions
	d_group_members		(input)  list of particle indices within the integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	ker_data		(input)  structure containing information for CUDA kernel launches
	res_data		(input)  structure containing information for resistance calculations

*/
void Precondition_Brownian_RFUmultiply(	
					float *d_y, // output
					float *d_x, // input
					const Scalar4 *d_pos,
					unsigned int *d_group_members,
					const int group_size, 
			      		const BoxDim box,
					void *pBuffer,
					KernelData *ker_data,
					ResistanceData *res_data
					){
	
	// Kernel data
	dim3 grid    = (ker_data->particle_grid);
	dim3 threads = (ker_data->particle_threads);

	// Pointer to scratch array (size 6N, same as d_x and d_y)
	float *d_z = (res_data->Scratch1);
	
	// Number of elements of the arrays
	int numel = 6 * group_size;
	
	// Variable required for Axpy
	float spAlpha = 1.0;
	
	//
	//// First incomplete Cholesky Solve: solve L'*y = x
	//cusparseScsrsv2_solve(
	//			res_data->spHandle, 
	//			res_data->trans_Lt, 
	//			numel, 
	//			res_data->nnz, 
	//			&spAlpha, 
	//			res_data->descr_L,
	//   			res_data->L_Val, 
	//			res_data->L_RowPtr, 
	//			res_data->L_ColInd, 
	//			res_data->info_Lt,
	//   			d_x, // input
	//			d_y, // output
	//			res_data->policy_Lt, 
	//			pBuffer
	//			);
	//
	//// Diagonal multiplication (zhoge: Actually dividing the diagonal elements by their square roots)
	//Precondition_DiagMult_kernel<<< grid, threads >>>(
	//							d_z, // output
	//							d_y, // input
	//							group_size, 
	//							res_data->Diag,  //input
	//							1
	//							);
	//
	//// Permute the input vector (zhoge: Undo the RCM permutation given -1)
	//Precondition_ApplyRCM_Vector_kernel<<<grid,threads>>>( 
	//							d_y, // output
	//							d_z, // input
	//							res_data->prcm,
	//							group_size,
	//							-1
	//							);
	//
	//// RFU multiplication and addition of Inn
	////
	//// d_z = RFU * d_y
	//Lubrication_RFU_kernel<<< grid, threads >>>(
	//					        d_z, // output (intermediate)
	//						d_y, // input
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
	//// d_z += Inn * d_y
	//Precondition_Inn_kernel<<< grid, threads >>>(
	//						d_z, // input/output (overwritten)
	//						d_y, // input
	//						res_data->HasNeigh,
	//						group_size
	//						);
	//
	//// Permute the output vector (zhoge: Apply the RCM given 1)
	//Precondition_ApplyRCM_Vector_kernel<<<grid,threads>>>( 
	//						        d_y, // output
	//							d_z, // input
	//							res_data->prcm,
	//							group_size,
	//							1
	//							);
	//
	//// Diagonal multiplication (zhoge: Divide again, now the diagonals are 1 if they were between 0 and 1)
	//Precondition_DiagMult_kernel<<< grid, threads >>>(
	//							d_z, // output
	//							d_y, // input
	//							group_size, 
	//							res_data->Diag,
	//							1
	//							);
	//
	//// Second incomplete Cholesky solve: solve L*y = x
	//cusparseScsrsv2_solve(
	//			res_data->spHandle, 
	//			res_data->trans_L, 
	//			numel, 
	//			res_data->nnz, 
	//			&spAlpha, 
	//			res_data->descr_L,
	//   			res_data->L_Val, 
	//			res_data->L_RowPtr, 
	//			res_data->L_ColInd, 
	//			res_data->info_L,
	//   			d_z, // input 
	//			d_y, // output
	//			res_data->policy_L, 
	//			pBuffer
	//			);
	//
	
	//debug: effectively turn off the preconditioner
	// d_y = RFU * d_x
	Lubrication_RFU_kernel<<< grid, threads >>>(
						        d_y, // output
							d_x, // input
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


	
	// Clean up
	d_z = NULL;

}

/*
	Undoes the square root of the pre-conditioner so that the resulting random 
	variable has the correct variance

		x = ( I - Inn ) * P^(T) * D * G

		G   = lower Cholesky factor
		D   = modified diagonal matrix
		P   = RCM re-ordering matrix
		Inn = modified identity tensor

	zhoge: Again, D and P^T are flipped relative to the FSD paper, but it is self-consistent in the code.

	!!! Works in-place

	d_x		(input/output) 	vector to be rescaled and reordered
	group_size	(input)		number of particles
	ker_data	(input)  	structure containing information for CUDA kernel launches
	res_data	(input)  	structure containing information for resistance calculations

*/
void Precondition_Brownian_Undo(	
				float *d_x,       // input/output
				int group_size,
				KernelData *ker_data,
				ResistanceData *res_data
				){

	// Kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Pointer to scratch array
	float *d_z = (res_data->Scratch1);

	// Number of elements in vectors
	int numel = 6*group_size;
	
	// Incomplete Cholesky multiplication
        float spAlpha = 1.0;
        float spBeta = 0.0;
        cusparseScsrmv(
                        res_data->spHandle,
                        res_data->trans_L,
                        numel,
                        numel,
                        res_data->nnz,
                        &spAlpha,
                        res_data->descr_L,
                        res_data->L_Val,
                        res_data->L_RowPtr,
                        res_data->L_ColInd,
                        d_x, // Input
                        &spBeta,
                        d_z  // Output
                        );

	// Diagonal preconditioner (zhoge: Multiply the square root given -1, confusing notation but correct)
	Precondition_DiagMult_kernel<<< grid, threads >>>(
								d_x,  // output
								d_z,  // input
								group_size, 
								res_data->Diag,
								-1 
								);
	
	// Permute the output vector (zhoge: Actually undo the RCM permutation given -1)
	Precondition_ApplyRCM_Vector_kernel<<< grid, threads >>>( 
								 	d_z, // output
								        d_x, // input
									res_data->prcm,
									group_size,
									-1
									);

	// Project out the components for particles with no neighbors
	Precondition_ImInn_kernel<<< grid, threads >>>(
							d_x, // output
							d_z, // input
							res_data->HasNeigh,
							group_size
							);

	// Clean pointers
	d_z = NULL;

}




/*
	Apply the preconditioner for the near-field lubrication in the saddle point solve

	Wrapper to perform the solves required of inverting the incomplete
	Cholesky representation of the approximate resistance tensor

		y = ( L * L' ) \ x

	!!! CAN work with in-place solve (i.e. pointers d_x = d_y -- needed for GMRES)
	
	d_y		(output) product of matrix and input vector
	d_x		(input)  input vector for multiplication
	d_Scratch	(input)  scratch space for calculations
	d_prcm		(input)  RCM re-ordering vector
	group_size	(input)  Number of particles
	nnz		(input)  Number of nonzero elements 
	d_L_RowPtr	(input)  CSR row pointer to RFU / lower Cholesky factor
	d_L_ColInd	(input)  CSR col indices to RFU / lower Cholesky factor
	d_L_Val		(input)  CSR values for RFU / lower Cholesky factor
	spHandle	(input)	 opaque handle for cuSPARSE operations
	spStatus	(input)	 status output for cuSPARSE operations
	descr_L		(input)  cuSPARSE matrix description for lower Cholesky factor
	info_L		(input)  cuSPARSE info for lower Cholesky factor
	info_Lt		(input)  cuSPARSE info for upper Cholesky factor
	trans_L		(input)  cuSPARSE transpose operation for lower Cholesky factor
	trans_Lt	(input)  cuSPARSE transpose operation for upper Cholesky factor
	policy_L	(input)  cuSPARSE solver policy for lower Cholesky factor
	policy_Lt	(input)  cuSPARSE solver policy for upper Cholesky factor
	pBufferSize	(input)  Buffer size for cuSPARSE operations
	grid		(input)  grid for CUDA kernel launch
	threads		(input)  threads for CUDA kernel launch



*/
void Precondition_Saddle_RFUmultiply(	
					float *d_y,       // output
					float *d_x,       // input
					float *d_Scratch, // intermediate storage
					const int *d_prcm,
					int group_size,
					unsigned int nnz,
					const int   *d_L_RowPtr,
					const int   *d_L_ColInd,
					const float *d_L_Val,
					cusparseHandle_t spHandle,
        				cusparseStatus_t spStatus,
					cusparseMatDescr_t descr_L,
					csrsv2Info_t info_L,
					csrsv2Info_t info_Lt,
					const cusparseOperation_t trans_L,
					const cusparseOperation_t trans_Lt,
					const cusparseSolvePolicy_t policy_L,
					const cusparseSolvePolicy_t policy_Lt,
					void *pBuffer,
					dim3 grid,
					dim3 threads
					){

	// Variable required for Axpy
	float spAlpha = 1.0;

	// Vector length
	int numel = 6 * group_size;
	
	// Permute the input vector
	Precondition_ApplyRCM_Vector_kernel<<<grid,threads>>>( 
								d_Scratch, // output
								d_x,       // input
								d_prcm,
								group_size,
								1
								);

	//
	// Incomplete Cholesky solve
	
	// first: solve L*y = x
	cusparseScsrsv2_solve(
				spHandle, 
				trans_L, 
				numel,
				nnz, 
				&spAlpha, 
				descr_L,
	   			d_L_Val, 
				d_L_RowPtr, 
				d_L_ColInd, 
				info_L,
	   			d_Scratch,  // input 
				d_y,        // output
				policy_L, 
				pBuffer
				);
	
	// second: solve L'*z = y
	cusparseScsrsv2_solve(
				spHandle, 
				trans_Lt, 
				numel, 
				nnz, 
				&spAlpha, 
				descr_L,
	   			d_L_Val, 
				d_L_RowPtr, 
				d_L_ColInd, 
				info_Lt,
	   			d_y,       // input
				d_Scratch, // output
				policy_Lt, 
				pBuffer
				);

	// Permute the output vector
	Precondition_ApplyRCM_Vector_kernel<<<grid,threads>>>( 
								d_y,       // output
								d_Scratch, // input
								d_prcm,
								group_size,
								-1
								);
	

}

