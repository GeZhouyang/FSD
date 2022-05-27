// Andrew Fiore
// Zhouyang Ge

#include "Lubrication.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Saddle.cuh"

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
	This file defines functions required to compute the lubrication interations,
	i.e. compute the action of the lubrication tensor on a vector
	
	zhoge: Simplified in May 2021. 
*/

/*!
	Matrix-vector product for the RFU lubrication tensor

	ALL PARTICLES SAME SIZE -- give one thread per particle

	\param d_Force	Generalized force (force/torque) on particles
	\param d_Velociy	Generalized velocity of particles
	\param d_pos  		particle positions
	\param group_size                Number of particles
	\param box		simulation box information
	\param d_group_members  array of particle indices
	\param d_n_neigh_lub    list of number of neighbors for each particle
	\param d_nlist_lub      neighborlist array
	\param d_headlist_lub   indices into the neighborlist for each particle
	\param d_ResTable_dist	distances for which the resistance function has been tabulated
	\param d_ResTable_vals	tabulated values of the resistance tensor
	\param ResTable_min	minimum table value
	\param ResTable_dr	table discretization (in log units)
	\param rlub		lubrication cutoff distance

*/

__global__ void Lubrication_RFU_kernel(
					Scalar *d_Force,          // output
					const Scalar *d_Velocity, // input
					const Scalar4 *d_pos,
					unsigned int *d_group_members,
					const int group_size, 
			      		const BoxDim box,
					const unsigned int *d_n_neigh, 
					unsigned int *d_nlist, 
					const unsigned int *d_headlist, 
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const float ResTable_min,
					const float ResTable_dr,
					const Scalar rlub
					){
  
  // Index for current thread 
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
  // Check that thread is within bounds, and only do work if so	
  if ( tidx < group_size ) {
	
    // Square of the cutoff radius
    Scalar rlubsq = rlub * rlub;
	
    // Particle info for this thread
    unsigned int curr_particle = d_group_members[ tidx ];
	
    // Position and (disturbance) velocity of current particle	
    Scalar4 posi = d_pos[ curr_particle ];
    Scalar3 ui, wi, uj, wj;
    ui.x = d_Velocity[ 6*curr_particle     ];
    ui.y = d_Velocity[ 6*curr_particle + 1 ];
    ui.z = d_Velocity[ 6*curr_particle + 2 ];
    wi.x = d_Velocity[ 6*curr_particle + 3 ];
    wi.y = d_Velocity[ 6*curr_particle + 4 ];
    wi.z = d_Velocity[ 6*curr_particle + 5 ];

    // Initialize force/torque
    Scalar3 fi = make_scalar3( 0.0, 0.0, 0.0 );
    Scalar3 li = make_scalar3( 0.0, 0.0, 0.0 );	
    	
    // Neighborlist arrays
    unsigned int head_idx = d_headlist[ curr_particle ]; // Location in head array for neighbors of current particle
    unsigned int n_neigh = d_n_neigh[ curr_particle ];   // Number of neighbors of the nearest particle
	
    // Loop over all the neighbors for the current particle and add those
    // pair entries to the lubrication resistance tensor
    // zhoge: Each GPU core takes care of one center particle and only modify the value of center particle.
    //        Attempts to modify neighbors may lead to race condition in CUDA.
    for (unsigned int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {
	
      // Get the current neighbor index
      unsigned int curr_neigh = d_nlist[ head_idx + neigh_idx ];

      // Position of neighbor particle
      Scalar4 posj = d_pos[ curr_neigh ];

      // (disturbance) Velocity of neighbor particle
      uj.x = d_Velocity[ 6*curr_neigh ];
      uj.y = d_Velocity[ 6*curr_neigh + 1 ];
      uj.z = d_Velocity[ 6*curr_neigh + 2 ];
      wj.x = d_Velocity[ 6*curr_neigh + 3 ];
      wj.y = d_Velocity[ 6*curr_neigh + 4 ];
      wj.z = d_Velocity[ 6*curr_neigh + 5 ];
	
      // Distance vector between current particle and neighbor
      Scalar3 R = make_scalar3( posj.x - posi.x, posj.y - posi.y, posj.z - posi.z );
			
      // Minimum image
      R = box.minImage(R);

      // Distance magnitude
      Scalar distSqr = dot(R,R);

      // Check that particles are within the hard-sphere cutoff (not all particles in
      // the HOOMD neighborlist are guaranteed to be within the cutoff) 
      if ( (distSqr < rlubsq) && (distSqr > 0.) ){	
				
	// Distance 
	Scalar dist = sqrtf( distSqr );

	// Interpolate the non-divergent part of the sums from tabulation
	Scalar XA11, XA12, YA11, YA12, YB11, YB12, XC11, XC12, YC11, YC12;

	// The block below may be commented out to test stress calculation
	// for dense suspensions. (zhoge)
	/*
	if ( dist <= ( 2.0 + ResTable_min ) ){
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
	*/
	if ( dist <= 2.001 ){
	  // In Stokes_ResistanceTable.cc, h_ResTable_dist.data[232] = 2.000997;
	  // Table is strided by 22
	  int i_regl = 232*22; //lubrication regularization (due to roughness) 
	  XA11 = d_ResTable_vals[ i_regl + 0 ];
	  XA12 = d_ResTable_vals[ i_regl + 1 ];
	  YA11 = d_ResTable_vals[ i_regl + 2 ];
	  YA12 = d_ResTable_vals[ i_regl + 3 ];
	  YB11 = d_ResTable_vals[ i_regl + 4 ];
	  YB12 = d_ResTable_vals[ i_regl + 5 ];
	  XC11 = d_ResTable_vals[ i_regl + 6 ];
	  XC12 = d_ResTable_vals[ i_regl + 7 ];
	  YC11 = d_ResTable_vals[ i_regl + 8 ];
	  YC12 = d_ResTable_vals[ i_regl + 9 ];
	}
	// End stress test (zhoge)
	else {

	  // Get the index of the nearest entry below the current distance in the distance array
	  // NOTE: Distances are logarithmically spaced in the tabulation, dr is distance in log
	  //	 space, i.e. ResTable_dr = log10( ResTable_dist[1] / ResTable_dist[0] )
	  int ind = log10f( ( dist - 2.0 ) /  ResTable_min ) / ResTable_dr;
						
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

	// Unit vector from curr-center to neigh-center
	Scalar3 r = make_scalar3( R.x / dist, R.y / dist, R.z / dist );
	
	// Dot product of r and U, i.e. axisymmetric projection
	Scalar rdui = ( r.x * ui.x + r.y * ui.y + r.z * ui.z );
	Scalar rduj = ( r.x * uj.x + r.y * uj.y + r.z * uj.z );
	Scalar rdwi = ( r.x * wi.x + r.y * wi.y + r.z * wi.z );
	Scalar rdwj = ( r.x * wj.x + r.y * wj.y + r.z * wj.z );

	// Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U,
	// (eps_ijk is the Levi-Civita symbol)
	// Px = eps_ijk*r_k = [  0   rz -ry ]  U = [Ux]
	//                    [ -rz  0   rx ]      [Uy]
	//      	      [  ry -rx  0  ],     [Uz].
	// The following four vectors are the dot products.
	// (Originally, Andrew incorrectly timed them all by -1.)
	Scalar3 epsrdui = make_scalar3( (  r.z * ui.y - r.y * ui.z ),
					( -r.z * ui.x + r.x * ui.z ),
					(  r.y * ui.x - r.x * ui.y ) );
	Scalar3 epsrdwi = make_scalar3( (  r.z * wi.y - r.y * wi.z ),
					( -r.z * wi.x + r.x * wi.z ),
					(  r.y * wi.x - r.x * wi.y ) );
	Scalar3 epsrduj = make_scalar3( (  r.z * uj.y - r.y * uj.z ),
					( -r.z * uj.x + r.x * uj.z ),
					(  r.y * uj.x - r.x * uj.y ) );
	Scalar3 epsrdwj = make_scalar3( (  r.z * wj.y - r.y * wj.z ),
					( -r.z * wj.x + r.x * wj.z ),
					(  r.y * wj.x - r.x * wj.y ) );

	// Compute the contributions to the force (F1)
	//
	// F1 = A11*U1 + A12*U2 + BT11*W1 + BT12*W2,
	//
	// where A11 = XA11*Pn + YA11*(I-Pn) = (XA11-YA11)*Pn + YA11,
	// with Pn_ij = r_i*r_j (similarly for A12),
	// and BT11_ij = B11_ji = YB11*(-Px), BT12_ij = B21_ji = YB21*(-Px).
	
	// Symmetry condition:
	Scalar YB21 = -YB12;
	
	fi += ( XA11 - YA11 ) * rdui * r + YA11 * ui    
	    + ( XA12 - YA12 ) * rduj * r + YA12 * uj    
	    + YB11 * (-epsrdwi)                         
	    + YB21 * (-epsrdwj);        		      

	// Compute the contributions to the torque (L1)
	//
	// L1 = B11*U1 + B12*U2 + C11*W1 + C12*W2,
	//
	// where C11 and C12 are just like A11 and A12.
	
	li += YB11 * epsrdui 
	    + YB12 * epsrduj 
	    + ( XC11 - YC11 ) * rdwi * r + YC11 * wi 
	    + ( XC12 - YC12 ) * rdwj * r + YC12 * wj;
	
      } // Check on distance
    } // Loop over neighbors

    // Write to output
    d_Force[ 6*curr_particle     ] = fi.x;
    d_Force[ 6*curr_particle + 1 ] = fi.y;
    d_Force[ 6*curr_particle + 2 ] = fi.z;
    d_Force[ 6*curr_particle + 3 ] = li.x;
    d_Force[ 6*curr_particle + 4 ] = li.y;
    d_Force[ 6*curr_particle + 5 ] = li.z;

  } // Check for thread in bounds

}



























/*!
	Matrix-vector product for the RSU lubrication tensor

	ALL PARTICLES SAME SIZE -- give one thread per particle

	\param d_pos  		particle positions
	\param box		simulation box information
	\param d_ResTable_dist	distances for which the resistance function has been tabulated
	\param d_ResTable_vals	tabulated values of the resistance tensor
	\param ResTable_dr	table discretization (in log units)
	\param N                Number of particles
	\param d_group_members  array of particle indices
	\param d_n_neigh        list of number of neighbors for each particle
	\param d_nlist_lub      neighborlist array
	\param d_headlist_lub   indices into the neighborlist for each particle
	\param d_offset         current particle's offsets into the output arrays
	\param d_NEPP		Number of non-zero entries per particle


*/
__global__ void Lubrication_RSU_kernel(
					Scalar *d_Stresslet,
					Scalar *d_Velocity,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size,
			      		BoxDim box,
					const unsigned int *d_n_neigh,
					unsigned int *d_nlist,
					const unsigned int *d_headlist,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const float ResTable_min,
					const float ResTable_dr,
					const Scalar rlub
					){

	// Index for current thread
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check that thread is within bounds, and only do work if so
	if ( tidx < group_size ) {

		// Square of the cutoff radius
		Scalar rlubsq = rlub * rlub;

		// Particle info for this thread
		unsigned int curr_particle = d_group_members[ tidx ];

		unsigned int head_idx = d_headlist[ curr_particle ]; // Location in head array for neighbors of current particle
		unsigned int n_neigh = d_n_neigh[ curr_particle ]; // Number of neighbors of the nearest particle

		// Neighbor counters
		unsigned int neigh_idx, curr_neigh;

		// Position and Velocity for current particle
		Scalar4 posi = d_pos[ curr_particle ];
		Scalar si[5] = { 0.0 };
		Scalar3 ui, wi, uj, wj;
		ui.x = d_Velocity[ 6*curr_particle ];
		ui.y = d_Velocity[ 6*curr_particle + 1 ];
		ui.z = d_Velocity[ 6*curr_particle + 2 ];
		wi.x = d_Velocity[ 6*curr_particle + 3 ];
		wi.y = d_Velocity[ 6*curr_particle + 4 ];
		wi.z = d_Velocity[ 6*curr_particle + 5 ];

		// Loop over all the neighbors for the current particle and add those
		// pair entries to the lubrication resistance tensor
		for (neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {

			// Get the current neighbor index
			curr_neigh = d_nlist[ head_idx + neigh_idx ];

			// Have to keep track of which set of particle indices we have for the sign
			// convention in Jeffrey and Onishi. For simplicity, in the JO notation of
			// pairs 11, 12, 21, 11, always set the particle with the lower global
			// index as 1 and the one with the higher as 2.
			//
			// Applies to the distance vector between the particles
			//float jo_sign = ( curr_particle < curr_neigh ) ? 1.0 : -1.0;

			// Position of neighbor particle
			Scalar4 posj = d_pos[ curr_neigh ];

			// Velocity of neighbor particle
			uj.x = d_Velocity[ 6*curr_neigh ];
			uj.y = d_Velocity[ 6*curr_neigh + 1 ];
			uj.z = d_Velocity[ 6*curr_neigh + 2 ];
			wj.x = d_Velocity[ 6*curr_neigh + 3 ];
			wj.y = d_Velocity[ 6*curr_neigh + 4 ];
			wj.z = d_Velocity[ 6*curr_neigh + 5 ];

			// Distance vector between current particle and neighbor
			Scalar3 R = make_scalar3( posj.x - posi.x, posj.y - posi.y, posj.z - posi.z );

			//// Sign convention (JO equation 1.5)
			//R *= jo_sign;

			// Minimum image
			R = box.minImage(R);

			// Distance magnitude
			Scalar distSqr = dot(R,R);

			// Check that particles are within the hard-sphere cutoff (not all particles in
			// the HOOMD neighborlist are guaranteed to be within the cutoff)
			if ( distSqr < rlubsq ){

				// Distance
				Scalar dist = sqrtf( distSqr );

				Scalar XG11, XG12, YG11, YG12, YH11, YH12;
	// The block below may be commented out to test stress calculation
	// for dense suspensions. (zhoge)
	/*
	if ( dist <= ( 2.0 + ResTable_min ) ){
	  // Table is strided by 22
	  XG11 = d_ResTable_vals[ 10 ];
	  XG12 = d_ResTable_vals[ 11 ];
	  YG11 = d_ResTable_vals[ 12 ];
	  YG12 = d_ResTable_vals[ 13 ];
	  YH11 = d_ResTable_vals[ 14 ];
	  YH12 = d_ResTable_vals[ 15 ];
	}
	*/
	if ( dist <= 2.001 ){
	  // In Stokes_ResistanceTable.cc, h_ResTable_dist.data[232] = 2.000997;
	  // Table is strided by 22
	  int i_regl = 232*22; //lubrication regularization (due to roughness) 
	  XG11 = d_ResTable_vals[ i_regl + 10 ];
	  XG12 = d_ResTable_vals[ i_regl + 11 ];
	  YG11 = d_ResTable_vals[ i_regl + 12 ];
	  YG12 = d_ResTable_vals[ i_regl + 13 ];
	  YH11 = d_ResTable_vals[ i_regl + 14 ];
	  YH12 = d_ResTable_vals[ i_regl + 15 ];
	}
	// End stress test (zhoge)

				else {

					// Get the index of the nearest entry below the current distance in the distance array
					// NOTE: Distances are logarithmically spaced in the tabulation
					int ind = log10f( ( dist - 2.0 ) / ResTable_min ) / ResTable_dr;

					// Get the values from the distance array for interpolation
					Scalar dist_lower = d_ResTable_dist[ ind ];
					Scalar dist_upper = d_ResTable_dist[ ind + 1 ];

					// Read the scalar resistance coefficients from the array (lower and upper values
					// for interpolation)
					//
					// Table is strided by 32 to coalesce reads
					Scalar XG11_lower = d_ResTable_vals[ 22 * ind + 10 ];
					Scalar XG12_lower = d_ResTable_vals[ 22 * ind + 11 ];
					Scalar YG11_lower = d_ResTable_vals[ 22 * ind + 12 ];
					Scalar YG12_lower = d_ResTable_vals[ 22 * ind + 13 ];
					Scalar YH11_lower = d_ResTable_vals[ 22 * ind + 14 ];
					Scalar YH12_lower = d_ResTable_vals[ 22 * ind + 15 ];

					Scalar XG11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 10 ];
					Scalar XG12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 11 ];
					Scalar YG11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 12 ];
					Scalar YG12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 13 ];
					Scalar YH11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 14 ];
					Scalar YH12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 15 ];

					// Linear interpolation of the Table values
					Scalar fac = ( dist - dist_lower  ) / ( dist_upper - dist_lower );

					XG11 = XG11_lower + ( XG11_upper - XG11_lower ) * fac;
					XG12 = XG12_lower + ( XG12_upper - XG12_lower ) * fac;
					YG11 = YG11_lower + ( YG11_upper - YG11_lower ) * fac;
					YG12 = YG12_lower + ( YG12_upper - YG12_lower ) * fac;
					YH11 = YH11_lower + ( YH11_upper - YH11_lower ) * fac;
					YH12 = YH12_lower + ( YH12_upper - YH12_lower ) * fac;

				}
				//madhu
				// Geometric quantities
				Scalar3 r = make_scalar3( R.x / dist, R.y / dist, R.z / dist );

				Scalar rdui = r.x * ui.x + r.y * ui.y + r.z * ui.z ;
				Scalar rduj = r.x * uj.x + r.y * uj.y + r.z * uj.z ;

				// epsr = [  0   rz -ry ]
                                //        [ -rz  0   rx ]
				// 	  [  ry -rx  0  ]
				//
				// Levi-Civita is left-handed because JO is left-handed (NO.)
				Scalar3 epsrdwi = make_scalar3( (  r.z * wi.y - r.y * wi.z ),
							        ( -r.z * wi.x + r.x * wi.z ),
							        (  r.y * wi.x - r.x * wi.y ) );
				Scalar3 epsrdwj = make_scalar3( (  r.z * wj.y - r.y * wj.z ),
							        ( -r.z * wj.x + r.x * wj.z ),
							        (  r.y * wj.x - r.x * wj.y ) );


				// Value, note that si[3] denotes the yy component
				//
				// S_ij += G^11_ijk * U1k + G^12_ijk * U2k
				//       = XG11 * ( ri * rj - 1/3 * delta_ij ) * rk*U1k 
				//       + XG12 * ( ri * rj - 1/3 * delta_ij ) * rk*U2k 
				//       + YG11 * ( ri * U1j + U1i * rj - 2.* ri * rj * rk*U1k)
				//       + YG12 * ( ri * U2j + U2i * rj - 2.* ri * rj * rk*U2k)
				
				si[0] += XG11 * ( r.x * r.x - 1.0 / 3.0 ) * rdui +
					 XG12 * ( r.x * r.x - 1.0 / 3.0 ) * rduj +
					 YG11 * (ui.x * r.x + r.x * ui.x - 2.0 * r.x * r.x * rdui ) +
					 YG12 * (uj.x * r.x + r.x * uj.x - 2.0 * r.x * r.x * rduj );
				
				si[1] += XG11 * ( r.x * r.y ) * rdui +
					 XG12 * ( r.x * r.y ) * rduj +
					 YG11 * (ui.x * r.y + r.x * ui.y - 2.0 * r.x * r.y * rdui ) +
					 YG12 * (uj.x * r.y + r.x * uj.y - 2.0 * r.x * r.y * rduj );
				
				si[2] += XG11 * ( r.x * r.z ) * rdui +
					 XG12 * ( r.x * r.z ) * rduj +
					 YG11 * (ui.x * r.z + r.x * ui.z - 2.0 * r.x * r.z * rdui ) +
					 YG12 * (uj.x * r.z + r.x * uj.z - 2.0 * r.x * r.z * rduj );

				si[4] += XG11 * ( r.y * r.z ) * rdui +
					 XG12 * ( r.y * r.z ) * rduj +
					 YG11 * (ui.y * r.z + r.y * ui.z - 2.0 * r.y * r.z * rdui ) +
					 YG12 * (uj.y * r.z + r.y * uj.z - 2.0 * r.y * r.z * rduj );

				si[3] += XG11 * ( r.y * r.y - 1.0 / 3.0 ) * rdui +
					 XG12 * ( r.y * r.y - 1.0 / 3.0 ) * rduj +
					 YG11 * (ui.y * r.y + r.y * ui.y - 2.0 * r.y * r.y * rdui ) +
					 YG12 * (uj.y * r.y + r.y * uj.y - 2.0 * r.y * r.y * rduj );

				//si[0] +=	jo_sign * XG11 * ( r.x * r.x - 1.0 / 3.0 ) * rdui +
				//		jo_sign * YG11 * ( ( ui.x * r.x ) + ( r.x * ui.x ) - 2.0 * r.x * r.x * rdui ) +
				//		jo_sign * XG12 * ( r.x * r.x - 1.0 / 3.0 ) * rduj +
				//		jo_sign * YG12 * ( ( uj.x * r.x ) + ( r.x * uj.x ) - 2.0 * r.x * r.x * rduj );
				//si[1] +=	jo_sign * XG11 * ( r.x * r.y ) * rdui +
				//		jo_sign * YG11 * ( ( ui.x * r.y ) + ( r.x * ui.y ) - 2.0 * r.x * r.y * rdui ) +
				//		jo_sign * XG12 * ( r.x * r.y ) * rduj +
				//		jo_sign * YG12 * ( ( uj.x * r.y ) + ( r.x * uj.y ) - 2.0 * r.x * r.y * rduj );
				//si[2] +=	jo_sign * XG11 * ( r.x * r.z ) * rdui +
				//		jo_sign * YG11 * ( ( ui.x * r.z ) + ( r.x * ui.z ) - 2.0 * r.x * r.z * rdui ) +
				//		jo_sign * XG12 * ( r.x * r.z ) * rduj +
				//		jo_sign * YG12 * ( ( uj.x * r.z ) + ( r.x * uj.z ) - 2.0 * r.x * r.z * rduj );
				//si[3] +=	jo_sign * XG11 * ( r.y * r.z ) * rdui +
				//		jo_sign * YG11 * ( ( ui.y * r.z ) + ( r.y * ui.z ) - 2.0 * r.y * r.z * rdui ) +
				//		jo_sign * XG12 * ( r.y * r.z ) * rduj +
				//		jo_sign * YG12 * ( ( uj.y * r.z ) + ( r.y * uj.z ) - 2.0 * r.y * r.z * rduj );
				//si[4] +=	jo_sign * XG11 * ( r.y * r.y - 1.0 / 3.0 ) * rdui +
				//		jo_sign * YG11 * ( ( ui.y * r.y ) + ( r.y * ui.y ) - 2.0 * r.y * r.y * rdui ) +
				//		jo_sign * XG12 * ( r.y * r.y - 1.0 / 3.0 ) * rduj +
				//		jo_sign * YG12 * ( ( uj.y * r.y ) + ( r.y * uj.y ) - 2.0 * r.y * r.y * rduj );

				// Value, note that si[3] denotes the yy component
				//
				// S_ij += H^11_ijk * W1k + H^12_ijk * W2k
				//       = YH11 * ( ri * eps_jkm * rm + rj * eps_ikm * rm ) * W1k
				//       + YH12 * ( ri * eps_jkm * rm + rj * eps_ikm * rm ) * W2k
				
				si[0] += YH11 * ( r.x * epsrdwi.x + epsrdwi.x * r.x ) +
					 YH12 * ( r.x * epsrdwj.x + epsrdwj.x * r.x );
				
				si[1] += YH11 * ( r.x * epsrdwi.y + epsrdwi.x * r.y ) +
					 YH12 * ( r.x * epsrdwj.y + epsrdwj.x * r.y );
				
				si[2] += YH11 * ( r.x * epsrdwi.z + epsrdwi.x * r.z ) +
					 YH12 * ( r.x * epsrdwj.z + epsrdwj.x * r.z );
				
				si[4] += YH11 * ( r.y * epsrdwi.z + epsrdwi.y * r.z ) +
					 YH12 * ( r.y * epsrdwj.z + epsrdwj.y * r.z );
				
				si[3] += YH11 * ( r.y * epsrdwi.y + epsrdwi.y * r.y ) +
					 YH12 * ( r.y * epsrdwj.y + epsrdwj.y * r.y );

			} // check if neighbor is within cutoff

		} // Loop over neighbors

		// Write to output
		//madhu
		//float sign = 1;
		d_Stresslet[ 5*curr_particle ]     -= si[0];
		d_Stresslet[ 5*curr_particle + 1 ] -= si[1];
		d_Stresslet[ 5*curr_particle + 2 ] -= si[2];
		d_Stresslet[ 5*curr_particle + 3 ] -= si[3];
		d_Stresslet[ 5*curr_particle + 4 ] -= si[4];
		//madhu

	} // Check for thread in bounds

}

/*!
	Matrix-vector product for the RFE lubrication tensor

	These simulations are constructed so that, if there is strain,
		x is the flow direction
		y is the gradient direction
		z is the vorticity direction
	therefore,
		Einf = 	[ 0 g 0 ]
			[ g 0 0 ]
			[ 0 0 0 ]
	where g is the shear rate. Therefore, the strain rate on each particle (due to the imposed straining flow)
	is identical, so the only needed quantity is the global shear rate.

	ALL PARTICLES SAME SIZE -- give one thread per particle

	\param d_pos  		particle positions
	\param box		simulation box information
	\param d_ResTable_dist	distances for which the resistance function has been tabulated
	\param d_ResTable_vals	tabulated values of the resistance tensor
	\param ResTable_dr	table discretization (in log units)
	\param N                Number of particles
	\param d_group_members  array of particle indices
	\param d_n_neigh        list of number of neighbors for each particle
	\param d_nlist_lub      neighborlist array
	\param d_headlist_lub   indices into the neighborlist for each particle
	\param d_offset         current particle's offsets into the output arrays
	\param d_NEPP		Number of non-zero entries per particle


*/
__global__ void Lubrication_RFE_kernel(
					Scalar *d_Force,
					Scalar shear_rate,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size,
			      		BoxDim box,
					const unsigned int *d_n_neigh,
					unsigned int *d_nlist,
					const unsigned int *d_headlist,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const float ResTable_min,
					const float ResTable_dr,
					const Scalar rlub
					){

  // Index for current thread
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  // Check that thread is within bounds, and only do work if so
  if ( tidx < group_size ) {

    // Square of the cutoff radius
    Scalar rlubsq = rlub * rlub;

    // Particle info for this thread
    unsigned int curr_particle = d_group_members[ tidx ];

    unsigned int head_idx = d_headlist[ curr_particle ]; // Location in head array for neighbors of current particle
    unsigned int n_neigh = d_n_neigh[ curr_particle ]; // Number of neighbors of the nearest particle

    // Neighbor counters
    unsigned int neigh_idx, curr_neigh;

    // Position and rate of strain of current particle
    Scalar4 posi = d_pos[ curr_particle ];
    Scalar Ei[5];
    Ei[0] = 0.0;
    Ei[1] = shear_rate;
    Ei[2] = 0.0;
    Ei[3] = 0.0;
    Ei[4] = 0.0;

    // Map to 3x3
    Scalar Eblocki[3][3] = {0.0};
    Eblocki[0][0] = (1.0/3.0) * ( 2.0 * Ei[0] - Ei[4] );
    Eblocki[0][1] = 0.5 * Ei[1];
    Eblocki[0][2] = 0.5 * Ei[2];
    Eblocki[1][0] = Eblocki[0][1];
    Eblocki[1][1] = (1.0/3.0) * ( -Ei[0] + 2.0 * Ei[4] );
    Eblocki[1][2] = 0.5 * Ei[3];
    Eblocki[2][0] = Eblocki[0][2];
    Eblocki[2][1] = Eblocki[1][2];
    Eblocki[2][2] = (-1.0/3.0) * ( Ei[0] + Ei[4] );

    Scalar3 fi = make_scalar3( 0.0, 0.0, 0.0 );
    Scalar3 li = make_scalar3( 0.0, 0.0, 0.0 );	
	
    // Loop over all the neighbors for the current particle and add those
    // pair entries to the lubrication resistance tensor
    for (neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {

      // Get the current neighbor index
      curr_neigh = d_nlist[ head_idx + neigh_idx ];
		
      // Position of neighbor particle
      Scalar4 posj = d_pos[ curr_neigh ];

      // Rate of strain of neighbor particle
      Scalar Ej[5];
      Ej[0] = 0.0;
      Ej[1] = shear_rate;
      Ej[2] = 0.0;
      Ej[3] = 0.0;
      Ej[4] = 0.0;

      // Map to 3x3
      Scalar Eblockj[3][3] = {0.0};
      Eblockj[0][0] = (1.0/3.0) * ( 2.0 * Ej[0] - Ej[4] );
      Eblockj[0][1] = 0.5 * Ej[1];
      Eblockj[0][2] = 0.5 * Ej[2];
      Eblockj[1][0] = Eblockj[0][1];
      Eblockj[1][1] = (1.0/3.0) * ( -Ej[0] + 2.0 * Ej[4] );
      Eblockj[1][2] = 0.5 * Ej[3];
      Eblockj[2][0] = Eblockj[0][2];
      Eblockj[2][1] = Eblockj[1][2];
      Eblockj[2][2] = (-1.0/3.0) * ( Ej[0] + Ej[4] );

      // Distance vector between current particle and neighbor
      Scalar3 R = make_scalar3( posj.x - posi.x, posj.y - posi.y, posj.z - posi.z );

      // Minimum image
      R = box.minImage(R);

      // Distance magnitude
      Scalar distSqr = dot(R,R);

      // Check that particles are within the hard-sphere cutoff (not all particles in
      // the HOOMD neighborlist are guaranteed to be within the cutoff)
      if ( distSqr < rlubsq ){

	// Distance
	Scalar dist = sqrtf( distSqr );

	Scalar XG11, XG12, YG11, YG12, YH11, YH12;
	// The block below may be commented out to test stress calculation
	// for dense suspensions. (zhoge)
	/*
	if ( dist <= ( 2.0 + ResTable_min ) ){
	  // Table is strided by 22
	  XG11 = d_ResTable_vals[ 10 ];
	  XG12 = d_ResTable_vals[ 11 ];
	  YG11 = d_ResTable_vals[ 12 ];
	  YG12 = d_ResTable_vals[ 13 ];
	  YH11 = d_ResTable_vals[ 14 ];
	  YH12 = d_ResTable_vals[ 15 ];
	}
	*/
	if ( dist <= 2.001 ){
	  // In Stokes_ResistanceTable.cc, h_ResTable_dist.data[232] = 2.000997;
	  // Table is strided by 22
	  int i_regl = 232*22; //lubrication regularization (due to roughness) 
	  XG11 = d_ResTable_vals[ i_regl + 10 ];
	  XG12 = d_ResTable_vals[ i_regl + 11 ];
	  YG11 = d_ResTable_vals[ i_regl + 12 ];
	  YG12 = d_ResTable_vals[ i_regl + 13 ];
	  YH11 = d_ResTable_vals[ i_regl + 14 ];
	  YH12 = d_ResTable_vals[ i_regl + 15 ];
	}
	else {

	  // Get the index of the nearest entry below the current distance in the distance array
	  // NOTE: Distances are logarithmically spaced in the tabulation
	  int ind = log10f( ( dist - 2.0 ) / ResTable_min ) / ResTable_dr;

	  // Get the values from the distance array for interpolation
	  Scalar dist_lower = d_ResTable_dist[ ind ];
	  Scalar dist_upper = d_ResTable_dist[ ind + 1 ];

	  // Read the scalar resistance coefficients from the array (lower and upper values
	  // for interpolation)
	  //
	  // Table is strided by 32 to coalesce reads
	  Scalar XG11_lower = d_ResTable_vals[ 22 * ind + 10 ];
	  Scalar XG12_lower = d_ResTable_vals[ 22 * ind + 11 ];
	  Scalar YG11_lower = d_ResTable_vals[ 22 * ind + 12 ];
	  Scalar YG12_lower = d_ResTable_vals[ 22 * ind + 13 ];
	  Scalar YH11_lower = d_ResTable_vals[ 22 * ind + 14 ];
	  Scalar YH12_lower = d_ResTable_vals[ 22 * ind + 15 ];

	  Scalar XG11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 10 ];
	  Scalar XG12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 11 ];
	  Scalar YG11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 12 ];
	  Scalar YG12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 13 ];
	  Scalar YH11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 14 ];
	  Scalar YH12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 15 ];

	  // Linear interpolation of the Table values
	  Scalar fac = ( dist - dist_lower  ) / ( dist_upper - dist_lower );

	  XG11 = XG11_lower + ( XG11_upper - XG11_lower ) * fac;
	  XG12 = XG12_lower + ( XG12_upper - XG12_lower ) * fac;
	  YG11 = YG11_lower + ( YG11_upper - YG11_lower ) * fac;
	  YG12 = YG12_lower + ( YG12_upper - YG12_lower ) * fac;
	  YH11 = YH11_lower + ( YH11_upper - YH11_lower ) * fac;
	  YH12 = YH12_lower + ( YH12_upper - YH12_lower ) * fac;

	}


	// Geometric quantities
	Scalar rx = R.x / dist;
	Scalar ry = R.y / dist;
	Scalar rz = R.z / dist;
	Scalar3 r = make_scalar3( rx, ry, rz );

	Scalar3 Edri = make_scalar3(    Eblocki[0][0] * r.x + Eblocki[0][1] * r.y + Eblocki[0][2] * r.z,
					Eblocki[1][0] * r.x + Eblocki[1][1] * r.y + Eblocki[1][2] * r.z,
					Eblocki[2][0] * r.x + Eblocki[2][1] * r.y + Eblocki[2][2] * r.z );
	Scalar3 Edrj = make_scalar3(    Eblockj[0][0] * r.x + Eblockj[0][1] * r.y + Eblockj[0][2] * r.z,
					Eblockj[1][0] * r.x + Eblockj[1][1] * r.y + Eblockj[1][2] * r.z,
					Eblockj[2][0] * r.x + Eblockj[2][1] * r.y + Eblockj[2][2] * r.z );
	//E multiply the normal vector twice
	Scalar rdEdri = r.x * Edri.x + r.y * Edri.y + r.z * Edri.z;
	Scalar rdEdrj = r.x * Edrj.x + r.y * Edrj.y + r.z * Edrj.z;


	// epsr = [  0   rz -ry ]
	//        [ -rz  0   rx ]
	// 	  [  ry -rx  0  ]
	//
	// epsr dot Edr
	Scalar3 epsrdEdri = make_scalar3( (  r.z * Edri.y - r.y * Edri.z ),
					  ( -r.z * Edri.x + r.x * Edri.z ),
					  (  r.y * Edri.x - r.x * Edri.y ) );
	Scalar3 epsrdEdrj = make_scalar3( (  r.z * Edrj.y - r.y * Edrj.z ),
					  ( -r.z * Edrj.x + r.x * Edrj.z ),
					  (  r.y * Edrj.x - r.x * Edrj.y ) );

	// Force on fluid by i (or 1)
	//
	// F1 = G^t_11 * (0-Einf) + G^t_12 * (0-Einf)  [G^t are in ijk]
	//    = -G_11 * Einf -G_21 * Einf              [G   are in jki]
	//    = XG11 * (-P_n:Einf) r -YG11 * (2*Einf*r -2*P_n:Einf r)
	//      XG21 * (-P_n:Einf) r -YG21 * (2*Einf*r -2*P_n:Einf r)
	//
	// Symmetry condition:
	Scalar XG21 = -XG12;
	Scalar YG21 = -YG12;
				
	fi= (XG11 - 2.0*YG11) * (-rdEdri) * r + 2.0 * YG11 * (-Edri)
	  + (XG21 - 2.0*YG21) * (-rdEdrj) * r + 2.0 * YG21 * (-Edrj);
				
	// Torque on fluid by i (or 1)
	//
	// L1 = H^t_11 * (0-Einf) + H^t_12 * (0-Einf)   [H^t in ijk]
	//    = -H_11 * Einf -H_21 * Einf               [H   in jki]
	//    = -YH11 * (-epsrdEdri*2) - YH21 * (-epsrdEdri*2)
	//
	// Symmetry condition:
	Scalar YH21 = YH12;

	li= YH11 * (2.0 * epsrdEdri)
	  + YH21 * (2.0 * epsrdEdrj);
				
	// Write to output
	d_Force[ 6*curr_particle     ] += fi.x; 
	d_Force[ 6*curr_particle + 1 ] += fi.y; 
	d_Force[ 6*curr_particle + 2 ] += fi.z; 
	d_Force[ 6*curr_particle + 3 ] += li.x; 
	d_Force[ 6*curr_particle + 4 ] += li.y; 
	d_Force[ 6*curr_particle + 5 ] += li.z; 
			    
      } // check if neighbor is within cutoff
    } // Loop over neighbors
  } // Check for thread in bounds
}









/*!
	Compute the product of the RSE tensor with a vector ( ALL PARTICLES SAME SIZE ) -- give one thread per particle

	These simulations are constructed so that, if there is strain,
		x is the flow direction
		y is the gradient direction
		z is the vorticity direction
	therefore,
		Einf = 	[ 0 g 0 ]
			[ g 0 0 ]
			[ 0 0 0 ]
	where g is the shear rate. Therefore, the strain rate on each particle (due to the imposed straining flow)
	is identical, so the only needed quantity is the global shear rate.

	\param d_pos  		particle positions
	\param box		simulation box information
	\param d_ResTable_dist	distances for which the resistance function has been tabulated
	\param d_ResTable_vals	tabulated values of the resistance tensor
	\param ResTable_dr	table discretization (in log units)
	\param group_size                Number of particles
	\param d_group_members  array of particle indices
	\param d_n_neigh        list of number of neighbors for each particle
	\param d_nlist_lub      neighborlist array
	\param d_headlist_lub   indices into the neighborlist for each particle
	\param d_offset         current particle's offsets into the output arrays
	\param d_NEPP		Number of non-zero entries per particle


*/
__global__ void Lubrication_RSE_kernel(
					Scalar *d_Stresslet,
					Scalar shear_rate,
					int group_size,
					unsigned int *d_group_members,
					const unsigned int *d_n_neigh,
					unsigned int *d_nlist,
					const unsigned int *d_headlist,
					Scalar4 *d_pos,
			      		BoxDim box,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const float ResTable_min,
					const float ResTable_dr
					){

	// Index for current thread
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check that thread is within bounds, and only do work if so
	if ( tidx < group_size ) {

		// Particle info for this thread
		unsigned int curr_particle = d_group_members[ tidx ];

		unsigned int head_idx = d_headlist[ curr_particle ]; // Location in head array for neighbors of current particle
		unsigned int n_neigh = d_n_neigh[ curr_particle ]; // Number of neighbors of the nearest particle

		// Neighbor counters
		unsigned int neigh_idx, curr_neigh;

		// Position and rate of strain of current particle
		Scalar4 posi = d_pos[ curr_particle ];
		Scalar Ei[5];
		Ei[0] = 0.0;
		Ei[1] = shear_rate;
		Ei[2] = 0.0;
		Ei[3] = 0.0;
		Ei[4] = 0.0;

		// Map to 3x3
		Scalar Eblocki[3][3] = {0.0};
		Eblocki[0][0] = (1.0/3.0) * ( 2.0 * Ei[0] - Ei[4] );
		Eblocki[0][1] = 0.5 * Ei[1];
		Eblocki[0][2] = 0.5 * Ei[2];
		Eblocki[1][0] = Eblocki[0][1];
		Eblocki[1][1] = (1.0/3.0) * ( -Ei[0] + 2.0 * Ei[4] );
		Eblocki[1][2] = 0.5 * Ei[3];
		Eblocki[2][0] = Eblocki[0][2];
		Eblocki[2][1] = Eblocki[1][2];
		Eblocki[2][2] = (-1.0/3.0) * ( Ei[0] + Ei[4] );

		// Initialize stresslet
		Scalar Si[5] = { 0.0 };

		// Loop over all the neighbors for the current particle and add those
		// pair entries to the lubrication resistance tensor
		for (neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {

			// Get the current neighbor index
			curr_neigh = d_nlist[ head_idx + neigh_idx ];

			// Position and rate of strain of neighbor particle
			Scalar4 posj = d_pos[ curr_neigh ];
			Scalar Ej[5];
			Ej[0] = 0.0;
			Ej[1] = shear_rate;
			Ej[2] = 0.0;
			Ej[3] = 0.0;
			Ej[4] = 0.0;

			// Map to 3x3
			Scalar Eblockj[3][3] = {0.0};
			Eblockj[0][0] = (1.0/3.0) * ( 2.0 * Ej[0] - Ej[4] );
			Eblockj[0][1] = 0.5 * Ej[1];
			Eblockj[0][2] = 0.5 * Ej[2];
			Eblockj[1][0] = Eblockj[0][1];
			Eblockj[1][1] = (1.0/3.0) * ( -Ej[0] + 2.0 * Ej[4] );
			Eblockj[1][2] = 0.5 * Ej[3];
			Eblockj[2][0] = Eblockj[0][2];
			Eblockj[2][1] = Eblockj[1][2];
			Eblockj[2][2] = (-1.0/3.0) * ( Ej[0] + Ej[4] );

			// Distance vector between current particle and neighbor
			Scalar3 R = make_scalar3( posj.x - posi.x, posj.y - posi.y, posj.z - posi.z );
			R = box.minImage(R);
			Scalar distSqr = dot(R,R);

			if ( distSqr < 16.0 ){

				// Distance
				Scalar dist = sqrtf( distSqr );

				Scalar XM11, XM12, YM11, YM12, ZM11, ZM12;
				// The block below may be commented out to test stress calculation
				// for dense suspensions. (zhoge)
				/*
				  if ( dist <= ( 2.0 + ResTable_min ) ){
				  // Table is strided by 22
				  XM11 = d_ResTable_vals[ 16 ];
				  XM12 = d_ResTable_vals[ 17 ];
				  YM11 = d_ResTable_vals[ 18 ];
				  YM12 = d_ResTable_vals[ 19 ];
				  ZM11 = d_ResTable_vals[ 20 ];
				  ZM12 = d_ResTable_vals[ 21 ];
				}
				*/
				if ( dist <= 2.001 ){
				  // In Stokes_ResistanceTable.cc, h_ResTable_dist.data[232] = 2.000997;
				  // Table is strided by 22
				  int i_regl = 232*22; //lubrication regularization (due to roughness) 
				  XM11 = d_ResTable_vals[ i_regl + 16 ];
				  XM12 = d_ResTable_vals[ i_regl + 17 ];
				  YM11 = d_ResTable_vals[ i_regl + 18 ];
				  YM12 = d_ResTable_vals[ i_regl + 19 ];
				  ZM11 = d_ResTable_vals[ i_regl + 20 ];
				  ZM12 = d_ResTable_vals[ i_regl + 21 ];
				}
				else {

					// Get the index of the nearest entry below the current distance in the distance array
					// NOTE: Distances are logarithmically spaced in the tabulation
					int ind = log10f( ( dist - 2.0 ) / ResTable_min ) / ResTable_dr;

					// Get the values from the distance array for interpolation
					Scalar dist_lower = d_ResTable_dist[ ind ];
					Scalar dist_upper = d_ResTable_dist[ ind + 1 ];

					// Read the scalar resistance coefficients from the array (lower and upper values
					// for interpolation)
					//
					// Table is strided by 2
					Scalar XM11_lower = d_ResTable_vals[ 22 * ind + 16 ];
                        		Scalar XM12_lower = d_ResTable_vals[ 22 * ind + 17 ];
                        		Scalar YM11_lower = d_ResTable_vals[ 22 * ind + 18 ];
                        		Scalar YM12_lower = d_ResTable_vals[ 22 * ind + 19 ];
                        		Scalar ZM11_lower = d_ResTable_vals[ 22 * ind + 20 ];
                        		Scalar ZM12_lower = d_ResTable_vals[ 22 * ind + 21 ];

                        		Scalar XM11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 16 ];
                        		Scalar XM12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 17 ];
                        		Scalar YM11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 18 ];
                        		Scalar YM12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 19 ];
                        		Scalar ZM11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 20 ];
                        		Scalar ZM12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 21 ];

					// Linear interpolation of the Table values
					Scalar fac = ( dist - dist_lower  ) / ( dist_upper - dist_lower );

                        		XM11 = XM11_lower + ( XM11_upper - XM11_lower ) * fac;
                        		XM12 = XM12_lower + ( XM12_upper - XM12_lower ) * fac;
                        		YM11 = YM11_lower + ( YM11_upper - YM11_lower ) * fac;
                        		YM12 = YM12_lower + ( YM12_upper - YM12_lower ) * fac;
                        		ZM11 = ZM11_lower + ( ZM11_upper - ZM11_lower ) * fac;
                        		ZM12 = ZM12_lower + ( ZM12_upper - ZM12_lower ) * fac;
				}

				// Geometric quantities
				Scalar rx = R.x / dist;
				Scalar ry = R.y / dist;
				Scalar rz = R.z / dist;
				Scalar3 r = make_scalar3( rx, ry, rz );

				// E dot r once
				Scalar3 Edri = make_scalar3( 	Eblocki[0][0] * r.x + Eblocki[0][1] * r.y + Eblocki[0][2] * r.z,
								Eblocki[1][0] * r.x + Eblocki[1][1] * r.y + Eblocki[1][2] * r.z,
								Eblocki[2][0] * r.x + Eblocki[2][1] * r.y + Eblocki[2][2] * r.z );
				Scalar3 Edrj = make_scalar3( 	Eblockj[0][0] * r.x + Eblockj[0][1] * r.y + Eblockj[0][2] * r.z,
								Eblockj[1][0] * r.x + Eblockj[1][1] * r.y + Eblockj[1][2] * r.z,
								Eblockj[2][0] * r.x + Eblockj[2][1] * r.y + Eblockj[2][2] * r.z );

				// E dot r twice
				Scalar rdEdri = r.x * Edri.x + r.y * Edri.y + r.z * Edri.z;
				Scalar rdEdrj = r.x * Edrj.x + r.y * Edrj.y + r.z * Edrj.z;

				// Value.
				//
				// R_ijkl = XM * 3/2 * ( ri*rj - 1/3 * delta_ij ) * ( rk*rl- 1/3 * delta_kl ) +
				//          YM * 1/2 * ( ri*delta_jl*rk + rj*delta_il*rk + ri*delta_jk*rl +
				//                       rj*delta_ik*rl - 4*ri*rj*rk*rl ) +
				//          ZM * 1/2 * ( delta_ik*delta_jl + delta_jk*delta_il -
				//                       delta_ij*delta_kl + ri*rj*delta_kl +
				//                       delta_ij*rk*rl + ri*rj*rk*rl - ri*delta_jl*rk -
				//                       rj*delta_il*rk - ri*delta_jk*rl - rj*delta_il*rl )

			//madhu
				Si[0] += 1.5 * XM11 * ( r.x*r.x - 1.0/3.0 )*rdEdri +
					 0.5 * YM11 * ( 2.0*r.x*Edri.x + 2.0*r.x*Edri.x - 4.0*rdEdri*r.x*r.x ) +
					 0.5 * ZM11 * ( 2*Eblocki[0][0] + ( 1.0 + r.x*r.x )*rdEdri - 2.0*r.x*Edri.x - 2.0*r.x*Edri.x ) +
					 1.5 * XM12 * ( r.x*r.x - 1.0/3.0 )*rdEdrj +
					 0.5 * YM12 * ( 2.0*r.x*Edrj.x + 2.0*r.x*Edrj.x - 4.0*rdEdrj*r.x*r.x ) +
					 0.5 * ZM12 * ( 2*Eblockj[0][0] + ( 1.0 + r.x*r.x )*rdEdrj - 2.0*r.x*Edrj.x - 2.0*r.x*Edrj.x );

				Si[1] += 1.5 * XM11 * ( r.x*r.y )*rdEdri +
					 0.5 * YM11 * ( 2.0*r.x*Edri.y + 2.0*r.y*Edri.x - 4.0*rdEdri*r.x*r.y ) +
				  //zhoge//0.5 * ZM11 * ( 2*Ei[1] + ( r.x*r.y )*rdEdri - 2.0*r.x*Edri.y - 2.0*r.y*Edri.x ) +
					 0.5 * ZM11 * ( 2*Eblocki[0][1] + ( r.x*r.y )*rdEdri - 2.0*r.x*Edri.y - 2.0*r.y*Edri.x ) +
					 1.5 * XM12 * ( r.x*r.y )*rdEdrj +
					 0.5 * YM12 * ( 2.0*r.x*Edrj.y + 2.0*r.y*Edrj.x - 4.0*rdEdrj*r.x*r.y ) +
				  //zhoge//0.5 * ZM12 * ( 2*Ej[1] + ( r.x*r.y )*rdEdrj - 2.0*r.x*Edrj.y - 2.0*r.y*Edrj.x );
					 0.5 * ZM12 * ( 2*Eblockj[0][1] + ( r.x*r.y )*rdEdrj - 2.0*r.x*Edrj.y - 2.0*r.y*Edrj.x );

				Si[2] += 1.5 * XM11 * ( r.x*r.z )*rdEdri +
					 0.5 * YM11 * ( 2.0*r.x*Edri.z + 2.0*r.z*Edri.x - 4.0*rdEdri*r.x*r.z ) +
				  //zhoge//0.5 * ZM11 * ( 2*Ei[2] + ( r.x*r.z )*rdEdri - 2.0*r.x*Edri.z - 2.0*r.z*Edri.x ) +
					 0.5 * ZM11 * ( 2*Eblocki[0][2] + ( r.x*r.z )*rdEdri - 2.0*r.x*Edri.z - 2.0*r.z*Edri.x ) +
					 1.5 * XM12 * ( r.x*r.z )*rdEdrj +
					 0.5 * YM12 * ( 2.0*r.x*Edrj.z + 2.0*r.z*Edrj.x - 4.0*rdEdrj*r.x*r.z ) +
				  //zhoge//0.5 * ZM12 * ( 2*Ej[2] + ( r.x*r.z )*rdEdrj - 2.0*r.x*Edrj.z - 2.0*r.z*Edrj.x );
				         0.5 * ZM12 * ( 2*Eblockj[0][2] + ( r.x*r.z )*rdEdrj - 2.0*r.x*Edrj.z - 2.0*r.z*Edrj.x );

				Si[4] += 1.5 * XM11 * ( r.y*r.z )*rdEdri +
					 0.5 * YM11 * ( 2.0*r.y*Edri.z + 2.0*r.z*Edri.y - 4.0*rdEdri*r.y*r.z ) +
				  //zhoge//0.5 * ZM11 * ( 2*Ei[3] + ( r.y*r.z )*rdEdri - 2.0*r.y*Edri.z - 2.0*r.z*Edri.y ) +
					 0.5 * ZM11 * ( 2*Eblocki[1][2] + ( r.y*r.z )*rdEdri - 2.0*r.y*Edri.z - 2.0*r.z*Edri.y ) +
					 1.5 * XM12 * ( r.y*r.z )*rdEdrj +
					 0.5 * YM12 * ( 2.0*r.y*Edrj.z + 2.0*r.z*Edrj.y - 4.0*rdEdrj*r.y*r.z ) +
				  //zhoge//0.5 * ZM12 * ( 2*Ej[3] + ( r.y*r.z )*rdEdrj - 2.0*r.y*Edrj.z - 2.0*r.z*Edrj.y );
				         0.5 * ZM12 * ( 2*Eblockj[1][2] + ( r.y*r.z )*rdEdrj - 2.0*r.y*Edrj.z - 2.0*r.z*Edrj.y );

				Si[3] += 1.5 * XM11 * ( r.y*r.y - 1.0/3.0 )*rdEdri +
					 0.5 * YM11 * ( 2.0*r.y*Edri.y + 2.0*r.y*Edri.y - 4.0*rdEdri*r.y*r.y ) +
					 0.5 * ZM11 * ( 2*Eblocki[1][1] + ( 1.0 + r.y*r.y )*rdEdri - 2.0*r.y*Edri.y - 2.0*r.y*Edri.y ) +
					 1.5 * XM12 * ( r.y*r.y - 1.0/3.0 )*rdEdrj +
					 0.5 * YM12 * ( 2.0*r.y*Edrj.y + 2.0*r.y*Edrj.y - 4.0*rdEdrj*r.y*r.y ) +
					 0.5 * ZM12 * ( 2*Eblockj[1][1] + ( 1.0 + r.y*r.y )*rdEdrj - 2.0*r.y*Edrj.y - 2.0*r.y*Edrj.y );

			//madhu
			} // Check for particles in bounds

		} // Loop over neighbors

		// Write to output
		d_Stresslet[ 5*curr_particle ]     += Si[0];
		d_Stresslet[ 5*curr_particle + 1 ] += Si[1];
		d_Stresslet[ 5*curr_particle + 2 ] += Si[2];
		d_Stresslet[ 5*curr_particle + 3 ] += Si[3];
		d_Stresslet[ 5*curr_particle + 4 ] += Si[4];

	} // Check for thread in bounds

}

/*!
	Build the product of the RSE tensor with a vector ( ALL PARTICLES SAME SIZE ) -- give one thread per particle

	\param d_pos  		particle positions
	\param box		simulation box information
	\param d_ResTable_dist	distances for which the resistance function has been tabulated
	\param d_ResTable_vals	tabulated values of the resistance tensor
	\param ResTable_dr	table discretization (in log units)
	\param group_size                Number of particles
	\param d_group_members  array of particle indices
	\param d_n_neigh        list of number of neighbors for each particle
	\param d_nlist_lub      neighborlist array
	\param d_headlist_lub   indices into the neighborlist for each particle
	\param d_offset         current particle's offsets into the output arrays
	\param d_NEPP		Number of non-zero entries per particle


*/
__global__ void Lubrication_RSEgeneral_kernel(
					Scalar *d_Stresslet,
					Scalar *d_Strain,
					int group_size,
					unsigned int *d_group_members,
					const unsigned int *d_n_neigh,
					unsigned int *d_nlist,
					const unsigned int *d_headlist,
					Scalar4 *d_pos,
			      		BoxDim box,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const float ResTable_min,
					const float ResTable_dr
					){

	// Index for current thread
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check that thread is within bounds, and only do work if so
	if ( tidx < group_size ) {

		// Particle info for this thread
		unsigned int curr_particle = d_group_members[ tidx ];

		unsigned int head_idx = d_headlist[ curr_particle ]; // Location in head array for neighbors of current particle
		unsigned int n_neigh = d_n_neigh[ curr_particle ]; // Number of neighbors of the nearest particle

		// Neighbor counters
		unsigned int neigh_idx, curr_neigh;



		for (int ii=0;ii<10;ii++)
		{
		//	printf("%f, ",d_Strain[ii]);
			d_Stresslet[ii]=0.0;
		}

		// Position and rate of strain of current particle
		Scalar4 posi = d_pos[ curr_particle ];
		Scalar Ei[5];
                //Ei[0] = d_Strain[ 5*group_size + 0 ];
                //Ei[1] = d_Strain[ 5*group_size + 1 ];
                //Ei[2] = d_Strain[ 5*group_size + 2 ];
                //Ei[3] = d_Strain[ 5*group_size + 3 ];
                //Ei[4] = d_Strain[ 5*group_size + 4 ];

		Ei[0] = d_Strain[  5*curr_particle + 0 ];
		Ei[1] = d_Strain[  5*curr_particle + 1 ];
		Ei[2] = d_Strain[  5*curr_particle + 2 ];
		Ei[3] = d_Strain[  5*curr_particle + 3 ];
		Ei[4] = d_Strain[  5*curr_particle + 4 ];

		// Map to 3x3
		Scalar Eblocki[3][3] = {0.0};
		Eblocki[0][0] = (1.0/3.0) * ( 2.0 * Ei[0] - Ei[4] );
		Eblocki[0][1] = 0.5 * Ei[1];
		Eblocki[0][2] = 0.5 * Ei[2];
		Eblocki[1][0] = Eblocki[0][1];
		Eblocki[1][1] = (1.0/3.0) * ( -Ei[0] + 2.0 * Ei[4] );
		Eblocki[1][2] = 0.5 * Ei[3];
		Eblocki[2][0] = Eblocki[0][2];
		Eblocki[2][1] = Eblocki[1][2];
		Eblocki[2][2] = (-1.0/3.0) * ( Ei[0] + Ei[4] );

		// Initialize stresslet
		Scalar Si[5] = { 0.0 };

		// Loop over all the neighbors for the current particle and add those
		// pair entries to the lubrication resistance tensor
		for (neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {

			// Get the current neighbor index
			curr_neigh = d_nlist[ head_idx + neigh_idx ];


			// Position and rate of strain of neighbor particle
			Scalar4 posj = d_pos[ curr_neigh ];
			Scalar Ej[5];
			Ej[0] = d_Strain[ 5*curr_neigh ];
			Ej[1] = d_Strain[ 5*curr_neigh + 1 ];
			Ej[2] = d_Strain[ 5*curr_neigh + 2 ];
			Ej[3] = d_Strain[ 5*curr_neigh + 3 ];
			Ej[4] = d_Strain[ 5*curr_neigh + 4 ];

			// Map to 3x3
			Scalar Eblockj[3][3] = {0.0};
			Eblockj[0][0] = (1.0/3.0) * ( 2.0 * Ej[0] - Ej[4] );
			Eblockj[0][1] = 0.5 * Ej[1];
			Eblockj[0][2] = 0.5 * Ej[2];
			Eblockj[1][0] = Eblockj[0][1];
			Eblockj[1][1] = (1.0/3.0) * ( -Ej[0] + 2.0 * Ej[4] );
			Eblockj[1][2] = 0.5 * Ej[3];
			Eblockj[2][0] = Eblockj[0][2];
			Eblockj[2][1] = Eblockj[1][2];
			Eblockj[2][2] = (-1.0/3.0) * ( Ej[0] + Ej[4] );

			// Distance vector between current particle and neighbor
			Scalar3 R = make_scalar3( posj.x - posi.x, posj.y - posi.y, posj.z - posi.z );
			R = box.minImage(R);
			Scalar distSqr = dot(R,R);

			if ( distSqr < 16.0 ){

				// Distance
				Scalar dist = sqrtf( distSqr );

				Scalar XM11, XM12, YM11, YM12, ZM11, ZM12;
				if ( dist <= (2.0 + ResTable_min ) ){
					XM11 = d_ResTable_vals[ 16 ];
                        		XM12 = d_ResTable_vals[ 17 ];
                        		YM11 = d_ResTable_vals[ 18 ];
                        		YM12 = d_ResTable_vals[ 19 ];
                        		ZM11 = d_ResTable_vals[ 20 ];
                        		ZM12 = d_ResTable_vals[ 21 ];
				}
				else {

					// Get the index of the nearest entry below the current distance in the distance array
					// NOTE: Distances are logarithmically spaced in the tabulation
					int ind = log10f( ( dist - 2.0 ) / ResTable_min ) / ResTable_dr;

					// Get the values from the distance array for interpolation
					Scalar dist_lower = d_ResTable_dist[ ind ];
					Scalar dist_upper = d_ResTable_dist[ ind + 1 ];

					// Read the scalar resistance coefficients from the array (lower and upper values
					// for interpolation)
					//
					// Table is strided by 2
					Scalar XM11_lower = d_ResTable_vals[ 22 * ind + 16 ];
                        		Scalar XM12_lower = d_ResTable_vals[ 22 * ind + 17 ];
                        		Scalar YM11_lower = d_ResTable_vals[ 22 * ind + 18 ];
                        		Scalar YM12_lower = d_ResTable_vals[ 22 * ind + 19 ];
                        		Scalar ZM11_lower = d_ResTable_vals[ 22 * ind + 20 ];
                        		Scalar ZM12_lower = d_ResTable_vals[ 22 * ind + 21 ];

                        		Scalar XM11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 16 ];
                        		Scalar XM12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 17 ];
                        		Scalar YM11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 18 ];
                        		Scalar YM12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 19 ];
                        		Scalar ZM11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 20 ];
                        		Scalar ZM12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 21 ];

					// Linear interpolation of the Table values
					Scalar fac = ( dist - dist_lower  ) / ( dist_upper - dist_lower );

                        		XM11 = XM11_lower + ( XM11_upper - XM11_lower ) * fac;
                        		XM12 = XM12_lower + ( XM12_upper - XM12_lower ) * fac;
                        		YM11 = YM11_lower + ( YM11_upper - YM11_lower ) * fac;
                        		YM12 = YM12_lower + ( YM12_upper - YM12_lower ) * fac;
                        		ZM11 = ZM11_lower + ( ZM11_upper - ZM11_lower ) * fac;
                        		ZM12 = ZM12_lower + ( ZM12_upper - ZM12_lower ) * fac;
				}


				// Geometric quantities
				Scalar rx = R.x / dist;
				Scalar ry = R.y / dist;
				Scalar rz = R.z / dist;
				Scalar3 r = make_scalar3( rx, ry, rz );

				Scalar3 Edri = make_scalar3( 	Eblocki[0][0] * r.x + Eblocki[0][1] * r.y + Eblocki[0][2] * r.z,
								Eblocki[1][0] * r.x + Eblocki[1][1] * r.y + Eblocki[1][2] * r.z,
								Eblocki[2][0] * r.x + Eblocki[2][1] * r.y + Eblocki[2][2] * r.z );
				Scalar3 Edrj = make_scalar3( 	Eblockj[0][0] * r.x + Eblockj[0][1] * r.y + Eblockj[0][2] * r.z,
								Eblockj[1][0] * r.x + Eblockj[1][1] * r.y + Eblockj[1][2] * r.z,
								Eblockj[2][0] * r.x + Eblockj[2][1] * r.y + Eblockj[2][2] * r.z );

				Scalar rdEdri = r.x * Edri.x + r.y * Edri.y + r.z * Edri.z;
				Scalar rdEdrj = r.x * Edrj.x + r.y * Edrj.y + r.z * Edrj.z;

				// Value
				//
				// R_ijkl = XM * 3/2 * ( ri*rj - 1/3 * delta_ij ) * ( rk*rl- 1/3 * delta_kl ) +
				//          YM * 1/2 * ( ri*delta_jl*rk + rj*delta_il*rk + ri*delta_jk*rl +
				//                       rj*delta_ik*rl - 4*ri*rj*rk*rl ) +
				//          ZM * 1/2 * ( delta_ik*delta_jl + delta_jk*delta_il -
				//                       delta_ij*delta_kl + ri*rj*delta_kl +
				//                       delta_ij*rk*rl + ri*rj*rk*rl - ri*delta_jl*rk -
				//                       rj*delta_il*rk - ri*delta_jk*rl - rj*delta_il*rl )


                                Si[0] +=        1.5 * XM11 * ( r.x*r.x - 1.0/3.0 )*rdEdri +
                                                0.5 * YM11 * ( 2.0*r.x*Edri.x + 2.0*r.x*Edri.x - 4.0*rdEdri*r.x*r.x ) +
                                                0.5 * ZM11 * ( 2*Eblocki[0][0] + ( 1.0 + r.x*r.x )*rdEdri - 2.0*r.x*Edri.x - 2.0*r.x*Edri.x ) +
                                                1.5 * XM12 * ( r.x*r.x - 1.0/3.0 )*rdEdrj +
                                                0.5 * YM12 * ( 2.0*r.x*Edrj.x + 2.0*r.x*Edrj.x - 4.0*rdEdrj*r.x*r.x ) +
                                                0.5 * ZM12 * ( 2*Eblockj[0][0] + ( 1.0 + r.x*r.x )*rdEdrj - 2.0*r.x*Edrj.x - 2.0*r.x*Edrj.x );

                                Si[1] +=        1.5 * XM11 * ( r.x*r.y )*rdEdri +
                                                0.5 * YM11 * ( 2.0*r.x*Edri.y + 2.0*r.y*Edri.x - 4.0*rdEdri*r.x*r.y ) +
                                                0.5 * ZM11 * ( 2*Ei[1] + ( r.x*r.y )*rdEdri - 2.0*r.x*Edri.y - 2.0*r.y*Edri.x ) +
                                                1.5 * XM12 * ( r.x*r.y )*rdEdrj +
                                                0.5 * YM12 * ( 2.0*r.x*Edrj.y + 2.0*r.y*Edrj.x - 4.0*rdEdrj*r.x*r.y ) +
                                                0.5 * ZM12 * ( 2*Ej[1] + ( r.x*r.y )*rdEdrj - 2.0*r.x*Edrj.y - 2.0*r.y*Edrj.x );

                                Si[2] +=        1.5 * XM11 * ( r.x*r.z )*rdEdri +
                                                0.5 * YM11 * ( 2.0*r.x*Edri.z + 2.0*r.z*Edri.x - 4.0*rdEdri*r.x*r.z ) +
                                                0.5 * ZM11 * ( 2*Ei[2] + ( r.x*r.z )*rdEdri - 2.0*r.x*Edri.z - 2.0*r.z*Edri.x ) +
                                                1.5 * XM12 * ( r.x*r.z )*rdEdrj +
                                                0.5 * YM12 * ( 2.0*r.x*Edrj.z + 2.0*r.z*Edrj.x - 4.0*rdEdrj*r.x*r.z ) +
                                                0.5 * ZM12 * ( 2*Ej[2] + ( r.x*r.z )*rdEdrj - 2.0*r.x*Edrj.z - 2.0*r.z*Edrj.x );

                                Si[3] +=        1.5 * XM11 * ( r.y*r.z )*rdEdri +
                                                0.5 * YM11 * ( 2.0*r.y*Edri.z + 2.0*r.z*Edri.y - 4.0*rdEdri*r.y*r.z ) +
                                                0.5 * ZM11 * ( 2*Ei[3] + ( r.y*r.z )*rdEdri - 2.0*r.y*Edri.z - 2.0*r.z*Edri.y ) +
                                                1.5 * XM12 * ( r.y*r.z )*rdEdrj +
                                                0.5 * YM12 * ( 2.0*r.y*Edrj.z + 2.0*r.z*Edrj.y - 4.0*rdEdrj*r.y*r.z ) +
                                                0.5 * ZM12 * ( 2*Ej[3] + ( r.y*r.z )*rdEdrj - 2.0*r.y*Edrj.z - 2.0*r.z*Edrj.y );

                                Si[4] +=        1.5 * XM11 * ( r.y*r.y - 1.0/3.0 )*rdEdri +
                                                0.5 * YM11 * ( 2.0*r.y*Edri.y + 2.0*r.y*Edri.y - 4.0*rdEdri*r.y*r.y ) +
                                                0.5 * ZM11 * ( 2*Eblocki[1][1] + ( 1.0 + r.y*r.y )*rdEdri - 2.0*r.y*Edri.y - 2.0*r.y*Edri.y ) +
                                                1.5 * XM12 * ( r.y*r.y - 1.0/3.0 )*rdEdrj +
                                                0.5 * YM12 * ( 2.0*r.y*Edrj.y + 2.0*r.y*Edrj.y - 4.0*rdEdrj*r.y*r.y ) +
                                                0.5 * ZM12 * ( 2*Eblockj[1][1] + ( 1.0 + r.y*r.y )*rdEdrj - 2.0*r.y*Edrj.y - 2.0*r.y*Edrj.y );


			} // Check for particles in bounds

		} // Loop over neighbors


		// Write to output
		d_Stresslet[ 5*curr_particle ]     += Si[0];
		d_Stresslet[ 5*curr_particle + 1 ] += Si[1];
		d_Stresslet[ 5*curr_particle + 2 ] += Si[2];
		d_Stresslet[ 5*curr_particle + 3 ] += Si[3];
		d_Stresslet[ 5*curr_particle + 4 ] += Si[4];

	} // Check for thread in bounds

}
/*!
	Matrix-vector product for the RFE lubrication tensor

	These simulations are constructed so that, if there is strain,
		x is the flow direction
		y is the gradient direction
		z is the vorticity direction
	therefore,
		Einf = 	[ 0 g 0 ]
			[ g 0 0 ]
			[ 0 0 0 ]
	where g is the shear rate. Therefore, the strain rate on each particle (due to the imposed straining flow)
	is identical, so the only needed quantity is the global shear rate.
	ALL PARTICLES SAME SIZE -- give one thread per particle
	\param d_pos  		particle positions
	\param box		simulation box information
	\param d_ResTable_dist	distances for which the resistance function has been tabulated
	\param d_ResTable_vals	tabulated values of the resistance tensor
	\param ResTable_dr	table discretization (in log units)
	\param N                Number of particles
	\param d_group_members  array of particle indices
	\param d_n_neigh        list of number of neighbors for each particle
	\param d_nlist_lub      neighborlist array
	\param d_headlist_lub   indices into the neighborlist for each particle
	\param d_offset         current particle's offsets into the output arrays
	\param d_NEPP		Number of non-zero entries per particle
*/
__global__ void Lubrication_RFEgeneral_kernel(
					Scalar *d_Force,
					Scalar *d_Strain,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size,
			      		BoxDim box,
					const unsigned int *d_n_neigh,
					unsigned int *d_nlist,
					const unsigned int *d_headlist,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const float ResTable_min,
					const float ResTable_dr,
					const Scalar rlub
					){
	// Index for current thread
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check that thread is within bounds, and only do work if so
	if ( tidx < group_size ) {

		// Square of the cutoff radius
		Scalar rlubsq = rlub * rlub;

		// Particle info for this thread
		unsigned int curr_particle = d_group_members[ tidx ];

		unsigned int head_idx = d_headlist[ curr_particle ]; // Location in head array for neighbors of current particle
		unsigned int n_neigh = d_n_neigh[ curr_particle ]; // Number of neighbors of the nearest particle

		// Neighbor counters
		unsigned int neigh_idx, curr_neigh;

		//madhu
		for(int ii=0;ii<6;ii++){
			d_Force[6*curr_particle+ii]=0.0;
		}
		// Position and rate of strain of current particle
		Scalar4 posi = d_pos[ curr_particle ];
		Scalar Ei[5];
		Ei[0] = d_Strain[ 5*curr_particle + 0 ];
		Ei[1] = d_Strain[ 5*curr_particle + 1 ];
		Ei[2] = d_Strain[ 5*curr_particle + 2 ];
		Ei[3] = d_Strain[ 5*curr_particle + 3 ];
		Ei[4] = d_Strain[ 5*curr_particle + 4 ];

		// Map to 3x3
		Scalar Eblocki[3][3] = {0.0};
		Eblocki[0][0] = (1.0/3.0) * ( 2.0 * Ei[0] - Ei[4] );
		Eblocki[0][1] = 0.5 * Ei[1];
		Eblocki[0][2] = 0.5 * Ei[2];
		Eblocki[1][0] = Eblocki[0][1];
		Eblocki[1][1] = (1.0/3.0) * ( -Ei[0] + 2.0 * Ei[4] );
		Eblocki[1][2] = 0.5 * Ei[3];
		Eblocki[2][0] = Eblocki[0][2];
		Eblocki[2][1] = Eblocki[1][2];
		Eblocki[2][2] = (-1.0/3.0) * ( Ei[0] + Ei[4] );

		// Initialize force/torque
		float fi[3] = { 0.0 };
		float li[3] = { 0.0 };

		// Loop over all the neighbors for the current particle and add those
		// pair entries to the lubrication resistance tensor
		for (neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {

			// Get the current neighbor index
			curr_neigh = d_nlist[ head_idx + neigh_idx ];
			// Have to keep track of which set of particle indices we have for the sign
			// convention in Jeffrey and Onishi. For simplicity, in the JO notation of
			// pairs 11, 12, 21, 11, always set the particle with the lower global
			// index as 1 and the one with the higher as 2.
			//
			// Applies to the distance vector between the particles
			float jo_sign = ( curr_particle < curr_neigh ) ? 1.0 : -1.0;
			jo_sign *= -1.0; // Accounts for using the RSU tensor to compute RFE
			// Position of neighbor particle
			Scalar4 posj = d_pos[ curr_neigh ];
			// Rate of strain of neighbor particle
			Scalar Ej[5];
			Ej[0] = d_Strain[ 5*curr_neigh ];
			Ej[1] = d_Strain[ 5*curr_neigh + 1 ];
			Ej[2] = d_Strain[ 5*curr_neigh + 2 ];
			Ej[3] = d_Strain[ 5*curr_neigh + 3 ];
			Ej[4] = d_Strain[ 5*curr_neigh + 4 ];

			// Map to 3x3
			Scalar Eblockj[3][3] = {0.0};
			Eblockj[0][0] = (1.0/3.0) * ( 2.0 * Ej[0] - Ej[4] );
			Eblockj[0][1] = 0.5 * Ej[1];
			Eblockj[0][2] = 0.5 * Ej[2];
			Eblockj[1][0] = Eblockj[0][1];
			Eblockj[1][1] = (1.0/3.0) * ( -Ej[0] + 2.0 * Ej[4] );
			Eblockj[1][2] = 0.5 * Ej[3];
			Eblockj[2][0] = Eblockj[0][2];
			Eblockj[2][1] = Eblockj[1][2];
			Eblockj[2][2] = (-1.0/3.0) * ( Ej[0] + Ej[4] );

			// Distance vector between current particle and neighbor
			Scalar3 R = make_scalar3( posj.x - posi.x, posj.y - posi.y, posj.z - posi.z );

			// Sign convention (JO equation 1.5)
			R *= jo_sign;

			// Minimum image
			R = box.minImage(R);
			// Distance magnitude
			Scalar distSqr = dot(R,R);
			// Check that particles are within the hard-sphere cutoff (not all particles in
			// the HOOMD neighborlist are guaranteed to be within the cutoff)
			if ( distSqr < rlubsq ){

				// Distance
				Scalar dist = sqrtf( distSqr );

				Scalar XG11, XG12, YG11, YG12, YH11, YH12;
				if ( dist <= ( 2.0 + ResTable_min ) ){
					// Table is strided by 22
					XG11 = d_ResTable_vals[ 10 ];
					XG12 = d_ResTable_vals[ 11 ];
					YG11 = d_ResTable_vals[ 12 ];
					YG12 = d_ResTable_vals[ 13 ];
					YH11 = d_ResTable_vals[ 14 ];
					YH12 = d_ResTable_vals[ 15 ];
				}
				else {

					// Get the index of the nearest entry below the current distance in the distance array
					// NOTE: Distances are logarithmically spaced in the tabulation
					int ind = log10f( ( dist - 2.0 ) / ResTable_min ) / ResTable_dr;

					// Get the values from the distance array for interpolation
					Scalar dist_lower = d_ResTable_dist[ ind ];
					Scalar dist_upper = d_ResTable_dist[ ind + 1 ];

					// Read the scalar resistance coefficients from the array (lower and upper values
					// for interpolation)
					//
					// Table is strided by 32 to coalesce reads
					Scalar XG11_lower = d_ResTable_vals[ 22 * ind + 10 ];
					Scalar XG12_lower = d_ResTable_vals[ 22 * ind + 11 ];
					Scalar YG11_lower = d_ResTable_vals[ 22 * ind + 12 ];
					Scalar YG12_lower = d_ResTable_vals[ 22 * ind + 13 ];
					Scalar YH11_lower = d_ResTable_vals[ 22 * ind + 14 ];
					Scalar YH12_lower = d_ResTable_vals[ 22 * ind + 15 ];

					Scalar XG11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 10 ];
					Scalar XG12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 11 ];
					Scalar YG11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 12 ];
					Scalar YG12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 13 ];
					Scalar YH11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 14 ];
					Scalar YH12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 15 ];

					// Linear interpolation of the Table values
					Scalar fac = ( dist - dist_lower  ) / ( dist_upper - dist_lower );

					XG11 = XG11_lower + ( XG11_upper - XG11_lower ) * fac;
					XG12 = XG12_lower + ( XG12_upper - XG12_lower ) * fac;
					YG11 = YG11_lower + ( YG11_upper - YG11_lower ) * fac;
					YG12 = YG12_lower + ( YG12_upper - YG12_lower ) * fac;
					YH11 = YH11_lower + ( YH11_upper - YH11_lower ) * fac;
					YH12 = YH12_lower + ( YH12_upper - YH12_lower ) * fac;
				}
				// Account for minus signs to symmetry of RSU and RFE
				XG11 *= -1.0;
				XG12 *= -1.0;
				YG11 *= -1.0;
				YG12 *= -1.0;
                                // Geometric quantities
                                Scalar rx = R.x / dist;
                                Scalar ry = R.y / dist;
                                Scalar rz = R.z / dist;
                                Scalar3 r = make_scalar3( rx, ry, rz );
                                Scalar3 Edri = make_scalar3(    Eblocki[0][0] * r.x + Eblocki[0][1] * r.y + Eblocki[0][2] * r.z,
                                                                Eblocki[1][0] * r.x + Eblocki[1][1] * r.y + Eblocki[1][2] * r.z,
                                                                Eblocki[2][0] * r.x + Eblocki[2][1] * r.y + Eblocki[2][2] * r.z );
                                Scalar3 Edrj = make_scalar3(    Eblockj[0][0] * r.x + Eblockj[0][1] * r.y + Eblockj[0][2] * r.z,
                                                                Eblockj[1][0] * r.x + Eblockj[1][1] * r.y + Eblockj[1][2] * r.z,
                                                                Eblockj[2][0] * r.x + Eblockj[2][1] * r.y + Eblockj[2][2] * r.z );
                                Scalar rdEdri = r.x * Edri.x + r.y * Edri.y + r.z * Edri.z;
                                Scalar rdEdrj = r.x * Edrj.x + r.y * Edrj.y + r.z * Edrj.z;
				// epsr = [  0   rz -ry ]
                                //        [ -rz  0   rx ]
				// 	  [  ry -rx  0  ]
				//
				// Levi-Civita is left-handed because JO is left-handed
				Scalar3 epsrdEdri = make_scalar3( -(  r.z * Edri.y - r.y * Edri.z ),
							          -( -r.z * Edri.x + r.x * Edri.z ),
							          -(  r.y * Edri.x - r.x * Edri.y ) );
				Scalar3 epsrdEdrj = make_scalar3( -(  r.z * Edrj.y - r.y * Edrj.z ),
							          -( -r.z * Edrj.x + r.x * Edrj.z ),
							          -(  r.y * Edrj.x - r.x * Edrj.y ) );


				// Value
				//
				// Gtilde_ijk = G_jki
				//
				// G_ijk = XG * ( ri * rj - 1/3 * delta_ij ) * rk +
				//         YG * [ ( delta_jk - rj *rk ) * ri + ( delta_ik - ri * rk ) * rj ]
				fi[0] += 	jo_sign * XG11 * rdEdri * r.x +
				  	    	jo_sign * 2.0 * YG11 * ( Edri.x - rdEdri * r.x ) -
				  	    	jo_sign * XG12 * rdEdrj * r.x -
				  	    	jo_sign * 2.0 * YG12 * ( Edrj.x - rdEdrj * r.x );
				fi[1] += 	jo_sign * XG11 * rdEdri * r.y +
				  	    	jo_sign * 2.0 * YG11 * ( Edri.y - rdEdri * r.y ) -
				  	    	jo_sign * XG12 * rdEdrj * r.y -
				  	    	jo_sign * 2.0 * YG12 * ( Edrj.y - rdEdrj * r.y );
				fi[2] += 	jo_sign * XG11 * rdEdri * r.z +
						jo_sign * 2.0 * YG11 * ( Edri.z - rdEdri * r.z ) -
						jo_sign * XG12 * rdEdrj * r.z -
						jo_sign * 2.0 * YG12 * ( Edrj.z - rdEdrj * r.z );
				// Value
				//
				// Htilde_ijk = H_jki
				//
				// H_ijk = YH * ( ri * eps_jkm * rm + rj * eps_ikm * rm )
				li[0] += 	2.0 * YH11 * ( epsrdEdri.x ) +
						2.0 * YH12 * ( epsrdEdrj.x );
				li[1] += 	2.0 * YH11 * ( epsrdEdri.y ) +
						2.0 * YH12 * ( epsrdEdrj.y );
				li[2] += 	2.0 * YH11 * ( epsrdEdri.z ) +
						2.0 * YH12 * ( epsrdEdrj.z );


			} // check if neighbor is within cutoff
		} // Loop over neighbors
		// Write to output
                d_Force[ 6*curr_particle     ] += fi[0];
                d_Force[ 6*curr_particle + 1 ] += fi[1];
                d_Force[ 6*curr_particle + 2 ] += fi[2];
                d_Force[ 6*curr_particle + 3 ] += li[0];
                d_Force[ 6*curr_particle + 4 ] += li[1];
                d_Force[ 6*curr_particle + 5 ] += li[2];

	} // Check for thread in bounds
}
