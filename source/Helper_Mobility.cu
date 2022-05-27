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


#include "Helper_Mobility.cuh"

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


/*! \file Helper_Mobility.cu
    	\brief Helper functions to perform additions etc., needed in 
		the mobility calculations
*/


/*! 

	Zero out the force grid
	
	grid		(input/output)	the grid going to be zero out
   	NxNyNz		(input)		dimension of the grid

*/
__global__ void Mobility_ZeroGrid_kernel(
						CUFFTCOMPLEX *grid,
						unsigned int NxNyNz
						){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if index is within bounds
	if ( tid < NxNyNz ) {
	
		grid[tid] = make_scalar2( 0.0, 0.0 );  
	
	}
}

/*!
	Linear combination helper function
	C = a*A + b*B
	C can be A or B, so that A or B will be overwritten
	The fourth element of Scalar4 is not changed!

	d_a			(input)  input vector, A
	d_b			(input)  input vector, B
	d_c			(output) output vector, C
	coeff_a			(input)  scaling factor for A, a
	coeff_b			(input)  scaling factor for B, b
	group_size		(input)  length of vectors
	d_group_members		(input)  index into vectors
*/
__global__ void Mobility_LinearCombination_kernel(
							Scalar4 *d_a,
							Scalar4 *d_b,
							Scalar4 *d_c,
							Scalar coeff_a,
							Scalar coeff_b,
							unsigned int group_size,
							unsigned int *d_group_members
							){

	// Thread index
	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is within bounds
	if (group_idx < group_size){

		// Get current vector element, using the index (if needed)
		unsigned int idx = d_group_members[group_idx];
		Scalar4 A4 = d_a[idx];
		Scalar4 B4 = d_b[idx];

		// Make scalar3 because we only want to sum the first
		// three components
		Scalar3 A = make_scalar3(A4.x, A4.y, A4.z);
		Scalar3 B = make_scalar3(B4.x, B4.y, B4.z);

		// Addition
		A = coeff_a * A + coeff_b * B;

		// Write out
		d_c[idx] = make_scalar4(A.x, A.y, A.z, d_c[idx].w);
	}
}

/*!
        Direct addition of two scalar4 arrays, where each thread does
	work on two adjacent scalar4 elements of the array

        C = a*A + b*B
        C can be A or B, so that A or B will be overwritten
        The fourth element of Scalar4 is changed!

        d_a		(input)  input vector, A
        d_b		(input)  input vector, B
        d_c		(output) output vector, C
        coeff_a		(input)  scaling factor for A, a
        coeff_b		(input)  scaling factor for B, b
        group_size	(input)  length of vectors

*/
__global__ void Mobility_Add4_kernel(
					Scalar4 *d_a,
					Scalar4 *d_b,
					Scalar4 *d_c,
					Scalar coeff_a,
					Scalar coeff_b,
					unsigned int group_size
					){

	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (idx < group_size) {

		// Get first element
                Scalar4 A = d_a[2*idx];
                Scalar4 B = d_b[2*idx];

		// Addition for 4 components of the first element
                A.x = coeff_a * A.x + coeff_b * B.x;
                A.y = coeff_a * A.y + coeff_b * B.y;
                A.z = coeff_a * A.z + coeff_b * B.z;
                A.w = coeff_a * A.w + coeff_b * B.w;

		// Write out first element
                d_c[2*idx] = make_scalar4(A.x, A.y, A.z, A.w);

		// Get second element
                A = d_a[2*idx+1];
                B = d_b[2*idx+1];

		// Addition for 4 components of the second element
                A.x = coeff_a * A.x + coeff_b * B.x;
                A.y = coeff_a * A.y + coeff_b * B.y;
                A.z = coeff_a * A.z + coeff_b * B.z;
                A.w = coeff_a * A.w + coeff_b * B.w;

		// Write out second element
                d_c[2*idx+1] = make_scalar4(A.x, A.y, A.z, A.w);
        }
}

/*!

        Helper function to convert velocity gradient to angular velocity and rate of strain

	d_delu		(input)  velocity gradient
	d_omegaE	(output) angular velocity and rate of strain
	group_size	(input)  number of particles

*/
__global__ void Mobility_D2WE_kernel(
					Scalar4 *d_delu,
					Scalar4 *d_omegaE,
					unsigned int group_size
					){

	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
        if (idx < group_size) {

		// Get the current velocity gradient
                Scalar4 D[2];
                D[0] = make_scalar4( d_delu[2*idx].x,   d_delu[2*idx].y,   d_delu[2*idx].z,   d_delu[2*idx].w );
                D[1] = make_scalar4( d_delu[2*idx+1].x, d_delu[2*idx+1].y, d_delu[2*idx+1].z, d_delu[2*idx+1].w );

		// Convert to angular velocity and rate of strain
                Scalar W[3];
                Scalar E[5];

                W[0] = 0.5 * ( D[0].w - D[1].w );
                W[1] = 0.5 * ( D[1].z - D[0].z );
                W[2] = 0.5 * ( D[0].y - D[1].y );

                E[0] = D[0].x;
                E[1] = 0.5 * ( D[0].y + D[1].y );
                E[2] = 0.5 * ( D[0].z + D[1].z );
                E[3] = 0.5 * ( D[0].w + D[1].w );
                E[4] = D[1].x;

		// Write output
                d_omegaE[2*idx]   = make_scalar4( W[0], W[1], W[2], 2*E[0]+E[4] );
                d_omegaE[2*idx+1] = make_scalar4( 2*E[1], 2*E[2], 2*E[3], 2*E[4]+E[0] );
                
        }
}

/*!

        Helper function to convert torque and stresslet to couplet

	d_couplet	(output) particle couplet
	d_ts		(input)  torque and stresslet
	group_size	(input)  number of particles

*/
__global__ void Mobility_TS2C_kernel(
					Scalar4 *d_couplet,
					Scalar4 *d_ts,
					unsigned int group_size
					){

	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
        if (idx < group_size) {

		// Get torque and stresslet
		//
		// Torque is first 3 elements of the 2 scalar4s
		// Stresslet is last 5 elements of the 2 scalar4s
                Scalar4 TS[2];
                TS[0] = make_scalar4( d_ts[2*idx].x,   d_ts[2*idx].y,   d_ts[2*idx].z,   d_ts[2*idx].w );
                TS[1] = make_scalar4( d_ts[2*idx+1].x, d_ts[2*idx+1].y, d_ts[2*idx+1].z, d_ts[2*idx+1].w );

                Scalar Lx = TS[0].x;
                Scalar Ly = TS[0].y;
                Scalar Lz = TS[0].z;

                Scalar Sxx = TS[0].w;
                Scalar Sxy = TS[1].x;
                Scalar Sxz = TS[1].y;
                Scalar Syz = TS[1].z;
                Scalar Syy = TS[1].w;

		// Compute the couplet from torque and stresslet
                Scalar C[8];
                C[0] = Sxx;          C[1] = Sxy + 0.5*Lz; C[2] = Sxz - 0.5*Ly;
                C[5] = Sxy - 0.5*Lz; C[4] = Syy;          C[3] = Syz + 0.5*Lx;
                C[6] = Sxz + 0.5*Ly; C[7] = Syz - 0.5*Lx;

		// Write output
                d_couplet[2*idx]   = make_scalar4( C[0], C[1], C[2], C[3] );
                d_couplet[2*idx+1] = make_scalar4( C[4], C[5], C[6], C[7] );
        }
}

/*!
	Kernel function to calculate position of each grid in reciprocal space

	gridk	(output) Fourier space lattice vectors and Stokes flow scaling coefficient
	Nx	(input)  number of grid points in x-direction
	Ny	(input)  number of grid points in y-direction
	Nz	(input)  number of grid points in z-direction
	NxNyNz	(input)  total number of grid points (NxNyNz = Nx*Ny*Nz)
	box	(input)  periodic box information
	xi	(input)  Ewald parameter
	eta	(input)  NUFFT parameter

*/
__global__ void Mobility_SetGridk_kernel(
						Scalar4 *gridk,
                        		        int Nx,
                        		        int Ny,
                        		        int Nz,
                        		        unsigned int NxNyNz,
                        		        BoxDim box,
                        		        Scalar xi,
                        		        Scalar eta
						){
        
	// Thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
        if ( tid < NxNyNz ) {

		// x,y,z coordinates from modulo arithmetic
                int i = tid / (Ny*Nz);
                int j = (tid - i * Ny * Nz) / Nz;
                int k = tid % Nz;

		// Get box and tilt factor .
		//
		// NOTE: tilt factor assumes only shear in XY
                Scalar3 L = box.getL();
                Scalar xy = box.getTiltFactorXY();
                Scalar4 gridk_value;

		// Grid coordinates in x,y,z directions.
		//
		// NOTE: Assumes only shear in XY
                gridk_value.x = (i < (Nx+1) / 2) ? i : i - Nx;
                gridk_value.y = ( ((j < (Ny+1) / 2) ? j : j - Ny) - xy * gridk_value.x * L.y / L.x ) / L.y; // Fixed by Zsigi 2015
                gridk_value.x = gridk_value.x / L.x;
                gridk_value.z = ((k < (Nz+1) / 2) ? k : k - Nz) / L.z;

		// Scale by 2*pi
                gridk_value.x *= 2.0*3.1416926536;
                gridk_value.y *= 2.0*3.1416926536;
                gridk_value.z *= 2.0*3.1416926536;

		// Compute dot(k,k) and xisq once
                Scalar k2 = gridk_value.x*gridk_value.x + gridk_value.y*gridk_value.y + gridk_value.z*gridk_value.z;
                Scalar xisq = xi * xi;

                // Scaling factor used in wave space sum
                if (i == 0 && j == 0 && k == 0){
                        gridk_value.w = 0.0;
                }
                else{
                        // Have to divide by Nx*Ny*Nz to normalize the FFTs
                        gridk_value.w = 6.0*3.1415926536 * (1.0 + k2/4.0/xisq) * expf( -(1-eta) * k2/4.0/xisq ) / ( k2 ) / Scalar( Nx*Ny*Nz );
                }

		// Write output
                gridk[tid] = gridk_value;

        }
}

