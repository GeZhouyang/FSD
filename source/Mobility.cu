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


#include "Mobility.cuh"
#include "Wrappers.cuh"

#include "Helper_Mobility.cuh"
#include "Helper_Saddle.cuh"

#include "hoomd/Saru.h"
#include "hoomd/TextureTools.h"

#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/version.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

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
	This file contains the functions required to compute the action of the
	Mobility tensor on a vector using the Ewald sum.
*/


//! Texture for reading table values
scalar4_tex_t tables1_tex;

/*! 

	Spread particle force and couplet to the grid ( ALL PARTICLES SAME SIZE ) -- give one block per particle
	
	d_pos			(input)  positions of the particles
	d_net_force		(input)  particle forces
	d_couplet		(input)  particle couplets
	gridX			(output) x-component of gridded force
	gridY			(output) y-component of gridded force
	gridZ			(output) z-component of gridded force
	gridXX			(output) xx-component of gridded couplet
	gridXY			(output) xy-component of gridded couplet
	gridXZ			(output) xz-component of gridded couplet
	gridYX			(output) yx-component of gridded couplet
	gridYY			(output) yy-component of gridded couplet
	gridYZ			(output) yz-component of gridded couplet
	gridZX			(output) zx-component of gridded couplet
	gridZY			(output) zy-component of gridded couplet
	group_size		(input)  size of the group, i.e. number of particles
	Nx			(input)  number of grid nodes in x direction
	Ny			(input)  number of grid nodes in y direction
	Nz			(input)  number of grid nodes in z direction
	d_group_members		(input)  index array to global HOOMD tag on each particle
	box			(input)  box information
	P			(input)  number of grid nodes in support of spreading Gaussians
	gridh			(input)  space between grid nodes in each dimension
	xi			(input)  Ewald splitting parameter
	eta			(input)  Spectral splitting parameter
	prefac			(input)  prefactor for Gaussian envelope
	expfac			(input)  decay factor for Gaussian envelope
*/
__global__ void Mobility_WaveSpace_Spread_kernel( 	
							Scalar4 *d_pos,
						    	Scalar4 *d_net_force,
							Scalar4 *d_couplet,
						    	CUFFTCOMPLEX *gridX,
						    	CUFFTCOMPLEX *gridY,
						    	CUFFTCOMPLEX *gridZ,
							CUFFTCOMPLEX *gridXX,
							CUFFTCOMPLEX *gridXY,
							CUFFTCOMPLEX *gridXZ,
							CUFFTCOMPLEX *gridYX,
							CUFFTCOMPLEX *gridYY,
							CUFFTCOMPLEX *gridYZ,
							CUFFTCOMPLEX *gridZX,
							CUFFTCOMPLEX *gridZY,
						    	int group_size,
						    	int Nx,
						    	int Ny,
						    	int Nz,
						    	unsigned int *d_group_members,
						    	BoxDim box,
						    	const int P,
						    	Scalar3 gridh,
						    	Scalar xi,
						    	Scalar eta,
							Scalar prefac,
							Scalar expfac
							){

	__shared__ float4 shared[4]; // 16 kb max
	
	float4 *force_shared = shared;
	float4 *couplet_shared = &shared[1];	
	float4 *pos_shared = &shared[3];

	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.z*blockDim.y;
	//int block_size = blockDim.x * blockDim.y * blockDim.z;
	
	// Global particle ID
	unsigned int idx = d_group_members[group_idx];
	
	// Initialize shared memory and get particle position
	if ( thread_offset == 0 ){
	  
	  Scalar4 tpos = d_pos[idx];
	  pos_shared[0].x = tpos.x; 
	  pos_shared[0].y = tpos.y; 
	  pos_shared[0].z = tpos.z;
	  pos_shared[0].w = 2.0;
		
	  Scalar4 tforce = d_net_force[idx];
	  force_shared[0].x = tforce.x;
	  force_shared[0].y = tforce.y;
	  force_shared[0].z = tforce.z;

	  couplet_shared[0] = make_scalar4( d_couplet[2*idx].x, d_couplet[2*idx].y, d_couplet[2*idx].z, d_couplet[2*idx].w );
	  couplet_shared[1] = make_scalar4( d_couplet[2*idx+1].x, d_couplet[2*idx+1].y, d_couplet[2*idx+1].z, d_couplet[2*idx+1].w );
	}
	__syncthreads();
	
	// Box dimension
	Scalar3 L = box.getL();
	Scalar3 Ld2 = L / 2.0;

	// Retrieve position from shared memory
        Scalar3 pos = make_scalar3( pos_shared[0].x, pos_shared[0].y, pos_shared[0].z );
        Scalar3 force = make_scalar3( force_shared[0].x, force_shared[0].y, force_shared[0].z );
	Scalar4 couplet[2];
	couplet[0] = make_scalar4( couplet_shared[0].x, couplet_shared[0].y, couplet_shared[0].z, couplet_shared[0].w );
	couplet[1] = make_scalar4( couplet_shared[1].x, couplet_shared[1].y, couplet_shared[1].z, couplet_shared[1].w );

	// Fractional position within box 
	Scalar3 pos_frac = box.makeFraction(pos);
	
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;
	
	int x = int( pos_frac.x );
	int y = int( pos_frac.y );
	int z = int( pos_frac.z );
 
	// Amount of work needed for each thread to cover support
        int3 n;
        n.x = ( P + blockDim.x - 1 ) / blockDim.x; // ceiling
        n.y = ( P + blockDim.y - 1 ) / blockDim.y;
        n.z = ( P + blockDim.z - 1 ) / blockDim.z;

        int3 t;

        int Pd2 = floorf( P / 2 );
        for( int ii = 0; ii < n.x; ii++ ){

                t.x = threadIdx.x + ii*blockDim.x;

                for( int jj = 0; jj < n.y; jj++ ){

                        t.y = threadIdx.y + jj*blockDim.y;

                        for( int kk = 0; kk < n.z; kk ++ ){

                                t.z = threadIdx.z + kk*blockDim.z;

                                if( ( t.x < P ) && ( t.y < P ) && ( t.z < P ) ){

                                        // Grid point associated with current thread
                                        //int x_inp = x + t.x - Pd2;
                                        //int y_inp = y + t.y - Pd2;
                                        //int z_inp = z + t.z - Pd2;

                                        int x_inp = x + t.x - Pd2 + 1 - (P % 2) * ( pos_frac.x - Scalar( x ) < 0.5 );
                                        int y_inp = y + t.y - Pd2 + 1 - (P % 2) * ( pos_frac.y - Scalar( y ) < 0.5 );
                                        int z_inp = z + t.z - Pd2 + 1 - (P % 2) * ( pos_frac.z - Scalar( z ) < 0.5 );

                                        x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp );
                                        y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp );
                                        z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp );
	
	                                Scalar3 pos_grid;
                                        pos_grid.x = gridh.x*x_inp - Ld2.x;
                                        pos_grid.y = gridh.y*y_inp - Ld2.y;
                                        pos_grid.z = gridh.z*z_inp - Ld2.z;

                                        pos_grid.x = pos_grid.x + box.getTiltFactorXY() * pos_grid.y; // shear lattic position

                                        int grid_idx = x_inp * Ny * Nz + y_inp * Nz + z_inp;

                                        // Distance from particle to grid node
                                        Scalar3 r = pos_grid - pos;
                                        r = box.minImage(r);
                                        Scalar rsq = r.x*r.x + r.y*r.y + r.z*r.z;

                                        // Magnitude of the force contribution to the current grid node
					Scalar Cfac = prefac * expf( -expfac * rsq );

                                        // Add force to the grid
                                        atomicAdd( &(gridX[grid_idx].x), Cfac * force.x );
                                        atomicAdd( &(gridY[grid_idx].x), Cfac * force.y );
                                        atomicAdd( &(gridZ[grid_idx].x), Cfac * force.z );
	
                                        atomicAdd( &(gridXX[grid_idx].x), Cfac * couplet[0].x );
                                        atomicAdd( &(gridXY[grid_idx].x), Cfac * couplet[0].y );
                                        atomicAdd( &(gridXZ[grid_idx].x), Cfac * couplet[0].z );
                                        atomicAdd( &(gridYZ[grid_idx].x), Cfac * couplet[0].w );
                                        atomicAdd( &(gridYY[grid_idx].x), Cfac * couplet[1].x );
                                        atomicAdd( &(gridYX[grid_idx].x), Cfac * couplet[1].y );
                                        atomicAdd( &(gridZX[grid_idx].x), Cfac * couplet[1].z );
                                        atomicAdd( &(gridZY[grid_idx].x), Cfac * couplet[1].w );


                                }
                        }
                }
        }

}

/*! 

	Apply the wave space scaling to the Fourier components of the gridded force and couplet 
	to get the Fourier components of the gridded velocity and velocity gradient. (Same Size
	Particles). 

	Gridded quantities are resacled and mapped in place. 

	gridX		(input/output) 	x-component of Fourier components of gridded force/velocity
	gridY		(input/output) 	y-component of Fourier components of gridded force/velocity
	gridZ		(input/output) 	z-component of Fourier components of gridded force/velocity
	gridXX		(input/output) 	xx-component of Fourier components of gridded couplet/velocity gradient
	gridXY		(input/output) 	xy-component of Fourier components of gridded couplet/velocity gradient
	gridXZ		(input/output) 	xz-component of Fourier components of gridded couplet/velocity gradient
	gridYX		(input/output) 	yx-component of Fourier components of gridded couplet/velocity gradient
	gridYY		(input/output) 	yy-component of Fourier components of gridded couplet/velocity gradient
	gridYZ		(input/output) 	yz-component of Fourier components of gridded couplet/velocity gradient
	gridZX		(input/output) 	zx-component of Fourier components of gridded couplet/velocity gradient
	gridZY		(input/output) 	zy-component of Fourier components of gridded couplet/velocity gradient
	gridk		(input) 	wave vector and scaling factor associated with each reciprocal grid node
	NxNyNz		(input) 	total number of grid nodes

*/
__global__ void Mobility_WaveSpace_Green_kernel(
						CUFFTCOMPLEX *gridX, 
						CUFFTCOMPLEX *gridY, 
						CUFFTCOMPLEX *gridZ, 
						CUFFTCOMPLEX *gridXX,
						CUFFTCOMPLEX *gridXY,
						CUFFTCOMPLEX *gridXZ,
						CUFFTCOMPLEX *gridYX,
						CUFFTCOMPLEX *gridYY,
						CUFFTCOMPLEX *gridYZ,
						CUFFTCOMPLEX *gridZX,
						CUFFTCOMPLEX *gridZY,
						Scalar4 *gridk, 
						unsigned int NxNyNz
						){

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < NxNyNz ) {

		//
		// Values associated with current wave-vector
		//

		// Read the FFT force from global memory
		Scalar2 fX = gridX[tid];  
		Scalar2 fY = gridY[tid];
		Scalar2 fZ = gridZ[tid];
		Scalar2 cXX = gridXX[tid];
		Scalar2 cXY = gridXY[tid];
		Scalar2 cXZ = gridXZ[tid];
		Scalar2 cYX = gridYX[tid];
		Scalar2 cYY = gridYY[tid];
		Scalar2 cYZ = gridYZ[tid];
		Scalar2 cZX = gridZX[tid];
		Scalar2 cZY = gridZY[tid];
		Scalar2 cZZ = make_scalar2( -(cXX.x+cYY.x), -(cXX.y+cYY.y) );		

		// Current wave-space vector 
		Scalar4 tk = gridk[tid];
		Scalar ksq = tk.x*tk.x + tk.y*tk.y + tk.z*tk.z;
		Scalar k = sqrtf( ksq );
		
		//
		// Geometric Quantities
		//

		// k.F 
		Scalar2 kdF = (tid==0) ? make_scalar2(0.0,0.0) : make_scalar2( ( tk.x*fX.x + tk.y*fY.x + tk.z*fZ.x ) / ksq,  ( tk.x*fX.y + tk.y*fY.y + tk.z*fZ.y ) / ksq );

		// C.k
		Scalar2 Cdkx = make_scalar2( (cXX.x*tk.x + cXY.x*tk.y + cXZ.x*tk.z), (cXX.y*tk.x + cXY.y*tk.y + cXZ.y*tk.z) );
		Scalar2 Cdky = make_scalar2( (cYX.x*tk.x + cYY.x*tk.y + cYZ.x*tk.z), (cYX.y*tk.x + cYY.y*tk.y + cYZ.y*tk.z) );
		Scalar2 Cdkz = make_scalar2( (cZX.x*tk.x + cZY.x*tk.y + cZZ.x*tk.z), (cZX.y*tk.x + cZY.y*tk.y + cZZ.y*tk.z) );

		// k.C.k
		Scalar2 kdcdk = (tid==0) ? make_scalar2(0.0,0.0) : make_scalar2( (Cdkx.x*tk.x + Cdky.x*tk.y + Cdkz.x*tk.z) / ksq, (Cdkx.y*tk.x + Cdky.y*tk.y + Cdkz.y*tk.z) / ksq );

		// Fk
		Scalar2 Fkxx = make_scalar2( fX.x*tk.x, fX.y*tk.x );
		Scalar2 Fkxy = make_scalar2( fX.x*tk.y, fX.y*tk.y );
		Scalar2 Fkxz = make_scalar2( fX.x*tk.z, fX.y*tk.z );
		Scalar2 Fkyx = make_scalar2( fY.x*tk.x, fY.y*tk.x );
		Scalar2 Fkyy = make_scalar2( fY.x*tk.y, fY.y*tk.y );
		Scalar2 Fkyz = make_scalar2( fY.x*tk.z, fY.y*tk.z );
		Scalar2 Fkzx = make_scalar2( fZ.x*tk.x, fZ.y*tk.x );
		Scalar2 Fkzy = make_scalar2( fZ.x*tk.y, fZ.y*tk.y );
		
		// kk (real only)
		Scalar kkxx = tk.x*tk.x;
		Scalar kkxy = tk.x*tk.y;
		Scalar kkxz = tk.x*tk.z;
		Scalar kkyx = tk.y*tk.x;
		Scalar kkyy = tk.y*tk.y;
		Scalar kkyz = tk.y*tk.z;
		Scalar kkzx = tk.z*tk.x;
		Scalar kkzy = tk.z*tk.y;

		// (C.k)k
		Scalar2 Cdkkxx = make_scalar2( Cdkx.x*tk.x, Cdkx.y*tk.x );
		Scalar2 Cdkkxy = make_scalar2( Cdkx.x*tk.y, Cdkx.y*tk.y );
		Scalar2 Cdkkxz = make_scalar2( Cdkx.x*tk.z, Cdkx.y*tk.z );
		Scalar2 Cdkkyx = make_scalar2( Cdky.x*tk.x, Cdky.y*tk.x );
		Scalar2 Cdkkyy = make_scalar2( Cdky.x*tk.y, Cdky.y*tk.y );
		Scalar2 Cdkkyz = make_scalar2( Cdky.x*tk.z, Cdky.y*tk.z );
		Scalar2 Cdkkzx = make_scalar2( Cdkz.x*tk.x, Cdkz.y*tk.x );
		Scalar2 Cdkkzy = make_scalar2( Cdkz.x*tk.y, Cdkz.y*tk.y );

		//
		// UF Part
		//

		// Scaling factor
		Scalar B = (tid==0) ? 0.0 : tk.w * ( sinf( k ) / k ) * ( sinf( k ) / k );
	
		// Velocity calculation
		gridX[tid] = make_scalar2( ( fX.x - tk.x * kdF.x ) * B, ( fX.y - tk.x * kdF.y ) * B );
		gridY[tid] = make_scalar2( ( fY.x - tk.y * kdF.x ) * B, ( fY.y - tk.y * kdF.y ) * B );
		gridZ[tid] = make_scalar2( ( fZ.x - tk.z * kdF.x ) * B, ( fZ.y - tk.z * kdF.y ) * B );

		//
		// UC Part
		//

		// Scaling factor (imaginary here!)
		B = (tid==0) ? 0.0 : tk.w * ( sinf( k ) / k ) * ( 3.0 * ( sinf(k) - k*cosf(k) ) / (ksq*k) );
	
		// Velocity calculation
		gridX[tid].x += -( Cdkx.y - tk.x*kdcdk.y ) * B;
		gridY[tid].x += -( Cdky.y - tk.y*kdcdk.y ) * B;
		gridZ[tid].x += -( Cdkz.y - tk.z*kdcdk.y ) * B;

		gridX[tid].y += ( Cdkx.x - tk.x*kdcdk.x ) * B;
		gridY[tid].y += ( Cdky.x - tk.y*kdcdk.x ) * B;
		gridZ[tid].y += ( Cdkz.x - tk.z*kdcdk.x ) * B;


		//
		// DF Part
		//

		// Scaling factor
		B = (tid==0) ? 0.0 : tk.w * (-1.0) * ( sinf( k ) / k ) * ( 3.0 * ( sinf(k) - k*cosf(k) ) / (ksq*k) );
		
		// Velocity gradient contribution
		gridXX[tid] = make_scalar2( -( Fkxx.y - kkxx*kdF.y) * B, ( Fkxx.x - kkxx*kdF.x) * B );
		gridXY[tid] = make_scalar2( -( Fkxy.y - kkxy*kdF.y) * B, ( Fkxy.x - kkxy*kdF.x) * B );
		gridXZ[tid] = make_scalar2( -( Fkxz.y - kkxz*kdF.y) * B, ( Fkxz.x - kkxz*kdF.x) * B );
		gridYX[tid] = make_scalar2( -( Fkyx.y - kkyx*kdF.y) * B, ( Fkyx.x - kkyx*kdF.x) * B );
		gridYY[tid] = make_scalar2( -( Fkyy.y - kkyy*kdF.y) * B, ( Fkyy.x - kkyy*kdF.x) * B );
		gridYZ[tid] = make_scalar2( -( Fkyz.y - kkyz*kdF.y) * B, ( Fkyz.x - kkyz*kdF.x) * B );
		gridZX[tid] = make_scalar2( -( Fkzx.y - kkzx*kdF.y) * B, ( Fkzx.x - kkzx*kdF.x) * B );
		gridZY[tid] = make_scalar2( -( Fkzy.y - kkzy*kdF.y) * B, ( Fkzy.x - kkzy*kdF.x) * B );

		//
		// DC Part
		//

		// Scaling factor
		B = (tid==0) ? 0.0 : tk.w * (-1.0) * (-9.0) * ( ( sinf(k) - k*cosf(k) ) / (ksq*k) ) * ( ( sinf(k) - k*cosf(k) ) / (ksq*k) );

		// Velocity gradient contribution 
		gridXX[tid].x += ( Cdkkxx.x - kkxx*kdcdk.x ) * B;
		gridXY[tid].x += ( Cdkkxy.x - kkxy*kdcdk.x ) * B;
		gridXZ[tid].x += ( Cdkkxz.x - kkxz*kdcdk.x ) * B;
		gridYX[tid].x += ( Cdkkyx.x - kkyx*kdcdk.x ) * B;
		gridYY[tid].x += ( Cdkkyy.x - kkyy*kdcdk.x ) * B;
		gridYZ[tid].x += ( Cdkkyz.x - kkyz*kdcdk.x ) * B;
		gridZX[tid].x += ( Cdkkzx.x - kkzx*kdcdk.x ) * B;
		gridZY[tid].x += ( Cdkkzy.x - kkzy*kdcdk.x ) * B;

		gridXX[tid].y += ( Cdkkxx.y - kkxx*kdcdk.y ) * B;
		gridXY[tid].y += ( Cdkkxy.y - kkxy*kdcdk.y ) * B;
		gridXZ[tid].y += ( Cdkkxz.y - kkxz*kdcdk.y ) * B;
		gridYX[tid].y += ( Cdkkyx.y - kkyx*kdcdk.y ) * B;
		gridYY[tid].y += ( Cdkkyy.y - kkyy*kdcdk.y ) * B;
		gridYZ[tid].y += ( Cdkkyz.y - kkyz*kdcdk.y ) * B;
		gridZX[tid].y += ( Cdkkzx.y - kkzx*kdcdk.y ) * B;
		gridZY[tid].y += ( Cdkkzy.y - kkzy*kdcdk.y ) * B;


	}
}

/*! 

	Interpolate linear velocity from the grid to particles ( Same Size Particles, Block Per Particle (support) )

	d_pos			(input)  positions of the particles
	d_vel			(output) particle velocity
	gridX			(input)  x-component of gridded velocity
	gridY			(input)  y-component of gridded velocity
	gridZ			(input)  z-component of gridded velocity
	group_size		(input)  size of the group, i.e. number of particles
	Nx			(input)  number of grid nodes in x direction
	Ny			(input)  number of grid nodes in y direction
	Nz			(input)  number of grid nodes in z direction
	xi			(input)  Ewald splitting parameter
	eta			(input)  Spectral splitting parameter
	d_group_members		(input)  index array to global HOOMD tag on each particle
	box			(input)  array containing box dimensions
	P			(input)  number of grid nodes in support of spreading Gaussians
	gridh			(input)  space between grid nodes in each dimension
	prefac			(input)  prefactor for Gaussian envelope
	expfac			(input)  decay factor for Gaussian envelope
*/
__global__ void Mobility_WaveSpace_ContractU(	
						Scalar4 *d_pos,
					 	Scalar4 *d_vel,
					 	CUFFTCOMPLEX *gridX,
					 	CUFFTCOMPLEX *gridY,
					 	CUFFTCOMPLEX *gridZ,
					 	int group_size,
					 	int Nx,
					 	int Ny,
					 	int Nz,
					 	Scalar xi,
					 	Scalar eta,
					 	unsigned int *d_group_members,
					 	BoxDim box,
					 	const int P,
					 	Scalar3 gridh,
					 	Scalar prefac,
					 	Scalar expfac
						){
	
	extern __shared__ float3 shared3[];
	
	float3 *velocity = shared3;
	float3 *pos_shared = &shared3[blockDim.x*blockDim.y*blockDim.z];
	
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.z*blockDim.y;
	int block_size = blockDim.x * blockDim.y * blockDim.z;
	
	// Global particle ID
	unsigned int idx = d_group_members[group_idx];
	
	// Initialize shared memory and get particle position
	velocity[thread_offset] = make_scalar3(0.0,0.0,0.0);
	if ( thread_offset == 0 ){
		Scalar4 tpos = d_pos[idx];
		pos_shared[0] = make_scalar3( tpos.x, tpos.y, tpos.z ); 
	}
	__syncthreads();

	// Box dimension
	Scalar3 L = box.getL();
	Scalar3 Ld2 = L / 2.0;
	
	// Retrieve position from shared memory
	Scalar3 pos = pos_shared[0];
	
	// Fractional position within box 
	Scalar3 pos_frac = box.makeFraction(pos);
	
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;
	
	int x = int( pos_frac.x );
	int y = int( pos_frac.y );
	int z = int( pos_frac.z );
	//x = ( pos_frac.x - float(x) < 0.5 ) ? x : x + 1;
	//y = ( pos_frac.y - float(y) < 0.5 ) ? y : y + 1;
	//z = ( pos_frac.z - float(z) < 0.5 ) ? z : z + 1;
	
	int3 n;
	n.x = ( P + blockDim.x - 1 ) / blockDim.x; // ceiling
	n.y = ( P + blockDim.y - 1 ) / blockDim.y;
	n.z = ( P + blockDim.z - 1 ) / blockDim.z;
	
	int3 t;
	
	int Pd2 = P / 2; // integer division does floor
	for( int ii = 0; ii < n.x; ii++ ){
	
	      	t.x = threadIdx.x + ii*blockDim.x;
	
	      	for( int jj = 0; jj < n.y; jj++ ){
	
	      		t.y = threadIdx.y + jj*blockDim.y;
	
	      		for( int kk = 0; kk < n.z; kk++ ){
	
	      			t.z = threadIdx.z + kk*blockDim.z;

				if( ( t.x < P ) && ( t.y < P ) && ( t.z < P ) ){
	
					// Grid point associated with current thread
					//int x_inp = x + t.x - Pd2;
					//int y_inp = y + t.y - Pd2;
					//int z_inp = z + t.z - Pd2;
					int x_inp = x + t.x - Pd2 + 1 - (P % 2) * ( pos_frac.x - Scalar( x ) < 0.5 );
					int y_inp = y + t.y - Pd2 + 1 - (P % 2) * ( pos_frac.y - Scalar( y ) < 0.5 );
					int z_inp = z + t.z - Pd2 + 1 - (P % 2) * ( pos_frac.z - Scalar( z ) < 0.5 );
					
					x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp );
					y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp );
					z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp );
	
                                        Scalar3 pos_grid;
                                        pos_grid.x = gridh.x*x_inp - Ld2.x;
                                        pos_grid.y = gridh.y*y_inp - Ld2.y;
                                        pos_grid.z = gridh.z*z_inp - Ld2.z;

                                        pos_grid.x = pos_grid.x + box.getTiltFactorXY() * pos_grid.y; // shear lattic position
					
					int grid_idx = x_inp * Ny * Nz + y_inp * Nz + z_inp;
					
					// Distance from particle to grid node
					Scalar3 r = pos_grid - pos;
					r = box.minImage(r);
					Scalar rsq = r.x*r.x + r.y*r.y + r.z*r.z;
	
	      				// Contraction Factor
					Scalar Cfac = prefac * expf( -expfac * rsq );

					// THIS IS THE SLOW STEP:
					velocity[thread_offset] += Cfac * make_scalar3( gridX[grid_idx].x, gridY[grid_idx].x, gridZ[grid_idx].x );

				}
	      		}
	      	}
	}
	
	int offs = block_size;
	int offs_prev; 
	while (offs > 1)
	{
	      offs_prev = offs; 
	      offs = ( offs + 1 ) / 2;
		__syncthreads();
	    	if (thread_offset + offs < offs_prev)
	        {
	        	velocity[thread_offset] += velocity[thread_offset + offs];
	        }
	    	
	}
	
	// Combine components of velocity
	if (thread_offset == 0){
		d_vel[idx] = make_scalar4(velocity[0].x, velocity[0].y, velocity[0].z, d_vel[idx].w);
	}
	
}


/*! 

	Interpolate velocity gradient from the grid to the particles ( Same Size Particles, Block Per Particle (support) )

	d_pos			(input)  positions of the particles
	d_delu			(output) particle velocity gradient
	gridXX			(input)  xx-component of gridded velocity gradient
	gridXY			(input)  xy-component of gridded velocity gradient
	gridXZ			(input)  xz-component of gridded velocity gradient
	gridYX			(input)  yx-component of gridded velocity gradient
	gridYY			(input)  yy-component of gridded velocity gradient
	gridYZ			(input)  yz-component of gridded velocity gradient
	gridZX			(input)  zx-component of gridded velocity gradient
	gridZY			(input)  zy-component of gridded velocity gradient
	group_size		(input)  size of the group, i.e. number of particles
	Nx			(input)  number of grid nodes in x direction
	Ny			(input)  number of grid nodes in y direction
	Nz			(input)  number of grid nodes in z direction
	xi			(input)  Ewald splitting parameter
	eta			(input)  Spectral splitting parameter
	d_group_members		(input)  index array to global HOOMD tag on each particle
	box			(input)  box information
	P			(input)  number of grid nodes in support of spreading Gaussians
	gridh			(input)  space between grid nodes in each dimension
	prefac			(input)  prefactor for Gaussian envelope
	expfac			(input)  decay factor for Gaussian envelope

*/
__global__ void Mobility_WaveSpace_ContractD(	
						Scalar4 *d_pos,
					 	Scalar4 *d_delu,
						CUFFTCOMPLEX *gridXX,
						CUFFTCOMPLEX *gridXY,
						CUFFTCOMPLEX *gridXZ,
						CUFFTCOMPLEX *gridYX,
						CUFFTCOMPLEX *gridYY,
						CUFFTCOMPLEX *gridYZ,
						CUFFTCOMPLEX *gridZX,
						CUFFTCOMPLEX *gridZY,
					 	int group_size,
					 	int Nx,
					 	int Ny,
					 	int Nz,
					 	Scalar xi,
					 	Scalar eta,
					 	unsigned int *d_group_members,
					 	BoxDim box,
					 	const int P,
					 	Scalar3 gridh,
					 	Scalar prefac,
					 	Scalar expfac
						){

	extern __shared__ float4 shared4[];
	
	float4 *velocity = shared4;
	float4 *pos_shared = &shared4[2*blockDim.x*blockDim.y*blockDim.z];
	
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.z*blockDim.y;
	int block_size = blockDim.x * blockDim.y * blockDim.z;
	
	// Global particle ID
	unsigned int idx = d_group_members[group_idx];
	
	// Initialize shared memory and get particle position
	velocity[thread_offset]   = make_scalar4(0.0,0.0,0.0,0.0);
	velocity[block_size+thread_offset] = make_scalar4(0.0,0.0,0.0,0.0);
	if ( thread_offset == 0 ){
		Scalar4 tpos = d_pos[idx];
		pos_shared[0] = make_scalar4( tpos.x, tpos.y, tpos.z, 1.0 ); 
	}
	__syncthreads();
	
	// Box dimension
	Scalar3 L = box.getL();
	Scalar3 Ld2 = L / 2.0;
	
	// Retrieve position from shared memory
	Scalar3 pos = make_scalar3( pos_shared[0].x, pos_shared[0].y, pos_shared[0].z );
	
	// Fractional position within box 
	Scalar3 pos_frac = box.makeFraction(pos);
	
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;
	
	int x = int( pos_frac.x );
	int y = int( pos_frac.y );
	int z = int( pos_frac.z );
	//x = ( pos_frac.x - float(x) < 0.5 ) ? x : x + 1;
	//y = ( pos_frac.y - float(y) < 0.5 ) ? y : y + 1;
	//z = ( pos_frac.z - float(z) < 0.5 ) ? z : z + 1;
	
	int3 n;
	n.x = ( P + blockDim.x - 1 ) / blockDim.x; // ceiling
	n.y = ( P + blockDim.y - 1 ) / blockDim.y;
	n.z = ( P + blockDim.z - 1 ) / blockDim.z;
	
	int3 t;
	
	int Pd2 = P / 2; // integer division does floor
	for( int ii = 0; ii < n.x; ++ii ){

	      	t.x = threadIdx.x + ii*blockDim.x;

		for( int jj = 0; jj < n.y; ++jj ){
	      	
			t.y = threadIdx.y + jj*blockDim.y;

			for( int kk = 0; kk < n.z; ++kk ){	
	      	
				t.z = threadIdx.z + kk*blockDim.z;
	
				if( ( t.x < P ) && ( t.y < P ) && ( t.z < P ) ){

					// Grid point associated with current thread
					//int x_inp = x + t.x - Pd2;
					//int y_inp = y + t.y - Pd2;
					//int z_inp = z + t.z - Pd2;
					int x_inp = x + t.x - Pd2 + 1 - (P % 2) * ( pos_frac.x - Scalar( x ) < 0.5 );
					int y_inp = y + t.y - Pd2 + 1 - (P % 2) * ( pos_frac.y - Scalar( y ) < 0.5 );
					int z_inp = z + t.z - Pd2 + 1 - (P % 2) * ( pos_frac.z - Scalar( z ) < 0.5 );
					
					x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp );
					y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp );
					z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp );
					
                                        Scalar3 pos_grid;
                                        pos_grid.x = gridh.x*x_inp - Ld2.x;
                                        pos_grid.y = gridh.y*y_inp - Ld2.y;
                                        pos_grid.z = gridh.z*z_inp - Ld2.z;

                                        pos_grid.x = pos_grid.x + box.getTiltFactorXY() * pos_grid.y; // shear lattic position
					
					int grid_idx = x_inp * Ny * Nz + y_inp * Nz + z_inp;
					
					// Distance from particle to grid node
					Scalar3 r = pos_grid - pos;
					r = box.minImage(r);
					Scalar rsq = r.x*r.x + r.y*r.y + r.z*r.z;
					Scalar R = sqrtf( rsq );
					
					// Contraction factor
					Scalar Cfac = prefac * expf( - expfac * rsq );
	
					// THIS IS THE SLOW STEP:
					velocity[thread_offset].x += Cfac * gridXX[grid_idx].x;
					velocity[thread_offset].y += Cfac * gridXY[grid_idx].x;
					velocity[thread_offset].z += Cfac * gridXZ[grid_idx].x;
					velocity[thread_offset].w += Cfac * gridYZ[grid_idx].x;	
					velocity[block_size + thread_offset].x += Cfac * gridYY[grid_idx].x;
					velocity[block_size + thread_offset].y += Cfac * gridYX[grid_idx].x;
					velocity[block_size + thread_offset].z += Cfac * gridZX[grid_idx].x;
					velocity[block_size + thread_offset].w += Cfac * gridZY[grid_idx].x;
 
				}
			}
		}
	}
				
	int offs = block_size;
	int offs_prev; 
	while (offs > 1)
	{
	      offs_prev = offs; 
	      offs = ( offs + 1 ) / 2;
		__syncthreads();
	    	if (thread_offset + offs < offs_prev)
	        {
	        	velocity[thread_offset].x += velocity[thread_offset + offs].x;
	        	velocity[thread_offset].y += velocity[thread_offset + offs].y;
	        	velocity[thread_offset].z += velocity[thread_offset + offs].z;
	        	velocity[thread_offset].w += velocity[thread_offset + offs].w;
	        	
			velocity[block_size + thread_offset].x += velocity[block_size + thread_offset + offs].x;
			velocity[block_size + thread_offset].y += velocity[block_size + thread_offset + offs].y;
			velocity[block_size + thread_offset].z += velocity[block_size + thread_offset + offs].z;
			velocity[block_size + thread_offset].w += velocity[block_size + thread_offset + offs].w;

	        }
	    	
	}
	
	// Combine components of velocity gradient (because of our definition, have to write out the transpose)
	if (thread_offset == 0){
		d_delu[2*idx]   = make_scalar4(velocity[0].x, velocity[block_size].y, velocity[block_size].z, velocity[block_size].w); //zhoge: transpose
		d_delu[2*idx+1] = make_scalar4(velocity[block_size].x, velocity[0].y, velocity[0].z, velocity[0].w);		       //zhoge: transpose
		//d_delu[2*idx]   = make_scalar4(velocity[0].x, velocity[0].y, velocity[0].z, velocity[0].w);
		//d_delu[2*idx+1] = make_scalar4(velocity[block_size].x, velocity[block_size].y, velocity[block_size].z, velocity[block_size].w);
	}

	
}

/*! 

	Wrap all the functions to compute wave space part of Mobility ( Same Size Particles ) and 
	drive the kernel functions. 

	d_pos			(input)  positions of the particles
	d_vel			(output) particle velocity
	d_delu			(output) particle velocity gradient
	d_net_force		(input)  forces on the particles
	d_couplet		(input)  particle couplets
	group_size		(input)  size of the group, i.e. number of particles
	d_group_members		(input)  index array to global HOOMD tag on each particle
	box			(input)  array containing box dimensions
	xi			(input)  Ewald splitting parameter
	eta			(input)  Spectral splitting parameter
	d_gridk			(input)  wave vector and scaling factor associated with each reciprocal grid node
	d_gridX			(input)  x-component of force projected onto grid
	d_gridY			(input)  y-component of force projected onto grid
	d_gridZ			(input)  z-component of force projected onto grid
	d_gridXX		(input)  xx-component of couplet projected onto grid
	d_gridXY		(input)  xy-component of couplet projected onto grid
	d_gridXZ		(input)  xz-component of couplet projected onto grid
	d_gridYX		(input)  yx-component of couplet projected onto grid
	d_gridYY		(input)  yy-component of couplet projected onto grid
	d_gridYZ		(input)  yz-component of couplet projected onto grid
	d_gridZX		(input)  zx-component of couplet projected onto grid
	d_gridZY		(input)  zy-component of couplet projected onto grid
	plan			(input)  Plan for cufft
	Nx			(input)  Number of grid/FFT nodes in x-direction
	Ny			(input)  Number of grid/FFT nodes in y-direction
	Nz			(input)  Number of grid/FFT nodes in z-direction
	NxNyNz			(input)  total number of grid/FFT nodes
	grid			(input)  block grid to use when launching kernels
	threads			(input)  number of threads per block for kernels
	gridBlockSize		(input)  number of threads per block
	gridNBlock		(input)  number of blocks
	P			(input)  number of nodes in support of each gaussian for k-space sum
	gridh			(input)  distance between grid nodes

*/
void gpu_stokes_Mwave_wrap(	
				Scalar4 *d_pos,
                        	Scalar4 *d_vel,
				Scalar4 *d_delu,
                        	Scalar4 *d_net_force,
				Scalar4 *d_couplet,
				unsigned int *d_group_members,
				unsigned int group_size,
                        	const BoxDim& box,
				Scalar xi,
				Scalar eta,
				Scalar4 *d_gridk,
				CUFFTCOMPLEX *d_gridX,
				CUFFTCOMPLEX *d_gridY,
				CUFFTCOMPLEX *d_gridZ,
				CUFFTCOMPLEX *d_gridXX,
				CUFFTCOMPLEX *d_gridXY,
				CUFFTCOMPLEX *d_gridXZ,
				CUFFTCOMPLEX *d_gridYX,
				CUFFTCOMPLEX *d_gridYY,
				CUFFTCOMPLEX *d_gridYZ,
				CUFFTCOMPLEX *d_gridZX,
				CUFFTCOMPLEX *d_gridZY,
				cufftHandle plan,
				const int Nx,
				const int Ny,
				const int Nz,
				unsigned int NxNyNz,
				dim3 grid,
				dim3 threads,
				int gridBlockSize,
				int gridNBlock,
				const int P,
				Scalar3 gridh
				){
   
	// Spreading and contraction stuff
	dim3 Cgrid( group_size, 1, 1);
	int B = ( P < 8 ) ? P : 8;
	dim3 Cthreads(B, B, B);
	
	Scalar quadW = gridh.x * gridh.y * gridh.z;
	Scalar xisq = xi * xi;
	Scalar prefac = ( 2.0 * xisq / 3.1415926536 / eta ) * sqrtf( 2.0 * xisq / 3.1415926536 / eta );
	Scalar expfac = 2.0 * xisq / eta;
	
	// Reset the grid ( remove any previously distributed forces )
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridX, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridY, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridZ, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridXX, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridXY, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridXZ, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridYX, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridYY, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridYZ, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridZX, NxNyNz );
	Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( d_gridZY, NxNyNz );
	
	// Spread forces onto grid (zhoge: check this out later, force spreading)
	Mobility_WaveSpace_Spread_kernel<<<Cgrid, Cthreads>>>(
							      d_pos,
							      d_net_force,
							      d_couplet,
							      d_gridX, d_gridY, d_gridZ,
							      d_gridXX, d_gridXY, d_gridXZ,
							      d_gridYX, d_gridYY, d_gridYZ,
							      d_gridZX, d_gridZY,
							      group_size,
							      Nx, Ny, Nz,
							      d_group_members,
							      box, P, gridh, xi, eta, prefac, expfac );
	
	// Perform FFT on gridded forces
	cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridXX, d_gridXX, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridXY, d_gridXY, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridXZ, d_gridXZ, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridYX, d_gridYX, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridYY, d_gridYY, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridYZ, d_gridYZ, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridZX, d_gridZX, CUFFT_FORWARD);
	cufftExecC2C(plan, d_gridZY, d_gridZY, CUFFT_FORWARD);
	
	// Apply wave space scaling to FFT'd forces
	Mobility_WaveSpace_Green_kernel<<<gridNBlock,gridBlockSize>>>( d_gridX, d_gridY, d_gridZ, d_gridXX, d_gridXY, d_gridXZ, d_gridYX, d_gridYY, d_gridYZ, d_gridZX, d_gridZY, d_gridk, NxNyNz );
	
	// Return rescaled forces to real space
	cufftExecC2C(plan, d_gridX, d_gridX, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridY, d_gridY, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridZ, d_gridZ, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridXX, d_gridXX, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridXY, d_gridXY, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridXZ, d_gridXZ, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridYX, d_gridYX, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridYY, d_gridYY, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridYZ, d_gridYZ, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridZX, d_gridZX, CUFFT_INVERSE);
	cufftExecC2C(plan, d_gridZY, d_gridZY, CUFFT_INVERSE);
	
	// Evaluate contribution of grid velocities at particle centers
	Mobility_WaveSpace_ContractU<<<Cgrid, Cthreads, (B*B*B+1)*sizeof(float3)>>>( d_pos, d_vel, d_gridX, d_gridY, d_gridZ, group_size, Nx, Ny, Nz, xi, eta, d_group_members, box, P, gridh, quadW*prefac, expfac );
	Mobility_WaveSpace_ContractD<<<Cgrid, Cthreads, (2*B*B*B+1)*sizeof(float4)>>>( d_pos, d_delu, d_gridXX, d_gridXY, d_gridXZ, d_gridYX, d_gridYY, d_gridYZ, d_gridZX, d_gridZY, group_size, Nx, Ny, Nz, xi, eta, d_group_members, box, P, gridh, quadW*prefac, expfac );
 
}

/*! 
	
	Kernel to compute the product of the real space mobility tensor with a vector.

	d_pos			(input)  positions of the particles
	d_vel			(output) particle velocity
	d_delu			(output) particle velocity gradient
	d_net_force		(input)  forces on the particles
	d_couplet		(input)  particle couplets
	group_size		(input)  number of particles
	xi			(input)  Ewald splitting parameter
	d_ewaldC1		(input)  Pre-tabulated form of the real-space Ewald sum
	ewald_cut		(input)  Cut-off distance for real-space interaction
	ewald_n			(input)  Number of entries in the Ewald table
	ewald_dr		(input)  Distance spacing used in computing the pre-tabulated tables
	d_group_members		(input)  index array to global HOOMD tag on each particle
	box			(input)  periodic box information
	d_nneigh		(input)  number of neighbors for each particle
	d_nlist			(input)  neighbor list for the real space interactions
	d_headlist		(input)  head list into the neighbor list for the real space interactions

*/
__global__ void Mobility_RealSpace_kernel(
						Scalar4 *d_pos,
			      			Scalar4 *d_vel,
						Scalar4 *d_delu,
			      			Scalar4 *d_net_force,
						Scalar4 *d_couplet,
			      			int group_size,
			      			Scalar xi,
			      			Scalar4 *d_ewaldC1, 
			      			Scalar2 self,
			      			Scalar ewald_cut,
			      			int ewald_n,
			      			Scalar ewald_dr,
			      			unsigned int *d_group_members,
			      			BoxDim box,
			      			const unsigned int *d_nneigh,
                              			const unsigned int *d_nlist,
                              			const unsigned int *d_headlist
						){
 
	// Index for current thread 
	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Initialize contribution to velocity
	Scalar4 u = make_scalar4( 0.0, 0.0, 0.0, 0.0 );
	Scalar4 D[2]; 
	D[0] = make_scalar4( 0.0, 0.0, 0.0, 0.0 ); 
	D[1] = make_scalar4( 0.0, 0.0, 0.0, 0.0 );
	
	if (group_idx < group_size) {
	  
		// Particle for this thread
		unsigned int idx = d_group_members[group_idx];
		
		// Number of neighbors for current particle
		unsigned int nneigh = d_nneigh[idx]; 
		unsigned int head_idx = d_headlist[idx];
		
		// Particle position and table ID
		Scalar4 posi = d_pos[idx];
		
		// Self contribution
		Scalar4 F = d_net_force[idx];
		Scalar4 C[2], cc;
                cc = d_couplet[2*idx];   C[0] = make_scalar4( cc.x, cc.y, cc.z, cc.w );
                cc = d_couplet[2*idx+1]; C[1] = make_scalar4( cc.x, cc.y, cc.z, cc.w );

		u = make_scalar4( self.x * F.x, self.x * F.y, self.x * F.z, 0.0 );
		D[0] = make_scalar4( self.y*(C[0].x - 4.*C[0].x), self.y*(C[0].y - 4.*C[1].y), self.y*(C[0].z - 4.*C[1].z), self.y*(C[0].w - 4.*C[1].w) );
		D[1] = make_scalar4( self.y*(C[1].x - 4.*C[1].x), self.y*(C[1].y - 4.*C[0].y), self.y*(C[1].z - 4.*C[0].z), self.y*(C[1].w - 4.*C[0].w) );

		// Minimum and maximum distance for pair calculation
		Scalar mindistSq = ewald_dr * ewald_dr;
		Scalar maxdistSq = ewald_cut * ewald_cut;
		
		for (int neigh_idx = 0; neigh_idx < nneigh; neigh_idx++) {
		
			// Statement might be necessary for bug on older architectures?
			unsigned int cur_j = d_nlist[ head_idx + neigh_idx ];
		
			// Position and size of neighbor particle
			Scalar4 posj = d_pos[cur_j];
		
			// Distance vector between current particle and neighbor
			Scalar3 r = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
			r = box.minImage(r);
			Scalar distSqr = dot(r,r);
		
			// Add neighbor contribution if it is within the real space cutoff radius
			if ( ( distSqr < maxdistSq ) && ( distSqr >= mindistSq ) ) {
		
				// Need distance 
				Scalar dist = sqrtf( distSqr );
				
				// Force on neighbor particle
				Scalar4 Fj = d_net_force[cur_j];
				Scalar4 Cj[2];
                                cc = d_couplet[2*cur_j];   Cj[0] = make_scalar4( cc.x, cc.y, cc.z, cc.w );
                                cc = d_couplet[2*cur_j+1]; Cj[1] = make_scalar4( cc.x, cc.y, cc.z, cc.w );

                                // Fetch relevant elements from textured table for real space interaction
                                int r_ind = __scalar2int_rd( ewald_n * ( dist - ewald_dr ) / ( ewald_cut - ewald_dr ) );
                                int offset1  = 2 * r_ind;
                                int offset2  = 2 * r_ind + 1;

                                Scalar4 tewaldC1m = texFetchScalar4(d_ewaldC1, tables1_tex, offset1);   // UF and UC
                                Scalar4 tewaldC1p = texFetchScalar4(d_ewaldC1, tables1_tex, offset1+2);
                                Scalar4 tewaldC2m = texFetchScalar4(d_ewaldC1, tables1_tex, offset2);   // DC
                                Scalar4 tewaldC2p = texFetchScalar4(d_ewaldC1, tables1_tex, offset2+2);
		
				// Linear interpolation of table
				Scalar fac = dist / ewald_dr - r_ind - Scalar(1.0);
	
                                Scalar f1 = tewaldC1m.x + ( tewaldC1p.x - tewaldC1m.x ) * fac;
                                Scalar f2 = tewaldC1m.y + ( tewaldC1p.y - tewaldC1m.y ) * fac;

                                Scalar g1 = tewaldC1m.z + ( tewaldC1p.z - tewaldC1m.z ) * fac;
                                Scalar g2 = tewaldC1m.w + ( tewaldC1p.w - tewaldC1m.w ) * fac;

                                Scalar h1 = tewaldC2m.x + ( tewaldC2p.x - tewaldC2m.x ) * fac;
                                Scalar h2 = tewaldC2m.y + ( tewaldC2p.y - tewaldC2m.y ) * fac;
                                Scalar h3 = tewaldC2m.z + ( tewaldC2p.z - tewaldC2m.z ) * fac;

				// Geometric quantities
				Scalar3 R = r / dist;

                                Scalar rdotf = R.x*Fj.x + R.y*Fj.y + R.z*Fj.z;

                                Scalar3 Cdotr;
                                Cdotr.x = ( Cj[0].x*R.x + Cj[0].y*R.y +           Cj[0].z*R.z );
                                Cdotr.y = ( Cj[1].y*R.x + Cj[1].x*R.y +           Cj[0].w*R.z );
                                Cdotr.z = ( Cj[1].z*R.x + Cj[1].w*R.y - (Cj[0].x+Cj[1].x)*R.z );

                                Scalar3 rdotC;
                                rdotC.x = ( Cj[0].x*R.x + Cj[1].y*R.y +           Cj[1].z*R.z );
                                rdotC.y = ( Cj[0].y*R.x + Cj[1].x*R.y +           Cj[1].w*R.z );
                                rdotC.z = ( Cj[0].z*R.x + Cj[0].w*R.y - (Cj[0].x+Cj[1].x)*R.z );

                                Scalar rdotCdotr = ( R.x*Cdotr.x + R.y*Cdotr.y + R.z*Cdotr.z );


				// Velocity
	
				//
				u.x += f1 * Fj.x + ( f2 - f1 ) * rdotf * R.x;
				u.y += f1 * Fj.y + ( f2 - f1 ) * rdotf * R.y;
				u.z += f1 * Fj.z + ( f2 - f1 ) * rdotf * R.z;
		
				//
				u.x += g1 * ( Cdotr.x - rdotCdotr * R.x ) + g2 * ( rdotC.x - 4.*rdotCdotr * R.x );
				u.y += g1 * ( Cdotr.y - rdotCdotr * R.y ) + g2 * ( rdotC.y - 4.*rdotCdotr * R.y );
				u.z += g1 * ( Cdotr.z - rdotCdotr * R.z ) + g2 * ( rdotC.z - 4.*rdotCdotr * R.z );

				// Velocity gradient

				//
				D[0].x += (-1.)*g1 * ( R.x*Fj.x - rdotf * R.x*R.x ) + (-1.)*g2 * ( rdotf + Fj.x*R.x - 4.*rdotf * R.x*R.x );
				D[0].y += (-1.)*g1 * ( R.x*Fj.y - rdotf * R.x*R.y ) + (-1.)*g2 * (         Fj.x*R.y - 4.*rdotf * R.x*R.y );
				D[0].z += (-1.)*g1 * ( R.x*Fj.z - rdotf * R.x*R.z ) + (-1.)*g2 * (         Fj.x*R.z - 4.*rdotf * R.x*R.z );

				D[0].w += (-1.)*g1 * ( R.y*Fj.z - rdotf * R.y*R.z ) + (-1.)*g2 * (         Fj.y*R.z - 4.*rdotf * R.y*R.z );
				D[1].x += (-1.)*g1 * ( R.y*Fj.y - rdotf * R.y*R.y ) + (-1.)*g2 * ( rdotf + Fj.y*R.y - 4.*rdotf * R.y*R.y );
				D[1].y += (-1.)*g1 * ( R.y*Fj.x - rdotf * R.y*R.x ) + (-1.)*g2 * (         Fj.y*R.x - 4.*rdotf * R.y*R.x );

				D[1].z += (-1.)*g1 * ( R.z*Fj.x - rdotf * R.z*R.x ) + (-1.)*g2 * (         Fj.z*R.x - 4.*rdotf * R.z*R.x );
				D[1].w += (-1.)*g1 * ( R.z*Fj.y - rdotf * R.z*R.y ) + (-1.)*g2 * (         Fj.z*R.y - 4.*rdotf * R.z*R.y );

				//
				D[0].x += h1 * ( Cj[0].x - 4.*Cj[0].x ) + h2 * ( R.x*Cdotr.x - rdotCdotr * R.x*R.x ) + h3 * ( rdotCdotr + Cdotr.x*R.x + R.x*rdotC.x + rdotC.x*R.x - 6.*rdotCdotr*R.x*R.x - Cj[0].x );
				D[0].y += h1 * ( Cj[0].y - 4.*Cj[1].y ) + h2 * ( R.x*Cdotr.y - rdotCdotr * R.x*R.y ) + h3 * (             Cdotr.x*R.y + R.x*rdotC.y + rdotC.x*R.y - 6.*rdotCdotr*R.x*R.y - Cj[1].y );
				D[0].z += h1 * ( Cj[0].z - 4.*Cj[1].z ) + h2 * ( R.x*Cdotr.z - rdotCdotr * R.x*R.z ) + h3 * (             Cdotr.x*R.z + R.x*rdotC.z + rdotC.x*R.z - 6.*rdotCdotr*R.x*R.z - Cj[1].z );
				
				D[0].w += h1 * ( Cj[0].w - 4.*Cj[1].w ) + h2 * ( R.y*Cdotr.z - rdotCdotr * R.y*R.z ) + h3 * (             Cdotr.y*R.z + R.y*rdotC.z + rdotC.y*R.z - 6.*rdotCdotr*R.y*R.z - Cj[1].w );
				D[1].x += h1 * ( Cj[1].x - 4.*Cj[1].x ) + h2 * ( R.y*Cdotr.y - rdotCdotr * R.y*R.y ) + h3 * ( rdotCdotr + Cdotr.y*R.y + R.y*rdotC.y + rdotC.y*R.y - 6.*rdotCdotr*R.y*R.y - Cj[1].x );
				D[1].y += h1 * ( Cj[1].y - 4.*Cj[0].y ) + h2 * ( R.y*Cdotr.x - rdotCdotr * R.y*R.x ) + h3 * (             Cdotr.y*R.x + R.y*rdotC.x + rdotC.y*R.x - 6.*rdotCdotr*R.y*R.x - Cj[0].y );

				D[1].z += h1 * ( Cj[1].z - 4.*Cj[0].z ) + h2 * ( R.z*Cdotr.x - rdotCdotr * R.z*R.x ) + h3 * (             Cdotr.z*R.x + R.z*rdotC.x + rdotC.z*R.x - 6.*rdotCdotr*R.z*R.x - Cj[0].z );
				D[1].w += h1 * ( Cj[1].w - 4.*Cj[0].w ) + h2 * ( R.z*Cdotr.y - rdotCdotr * R.z*R.y ) + h3 * (             Cdotr.z*R.y + R.z*rdotC.y + rdotC.z*R.y - 6.*rdotCdotr*R.z*R.y - Cj[0].w );

			}
		
		}

		// Write to output
		d_vel[idx] = u;
		//d_delu[2*idx]   = make_scalar4( D[0].x, D[0].y, D[0].z, D[0].w );
		//d_delu[2*idx+1] = make_scalar4( D[1].x, D[1].y, D[1].z, D[1].w );
		d_delu[2*idx]   = make_scalar4( D[0].x, D[1].y, D[1].z, D[1].w );
		d_delu[2*idx+1] = make_scalar4( D[1].x, D[0].y, D[0].z, D[0].w );

	}   

}

/*

	Compute Mreal multiplication with T/S input and W/E output (instead of C and D). Required for the
	Lanczos iteration on the the real space contribution to the far-field Brownian slip velocity.

		U = Mreal * F

	d_pos			(input)  particle positions
	d_vel			(output) linear velocity of particles
	d_AngvelStrain		(output) angular velocity and rate of strain of particles
	d_net_force		(input)  linear force on particles
	d_TorqueStress		(input)  torque and stress on particles
	d_couplet		(input)  storage space for couplet
	d_delu			(input)  storage space for velocity gradient
	d_group_members		(input)  particle index within group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	xi			(input)  Ewald splitting parameter
	ewald_cut		(input)  real space Ewald cutoff
	ewald_dr		(input)  real space Ewald tabulation discretization
	ewald_n			(input)  number of entries in real space Ewald tabulation
	d_ewaldC1		(input)  real space Ewald tabulation
	self			(input)  Ewald self piece
	d_nneigh		(input)  number of neighbors for real space Ewald sum
	d_nlist			(input)  neighbor list for real space Ewald sum
	d_headlist		(input)  head list into neighbor list for real space Ewald sum
	grid			(input)  grid for CUDA kernel launches
	threads			(input)  threads for CUDA kernel launches


*/
void Mobility_RealSpaceFTS(	
				Scalar4 *d_pos,
                        	Scalar4 *d_vel,
				Scalar4 *d_AngvelStrain,
                        	Scalar4 *d_net_force,
				Scalar4 *d_TorqueStress,
				Scalar4 *d_couplet,
				Scalar4 *d_delu,
				unsigned int *d_group_members,
				unsigned int group_size,
                        	const BoxDim& box,
				Scalar xi,
				Scalar ewald_cut,
				Scalar ewald_dr,
				int ewald_n,
				Scalar4 *d_ewaldC1, 
				Scalar2 self,
				const unsigned int *d_nneigh,
                        	const unsigned int *d_nlist,
                        	const unsigned int *d_headlist,
				dim3 grid,
				dim3 threads
				){

	// NEED THIS FOR CUSP RESULT TO BE OK
	cudaMemcpy( d_vel, d_net_force, group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

	// Map Torque/Stresslet to Couplet
        Mobility_TS2C_kernel<<<grid, threads>>>(d_couplet, d_TorqueStress, group_size);
	
	// Add the real space contribution to the velocity
	//
	// Real space calculation takes care of self contributions
	Mobility_RealSpace_kernel<<<grid, threads>>>(d_pos, d_vel, d_delu, d_net_force, d_couplet, group_size, xi, d_ewaldC1, self, ewald_cut, ewald_n, ewald_dr, d_group_members, box, d_nneigh, d_nlist, d_headlist );
	
	// Map velocity gradient back to angular velocity and rate of strain
        Mobility_D2WE_kernel<<<grid, threads>>>(d_delu, d_AngvelStrain, group_size);

}


/*!
	Wrap all the functions to compute U = M * F ( SAME SIZE PARTICLES )
	Drive GPU kernel functions
	\param d_vel array of particle velocities
	\param d_net_force array of net forces

		[ vel          ] = M * [ net_force    ]
		[ AngvelStrain ]       [ TorqueStress ]

	d_vel = M * d_net_force
	
	d_pos			(input)  positions of the particles
	d_vel			(output) linear velocity of particles
	d_AngvelStrain		(output) angular velocity and straing of particles
	d_net_force		(input)  linear force on particles
	d_TorqueStress		(input)	 Torque and stresslet on the particles
	d_group_members		(input)  index array to global HOOMD tag on each particle
	group_size		(input)  size of the group, i.e. number of particles
	box			(input)  periodic box information
	ker_data		(input)  structure containing information for kernel launches
	mob_data		(input)  structure containing informaiton for mobility calculations

*/
void Mobility_MobilityUD(	
				Scalar4 *d_pos,
                        	Scalar4 *d_vel,
				Scalar4 *d_AngvelStrain,
                        	Scalar4 *d_net_force,
				Scalar4 *d_TorqueStress,
				unsigned int *d_group_members,
				unsigned int group_size,
                        	const BoxDim& box,
				KernelData *ker_data,
				MobilityData *mob_data,
				WorkData *work_data
				){

	// Kernel Data
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// NEED THIS FOR CUSP RESULT TO BE OK
	cudaMemcpy( d_vel, d_net_force, group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

	// Real and wave space velocity
	Scalar4 *d_vel1 = (work_data->mob_vel1);
	Scalar4 *d_vel2 = (work_data->mob_vel2);
	Scalar4 *d_delu1 = (work_data->mob_delu1);
	Scalar4 *d_delu2 = (work_data->mob_delu2);

	Scalar4 *d_couplet = (work_data->mob_couplet);
	Scalar4 *d_delu = (work_data->mob_delu);

	// Map Torque/Stresslet to Couplet
        Mobility_TS2C_kernel<<<grid, threads>>>(d_couplet, d_TorqueStress, group_size);
	
	// Compute the wave space contribution to the velocity
	gpu_stokes_Mwave_wrap( 
				d_pos,
				d_vel1,
				d_delu1,
				d_net_force,
				d_couplet,
				d_group_members,
				group_size,
				box,
				mob_data->xi,
				mob_data->eta,
				mob_data->gridk,
				mob_data->gridX,
				mob_data->gridY,
				mob_data->gridZ,
				mob_data->gridXX,
				mob_data->gridXY,
				mob_data->gridXZ,
				mob_data->gridYX,
				mob_data->gridYY,
				mob_data->gridYZ,
				mob_data->gridZX,
				mob_data->gridZY,
				mob_data->plan,
				mob_data->Nx,
				mob_data->Ny,
				mob_data->Nz,
				ker_data->NxNyNz,
				ker_data->particle_grid,
				ker_data->particle_threads,
				ker_data->grid_threads,
				ker_data->grid_grid,
				mob_data->P,
				mob_data->gridh
				);
		
	// Compute the real space contribution to the velocity
	//
	// Real space calculation takes care of self contributions
	Mobility_RealSpace_kernel<<<grid, threads>>>(
							d_pos,
							d_vel2,
							d_delu2,
							d_net_force,
							d_couplet,
							group_size,
							mob_data->xi,
							mob_data->ewald_table,
							mob_data->self,
							mob_data->ewald_cut,
							mob_data->ewald_n,
							mob_data->ewald_dr,
							d_group_members,
							box,
							mob_data->nneigh,
							mob_data->nlist,
							mob_data->headlist
							);
	
	// Add real and wave space parts together
	Mobility_LinearCombination_kernel<<<grid, threads>>>(d_vel1, d_vel2, d_vel, 1.0, 1.0, group_size, d_group_members);
	Mobility_Add4_kernel<<<grid, threads>>>(d_delu1, d_delu2, d_delu, 1.0, 1.0, group_size);

	// Map velocity gradient back to angular velocity and rate of strain
        Mobility_D2WE_kernel<<<grid, threads>>>(d_delu, d_AngvelStrain, group_size);

	// Free memory
	d_vel1 = NULL;
	d_vel2 = NULL;
	d_delu1 = NULL;
	d_delu2 = NULL;
	d_couplet = NULL;
	d_delu = NULL;

}

/*!
	Wrap all the functions to compute U = M * F ( SAME SIZE PARTICLES )
	Drive GPU kernel functions

	!!! 
		This is the generalized grand mobility, so the forces are stored as the
		set of generalized forces on each particle followed by the stresslet
		on each particle, and similary the velocities are stored as the 
		generalized velocity on each particle followed by the stresslet on 
		each particle 
	!!!
	
	This function calls the appropriate functions to rearrange the data for the 
	original formulation of M and its associated data storage	

	d_generalU       	(output) generalized particle velocity
	d_generalF       	(input)  generalized particle force
	d_pos			(input)  positions of the particles
	d_group_members		(input)  index array to global HOOMD tag on each particle
	group_size		(input)  size of the group, i.e. number of particles
	box			(input)  periodic box information
	ker_data		(input)  structure containing information for kernel launches
	mob_data		(input)  structure containing informaiton for mobility calculations
*/
void Mobility_GeneralizedMobility(	
					float *d_generalU, // output
					float *d_generalF, // input
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					unsigned int group_size,
                        		const BoxDim& box,
					KernelData *ker_data,
					MobilityData *mob_data,
					WorkData *work_data
					){

	// Get kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Storage arrays for the velocities
        Scalar4 *d_vel = (work_data->mob_vel);
	Scalar4 *d_AngvelStrain = (work_data->mob_AngvelStrain);
        Scalar4 *d_net_force = (work_data->mob_net_force);
	Scalar4 *d_TorqueStress = (work_data->mob_TorqueStress);

	// Convert generalized force to force/torque/stresslet
	Saddle_SplitGeneralizedF_kernel<<<grid,threads>>>(
						 	d_generalF, 
							d_net_force,
							d_TorqueStress,
							group_size
							);

	// Call the Mobility wrapper
	Mobility_MobilityUD(	
				d_pos,
                        	d_vel,
				d_AngvelStrain,
                        	d_net_force,
				d_TorqueStress,
				d_group_members,
				group_size,
                        	box,
				ker_data,
				mob_data,
				work_data
				);

	// Convert velocity/angular velocity/rate of strain to generalized velocity
	Saddle_MakeGeneralizedU_kernel<<<grid,threads>>>(
							d_generalU, 
							d_vel,
							d_AngvelStrain,
							group_size
							);
	
	// Clean up
	d_vel = NULL;
	d_AngvelStrain = NULL;
	d_net_force = NULL;
	d_TorqueStress = NULL;
 
}
