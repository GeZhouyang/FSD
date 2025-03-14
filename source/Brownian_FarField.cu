// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Zhouyang Ge

#include "Brownian_FarField.cuh"
#include "Mobility.cuh"

#include "Helper_Brownian.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Saddle.cuh"

#include "hoomd/Saru.h"
using namespace hoomd;

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <cuda_runtime.h>

#include "lapacke.h"
#include "cublas_v2.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! 
	\file Brownian_FarField.cu
    	\brief Defines functions for PSE calculation of the far-field Brownian Displacements

    	Uses LAPACKE to perform the final square root of the tridiagonal matrix
	resulting from the Lanczos Method
*/


/*!
  	Generate random numbers on particles
	
	d_psi			(output) random vector
        group_size		(input)  number of particles
	d_group_members		(input)  index to particle arrays
	seed			(input)  seed for random number generation
*/
__global__ void Brownian_FarField_RNG_Particle_kernel(Scalar4 *d_psi,
						      unsigned int group_size,
						      unsigned int *d_group_members,
						      const unsigned int seed
						      )
{
  // Thread index
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Check if thread is in bounds, and if so do work
  if (idx < group_size)
    {
      // Initialize random number generator
      detail::Saru s(idx, seed);
      
      // Square root of 3 (variance of uniform distribution in [-1,1] is 1/3)
      float sqrt3 = 1.732050807568877;
      		
      //
      float randomx, randomy, randomz, randomw;
      
      // First bit (force)
      randomx = s.f( -sqrt3, sqrt3 );
      randomy = s.f( -sqrt3, sqrt3 );
      randomz = s.f( -sqrt3, sqrt3 );
      randomw = s.f( -sqrt3, sqrt3 );
      
      d_psi[ idx ] = make_scalar4( randomx, randomy, randomz, randomw );
      
      // Second bit (torque and Sxx)
      randomx = s.f( -sqrt3, sqrt3 );
      randomy = s.f( -sqrt3, sqrt3 );
      randomz = s.f( -sqrt3, sqrt3 );
      randomw = s.f( -sqrt3, sqrt3 );
      
      d_psi[ group_size + 2*idx ] = make_scalar4( randomx, randomy, randomz, randomw );
      		
      // Third bit (Sxy, Sxz, Syz, Syy)
      randomx = s.f( -sqrt3, sqrt3 );
      randomy = s.f( -sqrt3, sqrt3 );
      randomz = s.f( -sqrt3, sqrt3 );
      randomw = s.f( -sqrt3, sqrt3 );
      
      d_psi[ group_size + 2*idx+1 ] = make_scalar4( randomx, randomy, randomz, randomw );

    }
}

/*!
	Fluctuating force calculation. Step 1: Random vector on wave space grid with proper conjugacy.
	Random vector is in the space of force density

	d_gridX		(output) x-component of vectors on grid
	d_gridY		(output) y-component of vectors on grid
	d_gridZ		(output) z-component of vectors on grid
	d_gridk		(input)  reciprocal lattice vectors for each grid point
	NxNyNz		(input)  total number of grid points
	Nx		(input)  number of grid points in x-direction
	Ny		(input)  number of grid points in y-direction
	Nz		(input)  number of grid points in z-direction
	seed		(input)  seed for random number generation
	T		(input)  simulation temperature
	dt		(input)  simulation time step size
	quadW		(input)  quadrature weight for spectral Ewald integration
*/
__global__ void Brownian_FarField_RNG_Grid1of2_kernel(  	CUFFTCOMPLEX *gridX,
								CUFFTCOMPLEX *gridY,
								CUFFTCOMPLEX *gridZ,
								Scalar4 *gridk,
								float *d_gauss2,
				        			unsigned int NxNyNz,
								int Nx,
								int Ny,
								int Nz,
				        			const unsigned int seed,
								Scalar T,
								Scalar dt,
								Scalar quadW
				             			)
{
  // Thread index
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Do work if thread is in bounds
  if ( idx < NxNyNz ) {
 
    //// Scaling factor for covaraince based on Fluctuation-Dissipation
    //// 	fac1 = sqrt( 2.0 * T / dt / quadW );
    //// Scaling factor for covariance of random uniform on [-1,1]
    //// 	fac2 = sqrt( 3.0 )
    //// Scaling factor because each number has real and imaginary part
    //// 	fac3 = 1 / sqrt( 2.0 )	
    //// Total scaling factor
    ////	fac = fac1 * fac2 * fac3 
    ////	    = sqrt( 3.0 * T / dt / quadW )
    //Scalar fac = sqrtf( 3.0 * T / dt / quadW );
    //
    //// Get random numbers 
    //detail::Saru s(idx, seed);
    //		
    //Scalar reX = s.f( -fac, fac );
    //Scalar reY = s.f( -fac, fac );
    //Scalar reZ = s.f( -fac, fac );
    //Scalar imX = s.f( -fac, fac );
    //Scalar imY = s.f( -fac, fac );
    //Scalar imZ = s.f( -fac, fac );

    //zhoge: use Gaussian random variables
    Scalar fac = sqrtf( T / dt / quadW ); //no need to times sqrt(3) for gaussian of std=1
    
    Scalar reX = fac * d_gauss2[ idx     ];
    Scalar reY = fac * d_gauss2[ idx + 1 ];
    Scalar reZ = fac * d_gauss2[ idx + 2 ];
    Scalar imX = fac * d_gauss2[ idx + 3 ];
    Scalar imY = fac * d_gauss2[ idx + 4 ];
    Scalar imZ = fac * d_gauss2[ idx + 5 ];

    
    // Indices for current grid point
    int kk = idx % Nz;
    int jj = ( ( idx - kk ) / Nz ) % Ny;
    int ii = ( ( idx - kk ) / Nz - jj ) / Ny;
		
    // Only have to do the work for half the grid points (zhoge: Not exactly half, can be bigger or smaller)
    if ( 	!( 2 * kk >= Nz + 1 ) &&  // Lower half of the cube across the z-plane
		!( ( kk == 0 ) && ( 2 * jj >= Ny + 1 ) ) && // lower half of the plane across the y-line
		!( ( kk == 0 ) && ( jj == 0 ) && ( 2 * ii >= Nx + 1 ) ) && // lower half of the line across the x-point
		!( ( kk == 0 ) && ( jj == 0 ) && ( ii == 0 ) ) // ignore origin
		) {

      // Is current grid point a nyquist point
      bool ii_nyquist = ( ( ii == Nx/2 ) && ( Nx/2 == (Nx+1)/2 ) );
      bool jj_nyquist = ( ( jj == Ny/2 ) && ( Ny/2 == (Ny+1)/2 ) );
      bool kk_nyquist = ( ( kk == Nz/2 ) && ( Nz/2 == (Nz+1)/2 ) );
			
      // Index of conjugate point
      int ii_conj, jj_conj, kk_conj;
      if ( ii == 0 || ii_nyquist ){
	ii_conj = ii;
      }
      else {
	ii_conj = Nx - ii;
      }
      if ( jj == 0 || jj_nyquist ){
	jj_conj = jj;
      }
      else {
	jj_conj = Ny - jj;
      }
      if ( kk == 0 || kk_nyquist ){
	kk_conj = kk;
      }
      else {
	kk_conj = Nz - kk;
      }
		
      // Index of conjugate grid point
      int conj_idx = ii_conj * Ny*Nz + jj_conj * Nz + kk_conj;
		
      // Nyquist points
      if ( ( ii == 0    && jj_nyquist && kk == 0 ) ||
	   ( ii_nyquist && jj == 0    && kk == 0 ) ||
	   ( ii_nyquist && jj_nyquist && kk == 0 ) ||
	   ( ii == 0    && jj == 0    && kk_nyquist ) ||
	   ( ii == 0    && jj_nyquist && kk_nyquist ) ||
	   ( ii_nyquist && jj == 0    && kk_nyquist ) ||
	   ( ii_nyquist && jj_nyquist && kk_nyquist ) ){
	
	// Since forces only have real part, they have to be rescaled to have variance 1	
	float sqrt2 = 1.414213562373095;
	gridX[idx] = make_scalar2( sqrt2*reX, 0.0 );
	gridY[idx] = make_scalar2( sqrt2*reY, 0.0 );
	gridZ[idx] = make_scalar2( sqrt2*reZ, 0.0 );

      }
      else {
		

	// Record Force
	gridX[idx] = make_scalar2( reX, imX );
	gridY[idx] = make_scalar2( reY, imY );
	gridZ[idx] = make_scalar2( reZ, imZ );
				
	// Conjugate points: F(k_conj) = conj( F(k) )
	gridX[conj_idx] = make_scalar2( reX, -imX );
	gridY[conj_idx] = make_scalar2( reY, -imY );
	gridZ[conj_idx] = make_scalar2( reZ, -imZ );
				
      } // Check for Nyquist
		
    } // Check if lower half of grid

  } // Check if thread in bounds
}


/*!
	Fluctuating force calculation. Step 2: Scaling to get action of square root of wave
					       space contribution to the Ewald sum

        zhoge: Sign error corrected (same error as in Mobility_WaveSpace_Green_kernel).
					     
	d_gridX		(input/output) x-component of vectors on grid
	d_gridY		(input/output) y-component of vectors on grid
	d_gridZ		(input/output) z-component of vectors on grid
	d_gridXX	(output) xx-component of vectors on grid
	d_gridXY	(output) xy-component of vectors on grid
	d_gridXZ	(output) xz-component of vectors on grid
	d_gridYX	(output) yx-component of vectors on grid
	d_gridYY	(output) yy-component of vectors on grid
	d_gridYZ	(output) yz-component of vectors on grid
	d_gridZX	(output) zx-component of vectors on grid
	d_gridZY	(output) zy-component of vectors on grid
	d_gridk		(input)  reciprocal lattice vectors for each grid point
	NxNyNz		(input)  total number of grid points
	Nx		(input)  number of grid points in x-direction
	Ny		(input)  number of grid points in y-direction
	Nz		(input)  number of grid points in z-direction
	seed		(input)  seed for random number generation
	T		(input)  simulation temperature
	dt		(input)  simulation time step size
	quadW		(input)  quadrature weight for spectral Ewald integration

*/
__global__ void Brownian_FarField_RNG_Grid2of2_kernel(  	
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

	// Thread ID
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
	if ( idx < NxNyNz ) {

		// Current wave-space vector 
		Scalar4 tk = gridk[idx];
		Scalar ksq = tk.x*tk.x + tk.y*tk.y + tk.z*tk.z;
		Scalar k = sqrtf( ksq );
		
		// Fluctuating force values
		Scalar2 fX = gridX[ idx ];
		Scalar2 fY = gridY[ idx ];
		Scalar2 fZ = gridZ[ idx ];
		
		// Scaling factors for the current grid
		Scalar B = ( idx == 0 ) ? 0.0 : sqrtf( tk.w );  //tk.w = H/(\eta*V*k^2) is the Hasimoto-Green factor
		Scalar SU = ( idx == 0 ) ? 0.0 : sinf( k ) / k; //j0 (real)
		Scalar SD = ( idx == 0 ) ? 0.0 : 3.0 * ( sinf(k) - k*cosf(k) ) / (ksq * k); //3i*j1/k (imaginary!)
		
		//// CONJUGATE!
		//SD = -1. * SD;  //zhoge: Without this gives the correct result
		
		// k dot f / k^2
		Scalar2 kdF = ( idx == 0 ) ? make_scalar2( 0.0, 0.0 ) : make_scalar2( ( tk.x*fX.x + tk.y*fY.x + tk.z*fZ.x ) / ksq,
										      ( tk.x*fX.y + tk.y*fY.y + tk.z*fZ.y ) / ksq );

		// BdW = B^0.5 * (I - kk)*f, where k is the normalized k (divided by ksq above)
		// (In the FSD paper, tk.w * (I - kk) is called \mathbb{B}.)
		Scalar2 BdWx, BdWy, BdWz;
		BdWx.x = ( fX.x - tk.x * kdF.x ) * B;
		BdWx.y = ( fX.y - tk.x * kdF.y ) * B;
		
		BdWy.x = ( fY.x - tk.y * kdF.x ) * B;
		BdWy.y = ( fY.y - tk.y * kdF.y ) * B;
		
		BdWz.x = ( fZ.x - tk.z * kdF.x ) * B;
		BdWz.y = ( fZ.y - tk.z * kdF.y ) * B;

		// BdW * k (should be \hat{k}, but that's considered in SD, which has an extra 1/k)
		Scalar2 BdWkxx = make_scalar2( BdWx.x*tk.x, BdWx.y*tk.x );
		Scalar2 BdWkxy = make_scalar2( BdWx.x*tk.y, BdWx.y*tk.y );
		Scalar2 BdWkxz = make_scalar2( BdWx.x*tk.z, BdWx.y*tk.z );
		Scalar2 BdWkyx = make_scalar2( BdWy.x*tk.x, BdWy.y*tk.x );
		Scalar2 BdWkyy = make_scalar2( BdWy.x*tk.y, BdWy.y*tk.y );
		Scalar2 BdWkyz = make_scalar2( BdWy.x*tk.z, BdWy.y*tk.z );
		Scalar2 BdWkzx = make_scalar2( BdWz.x*tk.x, BdWz.y*tk.x );
		Scalar2 BdWkzy = make_scalar2( BdWz.x*tk.y, BdWz.y*tk.y );
		
		// Velocity
		gridX[idx].x = SU * BdWx.x;
		gridX[idx].y = SU * BdWx.y;
		
		gridY[idx].x = SU * BdWy.x;
		gridY[idx].y = SU * BdWy.y;
		
		gridZ[idx].x = SU * BdWz.x;
		gridZ[idx].y = SU * BdWz.y;
		
		// Velocity Gradient
		gridXX[idx].x = - SD * BdWkxx.y;
		gridXX[idx].y = + SD * BdWkxx.x;
		
		gridXY[idx].x = - SD * BdWkxy.y;
		gridXY[idx].y = + SD * BdWkxy.x;
		
		gridXZ[idx].x = - SD * BdWkxz.y;
		gridXZ[idx].y = + SD * BdWkxz.x;
		
		gridYX[idx].x = - SD * BdWkyx.y;
		gridYX[idx].y = + SD * BdWkyx.x;
		
		gridYY[idx].x = - SD * BdWkyy.y;
		gridYY[idx].y = + SD * BdWkyy.x;
		
		gridYZ[idx].x = - SD * BdWkyz.y;
		gridYZ[idx].y = + SD * BdWkyz.x;
		
		gridZX[idx].x = - SD * BdWkzx.y;
		gridZX[idx].y = + SD * BdWkzx.x;
		
		gridZY[idx].x = - SD * BdWkzy.y;
		gridZY[idx].y = + SD * BdWkzy.x;

    	}
}


/*!
	Use Lanczos method to compute Mreal^0.5 * psi (get the real space contribution
        to the Brownian displacement.

	This method is detailed in the publication:
	Edmond Chow and Yousef Saad, PRECONDITIONED KRYLOV SUBSPACE METHODS FOR
	SAMPLING MULTIVARIATE GAUSSIAN DISTRIBUTIONS, SIAM J. Sci. Comput., 2014

	d_psi			(input)		Random vector for multiplication
	d_pos			(input)		particle positions
	d_group_members		(input)		indices for particles within the group
	group_size		(input)		number of particles
	box			(input)		periodic box information
	dt			(input)		integration time step
	d_M12psi		(output)	Product of sqrt(M) * psi
	T			(input)		temperature
	seed			(input)		seed for random number generator
	xi			(input)		Ewald splitting parameter
	ewald_cut		(input)		cutoff radius for real space Ewald sum
	ewald_dr		(input)		discretization of real space Ewald tabulation
	ewald_n			(input)		number of tabulated distances for real space Ewald tabulation
	d_ewaldC1		(input)		tabulated real space Ewald sums
	d_nneigh		(input)		number of neighbors for real space Ewald sum
	d_nlist			(input)		neighbor list for real space Ewald sum
	d_headlist		(input)		head list for neighbor list of real space Ewald sum
	m			(input/output)	number of iterations suggested/required
	tol			(input) 	calculation error tolerance
	grid			(input)		grid for CUDA kernel launches
	threads			(input)		threads for CUDA kernel launches
	gridh			(input)		grid discretization
	self			(input) 	self piece for Ewald sum
	work_data		(input)		data structure with points to workspaces

*/



//zhoge: Re-implemented Chow & Saad (2014) method to sample correlated noise (far-field).

__global__ void Type_conversion_Append_zero_kernel( float   *d_x,  //input
						    Scalar4 *d_y,  //output
						    unsigned int group_size )
{
  // Thread index
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
  // Check if thread is within bounds
  if ( idx < group_size )
    {
      // [ F0x,F0y,F0z,     F1x,F1y,F1z,    ... ] ->
      // [ F0x,F0y,F0z,0], [F1x,F1y,F1z,0], ... ]
      d_y[idx] = make_scalar4( d_x[ 3*idx     ],
			       d_x[ 3*idx + 1 ],
			       d_x[ 3*idx + 2 ],
			       0.0 );
    }
}

__global__ void Type_conversion_Append_five_zeros_kernel( float   *d_x,  //input
							  Scalar4 *d_y,  //output
							  unsigned int group_size )
{
  // Thread index
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
  // Check if thread is within bounds
  if ( idx < group_size )
    {
      // [ T0x,T0y,T0z,                T1x,T1y,T1z,              ... ] ->
      // [ T0x,T0y,T0z,0], [0,0,0,0], [T1x,T1y,T1z,0], [0,0,0,0] ... ]
      d_y[2*idx] = make_scalar4( d_x[ 3*idx     ],
				 d_x[ 3*idx + 1 ],
				 d_x[ 3*idx + 2 ],
				 0.0 );
      d_y[2*idx+1] = make_scalar4( 0.0, 0.0, 0.0, 0.0 );
    }
}


__global__ void Type_conversion_Pop_zero_kernel( float   *d_x,  //output
						 Scalar4 *d_y,  //input
						 unsigned int group_size )
{
  // Thread index
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
  // Check if thread is within bounds
  if ( idx < group_size )
    {
      // [ F0x,F0y,F0z,     F1x,F1y,F1z,    ... ] <-
      // [ F0x,F0y,F0z,0], [F1x,F1y,F1z,0], ... ]
      d_x[ 3*idx     ] = d_y[ idx ].x;
      d_x[ 3*idx + 1 ] = d_y[ idx ].y;
      d_x[ 3*idx + 2 ] = d_y[ idx ].z;
    }
}


__global__ void Type_conversion_Pop_strain_kernel( float   *d_x,  //output
						   Scalar4 *d_y,  //input
						   unsigned int group_size )
{
  // Thread index
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
  // Check if thread is within bounds
  if ( idx < group_size )
    {
      // Pop the strain, keep the omega
      d_x[ 3*idx     ] = d_y[ 2*idx ].x;
      d_x[ 3*idx + 1 ] = d_y[ 2*idx ].y;
      d_x[ 3*idx + 2 ] = d_y[ 2*idx ].z;
    }
}


void Lanczos_process( float *d_vm,  //input
		      float *d_v,   //input
		      float *d_vp,  //output
		      float *alpha, //output
		      float *beta,  //output/input
		      float tol_beta,
		      int numel,
		      Scalar4 *d_pos,
		      unsigned int *d_group_members,
		      const int group_size, 
		      const BoxDim box,
		      //void *pBuffer,
		      KernelData *ker_data,
		      //ResistanceData *res_data,
		      MobilityData *mob_data,
		      WorkData *work_data )
{
  // cuBLAS handle
  cublasHandle_t blasHandle = work_data->blasHandle;

  // Kernel information
  dim3 grid    = ker_data->particle_grid;
  dim3 threads = ker_data->particle_threads;

  // Apply A to d_v (d_vp = A * d_v), which takes a few extra steps due to different data structures in the mobility.
  Scalar4 *d_psi  = work_data->bro_ff_psi;
  Scalar4 *d_Mpsi = work_data->bro_ff_Mpsi;

  // Copy d_v to d_psi (but be careful with the zeros)
  Type_conversion_Append_zero_kernel<<<grid,threads>>>( d_v,   //input (the first 3N floats)
							d_psi, //output (the first N Scalar4's)
							group_size );
  // if input 6N Gaussians
  //Type_conversion_Append_five_zeros_kernel<<<grid,threads>>>( &d_v[3*group_size],  //input (the next 3N floats)
  //							      &d_psi[group_size],  //output (the next 2N Scalar4's)
  //							      group_size );
  // if input 11N Gaussians
  cudaMemcpy( &d_psi[group_size], &d_v[3*group_size], 2*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

  // Do the mobility calculation, [U D] = M * [F C]
  Mobility_RealSpaceFTS( d_pos,
			 d_Mpsi,               //output: linear velocity of particles (Scalar4)
			 &d_Mpsi[group_size],  //output: angular velocity and rate of strain of particles (Scalar4)
			 d_psi,                //input:  linear force on particles (Scalar4)
			 &d_psi[group_size],   //input:  torque and stress on particles (Scalar4)
			 work_data->mob_couplet,
			 work_data->mob_delu,  //zhoge: modified inside !!!
			 d_group_members,
			 group_size,
			 box,
			 mob_data->xi,
			 mob_data->ewald_cut,
			 mob_data->ewald_dr,
			 mob_data->ewald_n,
			 mob_data->ewald_table, //zhoge: called mob_data->d_ewaldC1 in the mobility
			 mob_data->self,
			 mob_data->nneigh,
			 mob_data->nlist,
			 mob_data->headlist,
			 ker_data->particle_grid,
			 ker_data->particle_threads );

  // Copy d_Mpsi to d_vp (but be careful with the zeros)
  Type_conversion_Pop_zero_kernel<<<grid,threads>>>( d_vp,   //output (the first 3N floats)
						     d_Mpsi, //input (the first N Scalar4's)
						     group_size );
  // if input 6N Gaussians
  //Type_conversion_Pop_strain_kernel<<<grid,threads>>>( &d_vp[3*group_size],  //output (the next 3N floats)
  //						       &d_Mpsi[group_size],  //input (the next 2N Scalar4's)
  //						       group_size );
  // if input 11N Gaussians
  cudaMemcpy( &d_vp[3*group_size], &d_Mpsi[group_size], 2*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );
  // End applying A to d_v.
  
  
  // Project out d_vm (d_vp = d_vp - beta * d_vm)
  float scale = -1.0 * beta[0];
  cublasSaxpy( blasHandle, numel, &scale, d_vm, 1, d_vp, 1 );  //d_vp is modified in place

  // The diagonal value associated with dv (alpha = d_v \cdot d_vp)
  cublasSdot( blasHandle, numel, d_v, 1, d_vp, 1, alpha );
  
  // Project out d_v (d_vp = d_vp - alpha * d_v)
  scale = -1.0 * alpha[0];
  cublasSaxpy( blasHandle, numel, &scale, d_v, 1, d_vp, 1 );  //d_vp is modified in place

  // The norm of d_vp (betap = || d_vp ||)
  cublasSnrm2( blasHandle, numel, d_vp, 1, &beta[1] );

  // Check if the norm has become very small and if so, set d_vp = d_v
  if ( beta[1] < tol_beta )
    {
      cudaMemcpy( d_vp, d_v, numel*sizeof(float), cudaMemcpyDeviceToDevice );
    }
  else  //otherwise normalize d_vp
    {
      scale = 1.0 / beta[1];
      cublasSscal( blasHandle, numel, &scale, d_vp, 1 );  //d_vp is modified in place
    }
  
}



void Brownian_FarField_Chow_Saad( Scalar *d_y,  //output: far-field Brownian slip (real part)
				  Scalar *d_x,  //input: random Gaussian variables
				  Scalar4 *d_pos,
				  unsigned int *d_group_members,
				  unsigned int group_size,
				  const BoxDim& box,
				  Scalar dt,
				  //void *pBuffer,
				  KernelData *ker_data,
				  BrownianData *bro_data,
				  //ResistanceData *res_data,
				  MobilityData *mob_data,
				  WorkData *work_data)
{
  // cuBLAS handle
  cublasHandle_t blasHandle = work_data->blasHandle;

  // Constants
  //int numel = 6 * group_size;      //size of v1,v2,...,vm_max, d_x, d_y
  int numel = 11 * group_size;     //size of v1,v2,...,vm_max, d_x, d_y
  int m = bro_data->m_Lanczos_ff;  //number of Lanczos iterations in step 1 (either same as last time or reset in Stokes.cc)
  int m_max = 100;                 //m_max-1 is the maximum size of Tm at the end of step 2 (set to 100 in Stokes.cc)

  //debug
  if ( m >= m_max-1 )
    {
      printf("Illegal condition: m >= m_max-1. Program aborted.");
      exit(1);
    }
  
  // Host vectors for the main and sub-diagonal values of Tm
  float *h_alpha  = (float *)malloc( (m_max)*sizeof(float) );
  float *h_beta   = (float *)malloc( (m_max)*sizeof(float) );
  float *h_alpha1 = (float *)malloc( (m_max)*sizeof(float) );  //buffer
  float *h_beta1  = (float *)malloc( (m_max)*sizeof(float) );  //buffer

  // Set the first element of beta to 0
  h_beta[0] = 0.0;

  // Set the tolerance for beta (less than 1e-6 even for single precision because ||vm|| can be << 1)
  float tol_beta = 1e-8; 

  // Buffer vector for checking convergence
  Scalar *d_y0 = work_data->bro_ff_UB_old1;  

  // Lanczos basis vectors V = [v0, v1, v2, ..., vm_max], v0 is a placeholder
  Scalar *d_V = work_data->bro_ff_V1;
	
  // Zero out v0
  float scale = 0.0;
  cublasSscal( blasHandle, numel, &scale, d_V, 1 );
  
  // Initialize v1 = d_x / ||d_x||
  float xnorm;
  cublasSnrm2( blasHandle, numel, d_x, 1, &xnorm );
  
  cudaMemcpy( &d_V[numel], d_x, numel*sizeof(float), cudaMemcpyDeviceToDevice );
  
  scale = 1.0 / xnorm;
  cublasSscal( blasHandle, numel, &scale, &d_V[numel], 1 ); 
  
  //
  // Step 1: Build Vm and Tm via the Lanczos process
  //
  for ( int j = 0; j < m; ++j )  //iterate at most m times 
    {
      // Find Vm and Tm that approximately satisfy Vm^T * A * Vm = Tm  
      Lanczos_process( &d_V[  j   *numel ],  //input
		       &d_V[ (j+1)*numel ],  //input
		       &d_V[ (j+2)*numel ],  //output
		       &h_alpha[ j ],        //output
		       &h_beta[  j ],        //input [j] / output [j+1]
		       tol_beta,
		       numel,
		       d_pos,
		       d_group_members,
		       group_size, 
		       box,
		       //pBuffer,
		       ker_data,
		       //res_data,
		       mob_data,
		       work_data );

      // Stop if beta becomes very small
      if ( h_beta[j+1] < tol_beta )
	{
	  m = j+1;  //plus 1 because one iteration was done when j=0
	  break;
	}		
    }  

  ////debug
  //printf("m = %i\n",m);
  //for (int i=0; i<m; ++i){
  //  printf("alpha[%2i] = %f\n",i,h_alpha[i]);
  //}
  //printf("\n");
  //for (int i=0; i<m; ++i){
  //  printf("beta[ %2i] = %f\n",i,h_beta[ i]);
  //}
  //exit(1);

  //
  // Step 2: Iteratively compute d_y until convergence
  //
  Sqrt_multiply( &d_V[ numel ],  //input
		 h_alpha,        //input
		 h_beta,	 //input
		 h_alpha1,	 //input (buffer)
		 h_beta1,        //input (buffer)
		 m,              //input 
		 d_y0,           //output
		 numel,
		 group_size,
		 ker_data,
		 work_data );

  float error = 1.0;
  float ynorm = 1.0;

  cublasSnrm2( blasHandle, numel, d_y0, 1, &ynorm );

  ////debug
  //printf("Step 2: norm of d_y0 %10.3e\n",ynorm); 
  //exit(1);
  
  while( error > bro_data->tol and m < m_max-1 )
    {
      // Iteratively increase m
      Lanczos_process( &d_V[  m   *numel ],  //input
		       &d_V[ (m+1)*numel ],  //input
		       &d_V[ (m+2)*numel ],  //output
		       &h_alpha[ m ],        //output
		       &h_beta[  m ],        //input [m] / output [m+1]
		       tol_beta,
		       numel,
		       d_pos,
		       d_group_members,
		       group_size, 
		       box,
		       //pBuffer,
		       ker_data,
		       //res_data,
		       mob_data,
		       work_data );

      // Compute the new approximate solution, d_y
      Sqrt_multiply( &d_V[ numel ],  //input
		     h_alpha,        //input
		     h_beta,	     //input
		     h_alpha1,	     //input (buffer)
		     h_beta1,        //input (buffer)
		     m+1,            //input 
		     d_y,            //output
		     numel,
		     group_size,
		     ker_data,
		     work_data );

      	
      // Compute relative error = || d_y0 - d_y || / || d_y ||
      scale = -1.0;
      cublasSaxpy( blasHandle, numel, &scale, d_y, 1, d_y0, 1 );  //d_y0 is modified in place
      cublasSnrm2( blasHandle, numel, d_y0, 1, &error );
      cublasSnrm2( blasHandle, numel, d_y,  1, &ynorm );
      error /= ynorm;

      ////debug
      //printf("Chow & Saad (far-field) iteration %3i, relative error %13.6e (norm of d_y %13.6e)\n",m,error,ynorm);

      // Update solution
      cudaMemcpy( d_y0, d_y, numel*sizeof(float), cudaMemcpyDeviceToDevice );

      // Stop if beta becomes very small (even if the error is not small enough)
      if ( h_beta[m+1] < tol_beta )
	{
	  ++m;
	  break;
	}

      // Increment m
      ++m;
	
    }
  
  ////debug
  //printf("\n");

  // Finalize
  if ( error > bro_data->tol )
    {
      printf("\nChow & Saad (far-field) didn't converge after %i iterations.\n",m-1);
      printf("Final relative error %13.6e\n",error);
      printf("Last beta %13.6e\n",h_beta[m]);
      //printf("\nProgram aborted.\n");
      //exit(1);
    }
  
  // Save the number of required iterations (minus 1 because incremented at the end)
  bro_data->m_Lanczos_ff = m-1;

  // Rescale by original norm of d_x and the thermal scale (2*kT/dt)^0.5
  xnorm *= sqrtf(2.0 * bro_data->T / dt);
  cublasSscal( blasHandle, numel, &xnorm, d_y, 1 );
  
  // Clean up
  free(h_alpha);
  free(h_alpha1);
  free(h_beta);
  free(h_beta1);
}






/*
	Compute the Brownian slip due to the far-field hydrodynamic interactions.
	zhoge: This also includes the self contribution.

	d_Uslip_ff		(output) Far-field Brownian slip velocity
	d_pos			(input)  particle positions
	d_group_members		(input)  indices for particles within the group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	dt			(input)  integration time step
	bro_data		(input)  Structure with information for Brownian calculations
	mob_data		(input)  Structure with information for Mobility calculations
	ker_data		(input)  Structure with information for kernel launches
*/
void Brownian_FarField_SlipVelocity(float *d_Uslip_ff,
				    Scalar4 *d_pos,
				    unsigned int *d_group_members,
				    unsigned int group_size,
				    const BoxDim& box,
				    Scalar dt,
				    BrownianData *bro_data,
				    MobilityData *mob_data,
				    KernelData *ker_data,
				    WorkData *work_data)
{
  // Kernel Stuff
  dim3 gridNBlock    = ker_data->grid_grid;
  dim3 gridBlockSize = ker_data->grid_threads;

  dim3 grid    = ker_data->particle_grid;
  dim3 threads = ker_data->particle_threads;

  // Initialize storage variables for velocity and angular velocity/rate of strain
  Scalar4 *d_vel = (work_data->mob_vel);
  Scalar4 *d_delu = (work_data->mob_delu);
  Scalar4 *d_delu1 = work_data->mob_delu1; //zhoge: need this because delu will be modified in Lanczos by Mobility_RealSpaceFTS

  // Real space contribution	
  Scalar4 *d_Mreal12psi = (work_data->bro_ff_UBreal);
  //Scalar4 *d_psi = (work_data->bro_ff_psi);

  //// Generate uniform distribution (-1,1) on d_psi (zhoge: actually from -sqrt(3) to sqrt(3))
  //// VERY IMPORTANT TO USE A DIFFERENT, DE-CORRELATED SEED FROM THE OTHER RANDOM FUNCTION FOR WAVE SPACE!!!!!
  //Brownian_FarField_RNG_Particle_kernel<<<grid, threads>>>(d_psi,  //output
  //							   group_size,
  //							   d_group_members,
  //							   bro_data->seed_ff_rs );

  //zhoge: use cuRand to generate Gaussian variables
  curandGenerator_t gen;
  float *d_gauss = work_data->bro_gauss;
  unsigned int N_random = 11*group_size + 6*ker_data->NxNyNz;     //11N for real, 6NxNyNz for imaginary
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);       //the default generator (xorwow)
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);       //too slow
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);        //fast
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);      //slower than MTGP32
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);   //fastest      
  curandSetPseudoRandomGeneratorSeed(gen, bro_data->seed_ff_rs);  //set the seed
  curandGenerateNormal(gen, d_gauss, N_random, 0.0f, 1.0f);       //mean 0, std 1
  curandDestroyGenerator(gen);  

  ////the first 11N floats goes to d_psi (this takes care of the type difference)
  //cudaMemcpy(d_psi, d_gauss, 11*group_size*sizeof(float), cudaMemcpyDeviceToDevice);  


  
  // Spreading and contraction stuff
  int P = (mob_data->P);
  Scalar xi = (mob_data->xi);
  Scalar eta = (mob_data->eta);
  Scalar3 gridh = (mob_data->gridh);

  dim3 Cgrid( group_size, 1, 1);
  int B = ( P < 8 ) ? P : 8;
  dim3 Cthreads(B, B, B);
		
  Scalar quadW = gridh.x * gridh.y * gridh.z;
  Scalar xisq = xi * xi;
  Scalar prefac = ( 2.0 * xisq / 3.1415926536 / eta ) * sqrtf( 2.0 * xisq / 3.1415926536 / eta );
  Scalar expfac = 2.0 * xisq / eta;
		
  // ***************
  // Wave Space Part
  // ***************
		
  // Reset the grid ( remove any previously distributed forces )
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridX,  ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridY,  ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridZ,  ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridXX, ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridXY, ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridXZ, ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridYX, ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridYY, ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridYZ, ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridZX, ker_data->NxNyNz );
  Mobility_ZeroGrid_kernel<<<gridNBlock,gridBlockSize>>>( mob_data->gridZY, ker_data->NxNyNz );
		
  // Apply random fluctuation to wave space grid (zhoge: fluctuation of the force density)
  Brownian_FarField_RNG_Grid1of2_kernel<<<gridNBlock,gridBlockSize>>>(mob_data->gridX,  //output
  								      mob_data->gridY,	//output
  								      mob_data->gridZ,	//output
  								      mob_data->gridk,
  								      &d_gauss[11*group_size], //zhoge: pre-generated Gaussian
  								      ker_data->NxNyNz,
  								      mob_data->Nx,
  								      mob_data->Ny,
  								      mob_data->Nz,
  								      bro_data->seed_ff_ws,
  								      bro_data->T,
  								      dt,
  								      quadW);
  
  Brownian_FarField_RNG_Grid2of2_kernel<<<gridNBlock,gridBlockSize>>>(mob_data->gridX,  //input/output
  								      mob_data->gridY,	//input/output
  								      mob_data->gridZ,	//input/output
  								      mob_data->gridXX,	//output
  								      mob_data->gridXY,	//output
  								      mob_data->gridXZ,	//output
  								      mob_data->gridYX,	//output
  								      mob_data->gridYY,	//output
  								      mob_data->gridYZ,	//output
  								      mob_data->gridZX,	//output
  								      mob_data->gridZY,	//output
  								      mob_data->gridk,
  								      ker_data->NxNyNz);
  		
  // Return rescaled forces to real space
  cufftExecC2C( mob_data->plan, mob_data->gridX,  mob_data->gridX,  CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridY,  mob_data->gridY,  CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridZ,  mob_data->gridZ,  CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridXX, mob_data->gridXX, CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridXY, mob_data->gridXY, CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridXZ, mob_data->gridXZ, CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridYX, mob_data->gridYX, CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridYY, mob_data->gridYY, CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridYZ, mob_data->gridYZ, CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridZX, mob_data->gridZX, CUFFT_INVERSE);
  cufftExecC2C( mob_data->plan, mob_data->gridZY, mob_data->gridZY, CUFFT_INVERSE);
  		
  // Evaluate contribution of grid velocities at particle centers
  Mobility_WaveSpace_ContractU<<<Cgrid, Cthreads, (B*B*B+1)*sizeof(float3)>>>(d_pos,
  									      d_vel,  //output
  									      mob_data->gridX,
  									      mob_data->gridY,
  									      mob_data->gridZ,
  									      group_size,
  									      mob_data->Nx,
  									      mob_data->Ny,
  									      mob_data->Nz,
  									      mob_data->xi,
  									      mob_data->eta,
  									      d_group_members,
  									      box,
  									      P,
  									      gridh,
  									      quadW*prefac,
  									      expfac);
  
  Mobility_WaveSpace_ContractD<<<Cgrid, Cthreads, (2*B*B*B+1)*sizeof(float4)>>>(d_pos,
  										d_delu,  //output
  										mob_data->gridXX,
  										mob_data->gridXY,
  										mob_data->gridXZ,
  										mob_data->gridYX,
  										mob_data->gridYY,
  										mob_data->gridYZ,
  										mob_data->gridZX,
  										mob_data->gridZY,
  										group_size,
  										mob_data->Nx,
  										mob_data->Ny,
  										mob_data->Nz,
  										mob_data->xi,
  										mob_data->eta,
  										d_group_members,
  										box,
  										P,
  										gridh,
  										quadW*prefac,
  										expfac);
  
  // Convert to Angular velocity and rate of strain (can be done in-place)
  Mobility_D2WE_kernel<<<grid, threads>>>(d_delu,  //input (wave)
					  d_delu1, //output (wave)
					  group_size);

  
  // ***************
  // Real Space Part
  // ***************
  
  //zhoge: Apply the Chow & Saad method to sample the far-field velocity (real part)
  Scalar *d_UB1 = work_data->bro_ff_UB_new1;

  Brownian_FarField_Chow_Saad( d_UB1,    //output
  			       d_gauss,  //input (use d_gauss because of cublas)
  			       d_pos,
  			       d_group_members,
  			       group_size,
  			       box,
  			       dt,
  			       //pBuffer,
  			       ker_data,
  			       bro_data,
  			       //res_data,
  			       mob_data,
  			       work_data);  //zhoge: work_data->mob_delu is modified inside (!!)

  // Convert type again to conform with the old implementation (be careful with the zeros)
  Type_conversion_Append_zero_kernel<<<grid,threads>>>( d_UB1,        //input (the first 3N floats)
							d_Mreal12psi, //output (the first N Scalar4's)
							group_size );
  // if input 6N Gaussians
  //Type_conversion_Append_five_zeros_kernel<<<grid,threads>>>( &d_UB1[3*group_size],       //input (the next 3N floats)
  //							      &d_Mreal12psi[group_size],  //output (the next 2N Scalar4's)
  //							      group_size );
  // if input 11N Gaussians
  cudaMemcpy( &d_Mreal12psi[group_size], &d_UB1[3*group_size], 2*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );
  // End new Chow & Saad

  
  
  		
  // Add to wave space part
  Mobility_LinearCombination_kernel<<<grid, threads>>>( d_Mreal12psi, //input  (real)
  							d_vel,	      //input  (wave)
  							d_vel,	      //output (total)
  							1.0,  //real coefficient
  							1.0,  //wave coefficient
  							group_size,
  							d_group_members);
  
  Mobility_Add4_kernel<<<grid, threads>>>( &d_Mreal12psi[group_size], //input  (real) 
  					   d_delu1,		      //input  (wave) 
  					   d_delu,		      //output (total)
  					   1.0,     //real coefficient
  					   1.0,     //wave coefficient
  					   group_size );
  	
  // Rearrange Output
  Saddle_MakeGeneralizedU_kernel<<<grid,threads>>>(d_Uslip_ff, //output
						   d_vel,  
						   d_delu,
						   group_size);

}
