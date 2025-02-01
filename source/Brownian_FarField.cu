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
#include "cblas.h"

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
 
    //zhoge//// Scaling factor for covaraince based on Fluctuation-Dissipation
    //zhoge//// 	fac1 = sqrt( 2.0 * T / dt / quadW );
    //zhoge//// Scaling factor for covariance of random uniform on [-1,1]
    //zhoge//// 	fac2 = sqrt( 3.0 )
    //zhoge//// Scaling factor because each number has real and imaginary part
    //zhoge//// 	fac3 = 1 / sqrt( 2.0 )	
    //zhoge//// Total scaling factor
    //zhoge////	fac = fac1 * fac2 * fac3 
    //zhoge////	    = sqrt( 3.0 * T / dt / quadW )
    //zhoge//Scalar fac = sqrtf( 3.0 * T / dt / quadW );
    //zhoge//
    //zhoge//// Get random numbers 
    //zhoge//detail::Saru s(idx, seed);
    //zhoge//		
    //zhoge//Scalar reX = s.f( -fac, fac );
    //zhoge//Scalar reY = s.f( -fac, fac );
    //zhoge//Scalar reZ = s.f( -fac, fac );
    //zhoge//Scalar imX = s.f( -fac, fac );
    //zhoge//Scalar imY = s.f( -fac, fac );
    //zhoge//Scalar imZ = s.f( -fac, fac );

    //zhoge: use Gaussian random variables
    Scalar fac = sqrtf( T / dt / quadW );
    
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
		
    // Only have to do the work for half the grid points	
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
				              			unsigned int NxNyNz,
					      			int Nx,
					      			int Ny,
					      			int Nz,
				              			const unsigned int seed,  //zhoge: unused
					      			Scalar T,
					      			Scalar dt,
					      			Scalar quadW
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
		Scalar B = ( idx == 0 ) ? 0.0 : sqrtf( tk.w );
		Scalar SU = ( idx == 0 ) ? 0.0 : sinf( k ) / k; // real
		Scalar SD = ( idx == 0 ) ? 0.0 : 3.0 * ( sinf(k) - k*cosf(k) ) / (ksq * k); // imaginary!

		// CONJUGATE!	
		SD = -1. * SD;
	
		// Square root of Green's function times dW
		Scalar2 kdF = ( idx == 0 ) ? make_scalar2( 0.0, 0.0 ) : make_scalar2( ( tk.x*fX.x + tk.y*fY.x + tk.z*fZ.x ) / ksq,  ( tk.x*fX.y + tk.y*fY.y + tk.z*fZ.y ) / ksq );
		
		Scalar2 BdWx, BdWy, BdWz;
		BdWx.x = ( fX.x - tk.x * kdF.x ) * B;
		BdWx.y = ( fX.y - tk.x * kdF.y ) * B;

		BdWy.x = ( fY.x - tk.y * kdF.x ) * B;
		BdWy.y = ( fY.y - tk.y * kdF.y ) * B;

		BdWz.x = ( fZ.x - tk.z * kdF.x ) * B;
		BdWz.y = ( fZ.y - tk.z * kdF.y ) * B;

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

	d_psi			(input)		Random vector for multiplication (zhoge: force, torque and stress)
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
void Brownian_FarField_Lanczos( 	
				Scalar4 *d_psi,
				Scalar4 *d_pos,
                                unsigned int *d_group_members,
                                unsigned int group_size,
                                const BoxDim& box,
                                Scalar dt,
			        Scalar4 *d_M12psi,
			        const Scalar T,
			        Scalar xi,
			        Scalar ewald_cut,
			        Scalar ewald_dr,
			        int ewald_n,
			        Scalar4 *d_ewaldC1, 
			        const unsigned int *d_nneigh,
                                const unsigned int *d_nlist,
                                const unsigned int *d_headlist,
			        int& m,
				Scalar tol,
			        dim3 grid,
			        dim3 threads,
			        Scalar3 gridh,
			        Scalar2 self,
				WorkData *work_data
				){

	// Dot product kernel specifications
	unsigned int thread_for_dot = 512; // Must be 2^n
	unsigned int grid_for_dot = (group_size/thread_for_dot) + 1;

	// Temp var for dot product.
	Scalar *dot_sum = (work_data->dot_sum);

	// Allocate storage
	// 
	int m_in = m;
	int m_max = 100; //zhoge: Can probably increase if not converging

        // Storage vectors for tridiagonal factorization
	float *alpha, *beta, *alpha_save, *beta_save;
        alpha = (float *)malloc( (m_max)*sizeof(float) );
        alpha_save = (float *)malloc( (m_max)*sizeof(float) );
        beta = (float *)malloc( (m_max+1)*sizeof(float) );
        beta_save = (float *)malloc( (m_max+1)*sizeof(float) );

	// Vectors for Lapacke and square root
	float *W;
	W = (float *)malloc( (m_max*m_max)*sizeof(float) );
	float *W1; // W1 = Lambda^(1/2) * ( W^T * e1 )
	W1 = (float *)malloc( (m_max)*sizeof(float) );
	float *Tm;
	Tm = (float *)malloc( m_max*sizeof(float) );
	Scalar *d_Tm = (work_data->bro_ff_Tm);

	// Vectors for Lanczos iterations
	Scalar4 *d_v = (work_data->bro_ff_v);
	Scalar4 *d_vj = (work_data->bro_ff_vj);
	Scalar4 *d_vjm1 = (work_data->bro_ff_vjm1);

	// Storage vector for M*vj
	Scalar4 *d_Mvj = (work_data->bro_ff_Mvj);

	// Storage array for V
	Scalar4 *d_V = (work_data->bro_ff_V);

	// Step-norm things
	Scalar4 *d_M12psi_old = (work_data->bro_ff_UB_old);
	Scalar4 *d_Mpsi = (work_data->bro_ff_Mpsi);
	Scalar psiMpsi;

	// Temporary pointer
	Scalar4 *d_temp;

	// Copy random vector to v0
	cudaMemcpy( d_vj, d_psi, 3*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );
	
        Scalar vnorm;
	Brownian_FarField_Dot1of2_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_vj, d_vj,
													    dot_sum, //output
													    group_size,
													    d_group_members);
	Brownian_FarField_Dot2of2_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, //input/output
												 grid_for_dot);
	cudaMemcpy(&vnorm, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);
	vnorm = sqrtf( vnorm );

	Scalar psinorm = vnorm;

    	// Compute psi * M * psi ( for step norm )
	Mobility_RealSpaceFTS(
				d_pos,
				d_Mpsi,               //output: linear velocity of particles
				&d_Mpsi[group_size],  //output: angular velocity and rate of strain of particles
				d_psi,                //input:  linear force on particles
				&d_psi[group_size],   //input:  torque and stress on particles
				(work_data->mob_couplet),
				(work_data->mob_delu),  //zhoge: modified inside !!!
				d_group_members,
				group_size,
				box,
				xi,
				ewald_cut,
				ewald_dr,
				ewald_n,
				d_ewaldC1,
				self,
				d_nneigh,
				d_nlist,
				d_headlist,
				grid,
				threads
				);
	
	//cudaMemcpy( d_Mvj, d_vj, 3*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );
    	
	Brownian_FarField_Dot1of2_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_psi, d_Mpsi, dot_sum, group_size, d_group_members);
    	Brownian_FarField_Dot2of2_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
    	cudaMemcpy(&psiMpsi, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);

	psiMpsi = psiMpsi / ( psinorm * psinorm );

        // First iteration, vjm1 = 0, vj = psi / norm( psi )
	Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_vj, d_vj, d_vjm1, 0.0, 0.0, group_size, d_group_members);
	Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_vj, d_vj, d_vj, 1.0/vnorm, 0.0, group_size, d_group_members);

	m = m_in - 1;
	m = m < 1 ? 1 : m;

	Scalar tempalpha;
	Scalar tempbeta = 0.0;

	tempbeta = 0.0;
	for ( int jj = 0; jj < m; ++jj ){

		// Store current basis vector
		cudaMemcpy( &d_V[jj*3*group_size], d_vj, 3*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

		// Store beta
		beta[jj] = tempbeta;

		// v = M*vj - betaj*vjm1
		Mobility_RealSpaceFTS(
					d_pos,
					d_Mvj,
					&d_Mvj[group_size],
					d_vj,
					&d_vj[group_size],
					(work_data->mob_couplet),
					(work_data->mob_delu),
					d_group_members,
					group_size,
					box,
					xi,
					ewald_cut,
					ewald_dr,
					ewald_n,
					d_ewaldC1,
					self,
					d_nneigh,
					d_nlist,
					d_headlist,
					grid,
					threads
					);
		
		//cudaMemcpy( d_Mvj, d_vj, 3*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );
		
		Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_Mvj, d_vjm1, d_v, 1.0, -1.0*tempbeta, group_size, d_group_members);

		// vj dot v
	        Brownian_FarField_Dot1of2_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_vj, d_v, dot_sum, group_size, d_group_members);
	        Brownian_FarField_Dot2of2_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
	        cudaMemcpy(&tempalpha, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);

		// Store updated alpha
		alpha[jj] = tempalpha;
	
		// v = v - alphaj*vj
		Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_v, d_vj, d_v, 1.0, -1.0*tempalpha, group_size, d_group_members);

		// v dot v 
	        Brownian_FarField_Dot1of2_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_v, d_v, dot_sum, group_size, d_group_members);
	        Brownian_FarField_Dot2of2_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
	        cudaMemcpy(&vnorm, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);
		vnorm = sqrtf( vnorm );

		// betajp1 = norm( v )
		tempbeta = vnorm;

		if ( vnorm < 1E-8 ){

		    m = jj;
		    break;
		}

		// vjp1 = v / betajp1
		Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_v, d_v, d_v, 1.0/tempbeta, 0.0, group_size, d_group_members);

		// Swap pointers
		d_temp = d_vjm1;
		d_vjm1 = d_vj;
		d_vj = d_v;
		d_v = d_temp;
				
	}

	// Save alpha, beta vectors (will be overwritten by lapack)
	for ( int ii = 0; ii < m; ++ii ){
		alpha_save[ii] = alpha[ii];
		beta_save[ii] = beta[ii];
	}
	beta_save[m] = beta[m];


	// Compute eigen-decomposition of tridiagonal matrix
	// 	alpha (input) - vector of entries on main diagonal
	//      alpha (output) - eigenvalues sorted in descending order
	//      beta (input) - vector of entries of sub-diagonal
	//      beta (output) - overwritten (zeros?)
	//      W - (output) - matrix of eigenvectors. ith column corresponds to ith eigenvalue
	// 	INFO (output) = 0 if operation was succesful
	int INFO = LAPACKE_spteqr( LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m );

	if ( INFO != 0 ){
	    printf("Eigenvalue decomposition #1 failed \n");
	    printf("INFO = %i \n", INFO);

	    printf("\n alpha: \n");
	    for( int ii = 0; ii < m; ++ii ){
		printf("%f \n", alpha_save[ii]);
	    } 
	    printf("\n beta: \n");
	    for( int ii = 0; ii < m; ++ii ){
		printf("%f \n", beta_save[ii]);
	    }
	    printf("%f \n", beta_save[m]); 

	    exit(EXIT_FAILURE);
	}


//	printf("    doing square root...\n");

	// Now, we have to compute Tm^(1/2) * e1
	// 	Tm^(1/2) = W * Lambda^(1/2) * W^T * e1
	//	         = W * Lambda^(1/2) * ( W^T * e1 )
	// The quantity in parentheses is the first row of W 
	// Lambda^(1/2) only has diagonal entries, so it's product with the first row of W
	//     is easy to compute.
	for ( int ii = 0; ii < m; ++ii ){
	    W1[ii] = sqrtf( alpha[ii] ) * W[ii];
	}

	// Tm = W * W1 = W * Lambda^(1/2) * W^T * e1
	float tempsum;
	for ( int ii = 0; ii < m; ++ii ){
	    tempsum = 0.0;
	    for ( int jj = 0; jj < m; ++jj ){
		int idx = m*ii + jj;

		tempsum += W[idx] * W1[jj];
	    }
	    Tm[ii] = tempsum;
	}

	// Copy matrix to GPU
	cudaMemcpy( d_Tm, Tm, m*sizeof(Scalar), cudaMemcpyHostToDevice );

	// Multiply basis vectors by Tm
	Brownian_FarField_LanczosMatrixMultiply_kernel<<<grid,threads>>>(d_V, d_Tm, d_M12psi, group_size, m);

	// Copy velocity
	cudaMemcpy( d_M12psi_old, d_M12psi, 3*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

	// Restore alpha, beta
	for ( int ii = 0; ii < m; ++ii ){
		alpha[ii] = alpha_save[ii];
		beta[ii] = beta_save[ii];
	}
	beta[m] = beta_save[m];


	//
	// Keep adding to basis until step norm is small enough
	//
	Scalar stepnorm = 1.0;
	int jj;
	while( stepnorm > tol && m < m_max ){
		m++;
		jj = m - 1;

		//
		// Do another Lanczos iteration
		//

		cudaMemcpy( &d_V[jj*3*group_size], d_vj, 3*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice ); // store current basis vector

		beta[jj] = tempbeta; // store beta

		// v = M*vj - betaj*vjm1
		Mobility_RealSpaceFTS(
					d_pos,
					d_Mvj,
					&d_Mvj[group_size],
					d_vj,
					&d_vj[group_size],
					(work_data->mob_couplet),
					(work_data->mob_delu),
					d_group_members,
					group_size,
					box,
					xi,
					ewald_cut,
					ewald_dr,
					ewald_n,
					d_ewaldC1,
					self,
					d_nneigh,
					d_nlist,
					d_headlist,
					grid,
					threads
					);
		
		//cudaMemcpy( d_Mvj, d_vj, 3*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );
		
		Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_Mvj, d_vjm1, d_v, 1.0, -1.0*tempbeta, group_size, d_group_members);

		// vj dot v
	        Brownian_FarField_Dot1of2_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_vj, d_v, dot_sum, group_size, d_group_members);
	        Brownian_FarField_Dot2of2_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
	        cudaMemcpy(&tempalpha, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);

		alpha[jj] = tempalpha; // store updated alpha
	
		// v = v - alphaj*vj
		Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_v, d_vj, d_v, 1.0, -1.0*tempalpha, group_size, d_group_members);

		// v dot v 
	        Brownian_FarField_Dot1of2_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_v, d_v, dot_sum, group_size, d_group_members);
	        Brownian_FarField_Dot2of2_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
	        cudaMemcpy(&vnorm, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);
		vnorm = sqrtf( vnorm );

		tempbeta = vnorm; // betajp1 = norm( v )

		beta[jj+1] = tempbeta;

		//printf("jj = %i, m = %i, beta = %f, vnorm = %f \n", jj, m, tempbeta, vnorm);

		if ( vnorm < 1E-8 ){
		    m = jj;
		    break;
		}

		// vjp1 = v / betajp1
		Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_v, d_v, d_v, 1.0/tempbeta, 0.0, group_size, d_group_members);

		// Swap pointers
		d_temp = d_vjm1;
		d_vjm1 = d_vj;
		d_vj = d_v;
		d_v = d_temp;
			
		// Save alpha, beta vectors (will be overwritten by lapack)
		for ( int ii = 0; ii < m; ++ii ){
			alpha_save[ii] = alpha[ii];
			beta_save[ii] = beta[ii];
		}
		beta_save[m] = beta[m];
	
		//
		// Square root calculation with addition of latest Lanczos iteration
		//
	
		// Compute eigen-decomposition of tridiagonal matrix
		int INFO = LAPACKE_spteqr( LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m );

		if ( INFO != 0 ){
		    printf("Eigenvalue decomposition #2 failed \n");
		    printf("INFO = %i \n", INFO); 
	    
	    	    printf("\n alpha: \n");
	    	    for( int ii = 0; ii < m; ++ii ){
	    	        printf("%f \n", alpha_save[ii]);
	    	    } 
	    	    printf("\n beta: \n");
	    	    for( int ii = 0; ii < m; ++ii ){
	    	        printf("%f \n", beta_save[ii]);
	    	    }
		    printf("%f \n", beta_save[m]); 
	    
		    exit(EXIT_FAILURE);
		}

		// Now, we have to compute Tm^(1/2) * e1
		for ( int ii = 0; ii < m; ++ii ){
		    W1[ii] = sqrtf( alpha[ii] ) * W[ii];
		}
		// Tm = W * W1 = W * Lambda^(1/2) * W^T * e1
		float tempsum;
		for ( int ii = 0; ii < m; ++ii ){
		    tempsum = 0.0;
		    for ( int jj = 0; jj < m; ++jj ){
			int idx = m*ii + jj;

			tempsum += W[idx] * W1[jj];
		    }
		    Tm[ii] = tempsum;
		}

		// Copy matrix to GPU
		cudaMemcpy( d_Tm, Tm, m*sizeof(Scalar), cudaMemcpyHostToDevice );

		// Multiply basis vectors by Tm -- velocity = Vm * Tm
		Brownian_FarField_LanczosMatrixMultiply_kernel<<<grid,threads>>>(d_V, d_Tm, d_M12psi, group_size, m);

		//
		// Compute step norm error
		//
    		Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_M12psi, d_M12psi_old, d_M12psi_old, 1.0, -1.0, group_size, d_group_members);
        	Brownian_FarField_Dot1of2_kernel<<< grid_for_dot, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(d_M12psi_old, d_M12psi_old, dot_sum, group_size, d_group_members);
        	Brownian_FarField_Dot2of2_kernel<<< 1, thread_for_dot, thread_for_dot*sizeof(Scalar) >>>(dot_sum, grid_for_dot);
        	cudaMemcpy(&stepnorm, dot_sum, sizeof(Scalar), cudaMemcpyDeviceToHost);

		stepnorm = sqrtf( stepnorm / psiMpsi );

		// DEBUG
		//printf("iteration: %i | StepNorm: %f | alpha: %f | beta: %f \n", m, stepnorm, tempalpha, tempbeta );

		// Copy velocity
		cudaMemcpy( d_M12psi_old, d_M12psi, 3*group_size*sizeof(Scalar4), cudaMemcpyDeviceToDevice );

		// Restore alpha, beta
		for ( int ii = 0; ii < m; ++ii ){
			alpha[ii] = alpha_save[ii];
			beta[ii] = beta_save[ii];
		}
		beta[m] = beta_save[m];
			
	}

	// Rescale by original norm of Psi and add thermal variance
	Brownian_Farfield_LinearCombinationFTS_kernel<<<grid, threads>>>(d_M12psi, d_M12psi, d_M12psi, psinorm * sqrtf(2.0*T/dt), 0.0, group_size, d_group_members);
        
	// Free the memory and clear pointers
	dot_sum = NULL;
	d_Mvj = NULL;
	d_v = NULL;
	d_vj = NULL;
	d_vjm1 = NULL;
	d_V = NULL;
	d_Tm = NULL;
	d_M12psi_old = NULL;
	d_Mpsi = NULL;

	d_temp = NULL;

	free(alpha);
	free(beta);
	free(alpha_save);
	free(beta_save);

	free(W);
	free(W1);
	free(Tm);
	
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
  dim3 gridNBlock = (ker_data->grid_grid);
  dim3 gridBlockSize = (ker_data->grid_threads);

  dim3 grid = (ker_data->particle_grid);
  dim3 threads = (ker_data->particle_threads);

  // Initialize storage variables for velocity and angular velocity/rate of strain
  Scalar4 *d_vel = (work_data->mob_vel);
  Scalar4 *d_delu = (work_data->mob_delu);
  Scalar4 *d_delu1 = (work_data->mob_delu1); //zhoge: need this because delu will be modified in the Lanczos by Mobility_RealSpaceFTS

  // Real space contribution	
  Scalar4 *d_Mreal12psi = (work_data->bro_ff_UBreal);
  Scalar4 *d_psi = (work_data->bro_ff_psi);

  //brownian single particle//// Generate uniform distribution (-1,1) on d_psi (zhoge: actually from -sqrt(3) to sqrt(3))
  //brownian single particle//// VERY IMPORTANT TO USE A DIFFERENT, DE-CORRELATED SEED FROM THE OTHER RANDOM FUNCTION FOR WAVE SPACE!!!!!
  //brownian single particle//Brownian_FarField_RNG_Particle_kernel<<<grid, threads>>>(d_psi,  //output
  //brownian single particle//							   group_size,
  //brownian single particle//							   d_group_members,
  //brownian single particle//							   bro_data->seed_ff_rs );

  //zhoge: use cuRand to generate Gaussian variables
  curandGenerator_t gen;
  float *d_gauss = (work_data->bro_gauss);
  unsigned int N_random = 12*group_size + 6*ker_data->NxNyNz;     //12*group_size because d_psi needs 3 Scalar4 per particle, 6*NxNyNz because 3 complex values per grid point
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);       //the default generator (xorwow)
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);       //too slow
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);        //fast
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);      //slower than MTGP32
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);   //fastest      
  curandSetPseudoRandomGeneratorSeed(gen, bro_data->seed_ff_rs);  //set the seed
  curandGenerateNormal(gen, d_gauss, N_random, 0.0f, 1.0f);       //mean 0, std 1
  curandDestroyGenerator(gen);  

  //the first 12N floats goes to d_psi (this takes care of the type difference)
  cudaMemcpy(d_psi, d_gauss, 12*group_size*sizeof(float), cudaMemcpyDeviceToDevice);  


  
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
		
  // Apply random fluctuation to wave space grid (zhoge: only contains force, no torque or stress)
  Brownian_FarField_RNG_Grid1of2_kernel<<<gridNBlock,gridBlockSize>>>(mob_data->gridX,
  								      mob_data->gridY,
  								      mob_data->gridZ,
  								      mob_data->gridk,
  								      &d_gauss[12*group_size], //zhoge: pre-generated Gaussian
  								      ker_data->NxNyNz,
  								      mob_data->Nx,
  								      mob_data->Ny,
  								      mob_data->Nz,
  								      bro_data->seed_ff_ws,
  								      bro_data->T,
  								      dt,
  								      quadW);
  
  Brownian_FarField_RNG_Grid2of2_kernel<<<gridNBlock,gridBlockSize>>>(mob_data->gridX,  //output
  								      mob_data->gridY,	//output
  								      mob_data->gridZ,	//output
  								      mob_data->gridXX,	//output
  								      mob_data->gridXY,	//output
  								      mob_data->gridXZ,	//output
  								      mob_data->gridYX,	//output
  								      mob_data->gridYY,	//output
  								      mob_data->gridYZ,	//output
  								      mob_data->gridZX,	//output
  								      mob_data->gridZY,	//output
  								      mob_data->gridk,
  								      ker_data->NxNyNz,
  								      mob_data->Nx,
  								      mob_data->Ny,
  								      mob_data->Nz,
  								      bro_data->seed_ff_ws,
  								      bro_data->T,
  								      dt,
  								      quadW);
  		
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

  // Compute the square root
  Brownian_FarField_Lanczos(d_psi,  //input
  			    d_pos,  //input
  			    d_group_members,
  			    group_size,
  			    box,
  			    dt,
  			    d_Mreal12psi,  //output (11N vector of generalized velocity)
  			    bro_data->T,
  			    mob_data->xi,
  			    mob_data->ewald_cut,
  			    mob_data->ewald_dr,
  			    mob_data->ewald_n,
  			    mob_data->ewald_table, 
  			    mob_data->nneigh,
  			    mob_data->nlist,
  			    mob_data->headlist,
  			    bro_data->m_Lanczos_ff,  //input/output
  			    bro_data->tol,
  			    ker_data->particle_grid,
  			    ker_data->particle_threads,
  			    mob_data->gridh,
  			    mob_data->self,
  			    work_data);  //zhoge: work_data->mob_delu is modified inside (!!)
  		
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
  	

  // ****************
  // Rearrange Output
  // ****************
  Saddle_MakeGeneralizedU_kernel<<<grid,threads>>>(d_Uslip_ff, //output
						   d_vel,  
						   d_delu, //angular velocity and rate of strain
						   group_size);
	
  // ********
  // Clean Up
  // ********
  
  d_vel = NULL;
  d_delu = NULL;
  d_Mreal12psi = NULL;
  d_psi = NULL;

}
