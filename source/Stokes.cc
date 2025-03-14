// Modified by Andrew Fiore
// Modified by Zhouyang Ge

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

using namespace std;

#include "Stokes.h"
#include "Stokes.cuh"  //zhoge: This includes HOOMDMath.h, which includes cmath

#include "DataStruct.h"

#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <quadmath.h>

#include <cusparse.h>
#include <cusolverSp.h>

/*! \file Stokes.cc
    \brief Contains code for the Stokes class
*/

/*! 
	sysdef		SystemDefinition this method will act on. Must not be NULL.
        group		The group of particles this integration method is to work on
	T		temperature
	seed		Seed for random number generator
	nlist_ewald	neighbor list for Ewald calculation
	xi		Ewald parameter
	m_error		Error tolerance for all calculations
	fileprefix	prefix for output of stresslet data
	period		frequency of output of stresslet data
	ndsr            non-dimensional shear rate (interparticle force)
	kappa           inverse Debye length (electrostatic repulsion)
	k_n             collision spring constant
*/
Stokes::Stokes(std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<ParticleGroup> group,
	       std::shared_ptr<Variant> T,
	       unsigned int seed,
	       std::shared_ptr<NeighborList> nlist_ewald,
	       Scalar xi,
	       Scalar error,
	       std::string fileprefix,
	       int period,
	       Scalar ndsr, Scalar kappa, Scalar k_n, Scalar beta_AF, Scalar epsq, Scalar sqm_B1, Scalar sqm_B2,
	       unsigned int N_mix, Scalar coef_B1_mask, Scalar coef_B2_mask,
	       Scalar rot_diff, Scalar T_ext, Scalar omega_ext)
  : IntegrationMethodTwoStep(sysdef, group),
    m_T(T),
    m_seed(seed),
    m_nlist_ewald(nlist_ewald),
    m_xi(xi),
    m_error(error),
    m_fileprefix(fileprefix),
    m_period(period),
    m_ndsr(ndsr), m_kappa(kappa), m_k_n(k_n), m_beta(beta_AF), m_epsq(epsq), m_sqm_B1(sqm_B1),m_sqm_B2(sqm_B2),
    m_N_mix(N_mix), m_coef_B1_mask(coef_B1_mask),m_coef_B2_mask(coef_B2_mask),
    m_rot_diff(rot_diff), m_T_ext(T_ext),m_omega_ext(omega_ext)  //zhoge
    {
	m_exec_conf->msg->notice(5) << "Constructing Stokes" << endl;

	// Hash the User's Seed to make it less likely to be a low positive integer
	m_seed = m_seed * 0x12345677 + 0x12345; m_seed ^= (m_seed >> 16); m_seed *= 0x45679;

	// only one GPU is supported
	if (!m_exec_conf->isCUDAEnabled())
	{
		m_exec_conf->msg->error() << "Creating a Stokes when CUDA is disabled" << endl;
		throw std::runtime_error("Error initializing Stokes");
	}

    }

//! Destructor for the Stokes class
Stokes::~Stokes()
    {
	// Print out
   	m_exec_conf->msg->notice(5) << "Destroying Stokes" << endl;

	// Clean up cuFFT plan
	cufftDestroy(plan);

	// Clean up cuSOLVER handle
	cusolverSpDestroy(soHandle);

	// Clean up cuSPARSE handle and descriptions
	cusparseDestroy(spHandle);

	cusparseDestroyMatDescr(descr_R);
	cusparseDestroyMatDescr(descr_L);

	cusparseDestroyCsric02Info(info_R);
	cusparseDestroyCsrsv2Info(info_L);
	cusparseDestroyCsrsv2Info(info_Lt);
	
	// Clean up cuBLAS handle
	cublasDestroy( blasHandle );
	
	// Free workspace
	FreeWorkSpaces();

    }

/*!
	Set the parameters for Spectral Ewald Method
*/
void Stokes::setParams()
{

	// Try two Lanczos iterations to start (number of iterations will adapt as needed)
	m_m_Lanczos_ff = 2;
	m_m_Lanczos_nf = 2;

	m_rfd_epsilon = m_error;

	// At first only need to add identity, then increase if necessary  (used in Precondition.cu for the Cholesky decomposition)
	m_ichol_relaxer = 1.0;

	// Real space cutoff
	m_ewald_cut = sqrtf( - logf( m_error ) ) / m_xi;
	
	// Number of grid points
	int kmax = int( 2.0 * sqrtf( - logf( m_error ) ) * m_xi ) + 1;
	
	const BoxDim& box = m_pdata->getBox(); // Only for box not changing with time.
	Scalar3 L = box.getL();

	// Check that rcut is not too large (otherwise interact with images)
	if ( ( m_ewald_cut > L.x/2.0 ) || ( m_ewald_cut > L.y/2.0 ) || ( m_ewald_cut > L.z/2.0 ) ){
		
		float max_cut;
		if ( ( L.x < L.y ) && ( L.x < L.z ) ){
			max_cut = L.x / 2.0;
		}
		else if ( ( L.y < L.x ) && ( L.y < L.z ) ){
			max_cut = L.y / 2.0;
		}
		else if ( ( L.z < L.x ) && ( L.z < L.y ) ){
			max_cut = L.z / 2.0;
		}
		else {
			max_cut = L.x / 2.0;
		}

		float new_xi = sqrtf( -logf( m_error ) ) / max_cut;

		printf("Real space Ewald cutoff radius is too large! \n");
		printf("    xi = %f \n    rcut = %f \n    box = ( %f %f %f ) \n", m_xi, m_ewald_cut, L.x, L.y, L.z );
		printf("Increase xi to %f or larger to fix. \n", new_xi );
		
		exit(EXIT_FAILURE);

	}
	// initially, at least two points for the smallest wave length (modified to be multiples of 2,3,5 later)
	m_Nx = int( kmax * L.x / ( 2.0 * 3.1415926536 ) * 2.0 ) + 1; 
	m_Ny = int( kmax * L.y / ( 2.0 * 3.1415926536 ) * 2.0 ) + 1; 
	m_Nz = int( kmax * L.z / ( 2.0 * 3.1415926536 ) * 2.0 ) + 1; 
	
	// Get list of int values between 8 and 512 that can be written as
	// 	(2^a)*(3^b)*(5^c)
	// Then sort list from low to high and figure out how many entries there are
	std::vector<int> Mlist;
	for ( int ii = 0; ii < 10; ++ii ){
		int pow2 = 1;
		for ( int i = 0; i < ii; ++i ){
			pow2 *= 2;
		}
		for ( int jj = 0; jj < 6; ++jj ){
			int pow3 = 1;
			for ( int j = 0; j < jj; ++j ){
				pow3 *= 3;
			}
			for ( int kk = 0; kk < 4; ++kk ){
				int pow5 = 1;
				for ( int k = 0; k < kk; ++k ){
					pow5 *= 5;
				}
				int Mcurr = pow2 * pow3 * pow5;
				if ( Mcurr >= 8 && Mcurr <= 512 ){
					Mlist.push_back(Mcurr);
				}
			}
		}
	}
	std::sort(Mlist.begin(),Mlist.end());
	const int nmult = Mlist.size();

	// Compute the number of grid points in each direction
	//
	// Number of grid points should be a power of 2,3,5 for most efficient FFTs
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nx <= Mlist[ii]){
			 m_Nx = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Ny <= Mlist[ii]){
			m_Ny = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nz <= Mlist[ii]){
			m_Nz = Mlist[ii];
			break;
		}
	}

	// Maximum number of FFT nodes is limited by available memory
	// Max Number = 512 * 512 * 512 = 134,217,728
	if ( m_Nx * m_Ny * m_Nz > 512*512*512 ){

		printf("Requested Number of Fourier Nodes Exceeds Max Dimension of 512^3\n");
		printf("Mx = %i \n", m_Nx);
		printf("My = %i \n", m_Ny);
		printf("Mz = %i \n", m_Nz);

		exit(EXIT_FAILURE);
	}

        // Maximum eigenvalue of A'*A to scale support, P, for spreading on 
	// deformed grids (Fiore and Swan, J. Chem. Phys., 2018)
        Scalar gamma = m_max_strain;
        Scalar gamma2 = gamma*gamma;
        Scalar lambda = 1.0 + gamma2/2.0 + gamma*sqrtf(1.0 + gamma2/4.0);

	// Grid spacing
	m_gridh = L / make_scalar3(m_Nx,m_Ny,m_Nz); 

	// Parameters for the Spectral Ewald Method (Lindbo and Tornberg, J. Comp. Phys., 2011)
	m_gaussm = 1.0;
	while ( erfcf( m_gaussm / sqrtf(2.0*lambda) ) > m_error ){
	    m_gaussm = m_gaussm + 0.01;
	}
	m_gaussP = int( m_gaussm*m_gaussm / 3.1415926536 )  + 1;

	Scalar w = m_gaussP*m_gridh.x / 2.0;	               // Gaussian width in simulation units
	Scalar xisq  = m_xi * m_xi;
	m_eta = (2.0*w/m_gaussm)*(2.0*w/m_gaussm) * ( xisq );  // Gaussian splitting parameter	

	// Check that the support size isn't somehow larger than the grid
	if ( m_gaussP > std::min( m_Nx, std::min( m_Ny, m_Nz ) ) ){

		printf("Quadrature Support Exceeds Available Grid\n");
		printf("( Mx, My, Mz ) = ( %i, %i, %i ) \n", m_Nx, m_Ny, m_Nz);
		printf("Support Size, P = %i \n", m_gaussP);

		exit(EXIT_FAILURE);
	}

	// Print summary to command line output
	printf("\n");
	printf("\n");
	m_exec_conf->msg->notice(2) << "--- NUFFT Hydrodynamics Statistics ---" << endl;
	m_exec_conf->msg->notice(2) << "Mx: " << m_Nx << endl;
	m_exec_conf->msg->notice(2) << "My: " << m_Ny << endl;
	m_exec_conf->msg->notice(2) << "Mz: " << m_Nz << endl;
	m_exec_conf->msg->notice(2) << "rcut: " << m_ewald_cut << endl;	
	m_exec_conf->msg->notice(2) << "Points per radius (x,y,z): " << m_Nx / L.x << ", " << m_Ny / L.y << ", " << m_Nz / L.z << endl;
	m_exec_conf->msg->notice(2) << "--- Gaussian Spreading Parameters ---"  << endl;
	m_exec_conf->msg->notice(2) << "gauss_m: " << m_gaussm << endl;
        m_exec_conf->msg->notice(2) << "gauss_P: " << m_gaussP << endl;
	m_exec_conf->msg->notice(2) << "gauss_eta: " << m_eta << endl; 
	m_exec_conf->msg->notice(2) << "gauss_w: " << w << endl; 
	m_exec_conf->msg->notice(2) << "gauss_gridh (x,y,z): " << L.x/m_Nx << ", " << L.y/m_Ny << ", " << L.z/m_Nz << endl;
	printf("\n");
	printf("\n");

	// Create plan for CUFFT on the GPU
	cufftPlan3d(&plan, m_Nx, m_Ny, m_Nz, CUFFT_C2C);

	// Prepare GPUArrays for grid vectors and gridded forces
	GPUArray<Scalar4> n_gridk(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridk.swap(n_gridk);
	GPUArray<CUFFTCOMPLEX> n_gridX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridX.swap(n_gridX);
	GPUArray<CUFFTCOMPLEX> n_gridY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridY.swap(n_gridY);
	GPUArray<CUFFTCOMPLEX> n_gridZ(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridZ.swap(n_gridZ);
	
	GPUArray<CUFFTCOMPLEX> n_gridXX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridXX.swap(n_gridXX);
	GPUArray<CUFFTCOMPLEX> n_gridXY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridXY.swap(n_gridXY);
	GPUArray<CUFFTCOMPLEX> n_gridXZ(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridXZ.swap(n_gridXZ);
	GPUArray<CUFFTCOMPLEX> n_gridYX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridYX.swap(n_gridYX);
	GPUArray<CUFFTCOMPLEX> n_gridYY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridYY.swap(n_gridYY);
	GPUArray<CUFFTCOMPLEX> n_gridYZ(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridYZ.swap(n_gridYZ);
	GPUArray<CUFFTCOMPLEX> n_gridZX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridZX.swap(n_gridZX);
	GPUArray<CUFFTCOMPLEX> n_gridZY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridZY.swap(n_gridZY);

	// Get list of reciprocal space vectors, and scaling factor for the wave space calculation at each grid point
	ArrayHandle<Scalar4> h_gridk(m_gridk, access_location::host, access_mode::readwrite);
	for (int i = 0; i < m_Nx; i++) {
	  for (int j = 0; j < m_Ny; j++) {
	    for (int k = 0; k < m_Nz; k++) {

	      // Index into grid vector storage array
	      int idx = i * m_Ny*m_Nz + j * m_Nz + k;

	      // k goes from -N/2 to N/2
	      h_gridk.data[idx].x = 2.0*3.1415926536 * ((i < ( m_Nx + 1 ) / 2) ? i : i - m_Nx) / L.x;
	      h_gridk.data[idx].y = 2.0*3.1415926536 * ((j < ( m_Ny + 1 ) / 2) ? j : j - m_Ny) / L.y;
	      h_gridk.data[idx].z = 2.0*3.1415926536 * ((k < ( m_Nz + 1 ) / 2) ? k : k - m_Nz) / L.z;

	      // k dot k
	      Scalar k2 =
		h_gridk.data[idx].x*h_gridk.data[idx].x +
		h_gridk.data[idx].y*h_gridk.data[idx].y +
		h_gridk.data[idx].z*h_gridk.data[idx].z;

	      // Scaling factor used in wave space sum
	      //
	      // Can't include k=0 term in the Ewald sum
	      if (i == 0 && j == 0 && k == 0){ h_gridk.data[idx].w = 0;}
	      else
		{	// Have to divide by Nx*Ny*Nz to normalize the FFTs
		  h_gridk.data[idx].w = 6.0*3.1415926536 * (1.0 + k2/4.0/xisq) *
		    expf( -(1-m_eta) * k2/4.0/xisq ) / ( k2 ) / Scalar( m_Nx*m_Ny*m_Nz );
		}
	      
	    }
	  }
	}

	// Store the coefficients for the real space part of Ewald summation
	//
	// Will precompute scaling factors for real space component of summation for a given
	//     discretization to speed up GPU calculations
	//
	// NOTE: Due to the potential sensitivity of the real space functions at smaller xi, the
	//       tabulation will be computed in quadruple precision, then truncated and stored
	//       in single precision
	m_ewald_dr = 0.001; 		           // Distance resolution
	m_ewald_n = m_ewald_cut / m_ewald_dr - 1;  // Number of entries in tabulation

	// Table discretization in quadruple precision
	__float128 dr = 0.00100000000000000000000000000000;

	// Factors needed to compute self contribution
        Scalar pi12 = 1.77245385091; // square root of pi
	Scalar pi = 3.1415926536;    // pi
        Scalar aa = 1.0;  	     // radius
	Scalar axi = aa * m_xi;      // a * xi
	Scalar axi2 = axi * axi;     // ( a * xi )^2

	// Compute self contribution
        m_self.x = (1. + 4.*pi12*axi*erfcf(2.*axi) - expf(-4.*axi2))/(4.*pi12*axi*aa);
	m_self.y = ( (-3.*erfcf(2.*aa*m_xi)*powf(aa,-3.))/10. - (3.*powf(aa,-6.)*powf(pi,-0.5)*powf(m_xi,-3.))/80. -
		     (9.*powf(aa,-4.)*powf(pi,-0.5)*powf(m_xi,-1.))/40. +
		     (3.*expf(-4.*powf(aa,2.)*powf(m_xi,2.))*powf(aa,-6.)*powf(pi,-0.5)*powf(m_xi,-3.)*
		      (1. + 10.*powf(aa,2.)*powf(m_xi,2.)))/80. ); 

	// Allocate storage for real space Ewald table
	int nR = m_ewald_n + 1; // number of entries in ewald table
	GPUArray<Scalar4> n_ewaldC1( 2*nR, m_exec_conf); 
	m_ewaldC1.swap(n_ewaldC1);
	//zhoge//ArrayHandle<Scalar4> h_ewaldC1(m_ewaldC1, access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar4> h_ewaldC1(m_ewaldC1, access_location::host, access_mode::overwrite); // GPUdebug

	// Functions are complicated so calculation should be done in quadruple precision, then truncated to single precision
	// in order to ensure accurate evaluation
	__float128 xi  = m_xi;
	__float128 Pi = 3.1415926535897932384626433832795;
	__float128 a = aa;

	// Fill tables
	for ( int kk = 0; kk < nR; kk++ ) 
	{

		// Initialize entries
		h_ewaldC1.data[ 2*kk ].x = 0.0; // UF1
		h_ewaldC1.data[ 2*kk ].y = 0.0; // UF2
		h_ewaldC1.data[ 2*kk ].z = 0.0; // UC1
		h_ewaldC1.data[ 2*kk ].w = 0.0; // UC2 
		h_ewaldC1.data[ 2*kk + 1 ].x = 0.0; // DC1
		h_ewaldC1.data[ 2*kk + 1 ].y = 0.0; // DC2
		h_ewaldC1.data[ 2*kk + 1 ].z = 0.0; // DC3
		h_ewaldC1.data[ 2*kk + 1 ].w = 0.0; // extra 

		// Distance for current entry
		__float128 r = __float128( kk ) * dr + dr;
		__float128 Imrr = 0.00000000000000000000000000000000;
		__float128 rr   = 0.00000000000000000000000000000000;
		__float128 g1   = 0.00000000000000000000000000000000;
		__float128 g2   = 0.00000000000000000000000000000000;
		__float128 h1   = 0.00000000000000000000000000000000;
		__float128 h2   = 0.00000000000000000000000000000000;
		__float128 h3   = 0.00000000000000000000000000000000;
		

		// Expression have been simplified assuming no overlap, touching, and overlap
		if ( r > 2.0*a ){

		  Imrr = -powq(a,-1) + (powq(a,2)*powq(r,-3))/2. + (3*powq(r,-1))/4. + (3*erfcq(r*xi)*powq(a,-2)*powq(r,-3)*(-12*powq(r,4) + powq(xi,-4)))/128. + 
		    powq(a,-2)*((9*r)/32. - (3*powq(r,-3)*powq(xi,-4))/128.) + 
		    (erfcq((2*a + r)*xi)*(128*powq(a,-1) + 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(36*r - 3*powq(r,-3)*powq(xi,-4))))/256. + 
		    (erfcq(2*a*xi - r*xi)*(128*powq(a,-1) - 64*powq(a,2)*powq(r,-3) - 96*powq(r,-1) + powq(a,-2)*(-36*r + 3*powq(r,-3)*powq(xi,-4))))/
		    256. + (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-2)*powq(xi,-3)*(1 + 6*powq(r,2)*powq(xi,2)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (8*r*powq(a,2)*powq(xi,2) - 16*powq(a,3)*powq(xi,2) + a*(2 - 28*powq(r,2)*powq(xi,2)) - 3*(r + 6*powq(r,3)*powq(xi,2))))/128. + 
		    (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (8*r*powq(a,2)*powq(xi,2) + 16*powq(a,3)*powq(xi,2) + a*(-2 + 28*powq(r,2)*powq(xi,2)) - 3*(r + 6*powq(r,3)*powq(xi,2))))/128.;

		  rr = -powq(a,-1) - powq(a,2)*powq(r,-3) + (3*powq(r,-1))/2. + (3*powq(a,-2)*powq(r,-3)*(4*powq(r,4) + powq(xi,-4)))/64. + 
		    (erfcq(2*a*xi - r*xi)*(64*powq(a,-1) + 64*powq(a,2)*powq(r,-3) - 96*powq(r,-1) + powq(a,-2)*(-12*r - 3*powq(r,-3)*powq(xi,-4))))/128. + 
		    (erfcq((2*a + r)*xi)*(64*powq(a,-1) - 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(12*r + 3*powq(r,-3)*powq(xi,-4))))/128. + 
		    (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-2)*powq(xi,-3)*(-1 + 2*powq(r,2)*powq(xi,2)))/32. - 
		    ((2*a + 3*r)*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (-1 - 8*a*r*powq(xi,2) + 8*powq(a,2)*powq(xi,2) + 2*powq(r,2)*powq(xi,2)))/64. + 
		    ((2*a - 3*r)*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (-1 + 8*a*r*powq(xi,2) + 8*powq(a,2)*powq(xi,2) + 2*powq(r,2)*powq(xi,2)))/64. - 
		    (3*erfcq(r*xi)*powq(a,-2)*powq(r,-3)*powq(xi,-4)*(1 + 4*powq(r,4)*powq(xi,4)))/64.;

		  g1 = (expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(9 + 15*powq(r,2)*powq(xi,2) - 30*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (18*a - 45*r - 3*(2*a + r)*(-16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) + 6*(2*a + r)*(-32*r*powq(a,3) + 32*powq(a,4) + 44*powq(a,2)*powq(r,2) - 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/
		    640. + (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
			    (-9*(2*a + 5*r) + 3*(2*a - r)*(16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) - 6*(2*a - r)*(32*r*powq(a,3) + 32*powq(a,4) + 44*powq(a,2)*powq(r,2) + 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/
		    640. + (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(3 + 3*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))/128. - 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 512*powq(a,6)*powq(xi,6) - 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/
		    1280. - (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 512*powq(a,6)*powq(xi,6) + 256*a*powq(r,5)*powq(xi,6) + 
										      100*powq(r,6)*powq(xi,6)))/1280.;

		  g2 = (-3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (18*a + 45*r - 3*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(xi,2) + 6*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(-2*a + r,2)*powq(xi,4)))/640. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (-18*a + 45*r + 3*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(xi,2) - 6*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(2*a + r,2)*powq(xi,4)))/640. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 - 15*powq(r,2)*powq(xi,2) + 4*(128*powq(a,6) - 80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) - 5*powq(r,6))*powq(xi,6)))/1280. + 
		    (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))/128. - 
		    (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-15 + 15*powq(r,2)*powq(xi,2) + 4*(-128*powq(a,6) + 80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) + 5*powq(r,6))*powq(xi,6)))/1280.;

		  h1 = (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
			(27 - 2*powq(xi,2)*(15*powq(r,2) + 2*powq(r,4)*powq(xi,2) - 4*powq(r,6)*powq(xi,4) + 48*powq(a,2)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))))/4096. + 
		    (3*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (270*a - 135*r + 6*(2*a + 5*r)*(12*powq(a,2) + 5*powq(r,2))*powq(xi,2) - 4*(144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) - 5*powq(r,5))*powq(xi,4) + 
		      8*powq(2*a - r,3)*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(xi,6)))/40960. + 
		    (3*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (-135*(2*a + r) - 6*(2*a - 5*r)*(12*powq(a,2) + 5*powq(r,2))*powq(xi,2) + 4*(-144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) + 5*powq(r,5))*powq(xi,4) - 
		      8*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,3)*powq(xi,6)))/40960. + 
		    (3*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(27 + 8*powq(xi,2)*(-6*powq(r,2) + 9*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 
											 12*powq(a,2)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 
									       16*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(-2*a + r,4)*powq(xi,8)))/81920. + 
		    (3*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 
									      16*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,4)*powq(xi,8)))/81920.;

		  h2 = (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
			(-45 - 78*powq(r,2)*powq(xi,2) + 28*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + 19*powq(r,2)*powq(xi,2) + 10*powq(r,4)*powq(xi,4)) - 56*powq(r,6)*powq(xi,6)))/4096. + 
		    (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(2*a + r) + 6*(-20*r*powq(a,2) + 8*powq(a,3) + 46*a*powq(r,2) + 13*powq(r,3))*powq(xi,2) - 
		      4*(2*a + r)*(-32*r*powq(a,3) + 16*powq(a,4) + 48*powq(a,2)*powq(r,2) - 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) + 
		      8*(2*a + r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 7*powq(r,4))*powq(-2*a + r,2)*powq(xi,6)))/8192. + 
		    (9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(-2*a + r) - 6*(20*r*powq(a,2) + 8*powq(a,3) + 46*a*powq(r,2) - 13*powq(r,3))*powq(xi,2) + 
		      4*(2*a - r)*(32*r*powq(a,3) + 16*powq(a,4) + 48*powq(a,2)*powq(r,2) + 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) - 
		      8*(2*a - r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 7*powq(r,4))*powq(2*a + r,2)*powq(xi,6)))/8192. - 
		    (9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 6*powq(r,2) + 9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 
												   2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 40*powq(a,2)*powq(r,6) + 7*powq(r,8))*powq(xi,6))))/16384. - 
		    (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 6*powq(r,2) + 9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 
												  2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 40*powq(a,2)*powq(r,6) + 7*powq(r,8))*powq(xi,6))))/16384. - 
		    (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(45 + 8*powq(xi,2)*(6*powq(r,2) - 9*powq(r,4)*powq(xi,2) - 14*powq(r,8)*powq(xi,6) + 
											 4*powq(a,2)*(-15 - 9*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))))/8192.;

		  h3 = (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
			(-45 + 18*powq(r,2)*powq(xi,2) - 4*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + powq(r,2)*powq(xi,2) - 2*powq(r,4)*powq(xi,4)) + 8*powq(r,6)*powq(xi,6)))/4096. + 
		    (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(2*a + r) + 6*(2*a - 3*r)*powq(-2*a + r,2)*powq(xi,2) - 4*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 8*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(2*a + r,2)*powq(xi,6)))/8192.\
		    + (9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		       (45*(-2*a + r) - 6*(2*a + 3*r)*powq(2*a + r,2)*powq(xi,2) + 4*(4*powq(a,2) + powq(r,2))*powq(2*a + r,3)*powq(xi,4) - 8*(4*powq(a,2) + powq(r,2))*powq(-2*a + r,2)*powq(2*a + r,3)*powq(xi,6)))/8192.\
		    - (9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*
										 (60*powq(a,2) + 6*powq(r,2) - 3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. - 
		    (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 
												  2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. + 
		    (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(6*powq(r,2) - 3*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 4*powq(a,2)*(15 - 9*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/
		    8192.;
		}
		else if ( r == 2.0*a ){
				
		  Imrr = -(powq(a,-5)*(3 + 16*a*xi*powq(Pi,-0.5))*powq(xi,-4))/2048. + (3*erfcq(2*a*xi)*powq(a,-5)*(-192*powq(a,4) + powq(xi,-4)))/1024. + 
		    erfcq(4*a*xi)*(powq(a,-1) - (3*powq(a,-5)*powq(xi,-4))/2048.) + 
		    (expq(-16*powq(a,2)*powq(xi,2))*powq(a,-4)*powq(Pi,-0.5)*powq(xi,-3)*(-1 - 64*powq(a,2)*powq(xi,2)))/256. + 
		    (3*expq(-4*powq(a,2)*powq(xi,2))*powq(a,-4)*powq(Pi,-0.5)*powq(xi,-3)*(1 + 24*powq(a,2)*powq(xi,2)))/256.;

		  rr = (powq(a,-5)*(3 + 16*a*xi*powq(Pi,-0.5))*powq(xi,-4))/1024. + erfcq(2*a*xi)*((-3*powq(a,-1))/8. - (3*powq(a,-5)*powq(xi,-4))/512.) + 
		    erfcq(4*a*xi)*(powq(a,-1) + (3*powq(a,-5)*powq(xi,-4))/1024.) + 
		    (expq(-16*powq(a,2)*powq(xi,2))*powq(a,-4)*powq(Pi,-0.5)*powq(xi,-3)*(1 - 32*powq(a,2)*powq(xi,2)))/128. + 
		    (3*expq(-4*powq(a,2)*powq(xi,2))*powq(a,-4)*powq(Pi,-0.5)*powq(xi,-3)*(-1 + 8*powq(a,2)*powq(xi,2)))/128.;

		  g1 = (expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(9 + 15*powq(r,2)*powq(xi,2) - 30*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (18*a - 45*r - 3*(2*a + r)*(-16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) + 6*(2*a + r)*(-32*r*powq(a,3) + 32*powq(a,4) + 44*powq(a,2)*powq(r,2) - 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/
		    640. + (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
			    (-9*(2*a + 5*r) + 3*(2*a - r)*(16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) - 6*(2*a - r)*(32*r*powq(a,3) + 32*powq(a,4) + 44*powq(a,2)*powq(r,2) + 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/
		    640. + (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(3 + 3*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))/128. - 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 512*powq(a,6)*powq(xi,6) - 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/
		    1280. - (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 512*powq(a,6)*powq(xi,6) + 256*a*powq(r,5)*powq(xi,6) + 
										      100*powq(r,6)*powq(xi,6)))/1280.;

		  g2 = (-3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (18*a + 45*r - 3*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(xi,2) + 6*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(-2*a + r,2)*powq(xi,4)))/640. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (-18*a + 45*r + 3*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(xi,2) - 6*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(2*a + r,2)*powq(xi,4)))/640. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 - 15*powq(r,2)*powq(xi,2) + 4*(128*powq(a,6) - 80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) - 5*powq(r,6))*powq(xi,6)))/1280. + 
		    (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))/128. - 
		    (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-15 + 15*powq(r,2)*powq(xi,2) + 4*(-128*powq(a,6) + 80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) + 5*powq(r,6))*powq(xi,6)))/1280.;
		
		  h1 = (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
			(27 - 2*powq(xi,2)*(15*powq(r,2) + 2*powq(r,4)*powq(xi,2) - 4*powq(r,6)*powq(xi,4) + 48*powq(a,2)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))))/4096. + 
		    (3*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (270*a - 135*r + 6*(2*a + 5*r)*(12*powq(a,2) + 5*powq(r,2))*powq(xi,2) - 4*(144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) - 5*powq(r,5))*powq(xi,4) + 
		      8*powq(2*a - r,3)*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(xi,6)))/40960. + 
		    (3*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (-135*(2*a + r) - 6*(2*a - 5*r)*(12*powq(a,2) + 5*powq(r,2))*powq(xi,2) + 4*(-144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) + 5*powq(r,5))*powq(xi,4) - 
		      8*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,3)*powq(xi,6)))/40960. + 
		    (3*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(27 + 8*powq(xi,2)*(-6*powq(r,2) + 9*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 
											 12*powq(a,2)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 
									       16*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(-2*a + r,4)*powq(xi,8)))/81920. + 
		    (3*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 
									      16*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,4)*powq(xi,8)))/81920.;

		  h2 = (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
			(-45 - 78*powq(r,2)*powq(xi,2) + 28*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + 19*powq(r,2)*powq(xi,2) + 10*powq(r,4)*powq(xi,4)) - 56*powq(r,6)*powq(xi,6)))/4096. + 
		    (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(2*a + r) + 6*(-20*r*powq(a,2) + 8*powq(a,3) + 46*a*powq(r,2) + 13*powq(r,3))*powq(xi,2) - 
		      4*(2*a + r)*(-32*r*powq(a,3) + 16*powq(a,4) + 48*powq(a,2)*powq(r,2) - 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) + 
		      8*(2*a + r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 7*powq(r,4))*powq(-2*a + r,2)*powq(xi,6)))/8192. + 
		    (9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(-2*a + r) - 6*(20*r*powq(a,2) + 8*powq(a,3) + 46*a*powq(r,2) - 13*powq(r,3))*powq(xi,2) + 
		      4*(2*a - r)*(32*r*powq(a,3) + 16*powq(a,4) + 48*powq(a,2)*powq(r,2) + 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) - 
		      8*(2*a - r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 7*powq(r,4))*powq(2*a + r,2)*powq(xi,6)))/8192. - 
		    (9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 6*powq(r,2) + 9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 
												   2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 40*powq(a,2)*powq(r,6) + 7*powq(r,8))*powq(xi,6))))/16384. - 
		    (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 6*powq(r,2) + 9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 
												  2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 40*powq(a,2)*powq(r,6) + 7*powq(r,8))*powq(xi,6))))/16384. - 
		    (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(45 + 8*powq(xi,2)*(6*powq(r,2) - 9*powq(r,4)*powq(xi,2) - 14*powq(r,8)*powq(xi,6) + 
											 4*powq(a,2)*(-15 - 9*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))))/8192.;

		  h3 = (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
			(-45 + 18*powq(r,2)*powq(xi,2) - 4*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + powq(r,2)*powq(xi,2) - 2*powq(r,4)*powq(xi,4)) + 8*powq(r,6)*powq(xi,6)))/4096. + 
		    (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(2*a + r) + 6*(2*a - 3*r)*powq(-2*a + r,2)*powq(xi,2) - 4*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 8*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(2*a + r,2)*powq(xi,6)))/8192.\
		    + (9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		       (45*(-2*a + r) - 6*(2*a + 3*r)*powq(2*a + r,2)*powq(xi,2) + 4*(4*powq(a,2) + powq(r,2))*powq(2*a + r,3)*powq(xi,4) - 8*(4*powq(a,2) + powq(r,2))*powq(-2*a + r,2)*powq(2*a + r,3)*powq(xi,6)))/8192.\
		    - (9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*
										 (60*powq(a,2) + 6*powq(r,2) - 3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. - 
		    (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 
												  2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. + 
		    (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(6*powq(r,2) - 3*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 4*powq(a,2)*(15 - 9*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/
		    8192.;
		}
		else if ( r < 2*a){

		  Imrr = (-9*r*powq(a,-2))/32. + powq(a,-1) - (powq(a,2)*powq(r,-3))/2. - (3*powq(r,-1))/4. + 
		    (3*erfcq(r*xi)*powq(a,-2)*powq(r,-3)*(-12*powq(r,4) + powq(xi,-4)))/128. + 
		    (erfcq((-2*a + r)*xi)*(-128*powq(a,-1) + 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(36*r - 3*powq(r,-3)*powq(xi,-4))))/
		    256. + (erfcq((2*a + r)*xi)*(128*powq(a,-1) + 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(36*r - 3*powq(r,-3)*powq(xi,-4))))/
		    256. + (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-2)*powq(xi,-3)*(1 + 6*powq(r,2)*powq(xi,2)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (8*r*powq(a,2)*powq(xi,2) - 16*powq(a,3)*powq(xi,2) + a*(2 - 28*powq(r,2)*powq(xi,2)) - 3*(r + 6*powq(r,3)*powq(xi,2))))/128. + 
		    (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (8*r*powq(a,2)*powq(xi,2) + 16*powq(a,3)*powq(xi,2) + a*(-2 + 28*powq(r,2)*powq(xi,2)) - 3*(r + 6*powq(r,3)*powq(xi,2))))/128.;

		  rr = ((2*a + 3*r)*powq(a,-2)*powq(2*a - r,3)*powq(r,-3))/16. + 
		    (erfcq((-2*a + r)*xi)*(-64*powq(a,-1) - 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(12*r + 3*powq(r,-3)*powq(xi,-4))))/128. + 
		    (erfcq((2*a + r)*xi)*(64*powq(a,-1) - 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(12*r + 3*powq(r,-3)*powq(xi,-4))))/128. + 
		    (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-2)*powq(xi,-3)*(-1 + 2*powq(r,2)*powq(xi,2)))/32. - 
		    ((2*a + 3*r)*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (-1 - 8*a*r*powq(xi,2) + 8*powq(a,2)*powq(xi,2) + 2*powq(r,2)*powq(xi,2)))/64. + 
		    ((2*a - 3*r)*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (-1 + 8*a*r*powq(xi,2) + 8*powq(a,2)*powq(xi,2) + 2*powq(r,2)*powq(xi,2)))/64. - 
		    (3*erfcq(r*xi)*powq(a,-2)*powq(r,-3)*powq(xi,-4)*(1 + 4*powq(r,4)*powq(xi,4)))/64.;

		  g1 = (-9*powq(a,-4)*powq(r,-4)*powq(xi,-6))/128. - (9*powq(a,-4)*powq(r,-2)*powq(xi,-4))/128. + 
		    (expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(9 + 15*powq(r,2)*powq(xi,2) - 30*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (18*a - 45*r - 3*(2*a + r)*(-16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) + 6*(2*a + r)*(-32*r*powq(a,3) + 32*powq(a,4) + 44*powq(a,2)*powq(r,2) - 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/
		    640. + (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
			    (-9*(2*a + 5*r) + 3*(2*a - r)*(16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) - 6*(2*a - r)*(32*r*powq(a,3) + 32*powq(a,4) + 44*powq(a,2)*powq(r,2) + 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/
		    640. + (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(3 + 3*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))/128. + 
		    (3*erfcq(2*a*xi - r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 512*powq(a,6)*powq(xi,6) - 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/
		    1280. - (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 512*powq(a,6)*powq(xi,6) + 256*a*powq(r,5)*powq(xi,6) + 
										      100*powq(r,6)*powq(xi,6)))/1280.;

		  g2 = (-3*r*powq(a,-3))/10. - (12*powq(a,2)*powq(r,-4))/5. + (3*powq(r,-2))/2. + (3*powq(a,-4)*powq(r,2))/32. - 
		    (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (18*a + 45*r - 3*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(xi,2) + 6*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(-2*a + r,2)*powq(xi,4)))/640. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (-18*a + 45*r + 3*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(xi,2) - 6*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(2*a + r,2)*powq(xi,4)))/640. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 - 15*powq(r,2)*powq(xi,2) + 4*(128*powq(a,6) - 80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) - 5*powq(r,6))*powq(xi,6)))/1280. + 
		    (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))/128. - 
		    (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-15 + 15*powq(r,2)*powq(xi,2) + 4*(-128*powq(a,6) + 80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) + 5*powq(r,6))*powq(xi,6)))/1280.;

		  h1 = (9*r*powq(a,-4))/64. - (3*powq(a,-3))/10. - (9*powq(a,2)*powq(r,-5))/10. + (3*powq(r,-3))/4. - (3*powq(a,-6)*powq(r,3))/512. + 
		    (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
		     (27 - 2*powq(xi,2)*(15*powq(r,2) + 2*powq(r,4)*powq(xi,2) - 4*powq(r,6)*powq(xi,4) + 48*powq(a,2)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))))/4096. + 
		    (3*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (270*a - 135*r + 6*(2*a + 5*r)*(12*powq(a,2) + 5*powq(r,2))*powq(xi,2) - 4*(144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) - 5*powq(r,5))*powq(xi,4) + 
		      8*powq(2*a - r,3)*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(xi,6)))/40960. + 
		    (3*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (-135*(2*a + r) - 6*(2*a - 5*r)*(12*powq(a,2) + 5*powq(r,2))*powq(xi,2) + 4*(-144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) + 5*powq(r,5))*powq(xi,4) - 
		      8*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,3)*powq(xi,6)))/40960. + 
		    (3*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(27 + 8*powq(xi,2)*(-6*powq(r,2) + 9*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 
											 12*powq(a,2)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 
									       16*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(-2*a + r,4)*powq(xi,8)))/81920. + 
		    (3*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 
									      16*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,4)*powq(xi,8)))/81920.;

		  h2 = (63*r*powq(a,-4))/64. - (3*powq(a,-3))/2. + (9*powq(a,2)*powq(r,-5))/2. - (3*powq(r,-3))/4. - (33*powq(a,-6)*powq(r,3))/512. + (9*powq(a,-6)*powq(r,-3)*powq(xi,-6))/128. - 
		    (27*powq(a,-4)*powq(r,-3)*powq(xi,-4))/64. + (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
								  (-45 - 78*powq(r,2)*powq(xi,2) + 28*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + 19*powq(r,2)*powq(xi,2) + 10*powq(r,4)*powq(xi,4)) - 56*powq(r,6)*powq(xi,6)))/4096. + 
		    (3*erfcq(2*a*xi - r*xi)*powq(a,-6)*powq(r,-3)*powq(xi,-6)*(-3 + 18*powq(a,2)*powq(xi,2)*(1 - 4*powq(r,4)*powq(xi,4)) + 128*powq(a,6)*powq(xi,6) + 64*powq(a,3)*powq(r,3)*powq(xi,6) + 
									       8*powq(r,6)*powq(xi,6)))/256. + (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
														(45*(2*a + r) + 6*(-20*r*powq(a,2) + 8*powq(a,3) + 46*a*powq(r,2) + 13*powq(r,3))*powq(xi,2) - 
														 4*(2*a + r)*(-32*r*powq(a,3) + 16*powq(a,4) + 48*powq(a,2)*powq(r,2) - 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) + 
														 8*(2*a + r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 7*powq(r,4))*powq(-2*a + r,2)*powq(xi,6)))/8192. + 
		    (9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(-2*a + r) - 6*(20*r*powq(a,2) + 8*powq(a,3) + 46*a*powq(r,2) - 13*powq(r,3))*powq(xi,2) + 
		      4*(2*a - r)*(32*r*powq(a,3) + 16*powq(a,4) + 48*powq(a,2)*powq(r,2) + 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) - 
		      8*(2*a - r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 7*powq(r,4))*powq(2*a + r,2)*powq(xi,6)))/8192. - 
		    (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 6*powq(r,2) + 9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 
												  2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 40*powq(a,2)*powq(r,6) + 7*powq(r,8))*powq(xi,6))))/16384. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(135 + 8*powq(xi,2)*(-6*(30*powq(a,2) + powq(r,2)) + 9*(4*powq(a,2) - 3*powq(r,2))*powq(r,2)*powq(xi,2) + 
												   2*(-768*powq(a,8) + 128*powq(a,6)*powq(r,2) + 256*powq(a,3)*powq(r,5) - 168*powq(a,2)*powq(r,6) + 11*powq(r,8))*powq(xi,6))))/16384. - 
		    (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(45 + 8*powq(xi,2)*(6*powq(r,2) - 9*powq(r,4)*powq(xi,2) - 14*powq(r,8)*powq(xi,6) + 
											 4*powq(a,2)*(-15 - 9*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))))/8192.;

		  h3 = (9*r*powq(a,-4))/64. + (9*powq(a,2)*powq(r,-5))/2. - (9*powq(r,-3))/4. - (9*powq(a,-6)*powq(r,3))/512. + 
		    (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*
		     (-45 + 18*powq(r,2)*powq(xi,2) - 4*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + powq(r,2)*powq(xi,2) - 2*powq(r,4)*powq(xi,4)) + 8*powq(r,6)*powq(xi,6)))/4096. + 
		    (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(2*a + r) + 6*(2*a - 3*r)*powq(-2*a + r,2)*powq(xi,2) - 4*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 8*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(2*a + r,2)*powq(xi,6)))/8192.\
		    + (9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		       (45*(-2*a + r) - 6*(2*a + 3*r)*powq(2*a + r,2)*powq(xi,2) + 4*(4*powq(a,2) + powq(r,2))*powq(2*a + r,3)*powq(xi,4) - 8*(4*powq(a,2) + powq(r,2))*powq(-2*a + r,2)*powq(2*a + r,3)*powq(xi,6)))/8192.\
		    - (9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*
										 (60*powq(a,2) + 6*powq(r,2) - 3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. - 
		    (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 
												  2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. + 
		    (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(6*powq(r,2) - 3*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 4*powq(a,2)*(15 - 9*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/
		    8192.;

		}

		// Save values to table
		h_ewaldC1.data[ 2*kk ].x = Scalar( Imrr ); // UF1
		h_ewaldC1.data[ 2*kk ].y = Scalar( rr );   // UF2
		h_ewaldC1.data[ 2*kk ].z = Scalar( g1/2. );  // UC1
		h_ewaldC1.data[ 2*kk ].w = Scalar( -g2/2. ); // UC2
		h_ewaldC1.data[ 2*kk + 1 ].x = Scalar( h1 ); // DC1
		h_ewaldC1.data[ 2*kk + 1 ].y = Scalar( h2 ); // DC2
		h_ewaldC1.data[ 2*kk + 1 ].z = Scalar( h3 ); // DC3


	} // kk loop over distances

	// Applied forces/torques 
	// Particle linear/angular velocities, plus stresslet
	unsigned int group_size = m_group->getNumMembers();

	GPUArray<float> n_AppliedForce(6*group_size, m_exec_conf);
	GPUArray<float> n_Velocity(   11*group_size, m_exec_conf);
	GPUArray<float> n_sqm_B1_mask(   group_size, m_exec_conf);
	GPUArray<float> n_sqm_B2_mask(   group_size, m_exec_conf);
	GPUArray<Scalar3> n_noise_ang(   group_size, m_exec_conf);
 
	m_AppliedForce.swap(n_AppliedForce);
	m_Velocity.swap(n_Velocity);
	m_sqm_B1_mask.swap(n_sqm_B1_mask);
	m_sqm_B2_mask.swap(n_sqm_B2_mask);
	m_noise_ang.swap(n_noise_ang);

}

/*  
	Write quantities to file

	Modified from code written by Zach Sherman in his Immersed Boundary code
*/
void Stokes::OutputData( unsigned int timestep, BoxDim box, Scalar current_shear_rate ){

  Scalar volume = box.getVolume();
  unsigned int N = m_pdata->getN();	

  float nden  = N/volume;              //number density
  float sr    = 1.0;                   //(maximal) shear rate
  float F_0   = sr/m_ndsr;             //electrostatic repulsion scale
  float Hamaker = F_0*m_beta;          //Hamaker constant for vdW
  float epsq    = m_epsq;              //square of the regularization term for vdW
  //float F_0   = 0.001;               //Brady's repulsion factor
  //float r_c   = 0.001;               //Brady's repulsion range

  // Initial output
  if (timestep == 0) {
    std::cout << endl;
    std::cout << "--- INITIAL OUTPUT --- " << endl;
    std::cout << endl;
    std::cout << "box volume " << volume << endl;
    std::cout << "number density " << nden << endl;
    std::cout << "shear rate " << sr << endl;
    std::cout << "max repulsive force " << F_0 << endl;
    std::cout << "---------------------- " << endl;
    std::cout << endl;
  }

  // Access needed data from CPU
  ArrayHandle<Scalar4> h_pos(    m_pdata->getPositions(), access_location::host, access_mode::read);
  ArrayHandle<Scalar3> h_ori(m_pdata->getAccelerations(), access_location::host, access_mode::read);
  ArrayHandle<float>   h_Velocity(m_Velocity,             access_location::host, access_mode::read);
  ArrayHandle<Scalar4> h_pos_gb(m_pdata->getVelocities(), access_location::host, access_mode::read);

  ArrayHandle<unsigned int> h_index_array(   m_group->getIndexArray(),        access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_tag_array(     m_pdata->getTags(),              access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_nneigh_ewald(  m_nlist_ewald->getNNeighArray(), access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_nlist_ewald(   m_nlist_ewald->getNListArray(),  access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_headlist_ewald(m_nlist_ewald->getHeadList(),    access_location::host, access_mode::read);

  // Format the timestep to a string
  std::ostringstream timestep_str;
  timestep_str << std::setw(10) << std::setfill('0') << timestep;

  // Construct the filenames
  std::string filename0 = "raw/stresslet." + timestep_str.str() + ".txt";
  std::string filename1 = "raw/position." + timestep_str.str() + ".txt";
  std::string filename2 = "overlap.txt";
  std::string filename3 = "raw/interparticle_stresslet." + timestep_str.str() + ".txt";
  std::string filename4 = "raw/velocity." + timestep_str.str() + ".txt";
  
  // Open the files
  FILE * file0;
  FILE * file1;
  FILE * file2;
  FILE * file3;
  FILE * file4;
	
  file0 = fopen(filename0.c_str(), "w");	
  file1 = fopen(filename1.c_str(), "w");
  file2 = fopen(filename2.c_str(), "a");
  file3 = fopen(filename3.c_str(), "w");
  file4 = fopen(filename4.c_str(), "w");
  
  // init post-process results
  Scalar max_overlap = 0.0;
  Scalar avr_overlap = 0.0;
  float  cnt_overlap = 0.0;
  float  min_gap     = 10.0;

  float velx,vely,velz,omgx,omgy,omgz;
    
  // Loop through particle indices/tags and write per-particle result to file
  for (unsigned int ii = 0; ii < N; ii++) {

    // Get the particle's global index in data arrays (idx) and ID (tag)
    unsigned int idx = h_index_array.data[ii];
    unsigned int tag = h_tag_array.data[idx];

    // position and orientation
    Scalar4 pos4 = h_pos.data[idx];
    Scalar3 ori  = h_ori.data[idx];
    Scalar4 pos5 = h_pos_gb.data[idx];
  
    // velocity
    float * vel_p = & h_Velocity.data[ 6*idx ];  
    velx = vel_p[0];
    vely = vel_p[1];
    velz = vel_p[2];
    omgx = vel_p[3];
    omgy = vel_p[4];
    omgz = vel_p[5];
    
    // stresslet
    float * stlt = & h_Velocity.data[ 6*N + 5*idx ];
    
    float stress_xx = nden * stlt[0];
    float stress_xy = nden * stlt[1];
    float stress_xz = nden * stlt[2];
    float stress_yz = nden * stlt[3];  //zhoge: Note, this is yz, not yy.
    float stress_yy = nden * stlt[4];
    float stress_zz = -stress_xx-stress_yy;  //By definition, stlt is traceless.    

    
    // Output the hydrodynamic stresslets	
    fprintf (file0, "%7i %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e \n", tag,
    	     stress_xx, stress_xy, stress_xz, stress_yy, stress_yz, stress_zz);

    // Output the position and orientation
    fprintf (file1, "%7i %15.6e %15.6e %15.6e %12.3e %12.3e %12.3e %15.6e %15.6e %15.6e \n",
	     tag, pos4.x, pos4.y, pos4.z, ori.x, ori.y, ori.z, pos5.x, pos5.y, pos5.z);

    // Output the translational and angular velocities
    fprintf (file4, "%7i %15.6e %15.6e %15.6e %12.3e %12.3e %12.3e \n",
	     tag, velx, vely, velz, omgx, omgy, omgz);
    
    float stress_repl_xx = 0.0;
    float stress_repl_xy = 0.0;
    float stress_repl_xz = 0.0;
    float stress_repl_yy = 0.0;
    float stress_repl_yz = 0.0;
    float stress_repl_zz = 0.0;

    // Neighborlist arrays
    unsigned int head_idx = h_headlist_ewald.data[ idx ]; // Location in head array for neighbors of current particle
    unsigned int n_neigh  = h_nneigh_ewald.data[ idx ];   // Number of neighbors of the nearest particle

    
    for (unsigned int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {

      // Get the current neighbor index
      unsigned int curr_neigh = h_nlist_ewald.data[ head_idx + neigh_idx ];
		  
      Scalar4 posj = h_pos.data[curr_neigh];  // position
      Scalar3 R = make_scalar3( posj.x - pos4.x, posj.y - pos4.y, posj.z - pos4.z );  // distance vector
      R = box.minImage(R);  //periodic BC
      Scalar  distSqr = dot(R,R);          // Distance magnitude
      Scalar  dist    = sqrtf( distSqr );  // Distance
      Scalar  gap = dist - 2.0;
		  
      if (gap < 0.0)  // register overlap
	{
	  Scalar overlap = - gap;
	  if (overlap > max_overlap) {max_overlap = overlap;}
	  avr_overlap += overlap;
	  cnt_overlap += 1.0;
	  min_gap = 0.0;
	  //debug
	  //std::cout << timestep << "," << idx << "," << curr_neigh << "," << gap << std::endl;

	  //debug
	  if (overlap > 0.1)
	    {
	      printf("Large overlap: %12.3e \n", overlap);
	      printf("Center particle position: %12.3e %12.3e %12.3e \n", pos4.x, pos4.y, pos4.z);
	      printf("Neighb particle position: %12.3e %12.3e %12.3e \n", posj.x, posj.y, posj.z);
	      printf("Distance vector: %12.3e %12.3e %12.3e \n", R.x, R.y, R.z);
	      printf("Program aborted. \n");
	      exit(1);
	    }
	}
      else
	{
	  if (gap < min_gap) {min_gap = gap;}
	}

      //
      // Compute stress tensor contribution from interparticle forces
      //
      if (dist <= 2.0 + 10.0/m_kappa) {
      	
      	float normalx = R.x/dist;  //from center to neighbor
      	float normaly = R.y/dist;  //from center to neighbor
      	float normalz = R.z/dist;  //from center to neighbor
      
      	float rmax = 2.0 + 1.0*sqrt(epsq);  //Below rmax, collision force is activated to model surface roughness.
      	float gap1 = dist - rmax;  //surface gap for interparticle force calculations
      	
      	if (gap1 >= 0.) {
      	  
      	  float F_app_mag = -F_0 * expf (-gap1*m_kappa);  //electrostatic repulsion, negative
      	  F_app_mag += Hamaker/(12.*(gap1*gap1 + epsq));  //van der Waals attraction, positive
      	  //float F_app_mag = F_0/r_c*exp( -(dist-2.0)/r_c )/(1.0-exp( -(dist-2.0)/r_c ));  //always positive  //brady
      	  
      	  stress_repl_xx += nden * (F_app_mag * normalx * R.x)/2.0;  //divide by 2 because pair  
      	  stress_repl_xy += nden * (F_app_mag * normalx * R.y)/2.0;  
      	  stress_repl_xz += nden * (F_app_mag * normalx * R.z)/2.0;  
      	  stress_repl_yy += nden * (F_app_mag * normaly * R.y)/2.0;  
      	  stress_repl_yz += nden * (F_app_mag * normaly * R.z)/2.0;  
      	  stress_repl_zz += nden * (F_app_mag * normalz * R.z)/2.0;  
      	}
      	else {
      
      	  float F_app_mag = Hamaker/(12.*epsq) - F_0 - m_k_n * abs(gap1); //net force at gap1=0 minus collision
      
      	  stress_repl_xx += nden * (F_app_mag * normalx * R.x)/2.0;  //divide by 2 because pair
      	  stress_repl_xy += nden * (F_app_mag * normalx * R.y)/2.0;  
      	  stress_repl_xz += nden * (F_app_mag * normalx * R.z)/2.0;  
      	  stress_repl_yy += nden * (F_app_mag * normaly * R.y)/2.0;  
      	  stress_repl_yz += nden * (F_app_mag * normaly * R.z)/2.0;  
      	  stress_repl_zz += nden * (F_app_mag * normalz * R.z)/2.0;
      	  
      	}
      	
      } //interparticle
      
    } //neighbor particle

    // Output the interparticle stresslets	
    fprintf (file3, "%7i %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e \n", tag,
    	     stress_repl_xx, stress_repl_xy, stress_repl_xz, stress_repl_yy, stress_repl_yz, stress_repl_zz);

  } //center particle
  
  if (cnt_overlap > 0.0)
    {
      avr_overlap /= cnt_overlap;
      cnt_overlap /= 2.0; //count pairs (ij and ji are the same pair)
    }			    
  // Output overlap stats
  fprintf (file2, "%9i %8.0f %12.3e %12.3e  %12.3e \n", timestep, cnt_overlap, max_overlap, avr_overlap, min_gap);
  
  
  // Close output files
  fclose (file0);
  fclose (file1);
  fclose (file2);
  fclose (file3);
  fclose (file4);
		
}


/*
	Allocate workspace variables
*/
void Stokes::AllocateWorkSpaces(){
	
	// Set up the arrays and memory for calculation work spaces
	// 
	// Total Memory required for the arrays declared in this function:
	// 	
	//	sizeof(float) = sizeof(int) = 4 bytes
	//	
	//	nnz = 468 * N 
	//	mmax = 100	
	//
	//	Variable		Length		Type
	//	--------		------		----
	//	dot_sum			512		float
	//	bro_ff_psi		3*N		float4
	//	bro_ff_UBreal		3*N		float4
	//	bro_ff_Tm		mmax		float
	//	bro_ff_v		3*N		float4
	//	bro_ff_vj		3*N		float4
	//	bro_ff_vjm1		3*N		float4
	//	bro_ff_Mvj 		3*N		float4
	//	bro_ff_V		3*mmax*N	float4
	//	bro_ff_UB_old		3*N		float4
	//	bro_ff_Mpsi	 	3*N		float4
	//	bro_nf_Tm		m_max 		float
	//	bro_nf_v		6*N		float
	//	bro_nf_V		(mmax+1)*6*N	float
	//	bro_nf_FB_old 		6*N	 	float
	//	bro_nf_psi 		6*N		float
	//	saddle_psi		6*N		float
	//	saddle_posPrime		N		float4
	//	saddle_rhs 		17*N		float
	//	saddle_solution 	17*N		float
	//	mob_couplet		2*N		float4
	//	mob_delu		2*N		float4
	//	mob_vel1		N		float4
	//	mob_vel2		N		float4
	//	mob_delu1		2*N		float4
	//	mob_delu2		2*N		float4
	//	mob_vel			N		float4
	//	mob_AngvelStrain	2*N		float4
	//	mob_net_force		N		float4
	//	mob_TorqueStress	2*N		float4
	//	precond_scratch 	N		int
	//	precond_map 		nnz		int
	//	precond_backup	 	nnz		float
	//
	//				2963*N+712 \approx 2963*N
	//
	//	Total size in bytes: 	11852 * N
	//	Total size in KB:	11.852 * N
	//	Total size in MB:	0.011852 * N
	//
	// Some examples for various numbers of particles
	//
	//	N	Size (MB)	Size (GB)
	//	---	---------	---------
	//	1E2	1.1852		0.0011852
	//	1E3	11.852		0.011852
	//	1E4	118.52		0.11852
	//	1E5	1185.2		1.1852
	//	1E6	11852		11.852
	
	// Get the number of particles
	unsigned int group_size = m_group->getNumMembers();

	// Dot product kernel specifications
	unsigned int thread_for_dot = 512; // Must be 2^n
	unsigned int grid_for_dot = (group_size/thread_for_dot) + 1;

	// Maximum number of iterations in the Lanczos method
	int m_max = 100;

	// Maximum number of non-zero enries
	int nnz = m_nnz;
	
	// Dot product partial sum
	cudaMalloc( (void**)&dot_sum, grid_for_dot*sizeof(Scalar) );

	//zhoge: 11N for the real space noise, 6NxNyNz for the wave sapce noise
	cudaMalloc( (void**)&m_work_bro_gauss,		(11*group_size + 6*m_Nx*m_Ny*m_Nz) * sizeof(float) );  

	// Variables for far-field Lanczos iteration
	cudaMalloc( (void**)&m_work_bro_ff_psi,		3*group_size * sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_bro_ff_UBreal,	3*group_size * sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_bro_ff_Mpsi, 	3*group_size * sizeof(Scalar4) );
	//zhoge: change to simple float/Scalar for the far-field Chow & Saad
	cudaMalloc( (void**)&m_work_bro_ff_V1,		(m_max+1) * 11 * group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_ff_UB_new1,	            11 * group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_ff_UB_old1,	            11 * group_size * sizeof(Scalar) );

	//zhoge: RFD storage (Brownian drift)
	cudaMalloc( (void**)&m_work_rfd_rhs, 17*group_size*sizeof(float) );
	cudaMalloc( (void**)&m_work_rfd_sol, 17*group_size*sizeof(float) );

	//zhoge: RK2 midstep storage
	cudaMalloc( (void**)&m_work_pos_rk1, group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_ori_rk1, group_size*sizeof(Scalar3) );


	// Variables for near-field Lanczos iteration	
	cudaMalloc( (void**)&m_work_bro_nf_Tm,		m_max * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_nf_V,		(m_max+1) * 6*group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_nf_FB_old, 	6*group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_nf_psi, 	6*group_size*sizeof(float) );

	cudaMalloc( (void**)&m_work_saddle_psi,		6*group_size*sizeof(float) );
	cudaMalloc( (void**)&m_work_saddle_posPrime,	group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_saddle_rhs, 	17*group_size*sizeof(float) );
	cudaMalloc( (void**)&m_work_saddle_solution, 	17*group_size*sizeof(float) );

	cudaMalloc( (void**)&m_work_mob_couplet,	2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_delu,		2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_vel1,		group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_vel2,		group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_delu1,		2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_delu2,		2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_vel,		group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_AngvelStrain,	2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_net_force,	group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_TorqueStress,	2*group_size*sizeof(Scalar4) );

	cudaMalloc( (void**)&m_work_precond_scratch, 	group_size*sizeof(int) );	
	cudaMalloc( (void**)&m_work_precond_map, 	nnz*sizeof(int) );
	cudaMalloc( (void**)&m_work_precond_backup, 	nnz*sizeof(Scalar) );

}


/*
	Free workspace variables
*/
void Stokes::FreeWorkSpaces(){
	
	// Dot product partial sum
	cudaFree( dot_sum );
	cudaFree( m_work_bro_gauss );  //zhoge

	// Variables for far-field Lanczos iteration	
	cudaFree( m_work_bro_ff_psi );
	cudaFree( m_work_bro_ff_UBreal );
	cudaFree( m_work_bro_ff_Mpsi );
	//zhoge
	cudaFree( m_work_bro_ff_V1 );
	cudaFree( m_work_bro_ff_UB_new1 );
	cudaFree( m_work_bro_ff_UB_old1 );

	cudaFree( m_work_rfd_rhs );
	cudaFree( m_work_rfd_sol );

	cudaFree( m_work_pos_rk1 );
	cudaFree( m_work_ori_rk1 );


	// Variables for near-field Lanczos iteration	
	cudaFree( m_work_bro_nf_Tm );
	cudaFree( m_work_bro_nf_V );
	cudaFree( m_work_bro_nf_FB_old );
	cudaFree( m_work_bro_nf_psi );

	cudaFree( m_work_saddle_psi );
	cudaFree( m_work_saddle_posPrime );
	cudaFree( m_work_saddle_rhs );
	cudaFree( m_work_saddle_solution );

	cudaFree( m_work_mob_couplet );
	cudaFree( m_work_mob_delu );
	cudaFree( m_work_mob_vel1 );
	cudaFree( m_work_mob_vel2 );
	cudaFree( m_work_mob_delu1 );
	cudaFree( m_work_mob_delu2 );
	cudaFree( m_work_mob_vel );
	cudaFree( m_work_mob_AngvelStrain );
	cudaFree( m_work_mob_net_force );
	cudaFree( m_work_mob_TorqueStress );

	cudaFree( m_work_precond_scratch );
	cudaFree( m_work_precond_map );
	cudaFree( m_work_precond_backup );
}

/*
	Modify entries in the resistance table by the specified friction type

	NOTE: This function must be called AFTER setResistanceTable()

	Friction table contains entries in the following order: (This is the order given in Stokes_ResistanceTable.cc)
		1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22
		XA11 XA12 YA11 YA12 YB11 YB12 XC11 XC12 YC11 YC12 XG11 XG12 YG11 YG12 YH11 YH12 XM11 XM12 YM11 YM12 ZM11 ZM12

	friction_type	string specifying type of friction to add
	h0 		Maximum distance for frictional contact
	alpha		list of strengths of frictional contact

*/
void Stokes::setFriction( std::string friction_type,
			  float h0, 
			  std::vector<float> &alpha) {

	// Get handles to the resistance data
    ArrayHandle<Scalar> h_ResTable_dist(m_ResTable_dist, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_ResTable_vals(m_ResTable_vals, access_location::host, access_mode::readwrite);

	// Add friction coefficients if necessary
	if ( friction_type.compare("NONE") == 0 ){
		return;
	}
	else if (friction_type.compare("TYPE1") == 0 ){

		// Loop over all distances in array
		for (int ii = 0; ii < 1000; ++ii )
		{
			
			// Current gap width
			float h = h_ResTable_dist.data[ii] - 2.0;

			// If current distance is less than frictional distance, add to it
			if ( h <= h0 ){

				// Powers of h and h0
				float h2 = h * h;

				float h02 = h0 * h0;
				float h03 = h0 * h02;

				// Friction coefficient
				float YA11 = alpha[0] * ( 2.0 / h03 * h2 - 3.0 / h02 * h + 1.0 / h );
				float YA12 = alpha[1] * ( 2.0 / h03 * h2 - 3.0 / h02 * h + 1.0 / h );
				float YB11 = alpha[2] * ( 2.0 / h03 * h2 - 3.0 / h02 * h + 1.0 / h );
				float YB12 = alpha[3] * ( 2.0 / h03 * h2 - 3.0 / h02 * h + 1.0 / h );
				float YC11 = alpha[4] * ( 2.0 / h03 * h2 - 3.0 / h02 * h + 1.0 / h );
				float YC12 = alpha[5] * ( 2.0 / h03 * h2 - 3.0 / h02 * h + 1.0 / h );

				// Add to arrays. Minus signs account for sign of coefficients (add in magnitude)
				int curr_offset = 22 * ii;

				h_ResTable_vals.data[ curr_offset + 3  ] += YA11;
				h_ResTable_vals.data[ curr_offset + 4  ] -= YA12;
				h_ResTable_vals.data[ curr_offset + 5  ] -= YB11;
				h_ResTable_vals.data[ curr_offset + 6  ] += YB12;
				h_ResTable_vals.data[ curr_offset + 9  ] += YC11;
				h_ResTable_vals.data[ curr_offset + 10 ] += YC12;

			}

		}

	}
	else {
		m_exec_conf->msg->error() << "Invalid Friction Type. Allowable types are: None, Type1" << endl;
		throw std::runtime_error("Error initializing Stokes");
	}


}

/* 
  Run the integration method.
  Particle positions and velocities are moved forward to timestep+1 
*/
void Stokes::integrateStepOne(unsigned int timestep)
{

  // profile this step
  if (m_prof)
    m_prof->push(m_exec_conf, "Stokes step 1 (no step 2)");
  
  // Consistency check
  unsigned int group_size = m_group->getNumMembers();
  assert(group_size <= m_pdata->getN());
  if (group_size == 0)
    return;
	
  BoxDim box = m_pdata->getBox();

  // Calculate the shear rate of the current timestep
  Scalar current_shear_rate = m_shear_func -> getShearRate(timestep);

  // Recompute neighbor lists ( if needed )	
  m_nlist_ewald->compute(timestep);

  
  // ****************
  // Initializations
  // ****************
  
  
  // Initialize particle (global) position and orientation on host
  std::random_device rd {};
  std::mt19937 gen {rd()};

  float randx,randy,randz;

  if (timestep == 0)
    {
      std::normal_distribution<float> gaussian {0.0, 1.0}; // mean=0, std=1	    

      ArrayHandle<Scalar4> h_pos0(    m_pdata->getPositions(), access_location::host, access_mode::readwrite); //may re-init
      ArrayHandle<Scalar3> h_ori0(m_pdata->getAccelerations(), access_location::host, access_mode::overwrite); //init orientation
      ArrayHandle<Scalar4> h_pos_gb0(m_pdata->getVelocities(), access_location::host, access_mode::overwrite); //init global pos

      
      //// read position and orientation from a file (if restart)
      //ifstream last_pos;
      //last_pos.open("restart_last.txt");
      //if(last_pos.fail()) // checks if file opended 
      //	{ 
      //	  printf("ERROR: Couldn't find the restart file.\n"); 
      //	  exit(1); 
      //	}
      //printf("Read position and orientation from `restart_last.txt`.\n");
      //unsigned int tag;
      //float posx,posy,posz, orix,oriy,oriz, pos_gbx,pos_gby,pos_gbz;

      
      for (unsigned int i=0; i<group_size; i++){
	
	// orientation uniformly distributed on a sphere
	randx = gaussian(gen);
	randy = gaussian(gen);
	randz = gaussian(gen);
	float randmag = sqrtf(randx*randx + randy*randy + randz*randz);
	h_ori0.data[i] = make_scalar3(randx/randmag, randy/randmag, randz/randmag);

	// orientation all aligned
	//h_ori0.data[i] = make_scalar3(1.,0.,0.);  
	//h_ori0.data[i] = make_scalar3(0.,1.,0.); 
	//h_ori0.data[i] = make_scalar3(0.,0.,1.);

	//// restart from the last time (tag may change, so be careful when postprocessing)
	////last_pos >> tag >> posx >> posy >> posz >> orix >> oriy >> oriz;
	//last_pos >> tag >> posx >> posy >> posz >> orix >> oriy >> oriz >> pos_gbx >> pos_gby >> pos_gbz;
	//h_pos0.data[i]    = make_scalar4(posx,    posy,    posz,    0.0);
	//h_ori0.data[i]    = make_scalar3(orix,    oriy,    oriz        );
	//h_pos_gb0.data[i] = make_scalar4(pos_gbx, pos_gby, pos_gbz, 0.0);

	// initialize global position from local position (start from scratch or if last pos_gb unavail)
	h_pos_gb0.data[i] = h_pos0.data[i];
	
      }
      ////mouad swim3
      //h_ori0.data[2] = make_scalar3(0.05,sqrtf(1.-2.*0.05*0.05),0.05);  
      //h_ori0.data[1] = make_scalar3(7./8., sqrtf(1.-49./64.), 0.);  
      //h_ori0.data[0] = make_scalar3(9.987e-01, 0., 0.05);  
      //h_ori0.data[1] = make_scalar3(9.987e-01, 0.05, 0.);  
      //h_ori0.data[0] = make_scalar3(1., 0., 0.);

    }

	
  //// Generate a list (3N) of random variables for the rotational diffusion (noise)
  //// from an identical and independent Gaussian distribution
  ArrayHandle<Scalar3> h_noise_ang(m_noise_ang, access_location::host, access_mode::overwrite);

  float noise_mean, noise_stdd;
  noise_mean = 0.0;
  noise_stdd = sqrtf(2.0*m_rot_diff/m_deltaT);  // divide by dt because it is a velocity
	
  std::normal_distribution<float> noise {noise_mean, noise_stdd};
	
  for (unsigned int i=0; i<group_size; i++){

    randx = noise(gen);
    randy = noise(gen);
    randz = noise(gen);

    h_noise_ang.data[i] = make_scalar3(randx,randy,randz);
  }


  // Initialize squirmer masks on the host, at every timestep (index to tag mapping may change)
  ArrayHandle<unsigned int> h_index_array(m_group->getIndexArray(), access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_tag_array(m_pdata->getTags(),         access_location::host, access_mode::read);
  ArrayHandle<float>        h_sqm_B1_mask(m_sqm_B1_mask,            access_location::host, access_mode::overwrite); 
  ArrayHandle<float>        h_sqm_B2_mask(m_sqm_B2_mask,            access_location::host, access_mode::overwrite);
	
  //// Only uncomment the following if testing ishikawa
  //ArrayHandle<Scalar4>      h_pos(m_pdata->getPositions(), access_location::host, access_mode::overwrite); 	
  //float r_ishi = 2.5 + float(timestep)*0.5;  //center-to-center distance 
  //float theta_ishi = M_PI/4.;  //angular position of the passive particle from the axis of the active particle (0,1,0)
	
  for (unsigned int idx = 0; idx < group_size; idx++){
    unsigned int i = h_index_array.data[idx]; // this is an identical mapping (i.e. redundant)
    //// debug 2023-03-26
    if (i != idx){
      printf("particle index: %10i %10i\n", i, idx);
      printf("i and idx don't match. Exit the program.\n");
      exit(1);
    }
	  
    if (h_tag_array.data[i] < m_N_mix)
      {
	h_sqm_B1_mask.data[i] = 1.0;  // unchanged
	h_sqm_B2_mask.data[i] = 1.0;  // unchanged
	      
	//// Only uncomment the following if testing ishikawa
	//h_pos.data[i] = {0.0, 0.0, 0.0, 0.0}; //active particle at the origin
      }
    else
      {
	h_sqm_B1_mask.data[i] = m_coef_B1_mask;  // masked
	h_sqm_B2_mask.data[i] = m_coef_B2_mask;  // masked

	//// Only uncomment the following if testing ishikawa
	//h_pos.data[i] = {0.,              r_ishi,          0.0, 0.0}; // 0 deg in the xy plane
	//h_pos.data[i] = {r_ishi/sqrt(2.), r_ishi/sqrt(2.), 0.0, 0.0}; //45 deg in the xy plane
	//h_pos.data[i] = {r_ishi*sin(theta_ishi), r_ishi*cos(theta_ishi), 0.0, 0.0}; //note the difference from conventions
	
      }
  } //for loop

  
  // Initial output (before updating)
  if ( ( m_period > 0 ) && ( int(timestep) == 0 ) ) {  
    OutputData(int(timestep), box, current_shear_rate);
  }

  
  // *****************************
  // Get Handles to Device Arrays
  // *****************************

  
  // Neighbor lists
  ArrayHandle<unsigned int> d_nneigh_ewald(   m_nlist_ewald->getNNeighArray(), access_location::device, access_mode::read );
  ArrayHandle<unsigned int> d_nlist_ewald(    m_nlist_ewald->getNListArray(),  access_location::device, access_mode::read );
  ArrayHandle<unsigned int> d_headlist_ewald( m_nlist_ewald->getHeadList(),    access_location::device, access_mode::read );
	
  // Pruned neighbor list for lubrication preconditioner (constructed in Precondition_Wrap)
  ArrayHandle<unsigned int> d_nneigh_pruned(   m_nneigh_pruned,   access_location::device, access_mode::readwrite );
  ArrayHandle<unsigned int> d_nlist_pruned(    m_nlist_pruned,    access_location::device, access_mode::readwrite );
  ArrayHandle<unsigned int> d_headlist_pruned( m_headlist_pruned, access_location::device, access_mode::readwrite );

  ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite); 

  // Grid vectors
  ArrayHandle<Scalar4>      d_gridk(m_gridk, access_location::device, access_mode::readwrite); //write if deform
  ArrayHandle<CUFFTCOMPLEX> d_gridX(m_gridX, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridY(m_gridY, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridZ(m_gridZ, access_location::device, access_mode::read); 
	
  ArrayHandle<CUFFTCOMPLEX> d_gridXX(m_gridXX, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridXY(m_gridXY, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridXZ(m_gridXZ, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridYX(m_gridYX, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridYY(m_gridYY, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridYZ(m_gridYZ, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridZX(m_gridZX, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridZY(m_gridZY, access_location::device, access_mode::read); 

  // Real space interaction tabulation
  ArrayHandle<Scalar4> d_ewaldC1(m_ewaldC1, access_location::device, access_mode::read);

  // Lubrication calculation stuff
  ArrayHandle<int>   d_L_RowInd( m_L_RowInd, access_location::device, access_mode::overwrite ); 
  ArrayHandle<int>   d_L_RowPtr( m_L_RowPtr, access_location::device, access_mode::overwrite ); 
  ArrayHandle<int>   d_L_ColInd( m_L_ColInd, access_location::device, access_mode::overwrite ); 
  ArrayHandle<float> d_L_Val(    m_L_Val,    access_location::device, access_mode::overwrite ); 	
  ArrayHandle<float> d_Diag(     m_Diag,     access_location::device, access_mode::overwrite ); 
  ArrayHandle<int>   d_HasNeigh( m_HasNeigh, access_location::device, access_mode::overwrite ); 
	
  ArrayHandle<float> d_ResTable_dist( m_ResTable_dist, access_location::device, access_mode::read );
  ArrayHandle<float> d_ResTable_vals( m_ResTable_vals, access_location::device, access_mode::read );
	
  ArrayHandle<unsigned int> d_nneigh_less( m_nneigh_less, access_location::device, access_mode::overwrite ); 
  ArrayHandle<unsigned int> d_NEPP(        m_NEPP,        access_location::device, access_mode::overwrite ); 
  ArrayHandle<unsigned int> d_offset(      m_offset,      access_location::device, access_mode::overwrite ); 

  ArrayHandle<float> d_Scratch1( m_Scratch1, access_location::device, access_mode::overwrite ); 
  ArrayHandle<float> d_Scratch2( m_Scratch2, access_location::device, access_mode::overwrite ); 
  ArrayHandle<float> d_Scratch3( m_Scratch3, access_location::device, access_mode::overwrite ); 
  ArrayHandle<int>   d_prcm(     m_prcm,     access_location::device, access_mode::overwrite );   

  
  // Particle index (may change in time, unlike tag)
  ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);       

  // Particle position and orientation
  ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),     access_location::device, access_mode::readwrite ); 
  ArrayHandle<Scalar3> d_ori(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite ); //zhoge:temp
  ArrayHandle<Scalar4> d_pos_gb(m_pdata->getVelocities(), access_location::device, access_mode::readwrite ); //zhoge:temp 

  // Linear/angular velocities and applied force/torque
  ArrayHandle<float> d_Velocity(    m_Velocity,     access_location::device, access_mode::readwrite); 
  ArrayHandle<float> d_AppliedForce(m_AppliedForce, access_location::device, access_mode::readwrite); 

  // Rotational noise
  ArrayHandle<Scalar3> d_noise_ang(m_noise_ang, access_location::device, access_mode::read);
  
  // squirmer masks
  ArrayHandle<float> d_sqm_B1_mask(m_sqm_B1_mask, access_location::device, access_mode::read); 
  ArrayHandle<float> d_sqm_B2_mask(m_sqm_B2_mask, access_location::device, access_mode::read);

	
  // ***********************
  // Set up data structures
  // ***********************

  
  // Randomize seeds for stochastic calculations
  srand( m_seed + timestep );  //zhoge: identical sequence if m_seed is fixed
  m_seed_ff_rs = rand();
  m_seed_ff_ws = rand();
  m_seed_nf = rand();
  m_seed_rfd = rand();

  ////debug Brownian
  //printf("random seeds %12i %12i %12i %12i\n",m_seed_ff_rs,m_seed_ff_ws,m_seed_nf,m_seed_rfd);
  //exit(1);

  // Initialize values in the data structure for Brownian calculation
  BrownianData bro_struct = {
			     m_error,  //tol
			     timestep,
			     m_seed_ff_rs,
			     m_seed_ff_ws,
			     m_seed_nf,
			     m_seed_rfd,
			     m_m_Lanczos_ff,
			     m_m_Lanczos_nf,
			     float(m_T->getValue(timestep)),
			     m_rfd_epsilon,
			     m_work_rfd_rhs,
			     m_work_rfd_sol
  };
  BrownianData *bro_data = &bro_struct;

  // Initialize values in the data structure for mobility calculations
  MobilityData mob_struct = {
			     m_xi,
			     m_ewald_cut,
			     m_ewald_dr,
			     m_ewald_n,
			     d_ewaldC1.data,
			     m_self,
			     d_nneigh_ewald.data,
			     d_nlist_ewald.data,
			     d_headlist_ewald.data,
			     m_eta,
			     m_gaussP,
			     m_gridh,
			     d_gridk.data,
			     d_gridX.data,
			     d_gridY.data,
			     d_gridZ.data,
			     d_gridXX.data,
			     d_gridXY.data,
			     d_gridXZ.data,
			     d_gridYX.data,
			     d_gridYY.data,
			     d_gridYZ.data,
			     d_gridZX.data,
			     d_gridZY.data,
			     plan,
			     m_Nx,
			     m_Ny,
			     m_Nz
  };
  MobilityData *mob_data = &mob_struct;

  // Initialize values in the data structure for the resistance calculations
  //
  // !!! The pointers to the neighbor list here are the EXACT SAME pointers
  //     used mobility structure. Replicated here for simplicity because
  //     it's only ever read, not modified. This way also leaves the option
  //     open to add different neighbor list structures for the lubrication
  //     and mobility calculations. 
  ResistanceData res_struct = {
			       4.0,  // lubrication cutoff, rlub 
			       2.1,  // lubrication (preconditioner) cutoff, rp
			       d_nneigh_ewald.data,
			       d_nlist_ewald.data,
			       d_headlist_ewald.data,
			       d_nneigh_pruned.data,
			       d_headlist_pruned.data,
			       d_nlist_pruned.data,
			       m_nnz,
			       d_nneigh_less.data,
			       d_NEPP.data,
			       d_offset.data,
			       d_L_RowInd.data,
			       d_L_RowPtr.data,
			       d_L_ColInd.data,
			       d_L_Val.data,
			       d_ResTable_dist.data,
			       d_ResTable_vals.data,
			       m_ResTable_min,
			       m_ResTable_dr,
			       soHandle,
			       spHandle,
			       spStatus,
			       descr_R,
			       descr_L,
			       trans_L,
			       trans_Lt,
			       info_R,
			       info_L,
			       info_Lt,
			       policy_R,
			       policy_L,
			       policy_Lt,
			       m_pBufferSize,
			       d_Scratch1.data,
			       d_Scratch2.data,
			       d_Scratch3.data,
			       d_prcm.data,
			       d_HasNeigh.data,
			       d_Diag.data,
			       m_ichol_relaxer,
			       false,
			       m_ndsr,    //non-dimensional shear rate
			       m_k_n,     //collision spring const
			       m_kappa,   //inverse Debye length for electrostatic repulsion
			       m_beta,    //ratio of the Hamaker const and elst force scale
			       m_epsq     //square of the vdw regularization
  };
  ResistanceData *res_data = &res_struct;

  // Initialize values in workspace data
  WorkData work_struct = {
			  blasHandle,      //zhoge: was in res_data
			  m_work_pos_rk1,  //zhoge: RK2 midstep storage
			  m_work_ori_rk1,  //zhoge: RK2 midstep storage
			  dot_sum,
			  m_work_bro_gauss,  //zhoge: Gaussian random variables
			  m_work_bro_ff_psi,
			  m_work_bro_ff_UBreal,
			  m_work_bro_ff_Mpsi,
			  m_work_bro_ff_V1,
			  m_work_bro_ff_UB_new1,
			  m_work_bro_ff_UB_old1,
			  m_work_bro_nf_Tm,
			  m_work_bro_nf_V,
			  m_work_bro_nf_FB_old,
			  m_work_bro_nf_psi,
			  m_work_saddle_psi,
			  m_work_saddle_posPrime,
			  m_work_saddle_rhs,
			  m_work_saddle_solution,
			  m_work_mob_couplet,
			  m_work_mob_delu,
			  m_work_mob_vel1,
			  m_work_mob_vel2,
			  m_work_mob_delu1,
			  m_work_mob_delu2,
			  m_work_mob_vel,
			  m_work_mob_AngvelStrain,
			  m_work_mob_net_force,
			  m_work_mob_TorqueStress,
			  m_work_precond_scratch,
			  m_work_precond_map,
			  m_work_precond_backup
  };
  WorkData *work_data = &work_struct;


  // Time-dependent external torque (constant if m_omega_ext == 0)
  Scalar T_ext = m_T_ext * cos(m_omega_ext * timestep * m_deltaT);

  
  // *********************************************
  // Perform the update on the GPU (in Stokes.cu)
  // *********************************************
  
	
  Stokes_StepOne( timestep,
		  m_period,
		  d_pos.data,            //input/output
		  d_ori.data,            //input/output: orientation
		  d_pos_gb.data,         //input/output: global position (not confined to box)
		  d_AppliedForce.data,   //input/output
		  d_Velocity.data,       //input/output: FSD velocity and stresslet (11N)
		  m_sqm_B1, m_sqm_B2,
		  d_sqm_B1_mask.data,
		  d_sqm_B2_mask.data,
		  m_rot_diff,
		  d_noise_ang.data,
		  T_ext,                 //external torque
		  m_deltaT,              //dt
		  m_error,	       
		  current_shear_rate,    
		  16,                    //cuda block size
		  d_image.data,	       
		  d_index_array.data,    
		  group_size,	       
		  box,		       
		  bro_data,	       
		  mob_data,	       
		  res_data,	       
		  work_data
		  );

	
  ////ishikawa Uncomment the following if testing ishikawa (output the velocity of the passive sphere)
  //for (unsigned int i=0; i<group_size; i++){
  //  unsigned int idx0 = i;
  //  
  //  if (h_tag_array.data[idx0] >= m_N_mix)
  //    {
  //      std::string filename4 = "raw/velocity_passive.txt";
  //      FILE * file4;
  //      file4 = fopen(filename4.c_str(), "a");
  //      //////////////////
  //      ArrayHandle<float> h_Velocity(m_Velocity, access_location::host, access_mode::read);
  //
  //      float * vel_p = & h_Velocity.data[ 6*idx0 ]; //velocity of the passive sphere
  //      
  //      float velx = vel_p[0];
  //      float vely = vel_p[1];
  //      float omgz = vel_p[5];
  //      // Output the translational and angular velocities
  //      fprintf (file4, "%7i %15.6e %15.6e %15.6e %15.6e \n",
  //	       timestep, r_ishi, velx, vely, omgz);
  //      // Close output files
  //      fclose (file4);
  //    }
  //} //for loop
  //// ----------------------------------------------------------------------------------
	
	
  // Save the number of iterations, but reset every so often so that it doesn't grow too large   //zhoge: chow & saad
  m_m_Lanczos_ff = ( ( timestep % 100 == 0 ) || ( bro_data->m_Lanczos_ff > 50 ) ) ? 3 : bro_data->m_Lanczos_ff; 
  m_m_Lanczos_nf = ( ( timestep % 100 == 0 ) || ( bro_data->m_Lanczos_nf > 50 ) ) ? 3 : bro_data->m_Lanczos_nf; 
  
  // Debug
  //m_m_Lanczos_ff = 40;
  //m_m_Lanczos_nf = 40;
	
  // Save the relaxation constant, but reset every so often (used in Precondition.cu for the Cholesky decomposition)
  m_ichol_relaxer = ( ( timestep % 5 == 0 ) || ( m_ichol_relaxer > 1024.0 ) ) ? 1.0 : res_data->ichol_relaxer;


  // Output if the period is set (*after* updating, because the indices may change at the next time step)
  if ( ( m_period > 0 ) && ( int(timestep+1) % m_period == 0 ) ) {
    OutputData(int(timestep+1), box, current_shear_rate);
  }

	

  if (m_exec_conf->isCUDAErrorCheckingEnabled())
    CHECK_CUDA_ERROR();

  // done profiling
  if (m_prof)
    m_prof->pop(m_exec_conf);

}

/*! \param timestep Current time step
	\post Nothing is done.
*/
void Stokes::integrateStepTwo(unsigned int timestep)
{
}

void export_Stokes(pybind11::module& m)
{
  pybind11::class_<Stokes, std::shared_ptr<Stokes> > (m, "Stokes", pybind11::base<IntegrationMethodTwoStep>())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<Variant>,
	 unsigned int, std::shared_ptr<NeighborList>, Scalar, Scalar, std::string, int,
	 Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, unsigned int, Scalar, Scalar,
	 Scalar, Scalar, Scalar >() )
    //In the line above, I added ndsr, kappa, k_n, beta_AF, epsq, sqm_B1, sqm_B2, N_mix, coef_B1_mask, coef_B2_mask,
    //rot_diff, T_ext, omega_ext (zhoge)
    .def("setT", &Stokes::setT)
    .def("setShear", &Stokes::setShear)
    .def("setParams", &Stokes::setParams)
    .def("OutputData", &Stokes::OutputData)
    .def("setResistanceTable", &Stokes::setResistanceTable)
    .def("setSparseMath", &Stokes::setSparseMath)
    .def("AllocateWorkSpaces", &Stokes::AllocateWorkSpaces)
    .def("setFriction", &Stokes::setFriction)
    ;
}

#ifdef WIN32
#pragma warning( pop )
#endif
