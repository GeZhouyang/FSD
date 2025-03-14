// Maintainer: Andrew Fiore
// Modified by Zhouyang Ge

/*! \file DataStruct.h
    \brief Defines data structures to hold related variables for the different
		parts of the calculation
*/


#include "hoomd/HOOMDMath.h"

#include <cufft.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include "cublas_v2.h"

#ifndef __DATA_STRUCT_H__
#define __DATA_STRUCT_H__

#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

//! Declare a structure to hold all of the kernel parameters
struct KernelData
{

	dim3 particle_grid;	//!< Particle-based calculations CUDA kernel grid dimension
	dim3 particle_threads;	//!< Particle-based calculations CUDA kernel block dimension

	int grid_grid;		//!< FFT Grid-based calcualtions CUDA kernel grid dimension
	int grid_threads;	//!< FFT Grid-based calcualtions CUDA kernel block dimension

	unsigned int NxNyNz;	//!< Total number of FFT grid points

};

//! Declare a structure to hold all of the Brownian calculation information
struct BrownianData
{
	float tol;	//!< Tolerance for the Brownian approximation (should be same as all other errors)

	unsigned int timestep;	//!< Simulation time step (used by RNG)
	
	unsigned int seed_ff_rs;	//!< Seed for the RNG for far-field Brownian calculation, real space
	unsigned int seed_ff_ws;	//!< Seed for the RNG for far-field Brownian calculation, wave space
	unsigned int seed_nf;		//!< Seed for the RNG for near-field Brownian calculation
	unsigned int seed_rfd;		//!< Seed for the RNG for RFD

	int m_Lanczos_ff;	//!< Number of Lanczos iterations for the far-field Brownian calculation
	int m_Lanczos_nf;	//!< Number of Lanczos iterations for the near-field Brownian calculation

	float T;	//!< Temperature

	float rfd_epsilon;	//!< epsilon for RFD approximation

  Scalar  *rfd_rhs;       //!< (DEVICE) RFD right-hand side 
  Scalar  *rfd_sol;       //!< (DEVICE) RFD solution vector

};

//! Declare a structure to hold all of the mobility calculation information
struct MobilityData
{
	Scalar xi;	//!< Ewald splitting parameter

	Scalar ewald_cut; 		//!< Ewald sum real space cutoff distance
	Scalar ewald_dr;  		//!< Ewald sum real space tabulation discretization
	int ewald_n;			//!< Ewald sum real space tabulation number of entries
	Scalar4 *ewald_table;		//!< Ewald sum real space table

	Scalar2 self;       		//!< Ewald sum self piece

	unsigned int *nneigh;		//!< Ewald sum real space number of neighbors
	unsigned int *nlist;		//!< Ewald sum real space neighbor list
	unsigned int *headlist;		//!< Ewald sum real space headlist

	Scalar eta;		//!< Ewald sum wave space spectral Ewald decay parameter
	int P;			//!< Ewald sum wave space spectral Ewald support size
	Scalar3 gridh;		//!< Ewald sum wave space grid spacing (in real space)
	Scalar4 *gridk;        	//!< Ewald sum wave space grid vectors
	CUFFTCOMPLEX *gridX;   	//!< Ewald sum wave space gridded force, x-component
	CUFFTCOMPLEX *gridY;
	CUFFTCOMPLEX *gridZ;
	CUFFTCOMPLEX *gridXX;  	//!< Ewald sum wave space gridded couplet, xx-component
	CUFFTCOMPLEX *gridXY;
	CUFFTCOMPLEX *gridXZ;
	CUFFTCOMPLEX *gridYX;
	CUFFTCOMPLEX *gridYY;
	CUFFTCOMPLEX *gridYZ;
	CUFFTCOMPLEX *gridZX;
	CUFFTCOMPLEX *gridZY;
	cufftHandle plan;	//!< Ewald sum wave space CUFFT plan
	int Nx;			//!< Ewald sum wave space number of grid vectors in each direction
	int Ny;			//!< Ewald sum wave space number of grid vectors in each direction
	int Nz;			//!< Ewald sum wave space number of grid vectors in each direction

};

//! Declare a structure to hold all of the resistance calculation information
struct ResistanceData
{

	float rlub;	//!< cutoff distance for lubrication
	float rp;	//!< cutoff distance for preconditioner

	unsigned int *nneigh;		//!< Lubrication interaction number of neighbors
	unsigned int *nlist;		//!< Lubrication interaction neighbor list
	unsigned int *headlist;		//!< Lubrication interaction headlist
	
	unsigned int *nneigh_pruned;	//!< Number of neighbors for pruned neighborlist
	unsigned int *headlist_pruned;	//!< Headlist for pruned neighborlist
	unsigned int *nlist_pruned;	//!< Pruned neighborlist

	int nnz; 			//!< Lubrication preconditioner Number of non-zero entries
	unsigned int *nneigh_less;	//!< Lubrication preconditioner Number of neighbors with index less than particle
	unsigned int *NEPP;		//!< Lubrication preconditioner Number of entries per-particle 
	unsigned int *offset;		//!< Lubrication preconditioner Offset into array
	
	int   *L_RowInd;	//!< Lubrication preconditioner, sparse storage, row indices
	int   *L_RowPtr;	//!< Lubrication preconditioner, sparse storage, row pointers
	int   *L_ColInd;	//!< Lubrication preconditioner, sparse storage, column indices
	float *L_Val;		//!< Lubrication preconditioner, sparse storage, values

	float *table_dist;	//!< Resistance tabulation distances
	float *table_vals; 	//!< Resistance tabulation values
	float table_min;	//!< Resistance tabulation shortest distance
	float table_dr;		//!< Resistance tabulation discretization

        cusolverSpHandle_t soHandle;		//!< Opaque handle to cuSOLVER
        cusparseHandle_t spHandle;		//!< Opaque handle to cuSPARSE
        cusparseStatus_t spStatus;		//!< Status output for cuSPARSE operations
        cusparseMatDescr_t    descr_R;		//!< Matrix description for the resistance tensor (preconditioner)
        cusparseMatDescr_t    descr_L;		//!< Matrix description for the IChol of resistance tensor
        cusparseOperation_t   trans_L;		//!< Specify to not transpose IChol
        cusparseOperation_t   trans_Lt;		//!< Specify to transpose IChol
        csric02Info_t         info_R;		//!< Info on the resistance tensor
        csrsv2Info_t          info_L;		//!< Info on the IChol matrix
        csrsv2Info_t          info_Lt;		//!< Info on the transpose fo the IChol matrix
        cusparseSolvePolicy_t policy_R;		//!< Solver policy for R
        cusparseSolvePolicy_t policy_L;		//!< Solver policy for L
        cusparseSolvePolicy_t policy_Lt;	//!< Solver policy for L^T
        
	int pBufferSize;			//!< Buffer size for cuSPARSE oeprations

	float *Scratch1;	//!< Scratch vector for in-place calculations (size 6*N)
	float *Scratch2;	//!< Scratch vector for in-place calculations (size 17*N)
	float *Scratch3;	//!< Scratch vector for re-ordering values (size nnz)

	int *prcm;	//!< Reverse-Cuthill-McKee permutation vector

	int *HasNeigh;	//!< List for whether a particle has neighbors or not
	float *Diag;	//!< Diagonal preconditioner for Brownian calculation

	float ichol_relaxer;	//!< magnitude of term to add to diagonal for IChol
	bool ichol_converged;	//!< flag for whether the incomplete Cholesky converged


  // Interparticle force parameters
  float m_ndsr;      //non-dimensional shear rate                       
  float m_k_n;	     //collision spring const                           
  float m_kappa;     //inverse Debye length for electrostatic repulsion 
  float m_beta;      // ratio of Hamaker constant and electrostatic force scale
  float m_epsq;      // square root of the regularization term for vdW
//  float m_sqm_B1;    // coef for the B1 mode of spherical squirmers
//  float m_sqm_B2;    // coef for the B2 mode of spherical squirmers

};

//! Declare a structure to hold work spaces required throughout the calculations
struct WorkData
{

  cublasHandle_t blasHandle;	//!< Opaque handle for cuBLAS operations  //zhoge: was in res_data

  //zhoge: RK2 midstep storage
  Scalar4 *pos_rk1;
  Scalar3 *ori_rk1;
  
	// Dot product partial sum
	Scalar *dot_sum; 	//!< Partial dot product sum
        float *bro_gauss;   //zhoge: Gaussian random variables

	// Variables for far-field Lanczos iteration	
	Scalar4 *bro_ff_psi;	//!< (DEVICE) random vector for far-field real space
	Scalar4 *bro_ff_UBreal;	//!< (DEVICE) real space far-field Brownian displacement
	Scalar4 *bro_ff_Mpsi;	//!< (DEVICE) Product of mobility with the random vector

  //zhoge: re-implement ff Chow & Saad
  Scalar *bro_ff_V1;	        //!< (DEVICE) Basis vectors for Lanczos iteration
  Scalar *bro_ff_UB_new1;	//!< (DEVICE) Old value of displacement
  Scalar *bro_ff_UB_old1;	//!< (DEVICE) Old value of displacement
  
	// Variables for near-field Lanczos iteration	
	Scalar *bro_nf_Tm;	//!< (DEVICE) Tri-diagonal matrix for square root calculation
	Scalar *bro_nf_V;	//!< (DEVICE) Basis vectors for Lanczos iteration
	Scalar *bro_nf_FB_old;	//!< (DEVICE) Old value of displacement
        Scalar *bro_nf_psi;	//!< (DEVICE) Random vector for near-field Brownian calculation

	Scalar  *saddle_psi;		//!< (DEVICE) Random vector for RFD
	Scalar4 *saddle_posPrime;       //!< (DEVICE) Displaced position for RFD
	Scalar  *saddle_rhs;            //!< (DEVICE) Saddle point solve right-hand side 
	Scalar  *saddle_solution;       //!< (DEVICE) Saddle point solve solution vector

	Scalar4 *mob_couplet;		//!< (DEVICE) Placeholder for couplet
	Scalar4 *mob_delu;		//!< (DEVICE) Placeholder for velocity gradient
	Scalar4 *mob_vel1;		//!< (DEVICE) Placeholder for velocity
	Scalar4 *mob_vel2;		//!< (DEVICE) Another
	Scalar4 *mob_delu1;		//!< (DEVICE) Placeholder for velocity gradient
	Scalar4 *mob_delu2;		//!< (DEVICE) Another
	Scalar4 *mob_vel;		//!< (DEVICE) Storage for velocity
	Scalar4 *mob_AngvelStrain;	//!< (DEVICE) Storage for angular velocity and rate of strain
	Scalar4 *mob_net_force;		//!< (DEVICE) Storage for net force
	Scalar4 *mob_TorqueStress;	//!< (DEVICE) Storage for torque and stresslet


	int    *precond_scratch;	//!< (DEVICE) Placeholder for preconditioning copies
	int    *precond_map;		//!< (DEVICE) Map for RCM reordering
	Scalar *precond_backup;		//!< (DEVICE) Backup IChol values if need to increase diagonal

};


#endif
