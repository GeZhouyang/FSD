// Modified by Andrew Fiore
// Modified by Zhouyang Ge

#ifndef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

#ifndef __STOKES_H__
#define __STOKES_H__

/*! \file Stokes.h
    \brief Declares the Stokes class
*/

#include <hoomd/Variant.h>
//#include <hoomd/ForceCompute.h> //zhoge//RK2//////
#include <hoomd/md/NeighborList.h>
#include <hoomd/md/IntegrationMethodTwoStep.h>

#include <cufft.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include "cublas_v2.h"

#include "ShearFunction.h"

#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h> // lets us pass arrays from python to c++

//! Integrates the system forward considering hydrodynamic interactions by GPU
/*! Implements overdamped integration (one step) through IntegrationMethodTwoStep interface, runs on the GPU
*/

class Stokes : public IntegrationMethodTwoStep
{
public:
  //! Constructs the integration method and associates it with the system
  Stokes(	std::shared_ptr<SystemDefinition> sysdef,
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
		Scalar rot_diff, Scalar T_ext, Scalar omega_ext  //zhoge
		);

  virtual ~Stokes();

  //! Set a new temperature
  /*! \param T new temperature to set */
  void setT(std::shared_ptr<Variant> T)
  {
    m_T = T;
  }

  //! Performs the first step of the integration
  virtual void integrateStepOne(unsigned int timestep);

  //! Performs the second step of the integration
  virtual void integrateStepTwo(unsigned int timestep);

  //! Set the table for resistance coefficients
  void setResistanceTable();

  //! Set up the sparse math functions
  void setSparseMath();

  //! Set the parameters for various parts of the calculation (Ewald Sum, Lubrication Tensor)
  void setParams();
		
  //! Write particle dipoles to file
  void OutputData(unsigned int timestep, BoxDim box, Scalar current_shear_rate);

  //! Allocate workspace variables
  void AllocateWorkSpaces();
		
  //! Free workspace variables
  void FreeWorkSpaces();

  //! Set the friction type
  void setFriction( std::string friction_type, float h0, std::vector<float> &alpha );

  //! Set the shear rate and shear frequency
  void setShear(std::shared_ptr<ShearFunction> shear_func, Scalar max_strain){
    m_shear_func = shear_func;
    m_max_strain = max_strain;
  }

        
protected:

  std::shared_ptr<ShearFunction> m_shear_func; //!< mutable shared pointer towards a ShearFunction object
  Scalar m_max_strain; //!< Maximum total strain before box resizing

  std::shared_ptr<Variant> m_T;	//!< The Temperature of the Stochastic Bath
  unsigned int m_seed;			//!< The seed for the RNG of the Stochastic Bath
  unsigned int m_seed_ff_rs;		//!< The seed for the RNG, far-field, real space
  unsigned int m_seed_ff_ws;		//!< The seed for the RNG, far-field, wave space
  unsigned int m_seed_nf;			//!< The seed for the RNG, near-field
  unsigned int m_seed_rfd;		//!< The seed for the RNG, random finite displacement
        
  cufftHandle plan;       //!< Used for the Fast Fourier Transformations performed on the GPU
        
  std::shared_ptr<NeighborList> m_nlist_ewald;  //!< The neighborlist to use for the mobility computation

  unsigned int m_shear_offset; //!< Offset time of the shear

  ////zhoge//RK2//////
  //std::shared_ptr<ForceCompute> m_force; //!< mutable shared pointer 
  

  // ************************************************************************
  // Declare all variables related to the far-field hydrodynamic calculations
  // ************************************************************************
 
  Scalar m_xi;                   //!< ewald splitting parameter xi
  Scalar m_ewald_cut;            //!< Real space cutoff
  GPUArray<Scalar4> m_ewaldC1;   //!< Real space Ewald coefficients table
  int m_ewald_n;                 //!< Number of entries in table of Ewald coefficients
  Scalar m_ewald_dr;             //!< Real space Ewald table spacing
 
  Scalar2 m_self; //!< self piece

  int m_Nx;  //!< Number of grid points in x direction
  int m_Ny;  //!< Number of grid points in y direction
  int m_Nz;  //!< Number of grid points in z direction
 
  GPUArray<Scalar4> m_gridk;        //!< k-vectors for each grid point
  GPUArray<CUFFTCOMPLEX> m_gridX;   //!< x component of the grid based force/velocity
  GPUArray<CUFFTCOMPLEX> m_gridY;   //!< y component of the grid based force/velocity
  GPUArray<CUFFTCOMPLEX> m_gridZ;   //!< z component of the grid based force/velocity
        
  GPUArray<CUFFTCOMPLEX> m_gridXX;   //!< xx component of the grid based couplet/velocity gradient
  GPUArray<CUFFTCOMPLEX> m_gridXY;   //!< xy component of the grid based couplet/velocity gradient
  GPUArray<CUFFTCOMPLEX> m_gridXZ;   //!< xz component of the grid based couplet/velocity gradient
  GPUArray<CUFFTCOMPLEX> m_gridYX;   //!< yx component of the grid based couplet/velocity gradient
  GPUArray<CUFFTCOMPLEX> m_gridYY;   //!< yy component of the grid based couplet/velocity gradient
  GPUArray<CUFFTCOMPLEX> m_gridYZ;   //!< yz component of the grid based couplet/velocity gradient
  GPUArray<CUFFTCOMPLEX> m_gridZX;   //!< zx component of the grid based couplet/velocity gradient
  GPUArray<CUFFTCOMPLEX> m_gridZY;   //!< zy component of the grid based couplet/velocity gradient
 
  Scalar m_gaussm;  //!< Gaussian width in standard deviations for wave space spreading/contraction
  int m_gaussP;     //!< Number of points in each dimension for Gaussian support 
  Scalar m_eta;     //!< Gaussian spreading parameter
  Scalar3 m_gridh;  //!< Size of the grid box in 3 direction
        
  int m_m_Lanczos_ff; //!< Number of Lanczos Iterations to use for calculation of far-field Brownian slip
  int m_m_Lanczos_nf; //!< Number of Lanczos Iterations to use for calculation of near-field Brownian force

  float m_rfd_epsilon;	//!< epsilon for RFD displacement

  Scalar m_error;  //!< Error tolerance for all calculations

  Scalar m_ndsr;   // non-dim shear rate (zhoge)
  Scalar m_kappa;  // inverse Debye length (zhoge)
  Scalar m_k_n;    // collision spring constant (zhoge)
  Scalar m_beta;   // ratio of Hamaker constant and electrostatic force scale
  Scalar m_epsq;   // square root of the regularization term for vdW
  Scalar m_sqm_B1; // coef for the B1 mode of spherical squirmers
  Scalar m_sqm_B2; // coef for the B2 mode of spherical squirmers
  unsigned int m_N_mix;  // number of particles in the first group (when having a mixture)
  Scalar m_coef_B1_mask; // coef for the B1 mask of spherical squirmers
  Scalar m_coef_B2_mask; // coef for the B2 mask of spherical squirmers
  GPUArray<float> m_sqm_B1_mask; // mask array for B1
  GPUArray<float> m_sqm_B2_mask; // mask array for B2
  GPUArray<Scalar3> m_noise_ang; // Gaussian noise for the angular velocity
  Scalar m_rot_diff;  // rotational diffusion coef due to noise
  Scalar m_T_ext;     // external torque
  Scalar m_omega_ext; // external torque angular frequency

  // ******************************************************************
  // Declare all variables for physical quantities (forces, velocities)
  // ******************************************************************

  GPUArray<float> m_AppliedForce; // Force and torque applied to the particles
  GPUArray<float> m_Velocity; // Linear velocity,  angular velocity, and stresslet of all particles

  // *********************************************************************************
  // Declare all variables related to the lubrication and required sparse calculations
  // *********************************************************************************

  float m_ResTable_min; 		  //!< Minimum distance in the lubrication tabulation 
  float m_ResTable_dr; 		  //!< Discretization of the lubrication table (in log space) 
  GPUArray<Scalar> m_ResTable_dist; //!< Distance values used in the lubrication function tabulation
  GPUArray<Scalar> m_ResTable_vals; //!< Lubrication function tabulation
			
  GPUArray<unsigned int> m_nneigh_pruned;		//!< Number of neighbors for pruned neighborlist
  GPUArray<unsigned int> m_headlist_pruned;	//!< Headlist for pruned neighborlist
  GPUArray<unsigned int> m_nlist_pruned;		//!< Pruned neighborlist
	
  GPUArray<unsigned int> m_nneigh_less;	//!< Number of neighbors with index less than particle ID
  GPUArray<unsigned int> m_NEPP;		//!< Number of non-zero entries per particle in sparse matrices
  GPUArray<unsigned int> m_offset;	//!< Particle offset into sparse matrix arrays

  int m_nnz; //!< Number of non-zero entries in RFU preconditioner

  GPUArray<int>   m_L_RowInd;	//!< Rnf sparse storage ( COO Format - Row Indices )
  GPUArray<int>   m_L_RowPtr;	//!< Rnf sparse storage ( CSR Format - Row Pointer )
  GPUArray<int>   m_L_ColInd;	//!< Rnf sparse storage ( COO/CSR Format - Col Indices )
  GPUArray<float> m_L_Val;	//!< L sparse storage ( COO/CSR Format - Values )
	
  GPUArray<float> m_Diag;		//!< Diagonal entries for preconditioner
  GPUArray<int>   m_HasNeigh;	//!< Whether a particle has neighbors or not	

  float m_ichol_relaxer;	//!< magnitude of term to add to diagonal before IChol to ensure convergence

  cusolverSpHandle_t soHandle; //!< opaque handle fo cuSOLVER operations
 
  cusparseHandle_t spHandle;       //!< Opaque handle for cuSPARSE operations
  cusparseStatus_t spStatus;       //!< cuSPARSE function success/failure output
  cusparseMatDescr_t descr_R;      //!< cuSPARSE matrix descriptor for resistance tensor
  cusparseMatDescr_t descr_L;      //!< cuSPARSE matrix descriptor for lower cholesky of R
  cusparseOperation_t trans_L;     //!< Transpose option for lower Cholesky factor, L
  cusparseOperation_t trans_Lt;    //!< Transpose option for upper Cholesky factor, L^T
  csric02Info_t info_R;            //!< Opaque solver information for cuSPARSE operations on R
  csrsv2Info_t info_L;             //!< Opaque solver information for cuSPARSE operations on L
  csrsv2Info_t info_Lt;            //!< Opaque solver information for cuSPARSE operations on L^T
  cusparseSolvePolicy_t policy_R;  //!< Solve level output for R
  cusparseSolvePolicy_t policy_L;  //!< Solve level output for L
  cusparseSolvePolicy_t policy_Lt; //!< Solve level output for L^T
  int m_pBufferSize;               //!< Buffer size for cuSPARSE calculations

  GPUArray<float> m_Scratch1;     //!< 6*N, Scratch storage for re-ordered matrix-vector multiplication 
  GPUArray<float> m_Scratch2;     //!< 17*N, Scratch storage for saddle point preconditioning
  GPUArray<float> m_Scratch3;	//!< nnz, Scratch Storage for Value reordering 
  GPUArray<int> m_prcm;           //!< matrix re-ordering vector using Reverse-Cuthill-Mckee (RCM)
		
  std::string m_fileprefix;	//!< output file prefix
  int m_period;			//!< frequency with which to write output files
	
  // *********************************************************************************
  // Work space variables for all calculations
  // *********************************************************************************

  cublasHandle_t blasHandle;	//!< opaque handle for cuBLAS operations

  
  // Dot product partial sum
  Scalar *dot_sum;
  float  *m_work_bro_gauss;  //zhoge: Gaussian random variables (type float for cuRand)
  
  // Variables for far-field Lanczos iteration
  Scalar4 *m_work_bro_ff_psi;
  Scalar4 *m_work_bro_ff_UBreal;
  Scalar4 *m_work_bro_ff_Mpsi;
  //zhoge
  Scalar *m_work_bro_ff_V1;
  Scalar *m_work_bro_ff_UB_new1;
  Scalar *m_work_bro_ff_UB_old1;

  Scalar  *m_work_rfd_rhs;
  Scalar  *m_work_rfd_sol;
  
  Scalar4  *m_work_pos_rk1;
  Scalar3  *m_work_ori_rk1;

  
  // Variables for near-field Lanczos iteration  
  Scalar *m_work_bro_nf_Tm;
  Scalar *m_work_bro_nf_V;
  Scalar *m_work_bro_nf_FB_old;
  Scalar *m_work_bro_nf_psi;

  Scalar  *m_work_saddle_psi;
  Scalar4 *m_work_saddle_posPrime;
  Scalar  *m_work_saddle_rhs;
  Scalar  *m_work_saddle_solution;

  Scalar4 *m_work_mob_couplet;
  Scalar4 *m_work_mob_delu;
  Scalar4 *m_work_mob_vel1;
  Scalar4 *m_work_mob_vel2;
  Scalar4 *m_work_mob_delu1;
  Scalar4 *m_work_mob_delu2;
  Scalar4 *m_work_mob_vel;
  Scalar4 *m_work_mob_AngvelStrain;
  Scalar4 *m_work_mob_net_force;
  Scalar4 *m_work_mob_TorqueStress;

  int    *m_work_precond_scratch;	
  int    *m_work_precond_map;
  Scalar *m_work_precond_backup;


};

//! Exports the Stokes class to python
void export_Stokes(pybind11::module& m);

#endif
