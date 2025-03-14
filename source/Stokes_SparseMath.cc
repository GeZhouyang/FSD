#include <stdio.h>
#include "Stokes.h"
         
#include <stdlib.h>
#include <cuda_runtime.h>

#include <cusparse.h>
#include <cusolverSp.h>

// Set up the sparse matrices using CUSPARSE
//
// ***IMPORTANT: To get the cuSPARSE libraries to work, have to link to
//               libcusparse.so in the FindHoomd makefile, like so:
//	set(HOOMD_LIBRARIES ${HOOMD_LIB} ${HOOMD_COMMON_LIBS} /usr/local/cuda/lib64/libcublas.so /usr/local/cuda/lib64/libcusparse.so )
void Stokes::setSparseMath(){

	// Set up the arrays and memory required to store the matrix information
	// 
	// Total Memory required for the arrays declared in this file:
	// 	
	//	sizeof(float) = sizeof(int) = 4
	//	
	//	nnz = 468 * N 
	//	
	//	Variable		Length		Type
	//	--------		------		----
	//	nneigh_pruned		N		int
	//	headlist_pruned 	N+1		int
	//	nlist_pruned		13*N		int
	//	L_RowInd		nnz		int
	//	L_RowPtr		6*N+1		int
	//	L_ColInd		nnz		int		
	//	L_Val			nnz		float
	//	HasNeigh		N		int
	//	Diag			6*N		float
	//	nneigh_less		N		int
	//	NEPP			2*N		int
	//	offset			N+1		int
	//	Scratch1		6*N		float
	//	Scratch2		17*N		float
	//	Scratch3		nnz		float
	//	prcm			6*N		int
	//
	//				1960*N+3 \approx 1960*N
	//
	//	Total size in bytes: 	7840 * N
	//	Total size in KB:	7.840 * N
	//	Total size in MB:	0.007840 * N
	//
	// Some examples for various numbers of particles
	//
	//	N	Size (MB)	Size (GB)
	//	---	---------	---------
	//	1E2	0.7840		0.0007840
	//	1E3	7.840		0.007840
	//	1E4	78.40		0.07840
	//	1E5	784.0		0.7840
	//	1E6	7840		7.840

	// For particles of equal size, in a closest-packed configuration, each particle
	// can have at most 12 neighbors within a distance of 3 or less
	unsigned int N = m_group->getNumMembers();
	unsigned int max_neigh = 13;

	// Maximum number of non-zero entries in the RFU preconditioner
	m_nnz = ( 36 * ( max_neigh + 1 ) ) * N; 

	// Pruned Neighborlist Arrays (lists of particles within the (shorter) preconditioner cutoff
	GPUArray<unsigned int> n_nneigh_pruned( N, m_exec_conf );
	GPUArray<unsigned int> n_headlist_pruned( N + 1, m_exec_conf );
	GPUArray<unsigned int> n_nlist_pruned( (max_neigh+1)*N, m_exec_conf );

	m_nneigh_pruned.swap( n_nneigh_pruned );
	m_headlist_pruned.swap( n_headlist_pruned );
	m_nlist_pruned.swap( n_nlist_pruned );

	// Prepare GPUArrays for sparse matrix constructions
	GPUArray<int>   n_L_RowInd( m_nnz, m_exec_conf );	//!< Rnf preconditioner sparse storage ( COO Format - Row Indices )
	GPUArray<int>   n_L_RowPtr( 6*N+1, m_exec_conf );	//!< Rnf preconditioner sparse storage ( CSR Format - Row Pointers )
	GPUArray<int>   n_L_ColInd( m_nnz, m_exec_conf );	//!< Rnf preconditioner sparse storage ( COO/CSR Format - Col Indices )
	GPUArray<float> n_L_Val(    m_nnz, m_exec_conf );	//!< Values of incomplete lower Cholesky of RFU (also the matrix itself)

	m_L_RowInd.swap(n_L_RowInd);
	m_L_RowPtr.swap(n_L_RowPtr);
	m_L_ColInd.swap(n_L_ColInd);
	m_L_Val.swap(   n_L_Val   );

	// Things required for diagonal preconditioning
	GPUArray<int>   n_HasNeigh( N, m_exec_conf );	//!< Whether a particle has neighbors or not
	GPUArray<float> n_Diag( 6*N, m_exec_conf );	//!< Diagonal preconditioner elements 
	m_Diag.swap( n_Diag );
	m_HasNeigh.swap( n_HasNeigh );

	// Index arrays needed to construct sparse matrices
	GPUArray<unsigned int> n_nneigh_less( N, m_exec_conf ); //!< Number of neighbors with index less than particle ID
	GPUArray<unsigned int> n_NEPP( 2*N, m_exec_conf );       //!< Number of non-zero entries per particle in sparse matrices
	GPUArray<unsigned int> n_offset( (N+1), m_exec_conf );   //!< Particle offset into sparse matrix arrays

	m_nneigh_less.swap( n_nneigh_less );
	m_NEPP.swap( n_NEPP );
	m_offset.swap( n_offset );

	//
	// Re-ordering vector and scratch space for sparse math operations
	//
	GPUArray<float> n_Scratch1( 6*N, m_exec_conf );		//!< Scratch storage for re-ordered matrix-vector multiplication 
	GPUArray<float> n_Scratch2( 17*N, m_exec_conf );	//!< Scratch storage for saddle point preconditioning
	GPUArray<float> n_Scratch3( m_nnz, m_exec_conf );	//!< Scratch Storage for Value reordering 
	GPUArray<int>   n_prcm( 6*N, m_exec_conf );		//!< matrix re-ordering vector using Reverse-Cuthill-Mckee (RCM)

	m_Scratch1.swap( n_Scratch1 );
	m_Scratch2.swap( n_Scratch2 );
	m_Scratch3.swap( n_Scratch3 );
	m_prcm.swap( n_prcm );

	//
	// Set up conctext for cuSOLVER (used to perform reverse-Cuthill-Mckee reordering
	//
	cusolverSpCreate(&soHandle);

	//
	// Set up matrices for cuSPARSE
	//

        // Initialize cuSPARSE
        cusparseCreate(&spHandle);

	// 1. Define the matrices for cuSPARSE, detailing the structure
	descr_R = 0;
	cusparseCreateMatDescr( &descr_R );
	cusparseSetMatIndexBase( descr_R, CUSPARSE_INDEX_BASE_ZERO );
	cusparseSetMatType( descr_R, CUSPARSE_MATRIX_TYPE_GENERAL );

	descr_L = 0;
	cusparseCreateMatDescr( &descr_L );
	cusparseSetMatDiagType( descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT );
	cusparseSetMatType( descr_L, CUSPARSE_MATRIX_TYPE_GENERAL );
	cusparseSetMatFillMode( descr_L, CUSPARSE_FILL_MODE_LOWER );
	cusparseSetMatIndexBase( descr_L, CUSPARSE_INDEX_BASE_ZERO );
 
	// 2. Operations for the triangular matrix solves
        trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
        trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

        // 3. Create info structures for cuSPARSE solves
        // 	We need one info for csric02 (incomplete Cholesky) 
	//	and two info's for csrsv2 (Lower and upper triangular solves)
        info_R = 0; // Info structures required for setting buffer size
        info_L = 0;
	info_Lt = 0;
        cusparseCreateCsric02Info(&info_R);
        cusparseCreateCsrsv2Info(&info_L);
        cusparseCreateCsrsv2Info(&info_Lt);

	// 4. Level output information for cuSPARSE solves
        policy_R = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
        policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
        policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	
	// Initialize cuBLAS
	cublasCreate( &blasHandle );  //zhoge: Was here probably because initially only used for lubrication

}
