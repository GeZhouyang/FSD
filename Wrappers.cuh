// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Mobility.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Brownian.cuh"
#include "Helper_Saddle.cuh"
#include "Saddle.cuh"

#include "DataStruct.h"

#include <stdio.h>

#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/print.h>
#include <cusp/array1d.h>
#include <cusp/linear_operator.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/gmres.h>
#include <cusp/krylov/cg.h>
#include <cusp/multiply.h>

#include <stdlib.h>

#include <cusparse.h>

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


/*! \file Wrappers.cuh
    \brief Defines functions wrappers used by CUSP to solve the linear equations where required.
*/


//! Shared memory array for partial sum of dot product kernel
extern __shared__ Scalar partial_sum[];

//! Construct class wrapper to use the Saddle point matrix as a matrix-free method in CUSP.
/*! 
	CUSP shell to apply matrix-free multiplication of the saddle point matrix
*/
class cuspSaddle : public cusp::linear_operator<float,cusp::device_memory>
{
public:

	typedef cusp::linear_operator<float,cusp::device_memory> super; //!< Defines size of linear operator

        // No need to specify their values because it's the relationship that matters
	Scalar *d_x;   //!< input vector (unspecified)
	Scalar *d_y;   //!< output vector (unspecified)

	Scalar4 *d_pos;			//!< Particle positions	
	unsigned int *d_group_members;	//!< list of particles in group to integrate
	unsigned int group_size; 	//!< Number of particles
	const BoxDim& box;		//!< Box dimensions

	KernelData *ker_data;		//!< Pointer to data structure for CUDA kernels
	MobilityData *mob_data;		//!< Pointer to data structure for mobility calculations
	ResistanceData *res_data;	//!< Pointer to data structure for resistance calculations
	WorkData *work_data;		//!< Pointer to data structure for workspaces
	
	// constructor
	cuspSaddle(	
			Scalar4 *d_pos,
			unsigned int *d_group_members,
			unsigned int group_size,
			const BoxDim& box,
			KernelData *ker_data,
			MobilityData *mob_data,
			ResistanceData *res_data,
			WorkData *work_data
			)
        	: super(17*group_size,17*group_size),
			d_pos(d_pos),
			d_group_members(d_group_members),
			group_size(group_size),
			box(box),
			ker_data(ker_data),
			mob_data(mob_data),
			res_data(res_data),
			work_data(work_data)
		{}

	// linear operator y = A*x
	//! Matrix multiplication part of CUSP wrapper
	template <typename VectorType1,
	         typename VectorType2>
	void operator()( VectorType1& x, VectorType2& y )
	{
	
		// Raw pointer to device memory for input and output arrays
		d_x = (float*)thrust::raw_pointer_cast(&x[0]);
		d_y = (float*)thrust::raw_pointer_cast(&y[0]);
	
		// Call the kernel	
		Saddle_Multiply( 
		                d_y, // output
				d_x, // input
				d_pos,
				d_group_members,
				group_size,
		                box,
				ker_data,
				mob_data,
				res_data,
				work_data
				);
	
	}
};


//! Construct class wrapper for the preconditioner to the saddle point matrix in CUSP.
/*! 
	CUSP shell to apply the action of the preconditioner to a vector. P^(-1) * vec
*/
class cuspSaddlePreconditioner : public cusp::linear_operator<float,cusp::device_memory>
{
public:

	typedef cusp::linear_operator<float,cusp::device_memory> super; //!< Defines size of linear operator
	
	Scalar *d_x;   //!< input vector
	Scalar *d_y;   //!< output vector

	unsigned int group_size;	//!< number of particles

	void *pBuffer;	//!< Buffer space for cuSPARSE calculations

	KernelData *ker_data;		//!< Pointer to data structure for CUDA kernels
	ResistanceData *res_data;	//!< Pointer to data structure for mobility calculations

	// constructor
	cuspSaddlePreconditioner(
			unsigned int group_size,
			void *pBuffer,
			KernelData *ker_data,
			ResistanceData *res_data
		)
        	: super(17*group_size,17*group_size),
			group_size(group_size),
			pBuffer(pBuffer),
			ker_data(ker_data),
			res_data(res_data)
		{}

	// Linear operator y = A*x, here A = P^(-1), where P is the preconditioner
	//
	//! Matrix multiplication part of CUSP wrapper
	template <typename VectorType1,
	         typename VectorType2>
	void operator()( VectorType1& x, VectorType2& y )
	{
	
		// Raw pointer to device memory for input and output arrays
		d_x = (float*)thrust::raw_pointer_cast(&x[0]);
		d_y = (float*)thrust::raw_pointer_cast(&y[0]);
	
		// Call the kernel
		Saddle_Preconditioner(	
					d_y, // output
					d_x, // input
					group_size,
					pBuffer,
					ker_data,
					res_data
					);
	
	}
};
