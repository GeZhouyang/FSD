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
// Modified by Andrew Fiore

/*! \file Stokes.cuh
    \brief Declares GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.
*/
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include "DataStruct.h"

//! Define the step_one kernel
#ifndef __MOBILITY_CUH__
#define __MOBILITY_CUH__

//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif


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
				);

__global__ void Mobility_RealSpace_kernel(
					Scalar4 *d_pos,
			      		Scalar4 *d_vel,
					Scalar4 *d_AngvelStrain,
			      		Scalar4 *d_net_force,
					Scalar4 *d_TorqueStress,
			      		int group_size,
			      		Scalar xi,
			      		Scalar4 *d_ewaldC1, 
			      		Scalar2 self, 
			      		Scalar ewald_cut,
			      		int ewald_n,
			      		Scalar ewald_dr,
			      		unsigned int *d_group_members,
			      		BoxDim box,
			      		const unsigned int *d_n_neigh,
                              		const unsigned int *d_nlist,
                              		const unsigned int *d_headlist 
					);

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
				const unsigned int *d_n_neigh,
                        	const unsigned int *d_nlist,
                        	const unsigned int *d_headlist,
				dim3 grid,
				dim3 threads );


__global__ void Mobility_WaveSpace_Spread_kernel( 	
						Scalar4 *d_pos,
				    		Scalar4 *d_net_force,
						Scalar4 *d_TorqueStress,
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
						);

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
						);

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
						);

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
						);

void Mobility_GeneralizedMobility(	
					float *d_generalU,
					float *d_generalF,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					unsigned int group_size,
                        		const BoxDim& box,
					KernelData *ker_data,
					MobilityData *mob_data,
					WorkData *work_data
					);

#endif
