//////////////////////////////////////////////////////////////////
// (c) Copyright 2006-  by Jeongnim Kim and Ken Esler           //
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   National Center for Supercomputing Applications &          //
//   Materials Computation Center                               //
//   University of Illinois, Urbana-Champaign                   //
//   Urbana, IL 61801                                           //
//   e-mail: jnkim@ncsa.uiuc.edu                                //
//                                                              //
// Supported by                                                 //
//   National Center for Supercomputing Applications, UIUC      //
//   Materials Computation Center, UIUC                         //
//////////////////////////////////////////////////////////////////
/** @file EinsplineSetHybrid.h
 *
 * Declaration of  EinsplineSetHybrid, hybrid function with CUDA
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SET_HYBRID_CUDA_H
#define QMCPLUSPLUS_EINSPLINE_SET_HYBRID_CUDA_H

#include <QMCWaveFunctions/EinsplineSet.h>
#include <QMCWaveFunctions/EinsplineSetExtended.h>
#include <einspline/multi_bspline_create_cuda.h>
#include <QMCWaveFunctions/AtomicOrbitalCuda.h>

namespace qmcplusplus {

  inline void create_multi_UBspline_3d_cuda (multi_UBspline_3d_d *in, 
					     multi_UBspline_3d_s_cuda* &out)
  { out = create_multi_UBspline_3d_s_cuda_conv (in); }

  inline void create_multi_UBspline_3d_cuda (multi_UBspline_3d_d *in, 
					     multi_UBspline_3d_d_cuda * &out)
  { out = create_multi_UBspline_3d_d_cuda(in); }

  inline void create_multi_UBspline_3d_cuda (multi_UBspline_3d_z *in, 
					     multi_UBspline_3d_c_cuda* &out)
  { out = create_multi_UBspline_3d_c_cuda_conv (in); }

  inline void create_multi_UBspline_3d_cuda (multi_UBspline_3d_z *in, 
					     multi_UBspline_3d_z_cuda * &out)
  { out = create_multi_UBspline_3d_z_cuda(in); }

  inline void create_multi_UBspline_3d_cuda (multi_UBspline_3d_z *in, 
					     multi_UBspline_3d_d_cuda * &out)
  { 
    app_error() << "Attempted to convert complex CPU spline into a real "
		<< " GPU spline.\n";
    abort();
  }

  inline void create_multi_UBspline_3d_cuda (multi_UBspline_3d_z *in, 
					     multi_UBspline_3d_s_cuda * &out)
  { 
    app_error() << "Attempted to convert complex CPU spline into a real "
		<< " GPU spline.\n";
    abort();
  }

  template<typename StorageType>
  class EinsplineSetHybrid : public EinsplineSetExtended<StorageType>
  {
    friend class EinsplineSetBuilder;
  protected:
    int a;
    //////////////////////
    // Type definitions //
    //////////////////////
    typedef typename EinsplineSetExtended<StorageType>::Walker_t     Walker_t;
    typedef typename EinsplineSetExtended<StorageType>::PosType      PosType;
    typedef typename EinsplineSetExtended<StorageType>::CudaRealType CudaRealType;
    typedef typename EinsplineSetExtended<StorageType>::CudaComplexType CudaComplexType;
    typedef typename EinsplineSetExtended<StorageType>::CudaStorageType CudaStorageType;

    vector<gpu::device_vector<CudaRealType> > AtomicSplineCoefs_GPU,
      AtomicPolyCoefs_GPU;
    gpu::device_vector<AtomicOrbitalCuda<CudaRealType> > AtomicOrbitals_GPU;

    // gpu::host_vector<AtomicPolyJob<CudaRealType> >   AtomicPolyJobs_CPU;
    // gpu::device_vector<AtomicPolyJob<CudaRealType> >   AtomicPolyJobs_GPU;
    // gpu::host_vector<AtomicSplineJob<CudaRealType> > AtomicSplineJobs_CPU;
    // gpu::device_vector<AtomicSplineJob<CudaRealType> > AtomicSplineJobs_GPU;

    gpu::device_vector<HybridJobType> HybridJobs_GPU;
    gpu::device_vector<CudaRealType>  IonPos_GPU;
    gpu::device_vector<CudaRealType>  CutoffRadii_GPU, PolyRadii_GPU;
    gpu::device_vector<HybridDataFloat> HybridData_GPU;

    gpu::device_vector<CudaRealType> Ylm_GPU;
    gpu::device_vector<CudaRealType*> Ylm_ptr_GPU, dYlm_dtheta_ptr_GPU, dYlm_dphi_ptr_GPU;
    gpu::host_vector<CudaRealType*> Ylm_ptr_CPU, dYlm_dtheta_ptr_CPU, dYlm_dphi_ptr_CPU;
    gpu::device_vector<CudaRealType> rhats_GPU;
    gpu::host_vector<CudaRealType> rhats_CPU;
    gpu::device_vector<int> JobType;
    
    // Vectors for 3D Bspline evaluation
    gpu::device_vector<CudaRealType> BsplinePos_GPU;
    gpu::host_vector<CudaRealType> BsplinePos_CPU;
    gpu::device_vector<CudaStorageType*> BsplineVals_GPU, BsplineGradLapl_GPU;
    gpu::host_vector<CudaStorageType*> BsplineVals_CPU, BsplineGradLapl_CPU;

    // The maximum lMax across all atomic orbitals
    int lMax;
    int numlm, NumOrbitals, Ylm_BS;
    // Stores the maximum number of walkers that can be handled by currently
    // allocated GPU memory.  Must resize if we have more walkers than this.
    int CurrentWalkers;

    //////////////////////////////
    /// Orbital storage objects //
    //////////////////////////////

    ////////////
    // Timers //
    ////////////
    // Data for vectorized evaluations

    void sort_electrons(vector<PosType> &pos);

  public:
    void initGPU();
    //    void registerTimers();

    // Resize cuda objects
    void resize_cuda(int numwalkers);

    // Vectorized evaluation functions
    void evaluate (vector<Walker_t*> &walkers, int iat,
		   gpu::device_vector<CudaRealType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, int iat,
		   gpu::device_vector<CudaComplexType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   gpu::device_vector<CudaRealType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos,
		   gpu::device_vector<CudaComplexType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   gpu::device_vector<CudaRealType*> &phi,
		   gpu::device_vector<CudaRealType*> &grad_lapl,
		   int row_stride);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   gpu::device_vector<CudaComplexType*> &phi,
		   gpu::device_vector<CudaComplexType*> &grad_lapl,
		   int row_stride);
    void evaluate (vector<PosType> &pos, gpu::device_vector<CudaRealType*> &phi);
    void evaluate (vector<PosType> &pos, gpu::device_vector<CudaComplexType*> &phi);
    
    string Type();
    
    SPOSetBase* makeClone() const;
    
    EinsplineSetHybrid();
  };
}

#endif
/***************************************************************************
* $RCSfile$   $Author: jeongnim.kim $
* $Revision: 5119 $   $Date: 2011-02-06 16:20:47 -0600 (Sun, 06 Feb 2011) $
* $Id: EinsplineSetHybrid.h 5119 2011-02-06 22:20:47Z jeongnim.kim $
***************************************************************************/
