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

#ifndef QMCPLUSPLUS_EINSPLINE_TRAITS_H
#define QMCPLUSPLUS_EINSPLINE_TRAITS_H

#include <einspline/multi_bspline_structs.h>
#ifdef QMC_CUDA
  #include <einspline/multi_bspline_create_cuda.h>
#endif

namespace qmcplusplus 
{

  ////////////////////////////////////////////////////////////////////
  // This is just a template trick to avoid template specialization //
  // in EinsplineSetExtended.                                       //
  ////////////////////////////////////////////////////////////////////
  template<typename StorageType, int dim>  struct MultiOrbitalTraits{};

  template<> struct MultiOrbitalTraits<double,2>
  {  
    typedef multi_UBspline_2d_d SplineType; 
#ifdef QMC_CUDA 
    typedef multi_UBspline_2d_d_cuda CudaSplineType;  
#endif
  };

  template<> struct MultiOrbitalTraits<double,3>
  {  
    typedef multi_UBspline_3d_d SplineType;  
#ifdef QMC_CUDA 
    typedef multi_UBspline_3d_d_cuda CudaSplineType; 
#endif
  };

  template<> struct MultiOrbitalTraits<complex<double>,2>
  {  
    typedef multi_UBspline_2d_z SplineType;  
#ifdef QMC_CUDA 
    typedef multi_UBspline_2d_z_cuda CudaSplineType;  
#endif
  };

  template<> struct MultiOrbitalTraits<complex<double>,3>
  {  
    typedef multi_UBspline_3d_z SplineType;  
#ifdef QMC_CUDA 
    typedef multi_UBspline_3d_z_cuda CudaSplineType;  
#endif
  };


  template<> struct MultiOrbitalTraits<float,2>
  {  
    typedef multi_UBspline_2d_s SplineType;  
#ifdef QMC_CUDA 
    typedef multi_UBspline_2d_s_cuda CudaSplineType;  
#endif
  };

  template<> struct MultiOrbitalTraits<float,3>
  {  
    typedef multi_UBspline_3d_s SplineType;  
#ifdef QMC_CUDA 
    typedef multi_UBspline_3d_s_cuda CudaSplineType;  
#endif
  };

  template<> struct MultiOrbitalTraits<complex<float>,2>
  {  
    typedef multi_UBspline_2d_c SplineType;  
#ifdef QMC_CUDA 
    typedef multi_UBspline_2d_c_cuda CudaSplineType;  
#endif
  };

  template<> struct MultiOrbitalTraits<complex<float>,3>
  {  
    typedef multi_UBspline_3d_c SplineType;  
#ifdef QMC_CUDA 
    typedef multi_UBspline_3d_c_cuda CudaSplineType;  
#endif
  };

  // Real evaluation functions
  inline void 
  EinsplineMultiEval (multi_UBspline_3d_d *restrict spline,
		      const TinyVector<double,3>& r,
		      Vector<double> &psi)
  {
    eval_multi_UBspline_3d_d (spline, r[0], r[1], r[2], psi.data());
  }

  inline void
  EinsplineMultiEval (multi_UBspline_3d_d *restrict spline,
		      TinyVector<double,3> r, 
		      vector<double> &psi)
  {
    eval_multi_UBspline_3d_d (spline, r[0], r[1], r[2], &(psi[0]));
  }

  inline void
  EinsplineMultiEval (multi_UBspline_3d_d *restrict spline,
		      const TinyVector<double,3>& r,
		      Vector<double> &psi,
		      Vector<TinyVector<double,3> > &grad)
  {
    eval_multi_UBspline_3d_d_vg (spline, r[0], r[1], r[2],
				 psi.data(),
				 (double*)grad.data());
  }


  inline void
  EinsplineMultiEval (multi_UBspline_3d_d *restrict spline,
		      const TinyVector<double,3>& r,
		      Vector<double> &psi,
		      Vector<TinyVector<double,3> > &grad,
		      Vector<Tensor<double,3> > &hess)
  {
    eval_multi_UBspline_3d_d_vgh (spline, r[0], r[1], r[2],
				  psi.data(), 
				  (double*)grad.data(), 
				  (double*)hess.data());
  }

  //////////////////////////////////
  // Complex evaluation functions //
  //////////////////////////////////
  inline void 
  EinsplineMultiEval (multi_UBspline_3d_z *restrict spline,
		      const TinyVector<double,3>& r,
		      Vector<complex<double> > &psi)
  {
    eval_multi_UBspline_3d_z (spline, r[0], r[1], r[2], psi.data());
  }

  inline void
  EinsplineMultiEval (multi_UBspline_3d_z *restrict spline,
		      const TinyVector<double,3>& r,
		      Vector<complex<double> > &psi,
		      Vector<TinyVector<complex<double>,3> > &grad)
  {
    eval_multi_UBspline_3d_z_vg (spline, r[0], r[1], r[2],
				 psi.data(), 
				 (complex<double>*)grad.data());
  }

  inline void
  EinsplineMultiEval (multi_UBspline_3d_z *restrict spline,
		      const TinyVector<double,3>& r,
		      Vector<complex<double> > &psi,
		      Vector<TinyVector<complex<double>,3> > &grad,
		      Vector<Tensor<complex<double>,3> > &hess)
  {
    eval_multi_UBspline_3d_z_vgh (spline, r[0], r[1], r[2],
				  psi.data(), 
				  (complex<double>*)grad.data(), 
				  (complex<double>*)hess.data());
  }
			   


#ifdef QMC_CUDA
  template<typename StoreType, typename CudaPrec> struct StorageTypeConverter;
  template<> struct StorageTypeConverter<double,double>
  {    typedef double CudaStorageType;         };
  template<> struct StorageTypeConverter<double,float>
  {    typedef float CudaStorageType;           };
  template<> struct StorageTypeConverter<complex<double>,float>
  {    typedef complex<float> CudaStorageType ; };
  template<> struct StorageTypeConverter<complex<double>,complex<double> >
  {    typedef complex<double> CudaStorageType; };

  template<typename T>
  struct AtomicSplineJob
  {
    T dist, SplineDelta;
    T rhat[OHMMS_DIM];
    int lMax, YlmIndex;
    T* SplineCoefs;
    T *phi, *grad_lapl;
    T PAD[3];
    //T PAD[(64 - (2*OHMMS_DIM*sizeof(T) + 2*sizeof(int) + 2*sizeof(T*)))/sizeof(T)];
  };

  template<typename T>
  struct AtomicPolyJob
  {
    T dist, SplineDelta;
    T rhat[OHMMS_DIM];
    int lMax, PolyOrder, YlmIndex;
    T* PolyCoefs;
    T *phi, *grad_lapl;
    T PAD[2];
    //T PAD[(64 - (2*OHMMS_DIM*sizeof(T) + 2*sizeof(int) + 2*sizeof(T*)))/sizeof(T)];
  };

#endif

}
#endif
