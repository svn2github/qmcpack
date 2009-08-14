//////////////////////////////////////////////////////////////////
// (c) Copyright 2006-  by Jeongnim Kim and Ken Esler           //
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   National Center for Supercomputing Applications &          //
//   Materials Computation Center                               //
//   University of Illinois, Urbana-Champaign                   //
//   Urbana, IL 61801                                           //
//   e-mail: jnkim@ncsa.uiuc.edu                                //
//   Tel:    217-244-6319 (NCSA) 217-333-3324 (MCC)             //
//                                                              //
// Supported by                                                 //
//   National Center for Supercomputing Applications, UIUC      //
//   Materials Computation Center, UIUC                         //
//////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_EINSPLINE_SET_H
#define QMCPLUSPLUS_EINSPLINE_SET_H

//#include <einspline/bspline.h>
#include "Configuration.h"
#include "QMCWaveFunctions/BasisSetBase.h"
#include "QMCWaveFunctions/SPOSetBase.h"
#include "Optimize/VarList.h"
#include "QMCWaveFunctions/EinsplineOrb.h"
#include "QMCWaveFunctions/AtomicOrbital.h"
#include "QMCWaveFunctions/MuffinTin.h"
#include "Utilities/NewTimer.h"
#include <einspline/multi_bspline_structs.h>
#include "Configuration.h"
#include "Numerics/e2iphi.h"

#include <einspline/multi_bspline_create_cuda.h>
#include "QMCWaveFunctions/AtomicOrbitalCuda.h"


namespace qmcplusplus {

  class EinsplineSetBuilder;

  class EinsplineSet : public SPOSetBase
  {
    friend class EinsplineSetBuilder;
  protected:
    //////////////////////
    // Type definitions //
    //////////////////////
    typedef CrystalLattice<RealType,OHMMS_DIM> UnitCellType;
    
    ///////////
    // Flags //
    ///////////
    /// True if all Lattice is diagonal, i.e. 90 degree angles
    bool Orthorhombic;
    /// True if we are using localize orbitals
    bool Localized;
    /// True if we are tiling the primitive cell
    bool Tiling;
    
    //////////////////////////
    // Lattice and geometry //
    //////////////////////////
    TinyVector<int,3> TileFactor;
    Tensor<int,OHMMS_DIM> TileMatrix;
    UnitCellType SuperLattice, PrimLattice;
    /// The "Twist" variables are in reduced coords, i.e. from 0 to1.
    /// The "k" variables are in Cartesian coordinates.
    PosType TwistVector, kVector;
    /// This stores which "true" twist vector this clone is using.
    /// "True" indicates the physical twist angle after untiling
    int TwistNum;
    /// metric tensor to handle generic unitcell
    Tensor<RealType,OHMMS_DIM> GGt;

    ///////////////////////////////////////////////
    // Muffin-tin orbitals from LAPW calculation //
    ///////////////////////////////////////////////
    vector<MuffinTinClass> MuffinTins;
    int NumValenceOrbs, NumCoreOrbs;
        
  public:  
    UnitCellType GetLattice();

    void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi);
    void evaluate(const ParticleSet& P, int iat, 
		  ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi);
    void evaluate(const ParticleSet& P, int first, int last,
		  ValueMatrix_t& psi, GradMatrix_t& dpsi, 
		  ValueMatrix_t& d2psi);

    void resetTargetParticleSet(ParticleSet& e);
    void resetSourceParticleSet(ParticleSet& ions);
    void setOrbitalSetSize(int norbs);
    string Type();
    EinsplineSet() :  
      TwistNum(0), NumValenceOrbs(0), NumCoreOrbs(0)
    {
      className = "EinsplineSet";
    }
  };

  class EinsplineSetLocal : public EinsplineSet
  {
    friend class EinsplineSetBuilder;
  protected:
    /////////////////////
    // Orbital storage //
    /////////////////////
    /// Store the orbital objects.  Using template class allows us to
    /// avoid making separate real and complex versions of this class.
    //std::vector<EinsplineOrb<ValueType,OHMMS_DIM>*> Orbitals;
    std::vector<EinsplineOrb<complex<RealType>,OHMMS_DIM>*> Orbitals;

  public:
    SPOSetBase* makeClone() const;
    void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi);
    void evaluate(const ParticleSet& P, int iat, 
		  ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi);
    void evaluate(const ParticleSet& P, int first, int last,
		  ValueMatrix_t& psi, GradMatrix_t& dpsi, 
		  ValueMatrix_t& d2psi);

    void resetParameters(const opt_variables_type& active);

    EinsplineSetLocal() 
    {
      className = "EinsplineSetLocal";
    }
  };



  ////////////////////////////////////////////////////////////////////
  // This is just a template trick to avoid template specialization //
  // in EinsplineSetExtended.                                       //
  ////////////////////////////////////////////////////////////////////
  template<typename StorageType, int dim>  struct MultiOrbitalTraits{};

  template<> struct MultiOrbitalTraits<double,2>
  {  
    typedef multi_UBspline_2d_d SplineType;  
    typedef multi_UBspline_2d_d_cuda CudaSplineType;  
  };

  template<> struct MultiOrbitalTraits<double,3>
  {  
    typedef multi_UBspline_3d_d SplineType;  
    typedef multi_UBspline_3d_d_cuda CudaSplineType; 
  };

  template<> struct MultiOrbitalTraits<complex<double>,2>
  {  
    typedef multi_UBspline_2d_z SplineType;  
    typedef multi_UBspline_2d_z_cuda CudaSplineType;  
  };

  template<> struct MultiOrbitalTraits<complex<double>,3>
  {  
    typedef multi_UBspline_3d_z SplineType;  
    typedef multi_UBspline_3d_z_cuda CudaSplineType;  
  };


  template<> struct MultiOrbitalTraits<float,2>
  {  
    typedef multi_UBspline_2d_s SplineType;  
    typedef multi_UBspline_2d_s_cuda CudaSplineType;  
  };

  template<> struct MultiOrbitalTraits<float,3>
  {  
    typedef multi_UBspline_3d_s SplineType;  
    typedef multi_UBspline_3d_s_cuda CudaSplineType;  
  };

  template<> struct MultiOrbitalTraits<complex<float>,2>
  {  
    typedef multi_UBspline_2d_c SplineType;  
    typedef multi_UBspline_2d_c_cuda CudaSplineType;  
  };

  template<> struct MultiOrbitalTraits<complex<float>,3>
  {  
    typedef multi_UBspline_3d_c SplineType;  
    typedef multi_UBspline_3d_c_cuda CudaSplineType;  
  };


  template<typename StoreType, typename CudaPrec> struct StorageTypeConverter;
  template<> struct StorageTypeConverter<double,double>
  {    typedef double CudaStorageType;         };
  template<> struct StorageTypeConverter<double,float>
  {    typedef float CudaStorageType;           };
  template<> struct StorageTypeConverter<complex<double>,float>
  {    typedef complex<float> CudaStorageType ; };
  template<> struct StorageTypeConverter<complex<double>,complex<double> >
  {    typedef complex<double> CudaStorageType; };

  



  ////////////////////////////////////////////////////////////////////
  // Template class for evaluating multiple extended Bloch orbitals // 
  // quickly.  Currently uses einspline library.                    //
  ////////////////////////////////////////////////////////////////////
  template<typename StorageType>
  class EinsplineSetExtended : public EinsplineSet
  {
    friend class EinsplineSetBuilder;
  protected:
    //////////////////////
    // Type definitions //
    //////////////////////
    //typedef CrystalLattice<RealType,OHMMS_DIM> UnitCellType;
    typedef typename StorageTypeConverter<StorageType,CUDA_PRECISION>::CudaStorageType CudaStorageType;
    typedef typename MultiOrbitalTraits<StorageType,OHMMS_DIM>::SplineType SplineType; 
    typedef typename MultiOrbitalTraits<CudaStorageType,OHMMS_DIM>::CudaSplineType CudaSplineType; 
    typedef typename OrbitalSetTraits<StorageType>::ValueVector_t StorageValueVector_t;
    typedef typename OrbitalSetTraits<StorageType>::GradVector_t  StorageGradVector_t;
    typedef typename OrbitalSetTraits<StorageType>::HessVector_t  StorageHessVector_t;
    typedef Vector<RealType>                                        RealValueVector_t;
    typedef Vector<complex<RealType> >                              ComplexValueVector_t;
    typedef Vector<TinyVector<RealType,OHMMS_DIM> >                 RealGradVector_t;
    typedef Vector<TinyVector<complex<RealType>,OHMMS_DIM> >        ComplexGradVector_t;
    typedef Vector<Tensor<RealType,OHMMS_DIM> >                     RealHessVector_t;
    typedef Vector<Tensor<complex<RealType>,OHMMS_DIM> >            ComplexHessVector_t;
    typedef Matrix<Tensor<RealType,OHMMS_DIM> >                     RealHessMatrix_t;
    typedef Matrix<Tensor<complex<RealType>,OHMMS_DIM> >            ComplexHessMatrix_t;
    typedef Matrix<RealType>                                        RealValueMatrix_t;
    typedef Matrix<complex<RealType> >                              ComplexValueMatrix_t;
    typedef Matrix<TinyVector<RealType,OHMMS_DIM> >                 RealGradMatrix_t;
    typedef Matrix<TinyVector<complex<RealType>,OHMMS_DIM> >        ComplexGradMatrix_t;
//     typedef typename OrbitalSetTraits<ReturnType >::ValueVector_t ReturnValueVector_t;
//     typedef typename OrbitalSetTraits<ReturnType >::GradVector_t  ReturnGradVector_t;
//     typedef typename OrbitalSetTraits<ReturnType >::HessVector_t  ReturnHessVector_t;

//     typedef typename OrbitalSetTraits<ReturnType >::ValueMatrix_t ReturnValueMatrix_t;
//     typedef typename OrbitalSetTraits<ReturnType >::GradMatrix_t  ReturnGradMatrix_t;
//     typedef typename OrbitalSetTraits<ReturnType >::HessMatrix_t  ReturnHessMatrix_t;
       
    //////////////////////////////
    /// Orbital storage objects //
    //////////////////////////////
    SplineType *MultiSpline;

    //////////////////////////////////////
    // Radial/Ylm orbitals around atoms //
    //////////////////////////////////////
    vector<AtomicOrbital<StorageType> > AtomicOrbitals;

    // First-order derivative w.r.t. the ion positions
    vector<TinyVector<SplineType*,OHMMS_DIM> > FirstOrderSplines;

    CudaSplineType *CudaMultiSpline;
    // Temporary storage for Eispline calls
    StorageValueVector_t StorageValueVector, StorageLaplVector;
    StorageGradVector_t  StorageGradVector;
    StorageHessVector_t  StorageHessVector;
    // Cuda equivalents of the above
    cuda_vector<CudaStorageType> CudaValueVector, CudaGradLaplVector;
    cuda_vector<CudaStorageType*> CudaValuePointers, CudaGradLaplPointers;
    void resizeCuda(int numWalkers);
    // Temporary storage used when blending functions        
    StorageValueVector_t BlendValueVector, BlendLaplVector;   
    StorageGradVector_t BlendGradVector;
        
    // True if we should unpack this orbital into two copies
    vector<bool>         MakeTwoCopies;
    // Cuda equivalent
    cuda_vector<int> CudaMakeTwoCopies;
    // k-points for each orbital
    Vector<TinyVector<RealType,OHMMS_DIM> > kPoints;
    // Cuda equivalent
    cuda_vector<TinyVector<CUDA_PRECISION,OHMMS_DIM > > CudakPoints;

    ///////////////////
    // Phase factors //
    ///////////////////
    Vector<RealType> phase;
    Vector<complex<RealType> > eikr;
    inline void computePhaseFactors(TinyVector<RealType,OHMMS_DIM> r);
    // For running at half G-vectors with real orbitals;  
    // 0 if the twist is zero, 1 if the twist is G/2.
    TinyVector<int,OHMMS_DIM> HalfG;

    void applyPhaseFactors (cuda_vector<CudaStorageType*> &storageVector,
			    cuda_vector<CudaRealType*> &phi);
    ////////////
    // Timers //
    ////////////
    NewTimer ValueTimer, VGLTimer, VGLMatTimer;
    NewTimer EinsplineTimer;

    // Data for vectorized evaluations
    host_vector<CudaPosType> hostPos, NLhostPos;
    cuda_vector<CudaPosType> cudaPos, NLcudaPos;
    host_vector<CudaRealType> hostSign, NLhostSign;
    cuda_vector<CudaRealType> cudaSign, NLcudaSign;
    // This stores the inverse of the lattice vector matrix in
    // GPU memory.
    cuda_vector<CudaRealType> Linv_cuda, L_cuda;

  public:
    void registerTimers();

    // Real return values
    void evaluate(const ParticleSet& P, int iat, RealValueVector_t& psi);
    void evaluate(const ParticleSet& P, int iat, RealValueVector_t& psi, 
		  RealGradVector_t& dpsi, RealValueVector_t& d2psi);
    void evaluate(const ParticleSet& P, int first, int last,
		  RealValueMatrix_t& psi, RealGradMatrix_t& dpsi, 
		  RealValueMatrix_t& d2psi);
    void evaluate (const ParticleSet& P, PosType r, vector<RealType> &psi);
#if !defined(QMC_COMPLEX)
    // This is the gradient of the orbitals w.r.t. the ion iat
    void evaluateGradSource (const ParticleSet &P, int first, int last, 
			 const ParticleSet &source, int iat_src, 
			 RealGradMatrix_t &gradphi);
    // Evaluate the gradient w.r.t. to ion iat of the gradient and
    // laplacian of the orbitals w.r.t. the electrons
    void evaluateGradSource (const ParticleSet &P, int first, int last, 
			     const ParticleSet &source, int iat_src,
			     RealGradMatrix_t &dphi,
			     RealHessMatrix_t  &dgrad_phi,
			     RealGradMatrix_t &dlaplphi);
    void evaluateGradSource (const ParticleSet &P, int first, int last,
			     const ParticleSet &source, int iat_src, 
			     ComplexGradMatrix_t &gradphi);
#endif

    // Complex return values
    void evaluate(const ParticleSet& P, int iat, ComplexValueVector_t& psi);
    void evaluate(const ParticleSet& P, int iat, ComplexValueVector_t& psi, 
		  ComplexGradVector_t& dpsi, ComplexValueVector_t& d2psi);
    void evaluate(const ParticleSet& P, int first, int last,
		  ComplexValueMatrix_t& psi, ComplexGradMatrix_t& dpsi, 
		  ComplexValueMatrix_t& d2psi);

    // Vectorized evaluation functions
    void evaluate (vector<Walker_t*> &walkers, int iat,
		   cuda_vector<CudaRealType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, int iat,
		   cuda_vector<CudaComplexType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   cuda_vector<CudaRealType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos,
		   cuda_vector<CudaComplexType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   cuda_vector<CudaRealType*> &phi,
		   cuda_vector<CudaRealType*> &grad_lapl,
		   int row_stride);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   cuda_vector<CudaComplexType*> &phi,
		   cuda_vector<CudaComplexType*> &grad_lapl,
		   int row_stride);
    void evaluate (vector<PosType> &pos, cuda_vector<CudaRealType*> &phi);
    void evaluate (vector<PosType> &pos, cuda_vector<CudaComplexType*> &phi);
    
    void resetParameters(const opt_variables_type& active);
    void resetTargetParticleSet(ParticleSet& e);
    void setOrbitalSetSize(int norbs);
    string Type();
    
    SPOSetBase* makeClone() const;
    
    EinsplineSetExtended() : 
      ValueTimer  ("EinsplineSetExtended::ValueOnly"),
      VGLTimer    ("EinsplineSetExtended::VGL"),
      VGLMatTimer ("EinsplineSetExtended::VGLMatrix"),
      EinsplineTimer("libeinspline"),
      MultiSpline(NULL), CudaMultiSpline(NULL)
    {
      className = "EinsplineSetExtended";
      TimerManager.addTimer (&ValueTimer);
      TimerManager.addTimer (&VGLTimer);
      TimerManager.addTimer (&VGLMatTimer);
      TimerManager.addTimer (&EinsplineTimer);
      for (int i=0; i<OHMMS_DIM; i++)
	HalfG[i] = 0;
    }
  };

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

    vector<cuda_vector<CudaRealType> > AtomicSplineCoefs_GPU;
    cuda_vector<AtomicOrbitalCuda<CudaRealType> > AtomicOrbitals_GPU;

    // host_vector<AtomicPolyJob<CudaRealType> >   AtomicPolyJobs_CPU;
    // cuda_vector<AtomicPolyJob<CudaRealType> >   AtomicPolyJobs_GPU;
    // host_vector<AtomicSplineJob<CudaRealType> > AtomicSplineJobs_CPU;
    // cuda_vector<AtomicSplineJob<CudaRealType> > AtomicSplineJobs_GPU;

    cuda_vector<HybridJobType> HybridJobs_GPU;
    cuda_vector<CudaRealType>  IonPos_GPU;
    cuda_vector<CudaRealType>  CutoffRadii_GPU, PolyRadii_GPU;
    cuda_vector<HybridDataFloat> HybridData_GPU;

    cuda_vector<CudaRealType> Ylm_GPU;
    cuda_vector<CudaRealType*> Ylm_ptr_GPU, dYlm_dtheta_ptr_GPU, dYlm_dphi_ptr_GPU;
    host_vector<CudaRealType*> Ylm_ptr_CPU, dYlm_dtheta_ptr_CPU, dYlm_dphi_ptr_CPU;
    cuda_vector<CudaRealType> rhats_GPU;
    host_vector<CudaRealType> rhats_CPU;
    cuda_vector<int> JobType;
    
    // Vectors for 3D Bspline evaluation
    cuda_vector<CudaRealType> BsplinePos_GPU;
    host_vector<CudaRealType> BsplinePos_CPU;
    cuda_vector<CudaStorageType*> BsplineVals_GPU, BsplineGradLapl_GPU;
    host_vector<CudaStorageType*> BsplineVals_CPU, BsplineGradLapl_CPU;

    // The maximum lMax across all atomic orbitals
    int lMax;
    int numlm, Ylm_BS;
    // Stores the maximum number of walkers that can be handled by currently
    // allocated GPU memory.  Must resize if we have more walkers than this.
    int CurrentWalkers;

    cuda_vector<CudaRealType> YlmData;
    //////////////////////////////
    /// Orbital storage objects //
    //////////////////////////////

    ////////////
    // Timers //
    ////////////
    // Data for vectorized evaluations

    void sort_electrons(vector<PosType> &pos);

  public:
    void init_cuda();
    //    void registerTimers();

    // Resize cuda objects
    void resize_cuda(int numwalkers);

    // Vectorized evaluation functions
    void evaluate (vector<Walker_t*> &walkers, int iat,
		   cuda_vector<CudaRealType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, int iat,
		   cuda_vector<CudaComplexType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   cuda_vector<CudaRealType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos,
		   cuda_vector<CudaComplexType*> &phi);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   cuda_vector<CudaRealType*> &phi,
		   cuda_vector<CudaRealType*> &grad_lapl,
		   int row_stride);
    void evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
		   cuda_vector<CudaComplexType*> &phi,
		   cuda_vector<CudaComplexType*> &grad_lapl,
		   int row_stride);
    void evaluate (vector<PosType> &pos, cuda_vector<CudaRealType*> &phi);
    void evaluate (vector<PosType> &pos, cuda_vector<CudaComplexType*> &phi);
    
    string Type();
    
    SPOSetBase* makeClone() const;
    
    EinsplineSetHybrid();
  };


  template<typename StorageType>
  inline void EinsplineSetExtended<StorageType>::computePhaseFactors(TinyVector<RealType,OHMMS_DIM> r)
  {
    for (int i=0; i<kPoints.size(); i++) phase[i] = -dot(r, kPoints[i]);
    eval_e2iphi(phase,eikr);
//#ifdef HAVE_MKL
//    for (int i=0; i<kPoints.size(); i++) 
//      phase[i] = -dot(r, kPoints[i]);
//    vzCIS(OrbitalSetSize, phase, (RealType*)eikr.data());
//#else
//    RealType s, c;
//    for (int i=0; i<kPoints.size(); i++) {
//      phase[i] = -dot(r, kPoints[i]);
//      sincos (phase[i], &s, &c);
//      eikr[i] = complex<RealType>(c,s);
//    }
//#endif
  }
  

  
}
#endif
