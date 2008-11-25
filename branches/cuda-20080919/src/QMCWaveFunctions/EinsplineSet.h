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
#include "QMCWaveFunctions/BasisSetBase.h"
#include "QMCWaveFunctions/SPOSetBase.h"
#include "Optimize/VarList.h"
#include "QMCWaveFunctions/EinsplineOrb.h"
#include "QMCWaveFunctions/MuffinTin.h"
#include "Utilities/NewTimer.h"
#include <einspline/multi_bspline_structs.h>
#include "Configuration.h"
#include "Numerics/e2iphi.h"

#include <einspline/multi_bspline_create_cuda.h>


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
    typedef typename MultiOrbitalTraits<StorageType,OHMMS_DIM>::SplineType SplineType; 
    typedef typename MultiOrbitalTraits<CUDA_PRECISION,OHMMS_DIM>::CudaSplineType CudaSplineType; 
    typedef typename OrbitalSetTraits<StorageType>::ValueVector_t StorageValueVector_t;
    typedef typename OrbitalSetTraits<StorageType>::GradVector_t  StorageGradVector_t;
    typedef typename OrbitalSetTraits<StorageType>::HessVector_t  StorageHessVector_t;
    typedef Vector<RealType>                                        RealValueVector_t;
    typedef Vector<complex<RealType> >                              ComplexValueVector_t;
    typedef Vector<TinyVector<RealType,OHMMS_DIM> >                 RealGradVector_t;
    typedef Vector<TinyVector<complex<RealType>,OHMMS_DIM> >        ComplexGradVector_t;
    typedef Vector<Tensor<RealType,OHMMS_DIM> >                     RealHessVector_t;
    typedef Vector<Tensor<complex<RealType>,OHMMS_DIM> >            ComplexHessVector_t;
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
       
    /////////////////////////////
    /// Orbital storage object //
    /////////////////////////////
    SplineType *MultiSpline;
    CudaSplineType *CudaMultiSpline;
    // Temporary storage for Eispline calls
    StorageValueVector_t StorageValueVector, StorageLaplVector;
    StorageGradVector_t  StorageGradVector;
    StorageHessVector_t  StorageHessVector;
    // Temporary storage used when blending functions        
    StorageValueVector_t BlendValueVector, BlendLaplVector;   
    StorageGradVector_t BlendGradVector;
        
    // True if we should unpack this orbital into two copies
    vector<bool>         MakeTwoCopies;
    // k-points for each orbital
    Vector<TinyVector<RealType,OHMMS_DIM> > kPoints;

    ///////////////////
    // Phase factors //
    ///////////////////
    Vector<RealType> phase;
    Vector<complex<RealType> > eikr;
    inline void computePhaseFactors(TinyVector<RealType,OHMMS_DIM> r);

    ////////////
    // Timers //
    ////////////
    NewTimer ValueTimer, VGLTimer, VGLMatTimer;
    NewTimer EinsplineTimer;

    // Data for vectorized evaluations
    host_vector<CudaPosType> hostPos;
    cuda_vector<CudaPosType> cudaPos;
    // This stores the inverse of the lattice vector matrix in
    // GPU memory.
    cuda_vector<CudaRealType> Linv_cuda;
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
    }
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
