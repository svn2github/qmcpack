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
/** @file EinsplineSetExtended.h
 *
 * Declaration of EinsplineSetExtended<StorageType>, a derived class from EinsplineSet
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SET_EXTENDED_H
#define QMCPLUSPLUS_EINSPLINE_SET_EXTENDED_H

#include <QMCWaveFunctions/EinsplineSet.h>

namespace qmcplusplus {

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
    typedef typename OrbitalSetTraits<StorageType>::ValueVector_t StorageValueVector_t;
    typedef typename OrbitalSetTraits<StorageType>::GradVector_t  StorageGradVector_t;
    typedef typename OrbitalSetTraits<StorageType>::HessVector_t  StorageHessVector_t;
    typedef Vector<double>                                        RealValueVector_t;
    typedef Vector<complex<double> >                              ComplexValueVector_t;
    typedef Vector<TinyVector<double,OHMMS_DIM> >                 RealGradVector_t;
    typedef Vector<TinyVector<complex<double>,OHMMS_DIM> >        ComplexGradVector_t;
    typedef Tensor<double,OHMMS_DIM>                            RealHessType;
    typedef Tensor<complex<double>,OHMMS_DIM>                   ComplexHessType;
    typedef Vector<RealHessType>                                  RealHessVector_t;
    typedef Matrix<RealHessType>                                  RealHessMatrix_t;
    typedef Vector<ComplexHessType>                               ComplexHessVector_t;
    typedef Matrix<ComplexHessType>                               ComplexHessMatrix_t;
    typedef Matrix<double>                                        RealValueMatrix_t;
    typedef Matrix<complex<double> >                              ComplexValueMatrix_t;
    typedef Matrix<TinyVector<double,OHMMS_DIM> >                 RealGradMatrix_t;
    typedef Matrix<TinyVector<complex<double>,OHMMS_DIM> >        ComplexGradMatrix_t;
    typedef TinyVector<RealHessType, 3>                           RealGGGType;
    typedef Vector<RealGGGType>                                   RealGGGVector_t;
    typedef Matrix<RealGGGType>                                   RealGGGMatrix_t;
    typedef TinyVector<ComplexHessType, 3>                        ComplexGGGType;
    typedef Vector<ComplexGGGType>                                ComplexGGGVector_t;
    typedef Matrix<ComplexGGGType>                                ComplexGGGMatrix_t;
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

    //////////////////////////////////////
    // Radial/Ylm orbitals around atoms //
    //////////////////////////////////////
    vector<AtomicOrbital<StorageType> > AtomicOrbitals;


    // First-order derivative w.r.t. the ion positions
    vector<TinyVector<SplineType*,OHMMS_DIM> > FirstOrderSplines;
    // Temporary storage for Eispline calls
    StorageValueVector_t StorageValueVector, StorageLaplVector;
    StorageGradVector_t  StorageGradVector;
    StorageHessVector_t  StorageHessVector;
    // Temporary storage used when blending functions        
    StorageValueVector_t BlendValueVector, BlendLaplVector;   
    StorageGradVector_t BlendGradVector;
    StorageHessVector_t  BlendHessVector;
        
    // True if we should unpack this orbital into two copies
    vector<bool>         MakeTwoCopies;
    // k-points for each orbital
    Vector<TinyVector<double,OHMMS_DIM> > kPoints;

    ///////////////////
    // Phase factors //
    ///////////////////
    Vector<double> phase;
    Vector<complex<double> > eikr;
    inline void computePhaseFactors(TinyVector<double,OHMMS_DIM> r);
    // For running at half G-vectors with real orbitals;  
    // 0 if the twist is zero, 1 if the twist is G/2.
    TinyVector<int,OHMMS_DIM> HalfG;

    ////////////
    // Timers //
    ////////////
    NewTimer ValueTimer, VGLTimer, VGLMatTimer;
    NewTimer EinsplineTimer;

#if defined(QMC_CUDA)
    // Cuda equivalents of the above
    typedef typename StorageTypeConverter<StorageType,CUDA_PRECISION>::CudaStorageType CudaStorageType;
    typedef typename MultiOrbitalTraits<CudaStorageType,OHMMS_DIM>::CudaSplineType CudaSplineType; 

    CudaSplineType *CudaMultiSpline;
    gpu::device_vector<CudaStorageType> CudaValueVector, CudaGradLaplVector;
    gpu::device_vector<CudaStorageType*> CudaValuePointers, CudaGradLaplPointers;
    void resize_cuda(int numWalkers);
    // Cuda equivalent
    gpu::device_vector<int> CudaMakeTwoCopies;
    // Cuda equivalent
    gpu::device_vector<TinyVector<CUDA_PRECISION,OHMMS_DIM > > CudakPoints,
      CudakPoints_reduced;
    void applyPhaseFactors (gpu::device_vector<CudaStorageType*> &storageVector,
			    gpu::device_vector<CudaRealType*> &phi);
    // Data for vectorized evaluations
    gpu::host_vector<CudaPosType> hostPos, NLhostPos;
    gpu::device_vector<CudaPosType> cudapos, NLcudapos;
    gpu::host_vector<CudaRealType> hostSign, NLhostSign;
    gpu::device_vector<CudaRealType> cudaSign, NLcudaSign;
    // This stores the inverse of the lattice vector matrix in
    // GPU memory.
    gpu::device_vector<CudaRealType> Linv_cuda, L_cuda;
#endif

  public:
    void registerTimers();
    PosType get_k(int orb) { return kPoints[orb]; }

    // Real return values
    void evaluate(const ParticleSet& P, int iat, RealValueVector_t& psi);
    void evaluate(const ParticleSet& P, int iat, RealValueVector_t& psi, 
		  RealGradVector_t& dpsi, RealValueVector_t& d2psi);
    void evaluate(const ParticleSet& P, int iat, RealValueVector_t& psi, 
		  RealGradVector_t& dpsi, RealHessVector_t& grad_grad_psi);
    void evaluate_notranspose(const ParticleSet& P, int first, int last,
		  RealValueMatrix_t& psi, RealGradMatrix_t& dpsi, 
		  RealValueMatrix_t& d2psi);
    void evaluate_notranspose(const ParticleSet& P, int first, int last,
                  RealValueMatrix_t& psi, RealGradMatrix_t& dpsi,
                  RealHessMatrix_t& grad_grad_psi);
    void evaluate_notranspose(const ParticleSet& P, int first, int last,
                  RealValueMatrix_t& psi, RealGradMatrix_t& dpsi,
                  RealHessMatrix_t& grad_grad_psi,
                  RealGGGMatrix_t& grad_grad_grad_logdet);

    //    void evaluate (const ParticleSet& P, const PosType& r, vector<double> &psi);
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
    void evaluate(const ParticleSet& P, int iat, ComplexValueVector_t& psi, 
		  ComplexGradVector_t& dpsi, ComplexHessVector_t& grad_grad_psi);
    void evaluate_notranspose(const ParticleSet& P, int first, int last,
		  ComplexValueMatrix_t& psi, ComplexGradMatrix_t& dpsi, 
		  ComplexValueMatrix_t& d2psi);
    void evaluate_notranspose(const ParticleSet& P, int first, int last,
                  ComplexValueMatrix_t& psi, ComplexGradMatrix_t& dpsi,
                  ComplexHessMatrix_t& grad_grad_psi);
    void evaluate_notranspose(const ParticleSet& P, int first, int last,
                  ComplexValueMatrix_t& psi, ComplexGradMatrix_t& dpsi,
                  ComplexHessMatrix_t& grad_grad_psi,
                  ComplexGGGMatrix_t& grad_grad_grad_logdet); 
#if defined(QMC_CUDA)
    void initGPU();

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
#endif
    
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
      MultiSpline(NULL)
#ifdef QMC_CUDA
      , CudaMultiSpline(NULL),
      cudapos("EinsplineSetExtended::cudapos"),
      NLcudapos("EinsplineSetExtended::NLcudapos"),
      cudaSign("EinsplineSetExtended::cudaSign"),
      NLcudaSign("EinsplineSetExtended::NLcudaSign"),
      Linv_cuda("EinsplineSetExtended::Linv_cuda"),
      L_cuda("EinsplineSetExtended::L_cuda"),
      CudaValueVector("EinsplineSetExtended::CudaValueVector"),
      CudaGradLaplVector("EinsplineSetExtended::CudaGradLaplVector"),
      CudaValuePointers("EinsplineSetExtended::CudaValuePointers"),
      CudaGradLaplPointers("EinsplineSetExtended::CudaGradLaplPointers"),
      CudaMakeTwoCopies("EinsplineSetExtended::CudaMakeTwoCopies"),
      CudakPoints("EinsplineSetExtended::CudakPoints"),
      CudakPoints_reduced("EinsplineSetExtended::CudakPoints_reduced")
#endif
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

}

#endif
/***************************************************************************
* $RCSfile$   $Author: jeongnim.kim $
* $Revision: 5119 $   $Date: 2011-02-06 16:20:47 -0600 (Sun, 06 Feb 2011) $
* $Id: EinsplineSetExtended.h 5119 2011-02-06 22:20:47Z jeongnim.kim $
***************************************************************************/
