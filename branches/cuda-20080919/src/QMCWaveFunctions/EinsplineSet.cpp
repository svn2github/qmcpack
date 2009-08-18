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

#include "QMCWaveFunctions/EinsplineSet.h"
#include <einspline/multi_bspline.h>
#include <einspline/multi_bspline_eval_cuda.h>
#include "Configuration.h"
#include "AtomicOrbitalCuda.h"
#ifdef HAVE_MKL
  #include <mkl_vml.h>
#endif

void apply_phase_factors(float kPoints[], int makeTwoCopies[], 
			 float pos[], float *phi_in[], float *phi_out[], 
			 int num_splines, int num_walkers);

void apply_phase_factors(float kPoints[], int makeTwoCopies[], 
			 float pos[], float *phi_in[], float *phi_out[], 
			 float *GL_in[], float *GL_out[],
			 int num_splines, int num_walkers, int row_stride);

namespace qmcplusplus {
 
  // Real CUDA routines
//   inline void
//   eval_multi_multi_UBspline_3d_cuda (multi_UBspline_3d_s_cuda *spline, 
// 				     float *pos, float *phi[], int N)
//   { eval_multi_multi_UBspline_3d_s_cuda  (spline, pos, phi, N); }

  inline void
  eval_multi_multi_UBspline_3d_cuda (multi_UBspline_3d_s_cuda *spline, 
				     float *pos, float *sign, float *phi[], int N)
  { eval_multi_multi_UBspline_3d_s_sign_cuda  (spline, pos, sign, phi, N); }


  inline void
  eval_multi_multi_UBspline_3d_cuda (multi_UBspline_3d_d_cuda *spline, 
				     double *pos, double *phi[], int N)
  { eval_multi_multi_UBspline_3d_d_cuda  (spline, pos, phi, N); }


//   inline void eval_multi_multi_UBspline_3d_vgl_cuda 
//   (multi_UBspline_3d_s_cuda *spline, float *pos, float Linv[],
//    float *phi[], float *grad_lapl[], int N, int row_stride)
//   {
//     eval_multi_multi_UBspline_3d_s_vgl_cuda
//       (spline, pos, Linv, phi, grad_lapl, N, row_stride);
//   }

  inline void eval_multi_multi_UBspline_3d_vgl_cuda 
  (multi_UBspline_3d_s_cuda *spline, float *pos, float *sign, float Linv[],
   float *phi[], float *grad_lapl[], int N, int row_stride)
  {
    eval_multi_multi_UBspline_3d_s_vgl_sign_cuda
      (spline, pos, sign, Linv, phi, grad_lapl, N, row_stride);
  }


  inline void eval_multi_multi_UBspline_3d_vgl_cuda 
  (multi_UBspline_3d_d_cuda *spline, double *pos, double Linv[],
   double *phi[], double *grad_lapl[], int N, int row_stride)
  {
    eval_multi_multi_UBspline_3d_d_vgl_cuda
      (spline, pos, Linv, phi, grad_lapl, N, row_stride);
  }

  // Complex CUDA routines
  inline void
  eval_multi_multi_UBspline_3d_cuda (multi_UBspline_3d_c_cuda *spline, 
				     float *pos, complex<float> *phi[], int N)
  { eval_multi_multi_UBspline_3d_c_cuda  (spline, pos, phi, N); }

  inline void
  eval_multi_multi_UBspline_3d_cuda (multi_UBspline_3d_z_cuda *spline, 
				     double *pos, complex<double> *phi[], 
				     int N)
  { eval_multi_multi_UBspline_3d_z_cuda  (spline, pos, phi, N); }


  inline void eval_multi_multi_UBspline_3d_vgl_cuda 
  (multi_UBspline_3d_c_cuda *spline, float *pos, float Linv[],
   complex<float> *phi[], complex<float> *grad_lapl[], int N, int row_stride)
  {
    eval_multi_multi_UBspline_3d_c_vgl_cuda
      (spline, pos, Linv, phi, grad_lapl, N, row_stride);
  }

  inline void eval_multi_multi_UBspline_3d_vgl_cuda 
  (multi_UBspline_3d_z_cuda *spline, double *pos, double Linv[],
   complex<double> *phi[], complex<double> *grad_lapl[], int N, int row_stride)
  {
    eval_multi_multi_UBspline_3d_z_vgl_cuda
      (spline, pos, Linv, phi, grad_lapl, N, row_stride);
  }




  EinsplineSet::UnitCellType
  EinsplineSet::GetLattice()
  {
    return SuperLattice;
  }
  
  void
  EinsplineSet::resetTargetParticleSet(ParticleSet& e)
  {
  }

  void
  EinsplineSet::resetSourceParticleSet(ParticleSet& ions)
  {
  }
  
  void
  EinsplineSet::setOrbitalSetSize(int norbs)
  {
    OrbitalSetSize = norbs;
  }
  
  void 
  EinsplineSet::evaluate (const ParticleSet& P, int iat, ValueVector_t& psi)
  {
    app_error() << "Should never instantiate EinsplineSet.\n";
    abort();
  }

  void 
  EinsplineSet::evaluate (const ParticleSet& P, int iat, 
			  ValueVector_t& psi, GradVector_t& dpsi, 
			  ValueVector_t& d2psi)
  {
    app_error() << "Should never instantiate EinsplineSet.\n";
    abort();
  }

  
  void 
  EinsplineSet::evaluate (const ParticleSet& P, int first, int last,
			  ValueMatrix_t& vals, GradMatrix_t& grads, 
			  ValueMatrix_t& lapls)
  {
    app_error() << "Should never instantiate EinsplineSet.\n";
    abort();
  }


  void 
  EinsplineSetLocal::evaluate (const ParticleSet& P, int iat, 
			       ValueVector_t& psi)
  {
    PosType r (P.R[iat]);
    PosType ru(PrimLattice.toUnit(P.R[iat]));
    ru[0] -= std::floor (ru[0]);
    ru[1] -= std::floor (ru[1]);
    ru[2] -= std::floor (ru[2]);
    for(int j=0; j<OrbitalSetSize; j++) {
      complex<double> val;
      Orbitals[j]->evaluate(ru, val);

      double phase = -dot(r, Orbitals[j]->kVec);
      double s,c;
      sincos (phase, &s, &c);
      complex<double> e_mikr (c,s);
      val *= e_mikr;
#ifdef QMC_COMPLEX
      psi[j] = val;
#else
      psi[j] = real(val);
#endif
    }
  }
  
  void 
  EinsplineSetLocal::evaluate (const ParticleSet& P, int iat, 
			       ValueVector_t& psi, GradVector_t& dpsi, 
			       ValueVector_t& d2psi)
  {
    PosType r (P.R[iat]);
    PosType ru(PrimLattice.toUnit(P.R[iat]));
    ru[0] -= std::floor (ru[0]);
    ru[1] -= std::floor (ru[1]);
    ru[2] -= std::floor (ru[2]);
    complex<double> val;
    TinyVector<complex<double>,3> gu;
    Tensor<complex<double>,3> hess;
    complex<double> eye (0.0, 1.0);
    for(int j=0; j<OrbitalSetSize; j++) {
      complex<double> u;
      TinyVector<complex<double>,3> gradu;
      complex<double> laplu;

      Orbitals[j]->evaluate(ru, val, gu, hess);
      u  = val;
      // Compute gradient in cartesian coordinates
      gradu = dot(PrimLattice.G, gu);
      laplu = trace(hess, GGt);      
      
      PosType k = Orbitals[j]->kVec;
      TinyVector<complex<double>,3> ck;
      ck[0]=k[0];  ck[1]=k[1];  ck[2]=k[2];
      double s,c;
      double phase = -dot(P.R[iat], k);
      sincos (phase, &s, &c);
      complex<double> e_mikr (c,s);
#ifdef QMC_COMPLEX
      psi[j]   = e_mikr * u;
      dpsi[j]  = e_mikr*(-eye * ck * u + gradu);
      d2psi[j] = e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu);
#else
      psi[j]   = real(e_mikr * u);
      dpsi[j]  = real(e_mikr*(-eye * ck * u + gradu));
      d2psi[j] = real(e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu));
#endif

    }
  }
  
  void 
  EinsplineSetLocal::evaluate (const ParticleSet& P, int first, int last,
			       ValueMatrix_t& vals, GradMatrix_t& grads, 
			       ValueMatrix_t& lapls)
  {
    for(int iat=first,i=0; iat<last; iat++,i++) {
      PosType r (P.R[iat]);
      PosType ru(PrimLattice.toUnit(r));
      ru[0] -= std::floor (ru[0]);
      ru[1] -= std::floor (ru[1]);
      ru[2] -= std::floor (ru[2]);
      complex<double> val;
      TinyVector<complex<double>,3> gu;
      Tensor<complex<double>,3> hess;
      complex<double> eye (0.0, 1.0);
      for(int j=0; j<OrbitalSetSize; j++) {
	complex<double> u;
	TinyVector<complex<double>,3> gradu;
	complex<double> laplu;

	Orbitals[j]->evaluate(ru, val, gu, hess);
	u  = val;
	gradu = dot(PrimLattice.G, gu);
	laplu = trace(hess, GGt);
	
	PosType k = Orbitals[j]->kVec;
	TinyVector<complex<double>,3> ck;
	ck[0]=k[0];  ck[1]=k[1];  ck[2]=k[2];
	double s,c;
	double phase = -dot(r, k);
	sincos (phase, &s, &c);
	complex<double> e_mikr (c,s);
#ifdef QMC_COMPLEX
	vals(j,i)  = e_mikr * u;
	grads(i,j) = e_mikr*(-eye*u*ck + gradu);
	lapls(i,j) = e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu);
#else
	vals(j,i)  = real(e_mikr * u);
	grads(i,j) = real(e_mikr*(-eye*u*ck + gradu));
	lapls(i,j) = real(e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu));
#endif

      }
    }
  }

  string 
  EinsplineSet::Type()
  {
    return "EinsplineSet";
  }




  ///////////////////////////////////////////
  // EinsplineSetExtended Member functions //
  ///////////////////////////////////////////

  inline void
  convert (complex<double> a, complex<double> &b)
  { b = a;  }

  inline void
  convert (complex<double> a, double &b)
  { b = a.real();  }

  inline void
  convert (double a, complex<double>&b)
  { b = complex<double>(a,0.0); }

  inline void
  convert (double a, double &b)
  { b = a; }

  template<typename T1, typename T2> void
  convertVec (TinyVector<T1,3> a, TinyVector<T2,3> &b)
  {
    for (int i=0; i<3; i++)
      convert (a[i], b[i]);
  }

  //////////////////////
  // Double precision //
  //////////////////////

  // Real evaluation functions
  inline void 
  EinsplineMultiEval (multi_UBspline_3d_d *restrict spline,
		      TinyVector<double,3> r, 
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
		      TinyVector<double,3> r,
		      Vector<double> &psi,
		      Vector<TinyVector<double,3> > &grad,
		      Vector<Tensor<double,3> > &hess)
  {
    eval_multi_UBspline_3d_d_vgh (spline, r[0], r[1], r[2],
				  psi.data(), 
				  (double*)grad.data(), 
				  (double*)hess.data());
  }

  // Complex evaluation functions 
  inline void 
  EinsplineMultiEval (multi_UBspline_3d_z *restrict spline,
		      TinyVector<double,3> r, 
		      Vector<complex<double> > &psi)
  {
    eval_multi_UBspline_3d_z (spline, r[0], r[1], r[2], psi.data());
  }


  inline void
  EinsplineMultiEval (multi_UBspline_3d_z *restrict spline,
		      TinyVector<double,3> r,
		      Vector<complex<double> > &psi,
		      Vector<TinyVector<complex<double>,3> > &grad,
		      Vector<Tensor<complex<double>,3> > &hess)
  {
    eval_multi_UBspline_3d_z_vgh (spline, r[0], r[1], r[2],
				  psi.data(), 
				  (complex<double>*)grad.data(), 
				  (complex<double>*)hess.data());
  }

  //////////////////////
  // Single precision //
  //////////////////////

  // Real evaluation functions
  inline void 
  EinsplineMultiEval (multi_UBspline_3d_s *restrict spline,
		      TinyVector<float,3> r, 
		      Vector<float> &psi)
  {
    eval_multi_UBspline_3d_s (spline, r[0], r[1], r[2], psi.data());
  }

  inline void
  EinsplineMultiEval (multi_UBspline_3d_s *restrict spline,
		      TinyVector<float,3> r,
		      Vector<float> &psi,
		      Vector<TinyVector<float,3> > &grad,
		      Vector<Tensor<float,3> > &hess)
  {
    eval_multi_UBspline_3d_s_vgh (spline, r[0], r[1], r[2],
				  psi.data(), 
				  (float*)grad.data(), 
				  (float*)hess.data());
  }

  // Complex evaluation functions 

  inline void 
  EinsplineMultiEval (multi_UBspline_3d_c *restrict spline,
		      TinyVector<float,3> r, 
		      Vector<complex<float> > &psi)
  {
    eval_multi_UBspline_3d_c (spline, r[0], r[1], r[2], psi.data());
  }


  inline void
  EinsplineMultiEval (multi_UBspline_3d_c *restrict spline,
		      TinyVector<float,3> r,
		      Vector<complex<float> > &psi,
		      Vector<TinyVector<complex<float>,3> > &grad,
		      Vector<Tensor<complex<float>,3> > &hess)
  {
    eval_multi_UBspline_3d_c_vgh (spline, r[0], r[1], r[2],
				  psi.data(), 
				  (complex<float>*)grad.data(), 
				  (complex<float>*)hess.data());
  }



			   

  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::resetParameters(const opt_variables_type& active)
  {

  }

  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::resetTargetParticleSet(ParticleSet& e)
  {
  }

  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::setOrbitalSetSize(int norbs)
  {
    OrbitalSetSize = norbs;
  }
  
  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::evaluate
  (const ParticleSet& P, int iat, RealValueVector_t& psi)
  {
    ValueTimer.start();
    PosType r (P.R[iat]);

    // Do core states first
    int icore = NumValenceOrbs;
    for (int tin=0; tin<MuffinTins.size(); tin++) {
      MuffinTins[tin].evaluateCore(r, StorageValueVector, icore);
      icore += MuffinTins[tin].get_num_core();
    }
    // Add phase to core orbitals
    for (int j=NumValenceOrbs; j<StorageValueVector.size(); j++) {
      PosType k = kPoints[j];
      double s,c;
      double phase = -dot(r, k);
      sincos (phase, &s, &c);
      complex<double> e_mikr (c,s);
      StorageValueVector[j] *= e_mikr;
    }

    // Check if we are inside a muffin tin.  If so, compute valence
    // states in the muffin tin.
    bool inTin = false;
    bool need2blend = false;
    double b(0.0);
    for (int tin=0; tin<MuffinTins.size() && !inTin; tin++) {
      MuffinTins[tin].inside(r, inTin, need2blend);
      if (inTin) {
	MuffinTins[tin].evaluate (r, StorageValueVector);
	if (need2blend) {
	  PosType disp = MuffinTins[tin].disp(r);
	  double dr = std::sqrt(dot(disp, disp));
	  MuffinTins[tin].blend_func(dr, b);
	}
	break;
      }
    }

    StorageValueVector_t &valVec = 
      need2blend ? BlendValueVector : StorageValueVector;

    if (!inTin || need2blend) {
      PosType ru(PrimLattice.toUnit(P.R[iat]));
      for (int i=0; i<OHMMS_DIM; i++)
	ru[i] -= std::floor (ru[i]);
      EinsplineTimer.start();
      EinsplineMultiEval (MultiSpline, ru, valVec);
      EinsplineTimer.stop();
      // Add e^ikr phase to B-spline orbitals
      for (int j=0; j<NumValenceOrbs; j++) {
	PosType k = kPoints[j];
	double s,c;
	double phase = -dot(r, k);
	sincos (phase, &s, &c);
	complex<double> e_mikr (c,s);
	valVec[j] *= e_mikr;
      }
    }
    int N = StorageValueVector.size();

    // If we are in a muffin tin, don't add the e^ikr term
    // We should add it to the core states, however

    if (need2blend) {
      int psiIndex = 0;
      for (int j=0; j<N; j++) {
	complex<double> psi1 = StorageValueVector[j];
	complex<double> psi2 =   BlendValueVector[j];
	
	complex<double> psi_val = b * psi1 + (1.0-b) * psi2;

	psi[psiIndex] = real(psi_val);
	psiIndex++;
	if (MakeTwoCopies[j]) {
	  psi[psiIndex] = imag(psi_val);
	  psiIndex++;
	}
      }

    }
    else {
      int psiIndex = 0;
      for (int j=0; j<N; j++) {
	complex<double> psi_val = StorageValueVector[j];
	psi[psiIndex] = real(psi_val);
	psiIndex++;
	if (MakeTwoCopies[j]) {
	  psi[psiIndex] = imag(psi_val);
	  psiIndex++;
	}
      }
    }
    ValueTimer.stop();
  }


  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::evaluate
  (const ParticleSet& P, int iat, ComplexValueVector_t& psi)
  {
    ValueTimer.start();
    PosType r (P.R[iat]);
    PosType ru(PrimLattice.toUnit(P.R[iat]));
    for (int i=0; i<OHMMS_DIM; i++)
      ru[i] -= std::floor (ru[i]);
    EinsplineTimer.start();
    EinsplineMultiEval (MultiSpline, ru, StorageValueVector);
    EinsplineTimer.stop();
    //computePhaseFactors(r);
    for (int i=0; i<psi.size(); i++) {
      PosType k = kPoints[i];
      double s,c;
      double phase = -dot(r, k);
      sincos (phase, &s, &c);
      complex<double> e_mikr (c,s);
      convert (e_mikr*StorageValueVector[i], psi[i]);
    }
    ValueTimer.stop();
  }

  // This is an explicit specialization of the above for real orbitals
  // with a real return value, i.e. simulations at the gamma or L 
  // point.
  template<> void
  EinsplineSetExtended<double>::evaluate
  (const ParticleSet &P, int iat, RealValueVector_t& psi)
  {
    ValueTimer.start();
    PosType r (P.R[iat]);
    PosType ru(PrimLattice.toUnit(P.R[iat]));
    int image[OHMMS_DIM];
    for (int i=0; i<OHMMS_DIM; i++) {
      RealType img = std::floor(ru[i]);
      ru[i] -= img;
      image[i] = (int) img;
    }
    EinsplineTimer.start();
    EinsplineMultiEval (MultiSpline, ru, psi);
    EinsplineTimer.stop();
    int sign = 0;
    for (int i=0; i<OHMMS_DIM; i++) 
      sign += HalfG[i] * image[i];
    if (sign & 1) 
      for (int j=0; j<psi.size(); j++)
	psi[j] *= -1.0;
    
    ValueTimer.stop();
  }

  template<> void
  EinsplineSetExtended<double>::evaluate
  (const ParticleSet &P, PosType r, vector<RealType> &psi)
  {
    ValueTimer.start();
    PosType ru(PrimLattice.toUnit(r));
    int image[OHMMS_DIM];
    for (int i=0; i<OHMMS_DIM; i++) {
      RealType img = std::floor(ru[i]);
      ru[i] -= img;
      image[i] = (int) img;
    }
    EinsplineTimer.start();
    EinsplineMultiEval (MultiSpline, ru, psi);
    EinsplineTimer.stop();
    int sign=0;
    for (int i=0; i<OHMMS_DIM; i++) 
      sign += HalfG[i]*image[i];
    if (sign & 1) 
      for (int j=0; j<psi.size(); j++)
	psi[j] *= -1.0;
    ValueTimer.stop();
  }

  template<> void
  EinsplineSetExtended<complex<double> >::evaluate
  (const ParticleSet &P, PosType r, vector<RealType> &psi)
  {
    cerr << "Not Implemented.\n";
  }


  // Value, gradient, and laplacian
  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::evaluate
  (const ParticleSet& P, int iat, RealValueVector_t& psi, 
   RealGradVector_t& dpsi, RealValueVector_t& d2psi)
  {
    VGLTimer.start();
    PosType r (P.R[iat]);
    complex<double> eye (0.0, 1.0);
    
    // Do core states first
    int icore = NumValenceOrbs;
    for (int tin=0; tin<MuffinTins.size(); tin++) {
      MuffinTins[tin].evaluateCore(r, StorageValueVector, StorageGradVector, 
				   StorageLaplVector, icore);
      icore += MuffinTins[tin].get_num_core();
    }
    
    // Add phase to core orbitals
    for (int j=NumValenceOrbs; j<StorageValueVector.size(); j++) {
      complex<double> u = StorageValueVector[j];
      TinyVector<complex<double>,OHMMS_DIM> gradu = StorageGradVector[j];
      complex<double> laplu = StorageLaplVector[j];
      PosType k = kPoints[j];
      TinyVector<complex<double>,OHMMS_DIM> ck;
      for (int n=0; n<OHMMS_DIM; n++)	  ck[n] = k[n];
      double s,c;
      double phase = -dot(r, k);
      sincos (phase, &s, &c);
      complex<double> e_mikr (c,s);
      StorageValueVector[j] = e_mikr*u;
      StorageGradVector[j]  = e_mikr*(-eye*u*ck + gradu);
      StorageLaplVector[j]  = e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu);
    }

    // Check muffin tins;  if inside evaluate the orbitals
    bool inTin = false;
    bool need2blend = false;
    PosType disp;
    double b, db, d2b;
    for (int tin=0; tin<MuffinTins.size(); tin++) {
      MuffinTins[tin].inside(r, inTin, need2blend);
      if (inTin) {
	MuffinTins[tin].evaluate (r, StorageValueVector, StorageGradVector, StorageLaplVector);
	if (need2blend) {
	  disp = MuffinTins[tin].disp(r);
	  double dr = std::sqrt(dot(disp, disp));
	  MuffinTins[tin].blend_func(dr, b, db, d2b);
	}
	break;
      }
    }

    StorageValueVector_t &valVec =  
      need2blend ? BlendValueVector : StorageValueVector;
    StorageGradVector_t &gradVec =  
      need2blend ? BlendGradVector : StorageGradVector;
    StorageValueVector_t &laplVec =  
      need2blend ? BlendLaplVector : StorageLaplVector;

    // Otherwise, evaluate the B-splines
    if (!inTin || need2blend) {
      PosType ru(PrimLattice.toUnit(P.R[iat]));
      for (int i=0; i<OHMMS_DIM; i++)
	ru[i] -= std::floor (ru[i]);
      EinsplineTimer.start();
      EinsplineMultiEval (MultiSpline, ru, valVec, gradVec, StorageHessVector);
      EinsplineTimer.stop();
      for (int j=0; j<NumValenceOrbs; j++) {
	gradVec[j] = dot (PrimLattice.G, gradVec[j]);
	laplVec[j] = trace (StorageHessVector[j], GGt);
      }
      // Add e^-ikr phase to B-spline orbitals
      for (int j=0; j<NumValenceOrbs; j++) {
	complex<double> u = valVec[j];
	TinyVector<complex<double>,OHMMS_DIM> gradu = gradVec[j];
	complex<double> laplu = laplVec[j];
	PosType k = kPoints[j];
	TinyVector<complex<double>,OHMMS_DIM> ck;
	for (int n=0; n<OHMMS_DIM; n++)	  ck[n] = k[n];
	double s,c;
	double phase = -dot(r, k);
	sincos (phase, &s, &c);
	complex<double> e_mikr (c,s);
	valVec[j]   = e_mikr*u;
	gradVec[j]  = e_mikr*(-eye*u*ck + gradu);
	laplVec[j]  = e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu);
      }
    }

    // Finally, copy into output vectors
    int psiIndex = 0;
    int N = StorageValueVector.size();
    if (need2blend) {
      for (int j=0; j<NumValenceOrbs; j++) {
	complex<double> psi_val, psi_lapl;
	TinyVector<complex<double>, OHMMS_DIM> psi_grad;
	PosType rhat = 1.0/std::sqrt(dot(disp,disp)) * disp;
	complex<double> psi1 = StorageValueVector[j];
	complex<double> psi2 =   BlendValueVector[j];
	TinyVector<complex<double>,OHMMS_DIM> dpsi1 = StorageGradVector[j];
	TinyVector<complex<double>,OHMMS_DIM> dpsi2 = BlendGradVector[j];
	complex<double> d2psi1 = StorageLaplVector[j];
	complex<double> d2psi2 =   BlendLaplVector[j];
	
	TinyVector<complex<double>,OHMMS_DIM> zrhat;
	for (int i=0; i<OHMMS_DIM; i++)
	  zrhat[i] = rhat[i];

	psi_val  = b * psi1 + (1.0-b)*psi2;
	psi_grad = b * dpsi1 + (1.0-b)*dpsi2 + db * (psi1 - psi2)* zrhat;
	psi_lapl = b * d2psi1 + (1.0-b)*d2psi2 +
	  2.0*db * (dot(zrhat,dpsi1) - dot(zrhat, dpsi2)) +
	  d2b * (psi1 - psi2);
	
	psi[psiIndex] = real(psi_val);
	for (int n=0; n<OHMMS_DIM; n++)
	  dpsi[psiIndex][n] = real(psi_grad[n]);
	d2psi[psiIndex] = real(psi_lapl);
	psiIndex++;
	if (MakeTwoCopies[j]) {
	  psi[psiIndex] = imag(psi_val);
	  for (int n=0; n<OHMMS_DIM; n++)
	    dpsi[psiIndex][n] = imag(psi_grad[n]);
	  d2psi[psiIndex] = imag(psi_lapl);
	  psiIndex++;
	}
      } 
      for (int j=NumValenceOrbs; j<N; j++) {
	complex<double> psi_val, psi_lapl;
	TinyVector<complex<double>, OHMMS_DIM> psi_grad;
	psi_val  = StorageValueVector[j];
	psi_grad = StorageGradVector[j];
	psi_lapl = StorageLaplVector[j];
	
	psi[psiIndex] = real(psi_val);
	for (int n=0; n<OHMMS_DIM; n++)
	  dpsi[psiIndex][n] = real(psi_grad[n]);
	d2psi[psiIndex] = real(psi_lapl);
	psiIndex++;
	if (MakeTwoCopies[j]) {
	  psi[psiIndex] = imag(psi_val);
	  for (int n=0; n<OHMMS_DIM; n++)
	    dpsi[psiIndex][n] = imag(psi_grad[n]);
	  d2psi[psiIndex] = imag(psi_lapl);
	  psiIndex++;
	}
      }
    }
    else {
      for (int j=0; j<N; j++) {
	complex<double> psi_val, psi_lapl;
	TinyVector<complex<double>, OHMMS_DIM> psi_grad;
	psi_val  = StorageValueVector[j];
	psi_grad = StorageGradVector[j];
	psi_lapl = StorageLaplVector[j];
	
	psi[psiIndex] = real(psi_val);
	for (int n=0; n<OHMMS_DIM; n++)
	  dpsi[psiIndex][n] = real(psi_grad[n]);
	d2psi[psiIndex] = real(psi_lapl);
	psiIndex++;
	if (MakeTwoCopies[j]) {
	  psi[psiIndex] = imag(psi_val);
	  for (int n=0; n<OHMMS_DIM; n++)
	    dpsi[psiIndex][n] = imag(psi_grad[n]);
	  d2psi[psiIndex] = imag(psi_lapl);
	  psiIndex++;
	}
      }
    }
    VGLTimer.stop();
  }
  
  // Value, gradient, and laplacian
  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::evaluate
  (const ParticleSet& P, int iat, ComplexValueVector_t& psi, 
   ComplexGradVector_t& dpsi, ComplexValueVector_t& d2psi)
  {
    VGLTimer.start();
    PosType r (P.R[iat]);
    PosType ru(PrimLattice.toUnit(P.R[iat]));
    for (int i=0; i<OHMMS_DIM; i++)
      ru[i] -= std::floor (ru[i]);
    EinsplineTimer.start();
    EinsplineMultiEval (MultiSpline, ru, StorageValueVector,
			StorageGradVector, StorageHessVector);
    EinsplineTimer.stop();
    //computePhaseFactors(r);
    complex<double> eye (0.0, 1.0);
    for (int j=0; j<psi.size(); j++) {
      complex<double> u, laplu;
      TinyVector<complex<double>, OHMMS_DIM> gradu;
      u = StorageValueVector[j];
      gradu = dot(PrimLattice.G, StorageGradVector[j]);
      laplu = trace(StorageHessVector[j], GGt);
      
      PosType k = kPoints[j];
      TinyVector<complex<double>,OHMMS_DIM> ck;
      for (int n=0; n<OHMMS_DIM; n++)	
	ck[n] = k[n];
      double s,c;
      double phase = -dot(r, k);
      sincos (phase, &s, &c);
      complex<double> e_mikr (c,s);
      convert(e_mikr * u, psi[j]);
      convertVec(e_mikr*(-eye*u*ck + gradu), dpsi[j]);
      convert(e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu), d2psi[j]);
    }
    VGLTimer.stop();
  }
  
  
  template<> void
  EinsplineSetExtended<double>::evaluate
  (const ParticleSet& P, int iat, RealValueVector_t& psi, 
   RealGradVector_t& dpsi, RealValueVector_t& d2psi)
  {
    VGLTimer.start();
    PosType r (P.R[iat]);
    PosType ru(PrimLattice.toUnit(P.R[iat]));
    int image[OHMMS_DIM];
    for (int i=0; i<OHMMS_DIM; i++) {
      RealType img = std::floor(ru[i]);
      ru[i] -= img;
      image[i] = (int) img;
    }
    EinsplineTimer.start();
    EinsplineMultiEval (MultiSpline, ru, psi, StorageGradVector, 
			StorageHessVector);
    int sign=0;
    for (int i=0; i<OHMMS_DIM; i++) 
      sign += HalfG[i]*image[i];
    if (sign & 1) 
      for (int j=0; j<psi.size(); j++) {
	psi[j] *= -1.0;
	StorageGradVector[j] *= -1.0;
	StorageHessVector[j] *= -1.0;
      }
    EinsplineTimer.stop();
    for (int i=0; i<psi.size(); i++) {
      dpsi[i]  = dot(PrimLattice.G, StorageGradVector[i]);
      d2psi[i] = trace(StorageHessVector[i], GGt);
    }
    VGLTimer.stop();
  }
  
  
  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::evaluate
  (const ParticleSet& P, int first, int last, RealValueMatrix_t& psi, 
   RealGradMatrix_t& dpsi, RealValueMatrix_t& d2psi)
  {
    complex<double> eye(0.0,1.0);
    VGLMatTimer.start();
    for (int iat=first,i=0; iat<last; iat++,i++) {
      PosType r (P.R[iat]);
      
      // Do core states first
      int icore = NumValenceOrbs;
      for (int tin=0; tin<MuffinTins.size(); tin++) {
	MuffinTins[tin].evaluateCore(r, StorageValueVector, StorageGradVector,
				     StorageLaplVector, icore);
	icore += MuffinTins[tin].get_num_core();
      }
      
      // Add phase to core orbitals
      for (int j=NumValenceOrbs; j<StorageValueVector.size(); j++) {
	complex<double> u = StorageValueVector[j];
	TinyVector<complex<double>,OHMMS_DIM> gradu = StorageGradVector[j];
	complex<double> laplu = StorageLaplVector[j];
	PosType k = kPoints[j];
	TinyVector<complex<double>,OHMMS_DIM> ck;
	for (int n=0; n<OHMMS_DIM; n++)	  ck[n] = k[n];
	double s,c;
	double phase = -dot(r, k);
	sincos (phase, &s, &c);
	complex<double> e_mikr (c,s);
	StorageValueVector[j] = e_mikr*u;
	StorageGradVector[j]  = e_mikr*(-eye*u*ck + gradu);
	StorageLaplVector[j]  = e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu);
      }
      
      // Check if we are in the muffin tin;  if so, evaluate
      bool inTin = false, need2blend = false;
      PosType disp;
      double b, db, d2b;
      for (int tin=0; tin<MuffinTins.size(); tin++) {
	MuffinTins[tin].inside(r, inTin, need2blend);
	if (inTin) {
	  MuffinTins[tin].evaluate (r, StorageValueVector, 
				    StorageGradVector, StorageLaplVector);
	  if (need2blend) {
	    disp = MuffinTins[tin].disp(r);
	    double dr = std::sqrt(dot(disp, disp));
	    MuffinTins[tin].blend_func(dr, b, db, d2b);
	  }
	  break;
	}
      }
      
      StorageValueVector_t &valVec =  
	need2blend ? BlendValueVector : StorageValueVector;
      StorageGradVector_t &gradVec =  
	need2blend ? BlendGradVector : StorageGradVector;
      StorageValueVector_t &laplVec =  
	need2blend ? BlendLaplVector : StorageLaplVector;
      
      // Otherwise, evaluate the B-splines
      if (!inTin || need2blend) {
	PosType ru(PrimLattice.toUnit(P.R[iat]));
	for (int i=0; i<OHMMS_DIM; i++)
	  ru[i] -= std::floor (ru[i]);
	EinsplineTimer.start();
	EinsplineMultiEval (MultiSpline, ru, valVec, gradVec, StorageHessVector);
	EinsplineTimer.stop();
	for (int j=0; j<NumValenceOrbs; j++) {
	  gradVec[j] = dot (PrimLattice.G, gradVec[j]);
	  laplVec[j] = trace (StorageHessVector[j], GGt);
	}
	// Add e^-ikr phase to B-spline orbitals
	for (int j=0; j<NumValenceOrbs; j++) {
	  complex<double> u = valVec[j];
	  TinyVector<complex<double>,OHMMS_DIM> gradu = gradVec[j];
	  complex<double> laplu = laplVec[j];
	  PosType k = kPoints[j];
	  TinyVector<complex<double>,OHMMS_DIM> ck;
	  for (int n=0; n<OHMMS_DIM; n++)	  ck[n] = k[n];
	  double s,c;
	  double phase = -dot(r, k);
	  sincos (phase, &s, &c);
	  complex<double> e_mikr (c,s);
	  valVec[j]   = e_mikr*u;
	  gradVec[j]  = e_mikr*(-eye*u*ck + gradu);
	  laplVec[j]  = e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu);
	}
      }
      
      // Finally, copy into output vectors
      int psiIndex = 0;
      int N = StorageValueVector.size();
      if (need2blend) {
	for (int j=0; j<NumValenceOrbs; j++) {
	  complex<double> psi_val, psi_lapl;
	  TinyVector<complex<double>, OHMMS_DIM> psi_grad;
	  PosType rhat = 1.0/std::sqrt(dot(disp,disp)) * disp;
	  complex<double> psi1 = StorageValueVector[j];
	  complex<double> psi2 =   BlendValueVector[j];
	  TinyVector<complex<double>,OHMMS_DIM> dpsi1 = StorageGradVector[j];
	  TinyVector<complex<double>,OHMMS_DIM> dpsi2 = BlendGradVector[j];
	  complex<double> d2psi1 = StorageLaplVector[j];
	  complex<double> d2psi2 =   BlendLaplVector[j];
	  
	  TinyVector<complex<double>,OHMMS_DIM> zrhat;
	  for (int n=0; n<OHMMS_DIM; n++)
	    zrhat[n] = rhat[n];
	  
	  psi_val  = b * psi1 + (1.0-b)*psi2;
	  psi_grad = b * dpsi1 + (1.0-b)*dpsi2 + db * (psi1 - psi2)* zrhat;
	  psi_lapl = b * d2psi1 + (1.0-b)*d2psi2 +
	    2.0*db * (dot(zrhat,dpsi1) - dot(zrhat, dpsi2)) +
	    d2b * (psi1 - psi2);
	  
	  psi(psiIndex,i) = real(psi_val);
	  for (int n=0; n<OHMMS_DIM; n++)
	    dpsi(i,psiIndex)[n] = real(psi_grad[n]);
	  d2psi(i,psiIndex) = real(psi_lapl);
	  psiIndex++;
	  if (MakeTwoCopies[j]) {
	    psi(psiIndex,i) = imag(psi_val);
	    for (int n=0; n<OHMMS_DIM; n++)
	      dpsi(i,psiIndex)[n] = imag(psi_grad[n]);
	    d2psi(i,psiIndex) = imag(psi_lapl);
	    psiIndex++;
	  }
	} 
	// Copy core states
	for (int j=NumValenceOrbs; j<N; j++) {
	  complex<double> psi_val, psi_lapl;
	  TinyVector<complex<double>, OHMMS_DIM> psi_grad;
	  psi_val  = StorageValueVector[j];
	  psi_grad = StorageGradVector[j];
	  psi_lapl = StorageLaplVector[j];
	  
	  psi(psiIndex,i) = real(psi_val);
	  for (int n=0; n<OHMMS_DIM; n++)
	    dpsi(i,psiIndex)[n] = real(psi_grad[n]);
	  d2psi(i,psiIndex) = real(psi_lapl);
	  psiIndex++;
	  if (MakeTwoCopies[j]) {
	    psi(psiIndex,i) = imag(psi_val);
	    for (int n=0; n<OHMMS_DIM; n++)
	      dpsi(i,psiIndex)[n] = imag(psi_grad[n]);
	    d2psi(i,psiIndex) = imag(psi_lapl);
	    psiIndex++;
	  }
	}
      }
      else { // No blending needed
	for (int j=0; j<N; j++) {
	  complex<double> psi_val, psi_lapl;
	  TinyVector<complex<double>, OHMMS_DIM> psi_grad;
	  psi_val  = StorageValueVector[j];
	  psi_grad = StorageGradVector[j];
	  psi_lapl = StorageLaplVector[j];
	  
	  psi(psiIndex,i) = real(psi_val);
	  for (int n=0; n<OHMMS_DIM; n++)
	    dpsi(i,psiIndex)[n] = real(psi_grad[n]);
	  d2psi(i,psiIndex) = real(psi_lapl);
	  psiIndex++;
	  // if (psiIndex >= dpsi.cols()) {
	  //   cerr << "Error:  out of bounds writing in EinsplineSet::evalate.\n"
	  // 	 << "psiIndex = " << psiIndex << "  dpsi.cols() = " << dpsi.cols() << endl;
	  // }
	  if (MakeTwoCopies[j]) {
	    psi(psiIndex,i) = imag(psi_val);
	    for (int n=0; n<OHMMS_DIM; n++)
	      dpsi(i,psiIndex)[n] = imag(psi_grad[n]);
	    d2psi(i,psiIndex) = imag(psi_lapl);
	    psiIndex++;
	  }
	}
      }
    }
    VGLMatTimer.stop();
  }
  
  
  
  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::evaluate
  (const ParticleSet& P, int first, int last, ComplexValueMatrix_t& psi, 
   ComplexGradMatrix_t& dpsi, ComplexValueMatrix_t& d2psi)
  {
    VGLMatTimer.start();
    for(int iat=first,i=0; iat<last; iat++,i++) {
      PosType r (P.R[iat]);
      PosType ru(PrimLattice.toUnit(P.R[iat]));
      for (int n=0; n<OHMMS_DIM; n++)
	ru[n] -= std::floor (ru[n]);
      EinsplineTimer.start();
      EinsplineMultiEval (MultiSpline, ru, StorageValueVector,
			  StorageGradVector, StorageHessVector);
      EinsplineTimer.stop();
      //computePhaseFactors(r);
      complex<double> eye (0.0, 1.0);
      for (int j=0; j<OrbitalSetSize; j++) {
	complex<double> u, laplu;
	TinyVector<complex<double>, OHMMS_DIM> gradu;
	u = StorageValueVector[j];
	gradu = dot(PrimLattice.G, StorageGradVector[j]);
	laplu = trace(StorageHessVector[j], GGt);
	
	PosType k = kPoints[j];
	TinyVector<complex<double>,OHMMS_DIM> ck;
	for (int n=0; n<OHMMS_DIM; n++)	
	  ck[n] = k[n];
	double s,c;
	double phase = -dot(r, k);
	sincos (phase, &s, &c);
	complex<double> e_mikr (c,s);
	convert(e_mikr * u, psi(j,i));
	convertVec(e_mikr*(-eye*u*ck + gradu), dpsi(i,j));
	convert(e_mikr*(-dot(k,k)*u - 2.0*eye*dot(ck,gradu) + laplu), d2psi(i,j));
      } 
    }
    VGLMatTimer.stop();
  }
  
  
  
  template<> void
  EinsplineSetExtended<double>::evaluate
  (const ParticleSet& P, int first, int last, RealValueMatrix_t& psi, 
   RealGradMatrix_t& dpsi, RealValueMatrix_t& d2psi)
  {
    VGLMatTimer.start();
    for(int iat=first,i=0; iat<last; iat++,i++) {
      PosType r (P.R[iat]);
      PosType ru(PrimLattice.toUnit(P.R[iat]));
      int image[OHMMS_DIM];
      for (int n=0; n<OHMMS_DIM; n++) {
	RealType img = std::floor(ru[n]);
	ru[n] -= img;
	image[n] = (int) img;
      }
      EinsplineTimer.start();
      EinsplineMultiEval (MultiSpline, ru, StorageValueVector,
			  StorageGradVector, StorageHessVector);
      EinsplineTimer.stop();

    int sign=0;
    for (int n=0; n<OHMMS_DIM; n++) 
      sign += HalfG[n]*image[n];

    if (sign & 1) 
      for (int j=0; j<OrbitalSetSize; j++) {
	StorageValueVector[j] *= -1.0;
	StorageGradVector[j] *= -1.0;
	StorageHessVector[j] *= -1.0;
      }
      for (int j=0; j<OrbitalSetSize; j++) {
        psi(j,i)   = StorageValueVector[j];
	dpsi(i,j)  = dot(PrimLattice.G, StorageGradVector[j]);
	d2psi(i,j) = trace(StorageHessVector[j], GGt);
      }
    }
    VGLMatTimer.stop();
  }
  

  //////////////////////////////////////////////
  // Vectorized evaluation routines using GPU //
  //////////////////////////////////////////////

  template<> void 
  EinsplineSetExtended<double>::evaluate 
  (vector<Walker_t*> &walkers, int iat,
   cuda_vector<CudaRealType*> &phi)
  {
    // app_log() << "Start EinsplineSet CUDA evaluation\n";
    int N = walkers.size();
    CudaRealType plus_minus[2] = {1.0, -1.0};
    if (cudaPos.size() < N) {
      hostPos.resize(N);
      cudaPos.resize(N);
      hostSign.resize(N);
      cudaSign.resize(N);
    }
    for (int iw=0; iw < N; iw++) {
      PosType r = walkers[iw]->R[iat];
      PosType ru(PrimLattice.toUnit(r));
      int image[OHMMS_DIM];
      for (int i=0; i<OHMMS_DIM; i++) {
	RealType img = std::floor(ru[i]);
	ru[i] -= img;
	image[i] = (int) img;
      }
      int sign = 0;
      for (int i=0; i<OHMMS_DIM; i++) 
	sign += HalfG[i] * image[i];
      
      hostSign[iw] = plus_minus[sign&1];
      hostPos[iw] = ru;
    }

    cudaPos = hostPos;
    cudaSign = hostSign;
    eval_multi_multi_UBspline_3d_cuda 
      (CudaMultiSpline, (CudaRealType*)(cudaPos.data()), cudaSign.data(), phi.data(), N);
  }

  template<> void 
  EinsplineSetExtended<complex<double> >::evaluate 
  (vector<Walker_t*> &walkers, int iat,
   cuda_vector<CudaRealType*> &phi)
  {
    //    app_log() << "Eval 1.\n";
    int N = walkers.size();

    if (CudaValuePointers.size() < N)
      resizeCuda(N);

    if (cudaPos.size() < N) {
      hostPos.resize(N);
      cudaPos.resize(N);
    }
    for (int iw=0; iw < N; iw++) {
      PosType r = walkers[iw]->R[iat];
      PosType ru(PrimLattice.toUnit(r));
      ru[0] -= std::floor (ru[0]);
      ru[1] -= std::floor (ru[1]);
      ru[2] -= std::floor (ru[2]);
      hostPos[iw] = ru;
    }

    cudaPos = hostPos;

    eval_multi_multi_UBspline_3d_cuda 
      (CudaMultiSpline, (float*)cudaPos.data(), CudaValuePointers.data(), N);

    // Now, add on phases
    for (int iw=0; iw < N; iw++) 
      hostPos[iw] = walkers[iw]->R[iat];
    cudaPos = hostPos;

    apply_phase_factors ((CUDA_PRECISION*) CudakPoints.data(),
			 CudaMakeTwoCopies.data(),
			 (CUDA_PRECISION*)cudaPos.data(),
			 (CUDA_PRECISION**)CudaValuePointers.data(),
			 phi.data(), CudaMultiSpline->num_splines, 
			 walkers.size());
  }


  template<> void 
  EinsplineSetExtended<double>::evaluate 
  (vector<Walker_t*> &walkers, vector<PosType> &newpos,
   cuda_vector<CudaRealType*> &phi)
  {
    // app_log() << "Start EinsplineSet CUDA evaluation\n";
    int N = newpos.size();
    CudaRealType plus_minus[2] = {1.0, -1.0};
    
    if (cudaPos.size() < N) {
      hostPos.resize(N);
      cudaPos.resize(N);
      hostSign.resize(N);
      cudaSign.resize(N);
    }

    for (int iw=0; iw < N; iw++) {
      PosType r = newpos[iw];
      PosType ru(PrimLattice.toUnit(r));
      int image[OHMMS_DIM];
      for (int i=0; i<OHMMS_DIM; i++) {
	RealType img = std::floor(ru[i]);
	ru[i] -= img;
	image[i] = (int) img;
      }
      int sign = 0;
      for (int i=0; i<OHMMS_DIM; i++) 
	sign += HalfG[i] * image[i];
      
      hostSign[iw] = plus_minus[sign&1];
      hostPos[iw] = ru;
    }

    cudaPos = hostPos;
    cudaSign = hostSign;
    eval_multi_multi_UBspline_3d_cuda 
      (CudaMultiSpline, (CudaRealType*)(cudaPos.data()), cudaSign.data(), 
       phi.data(), N);
    //app_log() << "End EinsplineSet CUDA evaluation\n";
  }

  template<typename T> void
  EinsplineSetExtended<T>::resizeCuda(int numWalkers)
  {
    CudaValuePointers.resize(numWalkers);
    CudaGradLaplPointers.resize(numWalkers);
    int N = CudaMultiSpline->num_splines;
    CudaValueVector.resize(N*numWalkers);
    CudaGradLaplVector.resize(4*N*numWalkers);
    host_vector<CudaStorageType*> hostValuePointers(numWalkers);
    host_vector<CudaStorageType*> hostGradLaplPointers(numWalkers);
    for (int i=0; i<numWalkers; i++) {
      hostValuePointers[i]    = &(CudaValueVector[i*N]);
      hostGradLaplPointers[i] = &(CudaGradLaplVector[4*i*N]);
    }
    CudaValuePointers    = hostValuePointers;
    CudaGradLaplPointers = hostGradLaplPointers;

    CudaMakeTwoCopies.resize(N);
    host_vector<int> hostMakeTwoCopies(N);
    for (int i=0; i<N; i++)
      hostMakeTwoCopies[i] = MakeTwoCopies[i];
    CudaMakeTwoCopies = hostMakeTwoCopies;

    CudakPoints.resize(N);
    host_vector<TinyVector<CUDA_PRECISION,OHMMS_DIM> > hostkPoints(N);
    for (int i=0; i<N; i++) {
      for (int j=0; j<OHMMS_DIM; j++)
	hostkPoints[i][j] = kPoints[i][j];
    }
    CudakPoints = hostkPoints;
  }

  template<> void 
  EinsplineSetExtended<complex<double> >::evaluate 
  (vector<Walker_t*> &walkers, vector<PosType> &newpos,
   cuda_vector<CudaRealType*> &phi)
  {
    //    app_log() << "Eval 2.\n";
    int N = walkers.size();
    if (CudaValuePointers.size() < N)
      resizeCuda(N);

    if (cudaPos.size() < N) {
      hostPos.resize(N);
      cudaPos.resize(N);
    }
    for (int iw=0; iw < N; iw++) {
      PosType r = newpos[iw];
      PosType ru(PrimLattice.toUnit(r));
      ru[0] -= std::floor (ru[0]);
      ru[1] -= std::floor (ru[1]);
      ru[2] -= std::floor (ru[2]);
      hostPos[iw] = ru;
    }

    cudaPos = hostPos;
    
    eval_multi_multi_UBspline_3d_cuda 
      (CudaMultiSpline, (float*)cudaPos.data(), CudaValuePointers.data(), N);
    
    // Now, add on phases
    for (int iw=0; iw < N; iw++) 
      hostPos[iw] = newpos[iw];
    cudaPos = hostPos;
    
    apply_phase_factors ((CUDA_PRECISION*) CudakPoints.data(),
			 CudaMakeTwoCopies.data(),
			 (CUDA_PRECISION*)cudaPos.data(),
			 (CUDA_PRECISION**)CudaValuePointers.data(),
			 phi.data(), CudaMultiSpline->num_splines, 
			 walkers.size());

  }

  template<> void
  EinsplineSetExtended<double>::evaluate
  (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
   cuda_vector<CudaRealType*> &phi, cuda_vector<CudaRealType*> &grad_lapl,
   int row_stride)
  {
    int N = walkers.size();
    CudaRealType plus_minus[2] = {1.0, -1.0};
    if (cudaPos.size() < N) {
      hostPos.resize(N);
      cudaPos.resize(N);
      hostSign.resize(N);
      cudaSign.resize(N);
    }
    for (int iw=0; iw < N; iw++) {
      PosType r = newpos[iw];
      PosType ru(PrimLattice.toUnit(r));
      int image[OHMMS_DIM];
      for (int i=0; i<OHMMS_DIM; i++) {
	RealType img = std::floor(ru[i]);
	ru[i] -= img;
	image[i] = (int) img;
      }
      int sign = 0;
      for (int i=0; i<OHMMS_DIM; i++) 
	sign += HalfG[i] * image[i];
      
      hostSign[iw] = plus_minus[sign&1];
      hostPos[iw] = ru;
    }
    
    cudaPos = hostPos;
    cudaSign = hostSign;

    eval_multi_multi_UBspline_3d_vgl_cuda
      (CudaMultiSpline, (CudaRealType*)cudaPos.data(), cudaSign.data(), 
       Linv_cuda.data(), phi.data(), grad_lapl.data(), N, row_stride);

    // host_vector<CudaRealType*> pointers;
    // pointers = phi;
    // CudaRealType data[N];
    // cudaMemcpy (data, pointers[0], N*sizeof(CudaRealType), cudaMemcpyDeviceToHost);
    // for (int i=0; i<N; i++)
    //   fprintf (stderr, "%1.12e\n", data[i]);
  }


  template<> void
  EinsplineSetExtended<complex<double> >::evaluate
  (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
   cuda_vector<CudaRealType*> &phi, cuda_vector<CudaRealType*> &grad_lapl,
   int row_stride)
  {
    //    app_log() << "Eval 3.\n";
    int N = walkers.size();
    int M = CudaMultiSpline->num_splines;

    if (CudaValuePointers.size() < N)
      resizeCuda(N);

    if (cudaPos.size() < N) {
      hostPos.resize(N);
      cudaPos.resize(N);
    }
    for (int iw=0; iw < N; iw++) {
      PosType r = newpos[iw];
      PosType ru(PrimLattice.toUnit(r));
      ru[0] -= std::floor (ru[0]);
      ru[1] -= std::floor (ru[1]);
      ru[2] -= std::floor (ru[2]);
      hostPos[iw] = ru;
    }

    cudaPos = hostPos;

    eval_multi_multi_UBspline_3d_c_vgl_cuda
      (CudaMultiSpline, (float*)cudaPos.data(),  Linv_cuda.data(), CudaValuePointers.data(), 
       CudaGradLaplPointers.data(), N, CudaMultiSpline->num_splines);


    // DEBUG
  //   TinyVector<double,OHMMS_DIM> r(hostPos[0][0], hostPos[0][1], hostPos[0][2]);
  //   Vector<complex<double > > psi(M);
  //   Vector<TinyVector<complex<double>,3> > grad(M);
  //   Vector<Tensor<complex<double>,3> > hess(M);
  //   EinsplineMultiEval (MultiSpline, r, psi, grad, hess);
      
  //   // complex<double> cpuSpline[M];
  //   // TinyVector<complex<double>,OHMMS_DIM> complex<double> cpuGrad[M];
  //   // Tensor cpuHess[M];
  //   // eval_multi_UBspline_3d_z_vgh (MultiSpline, hostPos[0][0], hostPos[0][1], hostPos[0][2],
  //   // 				  cpuSpline);

  //   host_vector<CudaStorageType*> pointers;
  //   pointers = CudaGradLaplPointers;
  //   complex<float> gpuSpline[4*M];
  //   cudaMemcpy(gpuSpline, pointers[0], 
  // 	       4*M * sizeof(complex<float>), cudaMemcpyDeviceToHost);

  // for (int i=0; i<M; i++)
  //   fprintf (stderr, "%10.6f %10.6f   %10.6f %10.6f\n",
  // 	     trace(hess[i],GGt).real(), gpuSpline[3*M+i].real(),
  // 	     trace(hess[i], GGt).imag(), gpuSpline[3*M+i].imag());

    // Now, add on phases
    for (int iw=0; iw < N; iw++) 
      hostPos[iw] = newpos[iw];
    cudaPos = hostPos;
    
    apply_phase_factors ((CUDA_PRECISION*) CudakPoints.data(),
			 CudaMakeTwoCopies.data(),
			 (CUDA_PRECISION*)cudaPos.data(),
			 (CUDA_PRECISION**)CudaValuePointers.data(), phi.data(), 
			 (CUDA_PRECISION**)CudaGradLaplPointers.data(), grad_lapl.data(),
			 CudaMultiSpline->num_splines,  walkers.size(), row_stride);
  }


  template<> void 
  EinsplineSetExtended<double>::evaluate 
  (vector<PosType> &pos, cuda_vector<CudaRealType*> &phi)
  { 
    int N = pos.size();
    CudaRealType plus_minus[2] = {1.0, -1.0};

    if (cudaPos.size() < N) {
      NLhostPos.resize(N);
      NLcudaPos.resize(N);
      NLhostSign.resize(N);
      NLcudaSign.resize(N);
    }
    for (int iw=0; iw < N; iw++) {
      PosType r = pos[iw];
      PosType ru(PrimLattice.toUnit(r));
      int image[OHMMS_DIM];
      for (int i=0; i<OHMMS_DIM; i++) {
	RealType img = std::floor(ru[i]);
	ru[i] -= img;
	image[i] = (int) img;
      }
      int sign = 0;
      for (int i=0; i<OHMMS_DIM; i++) 
	sign += HalfG[i] * image[i];
      
      NLhostSign[iw] = plus_minus[sign&1];

      NLhostPos[iw] = ru;
    }

    NLcudaPos  = NLhostPos;
    NLcudaSign = NLhostSign;
    eval_multi_multi_UBspline_3d_cuda 
      (CudaMultiSpline, (CudaRealType*)(NLcudaPos.data()), 
       NLcudaSign.data(), phi.data(), N);    
  }

  template<> void 
  EinsplineSetExtended<double>::evaluate 
  (vector<PosType> &pos, cuda_vector<CudaComplexType*> &phi)
  { 
    app_error() << "EinsplineSetExtended<complex<double> >::evaluate "
		<< "not yet implemented.\n";
    abort();
  }



  template<> void 
  EinsplineSetExtended<complex<double> >::evaluate 
  (vector<PosType> &pos, cuda_vector<CudaRealType*> &phi)
  { 
    //    app_log() << "Eval 4.\n";
    int N = pos.size();

    if (CudaValuePointers.size() < N)
      resizeCuda(N);

    if (cudaPos.size() < N) {
      hostPos.resize(N);
      cudaPos.resize(N);
    }
    for (int iw=0; iw < N; iw++) {
      PosType r = pos[iw];
      PosType ru(PrimLattice.toUnit(r));
      ru[0] -= std::floor (ru[0]);
      ru[1] -= std::floor (ru[1]);
      ru[2] -= std::floor (ru[2]);
      hostPos[iw] = ru;
    }

    cudaPos = hostPos;
    eval_multi_multi_UBspline_3d_cuda 
      (CudaMultiSpline, (CUDA_PRECISION*) cudaPos.data(), 
       CudaValuePointers.data(), N);
    
    // Now, add on phases
    for (int iw=0; iw < N; iw++) 
      hostPos[iw] = pos[iw];
    cudaPos = hostPos;
    
    apply_phase_factors ((CUDA_PRECISION*) CudakPoints.data(),
			 CudaMakeTwoCopies.data(),
			 (CUDA_PRECISION*)cudaPos.data(),
			 (CUDA_PRECISION**)CudaValuePointers.data(),
			 phi.data(), CudaMultiSpline->num_splines, N);
  }

  template<> void 
  EinsplineSetExtended<complex<double> >::evaluate 
  (vector<PosType> &pos, cuda_vector<CudaComplexType*> &phi)
  { 
    app_error() << "EinsplineSetExtended<complex<double> >::evaluate "
		<< "not yet implemented.\n";
    abort();
  }




  template<typename StorageType> string
  EinsplineSetExtended<StorageType>::Type()
  {
    return "EinsplineSetExtended";
  }


  template<typename StorageType> void
  EinsplineSetExtended<StorageType>::registerTimers()
  {
    ValueTimer.reset();
    VGLTimer.reset();
    VGLMatTimer.reset();
    EinsplineTimer.reset();
    TimerManager.addTimer (&ValueTimer);
    TimerManager.addTimer (&VGLTimer);
    TimerManager.addTimer (&VGLMatTimer);
    TimerManager.addTimer (&EinsplineTimer);
  }


  



  SPOSetBase*
  EinsplineSetLocal::makeClone() const 
  {
    return new EinsplineSetLocal(*this);
  }

  void
  EinsplineSetLocal::resetParameters(const opt_variables_type& active)
  {
  }

  template<typename StorageType> SPOSetBase*
  EinsplineSetExtended<StorageType>::makeClone() const
  {
    EinsplineSetExtended<StorageType> *clone = 
      new EinsplineSetExtended<StorageType> (*this);
    clone->registerTimers();
    return clone;
  }

  template class EinsplineSetExtended<complex<double> >;
  template class EinsplineSetExtended<        double  >;


  
  ////////////////////////////////
  // Hybrid evaluation routines //
  ////////////////////////////////
  // template<typename StorageType> void
  // EinsplineSetHybrid<StorageType>::sort_electrons(vector<PosType> &pos)
  // {
  //   int nw = pos.size();
  //   if (nw > CurrentWalkers)
  //     resize_cuda(nw);

  //   AtomicPolyJobs_CPU.clear();
  //   AtomicSplineJobs_CPU.clear();
  //   rhats_CPU.clear();
  //   int numAtomic = 0;

  //   // First, sort electrons into three categories:
  //   // 1) Interstitial region with 3D-Bsplines
  //   // 2) Atomic region near origin:      polynomial  radial functions
  //   // 3) Atomic region not near origin:  1D B-spline radial functions

  //   for (int i=0; i<newpos.size(); i++) {
  //     PosType r = newpos[i];
  //     // Note: this assumes that the atomic radii are smaller than the simulation cell radius.
  //     for (int j=0; j<AtomicOrbitals.size(); j++) {
  // 	AtomicOrbital<complex<double> > &orb = AtomicOrbitals[j];
  // 	PosType dr = r - orb.Pos;
  // 	PosType u = PrimLattice.toUnit(dr);
  // 	for (int k=0; k<OHMMS_DIM; k++) 
  // 	  u[k] -= round(u[k]);
  // 	dr = PrimLattice.toCart(u);
  // 	RealType dist2 = dot (dr,dr);
  // 	if (dist2 < orb.PolyRadius * orb.PolyRadius) {
  // 	  AtomicPolyJob<CudaRealType> job;
  // 	  RealType dist = std::sqrt(dist2);
  // 	  job.dist = dist;
  // 	  RealType distInv = 1.0/dist;
  // 	  for (int k=0; k<OHMMS_DIM; k++) {
  // 	    CudaRealType x = distInv*dr[k];
  // 	    job.rhat[k] = distInv * dr[k];
  // 	    rhats_CPU.push_back(x);
  // 	  }
  // 	  job.lMax = orb.lMax;
  // 	  job.YlmIndex  = i;
  // 	  job.PolyOrder = orb.PolyOrder;
  // 	  //job.PolyCoefs = orb.PolyCoefs;
  // 	  AtomicPolyJobs_CPU.push_back(job);
  // 	  numAtomic++;
  // 	}
  // 	else if (dist2 < orb.CutoffRadius*orb.CutoffRadius) {
  // 	  AtomicSplineJob<CudaRealType> job;
  // 	   RealType dist = std::sqrt(dist2);
  // 	  job.dist = dist;
  // 	  RealType distInv = 1.0/dist;
  // 	  for (int k=0; k<OHMMS_DIM; k++) {
  // 	    CudaRealType x = distInv*dr[k];
  // 	    job.rhat[k] = distInv * dr[k];
  // 	    rhats_CPU.push_back(x);
  // 	  }
  // 	  job.lMax      = orb.lMax;
  // 	  job.YlmIndex  = i;
  // 	  job.phi       = phi[i];
  // 	  job.grad_lapl = grad_lapl[i];
  // 	  //job.PolyCoefs = orb.PolyCoefs;
  // 	  AtomicSplineJobs_CPU.push_back(job);
  // 	  numAtomic++;
  // 	}
  // 	else { // Regular 3D B-spline job
  // 	  BsplinePos_CPU.push_back (r);
	  
  // 	}
  //     }
  //   }
    

  // }


  ///////////////////////////////
  // Real StorageType versions //
  ///////////////////////////////

  template<> void
  EinsplineSetHybrid<double>::resize_cuda(int numwalkers)
  {
    CurrentWalkers = numwalkers;

    // Resize Ylm temporaries
    // Find lMax;
    lMax=-1;
    for (int i=0; i<AtomicOrbitals.size(); i++) 
      lMax = max(AtomicOrbitals[i].lMax, lMax);
    
    numlm = (lMax+1)*(lMax+1);
    Ylm_BS = ((numlm+15)/16) * 16;
    
    Ylm_GPU.resize(numwalkers*Ylm_BS*3);
    Ylm_ptr_GPU.resize        (numwalkers);   Ylm_ptr_CPU.resize       (numwalkers);
    dYlm_dtheta_ptr_GPU.resize(numwalkers);  dYlm_dtheta_ptr_CPU.resize(numwalkers);
    dYlm_dphi_ptr_GPU.resize  (numwalkers);  dYlm_dphi_ptr_CPU.resize  (numwalkers);

    rhats_CPU.resize(OHMMS_DIM*numwalkers);
    rhats_GPU.resize(OHMMS_DIM*numwalkers);
    HybridJobs_GPU.resize(numwalkers);
    HybridData_GPU.resize(numwalkers);

    for (int iw=0; iw<numwalkers; iw++) {
      Ylm_ptr_CPU[iw]         = &(Ylm_GPU[0]) + (3*iw+0)*Ylm_BS;
      dYlm_dtheta_ptr_CPU[iw] = &(Ylm_GPU[0]) + (3*iw+1)*Ylm_BS;
      dYlm_dphi_ptr_CPU[iw]   = &(Ylm_GPU[0]) + (3*iw+2)*Ylm_BS;
    }
    
    Ylm_ptr_GPU         = Ylm_ptr_CPU;
    dYlm_dtheta_ptr_GPU = dYlm_dtheta_ptr_CPU;
    dYlm_dphi_ptr_GPU   = dYlm_dphi_ptr_CPU;
    
    // Resize AtomicJob temporaries
    
    // AtomicPolyJobs_GPU.resize(numwalkers);
    // AtomicSplineJobs_GPU.resize(numwalkers);
  }


  template<> void
  EinsplineSetHybrid<complex<double> >::resize_cuda(int numwalkers)
  {
    CurrentWalkers = numwalkers;

    // Resize Ylm temporaries
    // Find lMax;
    lMax=-1;
    for (int i=0; i<AtomicOrbitals.size(); i++) 
      lMax = max(AtomicOrbitals[i].lMax, lMax);
    
    numlm = (lMax+1)*(lMax+1);
    Ylm_BS = ((numlm+15)/16) * 16;
    
    Ylm_GPU.resize(numwalkers*Ylm_BS*3);
    Ylm_ptr_GPU.resize        (numwalkers);   Ylm_ptr_CPU.resize       (numwalkers);
    dYlm_dtheta_ptr_GPU.resize(numwalkers);  dYlm_dtheta_ptr_CPU.resize(numwalkers);
    dYlm_dphi_ptr_GPU.resize  (numwalkers);  dYlm_dphi_ptr_CPU.resize  (numwalkers);
    rhats_CPU.resize(numwalkers);
    rhats_GPU.resize(numwalkers);

    for (int iw=0; iw<numwalkers; iw++) {
      Ylm_ptr_CPU[iw]         = &(Ylm_GPU[0]) + (3*iw+0)*Ylm_BS;
      dYlm_dtheta_ptr_CPU[iw] = &(Ylm_GPU[0]) + (3*iw+1)*Ylm_BS;
      dYlm_dphi_ptr_CPU[iw]   = &(Ylm_GPU[0]) + (3*iw+2)*Ylm_BS;
    }
    
    Ylm_ptr_GPU         = Ylm_ptr_CPU;
    dYlm_dtheta_ptr_GPU = dYlm_dtheta_ptr_CPU;
    dYlm_dphi_ptr_GPU   = dYlm_dphi_ptr_CPU;

    // Resize AtomicJob temporaries
    // AtomicPolyJobs_GPU.resize(numwalkers);
    // AtomicSplineJobs_GPU.resize(numwalkers);
  }


  // Vectorized evaluation functions
  template<> void
  EinsplineSetHybrid<double>::evaluate (vector<Walker_t*> &walkers, int iat,
					cuda_vector<CudaRealType*> &phi)
  {
    app_error() << "EinsplineSetHybrid<double>::evaluate (vector<Walker_t*> &walkers, int iat,\n"
		<< " cuda_vector<CudaRealType*> &phi) not implemented.\n";
    abort();
  }

  
  template<> void
  EinsplineSetHybrid<double>::evaluate (vector<Walker_t*> &walkers, int iat,
				cuda_vector<CudaComplexType*> &phi)
  {
    app_error() << "EinsplineSetHybrid<double>::evaluate (vector<Walker_t*> &walkers, int iat,\n"
		<< " cuda_vector<CudaComplexType*> &phi) not implemented.\n";
    abort();
  }

  
  template<> void
  EinsplineSetHybrid<double>::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
					cuda_vector<CudaRealType*> &phi)
  { 
    app_error() << "EinsplineSetHybrid<double>::evaluate \n"
		<< " (vector<Walker_t*> &walkers, vector<PosType> &newpos, \n"
		<< " cuda_vector<CudaRealType*> &phi) not implemented.\n";
    abort();
  }

  template<> void
  EinsplineSetHybrid<double>::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos,
  		 cuda_vector<CudaComplexType*> &phi)
  {
    app_error() << "EinsplineSetHybrid<double>::evaluate \n"
		<< "  (vector<Walker_t*> &walkers, vector<PosType> &newpos,\n"
		<< "   cuda_vector<CudaComplexType*> &phi) not implemented.\n";
    abort();
  }

  

  template<> void
  EinsplineSetHybrid<double>::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
					cuda_vector<CudaRealType*> &phi,
					cuda_vector<CudaRealType*> &grad_lapl,
					int row_stride)
  { 
    int N = newpos.size();
    if (cudaPos.size() < N) {
      resize_cuda(N);
      hostPos.resize(N);
      cudaPos.resize(N);
    }
    for (int iw=0; iw < N; iw++) 
      hostPos[iw] = newpos[iw];
    cudaPos = hostPos;
    
    // hostPos = cudaPos;
    // for (int i=0; i<newpos.size(); i++)
    //   cerr << "newPos[" << i << "] = " << newpos[i] << endl;

    // host_vector<CudaRealType> IonPos_CPU(IonPos_GPU.size());
    // IonPos_CPU = IonPos_GPU;
    // for (int i=0; i<IonPos_CPU.size()/3; i++)
    //   fprintf (stderr, "ion[%d] = [%10.6f %10.6f %10.6f]\n",
    // 	       i, IonPos_CPU[3*i+0], IonPos_CPU[3*i+2], IonPos_CPU[3*i+2]);
    // cerr << "cudaPos.size()        = " << cudaPos.size() << endl;
    // cerr << "IonPos.size()         = " << IonPos_GPU.size() << endl;
    // cerr << "PolyRadii.size()      = " << PolyRadii_GPU.size() << endl;
    // cerr << "CutoffRadii.size()    = " << CutoffRadii_GPU.size() << endl;
    // cerr << "AtomicOrbitals.size() = " << AtomicOrbitals.size() << endl;
    // cerr << "L_cuda.size()         = " << L_cuda.size() << endl;
    // cerr << "Linv_cuda.size()      = " << Linv_cuda.size() << endl;
    // cerr << "HybridJobs_GPU.size() = " << HybridJobs_GPU.size() << endl;
    // cerr << "rhats_GPU.size()      = " << rhats_GPU.size() << endl;

    MakeHybridJobList ((float*) cudaPos.data(), N, IonPos_GPU.data(), 
		       PolyRadii_GPU.data(), CutoffRadii_GPU.data(),
		       AtomicOrbitals.size(), L_cuda.data(), Linv_cuda.data(),
		       HybridJobs_GPU.data(), rhats_GPU.data(),
		       HybridData_GPU.data());

    CalcYlmRealCuda (rhats_GPU.data(), HybridJobs_GPU.data(),
		     Ylm_ptr_GPU.data(), dYlm_dtheta_ptr_GPU.data(), 
		     dYlm_dphi_ptr_GPU.data(), lMax, newpos.size());

    // evaluateHybridSplineReal (HybridJobs_GPU.data(), Ylm_ptr_GPU.data(), 
    // 			      AtomicOrbitals_GPU.data(), HybridData_GPU.data(),
    // 			      phi.data(), NumOrbitals, newpos.size(), lMax);
    evaluateHybridSplineReal (HybridJobs_GPU.data(), rhats_GPU.data(), Ylm_ptr_GPU.data(),
			      dYlm_dtheta_ptr_GPU.data(), dYlm_dphi_ptr_GPU.data(),
			      AtomicOrbitals_GPU.data(), HybridData_GPU.data(),
			      phi.data(), grad_lapl.data(), row_stride, 
			      NumOrbitals, newpos.size(), lMax);


    host_vector<CudaRealType*> phi_CPU (phi.size()), grad_lapl_CPU(phi.size());
    phi_CPU = phi;
    grad_lapl_CPU = grad_lapl;
    host_vector<CudaRealType> vals_CPU(NumOrbitals), GL_CPU(4*row_stride);
    host_vector<HybridJobType> HybridJobs_CPU(HybridJobs_GPU.size());
    HybridJobs_CPU = HybridJobs_GPU;
    host_vector<HybridDataFloat> HybridData_CPU(HybridData_GPU.size());
    HybridData_CPU = HybridData_GPU;
    for (int iw=0; iw<newpos.size(); iw++) 
      if (HybridJobs_CPU[iw] != BSPLINE_3D_JOB) {
	ValueVector_t CPUvals(NumOrbitals), CPUlapl(NumOrbitals);
	GradVector_t CPUgrad(NumOrbitals);
	HybridDataFloat &d = HybridData_CPU[iw];
	AtomicOrbital<double> &atom = AtomicOrbitals[d.ion];
	atom.evaluate (newpos[iw], CPUvals, CPUgrad, CPUlapl);

	cudaMemcpy (&vals_CPU[0], phi_CPU[iw], NumOrbitals*sizeof(float),
		    cudaMemcpyDeviceToHost);
	cudaMemcpy (&GL_CPU[0], grad_lapl_CPU[iw], 4*row_stride*sizeof(float),
		    cudaMemcpyDeviceToHost);
	fprintf (stderr, " %d %2.0f %2.0f %2.0f  %8.5f  %d %d\n",
		 iw, d.img[0], d.img[1], d.img[2], d.dist, d.ion, d.lMax);
	for (int j=0; j<NumOrbitals; j++) {
	  fprintf (stderr, "val[%2d]  = %10.5e %10.5e\n", 
		   j, vals_CPU[j], CPUvals[j]);
	  fprintf (stderr, "lapl[%2d] = %10.5e %10.5e\n", 
		   j, GL_CPU[3*row_stride+j], CPUlapl[j]);
	}
      }

#ifdef HYBRID_DEBUG
    host_vector<float> Ylm_CPU(Ylm_GPU.size());
    Ylm_CPU = Ylm_GPU;
    
    rhats_CPU = rhats_GPU;
    for (int i=0; i<rhats_CPU.size()/3; i++)
      fprintf (stderr, "rhat[%d] = [%10.6f %10.6f %10.6f]\n",
	       i, rhats_CPU[3*i+0], rhats_CPU[3*i+1], rhats_CPU[3*i+2]);

    host_vector<HybridJobType> HybridJobs_CPU(HybridJobs_GPU.size());
    HybridJobs_CPU = HybridJobs_GPU;
        
    host_vector<HybridDataFloat> HybridData_CPU(HybridData_GPU.size());
    HybridData_CPU = HybridData_GPU;

    cerr << "Before loop.\n";
    for (int i=0; i<newpos.size(); i++) 
      if (HybridJobs_CPU[i] != BSPLINE_3D_JOB) {
	cerr << "Inside if.\n";
	PosType rhat(rhats_CPU[3*i+0], rhats_CPU[3*i+1], rhats_CPU[3*i+2]);
	AtomicOrbital<double> &atom = AtomicOrbitals[HybridData_CPU[i].ion];
	int numlm = (atom.lMax+1)*(atom.lMax+1);
	vector<double> Ylm(numlm), dYlm_dtheta(numlm), dYlm_dphi(numlm);
	atom.CalcYlm (rhat, Ylm, dYlm_dtheta, dYlm_dphi);
	
	for (int lm=0; lm < numlm; lm++) {
	  fprintf (stderr, "lm=%3d  Ylm_CPU=%8.5f  Ylm_GPU=%8.5f\n",
		   lm, Ylm[lm], Ylm_CPU[3*i*Ylm_BS+lm]);
	}
      }

    fprintf (stderr, " N  img      dist    ion    lMax\n");
    for (int i=0; i<HybridData_CPU.size(); i++) {
      HybridDataFloat &d = HybridData_CPU[i];
      fprintf (stderr, " %d %2.0f %2.0f %2.0f  %8.5f  %d %d\n",
	       i, d.img[0], d.img[1], d.img[2], d.dist, d.ion, d.lMax);
    }
#endif

    // int N = newpos.size();
    // if (N > CurrentWalkers)
    //   resize_cuda(N);

    // AtomicPolyJobs_CPU.clear();
    // AtomicSplineJobs_CPU.clear();
    // rhats_CPU.clear();
    // BsplinePos_CPU.clear();
    // BsplineVals_CPU.clear();
    // BsplineGradLapl_CPU.clear();
    // int numAtomic = 0;

    // // First, sort electrons into three categories:
    // // 1) Interstitial region with 3D B-splines
    // // 2) Atomic region near origin:      polynomial  radial functions
    // // 3) Atomic region not near origin:  1D B-spline radial functions

    // for (int i=0; i<newpos.size(); i++) {
    //   PosType r = newpos[i];
    //   // Note: this assumes that the atomic radii are smaller than the simulation cell radius.
    //   for (int j=0; j<AtomicOrbitals.size(); j++) {
    // 	AtomicOrbital<complex<double> > &orb = AtomicOrbitals[j];
    // 	PosType dr = r - orb.Pos;
    // 	PosType u = PrimLattice.toUnit(dr);
    // 	for (int k=0; k<OHMMS_DIM; k++) 
    // 	  u[k] -= round(u[k]);
    // 	dr = PrimLattice.toCart(u);
    // 	RealType dist2 = dot (dr,dr);
    // 	if (dist2 < orb.PolyRadius * orb.PolyRadius) {
    // 	  AtomicPolyJob<CudaRealType> job;
    // 	  RealType dist = std::sqrt(dist2);
    // 	  job.dist = dist;
    // 	  RealType distInv = 1.0/dist;
    // 	  for (int k=0; k<OHMMS_DIM; k++) {
    // 	    CudaRealType x = distInv*dr[k];
    // 	    job.rhat[k] = distInv * dr[k];
    // 	    rhats_CPU.push_back(x);
    // 	  }
    // 	  job.lMax = orb.lMax;
    // 	  job.YlmIndex  = i;
    // 	  job.PolyOrder = orb.PolyOrder;
    // 	  //job.PolyCoefs = orb.PolyCoefs;
    // 	  AtomicPolyJobs_CPU.push_back(job);
    // 	  numAtomic++;
    // 	}
    // 	else if (dist2 < orb.CutoffRadius*orb.CutoffRadius) {
    // 	  AtomicSplineJob<CudaRealType> job;
    // 	   RealType dist = std::sqrt(dist2);
    // 	  job.dist = dist;
    // 	  RealType distInv = 1.0/dist;
    // 	  for (int k=0; k<OHMMS_DIM; k++) {
    // 	    CudaRealType x = distInv*dr[k];
    // 	    job.rhat[k] = distInv * dr[k];
    // 	    rhats_CPU.push_back(x);
    // 	  }
    // 	  job.lMax      = orb.lMax;
    // 	  job.YlmIndex  = i;
    // 	  job.phi       = phi[i];
    // 	  job.grad_lapl = grad_lapl[i];
    // 	  //job.PolyCoefs = orb.PolyCoefs;
    // 	  AtomicSplineJobs_CPU.push_back(job);
    // 	  numAtomic++;
    // 	}
    // 	else { // Regular 3D B-spline job
    // 	  BsplinePos_CPU
    // 	}
    //   }
    // }

    // //////////////////////////////////
    // // First, evaluate 3D B-splines //
    // //////////////////////////////////
    // int N = newpos.size();
    // CudaRealType plus_minus[2] = {1.0, -1.0};
    
    // if (cudaPos.size() < N) {
    //   hostPos.resize(N);
    //   cudaPos.resize(N);
    //   hostSign.resize(N);
    //   cudaSign.resize(N);
    // }

    // for (int iw=0; iw < N; iw++) {
    //   PosType r = newpos[iw];
    //   PosType ru(PrimLattice.toUnit(r));
    //   int image[OHMMS_DIM];
    //   for (int i=0; i<OHMMS_DIM; i++) {
    // 	RealType img = std::floor(ru[i]);
    // 	ru[i] -= img;
    // 	image[i] = (int) img;
    //   }
    //   int sign = 0;
    //   for (int i=0; i<OHMMS_DIM; i++) 
    // 	sign += HalfG[i] * image[i];
      
    //   hostSign[iw] = plus_minus[sign&1];
    //   hostPos[iw] = ru;
    // }

    // cudaPos = hostPos;
    // cudaSign = hostSign;
    // eval_multi_multi_UBspline_3d_cuda 
    //   (CudaMultiSpline, (CudaRealType*)(cudaPos.data()), cudaSign.data(), 
    //    phi.data(), N);

    // ////////////////////////////////////////////////////////////
    // // Next, evaluate spherical harmonics for atomic orbitals //
    // ////////////////////////////////////////////////////////////

    // // Evaluate Ylms
    // if (rhats_CPU.size()) {
    //   rhats_GPU = rhats_CPU;
    //   CalcYlmComplexCuda(rhats_GPU.data(), Ylm_ptr_GPU.data(), dYlm_dtheta_ptr_GPU.data(),
    // 			 dYlm_dphi_ptr_GPU.data(), lMax, numAtomic);
    //   cerr << "Calculated Ylms.\n";
    // }

    // ///////////////////////////////////////
    // // Next, evaluate 1D spline orbitals //
    // ///////////////////////////////////////
    // if (AtomicSplineJobs_CPU.size()) {
    //   AtomicSplineJobs_GPU = AtomicSplineJobs_CPU;

    // }
    // ///////////////////////////////////////////
    // // Next, evaluate 1D polynomial orbitals //
    // ///////////////////////////////////////////
    // if (AtomicSplineJobs_CPU.size()) {
    //   AtomicPolyJobs_GPU = AtomicPolyJobs_CPU;
    // }



  }

  template<> void
  EinsplineSetHybrid<double>::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
					cuda_vector<CudaComplexType*> &phi,
					cuda_vector<CudaComplexType*> &grad_lapl,
					int row_stride)
  {
  }

  template<> void
  EinsplineSetHybrid<double>::evaluate (vector<PosType> &pos, cuda_vector<CudaRealType*> &phi)
  {
  }

  template<> void
  EinsplineSetHybrid<double>::evaluate (vector<PosType> &pos, cuda_vector<CudaComplexType*> &phi)
  {
  }
  
  template<> string
  EinsplineSetHybrid<double>::Type()
  {
  }
  
  
  template<typename StorageType> SPOSetBase*
  EinsplineSetHybrid<StorageType>::makeClone() const
  {
    EinsplineSetHybrid<StorageType> *clone = 
      new EinsplineSetHybrid<StorageType> (*this);
    clone->registerTimers();
    return clone;
  }
  

  //////////////////////////////////
  // Complex StorageType versions //
  //////////////////////////////////

  template<> void
  EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, int iat,
					cuda_vector<CudaRealType*> &phi)
  {
    app_error() << "EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, int iat,\n"
		<< "			                            cuda_vector<CudaRealType*> &phi)\n"
		<< "not yet implemented.\n";
  }

  
  template<> void
  EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, int iat,
						  cuda_vector<CudaComplexType*> &phi)
  {
   app_error() << "EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, int iat,\n"
		<< "			                            cuda_vector<CudaComplexType*> &phi)\n"
	       << "not yet implemented.\n";
  }

  
  template<> void
  EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
						  cuda_vector<CudaRealType*> &phi)
  { 
   app_error() << "EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos,\n"
		<< "			                            cuda_vector<CudaRealType*> &phi)\n"
		<< "not yet implemented.\n";
  }

  template<> void
  EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos,
						  cuda_vector<CudaComplexType*> &phi)
  {
   app_error() << "EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, vector<PosType> ,\n"
		<< "			                            cuda_vector<CudaComplexType*> &phi)\n"
		<< "not yet implemented.\n";
  }

  template<> void
  EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
						  cuda_vector<CudaRealType*> &phi,
						  cuda_vector<CudaRealType*> &grad_lapl,
						  int row_stride)
  { 
    // int N = newpos.size();
    // if (N > CurrentWalkers)
    //   resize_cuda(N);

    // AtomicPolyJobs_CPU.clear();
    // AtomicSplineJobs_CPU.clear();
    // rhats_CPU.clear();
    // int numAtomic = 0;

    // // First, sort electrons into three categories:
    // // 1) Interstitial region with 3D-Bsplines
    // // 2) Atomic region near origin:      polynomial  radial functions
    // // 3) Atomic region not near origin:  1D B-spline radial functions

    // for (int i=0; i<newpos.size(); i++) {
    //   PosType r = newpos[i];
    //   // Note: this assumes that the atomic radii are smaller than the simulation cell radius.
    //   for (int j=0; j<AtomicOrbitals.size(); j++) {
    // 	AtomicOrbital<complex<double> > &orb = AtomicOrbitals[j];
    // 	PosType dr = r - orb.Pos;
    // 	PosType u = PrimLattice.toUnit(dr);
    // 	for (int k=0; k<OHMMS_DIM; k++) 
    // 	  u[k] -= round(u[k]);
    // 	dr = PrimLattice.toCart(u);
    // 	RealType dist2 = dot (dr,dr);
    // 	if (dist2 < orb.PolyRadius * orb.PolyRadius) {
    // 	  AtomicPolyJob<CudaRealType> job;
    // 	  RealType dist = std::sqrt(dist2);
    // 	  job.dist = dist;
    // 	  RealType distInv = 1.0/dist;
    // 	  for (int k=0; k<OHMMS_DIM; k++) {
    // 	    CudaRealType x = distInv*dr[k];
    // 	    job.rhat[k] = distInv * dr[k];
    // 	    rhats_CPU.push_back(x);
    // 	  }
    // 	  job.lMax = orb.lMax;
    // 	  job.YlmIndex  = i;
    // 	  job.PolyOrder = orb.PolyOrder;
    // 	  //job.PolyCoefs = orb.PolyCoefs;
    // 	  AtomicPolyJobs_CPU.push_back(job);
    // 	  numAtomic++;
    // 	}
    // 	else if (dist2 < orb.CutoffRadius*orb.CutoffRadius) {
    // 	  AtomicSplineJob<CudaRealType> job;
    // 	   RealType dist = std::sqrt(dist2);
    // 	  job.dist = dist;
    // 	  RealType distInv = 1.0/dist;
    // 	  for (int k=0; k<OHMMS_DIM; k++) {
    // 	    CudaRealType x = distInv*dr[k];
    // 	    job.rhat[k] = distInv * dr[k];
    // 	    rhats_CPU.push_back(x);
    // 	  }
    // 	  job.lMax      = orb.lMax;
    // 	  job.YlmIndex  = i;
    // 	  job.phi       = phi[i];
    // 	  job.grad_lapl = grad_lapl[i];
    // 	  //job.PolyCoefs = orb.PolyCoefs;
    // 	  AtomicSplineJobs_CPU.push_back(job);
    // 	  numAtomic++;
    // 	}
    // 	else { // Regular 3D B-spline job

    // 	}
    //   }
    // }

    // //////////////////////////////////
    // // First, evaluate 3D B-splines //
    // //////////////////////////////////

    // int N = walkers.size();
    // int M = CudaMultiSpline->num_splines;

    // if (CudaValuePointers.size() < N)
    //   resizeCuda(N);

    // if (cudaPos.size() < N) {
    //   hostPos.resize(N);
    //   cudaPos.resize(N);
    // }
    // for (int iw=0; iw < N; iw++) {
    //   PosType r = newpos[iw];
    //   PosType ru(PrimLattice.toUnit(r));
    //   ru[0] -= std::floor (ru[0]);
    //   ru[1] -= std::floor (ru[1]);
    //   ru[2] -= std::floor (ru[2]);
    //   hostPos[iw] = ru;
    // }

    // cudaPos = hostPos;

    // eval_multi_multi_UBspline_3d_c_vgl_cuda
    //   (CudaMultiSpline, (float*)cudaPos.data(),  Linv_cuda.data(), CudaValuePointers.data(), 
    //    CudaGradLaplPointers.data(), N, CudaMultiSpline->num_splines);

    // // Now, add on phases
    // for (int iw=0; iw < N; iw++) 
    //   hostPos[iw] = newpos[iw];
    // cudaPos = hostPos;
    
    // apply_phase_factors ((CUDA_PRECISION*) CudakPoints.data(),
    // 			 CudaMakeTwoCopies.data(),
    // 			 (CUDA_PRECISION*)cudaPos.data(),
    // 			 (CUDA_PRECISION**)CudaValuePointers.data(), phi.data(), 
    // 			 (CUDA_PRECISION**)CudaGradLaplPointers.data(), grad_lapl.data(),
    // 			 CudaMultiSpline->num_splines,  walkers.size(), row_stride);


    // ////////////////////////////////////////////////////////////
    // // Next, evaluate spherical harmonics for atomic orbitals //
    // ////////////////////////////////////////////////////////////

    // // Evaluate Ylms
    // if (rhats_CPU.size()) {
    //   rhats_GPU = rhats_CPU;
    //   CalcYlmComplexCuda(rhats_GPU.data(), Ylm_ptr_GPU.data(), dYlm_dtheta_ptr_GPU.data(),
    // 			 dYlm_dphi_ptr_GPU.data(), lMax, numAtomic);
    //   cerr << "Calculated Ylms.\n";
    // }

    // ///////////////////////////////////////
    // // Next, evaluate 1D spline orbitals //
    // ///////////////////////////////////////
    // if (AtomicSplineJobs_CPU.size()) {
    //   AtomicSplineJobs_GPU = AtomicSplineJobs_CPU;

    // }
    // ///////////////////////////////////////////
    // // Next, evaluate 1D polynomial orbitals //
    // ///////////////////////////////////////////
    // if (AtomicSplineJobs_CPU.size()) {
    //   AtomicPolyJobs_GPU = AtomicPolyJobs_CPU;
    // }

  }

  template<> void
  EinsplineSetHybrid<complex<double> >::evaluate (vector<Walker_t*> &walkers, vector<PosType> &newpos, 
  		 cuda_vector<CudaComplexType*> &phi,
  		 cuda_vector<CudaComplexType*> &grad_lapl,
  		 int row_stride)
  {
    app_error() << "EinsplineSetHybrid<complex<double> >::evaluate \n"
		<< "(vector<Walker_t*> &walkers, vector<PosType> &newpos, \n"
		<< " cuda_vector<CudaComplexType*> &phi,\n"
		<< " cuda_vector<CudaComplexType*> &grad_lapl,\n"
		<< " int row_stride)\n"
		<< "not yet implemented.\n";
  }

  template<> void
  EinsplineSetHybrid<complex<double> >::evaluate 
  (vector<PosType> &pos, cuda_vector<CudaRealType*> &phi)
  {
  }

  template<> void
  EinsplineSetHybrid<complex<double> >::evaluate 
  (vector<PosType> &pos, cuda_vector<CudaComplexType*> &phi)
  {
    app_error() << "EinsplineSetHybrid<complex<double> >::evaluate \n"
		<< "(vector<PosType> &pos, cuda_vector<CudaComplexType*> &phi)\n"
		<< "not yet implemented.\n";
  }
  
  template<> string
  EinsplineSetHybrid<complex<double> >::Type()
  {
  }
  






  template<>
  EinsplineSetHybrid<double>::EinsplineSetHybrid() :
    CurrentWalkers(0)
  {
    ValueTimer.set_name ("EinsplineSetHybrid::ValueOnly");
    VGLTimer.set_name ("EinsplineSetHybrid::VGL");
    ValueTimer.set_name ("EinsplineSetHybrid::VGLMat");
    EinsplineTimer.set_name ("EinsplineSetHybrid::Einspline");
    className = "EinsplineSeHybrid";
    TimerManager.addTimer (&ValueTimer);
    TimerManager.addTimer (&VGLTimer);
    TimerManager.addTimer (&VGLMatTimer);
    TimerManager.addTimer (&EinsplineTimer);
    for (int i=0; i<OHMMS_DIM; i++)
      HalfG[i] = 0;
  }

  template<>
  EinsplineSetHybrid<complex<double > >::EinsplineSetHybrid() :
    CurrentWalkers(0)
  {
    ValueTimer.set_name ("EinsplineSetHybrid::ValueOnly");
    VGLTimer.set_name ("EinsplineSetHybrid::VGL");
    ValueTimer.set_name ("EinsplineSetHybrid::VGLMat");
    EinsplineTimer.set_name ("EinsplineSetHybrid::Einspline");
    className = "EinsplineSeHybrid";
    TimerManager.addTimer (&ValueTimer);
    TimerManager.addTimer (&VGLTimer);
    TimerManager.addTimer (&VGLMatTimer);
    TimerManager.addTimer (&EinsplineTimer);
    for (int i=0; i<OHMMS_DIM; i++)
      HalfG[i] = 0;
  }

  template<> void
  EinsplineSetHybrid<double>::init_cuda()
  {
    // Setup B-spline Acuda matrix in constant memory
    init_atomic_cuda();

    host_vector<AtomicOrbitalCuda<CudaRealType> > AtomicOrbitals_CPU;
    const int BS=16;
    NumOrbitals = getOrbitalSetSize();
    // Bump up the stride to be a multiple of 512-bit bus width
    int lm_stride = ((NumOrbitals+BS-1)/BS)*BS;
    
    AtomicSplineCoefs_GPU.resize(AtomicOrbitals.size());
    vector<CudaRealType> IonPos_CPU, CutoffRadii_CPU, PolyRadii_CPU;

    for (int iat=0; iat<AtomicOrbitals.size(); iat++) {
      app_log() << "Copying atomic orbitals for ion " << iat << " to GPU memory.\n";
      AtomicOrbital<double> &atom = AtomicOrbitals[iat];
      for (int i=0; i<OHMMS_DIM; i++) 
	IonPos_CPU.push_back(atom.Pos[i]);
      CutoffRadii_CPU.push_back(atom.CutoffRadius);
      PolyRadii_CPU.push_back(atom.PolyRadius);
      AtomicOrbitalCuda<CudaRealType> atom_cuda;
      atom_cuda.lMax = atom.lMax;
      int numlm = (atom.lMax+1)*(atom.lMax+1);
      atom_cuda.lm_stride = lm_stride;
      atom_cuda.spline_stride = numlm * lm_stride;

      AtomicOrbital<double>::SplineType &cpu_spline = 
	*atom.get_radial_spline();
      atom_cuda.spline_dr_inv = cpu_spline.x_grid.delta_inv;
      int Ngrid = cpu_spline.x_grid.num;
      int spline_size = atom_cuda.spline_stride * (Ngrid+2);
      host_vector<float> spline_coefs(spline_size);
      AtomicSplineCoefs_GPU[iat].resize(spline_size);
      atom_cuda.spline_coefs = &AtomicSplineCoefs_GPU[iat][0];
      // Reorder and copy splines to GPU memory
      for (int igrid=0; igrid<Ngrid+2; igrid++)
	for (int lm=0; lm<numlm; lm++)
	  for (int orb=0; orb<NumOrbitals; orb++)
	    spline_coefs[igrid*atom_cuda.spline_stride +
			 lm   *atom_cuda.lm_stride + orb] =
	      cpu_spline.coefs[igrid*cpu_spline.x_stride + orb*numlm +lm];
      
      AtomicSplineCoefs_GPU[iat] = spline_coefs;
      
      AtomicOrbitals_CPU.push_back(atom_cuda);
    }
    AtomicOrbitals_GPU = AtomicOrbitals_CPU;
    IonPos_GPU      = IonPos_CPU;
    CutoffRadii_GPU = CutoffRadii_CPU;
    PolyRadii_GPU   = PolyRadii_CPU;
  }

  template<> void
  EinsplineSetHybrid<complex<double> >::init_cuda()
  {
    // Setup B-spline Acuda matrix in constant memory
    init_atomic_cuda();

    host_vector<AtomicOrbitalCuda<CudaRealType> > AtomicOrbitals_CPU;
    const int BS=16;
    NumOrbitals = getOrbitalSetSize();
    // Bump up the stride to be a multiple of 512-bit bus width
    int lm_stride = ((2*NumOrbitals+BS-1)/BS)*BS;
    
    AtomicSplineCoefs_GPU.resize(AtomicOrbitals.size());

    for (int iat=0; iat<AtomicOrbitals.size(); iat++) {
      AtomicOrbital<complex<double> > &atom = AtomicOrbitals[iat];
      AtomicOrbitalCuda<CudaRealType> atom_cuda;
      atom_cuda.lMax = atom.lMax;
      int numlm = (atom.lMax+1)*(atom.lMax+1);
      atom_cuda.lm_stride = lm_stride;
      atom_cuda.spline_stride = numlm * lm_stride;

      AtomicOrbital<complex<double> >::SplineType &cpu_spline = 
	*atom.get_radial_spline();
      int Ngrid = cpu_spline.x_grid.num;
      int spline_size = atom_cuda.spline_stride * (Ngrid+2);
      host_vector<float> spline_coefs(spline_size);
      AtomicSplineCoefs_GPU[iat].resize(spline_size);
      atom_cuda.spline_coefs = &AtomicSplineCoefs_GPU[0][0];
      // Reorder and copy splines to GPU memory
      for (int igrid=0; igrid<Ngrid+2; igrid++)
	for (int lm=0; lm<numlm; lm++)
	  for (int orb=0; orb<NumOrbitals; orb++) {
	    spline_coefs[2*(igrid*atom_cuda.spline_stride + lm*atom_cuda.lm_stride + orb)+0] =
	      cpu_spline.coefs[igrid*cpu_spline.x_stride + orb*numlm +lm].real();
	    spline_coefs[2*(igrid*atom_cuda.spline_stride + lm*atom_cuda.lm_stride + orb)+1] =
	      cpu_spline.coefs[igrid*cpu_spline.x_stride + orb*numlm +lm].imag();
	  }
      
      AtomicSplineCoefs_GPU[iat] = spline_coefs;
      AtomicOrbitals_CPU.push_back(atom_cuda);

    }
    AtomicOrbitals_GPU = AtomicOrbitals_CPU;

  }



  template class EinsplineSetHybrid<complex<double> >;
  template class EinsplineSetHybrid<        double  >;
  

}
