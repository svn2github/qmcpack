#ifndef TWO_BODY_JASTROW_ORBITAL_BSPLINE_H
#define TWO_BODY_JASTROW_ORBITAL_BSPLINE_H

#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrowOrbital.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "Configuration.h"

namespace qmcplusplus {

  template<typename T>
  struct CudaSpline
  {
    cuda_vector<T> coefs;
    T rMax;

    template<typename T2>
    CudaSpline (BsplineFunctor<T2> &func)
    {
      int num_coefs = func.SplineCoefs.size();
      host_vector<T> coefs_h(num_coefs);
      for (int i=0; i<num_coefs; i++) {
	coefs_h[i] = func.SplineCoefs[i];
	app_log() << "coefs_h[" << i << "] = " << coefs_h[i] << endl;
      }
      coefs = coefs_h;
      rMax = func.Rcut;
    }
  };


  class TwoBodyJastrowOrbitalBspline : 
    public TwoBodyJastrowOrbital<BsplineFunctor<OrbitalBase::RealType> > 
  {
  private:
    typedef float CudaReal;
    //typedef double CudaReal;

    int ROffset;
    vector<CudaSpline<CudaReal>*> GPUSplines, UniqueSplines;
    ParticleSet &PtclRef;
    cuda_vector<CudaReal> L, Linv;

    cuda_vector<CudaReal*> RlistGPU, UpdateListGPU;
    cuda_vector<CudaReal> SumGPU, RnewGPU, GradLaplGPU;

    host_vector<CudaReal*> RlistHost, UpdateListHost;
    host_vector<CudaReal> SumHost, RnewHost, GradLaplHost;
  public:
    typedef BsplineFunctor<OrbitalBase::RealType> FT;
    typedef ParticleSet::Walker_t     Walker_t;

    void addFunc(const string& aname, int ia, int ib, FT* j);
    void recompute(vector<Walker_t*> &walkers);
    void reserve (PointerPool<cuda_vector<CudaRealType> > &pool);
    void addLog (vector<Walker_t*> &walkers, vector<RealType> &logPsi);
    void update (vector<Walker_t*> &walkers, int iat);
    void ratio (vector<Walker_t*> &walkers, int iat, vector<PosType> &new_pos,
		vector<ValueType> &psi_ratios,	vector<GradType>  &grad,
		vector<ValueType> &lapl);
    void gradLapl (vector<Walker_t*> &walkers, GradMatrix_t &grads,
		   ValueMatrix_t &lapl);

    TwoBodyJastrowOrbitalBspline(ParticleSet& pset) :
      TwoBodyJastrowOrbital<BsplineFunctor<OrbitalBase::RealType> > (pset),
      PtclRef(pset)
    {
      int nsp = NumGroups = pset.groups();
      GPUSplines.resize(nsp*nsp,0);
      host_vector<CudaReal> LHost(OHMMS_DIM*OHMMS_DIM), 
	LinvHost(OHMMS_DIM*OHMMS_DIM);
      for (int i=0; i<OHMMS_DIM; i++)
	for (int j=0; j<OHMMS_DIM; j++) {
	  LHost[OHMMS_DIM*i+j]    = (CudaReal)pset.Lattice.a(i)[j];
	  LinvHost[OHMMS_DIM*i+j] = (CudaReal)pset.Lattice.b(i)[j];
	}
//       fprintf (stderr, "Identity should follow:\n");
//       for (int i=0; i<3; i++){
// 	for (int j=0; j<3; j++) {
// 	  CudaReal val = 0.0f;
// 	  for (int k=0; k<3; k++)
// 	    val += LinvHost[3*i+k]*LHost[3*k+j];
// 	  fprintf (stderr, "  %8.3f", val);
// 	}
// 	fprintf (stderr, "\n");
//       }


      L = LHost;
      Linv = LinvHost;
    }
  };
}


#endif
