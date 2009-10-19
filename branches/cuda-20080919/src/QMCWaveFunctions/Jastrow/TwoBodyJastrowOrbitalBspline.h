#ifndef TWO_BODY_JASTROW_ORBITAL_BSPLINE_H
#define TWO_BODY_JASTROW_ORBITAL_BSPLINE_H

#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrowOrbital.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "Configuration.h"
#include "QMCWaveFunctions/Jastrow/CudaSpline.h"
#include "QMCWaveFunctions/Jastrow/BsplineJastrowCuda.h"

namespace qmcplusplus {

  class TwoBodyJastrowOrbitalBspline : 
    public TwoBodyJastrowOrbital<BsplineFunctor<OrbitalBase::RealType> > 
  {
  private:
    typedef CUDA_PRECISION CudaReal;
    //typedef double CudaReal;

    vector<CudaSpline<CudaReal>*> GPUSplines, UniqueSplines;
    int MaxCoefs;
    ParticleSet &PtclRef;
    cuda_vector<CudaReal> L, Linv;

    cuda_vector<CudaReal*> UpdateListGPU;
    cuda_vector<CudaReal> SumGPU, GradLaplGPU, OneGradGPU;

    host_vector<CudaReal*> UpdateListHost;
    host_vector<CudaReal> SumHost, GradLaplHost, OneGradHost;
    host_vector<CudaReal> SplineDerivsHost;
    cuda_vector<CudaReal> SplineDerivsGPU;
    host_vector<CudaReal*> DerivListHost;
    cuda_vector<CudaReal*> DerivListGPU;

    host_vector<CudaReal*> NL_SplineCoefsListHost;
    cuda_vector<CudaReal*> NL_SplineCoefsListGPU;
    host_vector<NLjobGPU<CudaReal> > NL_JobListHost;
    cuda_vector<NLjobGPU<CudaReal> > NL_JobListGPU;
    host_vector<int> NL_NumCoefsHost, NL_NumQuadPointsHost;
    cuda_vector<int> NL_NumCoefsGPU,  NL_NumQuadPointsGPU;
    host_vector<CudaReal> NL_rMaxHost, NL_QuadPointsHost, NL_RatiosHost;
    cuda_vector<CudaReal> NL_rMaxGPU,  NL_QuadPointsGPU,  NL_RatiosGPU;
  public:
    typedef BsplineFunctor<OrbitalBase::RealType> FT;
    typedef ParticleSet::Walker_t     Walker_t;

    
    void checkInVariables(opt_variables_type& active);
    void addFunc(const string& aname, int ia, int ib, FT* j);
    void recompute(MCWalkerConfiguration &W, bool firstTime);
    void reserve (PointerPool<cuda_vector<CudaRealType> > &pool);
    void addLog (MCWalkerConfiguration &W, vector<RealType> &logPsi);
    void update (vector<Walker_t*> &walkers, int iat);
    void update (const vector<Walker_t*> &walkers, const vector<int> &iatList) 
    { /* This function doesn't really need to return the ratio */ }

    void ratio (MCWalkerConfiguration &W, int iat,
		vector<ValueType> &psi_ratios,	vector<GradType>  &grad,
		vector<ValueType> &lapl);
    void ratio (vector<Walker_t*> &walkers,    vector<int> &iatList,
		vector<PosType> &rNew, vector<ValueType> &psi_ratios, 
		vector<GradType>  &grad, vector<ValueType> &lapl)
    { /* This function doesn't really need to return the ratio */ }

    void addGradient(MCWalkerConfiguration &W, int iat, 
		     vector<GradType> &grad);
    void gradLapl (MCWalkerConfiguration &W, GradMatrix_t &grads,
		   ValueMatrix_t &lapl);
    void NLratios (MCWalkerConfiguration &W,  vector<NLjob> &jobList,
		   vector<PosType> &quadPoints, vector<ValueType> &psi_ratios);
    
    void resetParameters(const opt_variables_type& active);
    
    // Evaluates the derivatives of log psi and laplacian log psi w.r.t.
    // the parameters for optimization.  First index of the ValueMatrix is
    // the parameter.  The second is the walker.
    void
    evaluateDerivatives (MCWalkerConfiguration &W, 
			 const opt_variables_type& optvars,
			 ValueMatrix_t &dlogpsi, 
			 ValueMatrix_t &dlapl_over_psi);

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
	  LinvHost[OHMMS_DIM*i+j] = (CudaReal)pset.Lattice.b(j)[i];
	}
      // for (int i=0; i<OHMMS_DIM; i++)
      // 	for (int j=0; j<OHMMS_DIM; j++) {
      // 	  double sum = 0.0;
      // 	  for (int k=0; k<OHMMS_DIM; k++)
      // 	    sum += LHost[OHMMS_DIM*i+k]*LinvHost[OHMMS_DIM*k+j];
	  
      // 	  if (i == j) sum -= 1.0;
      // 	  if (std::fabs(sum) > 1.0e-5) {
      // 	    app_error() << "sum = " << sum << endl;
      // 	    app_error() << "Linv * L != identity.\n";
      // 	    abort();
      // 	  }
      // 	}

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
