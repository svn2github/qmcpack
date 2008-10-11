#include "TwoBodyJastrowOrbitalBspline.h"
#include "CudaSpline.h"
#include "Lattice/ParticleBConds.h"

namespace qmcplusplus {

  void
  TwoBodyJastrowOrbitalBspline::recompute(vector<Walker_t*> &walkers)
  {
  }
  
  void
  TwoBodyJastrowOrbitalBspline::reserve 
  (PointerPool<cuda_vector<CudaRealType> > &pool)
  {
    ROffset = pool.reserve(3*N);
  }
  
  void 
  TwoBodyJastrowOrbitalBspline::addFunc(const string& aname, int ia, int ib, FT* j)
  {
    TwoBodyJastrowOrbital<BsplineFunctor<OrbitalBase::RealType> >::addFunc(aname, ia, ib, j);
    CudaSpline<CudaReal> *newSpline = new CudaSpline<CudaReal>(*j);
    UniqueSplines.push_back(newSpline);

    if(ia==ib) {
      if(ia==0) { //first time, assign everything
	int ij=0;
	for(int ig=0; ig<NumGroups; ++ig) 
	  for(int jg=0; jg<NumGroups; ++jg, ++ij) 
	    if(GPUSplines[ij]==0) GPUSplines[ij]=newSpline;
      }
    }
    else {
      GPUSplines[ia*NumGroups+ib]=newSpline;
      GPUSplines[ib*NumGroups+ia]=newSpline; 
    }
  }
  

  void
  TwoBodyJastrowOrbitalBspline::addLog (vector<Walker_t*> &walkers, 
					vector<RealType> &logPsi)
  {
    app_log() << "TwoBodyJastrowOrbitalBspline::addLog.\n";
    if (SumGPU.size() < walkers.size()) {
      SumGPU.resize(walkers.size());
      SumHost.resize(walkers.size());
      RlistGPU.resize(walkers.size());
      RlistHost.resize(walkers.size());
    }

    int numR = 3*N*walkers.size();
    if (RGPU.size() < numR) {
      RHost.resize(numR);
      RGPU.resize(numR);
    }
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t &walker = *(walkers[iw]);
      SumHost[iw] = 0.0;
      for (int ptcl=0; ptcl<N; ptcl++)
	for (int dim=0; dim<OHMMS_DIM; dim++)
	  RHost[OHMMS_DIM*(N*iw+ptcl)+dim] = walker.R[ptcl][dim];
      //app_log() << "RHost[0] = " << RHost[OHMMS_DIM*N*iw] << endl;
      RlistHost[iw] = &(RGPU[OHMMS_DIM*N*iw]);
    }
    SumGPU = SumHost;
    RGPU = RHost;
    RlistGPU = RlistHost;

    DTD_BConds<double,3,SUPERCELL_BULK> bconds;


    double host_sum = 0.0;
    for (int group1=0; group1<PtclRef.groups(); group1++) {
      int first1 = PtclRef.first(group1);
      int last1  = PtclRef.last(group1) -1;
      for (int group2=group1; group2<PtclRef.groups(); group2++) {
	int first2 = PtclRef.first(group2);
	int last2  = PtclRef.last(group2) -1;

	double factor = (group1 == group2) ? 0.5 : 1.0;
	for (int e1=first1; e1 <= last1; e1++)
	  for (int e2=first2; e2 <= last2; e2++) {
	    PosType disp = walkers[0]->R[e2] - walkers[0]->R[e1];
	    double dist = std::sqrt(bconds.apply(PtclRef.Lattice, disp));
	    if (e1 != e2)
	      host_sum -= factor * F[group2*NumGroups + group1]->evaluate(dist);
	  }
	
	  CudaSpline<CudaReal> &spline = *(GPUSplines[group1*NumGroups+group2]);
	  two_body_sum (RlistGPU.data(), first1, last1, first2, last2, 
			spline.coefs.data(), spline.coefs.size(),
			spline.rMax, L.data(), Linv.data(),
			SumGPU.data(), walkers.size());
      }
    }
    // Copy data back to CPU memory
    
    SumHost = SumGPU;
    for (int iw=0; iw<walkers.size(); iw++) {
      logPsi[iw] -= SumHost[iw];
    }
    fprintf (stderr, "host = %25.16f\n", host_sum);
    fprintf (stderr, "cuda = %25.16f\n", logPsi[0]);
  }
  
  void
  TwoBodyJastrowOrbitalBspline::update (vector<Walker_t*> &walkers, int iat)
  {
    
  }
  
  void
  TwoBodyJastrowOrbitalBspline::ratio
  (vector<Walker_t*> &walkers, int iat, vector<PosType> &new_pos,
   vector<ValueType> &psi_ratios,	vector<GradType>  &grad,
   vector<ValueType> &lapl)
  {
    
  }

  void
  TwoBodyJastrowOrbitalBspline::gradLapl (vector<Walker_t*> &walkers, 
					  GradMatrix_t &grads,
					  ValueMatrix_t &lapl)
  {

  }
  
}
