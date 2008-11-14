#include "TwoBodyJastrowOrbitalBspline.h"
#include "CudaSpline.h"
#include "Lattice/ParticleBConds.h"

namespace qmcplusplus {

  void
  TwoBodyJastrowOrbitalBspline::recompute(MCWalkerConfiguration &W)
  {
  }
  
  void
  TwoBodyJastrowOrbitalBspline::reserve 
  (PointerPool<cuda_vector<CudaRealType> > &pool)
  {
  }

  void 
  TwoBodyJastrowOrbitalBspline::checkInVariables(opt_variables_type& active)
  {
    TwoBodyJastrowOrbital<BsplineFunctor<OrbitalBase::RealType> >::checkInVariables(active);
    for (int i=0; i<NumGroups*NumGroups; i++)
      GPUSplines[i]->set(*F[i]);
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
  TwoBodyJastrowOrbitalBspline::addLog (MCWalkerConfiguration &W, 
					vector<RealType> &logPsi)
  {
    vector<Walker_t*> &walkers = W.WalkerList;
    app_log() << "TwoBodyJastrowOrbitalBspline::addLog.\n";
    if (SumGPU.size() < walkers.size()) {
      SumGPU.resize(walkers.size());
      SumHost.resize(walkers.size());
      UpdateListHost.resize(walkers.size());
      UpdateListGPU.resize(walkers.size());
    }

    int numGL = 4*N*walkers.size();
    if (GradLaplGPU.size()  < numGL) {
      GradLaplGPU.resize(numGL);
      GradLaplHost.resize(numGL);
    }
    CudaReal RHost[OHMMS_DIM*N*walkers.size()];
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t &walker = *(walkers[iw]);
      SumHost[iw] = 0.0;
      CudaReal *dest = (CudaReal*)walker.R_GPU.data();
    }
    SumGPU = SumHost;

//     DTD_BConds<double,3,SUPERCELL_BULK> bconds;
//     double host_sum = 0.0;

    for (int group1=0; group1<PtclRef.groups(); group1++) {
      int first1 = PtclRef.first(group1);
      int last1  = PtclRef.last(group1) -1;
      for (int group2=group1; group2<PtclRef.groups(); group2++) {
	int first2 = PtclRef.first(group2);
	int last2  = PtclRef.last(group2) -1;

// 	double factor = (group1 == group2) ? 0.5 : 1.0;
// 	for (int e1=first1; e1 <= last1; e1++)
// 	  for (int e2=first2; e2 <= last2; e2++) {
// 	    PosType disp = walkers[0]->R[e2] - walkers[0]->R[e1];
// 	    double dist = std::sqrt(bconds.apply(PtclRef.Lattice, disp));
// 	    if (e1 != e2)
// 	      host_sum -= factor * F[group2*NumGroups + group1]->evaluate(dist);
// 	  }
	
	  CudaSpline<CudaReal> &spline = *(GPUSplines[group1*NumGroups+group2]);
	  two_body_sum (W.RList_GPU.data(), first1, last1, first2, last2, 
			spline.coefs.data(), spline.coefs.size(),
			spline.rMax, L.data(), Linv.data(),
			SumGPU.data(), walkers.size());
      }
    }
    // Copy data back to CPU memory
    
    SumHost = SumGPU;
    for (int iw=0; iw<walkers.size(); iw++) 
      logPsi[iw] -= SumHost[iw];
//     fprintf (stderr, "host = %25.16f\n", host_sum);
    fprintf (stderr, "cuda = %25.16f\n", logPsi[10]);
  }
  
  void
  TwoBodyJastrowOrbitalBspline::update (vector<Walker_t*> &walkers, int iat)
  {
    // for (int iw=0; iw<walkers.size(); iw++) 
    //   UpdateListHost[iw] = (CudaReal*)walkers[iw]->R_GPU.data();
    // UpdateListGPU = UpdateListHost;
    
    // two_body_update(UpdateListGPU.data(), N, iat, walkers.size());
  }
  
  void
  TwoBodyJastrowOrbitalBspline::ratio
  (MCWalkerConfiguration &W, int iat,
   vector<ValueType> &psi_ratios,	vector<GradType>  &grad,
   vector<ValueType> &lapl)
  {
    vector<Walker_t*> &walkers = W.WalkerList;
#ifdef CPU_RATIO
    DTD_BConds<double,3,SUPERCELL_BULK> bconds;

    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t &walker = *(walkers[iw]);
      double sum = 0.0;

      int group2 = PtclRef.GroupID (iat);
      for (int group1=0; group1<PtclRef.groups(); group1++) {
	int first1 = PtclRef.first(group1);
	int last1  = PtclRef.last(group1);
	  double factor = (group1 == group2) ? 0.5 : 1.0;
	  int id = group1*NumGroups + group2;
	  FT* func = F[id];
	  for (int ptcl1=first1; ptcl1<last1; ptcl1++) {
	    PosType disp = walkers[iw]->R[ptcl1] - walkers[iw]->R[iat];
	    double dist = std::sqrt(bconds.apply(PtclRef.Lattice, disp));
	    sum += factor*func->evaluate(dist);
	    disp = walkers[iw]->R[ptcl1] - new_pos[iw];
	    dist = std::sqrt(bconds.apply(PtclRef.Lattice, disp));
	    sum -= factor*func->evaluate(dist);

	  }
      }
      psi_ratios[iw] *= std::exp(-sum);
    }
#else

    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t &walker = *(walkers[iw]);
      SumHost[iw] = 0.0;
    }
    SumGPU = SumHost;

    int newGroup = PtclRef.GroupID[iat];

    for (int group=0; group<PtclRef.groups(); group++) {
      int first = PtclRef.first(group);
      int last  = PtclRef.last(group) -1;
	
      CudaSpline<CudaReal> &spline = *(GPUSplines[group*NumGroups+newGroup]);
      two_body_ratio (W.RList_GPU.data(), first, last, N, 
      		      (CudaReal*)W.Rnew_GPU.data(), iat, 
      		      spline.coefs.data(), spline.coefs.size(),
      		      spline.rMax, L.data(), Linv.data(),
      		      SumGPU.data(), walkers.size());
    }
    // Copy data back to CPU memory
    
    SumHost = SumGPU;
    for (int iw=0; iw<walkers.size(); iw++) 
      psi_ratios[iw] *= std::exp(-SumHost[iw]);
#endif
  }

  
  void
  TwoBodyJastrowOrbitalBspline::NLratios 
  (MCWalkerConfiguration &W,  vector<NLjob> &jobList,
   vector<PosType> &quadPoints, vector<ValueType> &psi_ratios)
  {
    vector<Walker_t*> &walkers = W.WalkerList;
    int njobs = jobList.size();
    if (NL_JobListHost.size() < njobs) {
      NL_JobListHost.resize(njobs);      
      NL_SplineCoefsListHost.resize(njobs);
      NL_NumCoefsHost.resize(njobs);
      NL_rMaxHost.resize(njobs);
    }
    if (NL_RatiosHost.size() < quadPoints.size()) {
      NL_QuadPointsHost.resize(OHMMS_DIM*quadPoints.size());
      NL_QuadPointsGPU.resize(OHMMS_DIM*quadPoints.size());
      NL_RatiosHost.resize(quadPoints.size());
      NL_RatiosGPU.resize(quadPoints.size());
    }
    int iratio = 0;
    for (int ijob=0; ijob < njobs; ijob++) {
      NLjob &job = jobList[ijob];
      NLjobGPU<CudaReal> &jobGPU = NL_JobListHost[ijob];
      jobGPU.R             = (CudaReal*)walkers[job.walker]->R_GPU.data();
      jobGPU.Elec          = job.elec;
      jobGPU.QuadPoints    = &(NL_QuadPointsGPU[OHMMS_DIM*iratio]);
      jobGPU.NumQuadPoints = job.numQuadPoints;
      jobGPU.Ratios        = &(NL_RatiosGPU[iratio]);
      iratio += job.numQuadPoints;
    }
    NL_JobListGPU         = NL_JobListHost;
    
    // Copy quad points to GPU
    for (int iq=0; iq<quadPoints.size(); iq++) {
      NL_RatiosHost[iq] = psi_ratios[iq];
      for (int dim=0; dim<OHMMS_DIM; dim++)
	NL_QuadPointsHost[OHMMS_DIM*iq + dim] = quadPoints[iq][dim];
    }
    NL_RatiosGPU = NL_RatiosHost;
    NL_QuadPointsGPU = NL_QuadPointsHost;
    
    // Now, loop over electron groups
    for (int group=0; group<PtclRef.groups(); group++) {
      int first = PtclRef.first(group);
      int last  = PtclRef.last(group) -1;
      for (int ijob=0; ijob<jobList.size(); ijob++) {
	int newGroup = PtclRef.GroupID[jobList[ijob].elec];
	CudaSpline<CudaReal> &spline = *(GPUSplines[group*NumGroups+newGroup]);
	NL_SplineCoefsListHost[ijob] = spline.coefs.data();
	NL_NumCoefsHost[ijob] = spline.coefs.size();
	NL_rMaxHost[ijob]     = spline.rMax;
      }
      NL_SplineCoefsListGPU = NL_SplineCoefsListHost;
      NL_NumCoefsGPU        = NL_NumCoefsHost;
      NL_rMaxGPU            = NL_rMaxHost;
      two_body_NLratios(NL_JobListGPU.data(), first, last,
			NL_SplineCoefsListGPU.data(), NL_NumCoefsGPU.data(),
			NL_rMaxGPU.data(), L.data(), Linv.data(), njobs);
    }
    NL_RatiosHost = NL_RatiosGPU;
    for (int i=0; i < psi_ratios.size(); i++)
      psi_ratios[i] = NL_RatiosHost[i];
  }


  void
  TwoBodyJastrowOrbitalBspline::gradLapl (MCWalkerConfiguration &W, 
					  GradMatrix_t &grad,
					  ValueMatrix_t &lapl)
  {
    vector<Walker_t*> &walkers = W.WalkerList;
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t &walker = *(walkers[iw]);
      SumHost[iw] = 0.0;
    }
    SumGPU = SumHost;

    for (int i=0; i<walkers.size()*4*N; i++)
      GradLaplHost[i] = 0.0;
    GradLaplGPU = GradLaplHost;


#ifdef CUDA_DEBUG
    vector<CudaReal> CPU_GradLapl(4*N);
    DTD_BConds<double,3,SUPERCELL_BULK> bconds;

    int iw = 0;

    for (int group1=0; group1<PtclRef.groups(); group1++) {
      int first1 = PtclRef.first(group1);
      int last1  = PtclRef.last(group1) -1;
      for (int ptcl1=first1; ptcl1<=last1; ptcl1++) {
	PosType grad(0.0, 0.0, 0.0);
	double lapl(0.0);
	for (int group2=0; group2<PtclRef.groups(); group2++) {
	  int first2 = PtclRef.first(group2);
	  int last2  = PtclRef.last(group2) -1;
	  int id = group2*NumGroups + group1;
	  FT* func = F[id];
	  for (int ptcl2=first2; ptcl2<=last2; ptcl2++) {
	    if (ptcl1 != ptcl2 ) {
	      PosType disp = walkers[iw]->R[ptcl2] - walkers[iw]->R[ptcl1];
	      double dist = std::sqrt(bconds.apply(PtclRef.Lattice, disp));
	      double u, du, d2u;
	      u = func->evaluate(dist, du, d2u);
	      du /= dist;
	      grad += disp * du;
	      lapl += d2u + 2.0*du;
	    }
	  }
	}
	CPU_GradLapl[ptcl1*4+0] = grad[0];
	CPU_GradLapl[ptcl1*4+1] = grad[1];
	CPU_GradLapl[ptcl1*4+2] = grad[2];
	CPU_GradLapl[ptcl1*4+3] = lapl;
      }
    }
#endif
    for (int group1=0; group1<PtclRef.groups(); group1++) {
      int first1 = PtclRef.first(group1);
      int last1  = PtclRef.last(group1) -1;
      for (int group2=0; group2<PtclRef.groups(); group2++) {
	int first2 = PtclRef.first(group2);
	int last2  = PtclRef.last(group2) -1;

	CudaSpline<CudaReal> &spline = *(GPUSplines[group1*NumGroups+group2]);
	two_body_grad_lapl (W.RList_GPU.data(), first1, last1, first2, last2, 
			    spline.coefs.data(), spline.coefs.size(),
			    spline.rMax, L.data(), Linv.data(),
			    GradLaplGPU.data(), 4*N, walkers.size());
      }
    }
    // Copy data back to CPU memory
    GradLaplHost = GradLaplGPU;

#ifdef CUDA_DEBUG
    fprintf (stderr, "GPU  grad = %12.5e %12.5e %12.5e   Lapl = %12.5e\n", 
	     GradLaplHost[0],  GradLaplHost[1], GradLaplHost[2], GradLaplHost[3]);
    fprintf (stderr, "CPU  grad = %12.5e %12.5e %12.5e   Lapl = %12.5e\n", 
	     CPU_GradLapl[0],  CPU_GradLapl[1], CPU_GradLapl[2], CPU_GradLapl[3]);
#endif

    for (int iw=0; iw<walkers.size(); iw++) {
      for (int ptcl=0; ptcl<N; ptcl++) {
	for (int i=0; i<OHMMS_DIM; i++) 
	  grad(iw,ptcl)[i] += GradLaplHost[4*N*iw + 4*ptcl + i];
	lapl(iw,ptcl) += GradLaplHost[4*N*iw+ + 4*ptcl +3];
      }
    }
  }
  
}
