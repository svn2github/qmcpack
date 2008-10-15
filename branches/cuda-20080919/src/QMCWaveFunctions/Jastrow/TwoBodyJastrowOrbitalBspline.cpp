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
    int elemSize = 
      std::max((unsigned long)1,sizeof(CudaReal)/sizeof(CUDA_PRECISION));
    ROffset = pool.reserve(3*(N+1) * elemSize);
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
      RnewHost.resize (OHMMS_DIM*walkers.size());
      RnewGPU.resize  (OHMMS_DIM*walkers.size());
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
      CudaReal *dest = (CudaReal*)&(walker.cuda_DataSet[ROffset]);
      for (int ptcl=0; ptcl<N; ptcl++)
      	for (int dim=0; dim<OHMMS_DIM; dim++)
      	  RHost[OHMMS_DIM*(iw*N+ptcl)+dim] = walker.R[ptcl][dim];
      cudaMemcpy(dest, &(RHost[OHMMS_DIM*iw*N]), 
		 OHMMS_DIM*N*sizeof(CudaReal), cudaMemcpyHostToDevice);
      RlistHost[iw] = dest;
    }
    
    SumGPU = SumHost;
    RlistGPU = RlistHost;

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
	  two_body_sum (RlistGPU.data(), first1, last1, first2, last2, 
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
    for (int iw=0; iw<walkers.size(); iw++) 
      UpdateListHost[iw] = (CudaReal*)&(walkers[iw]->cuda_DataSet[ROffset]);
    UpdateListGPU = UpdateListHost;
    
    two_body_update(UpdateListGPU.data(), N, iat, walkers.size());
  }
  
  void
  TwoBodyJastrowOrbitalBspline::ratio
  (vector<Walker_t*> &walkers, int iat, vector<PosType> &new_pos,
   vector<ValueType> &psi_ratios,	vector<GradType>  &grad,
   vector<ValueType> &lapl)
  {
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
      for (int dim=0; dim<OHMMS_DIM; dim++)
	RnewHost[OHMMS_DIM*iw+dim] = new_pos[iw][dim];
    }
    SumGPU = SumHost;
    RnewGPU = RnewHost;

    int newGroup = PtclRef.GroupID[iat];

    for (int group=0; group<PtclRef.groups(); group++) {
      int first = PtclRef.first(group);
      int last  = PtclRef.last(group) -1;
	
      CudaSpline<CudaReal> &spline = *(GPUSplines[group*NumGroups+newGroup]);
      two_body_ratio (RlistGPU.data(), first, last, N, 
      		      RnewGPU.data(), iat, 
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
  TwoBodyJastrowOrbitalBspline::gradLapl (vector<Walker_t*> &walkers, 
					  GradMatrix_t &grad,
					  ValueMatrix_t &lapl)
  {
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
	two_body_grad_lapl (RlistGPU.data(), first1, last1, first2, last2, 
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