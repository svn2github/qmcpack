#include "OneBodyJastrowOrbitalBspline.h"
#include "CudaSpline.h"
#include "Lattice/ParticleBConds.h"

namespace qmcplusplus {

  void
  OneBodyJastrowOrbitalBspline::recompute(MCWalkerConfiguration &W)
  {
  }
  
  void
  OneBodyJastrowOrbitalBspline::reserve 
  (PointerPool<cuda_vector<CudaRealType> > &pool)
  {
  }

  void 
  OneBodyJastrowOrbitalBspline::checkInVariables(opt_variables_type& active)
  {
    OneBodyJastrowOrbital<BsplineFunctor<OrbitalBase::RealType> >::checkInVariables(active);
    for (int i=0; i<NumCenterGroups; i++)
      GPUSplines[i]->set(*Fs[i]);
  }
  
  void 
  OneBodyJastrowOrbitalBspline::addFunc(int i, FT* j)
  {
    OneBodyJastrowOrbital<BsplineFunctor<OrbitalBase::RealType> >::addFunc(i, j);
    CudaSpline<CudaReal> *newSpline = new CudaSpline<CudaReal>(*j);
    UniqueSplines.push_back(newSpline);

    if(i==0) { //first time, assign everything
      for(int ig=0; ig<NumCenterGroups; ++ig) 
	if(GPUSplines[ig]==0) GPUSplines[ig]=newSpline;
    }
    else 
      GPUSplines[i]=newSpline;
  }
  

  void
  OneBodyJastrowOrbitalBspline::addLog (MCWalkerConfiguration &W, 
					vector<RealType> &logPsi)
  {
    app_log() << "OneBodyJastrowOrbitalBspline::addLog.\n";
    vector<Walker_t*> &walkers = W.WalkerList;
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
    }
    
    SumGPU = SumHost;
    int efirst = 0;
    int elast = N-1;

    for (int cgroup=0; cgroup<NumCenterGroups; cgroup++) {
      int cfirst = CenterFirst[cgroup];
      int clast  = CenterLast[cgroup];
      
      CudaSpline<CudaReal> &spline = *(GPUSplines[cgroup]);
      if (GPUSplines[cgroup]) {
	one_body_sum (C.data(), W.RList_GPU.data(), cfirst, clast, efirst, elast, 
		      spline.coefs.data(), spline.coefs.size(),
		      spline.rMax, L.data(), Linv.data(),
		      SumGPU.data(), walkers.size());
      }
      // Copy data back to CPU memory
    }
    SumHost = SumGPU;
    for (int iw=0; iw<walkers.size(); iw++) 
      logPsi[iw] -= SumHost[iw];

#ifdef CUDA_DEBUG
    DTD_BConds<double,3,SUPERCELL_BULK> bconds;
    double host_sum = 0.0;

    for (int cptcl=0; cptcl < CenterRef.getTotalNum(); cptcl++) {
      PosType c = CenterRef.R[cptcl];
      for (int eptcl=0; eptcl<N; eptcl++) {
	PosType r = walkers[0]->R[eptcl];
	PosType disp = r - c;
	double dist = std::sqrt(bconds.apply(ElecRef.Lattice, disp));
	host_sum -= Fs[cptcl]->evaluate(dist);
      }
    }

    fprintf (stderr, "host = %25.16f\n", host_sum);
    fprintf (stderr, "cuda = %25.16f\n", logPsi[0]);
#endif
  }
  
  void
  OneBodyJastrowOrbitalBspline::update (vector<Walker_t*> &walkers, int iat)
  {
    // for (int iw=0; iw<walkers.size(); iw++) 
    //   UpdateListHost[iw] = (CudaReal*)walkers[iw]->R_GPU.data();
    // UpdateListGPU = UpdateListHost;
    
    // one_body_update(UpdateListGPU.data(), N, iat, walkers.size());
  }
  
  void
  OneBodyJastrowOrbitalBspline::ratio
  (MCWalkerConfiguration &W, int iat,
   vector<ValueType> &psi_ratios,	vector<GradType>  &grad,
   vector<ValueType> &lapl)
  {
    vector<Walker_t*> &walkers = W.WalkerList;
    // Copy new particle positions to GPU
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t &walker = *(walkers[iw]);
      SumHost[iw] = 0.0;
    }
    SumGPU = SumHost;
    
    for (int group=0; group<NumCenterGroups; group++) {
      int first = CenterFirst[group];
      int last  = CenterLast[group];
      
      if (GPUSplines[group]) {
	CudaSpline<CudaReal> &spline = *(GPUSplines[group]);
	one_body_ratio (C.data(), W.RList_GPU.data(), first, last, N, 
			(CudaReal*)W.Rnew_GPU.data(), iat, 
			spline.coefs.data(), spline.coefs.size(),
			spline.rMax, L.data(), Linv.data(),
			SumGPU.data(), walkers.size());
      }
    }
    // Copy data back to CPU memory
    SumHost = SumGPU;

    for (int iw=0; iw<walkers.size(); iw++) 
      psi_ratios[iw] *= std::exp(-SumHost[iw]);

#ifdef CUDA_DEBUG
    DTD_BConds<double,3,SUPERCELL_BULK> bconds;
    int iw = 0;

    Walker_t &walker = *(walkers[iw]);
    double host_sum = 0.0;
    
    for (int cptcl=0; cptcl<CenterRef.getTotalNum(); cptcl++) {
      FT* func = Fs[cptcl];
      PosType disp = new_pos[iw] - CenterRef.R[cptcl];
      double dist = std::sqrt(bconds.apply(ElecRef.Lattice, disp));
      host_sum += func->evaluate(dist);
      disp = walkers[iw]->R[iat] - CenterRef.R[cptcl];
      dist = std::sqrt(bconds.apply(ElecRef.Lattice, disp));
      host_sum -= func->evaluate(dist);
    }
    fprintf (stderr, "Host sum = %18.12e\n", host_sum);
    fprintf (stderr, "CUDA sum = %18.12e\n", SumHost[0]);
    
#endif
    
  }


  void
  OneBodyJastrowOrbitalBspline::NLratios 
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
    for (int group=0; group<NumCenterGroups; group++) {
      int first = CenterFirst[group];
      int last  = CenterLast[group];
      if (GPUSplines[group]) {
	CudaSpline<CudaReal> &spline = *(GPUSplines[group]);
	one_body_NLratios(NL_JobListGPU.data(), C.data(), first, last,
			  spline.coefs.data(), spline.coefs.size(),
			  spline.rMax, L.data(), Linv.data(), njobs);
      }
    }
    NL_RatiosHost = NL_RatiosGPU;
    for (int i=0; i < psi_ratios.size(); i++)
      psi_ratios[i] = NL_RatiosHost[i];
  }


  void
  OneBodyJastrowOrbitalBspline::gradLapl (MCWalkerConfiguration &W, 
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
    
    for (int eptcl=0; eptcl<N; eptcl++) {
      PosType grad(0.0, 0.0, 0.0);
      double lapl(0.0);
      for (int cptcl=0; cptcl<CenterRef.getTotalNum(); cptcl++) {
	FT* func = Fs[cptcl];
	PosType disp = walkers[iw]->R[eptcl] - CenterRef.R[cptcl];
	double dist = std::sqrt(bconds.apply(ElecRef.Lattice, disp));
	double u, du, d2u;
	u = func->evaluate(dist, du, d2u);
	du /= dist;
	grad += disp * du;
	lapl += d2u + 2.0*du;
      }
      CPU_GradLapl[eptcl*4+0] = grad[0];
      CPU_GradLapl[eptcl*4+1] = grad[1];
      CPU_GradLapl[eptcl*4+2] = grad[2];
      CPU_GradLapl[eptcl*4+3] = lapl;
    }
  
#endif
    for (int cgroup=0; cgroup<NumCenterGroups; cgroup++) {
      int cfirst = CenterFirst[cgroup];
      int clast  = CenterLast[cgroup];
      int efirst = 0;
      int elast  = N-1;
      if (GPUSplines[cgroup]) {
	CudaSpline<CudaReal> &spline = *(GPUSplines[cgroup]);
	one_body_grad_lapl (C.data(), W.RList_GPU.data(), 
			    cfirst, clast, efirst, elast, 
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
