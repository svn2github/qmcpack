#include "OneBodyJastrowOrbitalBspline.h"
#include "CudaSpline.h"
#include "Lattice/ParticleBConds.h"

namespace qmcplusplus {

  void
  OneBodyJastrowOrbitalBspline::recompute(vector<Walker_t*> &walkers)
  {
  }
  
  void
  OneBodyJastrowOrbitalBspline::reserve 
  (PointerPool<cuda_vector<CudaRealType> > &pool)
  {
    int elemSize = 
      std::max((unsigned long)1,sizeof(CudaReal)/sizeof(CUDA_PRECISION));
    ROffset = pool.reserve(3*(N+1) * elemSize);
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
  OneBodyJastrowOrbitalBspline::addLog (vector<Walker_t*> &walkers, 
					vector<RealType> &logPsi)
  {
    app_log() << "OneBodyJastrowOrbitalBspline::addLog.\n";
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
    int efirst = 0;
    int elast = N-1;

    for (int cgroup=0; cgroup<NumCenterGroups; cgroup++) {
      int cfirst = CenterFirst[cgroup];
      int clast  = CenterLast[cgroup];
      
      CudaSpline<CudaReal> &spline = *(GPUSplines[cgroup]);
      if (GPUSplines[cgroup]) {
	one_body_sum (C.data(), RlistGPU.data(), cfirst, clast, efirst, elast, 
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
    for (int iw=0; iw<walkers.size(); iw++) 
      UpdateListHost[iw] = (CudaReal*)&(walkers[iw]->cuda_DataSet[ROffset]);
    UpdateListGPU = UpdateListHost;
    
    one_body_update(UpdateListGPU.data(), N, iat, walkers.size());
  }
  
  void
  OneBodyJastrowOrbitalBspline::ratio
  (vector<Walker_t*> &walkers, int iat, vector<PosType> &new_pos,
   vector<ValueType> &psi_ratios,	vector<GradType>  &grad,
   vector<ValueType> &lapl)
  {
    // Copy new particle positions to GPU
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t &walker = *(walkers[iw]);
      SumHost[iw] = 0.0;
      for (int dim=0; dim<OHMMS_DIM; dim++)
	RnewHost[OHMMS_DIM*iw+dim] = new_pos[iw][dim];
    }
    SumGPU = SumHost;
    RnewGPU = RnewHost;
    
    for (int group=0; group<NumCenterGroups; group++) {
      int first = CenterFirst[group];
      int last  = CenterLast[group];
      
      if (GPUSplines[group]) {
	CudaSpline<CudaReal> &spline = *(GPUSplines[group]);
	one_body_ratio (C.data(), RlistGPU.data(), first, last, N, 
			RnewGPU.data(), iat, 
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
  OneBodyJastrowOrbitalBspline::gradLapl (vector<Walker_t*> &walkers, 
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
	one_body_grad_lapl (C.data(), RlistGPU.data(), 
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
