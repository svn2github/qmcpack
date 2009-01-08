//////////////////////////////////////////////////////////////////
// (c) Copyright 2003- by Ken Esler
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Ken Esler
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: esler@uiuc.edu
//
// Supported by 
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#include "QMCDrivers/DMC/DMCcuda.h"
#include "OhmmsApp/RandomNumberControl.h"
#include "Utilities/RandomGenerator.h"
#include "ParticleBase/RandomSeqGenerator.h"

namespace qmcplusplus { 

  /// Constructor.
  DMCcuda::DMCcuda(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h):
    QMCDriver(w,psi,h), myWarmupSteps(0), UseDrift("yes")
  { 
    RootName = "dmc";
    QMCType ="DMCcuda";
    QMCDriverMode.set(QMC_UPDATE_MODE,1);
    QMCDriverMode.set(QMC_WARMUP,0);
    m_param.add(myWarmupSteps,"warmupSteps","int");
    m_param.add(nTargetSamples,"targetWalkers","int");
  }
  

  bool DMCcuda::run() 
  { 
    resetRun();
    IndexType block = 0;
    IndexType nAcceptTot = 0;
    IndexType nRejectTot = 0;
    int nat = W.getTotalNum();
    int nw  = W.getActiveWalkers();
    
    vector<RealType>  LocalEnergy(nw), oldScale(nw), newScale(nw);
    vector<PosType>   delpos(nw);
    vector<PosType>   dr(nw);
    vector<PosType>   newpos(nw);
    vector<ValueType> ratios(nw), rplus(nw), rminus(nw);
    vector<PosType>  oldG(nw), newG(nw);
    vector<ValueType> oldL(nw), newL(nw);
    vector<Walker_t*> accepted(nw);
    Matrix<ValueType> lapl(nw, nat);
    Matrix<GradType>  grad(nw, nat);

    do {
      IndexType step = 0;
      nAccept = nReject = 0;
      Estimators->startBlock(nSteps);
      do {
        step++;
	CurrentStep++;
        for(int iat=0; iat<nat; iat++) {
	  Psi.getGradient (W, iat, oldG);

          //create a 3N-Dimensional Gaussian with variance=1
          makeGaussRandomWithEngine(delpos,Random);
          for(int iw=0; iw<nw; iw++) {
	    oldScale[iw] = getDriftScale(m_tauovermass,oldG[iw]);
	    dr[iw] = (m_sqrttau*delpos[iw]) + (oldScale[iw]*oldG[iw]);
            newpos[iw]=W[iw]->R[iat] + dr[iw];
	    ratios[iw] = 1.0;
	  }
	  W.proposeMove_GPU(newpos, iat);
	  
          Psi.ratio(W,iat,ratios,newG, newL);
	  	  
          accepted.clear();
	  vector<bool> acc(nw, false);
          for(int iw=0; iw<nw; ++iw) {
	    PosType drOld = 
	      newpos[iw] - (W[iw]->R[iat] + oldScale[iw]*oldG[iw]);
	    RealType logGf = -m_oneover2tau * dot(drOld, drOld);
	    newScale[iw]   = getDriftScale(m_tauovermass,newG[iw]);
	    PosType drNew  = 
	      (newpos[iw] + newScale[iw]*newG[iw]) - W[iw]->R[iat];
	    RealType logGb =  -m_oneover2tau * dot(drNew, drNew);
	    RealType x = logGb - logGf;
	    RealType prob = ratios[iw]*ratios[iw]*std::exp(x);
	    
            if(Random() < prob) {
              accepted.push_back(W[iw]);
	      nAccept++;
	      W[iw]->R[iat] = newpos[iw];
	      acc[iw] = true;
	    }
	    else 
	      nReject++;
	  }
	  W.acceptMove_GPU(acc);
	  if (accepted.size())
	    Psi.update(accepted,iat);
	}

	Psi.gradLapl(W, grad, lapl);
	H.evaluate (W, LocalEnergy);
	Estimators->accumulate(W);
      } while(step<nSteps);
      Psi.recompute(W);
      
      double accept_ratio = (double)nAccept/(double)(nAccept+nReject);
      Estimators->stopBlock(accept_ratio);

      nAcceptTot += nAccept;
      nRejectTot += nReject;
      ++block;
      
      recordBlock(block);
    } while(block<nBlocks);
    //finalize a qmc section
    return finalize(block);
  }


  void DMCcuda::resetRun()
  {
    SpeciesSet tspecies(W.getSpeciesSet());
    int massind=tspecies.addAttribute("mass");
    RealType mass = tspecies(massind,0);
    RealType oneovermass = 1.0/mass;
    RealType oneoversqrtmass = std::sqrt(oneovermass);
    m_oneover2tau = 0.5/Tau;
    m_sqrttau = std::sqrt(Tau/mass);

    // Compute the size of data needed for each walker on the GPU card
    PointerPool<Walker_t::cuda_Buffer_t > pool;
    Psi.reserve (pool);
    app_log() << "Each walker requires " << pool.getTotalSize() * sizeof(CudaRealType)
	      << " bytes in GPU memory.\n";

    // Now allocate memory on the GPU card for each walker
    for (int iw=0; iw<W.WalkerList.size(); iw++) {
      Walker_t &walker = *(W.WalkerList[iw]);
      pool.allocate(walker.cuda_DataSet);
    }
    vector<RealType> logPsi(W.WalkerList.size(), 0.0);
    Psi.evaluateLog(W.WalkerList, logPsi);
    Estimators->start(nBlocks, true);
  }

  bool 
  DMCcuda::put(xmlNodePtr q){
    //nothing to add
    return true;
  }
}

/***************************************************************************
 * $RCSfile: DMCParticleByParticle.cpp,v $   $Author: jnkim $
 * $Revision: 1.25 $   $Date: 2006/10/18 17:03:05 $
 * $Id: DMCcuda.cpp,v 1.25 2006/10/18 17:03:05 jnkim Exp $ 
 ***************************************************************************/
