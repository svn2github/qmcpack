//////////////////////////////////////////////////////////////////
// (c) Copyright 2003- by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Jeongnim Kim
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: jnkim@ncsa.uiuc.edu
//
// Supported by 
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#include "QMCDrivers/VMC/VMCcuda.h"
#include "OhmmsApp/RandomNumberControl.h"
#include "Utilities/RandomGenerator.h"
#include "ParticleBase/RandomSeqGenerator.h"

namespace qmcplusplus { 

  /// Constructor.
  VMCcuda::VMCcuda(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h):
    QMCDriver(w,psi,h), myWarmupSteps(0), UseDrift("yes")
  { 
    RootName = "vmc";
    QMCType ="VMCcuda";
    QMCDriverMode.set(QMC_UPDATE_MODE,1);
    QMCDriverMode.set(QMC_WARMUP,0);
    m_param.add(UseDrift,"useDrift","string"); m_param.add(UseDrift,"usedrift","string");
    m_param.add(myWarmupSteps,"warmupSteps","int");
    m_param.add(nTargetSamples,"targetWalkers","int");
  }
  
  bool VMCcuda::run() { 

    resetRun();

    IndexType block = 0;
    IndexType nAcceptTot = 0;
    IndexType nRejectTot = 0;
    IndexType updatePeriod=(QMCDriverMode[QMC_UPDATE_MODE])?Period4CheckProperties:(nBlocks+1)*nSteps;

    int nat=W.getTotalNum();
    int nw=W.getActiveWalkers();

    vector<PosType> delpos(nw);
    vector<PosType> newpos(nw);
    vector<ValueType> ratios(nw);
    vector<GradType> oldG(nw), newG(nw);
    vector<ValueType> oldL(nw), newL(nw);
    vector<Walker_t*> accepted(nw);
    Matrix<ValueType> lapl(nw, nat);
    Matrix<GradType>  grad(nw, nat);
    double Esum;

    do {
      //Mover->startBlock(nSteps);
      IndexType step = 0;
      nAccept = nReject = 0;
      Esum = 0.0;
      clock_t block_start = clock();
      do
      {
	// cerr << "step = " << step << endl;
        ++step;++CurrentStep;
        for(int iat=0; iat<nat; ++iat)
        {
          //calculate drift
          //Psi.getGradient(W.WalkerList,iat,oldG);

          //create a 3N-Dimensional Gaussian with variance=1
          makeGaussRandomWithEngine(delpos,Random);
          for(int iw=0; iw<nw; ++iw) {
            newpos[iw]=W[iw]->R[iat] + m_sqrttau*delpos[iw];
	    ratios[iw] = 1.0;
	  }

          // Psi.ratio(W.WalkerList,iat,newpos,ratios,newG);
          //Psi.ratio(W.WalkerList,iat,newpos,ratios);
#ifdef CUDA_DEBUG
	  vector<RealType> logPsi1(W.WalkerList.size(), 0.0);
	  Psi.evaluateLog(W.WalkerList, logPsi1);
#endif
          Psi.ratio(W.WalkerList,iat,newpos,ratios,newG, newL);
	  
          accepted.clear();
	  vector<bool> acc(nw, false);
          for(int iw=0; iw<nw; ++iw) {
            if(ratios[iw]*ratios[iw] > Random()) {
              accepted.push_back(W[iw]);
	      nAccept++;
	      W[iw]->R[iat] = newpos[iw];
	      acc[iw] = true;
	    }
	    else 
	      nReject++;
	  }
	  if (accepted.size())
	    Psi.update(accepted,iat);

#ifdef CUDA_DEBUG
	  vector<RealType> logPsi2(W.WalkerList.size(), 0.0);
	  Psi.evaluateLog(W.WalkerList, logPsi2);
	  for (int iw=0; iw<nw; iw++) {
	    if (acc[iw])
	      cerr << "ratio[" << iw << "] = " << ratios[iw]
		   << "  exp(Log2-Log1) = " << std::exp(logPsi2[iw]-logPsi1[iw]) << endl;
	  }
#endif
	  
	}
	double Energy = 0.0;
	// Psi.recompute(W.WalkerList);
	Psi.gradLapl(W.WalkerList, grad, lapl);

	// GradType psiPlus, psiMinus; 	
	// for (int i=0; i<3; i++) {
	//   W[0]->R[0][0] += 1.0e-3;

	for (int iw=0; iw<nw; iw++)
	  for (int iat=0; iat<nat; iat++)
	    Energy -= 0.5*(dot (grad(iw,iat),grad(iw,iat))  + lapl(iw,iat));
	Energy /= (double)nw;
	// app_log() << "Step KE before = " << Energy << endl;
	vector<RealType> logPsi(W.WalkerList.size(), 0.0);

	Esum += Energy;

        //Mover->advanceWalkers(W.begin(),W.end(),true); //step==nSteps);
        //Estimators->accumulate(W);
        //if(CurrentStep%updatePeriod==0) Mover->updateWalkers(W.begin(),W.end());
        //if(CurrentStep%myPeriod4WalkerDump==0) W.saveEnsemble();
      } while(step<nSteps);
      Psi.recompute(W.WalkerList);

      // vector<RealType> logPsi(W.WalkerList.size(), 0.0);
      // Psi.evaluateLog(W.WalkerList, logPsi);
      
      double accept_ratio = (double)nAccept/(double)(nAccept+nReject);
      nAcceptTot += nAccept;
      nRejectTot += nReject;
      ++block;

      recordBlock(block);

      clock_t block_end = clock();
      double block_time = (double)(block_end-block_start)/CLOCKS_PER_SEC;
      fprintf (stderr, "Block energy = %10.5f    Block accept ratio = %5.3f  Block time = %8.3f\n",
	       Esum/(double)nSteps, accept_ratio, block_time);


      ////periodically re-evaluate everything for pbyp
      //if(QMCDriverMode[QMC_UPDATE_MODE] && CurrentStep%100 == 0) 
      //  Mover->updateWalkers(W.begin(),W.end());

    } while(block<nBlocks);

    //Mover->stopRun();

    //finalize a qmc section
    return finalize(block);
  }

  void VMCcuda::resetRun()
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
    
    //int samples_tot=W.getActiveWalkers()*nBlocks*nSteps*myComm->size();
    //myPeriod4WalkerDump=(nTargetSamples>0)?samples_tot/nTargetSamples:Period4WalkerDump;
    //if(myPeriod4WalkerDump==0 || QMCDriverMode[QMC_WARMUP]) 
    //  myPeriod4WalkerDump=(nBlocks+1)*nSteps;
    //W.clearEnsemble();
    //samples_tot=W.getActiveWalkers()*((nBlocks*nSteps)/myPeriod4WalkerDump);
    //W.setNumSamples(samples_tot);

    ////do a warmup run
    //for(int prestep=0; prestep<myWarmupSteps; ++prestep)
    //  Mover->advanceWalkers(W.begin(),W.end(),true); 
    //myWarmupSteps=0;
  }

  bool 
  VMCcuda::put(xmlNodePtr q){
    //nothing to add
    return true;
  }
}

/***************************************************************************
 * $RCSfile: VMCParticleByParticle.cpp,v $   $Author: jnkim $
 * $Revision: 1.25 $   $Date: 2006/10/18 17:03:05 $
 * $Id: VMCParticleByParticle.cpp,v 1.25 2006/10/18 17:03:05 jnkim Exp $ 
 ***************************************************************************/
