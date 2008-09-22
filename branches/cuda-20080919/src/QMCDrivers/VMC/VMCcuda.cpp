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

namespace qmcplusplus { 

  /// Constructor.
  VMCcuda::VMCcuda(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h):
    QMCDriver(w,psi,h), myWarmupSteps(0), Mover(0), UseDrift("yes")
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

    Mover->startRun(nBlocks,true);

    IndexType block = 0;
    IndexType nAcceptTot = 0;
    IndexType nRejectTot = 0;
    IndexType updatePeriod=(QMCDriverMode[QMC_UPDATE_MODE])?Period4CheckProperties:(nBlocks+1)*nSteps;

    int nat=W.getTotalNum();
    int nw=W.getActiveWalkers();

    vector<PosType> delpos(nw);
    vector<PosType> newpos(nw);
    vector<ValueType> ratios(nw);
    vector<GradType> oldG(nw);
    vector<GradType> newG(nw);
    vector<Walker_t*> moved(nw);
    do {
      Mover->startBlock(nSteps);
      IndexType step = 0;
      do
      {
        ++step;++CurrentStep;
        for(int iat=0; iat<nat; ++iat)
        {
          //calculate drift
          Psi.getGradient(W->WalkerList,iat,oldG);

          //create a 3N-Dimensional Gaussian with variance=1
          makeGaussRandomWithEngine(delpos,RandomGen);
          for(int iw=0; iw<nw; ++iw)
            newPos[iw]=W[iw]->R[iat]+m_sqrttau*deltaR[iw];

          Psi.ratio(W->WalkerList,iat,newpos,ratios,newG);

          moved.clear();
          for(int iw=0; iw<nw: ++iw)
            if(ratios[iw]*ratios[iw]<Random()) 
            {
              moved.push_back(W[iw]);
            }

          Psi.update(moved,iat);
        }


        //Mover->advanceWalkers(W.begin(),W.end(),true); //step==nSteps);
        //Estimators->accumulate(W);
        //if(CurrentStep%updatePeriod==0) Mover->updateWalkers(W.begin(),W.end());
        //if(CurrentStep%myPeriod4WalkerDump==0) W.saveEnsemble();
      } while(step<nSteps);

      nAcceptTot += nAccept;
      nRejectTot += nReject;
      ++block;

      recordBlock(block);

      ////periodically re-evaluate everything for pbyp
      //if(QMCDriverMode[QMC_UPDATE_MODE] && CurrentStep%100 == 0) 
      //  Mover->updateWalkers(W.begin(),W.end());

    } while(block<nBlocks);

    Mover->stopRun();

    //finalize a qmc section
    return finalize(block);
  }

  void VMCcuda::resetRun()
  {

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
