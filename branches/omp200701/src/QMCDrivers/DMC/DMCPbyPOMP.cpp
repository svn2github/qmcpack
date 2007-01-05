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
#include "QMCDrivers/DMC/DMCPbyPOMP.h"
#include "QMCDrivers/DMC/DMCUpdatePbyP.h"
#include "QMCDrivers/DMC/DMCNonLocalUpdate.h"
#include "QMCDrivers/DMC/DMCUpdateAll.h"
#include "Estimators/DMCEnergyEstimator.h"
#include "QMCApp/HamiltonianPool.h"
#include "Message/Communicate.h"
#include "Message/OpenMP.h"

namespace qmcplusplus { 

  /// Constructor.
  DMCPbyPOMP::DMCPbyPOMP(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h,
      HamiltonianPool& hpool):
    QMCDriver(w,psi,h), CloneManager(hpool),
    KillNodeCrossing(0),
    Reconfiguration("no"), BenchMarkRun("no"){
    RootName = "dummy";
    QMCType ="DMCPbyPOMP";

    QMCDriverMode.set(QMC_UPDATE_MODE,1);
    QMCDriverMode.set(QMC_MULTIPLE,1);

    m_param.add(KillWalker,"killnode","string");
    m_param.add(BenchMarkRun,"benchmark","string");
    m_param.add(Reconfiguration,"reconfiguration","string");

    Estimators = new ScalarEstimatorManager(H);
    Estimators->add(new DMCEnergyEstimator,"elocal");
  }

  void DMCPbyPOMP::resetUpdateEngines() {


    makeClones(W,Psi,H);

    if(Movers.empty()) {

      Movers.resize(NumThreads,0);
      branchClones.resize(NumThreads,0);
      Rng.resize(NumThreads,0);

      FairDivideLow(W.getActiveWalkers(),NumThreads,wPerNode);
      app_log() << "  Initial partition of walkers ";
      std::copy(wPerNode.begin(),wPerNode.end(),ostream_iterator<int>(app_log()," "));
      app_log() << endl;
#pragma omp parallel  
      {
        int ip = omp_get_thread_num();
        if(ip) {
          hClones[ip]->add2WalkerProperty(*wClones[ip]);
        }
        Rng[ip]=new RandomGenerator_t();
        Rng[ip]->init(ip,NumThreads,-1);

        branchClones[ip] = new BranchEngineType(*branchEngine);
        if(QMCDriverMode[QMC_UPDATE_MODE])
        {
          if(NonLocalMove == "yes")
          {
            DMCNonLocalUpdatePbyP* nlocMover= 
              new DMCNonLocalUpdatePbyP(*wClones[ip],*psiClones[ip],*hClones[ip],*Rng[ip]); 
            nlocMover->put(qmcNode);
            Movers[ip]=nlocMover;
          } 
          else
          {
            Movers[ip]= new DMCUpdatePbyPWithRejection(*wClones[ip],*psiClones[ip],*hClones[ip],*Rng[ip]); 
          }
          Movers[ip]->resetRun(branchClones[ip]);
          Movers[ip]->initWalkersForPbyP(W.begin()+wPerNode[ip],W.begin()+wPerNode[ip+1]);
        } 
        else
        {
          if(NonLocalMove == "yes") {
            app_log() << "  Non-local update is used." << endl;
            DMCNonLocalUpdate* nlocMover= new DMCNonLocalUpdate(W,Psi,H,Random);
            nlocMover->put(qmcNode);
            Movers[ip]=nlocMover;
          } else {
            if(KillNodeCrossing) {
              Movers[ip] = new DMCUpdateAllWithKill(W,Psi,H,Random);
            } else {
              Movers[ip] = new DMCUpdateAllWithRejection(W,Psi,H,Random);
            }
          }
          Movers[ip]->resetRun(branchClones[ip]);
          Movers[ip]->initWalkers(W.begin()+wPerNode[ip],W.begin()+wPerNode[ip+1]);
        }
      }
    }

    bool fixW = (Reconfiguration == "yes");
    if(fixW) 
    {
      app_log() << "  DMC/OMP PbyP Update with reconfigurations" << endl;
      for(int ip=0; ip<Movers.size(); ip++) Movers[ip]->MaxAge=0;
      if(BranchInterval<0)
      {
        BranchInterval=nSteps;
      }
    } 
    else 
    {
      app_log() << "  DMC/OMP PbyP update with a fluctuating population" << endl;
      for(int ip=0; ip<Movers.size(); ip++) Movers[ip]->MaxAge=1;
      if(BranchInterval<0) BranchInterval=1;
    }
    branchEngine->initWalkerController(Tau,fixW);
  }

  bool DMCPbyPOMP::run() {

    resetUpdateEngines();

    //set the collection mode for the estimator
    Estimators->setCollectionMode(branchEngine->SwapMode);
    Estimators->reportHeader(AppendRun);
    Estimators->reset();

    IndexType block = 0;

    do // block
    {
      Estimators->startBlock();
      for(int ip=0; ip<NumThreads; ip++) Movers[ip]->startBlock();
      IndexType step = 0;
      do  //step
      {
#pragma omp parallel  
        {
          int ip = omp_get_thread_num();
          IndexType interval = 0;
          do // interval
          {
            Movers[ip]->advanceWalkers(W.begin()+wPerNode[ip],W.begin()+wPerNode[ip+1]);
            ++interval;
          } while(interval<BranchInterval);
        }//#pragma omp parallel

        Movers[0]->setMultiplicity(W.begin(),W.end());
        Estimators->accumulate(W);
        branchEngine->branch(CurrentStep,W, branchClones);

        ++step; 
        CurrentStep+=BranchInterval;
      } while(step<nSteps);

      Estimators->stopBlock(acceptRatio());

      block++;
      recordBlock(block);

      if(QMCDriverMode[QMC_UPDATE_MODE] && CurrentStep%100 == 0) 
      {
#pragma omp parallel  
        {
          int ip = omp_get_thread_num();
          Movers[ip]->updateWalkers(W.begin()+wPerNode[ip], W.begin()+wPerNode[ip+1]);
        }
      }

    } while(block<nBlocks);

    return finalize(block);
  }

  void DMCPbyPOMP::benchMark() { 
    
    //set the collection mode for the estimator
    Estimators->setCollectionMode(branchEngine->SwapMode);

    IndexType PopIndex = Estimators->addColumn("Population");
    IndexType EtrialIndex = Estimators->addColumn("Etrial");
    Estimators->reportHeader(AppendRun);
    Estimators->reset();

    IndexType block = 0;
    RealType Eest = branchEngine->E_T;

    //resetRun();

    for(int ip=0; ip<NumThreads; ip++) {
      char fname[16];
      sprintf(fname,"test.%i",ip);
      ofstream fout(fname);
    }

    for(int istep=0; istep<nSteps; istep++) {

      FairDivideLow(W.getActiveWalkers(),NumThreads,wPerNode);
#pragma omp parallel  
      {
        int ip = omp_get_thread_num();
        Movers[ip]->benchMark(W.begin()+wPerNode[ip],W.begin()+wPerNode[ip+1],ip);
      }
    }
  }
  
  bool 
  DMCPbyPOMP::put(xmlNodePtr q){ 
    //nothing to do
    return true;
  }

//  void DMCPbyPOMP::dmcWithBranching() {
//
//    RealType Eest = branchEngine->E_T;
//
//    resetRun();
//
//
//    IndexType block = 0;
//    do {//start a block
//      for(int ip=0; ip<NumThreads; ip++) {
//        Movers[ip]->startBlock();
//      } 
//
//      FairDivideLow(W.getActiveWalkers(),NumThreads,wPerNode);
//      Estimators->startBlock();
//      IndexType step = 0;
//      IndexType pop_acc=0; 
//      do {
//#pragma omp parallel  
//        {
//          int ip = omp_get_thread_num();
//          Movers[ip]->resetEtrial(Eest);
//          Movers[ip]->advanceWalkers(W.begin()+wPerNode[ip], W.begin()+wPerNode[ip+1]);
//        }
//
//        ++step; ++CurrentStep;
//        Estimators->accumulate(W);
//
//        //int cur_pop = branchEngine->branch(CurrentStep,W, branchClones);
//        //pop_acc += cur_pop;
//        //Eest = branchEngine->CollectAndUpdate(cur_pop, Eest); 
//
//        FairDivideLow(W.getActiveWalkers(),NumThreads,wPerNode);
//
//        for(int ip=0; ip<NumThreads; ip++) Movers[ip]->resetEtrial(Eest); 
//
//        if(CurrentStep%100 == 0) {
//#pragma omp parallel  
//          {
//            int ip = omp_get_thread_num();
//            Movers[ip]->updateWalkers(W.begin()+wPerNode[ip], W.begin()+wPerNode[ip+1]);
//          }
//        }
//      } while(step<nSteps);
//      
//      Estimators->stopBlock(acceptRatio());
//      
//      //update estimator
//      //Estimators->setColumn(PopIndex,static_cast<RealType>(pop_acc)/static_cast<RealType>(nSteps));
//      //Estimators->setColumn(EtrialIndex,Eest); 
//      //Eest = Estimators->average(0);
//      //RealType totmoves=1.0/static_cast<RealType>(step*W.getActiveWalkers());
//
//      //Need MPI-IO
//      //app_log() 
//      //  << setw(4) << block 
//      //  << setw(20) << static_cast<RealType>(nAllRejected)*totmoves
//      //  << setw(20) << static_cast<RealType>(nNodeCrossing)*totmoves << endl;
//      block++;
//
//      recordBlock(block);
//
//    } while(block<nBlocks);
//  }
//


}

/***************************************************************************
 * $RCSfile: DMCPbyPOMP.cpp,v $   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
