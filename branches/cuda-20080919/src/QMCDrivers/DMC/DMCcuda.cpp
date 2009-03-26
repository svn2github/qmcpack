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
#include "QMCDrivers/DMC/DMCUpdatePbyP.h"
#include "QMCDrivers/QMCUpdateBase.h"
#include "OhmmsApp/RandomNumberControl.h"
#include "Utilities/RandomGenerator.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "QMCDrivers/DriftOperators.h"

namespace qmcplusplus { 

  /// Constructor.
  DMCcuda::DMCcuda(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h):
    QMCDriver(w,psi,h), myWarmupSteps(0), Mover(0)
  { 
    RootName = "dmc";
    QMCType ="DMCcuda";
    QMCDriverMode.set(QMC_UPDATE_MODE,1);
    QMCDriverMode.set(QMC_WARMUP,0);
    m_param.add(myWarmupSteps,"warmupSteps","int");
    m_param.add(nTargetSamples,"targetWalkers","int");
    m_param.add(NonLocalMove,"nonlocalmove","string");
    m_param.add(NonLocalMove,"nonlocalmoves","string");
  }
  

  bool DMCcuda::run() 
  { 
    bool NLmove = NonLocalMove == "yes";
    if (NLmove) 
      app_log() << "  Using Casula nonlocal moves in DMCcuda.\n";
      
    resetRun();
    Mover->MaxAge = 2;
    IndexType block = 0;
    IndexType nAcceptTot = 0;
    IndexType nRejectTot = 0;
    int nat = W.getTotalNum();
    int nw  = W.getActiveWalkers();
    
    vector<RealType>  LocalEnergy(nw), LocalEnergyOld(nw), 
      oldScale(nw), newScale(nw);
    vector<PosType>   delpos(nw);
    vector<PosType>   dr(nw);
    vector<PosType>   newpos(nw);
    vector<ValueType> ratios(nw), rplus(nw), rminus(nw), R2prop(nw), R2acc(nw);
    vector<PosType>  oldG(nw), newG(nw);
    vector<ValueType> oldL(nw), newL(nw);
    vector<Walker_t*> accepted(nw);
    Matrix<ValueType> lapl(nw, nat);
    Matrix<GradType>  grad(nw, nat);
    vector<ValueType> V2(nw), V2bar(nw);
    vector<vector<NonLocalData> > Txy(nw);

    for (int iw=0; iw<nw; iw++)
      W[iw]->Weight = 1.0;
    do {
      IndexType step = 0;
      nAccept = nReject = 0;
      Estimators->startBlock(nSteps);
      
      do {
        step++;
	CurrentStep++;
	nw = W.getActiveWalkers();
	LocalEnergy.resize(nw);       oldScale.resize(nw);
	newScale.resize(nw);          delpos.resize(nw);
	dr.resize(nw);                newpos.resize(nw);
	ratios.resize(nw);            rplus.resize(nw);
	rminus.resize(nw);            oldG.resize(nw);
	newG.resize(nw);              oldL.resize(nw);
	newL.resize(nw);              accepted.resize(nw);
	lapl.resize(nw, nat);         grad.resize(nw, nat);
	R2prop.resize(nw,0.0);        R2acc.resize(nw,0.0);
	V2.resize(nw,0.0);            V2bar.resize(nw,0.0);

	W.updateLists_GPU();

	if (NLmove) {
	  Txy.resize(nw);
	  for (int iw=0; iw<nw; iw++) {
	    Txy[iw].clear();
	    Txy[iw].push_back(NonLocalData(-1, 1.0, PosType()));
	  }
	}


	for (int iw=0; iw<nw; iw++)
	  W[iw]->Age++;
	
        for(int iat=0; iat<nat; iat++) {
	  Psi.getGradient (W, iat, oldG);
	  
          //create a 3N-Dimensional Gaussian with variance=1
          makeGaussRandomWithEngine(delpos,Random);
          for(int iw=0; iw<nw; iw++) {
	    delpos[iw] *= m_sqrttau;
	    oldScale[iw] = getDriftScale(m_tauovermass,oldG[iw]);
	    dr[iw] = delpos[iw] + (oldScale[iw]*oldG[iw]);
            newpos[iw]=W[iw]->R[iat] + dr[iw];
	    ratios[iw] = 1.0;
	    R2prop[iw] += dot(delpos[iw], delpos[iw]);
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

	    V2[iw]    += m_tauovermass * m_tauovermass * dot(oldG[iw],oldG[iw]);
	    V2bar[iw] +=  newScale[iw] *  newScale[iw] * dot(oldG[iw],oldG[iw]);

	    
            if(Random() < prob && ratios[iw] > 0.0) {
              accepted.push_back(W[iw]);
	      nAccept++;
	      W[iw]->R[iat] = newpos[iw];
	      W[iw]->Age = 0;
	      acc[iw] = true;
	      R2acc[iw] += dot(delpos[iw], delpos[iw]);
	    }
	    else 
	      nReject++;
	  }
	  W.acceptMove_GPU(acc);
	  if (accepted.size())
	    Psi.update(accepted,iat);
	}
	//	Psi.recompute(W, false);
	Psi.gradLapl(W, grad, lapl);
	if (NLmove)	  H.evaluate (W, LocalEnergy, Txy);
	else    	  H.evaluate (W, LocalEnergy);
	if (CurrentStep == 1)
	  LocalEnergyOld = LocalEnergy;
	
	if (NLmove) {
	  // Now, attempt nonlocal move
	  accepted.clear();
	  vector<int> iatList;
	  vector<PosType> accPos;
	  for (int iw=0; iw<nw; iw++) {
	    int ibar = NLop.selectMove(Random(), Txy[iw]);
	    if (ibar) {
	      accepted.push_back(W[iw]);
	      int iat = Txy[iw][ibar].PID;
	      iatList.push_back(iat);
	      accPos.push_back(W[iw]->R[iat] + Txy[iw][ibar].Delta);
	    }
	  }
	  if (accepted.size()) {
	    Psi.ratio(accepted,iatList, accPos, ratios, newG, newL);
	    Psi.update(accepted,iatList);
	    for (int i=0; i<accepted.size(); i++)
	      accepted[i]->R[iatList[i]] = accPos[i];
	    W.copyWalkersToGPU();
	  }
	}

	// Now branch
	for (int iw=0; iw<nw; iw++) {
	  RealType scNew = std::sqrt(V2bar[iw] / V2[iw]);
	  RealType scOld = (CurrentStep == 1) ? scNew : W[iw]->getPropertyBase()[DRIFTSCALE];
	  W[iw]->getPropertyBase()[DRIFTSCALE] = scNew;
	  // fprintf (stderr, "iw = %d  scNew = %1.8f  scOld = %1.8f\n", iw, scNew, scOld);
	  W[iw]->Weight *= branchEngine->branchWeight(LocalEnergy[iw], LocalEnergyOld[iw],
						      scNew, scOld);
	  W[iw]->getPropertyBase()[R2ACCEPTED] = R2acc[iw];
	  W[iw]->getPropertyBase()[R2PROPOSED] = R2prop[iw];
	}
	Mover->setMultiplicity(W.begin(), W.end());
	branchEngine->branch(CurrentStep,W);
	nw = W.getActiveWalkers();
	LocalEnergyOld.resize(nw);
	for (int iw=0; iw<nw; iw++)
	  LocalEnergyOld[iw] = W[iw]->getPropertyBase()[LOCALENERGY];
      } while(step<nSteps);
      Psi.recompute(W, true);


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



  bool DMCcuda::runWithNonlocal() 
  { 
    resetRun();
    Mover->MaxAge = 2;
    IndexType block = 0;
    IndexType nAcceptTot = 0;
    IndexType nRejectTot = 0;
    int nat = W.getTotalNum();
    int nw  = W.getActiveWalkers();
    
    vector<RealType>  LocalEnergy(nw), LocalEnergyOld(nw), 
      oldScale(nw), newScale(nw);
    vector<PosType>   delpos(nw);
    vector<PosType>   dr(nw);
    vector<PosType>   newpos(nw);
    vector<ValueType> ratios(nw), rplus(nw), rminus(nw), R2prop(nw), R2acc(nw);
    vector<PosType>  oldG(nw), newG(nw);
    vector<ValueType> oldL(nw), newL(nw);
    vector<Walker_t*> accepted(nw);
    Matrix<ValueType> lapl(nw, nat);
    Matrix<GradType>  grad(nw, nat);
    vector<vector<NonLocalData> > Txy(nw);
    for (int iw=0; iw<nw; iw++)
      W[iw]->Weight = 1.0;
    do {
      IndexType step = 0;
      nAccept = nReject = 0;
      Estimators->startBlock(nSteps);
      
      do {
        step++;
	CurrentStep++;
	nw = W.getActiveWalkers();
	LocalEnergy.resize(nw);    	oldScale.resize(nw);
	newScale.resize(nw);    	delpos.resize(nw);
	dr.resize(nw);                  newpos.resize(nw);
	ratios.resize(nw);	        rplus.resize(nw);
	rminus.resize(nw);              oldG.resize(nw);
	newG.resize(nw);                oldL.resize(nw);
	newL.resize(nw);                accepted.resize(nw);
	lapl.resize(nw, nat);           grad.resize(nw, nat);
	R2prop.resize(nw,0.0);          R2acc.resize(nw,0.0);

	W.updateLists_GPU();

	Txy.resize(nw);
	for (int iw=0; iw<nw; iw++) {
	  Txy[iw].clear();
	  Txy[iw].push_back(NonLocalData(-1, 1.0, PosType()));
	  W[iw]->Age++;
	}
	
        for(int iat=0; iat<nat; iat++) {
	  Psi.getGradient (W, iat, oldG);
	  
          //create a 3N-Dimensional Gaussian with variance=1
          makeGaussRandomWithEngine(delpos,Random);
          for(int iw=0; iw<nw; iw++) {
	    delpos[iw] *= m_sqrttau;
	    oldScale[iw] = getDriftScale(m_tauovermass,oldG[iw]);
	    dr[iw] = delpos[iw] + (oldScale[iw]*oldG[iw]);
            newpos[iw]=W[iw]->R[iat] + dr[iw];
	    ratios[iw] = 1.0;
	    R2prop[iw] += dot(delpos[iw], delpos[iw]);
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
	    
            if(Random() < prob && ratios[iw] > 0.0) {
              accepted.push_back(W[iw]);
	      nAccept++;
	      W[iw]->R[iat] = newpos[iw];
	      W[iw]->Age = 0;
	      acc[iw] = true;
	      R2acc[iw] += dot(delpos[iw], delpos[iw]);
	    }
	    else 
	      nReject++;
	  }
	  W.acceptMove_GPU(acc);
	  if (accepted.size())
	    Psi.update(accepted,iat);
	}
	//	Psi.recompute(W, false);
	Psi.gradLapl(W, grad, lapl);
	H.evaluate (W, LocalEnergy, Txy);
	if (CurrentStep == 1)
	  LocalEnergyOld = LocalEnergy;
	
	// Now, attempt nonlocal move
	accepted.clear();
	vector<int> iatList;
	vector<PosType> accPos;
	for (int iw=0; iw<nw; iw++) {
	  int ibar = NLop.selectMove(Random(), Txy[iw]);
	  // cerr << "Txy[iw].size() = " << Txy[iw].size() << endl;
	  if (ibar) {
	    accepted.push_back(W[iw]);
	    int iat = Txy[iw][ibar].PID;
	    iatList.push_back(iat);
	    accPos.push_back(W[iw]->R[iat] + Txy[iw][ibar].Delta);
	  }
	}
	if (accepted.size()) {
	  //   W.proposeMove_GPU(newpos, iatList);
	  Psi.ratio(accepted,iatList, accPos, ratios, newG, newL);
	  Psi.update(accepted,iatList);
	  for (int i=0; i<accepted.size(); i++)
	    accepted[i]->R[iatList[i]] = accPos[i];
	  W.copyWalkersToGPU();
	}

	// Now branch
	for (int iw=0; iw<nw; iw++) {
	  W[iw]->Weight *= branchEngine->branchWeight(LocalEnergy[iw], LocalEnergyOld[iw]);
	  W[iw]->getPropertyBase()[R2ACCEPTED] = R2acc[iw];
	  W[iw]->getPropertyBase()[R2PROPOSED] = R2prop[iw];
	}
	Mover->setMultiplicity(W.begin(), W.end());
	branchEngine->branch(CurrentStep,W);
	nw = W.getActiveWalkers();
	LocalEnergyOld.resize(nw);
	for (int iw=0; iw<nw; iw++)
	  LocalEnergyOld[iw] = W[iw]->getPropertyBase()[LOCALENERGY];
      } while(step<nSteps);
      Psi.recompute(W, true);


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



  void DMCcuda::resetUpdateEngine()
  {
    if(Mover==0) //disable switching update modes for DMC in a run  
    {
      //load walkers if they were saved
      W.loadEnsemble();
      
      branchEngine->initWalkerController(W,Tau,false);
      
      Mover = new DMCUpdatePbyPWithRejection(W,Psi,H,Random); 
      Mover->resetRun(branchEngine,Estimators);
      //Mover->initWalkersForPbyP(W.begin(),W.end());
      branchEngine->checkParameters(W);
    }
    //    Mover->updateWalkers(W.begin(),W.end());
    
    app_log() << "  DMC PbyP Update with a fluctuating population" << endl;
    Mover->MaxAge=1;
    app_log() << "  Steps per block = " << nSteps << endl;
    app_log() << "  Number of blocks = " << nBlocks << endl;
  }
    


  void DMCcuda::resetRun()
  {
    resetUpdateEngine();
    SpeciesSet tspecies(W.getSpeciesSet());
    int massind=tspecies.addAttribute("mass");
    RealType mass = tspecies(massind,0);
    RealType oneovermass = 1.0/mass;
    RealType oneoversqrtmass = std::sqrt(oneovermass);
    m_oneover2tau = 0.5/Tau;
    m_sqrttau = std::sqrt(Tau/mass);
    m_tauovermass = Tau/mass;

    // Compute the size of data needed for each walker on the GPU card
    PointerPool<Walker_t::cuda_Buffer_t > pool;
    Psi.reserve (pool);
    app_log() << "Each walker requires " 
	      << pool.getTotalSize() * sizeof(CudaRealType)
	      << " bytes in GPU memory.\n";

    // Now allocate memory on the GPU card for each walker
    int cudaSize = pool.getTotalSize();
    for (int iw=0; iw<W.WalkerList.size(); iw++) {
      Walker_t &walker = *(W.WalkerList[iw]);
      walker.resizeCuda(cudaSize);
      //pool.allocate(walker.cuda_DataSet);
    }
    W.copyWalkersToGPU();
    W.updateLists_GPU();
    vector<RealType> logPsi(W.WalkerList.size(), 0.0);
    Psi.evaluateLog(W, logPsi);
    Psi.recompute(W);
    Estimators->start(nBlocks, true);
  }

  bool 
  DMCcuda::put(xmlNodePtr q){
    //nothing to add
    NLop.put(q);
    return true;
  }
}

/***************************************************************************
 * $RCSfile: DMCParticleByParticle.cpp,v $   $Author: jnkim $
 * $Revision: 1.25 $   $Date: 2006/10/18 17:03:05 $
 * $Id: DMCcuda.cpp,v 1.25 2006/10/18 17:03:05 jnkim Exp $ 
 ***************************************************************************/
