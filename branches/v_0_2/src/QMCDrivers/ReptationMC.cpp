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
//   Tel:    217-244-6319 (NCSA) 217-333-3324 (MCC)
//
// Supported by 
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//   Department of Physics, Ohio State University
//   Ohio Supercomputer Center
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#include "QMCDrivers/ReptationMC.h"
#include "QMCDrivers/PolymerChain.h"
#include "Utilities/OhmmsInfo.h"
#include "Particle/MCWalkerConfiguration.h"
#include "Particle/DistanceTable.h"
#include "Particle/HDFWalkerIO.h"
#include "ParticleBase/ParticleUtility.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "Message/Communicate.h"
#include "Utilities/Clock.h"
namespace ohmmsqmc {

  ReptationMC::ReptationMC(MCWalkerConfiguration& w, 
			   TrialWaveFunction& psi, 
			   QMCHamiltonian& h):
    QMCDriver(w,psi,h), 
    UseBounce(false),
    ClonePolymer(true),
    PolymerLength(21),
    NumCuts(1),
    NumTurns(0)
    { 
      RootName = "rmc";
      QMCType ="rmc";
      m_param.add(PolymerLength,"chains","int");
      m_param.add(NumCuts,"cuts","int");
      m_param.add(UseBounce,"bounce","int");
      m_param.add(ClonePolymer,"clone","int");
    }

  ReptationMC::~ReptationMC() {
    for(int i=0; i<Polymers.size(); i++) delete Polymers[i];
  }
  
  ///initialize polymers
  void ReptationMC::initPolymers() {
    
    //overwrite the number of cuts for Bounce algorithm
    if(UseBounce) NumCuts = 1;

    RealType g = sqrt(Tau);
    LOGMSG("Moving " << NumCuts << " for each reptation step")
    MCWalkerConfiguration::iterator it(W.begin());
    MCWalkerConfiguration::iterator it_end(W.end());

    while(it != it_end) {
      (*it)->Properties(WEIGHT)=1.0;     
      PolymerChain* achain = new PolymerChain((*it),PolymerLength,NumCuts);
      Polymers.push_back(achain);

      Walker_t* cur=(*achain)[0];
      W.R = cur->R;
      DistanceTable::update(W);
      ValueType logpsi(Psi.evaluateLog(W));
      cur->Properties(LOCALENERGY) = H.evaluate(W);
      H.copy(cur->getEnergyBase());
      cur->Properties(LOCALPOTENTIAL) = H.getLocalPotential();
      cur->Drift = W.G;

      if(!ClonePolymer) {
	
	for(int i=0; i<NumCuts-1; i++ ) {
	  //create a 3N-Dimensional Gaussian with variance=1
	  makeGaussRandom(deltaR);
	  W.R = cur->R + g*deltaR + Tau*cur->Drift;
	  
	  //update the distance table associated with W
	  DistanceTable::update(W);
	  
	  //evaluate wave function
	  ValueType logpsic(Psi.evaluateLog(W));
	  cur = (*achain)[i+1];	  
	  cur->Properties(LOCALENERGY) = H.evaluate(W);
	  H.copy(cur->getEnergyBase());
	  cur->Properties(LOCALPOTENTIAL) = H.getLocalPotential();
	  cur->R = W.R;
	  cur->Drift = W.G;
	}
      }
      ++it;
    }
  }

  bool ReptationMC::run() { 

    Estimators->reportHeader();

    initPolymers();

    IndexType block = 0;
    Pooma::Clock timer;
    IndexType accstep=0;
    IndexType nAcceptTot = 0;
    IndexType nRejectTot = 0;
    
    LocalPotentialEstimator pe(&Polymers);
    pe.resetReportSettings(RootName);

    //accumulate configuration: probably need to reorder
    HDFWalkerOutput WO(RootName);

      do {

	IndexType step = 0;
	timer.start();
	NumTurns = 0;

	do {
          movePolymers();
	  step++; accstep++;

	  MCWalkerConfiguration::iterator it=W.begin();
	  MCWalkerConfiguration::iterator it_end=W.end();
	  int ilink=0;
	  while(it != it_end) {
	    Polymers[ilink]->average(**it);
	    ++ilink; ++it;
	  }

	  Estimators->accumulate(W);
	  pe.accumulate();

// 	  cout << step << " ";
// 	  for(int i=Polymers[0]->Middle-1; i>=0 ; i--)
// 	    cout << Polymers[0]->PEavg[i] << " ";
// 	  cout << endl;
	} while(step<nSteps);
	timer.stop();
	
	nAcceptTot += nAccept;
	nRejectTot += nReject;

        RealType acceptedR = static_cast<RealType>(nAccept)/static_cast<RealType>(nAccept+nReject); 
	Estimators->flush();
	Estimators->setColumn(AcceptIndex,acceptedR);
	Estimators->report(accstep);
	pe.report(accstep);

        //change NumCuts to make accstep ~ 50%
	LogOut->getStream() 
	  << "Block " << block << " " 
	  << timer.cpu_time() << " " << NumTurns << " " << Polymers[0]->getID() << endl;

	nAccept = 0; nReject = 0;
	block++;

	if(pStride) WO.get(W);

      } while(block<nBlocks);
      
      LogOut->getStream() 
	<< "ratio = " 
	<< static_cast<double>(nAcceptTot)/static_cast<double>(nAcceptTot+nRejectTot)
	<< endl;

      Estimators->finalize();
      return true;
  }

  bool 
  ReptationMC::put(xmlNodePtr q){
    //nothing to do yet
    return true;
  }
  
  void 
  ReptationMC::movePolymers(){
    
    //Pooma::Clock timer;
    //RealType oneovertau = 1.0/Tau;
    //RealType oneover2tau = 0.5*oneovertau;
    RealType tauover2 = 0.5*Tau;
    RealType g = sqrt(Tau);
    
    typedef MCWalkerConfiguration::PropertyContainer_t PropertyContainer_t;
    typedef MCWalkerConfiguration::Walker_t Walker_t;

    for(int ilink=0; ilink<Polymers.size(); ilink++) {

      PolymerChain& polymer = *(Polymers[ilink]);

      if(!UseBounce && Random()<0.5) {
	polymer.flip(); 	  
	NumTurns++;
      }

      Walker_t* anchor = polymer.makeEnds();
      
      //save the local energies of the anchor and tails
      //eloc_xp = the energy of the front
      //eloc_yp = the energy of the proposed move
      //eloc_x = the energy of the tail
      //eloc_y = the energy of the tail-1
      RealType eloc_xp=anchor->Properties(LOCALENERGY);
      RealType eloc_x = polymer.tails[0]->Properties(LOCALENERGY);
      RealType eloc_y = polymer.tails[1]->Properties(LOCALENERGY);

      NumCuts = polymer.NumCuts;
      RealType Wpolymer=0.0;

      for(int i=0; i<NumCuts; ) {

	Walker_t* head=polymer.heads[i];

	//create a 3N-Dimensional Gaussian with variance=1
	makeGaussRandom(deltaR);
	W.R = anchor->R + g*deltaR + Tau* anchor->Drift;

	//update the distance table associated with W
	DistanceTable::update(W);

	//evaluate wave function
	ValueType logpsi(Psi.evaluateLog(W));
	
	//update the properties of the front chain
	RealType eloc_yp = head->Properties(LOCALENERGY) = H.evaluate(W);
	H.copy(head->getEnergyBase());
	head->Properties(LOCALPOTENTIAL) = H.getLocalPotential();
	head->R = W.R;

	//ValueType vsq = Dot(W.G,W.G);
	//ValueType scale = ((-1.0+sqrt(1.0+2.0*Tau*vsq))/vsq);
	//head->Drift = scale*W.G;
	head->Drift = W.G;

	//\f${x-y-\tau\nabla \ln \Psi_{T}(y))\f$
	//deltaR = anchor->R - W.R - heads[i]->Drift;
	//Gdrift *= exp(-oneover2tau*Dot(deltaR,deltaR));
	/* 
	   \f$ X= \{R_0, R_1, ... , R_M\}\f$
	   \f$ X' = \{R_1, .., R_M, R_{M+1}\}\f$
	   \f[ G_B(R_{M+1}\leftarrow R_{M}, \tau)/G_B(R_{0}\leftarrow R_{1}, \tau)
	   = exp\(-\tau/2[E_L(R_{M+1})+E_L(R_M)-E_L(R_1)-E_L(R_0)]\)\f]
	   *
	   -  eloc_yp = \f$E_L(R_{M+1})\f$
	   -  eloc_xp = \f$E_L(R_{M})\f$
	   -  eloc_y = \f$E_L(R_{1})\f$
	   -  eloc_x = \f$E_L(R_{0})\f$
	*/
	//Wpolymer *= exp(-oneover2tau*(eloc_yp+eloc_xp-eloc_x-eloc_y));
	Wpolymer +=(eloc_yp+eloc_xp-eloc_x-eloc_y);

	//move the anchor and swap the local energies for Wpolymer
	anchor=head;

	//increment the index
	i++;
	if(i<NumCuts) {
	  eloc_xp  = eloc_yp;
	  eloc_x = eloc_y;
	  eloc_y = polymer.tails[i+1]->Properties(LOCALENERGY);
	}
      }
      
      Wpolymer = exp(-tauover2*Wpolymer);
      double accept = std::min(1.0,Wpolymer);
      if(Random() < accept){//move accepted
	polymer.updateEnds();
	++nAccept;
      } else {
	++nReject; 
	if(UseBounce) {
	  NumTurns++;
	  polymer.flip();
	}
      }

//       RealType Bounce =  UseBounce ? 1.0-accept: 0.5;
//       if(Random()<Bounce) {
// 	polymer.flip();
// 	LogOut->getStream() << "Bounce = " << Bounce << " " << NumTurns << " " << polymer.MoveHead << endl;
// 	NumTurns++;//increase the number of turns
//       }
    }
  }

}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
