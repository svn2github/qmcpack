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
#include "QMCDrivers/MolecuDMC.h"
#include "Utilities/OhmmsInfo.h"
#include "Particle/MCWalkerConfiguration.h"
#include "Particle/DistanceTable.h"
#include "Particle/HDFWalkerIO.h"
#include "ParticleBase/ParticleUtility.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "QMCDrivers/MolecuFixedNodeBranch.h"
#include "Message/Communicate.h"
#include "Utilities/Clock.h"

namespace ohmmsqmc {

  MolecuDMC::MolecuDMC(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h):
    QMCDriver(w,psi,h),BranchInfo("default"){ 
    RootName = "dmc";
    QMCType ="dmc";
  }

  void MolecuDMC::setBranchInfo(const string& afile) {
    BranchInfo=afile;
  }
  
  bool MolecuDMC::put(xmlNodePtr q){
    return true;
  }
  
  /** Advance the walkers nblocks*nsteps timesteps. 
   * @param nblocks number of blocks
   * @param nsteps number of steps
   * @param tau the timestep
   *
   * For each timestep:
   * <ul>
   * <li> Move all the particles of a walker.
   * <li> Calculate the properties for the new walker configuration.
   * <li> Accept/reject the new configuration.
   * <li> Accumulate the estimators.
   * <li> Update the trial energy \f$ E_T. \f$
   * <li> Branch the population of walkers (birth/death algorithm).
   * </ul>
   * For each block:
   * <ul>
   * <li> Flush the estimators and print to file.
   * <li> Update the estimate of the local energy.
   * <li> (Optional) Print the ensemble of walker configurations.
   * </ul>
   * Default mode: Print the ensemble of walker configurations 
   * at the end of the run.
   */
  bool MolecuDMC::run() { 

    //add columns
    IndexType PopIndex = Estimators->addColumn("Population");
    IndexType EtrialIndex = Estimators->addColumn("E_T");
    //write the header
    Estimators->reportHeader();

    MolecuFixedNodeBranch<RealType> brancher(Tau,W.getActiveWalkers());
    brancher.put(qmcNode,LogOut);

    if(BranchInfo != "default")  brancher.read(BranchInfo);

    MCWalkerConfiguration::iterator it(W.begin()); 
    MCWalkerConfiguration::iterator it_end(W.end()); 
    while(it != it_end) {
      (*it)->Properties(WEIGHT) = 1.0;
      (*it)->Properties(MULTIPLICITY) = 1.0;
      ++it;
    }
    
    /*if VMC/DMC directly preceded DMC (Counter > 0) then
      use the average value of the energy estimator for
      the reference energy of the brancher*/
    if(Counter) {
      RealType e_ref = W.getLocalEnergy();
      LOGMSG("Overwriting the reference energy by the local energy " << e_ref)  
      brancher.setEguess(e_ref);
    }
    
    IndexType block = 0;
    
    Pooma::Clock timer;
    int Population = W.getActiveWalkers();
    int tPopulation = W.getActiveWalkers();
    RealType Eest = brancher.E_T;
    IndexType accstep=0;
    IndexType nAcceptTot = 0;
    IndexType nRejectTot = 0;
    
    do {
      IndexType step = 0;
      timer.start();
      do {
        Population = W.getActiveWalkers();
        advanceWalkerByWalker(brancher);
        step++; accstep++;
        Estimators->accumulate(W);
        Eest = brancher.update(Population,Eest);
        brancher.branch(accstep,W);
      } while(step<nSteps);
      timer.stop();
      
      nAcceptTot += nAccept;
      nRejectTot += nReject;
      Estimators->flush();
      
      Estimators->setColumn(PopIndex,static_cast<RealType>(Population));
      Estimators->setColumn(EtrialIndex,Eest);
      Estimators->setColumn(AcceptIndex,
      	            static_cast<RealType>(nAccept)/static_cast<RealType>(nAccept+nReject));
      Estimators->report(accstep);
      LogOut->getStream() << "Block " << block << " " << timer.cpu_time()
      		    << " " << Population << endl;
      Eest = Estimators->average(0);
      
      nAccept = 0; nReject = 0;
      block++;
      if(pStride) {
        //create an output engine: could accumulate the configurations
        HDFWalkerOutput WO(RootName);
        WO.get(W);
        brancher.write(WO.getGroupID());
      }
      W.reset();
    } while(block<nBlocks);
    
    LogOut->getStream() 
      << "ratio = " << static_cast<double>(nAcceptTot)/static_cast<double>(nAcceptTot+nRejectTot)
      << endl;
    
    if(!pStride) {
      //create an output engine: could accumulate the configurations
      HDFWalkerOutput WO(RootName);
      WO.get(W);
      brancher.write(WO.getGroupID());
    }

    Estimators->finalize();
    return true;
  }

  /**  Advance all the walkers one timstep. 
   * 
   Propose a move for each walker from its old 
   position \f${\bf R'}\f$ to a new position \f${\bf R}\f$ 
   \f[ 
   {\bf R'} + {\bf \chi} + 
   \tau {\bf v_{drift}}({\bf R'}) =  {\bf R},
   \f]
   where \f$ {\bf \chi} \f$ is a 3N-diminsional 
   gaussian of mean zero and variance \f$ \tau \f$
   and \f$ {\bf v_{drift}} \f$ is the drift velocity
   \f[
   {\bf v_{drift}}({\bf R'}) = {\bf \nabla} 
   \ln |\Psi_T({\bf R'})| = \Psi_T({\bf R'})^{-1} 
   {\bf \nabla} \Psi_T({\bf R'}). 
   \f]
   For DMC it is necessary to check if the walker 
   crossed the nodal surface, if this is the case 
   then reject the move, otherwise Metropolis 
   accept/reject with probability
   \f[
   P_{accept}(\mathbf{R'}\rightarrow\mathbf{R}) = 
   \min\left[1,\frac{G(\mathbf{R}\rightarrow\mathbf{R'})
   \Psi_T(\mathbf{R})^2}{G(\mathbf{R'}\rightarrow\mathbf{R})
   \Psi_T(\mathbf{R'})^2}\right],
   \f] 
   where \f$ G \f$ is the drift-diffusion Green's function 
   \f[
   G(\mathbf{R'} \rightarrow 
   \mathbf{R}) = (2\pi\tau)^{-3/2}\exp \left[ -
   (\mathbf{R}-\mathbf{R'}-\tau \mathbf{v_{drift}}
   (\mathbf{R'}))^2/2\tau \right].
   \f]
   If the move is accepted, update the walker configuration and
   properties.  For rejected moves, do not update except for the
   Age which needs to be incremented by one.
   *
   Assign a weight and multiplicity for each walker
   \f[ weight = \exp \left[-\tau(E_L(\mathbf{R})+
   E_L(\mathbf{R})-2E_T)/2 \right]. \f]
   \f[ multiplicity = \exp \left[-\tau(E_L(\mathbf{R})+
   E_L(\mathbf{R})-2E_T)/2 \right] + \nu, \f]
   where \f$ \nu \f$ is a uniform random number.
   *
   Due to the fact that the drift velocity diverges on the nodal
   surface of the trial function \f$ \Psi_T \f$, it is possible
   for walkers close to the nodes to make excessively large proposed
   moves \f$ {\bf R'} \longrightarrow {\bf R} \f$.  With the
   accept/reject step this can lead to persistent configurations;
   a remedy is to impose a cutoff on the magnitude of the drift
   velocity.  We use the smooth cutoff proposed by Umrigar, 
   Nightingale and Runge
   [J. Chem. Phys., {\textbf 99}, 2865, (1993)]
   \f[
   {\bf \bar{v}_{drift}} = \frac{-1+\sqrt{1+2 \tau v^2_{drift}}}
   {\tau v^2_{drift}}{\bf v_{drift}},
   \f]
   where \f$ {\bf v_{drift}} \f$ is evaluated at 
   \f$ {\bf R'} \f$ and the magnitude of the drift
   \f$ \tau {\bf v_{drift}} \f$ is unchanged for small
   \f$ \tau v^2_{drift} \f$ and is limited to \f$ \sqrt{2\tau} \f$
   for large \f$ \tau v^2_{drift} \f$. 
   */
  template<class BRANCHER>
  void 
  MolecuDMC::advanceWalkerByWalker(BRANCHER& Branch) {
    

    //Pooma::Clock timer;
    RealType oneovertau = 1.0/Tau;
    RealType oneover2tau = 0.5*oneovertau;
    RealType g = sqrt(Tau);

    //MCWalkerConfiguration::PropertyContainer_t Properties;
    MCWalkerConfiguration::iterator it(W.begin()); 
    MCWalkerConfiguration::iterator it_end(W.end()); 
    while(it != it_end) {
      
      (*it)->Properties(WEIGHT) = 1.0;
      (*it)->Properties(MULTIPLICITY) = 1.0;
      
      //copy the properties of the working walker
      W.Properties = (*it)->Properties;

      //save old local energy
      RealType eold = W.Properties(LOCALENERGY);
      RealType emixed = eold;

      //create a 3N-Dimensional Gaussian with variance=1
      makeGaussRandom(deltaR);
      
      W.R = g*deltaR + (*it)->R + (*it)->Drift;
      
      //update the distance table associated with W
      DistanceTable::update(W);
      
      //evaluate wave function
      ValueType logpsi(Psi.evaluateLog(W));
      //update the properties
      W.Properties(LOCALENERGY) = H.evaluate(W);
      W.Properties(LOGPSI) =logpsi;
      W.Properties(SIGN) = Psi.getSign();
      bool accepted=false; 

      //deltaR = W.R - (*it)->R - (*it)->Drift;
      RealType logGf = -0.5*Dot(deltaR,deltaR);
      
      //scale the drift term to prevent persistent cofigurations
      ValueType vsq = Dot(W.G,W.G);
      
      //converting gradients to drifts, D = tau*G (reuse G)
      //   W.G *= Tau;//original implementation with bare drift
      ValueType scale = ((-1.0+sqrt(1.0+2.0*Tau*vsq))/vsq);
      drift = scale*W.G;
      deltaR = (*it)->R - W.R - drift;

      //RealType backwardGF = exp(-oneover2tau*Dot(deltaR,deltaR));
      RealType logGb = -oneover2tau*Dot(deltaR,deltaR);
      
      //set acceptance probability
      RealType prob= std::min(exp(logGb-logGf +2.0*(W.Properties(LOGPSI)-(*it)->Properties(LOGPSI))),1.0);
      
      if(Random() > prob){
	(*it)->Properties(AGE)++;
	emixed += emixed;
      } else {
	accepted=true;  
	W.Properties(AGE) = 0;
	(*it)->R = W.R;
	(*it)->Drift = drift;
	(*it)->Properties = W.Properties;
	(*it)->Properties(LOCALPOTENTIAL) = H.getLocalPotential();
	H.copy((*it)->getEnergyBase());
	emixed += W.Properties(LOCALENERGY);
      }
      
      //calculate the weight and multiplicity
      ValueType M = Branch.branchGF(Tau,emixed*0.5,1.0-prob);
      if((*it)->Properties(AGE) > 3.0) M = min(0.5,M);
      if((*it)->Properties(AGE) > 0.9) M = min(1.0,M);
      (*it)->Properties(WEIGHT) = M; 
      (*it)->Properties(MULTIPLICITY) = M + Random();
      
      //node-crossing: kill it for the time being
      if(Branch(W.Properties(SIGN),(*it)->Properties(SIGN))) {
	accepted=false;     
	(*it)->Properties(WEIGHT) = 0.0; 
	(*it)->Properties(MULTIPLICITY) = 0.0;
      }

      if(accepted) 
        ++nAccept;
      else 
        ++nReject;
      ++it;
    }
  }
}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
