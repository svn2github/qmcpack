/////////////////////////////////////////////////////////////////
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
#include "QMCDrivers/QMCDriver.h"
#include "Utilities/OhmmsInfo.h"
#include "Particle/MCWalkerConfiguration.h"
#include "Particle/DistanceTable.h"
#include "Particle/HDFWalkerIO.h"
#include "ParticleBase/ParticleUtility.h"
#include "ParticleBase/RandomSeqGenerator.h"

namespace ohmmsqmc {

  int QMCDriver::Counter = -1;
  
  QMCDriver::QMCDriver(MCWalkerConfiguration& w, 
		       TrialWaveFunction& psi, 
		       QMCHamiltonian& h):
    AcceptIndex(-1),
    Tau(0.001), FirstStep(0.0),
    nBlocks(100), nSteps(1000), pStride(false), 
    nAccept(0), nReject(0), nTargetWalkers(0),
    QMCType("invalid"), 
    W(w), Psi(psi), H(h), Estimators(0),
    LogOut(0), qmcNode(NULL)
  { 
    
    //Counter++; 
    m_param.add(nSteps,"steps","int");
    m_param.add(nBlocks,"blocks","int");
    m_param.add(nTargetWalkers,"walkers","int");
    m_param.add(Tau,"Tau","AU");
    m_param.add(FirstStep,"FirstStep","AU");
  }

  QMCDriver::~QMCDriver() { 
    
    if(Estimators) {
      if(Estimators->size()) W.setLocalEnergy(Estimators->average(0));
      delete Estimators;
    }
    
    if(LogOut) delete LogOut;
  }

  /** process a <qmc/> element 
   * @param cur xmlNode with qmc tag
   *
   * This function is called before QMCDriver::run and following actions are taken:
   * - Initialize basic data to execute run function.
   * -- distance tables
   * -- resize deltaR and drift with the number of particles
   * -- assign cur to qmcNode
   * - process input file
   *   -- putQMCInfo: <parameter/> s for QMC
   *   -- put : extra data by derived classes
   * - initialize Estimators
   * The virtual function put(xmlNodePtr cur) is where QMC algorithm-dependent
   * data are registered and initialized.
   */
  void QMCDriver::process(xmlNodePtr cur) {

    Counter++; 

    W.setUpdateMode(MCWalkerConfiguration::Update_Particle);

    deltaR.resize(W.getTotalNum());
    drift.resize(W.getTotalNum());

    qmcNode=cur;

    //create estimator
    putQMCInfo(qmcNode);
    put(qmcNode);

    if(Estimators == 0) {
      Estimators =new ScalarEstimatorManager(H);
    }

    Estimators->put(qmcNode);
    //set the stride for the scalar estimators 
    Estimators->setStride(nSteps);

    Estimators->resetReportSettings(RootName);
    AcceptIndex=Estimators->addColumn("AcceptRatio");
  }

  /**Sets the root file name for all output files.!
   * \param aname the root file name
   *
   * All output files will be of
   * the form "aname.s00X.suffix", where "X" is number
   * of previous QMC runs for the simulation and "suffix"
   * is the suffix for the output file. 
   */
  void QMCDriver::setFileRoot(const string& aname) {
    RootName = aname;
    
    char logfile[128];
    sprintf(logfile,"%s.%s",RootName.c_str(),QMCType.c_str());
    
    if(LogOut) delete LogOut;
    LogOut = new OhmmsInform(" ",logfile);
    
    LogOut->getStream() << "Starting a " << QMCType << " run " << endl;
  }

  ///** initialize estimators and other internal data */  
  //void QMCDriver::getReady() {
  //  
  //  //Estimators.resetReportSettings(RootName);
  //  //AcceptIndex = Estimators.addColumn("AcceptRatio");
  //  //Estimators.reportHeader();
  //}
  

  /** Add walkers to the end of the ensemble of walkers.  
   *@param nwalkers number of walkers to add
   *@return true, if the walker configuration is not empty.
   *
   * Assign positions to any new 
   * walkers \f[ {\bf R}[i] = {\bf R}[i-1] + g{\bf \chi}, \f]
   * where \f$ g \f$ is a constant and \f$ {\bf \chi} \f$
   * is a 3N-dimensional gaussian.
   * As a last step, for each walker calculate 
   * the properties given the new configuration
   <ul>
   <li> Local Energy \f$ E_L({\bf R} \f$
   <li> wavefunction \f$ \Psi({\bf R}) \f$
   <li> wavefunction squared \f$ \Psi^2({\bf R}) \f$
   <li> weight\f$ w({\bf R}) = 1.0\f$  
   <li> drift velocity \f$ {\bf v_{drift}}({\bf R})) \f$
   </ul>
  */
  void 
  QMCDriver::addWalkers(int nwalkers) {

    if(nwalkers>0) {
      //add nwalkers walkers to the end of the ensemble
      int nold = W.getActiveWalkers();

      LOGMSG("Adding " << nwalkers << " walkers to " << nold << " existing sets")

      W.createWalkers(nwalkers);
      LogOut->getStream() <<"Added " << nwalkers << " walkers" << endl;
      
      ParticleSet::ParticlePos_t rv(W.getTotalNum());
      RealType g = FirstStep;
      
      MCWalkerConfiguration::iterator it(W.begin()), it_end(W.end()),itprev;
      int iw = 0;
      while(it != it_end) {
	if(iw>=nold) {
	  makeGaussRandom(rv);
	  if(iw)
	    (*it)->R = (*itprev)->R+g*rv;
	  else
	    (*it)->R = W.R+g*rv;
	}
	itprev = it;
	++it;++iw;
      }
    } else {
      LOGMSG("Using the existing " << W.getActiveWalkers() << " walkers")
    }

    //calculate local energies and wave functions:
    //can be redundant but the overhead is small
    //W.Energy.resize(W.getActiveWalkers(),H.size()+1);
    LOGMSG("Evaluate all the walkers before starting")
    MCWalkerConfiguration::iterator it(W.begin()),it_end(W.end());
    int iwalker=0;

    int numCopies= (H1.empty())?1:H1.size();
    int numValues= H.size();

    while(it != it_end) {

      (*it)->resizeDynProperty(numCopies,numValues);
      W.R = (*it)->R;
      DistanceTable::update(W);

      //ValueType psi = Psi.evaluate(W);
      ValueType logpsi(Psi.evaluateLog(W));
      ValueType vsq = Dot(W.G,W.G);
      ValueType scale = ((-1.0+sqrt(1.0+2.0*Tau*vsq))/vsq);
      (*it)->Drift = scale*W.G;

      RealType ene = H.evaluate(W);
      (*it)->Properties(LOCALPOTENTIAL) = H.getLocalPotential();
      (*it)->resetProperty(logpsi,Psi.getSign(),ene);

      H.copy((*it)->getEnergyBase());
      ++it;++iwalker;
    }
  }
  
  /** Parses the xml input file for parameter definitions for a single qmc simulation.
   * \param q the current xmlNode
   *
   Available parameters added to the ParameterSeet
   <ul>
   <li> blocks: number of blocks, default 100
   <li> steps: number of steps per block, default 1000
   <li> walkers: target population of walkers, default 100
   <li> Tau: the timestep, default 0.001
   <li> stride: flag for printing the ensemble of walkers,  default false
   <ul>
   <li> true: print every block
   <li> false: print at the end of the run
   </ul>
   </ul>
   In addition, sets the stride for the scalar estimators
   such that the scalar estimators flush and print to
   file every block and calls the function to initialize
   the walkers.
   *Derived classes can add their parameters.
   */
  bool 
  QMCDriver::putQMCInfo(xmlNodePtr cur) {
    
    int defaultw = 100;
    int targetw = 0;
     
    m_param.get(cout);
    
    //  nTargetWalkers=0;
    if(cur) {

      xmlAttrPtr att = cur->properties;
      while(att != NULL) {
	string aname((const char*)(att->name));
	const char* vname=(const char*)(att->children->content);
	if(aname == "blocks") nBlocks = atoi(vname);
	else if(aname == "steps") nSteps = atoi(vname);
	att=att->next;
      }
      
      xmlNodePtr tcur=cur->children;
      //initialize the parameter set
      m_param.put(cur);
      //determine how often to print walkers to hdf5 file
      while(tcur != NULL) {
	string cname((const char*)(tcur->name));
	if(cname == "record") {
	  int stemp;
	  att = tcur->properties;
	  while(att != NULL) {
	    string aname((const char*)(att->name));
	    if(aname == "stride") stemp=atoi((const char*)(att->children->content));
	    att=att->next;
	  }
	  if(stemp >= 0){
	    pStride = true;
	    LogOut->getStream() << "print walker ensemble every block." << endl;
	  } else {
	    LogOut->getStream() << "print walker ensemble after last block." << endl;
	  }
	}
	tcur=tcur->next;
      }
    }
    
    LogOut->getStream() << "timestep = " << Tau << endl;
    LogOut->getStream() << "blocks = " << nBlocks << endl;
    LogOut->getStream() << "steps = " << nSteps << endl;
    LogOut->getStream() << "FirstStep = " << FirstStep << endl;
    LogOut->getStream() << "walkers = " << W.getActiveWalkers() << endl;
    m_param.get(cout);

    /*check to see if the target population is different 
      from the current population.*/ 
    int nw  = W.getActiveWalkers();
    int ndiff = 0;
    if(nw) {
      ndiff = nTargetWalkers-nw;
    } else {
      ndiff=(nTargetWalkers)? nTargetWalkers:defaultw;
    }

    addWalkers(ndiff);

    //always true
    return (W.getActiveWalkers()>0);
  }
}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
