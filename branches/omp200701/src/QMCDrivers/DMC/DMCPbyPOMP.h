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
#ifndef QMCPLUSPLUS_DMC_PARTICLEBYPARTICLE_OPNEMP_H
#define QMCPLUSPLUS_DMC_PARTICLEBYPARTICLE_OPNEMP_H
#include "QMCDrivers/QMCDriver.h" 
#include "QMCDrivers/CloneManager.h" 
namespace qmcplusplus {

  class DMCUpdateBase;

  /** @ingroup QMCDrivers 
   *@brief A dummy QMCDriver for testing
   */
  class DMCPbyPOMP: public QMCDriver, public CloneManager {
  public:

    /// Constructor.
    DMCPbyPOMP(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h,
        HamiltonianPool& hpool);

    bool run();
    bool put(xmlNodePtr cur);
 
  private:
    ///Index to determine what to do when node crossing is detected
    IndexType KillNodeCrossing;
    ///Interval between branching
    IndexType BranchInterval;
    ///hdf5 file name for Branch conditions
    string BranchInfo;
    ///input string to determine kill walkers or not
    string KillWalker;
    ///input string to determine swap walkers among mpi processors
    string SwapWalkers;
    ///input string to determine to use reconfiguration
    string Reconfiguration;
    ///input string to determine to use nonlocal move
    string NonLocalMove;
    ///input string to benchmark OMP performance
    string BenchMarkRun;

    void resetRun();
    void benchMark();
    bool runDMCBlocks();
    //void dmcWithBranching();
    //void dmcWithReconfiguration();

    /// Copy Constructor (disabled)
    DMCPbyPOMP(const DMCPbyPOMP& a): QMCDriver(a), CloneManager(a) { }
    /// Copy operator (disabled).
    DMCPbyPOMP& operator=(const DMCPbyPOMP&) { return *this;}
  };
}

#endif
/***************************************************************************
 * $RCSfile: DMCPbyPOMP.h,v $   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
