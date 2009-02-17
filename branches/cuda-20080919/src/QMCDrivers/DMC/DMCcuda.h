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
#ifndef QMCPLUSPLUS_DMC_CUDA_H
#define QMCPLUSPLUS_DMC_CUDA_H
#include "QMCDrivers/QMCDriver.h" 
namespace qmcplusplus {

  class QMCUpdateBase;

  /** @ingroup QMCDrivers  PbyP
   *@brief Implements the DMC algorithm using particle-by-particle move. 
   */
  class DMCcuda: public QMCDriver {
  public:
    /// Constructor.
    DMCcuda(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h);
    bool run();
    bool put(xmlNodePtr cur);
 
  private:
    /// tau/mass
    RealType m_tauovermass;
    ///number of warmup steps
    int myWarmupSteps;
    ///period for walker dump
    int myPeriod4WalkerDump;
    ///update engine
    QMCUpdateBase* Mover;
    /// Copy Constructor (disabled)
    DMCcuda(const DMCcuda& a): QMCDriver(a) { }
    /// Copy operator (disabled).
    DMCcuda& operator=(const DMCcuda&) { return *this;}
    ///hide initialization from the main function
    void resetRun();
  };
}

#endif
/***************************************************************************
 * $RCSfile: DMCcuda.h,v $   $Author: jnkim $
 * $Revision: 1.5 $   $Date: 2006/07/17 14:29:40 $
 * $Id: DMCcuda.h,v 1.5 2006/07/17 14:29:40 jnkim Exp $ 
 ***************************************************************************/
