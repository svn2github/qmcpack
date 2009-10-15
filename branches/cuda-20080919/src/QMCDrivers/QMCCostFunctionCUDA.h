//////////////////////////////////////////////////////////////////
// (c) Copyright 2005- by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Ken Esler
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: kpesler@gmail.com
//
// Supported by 
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_COSTFUNCTION_CUDA_H
#define QMCPLUSPLUS_COSTFUNCTION_CUDA_H

#include "QMCDrivers/QMCCostFunctionBase.h"
#include "QMCDrivers/CloneManager.h"

namespace qmcplusplus {

  /** @ingroup QMCDrivers
   * @brief Implements wave-function optimization
   *
   * Optimization by correlated sampling method with configurations 
   * generated from VMC running on a single thread.
   */
  class QMCCostFunctionCUDA: public QMCCostFunctionBase, public CloneManager
  {
  public:

    ///Constructor.
    QMCCostFunctionCUDA( MCWalkerConfiguration& w, TrialWaveFunction& psi, 
        QMCHamiltonian& h, HamiltonianPool& hpool);

    ///Destructor
    ~QMCCostFunctionCUDA();

    void getConfigurations(const string& aroot);
    void checkConfigurations();
  protected:
    vector<QMCHamiltonian*> H_KE_Node;
    vector<Matrix<Return_t>*> RecordsOnNode;
    void resetPsi();
    Return_t correlatedSampling();
  };
}
#endif
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1804 $   $Date: 2007-02-24 14:49:09 -0600 (Sat, 24 Feb 2007) $
 * $Id: QMCCostFunctionCUDA.h 1804 2007-02-24 20:49:09Z jnkim $ 
 ***************************************************************************/
