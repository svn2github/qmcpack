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
#include "Estimators/CSEnergyEstimator.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "ParticleBase/ParticleAttribOps.h"
#include "Message/CommOperators.h"
#include "QMCDrivers/DriftOperators.h"

namespace qmcplusplus {

  /** constructor
   * @param h QMCHamiltonian to define the components
   * @param hcopy number of copies of QMCHamiltonians
   */
  CSEnergyEstimator::CSEnergyEstimator(QMCHamiltonian& h, int hcopy) 
  {
    NumCopies=hcopy;
    NumObservables = h.size();
    d_data.resize(NumCopies*3+NumCopies*(NumCopies-1)/2);
  }

  CSEnergyEstimator::CSEnergyEstimator(const CSEnergyEstimator& mest): 
    ScalarEstimatorBase(mest)
  {
    NumCopies=mest.NumCopies;
    d_data.resize(mest.d_data.size());
  }

  ScalarEstimatorBase* CSEnergyEstimator::clone()
  {
    return new CSEnergyEstimator(*this);
  }

  /**  add the local energy, variance and all the Hamiltonian components to the scalar record container
   *@param record storage of scalar records (name,value)
   */
  void 
  CSEnergyEstimator::add2Record(RecordNamedProperty<RealType>& record, BufferType& msg) {
    FirstIndex = record.add("LE0");
    int dummy=record.add("LESQ0");
    dummy=record.add("WPsi0");
    char aname[32];
    for(int i=1; i<NumCopies; i++)
    {
      sprintf(aname,"LE%i",i);   
      dummy=record.add(aname);
      sprintf(aname,"LESQ%i",i);   
      dummy=record.add(aname);
      sprintf(aname,"WPsi%i",i);   
      dummy=record.add(aname);
    }

    for(int i=0; i<NumCopies-1; i++) {
      for(int j=i+1; j<NumCopies; j++) {
        sprintf(aname,"DiffS%iS%i",i,j); 
        dummy=record.add(aname);
      }
    }

    msg.add(d_data.begin(),d_data.end());
  }

  void 
  CSEnergyEstimator::accumulate(const Walker_t& awalker, RealType wgt) 
  {
    int ii=0;
    for(int i=0; i<NumCopies; i++) 
    {
      //get the pointer to the i-th row
      const RealType* restrict prop=awalker.getPropertyBase(i);
      RealType uw = prop[UMBRELLAWEIGHT];
      RealType e = prop[LOCALENERGY];
      d_data[ii++]+=uw*e;
      d_data[ii++]+=uw*e*e;
      d_data[ii++]+=uw;
    }
    //TinyVector<RealType,4> e,uw;
    //for(int i=0; i<NumCopies; i++) 
    //{
    //  //get the pointer to the i-th row
    //  const RealType* restrict prop=awalker.getPropertyBase(i);
    //  uw[i] = prop[UMBRELLAWEIGHT];
    //  e[i] = prop[LOCALENERGY];
    //  d_data[ii++]+=uw[i]*e[i];
    //  d_data[ii++]+=uw[i]*e[i]*e[i];
    //  d_data[ii++]+=uw[i];
    //}

    //for(int i=0; i<NumCopies-1; i++) 
    //  for(int j=i+1; j<NumCopies; j++)
    //    d_data[ii++]+=uw[i]*e[i]-uw[j]*e[j];
  }

  void 
  CSEnergyEstimator::evaluateDiff() 
  {
    int ii=0;
    for(int i=0; i<NumCopies; i++,ii+=3) 
    {
      RealType r= d_wgt/d_data[ii+2];
      d_data[ii] *= r;
      d_data[ii+1] *= r;
    }

    //d_wgt=1.0;
    for(int i=0; i<NumCopies-1; i++) 
      for(int j=i+1; j<NumCopies; j++)
        d_data[ii++]+=d_data[j*3]-d_data[i*3];
  }

  ///Set CurrentWalker to zero so that accumulation is done in a vectorized way
  void CSEnergyEstimator::reset() 
  {
    d_wgt=0.0;
    std::fill(d_data.begin(), d_data.end(),0.0);
  }

  void CSEnergyEstimator::report(RecordNamedProperty<RealType>& record, RealType wgtinv)
  {
  }

  /** calculate the averages and reset to zero
   * @param record a container class for storing scalar records (name,value)
   * @param wgtinv the inverse weight
   *
   * Disable collection. CSEnergyEstimator does not need to communiate at all.
   */
  void CSEnergyEstimator::report(RecordNamedProperty<RealType>& record, 
      RealType wgtinv, BufferType& msg) 
  {
    msg.get(d_data.begin(),d_data.end());
    report(record,wgtinv);
  }
}
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1926 $   $Date: 2007-04-20 12:30:26 -0500 (Fri, 20 Apr 2007) $
 * $Id: CSEnergyEstimator.cpp 1926 2007-04-20 17:30:26Z jnkim $
 ***************************************************************************/
