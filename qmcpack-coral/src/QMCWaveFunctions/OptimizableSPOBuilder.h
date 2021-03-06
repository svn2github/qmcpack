//////////////////////////////////////////////////////////////////
// (c) Copyright 2010-  by Ken Esler and Jeongnim Kim
//////////////////////////////////////////////////////////////////
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

#ifndef OPTIMIZABLE_SPO_BUILDER_H
#define OPTIMIZABLE_SPO_BUILDER_H

#include "QMCWaveFunctions/BasisSetBase.h"
#include "QMCWaveFunctions/OptimizableSPOSet.h"


namespace qmcplusplus
{
class OptimizableSPOBuilder : public BasisSetBuilder
{
protected:
  typedef map<string,ParticleSet*> PtclPoolType;
  typedef map<string,SPOSetBase*>  SPOPoolType;
  ParticleSet *targetPtcl;
public:
  OptimizableSPOBuilder(ParticleSet& p, PtclPoolType& psets,
                        xmlNodePtr cur);

  bool put (xmlNodePtr cur);

  /** initialize the Antisymmetric wave function for electrons
   *@param cur the current xml node
   */
  SPOSetBase* createSPOSet(xmlNodePtr cur);
  //    SPOSetBase* createSPOSet(xmlNodePtr cur, SPOPool_t& spo_pool);

};
}

#endif
