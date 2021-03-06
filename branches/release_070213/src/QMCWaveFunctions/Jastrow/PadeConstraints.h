//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_PADE_COMBO_CONSTRAINTS_H
#define QMCPLUSPLUS_PADE_COMBO_CONSTRAINTS_H
#include "QMCWaveFunctions/Jastrow/PadeFunctors.h"
#include "QMCWaveFunctions/OrbitalConstraintsBase.h"
#include "QMCWaveFunctions/ComboOrbital.h"
#include "QMCWaveFunctions/Jastrow/NumericalJastrowFunctor.h"

namespace qmcplusplus {

  struct PadeConstraints: public OrbitalConstraintsBase {
    typedef PadeFunctor<RealType> FuncType;
    bool IgnoreSpin;
    RealType B;
    string ID;
    vector<FuncType*> FuncList;

    ~PadeConstraints();

    PadeConstraints(ParticleSet& p, TrialWaveFunction& psi, bool nospin=true);

    inline void apply() {
      for(int i=0; i<FuncList.size(); i++) {
        FuncList[i]->B0=B;
        FuncList[i]->reset();
      }
    }

    void addOptimizables(VarRegistry<RealType>& outVars) {
      outVars.add(ID,&B,1);
    }

    OrbitalBase* createTwoBody();
    OrbitalBase* createOneBody(ParticleSet& source);
    void addExtra2ComboOrbital(ComboOrbital* jcombo) {}
    bool put(xmlNodePtr cur);
  };

  struct ScaledPadeConstraints: public OrbitalConstraintsBase {
    typedef ScaledPadeFunctor<RealType> FuncType;
    bool IgnoreSpin;
    RealType B;
    RealType C;
    string BID;
    string CID;
    vector<FuncType*> FuncList;

    ScaledPadeConstraints(ParticleSet& p, TrialWaveFunction& psi, bool nospin=true);

    ~ScaledPadeConstraints();

    inline void apply() {
      for(int i=0; i<FuncList.size(); i++) {
        FuncList[i]->B=B;
        FuncList[i]->C=C;
        FuncList[i]->reset();
      }
    }


    void addOptimizables(VarRegistry<RealType>& outVars) {
      outVars.add(BID,&B,1);
      outVars.add(CID,&C,1);
    }

    OrbitalBase* createTwoBody();
    OrbitalBase* createOneBody(ParticleSet& source);
    void addExtra2ComboOrbital(ComboOrbital* jcombo) {}
    bool put(xmlNodePtr cur);
  };

  //struct PadeOnGridConstraints: public OrbitalConstraintsBase {
  //  ///analytic functor
  //  typedef PadeFunctor<RealType> InFuncType;
  //  ///numerical functor
  //  typedef NumericalJastrow<RealType> FuncType;
  //  bool IgnoreSpin;
  //  RealType B;
  //  string ID;
  //  vector<InFuncType*> InFuncList;
  //  vector<FuncType*> FuncList;

  //  ~PadeOnGridConstraints();
  //  PadeOnGridConstraints(bool nospin=true):IgnoreSpin(nospin) {}
  //  void apply();
  //  void addOptimizables(VarRegistry<RealType>& outVars);
  //  OrbitalBase* createTwoBody(ParticleSet& target);
  //  OrbitalBase* createOneBody(ParticleSet& target, ParticleSet& source);
  //  inline void addTwoBodyPart(ParticleSet& target, ComboOrbital* jcombo) {
  //    OrbitalBase* j2 = createTwoBody(target);
  //    if (j2) jcombo->Psi.push_back(j2);
  //  }
  //  bool put(xmlNodePtr cur);
  //};

}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
