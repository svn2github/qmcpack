//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
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
#include "QMCWaveFunctions/Jastrow/PadeJastrowBuilder.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrowOrbital.h"
#include "QMCWaveFunctions/Jastrow/OneBodyJastrowOrbital.h"
#include "QMCWaveFunctions/DiffOrbitalBase.h"
#include "Utilities/IteratorUtility.h"
#include "Utilities/ProgressReportEngine.h"
#include "OhmmsData/AttributeSet.h"
//#include "QMCWaveFunctions/Jastrow/DiffTwoBodyJastrowOrbital.h"
//#include "QMCWaveFunctions/Jastrow/DiffOneBodyJastrowOrbital.h"

namespace qmcplusplus 
{

  PadeJastrowBuilder::PadeJastrowBuilder(ParticleSet& target, TrialWaveFunction& psi, 
      PtclPoolType& psets):
    OrbitalBuilderBase(target,psi),ptclPool(psets)
  {
    ClassName="PadeJastrowBuilder";
  }

  bool PadeJastrowBuilder::put(xmlNodePtr cur) 
  {

    ReportEngine PRE(ClassName,"put()");

    string sourceOpt=targetPtcl.getName();
    string jname="PadeJastrow";
    string spin="no";
    string id_b="jee_b";
    RealType pade_b=1.0;
    OhmmsAttributeSet pattrib;
    pattrib.add(jname,"name");
    pattrib.add(spin,"spin");
    pattrib.add(sourceOpt,"source");
    pattrib.put(cur);

    cur=cur->children;
    while(cur != NULL)
    {
      {//just to hide this
        string pname="0";
        OhmmsAttributeSet aa;
        aa.add(pname,"name");
        aa.add(id_b,"id");
        aa.put(cur);
        if(pname[0]=='B') putContent(pade_b,cur);
      }

      xmlNodePtr cur1=cur->children;
      while(cur1!= NULL)
      {
        string pname="0";
        OhmmsAttributeSet aa;
        aa.add(pname,"name");
        aa.add(id_b,"id");
        aa.put(cur1);
        if(pname[0]=='B') putContent(pade_b,cur1);
        cur1=cur1->next;
      }
      cur=cur->next;
    }

    app_log() << "PadeJastrowBuilder " << id_b << " = " << pade_b << endl;

    typedef PadeFunctor<RealType> FuncType;

    typedef TwoBodyJastrowOrbital<FuncType> JeeType;
    JeeType *J2 = new JeeType(targetPtcl);

    SpeciesSet& species(targetPtcl.getSpeciesSet());
    RealType q=species(0,species.addAttribute("charge"));

    if(spin == "no") 
    {
      RealType cusp=-0.5*q*q;
      FuncType *func=new FuncType(cusp,pade_b);
      func->setIDs("jee_cusp",id_b);//set the ID's

      J2->addFunc("pade_uu",0,0,func);

      //DerivFuncType *dfunc=new DerivFuncType(cusp,B);
      //dJ2->addFunc("pade_uu",0,0,dfunc);
      //dFuncList.push_back(dfunc);
      app_log() << "    Adding Spin-independent Pade Two-Body Jastrow Cusp " << cusp<< "\n";
    } 
    else 
    {
      //build uu functor
      RealType cusp_uu=-0.25*q*q;
      FuncType *funcUU=new FuncType(cusp_uu,pade_b);
      funcUU->setIDs("pade_uu",id_b);//set the ID's

      //build ud functor
      RealType cusp_ud=-0.5*q*q;
      FuncType *funcUD=new FuncType(cusp_ud,pade_b);
      funcUD->setIDs("pade_ud",id_b);//set the ID's

      J2->addFunc("pade_uu",0,0,funcUU);

      //DerivFuncType *dfuncUU=new DerivFuncType(cusp_uu,B);
      //DerivFuncType *dfuncUD=new DerivFuncType(cusp_ud,B);
      //dJ2->addFunc("pade_uu",0,0,dfuncUU);
      //dJ2->addFunc("pade_ud",0,1,dfuncUD);
      app_log() << "    Adding Spin-dependent Pade Two-Body Jastrow " << "\n";
      app_log() << "      parallel spin     " << cusp_uu << "\n";
      app_log() << "      antiparallel spin " << cusp_ud << "\n";
    }

    targetPsi.addOrbital(J2,"J2_pade");

    if(sourceOpt != targetPtcl.getName())
    {
      map<string,ParticleSet*>::iterator pa_it(ptclPool.find(sourceOpt));
      if(pa_it == ptclPool.end()) 
      {
        PRE.warning("PadeJastrowBuilder::put failed. "+sourceOpt+" does not exist.");
        return true;
      }
      ParticleSet& sourcePtcl= (*(*pa_it).second);

      app_log() << "  PadeBuilder::Adding Pade One-Body Jastrow with effective ionic charges." << endl;
      typedef OneBodyJastrowOrbital<FuncType> JneType;
      JneType* J1 = new JneType(sourcePtcl,targetPtcl);

      //typedef OneBodyJastrowOrbital<DerivFuncType> DerivJneType;
      //DerivJneType* dJ1=new DerivJneType(sourcePtcl,targetPtcl);

      SpeciesSet& Species(sourcePtcl.getSpeciesSet());
      int ng=Species.getTotalNum();
      int icharge = Species.addAttribute("charge");
      for(int ig=0; ig<ng; ++ig) 
      {
        RealType zeff=Species(icharge,ig);
        ostringstream j1id;
        j1id<<"pade_"<<Species.speciesName[ig];

        RealType sc=std::pow(2*zeff,0.25);
        FuncType *func=new FuncType(-zeff,pade_b,sc);
        func->setIDs(j1id.str(),id_b);

        J1->addFunc(ig,func);

        //DerivFuncType *dfunc=new DerivFuncType(-zeff,B,sc);
        //dJ1->addFunc(ig,dfunc);
        //dFuncList.push_back(dfunc);

        app_log() << "    " << Species.speciesName[ig] <<  " Zeff = " << zeff << " B= " << pade_b*sc << endl;
      }
      targetPsi.addOrbital(J1,"J1_pade");
    }
    return true;
  }

}
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 2930 $   $Date: 2008-07-31 10:30:42 -0500 (Thu, 31 Jul 2008) $
 * $Id: PadeConstraints.cpp 2930 2008-07-31 15:30:42Z jnkim $ 
 ***************************************************************************/
