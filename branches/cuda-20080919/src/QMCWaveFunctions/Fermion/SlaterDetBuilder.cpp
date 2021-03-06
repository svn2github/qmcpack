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
#include "QMCWaveFunctions/Fermion/SlaterDetBuilder.h"
#include "QMCWaveFunctions/BasisSetFactory.h"
#include "Utilities/ProgressReportEngine.h"
#include "OhmmsData/AttributeSet.h"

namespace qmcplusplus {

  SlaterDetBuilder::SlaterDetBuilder(ParticleSet& els, TrialWaveFunction& psi,
     PtclPoolType& psets):
    OrbitalBuilderBase(els,psi), 
    ptclPool(psets), 
    myBasisSetFactory(0)
  { 
    ClassName="SlaterDetBuilder";
  }   
  SlaterDetBuilder::~SlaterDetBuilder()
  {
    DEBUG_MEMORY("SlaterDetBuilder::~SlaterDetBuilder");
    if(myBasisSetFactory)
    {
      delete myBasisSetFactory;
    }
  }

  bool SlaterDetBuilder::put(xmlNodePtr cur)
  {
    ReportEngine PRE(ClassName,"put(xmlNodePtr)");

    ///save the current node
    xmlNodePtr curRoot=cur;
    bool success=true;
    cur = cur->children;
    string cname, tname;
    while(cur != NULL) 
    {
      getNodeName(cname,cur);
      if(cname == basisset_tag) 
      {
        if(myBasisSetFactory == 0) 
        {
          myBasisSetFactory = new BasisSetFactory(targetPtcl,targetPsi, ptclPool);
          myBasisSetFactory->setReportLevel(ReportLevel);
        }
        myBasisSetFactory->createBasisSet(cur,curRoot);
        //Children.push_back(bsFactory);
      } 
      else if(cname == sd_tag) 
      {
        int is=SlaterDetSet.size();
        //add a new SlaterDeterminant
        SlaterDetSet.push_back(new SlaterDeterminant_t);
        sdet_coeff.push_back(1.0);
        int firstIndex = 0;
        xmlNodePtr tcur = cur->children;
        while(tcur != NULL) {
          getNodeName(tname,tcur);
          if(tname == param_tag) {
            putContent(sdet_coeff[is],tcur);
          } else if(tname == det_tag) {
            firstIndex = putDeterminant(tcur, firstIndex);
          }
          tcur = tcur->next;
        }
      }
      cur = cur->next;
    }
    
    if(SlaterDetSet.empty()) 
    {
      //fatal
      PRE.error("Failed to create a SlaterDeterminant.",true);
      return false;
    }

    if(SlaterDetSet.size()>1)
      buildMultiSlaterDetermiant();
    else
      buildSlaterDetermiant();

    delete myBasisSetFactory;
    myBasisSetFactory=0;

    return success;
  }


  int SlaterDetBuilder::putDeterminant(xmlNodePtr cur, int firstIndex) {

    ReportEngine PRE(ClassName,"putDeterminant(xmlNodePtr,int)");

    string basisName("invalid");
    string detname("0"), refname("0");
    OhmmsAttributeSet aAttrib;
    aAttrib.add(basisName,basisset_tag);
    aAttrib.add(detname,"id");
    aAttrib.add(refname,"ref");
    aAttrib.put(cur);

    //index of the last SlaterDeterminant
    int dIndex=DetSet.size();
    if(refname[0] == '0') { //create one and use detname
      if(detname[0] =='0') { //no id is given, assign one
        char newname[8];
        sprintf(newname,"det%d",dIndex);
        detname=newname;
        //add attributed id and ref
        xmlNewProp(cur,(const xmlChar*)"id",(const xmlChar*)newname);
        xmlNewProp(cur,(const xmlChar*)"ref",(const xmlChar*)newname);
      }
      else
      {
        //add reference name
        xmlNewProp(cur,(const xmlChar*)"ref",(const xmlChar*)detname.c_str());
      }
    }

    map<string,SPOSetBasePtr>::iterator lit(SPOSet.find(detname));
    Det_t* adet=0;
    SPOSetBasePtr psi;
    if(lit == SPOSet.end()) {
#if defined(ENABLE_SMARTPOINTER)
      psi.reset(myBasisSetFactory->createSPOSet(cur)); 
#else
      psi = myBasisSetFactory->createSPOSet(cur); 
#endif
      psi->put(cur);
      psi->checkObject();
      SPOSet[detname]=psi;
      app_log() << "  Creating a new SPO set " << detname << endl;
    } else {
      app_log() << "  Reusing a SPO set " << detname << endl;
      psi = (*lit).second;
    }

    if(psi->getOrbitalSetSize()) {
      map<string,Det_t*>::iterator dit(DetSet.find(detname));
      if(dit == DetSet.end()) {
        app_log() << "  Creating a Dirac Determinant " << detname << " First Index = " 
          << firstIndex << endl;
        adet = new Det_t(psi,firstIndex);
        adet->set(firstIndex,psi->getOrbitalSetSize());
        DetSet[detname]=adet;
      } else {
        app_log() << "  Reusing a Dirac Determinant " << detname << " First Index = " 
          << firstIndex << endl;
        adet = (*dit).second;
      }
      firstIndex += psi->getOrbitalSetSize();
    }

    //only if a determinant is not 0
    if(adet) SlaterDetSet.back()->add(adet);

    return firstIndex;
  }

  void SlaterDetBuilder::buildSlaterDetermiant() {
    if(SlaterDetSet.empty()) return;
    //add a SlaterDeterminant to the trial wavefuntion
    targetPsi.addOrbital(SlaterDetSet[0],"SlaterDet");
  }

  void SlaterDetBuilder::buildMultiSlaterDetermiant() {
//    MultiSlaterDeterminant<LCOrbitalSet> *multidet= new MultiSlaterDeterminant<LCOrbitalSet>;
//    for(int i=0; i<SlaterDetSet.size(); i++) {
//      multidet->add(SlaterDetSet[i],sdet_coeff[i]);
//    }
//    multidet->setOptimizable(true);
//    //add a MultiDeterminant to the trial wavefuntion
//    targetPsi.addOrbital(multidet);
  }
}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
