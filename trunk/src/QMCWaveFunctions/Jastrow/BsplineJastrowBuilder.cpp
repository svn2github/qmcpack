//////////////////////////////////////////////////////////////////
// (c) Copyright 2008-  by Ken Esler and Jeongnim Kim
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
#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"
#include "QMCWaveFunctions/Jastrow/BsplineJastrowBuilder.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "QMCWaveFunctions/Jastrow/OneBodyJastrowOrbital.h"
#include "QMCWaveFunctions/Jastrow/DiffOneBodyJastrowOrbital.h"
#include "QMCWaveFunctions/Jastrow/OneBodySpinJastrowOrbital.h"
#include "QMCWaveFunctions/Jastrow/DiffOneBodySpinJastrowOrbital.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrowOrbital.h"
#include "QMCWaveFunctions/Jastrow/DiffTwoBodyJastrowOrbital.h"
#ifdef QMC_CUDA
#include "QMCWaveFunctions/Jastrow/OneBodyJastrowOrbitalBspline.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrowOrbitalBspline.h"
#endif
#include "LongRange/LRRPAHandlerTemp.h"
#include "QMCWaveFunctions/Jastrow/LRBreakupUtilities.h"
#include "Utilities/ProgressReportEngine.h"
#include "LongRange/LRJastrowSingleton.h"

namespace qmcplusplus
{

template<typename OBJT, typename DOBJT>
bool BsplineJastrowBuilder::createOneBodyJastrow(xmlNodePtr cur)
{
  ReportEngine PRE(ClassName,"createOneBodyJastrow(xmlNodePtr)");
  string j1name("J1");
  {
    OhmmsAttributeSet a;
    a.add(j1name,"name");
    a.put(cur);
  }
  int taskid=(targetPsi.is_manager())?targetPsi.getGroupID():-1;
  OBJT* J1 =new OBJT(*sourcePtcl,targetPtcl);
  DOBJT *dJ1 = new DOBJT(*sourcePtcl, targetPtcl);
  xmlNodePtr kids = cur->xmlChildrenNode;
  // Find the number of the source species
  SpeciesSet &sSet = sourcePtcl->getSpeciesSet();
  SpeciesSet &tSet = targetPtcl.getSpeciesSet();
  int numSpecies = sSet.getTotalNum();
  bool success=false;
  bool Opt(false);
  while (kids != NULL)
  {
    std::string kidsname = (char*)kids->name;
    if (kidsname == "correlation")
    {
      RealType cusp=0.0;
      string speciesA;
      string speciesB;
      OhmmsAttributeSet rAttrib;
      rAttrib.add(speciesA,"elementType");
      rAttrib.add(speciesA,"speciesA");
      rAttrib.add(speciesB,"speciesB");
      rAttrib.add(cusp,"cusp");
      rAttrib.put(kids);
      BsplineFunctor<RealType> *functor = new BsplineFunctor<RealType>(cusp);
      functor->elementType = speciesA;
      int ig = sSet.findSpecies (speciesA);
      functor->cutoff_radius = sourcePtcl->Lattice.WignerSeitzRadius;
      int jg=-1;
      if(speciesB.size())
        jg=tSet.findSpecies(speciesB);
      if (ig < numSpecies)
      {
        //ignore
        functor->put (kids);
        if (functor->cutoff_radius < 1.0e-6)
        {
          app_log()  << "  BsplineFunction rcut is currently zero.\n"
                     << "  Setting to Wigner-Seitz radius = "
                     << sourcePtcl->Lattice.WignerSeitzRadius << endl;
          functor->cutoff_radius = sourcePtcl->Lattice.WignerSeitzRadius;
          functor->reset();
        }
        J1->addFunc (ig,functor,jg);
        success = true;
        dJ1->addFunc(ig,functor,jg);
        Opt=(!functor->notOpt or Opt);
        char fname[128];
        if(ReportLevel)
        {
          if(taskid > -1)
          {
            if(speciesB.size())
              sprintf(fname,"%s.%s%s.g%03d.dat",j1name.c_str(),speciesA.c_str(),speciesB.c_str(),taskid);
            else
              sprintf(fname,"%s.%s.g%03d.dat",j1name.c_str(),speciesA.c_str(),taskid);
          }
          else
            ReportLevel=0;
        }
        functor->setReportLevel(ReportLevel,fname);
        functor->print();
      }
    }
    kids = kids->next;
  }
  if(success)
  {
    J1->dPsi=dJ1;
    targetPsi.addOrbital(J1,"J1_bspline");
    J1->setOptimizable(Opt);
    return true;
  }
  else
  {
    PRE.warning("BsplineJastrowBuilder failed to add an One-Body Jastrow.");
    delete J1;
    delete dJ1;
    return false;
  }
}

/** class to initialize bsplin functions
 */
template<typename T>
struct BsplineInitializer
{

  vector<T> rpaValues;

  /** initialize with RPA
   * @param ref particleset
   * @param bfunc bspline function to be initialized
   * @param nopbc true, if not periodic
   */
  inline void initWithRPA(ParticleSet& P,  BsplineFunctor<T>& bfunc, T fac)
  {
    if(P.Lattice.SuperCellEnum==SUPERCELL_OPEN) // for open systems, do nothing
    {
      return;
      //T vol=std::pow(bfunc.cutoff_radius,3);
      //rpa.reset(ref.getTotalNum(),vol*0.5);
    }
    int npts=bfunc.NumParams;
    if(rpaValues.empty())
    {
      rpaValues.resize(npts);
      LRRPAHandlerTemp<RPABreakup<T>,LPQHIBasis> rpa(P,-1.0);
      rpa.Breakup(P,-1.0);
      T dr=bfunc.cutoff_radius/static_cast<T>(npts);
      T r=0;
      for (int i=0; i<npts; i++)
      {
        rpaValues[i]=rpa.evaluate(r,1.0/r); //y[i]=fac*rpa.evaluate(r,1.0/r);
        r += dr;
      }
    }
    T last=rpaValues[npts-1];
    //vector<T>  y(npts);
    //for (int i=0; i<npts; i++) y[i]=fac*rpaValues[i];
    //T last=y[npts-1];
    for(int i=0; i<npts; i++)
      bfunc.Parameters[i]=fac*(rpaValues[i]-last);
    bfunc.reset();
  }
};

bool BsplineJastrowBuilder::put(xmlNodePtr cur)
{
  ReportEngine PRE(ClassName,"put(xmlNodePtr)");
  bool PrintTables=false;
  typedef BsplineFunctor<RealType> RadFuncType;
  // Create a one-body Jastrow
  if (sourcePtcl)
  {
    string j1spin("no");
    OhmmsAttributeSet jAttrib;
    jAttrib.add(j1spin,"spin");
    jAttrib.put(cur);
#ifdef QMC_CUDA
    return createOneBodyJastrow<OneBodyJastrowOrbitalBspline,DiffOneBodySpinJastrowOrbital<RadFuncType> >(cur);
#else
    //if(sourcePtcl->IsGrouped)
    //{
    //  app_log() << "Creating OneBodySpinJastrowOrbital<T> " << endl;
    //  return createOneBodyJastrow<OneBodySpinJastrowOrbital<RadFuncType>,DiffOneBodySpinJastrowOrbital<RadFuncType> >(cur);
    //}
    //else
    //{
    //  app_log() << "Creating OneBodyJastrowOrbital<T> " << endl;
    //  return createOneBodyJastrow<OneBodyJastrowOrbital<RadFuncType>,DiffOneBodyJastrowOrbital<RadFuncType> >(cur);
    //}
    if(j1spin=="yes")
      return createOneBodyJastrow<OneBodySpinJastrowOrbital<RadFuncType>,DiffOneBodySpinJastrowOrbital<RadFuncType> >(cur);
    else
      return createOneBodyJastrow<OneBodyJastrowOrbital<RadFuncType>,DiffOneBodyJastrowOrbital<RadFuncType> >(cur);
#endif
  }
  else // Create a two-body Jastrow
  {
    string init_mode("0");
    {
      OhmmsAttributeSet hAttrib;
      hAttrib.add(init_mode,"init");
      hAttrib.put(cur);
    }
    BsplineInitializer<RealType> j2Initializer;
    xmlNodePtr kids = cur->xmlChildrenNode;
#ifdef QMC_CUDA
    typedef TwoBodyJastrowOrbitalBspline J2Type;
#else
    typedef TwoBodyJastrowOrbital<BsplineFunctor<RealType> > J2Type;
#endif
    typedef DiffTwoBodyJastrowOrbital<BsplineFunctor<RealType> > dJ2Type;
    int taskid=(targetPsi.is_manager())?targetPsi.getGroupID():-1;
    J2Type *J2 = new J2Type(targetPtcl,taskid);
    dJ2Type *dJ2 = new dJ2Type(targetPtcl);
    SpeciesSet& species(targetPtcl.getSpeciesSet());
    int chargeInd=species.addAttribute("charge");
    //std::map<std::string,RadFuncType*> functorMap;
    bool Opt(false);
    while (kids != NULL)
    {
      std::string kidsname((const char*)kids->name);
      if (kidsname == "correlation")
      {
        OhmmsAttributeSet rAttrib;
        RealType cusp=-1e10;
        string pairType("0");
        string spA(species.speciesName[0]);
        string spB(species.speciesName[0]);
        rAttrib.add(spA,"speciesA");
        rAttrib.add(spB,"speciesB");
        rAttrib.add(pairType,"pairType");
        rAttrib.add(cusp,"cusp");
        rAttrib.put(kids);
        if(pairType[0]=='0')
        {
          pairType=spA+spB;
        }
        else
        {
          PRE.warning("pairType is deprecated. Use speciesA/speciesB");
          //overwrite the species
          spA=pairType[0];
          spB=pairType[1];
        }
        int ia = species.findSpecies(spA);
        int ib = species.findSpecies(spB);
        if(ia==species.size() || ib == species.size())
        {
          PRE.error("Failed. Species are incorrect.",true);
        }
        if(cusp<-1e6)
        {
          RealType qq=species(chargeInd,ia)*species(chargeInd,ib);
          cusp = (ia==ib)? -0.25*qq:-0.5*qq;
        }
        app_log() << "  BsplineJastrowBuilder adds a functor with cusp = " << cusp << endl;
        RadFuncType *functor = new RadFuncType(cusp);
        functor->cutoff_radius = targetPtcl.Lattice.WignerSeitzRadius;
        bool initialized_p=functor->put(kids);
        functor->elementType=pairType;
        if (functor->cutoff_radius < 1.0e-6)
        {
          app_log()  << "  BsplineFunction rcut is currently zero.\n"
                     << "  Setting to Wigner-Seitz radius = "
                     << targetPtcl.Lattice.WignerSeitzRadius << endl;
          functor->cutoff_radius = targetPtcl.Lattice.WignerSeitzRadius;
          functor->reset();
        }
        //RPA INIT
        if(!initialized_p && init_mode =="rpa")
        {
          app_log() << "  Initializing Two-Body with RPA Jastrow " << endl;
          j2Initializer.initWithRPA(targetPtcl,*functor,-cusp/0.5);
        }
        J2->addFunc(ia,ib,functor);
        dJ2->addFunc(ia,ib,functor);
        Opt=(!functor->notOpt or Opt);
        char fname[32];
        if(ReportLevel)
        {
          if(taskid > -1)
            sprintf(fname,"J2.%s.g%03d.dat",pairType.c_str(),taskid);
          else
            ReportLevel=0;
        }
        functor->setReportLevel(ReportLevel,fname);
        functor->print();
      }
      kids = kids->next;
    }
    //dJ2->initialize();
    //J2->setDiffOrbital(dJ2);
    J2->dPsi=dJ2;
    targetPsi.addOrbital(J2,"J2_bspline");
    J2->setOptimizable(Opt);
  }
  return true;
}

}
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1691 $   $Date: 2007-02-01 15:51:50 -0600 (Thu, 01 Feb 2007) $
 * $Id: BsplineConstraints.h 1691 2007-02-01 21:51:50Z jnkim $
 ***************************************************************************/
