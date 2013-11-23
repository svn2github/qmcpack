/////////////////////////////////////////////////////////////////
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
/**@file HamiltonianFactory.cpp
 *@brief Definition of a HamiltonianFactory
 */
#include "QMCHamiltonians/HamiltonianFactory.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/ConservedEnergy.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "QMCHamiltonians/CoulombPBCAATemp.h"
#include "QMCHamiltonians/CoulombPBCABTemp.h"
#if OHMMS_DIM == 3
#include "QMCHamiltonians/LocalCorePolPotential.h"
#include "QMCHamiltonians/ECPotentialBuilder.h"
#include "QMCHamiltonians/ChiesaCorrection.h"
#if defined(HAVE_LIBFFTW)
#include "QMCHamiltonians/MPC.h"
#include "QMCHamiltonians/VHXC.h"
#endif
#endif

#include "OhmmsData/AttributeSet.h"

#ifdef QMC_CUDA
#include "QMCHamiltonians/CoulombPBCAA_CUDA.h"
#include "QMCHamiltonians/CoulombPBCAB_CUDA.h"
#include "QMCHamiltonians/CoulombPotential_CUDA.h"
#include "QMCHamiltonians/MPC_CUDA.h"
#endif

//#include <iostream>
namespace qmcplusplus
{
HamiltonianFactory::HamiltonianFactory(ParticleSet* qp,
                                       PtclPoolType& pset, OrbitalPoolType& oset, Communicate* c)
  : MPIObjectBase(c), targetPtcl(qp), targetH(0)
  , ptclPool(pset),psiPool(oset), myNode(NULL), psiName("psi0")
{
  //PBCType is zero or 1 but should be generalized
  PBCType=targetPtcl->Lattice.SuperCellEnum;
  ClassName="HamiltonianFactory";
  myName="psi0";
}

/** main hamiltonian build function
 * @param cur element node <hamiltonian/>
 * @param buildtree if true, build xml tree for a reuse
 *
 * A valid hamiltonian node contains
 * \xmlonly
 *  <hamiltonian target="e">
 *    <pairpot type="coulomb" name="ElecElec" source="e"/>
 *    <pairpot type="coulomb" name="IonElec" source="i"/>
 *    <pairpot type="coulomb" name="IonIon" source="i" target="i"/>
 *  </hamiltonian>
 * \endxmlonly
 */
bool HamiltonianFactory::build(xmlNodePtr cur, bool buildtree)
{
  if(cur == NULL)
    return false;
  string htype("generic"), source("i"), defaultKE("yes");
  OhmmsAttributeSet hAttrib;
  hAttrib.add(htype,"type");
  hAttrib.add(source,"source");
  hAttrib.add(defaultKE,"default");
  hAttrib.put(cur);
  renameProperty(source);
  bool attach2Node=false;
  if(buildtree)
  {
    if(myNode == NULL)
    {
//#if (LIBXMLD_VERSION < 20616)
//        app_warning() << "   Workaround of libxml2 bug prior to 2.6.x versions" << endl;
//        myNode = xmlCopyNode(cur,2);
//#else
//        app_warning() << "   using libxml2 2.6.x versions" << endl;
//        myNode = xmlCopyNode(cur,1);
//#endif
      myNode = xmlCopyNode(cur,1);
    }
    else
    {
      attach2Node=true;
    }
  }
  if(targetH==0)
  {
    targetH  = new QMCHamiltonian;
    targetH->setName(myName);
    targetH->addOperator(new BareKineticEnergy<double>(*targetPtcl),"Kinetic");
  }
  xmlNodePtr cur_saved(cur);
  cur = cur->children;
  while(cur != NULL)
  {
    string cname((const char*)cur->name);
    string potType("0");
    string potName("any");
    string potUnit("hartree");
    string estType("coulomb");
    string sourceInp(targetPtcl->getName());
    string targetInp(targetPtcl->getName());
    OhmmsAttributeSet attrib;
    attrib.add(sourceInp,"source");
    attrib.add(sourceInp,"sources");
    attrib.add(targetInp,"target");
    attrib.add(potType,"type");
    attrib.add(potName,"name");
    attrib.add(potUnit,"units");
    attrib.add(estType,"potential");
    attrib.put(cur);
    renameProperty(sourceInp);
    renameProperty(targetInp);
    if(cname == "pairpot")
    {
      if(potType == "coulomb")
      {
        addCoulombPotential(cur);
      }
#if OHMMS_DIM==3
      /*
      else if (potType == "HFDBHE_smoothed") {
        HFDBHE_smoothed_phy* HFD = new HFDBHE_smoothed_phy(*targetPtcl);
        targetH->addOperator(HFD,"HFD-B(He)",true);
        HFD->addCorrection(*targetH);
      }
      */
      else if (potType == "MPC" || potType == "mpc")
        addMPCPotential(cur);
      else if (potType == "VHXC" || potType == "vhxc")
        addVHXCPotential(cur);
      else if(potType == "pseudo")
      {
        addPseudoPotential(cur);
      }
#endif
    }
    else if(cname == "constant")
    {
      //just to support old input
      if(potType == "coulomb")
        addCoulombPotential(cur);
    }
    else if (cname == "Kinetic")
    {
      string TargetName="e";
      string SourceName = "I";
      OhmmsAttributeSet hAttrib;
      hAttrib.add(TargetName,"Dependant");
      hAttrib.add(SourceName, "Independant");
      hAttrib.put(cur);
    }
    if(attach2Node)
      xmlAddChild(myNode,xmlCopyNode(cur,1));
    cur = cur->next;
  }
  //add observables with physical and simple estimators
  int howmany=targetH->addObservables(*targetPtcl);
  //do correction
  bool dmc_correction=false;
  //evaluate the observables again
  if(dmc_correction)
    howmany=targetH->addObservables(*targetPtcl);
  return true;
}

void
HamiltonianFactory::addMPCPotential(xmlNodePtr cur, bool isphysical)
{
#if OHMMS_DIM ==3 && defined(HAVE_LIBFFTW)
  string a("e"), title("MPC"), physical("no");
  OhmmsAttributeSet hAttrib;
  double cutoff = 30.0;
  hAttrib.add(title,"id");
  hAttrib.add(title,"name");
  hAttrib.add(cutoff,"cutoff");
  hAttrib.add(physical,"physical");
  hAttrib.put(cur);
  renameProperty(a);
  isphysical = (physical=="yes" || physical == "true");
#ifdef QMC_CUDA
  MPC_CUDA *mpc = new MPC_CUDA (*targetPtcl, cutoff);
#else
  MPC *mpc = new MPC (*targetPtcl, cutoff);
#endif
  targetH->addOperator(mpc, "MPC", isphysical);
#else
  APP_ABORT("HamiltonianFactory::addMPCPotential MPC is disabled because FFTW3 was not found during the build process.");
#endif // defined(HAVE_LIBFFTW)
}

void
HamiltonianFactory::addVHXCPotential(xmlNodePtr cur)
{
#if OHMMS_DIM==3 && defined(HAVE_LIBFFTW)
  string a("e"), title("VHXC");
  OhmmsAttributeSet hAttrib;
  bool physical = true;
  hAttrib.add(title,"id");
  hAttrib.add(title,"name");
  hAttrib.add(physical,"physical");
  hAttrib.put(cur);
  renameProperty(a);
  VHXC *vhxc = new VHXC (*targetPtcl);
  app_log() << "physical = " << physical << endl;
  targetH->addOperator(vhxc, "VHXC", physical);
#else
  APP_ABORT("HamiltonianFactory::addVHXCPotential VHXC is disabled because FFTW3 was not found during the build process.");
#endif // defined(HAVE_LIBFFTW)
}



void
HamiltonianFactory::addCoulombPotential(xmlNodePtr cur)
{
  string targetInp(targetPtcl->getName());
  string sourceInp(targetPtcl->getName());
  string title("ElecElec"),pbc("yes");
  string forces("no");
  bool physical = true;
  bool doForce = false;
  OhmmsAttributeSet hAttrib;
  hAttrib.add(title,"id");
  hAttrib.add(title,"name");
  hAttrib.add(targetInp,"target");
  hAttrib.add(sourceInp,"source");
  hAttrib.add(pbc,"pbc");
  hAttrib.add(physical,"physical");
  hAttrib.add(forces,"forces");
  hAttrib.put(cur);
  bool applyPBC= (PBCType && pbc=="yes");
  bool doForces = (forces == "yes") || (forces == "true");
  ParticleSet *ptclA=targetPtcl;
  if(sourceInp != targetPtcl->getName())
  {
    //renameProperty(sourceInp);
    PtclPoolType::iterator pit(ptclPool.find(sourceInp));
    if(pit == ptclPool.end())
    {
      ERRORMSG("Missing source ParticleSet" << sourceInp);
      APP_ABORT("HamiltonianFactory::addCoulombPotential");
      return;
    }
    ptclA = (*pit).second;
  }
  if(sourceInp == targetInp) // AA type
  {
    if(ptclA->getTotalNum() == 1)
    {
      app_log() << "  CoulombAA for " << sourceInp << " is not created.  Number of particles == 1" << endl;
      return;
    }
    bool quantum = (sourceInp==targetPtcl->getName());
#ifdef QMC_CUDA
    if(applyPBC)
      targetH->addOperator(new CoulombPBCAA_CUDA(*ptclA,quantum,doForces),title,physical);
    else
    {
      if(quantum)
        targetH->addOperator(new CoulombPotentialAA_CUDA(ptclA,true), title, physical);
      else
        targetH->addOperator(new CoulombPotential<double>(ptclA,0,quantum), title, physical);
    }
#else
    if(applyPBC)
      targetH->addOperator(new CoulombPBCAATemp(*ptclA,quantum,doForces),title,physical);
    else
      targetH->addOperator(new CoulombPotential<double>(ptclA,0,quantum), title, physical);
#endif
  }
  else //X-e type, for X=some other source
  {
#ifdef QMC_CUDA
    if(applyPBC)
      targetH->addOperator(new CoulombPBCAB_CUDA(*ptclA,*targetPtcl),title);
    else
      targetH->addOperator(new CoulombPotentialAB_CUDA(ptclA,targetPtcl),title);
#else
    if(applyPBC)
      targetH->addOperator(new CoulombPBCABTemp(*ptclA,*targetPtcl),title);
    else
      targetH->addOperator(new CoulombPotential<double>(ptclA,targetPtcl,true),title);
#endif
  }
}

// void
// HamiltonianFactory::addPulayForce (xmlNodePtr cur) {
//   string a("ion0"),targetName("e"),title("Pulay");
//   OhmmsAttributeSet hAttrib;
//   hAttrib.add(a,"source");
//   hAttrib.add(targetName,"target");

//   PtclPoolType::iterator pit(ptclPool.find(a));
//   if(pit == ptclPool.end()) {
//     ERRORMSG("Missing source ParticleSet" << a)
//     return;
//   }

//   ParticleSet* source = (*pit).second;
//   pit = ptclPool.find(targetName);
//   if(pit == ptclPool.end()) {
//     ERRORMSG("Missing target ParticleSet" << targetName)
//     return;
//   }
//   ParticleSet* target = (*pit).second;

//   targetH->addOperator(new PulayForce(*source, *target), title, false);

// }

void
HamiltonianFactory::addForceHam(xmlNodePtr cur)
{
}

void
HamiltonianFactory::addPseudoPotential(xmlNodePtr cur)
{
#if OHMMS_DIM == 3
  string src("i"),title("PseudoPot"),wfname("invalid"),format("xml");
  OhmmsAttributeSet pAttrib;
  pAttrib.add(title,"name");
  pAttrib.add(src,"source");
  pAttrib.add(wfname,"wavefunction");
  pAttrib.add(format,"format"); //temperary tag to switch between format
  pAttrib.put(cur);
  if(format == "old")
  {
    APP_ABORT("pseudopotential Table format is not supported.");
  }
  renameProperty(src);
  renameProperty(wfname);
  PtclPoolType::iterator pit(ptclPool.find(src));
  if(pit == ptclPool.end())
  {
    ERRORMSG("Missing source ParticleSet" << src)
    return;
  }
  ParticleSet* ion=(*pit).second;
  OrbitalPoolType::iterator oit(psiPool.find(wfname));
  TrialWaveFunction* psi=0;
  if(oit == psiPool.end())
  {
    if(psiPool.empty())
      return;
    app_error() << "  Cannot find " << wfname << " in the Wavefunction pool. Using the first wavefunction."<< endl;
    psi=(*(psiPool.begin())).second->targetPsi;
  }
  else
  {
    psi=(*oit).second->targetPsi;
  }
  //remember the TrialWaveFunction used by this pseudopotential
  psiName=wfname;
  app_log() << endl << "  ECPotential builder for pseudopotential "<< endl;
  ECPotentialBuilder ecp(*targetH,*ion,*targetPtcl,*psi,myComm);
  ecp.put(cur);
#else
  APP_ABORT("HamiltonianFactory::addPseudoPotential\n pairpot@type=\"pseudo\" is invalid if DIM != 3");
#endif
}

void
HamiltonianFactory::addCorePolPotential(xmlNodePtr cur)
{
#if OHMMS_DIM == 3
  string src("i"),title("CorePol");
  OhmmsAttributeSet pAttrib;
  pAttrib.add(title,"name");
  pAttrib.add(src,"source");
  pAttrib.put(cur);
  PtclPoolType::iterator pit(ptclPool.find(src));
  if(pit == ptclPool.end())
  {
    ERRORMSG("Missing source ParticleSet" << src)
    return;
  }
  ParticleSet* ion=(*pit).second;
  QMCHamiltonianBase* cpp=(new LocalCorePolPotential(*ion,*targetPtcl));
  cpp->put(cur);
  targetH->addOperator(cpp, title);
#else
  APP_ABORT("HamiltonianFactory::addCorePolPotential\n pairpot@type=\"cpp\" is invalid if DIM != 3");
#endif
}

//  void
//  HamiltonianFactory::addConstCoulombPotential(xmlNodePtr cur, string& nuclei)
//  {
//    OhmmsAttributeSet hAttrib;
//    string hname("IonIon");
//    string forces("no");
//    hAttrib.add(forces,"forces");
//    hAttrib.add(hname,"name");
//    hAttrib.put(cur);
//    bool doForces = (forces == "yes") || (forces == "true");
//
//    app_log() << "  Creating Coulomb potential " << nuclei << "-" << nuclei << endl;
//    renameProperty(nuclei);
//    PtclPoolType::iterator pit(ptclPool.find(nuclei));
//    if(pit != ptclPool.end()) {
//      ParticleSet* ion=(*pit).second;
//      if(PBCType)
//      {
//#ifdef QMC_CUDA
//	targetH->addOperator(new CoulombPBCAA_CUDA(*ion,false,doForces),hname);
//#else
//	targetH->addOperator(new CoulombPBCAATemp(*ion,false,doForces),hname);
//#endif
//      } else {
//        if(ion->getTotalNum()>1)
//          targetH->addOperator(new CoulombPotential<double>(ion),hname);
//          //targetH->addOperator(new IonIonPotential(*ion),hname);
//      }
//    }
//  }

void
HamiltonianFactory::addModInsKE(xmlNodePtr cur)
{
}

void HamiltonianFactory::renameProperty(const string& a, const string& b)
{
  RenamedProperty[a]=b;
}

void HamiltonianFactory::setCloneSize(int np)
{
  myClones.resize(np,0);
}

//TrialWaveFunction*
//HamiltonianFactory::cloneWaveFunction(ParticleSet* qp, int ip) {
//  HamiltonianFactory* aCopy= new HamiltonianFactory(qp,ptclPool);
//  aCopy->put(myNode,false);
//  myClones[ip]=aCopy;
//  return aCopy->targetPsi;
//}

void HamiltonianFactory::renameProperty(string& aname)
{
  map<string,string>::iterator it(RenamedProperty.find(aname));
  if(it != RenamedProperty.end())
  {
    aname=(*it).second;
  }
}
HamiltonianFactory::~HamiltonianFactory()
{
  //clean up everything
}

HamiltonianFactory*
HamiltonianFactory::clone(ParticleSet* qp, TrialWaveFunction* psi,
                          int ip, const string& aname)
{
  HamiltonianFactory* aCopy=new HamiltonianFactory(qp, ptclPool, psiPool, myComm);
  aCopy->setName(aname);
  aCopy->renameProperty("e",qp->getName());
  aCopy->renameProperty(psiName,psi->getName());
  aCopy->build(myNode,false);
  myClones[ip]=aCopy;
  //aCopy->get(app_log());
  return aCopy;
}

bool HamiltonianFactory::put(xmlNodePtr cur)
{
  return build(cur,true);
}

void HamiltonianFactory::reset() { }
}
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 5820 $   $Date: 2013-05-03 15:58:44 -0400 (Fri, 03 May 2013) $
 * $Id: HamiltonianFactory.cpp 5820 2013-05-03 19:58:44Z jnkim $
 ***************************************************************************/
