/////////////////////////////////////////////////////////////////
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
/**@file HamiltonianFactory.cpp
 *@brief Definition of a HamiltonianFactory 
 */
#include "QMCHamiltonians/HamiltonianFactory.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "QMCHamiltonians/IonIonPotential.h"
#include "QMCHamiltonians/NumericalRadialPotential.h"
#if OHMMS_DIM == 3
  #include "QMCHamiltonians/LocalCorePolPotential.h"
  #include "QMCHamiltonians/ECPotentialBuilder.h"
#endif
#if defined(HAVE_LIBFFTW_LS)
  #include "QMCHamiltonians/ModInsKineticEnergy.h"
  #include "QMCHamiltonians/MomentumDistribution.h"
  #include "QMCHamiltonians/DispersionRelation.h"
#endif
#include "QMCHamiltonians/CoulombPBCAATemp.h"
#include "QMCHamiltonians/CoulombPBCABTemp.h"
#include "QMCHamiltonians/Pressure.h"
#include "QMCHamiltonians/RPAPressure.h"
#include "QMCHamiltonians/PsiValue.h"
#include "QMCHamiltonians/DMCPsiValue.h"
#include "QMCHamiltonians/PsiOverlap.h"
#include "QMCHamiltonians/HePressure.h"
#include "QMCHamiltonians/HFDHE2Potential.h"
#include "QMCHamiltonians/HeEPotential.h"
#include "QMCHamiltonians/ForceBase.h"
// #include "QMCHamiltonians/ZeroVarObs.h"
#include "QMCHamiltonians/ForceCeperley.h"
#include "QMCHamiltonians/PulayForce.h"
#include "QMCHamiltonians/HFDHE2Potential_tail.h"
#include "QMCHamiltonians/HeEPotential_tail.h"
#include "QMCHamiltonians/HFDHE2_Moroni1995.h"
#include "QMCHamiltonians/HFDBHE_smoothed.h"
#include "QMCHamiltonians/HeSAPT_smoothed.h"
#include "QMCHamiltonians/ForwardWalking.h"
#include "QMCHamiltonians/trialDMCcorrection.h"
#include "QMCHamiltonians/ChiesaCorrection.h"
#include "QMCHamiltonians/PairCorrEstimator.h"
#include "QMCHamiltonians/DensityEstimator.h"
#include "QMCHamiltonians/SkEstimator.h"
#if defined(HAVE_LIBFFTW)
  #include "QMCHamiltonians/MPC.h"
#endif
#include "OhmmsData/AttributeSet.h"

namespace qmcplusplus {
  HamiltonianFactory::HamiltonianFactory(ParticleSet* qp, 
    PtclPoolType& pset, OrbitalPoolType& oset, Communicate* c): 
    MPIObjectBase(c),
    targetPtcl(qp), targetH(0), 
  ptclPool(pset),psiPool(oset), myNode(NULL), psiName("psi0") 
  {
    //PBCType is zero or 1 but should be generalized 
    PBCType=targetPtcl->Lattice.SuperCellEnum;
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
  bool HamiltonianFactory::build(xmlNodePtr cur, bool buildtree) {

    if(cur == NULL) return false;

    string htype("generic"), source("i"), defaultKE("yes");
    OhmmsAttributeSet hAttrib;
    hAttrib.add(htype,"type"); 
    hAttrib.add(source,"source");
    hAttrib.add(defaultKE,"default");
    hAttrib.put(cur);

    renameProperty(source);

    bool attach2Node=false;
    if(buildtree) {
      if(myNode == NULL) {
//#if (LIBXMLD_VERSION < 20616)
//        app_warning() << "   Workaround of libxml2 bug prior to 2.6.x versions" << endl;
//        myNode = xmlCopyNode(cur,2);
//#else
//        app_warning() << "   using libxml2 2.6.x versions" << endl;
//        myNode = xmlCopyNode(cur,1);
//#endif
        myNode = xmlCopyNode(cur,1);
      } else {
        attach2Node=true;
      }
    }

    if(targetH==0) {
      targetH  = new QMCHamiltonian;
      targetH->setName(myName);
      
      if(defaultKE == "yes"){
        targetH->addOperator(new BareKineticEnergy ,"Kinetic");
      } else if(defaultKE == "multi"){
        //Multicomponent wavefunction. Designed for 2 species.
	app_log()<<" Multicomponent system. You must add the Kinetic energy term first!"<<endl;
      } else  {
        double mass(1.0);
        string tgt("mass");
        int indx1 = targetPtcl->mySpecies.findSpecies(defaultKE);
        int indx2 = targetPtcl->mySpecies.addAttribute(tgt);
        mass = targetPtcl->mySpecies(indx2,indx1);
        cout<<"  Kinetic energy operator:: Mass "<<mass<<endl;
        targetH->addOperator(new BareKineticEnergy(mass),"Kinetic");
      }
    }

    xmlNodePtr cur2(cur);
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
          if(targetInp == targetPtcl->getName())
            addCoulombPotential(cur);
          else 
            addConstCoulombPotential(cur,sourceInp);
        } 
	else if (potType == "HeSAPT_smoothed") {
	  HeSAPT_smoothed* SAPT = new HeSAPT_smoothed(*targetPtcl);
	  targetH->addOperator(SAPT,"HeSAPT",true);
	  targetH->addOperator(SAPT->makeDependants(*targetPtcl),SAPT->depName,false);
	  app_log() << "  Adding " << SAPT->depName << endl;
	}
	else if (potType == "HFDHE2_Moroni1995") {
	  HFDHE2_Moroni1995_physical* HFD = new HFDHE2_Moroni1995_physical(*targetPtcl);
	  targetH->addOperator(HFD,"HFD-HE2",true);
	  targetH->addOperator(HFD->makeDependants(*targetPtcl),HFD->depName,false);
	  app_log() << "  Adding " << HFD->depName << endl;
	}
	else if (potType == "HFDBHE_smoothed") {
	  HFDBHE_smoothed* HFD = new HFDBHE_smoothed(*targetPtcl);
	  targetH->addOperator(HFD,"HFD-B(He)",true);
	  targetH->addOperator(HFD->makeDependants(*targetPtcl),HFD->depName,false);
	  app_log() << "  Adding " << HFD->depName << endl;
	}
	else if (potType == "MPC" || potType == "mpc")
	  addMPCPotential(cur);
        else if(potType == "HFDHE2") 
        {
          HFDHE2Potential* HFD = new HFDHE2Potential(*targetPtcl);
          targetH->addOperator(HFD,"HFDHE2",true);
          //HFD->addCorrection(*targetPtcl,*targetH);
          targetH->addOperator(HFD->makeDependants(*targetPtcl),HFD->depName,false);
          app_log() << "  Adding HFDHE2Potential(Au) " << endl;
        }
        else if(potType == "pseudo") 
        {
          addPseudoPotential(cur);
        } 
        else if(potType == "cpp") 
        {
          addCorePolPotential(cur);
        }
        else if(potType.find("num") < potType.size())
        {
          if(sourceInp == targetInp)//only accept the pair-potential for now
          {
            NumericalRadialPotential* apot=new NumericalRadialPotential(*targetPtcl);
            apot->put(cur);
            targetH->addOperator(apot,potName);
          }
        }
	else if(potType == "eHe")
	{
	  string SourceName = "e";
	  OhmmsAttributeSet hAttrib;
	  hAttrib.add(SourceName, "source");
	  hAttrib.put(cur);

	  PtclPoolType::iterator pit(ptclPool.find(SourceName));
	  if(pit == ptclPool.end()) 
	  {
            APP_ABORT("Unknown source \"" + SourceName + "\" for e-He Potential.");
	  }
	  ParticleSet* source = (*pit).second;
	  
	  HeePotential* eHetype = new HeePotential(*targetPtcl, *source);
	  targetH->addOperator(eHetype,potName,true);
// 	  targetH->addOperator(eHetype->makeDependants(*targetPtcl),potName,false);
	  
	}
      } 
      else if(cname == "constant") 
      { //ugly!!!
        if(potType == "coulomb")  addConstCoulombPotential(cur,sourceInp);
      } 
      else if(cname == "modInsKE") 
      {
        addModInsKE(cur);
      }
      else if(cname == "estimator")
      {
        if(potType == "Force")
        {
          addForceHam(cur);
        }
	else if(potType == "gofr")
        {
          PairCorrEstimator* apot=new PairCorrEstimator(*targetPtcl,sourceInp);
          apot->put(cur);
          targetH->addOperator(apot,potName,false);
        }
	else if(potType == "density")
        {
          if(PBCType)//only if perioidic 
          {
            DensityEstimator* apot=new DensityEstimator(*targetPtcl);
            apot->put(cur);
            targetH->addOperator(apot,potName,false);
          }
        }
	else if(potType == "sk")
        {
          if(PBCType)//only if perioidic 
          {
            SkEstimator* apot=new SkEstimator(*targetPtcl);
            apot->put(cur);
            targetH->addOperator(apot,potName,false);
          }
        }
	else if(potType == "chiesa")
	{
	  string PsiName="psi0";
	  string SourceName = "e";
	  OhmmsAttributeSet hAttrib;
	  hAttrib.add(PsiName,"psi"); 
	  hAttrib.add(SourceName, "source");
	  hAttrib.put(cur);

	  PtclPoolType::iterator pit(ptclPool.find(SourceName));
	  if(pit == ptclPool.end()) 
	  {
            APP_ABORT("Unknown source \""+SourceName+"\" for Chiesa correction.");
	  }
	  ParticleSet &source = *pit->second;

	  OrbitalPoolType::iterator psi_it(psiPool.find(PsiName));
	  if(psi_it == psiPool.end()) 
          {
            APP_ABORT("Unknown psi \""+PsiName+"\" for Chiesa correction.");
          }

	  const TrialWaveFunction &psi = *psi_it->second->targetPsi;
	  ChiesaCorrection *chiesa = new ChiesaCorrection (source, psi);
	  targetH->addOperator(chiesa,"KEcorr",false);
	}  
        else if(potType == "Pressure")
        {
          if(estType=="coulomb")
          {
            Pressure* BP = new Pressure(*targetPtcl);
            BP-> put(cur);
            targetH->addOperator(BP,"Pressure",false);

            int nlen(100);
            attrib.add(nlen,"truncateSum");
            attrib.put(cur);
            //             DMCPressureCorr* DMCP = new DMCPressureCorr(*targetPtcl,nlen);
            //             targetH->addOperator(DMCP,"PressureSum",false);

          } 
	  else if (estType=="HFDHE2")
	  {
            HePressure* BP = new HePressure(*targetPtcl);
            BP-> put(cur);
            targetH->addOperator(BP,"HePress",false);
          } 
	  else if (estType=="RPAZVZB")
	  {
            RPAPressure* BP= new RPAPressure(*targetPtcl);
            
            ParticleSet* Isource;
            bool withSource=false;
            xmlNodePtr tcur = cur->children;
            while(tcur != NULL) {
              string cname((const char*)tcur->name);
              if(cname == "OneBody") 
              {
                string PsiName="psi0";
                withSource=true;
//                 string in0("ion0");
                OhmmsAttributeSet hAttrib;
//                 hAttrib.add(in0,"source");
                hAttrib.add(PsiName,"psi"); 
                hAttrib.put(tcur);
//                 renameProperty(a);
                PtclPoolType::iterator pit(ptclPool.find(sourceInp));
                if(pit == ptclPool.end()) 
		{
                  ERRORMSG("Missing source ParticleSet" << sourceInp)
                }
                Isource = (*pit).second;
                BP-> put(cur, *targetPtcl,*Isource,*(psiPool[PsiName]->targetPsi));
              }
              tcur = tcur->next; 
            }
            if (!withSource) BP-> put(cur, *targetPtcl);
            targetH->addOperator(BP,BP->MyName,false);
            
            int nlen(100);
            attrib.add(nlen,"truncateSum");
            attrib.put(cur);
//             DMCPressureCorr* DMCP = new DMCPressureCorr(*targetPtcl,nlen);
//             targetH->addOperator(DMCP,"PressureSum",false);
          }
        }
	else if(potType=="psi")
	{
	  int pwr=2;
	  OhmmsAttributeSet hAttrib;
	  hAttrib.add(pwr,"power"); 
	  hAttrib.put(cur);
	  PsiValue* PV = new PsiValue(pwr);
	  PV->put(cur,targetPtcl,ptclPool,myComm);
	  targetH->addOperator(PV,"PsiValue",false);
	}
	else if(potType=="overlap")
	{
	  int pwr=1;
	  OhmmsAttributeSet hAttrib;
	  hAttrib.add(pwr,"power"); 
	  hAttrib.put(cur);
	  
	  PsiOverlapValue* PV = new PsiOverlapValue(pwr);
	  PV->put(cur,targetPtcl,ptclPool,myComm);
	  targetH->addOperator(PV,"PsiRatio",false);
	}
	else if(potType=="DMCoverlap")
	{
	  DMCPsiValue* PV = new DMCPsiValue( );
	  PV->put(cur,targetPtcl,ptclPool,myComm);
	  targetH->addOperator(PV,"DMCPsiRatio",false);
	}
	
	

//         else if (potType=="ForwardWalking"){
//           app_log()<<"  Adding Forward Walking Operator"<<endl;
//           ForwardWalking* FW=new ForwardWalking();
//           FW->put(cur,*targetH,*targetPtcl);
//           targetH->addOperator(FW,"ForwardWalking",false);
//           
//         }
      } 
      else if (cname == "Kinetic")
      {
      	  string TargetName="e";
	  string SourceName = "I";
	  OhmmsAttributeSet hAttrib;
	  hAttrib.add(TargetName,"Dependant"); 
	  hAttrib.add(SourceName, "Independant");
	  hAttrib.put(cur);
	  //hand two particle sets to the Hamiltonian. One is independant(heavy ions, not parameterized by electron positions), the other is dependant (parameterized by the others positions). This Hamiltonian element must also correct the Drift/Gradient and Laplacians in the Particlesets.
	  
	  
      }
      
      //else if(cname == "harmonic") 
      //{
      //  PtclPoolType::iterator pit(ptclPool.find(sourceInp));
      //  if(pit != ptclPool.end()) 
      //  {
      //    ParticleSet* ion=(*pit).second;
      //    targetH->addOperator(new HarmonicPotential(*ion, *targetPtcl),"Harmonic");
      //    app_log() << "  Adding HarmonicPotential " << endl;
      //  }
      //} 

      //const xmlChar* t = xmlGetProp(cur,(const xmlChar*)"type");
      //if(t != NULL) { // accept only if it has type
      //  string pot_type((const char*)t);
      //  string nuclei("i");

      //  const xmlChar* sptr = xmlGetProp(cur,(const xmlChar*)"source");
      //  if(sptr != NULL) nuclei=(const char*)sptr;
      //  renameProperty(nuclei);

      //  if(cname == "pairpot") {
      //    if(pot_type == "coulomb") {
      //      bool sameTarget=true;
      //      string aNewTarget(targetPtcl->getName());
      //      const xmlChar* aptr = xmlGetProp(cur,(const xmlChar*)"target");
      //      if(aptr != NULL) {
      //        aNewTarget=(const char*)aptr;
      //        renameProperty(aNewTarget);
      //        sameTarget= (aNewTarget == targetPtcl->getName());
      //      } 
      //      cout << "This is most likely problem " << aNewTarget << " " << targetPtcl->getName() << " " << targetPtcl->parent() << endl;
      //      if(sameTarget) 
      //        addCoulombPotential(cur);
      //      else {
      //        app_log() << "  Creating Coulomb potential " << nuclei << "-" << nuclei << endl;
      //        addConstCoulombPotential(cur,nuclei);
      //      }
      //    } else if(pot_type == "pseudo") {
      //      addPseudoPotential(cur);
      //    } else if(pot_type == "cpp") {
      //      addCorePolPotential(cur);
      //    }
      //  } 
      //  else if(cname == "harmonic") {
      //    PtclPoolType::iterator pit(ptclPool.find(nuclei));
      //    if(pit != ptclPool.end()) {
      //      ParticleSet* ion=(*pit).second;
      //      targetH->addOperator(new HarmonicPotential(*ion, *targetPtcl),"Harmonic");
      //      app_log() << "  Adding HarmonicPotential " << endl;
      //    }
      //  } else if(cname == "constant") { 
      //    if(pot_type == "coulomb") { //ugly!!!
      //      addConstCoulombPotential(cur,nuclei);
      //    }
      //  } else if(cname == "modInsKE") {
      //    addModInsKE(cur);
      //  }
      //}
      if(attach2Node) xmlAddChild(myNode,xmlCopyNode(cur,1));
      cur = cur->next;
    }

    //ATTENTION FORWARD WALKING IS BROKEN 
    //targetH->setTempObservables(targetPtcl->PropertyList);
    targetH->addObservables(*targetPtcl);
    
    ///This is officially ugly, but we need to add all observables (previous line) 
    ///before the forward walker is initialized otherwise we can't find them.
    cur2 = cur2->children;
    bool FoundET(false);
    while(cur2 != NULL) {
      string cname((const char*)cur2->name);
      string potType("Null");
      OhmmsAttributeSet attrib;
      attrib.add(potType,"type");
      attrib.put(cur2);
      if((cname == "estimator")&&(potType=="ZeroVarObs"))
      {
        app_log()<<"  Adding ZeroVarObs Operator"<<endl;
//         ZeroVarObs* FW=new ZeroVarObs();
//         FW->put(cur2,*targetH,*targetPtcl);
//         targetH->addOperator(FW,"ZeroVarObs",false);
      }
      else if((cname == "estimator")&&(potType=="ForwardWalking"))
      {
        app_log()<<"  Adding Forward Walking Operator"<<endl;
        ForwardWalking* FW=new ForwardWalking();
        FW->put(cur2,*targetH,*targetPtcl);
        targetH->addOperator(FW,"ForwardWalking",false);
      }
      else if((cname == "estimator")&&(potType == "DMCCorrection"))
      {
	TrialDMCCorrection* TE = new TrialDMCCorrection();
	TE->put(cur2,*targetH,*targetPtcl);
	targetH->addOperator(TE,"DMC_CORR",false);
	targetH->setTempObservables(targetPtcl->PropertyList);
      }
      cur2 = cur2->next;
    }
    //targetH->addObservables(targetPtcl->PropertyList);
    targetH->addObservables(*targetPtcl);
    return true;
  }

  void
  HamiltonianFactory::addMPCPotential(xmlNodePtr cur) 
  {
#if defined(HAVE_LIBFFTW)
    string a("e"), title("MPC");
    OhmmsAttributeSet hAttrib;
    bool physical = true;
    double cutoff = 30.0;
    hAttrib.add(title,"id"); 
    hAttrib.add(title,"name"); 
    hAttrib.add(cutoff,"cutoff");
    hAttrib.add(physical,"physical");
    hAttrib.put(cur);

    renameProperty(a);

    MPC *mpc = new MPC (*targetPtcl, cutoff);
    targetH->addOperator(mpc, "MPC", physical);
#else
    APP_ABORT("HamiltonianFactory::addMPCPotential MPC is disabled because FFTW3 was not found during the build process.");
#endif // defined(HAVE_LIBFFTW)
  }

  void 
  HamiltonianFactory::addCoulombPotential(xmlNodePtr cur) {

    string a("e"),title("ElecElec"),pbc("yes");
    bool physical = true;
    OhmmsAttributeSet hAttrib;
    hAttrib.add(title,"id"); hAttrib.add(title,"name"); 
    hAttrib.add(a,"source"); 
    hAttrib.add(pbc,"pbc"); 
    hAttrib.add(physical,"physical");
    hAttrib.put(cur);
    

    renameProperty(a);

    PtclPoolType::iterator pit(ptclPool.find(a));
    if(pit == ptclPool.end()) {
      ERRORMSG("Missing source ParticleSet" << a)
      return;
    }

    ParticleSet* source = (*pit).second;

    bool applyPBC= (PBCType && pbc=="yes");

    //CHECK PBC and create CoulombPBC for el-el
    if(source == targetPtcl) {
      if(source->getTotalNum()>1)  {
        if(applyPBC) {
          //targetH->addOperator(new CoulombPBCAA(*targetPtcl),title);
          targetH->addOperator(new CoulombPBCAATemp(*targetPtcl,true),
			       title,physical);
        } else {
          targetH->addOperator(new CoulombPotentialAA(*targetPtcl),
			       title,physical);
        }
      }
    } else {
      if(applyPBC) {
        //targetH->addOperator(new CoulombPBCAB(*source,*targetPtcl),title);
        targetH->addOperator(new CoulombPBCABTemp(*source,*targetPtcl),title);
      } else {
        targetH->addOperator(new CoulombPotentialAB(*source,*targetPtcl),title);
      }
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
  HamiltonianFactory::addForceHam(xmlNodePtr cur) {
    string a("ion0"),targetName("e"),title("ForceBase"),pbc("yes"),
      PsiName="psi0";
    OhmmsAttributeSet hAttrib;
    string mode("bare");
    //hAttrib.add(title,"id");
    //hAttrib.add(title,"name"); 
    hAttrib.add(a,"source"); 
    hAttrib.add(targetName,"target"); 
    hAttrib.add(pbc,"pbc"); 
    hAttrib.add(mode,"mode"); 
    hAttrib.add(PsiName, "psi");
    hAttrib.put(cur);
    cerr << "HamFac forceBase mode " << mode << endl;
    renameProperty(a);

    PtclPoolType::iterator pit(ptclPool.find(a));
    if(pit == ptclPool.end()) {
      ERRORMSG("Missing source ParticleSet" << a)
      return;
    }
    ParticleSet* source = (*pit).second;
    pit = ptclPool.find(targetName);
    if(pit == ptclPool.end()) {
      ERRORMSG("Missing target ParticleSet" << targetName)
      return;
    }
    ParticleSet* target = (*pit).second;

    //bool applyPBC= (PBCType && pbc=="yes");

    if(mode=="bare") 
      targetH->addOperator(new BareForce(*source, *target), title, false);
    else if(mode=="cep") 
      targetH->addOperator(new ForceCeperley(*source, *target), title, false);
    else if(mode=="pulay") {
      OrbitalPoolType::iterator psi_it(psiPool.find(PsiName));
      if(psi_it == psiPool.end()) {
	APP_ABORT("Unknown psi \""+PsiName+"\" for Chiesa correction.");
      }
      TrialWaveFunction &psi = *psi_it->second->targetPsi;
      targetH->addOperator(new PulayForce(*source, *target, psi), title, false);
    }
    else {
      ERRORMSG("Failed to recognize Force mode " << mode);
      //} else if(mode=="FD") {
      //  targetH->addOperator(new ForceFiniteDiff(*source, *target), title, false);
    }
  }

  void 
  HamiltonianFactory::addPseudoPotential(xmlNodePtr cur) {

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
    if(pit == ptclPool.end()) {
      ERRORMSG("Missing source ParticleSet" << src)
      return;
    }

    ParticleSet* ion=(*pit).second;

    OrbitalPoolType::iterator oit(psiPool.find(wfname));
    TrialWaveFunction* psi=0;
    if(oit == psiPool.end()) {
      if(psiPool.empty()) return;
      app_error() << "  Cannot find " << wfname << " in the Wavefunction pool. Using the first wavefunction."<< endl;
      psi=(*(psiPool.begin())).second->targetPsi;
    } else {
      psi=(*oit).second->targetPsi;
    }

    //remember the TrialWaveFunction used by this pseudopotential
    psiName=wfname;

    //if(format == "old") {
    //  app_log() << "  Using OLD NonLocalPseudopotential "<< endl;
    //  targetH->addOperator(new NonLocalPPotential(*ion,*targetPtcl,*psi), title);
    //}
    //else  {
    app_log() << endl << "  ECPotential builder for pseudopotential "<< endl;
    ECPotentialBuilder ecp(*targetH,*ion,*targetPtcl,*psi,myComm);
    ecp.put(cur);
#endif
    //}
  }

  void 
  HamiltonianFactory::addCorePolPotential(xmlNodePtr cur) {
#if OHMMS_DIM == 3
    string src("i"),title("CorePol");

    OhmmsAttributeSet pAttrib;
    pAttrib.add(title,"name");
    pAttrib.add(src,"source");
    pAttrib.put(cur);

    PtclPoolType::iterator pit(ptclPool.find(src));
    if(pit == ptclPool.end()) {
      ERRORMSG("Missing source ParticleSet" << src)
      return;
    }
    ParticleSet* ion=(*pit).second;

    QMCHamiltonianBase* cpp=(new LocalCorePolPotential(*ion,*targetPtcl));
    cpp->put(cur); 
    targetH->addOperator(cpp, title);
#endif
  }

  void 
  HamiltonianFactory::addConstCoulombPotential(xmlNodePtr cur, string& nuclei){
    OhmmsAttributeSet hAttrib;
    string forces("no");
    hAttrib.add(forces,"forces");
    hAttrib.put(cur);
    bool doForces = (forces == "yes") || (forces == "true");

    app_log() << "  Creating Coulomb potential " << nuclei << "-" << nuclei << endl;
    renameProperty(nuclei);
    PtclPoolType::iterator pit(ptclPool.find(nuclei));
    if(pit != ptclPool.end()) {
      ParticleSet* ion=(*pit).second;
      if(ion->getTotalNum()>1) 
        if(PBCType){
          //targetH->addOperator(new CoulombPBCAA(*ion),"IonIon");
          targetH->addOperator(new CoulombPBCAATemp(*ion,false,doForces),"IonIon");
        } else {
          targetH->addOperator(new IonIonPotential(*ion),"IonIon");
        }
     }
  }

  void
  HamiltonianFactory::addModInsKE(xmlNodePtr cur) {
#if defined(HAVE_LIBFFTW_LS)
    typedef QMCTraits::RealType    RealType;
    typedef QMCTraits::IndexType   IndexType;
    typedef QMCTraits::PosType     PosType;
    
    string Dimensions, DispRelType, PtclSelType, MomDistType;
    RealType Cutoff, GapSize(0.0), FermiMomentum(0.0);
    
    OhmmsAttributeSet pAttrib;
    pAttrib.add(Dimensions, "dims");
    pAttrib.add(DispRelType, "dispersion");
    pAttrib.add(PtclSelType, "selectParticle");
    pAttrib.add(Cutoff, "cutoff");
    pAttrib.add(GapSize, "gapSize");
    pAttrib.add(FermiMomentum, "kf");
    pAttrib.add(MomDistType, "momdisttype");
    pAttrib.put(cur);
    
    if (MomDistType == "") MomDistType = "FFT";
    
    TrialWaveFunction* psi;
    psi = (*(psiPool.begin())).second->targetPsi;
    
    Vector<PosType> LocLattice;
    Vector<IndexType> DimSizes;
    Vector<RealType> Dispersion;

    if (Dimensions == "3") {
      gen3DLattice(Cutoff, *targetPtcl, LocLattice, Dispersion, DimSizes);
    } else if (Dimensions == "1" || Dimensions == "1averaged") {
      gen1DLattice(Cutoff, *targetPtcl, LocLattice, Dispersion, DimSizes);
    } else if (Dimensions == "homogeneous") {
      genDegenLattice(Cutoff, *targetPtcl, LocLattice, Dispersion, DimSizes);
    } else {
      ERRORMSG("Dimensions value not recognized!")
    }
    
    if (DispRelType == "freeParticle") {
      genFreeParticleDispersion(LocLattice, Dispersion);
    } else if (DispRelType == "simpleModel") {
      genSimpleModelDispersion(LocLattice, Dispersion, GapSize, FermiMomentum);
    } else if (DispRelType == "pennModel") {
      genPennModelDispersion(LocLattice, Dispersion, GapSize, FermiMomentum);
    } else if (DispRelType == "debug") {
      genDebugDispersion(LocLattice, Dispersion);  
    } else {
      ERRORMSG("Dispersion relation not recognized");
    }
    
    PtclChoiceBase* pcp;
    if (PtclSelType == "random") {
      pcp = new RandomChoice(*targetPtcl);
    } else if (PtclSelType == "randomPerWalker") {
      pcp = new RandomChoicePerWalker(*targetPtcl);
    } else if (PtclSelType == "constant") {
      pcp = new StaticChoice(*targetPtcl);
    } else {
      ERRORMSG("Particle choice policy not recognized!");
    }
    
    MomDistBase* mdp;
    if (MomDistType == "direct") {
      mdp = new RandomMomDist(*targetPtcl, LocLattice, pcp);
    } else if (MomDistType == "FFT" || MomDistType =="fft") { 
      if (Dimensions == "3") {
	mdp = new ThreeDimMomDist(*targetPtcl, DimSizes, pcp);
      } else if (Dimensions == "1") {
	mdp = new OneDimMomDist(*targetPtcl, DimSizes, pcp);
      } else if (Dimensions == "1averaged") {
	mdp = new AveragedOneDimMomDist(*targetPtcl, DimSizes, pcp);
      } else {
	ERRORMSG("Dimensions value not recognized!");
      }
    } else {
      ERRORMSG("MomDistType value not recognized!");
    }
    delete pcp;
    
    QMCHamiltonianBase* modInsKE = new ModInsKineticEnergy(*psi, Dispersion, mdp);
    modInsKE->put(cur);
    targetH->addOperator(modInsKE, "ModelInsKE");
    
    delete mdp;
#else
    app_error() << "  ModelInsulatorKE cannot be used without FFTW " << endl;
#endif
  }
  
  void HamiltonianFactory::renameProperty(const string& a, const string& b){
    RenamedProperty[a]=b;
  }

  void HamiltonianFactory::setCloneSize(int np) {
    myClones.resize(np,0);
  }

  //TrialWaveFunction*
  //HamiltonianFactory::cloneWaveFunction(ParticleSet* qp, int ip) {
  //  HamiltonianFactory* aCopy= new HamiltonianFactory(qp,ptclPool);
  //  aCopy->put(myNode,false);
  //  myClones[ip]=aCopy;
  //  return aCopy->targetPsi;
  //}

  void HamiltonianFactory::renameProperty(string& aname) {
    map<string,string>::iterator it(RenamedProperty.find(aname));
    if(it != RenamedProperty.end()) {
      aname=(*it).second;
    }
  }
  HamiltonianFactory::~HamiltonianFactory() {
    //clean up everything
  }

  HamiltonianFactory*
  HamiltonianFactory::clone(ParticleSet* qp, TrialWaveFunction* psi, 
      int ip, const string& aname) {
    HamiltonianFactory* aCopy=new HamiltonianFactory(qp, ptclPool, psiPool, myComm);
    aCopy->setName(aname);

    aCopy->renameProperty("e",qp->getName());
    aCopy->renameProperty(psiName,psi->getName());
    aCopy->build(myNode,false);
    myClones[ip]=aCopy;
    aCopy->get(app_log());
    return aCopy;
  }

  bool HamiltonianFactory::get(std::ostream& os) const {
    targetH->get(os);
    return true;
  }

  bool HamiltonianFactory::put(std::istream& ) {
    return true;
  }

  bool HamiltonianFactory::put(xmlNodePtr cur) {
    return build(cur,true);
  }

  void HamiltonianFactory::reset() { }
}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
