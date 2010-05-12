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
#include "QMCWaveFunctions/BasisSetFactory.h"
#include "QMCWaveFunctions/Fermion/SlaterDetBuilder.h"
#include "Utilities/ProgressReportEngine.h"
#include "OhmmsData/AttributeSet.h"

#include "QMCWaveFunctions/MultiSlaterDeterminant.h"
//this is only for Bryan
#if defined(BRYAN_MULTIDET_TRIAL)
#include "QMCWaveFunctions/Fermion/DiracDeterminantIterative.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminantTruncation.h"
#include "QMCWaveFunctions/Fermion/MultiDiracDeterminantBase.h"
#endif
//Cannot use complex and released node
#if !defined(QMC_COMPLEX)
#include "QMCWaveFunctions/Fermion/RNDiracDeterminantBase.h"
#include "QMCWaveFunctions/Fermion/RNDiracDeterminantBaseAlternate.h"
#endif
#ifdef QMC_CUDA
  #include "QMCWaveFunctions/Fermion/DiracDeterminantCUDA.h"
#endif
#if QMC_BUILD_LEVEL>2 && OHMMS_DIM==3
#include "QMCWaveFunctions/Fermion/SlaterDetWithBackflow.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminantWithBackflow.h"
#endif
#include<vector>
//#include "QMCWaveFunctions/Fermion/ci_node.h"
#include "QMCWaveFunctions/Fermion/ci_configuration.h"
#include "QMCWaveFunctions/Fermion/SPOSetProxy.h"
#include "QMCWaveFunctions/Fermion/SPOSetProxyForMSD.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminantOpt.h"


namespace qmcplusplus
{

  SlaterDetBuilder::SlaterDetBuilder(ParticleSet& els, TrialWaveFunction& psi,
      PtclPoolType& psets)
    : OrbitalBuilderBase(els,psi), ptclPool(psets)
      , myBasisSetFactory(0), slaterdet_0(0), multislaterdet_0(0)
  {
    ClassName="SlaterDetBuilder";
#if QMC_BUILD_LEVEL>2 && OHMMS_DIM==3
    BFTrans=0;
    UseBackflow=false;
#endif
  }
  SlaterDetBuilder::~SlaterDetBuilder()
  {
    DEBUG_MEMORY("SlaterDetBuilder::~SlaterDetBuilder");
    if (myBasisSetFactory)
      {
        delete myBasisSetFactory;
      }
  }


  /** process <determinantset>
   *
   * determinantset 
   * - basiset 0..1
   * - sposet 0..* 
   * - slaterdeterminant
   *   - determinant 0..*
   * - ci
   */
  bool SlaterDetBuilder::put(xmlNodePtr cur)
  {

    ReportEngine PRE(ClassName,"put(xmlNodePtr)");

    ///save the current node
    xmlNodePtr curRoot=cur;
    xmlNodePtr BFnode;
    bool success=true;
    string cname, tname;
    std::map<string,SPOSetBasePtr> spomap;
    bool multiDet=false;

    //check the basis set
    cur = curRoot->children;
    while (cur != NULL)//check the basis set
    {
      getNodeName(cname,cur);
      if (cname == basisset_tag)
      {
        if (myBasisSetFactory == 0)
        {
          myBasisSetFactory = new BasisSetFactory(targetPtcl,targetPsi, ptclPool);
          myBasisSetFactory->setReportLevel(ReportLevel);
        }
        myBasisSetFactory->createBasisSet(cur,curRoot);
      }
      else if (cname == sposet_tag) {
	app_log() << "Creating SPOSet in SlaterDetBuilder::put(xmlNodePtr cur).\n";
	string spo_name;
	OhmmsAttributeSet spoAttrib;
	spoAttrib.add (spo_name, "name");
	spoAttrib.put(cur);
	app_log() << "spo_name = " << spo_name << endl;

	if (myBasisSetFactory == 0)
        {
          myBasisSetFactory = new BasisSetFactory(targetPtcl,targetPsi, ptclPool);
          myBasisSetFactory->setReportLevel(ReportLevel);
          myBasisSetFactory->createBasisSet(cur,cur);
        }
        //myBasisSetFactory->createBasisSet(cur,cur);
	SPOSetBasePtr spo = myBasisSetFactory->createSPOSet(cur);
	spo->put(cur, spomap);
	if (spomap.find(spo_name) != spomap.end()) {
	  app_error() << "SPOSet name \"" << spo_name << "\" is already in use.\n";
	  abort();
	}
	spomap[spo_name] = spo;
	assert(spomap.find(spo_name) != spomap.end());
	//	slaterdet_0->add(spo,spo_name);
      }
#if QMC_BUILD_LEVEL>2 && OHMMS_DIM==3 
      else if(cname == backflow_tag) {
        app_log() <<"Creating Backflow transformation in SlaterDetBuilder::put(xmlNodePtr cur).\n";

        // to simplify the logic inside DiracDeterminantWithBackflow,
        // I'm requiring that only a single <backflow> block appears
        // in the xml file
        if(BFTrans != 0) {
           APP_ABORT("Only a single backflow block is allowed in the xml. Please collect all transformations into a single block. \n");
        }
        UseBackflow=true;
        // creating later due to problems with ParticleSets
        //BFTrans = new BackflowTransformation(targetPtcl,ptclPool);
        BFTrans = NULL;
        BFnode=cur;
// read xml later, in case some ParticleSets are read from hdf5 file.
        //BFTrans->put(cur);  
      }
#endif
      cur = cur->next;
    }

    //missing basiset, e.g. einspline
    if (myBasisSetFactory == 0)
    {
      myBasisSetFactory = new BasisSetFactory(targetPtcl,targetPsi, ptclPool);
      myBasisSetFactory->setReportLevel(ReportLevel);
      myBasisSetFactory->createBasisSet(curRoot,curRoot);
    }

    //add sposet
    
    cur = curRoot->children;
    while (cur != NULL)//check the basis set
    {
      getNodeName(cname,cur);
      if (cname == sd_tag)
      {
        multiDet=false;
        if(slaterdet_0)
        {
          APP_ABORT("slaterdet is already instantiated.");
        }
#if QMC_BUILD_LEVEL>2 && OHMMS_DIM==3
        if(UseBackflow) 
          slaterdet_0 = new SlaterDetWithBackflow(targetPtcl,BFTrans);
        else 
#endif
          slaterdet_0 = new SlaterDeterminant_t(targetPtcl);
	// Copy any entries in sposetmap into slaterdet_0
	std::map<string,SPOSetBasePtr>::iterator iter;
	for (iter=spomap.begin(); iter!=spomap.end(); iter++) {
	  cerr << "Adding SPO \"" << iter->first << "\" to slaterdet_0.\n";
	  slaterdet_0->add(iter->second,iter->first);
	}


        int spin_group = 0;
        xmlNodePtr tcur = cur->children;
        while (tcur != NULL)
        {
          getNodeName(tname,tcur);
          if (tname == det_tag || tname == rn_tag)
          {
            if(putDeterminant(tcur, spin_group)) spin_group++;
          }
          tcur = tcur->next;
        }
      } else if(cname == multisd_tag) {
        multiDet=true;
        if(slaterdet_0)
        {
          APP_ABORT("can't combine slaterdet with multideterminant.");
        }
        if(multislaterdet_0)
        {
          APP_ABORT("multideterminant is already instantiated.");
        }

#if QMC_BUILD_LEVEL>2 && OHMMS_DIM==3
        if(UseBackflow) {
           APP_ABORT("Backflow is not implemented with multi determinants.");
        }
#endif

        string spo_alpha;
        string spo_beta;
        OhmmsAttributeSet spoAttrib;
        spoAttrib.add (spo_alpha, "spo_up");
        spoAttrib.add (spo_beta, "spo_dn");
        spoAttrib.put(cur);
        if(spo_alpha == spo_beta) {
          app_error() << "In SlaterDetBuilder: In MultiSlaterDeterminant construction, SPO sets must be different. spo_up: " <<spo_alpha <<"  spo_dn: " <<spo_beta <<"\n";
          abort();
        }    
        if (spomap.find(spo_alpha) == spomap.end()) {
          app_error() << "In SlaterDetBuilder: SPOSet \"" << spo_alpha << "\" is not found. Expected for MultiSlaterDeterminant.\n";
          abort();
        }
        if (spomap.find(spo_beta) == spomap.end()) {
          app_error() << "In SlaterDetBuilder: SPOSet \"" << spo_beta << "\" is not found. Expected for MultiSlaterDeterminant.\n";
          abort();
        }

        bool usetree=false;
        if(usetree) {
          SPOSetProxy* spo_up;
          SPOSetProxy* spo_dn;
          spo_up=new SPOSetProxy(spomap.find(spo_alpha)->second,targetPtcl.first(0),targetPtcl.last(0));
          spo_dn=new SPOSetProxy(spomap.find(spo_beta)->second,targetPtcl.first(1),targetPtcl.last(1));

          DiracDeterminantBase* up_det=0;
          DiracDeterminantBase* dn_det=0;
          app_log() <<"Creating base determinant (up) for MSD expansion. \n";
          up_det = new DiracDeterminantBase((SPOSetBasePtr) spo_up,0);
          up_det->set(targetPtcl.first(0),targetPtcl.last(0)-targetPtcl.first(0));
          app_log() <<"Creating base determinant (down) for MSD expansion. \n";
          dn_det = new DiracDeterminantBase((SPOSetBasePtr) spo_dn,1);
          dn_det->set(targetPtcl.first(1),targetPtcl.last(1)-targetPtcl.first(1));
          APP_ABORT("MultiSlaterDeterminant implementation with excitation tree is incomplete. \n");
          //multislaterdet_0 = new MultiSlaterDeterminantWithTree(targetPtcl,spo_up,spo_dn,up_det,dn_det);
          //success = createCINODE(multislaterdet_0,cur);
        } else {
          app_log() <<"Using a list of dirac determinants for MultiSlaterDeterminant expansion. \n";
          SPOSetProxyForMSD* spo_up;
          SPOSetProxyForMSD* spo_dn;
          spo_up=new SPOSetProxyForMSD(spomap.find(spo_alpha)->second,targetPtcl.first(0),targetPtcl.last(0));
          spo_dn=new SPOSetProxyForMSD(spomap.find(spo_beta)->second,targetPtcl.first(1),targetPtcl.last(1));

          multislaterdet_0 = new MultiSlaterDeterminant(targetPtcl,spo_up,spo_dn);
          success = createMSD(multislaterdet_0,cur);
        }
      }
      cur = cur->next;
    }

    if (!multiDet && !slaterdet_0)
    {
      //fatal
      PRE.error("Failed to create a SlaterDeterminant.",true);
      return false;
    }
    if(multiDet && !multislaterdet_0)
    {
      //fatal
      PRE.error("Failed to create a MultiSlaterDeterminant.",true);
      return false;
    }

    // change DistanceTables if using backflow
#if QMC_BUILD_LEVEL>2 && OHMMS_DIM==3
    if(UseBackflow)   { 
       BFTrans = new BackflowTransformation(targetPtcl,ptclPool);
  // HACK HACK HACK, until I figure out a solution      
       SlaterDetWithBackflow* tmp = (SlaterDetWithBackflow*) slaterdet_0;
       tmp->BFTrans = BFTrans;
       for(int i=0; i<tmp->Dets.size(); i++) {
         DiracDeterminantWithBackflow* tmpD = (DiracDeterminantWithBackflow*) tmp->Dets[i]; 
         tmpD->BFTrans = BFTrans;
       }
       BFTrans->put(BFnode);
       tmp->resetTargetParticleSet(BFTrans->QP);
    }
#endif
    //only single slater determinant
    if(multiDet) 
      targetPsi.addOrbital(multislaterdet_0,"MultiSlaterDeterminant");
    else
      targetPsi.addOrbital(slaterdet_0,"SlaterDet");

    delete myBasisSetFactory;
    myBasisSetFactory=0;

    return success;
  }


  bool SlaterDetBuilder::putDeterminant(xmlNodePtr cur, int spin_group)
  {

    ReportEngine PRE(ClassName,"putDeterminant(xmlNodePtr,int)");

    string basisName("invalid");
    string detname("0"), refname("0");
    string s_detSize("0");
    string detMethod("");
    OhmmsAttributeSet aAttrib;
    aAttrib.add(basisName,basisset_tag);
    aAttrib.add(detname,"id");
    aAttrib.add(refname,"ref");
    aAttrib.add(detMethod,"DetMethod");
    aAttrib.add(s_detSize,"DetSize");

    string s_cutoff("0.0");
    string s_radius("0.0");
    int s_smallnumber(-999999);
    int rntype(0);
    aAttrib.add(s_cutoff,"Cutoff");
    aAttrib.add(s_radius,"Radius");
    aAttrib.add(s_smallnumber,"smallnumber");
    aAttrib.add(s_smallnumber,"eps");
    aAttrib.add(rntype,"primary");
    aAttrib.put(cur);


    map<string,SPOSetBasePtr>& spo_ref(slaterdet_0->mySPOSet);
    map<string,SPOSetBasePtr>::iterator lit(spo_ref.find(detname));
    SPOSetBasePtr psi;
    if (lit == spo_ref.end())
    {
      // cerr << "Didn't find sposet named \"" << detname << "\"\n";
#if defined(ENABLE_SMARTPOINTER)
      psi.reset(myBasisSetFactory->createSPOSet(cur));
#else
      psi = myBasisSetFactory->createSPOSet(cur);
#endif
      psi->put(cur);
      psi->checkObject();
      slaterdet_0->add(psi,detname);
      //SPOSet[detname]=psi;
      app_log() << "  Creating a new SPO set " << detname << endl;
    }
    else
    {
      app_log() << "  Reusing a SPO set " << detname << endl;
      psi = (*lit).second;
    }

    int firstIndex=targetPtcl.first(spin_group);
    int lastIndex=targetPtcl.last(spin_group);
    if(firstIndex==lastIndex) return true;

//    app_log() << "  Creating DiracDeterminant " << detname << " group=" << spin_group << " First Index = " << firstIndex << endl;
//    app_log() <<"   My det method is "<<detMethod<<endl;
//#if defined(BRYAN_MULTIDET_TRIAL)
//    if (detMethod=="Iterative")
//    {
//      //   string s_cutoff("0.0");
//      //   aAttrib.add(s_cutoff,"Cutoff");
//      app_log()<<"My cutoff is "<<s_cutoff<<endl;
//
//      double cutoff=std::atof(s_cutoff.c_str());
//      DiracDeterminantIterative *adet= new DiracDeterminantIterative(psi,firstIndex);
//      adet->set_iterative(firstIndex,psi->getOrbitalSetSize(),cutoff);
//      slaterdet_0->add(adet,spin_group);
//    }
//    else if (detMethod=="Truncation")
//    {
//      //   string s_cutoff("0.0");
//      //   aAttrib.add(s_cutoff,"Cutoff");
//      DiracDeterminantTruncation *adet= new DiracDeterminantTruncation(psi,firstIndex);
//      double cutoff=std::atof(s_cutoff.c_str());
//      double radius=std::atof(s_radius.c_str());
//      //   adet->set(firstIndex,psi->getOrbitalSetSize());
//      adet->set_truncation(firstIndex,psi->getOrbitalSetSize(),cutoff,radius);
//      slaterdet_0->add(adet,spin_group);
//    }
//    else if (detMethod=="Multi")
//    {
//      app_log()<<"BUILDING DIRAC DETERM "<<firstIndex<<endl;
//      MultiDiracDeterminantBase *adet = new MultiDiracDeterminantBase(psi,firstIndex);
//      int detSize=std::atof(s_detSize.c_str());
//      adet-> set_Multi(firstIndex,detSize,psi->getOrbitalSetSize());
//      slaterdet_0->add(adet,spin_group);
//    }
//    else
//      slaterdet_0->add(new Det_t(psi,firstIndex),spin_group);
//    }
//#else
    string dname;
    getNodeName(dname,cur);
    DiracDeterminantBase* adet=0;
#if !defined(QMC_COMPLEX)
    if (rn_tag == dname)
    {
      double bosonicEpsilon=s_smallnumber;
      app_log()<<"  BUILDING Released Node Determinant logepsilon="<<bosonicEpsilon<<endl;
      if (rntype==0)
        adet = new RNDiracDeterminantBase(psi,firstIndex);
      else
        adet = new RNDiracDeterminantBaseAlternate(psi,firstIndex);
      adet->setLogEpsilon(bosonicEpsilon);  
    }
    else
#endif
    {
#ifdef QMC_CUDA
      adet = new DiracDeterminantCUDA(psi,firstIndex);
#else
#if QMC_BUILD_LEVEL>2 && OHMMS_DIM==3
      if(UseBackflow) 
        adet = new DiracDeterminantWithBackflow(psi,BFTrans,firstIndex);
      else 
#endif
	if (psi->Optimizable)
	  adet = new DiracDeterminantOpt(targetPtcl, psi, firstIndex);
	else
	  adet = new DiracDeterminantBase(psi,firstIndex);
#endif
    }
    adet->set(firstIndex,lastIndex-firstIndex);
    slaterdet_0->add(adet,spin_group);
    if (psi->Optimizable)
      slaterdet_0->Optimizable = true;
    return true;
  }

  bool SlaterDetBuilder::createCINODE(MultiSlaterDeterminant* multiSD, xmlNodePtr cur)
  {
     bool success=true;

/*********************************
    1. read configurations and coefficients from xml
    2. get unique set of determinants
    3. create excitation tree for both spin channels
    4. build mapping from original expansion to location in the tree
*********************************/

     vector<configuration> confgList_up, uniqueConfg_up;    
     vector<configuration> confgList_dn, uniqueConfg_dn;    
     configuration baseC_up;
     configuration baseC_dn;
     vector<RealType>& coeff = multiSD->C;


     xmlNodePtr curRoot=cur,DetListNode;
     string cname;
     cur = curRoot->children;
     while (cur != NULL)//check the basis set
     {
       getNodeName(cname,cur);
       if(cname == "detlist")
       {
         DetListNode=cur;
         app_log() <<"Found determinant list. \n";
       }
       cur = cur->next;
     }

     int NCA,NCB,NEA,NEB,nstates,ndets=0,count=0;
     string Dettype="DETS";
     OhmmsAttributeSet spoAttrib;
     spoAttrib.add (NCA, "nca");
     spoAttrib.add (NCB, "ncb");
     spoAttrib.add (NEA, "nea");
     spoAttrib.add (NEB, "neb");
     spoAttrib.add (ndets, "size");
     spoAttrib.add (Dettype, "type");
     spoAttrib.add (nstates, "nstates");
     spoAttrib.put(DetListNode);

/*
app_log() <<NCA <<"  "
          <<NCB <<"  "
          <<NEA <<"  "
          <<NEB <<"  "
          <<ndets <<"  "
          <<nstates <<"  "
          <<multiSD->nels_up <<"  "
          <<multiSD->nels_dn <<"  "
          <<multiSD->spo_up->refPhi->getOrbitalSetSize() <<"  "
          <<multiSD->spo_up->refPhi->getBasisSetSize() <<"  "
          <<multiSD->spo_dn->refPhi->getOrbitalSetSize() <<"  "
          <<multiSD->spo_dn->refPhi->getBasisSetSize() <<"  "
          <<Dettype <<endl;
*/

     if(ndets==0) { 
       APP_ABORT("size==0 in detlist is not allowed. Use slaterdeterminant in this case.\n");
     }
      
     if(Dettype != "DETS" && Dettype != "Determinants") { 
       APP_ABORT("Only allowed type in detlist is DETS. CSF not implemented yet.\n");
     }

     if(multiSD->nels_up != (NCA+NEA)) {
       APP_ABORT("Number of up electrons in ParticleSet doesn't agree with NCA+NEA in detlist.");
     }
     if(multiSD->nels_dn != (NCB+NEB)) {
       APP_ABORT("Number of down electrons in ParticleSet doesn't agree with NCB+NEB in detlist.");
     }
     if(multiSD->spo_up->refPhi->getOrbitalSetSize() < NCA+nstates) {
       APP_ABORT("Number of states in SPOSet is smaller than NCA+nstates in detlist.");
     }
     if(multiSD->spo_dn->refPhi->getOrbitalSetSize() < NCB+nstates) {
       APP_ABORT("Number of states in SPOSet is smaller than NCB+nstates in detlist.");
     }

     cur = DetListNode->children;
     configuration dummyC_alpha;
     configuration dummyC_beta;
     dummyC_alpha.taken=false;
     dummyC_alpha.nExct=0;
     dummyC_alpha.occup.resize(NCA+nstates,false);
     for(int i=0; i<NCA+NEA; i++) dummyC_alpha.occup[i]=true;
     dummyC_beta.taken=false;
     dummyC_beta.nExct=0;
     dummyC_beta.occup.resize(NCB+nstates,false);
     for(int i=0; i<NCB+NEB; i++) dummyC_beta.occup[i]=true;
     bool foundHF=false;

     // insert HF at the beggining
     confgList_up.push_back(dummyC_alpha);
     confgList_up.back().nExct=0;
     confgList_up.back().taken=true;;
     confgList_dn.push_back(dummyC_beta);
     confgList_dn.back().nExct=0;
     confgList_dn.back().taken=true;;
     coeff.push_back(1.0);

     while (cur != NULL)//check the basis set
     {
       getNodeName(cname,cur);
       if(cname == "configuration" || cname == "ci")
       {
         RealType ci=0.0;
         string alpha,beta;
         OhmmsAttributeSet confAttrib;
         confAttrib.add(ci,"coeff");
         confAttrib.add(alpha,"alpha");
         confAttrib.add(beta,"beta");
         confAttrib.put(cur);

         int nq=0,na,nr;
         if(alpha.size() < NCA+nstates)
         {
           cerr<<"alpha: " <<alpha <<endl;
           APP_ABORT("Found incorrect alpha determinant label. size < nca+nstates");
         }

         for(int i=0; i<NCA+nstates; i++)
         {
           if(alpha[i] != '0' && alpha[i] != '1') {
             cerr<<alpha <<endl;
             APP_ABORT("Found incorrect determinant label.");
           }
           if(alpha[i] == '1') nq++;
         } 
         if(nq != NCA+NEA) {
             cerr<<"alpha: " <<alpha <<endl;
             APP_ABORT("Found incorrect alpha determinant label. noccup != nca+nea");
         }

         nq=0; 
         if(beta.size() < NCB+nstates)
         {
           cerr<<"beta: " <<beta <<endl;
           APP_ABORT("Found incorrect beta determinant label. size < ncb+nstates");
         }
         for(int i=0; i<NCB+nstates; i++)
         {
           if(beta[i] != '0' && beta[i] != '1') {
             cerr<<beta <<endl;
             APP_ABORT("Found incorrect determinant label.");
           }
           if(beta[i] == '1') nq++;
         } 
         if(nq != NCB+NEB) {
             cerr<<"beta: " <<beta <<endl;
             APP_ABORT("Found incorrect beta determinant label. noccup != ncb+neb");
         }
 
         //app_log() <<"Found determinant configuration: \n"
         //          <<"alpha: " <<alpha <<endl 
         //          <<"beta: " <<beta <<endl 
         //          <<"c: " <<ci <<endl;
         bool isHF=true;
         for(int i=0; i<NCB+NEB; i++)
           if(beta[i] == '0' ) { isHF=false;  break; }
         for(int i=0; i<NCA+NEA; i++)
           if(alpha[i] == '0' ) { isHF=false;  break; }
         
         if(isHF) {
           foundHF=true;
           count++;
           coeff[0]=ci; // first element is reserved for base determinant
         } else {
           count++;
           coeff.push_back(ci);
           confgList_up.push_back(dummyC_alpha);
           na=0;
           nr=0;
           for(int i=0; i<NCA; i++) confgList_up.back().occup[i]=true;
           for(int i=NCA; i<NCA+nstates; i++) {
             confgList_up.back().occup[i]= (alpha[i]=='1');
             if(confgList_up.back().occup[i]^dummyC_alpha.occup[i]) {
               if(dummyC_alpha.occup[i])
                 nr++;
               else
                 na++;
             }
           } 
           if(nr != na) {
             cerr<<alpha <<endl;
             APP_ABORT("Found incorrect determinant label.");
           }  
           confgList_up.back().nExct=na;    
         
           confgList_dn.push_back(dummyC_beta);
           na=0;
           nr=0;
           for(int i=0; i<NCB; i++) confgList_dn.back().occup[i]=true;
           for(int i=NCB; i<NCB+nstates; i++) {
             confgList_dn.back().occup[i]=(beta[i]=='1');
              if(confgList_dn.back().occup[i]^dummyC_beta.occup[i]) {
               if(dummyC_beta.occup[i])
                 nr++;
               else
                 na++;
             }
           } 
           if(nr != na) {
             cerr<<beta <<endl;
             APP_ABORT("Found incorrect determinant label.");
           }
           confgList_dn.back().nExct=na;
         } // not HF
       }
       cur = cur->next;
     }

     if(!foundHF) {
       APP_ABORT("Problems with determinant configurations. HF state must be in the list. \n");
     }
     if(count != ndets) {
       cerr<<"count, ndets: " <<count <<"  " <<ndets <<endl;
       APP_ABORT("Problems reading determinant configurations. Found a number of determinants inconsistent with xml file size parameter.\n");
     }

     if(confgList_up.size() != ndets || confgList_dn.size() != ndets || coeff.size() != ndets) {
       APP_ABORT("Problems reading determinant configurations.");
     }

     multiSD->C2node_up.resize(coeff.size());
     multiSD->C2node_dn.resize(coeff.size());

     app_log() <<"Found " <<coeff.size() <<" terms in the MSD expansion.\n";
  
     for(int i=0; i<confgList_up.size(); i++)
     {
       bool found=false;
       int k=-1;
       for(int j=0; j<uniqueConfg_up.size(); j++)
       {
         if(confgList_up[i] == uniqueConfg_up[j]) {
           found=true;
           k=j;
           break;
         } 
       }  
       if(found) {
         multiSD->C2node_up[i]=k;
       } else {
         uniqueConfg_up.push_back(confgList_up[i]);
         multiSD->C2node_up[i]=uniqueConfg_up.size()-1;
       }
     }   
     for(int i=0; i<confgList_dn.size(); i++)
     {
       bool found=false;
       int k=-1;
       for(int j=0; j<uniqueConfg_dn.size(); j++)
       {
         if(confgList_dn[i] == uniqueConfg_dn[j]) {
           found=true;
           k=j;
           break;
         } 
       }
       if(found) {
         multiSD->C2node_dn[i]=k;
       } else {
         uniqueConfg_dn.push_back(confgList_dn[i]);
         multiSD->C2node_dn[i]=uniqueConfg_dn.size()-1;
       }
     }

     app_log() <<"Found " <<uniqueConfg_up.size() <<" unique up determinants.\n";
     app_log() <<"Found " <<uniqueConfg_dn.size() <<" unique down determinants.\n";
/*
     std::vector<int> map2det_up(uniqueConfg_up.size());
     std::vector<int> map2det_dn(uniqueConfg_dn.size());

     multiSD->detRatios_up.resize(uniqueConfg_up.size());
     multiSD->detRatios_dn.resize(uniqueConfg_dn.size());
     multiSD->resize();

     multiSD->tree_up->build_tree(uniqueConfg_up,dummyC_alpha,map2det_up);
     multiSD->tree_dn->build_tree(uniqueConfg_dn,dummyC_beta,map2det_dn);

     for(int i=0; i<multiSD->C2node_up.size(); i++)
     {
        int t = map2det_up[multiSD->C2node_up[i]];
        multiSD->C2node_up[i]=t;   
     } 
     for(int i=0; i<multiSD->C2node_dn.size(); i++)
     {
        int t = map2det_dn[multiSD->C2node_dn[i]];
        multiSD->C2node_dn[i]=t;
     }

     std::cerr.flush();
     std::cout.flush();
     targetPtcl.update();
     multiSD->test(targetPtcl); 
*/
     return success;
  }

    bool SlaterDetBuilder::createMSD(MultiSlaterDeterminant* multiSD, xmlNodePtr cur)
  {
     bool success=true;

/*********************************
    1. read configurations and coefficients from xml
    2. get unique set of determinants
    3. create excitation tree for both spin channels
    4. build mapping from original expansion to location in the tree
*********************************/

     vector<configuration> confgList_up, uniqueConfg_up;
     vector<configuration> confgList_dn, uniqueConfg_dn;
     configuration baseC_up;
     configuration baseC_dn;
     vector<RealType>& coeff = multiSD->C;

     string optCI="no";
     OhmmsAttributeSet ciAttrib;
     ciAttrib.add (optCI,"optimize");
     ciAttrib.add (optCI,"Optimize");
     ciAttrib.put(cur);

     bool optimizeCI = optCI=="yes"; 

     xmlNodePtr curRoot=cur,DetListNode;
     string cname;
     cur = curRoot->children;
     while (cur != NULL)//check the basis set
     {
       getNodeName(cname,cur);
       if(cname == "detlist")
       {
         DetListNode=cur;
         app_log() <<"Found determinant list. \n";
       }
       cur = cur->next;
     }

     int NCA,NCB,NEA,NEB,nstates,ndets=0,count=0;
     string Dettype="DETS";
     OhmmsAttributeSet spoAttrib;
     spoAttrib.add (NCA, "nca");
     spoAttrib.add (NCB, "ncb");
     spoAttrib.add (NEA, "nea");
     spoAttrib.add (NEB, "neb");
     spoAttrib.add (ndets, "size");
     spoAttrib.add (Dettype, "type");
     spoAttrib.add (nstates, "nstates");
     spoAttrib.put(DetListNode);

     if(ndets==0) {
       APP_ABORT("size==0 in detlist is not allowed. Use slaterdeterminant in this case.\n");
     }

     if(Dettype != "DETS" && Dettype != "Determinants") {
       APP_ABORT("Only allowed type in detlist is DETS. CSF not implemented yet.\n");
     }
    
// cheating until I fix the converter
     NCA = multiSD->nels_up-NEA;
     NCB = multiSD->nels_dn-NEB;

     if(multiSD->nels_up != (NCA+NEA)) {
       APP_ABORT("Number of up electrons in ParticleSet doesn't agree with NCA+NEA in detlist.");
     }
     if(multiSD->nels_dn != (NCB+NEB)) {
       APP_ABORT("Number of down electrons in ParticleSet doesn't agree with NCB+NEB in detlist.");
     }
     if(multiSD->spo_up->refPhi->getOrbitalSetSize() < NCA+nstates) {
       APP_ABORT("Number of states in SPOSet is smaller than NCA+nstates in detlist.");
     }
     if(multiSD->spo_dn->refPhi->getOrbitalSetSize() < NCB+nstates) {
       APP_ABORT("Number of states in SPOSet is smaller than NCB+nstates in detlist.");
     }

     cur = DetListNode->children;
     configuration dummyC_alpha;
     configuration dummyC_beta;
     dummyC_alpha.occup.resize(NCA+nstates,false);
     for(int i=0; i<NCA+NEA; i++) dummyC_alpha.occup[i]=true;
     dummyC_beta.occup.resize(NCB+nstates,false);
     for(int i=0; i<NCB+NEB; i++) dummyC_beta.occup[i]=true;

     app_log() <<"alpha reference: \n" <<dummyC_alpha;
     app_log() <<"beta reference: \n" <<dummyC_beta;

     while (cur != NULL)//check the basis set
     {
       getNodeName(cname,cur);
       if(cname == "configuration" || cname == "ci")
       {
         RealType ci=0.0;
         string alpha,beta;
         OhmmsAttributeSet confAttrib;
         confAttrib.add(ci,"coeff");
         confAttrib.add(alpha,"alpha");
         confAttrib.add(beta,"beta");
         confAttrib.put(cur);

         int nq=0,na,nr;
         if(alpha.size() < nstates)
         {
           cerr<<"alpha: " <<alpha <<endl;
           APP_ABORT("Found incorrect alpha determinant label. size < nca+nstates");
         }

         for(int i=0; i<nstates; i++)
         {
           if(alpha[i] != '0' && alpha[i] != '1') {
             cerr<<alpha <<endl;
             APP_ABORT("Found incorrect determinant label.");
           }
           if(alpha[i] == '1') nq++;
         }
         if(nq != NEA) {
             cerr<<"alpha: " <<alpha <<endl;
             APP_ABORT("Found incorrect alpha determinant label. noccup != nca+nea");
         }

         nq=0;
         if(beta.size() < nstates)
         {
           cerr<<"beta: " <<beta <<endl;
           APP_ABORT("Found incorrect beta determinant label. size < ncb+nstates");
         }
         for(int i=0; i<nstates; i++)
         {
           if(beta[i] != '0' && beta[i] != '1') {
             cerr<<beta <<endl;
             APP_ABORT("Found incorrect determinant label.");
           }
           if(beta[i] == '1') nq++;
         }
         if(nq != NEB) {
             cerr<<"beta: " <<beta <<endl;
             APP_ABORT("Found incorrect beta determinant label. noccup != ncb+neb");
         }

         count++;
         coeff.push_back(ci);
         confgList_up.push_back(dummyC_alpha);
         for(int i=0; i<NCA; i++) confgList_up.back().occup[i]=true;
         for(int i=NCA; i<NCA+nstates; i++) 
           confgList_up.back().occup[i]= (alpha[i-NCA]=='1');
         confgList_dn.push_back(dummyC_beta);
         for(int i=0; i<NCB; i++) confgList_dn.back().occup[i]=true;
         for(int i=NCB; i<NCB+nstates; i++) 
           confgList_dn.back().occup[i]=(beta[i-NCB]=='1');
       }
       cur = cur->next;
     }

     if(count != ndets) {
       cerr<<"count, ndets: " <<count <<"  " <<ndets <<endl;
       APP_ABORT("Problems reading determinant configurations. Found a number of determinants inconsistent with xml file size parameter.\n");
     }
     if(confgList_up.size() != ndets || confgList_dn.size() != ndets || coeff.size() != ndets) {
       APP_ABORT("Problems reading determinant configurations.");
     }

     multiSD->C2node_up.resize(coeff.size());
     multiSD->C2node_dn.resize(coeff.size());

     app_log() <<"Found " <<coeff.size() <<" terms in the MSD expansion.\n";

     for(int i=0; i<confgList_up.size(); i++)
     {
       bool found=false;
       int k=-1;
       for(int j=0; j<uniqueConfg_up.size(); j++)
       {
         if(confgList_up[i] == uniqueConfg_up[j]) {
           found=true;
           k=j;
           break;
         }
       }
       if(found) {
         multiSD->C2node_up[i]=k;
       } else {
         uniqueConfg_up.push_back(confgList_up[i]);
         multiSD->C2node_up[i]=uniqueConfg_up.size()-1;
       }
     }
     for(int i=0; i<confgList_dn.size(); i++)
     {
       bool found=false;
       int k=-1;
       for(int j=0; j<uniqueConfg_dn.size(); j++)
       {
         if(confgList_dn[i] == uniqueConfg_dn[j]) {
           found=true;
           k=j;
           break;
         }
       }
       if(found) {
         multiSD->C2node_dn[i]=k;
       } else {
         uniqueConfg_dn.push_back(confgList_dn[i]);
         multiSD->C2node_dn[i]=uniqueConfg_dn.size()-1;
       }
     }
     app_log() <<"Found " <<uniqueConfg_up.size() <<" unique up determinants.\n";
     app_log() <<"Found " <<uniqueConfg_dn.size() <<" unique down determinants.\n";

     multiSD->resize(uniqueConfg_up.size(),uniqueConfg_dn.size());
     SPOSetProxyForMSD* spo = multiSD->spo_up;
     spo->occup.resize(uniqueConfg_up.size(),multiSD->nels_up);
     for(int i=0; i<uniqueConfg_up.size(); i++)
     {
       int nq=0;
       configuration& ci = uniqueConfg_up[i];
       for(int k=0; k<ci.occup.size(); k++) {
         if(ci.occup[k]) { 
           spo->occup(i,nq++) = k;
         }
       }
       DiracDeterminantBase* adet = new DiracDeterminantBase((SPOSetBasePtr) spo,0);
       adet->set(multiSD->FirstIndex_up,multiSD->nels_up);
       multiSD->dets_up.push_back(adet);
     }
     spo = multiSD->spo_dn;
     spo->occup.resize(uniqueConfg_dn.size(),multiSD->nels_dn);
     for(int i=0; i<uniqueConfg_dn.size(); i++)
     {
       int nq=0;
       configuration& ci = uniqueConfg_dn[i];
       for(int k=0; k<ci.occup.size(); k++) {
         if(ci.occup[k]) {
           spo->occup(i,nq++) = k;
         }
       }
       DiracDeterminantBase* adet = new DiracDeterminantBase((SPOSetBasePtr) spo,0);
       adet->set(multiSD->FirstIndex_dn,multiSD->nels_dn);
       multiSD->dets_dn.push_back(adet);
     }

     if(optimizeCI) {
       app_log() <<"CI coefficients are optimizable. ";
       multiSD->Optimizable=true;
       for(int i=0; i<coeff.size(); i++) {
         std::stringstream sstr;
         sstr << "CIcoeff" << "_" << i;
         multiSD->myVars.insert(sstr.str(),coeff[i],true,optimize::LINEAR_P);
       }
     }

     return success;
  }

 // void SlaterDetBuilder::buildMultiSlaterDetermiant()
 // {
 //   MultiSlaterDeterminant *multidet= new MultiSlaterDeterminant;
 //   for (int i=0; i<SlaterDetSet.size(); i++)
 //     {
 //       multidet->add(SlaterDetSet[i],sdet_coeff[i]);
 //     }
 //   // multidet->setOptimizable(true);
 //   //add a MultiDeterminant to the trial wavefuntion
 //   targetPsi.addOrbital(multidet,"MultiSlateDet");
 // }
}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$
 ***************************************************************************/
