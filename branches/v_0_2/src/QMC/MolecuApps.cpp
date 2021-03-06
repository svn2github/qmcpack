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
#include "Numerics/Spline3D/Config.h"
#include "QMC/MolecuApps.h"
#include "Utilities/OhmmsInfo.h"
#include "Particle/HDFWalkerIO.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"
#include "ParticleIO/XMLParticleIO.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "QMCHamiltonians/PolarizationPotential.h"
#include "QMCHamiltonians/WOS/WOSPotential.h"
#include "QMCHamiltonians/WOS/Device.h"
#include "QMCHamiltonians/IonIonPotential.h"
#include "QMCHamiltonians/LocalPPotential.h"
#include "QMCHamiltonians/NonLocalPPotential.h"
#include "QMCHamiltonians/HarmonicPotential.h"
#include "QMCHamiltonians/GeCorePolPotential.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCWaveFunctions/AtomicOrbitals/HFAtomicSTOSetBuilder.h"
#include "QMCWaveFunctions/AtomicOrbitals/HeSTOClementiRottie.h"
#include "QMCWaveFunctions/MolecularOrbitals/MolecularOrbitalBuilder.h"
//#include "QMCWaveFunctions/MolecularOrbitals/NumericalMolecularOrbitals.h"
#include "QMCWaveFunctions/JastrowBuilder.h"
#include "QMCTools/QMCUtilities.h"

namespace ohmmsqmc {

  MolecuApps::MolecuApps(int argc, char** argv): QMCApps(argc,argv) { 
    el.setName("e");
    int iu = el.Species.addSpecies("u");
    int id = el.Species.addSpecies("d");
    int icharge = el.Species.addAttribute("charge");
    el.Species(icharge,iu) = -1;
    el.Species(icharge,id) = -1;
  }

  ///destructor
  MolecuApps::~MolecuApps() {
    DEBUGMSG("MolecuApps::~MolecuApps")
  }

  bool MolecuApps::init() {

    //xmlXPathObjectPtr result
    //  = xmlXPathEvalExpression((const xmlChar*)"//include",m_context);
    //if(xmlXPathNodeSetIsEmpty(result->nodesetval)) {
    //  if(!setParticleSets(m_root)) {
    //     ERRORMSG("Failed to initialize the ions and electrons. Exit now.")
    //     return false;
    //  }
    //} else {
    //  xmlNodePtr cur = result->nodesetval->nodeTab[0];
    //  const xmlChar* a= xmlGetProp(cur,(const xmlChar*)"href");
    //  if(a) {
    //    GamesXmlParser a(ion,el,Psi);
    //    a.parse((const char*)a);
    //  }
    //}
    //xmlXPathFreeObject(result);

    if(!setParticleSets(m_root)) {
       ERRORMSG("Failed to initialize the ions and electrons. Exit now.")
       return false;
    }
    setWavefunctions(m_root);

    if(!setHamiltonian(m_root)) {
      ERRORMSG("Failed to initialize the Hamitonians. Exit now.")
	return false;
    }

    setMCWalkers(m_root);
    
    /*If there is no configuration file manually assign 
      the walker conigurations.  For the ith particle
      R[i] = (rcos(phi)sin(theta),rsin(phi)sin(theta),rcos(theta).
      Initially all the walkers have the same configuration, but 
      in the function QMCDriver::addWalkers loop over all the walkers
      and add a gaussian -> R[i] = R[i] + g\chi.
    if(!setMCWalkers(m_root)) {
      int nup = el.last(0);
      int nions = ion.getTotalNum();
      double r=0.012;
      for (int ipart=0; ipart<el.getTotalNum(); ++ipart) {
	double costheta=2*Random()-1;
	double sintheta=sqrt(1-costheta*costheta);
	double phi=2*3.141592653*Random();
	el.R[ipart] += 
	  MCWalkerConfiguration::PosType(r*cos(phi)*sintheta,
					 r*sin(phi)*sintheta,r*costheta);
      }
    }
     */
    string fname(myProject.CurrentRoot());
    fname.append(".debug");
    ofstream fout(fname.c_str());
    fout << "Ionic configuration : " << ion.getName() << endl;
    ion.get(fout);
    fout << "Electronic configuration : " << el.getName() << endl;
    el.get(fout);
    Psi.VarList.print(fout);
    return true;    
  }   

  bool  MolecuApps::setParticleSets(xmlNodePtr aroot) {

    bool init_els = determineNumOfElectrons(el,m_context);

    xmlXPathObjectPtr result
      = xmlXPathEvalExpression((const xmlChar*)"//particleset",m_context);

    xmlNodePtr el_ptr=NULL, ion_ptr=NULL;
    for(int i=0; i<result->nodesetval->nodeNr; i++) {
      xmlNodePtr cur=result->nodesetval->nodeTab[i];
      xmlChar* aname= xmlGetProp(cur,(const xmlChar*)"name");
      if(aname) {
	char fc = aname[0];
	if(fc == 'e') { el_ptr=cur;}
	else if(fc == 'i') {ion_ptr=cur;}
      }
    }

    bool donotresize = false;
    if(init_els) {
      el.setName("e");
      XMLReport("The configuration for electrons is already determined by the wave function")
      donotresize = true;
    } 

    if(ion_ptr) {
      XMLParticleParser pread(ion);
      pread.put(ion_ptr);
    }

    if(el_ptr) {
      XMLParticleParser pread(el,donotresize);
      pread.put(el_ptr);
    }

    xmlXPathFreeObject(result);

    if(!ion.getTotalNum()) {
      ion.setName("i");
      ion.create(1);
      ion.R[0] = 0.0;
    }

    return true;
  }

  /** Initialize the Hamiltonian
   */
  bool MolecuApps::setHamiltonian(xmlNodePtr aroot){

    string ptype("molecule");
    string mtype("analytic");
    double Efield = 0.0;
    xmlXPathObjectPtr result
      = xmlXPathEvalExpression((const xmlChar*)"//hamiltonian",m_context);

    xmlNodePtr pol=NULL;
    xmlNodePtr cpp=NULL;
    if(xmlXPathNodeSetIsEmpty(result->nodesetval)) {
      return false;
    } else {
      xmlNodePtr cur = result->nodesetval->nodeTab[0];
      xmlChar* att= xmlGetProp(cur,(const xmlChar*)"type");
      if(att) {
	ptype = (const char*)att;
	if(ptype=="polarization"){ pol=cur;}
	if(ptype=="cpp"){ cpp=cur;}
      }
    }

    xmlXPathFreeObject(result);

    XMLReport("Found Potential of type " << ptype)
    //always add kinetic energy first
    H.add(new BareKineticEnergy, "Kinetic");
    H.add(new CoulombPotentialAA(el),"ElecElec");

    if(ptype == "molecule" || ptype=="coulomb"){
      H.add(new CoulombPotentialAB(ion,el),"Coulomb");
      if(ion.getTotalNum()>1) 
	H.add(new IonIonPotential(ion),"IonIon");
    } else if(ptype == "harmonic") {
      H.add(new HarmonicPotential(ion,el),"Coulomb");
    } else if(ptype == "siesta" || ptype=="pseudo") {
      H.add(new NonLocalPPotential(ion,el,Psi),"NonLocal");
      //H.add(new LocalPPotential(ion,el), "PseudoPot");
      if(ion.getTotalNum()>1) 
	H.add(new IonIonPotential(ion),"IonIon");
    } else if(ptype == "cpp") {
      xmlChar* att2=xmlGetProp(cpp,(const xmlChar*)"species");
      string stype("Ge");
      if(att2) stype = (const char*)att2;
      H.add(new LocalPPotential(ion,el), "PseudoPot");
      H.add(new GeCorePolPotential(ion,el), "GeCPP");
      if(ion.getTotalNum()>1) 
	H.add(new IonIonPotential(ion),"IonIon");
    } else if(ptype == "polarization"){
      xmlChar* att2=xmlGetProp(pol,(const xmlChar*)"method");
      mtype = (const char*)att2;
      if(mtype == "wos") {
	xmlNodePtr cur1 = pol->children;
	while(cur1 != NULL) {
	  string cname((const char*)(cur1->name));
	  if(cname=="wos"){
	    int nruns = atoi((char*)xmlGetProp(cur1,(const xmlChar*)"Vruns"));
	    int mruns = atoi((char*)xmlGetProp(cur1,(const xmlChar*)"Druns"));
	    int mode = atoi((char*)xmlGetProp(cur1,(const xmlChar*)"mode"));
	    double dz = atof((char*)xmlGetProp(cur1,(const xmlChar*)"dz"));
	    double dv = atof((char*)xmlGetProp(cur1,(const xmlChar*)"dv"));
	    double dr = atof((char*)xmlGetProp(cur1,(const xmlChar*)"dr"));
	    double L = 10000;
	    posvec_t rmin(-L,-L,-dz);
	    posvec_t rmax( L, L, dz);
	    double eps = 1.0; double rho = 0.0;
	    std::vector<double> Vapp(6,0.0); Vapp[4] = -dv; Vapp[5] = dv;
	    Device* device = new Device(dr,eps,rho,Vapp,rmin,rmax);
	    H.add(new WOSPotential(mode,nruns,mruns,device,ion,el),"wos");
	  }
	  cur1=cur1->next;
	} 
      } else if(mtype == "analytic"){
	H.add(new CoulombPotentialAA(el),"ElecElec");
	H.add(new CoulombPotentialAB(ion,el),"Coulomb");

	xmlNodePtr cur1 = pol->children;
	while(cur1 != NULL ){
	  string cname((const char*)(cur1->name));
	  if(cname=="analytic"){
	    double Ez = atof((char*)xmlGetProp(cur1,(const xmlChar*)"Ez"));
	    cout << "Adding Electric field " << Ez << endl;
	    H.add(new PolarizationPotential(Ez),"Polarization");
	  }
	  cur1= cur1->next;
	}
      }
    } else {
      ERRORMSG(ptype << " is not supported.")
	return false;
    }


    return true;
  }
  

  /** Find a xmlnode Wavefunction and initialize the TrialWaveFunction by adding components.
   *@param root xml node
   *
   *This function substitutes TrialWaveFunctionBuilder
   *that is intended as a Builder for any TrialWaveFunction.
   *Since MolecuApps is specialized for molecular systems,
   *the type of allowed many-body wave functions can be decided here.
   *
   *Allowed many-body wave functions for MolecuApps are
   <ul>
   <li> DeterminantSet: only one kind of SlateDetermant can be used.
   <ul>
   <li> STO-He-Optimized: HePresetHF (specialized for He, test only)
   <li> STO-Clementi-Rottie: HFAtomicSTOSet (Hartree-Fock orbitals for atoms)
   <li> MolecularOrbital: molecular orbitals with radial orbitals 
   </ul>
   <li> Jastrow: any of them can be used
   <ul>
   <li> One-Body Jastrow
   <li> Two-Body Jastrow
   </ul>
   </ul>
   The number of terms is arbitrary.
   */
  bool MolecuApps::setWavefunctions(xmlNodePtr aroot) {

    xmlXPathObjectPtr result
      = xmlXPathEvalExpression((const xmlChar*)"//wavefunction",m_context);

    ///make a temporary array to pass over JastrowBuilder
    map<string,ParticleSet*> selected;
    selected[ion.getName()]=&ion;
    selected[el.getName()]=&el;

    bool foundwfs=true;
    if(xmlXPathNodeSetIsEmpty(result->nodesetval)) {
      ERRORMSG("Wavefunction is missing. Exit." << endl)
      foundwfs=false;
    } else {
      xmlNodePtr cur = result->nodesetval->nodeTab[0]->children;
      while(cur != NULL) {
	string cname((const char*)(cur->name));
	if (cname == OrbitalBuilderBase::detset_tag) {
	  string orbtype=(const char*)(xmlGetProp(cur, (const xmlChar *)"type"));
	  LOGMSG("Slater-determinant terms using " << orbtype)
	  if(orbtype == "STO-Clementi-Rottie") {
            HFAtomicSTOSetBuilder a(el,Psi,ion);
	    a.put(cur);
	  } else if(orbtype == "STO-He-Optimized") {
	    HePresetHFBuilder a(el,Psi,ion);
	    //a.put(cur);
	  } else if(orbtype == "MolecularOrbital") {
	    MolecularOrbitalBuilder a(el,Psi,selected);
	    a.put(cur);
          }
	  XMLReport("Done with the initialization of SlaterDeterminant using " << orbtype)
        } // if DeterminantSet    
	else if (cname ==  OrbitalBuilderBase::jastrow_tag) {
	  JastrowBuilder a(el,Psi,selected);
	  a.put(cur);
	}
	cur = cur->next;
      }
      LOGMSG("Completed the initialization of a many-body wave function")
    }
    xmlXPathFreeObject(result);
    return foundwfs;
  }
}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
