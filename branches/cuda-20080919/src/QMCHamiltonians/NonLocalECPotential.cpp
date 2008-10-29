//////////////////////////////////////////////////////////////////
// (c) Copyright 2005- by Jeongnim Kim and Simone Chiesa
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
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "Utilities/IteratorUtility.h"
#ifdef QMC_CUDA
  #include "NLPP.h"
#endif

namespace qmcplusplus {

  void NonLocalECPotential::resetTargetParticleSet(ParticleSet& P) 
  {
    d_table = DistanceTable::add(IonConfig,P);
  }

  /** constructor
   *\param ions the positions of the ions
   *\param els the positions of the electrons
   *\param psi trial wavefunction
   */
  NonLocalECPotential::NonLocalECPotential(ParticleSet& ions, ParticleSet& els,
      TrialWaveFunction& psi): IonConfig(ions), d_table(0), Psi(psi)
  { 
    d_table = DistanceTable::add(ions,els);
    NumIons=ions.getTotalNum();
    //els.resizeSphere(NumIons);
    PP.resize(NumIons,0);
    PPset.resize(IonConfig.getSpeciesSet().getTotalNum(),0);
    setupCuda(els);
  }

  ///destructor
  NonLocalECPotential::~NonLocalECPotential() 
  { 
    delete_iter(PPset.begin(),PPset.end());
    //map<int,NonLocalECPComponent*>::iterator pit(PPset.begin()), pit_end(PPset.end());
    //while(pit != pit_end) {
    //   delete (*pit).second; ++pit;
    //}
  }

  NonLocalECPotential::Return_t
  NonLocalECPotential::evaluate(ParticleSet& P) { 
    Value=0.0;
    //loop over all the ions
    for(int iat=0; iat<NumIons; iat++) {
      if(PP[iat]) {
        PP[iat]->randomize_grid(*(P.Sphere[iat]),UpdateMode[PRIMARY]);
        Value += PP[iat]->evaluate(P,iat,Psi);
      }
    }
    return Value;
  }

  NonLocalECPotential::Return_t
  NonLocalECPotential::evaluate(ParticleSet& P, vector<NonLocalData>& Txy) { 
    Value=0.0;
    //loop over all the ions
    for(int iat=0; iat<NumIons; iat++) {
      if(PP[iat]) {
        PP[iat]->randomize_grid(*(P.Sphere[iat]),UpdateMode[PRIMARY]);
        Value += PP[iat]->evaluate(P,Psi,iat,Txy);
      }
    }
    return Value;
  }

  void 
  NonLocalECPotential::add(int groupID, NonLocalECPComponent* ppot) {
    //map<int,NonLocalECPComponent*>::iterator pit(PPset.find(groupID));
    //ppot->myTable=d_table;
    //if(pit  == PPset.end()) {
    //  for(int iat=0; iat<PP.size(); iat++) {
    //    if(IonConfig.GroupID[iat]==groupID) PP[iat]=ppot;
    //  }
    //  PPset[groupID]=ppot;
    //}
    ppot->myTable=d_table;
    for(int iat=0; iat<PP.size(); iat++) 
      if(IonConfig.GroupID[iat]==groupID) PP[iat]=ppot;
    PPset[groupID]=ppot;
  }

  QMCHamiltonianBase* NonLocalECPotential::makeClone(ParticleSet& qp, TrialWaveFunction& psi)
  {
    NonLocalECPotential* myclone=new NonLocalECPotential(IonConfig,qp,psi);

    for(int ig=0; ig<PPset.size(); ++ig)
    {
      if(PPset[ig]) myclone->add(ig,PPset[ig]->makeClone());
    }

    //resize sphere
    qp.resizeSphere(IonConfig.getTotalNum());
    for(int ic=0; ic<IonConfig.getTotalNum(); ic++) {
      if(PP[ic] && PP[ic]->nknot) qp.Sphere[ic]->resize(PP[ic]->nknot);
    }
    return myclone;
  }


  void NonLocalECPotential::setRandomGenerator(RandomGenerator_t* rng)
  {
    for(int ig=0; ig<PPset.size(); ++ig)
      if(PPset[ig]) PPset[ig]->setRandomGenerator(rng);
    //map<int,NonLocalECPComponent*>::iterator pit(PPset.begin()), pit_end(PPset.end());
    //while(pit != pit_end) {
    //  (*pit).second->setRandomGenerator(rng);
    //  ++pit;
    //}
  }

  void NonLocalECPotential::setupCuda(ParticleSet &elecs)
  {
    SpeciesSet &sSet = IonConfig.getSpeciesSet();
    NumIonGroups = sSet.getTotalNum();
    host_vector<CUDA_PRECISION> LHost(OHMMS_DIM*OHMMS_DIM), 
      LinvHost(OHMMS_DIM*OHMMS_DIM);
    for (int i=0; i<OHMMS_DIM; i++)
      for (int j=0; j<OHMMS_DIM; j++) {
	LHost[OHMMS_DIM*i+j]    = (CUDA_PRECISION)elecs.Lattice.a(i)[j];
	LinvHost[OHMMS_DIM*i+j] = (CUDA_PRECISION)elecs.Lattice.b(i)[j];
      }
    L = LHost;
    Linv = LinvHost;
    NumElecs = elecs.getTotalNum();
    
    // Copy ion positions to GPU, sorting by GroupID
    host_vector<CUDA_PRECISION> Ion_host(OHMMS_DIM*IonConfig.getTotalNum());
    int index=0;
    for (int group=0; group<NumIonGroups; group++) {
      IonFirst.push_back(index);
      for (int i=0; i<IonConfig.getTotalNum(); i++) {
	if (IonConfig.GroupID[i] == group) {
	  for (int dim=0; dim<OHMMS_DIM; dim++) 
	    Ion_host[OHMMS_DIM*i+dim] = IonConfig.R[i][dim];
	  index++;
	}
      }
      IonLast.push_back(index-1);
    }
    Ions_GPU = Ion_host;
  }

  void NonLocalECPotential::addEnergy(vector<Walker_t*> &walkers, 
				      vector<RealType> &LocalEnergy)
  {
    int nw = walkers.size();
    // Copy electron positions to GPU memory
    if (R_host.size() < OHMMS_DIM*NumElecs*nw) {
      R_host.resize(OHMMS_DIM*NumElecs*nw);
      R_GPU.resize(OHMMS_DIM*NumElecs*nw);
      Rlist_host.resize(nw);
      Rlist_GPU.resize(nw);
      for (int iw=0; iw<nw; iw++)
	Rlist_host[iw] = &(R_GPU[OHMMS_DIM*NumElecs*iw]);
      Rlist_GPU = Rlist_host;
      
      int maxPairs = 0;
      for (int sp=0; sp<NumIonGroups; sp++)
	maxPairs = max(maxPairs, IonLast[sp]-IonFirst[sp]+1);
      maxPairs *= NumElecs;
      Pairs_GPU.resize(maxPairs*nw);
      Dist_GPU.resize(maxPairs*nw);
      host_vector<int2*> Pairlist_host(nw);
      host_vector<CUDA_PRECISION*> Distlist_host(nw);
      Pairlist_GPU.resize(nw);
      Distlist_GPU.resize(nw);
      NumPairs_GPU.resize(nw);
      for (int iw=0; iw<nw; iw++) {
	Pairlist_host[iw] = &(Pairs_GPU[maxPairs*iw]);
	Distlist_host[iw] = &(Dist_GPU[maxPairs*iw]);
      }
      Pairlist_GPU = Pairlist_host;
      Distlist_GPU = Distlist_host;
    }
    for (int iw=0; iw<nw; iw++) 
      for (int e=0; e<NumElecs; e++)
	for (int dim=0; dim<OHMMS_DIM; dim++)
	  R_host[OHMMS_DIM*NumElecs*iw + OHMMS_DIM*e + dim] = 
	    walkers[iw]->R[e][dim];
    R_GPU = R_host;

    // Loop over the ionic species
    for (int sp=0; sp<NumIonGroups; sp++) 
      if (PPset[sp]) {
	// First, we need to determine which ratios need to be updated
	app_log() << "Before find_core_electrons.\n";
	cerr << "PPset.size() = " << PPset.size() << endl;
	fprintf (stderr, "PPset[sp] = %p\n", PPset[sp]);
	cerr << "IonFirst = " << IonFirst[sp] << endl;      
	cerr << "IonLast  = " << IonLast[sp] << endl;
	cerr << "Ions_GPU.size() = " << Ions_GPU.size() << endl;
	cerr << "Rmax = " << PPset[sp]->Rmax << endl;
	cerr << "PairList_GPU.size() = " <<Pairlist_GPU.size() << endl;
	find_core_electrons (Rlist_GPU.data(), NumElecs,
			     Ions_GPU.data(), IonFirst[sp], IonLast[sp],
			     ( CUDA_PRECISION)PPset[sp]->Rmax, 
			     L.data(), Linv.data(), 
			     Pairlist_GPU.data(), Distlist_GPU.data(),
			     NumPairs_GPU.data(), walkers.size());
	app_log() << "After find_core_electrons.\n";
	
	host_vector<int> NumPairs_host;
	host_vector<CUDA_PRECISION> Dist_host;
	NumPairs_host = NumPairs_GPU;
	Dist_host = Dist_GPU;
	app_log() << "Walker[0] has " << NumPairs_host[0] << " pairs\n";
	app_log() << "Dist_GPU[0] is " << Dist_host[0] << " pairs\n";
      }
  }

}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
