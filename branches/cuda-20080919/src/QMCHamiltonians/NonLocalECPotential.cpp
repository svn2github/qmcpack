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
#include "Lattice/ParticleBConds.h"
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
					   TrialWaveFunction& psi) : 
    IonConfig(ions), d_table(0), Psi(psi), CurrentNumWalkers(0)
								    
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
	    Ion_host[OHMMS_DIM*index+dim] = IonConfig.R[i][dim];
	  SortedIons.push_back(IonConfig.R[i]);
	  index++;
	}
      }
      IonLast.push_back(index-1);
    }
    Ions_GPU = Ion_host;

  }

  void NonLocalECPotential::resizeCuda(int nw)
  {
    R_host.resize(OHMMS_DIM*NumElecs*nw);
    R_GPU.resize(OHMMS_DIM*NumElecs*nw);
    Rlist_host.resize(nw);
    Rlist_GPU.resize(nw);
    for (int iw=0; iw<nw; iw++)
      Rlist_host[iw] = &(R_GPU[OHMMS_DIM*NumElecs*iw]);
    Rlist_GPU = Rlist_host;

    MaxPairs = 2 * NumElecs;
        
    // Note: this will not cover pathological systems in which all
    // the cores overlap
    Elecs_GPU.resize(MaxPairs*nw);
    Dist_GPU.resize(MaxPairs*nw);
    host_vector<int*> Eleclist_host(nw);
    host_vector<CUDA_PRECISION*> Distlist_host(nw);
    Eleclist_GPU.resize(nw);
    Distlist_GPU.resize(nw);
    NumPairs_GPU.resize(nw);
    for (int iw=0; iw<nw; iw++) {
      Eleclist_host[iw] = &(Elecs_GPU[MaxPairs*iw]);
      Distlist_host[iw] = &(Dist_GPU[MaxPairs*iw]);
    }
    Eleclist_GPU = Eleclist_host;
    Distlist_GPU = Distlist_host;
    
    // Resize ratio positions vector

    // Compute maximum number of knots
    MaxKnots = 0;
    for (int i=0; i<PPset.size(); i++)
      if (PPset[i])
	MaxKnots = max(MaxKnots,PPset[i]->nknot);

    RatiosPerWalker = MaxPairs * MaxKnots;
    RatioPos_GPU.resize(OHMMS_DIM * RatiosPerWalker * nw);
    CosTheta_GPU.resize(RatiosPerWalker * nw);
    host_vector<CUDA_PRECISION*> RatioPoslist_host(nw);
    host_vector<CUDA_PRECISION*> Ratiolist_host(nw);
    host_vector<CUDA_PRECISION*> CosThetalist_host(nw);
    for (int iw=0; iw<nw; iw++) {
      RatioPoslist_host[iw] = 
	&(RatioPos_GPU[OHMMS_DIM * RatiosPerWalker * iw]);
      CosThetalist_host[iw] = &(CosTheta_GPU[RatiosPerWalker*iw]);
    }
    RatioPoslist_GPU = RatioPoslist_host;
    CosThetalist_GPU = CosThetalist_host;

    QuadPoints_GPU.resize(NumIonGroups);
    QuadPoints_host.resize(NumIonGroups);
    for (int i=0; i<NumIonGroups; i++) 
      if (PPset[i]) {
	QuadPoints_GPU[i].resize(OHMMS_DIM*PPset[i]->nknot);
	QuadPoints_host[i].resize(OHMMS_DIM*PPset[i]->nknot);
      }
    CurrentNumWalkers = nw;
  }

  void NonLocalECPotential::addEnergy(vector<Walker_t*> &walkers, 
				      vector<RealType> &LocalEnergy)
  {
    int nw = walkers.size();
    if (CurrentNumWalkers < nw)
      resizeCuda(nw);

    // Copy electron positions to GPU
    for (int iw=0; iw<nw; iw++) 
      for (int e=0; e<NumElecs; e++)
	for (int dim=0; dim<OHMMS_DIM; dim++)
	  R_host[OHMMS_DIM*NumElecs*iw + OHMMS_DIM*e + dim] = 
	    walkers[iw]->R[e][dim];
    R_GPU = R_host;

    // Loop over the ionic species
    vector<RealType> esum(walkers.size(), 0.0);
    for (int sp=0; sp<NumIonGroups; sp++) 
      if (PPset[sp]) {
	NonLocalECPComponent &pp = *PPset[sp];
	PPset[sp]->randomize_grid(QuadPoints_host[sp]);
	QuadPoints_GPU[sp] = QuadPoints_host[sp];

	// First, we need to determine which ratios need to be updated
	find_core_electrons 
	  (Rlist_GPU.data(), NumElecs, 
	   Ions_GPU.data(), IonFirst[sp], IonLast[sp],
	   (CUDA_PRECISION)PPset[sp]->Rmax, L.data(), Linv.data(), 
	   QuadPoints_GPU[sp].data(), PPset[sp]->nknot,
	   Eleclist_GPU.data(), RatioPoslist_GPU.data(),
	   Distlist_GPU.data(), CosThetalist_GPU.data(),
	   NumPairs_GPU.data(), walkers.size());

	// Concatenate work into job list
	RatioPos_host = RatioPos_GPU;
	NumPairs_host = NumPairs_GPU;
	Elecs_host    = Elecs_GPU;
	CosTheta_host = CosTheta_GPU;
	Dist_host      = Dist_GPU;
	int numQuad = PPset[sp]->nknot;

	JobList.clear();
	QuadPosList.clear();
	for (int iw=0; iw<nw; iw++) {
	  CUDA_PRECISION *pos_host = &(RatioPos_host[OHMMS_DIM*RatiosPerWalker*iw]);
	  int *elecs = &(Elecs_host[iw*MaxPairs]);
	  for (int ie=0; ie<NumPairs_host[iw]; ie++) {
	    JobList.push_back(NLjob(iw,*(elecs++),numQuad));
	    for (int iq=0; iq<numQuad; iq++) {
	      PosType r; 
	      for (int dim=0; dim<OHMMS_DIM; dim++)
		r[dim] = *(pos_host++);
	      QuadPosList.push_back(r);
	    }
	  }
	}
	RatioList.resize(QuadPosList.size());

	RealType vrad[pp.nchannel];
	RealType lpol[pp.lmax+1];

	Psi.NLratios(walkers, JobList, QuadPosList, RatioList);
	int ratioIndex=0;
	for (int iw=0; iw<nw; iw++) {
	  CUDA_PRECISION *cos_ptr = &(CosTheta_host[RatiosPerWalker*iw]);
	  CUDA_PRECISION *dist_ptr = &(Dist_host[MaxPairs*iw]);
	  for (int ie=0; ie<NumPairs_host[iw]; ie++) {
	    RealType dist = *(dist_ptr++);

	    for (int ip=0; ip<pp.nchannel; ip++) 
	      vrad[ip] = pp.nlpp_m[ip]->splint(dist) * pp.wgt_angpp_m[ip];
	      
	    for (int iq=0; iq<numQuad; iq++) {
	      RealType costheta = *(cos_ptr++);
	      RealType ratio    = RatioList[ratioIndex++] * pp.sgridweight_m[iq];
	      RealType lpolprev=0.0;
	      lpol[0] = 1.0;

	      for (int l=0 ; l< pp.lmax ; l++){
		//Not a big difference
		// lpol[l+1]=(2*l+1)*costheta*lpol[l]-l*lpolprev;
		// lpol[l+1]/=(RealType)(l+1);
		lpol[l+1]  = pp.Lfactor1[l]*costheta*lpol[l]-l*lpolprev; 
		lpol[l+1] *= pp.Lfactor2[l]; 
		lpolprev=lpol[l];
	      }

	      for (int ip=0 ; ip<pp.nchannel ; ip++) 
		esum[iw] +=  vrad[ip] * lpol[pp.angpp_m[ip]] * ratio;
	    }
	  }
	}
      }
    for (int iw=0; iw<walkers.size(); iw++){
      walkers[iw]->getPropertyBase()[NUMPROPERTIES+myIndex] = esum[iw];
      LocalEnergy[iw] += esum[iw];
    }
  


#ifdef CUDA_DEBUG
	host_vector<int> Elecs_host;
	host_vector<int> NumPairs_host;
	host_vector<CUDA_PRECISION> RatioPos_host;
	RatioPos_host = RatioPos_GPU;
	NumPairs_host = NumPairs_GPU;
	Elecs_host    = Elecs_GPU;

	DTD_BConds<double,3,SUPERCELL_BULK> bconds;
	Walker_t &w = *walkers[0];
	int index = 0;
	int numPairs = 0;
	for (int i=IonFirst[sp]; i<=IonLast[sp]; i++) {
	  PosType ion = SortedIons[i];
	  for (int e=0; e<NumElecs; e++) {
	    PosType elec = w.R[e];
	    PosType disp = elec-ion;
	    double dist = 
	      std::sqrt(bconds.apply(IonConfig.Lattice, disp));
	    
	    if (dist < PPset[sp]->Rmax) {
	      numPairs++;
	      // fprintf (stderr, "i_CPU=%d  i_GPU=%d  elec_CPU=%d  elec_GPU=%d\n",
	      // 	       i, Elecs_host[index],
	      // 	       e, Elecs_host[index]);
	      
	      // int nknot = PPset[sp]->nknot;
	      // for (int k=0; k<PPset[sp]->nknot; k++) {
	      // 	PosType r = ion + dist * PPset[sp]->rrotsgrid_m[k];
	      // 	fprintf (stderr, "CPU %d %12.6f %12.6f %12.6f\n",
	      // 		 index, r[0], r[1], r[2]);
	      // 	fprintf (stderr, "GPU %d %12.6f %12.6f %12.6f\n", index,
	      // 		 RatioPos_host[3*index*nknot+3*k+0], 
	      // 		 RatioPos_host[3*index*nknot+3*k+1],
	      // 		 RatioPos_host[3*index*nknot+3*k+2]);
	    }
	    index ++;
	  }
	}
      
	cerr << "numPairs = " << numPairs << endl;
	cerr << "NumPairs_host[0] =" << NumPairs_host[0] << endl;
#endif
  }
// 	for (int i=0; i<NumPairs_host[0]*PPset[sp]->nknot; i++) {
// 	  fprintf (stderr, "%d %12.6f %12.6f %12.6f\n", i,
// 		   RatioPos_host[3*i+0],
// 		   RatioPos_host[3*i+1],
// 		   RatioPos_host[3*i+2]);
//	}

	// host_vector<CUDA_PRECISION> Dist_host;
	// NumPairs_host = NumPairs_GPU;
	// Dist_host = Dist_GPU;

}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
