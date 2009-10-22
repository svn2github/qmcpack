//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim and Kris Delaney
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
#include "QMCHamiltonians/CoulombPBCABTemp.h"
#include "Particle/DistanceTable.h"
#include "Particle/DistanceTableData.h"
#include "Message/Communicate.h"
#include "Utilities/ProgressReportEngine.h"

namespace qmcplusplus {

  CoulombPBCABTemp::CoulombPBCABTemp(ParticleSet& ions, ParticleSet& elns,
				     bool cloning): 
    PtclA(ions), myConst(0.0), myGrid(0),V0(0), ElecRef(elns), IonRef(ions),
    SumGPU("CoulombPBCABTemp::SumGPU"),
    IGPU("CoulombPBCABTemp::IGPU"),
    L("CoulombPBCABTemp::L"),
    Linv("CoulombPBCABTemp::Linv"),
    kpointsGPU("CoulombPBCABTemp::kpointsGPU"),
    kshellGPU("CoulombPBCABTemp::kshellGPU"),
    FkGPU("CoulombPBCABTemp::FkGPU"),
    RhoklistGPU("CoulombPBCABTemp::RhoklistGPU"),
    RhokElecGPU("CoulombPBCABTemp::RhokElecGPU")
    {
      ReportEngine PRE("CoulombPBCABTemp","CoulombPBCABTemp");
      //Use singleton pattern 
      //AB = new LRHandlerType(ions);
      myTableIndex=elns.addTable(ions);

      SpeciesSet &sSet = ions.getSpeciesSet();
      NumIonSpecies = sSet.getTotalNum();
      initBreakup(elns, cloning);
      app_log() << "  Maximum K shell " << AB->MaxKshell << endl;
      app_log() << "  Number of k vectors " << AB->Fk.size() << endl;
      NumIons  = ions.getTotalNum();
      NumElecs = elns.getTotalNum();


      // CUDA setup
#ifdef QMC_CUDA
      gpu::host_vector<CUDA_PRECISION> LHost(9), LinvHost(9);
      for (int i=0; i<3; i++)
	for (int j=0; j<3; j++) {
	  LHost[3*i+j]    = elns.Lattice.a(j)[i];
	  LinvHost[3*i+j] = elns.Lattice.b(i)[j];
	}
      L    = LHost;
      Linv = LinvHost;
      
      // Copy center positions to GPU, sorting by GroupID
      gpu::host_vector<CUDA_PRECISION> I_host(OHMMS_DIM*NumIons);
      int index=0;
      for (int cgroup=0; cgroup<NumIonSpecies; cgroup++) {
	IonFirst.push_back(index);
	for (int i=0; i<NumIons; i++) {
	  if (ions.GroupID[i] == cgroup) {
	    for (int dim=0; dim<OHMMS_DIM; dim++) 
	      I_host[OHMMS_DIM*index+dim] = ions.R[i][dim];
	    SortedIons.push_back(ions.R[i]);
	    index++;
	  }
	}
	IonLast.push_back(index-1);
      }
      IGPU = I_host;
      SRSplines.resize(NumIonSpecies,0);
      setupLongRangeGPU();
#endif
    }

  QMCHamiltonianBase* CoulombPBCABTemp::makeClone(ParticleSet& qp, TrialWaveFunction& psi)
  {
    CoulombPBCABTemp* myclone=new CoulombPBCABTemp(PtclA, qp, true);
    if(myGrid) myclone->myGrid=new GridType(*myGrid);
    for(int ig=0; ig<Vspec.size(); ++ig)
    {
      if(Vspec[ig]) 
      {
        RadFunctorType* apot=Vspec[ig]->makeClone();
        myclone->Vspec[ig]=apot;
        for(int iat=0; iat<PtclA.getTotalNum(); ++iat)
        {
          if(PtclA.GroupID[iat]==ig) myclone->Vat[iat]=apot;
        }
      }
      myclone->V0Spline = V0Spline;
      myclone->SRSplines[ig] = SRSplines[ig];

    }
    return myclone;
  }

  CoulombPBCABTemp:: ~CoulombPBCABTemp() 
  {
    //probably need to clean up
  }

  void CoulombPBCABTemp::resetTargetParticleSet(ParticleSet& P) {
    int tid=P.addTable(PtclA);
    if(tid != myTableIndex)
    {
      APP_ABORT("CoulombPBCABTemp::resetTargetParticleSet found inconsistent table index");
    }
    AB->resetTargetParticleSet(P);
  }

  CoulombPBCABTemp::Return_t 
    CoulombPBCABTemp::evaluate(ParticleSet& P) 
    {
      // HACK HACK HACK
      return Value = evalLR(P)+evalSR(P)+myConst;
      // return Value = evalSR(P) + myConst;
    }

  CoulombPBCABTemp::Return_t 
    CoulombPBCABTemp::registerData(ParticleSet& P, BufferType& buffer) 
    {
      P.SK->DoUpdate=true;

      SRpart.resize(NptclB);
      LRpart.resize(NptclB);

      Value=evaluateForPyP(P);

      buffer.add(SRpart.begin(),SRpart.end());
      buffer.add(LRpart.begin(),LRpart.end());
      buffer.add(Value);
      return Value;
    }

  CoulombPBCABTemp::Return_t 
    CoulombPBCABTemp::updateBuffer(ParticleSet& P, BufferType& buffer) 
    {
      Value=evaluateForPyP(P);
      buffer.put(SRpart.begin(),SRpart.end());
      buffer.put(LRpart.begin(),LRpart.end());
      buffer.put(Value);
      return Value;
    }

  void CoulombPBCABTemp::copyFromBuffer(ParticleSet& P, BufferType& buffer) 
  {
    buffer.get(SRpart.begin(),SRpart.end());
    buffer.get(LRpart.begin(),LRpart.end());
    buffer.get(Value);
  }

  void CoulombPBCABTemp::copyToBuffer(ParticleSet& P, BufferType& buffer) 
  {
    buffer.put(SRpart.begin(),SRpart.end());
    buffer.put(LRpart.begin(),LRpart.end());
    buffer.put(Value);
  }

  CoulombPBCABTemp::Return_t 
    CoulombPBCABTemp::evaluateForPyP(ParticleSet& P) 
    {
      const DistanceTableData* d_ab=P.DistTables[myTableIndex];
      Return_t res=myConst;
      SRpart=0.0;
      for(int iat=0; iat<NptclA; ++iat)
      {
        RealType z=Zat[iat];
        RadFunctorType* rVs=Vat[iat];
        for(int nn=d_ab->M[iat], jat=0; nn<d_ab->M[iat+1]; nn++,jat++) 
        {
          RealType e=z*Qat[jat]*d_ab->rinv(nn)*rVs->splint(d_ab->r(nn));
          SRpart[jat]+=e;
          res+=e;
        }
      }
      LRpart=0.0;
      const StructFact& RhoKA(*(PtclA.SK));
      const StructFact& RhoKB(*(P.SK));
      // const StructFact& RhoKB(*(PtclB->SK));
      for(int i=0; i<NumSpeciesA; i++) 
      {
        RealType z=Zspec[i];
        for(int jat=0; jat<P.getTotalNum(); ++jat)
        {
          RealType e=z*Qat[jat]*AB->evaluate(RhoKA.KLists.kshell, RhoKA.rhok[i],RhoKB.eikr[jat]);
          LRpart[jat]+=e;
          res+=e;
        }
      }
      return res;
    }


  CoulombPBCABTemp::Return_t 
    CoulombPBCABTemp::evaluatePbyP(ParticleSet& P, int active)
    {
      const std::vector<DistanceTableData::TempDistType> &temp(P.DistTables[myTableIndex]->Temp);
      RealType q=Qat[active];
      SRtmp=0.0;
      for(int iat=0; iat<NptclA; ++iat)
      {
        SRtmp+=Zat[iat]*q*temp[iat].rinv1*Vat[iat]->splint(temp[iat].r1);
      }

      LRtmp=0.0;
      const StructFact& RhoKA(*(PtclA.SK));
      //const StructFact& RhoKB(*(PtclB->SK));
      const StructFact& RhoKB(*(P.SK));
      for(int i=0; i<NumSpeciesA; i++) 
        LRtmp+=Zspec[i]*q*AB->evaluate(RhoKA.KLists.kshell, RhoKA.rhok[i],RhoKB.eikr_temp.data());
      return NewValue=Value+(SRtmp-SRpart[active])+(LRtmp-LRpart[active]);
      return NewValue=Value+(SRtmp-SRpart[active]);
    }

  void CoulombPBCABTemp::acceptMove(int active)
  {
    SRpart[active]=SRtmp;
    LRpart[active]=LRtmp;
    Value=NewValue;
  }

  void CoulombPBCABTemp::initBreakup(ParticleSet& P, bool cloning) 
  {
    SpeciesSet& tspeciesA(PtclA.getSpeciesSet());
    SpeciesSet& tspeciesB(P.getSpeciesSet());

    int ChargeAttribIndxA = tspeciesA.addAttribute("charge");
    int MemberAttribIndxA = tspeciesA.addAttribute("membersize");
    int ChargeAttribIndxB = tspeciesB.addAttribute("charge");
    int MemberAttribIndxB = tspeciesB.addAttribute("membersize");

    NptclA = PtclA.getTotalNum();
    NptclB = P.getTotalNum();

    NumSpeciesA = tspeciesA.TotalNum;
    NumSpeciesB = tspeciesB.TotalNum;

    //Store information about charges and number of each species
    Zat.resize(NptclA); Zspec.resize(NumSpeciesA);
    Qat.resize(NptclB); Qspec.resize(NumSpeciesB);

    NofSpeciesA.resize(NumSpeciesA);
    NofSpeciesB.resize(NumSpeciesB);

    for(int spec=0; spec<NumSpeciesA; spec++) { 
      Zspec[spec] = tspeciesA(ChargeAttribIndxA,spec);
      NofSpeciesA[spec] = static_cast<int>(tspeciesA(MemberAttribIndxA,spec));
    }
    for(int spec=0; spec<NumSpeciesB; spec++) {
      Qspec[spec] = tspeciesB(ChargeAttribIndxB,spec);
      NofSpeciesB[spec] = static_cast<int>(tspeciesB(MemberAttribIndxB,spec));
    }

    RealType totQ=0.0;
    for(int iat=0; iat<NptclA; iat++)
      totQ+=Zat[iat] = Zspec[PtclA.GroupID[iat]];
    for(int iat=0; iat<NptclB; iat++)
      totQ+=Qat[iat] = Qspec[P.GroupID[iat]];

    if(totQ>numeric_limits<RealType>::epsilon()) {
      LOGMSG("PBCs not yet finished for non-neutral cells");
      OHMMS::Controller->abort();
    }

    ////Test if the box sizes are same (=> kcut same for fixed dimcut)
    kcdifferent = (std::abs(PtclA.Lattice.LR_kc - P.Lattice.LR_kc) > numeric_limits<RealType>::epsilon());
    minkc = std::min(PtclA.Lattice.LR_kc,P.Lattice.LR_kc);

    //AB->initBreakup(*PtclB);
    //initBreakup is called only once
    //AB = LRCoulombSingleton::getHandler(*PtclB);
    AB = LRCoulombSingleton::getHandler(P);
    myConst=evalConsts();
    myRcut=AB->Basis.get_rc();

    if(V0==0) {
      V0 = LRCoulombSingleton::createSpline4RbyVs(AB,myRcut,myGrid);
      if(Vat.size()) {
        app_log() << "  Vat is not empty. Something is wrong" << endl;
        OHMMS::Controller->abort();
      }
      Vat.resize(NptclA,V0);
      Vspec.resize(NumSpeciesA,0);

#ifdef QMC_CUDA
      if (!cloning) {
	V0Spline = new TextureSpline;
	V0Spline->set(V0->data(), V0->size(), V0->grid().rmin(), 
		      V0->grid().rmax());
      }
      SRSplines.resize(NumIonSpecies, V0Spline);
#endif
    }
  }

  void CoulombPBCABTemp::add(int groupID, RadFunctorType* ppot) {
    if(myGrid ==0)
    {
      myGrid = new LinearGrid<RealType>;
      int ng=static_cast<int>(myRcut/1e-3)+1;
      app_log() << "  CoulombPBCABTemp::add \n Setting a linear grid=[0," 
        << myRcut << ") number of grid =" << ng << endl;
      myGrid->set(0,myRcut,ng);
    }

    //add a numerical functor
    if(Vspec[groupID]==0){
      int ng=myGrid->size();
      vector<RealType> v(ng);
      v[0]=0.0;
      for(int ig=1; ig<ng-1; ig++) {
        RealType r=(*myGrid)[ig];
        //need to multiply r for the LR
        v[ig]=r*AB->evaluateLR(r)+ppot->splint(r);
      }
      v[ng-1]=0.0;


      RadFunctorType* rfunc=new RadFunctorType(myGrid,v);
      RealType deriv=(v[1]-v[0])/((*myGrid)[1]-(*myGrid)[0]);
      rfunc->spline(0,deriv,ng-1,0.0);
      Vspec[groupID]=rfunc;
      // Setup CUDA spline
      SRSplines[groupID] = new TextureSpline();
      SRSplines[groupID]->set(rfunc->data(), rfunc->size(), 
			      rfunc->grid().rmin(), rfunc->grid().rmax());
      for(int iat=0; iat<NptclA; iat++) {
        if(PtclA.GroupID[iat]==groupID) Vat[iat]=rfunc;
      }
    }
  }

  CoulombPBCABTemp::Return_t
    CoulombPBCABTemp::evalLR(ParticleSet& P) {
      RealType res=0.0;
      const StructFact& RhoKA(*(PtclA.SK));
      const StructFact& RhoKB(*(P.SK));
      for(int i=0; i<NumSpeciesA; i++) {
        RealType esum=0.0;
        for(int j=0; j<NumSpeciesB; j++) {
          esum += Qspec[j]*AB->evaluate(RhoKA.KLists.kshell, RhoKA.rhok[i],RhoKB.rhok[j]);
        } //speceln
        res += Zspec[i]*esum;
      }//specion
      return res;
    }

  CoulombPBCABTemp::Return_t
    CoulombPBCABTemp::evalSR(ParticleSet& P) 
    {
      const DistanceTableData &d_ab(*P.DistTables[myTableIndex]);
      RealType res=0.0;
      //Loop over distinct eln-ion pairs
      for(int iat=0; iat<NptclA; iat++)
      {
        RealType esum = 0.0;
        RadFunctorType* rVs=Vat[iat];
        for(int nn=d_ab.M[iat], jat=0; nn<d_ab.M[iat+1]; ++nn,++jat) 
        {
          //if(d_ab->r(nn)>=myRcut) continue;
          esum += Qat[jat]*d_ab.rinv(nn)*rVs->splint(d_ab.r(nn));
        }
        //Accumulate pair sums...species charge for atom i.
	res += Zat[iat]*esum;
      }
      return res;
    }

  CoulombPBCABTemp::Return_t
    CoulombPBCABTemp::evalConsts() {
      LRHandlerType::BreakupBasisType &Basis(AB->Basis);
      const Vector<RealType> &coefs(AB->coefs);
      RealType v0_ = Basis.get_rc()*Basis.get_rc()*0.5;
      for(int n=0; n<coefs.size(); n++)
        v0_ -= coefs[n]*Basis.hintr2(n);
      v0_ *= 2.0*TWOPI/Basis.get_CellVolume(); //For charge q1=q2=1

      //Can simplify this if we know a way to get number of particles with each
      //groupID.
      RealType Consts=0.0;
      for(int i=0; i<NumSpeciesA; i++) {
        RealType q=Zspec[i]*NofSpeciesA[i];
        for(int j=0; j<NumSpeciesB; j++) {
          Consts += -v0_*Qspec[j]*NofSpeciesB[j]*q;
        }
      }

      app_log() << "   Constant of PBCAB " << Consts << endl;
      return Consts;
    }


  void
  CoulombPBCABTemp::setupLongRangeGPU()
  {
    StructFact &SK = *(ElecRef.SK);
    Numk = SK.KLists.numk;
    gpu::host_vector<CUDA_PRECISION> kpointsHost(OHMMS_DIM*Numk);
    for (int ik=0; ik<Numk; ik++)
      for (int dim=0; dim<OHMMS_DIM; dim++)
	kpointsHost[ik*OHMMS_DIM+dim] = SK.KLists.kpts_cart[ik][dim];
    kpointsGPU = kpointsHost;
    
    gpu::host_vector<CUDA_PRECISION> FkHost(Numk);
    for (int ik=0; ik<Numk; ik++)
      FkHost[ik] = AB->Fk[ik];
    FkGPU = FkHost;

    // Now compute Rhok for the ions
    RhokIonsGPU.resize(NumIonSpecies);
    
    gpu::host_vector<CUDA_PRECISION> RhokIons_host(2*Numk);
    for (int sp=0; sp<NumIonSpecies; sp++) {
      for (int ik=0; ik < Numk; ik++) {
	PosType k = SK.KLists.kpts_cart[ik];
	RhokIons_host[2*ik+0] = 0.0;
	RhokIons_host[2*ik+1] = 0.0;
	for (int ion=IonFirst[sp]; ion<=IonLast[sp]; ion++) {
	  PosType ipos = SortedIons[ion];
	  RealType phase = dot(k,ipos);
	  double s,c;
	  sincos(phase, &s, &c);
	  RhokIons_host[2*ik+0] += c;
	  RhokIons_host[2*ik+1] += s;
	}
      }
      RhokIonsGPU[sp].set_name ("CoulombPBCABTemp::RhokIonsGPU");
      RhokIonsGPU[sp] = RhokIons_host;
    }
  }

  void 
  CoulombPBCABTemp::addEnergy(MCWalkerConfiguration &W, 
			      vector<RealType> &LocalEnergy)
  {
    vector<Walker_t*> &walkers = W.WalkerList;

    // Short-circuit for constant contribution (e.g. fixed ions)
    // if (!is_active) {
    //   for (int iw=0; iw<walkers.size(); iw++) {
    // 	walkers[iw]->getPropertyBase()[NUMPROPERTIES+myIndex] = Value;
    // 	LocalEnergy[iw] += Value;
    //   }
    //   return;
    // }

    int nw = walkers.size();
    int N = NumElecs;
    if (SumGPU.size() < nw) {
      SumGPU.resize(nw);
      SumHost.resize(nw);
      RhokElecGPU.resize(2*nw*Numk);
      RhokIonsGPU.resize(NumIonSpecies);
      for (int sp=0; sp<NumIonSpecies; sp++)
	RhokIonsGPU.resize(2*Numk);
      SumHost.resize(nw);
      RhoklistGPU.resize(nw);
      RhoklistHost.resize(nw);
    }
    for (int iw=0; iw<nw; iw++) {
      RhoklistHost[iw] = &(RhokElecGPU.data()[2*Numk*iw]);
      SumHost[iw] = 0.0;
    }
    RhoklistGPU = RhoklistHost;
    SumGPU = SumHost;

    // First, do short-range part
    vector<double> esum(nw, 0.0);
    for (int sp=0; sp<NumIonSpecies; sp++) {
      if (SRSplines[sp]) {
	CoulombAB_SR_Sum
	  (W.RList_GPU.data(), N, IGPU.data(), IonFirst[sp], IonLast[sp],
	   SRSplines[sp]->rMax, SRSplines[sp]->NumPoints, 
	   SRSplines[sp]->MyTexture, L.data(), Linv.data(), SumGPU.data(), nw);
	SumHost = SumGPU;
	for (int iw=0; iw<nw; iw++)
	  esum[iw] += Zspec[sp]*Qspec[0]* SumHost[iw];
      }
    }
    
    // Now, do long-range part:
    int first = 0;
    int last  = N-1;
    eval_rhok_cuda(W.RList_GPU.data(), first, last, kpointsGPU.data(), Numk, 
    		   RhoklistGPU.data(), nw);
    
    for (int sp=0; sp<NumIonSpecies; sp++) {
      for (int iw=0; iw<nw; iw++)
	SumHost[iw] = 0.0;
      SumGPU = SumHost;
      eval_vk_sum_cuda(RhoklistGPU.data(), RhokIonsGPU[sp].data(),
		       FkGPU.data(), Numk, SumGPU.data(), nw);
      SumHost = SumGPU;
      for (int iw=0; iw<nw; iw++)
	esum[iw] += Zspec[sp]*Qspec[0]* SumHost[iw];
    }

// #ifdef DEBUG_CUDA_RHOK
//     gpu::host_vector<CUDA_PRECISION> RhokHost;
//     RhokHost = RhokGPU;
//     for (int ik=0; ik<Numk; ik++) {
//       complex<double> rhok(0.0, 0.0);
//       PosType k = PtclRef.SK->KLists.kpts_cart[ik];
//       for (int ir=0; ir<N; ir++) {
//     	PosType r = walkers[0]->R[ir];
//     	double s, c;
//     	double phase = dot(k,r);
//     	sincos(phase, &s, &c);
//     	rhok += complex<double>(c,s);
//       }

      
//       fprintf (stderr, "GPU:   %d   %14.6f  %14.6f\n", 
//     	       ik, RhokHost[2*ik+0], RhokHost[2*ik+1]);
//       fprintf (stderr, "CPU:   %d   %14.6f  %14.6f\n", 
//     	       ik, rhok.real(), rhok.imag());
//     }
// #endif
    
//     for (int sp1=0; sp1<NumSpecies; sp1++)
//       for (int sp2=sp1; sp2<NumSpecies; sp2++) 
//     	eval_vk_sum_cuda(RhoklistsGPU[sp1].data(), RhoklistsGPU[sp2].data(),
//     			 FkGPU.data(), Numk, SumGPU.data(), nw);

    SumHost = SumGPU;
    for (int iw=0; iw<walkers.size(); iw++) {
      // fprintf (stderr, "Energy = %18.6f\n", SumHost[iw]);
      walkers[iw]->getPropertyBase()[NUMPROPERTIES+myIndex] = 
	esum[iw] + myConst;
      LocalEnergy[iw] += esum[iw] + myConst;
    }
  }

}

/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/

