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

  CoulombPBCABTemp::CoulombPBCABTemp(ParticleSet& ions, ParticleSet& elns): 
    PtclA(ions), myConst(0.0), myGrid(0),V0(0), ElecRef(elns), IonRef(ions)
    {
      ReportEngine PRE("CoulombPBCABTemp","CoulombPBCABTemp");
      //Use singleton pattern 
      //AB = new LRHandlerType(ions);
      myTableIndex=elns.addTable(ions);
      initBreakup(elns);
      app_log() << "  Maximum K shell " << AB->MaxKshell << endl;
      app_log() << "  Number of k vectors " << AB->Fk.size() << endl;
    }

  QMCHamiltonianBase* CoulombPBCABTemp::makeClone(ParticleSet& qp, TrialWaveFunction& psi)
  {
    CoulombPBCABTemp* myclone=new CoulombPBCABTemp(PtclA,qp);
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
      return Value = evalLR(P)+evalSR(P)+myConst;
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

  void CoulombPBCABTemp::initBreakup(ParticleSet& P) {
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
    }
#ifdef QMC_CUDA
    // SRSpline.set(V0->data(), V0->size(), V0->grid().rmin(), 
    // 		 V0->grid().rmax());
    // setupLongRangeGPU(P);
#endif
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
  CoulombPBCABTemp::setupLongRangeGPU(ParticleSet &P)
  {
    StructFact &SK = *(P.SK);
    Numk = SK.KLists.numk;
    host_vector<CUDA_PRECISION> kpointsHost(OHMMS_DIM*Numk);
    for (int ik=0; ik<Numk; ik++)
      for (int dim=0; dim<OHMMS_DIM; dim++)
	kpointsHost[ik*OHMMS_DIM+dim] = SK.KLists.kpts_cart[ik][dim];
    kpointsGPU = kpointsHost;
    
    host_vector<CUDA_PRECISION> FkHost(Numk);
    for (int ik=0; ik<Numk; ik++)
      FkHost[ik] = AB->Fk[ik];
    FkGPU = FkHost;
  }

  void 
  CoulombPBCABTemp::addEnergy(vector<Walker_t*> &walkers, 
			      vector<RealType> &LocalEnergy)
  {
    return;
    // Short-circuit for constant contribution (e.g. fixed ions)
    // if (!is_active) {
    //   for (int iw=0; iw<walkers.size(); iw++) {
    // 	walkers[iw]->getPropertyBase()[NUMPROPERTIES+myIndex] = Value;
    // 	LocalEnergy[iw] += Value;
    //   }
    //   return;
    // }

    int nw = walkers.size();
    int N = NumCenters;
    if (RGPU.size() < OHMMS_DIM*nw*N) {
      RGPU.resize(OHMMS_DIM*nw*N);   
      SumGPU.resize(nw);
      RhokGPU.resize(2*nw*Numk*NumSpecies);
      RHost.resize(OHMMS_DIM*nw*N);  SumHost.resize(nw);
      RlistGPU.resize(nw);           RlistHost.resize(nw);
      RhoklistsGPU.resize(NumSpecies);
      RhoklistsHost.resize(NumSpecies);
      for (int sp=0; sp<NumSpecies; sp++) {
	RhoklistsGPU[sp].resize(nw);
	RhoklistsHost[sp].resize(nw);
      }
    }
    for (int iw=0; iw<nw; iw++) {
      for (int sp=0; sp<NumSpecies; sp++) 
	RhoklistsHost[sp][iw] = &(RhokGPU[2*nw*Numk*sp + 2*Numk*iw]);
      RlistHost[iw] = &(RGPU[OHMMS_DIM*N*iw]);
      for (int iat=0; iat<N; iat++)
	for (int dim=0; dim<OHMMS_DIM; dim++)
	  RHost[(iw*N+iat)*OHMMS_DIM + dim] = walkers[iw]->R[iat][dim];
    }
    for (int sp=0; sp<NumSpecies; sp++)
      RhoklistsGPU[sp] = RhoklistsHost[sp];
    RlistGPU = RlistHost;
    RGPU = RHost;  

    // First, do short-range part
    CoulombAA_SR_Sum(RlistGPU.data(), N, SRSpline.rMax, SRSpline.NumPoints, 
		     SRSpline.MyTexture, L.data(), Linv.data(), SumGPU.data(), 
		     nw);
    
    // Now, do long-range part:
    for (int sp=0; sp<NumSpecies; sp++) {
      int first = ElecRef.first(sp);
      int last  = ElecRef.last(sp)-1;
      eval_rhok_cuda(RlistGPU.data(), first, last, 
    		     kpointsGPU.data(), Numk, 
    		     RhoklistsGPU[sp].data(), walkers.size());
    }

#ifdef DEBUG_CUDA_RHOK
    host_vector<CUDA_PRECISION> RhokHost;
    RhokHost = RhokGPU;
    for (int ik=0; ik<Numk; ik++) {
      complex<double> rhok(0.0, 0.0);
      PosType k = PtclRef.SK->KLists.kpts_cart[ik];
      for (int ir=0; ir<N; ir++) {
    	PosType r = walkers[0]->R[ir];
    	double s, c;
    	double phase = dot(k,r);
    	sincos(phase, &s, &c);
    	rhok += complex<double>(c,s);
      }

      
      fprintf (stderr, "GPU:   %d   %14.6f  %14.6f\n", 
    	       ik, RhokHost[2*ik+0], RhokHost[2*ik+1]);
      fprintf (stderr, "CPU:   %d   %14.6f  %14.6f\n", 
    	       ik, rhok.real(), rhok.imag());
    }
#endif
    
    for (int sp1=0; sp1<NumSpecies; sp1++)
      for (int sp2=sp1; sp2<NumSpecies; sp2++) 
    	eval_vk_sum_cuda(RhoklistsGPU[sp1].data(), RhoklistsGPU[sp2].data(),
    			 FkGPU.data(), Numk, SumGPU.data(), nw);

    SumHost = SumGPU;
    for (int iw=0; iw<walkers.size(); iw++) {
      // fprintf (stderr, "Energy = %18.6f\n", SumHost[iw]);
      walkers[iw]->getPropertyBase()[NUMPROPERTIES+myIndex] = 
	SumHost[iw] + myConst;
      LocalEnergy[iw] += SumHost[iw] + myConst;
    }
  }

}

/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/

