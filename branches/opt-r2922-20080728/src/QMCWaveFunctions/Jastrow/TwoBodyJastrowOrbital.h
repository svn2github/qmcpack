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
// -*- C++ -*-
#ifndef QMCPLUSPLUS_COMMON_TWOBODYJASTROW_H
#define QMCPLUSPLUS_COMMON_TWOBODYJASTROW_H
#include "Configuration.h"
#include  <map>
#include "QMCWaveFunctions/OrbitalBase.h"
#include "QMCWaveFunctions/DiffOrbitalBase.h"
#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"

namespace qmcplusplus {

  /** @ingroup OrbitalComponent
   *  @brief Specialization for two-body Jastrow function using multiple functors
   *
   *Each pair-type can have distinct function \f$u(r_{ij})\f$.
   *For electrons, distinct pair correlation functions are used 
   *for spins up-up/down-down and up-down/down-up.
   */ 
  template<class FT>
  class TwoBodyJastrowOrbital: public OrbitalBase {

    const DistanceTableData* d_table;

    int N,NN;
    int NumGroups;
    RealType DiffVal, DiffValSum;
    ParticleAttrib<RealType> U,d2U,curLap,curVal;
    ParticleAttrib<PosType> dU,curGrad;
    RealType *FirstAddressOfdU, *LastAddressOfdU;
    Matrix<int> PairID;

    std::map<std::string,FT*> J2Unique;

  public:

    typedef FT FuncType;

    ///container for the Jastrow functions 
    vector<FT*> F;

    TwoBodyJastrowOrbital(ParticleSet& p) {
      d_table=DistanceTable::add(p);
      init(p);
    }

    ~TwoBodyJastrowOrbital(){ }

    void init(ParticleSet& p) {
      N=p.getTotalNum();
      NN=N*N;
      U.resize(NN+1);
      d2U.resize(NN);
      dU.resize(NN);
      curGrad.resize(N);
      curLap.resize(N);
      curVal.resize(N);

      FirstAddressOfdU = &(dU[0][0]);
      LastAddressOfdU = FirstAddressOfdU + dU.size()*DIM;

      PairID.resize(N,N);
      int nsp=NumGroups=p.groups();
      for(int i=0; i<N; ++i)
	for(int j=0; j<N; ++j) 
	  PairID(i,j) = p.GroupID[i]*nsp+p.GroupID[j];
      F.resize(nsp*nsp,0);
    }

    void addFunc(const string& aname, int ia, int ib, FT* j)
    {
      if(ia==ib)
      {
        if(ia==0)//first time, assign everything
        {
          int ij=0;
          for(int ig=0; ig<NumGroups; ++ig) 
            for(int jg=0; jg<NumGroups; ++jg, ++ij) 
              if(F[ij]==0) F[ij]=j;
        }
      }
      else 
      {
        F[ia*NumGroups+ib]=j;
        if(ia<ib) F[ib*NumGroups+ia]=j; 
      }

      J2Unique[aname]=j;
    }

    //evaluate the distance table with els
    void resetTargetParticleSet(ParticleSet& P) 
    {
      d_table = DistanceTable::add(P);
    }

    /** check in an optimizable parameter
     * @param o a super set of optimizable variables
     */
    void checkInVariables(opt_variables_type& active)
    {
      myVars.clear();
      typename std::map<std::string,FT*>::iterator it(J2Unique.begin()),it_end(J2Unique.end());
      while(it != it_end) 
      {
        (*it).second->checkInVariables(active);
        (*it).second->checkInVariables(myVars);
        ++it;
      }
    }

    /** check out optimizable variables
     */
    void checkOutVariables(const opt_variables_type& active)
    {
      myVars.getIndex(active);
      Optimizable=myVars.is_optimizable();
      typename std::map<std::string,FT*>::iterator it(J2Unique.begin()),it_end(J2Unique.end());
      while(it != it_end) 
      {
        (*it++).second->checkOutVariables(active);
      }
      //if(dPsi) dPsi->checkOutVariables(active);
    }

    ///reset the value of all the unique Two-Body Jastrow functions
    void resetParameters(const opt_variables_type& active)
    { 
      if(!Optimizable) return;
      typename std::map<std::string,FT*>::iterator it(J2Unique.begin()),it_end(J2Unique.end());
      while(it != it_end) 
      {
        (*it++).second->resetParameters(active); 
      }
      //if(dPsi) dPsi->resetParameters(optVariables);
      for(int i=0; i<myVars.size(); ++i)
      {
        int ii=myVars.Index[i];
        if(ii>=0) myVars[i]= active[ii];
      }
    }

    /** print the state, e.g., optimizables */
    void reportStatus(ostream& os)
    {
      typename std::map<std::string,FT*>::iterator it(J2Unique.begin()),it_end(J2Unique.end());
      while(it != it_end) 
      {
        (*it).second->myVars.print(os);
        ++it;
      }
    }


    /** 
     *@param P input configuration containing N particles
     *@param G a vector containing N gradients
     *@param L a vector containing N laplacians
     *@param G returns the gradient \f$G[i]={\bf \nabla}_i J({\bf R})\f$
     *@param L returns the laplacian \f$L[i]=\nabla^2_i J({\bf R})\f$
     *@return \f$exp(-J({\bf R}))\f$
     *@brief While evaluating the value of the Jastrow for a set of
     *particles add the gradient and laplacian contribution of the
     *Jastrow to G(radient) and L(aplacian) for local energy calculations
     *such that \f[ G[i]+={\bf \nabla}_i J({\bf R}) \f] 
     *and \f[ L[i]+=\nabla^2_i J({\bf R}). \f]
     *@note The DistanceTableData contains only distinct pairs of the 
     *particles belonging to one set, e.g., SymmetricDTD.
     */
    ValueType evaluateLog(ParticleSet& P,
		          ParticleSet::ParticleGradient_t& G, 
		          ParticleSet::ParticleLaplacian_t& L) {
      LogValue=0.0;
      RealType dudr, d2udr2;
      PosType gr;
      for(int i=0; i<d_table->size(SourceIndex); i++) {
	for(int nn=d_table->M[i]; nn<d_table->M[i+1]; nn++) {
	  int j = d_table->J[nn];
	  //LogValue -= F[d_table->PairID[nn]]->evaluate(d_table->r(nn), dudr, d2udr2);
	  RealType uij = F[d_table->PairID[nn]]->evaluate(d_table->r(nn), dudr, d2udr2);
          LogValue -= uij;
	  U[i*N+j]=uij; U[j*N+i]=uij; //save for the ratio
	  //multiply 1/r
	  dudr *= d_table->rinv(nn);
	  gr = dudr*d_table->dr(nn);
	  //(d^2 u \over dr^2) + (2.0\over r)(du\over\dr)
	  RealType lap(d2udr2+2.0*dudr);

	  //multiply -1
	  G[i] += gr;
	  G[j] -= gr;
	  L[i] -= lap; 
	  L[j] -= lap; 
	}
      }
      return LogValue;
    }

    ValueType evaluate(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G, 
		       ParticleSet::ParticleLaplacian_t& L) {
      return std::exp(evaluateLog(P,G,L));
    }

    ValueType ratio(ParticleSet& P, int iat) {
      DiffVal=0.0;
      //ValueType d(0.0);
      const int* pairid(PairID[iat]);
      for(int jat=0, ij=iat*N; jat<N; jat++,ij++) {
        if(jat == iat) {
          curVal[jat]=0.0;
        } else {
          curVal[jat]=F[pairid[jat]]->evaluate(d_table->Temp[jat].r1);
          DiffVal += U[ij]-curVal[jat];
          //DiffVal += U[ij]-F[pairid[jat]]->evaluate(d_table->Temp[jat].r1);
        }
      }
      return std::exp(DiffVal);
    }

    /** later merge the loop */
    ValueType ratio(ParticleSet& P, int iat,
		    ParticleSet::ParticleGradient_t& dG,
		    ParticleSet::ParticleLaplacian_t& dL)  {
      register RealType dudr, d2udr2,u;
      register PosType gr;
      DiffVal = 0.0;      
      const int* pairid = PairID[iat];
      for(int jat=0, ij=iat*N; jat<N; jat++,ij++) {
	if(jat==iat) {
	  curVal[jat] = 0.0;curGrad[jat]=0.0; curLap[jat]=0.0;
	} else {
	  curVal[jat] = F[pairid[jat]]->evaluate(d_table->Temp[jat].r1, dudr, d2udr2);
	  dudr *= d_table->Temp[jat].rinv1;
	  curGrad[jat] = -dudr*d_table->Temp[jat].dr1;
	  curLap[jat] = -(d2udr2+2.0*dudr);
	  DiffVal += (U[ij]-curVal[jat]);
	}
      }
      PosType sumg,dg;
      RealType suml=0.0,dl;
      for(int jat=0,ij=iat*N,ji=iat; jat<N; jat++,ij++,ji+=N) {
	sumg += (dg=curGrad[jat]-dU[ij]);
	suml += (dl=curLap[jat]-d2U[ij]);
        dG[jat] -= dg;
	dL[jat] += dl;
      }
      dG[iat] += sumg;
      dL[iat] += suml;     
      return std::exp(DiffVal);
    }

    /** later merge the loop */
    ValueType logRatio(ParticleSet& P, int iat,
		    ParticleSet::ParticleGradient_t& dG,
		    ParticleSet::ParticleLaplacian_t& dL)  {
      register RealType dudr, d2udr2,u;
      register PosType gr;
      DiffVal = 0.0;      
      const int* pairid = PairID[iat];
      for(int jat=0, ij=iat*N; jat<N; jat++,ij++) {
	if(jat==iat) {
	  curVal[jat] = 0.0;curGrad[jat]=0.0; curLap[jat]=0.0;
	} else {
	  curVal[jat] = F[pairid[jat]]->evaluate(d_table->Temp[jat].r1, dudr, d2udr2);
	  dudr *= d_table->Temp[jat].rinv1;
	  curGrad[jat] = -dudr*d_table->Temp[jat].dr1;
	  curLap[jat] = -(d2udr2+2.0*dudr);
	  DiffVal += (U[ij]-curVal[jat]);
	}
      }
      PosType sumg,dg;
      RealType suml=0.0,dl;
      for(int jat=0,ij=iat*N,ji=iat; jat<N; jat++,ij++,ji+=N) {
	sumg += (dg=curGrad[jat]-dU[ij]);
	suml += (dl=curLap[jat]-d2U[ij]);
        dG[jat] -= dg;
	dL[jat] += dl;
      }
      dG[iat] += sumg;
      dL[iat] += suml;     
      return DiffVal;
    }

    inline void restore(int iat) {}

    void acceptMove(ParticleSet& P, int iat) { 
      DiffValSum += DiffVal;
      for(int jat=0,ij=iat*N,ji=iat; jat<N; jat++,ij++,ji+=N) {
	dU[ij]=curGrad[jat]; 
	//dU[ji]=-1.0*curGrad[jat];
	dU[ji]=curGrad[jat]*-1.0;
	d2U[ij]=d2U[ji] = curLap[jat];
	U[ij] =  U[ji] = curVal[jat];
      }
    }


    inline void update(ParticleSet& P, 		
		       ParticleSet::ParticleGradient_t& dG, 
		       ParticleSet::ParticleLaplacian_t& dL,
		       int iat) {
      DiffValSum += DiffVal;
      GradType sumg,dg;
      ValueType suml=0.0,dl;
      for(int jat=0,ij=iat*N,ji=iat; jat<N; jat++,ij++,ji+=N) {
	sumg += (dg=curGrad[jat]-dU[ij]);
	suml += (dl=curLap[jat]-d2U[ij]);
	dU[ij]=curGrad[jat]; 
	//dU[ji]=-1.0*curGrad[jat];
	dU[ji]=curGrad[jat]*-1.0;
	d2U[ij]=d2U[ji] = curLap[jat];
	U[ij] =  U[ji] = curVal[jat];
        dG[jat] -= dg;
	dL[jat] += dl;
      }
      dG[iat] += sumg;
      dL[iat] += suml;     
    }


    inline void evaluateLogAndStore(ParticleSet& P, 
		       ParticleSet::ParticleGradient_t& dG, 
		       ParticleSet::ParticleLaplacian_t& dL) {
      RealType dudr, d2udr2,u;
      LogValue=0.0;
      GradType gr;
      for(int i=0; i<d_table->size(SourceIndex); i++) {
	for(int nn=d_table->M[i]; nn<d_table->M[i+1]; nn++) {
	  int j = d_table->J[nn];
	  //ValueType sumu = F.evaluate(d_table->r(nn));
	  u = F[d_table->PairID[nn]]->evaluate(d_table->r(nn), dudr, d2udr2);
	  LogValue -= u;
	  dudr *= d_table->rinv(nn);
	  gr = dudr*d_table->dr(nn);
	  //(d^2 u \over dr^2) + (2.0\over r)(du\over\dr)\f$
	  RealType lap = d2udr2+2.0*dudr;
	  int ij = i*N+j, ji=j*N+i;
	  U[ij]=u; U[ji]=u;
	  //dU[ij] = gr; dU[ji] = -1.0*gr;
	  dU[ij] = gr; dU[ji] = gr*-1.0;
	  d2U[ij] = -lap; d2U[ji] = -lap;

	  //add gradient and laplacian contribution
	  dG[i] += gr;
	  dG[j] -= gr;
	  dL[i] -= lap; 
	  dL[j] -= lap; 
	}
      }
    }

    inline ValueType registerData(ParticleSet& P, PooledData<RealType>& buf){

      evaluateLogAndStore(P,P.G,P.L);
      //LogValue=0.0;
      //RealType dudr, d2udr2,u;
      //GradType gr;
      //PairID.resize(d_table->size(SourceIndex),d_table->size(SourceIndex));
      //int nsp=P.groups();
      //for(int i=0; i<d_table->size(SourceIndex); i++)
      //  for(int j=0; j<d_table->size(SourceIndex); j++) 
      //    PairID(i,j) = P.GroupID[i]*nsp+P.GroupID[j];

      //for(int i=0; i<d_table->size(SourceIndex); i++) {
      //  for(int nn=d_table->M[i]; nn<d_table->M[i+1]; nn++) {
      //    int j = d_table->J[nn];
      //    //ValueType sumu = F.evaluate(d_table->r(nn));
      //    u = F[d_table->PairID[nn]]->evaluate(d_table->r(nn), dudr, d2udr2);
      //    LogValue -= u;
      //    dudr *= d_table->rinv(nn);
      //    gr = dudr*d_table->dr(nn);
      //    //(d^2 u \over dr^2) + (2.0\over r)(du\over\dr)\f$
      //    RealType lap = d2udr2+2.0*dudr;
      //    int ij = i*N+j, ji=j*N+i;
      //    U[ij]=u; U[ji]=u;
      //    //dU[ij] = gr; dU[ji] = -1.0*gr;
      //    dU[ij] = gr; dU[ji] = gr*-1.0;
      //    d2U[ij] = -lap; d2U[ji] = -lap;

      //    //add gradient and laplacian contribution
      //    P.G[i] += gr;
      //    P.G[j] -= gr;
      //    P.L[i] -= lap; 
      //    P.L[j] -= lap; 
      //  }
      //}

      U[NN]= real(LogValue);
      buf.add(U.begin(), U.end());
      buf.add(d2U.begin(), d2U.end());
      buf.add(FirstAddressOfdU,LastAddressOfdU);

      return LogValue;
    }

    inline ValueType updateBuffer(ParticleSet& P, PooledData<RealType>& buf,
        bool fromscratch=false){
      evaluateLogAndStore(P,P.G,P.L);
      //RealType dudr, d2udr2,u;
      //LogValue=0.0;
      //GradType gr;
      //PairID.resize(d_table->size(SourceIndex),d_table->size(SourceIndex));
      //int nsp=P.groups();
      //for(int i=0; i<d_table->size(SourceIndex); i++)
      //  for(int j=0; j<d_table->size(SourceIndex); j++) 
      //    PairID(i,j) = P.GroupID[i]*nsp+P.GroupID[j];

      //for(int i=0; i<d_table->size(SourceIndex); i++) {
      //  for(int nn=d_table->M[i]; nn<d_table->M[i+1]; nn++) {
      //    int j = d_table->J[nn];
      //    //ValueType sumu = F.evaluate(d_table->r(nn));
      //    u = F[d_table->PairID[nn]]->evaluate(d_table->r(nn), dudr, d2udr2);
      //    LogValue -= u;
      //    dudr *= d_table->rinv(nn);
      //    gr = dudr*d_table->dr(nn);
      //    //(d^2 u \over dr^2) + (2.0\over r)(du\over\dr)\f$
      //    RealType lap = d2udr2+2.0*dudr;
      //    int ij = i*N+j, ji=j*N+i;
      //    U[ij]=u; U[ji]=u;
      //    //dU[ij] = gr; dU[ji] = -1.0*gr;
      //    dU[ij] = gr; dU[ji] = gr*-1.0;
      //    d2U[ij] = -lap; d2U[ji] = -lap;

      //    //add gradient and laplacian contribution
      //    P.G[i] += gr;
      //    P.G[j] -= gr;
      //    P.L[i] -= lap; 
      //    P.L[j] -= lap; 
      //  }
      //}

      U[NN]= real(LogValue);
      buf.put(U.begin(), U.end());
      buf.put(d2U.begin(), d2U.end());
      buf.put(FirstAddressOfdU,LastAddressOfdU);
      return LogValue;
    }
    
    inline void copyFromBuffer(ParticleSet& P, PooledData<RealType>& buf) {
      buf.get(U.begin(), U.end());
      buf.get(d2U.begin(), d2U.end());
      buf.get(FirstAddressOfdU,LastAddressOfdU);
      DiffValSum=0.0;
    }

    inline ValueType evaluate(ParticleSet& P, PooledData<RealType>& buf) {
      RealType x = (U[NN] += DiffValSum);
      buf.put(U.begin(), U.end());
      buf.put(d2U.begin(), d2U.end());
      buf.put(FirstAddressOfdU,LastAddressOfdU);
      return std::exp(x); 
    }

    OrbitalBasePtr makeClone(ParticleSet& tqp) const
    {
      TwoBodyJastrowOrbital<FT>* j2copy=new TwoBodyJastrowOrbital<FT>(tqp);
      map<const FT*,FT*> fcmap;
      for(int ig=0; ig<NumGroups; ++ig)
        for(int jg=ig; jg<NumGroups; ++jg)
        {
          int ij=ig*NumGroups+jg;
          if(F[ij]==0) continue;
          typename map<const FT*,FT*>::iterator fit=fcmap.find(F[ij]);
          if(fit == fcmap.end())
          {
            FT* fc=new FT(*F[ij]);
            stringstream aname;
            aname<<ig<<jg;
            j2copy->addFunc(aname.str(),ig,jg,fc);
            fcmap[F[ij]]=fc;
          }
        }
      return j2copy;
    }

    void copyFrom(const OrbitalBase& old)
    {
      //nothing to do
    }
  };
}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/

