//////////////////////////////////////////////////////////////////
// (c) Copyright 1998-2002,2003- by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Jeongnim Kim
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

#include "QMCWaveFunctions/Fermion/DiracDeterminantBase.h"
#include "Numerics/DeterminantOperators.h"
#include "Numerics/OhmmsBlas.h"

namespace qmcplusplus {

  /** constructor
   *@param spos the single-particle orbital set
   *@param first index of the first particle
   */
  DiracDeterminantBase::DiracDeterminantBase(SPOSetBasePtr const &spos, int first): 
    NP(0), Phi(spos), FirstIndex(first) 
  {
    Optimizable=false;
    OrbitalName="DiracDeterminantBase";
  }

  ///default destructor
  DiracDeterminantBase::~DiracDeterminantBase() {}

  DiracDeterminantBase& DiracDeterminantBase::operator=(const DiracDeterminantBase& s) {
    NP=0;
    resize(s.NumPtcls, s.NumOrbitals);
    return *this;
  }

  /** set the index of the first particle in the determinant and reset the size of the determinant
   *@param first index of first particle
   *@param nel number of particles in the determinant
   */
  void DiracDeterminantBase::set(int first, int nel) {
    FirstIndex = first;
    resize(nel,nel);
  }


  ///reset the size: with the number of particles and number of orbtials
  void DiracDeterminantBase::resize(int nel, int morb) {
    int norb=morb;
    if(norb <= 0) norb = nel; // for morb == -1 (default)
    psiM.resize(nel,norb);
    dpsiM.resize(nel,norb);
    d2psiM.resize(nel,norb);
    psiM_temp.resize(nel,norb);
    dpsiM_temp.resize(nel,norb);
    d2psiM_temp.resize(nel,norb);
    psiMinv.resize(nel,norb);
    psiV.resize(norb);
    WorkSpace.resize(nel);
    Pivot.resize(nel);
    LastIndex = FirstIndex + nel;
    NumPtcls=nel;
    NumOrbitals=norb;
  }

  DiracDeterminantBase::ValueType 
    DiracDeterminantBase::registerData(ParticleSet& P, PooledData<RealType>& buf) 
    {

    if(NP == 0) {//first time, allocate once
      //int norb = cols();
      dpsiV.resize(NumOrbitals);
      d2psiV.resize(NumOrbitals);
      workV1.resize(NumOrbitals);
      workV2.resize(NumOrbitals);
      NP=P.getTotalNum();
      myG.resize(NP);
      myL.resize(NP);
      myG_temp.resize(NP);
      myL_temp.resize(NP);
      FirstAddressOfG = &myG[0][0];
      LastAddressOfG = FirstAddressOfG + NP*DIM;
      FirstAddressOfdV = &(dpsiM(0,0)[0]); //(*dpsiM.begin())[0]);
      LastAddressOfdV = FirstAddressOfdV + NumPtcls*NumOrbitals*DIM;
    }

    myG=0.0;
    myL=0.0;

    //ValueType x=evaluate(P,myG,myL); 
    LogValue=evaluateLog(P,myG,myL); 

    P.G += myG;
    P.L += myL;

    //add the data: determinant, inverse, gradient and laplacians
    buf.add(psiM.first_address(),psiM.last_address());
    buf.add(FirstAddressOfdV,LastAddressOfdV);
    buf.add(d2psiM.first_address(),d2psiM.last_address());
    buf.add(myL.first_address(), myL.last_address());
    buf.add(FirstAddressOfG,LastAddressOfG);
    buf.add(LogValue);
    buf.add(PhaseValue);

    return LogValue;
  }

  DiracDeterminantBase::ValueType DiracDeterminantBase::updateBuffer(ParticleSet& P, 
      PooledData<RealType>& buf, bool fromscratch) {

    myG=0.0;
    myL=0.0;

    if(fromscratch)
      LogValue=evaluateLog(P,myG,myL);
    else
    {
      Phi->evaluate(P, FirstIndex, LastIndex, psiM_temp,dpsiM, d2psiM);    
      if(NumPtcls==1) {
        ValueType y=1.0/psiM_temp(0,0);
        psiM(0,0)=y;
        GradType rv = y*dpsiM(0,0);
        myG(FirstIndex) += rv;
        myL(FirstIndex) += y*d2psiM(0,0) - dot(rv,rv);
      } else {
        const ValueType* restrict yptr=psiM.data();
        const ValueType* restrict d2yptr=d2psiM.data();
        const GradType* restrict dyptr=dpsiM.data();
        for(int i=0, iat=FirstIndex; i<NumPtcls; i++, iat++) 
        {
          GradType rv;
          ValueType lap=0.0;
          for(int j=0; j<NumOrbitals; j++,yptr++) {
            rv += *yptr * *dyptr++;
            lap += *yptr * *d2yptr++;
          }
          myG(iat) += rv;
          myL(iat) += lap - dot(rv,rv);
        }
      }
    }

    P.G += myG;
    P.L += myL;

    buf.put(psiM.first_address(),psiM.last_address());
    buf.put(FirstAddressOfdV,LastAddressOfdV);
    buf.put(d2psiM.first_address(),d2psiM.last_address());
    buf.put(myL.first_address(), myL.last_address());
    buf.put(FirstAddressOfG,LastAddressOfG);
    buf.put(LogValue);
    buf.put(PhaseValue);

    return LogValue;
  }

  void DiracDeterminantBase::copyFromBuffer(ParticleSet& P, PooledData<RealType>& buf) {

    buf.get(psiM.first_address(),psiM.last_address());
    buf.get(FirstAddressOfdV,LastAddressOfdV);
    buf.get(d2psiM.first_address(),d2psiM.last_address());
    buf.get(myL.first_address(), myL.last_address());
    buf.get(FirstAddressOfG,LastAddressOfG);
    buf.get(LogValue);
    buf.get(PhaseValue);

    //re-evaluate it for testing
    //Phi.evaluate(P, FirstIndex, LastIndex, psiM, dpsiM, d2psiM);
    //CurrentDet = Invert(psiM.data(),NumPtcls,NumOrbitals);
    //need extra copy for gradient/laplacian calculations without updating it
    psiM_temp = psiM;
    dpsiM_temp = dpsiM;
    d2psiM_temp = d2psiM;
  }

  /** dump the inverse to the buffer
  */
  void DiracDeterminantBase::dumpToBuffer(ParticleSet& P, PooledData<RealType>& buf) {
    APP_ABORT("DiracDeterminantBase::dumpToBuffer");
    buf.add(psiM.first_address(),psiM.last_address());
  }

  /** copy the inverse from the buffer
  */
  void DiracDeterminantBase::dumpFromBuffer(ParticleSet& P, PooledData<RealType>& buf) {
    APP_ABORT("DiracDeterminantBase::dumpFromBuffer");
    buf.get(psiM.first_address(),psiM.last_address());
  }

  /** return the ratio only for the  iat-th partcle move
   * @param P current configuration
   * @param iat the particle thas is being moved
   */
  DiracDeterminantBase::ValueType DiracDeterminantBase::ratio(ParticleSet& P, int iat) {
    UseRatioOnly=true;
    WorkingIndex = iat-FirstIndex;
    Phi->evaluate(P, iat, psiV);
#ifdef DIRAC_USE_BLAS
    return curRatio = BLAS::dot(NumOrbitals,psiM[iat-FirstIndex],&psiV[0]);
#else
    return curRatio = DetRatio(psiM, psiV.begin(),iat-FirstIndex);
#endif
  }

  /** return the ratio
   * @param P current configuration
   * @param iat particle whose position is moved
   * @param dG differential Gradients
   * @param dL differential Laplacians
   *
   * Data member *_temp contain the data assuming that the move is accepted
   * and are used to evaluate differential Gradients and Laplacians.
   */
  DiracDeterminantBase::ValueType DiracDeterminantBase::ratio(ParticleSet& P, int iat,
      ParticleSet::ParticleGradient_t& dG, 
      ParticleSet::ParticleLaplacian_t& dL) {
    UseRatioOnly=false;
    Phi->evaluate(P, iat, psiV, dpsiV, d2psiV);
    WorkingIndex = iat-FirstIndex;

#ifdef DIRAC_USE_BLAS
    curRatio = BLAS::dot(NumOrbitals,psiM_temp[WorkingIndex],&psiV[0]);
#else
    curRatio= DetRatio(psiM_temp, psiV.begin(),WorkingIndex);
#endif

    if(abs(curRatio)<numeric_limits<RealType>::epsilon()) 
    {
      UseRatioOnly=true;//do not update with restore
      return 0.0;
    }

    //update psiM_temp with the row substituted
    DetUpdate(psiM_temp,psiV,workV1,workV2,WorkingIndex,curRatio);

    //update dpsiM_temp and d2psiM_temp 
    for(int j=0; j<NumOrbitals; j++) {
      dpsiM_temp(WorkingIndex,j)=dpsiV[j];
      d2psiM_temp(WorkingIndex,j)=d2psiV[j];
    }

    int kat=FirstIndex;

    const ValueType* restrict yptr=psiM_temp.data();
    const ValueType* restrict d2yptr=d2psiM_temp.data();
    const GradType* restrict dyptr=dpsiM_temp.data();
    for(int i=0; i<NumPtcls; i++,kat++) {
      //This mimics gemm with loop optimization
      GradType rv;
      ValueType lap=0.0;
      for(int j=0; j<NumOrbitals; j++,yptr++) {
        rv += *yptr * *dyptr++;
        lap += *yptr * *d2yptr++;
      }

      //using inline dot functions
      //GradType rv=dot(psiM_temp[i],dpsiM_temp[i],NumOrbitals);
      //ValueType lap=dot(psiM_temp[i],d2psiM_temp[i],NumOrbitals);

      //Old index: This is not pretty
      //GradType rv =psiM_temp(i,0)*dpsiM_temp(i,0);
      //ValueType lap=psiM_temp(i,0)*d2psiM_temp(i,0);
      //for(int j=1; j<NumOrbitals; j++) {
      //  rv += psiM_temp(i,j)*dpsiM_temp(i,j);
      //  lap += psiM_temp(i,j)*d2psiM_temp(i,j);
      //}
      lap -= dot(rv,rv);
      dG[kat] += rv - myG[kat];  myG_temp[kat]=rv;
      dL[kat] += lap -myL[kat];  myL_temp[kat]=lap;
    }

    return curRatio;
  }

  DiracDeterminantBase::ValueType DiracDeterminantBase::logRatio(ParticleSet& P, int iat,
      ParticleSet::ParticleGradient_t& dG, 
      ParticleSet::ParticleLaplacian_t& dL) {
    APP_ABORT("  logRatio is not allowed");
    //THIS SHOULD NOT BE CALLED
    ValueType r=ratio(P,iat,dG,dL);
    return LogValue = evaluateLogAndPhase(r,PhaseValue);
  }


  /** move was accepted, update the real container
  */
  void DiracDeterminantBase::acceptMove(ParticleSet& P, int iat) 
  {
    PhaseValue += evaluatePhase(curRatio);
    LogValue +=std::log(std::abs(curRatio));
    //CurrentDet *= curRatio;
    if(UseRatioOnly) 
    {
      DetUpdate(psiM,psiV,workV1,workV2,WorkingIndex,curRatio);
    } 
    else 
    {
      myG = myG_temp;
      myL = myL_temp;
      psiM = psiM_temp;
      std::copy(dpsiV.begin(),dpsiV.end(),dpsiM[WorkingIndex]);
      std::copy(d2psiV.begin(),d2psiV.end(),d2psiM[WorkingIndex]);
      //for(int j=0; j<NumOrbitals; j++) {
      //  dpsiM(WorkingIndex,j)=dpsiV[j];
      //  d2psiM(WorkingIndex,j)=d2psiV[j];
      //}
    }
    curRatio=1.0;
  }

  /** move was rejected. copy the real container to the temporary to move on
  */
  void DiracDeterminantBase::restore(int iat) {
    if(!UseRatioOnly) {
      psiM_temp = psiM;
      std::copy(dpsiM[WorkingIndex],dpsiM[WorkingIndex+1],dpsiM_temp[WorkingIndex]);
      std::copy(d2psiM[WorkingIndex],d2psiM[WorkingIndex+1],d2psiM_temp[WorkingIndex]);
      //for(int j=0; j<NumOrbitals; j++) {
      //  dpsiM_temp(WorkingIndex,j)=dpsiM(WorkingIndex,j);
      //  d2psiM_temp(WorkingIndex,j)=d2psiM(WorkingIndex,j);
      //}
    }
    curRatio=1.0;
  }

  void DiracDeterminantBase::update(ParticleSet& P, 
      ParticleSet::ParticleGradient_t& dG, 
      ParticleSet::ParticleLaplacian_t& dL,
      int iat) {

    DetUpdate(psiM,psiV,workV1,workV2,WorkingIndex,curRatio);
    for(int j=0; j<NumOrbitals; j++) {
      dpsiM(WorkingIndex,j)=dpsiV[j];
      d2psiM(WorkingIndex,j)=d2psiV[j];
    }

    int kat=FirstIndex;
    for(int i=0; i<NumPtcls; i++,kat++) {
      GradType rv=dot(psiM[i],dpsiM[i],NumOrbitals);
      ValueType lap=dot(psiM[i],d2psiM[i],NumOrbitals);
      //GradType rv =psiM(i,0)*dpsiM(i,0);
      //ValueType lap=psiM(i,0)*d2psiM(i,0);
      //for(int j=1; j<NumOrbitals; j++) {
      //  rv += psiM(i,j)*dpsiM(i,j);
      //  lap += psiM(i,j)*d2psiM(i,j);
      //}
      lap -= dot(rv,rv);
      dG[kat] += rv - myG[kat]; myG[kat]=rv;
      dL[kat] += lap -myL[kat]; myL[kat]=lap;
    }

    PhaseValue += evaluatePhase(curRatio);
    LogValue +=std::log(std::abs(curRatio));
    curRatio=1.0;
  }

  DiracDeterminantBase::ValueType 
    DiracDeterminantBase::evaluateLog(ParticleSet& P, PooledData<RealType>& buf) 
    {
      buf.put(psiM.first_address(),psiM.last_address());
      buf.put(FirstAddressOfdV,LastAddressOfdV);
      buf.put(d2psiM.first_address(),d2psiM.last_address());
      buf.put(myL.first_address(), myL.last_address());
      buf.put(FirstAddressOfG,LastAddressOfG);
      buf.put(LogValue);
      buf.put(PhaseValue);
      return LogValue;
    }


  /** Calculate the value of the Dirac determinant for particles
   *@param P input configuration containing N particles
   *@param G a vector containing N gradients
   *@param L a vector containing N laplacians
   *@return the value of the determinant
   *
   *\f$ (first,first+nel). \f$  Add the gradient and laplacian 
   *contribution of the determinant to G(radient) and L(aplacian)
   *for local energy calculations.
   */ 
  DiracDeterminantBase::ValueType
    DiracDeterminantBase::evaluate(ParticleSet& P, 
        ParticleSet::ParticleGradient_t& G, 
        ParticleSet::ParticleLaplacian_t& L){

      APP_ABORT("  DiracDeterminantBase::evaluate is distabled");

      Phi->evaluate(P, FirstIndex, LastIndex, psiM,dpsiM, d2psiM);

      ValueType CurrentDet;
      if(NumPtcls==1) {
        CurrentDet=psiM(0,0);
        ValueType y=1.0/CurrentDet;
        psiM(0,0)=y;
        GradType rv = y*dpsiM(0,0);
        G(FirstIndex) += rv;
        L(FirstIndex) += y*d2psiM(0,0) - dot(rv,rv);
      } else {
        CurrentDet = Invert(psiM.data(),NumPtcls,NumOrbitals, WorkSpace.data(), Pivot.data());
        //CurrentDet = Invert(psiM.data(),NumPtcls,NumOrbitals);
        
        const ValueType* restrict yptr=psiM.data();
        const ValueType* restrict d2yptr=d2psiM.data();
        const GradType* restrict dyptr=dpsiM.data();
        for(int i=0, iat=FirstIndex; i<NumPtcls; i++, iat++) {
          GradType rv;
          ValueType lap=0.0;
          for(int j=0; j<NumOrbitals; j++,yptr++) {
            rv += *yptr * *dyptr++;
            lap += *yptr * *d2yptr++;
          }
          //Old index
          //    GradType rv = psiM(i,0)*dpsiM(i,0);
          //    ValueType lap=psiM(i,0)*d2psiM(i,0);
          //    for(int j=1; j<NumOrbitals; j++) {
          //      rv += psiM(i,j)*dpsiM(i,j);
          //      lap += psiM(i,j)*d2psiM(i,j);
          //    }
          G(iat) += rv;
          L(iat) += lap - dot(rv,rv);
        }
      }
      return CurrentDet;
    }


  DiracDeterminantBase::ValueType
    DiracDeterminantBase::evaluateLog(ParticleSet& P, 
        ParticleSet::ParticleGradient_t& G, 
        ParticleSet::ParticleLaplacian_t& L)
    {

      Phi->evaluate(P, FirstIndex, LastIndex, psiM,dpsiM, d2psiM);

      if(NumPtcls==1) 
      {
        //CurrentDet=psiM(0,0);
        ValueType det=psiM(0,0);
        ValueType y=1.0/det;
        psiM(0,0)=y;
        GradType rv = y*dpsiM(0,0);
        G(FirstIndex) += rv;
        L(FirstIndex) += y*d2psiM(0,0) - dot(rv,rv);
        LogValue = evaluateLogAndPhase(det,PhaseValue);
      } else {
        LogValue=InvertWithLog(psiM.data(),NumPtcls,NumOrbitals,WorkSpace.data(),Pivot.data(),PhaseValue);
        const ValueType* restrict yptr=psiM.data();
        const ValueType* restrict d2yptr=d2psiM.data();
        const GradType* restrict dyptr=dpsiM.data();
        for(int i=0, iat=FirstIndex; i<NumPtcls; i++, iat++) {
          GradType rv;
          ValueType lap=0.0;
          for(int j=0; j<NumOrbitals; j++,yptr++) {
            rv += *yptr * *dyptr++;
            lap += *yptr * *d2yptr++;
          }
          G(iat) += rv;
          L(iat) += lap - dot(rv,rv);
        }
      }
      return LogValue;
    }

  OrbitalBasePtr DiracDeterminantBase::makeClone(ParticleSet& tqp) const
  {
    APP_ABORT(" Cannot use DiracDeterminantBase::makeClone");
    return 0;
    //SPOSetBase* sposclone=Phi->makeClone();
    //DiracDeterminantBase* dclone= new DiracDeterminantBase(sposclone);
    //dclone->set(FirstIndex,LastIndex-FirstIndex);
    //dclone->resetTargetParticleSet(tqp);
    //return dclone;
  }

  DiracDeterminantBase::DiracDeterminantBase(const DiracDeterminantBase& s): 
    OrbitalBase(s), NP(0),Phi(s.Phi),FirstIndex(s.FirstIndex)
  {
    this->resize(s.NumPtcls,s.NumOrbitals);
  }

  SPOSetBasePtr  DiracDeterminantBase::clonePhi() const
  {
    return Phi->makeClone();
  }

  /////////////////////////////////////
  // Vectorized evaluation functions //
  /////////////////////////////////////
  void 
  DiracDeterminantBase::update (vector<Walker_t*> &walkers, int iat)
  {
    // cerr << "AOffset         = " << AOffset << endl;
    // cerr << "AinvOffset      = " << AinvOffset << endl;
    // cerr << "newRowOffset    = " << newRowOffset << endl;
    // cerr << "AinvDeltaOffset = " << AinvDeltaOffset << endl;
    // cerr << "AinvColkOffset  = " << AinvColkOffset << endl;

    if (AList.size() < walkers.size())
      resizeLists(walkers.size());
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t::cuda_Buffer_t &data = walkers[iw]->cuda_DataSet;
      AList[iw]         =  &(data[AOffset]);
      AinvList[iw]      =  &(data[AinvOffset]);
      newRowList[iw]    =  &(data[newRowOffset]);
      AinvDeltaList[iw] =  &(data[AinvDeltaOffset]);
      AinvColkList[iw]  =  &(data[AinvColkOffset]);
    }
    // Copy pointers to the GPU
    AList_d         = AList;
    AinvList_d      = AinvList;
    newRowList_d    = newRowList;
    AinvDeltaList_d = AinvDeltaList;
    AinvColkList_d  = AinvColkList;
    // Call kernel wrapper function
    update_inverse_cuda(&(AList_d[0]),&(AinvList_d[0]), &(newRowList_d[0]),
    			&(AinvDeltaList_d[0]), &(AinvColkList_d[0]),
    			NumPtcls, NumPtcls, iat-FirstIndex, walkers.size());

#ifdef DEBUG_CUDA
    
    float Ainv[NumPtcls][NumPtcls], A[NumPtcls][NumPtcls];
    float new_row[NumPtcls], Ainv_delta[NumPtcls];
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t::cuda_Buffer_t &data = walkers[iw]->cuda_DataSet;
      cudaMemcpy (A, &(data[AOffset]),
		  NumPtcls*NumPtcls*sizeof(CudaValueType), cudaMemcpyDeviceToHost);
      cudaMemcpy (Ainv, &(data[AinvOffset]),
		  NumPtcls*NumPtcls*sizeof(CudaValueType), cudaMemcpyDeviceToHost);
      cudaMemcpy (new_row, &(data[newRowOffset]),
		  NumPtcls*sizeof(CudaValueType), cudaMemcpyDeviceToHost);

      // for (int i=0; i<NumPtcls; i++) 
      //  	cerr << "new_row(" << i << ") = " << new_row[i] 
      // 	     << "  old_row = " << A[iat-FirstIndex][i] << endl;

      // float Ainv_k[NumPtcls];
      // for (int i=0; i<NumPtcls; i++) {
      // 	Ainv_delta[i] = 0.0f;
      // 	Ainv_k[i] = Ainv[i][iat-FirstIndex];
      // 	for (int j=0; j<NumPtcls; j++)
      // 	  Ainv_delta[i] += Ainv[j][i] * (new_row[j] - A[(iat-FirstIndex)][j]);
      // }
      // double prefact = 1.0/(1.0+Ainv_delta[iat-FirstIndex]);
      // for (int i=0; i<NumPtcls; i++)
      // 	for (int j=0; j<NumPtcls; j++)
      // 	  Ainv[i][j] += prefact * Ainv_delta[j]*Ainv_k[i];
      // for (int j=0; j<NumPtcls; j++)
      // 	A[iat-FirstIndex][j] = new_row[j];

      for (int i=0; i<NumPtcls; i++) 
	for (int j=0; j<NumPtcls; j++) {
	  float val = 0.0;
	  for (int k=0; k<NumPtcls; k++)
	    val += Ainv[i][k]*A[k][j];
	  if (i==j && (std::fabs(val-1.0) > 1.0e-2))
	    cerr << "Error in inverse at (i,j) = (" << i << ", " << j 
		 << ")  val = " << val << "  walker = " << iw << endl;
	  else if ((i!=j) && (std::fabs(val) > 1.0e-2))
	    cerr << "Error in inverse at (i,j) = (" << i << ", " << j 
		 << ")  val = " << val << "  walker = " << iw << endl;
	}
    }
#endif
    
    // Copy gradLapl into matrices
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t::cuda_Buffer_t &data = walkers[iw]->cuda_DataSet;
      int off = 4*(iat-FirstIndex)*NumOrbitals;
      cudaMemcpy (&(data[gradLaplOffset+off]), &(data[newGradLaplOffset]),
    		  4*NumOrbitals*sizeof(CudaValueType), cudaMemcpyDeviceToDevice);
    }
  }
  


  void 
  DiracDeterminantBase::addLog (vector<Walker_t*> &walkers, vector<RealType> &logPsi)
  {
    app_log() << "addLog called.  NumPtcls = " << NumPtcls 
	      << "   NumOrbitals = " << NumOrbitals << endl;
    if (AList.size() < walkers.size())
      resizeLists(walkers.size());

    // Fill in the A matrix row by row   
    for (int iat=FirstIndex; iat<LastIndex; iat++) {
      int off = (iat-FirstIndex)*NumOrbitals;
      for (int iw=0; iw<walkers.size(); iw++) {
	Walker_t::cuda_Buffer_t& data = walkers[iw]->cuda_DataSet;
	newRowList[iw]    =  &(data[AOffset+off]);
      }
      newRowList_d = newRowList;
      Phi->evaluate (walkers, iat, newRowList_d);
    }
    // Now, compute determinant
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t::cuda_Buffer_t& data = walkers[iw]->cuda_DataSet;
      // HACK HACK HACK
      host_vector<CUDA_PRECISION> host_data(data);
      host_data = data;
      Vector<double> A(NumPtcls*NumOrbitals);
      for (int i=0; i<NumPtcls*NumOrbitals; i++)
	A[i] = host_data[AOffset+i];
      logPsi[iw] += std::log(Invert(A.data(), NumPtcls, NumOrbitals));
      int N = NumPtcls;
      bool passed = true;

      for (int i=0; i<NumPtcls*NumOrbitals; i++)
	host_data[AinvOffset+i] = A[i];
      data = host_data;

      for (int i=0; i<N; i++)
	for (int j=0; j<N; j++) {
	  double val = 0.0;
	  for (int k=0; k<N; k++) {
	    double aval = host_data[AOffset+i*N+k];
	    double ainv = host_data[AinvOffset+k*N+j];
	    val += aval * ainv;
	  }
	  if (i == j) {
	    if (std::fabs(val - 1.0) > 1.0e-3) {
	      app_error() << "Error in inverse, (i,j) = " << i << ", " << j << ".\n";
	      passed = false;
	    }
	  }
	  else
	    if (std::fabs(val) > 1.0e-3) {
	      app_error() << "Error in inverse, (i,j) = " << i << ", " << j << ".\n";
	      passed = false;
	    }
	
	}
      if (!passed)
	app_log() << (passed ? "Passed " : "Failed " ) << "inverse test.\n";
    }


  }

  void 
  DiracDeterminantBase::addGradient(vector<Walker_t*> &walkers, int iat,
				    vector<GradType> &grad)
  {
    cerr << "DiracDeterminantBase::addGradient.\n";
  }

  void DiracDeterminantBase::ratio (vector<Walker_t*> &walkers, 
				    int iat, vector<PosType> &new_pos,
				    vector<ValueType> &psi_ratios)
  {
    if (AList.size() < walkers.size())
      resizeLists(walkers.size());

    // First evaluate orbitals
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t::cuda_Buffer_t& data = walkers[iw]->cuda_DataSet;
      AinvList[iw]      =  &(data[AinvOffset]);
      newRowList[iw]    =  &(data[newRowOffset]);
    }
    newRowList_d = newRowList;
    Phi->evaluate (walkers, new_pos, newRowList_d);
    AinvList_d   = AinvList;

    // Now evaluate ratios
    determinant_ratios_cuda 
      (&(AinvList_d[0]), &(newRowList_d[0]), &(ratio_d[0]), 
	 NumPtcls, NumPtcls, iat, walkers.size());
    
    // Copy back to host
    ratio_host = ratio_d;

    for (int iw=0; iw<psi_ratios.size(); iw++)
      psi_ratios[iw] *= ratio_host[iw];
  }


  void DiracDeterminantBase::ratio (vector<Walker_t*> &walkers, int iat, 
				    vector<PosType> &new_pos, 
				    vector<ValueType> &psi_ratios, 
				    vector<GradType>  &grad)
  {
    

  }

  void DiracDeterminantBase::ratio (vector<Walker_t*> &walkers, int iat, 
				    vector<PosType> &new_pos, 
				    vector<ValueType> &psi_ratios, 
				    vector<GradType>  &grad,
				    vector<ValueType> &lapl)
  {
    if (AList.size() < walkers.size())
      resizeLists(walkers.size());

    // First evaluate orbitals
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t::cuda_Buffer_t& data = walkers[iw]->cuda_DataSet;
      AinvList[iw]        =  &(data[AinvOffset]);
      newRowList[iw]      =  &(data[newRowOffset]);
      newGradLaplList[iw] =  &(data[newGradLaplOffset]);
    }
    newRowList_d = newRowList;
    newGradLaplList_d = newGradLaplList;
    cerr << "In grad_lapl ratio.\n";
    Phi->evaluate (walkers, new_pos, newRowList_d, newGradLaplList_d, NumOrbitals);
    //Phi->evaluate (walkers, new_pos, newRowList_d);

    AinvList_d   = AinvList;
    
    // Now evaluate ratios
    determinant_ratios_cuda 
      (&(AinvList_d[0]), &(newRowList_d[0]), &(ratio_d[0]), 
	 NumPtcls, NumPtcls, iat, walkers.size());
    
    // Copy back to host
    ratio_host = ratio_d;
    
    for (int iw=0; iw<psi_ratios.size(); iw++)
      psi_ratios[iw] *= ratio_host[iw];

    // Calculate gradient and laplacian
    

  }

  void 
  DiracDeterminantBase::gradLapl (vector<Walker_t*> &walkers, GradMatrix_t &grads,
				  ValueMatrix_t &lapl)
  {
    if (AList.size() < walkers.size())
      resizeLists(walkers.size());

    // First evaluate orbitals
    for (int iw=0; iw<walkers.size(); iw++) {
      Walker_t::cuda_Buffer_t& data = walkers[iw]->cuda_DataSet;
      AinvList[iw]        =  &(data[AinvOffset]);
      gradLaplList[iw]    =  &(data[gradLaplOffset]);
      newGradLaplList[iw] =  &(gradLapl_d[4*NumPtcls*iw]);
    }
    AinvList_d      = AinvList;
    gradLaplList_d = gradLaplList;
    newGradLaplList_d = newGradLaplList;

    calc_grad_lapl (&(AinvList_d[0]), &(gradLaplList_d[0]),
		    &(newGradLaplList_d[0]), NumOrbitals, 
		    NumOrbitals, walkers.size());
    // Now copy data into the output matrices
    gradLapl_host = gradLapl_d;
    // for (int i=0; i<gradLapl_host.size(); i++)
    //   cerr << "i = " << i << "  gradLapl_host = " << gradLapl_host[i] << endl;

    for (int iw=0; iw<walkers.size(); iw++) {
      for(int iat=0; iat < NumPtcls; iat++) {
	grads(iw,iat+FirstIndex)[0] += gradLapl_host[4*(iw*NumPtcls + iat)+0];
	grads(iw,iat+FirstIndex)[1] += gradLapl_host[4*(iw*NumPtcls + iat)+1];
	grads(iw,iat+FirstIndex)[2] += gradLapl_host[4*(iw*NumPtcls + iat)+2];
	lapl(iw,iat+FirstIndex)     += gradLapl_host[4*(iw*NumPtcls + iat)+3];
      }
    }

  }
}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
