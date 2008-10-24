//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim and Kris Delaney
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
#ifndef QMCPLUSPLUS_COULOMBPBCAA_TEMP_H
#define QMCPLUSPLUS_COULOMBPBCAA_TEMP_H
#include "QMCHamiltonians/QMCHamiltonianBase.h"
#include "QMCHamiltonians/CudaCoulomb.h"
#include "LongRange/LRCoulombSingleton.h"

namespace qmcplusplus {

  /** @ingroup hamiltonian
   *\brief Calculates the AA Coulomb potential using PBCs
   *
   * Functionally identical to CoulombPBCAA but uses a templated version of
   * LRHandler.
   */
  struct CoulombPBCAATemp: public QMCHamiltonianBase {

    typedef LRCoulombSingleton::LRHandlerType LRHandlerType;
    typedef LRCoulombSingleton::GridType       GridType;
    typedef LRCoulombSingleton::RadFunctorType RadFunctorType;
    LRHandlerType* AA;
    GridType* myGrid;
    RadFunctorType* rVs;
    ParticleSet& PtclRef;

    bool is_active;
    bool FirstTime;
    int NumSpecies;
    int ChargeAttribIndx;
    int MemberAttribIndx;
    int NumCenters;
    RealType myConst;
    RealType myRcut;
    string PtclRefName;
    vector<RealType> Zat,Zspec; 
    vector<int> NofSpecies;
    vector<int> SpeciesID;

    Matrix<RealType> SR2;
    Vector<RealType> dSR;
    Vector<ComplexType> del_eikr;

    /** constructor */
    CoulombPBCAATemp(ParticleSet& ref, bool active);

    ~CoulombPBCAATemp();

    void resetTargetParticleSet(ParticleSet& P);

    Return_t evaluate(ParticleSet& P);

    inline Return_t evaluate(ParticleSet& P, vector<NonLocalData>& Txy) {
      return evaluate(P);
    }
    Return_t registerData(ParticleSet& P, BufferType& buffer);
    Return_t updateBuffer(ParticleSet& P, BufferType& buffer);
    void copyFromBuffer(ParticleSet& P, BufferType& buf);
    void copyToBuffer(ParticleSet& P, BufferType& buf);
    Return_t evaluatePbyP(ParticleSet& P, int iat);
    void acceptMove(int iat);

    /** Do nothing */
    bool put(xmlNodePtr cur) {
      return true;
    }

    bool get(std::ostream& os) const {
      os << "CoulombPBCAA potential: " << PtclRefName;
      return true;
    }

    QMCHamiltonianBase* makeClone(ParticleSet& qp, TrialWaveFunction& psi);

    void initBreakup(ParticleSet& P);

    Return_t evalSR(ParticleSet& P);
    Return_t evalLR(ParticleSet& P);
    Return_t evalConsts();
    Return_t evaluateForPbyP(ParticleSet& P);
    
    //////////////////////////////////
    // Vectorized evaluation on GPU //
    //////////////////////////////////
    //// Short-range part
    TextureSpline SRSpline;
    cuda_vector<CUDA_PRECISION*> RlistGPU;
    cuda_vector<CUDA_PRECISION>  RGPU, SumGPU;
    cuda_vector<CUDA_PRECISION>  L, Linv;
    host_vector<CUDA_PRECISION>  RHost, SumHost;
    host_vector<CUDA_PRECISION*> RlistHost;
    //// Long-range part
    int Numk;
    cuda_vector<CUDA_PRECISION> kpointsGPU;
    cuda_vector<int>            kshellGPU;
    // This has the same lengths as KshellGPU
    cuda_vector<CUDA_PRECISION> FkGPU;
    // The first vector index is the species number
    // Complex, stored as float2
    cuda_vector<CUDA_PRECISION*> RhoklistGPU;
    host_vector<CUDA_PRECISION*> RhoklistHost;
    cuda_vector<CUDA_PRECISION> RhokGPU;
    void setupLongRangeGPU(ParticleSet &P);
    void addEnergy(vector<Walker_t*> &walkers, 
		   vector<RealType> &LocalEnergy);
  };

}
#endif

/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/

