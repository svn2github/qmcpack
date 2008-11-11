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
#ifndef QMCPLUSPLUS_SLATERDETERMINANT_WITHBASE_H
#define QMCPLUSPLUS_SLATERDETERMINANT_WITHBASE_H
#include "QMCWaveFunctions/OrbitalBase.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminantBase.h"

namespace qmcplusplus {

  class SlaterDet: public OrbitalBase {

  public:

    typedef DiracDeterminantBase Determinant_t;

    /// constructor
    SlaterDet();

    ///destructor
    ~SlaterDet();

    ///add a new DiracDeterminant to the list of determinants
    void add(Determinant_t* det);

    void checkInVariables(opt_variables_type& active);

    void checkOutVariables(const opt_variables_type& active);

    ///reset all the Dirac determinants, Optimizable is true
    void resetParameters(const opt_variables_type& optVariables);

    void reportStatus(ostream& os);

    void resetTargetParticleSet(ParticleSet& P);

    ValueType 
    evaluate(ParticleSet& P, 
	     ParticleSet::ParticleGradient_t& G, 
	     ParticleSet::ParticleLaplacian_t& L);

    ValueType 
    evaluateLog(ParticleSet& P, 
	        ParticleSet::ParticleGradient_t& G, 
	        ParticleSet::ParticleLaplacian_t& L);

    ///return the total number of Dirac determinants
    inline int size() const { return Dets.size();}

    ///return the column dimension of the i-th Dirac determinant
    inline int size(int i) const { return Dets[i]->cols();}

    ValueType registerData(ParticleSet& P, PooledData<RealType>& buf);
    ValueType updateBuffer(ParticleSet& P, PooledData<RealType>& buf, bool fromscratch=false);
    void copyFromBuffer(ParticleSet& P, PooledData<RealType>& buf);
    void dumpToBuffer(ParticleSet& P, PooledData<RealType>& buf);
    void dumpFromBuffer(ParticleSet& P, PooledData<RealType>& buf);
    ValueType evaluateLog(ParticleSet& P, PooledData<RealType>& buf);

    inline ValueType ratio(ParticleSet& P, int iat,
			   ParticleSet::ParticleGradient_t& dG, 
			   ParticleSet::ParticleLaplacian_t& dL) { 
      return Dets[DetID[iat]]->ratio(P,iat,dG,dL);
    }

    inline ValueType logRatio(ParticleSet& P, int iat,
			   ParticleSet::ParticleGradient_t& dG, 
			   ParticleSet::ParticleLaplacian_t& dL) { 
      ValueType r = Dets[DetID[iat]]->ratio(P,iat,dG,dL);
      return evaluateLogAndPhase(r,PhaseValue);
    }
    
    inline void restore(int iat) {
      return Dets[DetID[iat]]->restore(iat);
    }

    inline void acceptMove(ParticleSet& P, int iat) {
      Dets[DetID[iat]]->acceptMove(P,iat);
    }

    inline ValueType
    ratio(ParticleSet& P, int iat) {
      return Dets[DetID[iat]]->ratio(P,iat);
    } 	  

    void update(ParticleSet& P, 
		ParticleSet::ParticleGradient_t& dG, 
		ParticleSet::ParticleLaplacian_t& dL,
		int iat) {
      return Dets[DetID[iat]]->update(P,dG,dL,iat);
    }

    OrbitalBasePtr makeClone(ParticleSet& tqp) const;

    /////////////////////////////////////////////////////
    // Functions for vectorized evaluation and updates //
    /////////////////////////////////////////////////////
    void recompute(MCWalkerConfiguration &W)
    {
      for (int id=0; id<Dets.size(); id++)
	Dets[id]->recompute(W);
    }

    void 
    reserve (PointerPool<cuda_vector<CudaRealType> > &pool)
    {
      for (int id=0; id<Dets.size(); id++)
	Dets[id]->reserve(pool);
    }

    void 
    addLog (MCWalkerConfiguration &W, vector<RealType> &logPsi)
    {
      for (int id=0; id<Dets.size(); id++)
    	Dets[id]->addLog(W, logPsi);
    }

    // void 
    // ratio (MCWalkerConfiguration &W, int iat, vector<PosType> &new_pos,
    // 	   vector<ValueType> &psi_ratios)
    // {
    //   Dets[DetID[iat]]->ratio(W, iat, new_pos, psi_ratios);
    // }

    // void 
    // ratio (MCWalkerConfiguration &W, int iat, vector<PosType> &new_pos,
    // 	   vector<ValueType> &psi_ratios,	vector<GradType>  &grad)
    // {
    //   Dets[DetID[iat]]->ratio(W, iat, new_pos, psi_ratios, grad);
    // }

    void 
    ratio (MCWalkerConfiguration &W, int iat, vector<PosType> &new_pos,
	   vector<ValueType> &psi_ratios,	vector<GradType>  &grad,
	   vector<ValueType> &lapl)
    {
      Dets[DetID[iat]]->ratio(W, iat, new_pos, psi_ratios, grad, lapl);
    }



    // void 
    // addGradient(MCWalkerConfiguration &W, int iat,
    // 		vector<GradType> &grad)
    // {
    //   Dets[DetID[iat]]->addGradient(W, iat, grad);
    // }

    void update (vector<Walker_t*> &walkers, int iat)
    {
      Dets[DetID[iat]]->update(walkers, iat);
    }

    
    void 
    gradLapl (MCWalkerConfiguration &W, GradMatrix_t &grads,
	      ValueMatrix_t &lapl) {
      for (int id=0; id<Dets.size(); id++)
	Dets[id]->gradLapl(W, grads, lapl);
    }

    void 
    NLratios (MCWalkerConfiguration &W,  vector<NLjob> &jobList,
	      vector<PosType> &quadPoints, vector<ValueType> &psi_ratios)
    {
      for (int id=0; id<Dets.size(); id++) 
	Dets[id]->NLratios(W, jobList, quadPoints, psi_ratios);
    }



    void 
    NLratios (MCWalkerConfiguration &W,  cuda_vector<CUDA_PRECISION*> &Rlist,
	      cuda_vector<int*> &ElecList, cuda_vector<int>             &NumCoreElecs,
	      cuda_vector<CUDA_PRECISION*> &QuadPosList,
	      cuda_vector<CUDA_PRECISION*> &RatioList,
	      int numQuadPoints)
    {
      for (int id=0; id<Dets.size(); id++)
	Dets[id]->NLratios(W, Rlist, ElecList, NumCoreElecs, QuadPosList,
			   RatioList, numQuadPoints);
    }

  private:
    vector<int> M;
    vector<int> DetID;
    ///container for the DiracDeterminants
    vector<Determinant_t*>  Dets;
  };
}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
