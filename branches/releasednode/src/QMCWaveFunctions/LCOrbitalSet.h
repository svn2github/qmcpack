//////////////////////////////////////////////////////////////////
// (c) Copyright 2006-  by Jeongnim Kim
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
#ifndef QMCPLUSPLUS_LINEARCOMIBINATIONORBITALSET_TEMP_H
#define QMCPLUSPLUS_LINEARCOMIBINATIONORBITALSET_TEMP_H

#include "QMCWaveFunctions/SPOSetBase.h"
#include "Numerics/DeterminantOperators.h"
#include "Numerics/MatrixOperators.h"

namespace qmcplusplus {

  /** decalaration of generic class to handle a linear-combination of basis function*/
  template<class BS, bool IDENTITY>
  class LCOrbitalSet: public SPOSetBase {
  };

  template<class BS>
  class LCOrbitalSet<BS,true>: public SPOSetBase {

  public:

    ///level of printing
    int ReportLevel;
    ///pointer to the basis set
    BS* myBasisSet;
    ValueMatrix_t Temp;
    ValueMatrix_t Tempv;

    /** constructor
     * @param bs pointer to the BasisSet
     * @param id identifier of this LCOrbitalSet
     */
    LCOrbitalSet(BS* bs=0,int rl=0): myBasisSet(0),ReportLevel(rl) 
    {
      if(bs) setBasisSet(bs);
    }

    /** destructor
     *
     * BasisSet is deleted by the object with ID == 0
     */
    ~LCOrbitalSet() {}

    SPOSetBase* makeClone() const 
    {
      LCOrbitalSet<BS,true>* myclone = new LCOrbitalSet<BS,true>(*this);
      myclone->myBasisSet = myBasisSet->makeClone();
      return myclone;
    }

    void resetParameters(const opt_variables_type& active) 
    {
      myBasisSet->resetParameters(active);
    }

    ///reset the target particleset
    void resetTargetParticleSet(ParticleSet& P) {
      myBasisSet->resetTargetParticleSet(P);
    }

    /** set the OrbitalSetSize
     */
    void setOrbitalSetSize(int norbs) {
      OrbitalSetSize=norbs;
      Tempv.resize(OrbitalSetSize,5);
    }

    /** set the basis set
     */
    void setBasisSet(BS* bs) {
      myBasisSet=bs;
      BasisSetSize=myBasisSet->getBasisSetSize();
      Temp.resize(BasisSetSize,5);
    }

    /** return the size of the basis set
     */
    inline int getBasisSetSize() const { 
      return (myBasisSet==0)? 0: myBasisSet->getBasisSetSize();
    }

    inline void 
    evaluate(const ParticleSet& P, int iat, ValueVector_t& psi) {
      myBasisSet->evaluateForPtclMove(P,iat);
      for(int j=0 ; j<OrbitalSetSize; j++) 
        psi[j] = myBasisSet->Phi[j];
    }

    inline void 
    evaluate(const ParticleSet& P, int iat, 
        ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi) {
      myBasisSet->evaluateAllForPtclMove(P,iat);
      for(int j=0; j<OrbitalSetSize; j++) psi[j]=myBasisSet->Phi[j];
      for(int j=0; j<OrbitalSetSize; j++) dpsi[j]=myBasisSet->dPhi[j];
      for(int j=0; j<OrbitalSetSize; j++) d2psi[j]=myBasisSet->d2Phi[j];
    }

    void evaluate_notranspose(const ParticleSet& P, int first, int last,
        ValueMatrix_t& logdet, GradMatrix_t& dlogdet, ValueMatrix_t& d2logdet) 
    {
      for(int i=0, iat=first; iat<last; i++,iat++)
      {
        myBasisSet->evaluateForWalkerMove(P,iat);
        std::copy(myBasisSet->Phi.data(),myBasisSet->Phi.data()+OrbitalSetSize,logdet[i]);
        std::copy(myBasisSet->dPhi.data(),myBasisSet->dPhi.data()+OrbitalSetSize,dlogdet[i]);
        std::copy(myBasisSet->d2Phi.data(),myBasisSet->d2Phi.data()+OrbitalSetSize,d2logdet[i]);
      }
    }
  };

  /** class to handle linear combinations of basis orbitals used to evaluate the Dirac determinants.
   *
   * LCOrbitalSet stands for (L)inear(C)ombinationOrbitals
   * Any single-particle orbital \f$ \psi_j \f$ that can be represented by
   * \f[
   * \psi_j ({\bf r}_i) = \sum_k C_{jk} \phi_{k}({\bf r}_i),
   * \f]
   * where \f$ \phi_{k} \f$ is the k-th basis.
   * A templated version is LCOrbitals.
   */
  template<class BS>
  class LCOrbitalSet<BS,false>: public SPOSetBase {

  public:

    ///level of printing
    int ReportLevel;
    ///pointer to the basis set
    BS* myBasisSet;

    ValueMatrix_t Temp;
    ValueMatrix_t Tempv;
    /** constructor
     * @param bs pointer to the BasisSet
     * @param id identifier of this LCOrbitalSet
     */
    LCOrbitalSet(BS* bs=0,int rl=0): myBasisSet(0),ReportLevel(rl) {
      if(bs) setBasisSet(bs);
    }

    /** destructor
     *
     * BasisSet is deleted by the object with ID == 0
     */
    ~LCOrbitalSet() {}

    SPOSetBase* makeClone() const 
    {
      LCOrbitalSet<BS,false>* myclone = new LCOrbitalSet<BS,false>(*this);
      myclone->myBasisSet = myBasisSet->makeClone();
      return myclone;
    }

    ///reset
    void resetParameters(const opt_variables_type& active) 
    {
      myBasisSet->resetParameters(active);
    }

    ///reset the target particleset
    void resetTargetParticleSet(ParticleSet& P) {
      myBasisSet->resetTargetParticleSet(P);
    }

    /** set the OrbitalSetSize
     */
    void setOrbitalSetSize(int norbs) {
      OrbitalSetSize=norbs;
      Tempv.resize(OrbitalSetSize,5);
    }

    /** set the basis set
     */
    void setBasisSet(BS* bs) {
      myBasisSet=bs;
      BasisSetSize=myBasisSet->getBasisSetSize();
      Temp.resize(BasisSetSize,5);
    }

    /** return the size of the basis set
     */
    inline int getBasisSetSize() const { 
      return (myBasisSet==0)? 0: myBasisSet->getBasisSetSize();
    }

    inline void 
    evaluate(const ParticleSet& P, int iat, ValueVector_t& psi) {
      myBasisSet->evaluateForPtclMove(P,iat);
      for(int j=0 ; j<OrbitalSetSize; j++) 
        psi[j] = dot(C[j],myBasisSet->Phi.data(),BasisSetSize);
      //overhead of blas::gemv is big
      //MatrixOperators::product(C,myBasisSet->Phi,psi.data());
      //overhead of blas::dot is too big, better to use the inline function
      //for((int j=0 ; j<OrbitalSetSize; j++)
      //  psi[j] = BLAS::dot(BasisSetSize,C[j],myBasisSet->Phi.data());
    }

    inline void 
    evaluate(const ParticleSet& P, int iat, 
        ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi) {
      myBasisSet->evaluateAllForPtclMove(P,iat);
      //optimal on tungsten
      const ValueType* restrict cptr=C.data();
      const typename BS::ValueType* restrict pptr=myBasisSet->Phi.data();
      const typename BS::ValueType* restrict d2ptr=myBasisSet->d2Phi.data();
      const typename BS::GradType* restrict dptr=myBasisSet->dPhi.data();
#pragma ivdep
      for(int j=0; j<OrbitalSetSize; j++) {
        register ValueType res=0.0, d2res=0.0;
        register GradType dres;
        for(int b=0; b<BasisSetSize; b++,cptr++) {
          res += *cptr*pptr[b];
          d2res += *cptr*d2ptr[b];
          dres += *cptr*dptr[b];
          //res += *cptr * (*pptr++);
          //d2res += *cptr* (*d2ptr++);
          //dres += *cptr* (*dptr++);
        }
        psi[j]=res; dpsi[j]=dres; d2psi[j]=d2res;
      }
      //blasI is not too good
      //for(int j=0 ; j<OrbitalSetSize; j++) {
      //  psi[j]   = dot(C[j],myBasisSet->Phi.data(),  BasisSetSize);
      //  dpsi[j]  = dot(C[j],myBasisSet->dPhi.data(), BasisSetSize);
      //  d2psi[j] = dot(C[j],myBasisSet->d2Phi.data(),BasisSetSize);
      //  //psi[j]   = dot(C[j],myBasisSet->Y[0],  BasisSetSize);
      //  //dpsi[j]  = dot(C[j],myBasisSet->dY[0], BasisSetSize);
      //  //d2psi[j] = dot(C[j],myBasisSet->d2Y[0],BasisSetSize);
      //}

    }

    void evaluate_notranspose(const ParticleSet& P, int first, int last,
        ValueMatrix_t& logdet, GradMatrix_t& dlogdet, ValueMatrix_t& d2logdet) 
    {
      const ValueType* restrict cptr=C.data();
#pragma ivdep
      for(int i=0,ij=0, iat=first; iat<last; i++,iat++){
        myBasisSet->evaluateForWalkerMove(P,iat);
        MatrixOperators::product(C,myBasisSet->Phi,logdet[i]);
        MatrixOperators::product(C,myBasisSet->d2Phi,d2logdet[i]);
        const typename BS::GradType* restrict dptr=myBasisSet->dPhi.data();
        for(int j=0,jk=0; j<OrbitalSetSize; j++) 
        {
          register GradType dres;
          for(int b=0; b<BasisSetSize; ++b) dres +=  cptr[jk++]*dptr[b];
          dlogdet(ij)=dres;
          ++ij;
        }
      }
    }
  };
}
#endif

/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
