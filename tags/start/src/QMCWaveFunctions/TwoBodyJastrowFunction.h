//////////////////////////////////////////////////////////////////
// (c) Copyright 2003  by Jeongnim Kim
//////////////////////////////////////////////////////////////////
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
#ifndef OHMMS_QMC_GENERIC_TWOBODYJASTROW_H
#define OHMMS_QMC_GENERIC_TWOBODYJASTROW_H
#include "Configuration.h"
#include "QMCWaveFunctions/OrbitalBase.h"

namespace ohmmsqmc {

  /**
     @defgroup TwoBodyJastrow
     generic implementation of TwoBodyJastrow<FT,SharedFunction>
     *@brief  The Two-Body Jastrow has the form 
     \f[ J_{ee}({\bf R}) = \sum_{i<j}^N u(r_{ij}) \f]
     where \f[ r_{ij} = |{\bf r}_i - {\bf r}_j| \f]
     and the summnation is over all distinct pairs
     *
     *The first template parameter FT is a functional \f$ u(r_{ij}), \f$ e.g.
     *PadeJastrow<T>
     *Requriement of the template function is 
     *ValueType evaluate(ValueType r, ValueType& dudr, ValueType& d2udr2).
     *The second template parameter SharedFunction is a boolean.
     *SharedFunction=false means that each pair type (ij) has a unique
     *functions, while SharedFunction=true means that all pair types
     *share the same functional
     *
     To calculate the Gradient use the identity
     \f[ 
     {\bf \nabla}_k(r_{ik}) = -\frac{{\bf r_{ik}}}{r_{ik}}. 
     \f]
     
     \f[ 
     {\bf \nabla}_k(J_{ee}({\bf R})) = 
     {\bf \nabla}_k \left[ \sum_{i=1}^{k-1} u(r_{ik}) 
     + \sum_{i=k+1}^N u(r_{ki})
     \right]
     \f]
     by application of the chain rule
     \f[ 
     {\bf \nabla}_k(J_{ee}({\bf R})) =  
     \sum_{i=1}^{k-1} \frac{du}{dr_{ik}}({\bf \hat{r_{ik}}})+
     +\sum_{i=k+1}^N \frac{du}{dr_{ki}}({\bf \hat{r_{ki}}}) 
     \f]
     which finally leads to the result
     \f[ 
     {\bf \nabla}_k(J_{ee}({\bf R}))= 
     -\sum_{i \neq k}^N \frac{du}{dr_{ik}}{\bf \hat{r_{ik}}}.
     \f]
     To calculate the Laplacian, use the identity
     \f[ 
     \nabla^2_k(r_{ik})=\frac{2}{r_{ik}}, 
     \f]
     and the vector product rule
     \f[ 
     \nabla^2 \cdot (f{\bf A}) = 
     f({\bf \nabla} \cdot {\bf A})+{\bf A}\cdot({\bf \nabla} f)
     \f]
     \f[
     \nabla^2_k (J_{ee}({\bf R}))  = 
     -\sum_{i \neq k}^N \left(\frac{du}{dr_{ik}}\right) {\bf \nabla}_k
     \cdot {\bf \hat{r_{ki}}} - {\bf \hat{r_{ki}}} \cdot 
     \left(\frac{d^2u}{dr_{ik}^2}\right){\bf \hat{r_{ki}}} 
     \f]
     which can be simplified to
     \f[\nabla^2_k (J_{ee}({\bf R})) = \sum_{i \neq k}^N \left(\frac{2}{r_{ik}}\frac{du}{dr_{ik}}
     + \frac{d^2u}{dr_{ik}^2}\right) \f]
     *
     */

  /**
     @ingroup TwoBodyJastrow
     @class TwoBodyJastrow
     @brief A generic TwoBodyJastrow function uses a distance table.
     *The indices i(sources) and j(targets) are the same, as indicated by the sum
     *\f$\sum_{i<j}\f$.
     *
     *@warning Note that -1 is multiplied globally after summing the pair terms.
     */

  template<class FT, bool SharedFunction>
  class TwoBodyJastrow: public OrbitalBase {};

  /**class specialized TwoBodyJastrow<FT,false>
   *@brief the function \f$u(r_{ij})\f$ depends on the pair type 
   *
   *For electrons, distinct pair correlation functions are used 
   *for spins up-up/down-down and up-down/down-up.
   */ 
  template<class FT>
  class TwoBodyJastrow<FT,false>: public OrbitalBase {

    const DistanceTableData* d_table;

  public:

    typedef FT FuncType;
    ///container for the Jastrow functions 
    vector<FT*> F;

    ///constructor
    TwoBodyJastrow(DistanceTableData* dtable): d_table(dtable) { }


    ~TwoBodyJastrow(){
      DEBUGMSG("TwoBodyJastrow::~TwoBodyJastrow")
	//for(int i=0; i<F.size(); i++) delete F[i];
	}

    ///reset the value of all the Two-Body Jastrow functions
    void reset() { 
      for(int i=0; i<F.size(); i++) F[i]->reset();
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
    ValueType evaluate(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G, 
		       ParticleSet::ParticleLaplacian_t& L) {

      ValueType sumu = 0.0;
      ValueType dudr, d2udr2;
      PosType gr;
      for(int i=0; i<d_table->size(SourceIndex); i++) {
	for(int nn=d_table->M[i]; nn<d_table->M[i+1]; nn++) {
	  int j = d_table->J[nn];
	  sumu += F[d_table->PairID[nn]]->evaluate(d_table->r(nn), dudr, d2udr2);
	  //multiply 1/r
	  dudr *= d_table->rinv(nn);
	  gr = dudr*d_table->dr(nn);
	  //(d^2 u \over dr^2) + (2.0\over r)(du\over\dr)
	  ValueType lap = d2udr2+2.0*dudr;

	  //multiply -1
	  G[i] += gr;
	  G[j] -= gr;
	  L[i] -= lap; 
	  L[j] -= lap; 
	}
      }
      return exp(-sumu);
    }

#ifdef USE_FASTWALKER
    void evaluate(WalkerSetRef& W,
		  ValueVectorType& psi,
		  WalkerSetRef::WalkerGradient_t& G,
		  WalkerSetRef::WalkerLaplacian_t& L) {

      ValueType dudr, d2udr2;
      int nw = W.walkers();
      const DistanceTableData::IndexVectorType& M = d_table->M;
      const DistanceTableData::IndexVectorType& J = d_table->J;
      const DistanceTableData::IndexVectorType& PairID = d_table->PairID;
      for(int iw=0; iw<nw; iw++) {
        ValueType sumu = 0.0;
	for(int i=0; i<d_table->size(SourceIndex); i++) {
	  for(int nn=M[i]; nn<M[i+1]; nn++) {
	    int j = J[nn];
	    sumu += F[PairID[nn]]->evaluate(d_table->r(iw,nn), dudr, d2udr2);
	    dudr *= d_table->rinv(iw,nn);
	    PosType gr = dudr*d_table->dr(iw,nn);
	    ValueType lap = d2udr2+2.0*dudr;
	    G(iw,i) += gr;
	    G(iw,j) -= gr;
	    L(iw,i) -= lap; 
	    L(iw,j) -= lap; 
	  }
	}
	psi[iw]*= exp(-sumu);
      }
    }
#else
    void evaluate(WalkerSetRef& W,
		  ValueVectorType& psi,
		  WalkerSetRef::WalkerGradient_t& G,
		  WalkerSetRef::WalkerLaplacian_t& L) {

      ValueType dudr, d2udr2;
      int nw = W.walkers();
      const DistanceTableData::IndexVectorType& M = d_table->M;
      const DistanceTableData::IndexVectorType& J = d_table->J;
      const DistanceTableData::IndexVectorType& PairID = d_table->PairID;
      vector<ValueType> sumu(nw,0.0);
      for(int i=0; i<d_table->size(SourceIndex); i++) {
	for(int nn=M[i]; nn<M[i+1]; nn++) {
	  int j = J[nn];
	  for(int iw=0; iw<nw; iw++) {
	    sumu[iw] += F[PairID[nn]]->evaluate(d_table->r(iw,nn), dudr, d2udr2);
	    dudr *= d_table->rinv(iw,nn);
	    PosType gr = dudr*d_table->dr(iw,nn);
	    ValueType lap = d2udr2+2.0*dudr;
	    G(iw,i) += gr;
	    G(iw,j) -= gr;
	    L(iw,i) -= lap; 
	    L(iw,j) -= lap; 
	  }
	}
      }
      for(int iw=0; iw<nw; iw++) psi[iw]*= exp(-sumu[iw]);
    }
#endif
  };

  /**class specialized TwoBodyJastrow<FT,true>
   *@brief an indentical function \f$u(r_{ij})\f$ for all the pair types 
   *
   *For electrons, a single pair correlation function is used for 
   *spins up-up/down-down/up-down/down-up.
   */ 
  template<class FT>
  class TwoBodyJastrow<FT,true>: public OrbitalBase {

    const DistanceTableData* d_table;

  public:

    typedef FT FuncType;
    FT F;

    ///constructor
    TwoBodyJastrow(DistanceTableData* dtable): d_table(dtable) { }

    ~TwoBodyJastrow(){
      DEBUGMSG("TwoBodyJastrow::~TwoBodyJastrow")
	}


    ///reset the value of the Two-Body Jastrow functions
    void reset() { 
      F.reset();
    }

    /** implements the virtual functions of OrbitalBase
	@param P the particle set
	@param G returns the gradient \f$G[i]={\bf \nabla}_i J({\bf R})\f$
	@param L returns the laplacian \f$L[i]=\nabla^2_i J({\bf R})\f$
	@return \f$exp(-J({\bf R}))\f$
	@note The DistanceTableData contains only distinct pairs of the 
	particles belonging to one set, e.g., SymmetricDTD.
    */
    inline ValueType evaluate(ParticleSet& P,
			      ParticleSet::ParticleGradient_t& G, 
			      ParticleSet::ParticleLaplacian_t& L) {

      ValueType sumu = 0.0;
      ValueType dudr, d2udr2;
      PosType gr;
      for(int i=0; i<d_table->size(SourceIndex); i++) {
	for(int nn=d_table->M[i]; nn<d_table->M[i+1]; nn++) {
	  int j = d_table->J[nn];
	  sumu += F.evaluate(d_table->r(nn), dudr, d2udr2);
	  //multiply 1/r
	  dudr *= d_table->rinv(nn);
	  gr = dudr*d_table->dr(nn);
	  //(d^2 u \over dr^2) + (2.0\over r)(du\over\dr)\f$
	  ValueType lap = d2udr2+2.0*dudr;

	  //multiply -1
	  G[i] += gr;
	  G[j] -= gr;
	  L[i] -= lap; 
	  L[j] -= lap; 
	}
      }
      return exp(-sumu);
    }

#ifdef USE_FASTWALKER
    inline void evaluate(WalkerSetRef& W,
			 ValueVectorType& psi,
			 WalkerSetRef::WalkerGradient_t& G,
			 WalkerSetRef::WalkerLaplacian_t& L) {
      
      ValueType dudr, d2udr2;
      int nw = W.walkers();
      const DistanceTableData::IndexVectorType& M = d_table->M;
      const DistanceTableData::IndexVectorType& J = d_table->J;
      const DistanceTableData::IndexVectorType& PairID = d_table->PairID;
      for(int iw=0; iw<nw; iw++) {
        ValueType sumu = 0.0;
	for(int i=0; i<d_table->size(SourceIndex); i++) {
	  for(int nn=M[i]; nn<M[i+1]; nn++) {
	    int j = J[nn];
	    sumu += F.evaluate(d_table->r(iw,nn), dudr, d2udr2);
	    dudr *= d_table->rinv(iw,nn);
	    PosType gr = dudr*d_table->dr(iw,nn);
	    ValueType lap = d2udr2+2.0*dudr;
	    G(iw,i) += gr;
	    G(iw,j) -= gr;
	    L(iw,i) -= lap; 
	    L(iw,j) -= lap; 
	  }
	}
	psi[iw]*= exp(-sumu);
      }
    }
#else
    inline void evaluate(WalkerSetRef& W,
			 ValueVectorType& psi,
			 WalkerSetRef::WalkerGradient_t& G,
			 WalkerSetRef::WalkerLaplacian_t& L) {
      
      ValueType dudr, d2udr2;
      int nw = W.walkers();
      const DistanceTableData::IndexVectorType& M = d_table->M;
      const DistanceTableData::IndexVectorType& J = d_table->J;
      const DistanceTableData::IndexVectorType& PairID = d_table->PairID;
      vector<ValueType> sumu(nw,0.0);
      for(int i=0; i<d_table->size(SourceIndex); i++) {
	for(int nn=M[i]; nn<M[i+1]; nn++) {
	  int j = J[nn];
	  for(int iw=0; iw<nw; iw++) {
	    sumu[iw] += F.evaluate(d_table->r(iw,nn), dudr, d2udr2);
	    dudr *= d_table->rinv(iw,nn);
	    PosType gr = dudr*d_table->dr(iw,nn);
	    ValueType lap = d2udr2+2.0*dudr;
	    G(iw,i) += gr;
	    G(iw,j) -= gr;
	    L(iw,i) -= lap; 
	    L(iw,j) -= lap; 
	  }
	}
      }
      for(int iw=0; iw<nw; iw++) psi[iw]*= exp(-sumu[iw]);
    }
#endif
  };

}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/

