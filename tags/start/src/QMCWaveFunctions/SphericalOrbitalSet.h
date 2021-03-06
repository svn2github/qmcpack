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
#ifndef OHMMS_QMC_SPHERICALORBITALSET_H
#define OHMMS_QMC_SPHERICALORBITALSET_H

#include "Particle/DistanceTableData.h"
#include "Numerics/SphericalTensor.h"

namespace ohmmsqmc {

  struct DummyGrid {
    inline int index(double r) {return 0;}
  };

  typedef TinyVector<int,4> QuantumNumberType;
  enum {q_n=0,q_l,q_m, q_s};

  /**class to represent a set of spherical orbitals centered at a common origin origin.
   *
   *Each basis orbital is represented by 
   \f[
    \phi_i({\bf r_{jC}}) =  R_{n_i l_i}(r_{jC})
    \Re (Y_{l_i}^{m_i}({\bf \hat{r_{jC}}}))
    \f]
    Here \f${\bf r_{jC}} = {\bf r_j - R_C}\f$ denotes the position of 
    particle \f$ j \f$ relative to the center \f$ C \f$.
    *
    The template (R)adial(O)rbital(T)ype should provide
    <ul>
    <li> evaluate(r,rinv)
    <li> Y
    <li> dY
    <li> d2Y
    </ul>
    The template (G)rid(T)ype should provide
    <ul>
    <li> index(r)
    </ul>
    Examples of ROT being GenericSTO<T> and OneDimGridFunctor<T>
    *
     Examples of GT being LogGrid<T>, LinearGrid<T>, LogZeroGrid<T>
     and NumericalGrid<T>
  */
  template<class ROT, class GT=DummyGrid>
  struct SphericalOrbitalSet {

    typedef DistanceTableData::RealType        RealType;
    typedef DistanceTableData::ValueType       ValueType;
    typedef DistanceTableData::PosType         PosType;
    typedef SphericalTensor<ValueType,PosType> SphericalHarmonics_t;
    typedef ROT                                RadialOrbital_t;

    ///reference to a DistanceTableData (ion-electron)
    const DistanceTableData* myTable;

    ///spherical tensor unique to this set of SphericalOrbitals
    SphericalHarmonics_t Ylm;
    ///index of the corresponding real Spherical Harmonic with quantum 
    ///numbers \f$ (l,m) \f$
    vector<int> LM;
    ///index of the corresponding radial orbital with quantum 
    ///numbers \f$ (n,l) \f$
    vector<int> NL;
    ///container for the radial grid
    vector<GT*> Grids;
    ///container for the radial orbitals
    vector<ROT*> Rnl;

    ///the constructor
    SphericalOrbitalSet(int lmax): Ylm(lmax){ }
    ~SphericalOrbitalSet() {
      myTable = NULL;
      //for(int i=0; i<Rnl.size(); i++) delete Rnl[i];
    }

    ///return the number of basis functions
    inline int basis() const { return NL.size(); }

    ///reset the DistanceTableData (ion-electron)
    inline void reset(DistanceTableData* atable) { 
      myTable = atable;
    }


    /**
       @brief For the center with index (source), evaluate
       all the basis functions beginning with index (offset). 
       *
       The results are stored in the matrices:
       \f[ y[i,k] = \phi_k(r_{ic}) \f]
       \f[ dy[i,k] = {\bf \nabla} \phi_k(r_{ic}) \f]
       \f[ d2y[i,k] = \nabla^2 \phi_k(r_{ic}) \f]
       *
       The basis functions can be written in the form
       \f[ \phi({\bf R}) = F(r)G(r,\theta,\phi), \f]
       where \f$ F(r) = \frac{R(r)}{r^l} \f$ is related to the radial orbital
       and \f$ G(r,\theta,\phi) = r^lS_l^m(\theta,\phi) \f$ is related to 
       the real spherical harmonic.  Evaluating the gradient and Laplacian
       leads to 
       \f[
       {\bf \nabla} \phi({\bf R}) = 
       \frac{dF(r)}{dr} G(r,\theta,\phi)\:{\bf \hat{r}} + 
       F(r) {\bf \nabla} G(r,\theta,\phi)
       \f]
       and
       \f[ 
       \nabla^2 \phi({\bf R}) = (\nabla^2 F) G +2 \nabla F \cdot \nabla G
       + F (\nabla^2 G) \f]
       where 
       \f[ 
       \nabla^2 G(r) = \frac{2}{r}\frac{dG(r)}{dr} + \frac{d^2G(r)}{dr^2} 
       \f]
       and 
       \f[ \nabla^2 G(r,\theta,\phi) = 0. \f]
       @param source index of the center \f$ I \f$
       @param first index of the first particle
       @param nptcl number of particles
       @param offset index of the first basis function
       @param y return vector \f$ y[i,j] = \phi_j({\bf r_i-R_I}) \f$
       @param dy return vector \f$ dy[i,j] = 
       {bf \nabla}_i \phi_j({\bf r_i-R_I}) \f$
       @param d2y return vector \f$ d2y[i,j] = 
       \nabla^2_i \phi_j({\bf r_i-R_I}) \f$  
    */ 

    template<class VM, class GM>
    inline void 
    evaluate(int source, int first, int nptcl, int offset, 
	     VM& y, GM& dy, VM& d2y) {

      int nn = myTable->M[source]+first;//first pair of the particle subset
      for(int i=0, iat=first; i<nptcl; i++, iat++, nn++) {
	RealType r = myTable->r(nn);
	RealType rinv = myTable->rinv(nn);
	PosType dr = myTable->dr(nn);
	Ylm.evaluate(dr);
	
	//find the indices for distinct grids
	for(int ir=0; ir<Grids.size(); ir++) {
	  Grids[ir]->index(r);
	}

	//spline them
	for(int nl=0; nl<Rnl.size(); nl++) {
	  Rnl[nl]->evaluate(r,rinv);
	}
	
	int bindex = offset;
	for(int ib=0; ib<NL.size(); ib++, bindex++) {
	  int nl = NL[ib];
	  int lm = LM[ib];
	  RealType drnloverr = rinv*Rnl[nl]->dY;
	  ValueType ang = Ylm.getYlm(lm);
	  PosType gr_rad = drnloverr*dr;
	  PosType gr_ang = Ylm.getGradYlm(lm);
	  y(iat,bindex)= ang*Rnl[nl]->Y;
	  dy(iat,bindex) = ang*gr_rad+Rnl[nl]->Y*gr_ang;
	  d2y(iat,bindex)= ang*(2.0*drnloverr+Rnl[nl]->d2Y) + 2.0*dot(gr_rad,gr_ang);
	}
      }
    }

    /**
       @brief For the center with index (source), evaluate
       all the basis functions beginning with index (offset). 
       @param source index of the source particle
       @param first starting index of the quantum particle
       @param nptcl the number of particles to be evaluate [first, first+npcl)
       @param offset the basis offset of the source
       @param nw the number of walkers
       @param nstride the number that makes up a composite index for (walker,particle)
       @param y function values, y((walker,particle),basis), basis includes the offset
       @param dy gradients, dy((walker,particle),basis)
       @param d2y laplacians, d2y((walker,particle),basis)
       *
       @note Implements a vectorized operation over particles and walkers.
    */

    template<class VM, class GM>
    inline void 
    evaluateW(int source, int first, int nptcl, int offset, 
	      int nw, int nstride,
	      VM& y, GM& dy, VM& d2y) {

      //composite index with the (particle,walker)
#ifdef USE_FASTWALKER
      //first pair of the particle subset
      int nn = myTable->M[source]+first;
      int iiw = 0;
      for(int i=0, iat=first; i<nptcl; i++, iat++, nn++) {
	for(int iw=0; iw < nw; iw++, iiw ++) {
#else
	  int nn0 = myTable->M[source]+first;
	  for(int iw=0; iw < nw; iw++) {
	    int iiw = iw*nstride+first;
	    for(int i=0, iat=first,nn=nn0; i<nptcl; i++, iat++, nn++, iiw++) {
#endif
	      RealType r = myTable->r(iw,nn);
	      RealType rinv = myTable->rinv(iw,nn);
	      PosType dr = myTable->dr(iw,nn);
	      Ylm.evaluate(dr);

	      //find the indices for distinct grids
	      for(int ir=0; ir<Grids.size(); ir++) {
		Grids[ir]->index(r);
	      }

	      //spline them
	      for(int nl=0; nl<Rnl.size(); nl++) {
		Rnl[nl]->evaluate(r,rinv);
	      }

	      int bindex = offset;
	      for(int ib=0; ib<NL.size(); ib++, bindex++) {
		int nl = NL[ib];
		int lm = LM[ib];
		RealType drnloverr = rinv*Rnl[nl]->dY;
		ValueType ang = Ylm.getYlm(lm);
		PosType gr_rad = drnloverr*dr;
		PosType gr_ang = Ylm.getGradYlm(lm);
		y(iiw,bindex)= ang*Rnl[nl]->Y;
		dy(iiw,bindex) = ang*gr_rad+Rnl[nl]->Y*gr_ang;
		d2y(iiw,bindex)= ang*(2.0*drnloverr+Rnl[nl]->d2Y)
		  +2.0*dot(gr_rad,gr_ang);
	      }
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
