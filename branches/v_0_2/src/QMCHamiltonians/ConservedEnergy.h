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
#ifndef OHMMS_QMC_CONSERVEDENERGY_H
#define OHMMS_QMC_CONSERVEDENERGY_H

#include "Particle/ParticleSet.h"
#include "Particle/WalkerSetRef.h"
#include "QMCHamiltonians/QMCHamiltonianBase.h"

namespace ohmmsqmc {

  /**
   *\brief A fake Hamiltonian to check the sampling
   * and numerical evaluation of the trial function.
   * 
   Integrating the expression
   \f[
   {\bf \nabla} \cdot (\Psi_T({\bf R}) {\bf \nabla}
   \Psi_T({\bf R})) = \Psi_T({\bf R}) \nabla^2 
   \Psi_T({\bf R}) + ({\bf \nabla} \Psi_T({\bf R}))^2
   \f]
   leads to (using the divergence theorem)
   \f[
   0 = \int d^3 {\bf R} \: \left[\Psi_T({\bf R}) \nabla^2
   \Psi_T({\bf R}) + ({\bf \nabla} \Psi_T({\bf R}))^2\right]
   \f]
   or, written in terms of the probability distribution
   \f$ |\Psi_T({\bf R})|^2 \f$ 
   \f[ 0 = \int d^3 {\bf R} \: |\Psi_T({\bf R})|^2
   \left[\frac{\nabla^2 \Psi_T({\bf R})}{\Psi_T({\bf R})} +
   \left(\frac{{\bf \nabla} \Psi_T({\bf R})}{\Psi_T({\bf R})}
   \right)^2\right],
   \f]
   where
   \f[
   \frac{\nabla^2 \Psi_T({\bf R})}{\Psi_T({\bf R})} =
   \nabla^2 \ln \Psi_T({\bf R}) +
   ({\bf \nabla} \ln \Psi_T({\bf R}))^2
   \f]
   \f[
   \frac{{\bf \nabla} \Psi_T({\bf R})}{\Psi_T({\bf R})} = 
   {\bf \nabla} \ln \Psi_T({\bf R})
   \f]
   it is possible to check the sampling and the evaluation
   of \f$ \Psi_T, \f$ e.g. the gradient and laplacian.  
   The expectation value of this estimator should fluctuate
   around zero. 
  */
  struct ConservedEnergy: public QMCHamiltonianBase {

    ConservedEnergy(){}
    ~ConservedEnergy() { }

    ValueType 
    evaluate(ParticleSet& P) {
      return 0.0;
    }

    inline ValueType 
    evaluate(ParticleSet& P, RealType& x) {
      RealType gradsq = 0.0;
      RealType lap = 0.0;
      for(int iat=0; iat<P.getTotalNum(); iat++) {
	gradsq += dot(P.G(iat),P.G(iat));
	lap += P.L(iat);
      }
      //pool.put(lap+2.0*gradsq);
      x = lap+2.0*gradsq;
      return 0.0;
    }

    void evaluate(WalkerSetRef& W, ValueVectorType& LE) {
    }
  };
}
#endif

/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/

