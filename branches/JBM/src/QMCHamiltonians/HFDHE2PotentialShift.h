//////////////////////////////////////////////////////////////////
// (c) Copyright 2003  by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   National Center for Supercutoffomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: jnkim@ncsa.uiuc.edu
//   Tel:    217-244-6319 (NCSA) 217-333-3324 (MCC)
//
// Supported by 
//   National Center for Supercutoffomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_HFDHE2POTENTIAL_SHIFT_H
#define QMCPLUSPLUS_HFDHE2POTENTIAL_SHIFT_H
#include "Particle/ParticleSet.h"
#include "Particle/WalkerSetRef.h"
#include "QMCHamiltonians/QMCHamiltonianBase.h"
#include "ParticleBase/ParticleAttribOps.h"




namespace qmcplusplus {

  /** @ingroup hamiltonian
   @brief Evaluate the He Pressure.
 
   where d is the dimension of space and /Omega is the volume.
  **/

  struct HFDHE2PotentialShift: public QMCHamiltonianBase {
    Return_t VShift,rc;
    DistanceTableData* d_table;
    string Pname;
    
    /** constructor
     *
     * HFDHE2PotentialShift operators need to be re-evaluated during optimization.
     */
    HFDHE2PotentialShift(ParticleSet& P ){
      Pname = P.getName();
      UpdateMode.set(OPTIMIZABLE,1);
      d_table = DistanceTable::add(P);
      rc = P.Lattice.WignerSeitzRadius;
      
      Return_t A = 18.63475757;
      Return_t alpha = -2.381392669;
      Return_t c1=1.460008056;
      Return_t c2=14.22016431;
      Return_t c3=187.2033646;
      Return_t D = 6.960524706;
      Return_t r2 = (rc*rc);
      Return_t rm2 = 1.0/r2;
      Return_t rm6 = std::pow(rm2,3);
      Return_t rm8 = rm6*rm2;
      Return_t rm10 = rm8*rm2;
      VShift = (A*std::exp(alpha*rc) - (c1*rm6+c2*rm8+c3*rm10));
    }
    ///destructor
    ~HFDHE2PotentialShift() { }

    void resetTargetParticleSet(ParticleSet& P) {
      Pname = P.getName();
      d_table = DistanceTable::add(P);
      rc = P.Lattice.WignerSeitzRadius;
      
      Return_t A = 18.63475757;
      Return_t alpha = -2.381392669;
      Return_t c1=1.460008056;
      Return_t c2=14.22016431;
      Return_t c3=187.2033646;
      Return_t D = 6.960524706;
      Return_t r2 = (rc*rc);
      Return_t rm2 = 1.0/r2;
      Return_t rm6 = std::pow(rm2,3);
      Return_t rm8 = rm6*rm2;
      Return_t rm10 = rm8*rm2;
      VShift = (A*std::exp(alpha*rc) - (c1*rm6+c2*rm8+c3*rm10));
    }

    inline Return_t evaluate(ParticleSet& P) {
      Value = 0.0;

      for(int i=0; i<d_table->getTotNadj(); i++) {
        Return_t RR = d_table->r(i);
        if ( RR < rc) {
           Value += VShift;
        }
      }
      return 0.0;
    }

    inline Return_t 
    evaluate(ParticleSet& P, vector<NonLocalData>& Txy) {
      return evaluate(P);
    }

    bool put(xmlNodePtr cur) {return true;}

    bool get(std::ostream& os) const {
      os <<  "HFDHE2Potential Shift(T/S): "<< Pname;
      return true;
    }

    QMCHamiltonianBase* makeClone(ParticleSet& qp, TrialWaveFunction& psi)
    {
      return new HFDHE2PotentialShift(qp);
    }
    
    void addObservables(PropertySetType& plist)
    {
      myIndex=plist.add("HFDHE2Shift");
    }

    void setObservables(PropertySetType& plist)
    {
      plist[myIndex]=Value;
    }
  };
}
#endif

/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1581 $   $Date: 2007-01-04 10:02:14 -0600 (Thu, 04 Jan 2007) $
 * $Id: HFDHE2PotentialShift.h 1581 2007-01-04 16:02:14Z jnkim $ 
 ***************************************************************************/

