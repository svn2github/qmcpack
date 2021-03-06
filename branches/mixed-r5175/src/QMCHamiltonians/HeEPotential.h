#ifndef QMCPLUSPLUS_HEEPOTENTIAL_H
#define QMCPLUSPLUS_HEEPOTENTIAL_H
#include "Particle/ParticleSet.h"
#include "Particle/WalkerSetRef.h"
#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"
#include "QMCHamiltonians/QMCHamiltonianBase.h"

#include "QMCHamiltonians/HeEPotential_tail.h"


namespace qmcplusplus {
  /** @ingroup hamiltonian
   *@brief He-e potential
   */
  struct HeePotential: public QMCHamiltonianBase {
    RealType rc;
    RealType A,B,C,trunc,TCValue;
//     HeePotential_tail* TCorr;
    DistanceTableData* d_table;
    ParticleSet* PtclRef, IRef;

    HeePotential(ParticleSet& P, ParticleSet& I): PtclRef(&P), IRef(I) {
//       Dependants=1;
//       depName = "Heetail";
      A=0.655;
      B=89099;
      C=12608;
    
      d_table = DistanceTable::add(I,P);
//       rc = P.Lattice.WignerSeitzRadius;
//       app_log()<<" RC is "<<rc<<endl;
//       if (rc>0) trunc= A*std::pow(rc,-4) * ( B/(C+std::pow(rc,6)) - 1 );
//       else trunc=0;
//       app_log()<<" trunc "<<trunc<<endl;
    }

    ~HeePotential() { }

//     QMCHamiltonianBase* makeDependants(ParticleSet& qp )
//     {
// //       TCorr = new HeePotential_tail(qp);
//       return TCorr;
//     }
    
    void resetTargetParticleSet(ParticleSet& P)  {
      d_table = DistanceTable::add(P);
      PtclRef=&P;
    }

    inline Return_t evaluate(ParticleSet& P) {
      Value = 0.0;
//       TCValue=0.0;
      for(int i=0; i<d_table->getTotNadj(); i++) {
	Return_t r1 = d_table->r(i);
//         if ( r1 < rc) {
	 Value+=A*std::pow(r1,-4) * ( B/(C+std::pow(r1,6)) - 1 ) ;
// 	 TCValue+=trunc;
// 	}
      }
//       TCorr->set_TC(TCValue);
      return Value;
    }

    inline Return_t evaluate(ParticleSet& P, vector<NonLocalData>& Txy) {
      return evaluate(P);
    }


    /** Do nothing */
    bool put(xmlNodePtr cur) {
        string tagName("HeePot");
        OhmmsAttributeSet Tattrib;
        Tattrib.add(tagName,"name");
        Tattrib.put(cur);
      if (tagName != "HeePot") myName=tagName;
      return true;
    }

    bool get(std::ostream& os) const {
      os << "HeePotential: " << PtclRef->getName();
      return true;
    }

    QMCHamiltonianBase* makeClone(ParticleSet& qp, TrialWaveFunction& psi)
    {
      HeePotential* HPclone = new HeePotential(qp,IRef);
      HPclone->myName = myName;
      return HPclone;
    }
    
    void addObservables(PropertySetType& plist)
    {
      myIndex=plist.add(myName);
    }

    void setObservables(PropertySetType& plist)
    {
      plist[myIndex]=Value;
    }
  };
}
#endif
