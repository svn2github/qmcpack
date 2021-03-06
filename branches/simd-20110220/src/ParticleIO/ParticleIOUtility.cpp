//////////////////////////////////////////////////////////////////
// (c) Copyright 2009- by Jeongnim Kim
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
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#include "ParticleIO/ParticleIOUtility.h"
namespace qmcplusplus {

  void expandSuperCell(ParticleSet& ref_, Tensor<int,3>& tmat)
  {
    typedef ParticleSet::SingleParticlePos_t SingleParticlePos_t;
    typedef ParticleSet::Tensor_t Tensor_t;

    Tensor<int,3> I(1,0,0,0,1,0,0,0,1);
    bool identity=true;
    int ij=0;
    while(identity&& ij<9)
    {
      identity=(I[ij]==tmat[ij]);
      ++ij;
    }

    if(identity) return;

    //convert2unit
    ref_.convert2Unit(ref_.R);
    ParticleSet::ParticleLayout_t PrimCell(ref_.Lattice);
    ref_.Lattice.set(dot(tmat,PrimCell.R));

    int natoms=ref_.getTotalNum();
    int numCopies = abs(tmat.det());
    ParticleSet::ParticlePos_t primPos(ref_.R);
    ParticleSet::ParticleIndex_t primTypes(ref_.GroupID);
    ref_.resize(natoms*numCopies);
    int maxCopies = 10;
    int index=0;
    //set the unit to the Cartesian
    ref_.R.InUnit=PosUnit::CartesianUnit;
    for(int ns=0; ns<ref_.getSpeciesSet().getTotalNum();++ns)
    {
      for (int i0=-maxCopies; i0<=maxCopies; i0++)    
        for (int i1=-maxCopies; i1<=maxCopies; i1++)
          for (int i2=-maxCopies; i2<=maxCopies; i2++) 
            for (int iat=0; iat < primPos.size(); iat++) 
            {
              if(primTypes[iat]!=ns) continue;
              //SingleParticlePos_t r     = primPos[iat];
              SingleParticlePos_t uPrim = primPos[iat];
              for (int i=0; i<3; i++)   uPrim[i] -= std::floor(uPrim[i]);
              SingleParticlePos_t r = PrimCell.toCart(uPrim) + (double)i0*PrimCell.a(0) 
                + (double)i1*PrimCell.a(1) + (double)i2*PrimCell.a(2);
              SingleParticlePos_t uSuper = ref_.Lattice.toUnit(r);
              if ((uSuper[0] >= -1.0e-6) && (uSuper[0] < 0.9999) &&
                  (uSuper[1] >= -1.0e-6) && (uSuper[1] < 0.9999) &&
                  (uSuper[2] >= -1.0e-6) && (uSuper[2] < 0.9999)) 
              {
                char buff[500];
                app_log() << "  Reduced coord    Cartesion coord    species.\n";
                snprintf (buff, 500, "  %10.4f  %10.4f %10.4f   %12.6f %12.6f %12.6f %d\n", 
                    uSuper[0], uSuper[1], uSuper[2], r[0], r[1], r[2], ns);
                app_log() << buff;
                ref_.R[index]= r;
                ref_.GroupID[index]= ns;//primTypes[iat];
                index++;
              }
            }
    }
  }
}

/***************************************************************************
 * $RCSfile$   $Author: qmc $
 * $Revision: 1048 $   $Date: 2006-05-18 13:49:04 -0500 (Thu, 18 May 2006) $
 * $Id: ParticleIOUtility.cpp 1048 2006-05-18 18:49:04Z qmc $ 
 ***************************************************************************/
