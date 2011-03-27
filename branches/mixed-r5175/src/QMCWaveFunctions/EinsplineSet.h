//////////////////////////////////////////////////////////////////
// (c) Copyright 2006-  by Jeongnim Kim and Ken Esler           //
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   National Center for Supercomputing Applications &          //
//   Materials Computation Center                               //
//   University of Illinois, Urbana-Champaign                   //
//   Urbana, IL 61801                                           //
//   e-mail: jnkim@ncsa.uiuc.edu                                //
//                                                              //
// Supported by                                                 //
//   National Center for Supercomputing Applications, UIUC      //
//   Materials Computation Center, UIUC                         //
//////////////////////////////////////////////////////////////////
/** @file EinsplineSet.h
 *
 * Declaration of EinsplineSet, a base class for SPOs using Einspline library
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SET_H
#define QMCPLUSPLUS_EINSPLINE_SET_H

#include <Configuration.h>
#include <QMCWaveFunctions/BasisSetBase.h>
#include <QMCWaveFunctions/SPOSetBase.h>
#include <QMCWaveFunctions/AtomicOrbital.h>
#include <QMCWaveFunctions/MuffinTin.h>
#include <Utilities/NewTimer.h>
#include <Numerics/e2iphi.h>
#include <QMCWaveFunctions/EinsplineTraits.h>
//#include <einspline/multi_bspline_structs.h>
//#ifdef QMC_CUDA
//  #include <einspline/multi_bspline_create_cuda.h>
//  #include "QMCWaveFunctions/AtomicOrbitalCuda.h"
//#endif

namespace qmcplusplus {

  class EinsplineSetBuilder;

  class EinsplineSet : public SPOSetBase
  {
    friend class EinsplineSetBuilder;
  protected:
    //////////////////////
    // Type definitions //
    //////////////////////
    typedef CrystalLattice<RealType,OHMMS_DIM> UnitCellType;
    
    ///////////
    // Flags //
    ///////////
    /// True if all Lattice is diagonal, i.e. 90 degree angles
    bool Orthorhombic;
    /// True if we are using localize orbitals
    bool Localized;
    /// True if we are tiling the primitive cell
    bool Tiling;
    
    //////////////////////////
    // Lattice and geometry //
    //////////////////////////
    TinyVector<int,3> TileFactor;
    Tensor<int,OHMMS_DIM> TileMatrix;
    UnitCellType SuperLattice, PrimLattice;
    /// The "Twist" variables are in reduced coords, i.e. from 0 to1.
    /// The "k" variables are in Cartesian coordinates.
    PosType TwistVector, kVector;
    /// This stores which "true" twist vector this clone is using.
    /// "True" indicates the physical twist angle after untiling
    int TwistNum;
    /// metric tensor to handle generic unitcell
    Tensor<RealType,OHMMS_DIM> GGt;

    ///////////////////////////////////////////////
    // Muffin-tin orbitals from LAPW calculation //
    ///////////////////////////////////////////////
    vector<MuffinTinClass> MuffinTins;
    int NumValenceOrbs, NumCoreOrbs;
        
  public:  

    inline string Type()
    {
      return "EinsplineSet";
    }

    EinsplineSet() :  TwistNum(0), NumValenceOrbs(0), NumCoreOrbs(0)
    {
      className = "EinsplineSet";
    }

    inline UnitCellType GetLattice() {return SuperLattice;}
    inline void resetTargetParticleSet(ParticleSet& e) {}
    inline void resetSourceParticleSet(ParticleSet& ions){}
    inline void resetParameters(const opt_variables_type& active){}
    inline void setOrbitalSetSize(int norbs) { OrbitalSetSize=norbs;}
  };
  
}
#endif
/***************************************************************************
* $RCSfile$   $Author: jeongnim.kim $
* $Revision: 5119 $   $Date: 2011-02-06 16:20:47 -0600 (Sun, 06 Feb 2011) $
* $Id: EinsplineSet.h 5119 2011-02-06 22:20:47Z jeongnim.kim $
***************************************************************************/
