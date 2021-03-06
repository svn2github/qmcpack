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
#ifndef OHMMS_QMC_SLATERTYPEORBITAL_MOLECULARORBITALS_H
#define OHMMS_QMC_SLATERTYPEORBITAL_MOLECULARORBITALS_H

#include "QMCWaveFunctions/OrbitalBuilderBase.h"
#include "QMCWaveFunctions/SphericalOrbitalSet.h"
#include "QMCWaveFunctions/MolecularOrbitals/MolecularOrbitalBasis.h"
#include "Numerics/SlaterTypeOrbital.h"

namespace ohmmsqmc {

  /**Class to add a set of Slater Type atomic orbital basis functions 
   *to the collection of basis functions.
   *
   @brief Example of a ROT (Radial Orbital Type)
  */
  class STOMolecularOrbitals: public OrbitalBuilderBase {
  public:

    typedef GenericSTO<ValueType>                      RadialOrbitalType;
    typedef SphericalOrbitalSet<RadialOrbitalType>     CenteredOrbitalType;
    typedef MolecularOrbitalBasis<CenteredOrbitalType> BasisSetType;

    ///constructor
    STOMolecularOrbitals(TrialWaveFunction& wfs, 
			 ParticleSet& ions, 
			 ParticleSet& els);

    ///implement vritual function
    bool put(xmlNodePtr cur);

    ///returns a BasisSet
    BasisSetType* addBasisSet(xmlNodePtr cur);

  private:

    BasisSetType*      BasisSet;
    DistanceTableData* d_table;
    map<string,int>    RnlID;
    map<string,int>    CenterID;

  };
}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
