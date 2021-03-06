//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim
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
#ifndef QMCPLUSPLUS_TRICUBIC_BSPLINESETBUILDER_H
#define QMCPLUSPLUS_TRICUBIC_BSPLINESETBUILDER_H

#include "QMCWaveFunctions/BasisSetBase.h"
#include "QMCWaveFunctions/GroupedOrbitalSet.h"
#include "Numerics/TricubicBsplineSet.h"

namespace qmcplusplus {

  class PWParameterSet;

  /**@ingroup WFSBuilder
   * A builder class for a set of Spline functions
   */
  class TricubicBsplineSetBuilder: public BasisSetBuilder {

  public:

    typedef TricubicBsplineSet<ValueType>              OrbitalGroupType;      
    typedef TricubicBsplineSet<ValueType>::StorageType StorageType;
    typedef GroupedOrbitalSet<OrbitalGroupType>        SPOSetType;             
    typedef map<string,ParticleSet*>                   PtclPoolType;

    /** constructor
     * @param p target ParticleSet
     * @param psets a set of ParticleSet objects
     */
    TricubicBsplineSetBuilder(ParticleSet& p, PtclPoolType& psets, xmlNodePtr cur);

    ///destructor
    ~TricubicBsplineSetBuilder();

    ///implement put function to process an xml node
    bool put(xmlNodePtr cur);

    /** initialize the Antisymmetric wave function for electrons
     *@param cur the current xml node
     */
    SPOSetBase* createSPOSet(xmlNodePtr cur);

  private:
    /** set of StorageType*
     *
     * Key is $1#$2 where $1 is the hdf5 file name and $2 is the band indenx
     */
    static map<string,StorageType*> BigDataSet;
    ///boolean to enable debug with EG
    bool DebugWithEG;
    ///if true, grid is open-ended [0,nx) x [0,ny) x [0, nz)
    bool OpenEndGrid;
    ///target ParticleSet
    ParticleSet& targetPtcl;
    ///reference to a ParticleSetPool
    PtclPoolType& ptclPool;
    ///save xml node
    xmlNodePtr rootNode;
    PosType LowerBox;
    PosType UpperBox;
    TinyVector<IndexType,DIM> BoxGrid;
    ///set of WFSetType*
    map<string,OrbitalGroupType*> myBasis;
    ///single-particle orbital sets
    map<string,SPOSetType*> mySPOSet;
    ///a function to test with EG
    SPOSetBase* createSPOSetWithEG();
    ///parameter set for h5 tags
    PWParameterSet* myParam;
  };
}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
