//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim
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
#ifndef QMCPLUSPLUS_MULTISLATERDETERMINANT_ORBITAL_H
#define QMCPLUSPLUS_MULTISLATERDETERMINANT_ORBITAL_H
#include <Configuration.h>
#include <QMCWaveFunctions/OrbitalBase.h>
#include <QMCWaveFunctions/Fermion/DiracDeterminantBase.h>
#include <QMCWaveFunctions/Fermion/SPOSetProxyForMSD.h>

namespace qmcplusplus
  {

  /** @ingroup OrbitalComponent
   *  @brief An AntiSymmetric OrbitalBase composed of a linear combination of SlaterDeterminants.
   *
   *\f[
   *MS({\bf R}) = \sum_n c_n S_n({\bf R})
   *\f].
   *
   *The (S)ingle(P)article(O)rbitalSet template parameter is an
   *engine which fills in single-particle orbital terms.
   *
   \f[
   \frac{\nabla_i\Psi}{\Psi} = \frac{\sum_{n=1}^M c_n \nabla_i D_n}
   {\sum_{n=1}^M c_n D_n}
   \f]
   \f[
   \frac{\sum_{n=1}^M c_n S_n(\sum_{j=1}^N(\nabla_i
   S^{ij}_n({\bf r_i}))(S^{-1})^{ji}_n}{\sum_{n=1}^M c_n S_n}
   \f]
   The Laplacian
   \f[
   \frac{\nabla^2_i\Psi}{\Psi} = \frac{\sum_{n=1}^M c_n S_n(\sum_{j=1}^N
   (\nabla_i^2S^{ij}_n({\bf r_i}))(S^{-1})^{ji}_n}{\sum_{n=1}^M c_n S_n}
   \f]
   */
  class MultiSlaterDeterminant: public OrbitalBase
    {

    public:

      typedef DiracDeterminantBase*    DiracDeterminantPtr;
      typedef SPOSetBase*              SPOSetBasePtr;
      typedef SPOSetProxyForMSD*             SPOSetProxyPtr;
    typedef OrbitalSetTraits<ValueType>::IndexVector_t IndexVector_t;
    typedef OrbitalSetTraits<ValueType>::ValueVector_t ValueVector_t;
    typedef OrbitalSetTraits<ValueType>::GradVector_t  GradVector_t;
    typedef OrbitalSetTraits<ValueType>::HessMatrix_t  HessMatrix_t;
    typedef OrbitalSetTraits<ValueType>::HessType      HessType;
    typedef Array<HessType,3>                          HessArray_t;
    typedef TinyVector<HessType, 3>                    GGGType;
    typedef Vector<GGGType>                            GGGVector_t;
    typedef Matrix<GGGType>                            GGGMatrix_t;
    typedef ParticleSet::Walker_t                      Walker_t;


      ///constructor
      MultiSlaterDeterminant(ParticleSet& targetPtcl, SPOSetProxyPtr upspo, SPOSetProxyPtr dnspo);

      ///destructor
      ~MultiSlaterDeterminant();

      void checkInVariables(opt_variables_type& active);
      void checkOutVariables(const opt_variables_type& active);
      void resetParameters(const opt_variables_type& active);
      void reportStatus(ostream& os);

      void resetTargetParticleSet(ParticleSet& P);

      ValueType
      evaluate(ParticleSet& P
               ,ParticleSet::ParticleGradient_t& G
               ,ParticleSet::ParticleLaplacian_t& L);

      RealType
      evaluateLog(ParticleSet& P //const DistanceTableData* dtable,
                  , ParticleSet::ParticleGradient_t& G
                  , ParticleSet::ParticleLaplacian_t& L);

      GradType evalGrad(ParticleSet& P, int iat);
      ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat);
      ValueType ratio(ParticleSet& P, int iat
                      , ParticleSet::ParticleGradient_t& dG,ParticleSet::ParticleLaplacian_t& dL);

      ValueType ratio(ParticleSet& P, int iat);
      void acceptMove(ParticleSet& P, int iat);
      void restore(int iat);

      void update(ParticleSet& P
                  , ParticleSet::ParticleGradient_t& dG, ParticleSet::ParticleLaplacian_t& dL
                  , int iat);
      RealType evaluateLog(ParticleSet& P,BufferType& buf);
      RealType registerData(ParticleSet& P, BufferType& buf);
      RealType updateBuffer(ParticleSet& P, BufferType& buf, bool fromscratch=false);
      void copyFromBuffer(ParticleSet& P, BufferType& buf);

      OrbitalBasePtr makeClone(ParticleSet& tqp) const;
      void evaluateDerivatives(ParticleSet& P,
                               const opt_variables_type& optvars,
                               vector<RealType>& dlogpsi,
                               vector<RealType>& dhpsioverpsi);

      void resize(int,int);

      /**
        add a new SlaterDeterminant with coefficient c to the
        list of determinants
        */
      //int NumOrbitals_ground,NumOrbitals_total;
      int nels_up, nels_dn;
      int NumUniqueDets_up;
      int NumUniqueDets_dn;
      vector<int> DetID;

      int FirstIndex_up, LastIndex_up;
      int FirstIndex_dn, LastIndex_dn;

      map<string,int> SPOSetID;

      SPOSetProxyPtr spo_up;
      SPOSetProxyPtr spo_dn;

      std::vector<DiracDeterminantPtr> dets_up;
      std::vector<DiracDeterminantPtr> dets_dn;

      // map determinant in linear combination to unique det list
      vector<int> C2node_up;
      vector<int> C2node_dn;

      vector<RealType> C;

      // lap(#uniqueDet,part#)
      ValueVector_t detValues_up;
      ValueVector_t detValues_dn;

// UGLY, how do I get around this? I want to use GradMatrix instead...
      // grads(#uniqueDet,part#)
      Vector<ParticleSet::ParticleGradient_t> grads_up;
      Vector<ParticleSet::ParticleGradient_t> grads_dn;

      // lap(#uniqueDet,part#)
      Vector<ParticleSet::ParticleLaplacian_t> lapls_up;
      Vector<ParticleSet::ParticleLaplacian_t> lapls_dn;

      // grads(#uniqueDet,part#)
      Vector<ParticleSet::ParticleGradient_t> tempgrad;

      // lap(#uniqueDet,part#)
      Vector<ParticleSet::ParticleLaplacian_t> templapl;

      ValueType curRatio;
      ValueVector_t detsRatios;
      ValueVector_t lapl_temp;
      GradVector_t grad_temp;

      ParticleSet::ParticleGradient_t myG;
      ParticleSet::ParticleLaplacian_t myL;

      opt_variables_type myVars;
    };

}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$
 ***************************************************************************/
