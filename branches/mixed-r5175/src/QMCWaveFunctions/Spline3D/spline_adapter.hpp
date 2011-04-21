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
/** @file spline_adapter.h
 *
 * Define and declare spline_adapter<EngT,IsOrtho,IsGamma,SingleK,PrecisionMatched>
 * EngT = einspline_engine<multi_UBspline_Xd_Y> with X=1,2,3 and Y=d,z,s,c
 * Specializations are needed for (IsGamma,SingleK,PrecisionMatched)
 * - spline_adapter<EngT,IsOrtho,true,true,true>  
 * - spline_adapter<EngT,IsOrtho,true,true,false> 
 * ToDo
 * - spline_adapter<EngT,IsOrtho,false,true,true>
 * - spline_adapter<EngT,IsOrtho,false,true,false>
 */
#ifndef QMCPLUSPLUS_EINSPLINE_ADAPTERS_H
#define QMCPLUSPLUS_EINSPLINE_ADAPTERS_H

#include <QMCWaveFunctions/OrbitalSetTraits.h>
#include <QMCWaveFunctions/EinsplineSet.h>
#include <spline/einspline_engine.hpp>

namespace qmcplusplus
{

  ///class to handler LatticeUnit
  template<bool IsOrtho> struct LatticeUnitHandler 
  {
    /** convert rin (Cartesian) to rout (reduced)
     *
     * Down convert rin to rout precision, if necessary
     */
    template<typename P1, typename P2>
      inline void toUnit(const P1& rin, P2& rout)
    {
    }
  };

  ///specialization of LatticeUnitHanlder for the orthorhombic cell
  template<> struct LatticeUnitHandler<true>
  {
    /** convert rin (Cartesian) to rout (reduced)
     *
     * Down convert rin to rout precision, if necessary
     */
    template<typename P1, typename P2>
      inline void toUnit(const P1& rin, P2& rout)
      {
      }
  };

  ///enumeration to handle specialization for spline_adapter
  enum {MixedPrecision, MultipleK, NotGamma} bspline_key;

  /** generic declaration of bspline_engin
   *
   * Template parameters
   * - EngT engine which evaluates quantities
   * - IsOrtho true, the cell is orthorombic
   * - IFLAG bspline_key using bitset<4>
   */
  template<typename EngT, bool IsOrtho, unsigned IFLAG> struct spline_adapter { };

  /** specialization for !(MixedPrecision) &!(MultipleK)& !(NotGamma)
   *
   * using IFLAG=0000
   */
  template<typename EngT, bool IsOrtho>
    class spline_adapter<EngT,IsOrtho,0> : public EinsplineSet
    {
      public:
        ///spliner
        EngT* spliner;
        ///lattice converter
        LatticeUnitHandler<IsOrtho> posConverter;

        ///constructor
        spline_adapter(): spliner(0){}

        void evaluate_notranspose(const ParticleSet& P, int first, int last
            , ValueMatrix_t& logdet, GradMatrix_t& dlogdet, ValueMatrix_t& d2logdet)
        {
          PosType ru;
          for(int i=0,iat=first; iat<last; ++i,++iat)
          {
            posConverter.toUnit(P.R[iat],ru);
            spliner->evaluate(ru, logdet[i], dlogdet[i], d2logdet[i]);
          }
        }

        //evaluate the values
        void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi)
        {
          PosType ru;
          posConverter.toUnit(P.R[iat],ru);
          spliner->evaluate(ru, psi);
        }

        void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi)
        {
          PosType ru;
          posConverter.toUnit(P.R[iat],ru);
          spliner->evaluate(ru,psi, dpsi, d2psi);
        }

        void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi, GradVector_t& dpsi, HessVector_t& grad_grad_psi)
        {
        }

    };

  /** specialization for MixedPrecision &!(MultipleK)& !(NotGamma)
   *
   * using IFLAG=0001
   */
  template<typename EngT, bool IsOrtho>
    class spline_adapter<EngT,IsOrtho,1> : public EinsplineSet
    {
      public:
        ///typedef real type
        typedef typename EngT::real_type real_type;
        ///typedef value type
        typedef typename EngT::value_type value_type;
        ///typedef for a position vector
        typedef TinyVector<real_type,OHMMS_DIM> pos_type;
        ///spliner
        EngT* spliner;
        ///lattice converter
        LatticeUnitHandler<IsOrtho> posConverter;
        ///temporary array to hold values
        Vector<value_type> valVec;
        ///temporary array to hold gradients
        Vector<TinyVector<value_type,OHMMS_DIM> > gradVec;
        ///temporary array to hold laplacians
        Vector<value_type> lapVec;

        ///constructor
        spline_adapter(): spliner(0){}

        void evaluate_notranspose(const ParticleSet& P, int first, int last
            , ValueMatrix_t& logdet, GradMatrix_t& dlogdet, ValueMatrix_t& d2logdet)
        {
          PosType ru;
          const int nv=valVec.size();
          for(int i=0,iat=first; iat<last; ++i,++iat)
          {
            posConverter.toUnit(P.R[iat],ru);
            spliner->evaluate(ru, valVec,gradVec,lapVec);
            convert(valVec.data(),logdet[i],nv);
            convert(gradVec.data(),dlogdet[i],nv);
            convert(lapVec.data(),d2logdet[i],nv);
          }
        }

        //evaluate the values
        void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi)
        {
          pos_type ru;
          posConverter.toUnit(P.R[iat],ru);
          spliner->evaluate(ru, valVec);
          convert(valVec,psi);
        }

        void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi)
        {
          pos_type ru;
          posConverter.toUnit(P.R[iat],ru);
          spliner->evaluate(ru,valVec, gradVec, lapVec);
          convert(valVec,psi);
          convert(gradVec,dpsi);
          convert(lapVec,d2psi);
        }

        void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi, GradVector_t& dpsi, HessVector_t& grad_grad_psi)
        {
        }

    };

}
#endif
/***************************************************************************
* $RCSfile$   $Author: jeongnim.kim $
* $Revision: 5119 $   $Date: 2011-02-06 16:20:47 -0600 (Sun, 06 Feb 2011) $
* $Id: spline_adapter.hpp 5119 2011-02-06 22:20:47Z jeongnim.kim $
***************************************************************************/
