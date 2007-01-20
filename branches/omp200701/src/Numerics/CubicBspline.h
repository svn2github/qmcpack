//////////////////////////////////////////////////////////////////
// (c) Copyright 2006-  Kenneth Esler and Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Modified by Jeongnim Kim for qmcpack
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
#ifndef QMCPLUSPLUS_CUBIC_B_SPLINE_H
#define QMCPLUSPLUS_CUBIC_B_SPLINE_H

#include "Numerics/CubicBsplineGrid.h"

template<class T, unsigned GRIDTYPE, bool PBC>
struct CubicBspline: public CubicBsplineGrid<T,GRIDTYPE,PBC>
{
  typedef typename CubicBsplineGrid<T,GRIDTYPE,PBC>::point_type point_type;
  typedef typename CubicBsplineGrid<T,GRIDTYPE,PBC>::value_type value_type;
  typedef typename CubicBsplineGrid<T,GRIDTYPE,PBC>::container_type container_type;

  using CubicBsplineGrid<T,GRIDTYPE,PBC>::i0;
  using CubicBsplineGrid<T,GRIDTYPE,PBC>::i1;
  using CubicBsplineGrid<T,GRIDTYPE,PBC>::i2;
  using CubicBsplineGrid<T,GRIDTYPE,PBC>::i3;
  using CubicBsplineGrid<T,GRIDTYPE,PBC>::tp;
  using CubicBsplineGrid<T,GRIDTYPE,PBC>::GridDeltaInv;
  using CubicBsplineGrid<T,GRIDTYPE,PBC>::GridDeltaInv2;

  ///coefficients
  point_type A[16], dA[12], d2A[8], d3A[4];

  // The control points
  container_type P;

  inline CubicBspline()
  {
    A[ 0] = -1.0/6.0; A[ 1] =  3.0/6.0; A[ 2] = -3.0/6.0; A[ 3] = 1.0/6.0;
    A[ 4] =  3.0/6.0; A[ 5] = -6.0/6.0; A[ 6] =  3.0/6.0; A[ 7] = 0.0/6.0;
    A[ 8] = -3.0/6.0; A[ 9] =  0.0/6.0; A[10] =  3.0/6.0; A[11] = 0.0/6.0;
    A[12] =  1.0/6.0; A[13] =  4.0/6.0; A[14] =  1.0/6.0; A[15] = 0.0/6.0;

    dA[0]=-0.5; dA[1]= 1.5; dA[ 2]=-1.5; dA[ 3]= 0.5;
    dA[4]= 1.0; dA[5]=-2.0; dA[ 6]= 1.0; dA[ 7]= 0.0;
    dA[8]=-0.5; dA[9]= 0.0; dA[10]= 0.5; dA[11]= 0.0;

    d2A[0]=-1.0; d2A[1]= 3.0; d2A[2]=-3.0; d2A[3]= 1.0;
    d2A[4]= 1.0; d2A[5]=-2.0; d2A[6]= 1.0; d2A[7]= 0.0;

    d3A[0]=-1.0; d3A[1]= 3.0; d3A[2]=-3.0; d3A[3]= 1.0;
  }

  void Init(T start, T end, const container_type& datain, bool closed)
  {
    this->spline(start,end,datain,P,closed);
  }

  inline value_type getValue(point_type x)
  {
    this->getGridPoint(x);
    return
      tp[0]*(A[ 0]*P[i0]+A[ 1]*P[i1]+A[ 2]*P[i2]+A[ 3]*P[i3])+
      tp[1]*(A[ 4]*P[i0]+A[ 5]*P[i1]+A[ 6]*P[i2]+A[ 7]*P[i3])+
      tp[2]*(A[ 8]*P[i0]+A[ 9]*P[i1]+A[10]*P[i2]+A[11]*P[i3])+
      tp[3]*(A[12]*P[i0]+A[13]*P[i1]+A[14]*P[i2]+A[15]*P[i3]);
  }

  inline value_type getDeriv(point_type x)
    {
      this->getGridPoint(x);
      return GridDeltaInv *
        (tp[1]*(dA[0]*P[i0]+dA[1]*P[i1]+dA[ 2]*P[i2]+dA[ 3]*P[i3])+
         tp[2]*(dA[4]*P[i0]+dA[5]*P[i1]+dA[ 6]*P[i2]+dA[ 7]*P[i3])+
         tp[3]*(dA[8]*P[i0]+dA[9]*P[i1]+dA[10]*P[i2]+dA[11]*P[i3]));
    }

  inline value_type getDeriv2(point_type x)
  {
    this->getGridPoint(x);
    return GridDeltaInv * GridDeltaInv*
      (tp[2]*(d2A[0]*P[i0]+d2A[1]*P[i1]+d2A[2]*P[i2]+d2A[3]*P[i3])+
       tp[3]*(d2A[4]*P[i0]+d2A[5]*P[i1]+d2A[6]*P[i2]+d2A[7]*P[i3]));
  }

  inline value_type getDeriv3(point_type x)
  {
    this->getGridPoint(x);
    return GridDeltaInv * GridDeltaInv* GridDeltaInv*
      (tp[3]*(d2A[0]*P[i0]+d2A[1]*P[i1]+d2A[2]*P[i2]+d2A[3]*P[i3]));
  }

  inline value_type operator()(T x) 
  {
    return getValue(x);
  }

  inline value_type splint(point_type x, value_type& dy, value_type& d2y)
  {
    this->getGridPoint(x);
    return interpolate(P[i0],P[i0+1],P[i0+2],P[i0+3],dy,d2y);
    //dy= GridDeltaInv *
    //  (tp[1]*(dA[0]*P[i0]+dA[1]*P[i1]+dA[ 2]*P[i2]+dA[ 3]*P[i3])+
    //   tp[2]*(dA[4]*P[i0]+dA[5]*P[i1]+dA[ 6]*P[i2]+dA[ 7]*P[i3])+
    //   tp[3]*(dA[8]*P[i0]+dA[9]*P[i1]+dA[10]*P[i2]+dA[11]*P[i3]));
    //d2y=GridDeltaInv * GridDeltaInv*
    //  (tp[2]*(d2A[0]*P[i0]+d2A[1]*P[i1]+d2A[2]*P[i2]+d2A[3]*P[i3])+
    //   tp[3]*(d2A[4]*P[i0]+d2A[5]*P[i1]+d2A[6]*P[i2]+d2A[7]*P[i3]));
    //return
    //  tp[0]*(A[ 0]*P[i0]+A[ 1]*P[i1]+A[ 2]*P[i2]+A[ 3]*P[i3])+
    //  tp[1]*(A[ 4]*P[i0]+A[ 5]*P[i1]+A[ 6]*P[i2]+A[ 7]*P[i3])+
    //  tp[2]*(A[ 8]*P[i0]+A[ 9]*P[i1]+A[10]*P[i2]+A[11]*P[i3])+
    //  tp[3]*(A[12]*P[i0]+A[13]*P[i1]+A[14]*P[i2]+A[15]*P[i3]);
  }
  inline value_type interpolate(value_type p0, value_type p1, value_type p2, value_type p3,
      value_type& dy, value_type& d2y)
  {
    dy= GridDeltaInv *
      (tp[1]*(dA[0]*p0+dA[1]*p1+dA[ 2]*p2+dA[ 3]*p3)+
       tp[2]*(dA[4]*p0+dA[5]*p1+dA[ 6]*p2+dA[ 7]*p3)+
       tp[3]*(dA[8]*p0+dA[9]*p1+dA[10]*p2+dA[11]*p3));
    d2y=GridDeltaInv2*
      (tp[2]*(d2A[0]*p0+d2A[1]*p1+d2A[2]*p2+d2A[3]*p3)+
       tp[3]*(d2A[4]*p0+d2A[5]*p1+d2A[6]*p2+d2A[7]*p3));
    return
      tp[0]*(A[ 0]*p0+A[ 1]*p1+A[ 2]*p2+A[ 3]*p3)+
      tp[1]*(A[ 4]*p0+A[ 5]*p1+A[ 6]*p2+A[ 7]*p3)+
      tp[2]*(A[ 8]*p0+A[ 9]*p1+A[10]*p2+A[11]*p3)+
      tp[3]*(A[12]*p0+A[13]*p1+A[14]*p2+A[15]*p3);
  }
};

#endif
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1580 $   $Date: 2007-01-04 10:00:43 -0600 (Thu, 04 Jan 2007) $
 * $Id: TricubicBsplineSet.h 1580 2007-01-04 16:00:43Z jnkim $
 ***************************************************************************/
