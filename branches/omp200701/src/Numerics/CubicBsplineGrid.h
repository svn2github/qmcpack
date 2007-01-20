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
#ifndef QMCPLUSPLUS_CUBIC_B_SPLINE_GRID_H
#define QMCPLUSPLUS_CUBIC_B_SPLINE_GRID_H
#include "Numerics/GridTraits.h"
#include <limits>

/** CubicBsplineGrid 
 *
 * Empty declaration to be specialized. Three template parameters are
 * - T data type
 * - GRIDTYPE enumeration of the grid type
 * - PBC true for periodic boundary condition
 */
template<class T, unsigned GRIDTYPE, bool PBC>
struct CubicBsplineGrid { };

/** specialization for linear grid with PBC */
template<class T>
struct CubicBsplineGrid<T,LINEAR_1DGRID,true>
{
  typedef typename GridTraits<T>::point_type point_type;
  typedef typename GridTraits<T>::value_type value_type;
  typedef std::vector<T>                     container_type;
  int i0,i1,i2,i3;
  point_type GridStart, GridEnd, GridDelta, GridDeltaInv, GridDeltaInv2, L, Linv;
  point_type curPoint;
  point_type tp[4];

  inline CubicBsplineGrid():curPoint(-10000){}

  inline bool getGridPoint(point_type x)
  {
    //test how expensive it is
    //if(fabs(curPoint-x) < std::numeric_limits<point_type>::epsilon())
    //  return true;
    curPoint=x;
    point_type delta = x - GridStart;
    delta -= std::floor(delta*Linv)*L;
    point_type ipart;
    point_type t = modf (delta*GridDeltaInv, &ipart);
    int i = (int) ipart;
    i0 = i;
    i1 = i+1;
    i2 = i+2;
    i3 = i+3;
    tp[0] = t*t*t;
    tp[1] = t*t;
    tp[2] = t;
    tp[3] = 1.0;
    return true;
  }

  void spline(point_type start, point_type end, const container_type& data, container_type& p, bool closed) 
  {
    GridStart=start;
    GridEnd=end;

    int N =data.size();
    if(closed) N--;

    p.resize(N+3);
    L=end-start;
    Linv=1.0/L;
    GridDelta=L/static_cast<T>(N);
    GridDeltaInv=1.0/GridDelta;
    GridDeltaInv2=1.0/GridDelta/GridDelta;

    T ratio = 0.25;
    container_type d(N), gamma(N), mu(N);
    for(int i=0; i<N; i++) d[i]=1.5*data[i];
    // First, eliminate leading coefficients
    gamma [0] = ratio;
    mu[0] = ratio;
    mu[N-1] = ratio;
    gamma[N-1] = 1.0;
    for (int row=1; row <(N-1); row++) {
      T diag = 1.0- mu[row-1]*ratio;
      T diagInv = 1.0/diag;
      gamma[row] = -ratio*gamma[row-1]*diagInv;
      mu[row] = diagInv*ratio;
      d[row]  = diagInv*(d[row]-ratio*d[row-1]);
      // Last row
      d[N-1] -= mu[N-1] * d[row-1];
      gamma[N-1] -= mu[N-1]*gamma[row-1];
      mu[N-1] = -mu[N-1]*mu[row-1];
    }
    // Last row:  gamma(N-1) hold diagonal element
    mu[N-1] += ratio;
    gamma[N-1] -= mu[N-1]*(mu[N-2]+gamma[N-2]);
    d[N-1] -= mu[N-1] * d[N-2];
    p[N] = d[N-1]/gamma[N-1];

    // Now go back upward, back substituting
    for (int row=N-2; row>=0; row--)
      p[row+1] = d[row] - mu[row]*p[row+2] - gamma[row]*p[N];

    p[0]=p[N];
    p[N+1]=p[1];
    p[N+2]=p[2];
  }


};
#endif
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1580 $   $Date: 2007-01-04 10:00:43 -0600 (Thu, 04 Jan 2007) $
 * $Id: TricubicBsplineSet.h 1580 2007-01-04 16:00:43Z jnkim $
 ***************************************************************************/
