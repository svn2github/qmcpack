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
#ifndef OHMMS_GRID_FUNCTOR_CUBIC_SPLINE_H
#define OHMMS_GRID_FUNCTOR_CUBIC_SPLINE_H

#include "Numerics/OneDimGridFunctor.h"
#include "Numerics/NRSplineFunctions.h"


/**Perform One-Dimensional Cubic Spline Interpolation. 
 *
 Given a function evaluated on a grid \f$ \{x_i\}, 
 i=1\ldots N, \f$ such that \f$ y_i = y(x_i), \f$ we would like to 
 interpolate for a point \f$ x \f$ in the interval \f$ [x_j,x_{j+1}]. \f$ 
 The linear interpolation formula 
 \f[
 y = Ay_j + By_{j+1} 
 \f]
 where
 \f[
 A = \frac{x_{j+1}-x}{x_{j+1}-x_j} \;\;\;\;\;\;\;\;\;\;\;\;\;
 B = 1-A = \frac{x-x_{j+1}}{x_{j+1}-x_j}
 \f]
 Satisfies the conditions at the endpoints \f$ x_j \mbox{ and } x_{j+1},\f$
 but suffers from some major drawbacks.  The problem with this approach is 
 that over the range of the function \f$ [x_1,x_N] \f$ we have a series of 
 piecewise linear equations with a zero second derivative within each interval 
 and an undefined or infinite second derivative at the interval boundaries, 
 the grid points \f$ \{x_i\}. \f$  Ideally we would like to construct an 
 interpolation function with a smooth first derivate and a continuous second 
 derivative both within the intervals and at the the grid points.

 By adding a cubic polynomial to the linear interpolation equation within
 each interval, we can construct an interpolation function that varies
 linearly in the second derivative.  Assume for a moment that we have the 
 values of the second derivative evaluated at each grid point, 
 \f$ y_i'' = d^2y(x_i)/dx^2, i=1\ldots N. \f$  Now we can construct a cubic 
 polynomial that has the correct second derivatives \f$y_j'' \mbox{ and } 
 y_{j+1}''\f$ at the endpoints and also evaluates to zero at the endpoints.
 The reason the polynomial must be zero at the endpoints is to not spoil 
 the agreement that is already built into the linear function.  A function 
 constructed from these principals is given by the equation
 \f[
 y = Ay_j + By_{j+1} + Cy_j'' + Dy_{j+1}''
 \f]
 where
 \f[
 C = \frac{1}{6}(A^3-A)(x_{j+1}-x_j)^2 \;\;\;\;\;\;\;
 D = \frac{1}{6}(B^3-B)(x_{j+1}-x_j)^2.
 \f]


 To explictly check that this function does indeed satisfy the conditions 
 at the endpoints take the derivatives
 \f[
 \frac{dy}{dx} = \frac{y_{j+1}-y_j}{x_{j+1}-x_j}
 - \frac{3A^2-1}{6}(x_{j+1}-x_j)y_j''
 + \frac{3B^2-1}{6}(x_{j+1}-x_j)y_{j+1}''
 \f]
 and
 \f[
 \frac{d^2y}{dx^2} = Ay_j'' + By_{j+1}''.
 \f]
 The second derivative is continuous across the boundary between two 
 intervals, e.g. \f$ [x_{j-1},x_j] \f$ and \f$ [x_j,x_{j+1}], \f$ and 
 obeys the conditions at the endpoints since at \f$ x=x_j, (A=1,B=0) \f$ 
 and at  \f$ x=x_{j+1}, (A=0,B=1). \f$


 We had made the assumption that the values of the second derivative are 
 known at the grid points, which they are not.  By imposing the condition
 that the first derivative is smooth and continuous across the boundary 
 between two intervals it is possible to derive a set of equations to 
 generate the \f$ y_i''\f$'s.  Evaluate the equation for the first 
 derivative at \f$x=x_j\f$ in the inverval \f$ [x_{j-1},x_j] \f$ and set
 it equal to the same equation evaluated at \f$x=x_j\f$ in the inverval 
 \f$ [x_j,x_{j+1}]; \f$ rearranging the terms

 \f[
 \frac{x_j-x_{j+1}}{6}y_{j+1}'' + \frac{x_{j+1}-x_{j-1}}{3}y_j''
 + \frac{x_{j+1}-x_j}{6}y_{j+1}'' = \frac{y_{j+1}-y_j}{x_{j+1}-x_j}
 -  \frac{y_j-y_{j+1}}{x_j-x_{j+1}},
 \f]
 where \f$ j=2\ldots N-1.\f$  To generate a unique solution for the system 
 of \f$N-2\f$ equations we have to impose boundary conditions at \f$x_1 
 \mbox{ and } x_N,\f$ the possibilities being either to set \f$y_1'' 
 \mbox{ and } y_N''\f$ to zero, the natural cubic spline, or, if you want 
 to make the first derivative at the boundaries to have a specified value,
 use \f$y_1' \mbox{ and } y_N'\f$ to calculate the second derivatives at 
 the endpoints using equation. 
 *
*/

template <class Td, 
	  class Tg = double, 
	  class CTd= Vector<Td>,
	  class CTg= Vector<Tg> >
class OneDimCubicSpline: public OneDimGridFunctor<Td,Tg,CTd,CTg> {

public:

  typedef OneDimGridFunctor<Td,Tg,CTd,CTg> base_type;
  typedef typename base_type::value_type  value_type;
  typedef typename base_type::point_type  point_type;
  typedef typename base_type::data_type data_type;
  typedef typename base_type::grid_type grid_type;

  point_type r_min;
  point_type r_max;
  value_type first_deriv;
  value_type last_deriv;

  OneDimCubicSpline(grid_type* gt = NULL):base_type(gt) { }

  template<class VV>
  OneDimCubicSpline(grid_type* gt, const VV& nv):
    base_type(gt),first_deriv(0.0),last_deriv(0.0)
  {
    m_Y.resize(nv.size());
    std::copy(nv.begin(), nv.end(), m_Y.data());
  }

  /*
  OneDimCubicSpline(grid_type* gt,
                    const std::vector<Td>& nv):
    base_type(gt),first_deriv(0.0),last_deriv(0.0) 
  { 
    m_Y.resize(nv.size());
    std::copy(nv.begin(), nv.end(), m_Y.data());	   
  }
  */


  /**
   *@param r the radial distance
   *@param du return the derivative
   *@param d2u return the 2nd derivative
   *@return the value of the function
   *@brief Use the formula for the Cubic Spline 
   *Interpolation to evaluate the function and its
   *derivatives.
   *@note Must first call the function setgrid to
   *determine the interval on the grid which contains r.
  */
  inline value_type 
  splint(point_type r, value_type& du, value_type& d2u) {

    if(r<r_min) {
      //linear-extrapolation returns y[0]+y'*(r-r[0])
      du = first_deriv; 
      d2u = 0.0; 
      return m_Y[0]+first_deriv*(r-r_min);
    } 

    if(r<r_max) {
      const double onesixth = 1.0/6.0;
      //first set Loc for the grid
      int klo = m_grid->Loc;
      int khi = klo+1;
      point_type h = m_grid->dr(klo);
      point_type hinv = 1.0/h;
      point_type h6 = h*onesixth;
      point_type hh6 = h6*h;
      point_type A = (m_grid->r(khi)-r)*hinv; 
      point_type dA = -hinv;
      point_type B = (r-m_grid->r(klo))*hinv; 
      point_type dB = hinv;
      point_type C = A*(A*A-1.0)*hh6; 
      point_type dC = -h6*(3*A*A-1.0);
      point_type D = B*(B*B-1.0)*hh6; 
      point_type dD = h6*(3*B*B-1.0);
      du = dA*m_Y[klo]+dB*m_Y[khi]+ dC*m_Y2[klo] + dD*m_Y2[khi];
      d2u = A*m_Y2[klo] + B*m_Y2[khi];
      return A*m_Y[klo]+B*m_Y[khi]+C*m_Y2[klo]+D*m_Y2[khi];
    } else {
      du = 0.0; d2u = 0.0; return 1e-20;
     }
  }

  /**
   *\param imin the index of the first valid data point
   *\param yp1 the derivative at the imin-th grid point
   *\param imax the index of the last valid data point
   *\param ypn the derivative at the imax-th grid point
   *\brief Evaluate the 2nd derivate on the grid points for 
   *splint given the boundary conditions.
   *
   *In general, a grid is shared by several OneDimCubicSpline objects
   *and each object can have its own range of valid grid points.
   *r_min and r_max are used to specify the range.
   */
  inline 
  void spline(int imin, value_type yp1, int imax, value_type ypn) {
    first_deriv = yp1;
    last_deriv = ypn;
    r_min = m_grid->r(imin);
    r_max = m_grid->r(imax);
    m_Y2.resize(size());
    m_Y2 = 0.0;
    NRCubicSpline(m_grid->data()+imin, m_Y.data()+imin, 
		  size()-imin, yp1, ypn, m_Y2.data()+imin);
  }
};
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
