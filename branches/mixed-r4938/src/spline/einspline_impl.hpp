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
/** @file einspline_impl.hpp
 * @brief define einspline class with static functions
 *
 */
#ifndef QMCPLUSPLUS_EINSPLINE_IMPL_H
#define QMCPLUSPLUS_EINSPLINE_IMPL_H

#ifndef QMCPLUSPLUS_EINSPLINE_ENGINE_HPP
#error "einspline_impl.hpp can be used only by einspline_engine.hpp"
#else
#include <einspline/multi_bspline.h>
#include <OhmmsPETE/Tensor.h>
#include <OhmmsPETE/TinyVector.h>

/** einspline::impl provides the function wrappers for einspline library
 *
 * - create_bspline_engine : create the spline engine based on the data type, size and BCs
 * - set_bspline(spline,ival,data_in): set a spline function at ival location
 *   - pointer to spline engine
 *   - ival the index of a state
 *   - data_in raw data
 * - evaluate functions
 */
namespace qmcplusplus { namespace einspline  { namespace impl {

#if  OHMMS_DIM==3

#define EINSPLINE_3D_INTERFACE(SPT,BCT,RT,DT)                                   \
SPT* create_bspline_engine(TinyVector<int,3> &npts                              \
    , TinyVector<RT,3>& start, TinyVector<RT,3>& end, BCT *xyzBC                \
    , int num_orbitals                                                          \
    )                                                                           \
{                                                                               \
  Ugrid xgrid, ygrid, zgrid;                                                    \
  xgrid.start = start[0];   xgrid.end = end[0];  xgrid.num =npts[0];            \
  ygrid.start = start[1];   ygrid.end = end[1];  ygrid.num =npts[1];            \
  zgrid.start = start[2];   zgrid.end = end[2];  zgrid.num =npts[2];            \
  return create_##SPT(xgrid, ygrid, zgrid, xyzBC[0], xyzBC[1], xyzBC[2], num_orbitals);      \
}                                                                               \
                                                                                \
SPT* create_bspline_engine(Ugrid* xyzgrid, BCT *xyzBC, int num_orbitals)        \
{                                                                               \
  return create_##SPT(xyzgrid[0], xyzgrid[1], xyzgrid[2]                        \
      , xyzBC[0], xyzBC[1], xyzBC[2], num_orbitals);                            \
}                                                                               \
                                                                                \
inline void set_bspline(SPT *spline, int ival, DT* data_in)                     \
{                                                                               \
  set_##SPT(spline, ival, data_in);                                             \
}                                                                               \
                                                                                \
inline void evaluate(SPT *spline, const TinyVector<RT,3>& r, DT*  psi)          \
{                                                                               \
  eval_##SPT(spline, r[0], r[1], r[2], psi);                                    \
}                                                                               \
                                                                                \
inline void evaluate(SPT *spline, const TinyVector<RT,3>& r                     \
    , DT* restrict psi, TinyVector<DT,3> * restrict grad)                       \
{                                                                               \
  eval_##SPT##_vg(spline, r[0], r[1], r[2], psi, grad->data());                 \
}                                                                               \
                                                                                \
inline void evaluate(SPT *spline, const TinyVector<RT,3>& r                     \
    , DT* restrict psi, TinyVector<DT,3> * restrict grad, DT* lap)              \
{                                                                               \
  eval_##SPT##_vgl(spline, r[0], r[1], r[2], psi, grad->data(),lap);            \
}                                                                               \
                                                                                \
inline void evaluate(SPT *spline, const TinyVector<RT,3>& r                     \
    , DT* restrict psi, TinyVector<DT,3>* restrict grad                         \
    , Tensor<DT,3>* restrict hess)                                              \
{                                                                               \
  eval_##SPT##_vgh(spline, r[0], r[1], r[2], psi, grad->data(),hess->data());   \
}                                                                               \


///declare interfaces(einspline engine, boundary type, precision, data type)
EINSPLINE_3D_INTERFACE(multi_UBspline_3d_d, BCtype_d, double, double)
EINSPLINE_3D_INTERFACE(multi_UBspline_3d_z, BCtype_z, double, std::complex<double>)

EINSPLINE_3D_INTERFACE(multi_UBspline_3d_s, BCtype_s, float, float)
EINSPLINE_3D_INTERFACE(multi_UBspline_3d_c, BCtype_c, float, std::complex<float>)


#else
  #error "Only three-dimensional interfaces are implemented."
#endif

}}}
#endif
#endif
