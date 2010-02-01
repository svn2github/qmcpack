//////////////////////////////////////////////////////////////////
// (c) Copyright 1998-2002 by Jeongnim Kim
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
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_H5DATATYPE_DEFINE_H
#define QMCPLUSPLUS_H5DATATYPE_DEFINE_H

#include <type_traits/scalar_traits.h>
#if defined(HAVE_LIBHDF5)
#include <hdf5.h>
#endif

namespace APPNAMESPACE {
#if defined(HAVE_LIBHDF5)
template <typename T>                              
inline hid_t                                     
get_h5_datatype(const T&) { return H5T_NATIVE_CHAR;}

#define BOOSTSUB_H5_DATATYPE(CppType, H5DTYPE)                         \
template<>                                                             \
inline hid_t                                                           \
get_h5_datatype< CppType >(const CppType&) { return H5DTYPE; }         

/// INTERNAL ONLY
BOOSTSUB_H5_DATATYPE(short, H5T_NATIVE_SHORT);

/// INTERNAL ONLY
BOOSTSUB_H5_DATATYPE(int, H5T_NATIVE_INT);

/// INTERNAL ONLY
BOOSTSUB_H5_DATATYPE(long, H5T_NATIVE_LONG);

/// INTERNAL ONLY
BOOSTSUB_H5_DATATYPE(float, H5T_NATIVE_FLOAT);

/// INTERNAL ONLY
BOOSTSUB_H5_DATATYPE(double, H5T_NATIVE_DOUBLE);

/// INTERNAL ONLY
BOOSTSUB_H5_DATATYPE(std::complex<double>, H5T_NATIVE_DOUBLE);

/// INTERNAL ONLY
BOOSTSUB_H5_DATATYPE(std::complex<float>, H5T_NATIVE_FLOAT);

#else
  typedef int hid_t;
  typedef std::size_t hsize_t;
  const int H5P_DEFAULT=0;

  //return a non-sense integer
  template <typename T>                              
    inline hid_t                                     
    get_h5_datatype(const T&) { return 0;}
#endif
}
#endif
