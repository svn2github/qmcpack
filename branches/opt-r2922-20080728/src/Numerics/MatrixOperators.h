//////////////////////////////////////////////////////////////////
// (c) Copyright 2005- by Jeongnim Kim
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
//   Department of Physics, Ohio State University
//   Ohio Supercomputer Center
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_MATRIXOPERATORS_H
#define QMCPLUSPLUS_MATRIXOPERATORS_H

#include "OhmmsPETE/OhmmsMatrix.h"
#include "OhmmsPETE/OhmmsVector.h"
#include "OhmmsPETE/TinyVector.h"
#include "Numerics/OhmmsBlas.h"
namespace qmcplusplus {

  template<typename T>
  inline T TESTDOT(const T* restrict f, const T* restrict l, const T* restrict b) 
  {
    T s=0;
    while(f!=l) s += (*f++)*(*b++);
    return s;
  }

  struct MatrixOperators {
    /** static function to perform C=AB for real matrices
     *
     * Call dgemm
     */
    inline static void product(const Matrix<double>& A,
        const Matrix<double>& B, Matrix<double>& C) {
      const char transa = 'N';
      const char transb = 'N';
      const double one=1.0;
      const double zero=0.0;
      dgemm(transa, transb, B.cols(), A.rows(), B.rows(),
          one, B.data(), B.cols(), A.data(), A.cols(),
          zero, C.data(), C.cols());
    }

    /** static function to perform C=AB for complex matrices
     *
     * Call zgemm
     */
    inline static void product(const Matrix<std::complex<double> >& A,
        const Matrix<std::complex<double> >& B, Matrix<std::complex<double> >& C) {
      const char transa = 'N';
      const char transb = 'N';
      const std::complex<double> zone(1.0,0.0);
      const std::complex<double> zero(0.0,0.0);
      zgemm(transa, transb, B.cols(), A.rows(), B.rows(),
          zone, B.data(), B.cols(), A.data(), A.cols(),
          zero, C.data(), C.cols());
    }

    /** static function to perform C=AB for complex matrices
     *
     * Call zgemm
     */
    inline static void product(const Matrix<double>& A,
        const Matrix<std::complex<double> >& B, Matrix<double>& C) {
        cerr << " Undefined C=AB with real A and complex B " << endl;
    }

    /** static function to perform y=Ax for generic matrix and vector
     */
    inline static void product(const Matrix<double>& A, const Vector<double>& x, double* restrict yptr) {
      const char transa = 'T';
      const double one=1.0;
      const double zero=0.0;
      dgemv(transa, A.cols(), A.rows(), one, A.data(), A.cols(), x.data(), 1, zero, yptr, 1);
    }

    /** static function to perform y=Ax for generic matrix and vector
     */
    inline static void product(const Matrix<double>& A, const double* restrict xptr, double* restrict yptr) {
      const char transa = 'T';
      const double one=1.0;
      const double zero=0.0;
      dgemv(transa, A.cols(), A.rows(), one, A.data(), A.cols(), xptr, 1, zero, yptr, 1);
    }

    /** static function to perform y=Ax for generic matrix and vector
     */
    template<unsigned D>
    inline static void product(const Matrix<double>& A, const TinyVector<double,D>* xvPtr, 
        TinyVector<double,D>* restrict yptr) {
      const double one=1.0;
      const double zero=0.0;
      const char transa = 'N';
      const char transb = 'N';
      dgemm(transa, transb, D, A.rows(), A.cols(), 
          one, xvPtr->begin(), D, A.data(), A.cols(),
          zero, yptr->begin(), D);
    }

    /** static function to perform y=Ax for generic matrix and vector
     */
    template<unsigned D>
    inline static void product(const Matrix<double>& A, const Vector<TinyVector<double,D> >& x, 
        TinyVector<double,D>* restrict yptr) {
      const double one=1.0;
      const double zero=0.0;
      const char transa = 'N';
      const char transb = 'N';
      dgemm(transa, transb, D, A.rows(), x.size(),
          one, x.data()->begin(), D, A.data(), A.cols(),
          zero, yptr->begin(), D);
    }

    /** static function to perform y=Ax for generic matrix and vector
     */
    template<unsigned D>
    inline static void product(const Matrix<std::complex<double> >& A, 
        const Vector<TinyVector<std::complex<double>,D> >& x, 
        TinyVector<std::complex<double>,D>* restrict yptr) {
      const char transa = 'N';
      const char transb = 'N';
      const std::complex<double> zone(1.0,0.0);
      const std::complex<double> zero(0.0,0.0);
      zgemm(transa, transb, D, A.rows(), x.size(),
          zone, x.data()->begin(), D, A.data(), A.cols(),
          zero, yptr->begin(), D);
    }


    /** static function to perform y=Ax for generic matrix and vector
     */
    inline static void product(const Matrix<std::complex<double> >& A, 
        const Vector<std::complex<double> >& x, 
        std::complex<double>* restrict yptr) {
      const char transa = 'T';
      const std::complex<double> zone(1.0,0.0);
      const std::complex<double> zero(0.0,0.0);
      zgemv(transa, A.cols(), A.rows(), zone, A.data(), A.cols(), x.data(), 1, zero, yptr, 1);
    }

    /** static function to perform y=Ax for generic matrix and vector
     */
    inline static void product(const Matrix<double>& A, 
        const Vector<std::complex<double> >& x, double* restrict yptr) {
        cerr << " Undefined C=AB with real A and complex x " << endl;
    }

    template<typename T>
    inline static void product(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, std::vector<int>& offset) {
      int nr=C.rows();
      int nb=offset.size()-1;
      for(int i=0; i<nr; i++) 
      {
        for(int b=0; b<nb; b++) 
        {
          int firstK=offset[b];
          int lastK=offset[b+1];
          const T* restrict firstY=A[i]+firstK;
          const T* restrict lastY=A[i]+lastK;
          for(int k=firstK; k<lastK; k++) 
          {
            C(i,k)=TESTDOT(firstY,lastY,B[k]+firstK);
          }
        }
      }
    }

//    template<typename T>
//      inline statis void product(const Matrix<T>& A, const T* restrict x, T* restrict y)
//      {
//        GEMV<T,0>::apply(A.data(),x,y,A.rows(),A.cols());
//      }
  };
}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
