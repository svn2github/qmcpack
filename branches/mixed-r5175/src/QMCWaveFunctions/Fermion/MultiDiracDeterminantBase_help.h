//////////////////////////////////////////////////////////////////
// (c) Copyright 1998-2002,2003- by Jeongnim Kim
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
/**@file DiracDeterminantBaseBase.h
 * @brief Declaration of DiracDeterminantBase with a S(ingle)P(article)O(rbital)SetBase
 */
#ifndef QMCPLUSPLUS_MULTIDIRACDETERMINANTWITHBASE_HELP_H
#define QMCPLUSPLUS_MULTIDIRACDETERMINANTWITHBASE_HELP_H

#include "OhmmsPETE/OhmmsMatrix.h"

namespace qmcplusplus {


 template <typename T>
 struct MyDeterminant {

    vector<T> M;
    vector<int> Pivot;

    void resize(int n) {
      M.resize(n*n);
      Pivot.resize(n);
    }

    inline T evaluate(T a11, T a12,
                      T a21, T a22)
    {
       return a11*a22-a21*a12;
    }

    inline T evaluate(T a11, T a12, T a13, 
                      T a21, T a22, T a23, 
                      T a31, T a32, T a33)
    {  
      //return a11*(a33*a22-a32*a23)-a21*(a33*a12-a32*a13)+a31*(a23*a12-a22*a13);
      return (a11*(a22*a33-a32*a23)-a21*(a12*a33-a32*a13)+a31*(a12*a23-a22*a13));
    }

    inline T evaluate(T a11, T a12, T a13, T a14, 
                      T a21, T a22, T a23, T a24,
                      T a31, T a32, T a33, T a34,
                      T a41, T a42, T a43, T a44)
    {  
      //return a11*evaluate(a22,a23,a24,a32,a33,a34,a42,a43,a44) - a21*evaluate(a12,a13,a14,a32,a33,a34,a42,a43,a44) + a31*evaluate(a12,a13,a14,a22,a23,a24,a42,a43,a44) - a41*evaluate(a12,a13,a14,a22,a23,a24,a32,a33,a34);
      return (a11*(a22*(a33*a44-a43*a34)-a32*(a23*a44-a43*a24)+a42*(a23*a34-a33*a24))-a21*(a12*(a33*a44-a43*a34)-a32*(a13*a44-a43*a14)+a42*(a13*a34-a33*a14))+a31*(a12*(a23*a44-a43*a24)-a22*(a13*a44-a43*a14)+a42*(a13*a24-a23*a14))-a41*(a12*(a23*a34-a33*a24)-a22*(a13*a34-a33*a14)+a32*(a13*a24-a23*a14)));
    }

    inline T evaluate(T a11, T a12, T a13, T a14, T a15,
                      T a21, T a22, T a23, T a24, T a25,
                      T a31, T a32, T a33, T a34, T a35,
                      T a41, T a42, T a43, T a44, T a45,
                      T a51, T a52, T a53, T a54, T a55)
    {
      //return a11*evaluate(a22,a23,a24,a25,a32,a33,a34,a35,a42,a43,a44,a45,a52,a53,a54,a55) 
      //     - a21*evaluate(a12,a13,a14,a15,a32,a33,a34,a35,a42,a43,a44,a45,a52,a53,a54,a55) 
      //     + a31*evaluate(a12,a13,a14,a15,a22,a23,a24,a25,a42,a43,a44,a45,a52,a53,a54,a55) 
      //     - a41*evaluate(a12,a13,a14,a15,a22,a23,a24,a25,a32,a33,a34,a35,a52,a53,a54,a55)
      //     + a51*evaluate(a12,a13,a14,a15,a22,a23,a24,a25,a32,a33,a34,a35,a42,a43,a44,a45);
      return (a11*(a22*(a33*(a44*a55-a54*a45)-a43*(a34*a55-a54*a35)+a53*(a34*a45-a44*a35))-a32*(a23*(a44*a55-a54*a45)-a43*(a24*a55-a54*a25)+a53*(a24*a45-a44*a25))+a42*(a23*(a34*a55-a54*a35)-a33*(a24*a55-a54*a25)+a53*(a24*a35-a34*a25))-a52*(a23*(a34*a45-a44*a35)-a33*(a24*a45-a44*a25)+a43*(a24*a35-a34*a25)))-a21*(a12*(a33*(a44*a55-a54*a45)-a43*(a34*a55-a54*a35)+a53*(a34*a45-a44*a35))-a32*(a13*(a44*a55-a54*a45)-a43*(a14*a55-a54*a15)+a53*(a14*a45-a44*a15))+a42*(a13*(a34*a55-a54*a35)-a33*(a14*a55-a54*a15)+a53*(a14*a35-a34*a15))-a52*(a13*(a34*a45-a44*a35)-a33*(a14*a45-a44*a15)+a43*(a14*a35-a34*a15)))+a31*(a12*(a23*(a44*a55-a54*a45)-a43*(a24*a55-a54*a25)+a53*(a24*a45-a44*a25))-a22*(a13*(a44*a55-a54*a45)-a43*(a14*a55-a54*a15)+a53*(a14*a45-a44*a15))+a42*(a13*(a24*a55-a54*a25)-a23*(a14*a55-a54*a15)+a53*(a14*a25-a24*a15))-a52*(a13*(a24*a45-a44*a25)-a23*(a14*a45-a44*a15)+a43*(a14*a25-a24*a15)))-a41*(a12*(a23*(a34*a55-a54*a35)-a33*(a24*a55-a54*a25)+a53*(a24*a35-a34*a25))-a22*(a13*(a34*a55-a54*a35)-a33*(a14*a55-a54*a15)+a53*(a14*a35-a34*a15))+a32*(a13*(a24*a55-a54*a25)-a23*(a14*a55-a54*a15)+a53*(a14*a25-a24*a15))-a52*(a13*(a24*a35-a34*a25)-a23*(a14*a35-a34*a15)+a33*(a14*a25-a24*a15)))+a51*(a12*(a23*(a34*a45-a44*a35)-a33*(a24*a45-a44*a25)+a43*(a24*a35-a34*a25))-a22*(a13*(a34*a45-a44*a35)-a33*(a14*a45-a44*a15)+a43*(a14*a35-a34*a15))+a32*(a13*(a24*a45-a44*a25)-a23*(a14*a45-a44*a15)+a43*(a14*a25-a24*a15))-a42*(a13*(a24*a35-a34*a25)-a23*(a14*a35-a34*a15)+a33*(a14*a25-a24*a15))));
    }

// mmorales: THERE IS SOMETHING WRONG WITH THIS ROUTINE, BUT I DON'T USE IT ANYMORE !!! 
    inline T evaluate(T a11, T a12, T a13, T a14, T a15, T a16,
                      T a21, T a22, T a23, T a24, T a25, T a26,
                      T a31, T a32, T a33, T a34, T a35, T a36,
                      T a41, T a42, T a43, T a44, T a45, T a46,
                      T a51, T a52, T a53, T a54, T a55, T a56, 
                      T a61, T a62, T a63, T a64, T a65, T a66) 
    {
      return (a11*(a22*(a33*(a44*(a55*a66-a65*a56)-a54*(a45*a66-a65*a46)+a64*(a45*a56-a55*a46))-a43*(a34*(a55*a66-a65*a56)-a54*(a35*a66-a65*a36)+a64*(a35*a56-a55*a36))+a53*(a34*(a45*a66-a65*a46)-a44*(a35*a66-a65*a36)+a64*(a35*a46-a45*a36))-a63*(a34*(a45*a56-a55*a46)-a44*(a35*a56-a55*a36)+a54*(a35*a46-a45*a36)))-a32*(a23*(a44*(a55*a66-a65*a56)-a54*(a45*a66-a65*a46)+a64*(a45*a56-a55*a46))-a43*(a24*(a55*a66-a65*a56)-a54*(a25*a66-a65*a26)+a64*(a25*a56-a55*a26))+a53*(a24*(a45*a66-a65*a46)-a44*(a25*a66-a65*a26)+a64*(a25*a46-a45*a26))-a63*(a24*(a45*a56-a55*a46)-a44*(a25*a56-a55*a26)+a54*(a25*a46-a45*a26)))+a42*(a23*(a34*(a55*a66-a65*a56)-a54*(a35*a66-a65*a36)+a64*(a35*a56-a55*a36))-a33*(a24*(a55*a66-a65*a56)-a54*(a25*a66-a65*a26)+a64*(a25*a56-a55*a26))+a53*(a24*(a35*a66-a65*a36)-a34*(a25*a66-a65*a26)+a64*(a25*a36-a35*a26))-a63*(a24*(a35*a56-a55*a36)-a34*(a25*a56-a55*a26)+a54*(a25*a36-a35*a26)))-a52*(a23*(a34*(a45*a66-a65*a46)-a44*(a35*a66-a65*a36)+a64*(a35*a46-a45*a36))-a33*(a24*(a45*a66-a65*a46)-a44*(a25*a66-a65*a26)+a64*(a25*a46-a45*a26))+a43*(a24*(a35*a66-a65*a36)-a34*(a25*a66-a65*a26)+a64*(a25*a36-a35*a26))-a63*(a24*(a35*a46-a45*a36)-a34*(a25*a46-a45*a26)+a44*(a25*a36-a35*a26)))+a62*(a23*(a34*(a45*a56-a55*a46)-a44*(a35*a56-a55*a36)+a54*(a35*a46-a45*a36))-a33*(a24*(a45*a56-a55*a46)-a44*(a25*a56-a55*a26)+a54*(a25*a46-a45*a26))+a43*(a24*(a35*a56-a55*a36)-a34*(a25*a56-a55*a26)+a54*(a25*a36-a35*a26))-a53*(a24*(a35*a46-a45*a36)-a34*(a25*a46-a45*a26)+a44*(a25*a36-a35*a26))))
-a21*(a12*(a33*(a44*(a55*a66-a65*a56)-a54*(a45*a66-a65*a46)+a64*(a45*a56-a55*a46))-a43*(a34*(a55*a66-a65*a56)-a54*(a35*a66-a65*a36)+a64*(a35*a56-a55*a36))+a53*(a34*(a45*a66-a65*a46)-a44*(a35*a66-a65*a36)+a64*(a35*a46-a45*a36))-a63*(a34*(a45*a56-a55*a46)-a44*(a35*a56-a55*a36)+a54*(a35*a46-a45*a36)))-a32*(a13*(a44*(a55*a66-a65*a56)-a54*(a45*a66-a65*a46)+a64*(a45*a56-a55*a46))-a43*(a14*(a55*a66-a65*a56)-a54*(a15*a66-a65*a16)+a64*(a15*a56-a55*a16))+a53*(a14*(a45*a66-a65*a46)-a44*(a15*a66-a65*a16)+a64*(a15*a46-a45*a16))-a63*(a14*(a45*a56-a55*a46)-a44*(a15*a56-a55*a16)+a54*(a15*a46-a45*a16)))+a42*(a13*(a34*(a55*a66-a65*a56)-a54*(a35*a66-a65*a36)+a64*(a35*a56-a55*a36))-a33*(a14*(a55*a66-a65*a56)-a54*(a15*a66-a65*a16)+a64*(a15*a56-a55*a16))+a53*(a14*(a35*a66-a65*a36)-a34*(a15*a66-a65*a16)+a64*(a15*a36-a35*a16))-a63*(a14*(a35*a56-a55*a36)-a34*(a15*a56-a55*a16)+a54*(a15*a36-a35*a16)))-a52*(a13*(a34*(a45*a66-a65*a46)-a44*(a35*a66-a65*a36)+a64*(a35*a46-a45*a36))-a33*(a14*(a45*a66-a65*a46)-a44*(a15*a66-a65*a16)+a64*(a15*a46-a45*a16))+a43*(a14*(a35*a66-a65*a36)-a34*(a15*a66-a65*a16)+a64*(a15*a36-a35*a16))-a63*(a14*(a35*a46-a45*a36)-a34*(a15*a46-a45*a16)+a44*(a15*a36-a35*a16)))+a62*(a13*(a34*(a45*a56-a55*a46)-a44*(a35*a56-a55*a36)+a54*(a35*a46-a45*a36))-a33*(a14*(a45*a56-a55*a46)-a44*(a15*a56-a55*a16)+a54*(a15*a46-a45*a16))+a43*(a14*(a35*a56-a55*a36)-a34*(a15*a56-a55*a16)+a54*(a15*a36-a35*a16))-a53*(a14*(a35*a46-a45*a36)-a34*(a15*a46-a45*a16)+a44*(a15*a36-a35*a16))))
+a31*(a12*(a23*(a44*(a55*a66-a65*a56)-a54*(a45*a66-a65*a46)+a64*(a45*a56-a55*a46))-a43*(a24*(a55*a66-a65*a56)-a54*(a25*a66-a65*a26)+a64*(a25*a56-a55*a26))+a53*(a24*(a45*a66-a65*a46)-a44*(a25*a66-a65*a26)+a64*(a25*a46-a45*a26))-a63*(a24*(a45*a56-a55*a46)-a44*(a25*a56-a55*a26)+a54*(a25*a46-a45*a26)))-a22*(a13*(a44*(a55*a66-a65*a56)-a54*(a45*a66-a65*a46)+a64*(a45*a56-a55*a46))-a43*(a14*(a55*a66-a65*a56)-a54*(a15*a66-a65*a16)+a64*(a15*a56-a55*a16))+a53*(a14*(a45*a66-a65*a46)-a44*(a15*a66-a65*a16)+a64*(a15*a46-a45*a16))-a63*(a14*(a45*a56-a55*a46)-a44*(a15*a56-a55*a16)+a54*(a15*a46-a45*a16)))+a42*(a13*(a24*(a55*a66-a65*a56)-a54*(a25*a66-a65*a26)+a64*(a25*a56-a55*a26))-a23*(a14*(a55*a66-a65*a56)-a54*(a15*a66-a65*a16)+a64*(a15*a56-a55*a16))+a53*(a14*(a25*a66-a65*a26)-a24*(a15*a66-a65*a16)+a64*(a15*a26-a25*a16))-a63*(a14*(a25*a56-a55*a26)-a24*(a15*a56-a55*a16)+a54*(a15*a26-a25*a16)))-a52*(a13*(a24*(a45*a66-a65*a46)-a44*(a25*a66-a65*a26)+a64*(a25*a46-a45*a26))-a23*(a14*(a45*a66-a65*a46)-a44*(a15*a66-a65*a16)+a64*(a15*a46-a45*a16))+a43*(a14*(a25*a66-a65*a26)-a24*(a15*a66-a65*a16)+a64*(a15*a26-a25*a16))-a63*(a14*(a25*a46-a45*a26)-a24*(a15*a46-a45*a16)+a44*(a15*a26-a25*a16)))+a62*(a13*(a24*(a45*a56-a55*a46)-a44*(a25*a56-a55*a26)+a54*(a25*a46-a45*a26))-a23*(a14*(a45*a56-a55*a46)-a44*(a15*a56-a55*a16)+a54*(a15*a46-a45*a16))+a43*(a14*(a25*a56-a55*a26)-a24*(a15*a56-a55*a16)+a54*(a15*a26-a25*a16))-a53*(a14*(a25*a46-a45*a26)-a24*(a15*a46-a45*a16)+a44*(a15*a26-a25*a16))))
-a41*(a12*(a23*(a34*(a55*a66-a65*a56)-a54*(a35*a66-a65*a36)+a64*(a35*a56-a55*a36))-a33*(a24*(a55*a66-a65*a56)-a54*(a25*a66-a65*a26)+a64*(a25*a56-a55*a26))+a53*(a24*(a35*a66-a65*a36)-a34*(a25*a66-a65*a26)+a64*(a25*a36-a35*a26))-a63*(a24*(a35*a56-a55*a36)-a34*(a25*a56-a55*a26)+a54*(a25*a36-a35*a26)))-a22*(a13*(a34*(a55*a66-a65*a56)-a54*(a35*a66-a65*a36)+a64*(a35*a56-a55*a36))-a33*(a14*(a55*a66-a65*a56)-a54*(a15*a66-a65*a16)+a64*(a15*a56-a55*a16))+a53*(a14*(a35*a66-a65*a36)-a34*(a15*a66-a65*a16)+a64*(a15*a36-a35*a16))-a63*(a14*(a35*a56-a55*a36)-a34*(a15*a56-a55*a16)+a54*(a15*a36-a35*a16)))+a32*(a13*(a24*(a55*a66-a65*a56)-a54*(a25*a66-a65*a26)+a64*(a25*a56-a55*a26))-a23*(a14*(a55*a66-a65*a56)-a54*(a15*a66-a65*a16)+a64*(a15*a56-a55*a16))+a53*(a14*(a25*a66-a65*a26)-a24*(a15*a66-a65*a16)+a64*(a15*a26-a25*a16))-a63*(a14*(a25*a56-a55*a26)-a24*(a15*a56-a55*a16)+a54*(a15*a26-a25*a16)))-a52*(a13*(a24*(a35*a66-a65*a36)-a34*(a25*a66-a65*a26)+a64*(a25*a36-a35*a26))-a23*(a14*(a35*a66-a65*a36)-a34*(a15*a66-a65*a16)+a64*(a15*a36-a35*a16))+a33*(a14*(a25*a66-a65*a26)-a24*(a15*a66-a65*a16)+a64*(a15*a26-a25*a16))-a63*(a14*(a25*a36-a35*a26)-a24*(a15*a36-a35*a16)+a34*(a15*a26-a25*a16)))+a62*(a13*(a24*(a35*a56-a55*a36)-a34*(a25*a56-a55*a26)+a54*(a25*a36-a35*a26))-a23*(a14*(a35*a56-a55*a36)-a34*(a15*a56-a55*a16)+a54*(a15*a36-a35*a16))+a33*(a14*(a25*a56-a55*a26)-a24*(a15*a56-a55*a16)+a54*(a15*a26-a25*a16))-a53*(a14*(a25*a36-a35*a26)-a24*(a15*a36-a35*a16)+a34*(a15*a26-a25*a16))))+a51*(a12*(a23*(a34*(a45*a66-a65*a46)-a44*(a35*a66-a65*a36)+a64*(a35*a46-a45*a36))-a33*(a24*(a45*a66-a65*a46)-a44*(a25*a66-a65*a26)+a64*(a25*a46-a45*a26))+a43*(a24*(a35*a66-a65*a36)-a34*(a25*a66-a65*a26)+a64*(a25*a36-a35*a26))-a63*(a24*(a35*a46-a45*a36)-a34*(a25*a46-a45*a26)+a44*(a25*a36-a35*a26)))-a22*(a13*(a34*(a45*a66-a65*a46)-a44*(a35*a66-a65*a36)+a64*(a35*a46-a45*a36))-a33*(a14*(a45*a66-a65*a46)-a44*(a15*a66-a65*a16)+a64*(a15*a46-a45*a16))+a43*(a14*(a35*a66-a65*a36)-a34*(a15*a66-a65*a16)+a64*(a15*a36-a35*a16))-a63*(a14*(a35*a46-a45*a36)-a34*(a15*a46-a45*a16)+a44*(a15*a36-a35*a16)))+a32*(a13*(a24*(a45*a66-a65*a46)-a44*(a25*a66-a65*a26)+a64*(a25*a46-a45*a26))-a23*(a14*(a45*a66-a65*a46)-a44*(a15*a66-a65*a16)+a64*(a15*a46-a45*a16))+a43*(a14*(a25*a66-a65*a26)-a24*(a15*a66-a65*a16)+a64*(a15*a26-a25*a16))-a63*(a14*(a25*a46-a45*a26)-a24*(a15*a46-a45*a16)+a44*(a15*a26-a25*a16)))-a42*(a13*(a24*(a35*a66-a65*a36)-a34*(a25*a66-a65*a26)+a64*(a25*a36-a35*a26))-a23*(a14*(a35*a66-a65*a36)-a34*(a15*a66-a65*a16)+a64*(a15*a36-a35*a16))+a33*(a14*(a25*a66-a65*a26)-a24*(a15*a66-a65*a16)+a64*(a15*a26-a25*a16))-a63*(a14*(a25*a36-a35*a26)-a24*(a15*a36-a35*a16)+a34*(a15*a26-a25*a16)))+a62*(a13*(a24*(a35*a46-a45*a36)-a34*(a25*a46-a45*a26)+a44*(a25*a36-a35*a26))-a23*(a14*(a35*a46-a45*a36)-a34*(a15*a46-a45*a16)+a44*(a15*a36-a35*a16))+a33*(a14*(a25*a46-a45*a26)-a24*(a15*a46-a45*a16)+a44*(a15*a26-a25*a16))-a43*(a14*(a25*a36-a35*a26)-a24*(a15*a36-a35*a16)+a34*(a15*a26-a25*a16))))
+a61*(a12*(a23*(a34*(a45*a56-a55*a46)-a44*(a35*a56-a55*a36)+a54*(a35*a46-a45*a36))-a33*(a24*(a45*a56-a55*a46)-a44*(a25*a56-a55*a26)+a54*(a25*a46-a45*a26))+a43*(a24*(a35*a56-a55*a36)-a34*(a25*a56-a55*a26)+a54*(a25*a36-a35*a26))-a53*(a24*(a35*a46-a45*a36)-a34*(a25*a46-a45*a26)+a44*(a25*a36-a35*a26)))-a22*(a13*(a34*(a45*a56-a55*a46)-a44*(a35*a56-a55*a36)+a54*(a35*a46-a45*a36))-a33*(a14*(a45*a56-a55*a46)-a44*(a15*a56-a55*a16)+a54*(a15*a46-a45*a16))+a43*(a14*(a35*a56-a55*a36)-a34*(a15*a56-a55*a16)+a54*(a15*a36-a35*a16))-a53*(a14*(a35*a46-a45*a36)-a34*(a15*a46-a45*a16)+a44*(a15*a36-a35*a16)))+a32*(a13*(a24*(a45*a56-a55*a46)-a44*(a25*a56-a55*a26)+a54*(a25*a46-a45*a26))-a23*(a14*(a45*a56-a55*a46)-a44*(a15*a56-a55*a16)+a54*(a15*a46-a45*a16))+a43*(a14*(a25*a56-a55*a26)-a24*(a15*a56-a55*a16)+a54*(a15*a26-a25*a16))-a53*(a14*(a25*a46-a45*a26)-a24*(a15*a46-a45*a16)+a44*(a15*a26-a25*a16)))-a42*(a13*(a24*(a35*a56-a55*a36)-a34*(a25*a56-a55*a26)+a54*(a25*a36-a35*a26))-a23*(a14*(a35*a56-a55*a36)-a34*(a15*a56-a55*a16)+a54*(a15*a36-a35*a16))+a33*(a14*(a25*a56-a55*a26)-a24*(a15*a56-a55*a16)+a54*(a15*a26-a25*a16))-a53*(a14*(a25*a36-a35*a26)-a24*(a15*a36-a35*a16)+a34*(a15*a26-a25*a16)))+a52*(a13*(a24*(a35*a46-a45*a36)-a34*(a25*a46-a45*a26)+a44*(a25*a36-a35*a26))-a23*(a14*(a35*a46-a45*a36)-a34*(a15*a46-a45*a16)+a44*(a15*a36-a35*a16))+a33*(a14*(a25*a46-a45*a26)-a24*(a15*a46-a45*a16)+a44*(a15*a26-a25*a16))-a43*(a14*(a25*a36-a35*a26)-a24*(a15*a36-a35*a16)+a34*(a15*a26-a25*a16))))); 
    } 

    inline T evaluate(Matrix<T>& dots, vector<int>::iterator it, int n) 
    {
      //vector<T>::iterator d(M.data());
      T* d = &(M[0]);
      for(int i=0; i<n; i++)
       for(int j=0; j<n; j++)
        //M(i,j) = dots(*(it+i),*(it+n+j));
         *(d++)= dots(*(it+i),*(it+n+j));
      return Determinant(&(M[0]),n,n,&(Pivot[0]));
    }

  };
    
}
#endif
/***************************************************************************
 * $RCSfile$   $Author: kesler $
 * $Revision: 3574 $   $Date: 2009-02-19 17:11:24 -0600 (Thu, 19 Feb 2009) $
 * $Id: DiracDeterminantBase.h 3574 2009-02-19 23:11:24Z kesler $ 
 ***************************************************************************/
