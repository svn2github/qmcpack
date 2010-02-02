//////////////////////////////////////////////////////////////////
// (c) Copyright 2009- by Ken Esler
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Ken Esler
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: esler@uiuc.edu
//
// Supported by 
//   National Center for Supercomputing Applications, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
/**@file DiracDeterminantCUDA.h
 * @brief Declaration of DiracDeterminantCUDA with a S(ingle)P(article)O(rbital)SetBase
 */
#ifndef QMCPLUSPLUS_DIRAC_DETERMINANT_CUDA_H
#define QMCPLUSPLUS_DIRAC_DETERMINANT_CUDA_H
#include "QMCWaveFunctions/Fermion/DiracDeterminantBase.h"
#include "QMCWaveFunctions/SPOSetBase.h"
#include "QMCWaveFunctions/Fermion/determinant_update.h"
#include "Numerics/CUDA/cuda_inverse.h"
#include "Utilities/NewTimer.h"

namespace qmcplusplus {
  class DiracDeterminantCUDA: public DiracDeterminantBase 
  {
  public:
    DiracDeterminantCUDA(SPOSetBasePtr const &spos, int first=0) :
      DiracDeterminantBase(spos, first) {}
    
    DiracDeterminantCUDA(const DiracDeterminantCUDA& s) :
      DiracDeterminantBase(s) { }
      
    
    
  };
}
#endif // QMCPLUSPLUS_DIRAC_DETERMINANT_CUDA_H
