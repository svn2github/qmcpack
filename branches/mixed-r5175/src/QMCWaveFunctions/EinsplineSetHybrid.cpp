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
/** @file EinsplineSetHybrid.cpp
 *
 * Implementation and instantiation of EinsplineSetHybrid
 * This is only used with QMC_CUDA=1
 */

#include <QMCWaveFunctions/EinsplineSetHybrid.h>

namespace qmcplusplus {

  ///////////////////////////////
  // Real StorageType versions //
  ///////////////////////////////
  template<> string
  EinsplineSetHybrid<double>::Type()
  { }
  
  
  template<typename StorageType> SPOSetBase*
  EinsplineSetHybrid<StorageType>::makeClone() const
  {
    EinsplineSetHybrid<StorageType> *clone = 
      new EinsplineSetHybrid<StorageType> (*this);
    clone->registerTimers();
    return clone;
  }
  

  //////////////////////////////////
  // Complex StorageType versions //
  //////////////////////////////////

  
  template<> string
  EinsplineSetHybrid<complex<double> >::Type()
  { }
  

  template<>
  EinsplineSetHybrid<double>::EinsplineSetHybrid() :
    CurrentWalkers(0)
  {
    ValueTimer.set_name ("EinsplineSetHybrid::ValueOnly");
    VGLTimer.set_name ("EinsplineSetHybrid::VGL");
    ValueTimer.set_name ("EinsplineSetHybrid::VGLMat");
    EinsplineTimer.set_name ("EinsplineSetHybrid::Einspline");
    className = "EinsplineSeHybrid";
    TimerManager.addTimer (&ValueTimer);
    TimerManager.addTimer (&VGLTimer);
    TimerManager.addTimer (&VGLMatTimer);
    TimerManager.addTimer (&EinsplineTimer);
    for (int i=0; i<OHMMS_DIM; i++)
      HalfG[i] = 0;
  }

  template<>
  EinsplineSetHybrid<complex<double > >::EinsplineSetHybrid() :
    CurrentWalkers(0)
  {
    ValueTimer.set_name ("EinsplineSetHybrid::ValueOnly");
    VGLTimer.set_name ("EinsplineSetHybrid::VGL");
    ValueTimer.set_name ("EinsplineSetHybrid::VGLMat");
    EinsplineTimer.set_name ("EinsplineSetHybrid::Einspline");
    className = "EinsplineSeHybrid";
    TimerManager.addTimer (&ValueTimer);
    TimerManager.addTimer (&VGLTimer);
    TimerManager.addTimer (&VGLMatTimer);
    TimerManager.addTimer (&EinsplineTimer);
    for (int i=0; i<OHMMS_DIM; i++)
      HalfG[i] = 0;
  }

  template class EinsplineSetHybrid<complex<double> >;
  template class EinsplineSetHybrid<        double  >;
}
/***************************************************************************
* $RCSfile$   $Author: jeongnim.kim $
* $Revision: 5119 $   $Date: 2011-02-06 16:20:47 -0600 (Sun, 06 Feb 2011) $
* $Id: EinsplineSetHybrid.cpp 5119 2011-02-06 22:20:47Z jeongnim.kim $
***************************************************************************/
