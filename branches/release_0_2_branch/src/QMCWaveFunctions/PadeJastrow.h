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
#ifndef QMCPLUSPLUS_JASTROWFUNCTIONS_H
#define QMCPLUSPLUS_JASTROWFUNCTIONS_H
#include "OhmmsData/libxmldefs.h"
#include "Optimize/VarList.h"

/** Pade functional of \f[ u(r) = \frac{a*r}{1+b*r} \f]
 *
 * Prototype of the template parameter of TwoBodyJastrow and OneBodyJastrow
 */
template<class T>
struct PadeJastrow {
  ///coefficients
  T A, B, AB, B2;
  ///reference to the pade function
  PadeJastrow<T>* RefFunc;
  ///constructor
  PadeJastrow(T a=1.0, T b=1.0): RefFunc(0) {reset(a,b);}

  ///constructor with a PadeJastrow
  PadeJastrow(PadeJastrow<T>* func): RefFunc(func) {
    reset(RefFunc->A, RefFunc->B);
  }

  /** reset the internal variables.
   *
   * When RefPade is not 0, use RefPade->B to reset the values
   */
  inline void reset() {
    if(RefFunc) { B = RefFunc->B; }
    AB = A*B; B2=2.0*B;
  }

  /** reset the internal variables.
   *@param a Pade Jastrow parameter a 
   *@param b Pade Jastrow parameter b 
   */
  void reset(T a, T b) {
    A=a; B=b; AB=a*b; B2=2.0*b;
  }

  /** evaluate the value at r
   * @param r the distance
   * @return \f$ u(r) = a*r/(1+b*r) \f$
   */
  inline T evaluate(T r) {
    return A*r/(1.0+B*r);
  }

  /** evaluate the value, first derivative and second derivative
   * @param r the distance
   * @param dudr return value  \f$ du/dr = a/(1+br)^2 \f$
   * @param d2udr2 return value  \f$ d^2u/dr^2 = -2ab/(1+br)^3 \f$
   * @return \f$ u(r) = a*r/(1+b*r) \f$ 
   */
  inline T evaluate(T r, T& dudr, T& d2udr2) {
    T u = 1.0/(1.0+B*r);
    dudr = A*u*u;
    d2udr2 = -B2*dudr*u;
    return A*u*r;
  }

  /**@param cur current xmlNode from which the data members are reset
   @param vlist VarRegistry<T1> to which the Pade variables A and B
   are added for optimization
   @brief T1 is the type of VarRegistry, typically double.  Read 
   in the Pade parameters from the xml input file.
  */
  template<class T1>
  void put(xmlNodePtr cur, VarRegistry<T1>& vlist){
    T Atemp,Btemp;
    string ida, idb;
    //jastrow[iab]->put(cur->xmlChildrenNode,wfs_ref.RealVars);
    xmlNodePtr tcur = cur->xmlChildrenNode;
    while(tcur != NULL) {
      //@todo Var -> <param(eter) role="opt"/>
      string cname((const char*)(tcur->name));
      if(cname == "parameter" || cname == "Var") {
	string aname((const char*)(xmlGetProp(tcur,(const xmlChar *)"name")));
	string idname((const char*)(xmlGetProp(tcur,(const xmlChar *)"id")));
	if(aname == "A") {
	  ida = idname;
	  putContent(Atemp,tcur);
	} else if(aname == "B"){
	  idb = idname;
	  putContent(Btemp,tcur);
	}
      }
      tcur = tcur->next;
    }
    reset(Atemp,Btemp);
    vlist.add(ida,&A,1);
    if(RefFunc == 0) vlist.add(idb,&B,1);
    XMLReport("Jastrow Parameters = (" << A << "," << B << ")") 
  }
};

/** Pade function of \f[ u(r) = \frac{a*r+c*r^2}{1+b*r} \f]
 *
 * Prototype of the template parameter of TwoBodyJastrow and OneBodyJastrow
 */
template<class T>
struct PadeJastrow2 {

  ///coefficients
  T A, B, C, C2;

  ///constructor
  PadeJastrow2(T a=1.0, T b=1.0, T c=1.0) {reset(a,b,c);}

  PadeJastrow2(PadeJastrow2<T>* func) {
    reset(1.0,1.0,1.0);
  }

  /**
   *@brief reset the internal variables.
   */
  inline void reset() {
    C2 = 2.0*C;
  }

  /** reset the internal variables.
   *@param a Pade Jastrow parameter a 
   *@param b Pade Jastrow parameter b 
   *@param c Pade Jastrow parameter c 
   */
  void reset(T a, T b, T c) {
    A = a; B=b; C = c; C2 = 2.0*C;
  }

  /**@param r the distance
     @return \f$ u(r) = a*r/(1+b*r) \f$
  */
  inline T evaluate(T r) {
    T br(B*r);
    return (A+br)*r/(1.0+br);
  }

  /** evaluate the value at r
   * @param r the distance
     @param dudr return value  \f$ du/dr = a/(1+br)^2 \f$
     @param d2udr2 return value  \f$ d^2u/dr^2 = -2ab/(1+br)^3 \f$
     @return \f$ u(r) = a*r/(1+b*r) \f$
  */
  inline T evaluate(T r, T& dudr, T& d2udr2) {
    T u = 1.0/(1.0+B*r);
    T v = A*r+B*r*r;
    T w = A+C2*r;
    dudr = u*(w-B*u*v);
    d2udr2 = 2.0*u*(C-B*dudr);
    return u*v;
  }

  /** process input xml node
   * @param cur current xmlNode from which the data members are reset
   * @param vlist VarRegistry<T1> to which the Pade variables A and B are added for optimization
   *
   * T1 is the type of VarRegistry, typically double.  
   * Read in the Pade parameters from the xml input file.
   */
  template<class T1>
  void put(xmlNodePtr cur, VarRegistry<T1>& vlist){
    T Atemp,Btemp, Ctemp;
    string ida, idb, idc;
    //jastrow[iab]->put(cur->xmlChildrenNode,wfs_ref.RealVars);
    xmlNodePtr tcur = cur->xmlChildrenNode;
    while(tcur != NULL) {
      //@todo Var -> <param(eter) role="opt"/>
      string cname((const char*)(tcur->name));
      if(cname == "parameter") {
	string aname((const char*)(xmlGetProp(tcur,(const xmlChar *)"name")));
	string idname((const char*)(xmlGetProp(tcur,(const xmlChar *)"id")));
	if(aname == "A") {
	  ida = idname;
	  putContent(Atemp,tcur);
	} else if(aname == "B"){
	  idb = idname;
	  putContent(Btemp,tcur);
	} else if(aname == "C") {
	  idc = idname;
	  putContent(Ctemp,tcur);
	}
      }
      tcur = tcur->next;
    }
    reset(Atemp,Btemp,Ctemp);
    vlist.add(ida,&A,1);
    vlist.add(idb,&B,1);
    vlist.add(idc,&C,1);
    XMLReport("Jastrow (A*r+C*r*r)/(1+Br) = (" << A << "," << B << "," << C << ")") 
  }
};
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/

