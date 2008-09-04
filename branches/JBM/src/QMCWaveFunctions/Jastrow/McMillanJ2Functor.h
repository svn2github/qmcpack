//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim
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
#ifndef QMCPLUSPLUS_MCMILLANJ2_H
#define QMCPLUSPLUS_MCMILLANJ2_H
#include "Numerics/OptimizableFunctorBase.h"

namespace qmcplusplus {
  /** ModPade Jastrow functional
   *
   * \f[ u(r) = \frac{1}{2*A}\left[1-\exp(-A*r)\right], \f]
   */
  template<class T>
      struct McMillanJ2Functor: public OptimizableFunctorBase 
  {

    typedef typename OptimizableFunctorBase::real_type real_type;

      ///coefficients
    real_type A;
    real_type B;

    string ID_A;
    string ID_B;

      /** constructor
     * @param a A coefficient
     * @param samespin boolean to indicate if this function is for parallel spins
       */
    McMillanJ2Functor(real_type a=5.0, real_type b=4.9133):ID_A("McA"),ID_B("McB")
    {
      reset(a,b);
    }

    OptimizableFunctorBase* makeClone() const 
    {
      return new McMillanJ2Functor<T>(*this);
    }
    inline void reset() {
      reset(A,B);
    }

      /** reset the internal variables.
     *
     * USE_resetParameters
       */
    void resetParameters(const opt_variables_type& active)
    {
      int ia=myVars.where(0); if(ia>-1) A=active[ia];
      int ib=myVars.where(1); if(ib>-1) B=active[ib];
      reset(A,B);
    }
    
    void checkInVariables(opt_variables_type& active)
    {
      active.insertFrom(myVars);
    }

    void checkOutVariables(const opt_variables_type& active)
    {
      myVars.getIndex(active);
    }

    
      /** reset the internal variables.
     *@param a New Jastrow parameter a 
       */
    inline void reset(real_type a, real_type b) {
      A = a;
      B = b;
    }

      /** evaluate the value at r
     * @param r the distance
     * @return \f$ u(r) = \frac{A}{r}\left[1-\exp(-\frac{r}{F})\right]\f$
       */
    inline real_type evaluate(real_type r) {
      return std::pow(B/r,A);
    }

      /**@param r the distance
    @param dudr first derivative
    @param d2udr2 second derivative
    @return the value
       */
    inline real_type evaluate(real_type r, real_type& dudr, real_type& d2udr2) {
      real_type wert = std::pow(B/r,A);
      dudr = -A*wert/r;
      d2udr2= (A+1.)*A*wert/(r*r);
      return wert;
    }

      /** return a value at r
       */
    real_type f(real_type r) {
      return evaluate(r);
    }

      /** return a derivative at r
       */
    real_type df(real_type r) {
      return -A*std::pow(B/r,A)/r;
    }

      /** Read in the parameter from the xml input file.
     * @param cur current xmlNode from which the data members are reset
       */
    bool put(xmlNodePtr cur) {
      xmlNodePtr tcur = cur->xmlChildrenNode;
      while(tcur != NULL) {
          //@todo Var -> <param(eter) role="opt"/>
        string cname((const char*)(tcur->name));
        if(cname == "parameter" || cname == "Var") {
          string aname((const char*)(xmlGetProp(tcur,(const xmlChar *)"name")));
//            string idname((const char*)(xmlGetProp(tcur,(const xmlChar *)"id")));
          if(aname == "a") {
            putContent(A,tcur);
          } else if(aname == "b") {
            ID_B = (const char*)(xmlGetProp(tcur,(const xmlChar *)"id"));
            putContent(B,tcur);
          }
        }
        tcur = tcur->next;
      }
      myVars.insert(ID_A,A);
      myVars.insert(ID_B,B);
      reset(A,B);
      return true;
    }

    
  };
  
//   template<class T>
//       struct ModMcMillanJ2Functor: public OptimizableFunctorBase
//   {
// 
//     typedef typename OptimizableFunctorBase::real_type real_type;
// 
//       ///coefficients
//     real_type A;
//     real_type B;
//     real_type RC;
//     real_type c0,c1,c2,c3,c4,c5,c6;
// 
//     string ID_A,ID_B,ID_RC;
// 
//       /** constructor
//      * @param a A coefficient
//      * @param samespin boolean to indicate if this function is for parallel spins
//        */
//     ModMcMillanJ2Functor(real_type a=4.9133, real_type b=5, real_type rc=1.0):ID_RC("Rcutoff"),ID_A("A"),ID_B("B")
//     {
//       A = a;
//       B = b;
//       RC = rc;
//       reset();
//     }
// 
//     
//     void checkInVariables(opt_variables_type& active)
//     {
//       active.insertFrom(myVars);
//     }
// 
//     void checkOutVariables(const opt_variables_type& active)
//     {
//       myVars.getIndex(active);
//     }
//     
//     OptimizableFunctorBase* makeClone() const 
//     {
//       return new ModMcMillanJ2Functor<T>(*this);
//     }
//     
//     void resetParameters(const opt_variables_type& active) 
//     {
//       int ia=myVars.where(0); if(ia>-1) A=active[ia];
//       int ib=myVars.where(1); if(ib>-1) B=active[ib];
//       int ic=myVars.where(2); if(ic>-1) RC=active[ic];
//       reset();
//     }
// 
//     inline void reset() {
//       c0 = std::pow(A,B);
//       c1 = -1.0*std::pow(A/RC,B);
//       c2 =  B*c0*std::pow(RC,-B-1);
//       c3 =  -0.5*(B+1)*B*c0*std::pow(RC,-B-2);
//       
//       c4 = -B*c0;
//       c5 = 2.0*c3;
//       c6 = B*(B+1.0)*c0;
//     }
// 
// 
//       /** evaluate the value at r
//      * @param r the distance
//      * @return \f$ u(r) = \frac{A}{r}\left[1-\exp(-\frac{r}{F})\right]\f$
//        */
//     inline real_type evaluate(real_type r) {
//       if (r<RC) {
//         real_type r1 = (r-RC);
//         return c0*std::pow(r,-B)+c1 + r1*(c2+r1*c3) ;
//       }
//       else {return 0.0;};
//     }
// 
//       /**@param r the distance
//     @param dudr first derivative
//     @param d2udr2 second derivative
//     @return the value
//        */
//     inline real_type evaluate(real_type r, real_type& dudr, real_type& d2udr2) {
//       if (r<RC){
//         real_type r1 = (r-RC);
//         real_type overr = 1.0/r;
//         real_type wert = std::pow(1.0/r,B);
//         dudr = c4*wert*overr + c2 + c5*r1;
//         d2udr2= c6*wert*overr*overr+ c5 ;
//         return c0*wert + c1 + r1*(c2+r1*c3) ;
//       } else {
//         dudr = 0.0;
//         d2udr2= 0.0;
//         return 0.0;
//       };
//     }
// 
//       /** return a value at r
//        */
//     real_type f(real_type r) {
//       return evaluate(r);
//     }
// 
//       /** return a derivative at r
//        */
//     real_type df(real_type r) {
//       if (r<RC) {return c4*std::pow(1.0/r,B+1) + c2 + c5*(r-RC) ;}
//       else {return 0.0;};
//     }
// 
//       /** Read in the parameter from the xml input file.
//      * @param cur current xmlNode from which the data members are reset
//        */
//     bool put(xmlNodePtr cur) {
//       RC = cutoff_radius;
//       xmlNodePtr tcur = cur->xmlChildrenNode;
//       while(tcur != NULL) {
//           //@todo Var -> <param(eter) role="opt"/>
//         string cname((const char*)(tcur->name));
//         if(cname == "parameter" || cname == "Var") {
//           string aname((const char*)(xmlGetProp(tcur,(const xmlChar *)"name")));
//           string idname((const char*)(xmlGetProp(tcur,(const xmlChar *)"id")));
//           if(aname == "A") {
//             ID_A = idname;
//             putContent(A,tcur);
//           } else if(aname == "B") {
//             ID_B = idname;
//             putContent(B,tcur);
//           }else if(aname == "RC")
//           {
//             ID_RC = idname;
//             putContent(RC,tcur);
//           }
//         }
//         tcur = tcur->next;
//       }
//       myVars.insert(ID_A,A);
//       myVars.insert(ID_B,B);
//       myVars.insert(ID_RC,RC);
//       reset();
//       LOGMSG("  Modified McMillan Parameters ")
//           LOGMSG("    A (" << ID_A << ") = " << A  << "  B (" << ID_B << ") =  " << B<< "  R_c (" << ID_RC << ") =  " << RC)
//       return true;
//     }
// 
//   };
  
  template<class T>
      struct ModMcMillanJ2Functor: public OptimizableFunctorBase
  {

    ///f(r) = A*r^-5 - A*r_c^-5 + A/(B*r_c^6) * tanh(B*(r-r_c))
    typedef typename OptimizableFunctorBase::real_type real_type;

      ///coefficients
    real_type A;
    real_type B;
    real_type RC;
    real_type cA,c0,c1,c2,c3,c4,c5;

    string ID_A,ID_B,ID_RC;

      /** constructor
     * @param a A coefficient
     * @param samespin boolean to indicate if this function is for parallel spins
       */
    ModMcMillanJ2Functor(real_type a=0.0, real_type b=0.0, real_type rc=0.0):ID_RC("Rcutoff"),ID_A("A"),ID_B("B")
    {
    }

    
    void checkInVariables(opt_variables_type& active)
    {
      active.insertFrom(myVars);
    }

    void checkOutVariables(const opt_variables_type& active)
    {
      myVars.getIndex(active);
    }
    
    OptimizableFunctorBase* makeClone() const 
    {
      return new ModMcMillanJ2Functor<T>(*this);
    }
    
    void resetParameters(const opt_variables_type& active) 
    {
      int ia=myVars.where(0); if(ia>-1) A=active[ia];
      int ib=myVars.where(1); if(ib>-1) B=active[ib];
      int ic=myVars.where(2); if(ic>-1) RC=active[ic];
      reset();
    }

    inline void reset() {
      cA = std::pow(A,5);
      c0 = cA*std::pow(RC,-5);
      c1 = 5*cA*std::pow(RC,-6)/B;
      c2 = -5*cA;
      c3 = B*c1;
      c4 = 30*cA;
      c5 = B*c3;
    }


      /** evaluate the value at r
     * @param r the distance
     * @return \f$ u(r) = \frac{A}{r}\left[1-\exp(-\frac{r}{F})\right]\f$
       */
    inline real_type evaluate(real_type r) {
      if (r<RC) {
        real_type r1 = (r-RC);
        real_type plZ = std::exp(B*r1);
        real_type miZ = 1.0/plZ;
        real_type tanhZ = (plZ-miZ)/(plZ+miZ);
        return cA*std::pow(r,-5)-c0 + c1*tanhZ;
      }
      else {return 0.0;};
    }

      /**@param r the distance
    @param dudr first derivative
    @param d2udr2 second derivative
    @return the value
       */
    inline real_type evaluate(real_type r, real_type& dudr, real_type& d2udr2) {
      if (r<RC){
        real_type prt = cA*std::pow(r,-5);
        real_type r1 = (r-RC);
        real_type plZ = std::exp(B*r1);
        real_type miZ = 1.0/plZ;
        real_type tanhZ = (plZ-miZ)/(plZ+miZ);
        real_type sechZ = 2.0/(plZ+miZ);
        real_type sechZ2 = sechZ*sechZ; 
        real_type ret = prt - c0 + c1*tanhZ;
        
        prt = prt/r;
        dudr = -5*prt + c3*sechZ2;
        prt = prt/r;
        d2udr2 = 30*prt + c5*sechZ2*tanhZ;
        return ret;
      } else {
        dudr = 0.0;
        d2udr2= 0.0;
        return 0.0;
      };
    }

      /** return a value at r
       */
    real_type f(real_type r) {
      return evaluate(r);
    }

      /** return a derivative at r
       */
    real_type df(real_type r) {
      if (r<RC) {
        real_type r1 = (r-RC);
        real_type plZ = std::exp(B*r1);
        real_type miZ = 1.0/plZ;
        real_type sechZ = 2.0/(plZ+miZ);
        real_type sechZ2 = sechZ*sechZ; 
        return c2*std::pow(r,-6) + c3*sechZ2;
      }
      else {return 0.0;};
    }

      /** Read in the parameter from the xml input file.
     * @param cur current xmlNode from which the data members are reset
       */
    bool put(xmlNodePtr cur) {
      RC = cutoff_radius;
      xmlNodePtr tcur = cur->xmlChildrenNode;
      while(tcur != NULL) {
          //@todo Var -> <param(eter) role="opt"/>
        string cname((const char*)(tcur->name));
        if(cname == "parameter" || cname == "Var") {
          string aname((const char*)(xmlGetProp(tcur,(const xmlChar *)"name")));
          string idname((const char*)(xmlGetProp(tcur,(const xmlChar *)"id")));
          if(aname == "A") {
            ID_A = idname;
            putContent(A,tcur);
          } else if(aname == "B") {
            ID_B = idname;
            putContent(B,tcur);
          }else if(aname == "RC")
          {
            ID_RC = idname;
            putContent(RC,tcur);
          }
        }
        tcur = tcur->next;
      }
      myVars.insert(ID_A,A);
      myVars.insert(ID_B,B);
      myVars.insert(ID_RC,RC);
      reset();
      LOGMSG("  Modified McMillan Parameters ")
          LOGMSG("    A (" << ID_A << ") = " << A  << "  B (" << ID_B << ") =  " << B<< "  R_c (" << ID_RC << ") =  " << RC)
          return true;
    }

  };
  
  template<class T>
      struct Mod2McMillanJ2Functor: public OptimizableFunctorBase
  {

    ///f(r) = A*r^-5 - A*r_c^-5 + A/(B*r_c^6) * tanh(B*(r-r_c))
    typedef typename OptimizableFunctorBase::real_type real_type;

      ///coefficients
    real_type A;
    real_type B;
    real_type RC;
    real_type cA,c0,c1,c2,c3,c4,c5;
    real_type alpha,V_a,D;

    string ID_A,ID_B,ID_RC,ID_D;

      /** constructor
     * @param a A coefficient
     * @param samespin boolean to indicate if this function is for parallel spins
       */
    Mod2McMillanJ2Functor(real_type a=0.0, real_type b=0.0, real_type rc=0.0):ID_RC("Rcutoff"),ID_A("A"),ID_B("B")
    {
    }

    
    void checkInVariables(opt_variables_type& active)
    {
      active.insertFrom(myVars);
    }

    void checkOutVariables(const opt_variables_type& active)
    {
      myVars.getIndex(active);
    }
    
    OptimizableFunctorBase* makeClone() const 
    {
      return new Mod2McMillanJ2Functor<T>(*this);
    }
    
    void resetParameters(const opt_variables_type& active) 
    {
      int ia=myVars.where(0); if(ia>-1) A=active[ia];
      int ib=myVars.where(1); if(ib>-1) B=active[ib];
      int ic=myVars.where(2); if(ic>-1) RC=active[ic];
      reset();
    }

    inline void reset() {
      cA = std::pow(A,5);
      c0 = cA*std::pow(RC,-5);
      c1 = 5*cA*std::pow(RC,-6)/B;
      c2 = -5*cA;
      c3 = B*c1;
      c4 = 30*cA;
      c5 = B*c3;
    }


      /** evaluate the value at r
     * @param r the distance
     * @return \f$ u(r) = \frac{A}{r}\left[1-\exp(-\frac{r}{F})\right]\f$
       */
    inline real_type evaluate(real_type r) {
      if (r<RC) {
        real_type r1 = (r-RC);
        real_type plZ = std::exp(B*r1);
        real_type miZ = 1.0/plZ;
        real_type tanhZ = (plZ-miZ)/(plZ+miZ);
        return cA*std::pow(r,-5)-c0 + c1*tanhZ;
      }
      else {return 0.0;};
    }

      /**@param r the distance
    @param dudr first derivative
    @param d2udr2 second derivative
    @return the value
       */
    inline real_type evaluate(real_type r, real_type& dudr, real_type& d2udr2) {
      if (r<RC){
        real_type prt = cA*std::pow(r,-5);
        real_type r1 = (r-RC);
        real_type plZ = std::exp(B*r1);
        real_type miZ = 1.0/plZ;
        real_type tanhZ = (plZ-miZ)/(plZ+miZ);
        real_type sechZ = 2.0/(plZ+miZ);
        real_type sechZ2 = sechZ*sechZ; 
        real_type ret = prt - c0 + c1*tanhZ;
        
        prt = prt/r;
        dudr = -5*prt + c3*sechZ2;
        prt = prt/r;
        d2udr2 = 30*prt + c5*sechZ2*tanhZ;
        return ret;
      } else {
        dudr = 0.0;
        d2udr2= 0.0;
        return 0.0;
      };
    }

      /** return a value at r
       */
    real_type f(real_type r) {
      return evaluate(r);
    }

      /** return a derivative at r
       */
    real_type df(real_type r) {
      if (r<RC) {
        real_type r1 = (r-RC);
        real_type plZ = std::exp(B*r1);
        real_type miZ = 1.0/plZ;
        real_type sechZ = 2.0/(plZ+miZ);
        real_type sechZ2 = sechZ*sechZ; 
        return c2*std::pow(r,-6) + c3*sechZ2;
      }
      else {return 0.0;};
    }

      /** Read in the parameter from the xml input file.
     * @param cur current xmlNode from which the data members are reset
       */
    bool put(xmlNodePtr cur) {
      RC = cutoff_radius;
      xmlNodePtr tcur = cur->xmlChildrenNode;
      while(tcur != NULL) {
          //@todo Var -> <param(eter) role="opt"/>
        string cname((const char*)(tcur->name));
        if(cname == "parameter" || cname == "Var") {
          string aname((const char*)(xmlGetProp(tcur,(const xmlChar *)"name")));
          string idname((const char*)(xmlGetProp(tcur,(const xmlChar *)"id")));
          if(aname == "A") {
            ID_A = idname;
            putContent(A,tcur);
          } else if(aname == "B") {
            ID_B = idname;
            putContent(B,tcur);
          }else if(aname == "RC")
          {
            ID_RC = idname;
            putContent(RC,tcur);
          }
        }
        tcur = tcur->next;
      }
      myVars.insert(ID_A,A);
      myVars.insert(ID_B,B);
      myVars.insert(ID_RC,RC);
      reset();
      LOGMSG("  Modified McMillan Parameters ")
          LOGMSG("    A (" << ID_A << ") = " << A  << "  B (" << ID_B << ") =  " << B<< "  R_c (" << ID_RC << ") =  " << RC)
          return true;
    }

  };
  
  
  template<class T>
      struct HFDHE2Functor: public OptimizableFunctorBase
  {

    ///almost the integral of the potential. Ignoring integral of cutoff term.
    typedef typename OptimizableFunctorBase::real_type real_type;

      ///coefficients
      real_type A,alpha,alphaA,Aoveralpha,c1,c2,c3,c1p,c2p,c3p,c1pp,c2pp,c3pp,D,twoD,D2,SHIFT;

    HFDHE2Functor()
    {
      A = 18.63475757;
      alpha = -2.381392669;
      alphaA = A*alpha;
      Aoveralpha = A/alpha;
      c1=1.460008056/5.0;
      c2=14.22016431/7.0;
      c3=187.2033646/9.0;
      c1p=c1*5.0;
      c2p=c2*7.0;
      c3p=c3*9.0;
      c1pp=c1p*6.0;
      c2pp=c2p*8.0;
      c3pp=c3p*10.0;
      D = 6.960524706;
      twoD = 2.0*D;
      D2 = D*D;
      SHIFT=0.0;
    }
    void checkInVariables(opt_variables_type& active)
    {
    }

    void checkOutVariables(const opt_variables_type& active)
    {
    }
    
    OptimizableFunctorBase* makeClone() const 
    {
      return new HFDHE2Functor<T>(*this);
    }
    
    void resetParameters(const opt_variables_type& active) 
    {
    }

    inline void reset() {
    }


      /** evaluate the value at r
     * @param r the distance
     * @return \f$ u(r) = \frac{A}{r}\left[1-\exp(-\frac{r}{F})\right]\f$
       */
    inline real_type evaluate(real_type r) {
      real_type rm1 = 1.0/r;
      real_type rm5 = std::pow(rm1,5);
      real_type rm6 = rm5*rm1;
      real_type rm7 = rm6*rm1;
      real_type rm8 = rm7*rm1;
      real_type rm9 = rm8*rm1;
      real_type rm10 = rm9*rm1;
      real_type rm11 = rm10*rm1;
      return -(Aoveralpha*std::exp(alpha*r) - (c1*rm5+c2*rm7+c3*rm9)*dampF(r))-SHIFT;
    }

      /**@param r the distance
    @param dudr first derivative
    @param d2udr2 second derivative
    @return the value
       */
    inline real_type evaluate(real_type r, real_type& dudr, real_type& d2udr2) {
      if (r<D){
        real_type expart = std::exp(alpha*r);
        real_type rm1 = 1.0/r;
        real_type rm2 = rm1*rm1;
        real_type rm3 = rm2*rm1;
        real_type rm4 = rm2*rm2;
        real_type rm5 = std::pow(rm1,5);
        real_type rm6 = rm5*rm1;
        real_type rm7 = rm6*rm1;
        real_type rm8 = rm7*rm1;
        real_type rm9 = rm8*rm1;
        real_type rm10 = rm9*rm1;
        real_type rm11 = rm10*rm1;
      
        real_type sumPart = (c1*rm5+c2*rm7+c3*rm9);
        real_type dsumPart = (c1p*rm6+c2p*rm8+c3p*rm10);
        real_type d2sumPart = (c1pp*rm7+c2pp*rm9+c3pp*rm11);
        real_type dDampF = twoD*rm3*(D-r);
        real_type d2DampF = twoD*rm4*(-3.0*D+2.0*r);
      
        real_type Value = Aoveralpha*expart + sumPart*dampF(r);
        real_type prt = (dsumPart - sumPart*dDampF);
      
        dudr = -(A*expart - prt*dampF(r));
        d2udr2 = -(alphaA*expart+(d2sumPart + sumPart*d2DampF - dsumPart*dDampF - prt*dDampF)*dampF(r));
        return -Value-SHIFT;
      }else{
        real_type expart = std::exp(alpha*r);
        real_type rm1 = 1.0/r;
        real_type rm5 = std::pow(rm1,5);
        real_type rm6 = rm5*rm1;
        real_type rm7 = rm6*rm1;
        real_type rm8 = rm7*rm1;
        real_type rm9 = rm8*rm1;
        real_type rm10 = rm9*rm1;
        real_type rm11 = rm10*rm1;
      
        real_type sumPart = (c1*rm5+c2*rm7+c3*rm9);
        real_type dsumPart = (c1p*rm6+c2p*rm8+c3p*rm10);
        real_type d2sumPart = (c1pp*rm7+c2pp*rm9+c3pp*rm11);

        real_type Value = Aoveralpha*expart + sumPart;
        dudr = -(A*expart - dsumPart);
        d2udr2 = -(alphaA*expart+d2sumPart);
        return -Value-SHIFT;
      }
    }

      /** return a value at r
       */
    real_type f(real_type r) {
      return evaluate(r);
    }

      /** return a derivative at r
       */
    real_type df(real_type r) {
      if (r<D){
        real_type expart = std::exp(alpha*r);
        real_type rm1 = 1.0/r;
        real_type rm2 = rm1*rm1;
        real_type rm3 = rm2*rm1;
        real_type rm4 = rm2*rm2;
        real_type rm5 = std::pow(rm1,5);
        real_type rm6 = rm5*rm1;
        real_type rm7 = rm6*rm1;
        real_type rm8 = rm7*rm1;
        real_type rm9 = rm8*rm1;
        real_type rm10 = rm9*rm1;
        real_type rm11 = rm10*rm1;
      
        real_type sumPart = (c1*rm5+c2*rm7+c3*rm9);
        real_type dsumPart = (c1p*rm6+c2p*rm8+c3p*rm10);
        real_type dDampF = twoD*rm3*(D-r);
        real_type prt = (dsumPart - sumPart*dDampF);
        
        return -(A*expart - prt*dampF(r));
      }else{
        real_type expart = std::exp(alpha*r);
        real_type rm1 = 1.0/r;
        real_type rm5 = std::pow(rm1,5);
        real_type rm6 = rm5*rm1;
        real_type rm7 = rm6*rm1;
        real_type rm8 = rm7*rm1;
        real_type rm9 = rm8*rm1;
        real_type rm10 = rm9*rm1;
        real_type rm11 = rm10*rm1;
      
        real_type sumPart = (c1*rm5+c2*rm7+c3*rm9);
        real_type dsumPart = (c1p*rm6+c2p*rm8+c3p*rm10);
        return -(A*expart - dsumPart);
      }
    }

    inline real_type dampF(real_type r) {
      if (r < D){
        real_type t1=(D/r - 1.0);
        return std::exp(-t1*t1);
      }
      else
        return 1.0;
    }
      /** Read in the parameter from the xml input file.
     * @param cur current xmlNode from which the data members are reset
       */
    bool put(xmlNodePtr cur) {
      SHIFT = evaluate(cutoff_radius);
      return true;
    }

  };
}
#endif
/***************************************************************************
 * $RCSfile$   $Author: dcyang2 $
 * $Revision: 2801 $   $Date: 2008-07-09 14:25:22 -0500 (Wed, 09 Jul 2008) $
 * $Id: McMillanJ2Functor.h 2801 2008-07-09 19:25:22Z dcyang2 $ 
 ***************************************************************************/
