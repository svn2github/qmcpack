//////////////////////////////////////////////////////////////////
// (c) Copyright 2003- by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Jeongnim Kim
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
//   Department of Physics, Ohio State University
//   Ohio Supercomputer Center
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_MULTICHAIN_H
#define QMCPLUSPLUS_MULTICHAIN_H
#include "Particle/MCWalkerConfiguration.h"
#include "OhmmsPETE/OhmmsVector.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "Utilities/IteratorUtility.h"
#include "Numerics/HDFNumericAttrib.h"
#include "QMCDrivers/SpaceWarp.h"
#include <deque>
namespace qmcplusplus {

  struct Bead: public MCWalkerConfiguration::Walker_t{

    typedef MCWalkerConfiguration::Walker_t Walker_t;
    typedef MCWalkerConfiguration::RealType RealType;
    typedef MCWalkerConfiguration::ParticlePos_t ParticlePos_t;
 
    Vector<int> BeadSignWgt;    
    vector<ParticlePos_t*> Gradients;
    Matrix<RealType> Action;
    RealType TransProb[2];

    inline Bead(const Bead& a) {
      makeCopyBead(a); 
    }

    inline Bead(const Walker_t& a){
      makeCopy(a);
      int rows=Properties.rows();
      Resize_Grad_and_Action(rows,R.size());
      BeadSignWgt.resize(rows);
    }

    inline Bead& operator=(const Bead& a) {
      makeCopyBead(a);
      return *this;
    }

    inline void makeCopyBead(const Bead& a){
      makeCopy(a);
      int rows=a.Gradients.size();
      Resize_Grad_and_Action(rows,a.size());
      Action=a.Action;
      for(int i=0; i<rows; i++) *Gradients[i] = *(a.Gradients[i]);
      BeadSignWgt.resize(rows);
      BeadSignWgt=a.BeadSignWgt;
      TransProb[0]=a.TransProb[0];
      TransProb[1]=a.TransProb[1];
    }

    inline void Resize_Grad_and_Action(int n, int m){
      int curg=Gradients.size();
      while(curg<n) {
        Gradients.push_back(new ParticlePos_t(m));
        ++curg;
      }
      Action.resize(n,3);
    }

    /** copy the restart data to buf 
     * @param buf buffer to write
     */
    inline void registerData(Buffer_t& buf) {
      buf.add(get_first_address(R),get_last_address(R));
      buf.add(get_first_address(Drift),get_last_address(Drift)); 
      vector<ParticlePos_t*>::iterator git(Gradients.begin()), git_end(Gradients.end());
      while(git != git_end) {
        buf.add(get_first_address(**git),get_last_address(**git)); ++git;
      }
      buf.add(BeadSignWgt.begin(),BeadSignWgt.end());
      buf.add(TransProb[0]);
      buf.add(TransProb[1]);
      buf.add(Action.begin(),Action.end());
      buf.add(Properties.begin(),Properties.end());
    }

    /** copy the restart data from buf 
     * @param buf buffer to read from
     */
    inline void copyFromBuffer(Buffer_t& buf) {
      buf.get(get_first_address(R),get_last_address(R));
      buf.get(get_first_address(Drift),get_last_address(Drift)); 
      vector<ParticlePos_t*>::iterator git(Gradients.begin()), git_end(Gradients.end());
      while(git != git_end) {
        buf.get(get_first_address(**git),get_last_address(**git)); ++git;
      }
      //buf.get(BeadSignWgt.begin(),BeadSignWgt.end());
      for(int i=0; i<BeadSignWgt.size(); i++) buf.get(BeadSignWgt[i]);
      buf.get(TransProb[0]);
      buf.get(TransProb[1]);
      buf.get(Action.begin(),Action.end());
      buf.get(Properties.begin(),Properties.end());
    }

    /** copy the restart data to buf 
     * @param buf buffer to write
     */
    inline void copyToBuffer(Buffer_t& buf) {
      buf.put(get_first_address(R),get_last_address(R));
      buf.put(get_first_address(Drift),get_last_address(Drift)); 
      vector<ParticlePos_t*>::iterator git(Gradients.begin()), git_end(Gradients.end());
      while(git != git_end) {
        buf.put(get_first_address(**git),get_last_address(**git)); ++git;
      }
      buf.put(BeadSignWgt.begin(),BeadSignWgt.end());
      buf.put(TransProb[0]);
      buf.put(TransProb[1]);
      buf.put(Action.begin(),Action.end());
      buf.put(Properties.begin(),Properties.end());
    }

    inline void getDrift(vector<RealType>& LogNorm) {
      int npsi(Properties.rows());
      //compute Drift
      RealType denom(0.e0),wgtpsi;
      Drift=0.e0;
      for(int ipsi=0; ipsi<npsi; ipsi++) {
        wgtpsi=BeadSignWgt[ipsi]*std::exp(2.0*( Properties(ipsi,LOGPSI)- LogNorm[ipsi]
                                          -Properties(0,LOGPSI)   + LogNorm[0]));
        denom += wgtpsi;
        Drift += (wgtpsi*(*Gradients[ipsi]));
      }
      denom=1.0/denom;
      Drift *= denom;
    }

    inline void getDrift(vector<RealType>& Jacobian, SpaceWarp& PtclWarp,vector<RealType>& LogNorm) {

      QMCTraits::PosType WarpDrift;
      int npsi(Properties.rows());
      int nptcl(Gradients[0]->size());
      //compute Drift
      RealType denom(0.e0),wgtpsi;
      Drift=0.e0; 
      for(int ipsi=0; ipsi<npsi; ipsi++) {
        wgtpsi=BeadSignWgt[ipsi]*Jacobian[ipsi]*std::exp(2.0*( Properties(ipsi,LOGPSI)-LogNorm[ipsi]
                                                         -Properties(0,LOGPSI)   +LogNorm[0]));
        denom += wgtpsi;
        for(int iptcl=0; iptcl< nptcl; iptcl++){
          WarpDrift=dot(  (*Gradients[ipsi])[iptcl],PtclWarp.get_Jacob_matrix(iptcl,ipsi)  )
            +5.0e-1*PtclWarp.get_grad_ln_Jacob(iptcl,ipsi) ;
          Drift[iptcl] += (wgtpsi*WarpDrift);
        }
      }
      denom=1.0/denom;
      //Drift = denom*Drift;
      Drift *= denom;
    }
  };

  //struct MultiChain: public std::deque<Bead*> {
  struct MultiChain {

    typedef std::deque<Bead*>               Container_t;
    typedef Container_t::iterator           iterator;
    typedef Container_t::const_iterator     const_iterator;
    typedef Container_t::reference          reference;
    typedef Container_t::const_reference    const_reference;

    typedef MCWalkerConfiguration::Walker_t Walker_t;
    typedef MCWalkerConfiguration::RealType RealType;
    typedef Bead::Buffer_t                  Buffer_t;

    /// Direction of growth
    int GrowthDirection;

    ///The index of the middle
    int Middle;    

    ///The index of the Last
    int Last;

    ///The number of H/Psi pairs
    int nPsi;

    RealType GlobalWgt;

    Vector<RealType> GlobalAction,UmbrellaWeight;
    Vector<int>      GlobalSignWgt,RefSign;

    Container_t      Beads;

    /**  constructor    
     * @param abead Bead used to generate len Beads to form a chain
     * @param len size of the chain
     * @param direction initial growth direction
     * @param npsi number of Psi/H pairs
     */
    //MultiChain(Walker_t* awalker,int len, int direction, int npsi): 
    MultiChain(Bead* abead,int len, int direction, int npsi): 
      GrowthDirection(direction), nPsi(npsi){
      //always add number of beads
      Middle = len/2;
      Last = len-1;
      for(int i=0; i<len; i++) {
        Beads.push_back(new Bead(*abead));
      }
      GlobalAction.resize(npsi);   GlobalAction=0.0;
      UmbrellaWeight.resize(npsi); UmbrellaWeight=1.0;
      GlobalSignWgt.resize(npsi);  GlobalSignWgt=0;
      RefSign.resize(npsi); RefSign=0;
    }
    
    /** destructor
     *
     * Need to clean up the walkers in the repository and the polymer chain
     */
    ~MultiChain() {
      delete_iter(Beads.begin(),Beads.end());
    }

    inline reference operator[](int n) { return Beads[n];}
    inline const_reference operator[](int n) const { return Beads[n];}
    inline iterator begin() { return Beads.begin();}
    inline iterator end() { return Beads.end();}
    inline reference front() { return Beads.front();}
    inline reference back() { return Beads.back();}
    INLINE_ALL void push_front(Bead* abead) {
      Beads.push_front(abead);
    }
    INLINE_ALL void push_back(Bead* abead) {
      Beads.push_back(abead);
    }
    INLINE_ALL void pop_back() {
      Beads.pop_back();
    }
    INLINE_ALL void pop_front() {
      Beads.pop_front();
    }
    inline void flip(){ 
      GrowthDirection = abs(GrowthDirection-1); //flip the direction
    }

    inline void setRefSign(int ipsi, RealType phase) {
      RefSign[ipsi]=(phase>M_PI_2)?-1:1;
    }

    inline int getSign(RealType phase) {
      return (phase>M_PI_2)?-1:1;
    }

    /** copy the restart data from buf 
     * @param buf buffer to read from
     */
    inline void copyFromBuffer(Buffer_t& buf) {
      int n(Beads.size());
      buf.get(n);
      buf.get(GrowthDirection);
      buf.get(Middle);
      buf.get(Last);
      buf.get(nPsi);
      buf.get(GlobalWgt);
      buf.get(GlobalAction.begin(),GlobalAction.end());
      buf.get(UmbrellaWeight.begin(),UmbrellaWeight.end());
      for(int i=0; i<GlobalSignWgt.size(); i++) buf.get(GlobalSignWgt[i]);
      for(int i=0; i<RefSign.size(); i++) buf.get(RefSign[i]);
      //buf.get(GlobalSignWgt.begin(),GlobalSignWgt.end());
      //buf.get(RefSign.begin(),RefSign.end());
    }

    /** add the restart data to buf 
     * @param buf buffer to write
     *
     * add takes care of memory allocation and assignment
     */
    inline void copyToBuffer(Buffer_t& buf) {
      double n= static_cast<double>(Beads.size());
      buf.add(n);
      buf.add(GrowthDirection);
      buf.add(Middle);
      buf.add(Last);
      buf.add(nPsi);
      buf.add(GlobalWgt);
      buf.add(GlobalAction.begin(),GlobalAction.end());
      buf.add(UmbrellaWeight.begin(),UmbrellaWeight.end());
      buf.add(GlobalSignWgt.begin(),GlobalSignWgt.end());
      buf.add(RefSign.begin(),RefSign.end());
    }

    /** read multi-chain configuration from a file
     * @param aroot root name
     * @return true, if the input file contains valid data
     */
    bool read(const string& aroot);

    /** read MultiChain tree a group *
     * @param grp hdf5 group 
     * @return true, if the input file contains valid data
     * 
     * Typically, grp is the file id
     */
    bool read(hid_t grp);

    /** write MultiChain tree to a group
     * @param grp hdf5 group
     *
     * Typically, grp is the file id
     */
    void write(hid_t grp);

  };
}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
