//////////////////////////////////////////////////////////////////
// (c) Copyright 2003  by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
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
#ifndef QMCPLUSPLUS_WALKER_H
#define QMCPLUSPLUS_WALKER_H

#include "OhmmsPETE/OhmmsMatrix.h"
#include "Utilities/PooledData.h"
#include "Utilities/PointerPool.h"
//#include "Utilities/cuda_allocator.h"
#include "CUDA/gpu_vector.h"
#include <cstdio>

namespace qmcplusplus {

  /** an enum denoting index of physical properties 
   *
   * LOCALPOTENTIAL should be always the last enumeation 
   */
  enum {LOGPSI=0,       /*!< log(fabs(psi)) instead of square of the many-body wavefunction \f$|\Psi|^2\f$ */
	SIGN,           /*!< value of the many-body wavefunction \f$\Psi(\{R\})\f$ */
        UMBRELLAWEIGHT, /*!< sum of wavefunction ratios for multiple H and Psi */
        R2ACCEPTED,     /*!< r^2 for accepted moves */
        R2PROPOSED,     /*!< r^2 for proposed moves */
        DRIFTSCALE,     /*!< scaling value for the drift */
	LOCALENERGY,    /*!< local energy, the sum of all the components */
	LOCALPOTENTIAL, /*!< local potential energy = local energy - kinetic energy */
	NUMPROPERTIES   /*!< the number of properties */
       };
  
  /** A container class to represent a walker.
   *
   * A walker stores the particle configurations {R}  and a property container.
   * The template (P)articleSet(A)ttribute is a generic container  of position types.
   * The template (G)radient(A)ttribute is a generic container of gradients types.
   * Data members for each walker
   * - ID : identity for a walker. default is 0. 
   * - Age : generation after a move is accepted.
   * - Weight : weight to take the ensemble averages
   * - Multiplicity : multiplicity for branching. Probably can be removed.
   * - Properties  : 2D container. The first index corresponds to the H/Psi index and second index >=NUMPROPERTIES.
   * - DataSet : anonymous container. 
   */
  template<typename T, typename PA, typename LA, typename GA=PA>
  struct Walker 
  {
    
    enum {DIM=PA::Type_t::Size};

    ///typedef for the property container, fixed size
    typedef Matrix<T>      PropertyContainer_t;
    typedef PooledData<T>  Buffer_t;

    ///id reserved for forward walking
    long ID;
    ///id reserved for forward walking
    long ParentID;
    ///DMCgeneration
    int Generation;
    ///Age of this walker age is incremented when a walker is not moved after a sweep
    int Age;
    ///Weight of the walker
    T Weight;
    /** Number of copies for branching
     *
     * When Multiplicity = 0, this walker will be destroyed.
     */
    T Multiplicity;

    /**the configuration vector (3N-dimensional vector to store
       the positions of all the particles for a single walker)*/
    PA R;

    ///** \f$ \nabla_i d\log \Psi for the i-th particle */
    GA Grad;
    ///** \f$ \nabla^2_i d\log \Psi for the i-th particle */
    LA Lap;

    ///drift of the walker \f$ Drift({\bf R}) = \tau v_{drift}({\bf R}) \f$
    GA Drift;

    ///scalar properties of a walker
    PropertyContainer_t  Properties;

    ///buffer for the data for particle-by-particle update
    Buffer_t DataSet;

    /// Data for GPU-vectorized versions
#ifdef QMC_CUDA
    static int cuda_DataSize;
    typedef gpu::device_vector<CUDA_PRECISION> cuda_Buffer_t;
    cuda_Buffer_t cuda_DataSet;
    // Note that R_GPU has size N+1.  The last element contains the
    // proposed position for single-particle moves.
    gpu::device_vector<TinyVector<CUDA_PRECISION,OHMMS_DIM> > R_GPU, Grad_GPU;
    gpu::device_vector<CUDA_PRECISION> Lap_GPU;
    inline void resizeCuda(int size) {
      if (cuda_DataSize != size)
	cuda_DataSize = size;
      cuda_DataSet.resize(size);
    }

#endif


    ///default constructor
    inline Walker() : ID(0),ParentID(0), Generation(0),Age(0),
		      Weight(1.0e0),Multiplicity(1.0e0), 
		      cuda_DataSet("Walker::walker_buffer"),
		      R_GPU("Walker::R_GPU"), Grad_GPU("Walker::Grad_GPU"),
		      Lap_GPU("Walker::Lap_GPU")
		      
    {
      Properties.resize(1,NUMPROPERTIES);
      reset();
    }

    ///create a walker for n-particles
    inline explicit Walker(int nptcl) : ID(0),ParentID(0), Generation(0),Age(0),
					Weight(1.0e0),Multiplicity(1.0e0), 
				      	cuda_DataSet("Walker::walker_buffer"),
				      	R_GPU("Walker::R_GPU"), 
					Grad_GPU("Walker::Grad_GPU"),
				      	Lap_GPU("Walker::Lap_GPU")
    {
      Properties.resize(1,NUMPROPERTIES);
      resize(nptcl);
      reset();
    }

    inline ~Walker() { 
    }
    
    ///assignment operator
    inline Walker& operator=(const Walker& a) {
      if(this != &a) makeCopy(a);
      return *this;
    }

    ///return the number of particles per walker
    inline int size() const { return R.size(); }

    ///resize for n particles
    inline void resize(int nptcl) 
    {
      R.resize(nptcl); Grad.resize(nptcl),Lap.resize(nptcl),Drift.resize(nptcl); 
#ifdef QMC_CUDA
      R_GPU.resize(nptcl);
      Grad_GPU.resize(nptcl);
      Lap_GPU.resize(nptcl);
#endif
      // R.resize(nptcl); Drift.resize(nptcl); 
    }

    ///copy the content of a walker
    inline void makeCopy(const Walker& a) 
    {    
      ID=a.ID;
      ParentID=a.ParentID;
      Generation=a.Generation;
      Age=a.Age;
      Weight=a.Weight;
      Multiplicity=a.Multiplicity;
      if(R.size()!=a.R.size()) resize(a.R.size());
      R = a.R;
      Drift = a.Drift;
      Properties.copy(a.Properties);
      DataSet=a.DataSet;
      cuda_DataSet = a.cuda_DataSet;
      R_GPU = a.R_GPU;
      Grad_GPU = a.Grad_GPU;
      Lap_GPU = a.Lap_GPU;
    }

    //return the address of the values of Hamiltonian terms
    inline T* restrict getPropertyBase() {
      return Properties.data();
    }

    //return the address of the values of Hamiltonian terms
    inline const T* restrict getPropertyBase() const {
      return Properties.data();
    }

    ///return the address of the i-th properties
    inline T* restrict getPropertyBase(int i) {
      return Properties[i];
    }

    ///return the address of the i-th properties
    inline const T* restrict getPropertyBase(int i) const {
      return Properties[i];
    }


    /** reset the property of a walker
     *@param logpsi \f$\log |\Psi|\f$
     *@param sigN  sign of the trial wavefunction
     *@param ene the local energy
     *
     *Assign the values and reset the age
     * but leave the weight and multiplicity 
     */
    inline void resetProperty(T logpsi, T sigN, T ene) 
    {
      Age=0;
      //Weight=1.0;
      Properties(LOGPSI)=logpsi;
      Properties(SIGN)=sigN;
      Properties(LOCALENERGY) = ene;
    }

    /** reset the property of a walker
     * @param logpsi \f$\log |\Psi|\f$
     * @param sigN  sign of the trial wavefunction
     * @param ene the local energy
     * @param r2a \f$r^2\f$ for the accepted moves
     * @param r2p \f$r^2\f$ for the proposed moves
     * @param vq \f$\bar{V}/V\f$ scaling to control node divergency in JCP 93
     *
     *Assign the values and reset the age
     * but leave the weight and multiplicity 
     */
    inline void resetProperty(T logpsi, T sigN, T ene, T r2a, T r2p, T vq) 
    {
      Age=0;
      Properties(LOGPSI)=logpsi;
      Properties(SIGN)=sigN;
      Properties(LOCALENERGY) = ene;
      Properties(R2ACCEPTED) = r2a;
      Properties(R2PROPOSED) = r2p;
      Properties(DRIFTSCALE) = vq;
    }
    /** marked to die
     *
     * Multiplicity and weight are set to zero.
     */
    inline void willDie() {
      Multiplicity=0;
      Weight=0.0;
    }

    /** reset the walker weight, multiplicity and age */
    inline void reset() {
      Age=0;
      Multiplicity=1.0e0;
      Weight=1.0e0;
    }

    inline void resizeProperty(int n, int m) {
      Properties.resize(n,m);
    }


    /** byte size for a packed message 
     *
     * ID, Age, Properties, R, Drift, DataSet is packed
     */
    inline int byteSize() 
    {
      int bsize = 2*sizeof(long)+2*sizeof(int)+(Properties.size()+DIM*2*R.size()+DataSet.size())*sizeof(T);
#ifdef QMC_CUDA
      bsize += 2 *sizeof (int); // size and N
      bsize += cuda_DataSize             * sizeof(CUDA_PRECISION); // cuda_DataSet
      bsize += R.size()      * OHMMS_DIM * sizeof(CUDA_PRECISION); // R_GPU
      bsize += R.size()      * OHMMS_DIM * sizeof(CUDA_PRECISION); // Grad_GPU
      bsize += R.size()      * 1         * sizeof(CUDA_PRECISION); // Lap_GPU
#endif   
      return bsize;
    }

    template<class Msg>
    inline Msg& putMessage(Msg& m) {
      int nat=R.size();
      m << ID << ParentID << Generation << Age;
      //m << Generation << Age;
      for(int iat=0; iat<nat;iat++) R[iat].putMessage(m);
      for(int iat=0; iat<nat;iat++) Drift[iat].putMessage(m);
      Properties.putMessage(m);
      DataSet.putMessage(m);
#ifdef QMC_CUDA
      // Pack GPU data
      gpu::host_vector<CUDA_PRECISION> host_data;
      gpu::host_vector<TinyVector<CUDA_PRECISION,OHMMS_DIM> > R_host;
      gpu::host_vector<CUDA_PRECISION> host_lapl;

      host_data = cuda_DataSet;
      R_host = R_GPU;
      int size = host_data.size();
      int N = R_host.size();
      m.Pack(size);
      m.Pack(N);
      m.Pack(&(host_data[0]), host_data.size());
      m.Pack(&(R_host[0][0]), OHMMS_DIM*R_host.size());
      R_host = Grad_GPU;
      m.Pack(&(R_host[0][0]), OHMMS_DIM*R_host.size());
      
      host_lapl = Lap_GPU;
      m.Pack(&(host_lapl[0]), host_lapl.size());
#endif

      return m;
    }

    template<class Msg>
    inline Msg& getMessage(Msg& m) {
      int nat=R.size();
      m>>ID >> ParentID >> Generation >> Age;
      //m>> Generation >> Age;
      for(int iat=0; iat<nat;iat++) R[iat].getMessage(m);
      for(int iat=0; iat<nat;iat++) Drift[iat].getMessage(m);
      Properties.getMessage(m);
      DataSet.getMessage(m);
#ifdef QMC_CUDA
      // Pack GPU data
      gpu::host_vector<CUDA_PRECISION> host_data;
      gpu::host_vector<TinyVector<CUDA_PRECISION,OHMMS_DIM> > R_host;
      gpu::host_vector<CUDA_PRECISION> host_lapl;

      int size, N;
      m.Unpack(size);
      m.Unpack(N);
      host_data.resize(size);
      R_host.resize(N);
      host_lapl.resize(N);

      m.Unpack(&(host_data[0]), size);
      cuda_DataSet = host_data;

      m.Unpack(&(R_host[0][0]), OHMMS_DIM*N);
      R_GPU = R_host;
      m.Unpack(&(R_host[0][0]), OHMMS_DIM*N);
      Grad_GPU = R_host;
      
      m.Unpack(&(host_lapl[0]), N);
      Lap_GPU = host_lapl;
#endif



      return m;
    }

  };

  template<class T, class PA, class LA>
  ostream& operator<<(ostream& out, const Walker<T,PA,LA>& rhs)
    {
      copy(rhs.Properties.begin(), rhs.Properties.end(), 
      	   ostream_iterator<double>(out," "));
      out << endl;
      out << rhs.R;
      return out;
    }
}

#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
