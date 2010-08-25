//////////////////////////////////////////////////////////////////
// (c) Copyright 2010-  by Jeongnim Kim and Kenneth P Esler
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
/**@file einspline_engine.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_ENGINE_HPP
#define QMCPLUSPLUS_EINSPLINE_ENGINE_HPP
#include <spline/bspline_traits.hpp>
#include <spline/einspline_impl.hpp>
#include <OhmmsPETE/OhmmsVector.h>
namespace qmcplusplus
{

  /** einspline_engine
   * @tparam EngT type of einspline engine, e.g., multi_UBspline_3d_d
   * Container and handler of einspline  libraries
   *
   * Each object owns a set of states which can be globally addressed
   * [first_index,last_index).
   */
  template<typename EngT>
  class einspline_engine
  {
  public:
    enum
    {
      DIM = bspline_engine_traits<EngT>::DIM
    };
    typedef typename bspline_engine_traits<EngT>::real_type real_type;
    typedef typename bspline_engine_traits<EngT>::value_type value_type;
    typedef typename bspline_engine_traits<EngT>::Spline_t Spline_t;
    typedef typename bspline_engine_traits<EngT>::BCtype_t BCtype_t;
    /// owner of Spliner
    bool own_spliner;
    ///the lower bound of the index
    int first_index;
    ///the upper bound of the index
    int last_index;
    ///the number of spline objects owned by this class
    int num_splines;
    ///spline engine
    Spline_t *spliner;
    ///grid
    Ugrid my_grid[DIM];
    ///boundary conditions
    BCtype_t bconds[DIM];

    ///values
    Vector<value_type> Val;
    ///gradients
    Vector<TinyVector<value_type, DIM> > Grad;
    ///laplacians
    Vector<value_type> Lap;
    ///hessians
    Vector<Tensor<value_type, DIM> > Hess;

    /** default constructor
     *
     * initialize bconds to be periodic
     */
    einspline_engine() :
      own_spliner(false), spliner(0), first_index(0), num_splines(0)
    {
    }

    einspline_engine(const TinyVector<int, DIM>& npts, int norbs, int first=0) :
      own_spliner(false), spliner(0), first_index(first), num_splines(norbs)
    {
      create_plan(npts, norbs);
    }

    /// copy constructor
    einspline_engine(einspline_engine<EngT>& rhs) :
      own_spliner(false), first_index(rhs.first_index), num_splines(rhs.num_splines), spliner(rhs.spliner)
    {
      Val.resize(num_splines);
      Grad.resize(num_splines);
      Lap.resize(num_splines);
      Hess.resize(num_splines);
    }

    ~einspline_engine()
    {
      if (own_spliner&& spliner) destroy_Bspline(spliner);
    }

    void set_defaults(const TinyVector<int, DIM>& npts, int norbs=0)
    {
      for (int i = 0; i < DIM; ++i)
      {
        bconds[i].lCode = bconds[i].rCode = PERIODIC;
        my_grid[i].start = 0.0;
        my_grid[i].end = 1.0;
        my_grid[i].num = npts[i];
      }
      if(norbs>0) num_splines=norbs;
    }

    /** allocate internal storage
     * @param npts grid
     * @param norbs number of orbitals owned byt this
     */
    void create_plan(const TinyVector<int, DIM>& npts, int norbs, int first=0)
    {
      first_index=first;
      last_index=first+norbs;
      set_defaults(npts, norbs);
      create_plan();
    }

    /** allocate internal storage for the coefficients
     */
    void create_plan()
    {
      if(own_spliner&&spliner) destroy_Bspline(spliner);
      spliner=einspline::impl::create_bspline_engine(my_grid, bconds, num_splines);
      own_spliner=true;
      Val.resize(num_splines);
      Grad.resize(num_splines);
      Lap.resize(num_splines);
      Hess.resize(num_splines);
    }

    /** set the ival-th spline function with the datain
     */
    template<typename T1>
    void set(int ival, T1* datain, size_t n)
    {
      if (ival >= num_splines)
      {
        APP_ABORT("Out of bound of the orbital index");
      }
      einspline::impl::set_bspline(spliner, ival, datain);
//      if (compare_types<value_type, T1>::same)
//        einspline::impl::set_bspline(spliner, ival, datain);
//      else
//      {//not sure if the conversion should be handled here
//        std::vector<value_type> converted(n);
//        for (int i = 0; i < n; ++i)
//          convert(datain[i], converted[i]);
//        einspline::impl::set_bspline(spliner, ival, converted.data());
//      }
    }

    template<typename ValT>
    inline void evaluate_v(const TinyVector<ValT, DIM>& r)
    {
      einspline::impl::evaluate(spliner, r, Val.data());
    }

    template<typename ValT>
    inline void evaluate_vg(const TinyVector<ValT, DIM>& r)
    {
      einspline::impl::evaluate(spliner, r, Val.data(), Grad.data());
    }

    template<typename ValT>
    inline void evaluate_vgl(const TinyVector<ValT, DIM>& r)
    {
      einspline::impl::evaluate(spliner, r, Val.data(), Grad.data(), Lap.data());
    }

    template<typename ValT>
    inline void evaluate_vgh(const TinyVector<ValT, DIM>& r)
    {
      if (spliner) einspline::impl::evaluate(spliner, r, Val.data(), Grad.data(), Hess.data());
    }

    private:

  };
}

#endif
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1770 $   $Date: 2007-02-17 17:45:38 -0600 (Sat, 17 Feb 2007) $
 * $Id: OrbitalBase.h 1770 2007-02-17 23:45:38Z jnkim $ 
 ***************************************************************************/
