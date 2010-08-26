#include <Configuration.h>
#include <spline/einspline_engine.hpp>
#include <OhmmsPETE/OhmmsArray.h>
#include <Message/Communicate.h>
#include <mpi/collectives.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/Timer.h>
#include <Utilities/IteratorUtility.h>
#include <Utilities/UtilityFunctions.h>
void report_num_threads(int level, int pid)
{
#pragma omp critical
  {
    printf("Level %d: parent %d number of threads in the team - %d %d\n"
        , level, pid, omp_get_num_threads(), omp_get_thread_num());
  }
}

namespace qmcplusplus {
  template<typename EngT>
    class einspline_omp
    {
      typedef einspline_engine<EngT> spliner_type;
      enum {DIM=spliner_type::DIM};

      typedef typename spliner_type::value_type value_type;

      vector<spliner_type*> Engines;
      vector<int> Offsets;

      public:
      einspline_omp(){}

      ~einspline_omp()
      {
        delete_iter(Engines.begin(),Engines.end());
      }

      void create_plan(const TinyVector<int,DIM>& ng, int nsplines, int np=1, bool randomize=false)
      {
        report_num_threads(1,0);

        if(Engines.size() < np)
        {
          delete_iter(Engines.begin(), Engines.end());
          Engines.resize(np,0);
        }

        Offsets.resize(Engines.size()+1);
        FairDivideLow(nsplines,Engines.size(),Offsets);
        
#pragma omp parallel
        {
          int ip=omp_get_thread_num();
          int ns_loc=Offsets[ip+1]-Offsets[ip];//nsplines/Engines.size();
          if(Engines[ip]) delete Engines[ip];
          Engines[ip]=new spliner_type(ng,ns_loc);
          if(randomize)
          {
            Array<typename spliner_type::value_type, 3> data(ng[0], ng[1], ng[2]);
            for (int i = 0; i < ns_loc; ++i)
            {
              for (int j = 0; j < data.size(); ++j)
                data(j) = Random();
              Engines[ip]->set(i, data.data(), data.size());
            }
          }
        }
      }

      template<typename T1>
      inline void evaluate_v(const TinyVector<T1,DIM>& here)
      {
        const int np=Engines.size();
//#pragma omp parallel for
#pragma omp for
        for(int ip=0; ip<np; ++ip) Engines[ip]->evaluate_v(here);
      }

      template<typename T1>
      inline void evaluate_v(const vector<TinyVector<T1,DIM> >& here)
      {
        const int np=Engines.size();
//#pragma omp parallel for
#pragma omp for
        for(int ip=0; ip<np; ++ip) 
        {
          for(int i=0; i<here.size(); ++i)
            Engines[ip]->evaluate_v(here[i]);
        }
      }

      template<typename T1>
      inline void evaluate_nested(const vector<TinyVector<T1,DIM> >& here)
      {
        const int np=Engines.size();
//#pragma omp parallel for
#pragma omp for
        for(int ip=0; ip<np; ++ip) 
        {
#pragma omp parallel num_threads(4)
          {
            value_type v[128];
#pragma omp for
            for(int i=0; i<here.size(); ++i)
              Engines[ip]->evaluate(here[i],v);
          }
        }
      }

      template<typename T1>
      inline void evaluate_vgh(const TinyVector<T1,DIM>& here)
      {
        const int np=Engines.size();
#pragma omp parallel for
        for(int ip=0; ip<np; ++ip) Engines[ip]->evaluate_vgh(here);
      }
    };
}

using namespace qmcplusplus;

int main(int argc, char **argv)
{

  OHMMS::Controller->initialize(argc,argv);
  Communicate* mycomm=OHMMS::Controller;
  OhmmsInfo Welcome("mixed",mycomm->rank());
  Random.init(0,1,11);
  //einspline_test<multi_UBspline_3d_s> sp_s;
  //einspline_test<multi_UBspline_3d_d> sp_d;
  //einspline_test<multi_UBspline_3d_c> sp_c;
  //einspline_test<multi_UBspline_3d_z> sp_z;

  typedef double value_type;
  typedef double real_type;

  TinyVector<int,3> ng(48);
  int num_splines=128;
  int num_samples=14;
  int niters=1000;
  int opt;

  while((opt = getopt(argc, argv, "hg:x:y:z:i:s:p:")) != -1) {
    switch(opt) {
      case 'h':
        printf("[-g grid| -x grid_x -y grid_y -z grid_z] -s states -p particles -i iterations\n");
        return 1;
      case 'g':
        ng=atoi(optarg);
        break;
      case 'x':
        ng[0]=atoi(optarg);
        break;
      case 'y':
        ng[1]=atoi(optarg);
        break;
      case 'z':
        ng[2]=atoi(optarg);
        break;
      case 's':
        num_splines=atoi(optarg);
        break;
      case 'p':
        num_samples=atoi(optarg);
        break;
      case 'i':
        niters=atoi(optarg);
        break;
    }
  }

  einspline_omp<multi_UBspline_3d_d> myeng;
  myeng.create_plan(ng,num_splines,omp_get_max_threads(),true);
  RandomGenerator_t rng;

  typedef vector<TinyVector<real_type,3> > pos_array_type;
  vector<pos_array_type*> randvec;
  for(int i=0; i<niters; ++i)
  {
    pos_array_type* av=new pos_array_type(num_samples);
    for(int j=0; j<num_samples; ++j)
      (*av)[j]=TinyVector<real_type,3>(rng(),rng(),rng());
    randvec.push_back(av);
  }

  omp_set_nested(true);
  omp_set_max_active_levels(2);

  Timer clock;
  double dt_v=0.0;
  clock.restart();
#pragma omp parallel 
  for (int i = 0; i < niters; ++i)
    for(int j=0; j<num_samples; ++j)
      myeng.evaluate_v(randvec[i]->operator[](j));
  dt_v+=clock.elapsed();

  double dt_v2=0.0;
  clock.restart();
#pragma omp parallel 
  for (int i = 0; i < niters; ++i)
    myeng.evaluate_v(*randvec[i]);
  dt_v2+=clock.elapsed();

  double dt_v3=0.0;
  clock.restart();
#pragma omp parallel 
  for (int i = 0; i < niters; ++i)
    myeng.evaluate_nested(*randvec[i]);
  dt_v3+=clock.elapsed();

  double dt_vgh=0.0;
  report_num_threads(1,0);
  for (int i = 0; i < niters; ++i)
  {
    TinyVector<real_type, 3> here(rng(), rng(), rng());
    clock.restart();
    myeng.evaluate_vgh(here);
    dt_vgh+=clock.elapsed();
  }

  //mycomm->barrier();
  //mpi::reduce(*mycomm,dt_v);
  //mpi::reduce(*mycomm,dt_vgh);
  //dt_v/=static_cast<double>(mycomm->size());
  //dt_vgh/=static_cast<double>(mycomm->size());

  num_splines *= mycomm->size();
  app_log().setf(std::ios::scientific, std::ios::floatfield);
  app_log().precision(6);
  app_log() << "einspline_omp " << setw(3) << omp_get_max_threads() 
    <<setw(6) << ng[0] << setw(6) << num_splines << setw(6) << num_samples
    << "  " << niters*num_samples*num_splines/dt_v
    << "  " << niters*num_samples*num_splines/dt_v2
    << "  " << niters*num_samples*num_splines/dt_v3
    << "  " << niters*num_splines/dt_vgh
    << endl;

  delete_iter(randvec.begin(),randvec.end());
  return 0;
}
