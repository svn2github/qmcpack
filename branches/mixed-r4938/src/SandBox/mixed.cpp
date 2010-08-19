#include <Configuration.h>
#include <spline/einspline_engine.hpp>
#include <OhmmsPETE/OhmmsArray.h>
#include <Message/OpenMP.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/Timer.h>
#include <Utilities/IteratorUtility.h>
using namespace qmcplusplus;

int main(int argc, char **argv)
{

  //einspline_test<multi_UBspline_3d_s> sp_s;
  //einspline_test<multi_UBspline_3d_d> sp_d;
  //einspline_test<multi_UBspline_3d_c> sp_c;
  //einspline_test<multi_UBspline_3d_z> sp_z;

  typedef double value_type;
  typedef double real_type;

  TinyVector<int,3> ng(48);
  int num_splines=128;
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
//      case 'p':
//        nsamples=atoi(optarg);
//        break;
      case 'i':
        niters=atoi(optarg);
        break;
    }
  }

  typedef einspline_engine<multi_UBspline_3d_d> spliner_type;

  spliner_type master_engine(ng, num_splines);
  {
    Array<value_type, 3> data(ng[0], ng[1], ng[2]);
    for (int i = 0; i < num_splines; ++i)
    {
      for (int j = 0; j < data.size(); ++j)
        data(j) = Random();
      master_engine.set(i, data.data(), data.size());
    }
  }


  {
    vector<spliner_type*> myeng(omp_get_max_threads());
    Timer clock;
#pragma parallel omp
    {
      int ip = omp_get_thread_num();
      myeng[ip] = new spliner_type(master_engine);
      RandomGenerator_t rng;
      for (int i = 0; i < niters; ++i)
      {
        TinyVector<real_type, 3> here(rng(), rng(), rng());
        myeng[ip]->evaluate_v(here);
      }
    }
    double dt=clock.elapsed();
    cout << "Timing = " << dt << " nops=" << niters*num_splines/dt << endl;
    delete_iter(myeng.begin(), myeng.end());
  }
//  einspline_engine<multi_UBspline_3d_s> spliner(ng,num_splines);
//  spliner.evaluate_v(here);
//  cout << spliner.Val << endl;
//  spliner.evaluate_vg(here);
//  spliner.evaluate_vgl(here);
//  spliner.evaluate_vgh(here);

  return 0;
}
