#ifndef CUDA_SPLINE_H
#define CUDA_SPLINE_H

#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"

namespace qmcplusplus {

  template<typename T>
  struct CudaSpline
  {
    cuda_vector<T> coefs;
    T rMax;

    template<typename T2>
    void set (BsplineFunctor<T2> &func)
    {
      int num_coefs = func.SplineCoefs.size();
      host_vector<T> coefs_h(num_coefs);
      for (int i=0; i<num_coefs; i++) {
	coefs_h[i] = func.SplineCoefs[i];
	app_log() << "coefs_h[" << i << "] = " << coefs_h[i] << endl;
      }
      coefs = coefs_h;
      rMax = func.Rcut;
    }

    template<typename T2>
    CudaSpline (BsplineFunctor<T2> &func)
    {
      set (func);
    }
  };
}

///////////////////////
// Two-Body routines //
///////////////////////

void
two_body_sum (float *R[], int e1_first, int e1_last, int e2_first, int e2_last,
	      float spline_coefs[], int numCoefs, float rMax,  
	      float lattice[], float latticeInv[], float sum[], int numWalkers);

void
two_body_sum (double *R[], int e1_first, int e1_last, int e2_first, int e2_last,
	      double spline_coefs[], int numCoefs, double rMax,  
	      double lattice[], double latticeInv[], double sum[], int numWalkers);

void
two_body_ratio (float *R[], int first, int last, int N,
		float Rnew[], int inew,
		float spline_coefs[], int numCoefs, float rMax,  
		float lattice[], float latticeInv[], float sum[], int numWalkers);

void
two_body_ratio (double *R[], int first, int last, int N,
		double Rnew[], int inew,
		double spline_coefs[], int numCoefs, double rMax,  
		double lattice[], double latticeInv[], double sum[], int numWalkers);

void
two_body_update(float *R[], int N, int iat, int numWalkers);

void
two_body_update(double *R[], int N, int iat, int numWalkers);


void
two_body_grad_lapl(float *R[], int e1_first, int e1_last, int e2_first, int e2_last,
		   float spline_coefs[], int numCoefs, float rMax,  
		   float lattice[], float latticeInv[], 
		   float gradLapl[], int row_stride, int numWalkers);

void
two_body_grad_lapl(double *R[], int e1_first, int e1_last, int e2_first, int e2_last,
		   double spline_coefs[], int numCoefs, double rMax,  
		   double lattice[], double latticeInv[], 
		   double gradLapl[], int row_stride, int numWalkers);

///////////////////////
// One-Body routines //
///////////////////////

void
one_body_sum (float C[], float *R[], int e1_first, int e1_last, int e2_first, int e2_last,
	      float spline_coefs[], int numCoefs, float rMax,  
	      float lattice[], float latticeInv[], float sum[], int numWalkers);

void
one_body_sum (double C[], double *R[], int e1_first, int e1_last, int e2_first, int e2_last,
	      double spline_coefs[], int numCoefs, double rMax,  
	      double lattice[], double latticeInv[], double sum[], int numWalkers);

void
one_body_ratio (float C[], float *R[], int first, int last, int N,
		float Rnew[], int inew,
		float spline_coefs[], int numCoefs, float rMax,  
		float lattice[], float latticeInv[], float sum[], int numWalkers);

void
one_body_ratio (double C[], double *R[], int first, int last, int N,
		double Rnew[], int inew,
		double spline_coefs[], int numCoefs, double rMax,  
		double lattice[], double latticeInv[], double sum[], int numWalkers);

void
one_body_update(float *R[], int N, int iat, int numWalkers);

void
one_body_update(double *R[], int N, int iat, int numWalkers);


void
one_body_grad_lapl(float C[], float *R[], int e1_first, int e1_last, int e2_first, int e2_last,
		   float spline_coefs[], int numCoefs, float rMax,  
		   float lattice[], float latticeInv[], 
		   float gradLapl[], int row_stride, int numWalkers);

void
one_body_grad_lapl(double C[], double *R[], int e1_first, int e1_last, int e2_first, int e2_last,
		   double spline_coefs[], int numCoefs, double rMax,  
		   double lattice[], double latticeInv[], 
		   double gradLapl[], int row_stride, int numWalkers);





#endif
