#ifndef CUDA_COULOMB_H
#define CUDA_COULOMB_H

class TextureSpline
{
public:
  float rMin, rMax;
  int NumPoints;
  cudaArray *myArray;
  void set (float data[], int numPoints, float rmin, float rmax);
  void set (double data[], int numPoints, double rmin, double rmax);
};

void
CoulombAA_SR_Sum(float *R[], int N, float rMax, int Ntex,
		 float lattice[], float latticeInv[], float sum[],
		 int numWalkers);

void
eval_rhok_cuda(float *R[], int numr, float kpoints[], 
	       int numk, float* rhok[], int numWalkers);

void
eval_vk_sum_cuda (float *rhok[], float vk[], int numk, float sum[],
		  int numWalkers);

void
eval_rhok_cuda(float *R[], int first, int last, float kpoints[], 
	       int numk, float* rhok[], int numWalkers);

void
eval_vk_sum_cuda (float *rhok1[], float *rhok2[], 
		  float vk[], int numk, float sum[],
		  int numWalkers);

#endif
