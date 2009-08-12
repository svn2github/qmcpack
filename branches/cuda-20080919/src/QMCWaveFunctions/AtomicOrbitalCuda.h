#ifndef ATOMIC_ORBITAL_CUDA_H
#define ATOMIC_ORBITAL_CUDA_H

typedef enum { BSPLINE_3D_JOB, ATOMIC_POLY_JOB, ATOMIC_SPLINE_JOB } HybridJobType;

struct HybridDataFloat
{
  // Integer specifying which image of the ion this electron is near
  float img[3];
  // Minimum image distance to the ion;
  float dist;
  // The index the ion this electron is near
  int ion;
  // lMax for this ion
  int lMax;
};

struct AtomicOrbitalCudaFloat
{
  int lMax;
  float *SplineCoefs, *PolyCoefs;
};


void
MakeHybridJobList (float* elec_list, int num_elecs, float* ion_list, 
		   float* poly_radii, float* spline_radii,
		   int num_ions, float *L, float *Linv,
		   HybridJobType *job_list, float *rhat_list);

void CalcYlmRealCuda (float *rhats, float **Ylm_ptr, float **dYlm_dtheta_ptr, float **dYlm_dphi_ptr, 
		      int lMax, int N);

void CalcYlmComplexCuda (float *rhats, float **Ylm_ptr, float **dYlm_dtheta_ptr, float **dYlm_dphi_ptr, 
			 int lMax, int N);

void CalcYlmRealCuda (float *rhats, float **Ylm_ptr, int lMax, int N);

void CalcYlmComplexCuda (float *rhats, float **Ylm_ptr, int lMax, int N);

#endif
