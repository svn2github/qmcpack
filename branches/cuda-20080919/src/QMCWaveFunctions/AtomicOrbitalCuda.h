#ifndef ATOMIC_ORBITAL_CUDA_H
#define ATOMIC_ORBITAL_CUDA_H

typedef enum { BSPLINE_3D_JOB, ATOMIC_POLY_JOB, ATOMIC_SPLINE_JOB } AtomicJobType;

void CalcYlmRealCuda (float *rhats, float **Ylm_ptr, float **dYlm_dtheta_ptr, float **dYlm_dphi_ptr, 
		      int lMax, int N);

void CalcYlmComplexCuda (float *rhats, float **Ylm_ptr, float **dYlm_dtheta_ptr, float **dYlm_dphi_ptr, 
			 int lMax, int N);

void CalcYlmRealCuda (float *rhats, float **Ylm_ptr, int lMax, int N);

void CalcYlmComplexCuda (float *rhats, float **Ylm_ptr, int lMax, int N);

#endif
