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
  int lMax;
  int PAD[2];
};

template<typename T>
class AtomicOrbitalCuda
{
public:
  int lMax, spline_stride, lm_stride;
  T spline_dr_inv;
  T *spline_coefs, *poly_coefs;
};

void init_atomic_cuda();

void
MakeHybridJobList (float* elec_list, int num_elecs, float* ion_list, 
		   float* poly_radii, float* spline_radii,
		   int num_ions, float *L, float *Linv,
		   HybridJobType *job_list, float *rhat_list,
		   HybridDataFloat *data_list);

void
evaluateHybridSplineReal (HybridJobType *job_types, 
			  float **Ylm_real, int Ylm_stride,
			  float** SplineCoefs, float gridInv, int grid_stride,
			  HybridDataFloat *data,
			  float **vals, int N, int numWalkers, int lMax);

template<typename T>
void CalcYlmRealCuda (T *rhats,  HybridJobType *job_type,
		      T **Ylm_ptr, T **dYlm_dtheta_ptr, T **dYlm_dphi_ptr, 
		      int lMax, int N);

template<typename T>
void CalcYlmComplexCuda (T *rhats,  HybridJobType *job_type,
			 T **Ylm_ptr, T **dYlm_dtheta_ptr, T **dYlm_dphi_ptr, 
			 int lMax, int N);

template<typename T>
void CalcYlmRealCuda (T *rhats,  HybridJobType *job_type,
		      T **Ylm_ptr, int lMax, int N);

template<typename T>
void CalcYlmComplexCuda (T *rhats,  HybridJobType *job_type,
			 T **Ylm_ptr, int lMax, int N);

#endif
