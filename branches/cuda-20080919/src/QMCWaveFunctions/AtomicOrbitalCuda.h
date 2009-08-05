#ifndef ATOMIC_ORBITAL_CUDA_H
#define ATOMIC_ORBITAL_CUDA_H

template<typename T>
void CalcYlmRealCuda (T *rhats, T **Ylm_ptr, T **dYlm_dtheta_ptr, T **dYlm_dphi_ptr, 
		      int lMax, int N);

template<typename T>
void CalcYlmComplexCuda (T *rhats, T **Ylm_ptr, T **dYlm_dtheta_ptr, T **dYlm_dphi_ptr, 
			 int lMax, int N);

template<typename T>
void CalcYlmRealCuda (T *rhats, T **Ylm_ptr, int lMax, int N);

template<typename T>
void CalcYlmComplexCuda (T *rhats, T **Ylm_ptr, int lMax, int N);

#endif
