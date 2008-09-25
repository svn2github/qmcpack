#ifndef CUDA_DETERMINANT_UPDATE_H
#define CUDA_DETERMINANT_UPDATE_H

void
update_inverse_cuda(float *A_g[], float *Ainv_g[], float *u_g[], 
		    float *Ainv_delta_g[], float *Ainv_colk_g[], 
		    int N, int rowstride, int iat, int numWalkers);

void
update_inverse_cuda(double *A_g[], double *Ainv_g[], double *u_g[], 
		    double *Ainv_delta_g[], double *Ainv_colk_g[], 
		    int N, int rowstride, int iat, int numWalkers);


void
determinant_ratios_cuda (float *Ainv_list[], float *new_row_list[],
			 float *ratios, int N, int row_stride, int iat,
			 int numWalkers);

#endif
