#ifndef CUDA_DETERMINANT_UPDATE_H
#define CUDA_DETERMINANT_UPDATE_H

struct updateJob
{
  void *A, *Ainv, *newRow, *AinvDelta, *AinvColk, *gradLapl, *newGradLapl, *dummy;
};



void
update_inverse_cuda(float *A_g[], float *Ainv_g[], float *u_g[], 
		    float *Ainv_delta_g[], float *Ainv_colk_g[], 
		    int N, int rowstride, int iat, int numWalkers);

void
update_inverse_cuda(updateJob jobList[], double dummy,
		    int N, int rowstride, int iat, int numWalkers);
void
update_inverse_cuda(updateJob jobList[], float dummy,
		    int N, int rowstride, int iat, int numWalkers);

void
update_inverse_cuda(double *A_g[], double *Ainv_g[], double *u_g[], 
		    double *Ainv_delta_g[], double *Ainv_colk_g[], 
		    int N, int rowstride, int iat, int numWalkers);


void
determinant_ratios_cuda (float *Ainv_list[], float *new_row_list[],
			 float *ratios, int N, int row_stride, int iat,
			 int numWalkers);

void
determinant_ratios_cuda (double *Ainv_list[], double *new_row_list[],
			 double *ratios, int N, int row_stride, int iat,
			 int numWalkers);


void
calc_many_ratios (float *Ainv_list[], float *new_row_list[],
		  float* ratio_list[], int num_ratio_list[],
		  int N, int row_stride, int elec_list[],
		  int numWalkers);

void
calc_many_ratios (double *Ainv_list[], double *new_row_list[],
		  double* ratio_list[], int num_ratio_list[],
		  int N, int row_stride, int elec_list[],
		  int numWalkers);

void
scale_grad_lapl(float *grad_list[], float *hess_list[],
		float *grad_lapl_list[], float Linv[], int N,
		int num_walkers);

void
calc_grad_lapl (float *Ainv_list[], float *grad_lapl_list[],
		float *out_list[], int N, int row_stride, int num_mats);

void
calc_grad_lapl (double *Ainv_list[], double *grad_lapl_list[],
		double *out_list[], int N, int row_stride, int num_mats);


void
multi_copy (float *dest[], float *src[], int len, int num);

void
multi_copy (double *dest[], double *src[], int len, int num);


void
calc_many_ratios (float *Ainv_list[], float *new_row_list[],
		  float* ratio_list[], int num_ratio_list[],
		  int N, int row_stride, int elec_list[],
		  int numWalkers);

void
calc_many_ratios (double *Ainv_list[], double *new_row_list[],
		  double* ratio_list[], int num_ratio_list[],
		  int N, int row_stride, int elec_list[],
		  int numWalkers);

void
determinant_ratios_grad_lapl_cuda (float *Ainv_list[], float *new_row_list[],
				   float *grad_lapl_list[], float ratios_grad_lapl[], 
				   int N, int row_stride, int iat, int numWalkers);

void
determinant_ratios_grad_lapl_cuda (double *Ainv_list[], double *new_row_list[],
				   double *grad_lapl_list[], double ratios_grad_lapl[], 
				   int N, int row_stride, int iat, int numWalkers);


void
calc_gradient (float *Ainv_list[], float *grad_lapl_list[],
	       float grad[], int N, int row_stride, int elec,
	       int numWalkers);

void
calc_gradient (double *Ainv_list[], double *grad_lapl_list[],
	       double grad[], int N, int row_stride, int elec,
	       int numWalkers);


#endif
