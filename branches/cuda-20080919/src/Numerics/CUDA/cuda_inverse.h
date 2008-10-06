#ifndef CUDA_INVERSE_H
#define CUDA_INVERSE_H

void
cuda_inverse_many (float *Alist_d[], float *worklist_d[],
		   int N, int num_mats);

#endif
