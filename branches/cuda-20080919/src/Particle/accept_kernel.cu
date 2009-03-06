template<typename T, int BS>
__global__ void accept_kernel (T* Rlist[], T Rnew[], 
			       int toAccept[], int iat, int N)
{
  int tid = threadIdx.x;
  __shared__ T* myR[BS];
  __shared__ T Rnew_shared[BS*3];
  __shared__ int accept_shared[BS];
  
  int block = blockIdx.x;
  
  for (int i=0; i<3; i++) 
    if ((3*block+i)*BS + tid < 3*N)
      Rnew_shared[i*BS + tid] = Rnew[(3*block+i)*BS + tid];
  __syncthreads();
  
  if (block*BS + tid < N) {
    myR[tid] = Rlist[block*BS+tid] + 3*iat;
    accept_shared[tid] = toAccept[block*BS+tid];
  }
  __syncthreads();
  
  // if (block*BS + tid < N && accept_shared[tid]) {
  //   myR[tid][0] = Rnew_shared[3*tid+0];
  //   myR[tid][1] = Rnew_shared[3*tid+1];
  //   myR[tid][2] = Rnew_shared[3*tid+2];
  // }
  // return;

  for (int i=0; i<3; i++) {
    int index = i*BS + tid;
    int iw = index / 3;
    int dim = index % 3;
    if (iw+block*BS < N && accept_shared[iw])
      myR[iw%BS][dim] = Rnew_shared[index];
  }
}

#include <cstdio>

void
accept_move_GPU_cuda (float* Rlist[], float new_pos[], 
		      int toAccept[], int iat, int N)
{
  const int BS=32;
  
  int NB = N / BS + ((N % BS) ? 1 : 0);
  
  dim3 dimBlock(BS);
  dim3 dimGrid(NB);
  
  accept_kernel<float,BS><<<dimGrid,dimBlock>>>
    (Rlist, new_pos, toAccept, iat, N);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in accept_move_GPU_cuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

}


void
accept_move_GPU_cuda (double* Rlist[], double new_pos[], 
		      int toAccept[], int iat, int N)
{
  const int BS=32;
  
  int NB = N / BS + ((N % BS) ? 1 : 0);
  
  dim3 dimBlock(BS);
  dim3 dimGrid(NB);
  
  accept_kernel<double,BS><<<dimGrid,dimBlock>>>
    (Rlist, new_pos, toAccept, iat, N);
}



