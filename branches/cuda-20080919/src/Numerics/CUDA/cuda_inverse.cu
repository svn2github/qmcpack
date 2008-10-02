#include <stdio.h>

template<typename T, int BS>
__global__ void
block_inverse (float A[], int N, int stride)
{
  __shared__ unsigned int ipiv[BS];
  __shared__ unsigned int kb;
  __shared__ T maxval[BS], mask[BS], pivotInv;
  __shared__ T Arowk[BS], Acolk[BS];
  ipiv[threadIdx.x] = threadIdx.x;
  mask[threadIdx.x] = 1.0f;
  __syncthreads();

  unsigned int tid = threadIdx.x;

  for (int k=0; k<N; k++) {
    // First, find locate of maximum of kth column, excluding the
    // first k rows through the mask.
    maxval[tid] = mask[tid] * fabsf(A[tid*stride + k]);
    __syncthreads();
    if (threadIdx.x < 16) maxval[threadIdx.x] = 
      max(maxval[threadIdx.x], maxval[threadIdx.x+16]);
    if (threadIdx.x < 8 ) maxval[threadIdx.x] = 
      max(maxval[threadIdx.x], maxval[threadIdx.x+8]);
    if (threadIdx.x < 4 ) maxval[threadIdx.x] = 
      max(maxval[threadIdx.x], maxval[threadIdx.x+4]);
    if (threadIdx.x < 2 ) maxval[threadIdx.x] = 
      max(maxval[threadIdx.x], maxval[threadIdx.x+2]);
    if (threadIdx.x < 1 ) maxval[threadIdx.x] = 
      max(maxval[threadIdx.x], maxval[threadIdx.x+1]);
    __syncthreads();
    if ((mask[tid] * fabsf(A[tid*stride + k])) > 0.999* maxval[0]) {
      kb = tid;
      pivotInv = 1.0f/A[tid*stride + k];
    }
    __syncthreads();
    // Now kb holds pivot row and pivot the value
    
    // Swap rows
    T tmp = A[k*stride+tid];
    A[k*stride +tid] = A[kb*stride+tid];
    A[kb*stride+tid] = tmp;
    
    // Swap pivot
    if (tid == 0) {
      int itmp = ipiv[kb];
      ipiv[kb] = ipiv[k];
      ipiv[k]  = itmp;
    }
    __syncthreads();

    // Col k update
    if (tid != k)
      A[stride*tid+k] = -pivotInv*A[stride*tid+k];
    else
      A[stride*k+k] = 0.0f;
    __syncthreads();

    // Rank-1 update
    Arowk[tid] = A[stride*k   + tid];
    Acolk[tid] = A[stride*tid +   k];
    __syncthreads();
    for (int i=0; i<N; i++) 
      A[i*stride+tid] += Arowk[tid]*Acolk[i];
    __syncthreads();

    // Row k update
    if (tid != k) 
      A[k*stride+tid] *= pivotInv;
    else {
      A[k*stride+k] = pivotInv;
      mask[k] = 0.0;
    }
    __syncthreads();
  }
  // Finally, do backward pivoting
  for (int i=0; i<N; i++) {
    Arowk[tid] = A[i*stride+tid];
    __syncthreads();
    A[i*stride+ipiv[tid]] = Arowk[tid];
  }
}

#include <stdio.h>

main()
{
  int N=32;
  float A[N*N], Acopy[N*N];
  float *A_d;
  
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      A[N*i+j] = Acopy[N*i+j] = (float) drand48();

  cudaMalloc ((void**)&A_d, N*N*sizeof(float));
  cudaMemcpy (A_d, A, N*N*sizeof(float),
	      cudaMemcpyHostToDevice);

  dim3 dimBlock(N);
  dim3 dimGrid(1);
  block_inverse<float,32><<<dimGrid,dimBlock>>> (A_d, N, N);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in block_inverse:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

  cudaMemcpy (A, A_d, N*N*sizeof(float),
	      cudaMemcpyDeviceToHost);

  float nrm = 0.0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) {
      float val = 0.0;
      for (int k=0; k<N; k++)
	val += A[i*N+k] * Acopy[k*N+j];
      float diff = (i==j) ? 1.0-val : val;
      nrm += diff*diff;
    }
  fprintf (stderr, "Error = %1.6e\n", nrm);
}
