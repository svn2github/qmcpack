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


template<typename T, int BS>
__device__ void
block_inverse1 (T A[BS][BS+1])
{
  __shared__ unsigned int ipiv[BS];
  __shared__ unsigned int kb;
  __shared__ T maxval[BS], mask[BS], pivotInv;
  __shared__ T Arowk[BS], Acolk[BS];
  ipiv[threadIdx.x] = threadIdx.x;
  mask[threadIdx.x] = 1.0f;
  __syncthreads();

  unsigned int tid = threadIdx.x;

  for (int k=0; k<BS; k++) {
    // First, find locate of maximum of kth column, excluding the
    // first k rows through the mask.
    maxval[tid] = mask[tid] * fabsf(A[tid][k]);
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
    if ((mask[tid] * fabsf(A[tid][k])) > 0.999* maxval[0]) {
      kb = tid;
      pivotInv = 1.0f/A[tid][k];
    }
    __syncthreads();
    // Now kb holds pivot row and pivot the value
    
    // Swap rows
    T tmp = A[k][tid];
    A[k][tid] = A[kb][tid];
    A[kb][tid] = tmp;
    
    // Swap pivot
    if (tid == 0) {
      int itmp = ipiv[kb];
      ipiv[kb] = ipiv[k];
      ipiv[k]  = itmp;
    }
    __syncthreads();

    // Col k update
    if (tid != k)
      A[tid][k] = -pivotInv*A[tid][k];
    else
      A[k][k] = 0.0f;
    __syncthreads();

    // Rank-1 update
    Arowk[tid] = A[k][tid];
    Acolk[tid] = A[tid][k];
    __syncthreads();
    for (int i=0; i<BS; i++) 
      A[i][tid] += Arowk[tid]*Acolk[i];
    __syncthreads();

    // Row k update
    if (tid != k) 
      A[k][tid] *= pivotInv;
    else {
      A[k][k] = pivotInv;
      mask[k] = 0.0;
    }
    __syncthreads();
  }
  // Finally, do backward pivoting
  for (int i=0; i<BS; i++) {
    Arowk[tid] = A[i][tid];
    __syncthreads();
    A[i][ipiv[tid]] = Arowk[tid];
  }
}


template<typename T, int BS>
__device__ void block_mul (float A[BS][BS+1],
			   float B[BS][BS+1],
			   float C[BS][BS+1])
{
  int tid = threadIdx.x;
  for (int row=0; row<BS; row++)
    C[row][tid] = 0.0f;
  __syncthreads();

  for (int k=0; k<BS; k++)
    for (int i=0; i<BS; i++)
      C[i][tid] += A[i][k]*B[k][tid];
}  


template<typename T, int BS>
__device__ void block_mul_add (float A[BS][BS+1],
			       float B[BS][BS+1],
			       float *C, int Cstride)
{
  int tid = threadIdx.x;
  __shared__ T Crow[BS];

  for (int i=0; i<BS; i++) {
    Crow[tid] = C[i*Cstride + tid];
    for (int k=0; k<BS; k++) 
      Crow[tid] += A[i][k]*B[k][tid];
    C[i*Cstride + tid] = Crow[tid];
  }
  // for (int i=0; i<N; i++)
  //   for (int k=0; k<N; k++)
  //     C[i,tid] += A(i,k)*B(k,tid);

}  

template<typename T, int BS>
__device__ void block_mul_set (float A[BS][BS+1],
			       float B[BS][BS+1],
			       float *C, int Cstride)
{
  int tid = threadIdx.x;
  __shared__ T Crow[BS];


  for (int i=0; i<BS; i++) {
    Crow[tid] = 0.0f;
    for (int k=0; k<BS; k++) 
      Crow[tid] += A[i][k]*B[k][tid];
    C[i*Cstride + tid] = Crow[tid];
  }


  // for (int k=0; k<BS; k++) {
  //   Crow[tid] = 0.0f;
  //   for (int i=0; i<BS; i++)
  //     Crow[tid] += A[i][k]*B[k][tid];
  //   C[k*Cstride + tid] = Crow[tid];
  // }
}  





template<typename T, int BS>
__global__ void
inverse (T A[], T Atmp[], T pivot_tmp[], int N, int stride)
{
  __shared__ T pivot[BS][BS+1], in[BS][BS+1];
  int NB = N/BS;
  if (N%BS) NB++;
  int tid = threadIdx.x;


  for (int kb=0; kb<NB; kb++) {
    // load pivot block
    int row = kb*BS;
    for (int j=0; j<BS; j++)
      if (row+tid < N)
	pivot[j][tid] = A[(row+j)*stride + row+tid];
    
    // invert pivot
    block_inverse1<T,BS> (pivot);

    // Column scaling
    int col = kb*BS;
    for (int jb=0; jb < NB; jb++) {
      int row = jb*BS;
      if (kb != jb) {
    	for (int j=0; j<BS; j++)
    	  in[j][tid] = -A[(row+j)*stride + col + tid];
    	block_mul_set<T,BS>(in, pivot, A+row*stride+col, stride);
      }
      else {
    	for (int j=0; j<BS; j++)
    	  A[(row+j)*stride + col+tid] = 0.0f;
      }
    }	

    // Save pivot to global memory here!
    // We use it for temporary space in the rank-1 update
    for (int j=0; j<BS; j++)
      pivot_tmp[j*BS+tid] = pivot[j][tid];


    // Copy Ato Atmp
    for (int ib=0; ib<NB; ib++)
      for (int row=0; row<N; row++)
    	Atmp[row*stride+ib*BS+tid] =  A[row*stride+ib*BS+tid];
    
    // Rank-1 update
    for (int ib=0; ib < NB; ib++) {
      for (int i=0; i<BS; i++)
    	in[i][tid] = A[(ib*BS+i)*stride + kb*BS + tid];
      for (int jb=0; jb<NB; jb++) {
    	for (int i=0; i<BS; i++) {
    	  pivot[i][tid] = A[(kb*BS+i)*stride + jb*BS + tid];
    	  // Atmp[(ib*BS+i)*stride + (jb*BS+tid)] = 
    	  //   A[(ib*BS+i)*stride + (jb*BS+tid)];
    	}
    	block_mul_add<T,BS>(in, pivot,  Atmp+(ib*BS)*stride + jb*BS,
    			    stride);
      }
    }
    // Copy Atmp back to A
    for (int ib=0; ib<NB; ib++)
      for (int row=0; row<N; row++)
    	A[row*stride+ib*BS+tid] =  Atmp[row*stride+ib*BS+tid];

    // Restore pivot from global memory here!
    for (int j=0; j<BS; j++)
      pivot[j][tid] = pivot_tmp[j*BS+tid];

    // Row-scaling
    for (int ib=0; ib<NB; ib++) {
      int row = kb*BS;
      int col = ib*BS;
      if (kb != ib) {
    	for (int j=0; j<BS; j++)
    	  in[j][tid] = A[(row+j)*stride + col+tid];
    	block_mul_set<T,BS>(pivot, in, A+row*stride+col, stride);
      }
      else {
    	for (int j=0; j<BS; j++) 
    	  A[(row+j)*stride + col+tid] = pivot[j][tid];
      }
    }	


  }
}


#define INVERSE_BS 32

void 
test_inverse()
{
  int N = 128;
  dim3 dimBlock(32);
  dim3 dimGrid(1);

  float *A_d, *Atmp_d, *work_d;
  
  cudaMalloc((void**)&A_d, N*N*sizeof(float));
  cudaMalloc((void**)&Atmp_d, N*N*sizeof(float));
  cudaMalloc((void**)&work_d, INVERSE_BS*INVERSE_BS*sizeof(float));
  
  float A[N*N], Ainv[N*N];
  for (int i=0; i<N*N; i++)
    A[i] = 1.0-2.0*drand48();
  cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
  
  inverse<float,INVERSE_BS><<<dimGrid,dimBlock>>> (A_d, Atmp_d, work_d, N, N);
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in block_inverse:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

  // Copy Ainv back to host memory
  
  cudaMemcpy(Ainv, A_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  float error = 0.0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) {
      float val = 0.0;
      for (int k=0; k<N; k++)
	val += Ainv[i*N+k]*A[k*N+j];
      float diff = (i==j) ? (1.0f-val) : val;
      error += diff*diff;
    }
  fprintf (stderr, "error = %1.8e\n", sqrt(error/(double)(N*N)));

}



#include <stdio.h>

main()
{
  test_inverse();

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
