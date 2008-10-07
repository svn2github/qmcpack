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
  mask[threadIdx.x] = (T)1.0;
  __syncthreads();

  unsigned int tid = threadIdx.x;

  for (int k=0; k<N; k++) {
    // First, find locate of maximum of kth column, excluding the
    // first k rows through the mask.
    maxval[tid] = mask[tid] * fabsf(A[tid*stride + k]);
    __syncthreads();
    for (int bs = BS>>1; bs>0; bs=bs>>1) {
      if (tid < bs) 
	maxval[tid] =  max(maxval[tid], maxval[tid+bs]);
      __syncthreads();
    }
    if ((mask[tid] * fabsf(A[tid*stride + k])) > 0.999* maxval[0]) {
      kb = tid;
      pivotInv = (T)1.0/A[tid*stride + k];
    }
    __syncthreads();
    // HACK HACK HACK
    //kb = k;
    //pivotInv = 1.0f/A[k*stride + k];
    //__syncthreads();

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
      A[stride*k+k] = (T)0.0;
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
__device__ T
block_inverse1 (T A[BS][BS+1])
{
  __shared__ unsigned int ipiv[BS];
  __shared__ unsigned int kb;
  __shared__ T maxval[BS], mask[BS], pivotInv;
  __shared__ T Arowk[BS], Acolk[BS];
  ipiv[threadIdx.x] = threadIdx.x;
  mask[threadIdx.x] = (T)1.0;
  __syncthreads();

  unsigned int tid = threadIdx.x;

  __shared__ T det;
  if (tid == 0)
    det = (T)1.0;


  for (int k=0; k<BS; k++) {
    // First, find locate of maximum of kth column, excluding the
    // first k rows through the mask.
    maxval[tid] = mask[tid] * fabsf(A[tid][k]);
    __syncthreads();

    for (int bs = BS>>1; bs>0; bs=bs>>1) {
      if (tid < bs) 
	maxval[tid] =  max(maxval[tid], maxval[tid+bs]);
      __syncthreads();
    }

    if ((mask[tid] * fabsf(A[tid][k])) == maxval[0]) {
      kb = tid;
      pivotInv = (T)1.0/A[tid][k];
      if (kb == k)	det *= A[tid][k];
      else              det *= -A[tid][k];
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
      A[k][k] = (T)0.0;
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
  return det;
}


template<typename T, int BS>
__device__ void block_mul (T A[BS][BS+1],
			   T B[BS][BS+1],
			   T C[BS][BS+1])
{
  int tid = threadIdx.x;
  for (int row=0; row<BS; row++)
    C[row][tid] = (T)0.0;
  __syncthreads();

  for (int k=0; k<BS; k++)
    for (int i=0; i<BS; i++)
      C[i][tid] += A[i][k]*B[k][tid];
}  


template<typename T, int BS>
__device__ void block_mul_add (T A[BS][BS+1],
			       T B[BS][BS+1],
			       T *C, int Cstride)
{
  int tid = threadIdx.x;
  __shared__ T Crow[BS];

  for (int i=0; i<BS; i++) {
    Crow[tid] = C[i*Cstride + tid];
    for (int k=0; k<BS; k++) 
      Crow[tid] += A[i][k]*B[k][tid];
    C[i*Cstride + tid] = Crow[tid];
  }
}  

template<typename T, int BS>
__device__ void block_mul_set (T A[BS][BS+1],
			       T B[BS][BS+1],
			       T *C, int Cstride)
{
  int tid = threadIdx.x;
  __shared__ T Crow[BS];


  for (int i=0; i<BS; i++) {
    Crow[tid] = (T)0.0;
    for (int k=0; k<BS; k++) 
      Crow[tid] += A[i][k]*B[k][tid];
    C[i*Cstride + tid] = Crow[tid];
  }
}  



template<typename T, int BS>
__global__ void
inverse (T A[], T work[], int N, int stride)
{
  T *Atmp = work;
  T *pivot_tmp = work+N*stride;

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
    	  A[(row+j)*stride + col+tid] = (T)0.0;
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



template<typename T, int BS>
__global__ void
inverse_many (T *A_list[], T *work_list[], int N, int stride)
{
  int tid = threadIdx.x;
  __shared__ T *A, *work;
  if (tid == 0) {
    A    = A_list[blockIdx.x];
    work = work_list[blockIdx.x];
  }
  __syncthreads();

  T *Atmp = work;
  T *pivot_tmp = work+N*stride;

  __shared__ T pivot[BS][BS+1], in[BS][BS+1];
  int NB = N/BS;
  if (N%BS) NB++;


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
    	  A[(row+j)*stride + col+tid] = (T)0.0;
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
    	for (int i=0; i<BS; i++) 
    	  pivot[i][tid] = A[(kb*BS+i)*stride + jb*BS + tid];
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

#define MAX_BLOCKS 32

template<typename T, int BS>
__global__ void
inverse_many_pivot (T *A_list[], T *work_list[], int N, int stride)
{
  int tid = threadIdx.x;
  __shared__ T *A, *work;
  T maxdet, blockdet, det;
  __shared__ int ipiv[MAX_BLOCKS];

  if (tid == 0) {
    A    = A_list[blockIdx.x];
    work = work_list[blockIdx.x];
    det = (T)1.0;
  }
  ipiv[tid] = tid;
  __syncthreads();

  T *Atmp = work;
  T *pivot_tmp = work+N*stride;

  __shared__ T pivot[BS][BS+1], in[BS][BS+1];


  int NB = N/BS;
  if (N%BS) NB++;

  for (int kb=0; kb<NB; kb++) {
    int imax = kb;
    maxdet = (T)0.0;
    // Find pivot block
    for (int block=kb; block<NB; block++) {
      // load pivot block
      int row = block*BS;
      for (int j=0; j<BS; j++)
    	if (row+tid < N)
    	  in[j][tid] = A[(row+j)*stride + kb*BS + tid];
      __syncthreads();
      // invert pivot
      blockdet = block_inverse1<T,BS> (in);
      __syncthreads();
      if (fabsf(blockdet) > fabs(maxdet)) {
      //if (block == kb) {
    	imax = block;
    	maxdet = blockdet;
    	for (int j=0; j<BS; j++)
    	  pivot[j][tid] = in[j][tid];
      }
    }
	
    // Now, swap row blocks
    for (int j=0; j<BS; j++) {
      int rowa = kb   * BS + j;
      int rowb = imax * BS + j;
      for (int n=0; n<NB; n++) {
    	int col = n*BS + tid;
    	T tmp = A[rowa*stride + col];
    	__syncthreads();
    	A[rowa*stride + col] = A[rowb*stride + col];
    	__syncthreads();
    	A[rowb*stride + col] = tmp;
      }
    }
    if (tid == 0) {
      int tmp = ipiv[kb];
      ipiv[kb] = ipiv[imax];
      ipiv[imax] = tmp;
      if (imax == kb)
    	det *= blockdet;
      else
    	det *= -blockdet;
    }

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
    	  A[(row+j)*stride + col+tid] = (T)0.0;
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
    	for (int i=0; i<BS; i++) 
    	  pivot[i][tid] = A[(kb*BS+i)*stride + jb*BS + tid];
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

  // Note:  the following assumes that N <= BS*BS
  // Finally, reverse pivoting
  for (int row=0; row<N; row++) {
    for (int block=0; block < NB; block++) 
      in[block][tid] = A[row*stride + BS*block + tid];
    __syncthreads();
    for (int block=0; block<NB; block++)
      A[row*stride + BS*ipiv[block] + tid] = in[block][tid];
  }
}

#define CONVERT_BS 256


template<typename Tdest, typename Tsrc>
__global__ void
convert (Tdest *dest_list[], Tsrc *src_list[], int len)
{
  __shared__ Tsrc *mysrc;
  __shared__ Tdest *mydest;
  if (threadIdx.x ==0) {
    mysrc = src_list[blockIdx.y];
    mydest = dest_list[blockIdx.y];
  }
  __syncthreads();
  int i = blockIdx.x * CONVERT_BS + threadIdx.x;
  if (i < len)
    mydest[i] = (Tdest)mysrc[i];

}





#define INVERSE_BS 16

void
cuda_inverse_many (float *Alist_d[], float *worklist_d[],
		   int N, int num_mats)
{
  dim3 dimBlock(INVERSE_BS);
  dim3 dimGrid(num_mats);
  
  inverse_many_pivot<float,INVERSE_BS><<<dimGrid,dimBlock>>> 
    (Alist_d, worklist_d, N, N);
}


size_t
cuda_inverse_many_worksize(int N)
{
  return (N*N + INVERSE_BS*INVERSE_BS);
}


size_t
cuda_inverse_many_double_worksize(int N)
{
  return 2*(2*N*N + INVERSE_BS*INVERSE_BS);
}

void
cuda_inverse_many_double (float *Alist_d[], float *worklist_d[],
			  int N, int num_mats)
{
  dim3 dimBlockConvert (CONVERT_BS);
  dim3 dimGridConvert(N*N/CONVERT_BS, num_mats);
  if (N*N % CONVERT_BS)
    dimGridConvert.x++;
  convert<<<dimGridConvert,dimBlockConvert>>> 
    ((double**)worklist_d, Alist_d, N*N);

  float *Alist_new[num_mats], *Alist_h[num_mats];
  float *worklist_h[num_mats];
  double *worklist_double_h[num_mats];

  cudaMemcpy (worklist_h, worklist_d, num_mats*sizeof(float*),
	      cudaMemcpyDeviceToHost);
  cudaMemcpy (Alist_h, Alist_d, num_mats*sizeof(float*),
	      cudaMemcpyDeviceToHost);

  for (int i=0; i<num_mats; i++) {
    Alist_new[i] = worklist_h[i];
    worklist_double_h[i] = (double*)(worklist_h[i]) +N*N;
  }
  cudaMemcpy (worklist_d, worklist_double_h, num_mats*sizeof(double*),
	      cudaMemcpyHostToDevice);
  cudaMemcpy (Alist_d, Alist_new, num_mats*sizeof(double*),
	      cudaMemcpyHostToDevice);


  dim3 dimBlock(INVERSE_BS);
  dim3 dimGrid(num_mats);
  
  inverse_many_pivot<double,INVERSE_BS><<<dimGrid,dimBlock>>> 
    ((double**)Alist_d, (double**)worklist_d, N, N);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in inverse_many_pivot<double,%d>:\n  %s\n",
	     INVERSE_BS, cudaGetErrorString(err));
    abort();
  }

  cudaMemcpy (Alist_d, Alist_h, num_mats*sizeof(float*),
	      cudaMemcpyHostToDevice);

  cudaMemcpy (worklist_d, worklist_h, num_mats*sizeof(float*),
	      cudaMemcpyHostToDevice);

  convert<<<dimGridConvert,dimBlockConvert>>> 
    (Alist_d, (double**) worklist_d, N*N);

}




void
cuda_inverse_many (double *Alist_d[], double *worklist_d[],
		   int N, int num_mats)
{
  dim3 dimBlock(INVERSE_BS);
  dim3 dimGrid(num_mats);
  
  inverse_many_pivot<double,INVERSE_BS><<<dimGrid,dimBlock>>> 
    (Alist_d, worklist_d, N, N);
}



//////////////////////////////////////////////////////
//                  Test routines                   //
//////////////////////////////////////////////////////



#ifdef CUDA_TEST_MAIN

void 
test_inverse()
{
  int N = 32;
  dim3 dimBlock(INVERSE_BS);
  dim3 dimGrid(1);

  float *A_d, *work_d;
  int lwork = N*N + INVERSE_BS * INVERSE_BS;


  cudaMalloc((void**)&A_d, N*N*sizeof(float));
  cudaMalloc((void**)&work_d, lwork*sizeof(float));
  
  float A[N*N], Ainv[N*N];
  for (int i=0; i<N*N; i++)
    A[i] = drand48();
  cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
  
  inverse<float,INVERSE_BS><<<dimGrid,dimBlock>>> (A_d, work_d, N, N);
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



void 
test_inverse_many()
{
  int numMats = 100;

  int N = 128;

  int lwork = N*N + INVERSE_BS * INVERSE_BS;
  fprintf (stderr, "lwork = %d\n", lwork);

  float **Alist, **worklist;
  float **Alist_d, **worklist_d;

  Alist    = (float**)malloc(numMats*sizeof(float*));
  worklist = (float**)malloc(numMats*sizeof(float*));
  cudaMalloc((void**)&Alist_d,    numMats*sizeof(float*));
  cudaMalloc((void**)&worklist_d, numMats*sizeof(float*));

  float A[N*N];
  for (int i=0; i<N*N; i++)
    A[i] = drand48();

  for (int mat=0; mat<numMats; mat++) {
    cudaMalloc ((void**)&(Alist[mat]),    N*N*sizeof(float));
    cudaMalloc ((void**)&(worklist[mat]), lwork*sizeof(float));
    cudaMemcpy(Alist[mat], A, N*N*sizeof(float), cudaMemcpyHostToDevice);
  }

  cudaMemcpy(Alist_d   ,    Alist, numMats*sizeof(float*), 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(worklist_d, worklist, numMats*sizeof(float*), 
	     cudaMemcpyHostToDevice);
  
  dim3 dimBlock(INVERSE_BS);
  dim3 dimGrid(numMats);

  clock_t start = clock();
  for (int i=0; i<1; i++) {
    inverse_many_pivot<float,INVERSE_BS><<<dimGrid,dimBlock>>> 
      (Alist_d, worklist_d, N, N);
    // inverse_many<float,INVERSE_BS><<<dimGrid,dimBlock>>> 
    //   (Alist_d, worklist_d, N, N);
    cudaThreadSynchronize();
  }
  clock_t end = clock();
  
  double time = (double)(end-start)/(double)CLOCKS_PER_SEC
    / (double)numMats;
  double rate = 1.0/time;
  fprintf (stderr, "Rate is %1.3f matrix inversions per second.\n",
	   rate);


  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in block_inverse:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

  // Copy Ainv back to host memory
  float Ainv[N*N];
  cudaMemcpy(Ainv, Alist[10], N*N*sizeof(float), cudaMemcpyDeviceToHost);

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
  test_inverse_many();

  // int N=32;
  // float A[N*N], Acopy[N*N];
  // float *A_d;
  
  // for (int i=0; i<N; i++)
  //   for (int j=0; j<N; j++)
  //     A[N*i+j] = Acopy[N*i+j] = (float) drand48();

  // cudaMalloc ((void**)&A_d, N*N*sizeof(float));
  // cudaMemcpy (A_d, A, N*N*sizeof(float),
  // 	      cudaMemcpyHostToDevice);

  // dim3 dimBlock(N);
  // dim3 dimGrid(1);
  // block_inverse<float,32><<<dimGrid,dimBlock>>> (A_d, N, N);

  // cudaThreadSynchronize();
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   fprintf (stderr, "CUDA error in block_inverse:\n  %s\n",
  // 	     cudaGetErrorString(err));
  //   abort();
  // }

  // cudaMemcpy (A, A_d, N*N*sizeof(float),
  // 	      cudaMemcpyDeviceToHost);

  // float nrm = 0.0;
  // for (int i=0; i<N; i++)
  //   for (int j=0; j<N; j++) {
  //     float val = 0.0;
  //     for (int k=0; k<N; k++)
  // 	val += A[i*N+k] * Acopy[k*N+j];
  //     float diff = (i==j) ? 1.0-val : val;
  //     nrm += diff*diff;
  //   }
  // fprintf (stderr, "Error = %1.6e\n", sqrt(nrm/(double)(N*N)));
}
#endif
