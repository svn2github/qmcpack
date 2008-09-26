#define DET_BLOCK_SIZE 64

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>


// The first kernel just computes AinvT * u and also stores the kth
// col of Ainv in global memory
template<typename T>
__global__ void
update_inverse_cuda1 (T *A_g[], T *Ainv_g[], T *u_g[], 
		      T *Ainv_delta_g[], T *Ainv_colk_g[], 
		      int N, int rowstride, int k)
{
  __shared__ T *A, *Ainv, *u, *Ainv_delta, *Ainv_colk;
  if (threadIdx.x==0) {
    A           = A_g[blockIdx.y];
    Ainv        = Ainv_g[blockIdx.y];
    u           = u_g[blockIdx.y];
    Ainv_delta  = Ainv_delta_g[blockIdx.y];
    Ainv_colk   = Ainv_colk_g[blockIdx.y];
  }

  __syncthreads();

  // Store the product Ainv * u in shared memory
  __shared__ T Ainv_delta_shared[DET_BLOCK_SIZE], 
    Ainv_colk_shared[DET_BLOCK_SIZE], u_shared[DET_BLOCK_SIZE],
    uold_shared[DET_BLOCK_SIZE];
  Ainv_delta_shared[threadIdx.x] = 0.0;
  int col = blockIdx.x*DET_BLOCK_SIZE + threadIdx.x;
  int numblocks = N / DET_BLOCK_SIZE;

  // If the column I need to pull from Ainv is in this thread block
  // domain, do the following
  if (blockIdx.x*DET_BLOCK_SIZE <= k && k < (blockIdx.x+1)*DET_BLOCK_SIZE) {
    for (int block=0; block<numblocks; block++) {
      u_shared[threadIdx.x] = u[block*DET_BLOCK_SIZE+threadIdx.x];
      uold_shared[threadIdx.x] = 
      	A[k*rowstride + block*DET_BLOCK_SIZE+threadIdx.x];
      // Write new row into A matrix
      A[k*rowstride + block*DET_BLOCK_SIZE+threadIdx.x] = u_shared[threadIdx.x];
      __syncthreads();
      for (int i=0; i<DET_BLOCK_SIZE; i++) {
      	int row = block*DET_BLOCK_SIZE + i;
	
      	T a = Ainv[row*rowstride+col];
      	if (col == k)
      	  Ainv_colk_shared[i] = a;
      	Ainv_delta_shared[threadIdx.x] += a*(u_shared[i]-uold_shared[i]);
      }
      __syncthreads();
      Ainv_colk[block*DET_BLOCK_SIZE+threadIdx.x] = Ainv_colk_shared[threadIdx.x];
    }
  }
  else {
    for (int block=0; block<numblocks; block++) {
      u_shared[threadIdx.x] = u[block*DET_BLOCK_SIZE+threadIdx.x];
      uold_shared[threadIdx.x] = 
  	A[k*rowstride + block*DET_BLOCK_SIZE+threadIdx.x];
      // Write new row into A matrix
      A[k*rowstride + block*DET_BLOCK_SIZE+threadIdx.x] = u_shared[threadIdx.x];
      __syncthreads();
      for (int i=0; i<DET_BLOCK_SIZE; i++) {
  	int row = block*DET_BLOCK_SIZE + i;
  	Ainv_delta_shared[threadIdx.x] += 
  	  Ainv[row*rowstride+col]*(u_shared[i]- uold_shared[i]);
      }
    }
  }

  __syncthreads();
  
  // Write the data back to global memory
  Ainv_delta[col]    = Ainv_delta_shared[threadIdx.x];
}


template<typename T>
__global__ void
update_inverse_cuda2 (T *Ainv_g[], T *u_g[], T *Ainv_delta_g[],
		      T *Ainv_colk_g[], int N, int rowstride, int k)
{
  __shared__ T *Ainv, *Ainv_delta, *Ainv_colk;
  if (threadIdx.x==0) {
    Ainv     = Ainv_g[blockIdx.y];
    Ainv_delta    = Ainv_delta_g[blockIdx.y];
    Ainv_colk = Ainv_colk_g[blockIdx.y];
  }
  __syncthreads();

  __shared__ T Ainv_delta_shared[DET_BLOCK_SIZE];
  __shared__ T  Ainv_colk_shared[DET_BLOCK_SIZE];
  int col = blockIdx.x*DET_BLOCK_SIZE + threadIdx.x;
  // Read the data back from global memory
  Ainv_delta_shared[threadIdx.x] = Ainv_delta[col];
  Ainv_colk_shared[threadIdx.x] = Ainv_colk[col];
  __shared__ T prefact;
  if (threadIdx.x == 0)
    prefact = -1.0f/(1.0f+Ainv_delta[k]);
  __syncthreads();
		   
  int numblocks = N / DET_BLOCK_SIZE;
  for (int block=0; block<numblocks; block++) {
    Ainv_colk_shared[threadIdx.x] = 
      prefact*Ainv_colk[block*DET_BLOCK_SIZE+threadIdx.x];
    __syncthreads();
    for (int i=0; i<DET_BLOCK_SIZE; i++) {
      int row = block*DET_BLOCK_SIZE + i;
      Ainv[row*rowstride+col] += 
	Ainv_delta_shared[threadIdx.x]*Ainv_colk_shared[i];
    }
  }
}


void
update_inverse_cuda(float *A_g[], float *Ainv_g[], float *u_g[], 
		    float *Ainv_delta_g[], float *Ainv_colk_g[], 
		    int N, int rowstride, int iat, int numWalkers)
{
  dim3 dimBlock(DET_BLOCK_SIZE);
  dim3 dimGrid(N/DET_BLOCK_SIZE, numWalkers);

  update_inverse_cuda1<float><<<dimGrid,dimBlock>>>
    (A_g, Ainv_g, u_g, Ainv_delta_g, Ainv_colk_g, N, rowstride, iat);
  update_inverse_cuda2<float><<<dimGrid,dimBlock>>>
    (Ainv_g, u_g, Ainv_delta_g, Ainv_colk_g, N, rowstride, iat);
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in update_inverse_cuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}

void
update_inverse_cuda(double *A_g[], double *Ainv_g[], double *u_g[], 
		    double *Ainv_delta_g[], double *Ainv_colk_g[], 
		    int N, int rowstride, int iat, int numWalkers)
{
  dim3 dimBlock(DET_BLOCK_SIZE);
  dim3 dimGrid(N/DET_BLOCK_SIZE, numWalkers);

  fprintf (stderr, "dimBlock = %d\n", dimBlock.x);
  fprintf (stderr, "dimGrid  = (%d, %d)\n", dimGrid.x, dimGrid.y);

  update_inverse_cuda1<double><<<dimGrid,dimBlock>>>
    (A_g, Ainv_g, u_g, Ainv_delta_g, Ainv_colk_g, N, rowstride, iat);
  update_inverse_cuda2<double><<<dimGrid,dimBlock>>>
    (Ainv_g, u_g, Ainv_delta_g, Ainv_colk_g, N, rowstride, iat);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in update_inverse_cuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

}


template<typename T, int BS>
__global__ void
calc_ratios (T *Ainv_list[], T *new_row_list[], 
	     T *ratio, int N, int row_stride, int elec)
{
  int tid = threadIdx.x;

  int col = /*blockIdx.x*BS * */tid;
  __shared__ T *Ainv, *new_row;

  if (tid == 0) {
    Ainv = Ainv_list[blockIdx.x];
    new_row = new_row_list[blockIdx.x];
  }
  __syncthreads();
  __shared__ T new_row_shared[BS];
   
  if (col < N) 
    new_row_shared[tid] = new_row[tid];
    
  __shared__ T Ainv_colk_shared[BS];
  // This is *highly* uncoallesced, but we just have to eat it to allow
  // other kernels to operate quickly.
  if (col < N)
    Ainv_colk_shared[tid] = Ainv[col*row_stride + elec];
  __syncthreads();

  __shared__ T Ainv_new_row[BS];
  if (col < N)
    Ainv_new_row[tid] = Ainv_colk_shared[tid] * new_row_shared[tid];
    
  __syncthreads();
    // Now, we have to dot
  for (unsigned int s=BS/2; s>0; s>>=1) {
    if (tid < s && (tid+s) < N)
      Ainv_new_row[tid] += Ainv_new_row[tid + s];
    __syncthreads();
  }
  if (tid == 0)      ratio[blockIdx.x] = Ainv_new_row[0];
}


void
determinant_ratios_cuda (float *Ainv_list[], float *new_row_list[],
			 float *ratios, int N, int row_stride, int iat,
			 int numWalkers)
{
  dim3 dimBlock(N);
  dim3 dimGrid(numWalkers);

  cudaThreadSynchronize();
  cudaError_t err1 = cudaGetLastError();
  if (err1 != cudaSuccess) {
    fprintf (stderr, "CUDA error before determinant_ratios_cuda:\n  %s\n",
	     cudaGetErrorString(err1));
    abort();
  }

  if (N <= 32) 
    calc_ratios<float,32><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 64)
    calc_ratios<float,64><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 128)
    calc_ratios<float,128><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 256)
    calc_ratios<float,256><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 512)
    calc_ratios<float,512><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 1024)
    calc_ratios<float,1024><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else {
    fprintf (stdout, "Error:  N too large for CUDA evaluation.\n");
    abort();
  }

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in determinant_ratios_cuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

}

void
determinant_ratios_cuda (double *Ainv_list[], double *new_row_list[],
			 double *ratios, int N, int row_stride, int iat,
			 int numWalkers)
{
  dim3 dimBlock(N);
  dim3 dimGrid(numWalkers);

  if (N <= 32) 
    calc_ratios<double,32><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 64)
    calc_ratios<double,64><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 128)
    calc_ratios<double,128><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 256)
    calc_ratios<double,256><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else if (N <= 512)
    calc_ratios<double,512><<<dimGrid,dimBlock>>>(Ainv_list, new_row_list, ratios, N, row_stride, iat);
  else {
    fprintf (stdout, "Error:  N too large for CUDA evaluation.\n");
    abort();
  }
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in determinant_ratios_cuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}

#define RATIO_BS 16

template<typename T>
__global__ void
all_ratios_kernel (T *Ainv_list[], T *new_mat_list[], 
		   T *ratio_list[], int N, int row_stride)
{
  __shared__ T *Ainv, *new_mat, *ratio;
  
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    Ainv    = Ainv_list[blockIdx.x];
    new_mat = new_mat_list[blockIdx.x];
    ratio   = ratio_list[blockIdx.x];
  }

  __shared__ float Ainv_block[RATIO_BS][RATIO_BS+1];
  // __shared__ float new_block[RATIO_BS][RATIO_BS+1];
  __shared__ float ratio_block[RATIO_BS][RATIO_BS+1];
  unsigned int numBlocks = N >> 4;
  if (N & 15)
    numBlocks++;

  for (unsigned int yBlock=0; yBlock<numBlocks; yBlock++) {
    ratio_block[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    for (unsigned int xBlock=0; xBlock<numBlocks; xBlock++) {
      unsigned int xIndex = yBlock * RATIO_BS + threadIdx.x;
      unsigned int yIndex = xBlock * RATIO_BS + threadIdx.y;
      unsigned int index  = yIndex*row_stride + xIndex;
      if ((xIndex < N) && (yIndex < N))
	Ainv_block[threadIdx.x][threadIdx.y] = Ainv[index];
      __syncthreads();
      xIndex = xBlock * RATIO_BS + threadIdx.x;
      yIndex = yBlock * RATIO_BS + threadIdx.y;
      index  = yIndex*row_stride + xIndex;

      if ((xIndex < N) && (yIndex < N))
	ratio_block[threadIdx.y][threadIdx.x] +=
	  new_mat[index] * Ainv_block[threadIdx.y][threadIdx.x];
    }
    __syncthreads();
    // Now, we have to do the reduction across the ratio_blocks
    
    if (threadIdx.x < 8)
      ratio_block[threadIdx.y][threadIdx.x] +=
	ratio_block[threadIdx.y][threadIdx.x+8];
    if (threadIdx.x < 4)
      ratio_block[threadIdx.y][threadIdx.x] +=
	ratio_block[threadIdx.y][threadIdx.x+4];
    if (threadIdx.x < 2)
      ratio_block[threadIdx.y][threadIdx.x] +=
	ratio_block[threadIdx.y][threadIdx.x+2];
    if (threadIdx.x < 1) 
      ratio_block[threadIdx.y][threadIdx.x] +=
	ratio_block[threadIdx.y][threadIdx.x+1];
    __syncthreads();

    if (threadIdx.y == 0 && (yBlock * RATIO_BS + threadIdx.x) < N)
      ratio[yBlock * RATIO_BS + threadIdx.x] = ratio_block[threadIdx.x][0];
  }      
}




void
calc_all_ratios (float *Ainv_list[], float *new_mat_list[],
		 float *ratio_list[], int N, int row_stride, int num_mats)
{
  dim3 dimBlock(RATIO_BS, RATIO_BS);
  dim3 dimGrid (num_mats);

  all_ratios_kernel<float><<<dimGrid,dimBlock>>>
    (Ainv_list, new_mat_list, ratio_list, N, row_stride);
}


#define SCALE_BS 64

__constant__ float GGt[3][3];


template<typename T>
__global__ void
scale_grad_lapl (T *grad_list[], T *hess_list[],
		 T *grad_lapl_list[], float Linv[], int N)
{
  __shared__ float gradBlock[3][SCALE_BS];
  __shared__ float hessBlock[6][SCALE_BS];
  __shared__ float outBlock [4][SCALE_BS];
  __shared__ float G[3][3];
  __shared__ float *grad, *hess, *out;
  
  if (threadIdx.x == 0) {
    grad = grad_list[blockIdx.y];
    hess = hess_list[blockIdx.y];
    out  = grad_lapl_list[blockIdx.y];
  }
  if (threadIdx.x < 9)
    G[threadIdx.x/3][threadIdx.x%3] = Linv[threadIdx.x];

  
  // Load the gradients into shared memory
  for (int i=0; i<3; i++) {
    unsigned int gIndex = (3 * blockIdx.x+i) * SCALE_BS + threadIdx.x;
    if (gIndex < 3*N)  gradBlock[i][threadIdx.x] = grad[gIndex];
  }
  // Load the hessian into shared memory
  for (int i=0; i<6; i++) {
    unsigned int hIndex = (6 * blockIdx.x+i) * SCALE_BS + threadIdx.x;
    if (hIndex < 6*N)  hessBlock[i][threadIdx.x] = grad[hIndex];
  }

}



template<typename T>
__global__ void
all_ratios_grad_lapl_kernel (T *Ainv_list[], T *grad_lapl_list[], 
			     T *out_list[], int N, int row_stride)
{
  __shared__ T *Ainv, *gl_array, *out;
  
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    Ainv     = Ainv_list[blockIdx.x];
    gl_array = grad_lapl_list[blockIdx.x];
    out      = out_list[blockIdx.x];
  }
  __syncthreads();

  __shared__ float Ainv_block[RATIO_BS][RATIO_BS+1];
  __shared__ float grad_lapl_block[4][RATIO_BS][RATIO_BS+1];
  unsigned int numBlocks = N >> 4;
  if (N & 15)
    numBlocks++;

  __syncthreads();
  for (unsigned int yBlock=0; yBlock<numBlocks; yBlock++) {
    __syncthreads();
    grad_lapl_block[0][threadIdx.y][threadIdx.x] = 0.0f;
    grad_lapl_block[1][threadIdx.y][threadIdx.x] = 0.0f;
    grad_lapl_block[2][threadIdx.y][threadIdx.x] = 0.0f;
    grad_lapl_block[3][threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    for (unsigned int xBlock=0; xBlock<numBlocks; xBlock++) {
      unsigned int xIndex = yBlock * RATIO_BS + threadIdx.x;
      unsigned int yIndex = xBlock * RATIO_BS + threadIdx.y;
      unsigned int index  = yIndex*row_stride + xIndex;
      if ((xIndex < N) && (yIndex < N))
       	Ainv_block[threadIdx.x][threadIdx.y] = Ainv[index];
      __syncthreads();
      xIndex = xBlock * RATIO_BS + threadIdx.x;
      yIndex = yBlock * RATIO_BS + threadIdx.y;
      index  = 4*yIndex*row_stride + xIndex;
      __syncthreads();
      if ((xIndex < N) && (yIndex < N)) {
	grad_lapl_block[0][threadIdx.y][threadIdx.x] +=
	  gl_array[index+0*row_stride] * Ainv_block[threadIdx.y][threadIdx.x];
	grad_lapl_block[1][threadIdx.y][threadIdx.x] +=
	  gl_array[index+1*row_stride] * Ainv_block[threadIdx.y][threadIdx.x];
	grad_lapl_block[2][threadIdx.y][threadIdx.x] +=
	  gl_array[index+2*row_stride] * Ainv_block[threadIdx.y][threadIdx.x];
	grad_lapl_block[3][threadIdx.y][threadIdx.x] +=
	  gl_array[index+3*row_stride] * Ainv_block[threadIdx.y][threadIdx.x];
      }
      __syncthreads();
    }
    // Now, we have to do the reduction across the lapl_blocks
    
    if (threadIdx.x < 8) {
      grad_lapl_block[0][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[0][threadIdx.y][threadIdx.x+8];
      grad_lapl_block[1][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[1][threadIdx.y][threadIdx.x+8];
      grad_lapl_block[2][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[2][threadIdx.y][threadIdx.x+8];
      grad_lapl_block[3][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[3][threadIdx.y][threadIdx.x+8];
    }
    __syncthreads();
    if (threadIdx.x < 4) {
      grad_lapl_block[0][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[0][threadIdx.y][threadIdx.x+4];
      grad_lapl_block[1][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[1][threadIdx.y][threadIdx.x+4];
      grad_lapl_block[2][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[2][threadIdx.y][threadIdx.x+4];
      grad_lapl_block[3][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[3][threadIdx.y][threadIdx.x+4];
    }
    __syncthreads();
    if (threadIdx.x < 2) {
      grad_lapl_block[0][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[0][threadIdx.y][threadIdx.x+2];
      grad_lapl_block[1][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[1][threadIdx.y][threadIdx.x+2];
      grad_lapl_block[2][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[2][threadIdx.y][threadIdx.x+2];
      grad_lapl_block[3][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[3][threadIdx.y][threadIdx.x+2];
    }
    __syncthreads();
    if (threadIdx.x < 1) {
      grad_lapl_block[0][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[0][threadIdx.y][threadIdx.x+1];
      grad_lapl_block[1][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[1][threadIdx.y][threadIdx.x+1];
      grad_lapl_block[2][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[2][threadIdx.y][threadIdx.x+1];
      grad_lapl_block[3][threadIdx.y][threadIdx.x] +=
    	grad_lapl_block[3][threadIdx.y][threadIdx.x+1];
    }
    __syncthreads();

    // unsigned int yIndex = yBlock * RATIO_BS + threadIdx.x;

    // if (threadIdx.y == 0 && yIndex < N) {
    //   out[4*yIndex+0] = grad_lapl_block[0][threadIdx.x][0];
    //   out[4*yIndex+1] = grad_lapl_block[1][threadIdx.x][0];
    //   out[4*yIndex+2] = grad_lapl_block[2][threadIdx.x][0];
    //   out[4*yIndex+3] = grad_lapl_block[3][threadIdx.x][0];
    // }
    //unsigned int yIndex = 4*yBlock*RATIO_BS + 4*threadIdx.y + threadIdx.x;

    unsigned int ix = 16*threadIdx.y + threadIdx.x;

    if (ix < 64) 
      out[64*yBlock + ix] = grad_lapl_block[ix&3][ix>>2][0];
    // IMPORTANT!!!
    __syncthreads();
  }      
}

void
calc_grad_lapl (float *Ainv_list[], float *grad_lapl_list[],
		float *out_list[], int N, int row_stride, int num_mats)
{
  dim3 dimBlock(RATIO_BS, RATIO_BS);
  dim3 dimGrid (num_mats);

  all_ratios_grad_lapl_kernel<float><<<dimGrid,dimBlock>>>
    (Ainv_list, grad_lapl_list, out_list, N, row_stride);
}



#include <stdlib.h>
#include <time.h>

void
test_all_ratios_kernel()
{
  int N = 128;

  float *A, *A_d, *Ainv, *Ainv_d, *ratio, *ratio_d;

  cudaMalloc ((void**)&A_d,    N*N*sizeof(float));
  cudaMalloc ((void**)&Ainv_d, N*N*sizeof(float));  
  cudaMalloc ((void**)&ratio_d, 1*N*sizeof(float));
  A     = (float *)malloc (N*N*sizeof(float));
  Ainv  = (float *)malloc (N*N*sizeof(float));
  ratio = (float *)malloc (1*N*sizeof(float));

  float ratio2[N];
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) {
      A[i*N+j] = 1.0f+drand48();
      Ainv[i*N+j] = 1.0f+drand48();
    }
  
  cudaMemcpy (A_d,     A,    N*N*sizeof(float), cudaMemcpyHostToDevice);  
  cudaMemcpy (Ainv_d,  Ainv, N*N*sizeof(float), cudaMemcpyHostToDevice);

  float **A_list, **A_list_d, **Ainv_list, **Ainv_list_d, **ratio_list, **ratio_list_d;
  int numMats = 2000;


  cudaMalloc ((void**)&A_list_d,     numMats*sizeof(float*));
  cudaMalloc ((void**)&Ainv_list_d,  numMats*sizeof(float*));
  cudaMalloc ((void**)&ratio_list_d, numMats*sizeof(float*));
  A_list     = (float **)malloc (numMats*sizeof(float*));
  Ainv_list  = (float **)malloc (numMats*sizeof(float*));
  ratio_list = (float **)malloc (numMats*sizeof(float*));

  for (int i=0; i<numMats; i++) {
    A_list[i] = A_d;
    Ainv_list[i] = Ainv_d;
    ratio_list[i] = ratio_d;
  }

  cudaMemcpy (A_list_d,    A_list,      numMats*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy (Ainv_list_d, Ainv_list,   numMats*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy (ratio_list_d, ratio_list, numMats*sizeof(float*), cudaMemcpyHostToDevice);

  clock_t start = clock();
  for (int i=0; i<1000; i++) 
    calc_all_ratios (Ainv_list_d, A_list_d, ratio_list_d, N, N, numMats);
  clock_t end = clock();
  double time = (double)(end-start)/(double)CLOCKS_PER_SEC;
  fprintf (stderr, "start = %d\n", start);
  fprintf (stderr, "end = %d\n", end);
  double rate = 1000.0/time;
  fprintf (stderr, "Rate = %1.2f generations per second.\n", rate);


  cudaMemcpy (ratio, ratio_d, N*sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i=0; i<N; i++) {
  //   ratio2[i] = 0.0f;
  //   for (int j=0; j<N; j++)
  //     ratio2[i] += A[i*N+j]*Ainv[j*N+i];
  //   fprintf (stderr, "%3d  %10.6f  %10.6f\n", i, ratio2[i], ratio[i]);
  // }
  

}



void
test_all_grad_lapl_kernel()
{
  int N = 128;

  float *A, *A_d, *Ainv, *Ainv_d, *ratio, *ratio_d;

  cudaMalloc ((void**)&A_d,     4*N*N*sizeof(float));
  cudaMalloc ((void**)&Ainv_d,  N*N*sizeof(float));  
  cudaMalloc ((void**)&ratio_d, 4*N*sizeof(float));
  A     = (float *)malloc (4*N*N*sizeof(float));
  Ainv  = (float *)malloc (1*N*N*sizeof(float));
  ratio = (float *)malloc (4*N*sizeof(float));

  float ratio2[4*N];
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) {
      Ainv[i*N+j] = 1.0f+drand48();
      for (int k=0; k<4; k++)
	A[4*(i*N+j)+k] = 1.0f+drand48();
    }
  
  cudaMemcpy (A_d,     A,    4*N*N*sizeof(float), cudaMemcpyHostToDevice);  
  cudaMemcpy (Ainv_d,  Ainv, 1*N*N*sizeof(float), cudaMemcpyHostToDevice);

  float **A_list, **A_list_d, **Ainv_list, **Ainv_list_d, **ratio_list, **ratio_list_d;
  int numMats = 2000;


  cudaMalloc ((void**)&A_list_d,     numMats*sizeof(float*));
  cudaMalloc ((void**)&Ainv_list_d,  numMats*sizeof(float*));
  cudaMalloc ((void**)&ratio_list_d, numMats*sizeof(float*));
  A_list     = (float **)malloc (numMats*sizeof(float*));
  Ainv_list  = (float **)malloc (numMats*sizeof(float*));
  ratio_list = (float **)malloc (numMats*sizeof(float*));

  for (int i=0; i<numMats; i++) {
    A_list[i] = A_d;
    Ainv_list[i] = Ainv_d;
    ratio_list[i] = ratio_d;
  }

  cudaMemcpy (A_list_d,    A_list,      numMats*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy (Ainv_list_d, Ainv_list,   numMats*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy (ratio_list_d, ratio_list, numMats*sizeof(float*), cudaMemcpyHostToDevice);

  struct timeval tstart, tend;
  gettimeofday(&tstart, NULL);
  for (int i=0; i<100; i++) 
    calc_grad_lapl (Ainv_list_d, A_list_d, ratio_list_d, N, N, numMats);
  cudaMemcpy (ratio, ratio_d, 4*N*sizeof(float), cudaMemcpyDeviceToHost);
  gettimeofday(&tend, NULL);
  double start = (double)tstart.tv_sec + 1.0e-6 * (double)tstart.tv_usec;  
  double end   = (double)tend.tv_sec   + 1.0e-6 * (double)tend.tv_usec;
  fprintf (stderr, "start = %f\n", start);
  fprintf (stderr, "end = %f\n", end);
  double rate = 100.0/(end-start);
  fprintf (stderr, "Rate = %1.2f generations per second.\n", rate);



  for (int i=0; i<N; i++) {
    for (int k=0; k<4; k++)
      ratio2[4*i+k] = 0.0f;
    for (int j=0; j<N; j++)
      for (int k=0; k<4; k++)
	ratio2[4*i+k] += A[(4*i+k)*N+j]*Ainv[j*N+i];
    for (int k=0; k<4; k++)
    fprintf (stderr, "%3d  %10.6f  %10.6f\n", 4*i+k, ratio2[4*i+k], ratio[4*i+k]);
  }
 

}





#ifdef CUDA_TEST_MAIN

// Compile with:
// nvcc -o test_all_ratios -DCUDA_TEST_MAIN ../src/QMCWaveFunctions/Fermion/determinant_update.cu
main()
{
  //test_all_ratios_kernel();
  test_all_grad_lapl_kernel();
}



#endif
