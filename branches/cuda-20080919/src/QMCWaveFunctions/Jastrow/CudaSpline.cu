#define MAX_SPLINES 100
#include <stdio.h>

bool AisInitialized = false;


// void
// createCudaSplines (float rmax, int N,
// 		   float f[], float df[], float d2f[],
// 		   int &fSpline, int &dfSpline, int &d2fSpline)
// {
//   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
//   cudaArray *fArray, *dfArray, *d2fArray;
//   cudaMallocArray(  &fArray, &channelDesc, N);
//   cudaMallocArray( &dfArray, &channelDesc, N);
//   cudaMallocArray(&d2fArray, &channelDesc, N);
  
//   cudaMemcpyToArray(fArray,  N,1,  f,N*sizeof(float),cudaMemcpyHostToDevice);
//   cudaMemcpyToArray(dfArray, N,1, df,N*sizeof(float),cudaMemcpyHostToDevice);
//   cudaMemcpyToArray(d2fArray,N,1,d2f,N*sizeof(float),cudaMemcpyHostToDevice);


//   cudaBindTextureToArray(texSplines[fSpline=curTex++], fArray);
//   cudaBindTextureToArray(texSplines[dfSpline=curTex++], dfArray);
//   cudaBindTextureToArray(texSplines[d2fSpline=curTex++], d2fArray);
// }


template<typename T>
__device__
T min_dist (T& x, T& y, T& z, 
	    T L[3][3], T Linv[3][3])
{
//   T u0 = Linv[0][0]*x + Linv[0][1]*y + Linv[0][2]*z;  
//   T u1 = Linv[1][0]*x + Linv[1][1]*y + Linv[1][2]*z;
//   T u2 = Linv[2][0]*x + Linv[2][1]*y + Linv[2][2]*z;

//   u0 -= rintf(u0);
//   u1 -= rintf(u1);
//   u2 -= rintf(u2);

//   x = L[0][0]*u0 + L[0][1]*u1 + L[0][2]*u2;
//   y = L[1][0]*u0 + L[1][1]*u1 + L[1][2]*u2;
//   z = L[2][0]*u0 + L[2][1]*u1 + L[2][2]*u2;

  T u0 = Linv[0][0]*x; u0 -= rintf(u0); x = L[0][0]*u0;
  T u1 = Linv[1][1]*y; u1 -= rintf(u1); y = L[1][1]*u1;
  T u2 = Linv[2][2]*z; u2 -= rintf(u2); z = L[2][2]*u2;

  return sqrtf(x*x + y*y + z*z);

//   T d2min = x*x + y*y + z*z;
//   for (T i=-1.0f; i<=1.001; i+=1.0f)
//     for (T j=-1.0f; j<=1.001; j+=1.0f)
//       for (T k=-1.0f; k<=1.001; k+=1.0f) {
// 	T xnew = L[0][0]*(u0+i) + L[0][1]*(u1+j) + L[0][2]*(u2+k);
// 	T ynew = L[1][0]*(u0+i) + L[1][1]*(u1+j) + L[1][2]*(u2+k);
// 	T znew = L[2][0]*(u0+i) + L[2][1]*(u1+j) + L[2][2]*(u2+k);
	
// 	T d2 = xnew*xnew + ynew*ynew + znew*znew;
// 	d2min = min (d2, d2min);
// 	if (d2 < d2min) {
// 	  d2min = d2;
// 	  x = xnew;
// 	  y = ynew;
// 	  z = znew;
// 	}
//       }
//   return sqrt(d2min);
}

__constant__ float AcudaSpline[48];
__constant__ double AcudaSpline_double[48];

void
cuda_spline_init()
{
  float A_h[48] = { -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
		     3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
		    -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
		     1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0,
		         0.0,     -0.5,      1.0,    -0.5,
		         0.0,      1.5,     -2.0,     0.0,
		         0.0,     -1.5,      1.0,     0.5,
		         0.0,      0.5,      0.0,     0.0,
		         0.0,      0.0,     -1.0,     1.0,
		         0.0,      0.0,      3.0,    -2.0,
		         0.0,      0.0,     -3.0,     1.0,
		         0.0,      0.0,      1.0,     0.0 };

  cudaMemcpyToSymbol(AcudaSpline, A_h, 48*sizeof(float), 0, cudaMemcpyHostToDevice);

  double A_d[48] = { -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
		     3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
		    -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
		     1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0,
		         0.0,     -0.5,      1.0,    -0.5,
		         0.0,      1.5,     -2.0,     0.0,
		         0.0,     -1.5,      1.0,     0.5,
		         0.0,      0.5,      0.0,     0.0,
		         0.0,      0.0,     -1.0,     1.0,
		         0.0,      0.0,      3.0,    -2.0,
		         0.0,      0.0,     -3.0,     1.0,
		         0.0,      0.0,      1.0,     0.0 };

  cudaMemcpyToSymbol(AcudaSpline_double, A_d, 48*sizeof(double), 0, cudaMemcpyHostToDevice);

  AisInitialized = true;
}


template<typename T>
__device__ T 
eval_1d_spline(T dist, T rmax, T drInv, T A[4][4], T coefs[])
{
  if (dist > rmax)  return (T)0.0;

  T s = dist * drInv;
  T sf = floorf (s);
  int index = (int)sf;
  T t = s - sf;

//   return (coefs[index+0]*(AcudaSpline[ 0]*t*t*t + AcudaSpline[ 1]*t*t + AcudaSpline[ 2]*t + AcudaSpline[ 3]) +
//   	  coefs[index+1]*(AcudaSpline[ 4]*t*t*t + AcudaSpline[ 5]*t*t + AcudaSpline[ 6]*t + AcudaSpline[ 7]) +
//   	  coefs[index+2]*(AcudaSpline[ 8]*t*t*t + AcudaSpline[ 9]*t*t + AcudaSpline[10]*t + AcudaSpline[11]) +
//   	  coefs[index+3]*(AcudaSpline[12]*t*t*t + AcudaSpline[13]*t*t + AcudaSpline[14]*t + AcudaSpline[15]));


  return (coefs[index+0]*(A[0][0]*t*t*t + A[0][1]*t*t + A[0][2]*t + A[0][3]) +
  	  coefs[index+1]*(A[1][0]*t*t*t + A[1][1]*t*t + A[1][2]*t + A[1][3]) +
  	  coefs[index+2]*(A[2][0]*t*t*t + A[2][1]*t*t + A[2][2]*t + A[2][3]) +
  	  coefs[index+3]*(A[3][0]*t*t*t + A[3][1]*t*t + A[3][2]*t + A[3][3]));
}


template<typename T>
__device__ void 
eval_1d_spline_vgl(T dist, T rmax, T drInv, T A[12][4], T coefs[],
		   T& u, T& du, T& d2u)
{
  if (dist > rmax) {
    u = du = d2u = (T)0.0;
    return;
  }

  T s = dist * drInv;
  T sf = floorf (s);
  int index = (int)sf;
  T t = s - sf;

  u = (coefs[index+0]*(A[0][0]*t*t*t + A[0][1]*t*t + A[0][2]*t + A[0][3]) +
       coefs[index+1]*(A[1][0]*t*t*t + A[1][1]*t*t + A[1][2]*t + A[1][3]) +
       coefs[index+2]*(A[2][0]*t*t*t + A[2][1]*t*t + A[2][2]*t + A[2][3]) +
       coefs[index+3]*(A[3][0]*t*t*t + A[3][1]*t*t + A[3][2]*t + A[3][3]));

 du = drInv *    
   (coefs[index+0]*(A[4][0]*t*t*t + A[4][1]*t*t + A[4][2]*t + A[4][3]) +
    coefs[index+1]*(A[5][0]*t*t*t + A[5][1]*t*t + A[5][2]*t + A[5][3]) +
    coefs[index+2]*(A[6][0]*t*t*t + A[6][1]*t*t + A[6][2]*t + A[6][3]) +
    coefs[index+3]*(A[7][0]*t*t*t + A[7][1]*t*t + A[7][2]*t + A[7][3]));
 
 d2u = drInv*drInv * 
   (coefs[index+0]*(A[ 8][0]*t*t*t + A[ 8][1]*t*t + A[ 8][2]*t + A[ 8][3]) +
    coefs[index+1]*(A[ 9][0]*t*t*t + A[ 9][1]*t*t + A[ 9][2]*t + A[ 9][3]) +
    coefs[index+2]*(A[10][0]*t*t*t + A[10][1]*t*t + A[10][2]*t + A[10][3]) +
    coefs[index+3]*(A[11][0]*t*t*t + A[11][1]*t*t + A[11][2]*t + A[11][3]));
}



#define MAX_COEFS 32
template<typename T, int BS >
__global__ void
two_body_sum_kernel(T *R[], int e1_first, int e1_last, 
		    int e2_first, int e2_last,
		    T spline_coefs[], int numCoefs, T rMax,  
		    T lattice[], T latticeInv[], T sum[])
{
  T dr = rMax/(T)(numCoefs-3);
  T drInv = 1.0/dr;

  int tid = threadIdx.x;
  __shared__ T *myR;
  if (tid == 0) 
    myR = R[blockIdx.x];

  __shared__ T coefs[MAX_COEFS];
  if (tid < numCoefs)
    coefs[tid] = spline_coefs[tid];
  __shared__ T r1[BS][3], r2[BS][3];
  __shared__ T L[3][3], Linv[3][3];
  if (tid < 9) {
    L[0][tid] = lattice[tid];
    Linv[0][tid] = latticeInv[tid];
  }
  

  __shared__ T A[4][4];
  if (tid < 16)
    A[tid>>2][tid&3] = AcudaSpline_double[tid];
  __syncthreads();


  int N1 = e1_last - e1_first + 1;
  int N2 = e2_last - e2_first + 1;
  int NB1 = N1/BS + ((N1 % BS) ? 1 : 0);
  int NB2 = N2/BS + ((N2 % BS) ? 1 : 0);

  T mysum = (T)0.0; 
  for (int b1=0; b1 < NB1; b1++) {
    // Load block of positions from global memory
    for (int i=0; i<3; i++)
      if ((3*b1+i)*BS + tid < 3*N1) 
  	r1[0][i*BS + tid] = myR[3*e1_first + (3*b1+i)*BS + tid];
    __syncthreads();
    int ptcl1 = e1_first+b1*BS + tid;
    for (int b2=0; b2 < NB2; b2++) {
      // Load block of positions from global memory
      for (int i=0; i<3; i++)
  	if ((3*b2+i)*BS + tid < 3*N2) 
	  r2[0][i*BS + tid] = myR[3*e2_first + (3*b2+i)*BS + tid];
      __syncthreads();
      // Now, loop over particles
      int end = (b2+1)*BS < N2 ? BS : N2-b2*BS;
      for (int j=0; j<end; j++) {
  	int ptcl2 = e2_first + b2*BS+j;
  	T dx, dy, dz;
  	dx = r2[j][0] - r1[tid][0];
  	dy = r2[j][1] - r1[tid][1];
  	dz = r2[j][2] - r1[tid][2];
  	T dist = min_dist(dx, dy, dz, L, Linv);
  	if (ptcl1 != ptcl2 && (ptcl1 < (N1+e1_first) ) && (ptcl2 < (N2+e2_first)))
	  mysum += eval_1d_spline (dist, rMax, drInv, A, coefs);
      }

    }
  }
  __shared__ T shared_sum[BS];
  shared_sum[tid] = mysum;
  __syncthreads();
  for (int s=BS>>1; s>0; s >>=1) {
    if (tid < s)
      shared_sum[tid] += shared_sum[tid+s];
    __syncthreads();
  }

  T factor = (e1_first == e2_first) ? 0.5 : 1.0;

  if (tid==0)
    sum[blockIdx.x] += factor*shared_sum[0];

}

void
two_body_sum (float *R[], int e1_first, int e1_last, int e2_first, int e2_last,
	      float spline_coefs[], int numCoefs, float rMax,  
	      float lattice[], float latticeInv[], float sum[], int numWalkers)
{
  if (!AisInitialized)
    cuda_spline_init();

  const int BS = 128;

  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  two_body_sum_kernel<float,BS><<<dimGrid,dimBlock>>>
    (R, e1_first, e1_last, e2_first, e2_last, 
     spline_coefs, numCoefs, rMax, lattice, latticeInv, sum);
}


void
two_body_sum (double *R[], int e1_first, int e1_last, int e2_first, int e2_last,
	      double spline_coefs[], int numCoefs, double rMax,  
	      double lattice[], double latticeInv[], double sum[], int numWalkers)
{
  if (!AisInitialized)
    cuda_spline_init();

  const int BS = 128;

  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  two_body_sum_kernel<double,BS><<<dimGrid,dimBlock>>>
    (R, e1_first, e1_last, e2_first, e2_last, 
     spline_coefs, numCoefs, rMax, lattice, latticeInv, sum);
}




template<typename T, int BS>
__global__ void
two_body_ratio_kernel(T *R[], int first, int last, int Ntotal,
		      T  Rnew[], int inew,
		      T spline_coefs[], int numCoefs, T rMax,  
		      T lattice[], T latticeInv[], T sum[])
{
  T dr = rMax/(T)(numCoefs-3);
  T drInv = 1.0/dr;

  int tid = threadIdx.x;
  __shared__ T *myR;
  __shared__ T myRnew[3], myRold[3];
  if (tid == 0) 
    myR = R[blockIdx.x];
  __syncthreads();
  if (tid < 3 ) {
    myRnew[tid] = Rnew[3*blockIdx.x+tid];
    myRold[tid] = myR[3*inew+tid];
    // Store RNew into R, so that we can copy it if we wish to accept the move
    myR[3*Ntotal+tid] = myRnew[tid];
  }

  __shared__ T coefs[MAX_COEFS];
  if (tid < numCoefs)
    coefs[tid] = spline_coefs[tid];
  __shared__ T r1[BS][3];
  __shared__ T L[3][3], Linv[3][3];
  if (tid < 9) {
    L[0][tid] = lattice[tid];
    Linv[0][tid] = latticeInv[tid];
  }
  
  __shared__ T A[4][4];
  if (tid < 16) 
    A[tid>>2][tid&3] = AcudaSpline_double[tid];
  __syncthreads();

  int N = last - first + 1;
  int NB = N/BS + ((N % BS) ? 1 : 0);

  double mysum = (T)0.0; 
  for (int b=0; b < NB; b++) {
    // Load block of positions from global memory
    for (int i=0; i<3; i++)
      if ((3*b+i)*BS + tid < 3*N) 
  	r1[0][i*BS + tid] = myR[3*first + (3*b+i)*BS + tid];
    __syncthreads();
    int ptcl1 = first+b*BS + tid;

    T dx, dy, dz;
    dx = myRnew[0] - r1[tid][0];
    dy = myRnew[1] - r1[tid][1];
    dz = myRnew[2] - r1[tid][2];
    T dist = min_dist(dx, dy, dz, L, Linv);
    T delta = eval_1d_spline (dist, rMax, drInv, A, coefs);

    dx = myRold[0] - r1[tid][0];
    dy = myRold[1] - r1[tid][1];
    dz = myRold[2] - r1[tid][2];
    dist = min_dist(dx, dy, dz, L, Linv);
    delta -= eval_1d_spline (dist, rMax, drInv, A, coefs);
    
    if (ptcl1 != inew && (ptcl1 < (N+first) ))
      mysum += delta;
  }
  
  __shared__ T shared_sum[BS];
  shared_sum[tid] = mysum;
  __syncthreads();
  for (int s=BS>>1; s>0; s >>=1) {
    if (tid < s)
      shared_sum[tid] += shared_sum[tid+s];
    __syncthreads();
  }

  //  float factor = (first <= inew && inew <= last) ? 0.5 : 1.0;
  if (tid==0)
    sum[blockIdx.x] += /* factor * */ shared_sum[0];
}




void
two_body_ratio (float *R[], int first, int last, int N,
		float Rnew[], int inew,
		float spline_coefs[], int numCoefs, float rMax,  
		float lattice[], float latticeInv[], float sum[], int numWalkers)
{
  if (!AisInitialized)
    cuda_spline_init();

  const int BS = 128;

  //  int numBlocks = numWalkers / BS + ((numWalkers %BS) ? 1 :
  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  two_body_ratio_kernel<float,BS><<<dimGrid,dimBlock>>>
    (R, first, last, N, Rnew, inew, spline_coefs, numCoefs, rMax, 
     lattice, latticeInv, sum);
}



void
two_body_ratio (double *R[], int first, int last, int N,
		double Rnew[], int inew,
		double spline_coefs[], int numCoefs, double rMax,  
		double lattice[], double latticeInv[], double sum[], int numWalkers)
{
  if (!AisInitialized)
    cuda_spline_init();

  dim3 dimBlock(128);
  dim3 dimGrid(numWalkers);

  two_body_ratio_kernel<double,128><<<dimGrid,dimBlock>>>
    (R, first, last, N, Rnew, inew, spline_coefs, numCoefs, rMax, 
     lattice, latticeInv, sum);
}


template<typename T>
__global__ void
two_body_update_kernel (T *R[], int N, int iat)
{
  __shared__ T* myR;
  if (threadIdx.x == 0)
    myR = R[blockIdx.x];
  __syncthreads();
  
  if (threadIdx.x < 3)
    myR[3*iat + threadIdx.x] = myR[3*N + threadIdx.x];
}

void
two_body_update(float *R[], int N, int iat, int numWalkers)
{
  dim3 dimBlock(3);
  dim3 dimGrid(numWalkers);

  two_body_update_kernel<float><<<dimGrid, dimBlock>>> (R, N, iat);
}

void
two_body_update(double *R[], int N, int iat, int numWalkers)
{
  dim3 dimBlock(3);
  dim3 dimGrid(numWalkers);

  two_body_update_kernel<double><<<dimGrid, dimBlock>>> (R, N, iat);
}





#define MAX_COEFS 32

template<typename T, int BS>
__global__ void
two_body_grad_lapl_kernel(T *R[], int e1_first, int e1_last, 
			  int e2_first, int e2_last,
			  T spline_coefs[], int numCoefs, T rMax,  
			  T lattice[], T latticeInv[], 
			  T gradLapl[], int row_stride)
{
  T dr = rMax/(T)(numCoefs-3);
  T drInv = 1.0/dr;
  
  T factor = (e1_first == e2_first) ? 0.5 : 1.0;


  int tid = threadIdx.x;
  __shared__ T *myR;
  if (tid == 0) 
    myR = R[blockIdx.x];

  __shared__ T coefs[MAX_COEFS];
  if (tid < numCoefs)
    coefs[tid] = spline_coefs[tid];
  __shared__ T r1[BS][3], r2[BS][3];
  __shared__ T L[3][3], Linv[3][3];
  if (tid < 9) {
    L[0][tid] = lattice[tid];
    Linv[0][tid] = latticeInv[tid];
  }
  

  __shared__ T A[12][4];
  if (tid < 16) {
    A[0+(tid>>2)][tid&3] = AcudaSpline[tid+0];
    A[4+(tid>>2)][tid&3] = AcudaSpline[tid+16];
    A[8+(tid>>2)][tid&3] = AcudaSpline[tid+32];
  }
  __syncthreads();


  int N1 = e1_last - e1_first + 1;
  int N2 = e2_last - e2_first + 1;
  int NB1 = N1/BS + ((N1 % BS) ? 1 : 0);
  int NB2 = N2/BS + ((N2 % BS) ? 1 : 0);

  __shared__ T sGradLapl[BS][4];
  for (int b1=0; b1 < NB1; b1++) {
    // Load block of positions from global memory
    for (int i=0; i<3; i++)
      if ((3*b1+i)*BS + tid < 3*N1) 
  	r1[0][i*BS + tid] = myR[3*e1_first + (3*b1+i)*BS + tid];
    __syncthreads();
    int ptcl1 = e1_first+b1*BS + tid;
    int offset = blockIdx.x * row_stride + 4*b1*BS;
    sGradLapl[tid][0] = sGradLapl[tid][1] = 
      sGradLapl[tid][2] = sGradLapl[tid][3] = (T)0.0;
    for (int b2=0; b2 < NB2; b2++) {
      // Load block of positions from global memory
      for (int i=0; i<3; i++)
  	if ((3*b2+i)*BS + tid < 3*N2) 
	  r2[0][i*BS + tid] = myR[3*e2_first + (3*b2+i)*BS + tid];
      __syncthreads();
      // Now, loop over particles
      int end = (b2+1)*BS < N2 ? BS : N2-b2*BS;
      for (int j=0; j<end; j++) {
  	int ptcl2 = e2_first + b2*BS+j;
  	T dx, dy, dz, u, du, d2u;
  	dx = r2[j][0] - r1[tid][0];
  	dy = r2[j][1] - r1[tid][1];
  	dz = r2[j][2] - r1[tid][2];
  	T dist = min_dist(dx, dy, dz, L, Linv);
	eval_1d_spline_vgl (dist, rMax, drInv, A, coefs, u, du, d2u);
  	if (ptcl1 != ptcl2 && (ptcl1 < (N1+e1_first) ) && (ptcl2 < (N2+e2_first))) {
	  du /= dist;
	  sGradLapl[tid][0] -= du * dx;
	  sGradLapl[tid][1] -= du * dy;
	  sGradLapl[tid][2] -= du * dz;
	  sGradLapl[tid][3] -= d2u + 2.0*du;
	}
      }
    }
    __syncthreads();
    for (int i=0; i<4; i++)
      if ((4*b1+i)*BS + tid < 4*N1)
	gradLapl[offset + i*BS +tid] += /*factor**/sGradLapl[0][i*BS+tid];
    __syncthreads();
  }
}


void
two_body_grad_lapl(float *R[], int e1_first, int e1_last, 
		   int e2_first, int e2_last,
		   float spline_coefs[], int numCoefs, float rMax,  
		   float lattice[], float latticeInv[], 
		   float gradLapl[], int row_stride, int numWalkers)
{
  const int BS=32;
  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  two_body_grad_lapl_kernel<float,BS><<<dimGrid,dimBlock>>>
    (R, e1_first, e1_last, e2_first, e2_last, spline_coefs, numCoefs, 
     rMax, lattice, latticeInv,  gradLapl, row_stride);
}


void
two_body_grad_lapl(double *R[], int e1_first, int e1_last, 
		   int e2_first, int e2_last,
		   double spline_coefs[], int numCoefs, double rMax,  
		   double lattice[], double latticeInv[], 
		   double gradLapl[], int row_stride, int numWalkers)
{
  const int BS=32;
  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  two_body_grad_lapl_kernel<double,BS><<<dimGrid,dimBlock>>>
    (R, e1_first, e1_last, e2_first, e2_last, spline_coefs, numCoefs, 
     rMax, lattice, latticeInv,  gradLapl, row_stride);
}






void test()
{
  dim3 dimBlock(32);
  dim3 dimGrid(1000);

  float *R[1000];
  float L[9], Linv[9];
  float spline_coefs[10];
  float dr = 0.1;
  float sum[1000];

  two_body_sum_kernel<float,32><<<dimGrid,dimBlock>>>(R, 0, 100, 0, 100, spline_coefs, 10, dr,
						      L, Linv, sum);



}
