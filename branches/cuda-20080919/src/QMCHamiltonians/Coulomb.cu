texture<float,1,cudaReadModeElementType> myTex;

texture<float,1,cudaReadModeElementType> shortTex;


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

template<typename T, int BS>
__global__ void
coulomb_AA_kernel(T *R[], int N, T rMax, int Ntex,
		  T lattice[], T latticeInv[], T sum[])
{
  int tid = threadIdx.x;
  __shared__ T *myR;

  __shared__ float lastsum;
  if (tid == 0) {
    myR = R[blockIdx.x];
    lastsum = sum[blockIdx.x];
  }
  __shared__ T L[3][3], Linv[3][3];
  if (tid < 9) {
    L[0][tid] = lattice[tid];
    Linv[0][tid] = latticeInv[tid];
  }

  __syncthreads();

  T nrm = (T)(Ntex-1)/rMax;
  __shared__ float r1[BS][3], r2[BS][3];
  int NB = N/BS + ((N%BS) ? 1 : 0);

  T mysum = (T)0.0; 

  // Do diagonal blocks first
  for (int b=0; b<NB; b++) {
    for (int i=0; i<3; i++)
      if ((3*b+i)*BS + tid < N)
	r1[0][i*BS+tid] = myR[(3*b+i)*BS + tid];
    int ptcl1 = b*BS + tid;
    if (ptcl1 < N) {
      int end = (b+1)*BS < N ? BS : N-b*BS;
      for (int p2=0; p2<end; p2++) {
	int ptcl2 = b*BS + p2;
	T dx, dy, dz;
	dx = r1[p2][0] - r1[tid][0];
	dy = r1[p2][1] - r1[tid][1];
	dz = r1[p2][2] - r1[tid][2];
	T dist = min_dist(dx, dy, dz, L, Linv);
	if (ptcl1 != ptcl2)
	  mysum += tex1D(shortTex, nrm*dist+0.5);
      }
    }
  }
  // Avoid double-counting on the diagonal blocks
  mysum *= 0.5;

  // Now do off-diagonal blocks
  for (int b1=0; b1<NB; b1++) {
    for (int i=0; i<3; i++)
      if ((3*b1+i)*BS + tid < N)
	r1[0][i*BS+tid] = myR[(3*b1+i)*BS + tid];
    int ptcl1 = b1*BS + tid;
    if (ptcl1 < N) {
      for (int b2=b1+1; b2<NB; b2++) {
	for (int i=0; i<3; i++)
	  if ((3*b2+i)*BS + tid < N)
	    r2[0][i*BS+tid] = myR[(3*b2+i)*BS + tid];
	int end = (b2+1)*BS < N ? BS : N-b2*BS;
	for (int j=0; j<end; j++) {
	  T dx, dy, dz;
	  dx = r2[j][0] - r1[tid][0];
	  dy = r2[j][1] - r1[tid][1];
	  dz = r2[j][2] - r1[tid][2];
	  T dist = min_dist(dx, dy, dz, L, Linv);
	  mysum += tex1D(shortTex, nrm*dist+0.5);
	}
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
  if (tid==0)
    sum[blockIdx.x] = lastsum + shared_sum[0];
}


void
CoulombAA_SR_Sum(float *R[], int N, float rMax, int Ntex,
		 float lattice[], float latticeInv[], float sum[],
		 int numWalkers)
{
  const int BS=32;
  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  coulomb_AA_kernel<float,BS><<<dimGrid,dimBlock>>>
    (R, N, rMax, Ntex, lattice, latticeInv, sum);
}


__global__ void
test_texture_kernel(float x[], float vals[], int Ntex, int Nvals)
{
  float nrm = (float)(Ntex-1)/(float)Ntex;

  for (int i=0; i<Nvals; i++)
    vals[i] = tex1D(myTex, nrm*x[i]+0.5);
}

#include <stdio.h>

void
TestTexture()
{
  int Ntex = 2000;
  int Npoints = 31415;

  cudaArray *myArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);

  cudaMallocArray(&myArray, &channelDesc, Ntex);
  float data[Ntex];
  for (int i=0; i<Ntex; i++) {
    double x = (double)i/(double)(Ntex-1) * 2.0*M_PI;
    data[i] = (float)sin(x);
  }
  cudaMemcpyToArray(myArray, 0, 0, data, Ntex*sizeof(float), cudaMemcpyHostToDevice);
  myTex.addressMode[0] = cudaAddressModeClamp;
  myTex.filterMode = cudaFilterModeLinear;
  myTex.normalized = false;

  cudaBindTextureToArray(myTex, myArray, channelDesc);

  float *x_d, *vals_d;
  cudaMalloc ((void**)&x_d, Npoints*sizeof(float));
  cudaMalloc ((void**)&vals_d, Npoints*sizeof(float));

  float x_host[Npoints];
  for (int i=0; i<Npoints; i++) 
    x_host[i] = (double)i/(double)(Npoints-1) * (double)Ntex;

  cudaMemcpy(x_d, x_host, Npoints*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(1);
  dim3 dimGrid(1);
  
  test_texture_kernel<<<dimGrid,dimBlock>>>(x_d, vals_d, Ntex, Npoints);
  
  float vals_host[Npoints];
  cudaMemcpy(vals_host, vals_d, Npoints*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i=0; i<Npoints; i++) 
    fprintf (stderr, "%18.10f %18.10f\n", sin(2.0*M_PI*x_host[i]/(double)Ntex), vals_host[i]);

}

main()
{
  TestTexture();

}
