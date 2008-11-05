#include "CudaCoulomb.h"

const int MAX_TEXTURES = 10;

texture<float,1,cudaReadModeElementType> myTex;
texture<float,1,cudaReadModeElementType> tex00, tex01, tex02, tex03, tex04, tex05, tex06, tex07, tex08, tex09;
bool textureInUse[MAX_TEXTURES] =  { false, false, false, false, false, 
				     false, false, false, false, false };


#define arraytexFetch(_u, _texnum, _return)\
switch(_texnum)\
{\
case 0:\
 _return = tex1D(tex00, (_u)); \
 break;\
case 1:\
 _return = tex1D(tex01, (_u)); \
 break;			       \
case 2:\
 _return = tex1D(tex02, (_u)); \
 break;			       \
case 3:\
 _return = tex1D(tex03, (_u)); \
 break;			       \
case 4:\
 _return = tex1D(tex04, (_u)); \
 break;			       \
case 5:\
 _return = tex1D(tex05, (_u)); \
 break;			       \
case 6:\
 _return = tex1D(tex06, (_u)); \
 break;			       \
case 7:\
 _return = tex1D(tex07, (_u)); \
 break;			       \
case 8:\
 _return = tex1D(tex08, (_u)); \
 break;			       \
case 9:\
 _return = tex1D(tex09, (_u)); \
 break;			       \
 }


#include <iostream>

TextureSpline::TextureSpline()
{
  cerr << "TextureSpline constructor.\n";
  int iTex = 0;
  while ( iTex < MAX_TEXTURES && textureInUse[iTex]) iTex++;
  if (iTex == MAX_TEXTURES) {
    cerr << "Unable to allocated a texture.  Increase MAX_TEXTURES "
	 << "in CudaCoulomb.cu.\n";
    abort();
  }
  MyTexture = iTex;
  textureInUse[iTex] = true;
}

TextureSpline::~TextureSpline()
{
  textureInUse[MyTexture] = false;
}


void
TextureSpline::set(double data[], int numPoints, 
		   double rmin, double rmax)
{
  rMin = rmin;
  rMax = rmax;
  NumPoints = numPoints;
  float data_Host[numPoints];
  for (int i=0; i<numPoints; i++)
    data_Host[i] = data[i];

  cudaChannelFormatDesc channelDesc = 
    cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  cudaMallocArray(&myArray, &channelDesc, numPoints);

  cudaMemcpyToArray(myArray, 0, 0, data_Host, numPoints*sizeof(float), 
		    cudaMemcpyHostToDevice);
  switch (MyTexture) {
  case 0:
    tex00.addressMode[0] = cudaAddressModeClamp;
    tex00.filterMode = cudaFilterModeLinear;
    tex00.normalized = false;
    cudaBindTextureToArray(tex00, myArray, channelDesc);
    break;
  case 1:
    tex01.addressMode[0] = cudaAddressModeClamp;
    tex01.filterMode = cudaFilterModeLinear;
    tex01.normalized = false;
    cudaBindTextureToArray(tex01, myArray, channelDesc);
    break;
  case 2:
    tex02.addressMode[0] = cudaAddressModeClamp;
    tex02.filterMode = cudaFilterModeLinear;
    tex02.normalized = false;
    cudaBindTextureToArray(tex02, myArray, channelDesc);
    break;
  case 3:
    tex03.addressMode[0] = cudaAddressModeClamp;
    tex03.filterMode = cudaFilterModeLinear;
    tex03.normalized = false;
    cudaBindTextureToArray(tex03, myArray, channelDesc);
    break;
  case 4:
    tex04.addressMode[0] = cudaAddressModeClamp;
    tex04.filterMode = cudaFilterModeLinear;
    tex04.normalized = false;
    cudaBindTextureToArray(tex04, myArray, channelDesc);
    break;
  case 5:
    tex05.addressMode[0] = cudaAddressModeClamp;
    tex05.filterMode = cudaFilterModeLinear;
    tex05.normalized = false;
    cudaBindTextureToArray(tex05, myArray, channelDesc);
    break;
  case 6:
    tex06.addressMode[0] = cudaAddressModeClamp;
    tex06.filterMode = cudaFilterModeLinear;
    tex06.normalized = false;
    cudaBindTextureToArray(tex06, myArray, channelDesc);
    break;
  case 7:
    tex07.addressMode[0] = cudaAddressModeClamp;
    tex07.filterMode = cudaFilterModeLinear;
    tex07.normalized = false;
    cudaBindTextureToArray(tex07, myArray, channelDesc);
    break;
  case 8:
    tex08.addressMode[0] = cudaAddressModeClamp;
    tex08.filterMode = cudaFilterModeLinear;
    tex08.normalized = false;
    cudaBindTextureToArray(tex08, myArray, channelDesc);
    break;
  case 9:
    tex09.addressMode[0] = cudaAddressModeClamp;
    tex09.filterMode = cudaFilterModeLinear;
    tex09.normalized = false;
    cudaBindTextureToArray(tex09, myArray, channelDesc);
    break;
  }

}


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
		  int textureNum, T lattice[], T latticeInv[], T sum[])
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
      if ((3*b+i)*BS + tid < 3*N)
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
	if (ptcl1 != ptcl2) {
	  float tval;
	  arraytexFetch(nrm*dist+0.5, textureNum, tval);
	  mysum += tval/dist;
	}
	//	  mysum += dist;
      }
    }
  }
  // Avoid double-counting on the diagonal blocks
  mysum *= 0.5;

  // Now do off-diagonal blocks
  for (int b1=0; b1<NB; b1++) {
    for (int i=0; i<3; i++)
      if ((3*b1+i)*BS + tid < 3*N)
	r1[0][i*BS+tid] = myR[(3*b1+i)*BS + tid];
    int ptcl1 = b1*BS + tid;
    if (ptcl1 < N) {
      for (int b2=b1+1; b2<NB; b2++) {
	for (int i=0; i<3; i++)
	  if ((3*b2+i)*BS + tid < 3*N)
	    r2[0][i*BS+tid] = myR[(3*b2+i)*BS + tid];
	int end = ((b2+1)*BS < N) ? BS : (N-b2*BS);
	for (int j=0; j<end; j++) {
	  T dx, dy, dz;
	  dx = r2[j][0] - r1[tid][0];
	  dy = r2[j][1] - r1[tid][1];
	  dz = r2[j][2] - r1[tid][2];
	  T dist = min_dist(dx, dy, dz, L, Linv);
	  float tval;
	  arraytexFetch(nrm*dist+0.5, textureNum, tval);
	  mysum += tval/dist;
	  //	  mysum += tex1D(shortTex[textureNum], nrm*dist+0.5)/dist;
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
    sum[blockIdx.x] = shared_sum[0];
}


void
CoulombAA_SR_Sum(float *R[], int N, float rMax, int Ntex,
		 int textureNum, float lattice[], float latticeInv[], 
		 float sum[], int numWalkers)
{
  const int BS=32;
  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  coulomb_AA_kernel<float,BS><<<dimGrid,dimBlock>>>
    (R, N, rMax, Ntex, textureNum, lattice, latticeInv, sum);
}



template<typename T, int BS>
__global__ void
coulomb_AB_kernel(T *R[], int Nelec, T I[], int Ifirst, int Ilast, 
		  T rMax, int Ntex, int textureNum, 
		  T lattice[], T latticeInv[], T sum[])
{
  int tid = threadIdx.x;
  __shared__ T *myR;

  int Nion = Ilast - Ifirst + 1;

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
  __shared__ float r[BS][3], i[BS][3];
  int NeBlocks = Nelec/BS + ((Nelec%BS) ? 1 : 0);
  int NiBlocks = Nion/BS +  ((Nion %BS) ? 1 : 0);

  T mysum = (T)0.0; 

  // Now do off-diagonal blocks
  for (int iBlock=0; iBlock<NiBlocks; iBlock++) {
    for (int j=0; j<3; j++)
      if ((3*iBlock+j)*BS + tid < 3*Nion)
	i[0][j*BS+tid] = I[3*Ifirst+(3*iBlock+j)*BS + tid];
    __syncthreads();
    int ion = iBlock*BS + tid;
    for (int eBlock=0; eBlock<NeBlocks; eBlock++) {
      for (int j=0; j<3; j++)
	if ((3*eBlock+j)*BS + tid < 3*Nelec)
	  r[0][j*BS+tid] = myR[(3*eBlock+j)*BS + tid];
      __syncthreads();
      int end = ((eBlock+1)*BS < Nelec) ? BS : (Nelec-eBlock*BS);
      if (ion < Nion) {
	for (int j=0; j<end; j++) {
	  T dx, dy, dz;
	  dx = r[j][0] - i[tid][0];
	  dy = r[j][1] - i[tid][1];
	  dz = r[j][2] - i[tid][2];
	  T dist = min_dist(dx, dy, dz, L, Linv);
	  float tval;
	  arraytexFetch(nrm*dist+0.5, textureNum, tval);
	  mysum += tval / dist;
	}
      }
      __syncthreads();
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
    sum[blockIdx.x] = shared_sum[0];
}


void
CoulombAB_SR_Sum(float *R[], int Nelec, float I[],  int Ifirst, int Ilast,
		 float rMax, int Ntex, int textureNum, 
		 float lattice[], float latticeInv[], 
		 float sum[], int numWalkers)
{
  const int BS=64;
  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  coulomb_AB_kernel<float,BS><<<dimGrid,dimBlock>>>
    (R, Nelec, I, Ifirst, Ilast, rMax, Ntex, textureNum, 
     lattice, latticeInv, sum);
}


template<typename T, int BS>
__global__ void
eval_rhok_kernel (T *R[], int numr,
		  T kpoints[], int numk, T* rhok[])
{
  int tid = threadIdx.x;
  __shared__ float r[BS][3], k[BS][3], *myR, *myrhok;
  if (tid == 0) {
    myR    =    R[blockIdx.x];
    myrhok = rhok[blockIdx.x];
  }
  __syncthreads();
  
  int NrBlock = numr/BS + ((numr%BS) ? 1 : 0);
  int NkBlock = numk/BS + ((numk%BS) ? 1 : 0);
  __shared__ float rhok_re[BS], rhok_im[BS], rhok_s[2*BS];
  
  for (int kBlock=0; kBlock<NkBlock; kBlock++) {
    for (int i=0; i<3; i++)
      if ((i+3*kBlock)*BS + tid < 3*numk)
	k[0][BS*i+tid] = kpoints[(i+3*kBlock)*BS+tid];
    rhok_re[tid] = rhok_im[tid] = 0.0f;
    for (int rBlock=0; rBlock<NrBlock; rBlock++) {
      for (int i=0; i<3; i++)
	if ((i+3*rBlock)*BS + tid < 3*numr)
	  r[0][BS*i+tid] = myR[(i+3*rBlock)*BS+tid];
      int end = ((rBlock+1)*BS < numr) ? BS : (numr-rBlock*BS);
      for (int j=0; j<end; j++) {
	T phase = (k[tid][0] * r[j][0] +
		   k[tid][1] * r[j][1] + 
		   k[tid][2] * r[j][2]);
	float s,c;
	__sincosf (phase, &s, &c);
	rhok_im[tid] += s;
	rhok_re[tid] += c;
      }
    }
    // Write rhok to global memory
    rhok_s[2*tid+0] = rhok_re[tid];
    rhok_s[2*tid+1] = rhok_im[tid];
    __syncthreads();
    if (2*(kBlock*BS)+tid < 2*numk)
      myrhok[2*(kBlock*BS)+tid] = rhok_s[tid];
    if (2*(kBlock*BS)+tid+BS < 2*numk)
      myrhok[2*(kBlock*BS)+tid+BS] = rhok_s[tid+BS];
  }
}


template<typename T, int BS>
__global__ void
eval_rhok_kernel (T *R[], int first, int last,
		  T kpoints[], int numk, T* rhok[])
{
  int tid = threadIdx.x;
  int numr = last-first+1;
  __shared__ float r[BS][3], k[BS][3], *myR, *myrhok;
  if (tid == 0) {
    myR    =    R[blockIdx.x];
    myrhok = rhok[blockIdx.x];
  }
  __syncthreads();
  
  int NrBlock = numr/BS + ((numr%BS) ? 1 : 0);
  int NkBlock = numk/BS + ((numk%BS) ? 1 : 0);
  __shared__ float rhok_re[BS], rhok_im[BS], rhok_s[2*BS];
  
  for (int kBlock=0; kBlock<NkBlock; kBlock++) {
    for (int i=0; i<3; i++)
      if ((i+3*kBlock)*BS + tid < 3*numk)
	k[0][BS*i+tid] = kpoints[(i+3*kBlock)*BS+tid];
    rhok_re[tid] = rhok_im[tid] = 0.0f;
    for (int rBlock=0; rBlock<NrBlock; rBlock++) {
      for (int i=0; i<3; i++)
	if ((i+3*rBlock)*BS + tid < 3*numr)
	  r[0][BS*i+tid] = myR[3*first+(i+3*rBlock)*BS+tid];
      int end = ((rBlock+1)*BS < numr) ? BS : (numr-rBlock*BS);
      for (int j=0; j<end; j++) {
	T phase = (k[tid][0] * r[j][0] +
		   k[tid][1] * r[j][1] + 
		   k[tid][2] * r[j][2]);
	float s,c;
	__sincosf (phase, &s, &c);
	rhok_im[tid] += s;
	rhok_re[tid] += c;
      }
    }
    // Write rhok to global memory
    rhok_s[2*tid+0] = rhok_re[tid];
    rhok_s[2*tid+1] = rhok_im[tid];
    __syncthreads();
    if (2*(kBlock*BS)+tid < 2*numk)
      myrhok[2*(kBlock*BS)+tid] = rhok_s[tid];
    if (2*(kBlock*BS)+tid+BS < 2*numk)
      myrhok[2*(kBlock*BS)+tid+BS] = rhok_s[tid+BS];
  }
}



void
eval_rhok_cuda(float *R[], int numr, float kpoints[], 
	       int numk, float* rhok[], int numWalkers)
{
  const int BS=32;

  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  eval_rhok_kernel<float,BS><<<dimGrid,dimBlock>>>
    (R, numr, kpoints, numk, rhok);
}


void
eval_rhok_cuda(float *R[], int first, int last, float kpoints[], 
	       int numk, float* rhok[], int numWalkers)
{
  const int BS=32;

  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  eval_rhok_kernel<float,BS><<<dimGrid,dimBlock>>>
    (R, first, last, kpoints, numk, rhok);
}


template<typename T, int BS>
__global__ void
vk_sum_kernel(T *rhok[], T vk[], int numk,
	      T sum[])
{
  int tid = threadIdx.x;
  __shared__ float *myrhok;
  if (tid == 0) 
    myrhok = rhok[blockIdx.x];

  __syncthreads();
  
  // Used to do coalesced global loads
  __shared__ T rhok_s[2*BS];

  int NB = numk/BS + ((numk%BS) ? 1 : 0);
  T mysum = 0.0f;
  for (int b=0; b<NB; b++) {
    if (2*b*BS + tid < 2*numk)
      rhok_s[tid] = myrhok[2*b*BS+tid];
    if ((2*b+1)*BS + tid < 2*numk)
      rhok_s[BS+tid] = myrhok[(2*b+1)*BS+tid];
    __syncthreads();

    if (b*BS + tid < numk) 
      mysum += vk[b*BS+tid] * (rhok_s[2*tid+0]*rhok_s[2*tid+0] +
			       rhok_s[2*tid+1]*rhok_s[2*tid+1]);
  }
  
  __shared__ T shared_sum[BS];
  shared_sum[tid] = mysum;
  __syncthreads();
  for (int s=(BS>>1); s>0; s >>= 1) {
    if (tid < s) 
      shared_sum[tid] += shared_sum[tid+s];
    __syncthreads();
  }

  // Not sure if this 0.25 factor is correct.
  if (tid == 0)
    sum[blockIdx.x] += 0.25*shared_sum[0];
}


template<typename T, int BS>
__global__ void
vk_sum_kernel2(T *rhok1[], T *rhok2[], T vk[], int numk,
	      T sum[])
{
  int tid = threadIdx.x;
  __shared__ float *myrhok1, *myrhok2;
  if (tid == 0) {
    myrhok1 = rhok1[blockIdx.x];
    myrhok2 = rhok2[blockIdx.x];
  }
  __syncthreads();
  
  // Used to do coalesced global loads
  __shared__ T rhok_s1[2*BS], rhok_s2[2*BS];

  int NB = numk/BS + ((numk%BS) ? 1 : 0);
  T mysum = 0.0f;
  for (int b=0; b<NB; b++) {
    if (2*b*BS + tid < 2*numk) {
      rhok_s1[tid] = myrhok1[2*b*BS+tid];
      rhok_s2[tid] = myrhok2[2*b*BS+tid];
    }
    if ((2*b+1)*BS + tid < 2*numk) {
      rhok_s1[BS+tid] = myrhok1[(2*b+1)*BS+tid]; 
      rhok_s2[BS+tid] = myrhok2[(2*b+1)*BS+tid]; 
    }
    __syncthreads();

    if (b*BS + tid < numk) 
      mysum += vk[b*BS+tid] * (rhok_s1[2*tid+0]*rhok_s2[2*tid+0] +
			       rhok_s1[2*tid+1]*rhok_s2[2*tid+1]);
  }
  
  __shared__ T shared_sum[BS];
  shared_sum[tid] = mysum;
  __syncthreads();
  for (int s=(BS>>1); s>0; s >>= 1) {
    if (tid < s) 
      shared_sum[tid] += shared_sum[tid+s];
    __syncthreads();
  }

  T factor = (myrhok1 == myrhok2) ? 0.5f : 1.0f;

  if (tid == 0)
    sum[blockIdx.x] += factor*shared_sum[0];
}




void
eval_vk_sum_cuda (float *rhok[], float vk[], int numk, float sum[],
		  int numWalkers)
{
  const int BS=64;

  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  vk_sum_kernel<float,BS><<<dimGrid,dimBlock>>>
    (rhok, vk, numk, sum);
}

void
eval_vk_sum_cuda (float *rhok1[], float *rhok2[], 
		  float vk[], int numk, float sum[],
		  int numWalkers)
{
  const int BS=64;

  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  vk_sum_kernel2<float,BS><<<dimGrid,dimBlock>>>
    (rhok1, rhok2, vk, numk, sum);
}


template<typename T, int BS>
__global__ void
vk_sum_kernel2(T *rhok1[], T rhok2[], T vk[], int numk,
	       T sum[])
{
  int tid = threadIdx.x;
  __shared__ float *myrhok1;
  if (tid == 0) 
    myrhok1 = rhok1[blockIdx.x];
  __syncthreads();
  
  // Used to do coalesced global loads
  __shared__ T rhok_s1[2*BS], rhok_s2[2*BS];

  int NB = numk/BS + ((numk%BS) ? 1 : 0);
  T mysum = 0.0f;
  for (int b=0; b<NB; b++) {
    if (2*b*BS + tid < 2*numk) {
      rhok_s1[tid] = myrhok1[2*b*BS+tid];
      rhok_s2[tid] =   rhok2[2*b*BS+tid];
    }
    if ((2*b+1)*BS + tid < 2*numk) {
      rhok_s1[BS+tid] = myrhok1[(2*b+1)*BS+tid]; 
      rhok_s2[BS+tid] =   rhok2[(2*b+1)*BS+tid]; 
    }
    __syncthreads();

    if (b*BS + tid < numk) 
      mysum += vk[b*BS+tid] * (rhok_s1[2*tid+0]*rhok_s2[2*tid+0] +
			       rhok_s1[2*tid+1]*rhok_s2[2*tid+1]);
  }
  
  __shared__ T shared_sum[BS];
  shared_sum[tid] = mysum;
  __syncthreads();
  for (int s=(BS>>1); s>0; s >>= 1) {
    if (tid < s) 
      shared_sum[tid] += shared_sum[tid+s];
    __syncthreads();
  }

  if (tid == 0)
    sum[blockIdx.x] += shared_sum[0];
}

void
eval_vk_sum_cuda (float *rhok1[], float rhok2[], 
		  float vk[], int numk, float sum[],
		  int numWalkers)
{
  const int BS=64;

  dim3 dimBlock(BS);
  dim3 dimGrid(numWalkers);

  vk_sum_kernel2<float,BS><<<dimGrid,dimBlock>>>
    (rhok1, rhok2, vk, numk, sum);
}



#ifdef CUDA_COULOMB_TEST


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
#endif
