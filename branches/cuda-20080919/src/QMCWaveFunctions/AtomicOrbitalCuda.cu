#include <cstdio>
#include <vector>
#include <complex>
#include "AtomicOrbitalCuda.h"
#include <einspline/multi_bspline.h>
#include <einspline/multi_bspline_create_cuda.h>

using namespace std;
__constant__ float  Dcuda[48];

bool atomic_cuda_initialized = false;

void
init_atomic_cuda()
{
  fprintf (stderr, "Initializing B-spline matrix.\n");
  if (atomic_cuda_initialized)
    return;

  atomic_cuda_initialized = true;

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

  cudaMemcpyToSymbol(Dcuda, A_h, 48*sizeof(float), 0, cudaMemcpyHostToDevice);
}
  


template<typename T, int BS> __global__ void
MakeHybridJobList_kernel (T* elec_list, int num_elecs, T* ion_list, 
			  T* poly_radii, T* spline_radii,
			  int num_ions, T *L, T *Linv,
			  HybridJobType *job_list, T *rhat_list,
			  HybridDataFloat *data_list)
{
  __shared__ T epos[BS][3], ipos[BS][3], L_s[3][3], Linv_s[3][3];
  __shared__ T rhat[BS][3];
  __shared__ T r_spline[BS], r_poly[BS];
  __shared__ HybridJobType jobs_shared[BS];
  int tid = threadIdx.x;
  
  if (tid < 9) {
    L_s[0][tid]    = L[tid];
    Linv_s[0][tid] = Linv[tid];
  }

  // Load electron positions
  for (int i=0; i<3; i++)
    if ((3*blockIdx.x+i)*BS + tid < 3*num_elecs)
      epos[0][i*BS+tid] = elec_list[(3*blockIdx.x+i)*BS + tid];
  
  int iBlocks = (num_ions+BS-1)/BS;
  jobs_shared[tid] = BSPLINE_3D_JOB;
  __shared__ HybridDataFloat data[BS];
  for (int ib=0; ib<iBlocks; ib++) {
    // Fetch ion positions into shared memory
    for (int j=0; j<3; j++) {
      int off = (3*ib+j)*BS+tid;
      if (off < 3*num_ions)
	ipos[0][j*BS+tid] = ion_list[off];
    }
    // Fetch radii into shared memory
    if (ib*BS+tid < num_ions) {
      r_spline[tid] = spline_radii[ib*BS+tid];
      r_poly[tid]   = poly_radii  [ib*BS+tid];
    }
    __syncthreads();
    int iend = min (BS, num_ions - ib*BS);
    T dr0, dr1, dr2, u0, u1, u2, img0, img1, img2;
    rhat[tid][0] = 1.0f;
    rhat[tid][1] = 0.0f;
    rhat[tid][2] = 0.0f;
    for (int ion=0; ion<iend; ion++) {
      // Find mininum image displacement
      dr0 = epos[tid][0] - ipos[ion][0];
      dr1 = epos[tid][1] - ipos[ion][1];
      dr2 = epos[tid][2] - ipos[ion][2];
      u0 = Linv_s[0][0]*dr0 + Linv_s[0][1]*dr1 + Linv_s[0][2]*dr2;
      u1 = Linv_s[1][0]*dr0 + Linv_s[1][1]*dr1 + Linv_s[1][2]*dr2;
      u2 = Linv_s[2][0]*dr0 + Linv_s[2][1]*dr1 + Linv_s[2][2]*dr2;
      img0 = rintf(u0);  img1 = rintf(u1);  img2 = rintf(u2);
      u0  -= img0;       u1  -= img1;       u2  -= img2;
      dr0 = L_s[0][0]*u0 + L_s[0][1]*u1 + L_s[0][2]*u2;
      dr1 = L_s[1][0]*u0 + L_s[1][1]*u1 + L_s[1][2]*u2;
      dr2 = L_s[2][0]*u0 + L_s[2][1]*u1 + L_s[2][2]*u2;
      
      T dist2 = dr0*dr0 + dr1*dr1 + dr2*dr2;
      T dist = sqrtf(dist2);
      
      // Compare with radii
      if (dist < r_poly[ion]) 
	jobs_shared[tid] =  ATOMIC_POLY_JOB;
      else if (dist < r_spline[ion]) 
	jobs_shared[tid] =  ATOMIC_SPLINE_JOB;
      // Compute rhat
      if (dist < r_spline[ion]) {
	data[tid].dist = dist;
	data[tid].img[0] = img0;
	data[tid].img[1] = img1;
	data[tid].img[2] = img2;
	data[tid].ion = ion;
	dist = 1.0f/dist;
	rhat[tid][0] =  dr0 * dist;
	rhat[tid][1] =  dr1 * dist;
	rhat[tid][2] =  dr2 * dist;
      }
    } 
  }
  __syncthreads();
  // Now write rhats and job types to global memory
  for (int i=0; i<3; i++) {
    int off = (3*blockIdx.x+i)*BS + tid;
    if (off < 3*num_elecs)
      rhat_list[off] = rhat[0][i*BS+tid];
  }
  if (blockIdx.x*BS+tid < num_elecs)
    job_list[blockIdx.x*BS+tid] = jobs_shared[tid];

  const int m = sizeof(HybridDataFloat)/sizeof(float);
  float *data_f = (float*)data_list;
  for (int i=0; i<m; i++) {
    int off = (blockIdx.x*BS+i)*m + tid;
    if (off < m*num_elecs)
      data_f[off] = ((float*)data)[i*m+tid];
  }
}


void
MakeHybridJobList (float* elec_list, int num_elecs, float* ion_list, 
		   float* poly_radii, float* spline_radii,
		   int num_ions, float *L, float *Linv,
		   HybridJobType *job_list, float *rhat_list,
		   HybridDataFloat *data_list)
{ 
  const int BS=32;
  int numBlocks = (num_elecs+BS-1)/BS;
  dim3 dimGrid(numBlocks);
  dim3 dimBlock(BS);

  MakeHybridJobList_kernel<float,BS><<<dimGrid,dimBlock>>> (elec_list, num_elecs,
							    ion_list, poly_radii, spline_radii,
							    num_ions, L, Linv, job_list, rhat_list,
							    data_list);
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in MakeHybridJobList:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}


// The spline coefficients should be reordered so that the
// orbital is the fastest index.

//////////////////////////////////////////////////////////
// Need to add k-points and use image to determine sign //
// for anti-periodic boundary conditions!!!!!!          //
//////////////////////////////////////////////////////////

template< typename T, int BS, int LMAX>__global__ void
evaluateHybridSplineReal_kernel (HybridJobType *job_types, 
				 T **YlmReal, AtomicOrbitalCuda<T> *orbitals,
				 HybridDataFloat *data, T **vals, int N)
{
  int tid = threadIdx.x;
  __shared__ HybridJobType myjob;
  if (tid == 0) myjob = job_types[blockIdx.x];
  __syncthreads();
  if (myjob == BSPLINE_3D_JOB)
    return;
  
  __shared__ T *myYlm, *myCoefs, *myVal;
  __shared__ HybridDataFloat myData;
  __shared__ AtomicOrbitalCuda<T> myOrbital;
  const int data_size = (sizeof(HybridDataFloat)     +3)/sizeof(float);
  const int orb_size  = (sizeof(AtomicOrbitalCuda<T>)+3)/sizeof(float);
  if (tid < data_size) 
    ((float*)&myData)[tid]    = ((float*)&(data[blockIdx.x]))[tid];
  if (tid < orb_size) 
    ((float*)&myOrbital)[tid] = ((float*)(&orbitals[myData.ion]))[tid];

  if (tid == 0) {
    myYlm   = YlmReal[blockIdx.x];
    myVal   = vals[blockIdx.x];
    myCoefs = myOrbital.spline_coefs;
  }
  __syncthreads();
  

  // Compute spline basis functions
  T unit = myData.dist * myOrbital.spline_dr_inv;
  T sf = floor(unit);
  T t  = unit - sf;
  int index= (int) sf;
  float4 tp;
  tp = make_float4(t*t*t, t*t, t, 1.0f);
  __shared__ float a[4];
  if (tid < 4) 
    a[tid] = Dcuda[4*tid+0]*tp.x + Dcuda[4*tid+1]*tp.y + Dcuda[4*tid+2]*tp.z + Dcuda[4*tid+3]*tp.w;
  __syncthreads();


  __shared__ T Ylm[(LMAX+1)*(LMAX+1)];
  int numlm = (myOrbital.lMax+1)*(myOrbital.lMax+1);
  int Yblocks = (numlm+BS-1)/BS;
  for (int ib=0; ib<Yblocks; ib++)
    if (ib*BS + tid < numlm)
      Ylm[ib*BS+tid] = myYlm[ib*BS + tid];
  __syncthreads();

  int numBlocks = (N+BS-1)/BS;
  int ustride  = 1*myOrbital.spline_stride;
  int ustride2 = 2*myOrbital.spline_stride;
  int ustride3 = 3*myOrbital.spline_stride;
  for (int block=0; block<numBlocks; block++) {
    T *c0 =  myCoefs + index*ustride + block*BS + tid;
    T val = T();
    for (int lm=0; lm<numlm; lm++) {
      float *c = c0 + lm*myOrbital.lm_stride;
      float u = (a[0] * c[0] + 
		 a[1] * c[ustride] + 
		 a[2] * c[ustride2] + 
		 a[3] * c[ustride3]);
      val +=  u * Ylm[lm];
    }
    int off = block*BS + tid;
    if (off < N)
      myVal[off] = val;
  }

  __syncthreads();
  
}

void
evaluateHybridSplineReal (HybridJobType *job_types, 
			  float **Ylm_real, AtomicOrbitalCuda<float> *orbitals,
			  HybridDataFloat *data,
			  float **vals, int N, int numWalkers, int lMax)
{
  const int BS=32;
  dim3 dimGrid(numWalkers);
  dim3 dimBlock(BS);
  
  if (lMax == 0) 
    evaluateHybridSplineReal_kernel<float,BS,0><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 1) 
    evaluateHybridSplineReal_kernel<float,BS,1><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 2) 
    evaluateHybridSplineReal_kernel<float,BS,2><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 3) 
    evaluateHybridSplineReal_kernel<float,BS,3><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 4) 
    evaluateHybridSplineReal_kernel<float,BS,4><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 5) 
    evaluateHybridSplineReal_kernel<float,BS,5><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 6) 
    evaluateHybridSplineReal_kernel<float,BS,6><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 7) 
    evaluateHybridSplineReal_kernel<float,BS,7><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 8) 
    evaluateHybridSplineReal_kernel<float,BS,8><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);
  else if (lMax == 9) 
    evaluateHybridSplineReal_kernel<float,BS,9><<<dimGrid,dimBlock>>> 
      (job_types, Ylm_real, orbitals, data, vals, N);

     
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in evaluateHybridSplineReal:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}


template< typename T, int BS, int LMAX>__global__ void
evaluateHybridSplineReal_kernel (HybridJobType *job_types, T* rhats, 
				 T **YlmReal, float **dYlm_dTheta, float **dYlm_dphi,
				 AtomicOrbitalCuda<T> *orbitals, HybridDataFloat *data, 
				 T **vals, T **grad_lapl, int row_stride, int N)
{
  int tid = threadIdx.x;
  __shared__ HybridJobType myjob;
  if (tid == 0) myjob = job_types[blockIdx.x];
  __syncthreads();
  if (myjob == BSPLINE_3D_JOB)
    return;
  
  __shared__ T *myYlm, *mydTheta, *mydPhi, *myCoefs, *myVal, *myGradLapl;
  __shared__ HybridDataFloat myData;
  __shared__ AtomicOrbitalCuda<T> myOrbital;
  const int data_size = (sizeof(HybridDataFloat)     +3)/sizeof(float);
  const int orb_size  = (sizeof(AtomicOrbitalCuda<T>)+3)/sizeof(float);
  if (tid < data_size) 
    ((float*)&myData)[tid]    = ((float*)&(data[blockIdx.x]))[tid];
  if (tid < orb_size) 
    ((float*)&myOrbital)[tid] = ((float*)(&orbitals[myData.ion]))[tid];

  if (tid == 0) {
    myYlm      = YlmReal[blockIdx.x];
    mydTheta   = dYlm_dTheta[blockIdx.x];
    mydPhi     = dYlm_dphi[blockIdx.x];
    myVal      = vals[blockIdx.x];
    myGradLapl = grad_lapl[blockIdx.x];
    myCoefs    = myOrbital.spline_coefs;
  }
  __shared__ T rhat[3], thetahat[3], phihat[3];
  __shared__ T sintheta, cosphi, sinphi, rInv, sinthetaInv;
  if (tid < 3) 
    rhat[tid] = rhats[3*blockIdx.x+tid];
  if (tid ==0) {
    rInv = 1.0f/myData.dist;
    sintheta = sqrtf(1.0-rhat[2]*rhat[2]);
    sinthetaInv = 1.0/sintheta;
    cosphi = rhat[0]*sinthetaInv;
    sinphi = rhat[1]*sinthetaInv;
    thetahat[0] = rhat[2]*cosphi;
    thetahat[1] = rhat[2]*sinphi;
    thetahat[2] = -sintheta;
    phihat[0]   = -sinphi;
    phihat[1]   = cosphi;
    phihat[2]   = 0.0f;
  }
  __syncthreads();

  // Compute spline basis functions
  T unit = myData.dist * myOrbital.spline_dr_inv;
  T sf = floor(unit);
  T t  = unit - sf;
  int index= (int) sf;
  float4 tp;
  tp = make_float4(t*t*t, t*t, t, 1.0f);
  __shared__ float a[12];
  if (tid < 12) 
    a[tid] = Dcuda[4*tid+0]*tp.x + Dcuda[4*tid+1]*tp.y + Dcuda[4*tid+2]*tp.z + Dcuda[4*tid+3]*tp.w;
  __syncthreads();


  __shared__ T Ylm[(LMAX+1)*(LMAX+1)], dTheta[(LMAX+1)*(LMAX+1)], dPhi[(LMAX+1)*(LMAX+1)],
    lpref[(LMAX+1)*(LMAX+1)];
  int numlm = (myOrbital.lMax+1)*(myOrbital.lMax+1);
  int Yblocks = (numlm+BS-1)/BS;
  for (int ib=0; ib<Yblocks; ib++)
    if (ib*BS + tid < numlm) {
      Ylm[ib*BS+tid]    = myYlm[ib*BS + tid];
      dTheta[ib*BS+tid] = mydTheta[ib*BS + tid];
      dPhi[ib*BS+tid]   = mydPhi[ib*BS + tid];
    }
  for (int l=0; l<=myOrbital.lMax; l++) 
    if (tid < 2*l+1) {
      int lm = l*l + tid;
      lpref[lm] = -rInv*rInv*(T)(l*(l+1));
    }
  __syncthreads();

  int numBlocks = (N+BS-1)/BS;
  int ustride  = 1*myOrbital.spline_stride;
  int ustride2 = 2*myOrbital.spline_stride;
  int ustride3 = 3*myOrbital.spline_stride;
  for (int block=0; block<numBlocks; block++) {
    T *c0 =  myCoefs + index*ustride + block*BS + tid;
    T val = T();
    T g_rhat=T(), g_thetahat=T(), g_phihat=T(), lap=T();
    for (int lm=0; lm<numlm; lm++) {
      float *c = c0 + lm*myOrbital.lm_stride;
      float u, du, d2u, coef;
      coef = c[0];
      u  = a[0]*coef;  du  = a[4]*coef;  d2u  = a[8]*coef;
      coef = c[ustride];
      u += a[1]*coef;  du += a[5]*coef;  d2u += a[9]*coef;
      coef = c[ustride2];
      u += a[2]*coef;  du += a[6]*coef;  d2u += a[10]*coef;
      coef = c[ustride3];
      u += a[3]*coef;  du += a[7]*coef;  d2u += a[11]*coef;
      du  *= myOrbital.spline_dr_inv;
      d2u *= myOrbital.spline_dr_inv * myOrbital.spline_dr_inv;
      val    +=   u * Ylm[lm];
      g_rhat +=  du*Ylm[lm];
      g_thetahat += u*rInv*dTheta[lm];
      g_phihat   += u*rInv*sinthetaInv*dPhi[lm];
      lap += Ylm[lm] * (lpref[lm] * u + d2u + 2.0*rInv*du);
    }
    int off = block*BS + tid;
    if (off < N) {
      myVal[off] = val;
      myGradLapl[3*row_stride+off] = lap;
    }
  }

  __syncthreads();
  
}

void
evaluateHybridSplineReal (HybridJobType *job_types, float *rhats,
			  float **Ylm_real, float **dYlm_dTheta, float **dYlm_dphi,
			  AtomicOrbitalCuda<float> *orbitals,
			  HybridDataFloat *data,
			  float **vals, float **grad_lapl,
			  int row_stride, int N, int numWalkers, int lMax)
{
  const int BS=32;
  dim3 dimGrid(numWalkers);
  dim3 dimBlock(BS);
  
  if (lMax == 0) 
    evaluateHybridSplineReal_kernel<float,BS,0><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 1) 
    evaluateHybridSplineReal_kernel<float,BS,1><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 2) 
    evaluateHybridSplineReal_kernel<float,BS,2><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 3) 
    evaluateHybridSplineReal_kernel<float,BS,3><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 4) 
    evaluateHybridSplineReal_kernel<float,BS,4><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 5) 
    evaluateHybridSplineReal_kernel<float,BS,5><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 6) 
    evaluateHybridSplineReal_kernel<float,BS,6><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 7) 
    evaluateHybridSplineReal_kernel<float,BS,7><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 8) 
    evaluateHybridSplineReal_kernel<float,BS,8><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);
  else if (lMax == 9) 
    evaluateHybridSplineReal_kernel<float,BS,9><<<dimGrid,dimBlock>>> 
      (job_types, rhats, Ylm_real, dYlm_dTheta, dYlm_dphi, orbitals, 
       data, vals, grad_lapl, row_stride, N);

     
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in evaluateHybridSplineReal:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}




template<typename T, int LMAX, int BS> __global__ void
CalcYlmComplex (T *rhats, HybridJobType  *job_types,
		T **Ylm_ptr, T **dYlm_dtheta_ptr, T **dYlm_dphi_ptr, int N)
{
  HybridJobType jt = job_types[blockIdx.x];
  if (jt == BSPLINE_3D_JOB)   return;

  const T fourPiInv = 0.0795774715459477f;
  int tid = threadIdx.x;
  const int numlm = (LMAX+1)*(LMAX+1);

  __shared__ T* Ylm[BS], *dtheta[BS], *dphi[BS];
  if (blockIdx.x*BS+tid < N) {
    Ylm[tid]    = Ylm_ptr[blockIdx.x*BS+tid];
    dtheta[tid] = dYlm_dtheta_ptr[blockIdx.x*BS+tid];
    dphi[tid]   = dYlm_dphi_ptr[blockIdx.x*BS+tid];
  }

  __shared__ T rhat[BS][3];
  for (int i=0; i<3; i++) {
    int off = (3*blockIdx.x + i)*BS + tid;
    if (off < 3*N)
      rhat[0][i*BS+tid] = rhats[off];
  }
  __syncthreads();

  T costheta = rhat[tid][2];
  T sintheta = sqrt(1.0f-costheta*costheta);
  T cottheta = costheta/sintheta;
  
  T cosphi, sinphi;
  cosphi=rhat[tid][0]/sintheta;
  sinphi=rhat[tid][1]/sintheta;
  __shared__ T phi[BS];
  phi[tid] = atan2f(sinphi, cosphi);

  __shared__ T XlmVec[BS][(LMAX+1)*(LMAX+2)/2], 
    dXlmVec[BS][(LMAX+1)*(LMAX+2)/2];
  
  // Create a map from lm linear index to l and m
  __shared__ int l_lm[numlm];
  __shared__ int m_lm[numlm];
  __shared__ T  floatm_lm[numlm];
  int off=0;
  for (int l=0; l<=LMAX; l++) {
    if (tid < 2*l+1) {
      l_lm[off+tid] = l;
      m_lm[off+tid] = tid-l;
      floatm_lm[off+tid] = (T)(tid-l);
    }
    off += 2*l+1;
  }

  T lsign = 1.0f;
  T dl = 0.0f;
  for (int l=0; l<=LMAX; l++) {
    int index=l*(l+3)/2;
    XlmVec[tid][index]  = lsign;  
    dXlmVec[tid][index] = dl * cottheta * XlmVec[tid][index];
    T dm = dl;
    for (int m=l; m>0; m--, index--) {
      T tmp = sqrt((dl+dm)*(dl-dm+1.0f));
      XlmVec[tid][index-1]  = 
	-(dXlmVec[tid][index] + dm*cottheta*XlmVec[tid][index])/ tmp;
      dXlmVec[tid][index-1] = 
	(dm-1.0f)*cottheta*XlmVec[tid][index-1] + XlmVec[tid][index]*tmp;
      dm -= 1.0f;
    }
    index = l*(l+1)/2;
    T sum = XlmVec[tid][index] * XlmVec[tid][index];
    for (int m=1; m<=l; m++) 
      sum += 2.0f*XlmVec[tid][index+m]*XlmVec[tid][index+m];
    // Now, renormalize the Ylms for this l
    T norm = sqrt((2.0f*dl+1.0f)*fourPiInv / sum);
    for (int m=0; m<=l; m++) {
      XlmVec[tid][index+m]  *= norm;
      dXlmVec[tid][index+m] *= norm;
    }
    lsign *= -1.0f;
    dl += 1.0f;
  }
  __syncthreads();

  // Multiply by azimuthal phase and store in Ylm
  int end = min (N-blockIdx.x*BS, BS);
  int nb = ((LMAX+1)*(LMAX+1)+BS-1)/BS;
  __shared__ T outbuff[3][BS][2];
  for (int i=0; i < end; i++) {
    // __shared__ T sincosphi[2*LMAX+1][2];
    // if (tid < LMAX
    for (int block=0; block<nb; block++) {
      int lm = block*BS + tid;
      if (lm < numlm) {
	int l = l_lm[lm];
	int m = m_lm[lm];
	T fm = floatm_lm[lm];
	T re, im;
	__sincosf(fm*phi[i], &im, &re);
	// Switch sign if m<0 and it's odd
	if (m<0 && (m&1)) {
	  re *= -1.0f;
	  im *= -1.0f;
	}
	int off = ((l*(l+1))>>1) + abs(m);
	// Ylm
	outbuff[0][tid][0] =     re *  XlmVec[i][off];
	outbuff[0][tid][1] =     im *  XlmVec[i][off];
	// dYlm_dtheta
	outbuff[1][tid][0] =     re * dXlmVec[i][off];
	outbuff[1][tid][1] =     im * dXlmVec[i][off];
	// dYlm_dphi
	outbuff[2][tid][0] = -fm*im *  XlmVec[i][off];
	outbuff[2][tid][1] =  fm*re *  XlmVec[i][off];
      }
      __syncthreads();
      // Now write back to global mem with coallesced writes
      int off = 2*block*BS + tid;
      if (off < 2*numlm) {
	Ylm[i][off]    = outbuff[0][0][tid];
	dtheta[i][off] = outbuff[1][0][tid];
	dphi[i][off]   = outbuff[2][0][tid];
      }
      off += BS;
      if (off < 2*numlm) {
	Ylm[i][off]    = outbuff[0][0][tid+BS];
	dtheta[i][off] = outbuff[1][0][tid+BS];
	dphi[i][off]   = outbuff[2][0][tid+BS];
      }
    }
  }



  // complex<T> e2imphi (1.0, 0.0);
  // complex<T> eye(0.0, 1.0);
  // for (int m=0; m<=l; m++) {
  //   Ylm[l*(l+1)+m]  =  XlmVec[tid][l+m]*e2imphi;
  //   Ylm[l*(l+1)-m]  =  XlmVec[tid][l-m]*conj(e2imphi);
  //   dYlm_dphi[l*(l+1)+m ]  =  (double)m * eye *XlmVec[tid][l+m]*e2imphi;
  //   dYlm_dphi[l*(l+1)-m ]  = -(double)m * eye *XlmVec[tid][l-m]*conj(e2imphi);
  //   dYlm_dtheta[l*(l+1)+m] = dXlmVec[tid][l+m]*e2imphi;
  //   dYlm_dtheta[l*(l+1)-m] = dXlmVec[tid][l-m]*conj(e2imphi);
  //   e2imphi *= e2iphi;
  // } 
  
  // dl += 1.0;
  // lsign *= -1.0;
  // YlmTimer.stop();
}


template<typename T, int LMAX, int BS> __global__ void
CalcYlmReal (T *rhats, HybridJobType* job_type,
	     T **Ylm_ptr, T **dYlm_dtheta_ptr, T **dYlm_dphi_ptr, int N)
{
  HybridJobType jt = job_type[blockIdx.x];
  if (jt == BSPLINE_3D_JOB)
    return;

  const T fourPiInv = 0.0795774715459477f;
  int tid = threadIdx.x;
  const int numlm = (LMAX+1)*(LMAX+2)/2;

  __shared__ T* Ylm[BS], *dtheta[BS], *dphi[BS];
  if (blockIdx.x*BS+tid < N) {
    Ylm[tid]    = Ylm_ptr[blockIdx.x*BS+tid];
    dtheta[tid] = dYlm_dtheta_ptr[blockIdx.x*BS+tid];
    dphi[tid]   = dYlm_dphi_ptr[blockIdx.x*BS+tid];
  }

  __shared__ T rhat[BS][3];
  for (int i=0; i<3; i++) {
    int off = (3*blockIdx.x + i)*BS + tid;
    if (off < 3*N)
      rhat[0][i*BS+tid] = rhats[off];
  }
  __syncthreads();

  T costheta = rhat[tid][2];
  T sintheta = sqrt(1.0f-costheta*costheta);
  T cottheta = costheta/sintheta;
  
  T cosphi, sinphi;
  cosphi=rhat[tid][0]/sintheta;
  sinphi=rhat[tid][1]/sintheta;
  __shared__ T phi[BS];
  phi[tid] = atan2f(sinphi, cosphi);

  __shared__ T XlmVec[BS][numlm], 
    dXlmVec[BS][numlm];
  
  // Create a map from lm linear index to l and m
  __shared__ int l_lm[numlm];
  __shared__ int m_lm[numlm];
  __shared__ T  floatm_lm[numlm];
  int off=0;
  for (int l=0; l<=LMAX; l++) {
    if (tid < l+1) {
      l_lm[off+tid] = l;
      m_lm[off+tid] = tid;
      floatm_lm[off+tid] = (T)tid;
    }
    off += l+1;
  }

  T lsign = 1.0f;
  T dl = 0.0f;
  for (int l=0; l<=LMAX; l++) {
    int index=l*(l+3)/2;
    XlmVec[tid][index]  = lsign;  
    dXlmVec[tid][index] = dl * cottheta * XlmVec[tid][index];
    T dm = dl;
    for (int m=l; m>0; m--, index--) {
      T tmp = sqrt((dl+dm)*(dl-dm+1.0f));
      XlmVec[tid][index-1]  = 
	-(dXlmVec[tid][index] + dm*cottheta*XlmVec[tid][index])/ tmp;
      dXlmVec[tid][index-1] = 
	(dm-1.0f)*cottheta*XlmVec[tid][index-1] + XlmVec[tid][index]*tmp;
      dm -= 1.0f;
    }
    index = l*(l+1)/2;
    T sum = XlmVec[tid][index] * XlmVec[tid][index];
    for (int m=1; m<=l; m++) 
      sum += 2.0f*XlmVec[tid][index+m]*XlmVec[tid][index+m];
    // Now, renormalize the Ylms for this l
    T norm = sqrt((2.0f*dl+1.0f)*fourPiInv / sum);
    for (int m=0; m<=l; m++) {
      XlmVec[tid][index+m]  *= norm;
      dXlmVec[tid][index+m] *= norm;
    }
    lsign *= -1.0f;
    dl += 1.0f;
  }
  __syncthreads();

  // Multiply by azimuthal phase and store in Ylm
  int end = min (N-blockIdx.x*BS, BS);
  int nb = (numlm+BS-1)/BS;
  __shared__ T outbuff[3][2*BS];
  for (int i=0; i < end; i++) {
    for (int block=0; block<nb; block++) {
      int lm = block*BS + tid;
      if (lm < numlm) {
	int l = l_lm[lm];
	int m = m_lm[lm];
	T fm = floatm_lm[lm];
	T re, im;
	__sincosf(fm*phi[i], &im, &re);
	int off = ((l*(l+1))>>1) + m;
	int iplus = l*(l+1)+m;
	int iminus = l*(l+1)-m;
	// Ylm
	outbuff[0][iplus] =     re *  XlmVec[i][off];
	// dYlm_dtheta
	outbuff[1][iplus] =     re * dXlmVec[i][off];
	// dYlm_dphi
	outbuff[2][iplus] = -fm*im *  XlmVec[i][off];
	if (m != 0) {
	  outbuff[0][iminus] =     im *  XlmVec[i][off];
	  outbuff[1][iminus] =     im * dXlmVec[i][off];
	  outbuff[2][iminus] =  fm*re *  XlmVec[i][off];
	}
      }
      __syncthreads();
      // Now write back to global mem with coallesced writes
      int off = block*BS + tid;
      if (off < (LMAX+1)*(LMAX+1)) {
	Ylm[i][off]    = outbuff[0][tid];
	dtheta[i][off] = outbuff[1][tid];
	dphi[i][off]   = outbuff[2][tid];
      }
      off += BS;
      if (off < (LMAX+1)*(LMAX+1)) {
	Ylm[i][off]    = outbuff[0][tid+BS];
	dtheta[i][off] = outbuff[1][tid+BS];
	dphi[i][off]   = outbuff[2][tid+BS];
      }
    }
  }
}

template<typename T>
void CalcYlmRealCuda (T *rhats, HybridJobType *job_type,
		      T **Ylm_ptr, T **dYlm_dtheta_ptr, T **dYlm_dphi_ptr, 
		      int lMax, int N)
{
  const int BS=32;
  int Nblocks = (N+BS-1)/BS;
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(BS);
  
  if (lMax == 0)
    return;
  else if (lMax == 1)
    CalcYlmReal<T,1,BS><<<dimGrid,dimBlock>>>(rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 2)
    CalcYlmReal<T,2,BS><<<dimGrid,dimBlock>>>(rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 3)
    CalcYlmReal<T,3,BS><<<dimGrid,dimBlock>>>(rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 4)
    CalcYlmReal<T,4,BS><<<dimGrid,dimBlock>>>(rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 5)
    CalcYlmReal<T,5,BS><<<dimGrid,dimBlock>>>(rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 6)
    CalcYlmReal<T,6,BS><<<dimGrid,dimBlock>>>(rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 7)
    CalcYlmReal<T,7,BS><<<dimGrid,dimBlock>>>(rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 8)
    CalcYlmReal<T,8,BS><<<dimGrid,dimBlock>>>(rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in CalcYlmRealCuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}

void CalcYlmComplexCuda (float *rhats, HybridJobType *job_type,
			 float **Ylm_ptr, float **dYlm_dtheta_ptr, float **dYlm_dphi_ptr, 
			 int lMax, int N)
{
  const int BS=32;
  int Nblocks = (N+BS-1)/BS;
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(BS);
  
  if (lMax == 0)
    return;
  else if (lMax == 1)
    CalcYlmComplex<float,1,BS><<<dimGrid,dimBlock>>>
      (rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 2)
    CalcYlmComplex<float,2,BS><<<dimGrid,dimBlock>>>
      (rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 3)
    CalcYlmComplex<float,3,BS><<<dimGrid,dimBlock>>>
      (rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 4)
    CalcYlmComplex<float,4,BS><<<dimGrid,dimBlock>>>
      (rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 5)
    CalcYlmComplex<float,5,BS><<<dimGrid,dimBlock>>>
      (rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 6)
    CalcYlmComplex<float,6,BS><<<dimGrid,dimBlock>>>
      (rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 7)
    CalcYlmComplex<float,7,BS><<<dimGrid,dimBlock>>>
      (rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);
  else if (lMax == 8)
    CalcYlmComplex<float,8,BS><<<dimGrid,dimBlock>>>
      (rhats,job_type,Ylm_ptr,dYlm_dtheta_ptr,dYlm_dphi_ptr,N);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in CalcYlmComplexCuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}





template<typename T, int LMAX, int BS> __global__ void
CalcYlmComplex (T *rhats, HybridJobType *job_types, T **Ylm_ptr, int N)
{
  HybridJobType jt = job_types[blockIdx.x];
  if (jt == BSPLINE_3D_JOB)   return;

  const T fourPiInv = 0.0795774715459477f;
  int tid = threadIdx.x;
  const int numlm = (LMAX+1)*(LMAX+1);

  __shared__ T* Ylm[BS];
  if (blockIdx.x*BS+tid < N) 
    Ylm[tid]    = Ylm_ptr[blockIdx.x*BS+tid];

  __shared__ T rhat[BS][3];
  for (int i=0; i<3; i++) {
    int off = (3*blockIdx.x + i)*BS + tid;
    if (off < 3*N)
      rhat[0][i*BS+tid] = rhats[off];
  }
  __syncthreads();

  T costheta = rhat[tid][2];
  T sintheta = sqrt(1.0f-costheta*costheta);
  T cottheta = costheta/sintheta;
  
  T cosphi, sinphi;
  cosphi=rhat[tid][0]/sintheta;
  sinphi=rhat[tid][1]/sintheta;
  __shared__ T phi[BS];
  phi[tid] = atan2f(sinphi, cosphi);

  __shared__ T XlmVec[BS][(LMAX+1)*(LMAX+2)/2], 
    dXlmVec[BS][(LMAX+1)*(LMAX+2)/2];
  
  // Create a map from lm linear index to l and m
  __shared__ int l_lm[numlm];
  __shared__ int m_lm[numlm];
  __shared__ T  floatm_lm[numlm];
  int off=0;
  for (int l=0; l<=LMAX; l++) {
    if (tid < 2*l+1) {
      l_lm[off+tid] = l;
      m_lm[off+tid] = tid-l;
      floatm_lm[off+tid] = (T)(tid-l);
    }
    off += 2*l+1;
  }

  T lsign = 1.0f;
  T dl = 0.0f;
  for (int l=0; l<=LMAX; l++) {
    int index=l*(l+3)/2;
    XlmVec[tid][index]  = lsign;  
    dXlmVec[tid][index] = dl * cottheta * XlmVec[tid][index];
    T dm = dl;
    for (int m=l; m>0; m--, index--) {
      T tmp = sqrt((dl+dm)*(dl-dm+1.0f));
      XlmVec[tid][index-1]  = 
	-(dXlmVec[tid][index] + dm*cottheta*XlmVec[tid][index])/ tmp;
      dXlmVec[tid][index-1] = 
	(dm-1.0f)*cottheta*XlmVec[tid][index-1] + XlmVec[tid][index]*tmp;
      dm -= 1.0f;
    }
    index = l*(l+1)/2;
    T sum = XlmVec[tid][index] * XlmVec[tid][index];
    for (int m=1; m<=l; m++) 
      sum += 2.0f*XlmVec[tid][index+m]*XlmVec[tid][index+m];
    // Now, renormalize the Ylms for this l
    T norm = sqrt((2.0f*dl+1.0f)*fourPiInv / sum);
    for (int m=0; m<=l; m++) {
      XlmVec[tid][index+m]  *= norm;
      dXlmVec[tid][index+m] *= norm;
    }
    lsign *= -1.0f;
    dl += 1.0f;
  }
  __syncthreads();

  // Multiply by azimuthal phase and store in Ylm
  int end = min (N-blockIdx.x*BS, BS);
  int nb = ((LMAX+1)*(LMAX+1)+BS-1)/BS;
  __shared__ T outbuff[BS][2];
  for (int i=0; i < end; i++) {
    // __shared__ T sincosphi[2*LMAX+1][2];
    // if (tid < LMAX
    for (int block=0; block<nb; block++) {
      int lm = block*BS + tid;
      if (lm < numlm) {
	int l = l_lm[lm];
	int m = m_lm[lm];
	T fm = floatm_lm[lm];
	T re, im;
	__sincosf(fm*phi[i], &im, &re);
	// Switch sign if m<0 and it's odd
	if (m<0 && (m&1)) {
	  re *= -1.0f;
	  im *= -1.0f;
	}
	int off = ((l*(l+1))>>1) + abs(m);
	// Ylm
	outbuff[tid][0] =     re *  XlmVec[i][off];
	outbuff[tid][1] =     im *  XlmVec[i][off];
      }
      __syncthreads();
      // Now write back to global mem with coallesced writes
      int off = 2*block*BS + tid;
      if (off < 2*numlm) 
	Ylm[i][off]    = outbuff[0][tid];
      off += BS;
      if (off < 2*numlm) 
	Ylm[i][off]    = outbuff[0][tid+BS];
    }
  }
}


template<typename T, int LMAX, int BS> __global__ void
CalcYlmReal (T *rhats, HybridJobType *job_types, T **Ylm_ptr, int N)
{
  HybridJobType jt = job_types[blockIdx.x];
  if (jt == BSPLINE_3D_JOB)   return;

  const T fourPiInv = 0.0795774715459477f;
  int tid = threadIdx.x;
  const int numlm = (LMAX+1)*(LMAX+2)/2;

  __shared__ T* Ylm[BS];
  if (blockIdx.x*BS+tid < N) 
    Ylm[tid]    = Ylm_ptr[blockIdx.x*BS+tid];

  __shared__ T rhat[BS][3];
  for (int i=0; i<3; i++) {
    int off = (3*blockIdx.x + i)*BS + tid;
    if (off < 3*N)
      rhat[0][i*BS+tid] = rhats[off];
  }
  __syncthreads();

  T costheta = rhat[tid][2];
  T sintheta = sqrt(1.0f-costheta*costheta);
  T cottheta = costheta/sintheta;
  
  T cosphi, sinphi;
  cosphi=rhat[tid][0]/sintheta;
  sinphi=rhat[tid][1]/sintheta;
  __shared__ T phi[BS];
  phi[tid] = atan2f(sinphi, cosphi);

  __shared__ T XlmVec[BS][numlm], 
    dXlmVec[BS][numlm];
  
  // Create a map from lm linear index to l and m
  __shared__ int l_lm[numlm];
  __shared__ int m_lm[numlm];
  __shared__ T  floatm_lm[numlm];
  int off=0;
  for (int l=0; l<=LMAX; l++) {
    if (tid < l+1) {
      l_lm[off+tid] = l;
      m_lm[off+tid] = tid;
      floatm_lm[off+tid] = (T)tid;
    }
    off += l+1;
  }

  T lsign = 1.0f;
  T dl = 0.0f;
  for (int l=0; l<=LMAX; l++) {
    int index=l*(l+3)/2;
    XlmVec[tid][index]  = lsign;  
    dXlmVec[tid][index] = dl * cottheta * XlmVec[tid][index];
    T dm = dl;
    for (int m=l; m>0; m--, index--) {
      T tmp = sqrt((dl+dm)*(dl-dm+1.0f));
      XlmVec[tid][index-1]  = 
	-(dXlmVec[tid][index] + dm*cottheta*XlmVec[tid][index])/ tmp;
      dXlmVec[tid][index-1] = 
	(dm-1.0f)*cottheta*XlmVec[tid][index-1] + XlmVec[tid][index]*tmp;
      dm -= 1.0f;
    }
    index = l*(l+1)/2;
    T sum = XlmVec[tid][index] * XlmVec[tid][index];
    for (int m=1; m<=l; m++) 
      sum += 2.0f*XlmVec[tid][index+m]*XlmVec[tid][index+m];
    // Now, renormalize the Ylms for this l
    T norm = sqrt((2.0f*dl+1.0f)*fourPiInv / sum);
    for (int m=0; m<=l; m++) {
      XlmVec[tid][index+m]  *= norm;
      dXlmVec[tid][index+m] *= norm;
    }
    lsign *= -1.0f;
    dl += 1.0f;
  }
  __syncthreads();

  // Multiply by azimuthal phase and store in Ylm
  int end = min (N-blockIdx.x*BS, BS);
  int nb = (numlm+BS-1)/BS;
  __shared__ T outbuff[2*BS];
  for (int i=0; i < end; i++) {
    for (int block=0; block<nb; block++) {
      int lm = block*BS + tid;
      if (lm < numlm) {
	int l = l_lm[lm];
	int m = m_lm[lm];
	T fm = floatm_lm[lm];
	T re, im;
	__sincosf(fm*phi[i], &im, &re);
	int off = ((l*(l+1))>>1) + m;
	int iplus = l*(l+1)+m;
	int iminus = l*(l+1)-m;
	// Ylm
	outbuff[iplus] =     re *  XlmVec[i][off];
	if (m != 0) 
	  outbuff[iminus] =     im *  XlmVec[i][off];
      }
      __syncthreads();
      // Now write back to global mem with coallesced writes
      int off = block*BS + tid;
      if (off < (LMAX+1)*(LMAX+1)) 
	Ylm[i][off]    = outbuff[tid];
      off += BS;
      if (off < (LMAX+1)*(LMAX+1)) 
	Ylm[i][off]    = outbuff[tid+BS];
    }
  }
}

void CalcYlmRealCuda (float *rhats, HybridJobType *job_types, 
		      float **Ylm_ptr, int lMax, int N)
{
  const int BS=32;
  int Nblocks = (N+BS-1)/BS;
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(BS);
  
  if (lMax == 0)
    return;
  else if (lMax == 1)
    CalcYlmReal<float,1,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 2)
    CalcYlmReal<float,2,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 3)
    CalcYlmReal<float,3,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 4)
    CalcYlmReal<float,4,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 5)
    CalcYlmReal<float,5,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 6)
    CalcYlmReal<float,6,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 7)
    CalcYlmReal<float,7,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 8)
    CalcYlmReal<float,8,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in CalcYlmRealCuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}

void CalcYlmComplexCuda (float *rhats, HybridJobType *job_types, float **Ylm_ptr, int lMax, int N)
{
  const int BS=32;
  int Nblocks = (N+BS-1)/BS;
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(BS);
  
  if (lMax == 0)
    return;
  else if (lMax == 1)
    CalcYlmComplex<float,1,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 2)
    CalcYlmComplex<float,2,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 3)
    CalcYlmComplex<float,3,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 4)
    CalcYlmComplex<float,4,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 5)
    CalcYlmComplex<float,5,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 6)
    CalcYlmComplex<float,6,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 7)
    CalcYlmComplex<float,7,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);
  else if (lMax == 8)
    CalcYlmComplex<float,8,BS><<<dimGrid,dimBlock>>>(rhats,job_types,Ylm_ptr,N);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in CalcYlmComplexCuda:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
}



void dummy_float()
{
  float *rhats(0), **Ylm_ptr(0), **dYlm_dtheta_ptr(0), **dYlm_dphi_ptr(0);
  HybridJobType* job_types(0);
  
  CalcYlmRealCuda(rhats,    job_types, Ylm_ptr, dYlm_dtheta_ptr, dYlm_dphi_ptr, 1, 1);
  CalcYlmComplexCuda(rhats, job_types, Ylm_ptr, dYlm_dtheta_ptr, dYlm_dphi_ptr, 1, 1);
  CalcYlmRealCuda(rhats,    job_types, Ylm_ptr, 1, 1);
  CalcYlmComplexCuda(rhats, job_types, Ylm_ptr, 1, 1);
}

// void dummy_double()
// {
//   double *rhats(0), **Ylm_ptr(0), **dYlm_dtheta_ptr(0), **dYlm_dphi_ptr(0);
//   CalcYlmRealCuda(rhats, Ylm_ptr, dYlm_dtheta_ptr, dYlm_dphi_ptr, 1, 1);
//   CalcYlmComplexCuda(rhats, Ylm_ptr, dYlm_dtheta_ptr, dYlm_dphi_ptr, 1, 1);
//   CalcYlmRealCuda(rhats, Ylm_ptr, 1, 1);
//   CalcYlmComplexCuda(rhats, Ylm_ptr, 1, 1);
// }


#ifdef TEST_GPU_YLM

class Vec3
{
private:
  double r[3];
public:
  inline double  operator[](int i) const { return r[i]; }
  inline double& operator[](int i) { return r[i];}
  Vec3(double x, double y, double z) 
  { r[0]=x; r[1]=y; r[2]=z; }
  Vec3() { }
};
  

// Fast implementation
// See Geophys. J. Int. (1998) 135,pp.307-309
void
CalcYlm (Vec3 rhat,
	 vector<complex<double> > &Ylm,
	 vector<complex<double> > &dYlm_dtheta,
	 vector<complex<double> > &dYlm_dphi,
	 int lMax)
{
  const double fourPiInv = 0.0795774715459477;
  
  double costheta = rhat[2];
  double sintheta = std::sqrt(1.0-costheta*costheta);
  double cottheta = costheta/sintheta;
  
  double cosphi, sinphi;
  cosphi=rhat[0]/sintheta;
  sinphi=rhat[1]/sintheta;
  
  complex<double> e2iphi(cosphi, sinphi);
  
  
  double lsign = 1.0;
  double dl = 0.0;
  double XlmVec[2*lMax+1], dXlmVec[2*lMax+1];
  for (int l=0; l<=lMax; l++) {
    XlmVec[2*l]  = lsign;  
    dXlmVec[2*l] = dl * cottheta * XlmVec[2*l];
    XlmVec[0]    = lsign*XlmVec[2*l];
    dXlmVec[0]   = lsign*dXlmVec[2*l];
    double dm = dl;
    double msign = lsign;
    for (int m=l; m>0; m--) {
      double tmp = std::sqrt((dl+dm)*(dl-dm+1.0));
      XlmVec[l+m-1]  = -(dXlmVec[l+m] + dm*cottheta*XlmVec[l+m])/ tmp;
      dXlmVec[l+m-1] = (dm-1.0)*cottheta*XlmVec[l+m-1] + XlmVec[l+m]*tmp;
      // Copy to negative m
      XlmVec[l-(m-1)]  = -msign* XlmVec[l+m-1];
      dXlmVec[l-(m-1)] = -msign*dXlmVec[l+m-1];
      msign *= -1.0;
      dm -= 1.0;
    }
    double sum = 0.0;
    for (int m=-l; m<=l; m++) 
      sum += XlmVec[l+m]*XlmVec[l+m];
    // Now, renormalize the Ylms for this l
    double norm = std::sqrt((2.0*dl+1.0)*fourPiInv / sum);
    for (int m=-l; m<=l; m++) {
      XlmVec[l+m]  *= norm;
      dXlmVec[l+m] *= norm;
    }
    
    // Multiply by azimuthal phase and store in Ylm
    complex<double> e2imphi (1.0, 0.0);
    complex<double> eye(0.0, 1.0);
    for (int m=0; m<=l; m++) {
      Ylm[l*(l+1)+m]  =  XlmVec[l+m]*e2imphi;
      Ylm[l*(l+1)-m]  =  XlmVec[l-m]*conj(e2imphi);
      dYlm_dphi[l*(l+1)+m ]  =  (double)m * eye *XlmVec[l+m]*e2imphi;
      dYlm_dphi[l*(l+1)-m ]  = -(double)m * eye *XlmVec[l-m]*conj(e2imphi);
      dYlm_dtheta[l*(l+1)+m] = dXlmVec[l+m]*e2imphi;
      dYlm_dtheta[l*(l+1)-m] = dXlmVec[l-m]*conj(e2imphi);
      e2imphi *= e2iphi;
    } 
    
    dl += 1.0;
    lsign *= -1.0;
  }
}

// Fast implementation
// See Geophys. J. Int. (1998) 135,pp.307-309
void
CalcYlm (Vec3 rhat,
	 vector<double> &Ylm,
	 vector<double> &dYlm_dtheta,
	 vector<double> &dYlm_dphi,
	 int lMax)
{
  const double fourPiInv = 0.0795774715459477;
    
  double costheta = rhat[2];
  double sintheta = std::sqrt(1.0-costheta*costheta);
  double cottheta = costheta/sintheta;
    
  double cosphi, sinphi;
  cosphi=rhat[0]/sintheta;
  sinphi=rhat[1]/sintheta;
    
  complex<double> e2iphi(cosphi, sinphi);
    
  double lsign = 1.0;
  double dl = 0.0;
  double XlmVec[2*lMax+1], dXlmVec[2*lMax+1];
  for (int l=0; l<=lMax; l++) {
    XlmVec[2*l]  = lsign;  
    dXlmVec[2*l] = dl * cottheta * XlmVec[2*l];
    XlmVec[0]    = lsign*XlmVec[2*l];
    dXlmVec[0]   = lsign*dXlmVec[2*l];
    double dm = dl;
    double msign = lsign;
    for (int m=l; m>0; m--) {
      double tmp = std::sqrt((dl+dm)*(dl-dm+1.0));
      XlmVec[l+m-1]  = -(dXlmVec[l+m] + dm*cottheta*XlmVec[l+m])/ tmp;
      dXlmVec[l+m-1] = (dm-1.0)*cottheta*XlmVec[l+m-1] + XlmVec[l+m]*tmp;
      // Copy to negative m
      XlmVec[l-(m-1)]  = -msign* XlmVec[l+m-1];
      dXlmVec[l-(m-1)] = -msign*dXlmVec[l+m-1];
      msign *= -1.0;
      dm -= 1.0;
    }
    double sum = 0.0;
    for (int m=-l; m<=l; m++) 
      sum += XlmVec[l+m]*XlmVec[l+m];
    // Now, renormalize the Ylms for this l
    double norm = std::sqrt((2.0*dl+1.0)*fourPiInv / sum);
    for (int m=-l; m<=l; m++) {
      XlmVec[l+m]  *= norm;
      dXlmVec[l+m] *= norm;
    }
      
    // Multiply by azimuthal phase and store in Ylm
    Ylm[l*(l+1)]         =  XlmVec[l];
    dYlm_dphi[l*(l+1) ]  = 0.0;
    dYlm_dtheta[l*(l+1)] = dXlmVec[l];
    complex<double> e2imphi = e2iphi;
    for (int m=1; m<=l; m++) {
      Ylm[l*(l+1)+m]         =  XlmVec[l+m]*e2imphi.real();
      Ylm[l*(l+1)-m]         =  XlmVec[l+m]*e2imphi.imag();
      dYlm_dphi[l*(l+1)+m ]  = -(double)m * XlmVec[l+m] *e2imphi.imag();
      dYlm_dphi[l*(l+1)-m ]  =  (double)m * XlmVec[l+m] *e2imphi.real();
      dYlm_dtheta[l*(l+1)+m] = dXlmVec[l+m]*e2imphi.real();
      dYlm_dtheta[l*(l+1)-m] = dXlmVec[l+m]*e2imphi.imag();
      e2imphi *= e2iphi;
    } 
      
    dl += 1.0;
    lsign *= -1.0;
  }
}




#include <stdlib.h>

void TestYlmComplex()
{
  int numr = 1000;
  const int BS=32;

  float *rhat_device, *Ylm_device, *dtheta_device, *dphi_device;
  float **Ylm_ptr, **dtheta_ptr, **dphi_ptr;
  const int lmax = 5;
  const int numlm = (lmax+1)*(lmax+1);

  cudaMalloc ((void**)&rhat_device, 3*sizeof(float)*numr);
  cudaMalloc ((void**)&Ylm_device, 2*numlm*sizeof(float)*numr);
  cudaMalloc ((void**)&dtheta_device, 2*numlm*sizeof(float)*numr);
  cudaMalloc ((void**)&dphi_device, 2*numlm*sizeof(float)*numr);
  cudaMalloc ((void**)&Ylm_ptr,    numr*sizeof(float*));
  cudaMalloc ((void**)&dtheta_ptr, numr*sizeof(float*));
  cudaMalloc ((void**)&dphi_ptr,   numr*sizeof(float*));
  
  float *Ylm_host[numr], *dtheta_host[numr], *dphi_host[numr];
  float rhost[3*numr];
  vector<Vec3> rlist;
  for (int i=0; i<numr; i++) {
    Vec3 r;
    r[0] = 2.0*drand48()-1.0;
    r[1] = 2.0*drand48()-1.0;
    r[2] = 2.0*drand48()-1.0;
    double nrm = 1.0/std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    r[0] *= nrm;        r[1] *= nrm;        r[2] *= nrm;
    rlist.push_back(r);
    rhost[3*i+0]=r[0];  rhost[3*i+1]=r[1];  rhost[3*i+2]=r[2];
    
    Ylm_host[i] = Ylm_device+2*i*numlm;    
    dtheta_host[i] = dtheta_device+2*i*numlm;
    dphi_host[i]   = dphi_device + 2*i*numlm;
  }
  
  cudaMemcpy(rhat_device, rhost, 3*numr*sizeof(float),  cudaMemcpyHostToDevice);
  cudaMemcpy(Ylm_ptr, Ylm_host, numr*sizeof(float*),    cudaMemcpyHostToDevice);
  cudaMemcpy(dtheta_ptr, dtheta_host, numr*sizeof(float*), 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(dphi_ptr,  dphi_host, numr*sizeof(float*), cudaMemcpyHostToDevice);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in cudaMemcpy:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }


  dim3 dimBlock(BS);
  dim3 dimGrid((numr+BS-1)/BS);
  
  clock_t start, end;

  start = clock();
  for (int i=0; i<10000; i++) {
    CalcYlmComplex<float,5,BS><<<dimGrid,dimBlock>>>
      (rhat_device, Ylm_ptr, dtheta_ptr, dphi_ptr, numr);
  }
  cudaThreadSynchronize();
  end = clock();
  fprintf (stderr, "Ylm rate = %1.8f\n",
	   10000*numr/((double)(end-start)/(double)CLOCKS_PER_SEC));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in CalcYlmComplex:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
  complex<float> Ylm[numr*numlm], dtheta[numr*numlm], dphi[numr*numlm];
  cudaMemcpy(Ylm, Ylm_device, 2*numr*numlm*sizeof(float), 
	     cudaMemcpyDeviceToHost);
  cudaMemcpy(dtheta, dtheta_device, 2*numr*numlm*sizeof(float), 
	     cudaMemcpyDeviceToHost);
  cudaMemcpy(dphi, dphi_device, 2*numr*numlm*sizeof(float), 
	     cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in cudaMemcpy:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

  int n = 999;
  vector<complex<double> > Ylm_cpu(numlm), dtheta_cpu(numlm), dphi_cpu(numlm);
  CalcYlm (rlist[n], Ylm_cpu, dtheta_cpu, dphi_cpu, lmax);
  fprintf (stderr, "Ylm:\n");
  for (int lm=0; lm<numlm; lm++) {
    fprintf(stderr, "%12.7f %12.7f   %12.7f %12.7f  %3.0f %3.0f\n",
	    Ylm_cpu[lm].real(), Ylm_cpu[lm].imag(), 
	    Ylm[lm+n*numlm].real(), Ylm[lm+n*numlm].imag(),
	    Ylm_cpu[lm].real()/Ylm[lm+n*numlm].real(),
	    Ylm_cpu[lm].imag()/Ylm[lm+n*numlm].imag());
  }

  fprintf (stderr, "dtheta:\n");
  for (int lm=0; lm<numlm; lm++) {
    fprintf(stderr, "%12.6f %12.6f   %12.6f %12.6f  %3.0f %3.0f\n",
	    dtheta_cpu[lm].real(), dtheta_cpu[lm].imag(), 
	    dtheta[lm+n*numlm].real(), dtheta[lm+n*numlm].imag(),
	    dtheta_cpu[lm].real()/dtheta[lm+n*numlm].real(),
	    dtheta_cpu[lm].imag()/dtheta[lm+n*numlm].imag());
  }

  fprintf (stderr, "dphi:\n");
  for (int lm=0; lm<numlm; lm++) {
    fprintf(stderr, "%12.6f %12.6f   %12.6f %12.6f  %3.0f %3.0f\n",
	    dphi_cpu[lm].real(), dphi_cpu[lm].imag(), 
	    dphi[lm+n*numlm].real(), dphi[lm+n*numlm].imag(),
	    dphi_cpu[lm].real()/dphi[lm+n*numlm].real(),
	    dphi_cpu[lm].imag()/dphi[lm+n*numlm].imag());
  }
}


void TestYlmReal()
{
  int numr = 1000;
  const int BS=32;

  float *rhat_device, *Ylm_device, *dtheta_device, *dphi_device;
  float **Ylm_ptr, **dtheta_ptr, **dphi_ptr;
  const int lmax = 5;
  const int numlm = (lmax+1)*(lmax+1);

  int block_size = ((numlm+15)/16)*16;

  cudaMalloc ((void**)&rhat_device,   3*sizeof(float)*numr);
  cudaMalloc ((void**)&Ylm_device,    block_size*sizeof(float)*numr);
  cudaMalloc ((void**)&dtheta_device, block_size*sizeof(float)*numr);
  cudaMalloc ((void**)&dphi_device,   block_size*sizeof(float)*numr);
  cudaMalloc ((void**)&Ylm_ptr,       numr*sizeof(float*));
  cudaMalloc ((void**)&dtheta_ptr,    numr*sizeof(float*));
  cudaMalloc ((void**)&dphi_ptr,      numr*sizeof(float*));
  
  float *Ylm_host[numr], *dtheta_host[numr], *dphi_host[numr];
  float rhost[3*numr];
  vector<Vec3> rlist;
  for (int i=0; i<numr; i++) {
    Vec3 r;
    r[0] = 2.0*drand48()-1.0;
    r[1] = 2.0*drand48()-1.0;
    r[2] = 2.0*drand48()-1.0;
    double nrm = 1.0/std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    r[0] *= nrm;        r[1] *= nrm;        r[2] *= nrm;
    rlist.push_back(r);
    rhost[3*i+0]=r[0];  rhost[3*i+1]=r[1];  rhost[3*i+2]=r[2];
    
    Ylm_host[i]    = Ylm_device    + i*block_size;    
    dtheta_host[i] = dtheta_device + i*block_size;
    dphi_host[i]   = dphi_device   + i*block_size;
  }
  
  cudaMemcpy(rhat_device, rhost, 3*numr*sizeof(float),  cudaMemcpyHostToDevice);
  cudaMemcpy(Ylm_ptr, Ylm_host, numr*sizeof(float*),    cudaMemcpyHostToDevice);
  cudaMemcpy(dtheta_ptr, dtheta_host, numr*sizeof(float*), 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(dphi_ptr,  dphi_host, numr*sizeof(float*), cudaMemcpyHostToDevice);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in cudaMemcpy:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }


  dim3 dimBlock(BS);
  dim3 dimGrid((numr+BS-1)/BS);
  
  clock_t start, end;

  start = clock();
  for (int i=0; i<10000; i++) {
    CalcYlmReal<float,lmax,BS><<<dimGrid,dimBlock>>>
      (rhat_device, Ylm_ptr, dtheta_ptr, dphi_ptr, numr);
  }
  cudaThreadSynchronize();
  end = clock();
  fprintf (stderr, "Ylm rate = %1.8f\n",
	   10000*numr/((double)(end-start)/(double)CLOCKS_PER_SEC));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in CalcYlmReal:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
  float Ylm[numr*block_size], dtheta[numr*block_size], dphi[numr*block_size];
  cudaMemcpy(Ylm, Ylm_device, numr*block_size*sizeof(float), 
	     cudaMemcpyDeviceToHost);
  cudaMemcpy(dtheta, dtheta_device, numr*block_size*sizeof(float), 
	     cudaMemcpyDeviceToHost);
  cudaMemcpy(dphi, dphi_device, numr*block_size*sizeof(float), 
	     cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in cudaMemcpy:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

  int n = 999;
  vector<double> Ylm_cpu(numlm), dtheta_cpu(numlm), dphi_cpu(numlm);
  CalcYlm (rlist[n], Ylm_cpu, dtheta_cpu, dphi_cpu, lmax);
  fprintf (stderr, "Ylm:\n");
  for (int lm=0; lm<numlm; lm++) {
    fprintf(stderr, "%12.7f %12.7f %3.0f\n",
	    Ylm_cpu[lm], 
	    Ylm[lm+n*block_size], 
	    Ylm_cpu[lm]/Ylm[lm+n*block_size]);
  }

  fprintf (stderr, "dtheta:\n");
  for (int lm=0; lm<numlm; lm++) {
    fprintf(stderr, "%12.6f %12.6f %3.0f \n",
	    dtheta_cpu[lm], 
	    dtheta[lm+n*block_size], 
	    dtheta_cpu[lm]/dtheta[lm+n*block_size]);
  }

  fprintf (stderr, "dphi:\n");
  for (int lm=0; lm<numlm; lm++) {
    fprintf(stderr, "%12.6f %12.6f %3.0f\n",
	    dphi_cpu[lm], 
	    dphi[lm+n*block_size], 
	    dphi_cpu[lm]/dphi[lm+n*block_size]);
  }
}




main()
{
  TestYlmComplex();
  TestYlmReal();
}
#endif
