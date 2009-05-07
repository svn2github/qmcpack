#include <cstdio>
#include <vector>
#include <complex>

using namespace std;

template<typename T, int LMAX, int BS> __global__ void
CalcYlmComplex (T rhats[], 
		T *Ylm_ptr[], T *dYlm_dtheta_ptr[], T *dYlm_dphi_ptr[], int N)
{
  const T fourPiInv = 0.0795774715459477;
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
  T sintheta = sqrt(1.0-costheta*costheta);
  T cottheta = costheta/sintheta;
  
  T cosphi, sinphi;
  cosphi=rhat[tid][0]/sintheta;
  sinphi=rhat[tid][1]/sintheta;
  __shared__ T phi[BS];
  phi[tid] = atan2f(sinphi, cosphi);

  T lsign = 1.0;
  T dl = 0.0;
  __shared__ T XlmVec[BS][(LMAX*(LMAX+1))>>1], 
    dXlmVec[BS][(LMAX*(LMAX+1))>>1];

  // Create a map from lm linear index to l and m
  __shared__ int l_lm[numlm];
  __shared__ int m_lm[numlm];
  __shared__ T  floatm_lm[numlm];
  int off=0;
  for (int l=0; l<=LMAX; l++) {
    if (tid < 2*l+1) {
      l_lm[off+tid] = l;
      m_lm[off+tid] = tid-l;
      floatm_lm[off+tid] = (T)tid-l;
    }
    off += 2*l+1;
  }

  for (int l=0; l<=LMAX; l++) {
    int index=l*(l+3)/2;
    XlmVec[tid][index]  = lsign;  
    dXlmVec[tid][index] = dl * cottheta * XlmVec[tid][index];
    T dm = dl;
    for (int m=l; m>0; m--, index--) {
      T tmp = sqrt((dl+dm)*(dl-dm+1.0));
      XlmVec[tid][index-1]  = 
	-(dXlmVec[tid][index] + dm*cottheta*XlmVec[tid][index])/ tmp;
      dXlmVec[tid][index-1] = 
	(dm-1.0)*cottheta*XlmVec[tid][index-1] + XlmVec[tid][index]*tmp;
      dm -= 1.0;
    }
    index = l*(l+1)/2;
    T sum = XlmVec[tid][index] * XlmVec[tid][index];
    for (int m=1; m<=l; m++) 
      sum += 2.0f*XlmVec[tid][index+m]*XlmVec[tid][index+m];
    // Now, renormalize the Ylms for this l
    T norm = sqrt((2.0*dl+1.0)*fourPiInv / sum);
    index = l*(l+1)/2;
    for (int m=0; m<=l; m++) {
      XlmVec[tid][index+m]  *= norm;
      dXlmVec[tid][index+m] *= norm;
    }
  }
  __syncthreads();

  // Multiply by azimuthal phase and store in Ylm
  int end = min (N-blockIdx.x*BS, BS);
  int nb = ((LMAX+1)*(LMAX+1)+BS-1)/BS;
  __shared__ T outbuff[3][BS][2];
  for (int i=0; i < end; i++) {
    for (int block=0; block<nb; block++) {
      int lm = block*BS + tid;
      if (lm < numlm) {
	int l = l_lm[lm];
	int m = m_lm[lm];
	T fm = floatm_lm[lm];
	T re, im;
	sincosf(fm*phi[i], &im, &re);
	// (-1)^m term
	if (m&1) {
	  re *= -1.0;
	  im *= -1.0;
	}
	// Conjugate for m < 0
	if (m<0)
	  im *= -1.0;
	int off = (l*(l+1))>>1 + m;
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
      // Now write back to global mem with coallesced writes
      int off = 2*block*BS + tid;
      if (off < 2*numlm) {
	Ylm[i][off]    = outbuff[0][0][off];
	dtheta[i][off] = outbuff[1][0][off];
	dphi[i][off]   = outbuff[2][0][off];
      }
      off += BS;
      if (off < 2*numlm) {
	Ylm[i][off]    = outbuff[0][0][off];
	dtheta[i][off] = outbuff[1][0][off];
	dphi[i][off]   = outbuff[2][0][off];
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
	 vector<complex<double> > &dYlm_dphi)
{
  int lMax = 5;
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



void TestYlm()
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
  
  CalcYlmComplex<float,5,BS><<<dimGrid,dimBlock>>>
    (rhat_device, Ylm_ptr, dtheta_ptr, dphi_ptr, numr);

  cudaThreadSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in CalcYlmComplex:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }
  complex<float> Ylm[numr*numlm];
  cudaMemcpy(Ylm, Ylm_device, 2*numr*numlm*sizeof(float), 
	     cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stderr, "CUDA error in cudaMemcpy:\n  %s\n",
	     cudaGetErrorString(err));
    abort();
  }

  vector<complex<double> > Ylm_cpu(numlm), dtheta_cpu(numlm), dphi_cpu(numlm);
  CalcYlm (rlist[0], Ylm_cpu, dtheta_cpu, dphi_cpu);
  for (int lm=0; lm<numlm; lm++) {
    fprintf(stderr, "%12.6f %12.6f   %12.6f %12.6f\n",
	    Ylm_cpu[lm].real(), Ylm_cpu[lm].imag(), 
	    Ylm[lm].real(), Ylm[lm].imag());
  }



}



main()
{
  TestYlm();


}
