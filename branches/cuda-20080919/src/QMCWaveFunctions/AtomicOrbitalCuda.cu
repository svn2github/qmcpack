template<typename T, int LMAX, int BS> __global__
CalcYlmComplex (T rhats[], 
		T *Ylm_ptr[], T *dYlm_dtheta_ptr[], T *dYlm_dphi_ptr[], int N)
{
  const T fourPiInv = 0.0795774715459477;
  int tid = threadIdx.x;
  const int numlm = (LMAX+1)*(LMAX+1);

  __shared__ T* Ylm[BS], dtheta[BS], dphi[BS];
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
  __synchthreads();

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
  __shared__ XlmVec[BS][(LMAX*(LMAX+1))>>1], 
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
	Ylm[off]    = outbuff[0][0][off];
	dtheta[off] = outbuff[1][0][off];
	dphi[off]   = outbuff[2][0][off];
      }
      off += BS;
      if (off < 2*numlm) {
	Ylm[off]    = outbuff[0][0][off];
	dtheta[off] = outbuff[1][0][off];
	dphi[off]   = outbuff[2][0][off];
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
