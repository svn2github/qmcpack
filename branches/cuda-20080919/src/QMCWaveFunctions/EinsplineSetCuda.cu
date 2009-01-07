template<typename T, int BS> __global__
void phase_factor_kernel (T kPoints[], int makeTwoCopies[], 
			  T pos[], T *phi_in[], T *phi_out[], 
			  int num_splines, int num_walkers)
{
  __shared__ T in_shared[BS][2*BS+1], out_shared[BS][BS+1], kPoints_s[BS][3],
    pos_s[BS][3];
  __shared__ T *phi_in_ptr[BS], *phi_out_ptr[BS];
  int tid = threadIdx.x;

  for (int i=0; i<3; i++) {
    int off = (3*blockIdx.x+i)*BS + tid;
    if (off < 3*num_walkers) 
      pos_s[0][i*BS + tid] =  pos[off];
  }
  
  phi_in_ptr[tid]  = phi_in[blockIdx.x*BS+tid];
  phi_out_ptr[tid] = phi_out[blockIdx.x*BS+tid];

  __syncthreads();

  int nb = (num_splines + BS-1)/num_splines;
  
  int outIndex=0;
  int outBlock=0;
  __shared__ int m2c[BS];
  int numWrite = min(BS, num_walkers-blockIdx.x*BS);
  for (int block=0; block<nb; block++) {
    // Load kpoints into shared memory
    for (int i=0; i<3; i++) {
      int off = (3*block+i)*BS + tid;
      if (off < 3*num_splines)
    	kPoints_s[0][i*BS+tid] = kPoints[off];
    }
    // // Load phi_in with coallesced reads
    int iend = min(BS,num_walkers-blockIdx.x*BS);
    for (int i=0; i<iend; i++) {
      if ((2*block+0)*BS+tid < num_splines)
	in_shared[i][tid   ] = phi_in_ptr[i][(2*block+0)*BS+tid];
      if ((2*block+1)*BS+tid < num_splines)
	in_shared[i][tid+BS] = phi_in_ptr[i][(2*block+1)*BS+tid];
    }
    // Load makeTwoCopies with coallesced reads
    if (block*BS+tid < num_splines)
      m2c[tid] = makeTwoCopies[block*BS + tid];
    __syncthreads();
    T s, c;
    int end = ((block+1)*BS <= num_splines) ? BS : (num_splines - block*BS);
    for (int i=0; i<end; i++) {
      T phase = -(pos_s[tid][0]*kPoints_s[i][0] +
    		  pos_s[tid][1]*kPoints_s[i][1] +
    		  pos_s[tid][2]*kPoints_s[i][2]);
      sincos(phase, &s, &c);
      T phi_real = in_shared[tid][2*i]*c - in_shared[tid][2*i+1]*s;
      T phi_imag = in_shared[tid][2*i]*s + in_shared[tid][2*i+1]*c;
      out_shared[tid][outIndex++] = phi_real;
      if (outIndex == BS) {
    	for (int j=0; j<numWrite; j++)
    	  phi_out_ptr[j][outBlock*BS+tid]= out_shared[j][tid];
    	outIndex = 0;
    	outBlock++;
      }
      __syncthreads();
      if (m2c[i]) 
    	out_shared[tid][outIndex++] = phi_imag;
      if (outIndex == BS) {
    	for (int j=0; j<numWrite; j++)
    	  phi_out_ptr[j][outBlock*BS+tid] = out_shared[j][tid];
    	outIndex = 0;
    	outBlock++;
      }
      __syncthreads();
    }

  }
  
  // Write remainining outputs
  for (int i=0; i<numWrite; i++)
    if (tid < outIndex)
      phi_out_ptr[i][outBlock*BS+tid] = out_shared[i][tid];
}

#include <cstdio>

void apply_phase_factors(float kPoints[], int makeTwoCopies[], 
			 float pos[], float *phi_in[], float *phi_out[], 
			 int num_splines, int num_walkers)
{
  fprintf (stderr, "Applying phase factors on GPU.\n");

  const int BS = 32;
  dim3 dimBlock(BS);
  dim3 dimGrid ((num_walkers+BS-1)/BS);

  phase_factor_kernel<float,BS><<<dimGrid,dimBlock>>>
    (kPoints, makeTwoCopies, pos, phi_in, phi_out, num_splines, num_walkers);
}
