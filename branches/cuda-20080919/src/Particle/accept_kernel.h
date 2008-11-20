#ifndef ACCEPT_KERNEL_H
#define ACCEPT_KERNEL_H

void
accept_move_GPU_cuda (float* Rlist[], float new_pos[], 
		      int toAccept[], int iat, int N);

#endif
