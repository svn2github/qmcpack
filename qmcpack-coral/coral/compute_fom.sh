#!/bin/env bash
#usage compute_fom.sh logfile

# CPU= 8 mpi * 2 omp * 16 samples_per_thread
# GPU= 1 mpi * 1 omp * 256 samples_per_thread
samples_per_node=256

# blocks*steps of the dmc block
mcsteps=100

#number of nodes
node=780

#use the timing in the log file
let work=$node*$samples_per_node*$mcsteps
tail -10 $1| grep Execution | grep -v Total | awk -v a=$work '{print FOM a/$5}'
