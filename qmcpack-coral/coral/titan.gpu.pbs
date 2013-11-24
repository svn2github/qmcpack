#!/bin/bash
#PBS -A mat044
#PBS -j oe
#PBS -N q.gpu
#PBS -l walltime=00:15:00,nodes=780
#PBS -l gres=widow3
#PBS -V

module load craype-accel-nvidia35

export OMP_NUM_THREADS=1
let NP=$PBS_NUM_NODES

cd $PBS_O_WORKDIR

aprun -n ${NP} -N 1 -d ${OMP_NUM_THREADS} ./qmcapp_cuda --async_swap input.gpu.xml &> gpu.w256.n780.p1x1.log
