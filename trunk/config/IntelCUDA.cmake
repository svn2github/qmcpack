#--------------------------------------------------------------------------
# Generic tool chain file with Intel Composer XE and CUDA
#  Enabled OpneMP 
#  -xSSE2 for SSE2 instructions
#
# How to customize it
#  INTEL_OPTS 
#  CMAKE_FIND_ROOT_PATH for the working environment
#--------------------------------------------------------------------------
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_C_COMPILER  icc)
set(GNU_OPTS "-DADD_ -DINLINE_ALL=inline")

set(INTEL_OPTS "-g -unroll -O3 -ip -openmp -opt-prefetch -ftz -xSSE2")
set(CMAKE_CXX_FLAGS "$ENV{CXX_FLAGS} ${GNU_OPTS} ${INTEL_OPTS} -restrict -Wno-deprecated")
set(CMAKE_C_FLAGS "$ENV{CC_FLAGS} ${INTEL_OPTS} -std=c99 -restrict -Wno-deprecated")

SET(CMAKE_Fortran_FLAGS "${INTEL_OPTS}")
SET(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS})

#--------------------------------------------------------------------------
# path where the libraries are located
# boost,hdf,szip,libxml2,fftw,essl
#--------------------------------------------------------------------------
set(CMAKE_FIND_ROOT_PATH
  $ENV{HDF5_HOME}
  $ENV{FFTW_HOME}
)

#--------------------------------------------------------------------------
# below is common for INTEL compilers and MKL library
#--------------------------------------------------------------------------
set(ENABLE_OPENMP 1)
set(HAVE_MPI 0)
set(HAVE_SSE 1)
set(HAVE_SSE2 1)
set(HAVE_SSE3 1)
set(HAVE_SSSE3 1)
set(USE_PREFETCH 1)
set(PREFETCH_AHEAD 10)
set(HAVE_MKL 1)
set(HAVE_MKL_VML 1)

set(CUDA_NVCC_FLAGS "-arch=sm_20;-Drestrict=__restrict__;-DNO_CUDA_MAIN;-O3")

include_directories($ENV{MKLROOT}/include)
link_libraries(-L$ENV{MKLROOT}/lib/intel64 -mkl=sequential)

INCLUDE(Platform/UnixPaths)

SET(CMAKE_CXX_LINK_SHARED_LIBRARY)
SET(CMAKE_CXX_LINK_MODULE_LIBRARY)
SET(CMAKE_C_LINK_SHARED_LIBRARY)
SET(CMAKE_C_LINK_MODULE_LIBRARY)

