############################
# generic toolchain file with Gnu + CUDA
#  enable OpenMP
#
# remove any offending compiler options in GNU_FLAGS and SIMD_FLAGS
############################
set(CMAKE_C_COMPILER  gcc)
set(CMAKE_CXX_COMPILER  g++)
set(GNU_OPTS "-DADD_ -DINLINE_ALL=inline -Drestrict=__restrict__")
set(GNU_FLAGS " -fomit-frame-pointer -malign-double  -fopenmp -O3 -finline-limit=1000 -fstrict-aliasing -funroll-all-loops ")
set(SIMD_FLAGS " -msse -msse2 -msse3")
set(CMAKE_CXX_FLAGS "${SIMD_FLAGS} ${GNU_FLAGS} -ftemplate-depth-60 ${GNU_OPTS} -Wno-deprecated ")
set(CMAKE_C_FLAGS "${SIMD_FLAGS} ${GNU_FLAGS} -std=c99")

# need for both c++ and c
SET(ENABLE_OPENMP 1)
SET(HAVE_MPI 0)
set(HAVE_CUDA 1)
SET(QMC_BUILD_STATIC 1)

SET(HAVE_SSE 1)
SET(HAVE_SSE2 1)
SET(HAVE_SSE3 1)
SET(HAVE_SSSE3 1)
SET(USE_PREFETCH 1)
SET(PREFETCH_AHEAD 12)

set(CMAKE_FIND_ROOT_PATH
  $ENV{HDF5_HOME}
  $ENV{FFTW_HOME}
)

set(CUDA_NVCC_FLAGS "-arch=sm_20;-Drestrict=__restrict__;-DNO_CUDA_MAIN;-O3")

INCLUDE(Platform/UnixPaths)

SET(CMAKE_CXX_LINK_SHARED_LIBRARY)
SET(CMAKE_CXX_LINK_MODULE_LIBRARY)
SET(CMAKE_C_LINK_SHARED_LIBRARY)
SET(CMAKE_C_LINK_MODULE_LIBRARY)
