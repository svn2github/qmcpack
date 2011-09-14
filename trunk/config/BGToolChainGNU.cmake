# the name of the target operating system
#SET(CMAKE_SYSTEM_NAME BlueGeneP)
SET(BGP 1 CACHE BOOL "On BlueGeneP")
SET(Linux 0)

SET(QMC_BUILD_STATIC 1)
SET(ENABLE_OPENMP 1)
SET(HAVE_MPI 1)
SET(HAVE_LIBESSL 1)

# set the compiler
#set(CMAKE_C_COMPILER  /opt/ibmcmp/vacpp/bg/9.0/bin/bgxlc_r)
#set(CMAKE_CXX_COMPILER  /opt/ibmcmp/vacpp/bg/9.0/bin/bgxlC_r)
set(CMAKE_C_COMPILER  /soft/apps/current/gcc-4.3.2/comm/default/bin/mpicc)             
set(CMAKE_CXX_COMPILER  /soft/apps/current/gcc-4.3.2/comm/default/bin/mpicxx )           

SET(CMAKE_CXX_FLAGS "-g -O2 -Drestrict=__restrict__  -Wno-deprecated  -fopenmp")
SET(CMAKE_C_FLAGS "-O2 -g -Drestrict=__restrict__  -std=gnu99 -fomit-frame-pointer -fopenmp ")

FOREACH(type SHARED_LIBRARY SHARED_MODULE EXE)
  SET(CMAKE_${type}_LINK_STATIC_C_FLAGS "-Wl,-Bstatic")
  SET(CMAKE_${type}_LINK_DYNAMIC_C_FLAGS "-Wl,-Bstatic")
  SET(CMAKE_${type}_LINK_STATIC_CXX_FLAGS "-Wl,-Bstatic")
  SET(CMAKE_${type}_LINK_DYNAMIC_CXX_FLAGS "-Wl,-Bstatic")
ENDFOREACH(type)

# set the search path for the environment coming with the compiler
# and a directory where you can install your own compiled software
set(BOOST_HOME /home/projects/qmcpack/boost-1.45_0)
set(EINSPLINE_HOME /home/projects/qmcpack/einspline_gcc)
set(LIBXML2_HOME /home/projects/qmcpack/libxml_gcc)
set(HDF5_HOME /soft/apps/hdf5-1.8.0)
set(FFTW_HOME /soft/apps/fftw-3.1.2-double)


set(LAPACK_LIBRARY /home/projects/qmcpack/liblapack_gcc.a)
set(BLAS_LIBRARY /soft/apps/ESSL-4.4.1-1/lib/libesslbg.a)
SET(FORTRAN_LIBRARIES 
/soft/apps/ibmcmp-aug2011/xlf/bg/11.1/lib/libxlf90_r.a
/soft/apps/ibmcmp-aug2011/xlf/bg/11.1/lib/libxlfmath.a
/soft/apps/ibmcmp-aug2011/xlf/bg/11.1/lib/libxlopt.a
/soft/apps/ibmcmp-aug2011/xlf/bg/11.1/lib/libxl.a
)
#link lapack, essl, mass
#link_libraries(/soft/apps/LAPACK/lapack_3.3_BGP.a /soft/apps/ESSL-4.4.1-1/lib/libesslbg.a 
#link_libraries(/home/projects/qmcpack/liblapack_gcc.a /soft/apps/ESSL-4.4.1-1/lib/libesslbg.a 
#   /soft/apps/ibmcmp-aug2011/xlf/bg/11.1/lib/libxlf90_r.a 
#   /soft/apps/ibmcmp-aug2011/xlf/bg/11.1/lib/libxlfmath.a
#   /soft/apps/ibmcmp-aug2011/xlf/bg/11.1/lib/libxlopt.a
#)

