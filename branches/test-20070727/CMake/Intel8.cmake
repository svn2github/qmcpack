#compiler flags for intel 8.x
SET(INTEL_COMPILER 1)
ADD_DEFINITIONS(-DADD_ -DINLINE_ALL=inline)

#enable Interprocedural (IP) Optimizations
#-ipo_obj force generation of real object files (requires -ipo)
#SET(CMAKE_CXX_FLAGS "-restrict -unroll -fno-alias -O3 -Ob=1 -ansi -ipo -ipo_obj -cxxlib-icc")
#SET(CMAKE_CC_FLAGS "-restrict -unroll -fno-alias -O3 -Ob=1 -ansi -ipo -ipo_obj")
#SET(CMAKE_CXX_FLAGS "-restrict -unroll -fno-alias -g -ansi")
#SET(CMAKE_CC_FLAGS "-restrict -unroll -fno-alias -g -ansi")
#SET(CMAKE_CXX_FLAGS "-restrict -unroll -fno-alias -O3 -cxxlib-icc")
#SET(CMAKE_CXX_FLAGS "-restrict -unroll -fno-alias -O3 -ansi -fno-fnalias -ivdep_parallel -Ob=2")
#SET(CMAKE_CXX_FLAGS "-restrict -unroll -fno-alias -O3 -ivdep_parallel -Ob=2")
#SET(CMAKE_CXX_FLAGS "-restrict -unroll -fno-alias -O3 -Ob=2 -qp")
#SET(CMAKE_CXX_FLAGS "-restrict -unroll -fno-alias -O3 -Ob=2 -cxxlib-icc")
SET(CMAKE_CXX_FLAGS "-restrict -unroll -fno-alias -O3 -Wno-deprecated")
SET(CMAKE_CC_FLAGS "-restrict -unroll -fno-alias -O3")

IF(QMC_OMP)
  SET(ENABLE_OPENMP 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
ENDIF(QMC_OMP)

#IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
#ELSE(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
IF(QMC_BITS MATCHES 64)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftz ")
  IF(HAVE_MPI)
  LINK_LIBRARIES(-lmpi)
  ENDIF(HAVE_MPI)
ENDIF(QMC_BITS MATCHES 64)
#ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF($ENV{CXX} MATCHES cmpi)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ccl icpc")
ENDIF($ENV{CXX} MATCHES cmpi)

#ifc -> ifort
#SET(FORTRAN_LIBS " -lifcore -lifport")
SET(F77 ifort)
SET(F77OPTFLAGS  -fpp2 -O3)
SET(F77FLAGS ${F77OPTFLAGS})

######################################################
#KCC needs to be used to build static libraries
######################################################
#SET(CMAKE_AR xild) 
#SET(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> -lib -o <TARGET> <OBJECTS>")
