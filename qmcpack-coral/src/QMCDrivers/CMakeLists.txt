####################################
# QMC Drivers which implements various QMC algorithms
####################################
SET(QMCDRIVERS 
  SimpleFixedNodeBranch.cpp
  BranchIO.cpp
  QMCDriver.cpp
  WaveFunctionTester.cpp
  WalkerControlBase.cpp
  CloneManager.cpp
  QMCUpdateBase.cpp
  VMC/VMCUpdatePbyP.cpp
  VMC/VMCUpdateAll.cpp
  VMC/VMCFactory.cpp
  DMC/DMCOMP.cpp
  DMC/DMCUpdateAll.cpp
  DMC/DMCUpdatePbyP.cpp
  DMC/DMCUpdatePbyPFast.cpp
  DMC/DMCFactory.cpp
  DMC/WalkerControlFactory.cpp
  DMC/WalkerReconfiguration.cpp
  ../Estimators/LocalEnergyEstimator.cpp
  ../Estimators/LocalEnergyEstimatorHDF.cpp
  ../Estimators/EstimatorManager.cpp
  ../Estimators/CollectablesEstimator.cpp
)

IF(QMC_CUDA)
   SET(QMCDRIVERS ${QMCDRIVERS} VMC/VMC_CUDA.cpp DMC/DMC_CUDA.cpp)
ENDIF(QMC_CUDA)

SET(QMCDRIVERS ${QMCDRIVERS} VMC/VMCSingleOMP.cpp)

IF(HAVE_MPI) 
  SET(QMCDRIVERS ${QMCDRIVERS} 
  DMC/WalkerControlMPI.cpp
  DMC/WalkerReconfigurationMPI.cpp)
ENDIF(HAVE_MPI) 

####################################
# create libqmc 
####################################
ADD_LIBRARY(qmcdriver ${QMCDRIVERS})

#IF(QMC_BUILD_STATIC)
#  ADD_LIBRARY(qmcdriver STATIC ${QMCDRIVERS})
#ELSE(QMC_BUILD_STATIC)
#  ADD_LIBRARY(qmcdriver SHARED ${QMCDRIVERS})
#ENDIF(QMC_BUILD_STATIC)
