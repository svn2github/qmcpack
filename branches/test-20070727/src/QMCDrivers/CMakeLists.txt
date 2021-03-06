####################################
# QMC Drivers which implements various QMC algorithms
####################################
SET(QMCDRIVERS 
  ../Estimators/LocalEnergyEstimator.cpp
  ../Estimators/MultipleEnergyEstimator.cpp
  ../Estimators/RQMCMultipleEstimator.cpp
  ../Estimators/CompositeEstimators.cpp
  ../Estimators/GofREstimator.cpp
  ../Estimators/SkEstimator.cpp
  ../Estimators/EstimatorManager.cpp
  SimpleFixedNodeBranch.cpp
  QMCDriver.cpp
  QMCOptimize.cpp
  QMCCostFunctionBase.cpp
  QMCCostFunctionSingle.cpp
  WaveFunctionTester.cpp
  WalkerControlBase.cpp
  CloneManager.cpp
  QMCUpdateBase.cpp
  VMC/VMCUpdatePbyP.cpp
  VMC/VMCUpdateAll.cpp
  VMC/VMCSingle.cpp
  VMC/VMCFactory.cpp
  DMC/DMCUpdateAll.cpp
  DMC/DMCUpdatePbyP.cpp
  DMC/DMCNonLocalUpdate.cpp
  DMC/DMC.cpp
  DMC/DMCFactory.cpp
  DMC/WalkerControlFactory.cpp
  DMC/WalkerReconfiguration.cpp
  CorrelatedSampling/CSUpdateBase.cpp
  CorrelatedSampling/CSVMC.cpp
  CorrelatedSampling/CSVMCUpdateAll.cpp
  CorrelatedSampling/CSVMCUpdatePbyP.cpp
  ../Estimators/CSEnergyEstimator.cpp
  ../Estimators/CSPolymerEstimator.cpp
)
#  ../Estimators/PairCorrEstimator.cpp

IF(ENABLE_OPENMP) 
  SET(QMCDRIVERS ${QMCDRIVERS} 
      VMC/VMCSingleOMP.cpp
      DMC/DMCOMP.cpp
      QMCCostFunctionOMP.cpp
     )
ENDIF(ENABLE_OPENMP) 

IF(HAVE_MPI) 
  SET(QMCDRIVERS ${QMCDRIVERS} 
  DMC/WalkerControlMPI.cpp
  DMC/WalkerReconfigurationMPI.cpp)
ENDIF(HAVE_MPI) 


IF(NOT QMCPLUSPLUS_RELEASE)
  SET(QMCDRIVERS ${QMCDRIVERS}
    PolymerEstimator.cpp
    RQMCMultiple.cpp
    MultiChain.cpp
    SpaceWarp.cpp
    RQMCMultiWarp.cpp
    VMC/VMCMultipleWarp.cpp
    VMC/VMCPbyPMultiWarp.cpp
  )
ENDIF(NOT QMCPLUSPLUS_RELEASE)

####################################
# create libqmc 
####################################
ADD_LIBRARY(qmcdriver ${QMCDRIVERS})
#IF(QMC_BUILD_STATIC)
#  ADD_LIBRARY(qmcdriver STATIC ${QMCDRIVERS})
#ELSE(QMC_BUILD_STATIC)
#  ADD_LIBRARY(qmcdriver SHARED ${QMCDRIVERS})
#ENDIF(QMC_BUILD_STATIC)
