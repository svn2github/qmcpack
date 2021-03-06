SET(WFBASE_SRCS
    TrialWaveFunction.cpp
    OrbitalBase.cpp
    DiffOrbitalBase.cpp
    OrbitalBuilderBase.cpp
    WaveFunctionFactory.cpp
    ComboOrbital.cpp
    OrbitalConstraintsBase.cpp
   )

SET(JASTROW_SRCS
  Jastrow/LRTwoBodyJastrow.cpp
  Jastrow/JAABuilder.cpp
  Jastrow/JABBuilder.cpp
  Jastrow/PadeJastrowBuilder.cpp
  Jastrow/WMJastrowBuilder.cpp
  Jastrow/AnyConstraints.cpp
  Jastrow/CBSOBuilder.cpp
  Jastrow/JastrowBuilder.cpp
  Jastrow/BsplineJastrowBuilder.cpp
  Jastrow/kSpaceJastrow.cpp
  Jastrow/kSpaceJastrowBuilder.cpp
  Jastrow/RPAJastrow.cpp
  Jastrow/singleRPAJastrowBuilder.cpp
  Jastrow/TwoBodyJastrowOrbitalBspline.cpp
  Jastrow/OneBodyJastrowOrbitalBspline.cpp
)

#  Jastrow/BsplineFunctor.cpp
# common ferminoic wavefunctions
SET(FERMION_SRCS
  SPOSetBase.cpp
  BasisSetFactory.cpp
  Fermion/DiracDeterminantBase.cpp
  Fermion/SlaterDet.cpp
  Fermion/SlaterDetBuilder.cpp
  PlaneWave/PWBasis.cpp
  PlaneWave/PWParameterSet.cpp
  PlaneWave/PWOrbitalBuilder.cpp
)

# plane-wave / HEG
IF(QMC_COMPLEX)
  SET(FERMION_SRCS ${FERMION_SRCS}
      ElectronGas/ElectronGasComplexOrbitalBuilder.cpp
      PlaneWave/PWOrbitalSet.cpp
     )
ELSE(QMC_COMPLEX)
  SET(FERMION_SRCS ${FERMION_SRCS}
      ElectronGas/ElectronGasOrbitalBuilder.cpp
      PlaneWave/PWRealOrbitalSet.cpp
     )
ENDIF(QMC_COMPLEX)

IF(QMC_CUDA)
   SET(FERMION_SRCS ${FERMION_SRCS}
       Fermion/determinant_update.cu 
       Fermion/DiracDeterminantCUDA.cpp
      )
   SET(JASTROW_SRCS ${JASTROW_SRCS}
       Jastrow/CudaSpline.cu
      )
   IF(HAVE_EINSPLINE)
      SET(FERMION_SRCS ${FERMION_SRCS}
         EinsplineSetCuda.cpp
         PhaseFactors.cu
         AtomicOrbitalCuda.cu
         )
   ENDIF(HAVE_EINSPLINE)
ENDIF(QMC_CUDA)

# wavefunctions only availbale to 3-dim problems
IF(OHMMS_DIM MATCHES 3)
  SET(FERMION_SRCS ${FERMION_SRCS}
      Jastrow/ThreeBodyGeminal.cpp
      Jastrow/ThreeBodyBlockSparse.cpp
      Jastrow/JastrowBasisBuilder.cpp
      Bspline3DSetBase.cpp
      Bspline3DSet.cpp
      Bspline3DSetTrunc.cpp
      TricubicBsplineSetBuilder.cpp
      TricubicBsplineSetBuilder.1.cpp
      TricubicBsplineSetBuilder.2.cpp
      MolecularOrbitals/STOBuilder.cpp
      MolecularOrbitals/GTOBuilder.cpp
      MolecularOrbitals/NGOBuilder.cpp
     )
  IF(HAVE_EINSPLINE)
    SET(FERMION_SRCS 
      ${FERMION_SRCS}
      EinsplineSet.cpp
      EinsplineSetBuilder.cpp
      EinsplineWrapper.cpp
      MuffinTin.cpp
      AtomicOrbital.cpp
     )
  ENDIF(HAVE_EINSPLINE)
    
  IF(NOT QMC_COMPLEX)
    SET(FERMION_SRCS ${FERMION_SRCS} AGPDeterminant.cpp AGPDeterminantBuilder.cpp)
  ENDIF(NOT QMC_COMPLEX)
ENDIF(OHMMS_DIM MATCHES 3)

####################################
# create libqmcwfs
####################################
IF(QMC_CUDA)
  CUDA_ADD_LIBRARY(qmcwfs ${WFBASE_SRCS} ${JASTROW_SRCS} ${FERMION_SRCS})
ELSE(QMC_CUDA)
  ADD_LIBRARY(qmcwfs ${WFBASE_SRCS} ${JASTROW_SRCS} ${FERMION_SRCS})
ENDIF(QMC_CUDA)
#IF(QMC_BUILD_STATIC)
#  ADD_LIBRARY(qmcwfs STATIC ${WFBASE_SRCS} ${JASTROW_SRCS} ${FERMION_SRCS})
#ELSE(QMC_BUILD_STATIC)
#  ADD_LIBRARY(qmcwfs SHARED ${WFBASE_SRCS} ${JASTROW_SRCS} ${FERMION_SRCS})
#ENDIF(QMC_BUILD_STATIC)
