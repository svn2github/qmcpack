QMC_COMMON_SRCS = \
  ../QMCWaveFunctions/OrbitalBuilderBase.cpp\
  ../QMCWaveFunctions/AtomicOrbitals/HFAtomicSTOSet.cpp \
  ../QMCWaveFunctions/AtomicOrbitals/HFAtomicSTOSetBuilder.cpp \
  ../QMCWaveFunctions/AtomicOrbitals/HeSTOClementiRottie.cpp\
  ../QMCWaveFunctions/MolecularOrbitals/STOMolecularOrbitals.cpp\
  ../QMCWaveFunctions/MolecularOrbitals/GTOMolecularOrbitals.cpp \
  ../QMCWaveFunctions/MolecularOrbitals/RGFBuilderBase.cpp\
  ../QMCWaveFunctions/MolecularOrbitals/STO2GridBuilder.cpp\
  ../QMCWaveFunctions/MolecularOrbitals/GTO2GridBuilder.cpp \
  ../QMCWaveFunctions/MolecularOrbitals/Any2GridBuilder.cpp \
  ../QMCWaveFunctions/MolecularOrbitals/NumericalRGFBuilder.cpp\
  ../QMCWaveFunctions/MolecularOrbitals/GridMolecularOrbitals.cpp\
  ../QMCWaveFunctions/MolecularOrbitals/MolecularOrbitalBuilder.cpp\
  ../QMCWaveFunctions/MolecularOrbitals/NumericalMolecularOrbitals.cpp\
  ../QMCWaveFunctions/ElectronGasOrbitalBuilder.cpp \
  ../QMCHamiltonians/LocalPPotential.cpp\
  ../QMCHamiltonians/NonLocalPPotential.cpp\
  ../QMCHamiltonians/LocalCorePolPotential.cpp \
  ../LongRange/LPQHIBasis.cpp

QMC_DRIVERS = \
  ../QMCDrivers/SimpleFixedNodeBranch.cpp\
  ../QMCDrivers/QMCDriver.cpp\
  ../QMCDrivers/DummyQMC.cpp\
  ../QMCDrivers/VMC.cpp\
  ../QMCDrivers/VMCParticleByParticle.cpp\
  ../QMCDrivers/MolecuDMC.cpp\
  ../QMCDrivers/DMCParticleByParticle.cpp\
  ../QMCDrivers/WalkerControlBase.cpp\
  ../QMCDrivers/QMCOptimize.cpp\
  ../QMCDrivers/QMCCostFunction.cpp\
  ../QMCDrivers/VMCMultiple.cpp\
  ../QMCDrivers/VMCPbyPMultiple.cpp\
  ../Estimators/ScalarEstimatorManager.cpp\
  ../Estimators/MultipleEnergyEstimator.cpp \
  ../QMCDrivers/RQMCMultiple.cpp\
  ../QMCDrivers/MultiChain.cpp\
  ../QMCDrivers/MultiChainIO.cpp\
  ../QMCDrivers/PolymerEstimator.cpp\
  ../QMCDrivers/ReptationMC.cpp

QMC_APP =\
  ParticleSetPool.cpp\
  WaveFunctionPool.cpp\
  HamiltonianPool.cpp\
  InitMolecularSystem.cpp\
  QMCAppBase.cpp\
  QMCMain.cpp\
  qmcapp.cpp

bin_PROGRAMS = qmcapp
qmcapp_SOURCES = \
$(QMC_COMMON_SRCS) \
$(QMC_DRIVERS) \
$(QMC_APP)

INCLUDES = -I$(top_srcdir)/src
qmcapp_LDADD = @QMCBASE_LIBS@ 
qmcapp_LDFLAGS = @LDFLAGS@ 
qmcapp_DEPENDENCIES = ${top_builddir}/src/libqmcbase.a

