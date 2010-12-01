//////////////////////////////////////////////////////////////////
// (c) Copyright 2005- by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Jeongnim Kim
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: jnkim@ncsa.uiuc.edu
//   Tel:    217-244-6319 (NCSA) 217-333-3324 (MCC)
//
// Supported by 
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#include "QMCDrivers/QMCOptimize.h"
#include "Particle/HDFWalkerIO.h"
#include "Particle/DistanceTable.h"
#include "OhmmsData/AttributeSet.h"
#include "Message/CommOperators.h"
#include "Optimize/CGOptimization.h"
#include "Optimize/testDerivOptimization.h"
#include "Optimize/DampedDynamics.h"
#include "QMCDrivers/VMC/VMCSingle.h"
#include "QMCDrivers/QMCCostFunctionSingle.h"
#if defined(ENABLE_OPENMP)
#include "QMCDrivers/VMC/VMCSingleOMP.h"
#include "QMCDrivers/QMCCostFunctionOMP.h"
#endif
#if defined(QMC_CUDA)
  #include "QMCDrivers/VMC/VMC_CUDA.h"
  #include "QMCDrivers/QMCCostFunctionCUDA.h"
#endif
#include "QMCApp/HamiltonianPool.h"

namespace qmcplusplus {

  QMCOptimize::QMCOptimize(MCWalkerConfiguration& w,
      TrialWaveFunction& psi, QMCHamiltonian& h, HamiltonianPool& hpool, WaveFunctionPool& ppool): QMCDriver(w,psi,h,ppool),
      PartID(0), NumParts(1), WarmupBlocks(10), 
      SkipSampleGeneration("no"), hamPool(hpool),
      optTarget(0), optSolver(0), vmcEngine(0),
      wfNode(NULL), optNode(NULL)
  { 
    //set the optimization flag
    QMCDriverMode.set(QMC_OPTIMIZE,1);
    //read to use vmc output (just in case)
    RootName = "pot";
    QMCType ="QMCOptimize";
    //default method is cg
    optmethod = "cg";
    m_param.add(WarmupBlocks,"warmupBlocks","int");
    m_param.add(SkipSampleGeneration,"skipVMC","string");
  }

  /** Clean up the vector */
  QMCOptimize::~QMCOptimize() 
  { 
    delete vmcEngine;
    delete optSolver;
    delete optTarget;
  }

  /** Add configuration files for the optimization
   * @param a root of a hdf5 configuration file
   */
  void QMCOptimize::addConfiguration(const string& a) {
    if(a.size()) ConfigFile.push_back(a);
  }

  /** Reimplement QMCDriver::run
   */
  bool
  QMCOptimize::run() 
  {
    optTarget->initCommunicator(myComm);

    //close files automatically generated by QMCDriver
    //branchEngine->finalize();

    //generate samples
    generateSamples();

    //cleanup walkers
    //W.destroyWalkers(W.begin(), W.end());

    app_log() << "<opt stage=\"setup\">" << endl;
    app_log() << "  <log>"<<endl;

    //reset the rootname
    optTarget->setRootName(RootName);
    optTarget->setWaveFunctionNode(wfNode);

    app_log() << "   Reading configurations from h5FileRoot " << endl;
    //get configuration from the previous run
    Timer t1;

    optTarget->getConfigurations(h5FileRoot);
    optTarget->checkConfigurations();

    app_log() << "  Execution time = " << t1.elapsed() << endl;
    app_log() << "  </log>"<<endl;
    app_log() << "</opt>" << endl;

    app_log() << "<opt stage=\"main\" walkers=\""<< optTarget->getNumSamples() << "\">" << endl;
    app_log() << "  <log>" << endl;

    optTarget->setTargetEnergy(branchEngine->getEref());

    t1.restart();
    bool success=optSolver->optimize(optTarget);
//     W.reset();
//     branchEngine->flush(0);
//     branchEngine->reset();
    app_log() << "  Execution time = " << t1.elapsed() << endl;;
    app_log() << "  </log>" << endl;
    optTarget->reportParameters();
    app_log() << "</opt>" << endl;
    app_log() << "</optimization-report>" << endl;

    MyCounter++;

    return (optTarget->getReportCounter() > 0);
  }

  void QMCOptimize::generateSamples() 
  {
    Timer t1;
    app_log() << "<optimization-report>" << endl;
    //if(WarmupBlocks) 
    //{
    //  app_log() << "<vmc stage=\"warm-up\" blocks=\"" << WarmupBlocks << "\">" << endl;
    //  //turn off QMC_OPTIMIZE
    //  vmcEngine->setValue("blocks",WarmupBlocks);
    //  vmcEngine->QMCDriverMode.set(QMC_WARMUP,1);
    //  vmcEngine->run();
    //  vmcEngine->setValue("blocks",nBlocks);
    //  app_log() << "  Execution time = " << t1.elapsed() << endl;
    //  app_log() << "</vmc>" << endl;
    //}

    if(W.getActiveWalkers()>NumOfVMCWalkers)
    {
      W.destroyWalkers(W.getActiveWalkers()-NumOfVMCWalkers);
      app_log() << "  QMCOptimize::generateSamples removed walkers." << endl;
      app_log() << "  Number of Walkers per node " << W.getActiveWalkers() << endl;
    }

    vmcEngine->QMCDriverMode.set(QMC_OPTIMIZE,1);
    vmcEngine->QMCDriverMode.set(QMC_WARMUP,0);

    //vmcEngine->setValue("recordWalkers",1);//set record 
    vmcEngine->setValue("current",0);//reset CurrentStep
    app_log() << "<vmc stage=\"main\" blocks=\"" << nBlocks << "\">" << endl;
    t1.restart();
//     W.reset();
//     branchEngine->flush(0);
//     branchEngine->reset();
    vmcEngine->run();
    app_log() << "  Execution time = " << t1.elapsed() << endl;
    app_log() << "</vmc>" << endl;

    //branchEngine->Eref=vmcEngine->getBranchEngine()->Eref;
    branchEngine->setTrialEnergy(vmcEngine->getBranchEngine()->getEref());
    //set the h5File to the current RootName
    h5FileRoot=RootName;
  }

  /** Parses the xml input file for parameter definitions for the wavefunction optimization.
   * @param q current xmlNode 
   * @return true if successful
   */
  bool
  QMCOptimize::put(xmlNodePtr q) {

    string vmcMove("pbyp");
    string useGPU("no");
    OhmmsAttributeSet oAttrib;
    oAttrib.add(vmcMove,"move");
    oAttrib.add(useGPU,"gpu");
    oAttrib.put(q);

    xmlNodePtr qsave=q;
    xmlNodePtr cur=qsave->children;
    int pid=OHMMS::Controller->rank();
    while(cur != NULL) {
      string cname((const char*)(cur->name));
      if(cname == "mcwalkerset") {
        mcwalkerNodePtr.push_back(cur);
      } else if(cname == "optimizer") {
        xmlChar* att= xmlGetProp(cur,(const xmlChar*)"method");
        if(att) { optmethod = (const char*)att; }
        optNode=cur;
      } else if(cname == "optimize") {
        xmlChar* att= xmlGetProp(cur,(const xmlChar*)"method");
        if(att) { optmethod = (const char*)att; }
      }
      cur=cur->next;
    }  

    //no walkers exist, add 10
    if(W.getActiveWalkers() == 0) addWalkers(omp_get_max_threads()); 

    NumOfVMCWalkers=W.getActiveWalkers();

    //create VMC engine
    if(vmcEngine ==0) {
#if defined (QMC_CUDA)
      if (useGPU == "yes")
	vmcEngine = new VMCcuda(W,Psi,H);
      else
#endif
//#if defined(ENABLE_OPENMP)
//      if(omp_get_max_threads()>1)
//        vmcEngine = new VMCSingleOMP(W,Psi,H,hamPool);
//      else
//#endif
//        vmcEngine = new VMCSingle(W,Psi,H);
      vmcEngine = new VMCSingleOMP(W,Psi,H,hamPool,psiPool);
      vmcEngine->setUpdateMode(vmcMove[0] == 'p');
      vmcEngine->initCommunicator(myComm);
    }

    vmcEngine->setStatus(RootName,h5FileRoot,AppendRun);
    vmcEngine->process(qsave);

    if(optSolver ==0)
    {
      if(optmethod == "anneal") 
      {
        app_log() << " Annealing optimization using DampedDynamics"<<endl;
        optSolver = new DampedDynamics<RealType>;
      }  
      else if((optmethod == "flexOpt")  |(optmethod == "flexopt")  | (optmethod == "macopt") )
      {
        app_log() << "Conjugate-gradient optimization using FlexOptimization"<<endl;
        app_log() << " This method has been removed. "<< endl;
        abort();
      } 
      else if (optmethod == "BFGS") 
      {
        app_log() << " This method is not implemented correctly yet. "<< endl;
        abort();
      } 
      else if (optmethod == "test")
      {
        app_log() << "Conjugate-gradient optimization using tester Optimization: "<<endl;
        optSolver = new testDerivOptimization<RealType>;
      }
      else
      {
        app_log() << " Conjugate-gradient optimization using CGOptimization"<<endl;
        optSolver = new CGOptimization<RealType>;
      }      //set the stream
      optSolver->setOstream(&app_log());
    }

    if(optNode == NULL) 
      optSolver->put(qmcNode);
    else
      optSolver->put(optNode);

    bool success=true;
    if(optTarget == 0) 
    {
#if defined (QMC_CUDA)
      if (useGPU == "yes")
	optTarget = new QMCCostFunctionCUDA(W,Psi,H,hamPool);
      else
#endif
#if defined(ENABLE_OPENMP)
	if(true /*omp_get_max_threads()>1*/)
      {
        optTarget = new QMCCostFunctionOMP(W,Psi,H,hamPool);
      }
      else
#endif
        optTarget = new QMCCostFunctionSingle(W,Psi,H);

      optTarget->setStream(&app_log());
      success=optTarget->put(q);
    }
    return success;
  }
  }
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1286 $   $Date: 2006-08-17 12:33:18 -0500 (Thu, 17 Aug 2006) $
 * $Id: QMCOptimize.cpp 1286 2006-08-17 17:33:18Z jnkim $ 
 ***************************************************************************/
