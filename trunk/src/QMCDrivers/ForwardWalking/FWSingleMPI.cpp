//////////////////////////////////////////////////////////////////
// (c) Copyright 2009- by Jeremy McMinis and Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Jeongnim Kim
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: jnkim@ncsa.uiuc.edu
//
// Supported by 
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#include "QMCDrivers/ForwardWalking/FWSingleMPI.h"
#include "QMCDrivers/ForwardWalkingStructure.h"

namespace qmcplusplus { 

  /// Constructor.
  FWSingleMPI::FWSingleMPI(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h, HamiltonianPool& hpool)
    : QMCDriver(w,psi,h), CloneManager(hpool), weightFreq(1), weightLength(1), fname(""), verbose(0)
      , WIDstring("WalkerID"), PIDstring("ParentID"),gensTransferred(1),startStep(0)
  { 
    RootName = "FW";
    QMCType ="FWSingleMPI";
    xmlrootName="";
    m_param.add(xmlrootName,"rootname","string");
    m_param.add(weightLength,"numbersteps","int");
    m_param.add(weightFreq,"skipsteps","int");
    m_param.add(verbose,"verbose","int");
    m_param.add(startStep,"ignore","int");
    hdf_ID_data.setID(WIDstring);
    hdf_PID_data.setID(PIDstring);
  }
  
  bool FWSingleMPI::run() 
  {  
    hdf_WGT_data.setFileName(xmlrootName); 
    if (myComm->rank()==0) 
    {
        Estimators->start(weightLength,1); 
        fillIDMatrix();
        hdf_WGT_data.makeFile();
        hdf_WGT_data.openFile();
        hdf_WGT_data.addFW(0);
        for (int i=0;i<Weights.size();i++) hdf_WGT_data.addStep(i,Weights[i]);
        hdf_WGT_data.closeFW();
        hdf_WGT_data.closeFile();
      for(int ill=1;ill<weightLength;ill++)
      {
        transferParentsOneGeneration();
        FWOneStep();
  //       WeightHistory.push_back(Weights);
        hdf_WGT_data.openFile();
        hdf_WGT_data.addFW(ill);
        for (int i=0;i<Weights.size();i++) hdf_WGT_data.addStep(i,Weights[i]);
        hdf_WGT_data.closeFW();
        hdf_WGT_data.closeFile();
      }
      if (verbose>0) app_log()<<" Done Computing Weights"<<endl;
    }
    int nprops = H.sizeOfObservables();//local energy, local potnetial and all hamiltonian elements
    int FirstHamiltonian = H.startIndex(); 

    int nelectrons = W[0]->R.size();
    int nfloats=OHMMS_DIM*nelectrons; 
    makeClones(W,Psi,H);
    
    vector<ForwardWalkingData* > FWvector;
    for(int ip=0; ip<NumThreads; ip++) FWvector.push_back(new ForwardWalkingData(nelectrons));
    
    hdf_OBS_data.setFileName(xmlrootName);
    if (myComm->rank()==0) hdf_OBS_data.makeFile();
    hdf_float_data.openFile(fname.str());

    int MPIoffset = myComm->rank();
    for(int step=MPIoffset;step<numSteps;step+=myComm->size())
    {
      hdf_float_data.setStep(step);
        
      
      vector<RealType> stepObservables(walkersPerBlock[step]*(nprops+2), 0);
      for(int wstep=0; wstep<walkersPerBlock[step];)
      {
        vector<float> ThreadsCoordinate(NumThreads*nfloats);
        int nwalkthread = hdf_float_data.getFloat(wstep*nfloats, (wstep+NumThreads)*nfloats, ThreadsCoordinate) / nfloats;
//         for(int j=0;j<ThreadsCoordinate.size();j++)cout<<ThreadsCoordinate[j]<<" ";
//         cout<<endl;
#pragma omp parallel for
        for(int ip=0; ip<nwalkthread; ip++) 
        {
          vector<float> SINGLEcoordinate(0);
          vector<float>::iterator TCB1(ThreadsCoordinate.begin()+ip*nfloats), TCB2(ThreadsCoordinate.begin()+(1+ip)*nfloats);
          
          SINGLEcoordinate.insert(SINGLEcoordinate.begin(),TCB1,TCB2);
          FWvector[ip]->fromFloat(SINGLEcoordinate);
          wClones[ip]->R=FWvector[ip]->Pos;
          wClones[ip]->update();
          RealType logpsi(psiClones[ip]->evaluateLog(*wClones[ip]));
          RealType eloc=hClones[ip]->evaluate( *wClones[ip] );
          hClones[ip]->auxHevaluate(*wClones[ip]);
          int indx=(wstep+ip)*(nprops+2);
          stepObservables[indx]= eloc;
          stepObservables[indx+1]= hClones[ip]->getLocalPotential();
          for(int i=0;i<nprops;i++) stepObservables[indx+i+2] = hClones[ip]->getObservable(i) ;
        }
        wstep+=nwalkthread;
      for(int ip=0; ip<NumThreads; ip++)  wClones[ip]->resetCollectables();
      }
      hdf_OBS_data.openFile();
      hdf_OBS_data.addStep(step, stepObservables);
      hdf_OBS_data.closeFile();
//       savedValues.push_back(stepObservables);
      hdf_float_data.endStep();
      if (verbose >1) cout<<"Done with step: "<<step<<endl;
    }
    if (myComm->rank()==0) 
    {
      vector<int> Dimensions(3);
      
      hdf_WGT_data.openFile();
      hdf_OBS_data.openFile();
      for(int ill=0;ill<weightLength;ill++)
      {    
        Dimensions[0]=ill;
        Dimensions[1]=(nprops+2);
        Dimensions[2]=numSteps;
        Estimators->startBlock(1);
        Estimators->accumulate(hdf_OBS_data,hdf_WGT_data,Dimensions);
        Estimators->stopBlock(getNumberOfSamples(ill));
      }
      hdf_OBS_data.closeFile();
      hdf_WGT_data.closeFile();
      Estimators->stop();
    }
    return true;
  }

  int FWSingleMPI::getNumberOfSamples(int omittedSteps)
  {
    int returnValue(0);
    for(int i=startStep;i<(numSteps-omittedSteps);i++) returnValue +=walkersPerBlock[i];
    return returnValue;
  }

  void FWSingleMPI::fillIDMatrix()
  {
    if (verbose>0) app_log()<<" There are "<<numSteps<<" steps"<<endl;
    IDs.resize(numSteps); PIDs.resize(numSteps); Weights.resize(numSteps);
    vector<vector<long> >::iterator stepIDIterator(IDs.begin());
    vector<vector<long> >::iterator stepPIDIterator(PIDs.begin());
    int st(0);
    hdf_ID_data.openFile(fname.str());
    hdf_PID_data.openFile(fname.str());
    do
    {
      hdf_ID_data.readAll(st,*(stepIDIterator));
      hdf_PID_data.readAll(st,*(stepPIDIterator));
      walkersPerBlock.push_back( (*stepIDIterator).size() );
      stepIDIterator++; stepPIDIterator++; st++;
      if (verbose>1) app_log()<<"step:"<<st<<endl;
    } while (st<numSteps);
    hdf_ID_data.closeFile();
    hdf_PID_data.closeFile();
    //     Weights.resize( IDs.size());
    for(int i=0;i<numSteps;i++) Weights[i].resize(IDs[i].size(),1);
    realPIDs = PIDs;
    realIDs = IDs;
  }


  void FWSingleMPI::resetWeights()
  {
    if (verbose>2) app_log()<<" Resetting Weights"<<endl;
    Weights.clear();
    Weights.resize(numSteps);
    for(int i=0;i<numSteps;i++) Weights[i].resize(IDs[i].size(),0);
  }

  void FWSingleMPI::FWOneStep()
  {
    //create an ordered version of the ParentIDs to make the weight calculation faster
    vector<vector<long> > orderedPIDs=(PIDs);
    vector<vector<long> >::iterator it(orderedPIDs.begin());
    do
    {
      std::sort((*it).begin(),(*it).end());
      it++;
    } while(it<orderedPIDs.end());

    if (verbose>2) app_log()<<" Done Sorting IDs"<<endl;
    resetWeights();
    vector<vector<long> >::iterator stepIDIterator(IDs.begin());
    vector<vector<long> >::iterator stepPIDIterator(orderedPIDs.begin() + gensTransferred);
    vector<vector<int> >::iterator stepWeightsIterator(Weights.begin());
    //we start comparing the next generations ParentIDs with the current generations IDs
    int i=0;
    do
    {
      if (verbose>2) app_log()<<"  calculating weights for gen:"<<gensTransferred<<" step:"<<i<<"/"<<orderedPIDs.size()<<endl;
      //       if (verbose>2) app_log()<<"Nsamples ="<<(*stepWeightsIteratoetWeights).size()<<endl;

      vector<long>::iterator IDit( (*stepIDIterator).begin()     );
      vector<long>::iterator PIDit( (*stepPIDIterator).begin()   );
      vector<int>::iterator  Wit( (*stepWeightsIterator).begin() );
      if (verbose>2) app_log()<<"ID size:"<<(*stepIDIterator).size()<<" PID size:"<<(*stepPIDIterator).size()<<" Weight size:"<<(*stepWeightsIterator).size()<<endl;

      do
      {
        if ((*PIDit)==(*IDit))
        {
          (*Wit)++;
          PIDit++;
        }
        else 
        {
          IDit++;
          Wit++;
          if (IDit==(*stepIDIterator).end())
          {
            IDit=(*stepIDIterator).begin();
            Wit=(*stepWeightsIterator).begin();
          }
        }
      }while(PIDit<(*stepPIDIterator).end());
      //       if (verbose>2) { printIDs((*stepIDIterator));printIDs((*stepPIDIterator));}
      //       if (verbose>2) printInts((*stepWeightsIterator));
      stepIDIterator++; stepPIDIterator++; stepWeightsIterator++; i++;

    }while(stepPIDIterator<orderedPIDs.end());

  }

  void FWSingleMPI::printIDs(vector<long> vi)
  {
    for (int j=0;j<vi.size();j++) app_log()<<vi[j]<<" ";
    app_log()<<endl;
  }
  void FWSingleMPI::printInts(vector<int> vi)
  {
    for (int j=0;j<vi.size();j++) app_log()<<vi[j]<<" ";
    app_log()<<endl;
  }

  void FWSingleMPI::transferParentsOneGeneration( )
  {

    vector<vector<long> >::reverse_iterator stepIDIterator(IDs.rbegin());
    vector<vector<long> >::reverse_iterator stepPIDIterator(PIDs.rbegin()), nextStepPIDIterator(realPIDs.rbegin());
    stepIDIterator+=gensTransferred;
    nextStepPIDIterator+=gensTransferred;
    int i(0);
    do
    {
      vector<long>::iterator hereID( (*stepIDIterator).begin() ) ;
      vector<long>::iterator nextStepPID( (*nextStepPIDIterator).begin() );
      vector<long>::iterator herePID( (*stepPIDIterator).begin() );
      if (verbose>2) app_log()<<"  calculating Parent IDs for gen:"<<gensTransferred<<" step:"<<i<<"/"<<PIDs.size()-gensTransferred<<endl;
      if (verbose>2) {printIDs((*nextStepPIDIterator));printIDs((*stepIDIterator));printIDs((*stepPIDIterator));}
      do
      {
        if ((*herePID)==(*hereID)) 
        {
          (*herePID)=(*nextStepPID);
          herePID++;
        }
        else
        {
          hereID++; nextStepPID++;
          if (hereID==(*stepIDIterator).end()) 
          {
            hereID=(*stepIDIterator).begin();
            nextStepPID=(*nextStepPIDIterator).begin();
            //             if (verbose>2) app_log()<<"resetting to beginning of parents"<<endl;
          }
        }
      }while(herePID<(*stepPIDIterator).end());
      stepIDIterator++;nextStepPIDIterator++;stepPIDIterator++;i++;
    }while(stepIDIterator<IDs.rend());
    gensTransferred++; //number of gens backward to compare parents
    if (verbose>2) app_log()<<"  Finished generation block"<<endl;
  }


  bool 
    FWSingleMPI::put(xmlNodePtr q){

      fname<<xmlrootName<<".storeConfig.h5";
      c_file = H5Fopen(fname.str().c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
      hsize_t numGrps = 0;
      H5Gget_num_objs(c_file, &numGrps);
      numSteps = static_cast<int> (numGrps)-3;
      app_log()<<"  Total number of steps in input file "<<numSteps<<endl;
      if (weightFreq<1) weightFreq=1;
      int numberDataPoints = weightLength/weightFreq;
      //     pointsToCalculate.resize(numberDataPoints);
      //     for(int i=0;i<numberDataPoints;i++) pointsToCalculate[i]=i*weightFreq;
      app_log()<<"  "<<numberDataPoints<<" sets of observables will be calculated each "<<weightFreq<<" steps"<<endl;
      app_log()<<"  Config Generations skipped for thermalization: "<<startStep<<endl;//<<" steps. At: ";
      //     for(int i=0;i<numberDataPoints;i++) app_log()<<pointsToCalculate[i]<<" ";
      app_log()<<endl;
      if (H5Fclose(c_file)>-1) c_file=-1;

      return true;
    }
}

/***************************************************************************
 * $RCSfile: VMCParticleByParticle.cpp,v $   $Author: jnkim $
 * $Revision: 1.25 $   $Date: 2006/10/18 17:03:05 $
 * $Id: VMCParticleByParticle.cpp,v 1.25 2006/10/18 17:03:05 jnkim Exp $ 
 ***************************************************************************/
