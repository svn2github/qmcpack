//////////////////////////////////////////////////////////////////
// (c) Copyright 2003- by Jeongnim Kim and Simone Chiesa
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
#include "QMCDrivers/VMC/VMCPbyPMultiWarp.h"
#include "Utilities/OhmmsInfo.h"
#include "Particle/MCWalkerConfiguration.h"
#include "Particle/HDFWalkerIO.h"
#include "ParticleBase/ParticleAttribOps.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "Estimators/MultipleEnergyEstimator.h"
#include "QMCApp/ParticleSetPool.h"
#include "QMCDrivers/DriftOperators.h"

namespace qmcplusplus { 

  /// Constructor.
  VMCPbyPMultiWarp::VMCPbyPMultiWarp(MCWalkerConfiguration& w, 
					       TrialWaveFunction& psi, 
					       QMCHamiltonian& h,
                                               ParticleSetPool& ptclPool):
    QMCDriver(w,psi,h), PtclPool(ptclPool){ 
    RootName = "vmc-PbyP-warp";
    QMCType ="vmc-PbyP-warp";
    QMCDriverMode.set(QMC_UPDATE_MODE,1);
    QMCDriverMode.set(QMC_MULTIPLE,1);
    equilBlocks=-1;
    m_param.add(equilBlocks,"equilBlocks","int");
    refSetName="invalid";
    m_param.add(refSetName,"reference","str");
    add_H_and_Psi(&h,&psi);
  }

  VMCPbyPMultiWarp::~VMCPbyPMultiWarp() {
    for(int i=0; i<G.size(); i++) delete G[i];
    for(int i=0; i<dL.size(); i++) delete dL[i];
  }
  
  bool VMCPbyPMultiWarp::run() { 

    vector<RealType> new_Jacobian(nPsi);

    //TEST CACHE
    //Estimators->reportHeader(AppendRun);
      
    //going to add routines to calculate how much we need
    //bool require_register =  W.createAuxDataSet();
    bool require_register =  W[0]->DataSet.size();

    vector<RealType>  Norm(nPsi),tmpNorm(nPsi);
    if(equilBlocks > 0){
      for(int ipsi=0; ipsi< nPsi; ipsi++){
        Norm[ipsi]=1.0; 
        tmpNorm[ipsi]=0.0;
      }
    }else{
      for(int ipsi=0; ipsi< nPsi; ipsi++) Norm[ipsi]=std::exp(branchEngine->LogNorm[ipsi]);
    }

    multiEstimator->initialize(W,WW,PtclWarp,H1,Psi1,Tau,Norm,require_register);

    //TEST CACHE
    //Estimators->reset();
    Estimators->start(nBlocks);
    //TEST CACHE

    IndexType block = 0;
    
    m_oneover2tau = 0.5/Tau;
    m_sqrttau = std::sqrt(Tau);
    RealType nPsi_minus_one = nPsi-1;

    //ParticleSet::ParticleGradient_t dG(W.getTotalNum());
    dG.resize(W.getTotalNum());

    IndexType nAcceptTot = 0;
    IndexType nRejectTot = 0;

    MCWalkerConfiguration::iterator it;
    MCWalkerConfiguration::iterator it_end(W.end());

    do {  //Blocks loop
      IndexType step = 0;
      nAccept = 0; nReject=0;
      IndexType nAllRejected = 0;

      Estimators->startBlock(nSteps);
      do {  //Steps loop
        it = W.begin();	 
        int iwalker=0; 
        while(it != it_end) {  //Walkers loop

          Walker_t& thisWalker(**it);

          Walker_t::Buffer_t& w_buffer(thisWalker.DataSet);

          W.R = thisWalker.R;
          w_buffer.rewind();
	  // Copy walker info in W
          W.copyFromBuffer(w_buffer);
          for(int ipsi=0; ipsi<nPsi; ipsi++)
            WW[ipsi]->copyFromBuffer(w_buffer);
          PtclWarp.copyFromBuffer(w_buffer);

          for(int ipsi=0; ipsi<nPsi; ipsi++){
	    // Copy wave function info in W and Psi1
	    Psi1[ipsi]->copyFromBuffer(*WW[ipsi],w_buffer);  
	    Psi1[ipsi]->G=WW[ipsi]->G;
	    Psi1[ipsi]->L=WW[ipsi]->L;
          }

	  // Point to the correct walker in the ratioij buffer
	  RealType *ratioijPtr=multiEstimator->RatioIJ[iwalker];

          //create a 3N-Dimensional Gaussian with variance=1
          makeGaussRandom(deltaR);
          bool moved = false;

          for(int iat=0; iat<W.getTotalNum(); iat++) {  //Particles loop

            PosType dr = m_sqrttau*deltaR[iat]+thisWalker.Drift[iat];
            PosType newpos = W.makeMove(iat,dr);
            /*
            cout << "========================" << endl;
            cout << "PARTICLE " << iat << endl;
            cout << "DRIFT        " << thisWalker.Drift[iat]<< endl;
            cout << "NEW POSITION " << W.R[iat] << endl;
            */
            //Compute the displacement due to space-warp
            PtclWarp.warp_one(iat,1);

	    for(int ipsi=0; ipsi<nPsi; ipsi++){
              //if(ipsi) { //need to call makeMove for the secondary particles
                dr=newpos+PtclWarp.get_displacement(iat,ipsi)-WW[ipsi]->R[iat];
                WW[ipsi]->makeMove(iat,dr);
              //}
              //Warp the particle position
              //WW[ipsi]->R[iat]=W.R[iat]+PtclWarp.get_displacement(ipsi);
	      // Compute ratios before and after the move
	      ratio[ipsi] = Psi1[ipsi]->ratio(*WW[ipsi],iat,dG,*dL[ipsi]); 
              logpsi2[ipsi]=std::log(ratio[ipsi]*ratio[ipsi]);
	      // Compute Gradient in new position
              *G[ipsi]=Psi1[ipsi]->G + dG;
	      // Initialize: sumratio[i]=(Psi[i]/Psi[i])^2=1.0
              new_Jacobian[ipsi]=thisWalker.Properties(ipsi,JACOBIAN)
                *PtclWarp.get_Jacobian(iat,ipsi)/PtclWarp.one_ptcl_Jacob(ipsi,iat);
	      sumratio[ipsi]=1.e0;
	    }
            /*
            for(int ipsi=0; ipsi<nPsi; ipsi++){
              cout << "NEW WARP POSITION " << ipsi << " " << WW[ipsi]->R[iat]<<endl;
            }
            for(int ipsi=0; ipsi<nPsi; ipsi++){
              cout << "NEW JACOBIAN " << ipsi << " " << new_Jacobian[ipsi]<<endl;
            }
            */

	    // Compute new (Psi[i]/Psi[j])^2 and their sum
	    int indexij(0);
	    for(int ipsi=0; ipsi< nPsi_minus_one; ipsi++){
	      for(int jpsi=ipsi+1; jpsi < nPsi; jpsi++, indexij++){
                //Ratio between Norm already in ratioijPtr from MultiEstimator->initialize.
                ratioij[indexij]=std::exp(logpsi2[jpsi]-logpsi2[ipsi])*ratioijPtr[indexij];
                RealType rji=ratioij[indexij]*new_Jacobian[jpsi]/new_Jacobian[ipsi];
		sumratio[ipsi] += rji;
		sumratio[jpsi] += 1.e0/rji;
	      }
	    }
            /*
            for(int ipsi=0; ipsi<nPsi; ipsi++){
              cout << "WF RATIO " << ipsi << " " << ratio[ipsi] << endl;
            }
            for(int ipsi=0; ipsi<nPsi; ipsi++){
              cout << "NEW GRADIENT " << ipsi << " " << (*G[ipsi])[iat]<<endl;
            }
            for(int ipsi=0; ipsi<nPsi; ipsi++){
              cout << "NEW LAPLACIAN " << ipsi << " " << (Psi1[ipsi]->L[iat]+ (*dL[ipsi])[iat]) <<endl;
            }
            */

	    // Evaluate new Umbrella Weight and new drift
            RealType logGf = -0.5*dot(deltaR[iat],deltaR[iat]);
            
            /* START COMMENT
	    drift=0.0;
            QMCTraits::PosType WarpDrift;
            RealType denom(0.e0);
            for(int ipsi=0; ipsi< nPsi ; ipsi++) {
              invsumratio[ipsi]=1.e0/sumratio[ipsi];
              denom += invsumratio[ipsi];
              for(int iptcl=0; iptcl< W.getTotalNum(); iptcl++){
                WarpDrift=dot(  (*G[ipsi])[iptcl],PtclWarp.get_Jacob_matrix(iptcl,ipsi)  )
                  +5.0e-1*PtclWarp.get_grad_ln_Jacob(iptcl,ipsi) ;
                drift[iptcl] += (invsumratio[ipsi]*WarpDrift);
              }
            } 
            
            //drift *= (Tau/denom);
            denom = Tau/denom;
            drift *= denom;
            END COMMENT*/
            //START NEW
            for(int ipsi=0; ipsi< nPsi ;ipsi++)
	      invsumratio[ipsi]=1.0/sumratio[ipsi];
            /*
            for(int ipsi=0; ipsi<nPsi; ipsi++){
              cout << "NEW WEIGHT " << ipsi << " " << invsumratio[ipsi]<<endl;
            }
            */
            setScaledDrift(Tau,*G[0],drift);
            /*
            cout << "NEXT GRAD " << "0" << " " << (*G[0])[iat+1]<<endl;
            cout << "NEXT DRIFT " << "0" << " " << drift[iat+1]<<endl;
            */
            drift*=invsumratio[0];
            for(int ipsi=1; ipsi< nPsi ;ipsi++) {               		
              setScaledDrift(Tau,*G[ipsi],dG);
              /*
              cout << "NEXT GRAD " << ipsi << " " << (*G[ipsi])[iat+1]<<endl;
              cout << "NEXT DRIFT " << ipsi << " " << dG[iat+1]<<endl;
              */
#ifndef QMC_COMPLEX
              drift+= (invsumratio[ipsi]*dG);
#else
              app_error() << " Operation is not implemented." << endl;  
#endif
            }
            //END NEW


            dr = thisWalker.R[iat]-newpos-drift[iat];
            RealType logGb = -m_oneover2tau*dot(dr,dr);

	    // td = Target Density ratio
	    //RealType td=pow(ratio[0],2)*sumratio[0]/(*it)->Properties(SUMRATIO);
	    RealType td=ratio[0]*ratio[0]*
              new_Jacobian[0]/thisWalker.Properties(0,JACOBIAN)* //This two are 1 when reference system is system 0
              sumratio[0]/(*it)->Multiplicity;
	    RealType prob = td*std::exp(logGb-logGf);

	    if(Random() < prob) { 
              //cout << "ACCEPTED" << endl << endl;
	      /* Electron move is accepted. Update:
		 -ratio (Psi[i]/Psi[j])^2 for this walker
		 -Gradient and laplacian for each Psi1[i]
		 -Drift
		 -buffered info for each Psi1[i]*/
	      moved = true;
	      ++nAccept;

	      W.acceptMove(iat);
              for(int ipsi=0; ipsi<nPsi; ipsi++){
                // Update warped position
                WW[ipsi]->acceptMove(iat);
              }
              // Update single particle jacobian
              PtclWarp.update_one_ptcl_Jacob(iat);
	      // Update Buffer for (Psi[i]/Psi[j])^2 
	      std::copy(ratioij.begin(),ratioij.end(),ratioijPtr);
	      // Update Umbrella weight
              UmbrellaWeight=invsumratio;
	      // Store sumratio for next Accept/Reject step to Multiplicity
	      //thisWalker.Properties(SUMRATIO)=sumratio[0];
	      thisWalker.Multiplicity=sumratio[0];
	      for(int ipsi=0; ipsi< nPsi; ipsi++){
		////Update local Psi1[i] buffer for the next move
		Psi1[ipsi]->acceptMove(*WW[ipsi],iat);  
		// Update G and L in Psi1[i]
		Psi1[ipsi]->G = *G[ipsi];
		Psi1[ipsi]->L += *dL[ipsi];
                thisWalker.Properties(ipsi,LOGPSI)+=std::log(abs(ratio[ipsi]));
                thisWalker.Properties(ipsi,JACOBIAN)=new_Jacobian[ipsi];
	      }
	      // Update Drift
	      (*it)->Drift = drift;
	    } else {
              //cout << "REJECTED" << endl << endl;
	      ++nReject; 
	      W.rejectMove(iat);
              // reject moves
              for(int ipsi=0; ipsi<nPsi; ipsi++) 
                WW[ipsi]->rejectMove(iat);
	      for(int ipsi=0; ipsi< nPsi; ipsi++)
		Psi1[ipsi]->rejectMove(iat);
	    }
	  }

	  if(moved) {
	    /* The walker moved: Info are copied back to buffers:
	       -copy (Psi[i]/Psi[j])^2 to ratioijBuffer
	       -Gradient and laplacian for each Psi1[i]
	       -Drift
	       -buffered info for each Psi1[i]
	       Physical properties are updated */
            (*it)->Age=0;
	    (*it)->R = W.R;

	    w_buffer.rewind();
            W.copyToBuffer(w_buffer);
            for(int ipsi=0; ipsi< nPsi; ipsi++) WW[ipsi]->copyToBuffer(w_buffer);
            PtclWarp.copyToBuffer(w_buffer);

	    for(int ipsi=0; ipsi< nPsi; ipsi++){
	      WW[ipsi]->G=Psi1[ipsi]->G;
	      WW[ipsi]->L=Psi1[ipsi]->L;
	      //RealType psi = Psi1[ipsi]->evaluate(*WW[ipsi],w_buffer);
	      RealType logpsi = Psi1[ipsi]->evaluateLog(*WW[ipsi],w_buffer);
	      RealType et = H1[ipsi]->evaluate(*WW[ipsi]);

	      //multiEstimator->updateSample(iwalker,ipsi,et,UmbrellaWeight[ipsi]);
              //Properties is used for UmbrellaWeight and UmbrellaEnergy
              thisWalker.Properties(ipsi,UMBRELLAWEIGHT)=UmbrellaWeight[ipsi];
              thisWalker.Properties(ipsi,LOCALENERGY)=et;
              //thisWalker.Properties(ipsi,LOGPSI)=std::log(abs(psi));
              thisWalker.Properties(ipsi,LOGPSI)=logpsi;
              thisWalker.Properties(ipsi,SIGN)=Psi1[ipsi]->getPhase();

              H1[ipsi]->saveProperty(thisWalker.getPropertyBase(ipsi));
	    }
	  }
	  else {
	    ++nAllRejected;
	  }
	  ++it; ++iwalker;
	}
	++step;++CurrentStep;
	Estimators->accumulate(W);
      } while(step<nSteps);

      //Modify Norm. 
      if(block < equilBlocks){
        for(int ipsi=0; ipsi< nPsi; ipsi++){
          //cout << "WGT " << multiEstimator->esum(ipsi,MultipleEnergyEstimator::WEIGHT_INDEX) << endl;
          tmpNorm[ipsi]+=multiEstimator->esum(ipsi,MultipleEnergyEstimator::WEIGHT_INDEX);
        }
        if(block==(equilBlocks-1) || block==(nBlocks-1)){
          RealType SumNorm(0.e0);
          for(int ipsi=0; ipsi< nPsi; ipsi++) SumNorm+=tmpNorm[ipsi];
          for(int ipsi=0; ipsi< nPsi; ipsi++){
            Norm[ipsi]=tmpNorm[ipsi]/SumNorm;
            branchEngine->LogNorm[ipsi]=std::log(Norm[ipsi]);
            //cout << "LOGNORM VMC " << branchEngine->LogNorm[ipsi] << endl;
          }
        }
      }

      Estimators->stopBlock(static_cast<RealType>(nAccept)/static_cast<RealType>(nAccept+nReject));

      nAcceptTot += nAccept;
      nRejectTot += nReject;

      nAccept = 0; nReject = 0;
      ++block;

      //record the current configuration
      recordBlock(block);

      //refresh the buffer
      multiEstimator->initialize(W,WW,PtclWarp,H1,Psi1,Tau,Norm,false);
    } while(block<nBlocks);

    //Need MPI-IO
    app_log()
      << "Ratio = " 
      << static_cast<RealType>(nAcceptTot)/static_cast<RealType>(nAcceptTot+nRejectTot)
      << endl;

    return finalize(block);
  }


  bool 
  VMCPbyPMultiWarp::put(xmlNodePtr q){
    //////////////////////////
    //Taken from VMCMultiple//
    //////////////////////////
    if(WW.empty()) {
      W.clearDistanceTables();
    }
    //qmcsystem
    vector<ParticleSet*> ionSets;
    DistanceTableData* dtableReference;
    xmlNodePtr cur=q->children;
    while(cur != NULL) {
      string cname((const char*)(cur->name));
      if(cname == "qmcsystem") {
	string source_name((const char*)xmlGetProp(cur,(const xmlChar*)"source"));
        ionSets.push_back(PtclPool.getParticleSet(source_name));
      }
      cur=cur->next;
    }

    ParticleSet* p(0);
    if(refSetName!="invalid"){
      p=PtclPool.getParticleSet(refSetName);
      if(p==0){
        cout << "The specified reference cannot be found. Stop." << endl;
        abort();
      }
    }else{
      refSetName=ionSets[0]->getName().c_str();
      p=PtclPool.getParticleSet(refSetName);
    }

    dtableReference=DistanceTable::add(*p,W);

    /*vector<DistanceTableData*> dtableList;
    string target_name(W.getName());
    xmlNodePtr cur=q->children;
    while(cur != NULL) {
      string cname((const char*)(cur->name));
      if(cname == "qmcsystem") {
        string source_name((const char*)xmlGetProp(cur,(const xmlChar*)"source"));
	dtableList.push_back(DistanceTable::getTable(source_name.c_str(),target_name.c_str()));
      }
      cur=cur->next;
    }*/

    nptcl=W.R.size();
    ///////////////////////////
    
    nPsi=Psi1.size();

    //PtclWarp.initialize(dtableList);
    PtclWarp.initialize(ionSets,dtableReference);
    JACOBIAN=W.addProperty("Jacobian");
    
    resize(nPsi,W.getTotalNum()); 
    PtclWarp.resize_one_ptcl_Jacob();

    if(branchEngine->LogNorm.size()==0)branchEngine->LogNorm.resize(nPsi);
    if(equilBlocks>0){
      for(int ipsi=0; ipsi<nPsi; ipsi++)branchEngine->LogNorm[ipsi]=0.e0;
    }
    //////////////////////////
    //for(int ipsi=0; ipsi<nPsi; ipsi++) H1[ipsi]->add2WalkerProperty(W);

    Estimators = branchEngine->getEstimatorManager();
    if(Estimators == 0) {
      Estimators = new EstimatorManager(myComm);
      multiEstimator = new MultipleEnergyEstimator(H,nPsi);
      Estimators->add(multiEstimator,"elocal");
    }

    //H1[0]->setPrimary(true);
    for(int ipsi=0; ipsi<nPsi; ipsi++) {
      H1[ipsi]->setPrimary(true);
    }

    //////////////////////////////
    //Taken from VMCMultipleWarp//
    //////////////////////////////

    if(WW.empty()){
      //WW.push_back(&W);
      char newname[128];
      //for(int ipsi=1; ipsi<nPsi; ipsi++){
      for(int ipsi=0; ipsi<nPsi; ipsi++){
	sprintf(newname,"%s%d", W.getName().c_str(),ipsi);
        ParticleSet* pclone=PtclPool.getParticleSet(newname);
        if(pclone == 0) {
          app_log() << "  Cloning particle set in VMCMultipleWarp " << newname << endl;
          pclone=new ParticleSet(W);
          pclone->setName(newname);
          PtclPool.addParticleSet(pclone);
        } else {
          app_log() << "  Cloned particle exists " << newname << endl;
        }
	WW.push_back(pclone);
	Psi1[ipsi]->resetTargetParticleSet(*WW[ipsi]);
        H1[ipsi]->resetTargetParticleSet(*WW[ipsi]);
      }
    }

    /*if(WW.empty()){
      WW.push_back(&W);
      char newname[128];
      for(int ipsi=1; ipsi<nPsi; ipsi++){
	sprintf(newname,"%s%d", W.getName().c_str(),ipsi);
        ParticleSet* pclone=PtclPool.getParticleSet(newname);
        if(pclone == 0) {
          app_log() << "  Cloning particle set in VMCMultipleWarp " << newname << endl;
          pclone=new ParticleSet(W);
          pclone->setName(newname);
          PtclPool.addParticleSet(pclone);
        } else {
          app_log() << "  Cloned particle exists " << newname << endl;
        }
	//Correct copy constructor????????
	WW.push_back(pclone);
	WW[ipsi]=pclone;
	Psi1[ipsi]->resetTargetParticleSet(*WW[ipsi]);
        H1[ipsi]->resetTargetParticleSet(*WW[ipsi]);
	//Psi1[ipsi]->resetTargetParticleSet(W);
        //H1[ipsi]->resetTargetParticleSet(W
        //WW[ipsi]->setUpdateMode(MCWalkerConfiguration::Update_Particle);
      }
    }*/
    
    return true;
  }
}

/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 1593 $   $Date: 2007-01-04 17:23:27 -0600 (Thu, 04 Jan 2007) $
 * $Id: VMCPbyPMultiWarp.cpp 1593 2007-01-04 23:23:27Z jnkim $ 
 ***************************************************************************/
