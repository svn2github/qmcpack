//////////////////////////////////////////////////////////////////
// (c) Copyright 2003- by Jeongnim Kim
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
#include "Configuration.h"
#include "Message/Communicate.h"
#include "Utilities/OhmmsInfo.h"
#include "Utilities/SimpleParser.h"
#include "OhmmsData/FileUtility.h"
#include "Platforms/sysutil.h"
#include "OhmmsApp/ProjectData.h"
#include "QMCApp/QMCMain.h"
//#include "tau/profiler.h"

/** @file qmcapp.cpp
 *@brief a main function for QMC simulation. 
 *
 * @ingroup qmcapp
 * @brief main function for qmcapp executable.
 *
 *Actual works are done by QMCAppBase and its derived classe.
 *For other simulations, one can derive a class from QMCApps, similarly to MolecuApps.
 */
int main(int argc, char **argv) {
  ///done with the option

  //TAU_PROFILE("int main(int, char **)", " ", TAU_DEFAULT);
  //TAU_INIT(&argc, &argv);

  OHMMS::Controller->initialize(argc,argv);
  // Write out free memory on each node on Linux.

  //check the options first
  int clones=1;
  vector<string> fgroup1,fgroup2;
  int i=1;
  while(i<argc)
  {
    string c(argv[i]);
    if(c.find("clones")<c.size())
    {
      clones=atoi(argv[++i]);
    }
    else if(c.find("xml")<c.size())
    {
      fgroup1.push_back(argv[i]);
    }
    else 
    {
      ifstream fin(argv[i]);
      bool valid=true;
      do 
      {
        vector<string> words;
        getwords(words,fin);
        if(words.size())
        {
          if(words[0].find("xml")<words[0].size())
          {
            int nc=1;
            if(words.size()>1) nc=atoi(words[1].c_str());
            while(nc)
            {
              fgroup2.push_back(words[0]);--nc;
            }
          }
        }
        else
          valid=false;
      } while(valid);
    }
    ++i;
  }
  int in_files=fgroup1.size();
  vector<string> inputs(in_files*clones+fgroup2.size());
  std::copy(fgroup2.begin(),fgroup2.end(),inputs.begin());
  i=fgroup2.size();
  for(int k=0; k<in_files; ++k)
    for(int c=0; c<clones; ++c) inputs[i++]=fgroup1[k];

  if(inputs.empty())
  {
    if(OHMMS::Controller->rank()==0)
    {
      cerr << "No input file is given" << endl;
      cerr << "usage: qmcapp [--clones int] input-files " << endl;
    }
    APP_ABORT("Missing input file");
    return 1;
  }

  //safe to move on
  Communicate* qmcComm=OHMMS::Controller;
  if(inputs.size()>1)
    qmcComm=new Communicate(*OHMMS::Controller,inputs.size());

  using namespace qmcplusplus;
  stringstream logname;
  logname<<getDateAndTime("%Y%m%dT%H%M");
  OhmmsInfo Welcome(logname.str(),qmcComm->rank(),qmcComm->getGroupID(),inputs.size());

//#if defined(MPIRUN_EXTRA_ARGUMENTS)
//  //broadcast the input file name to other nodes
//  MPI_Bcast(fname.c_str(),fname.size(),MPI_CHAR,0,OHMMS::Controller->getID());
//#endif

  QMCMain *qmc=0;
  bool validInput=false;
  app_log() << "  Input file(s): ";
  for(int k=0; k<inputs.size(); ++k) app_log() << inputs[k] << " ";
  app_log() << endl;
  qmc = new QMCMain(qmcComm);
  if(inputs.size()>1)
    validInput=qmc->parse(inputs[qmcComm->getGroupID()]);
  else
    validInput=qmc->parse(inputs[0]);
  if(validInput) qmc->execute();
  if(qmc) delete qmc;

  OHMMS::Controller->finalize();
  return 0;
}
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/