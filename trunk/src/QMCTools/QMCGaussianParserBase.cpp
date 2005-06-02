#include "QMCTools/QMCGaussianParserBase.h"
#include "ParticleIO/XMLParticleIO.h"
#include "Utilities/OhmmsInfo.h"
#include <iterator>
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
using namespace std;
#include "QMCWaveFunctions/MolecularOrbitals/GTO2GridBuilder.h"
#include "QMCApp/InitMolecularSystem.h"

//std::vector<std::string> QMCGaussianParserBase::IonName;
std::map<int,std::string> QMCGaussianParserBase::IonName;
std::vector<std::string> QMCGaussianParserBase::gShellType;
std::vector<int> QMCGaussianParserBase::gShellID;

QMCGaussianParserBase::QMCGaussianParserBase(): 
  Title("sample"),basisType("Gaussian"),basisName("generic"),
  Normalized("no"),gridPtr(0)
{
}

QMCGaussianParserBase::QMCGaussianParserBase(int argc, char** argv):
  BohrUnit(true),SpinRestricted(false),NumberOfAtoms(0),NumberOfEls(0),
  NumberOfAlpha(0),NumberOfBeta(0),SizeOfBasisSet(0),
  Title("sample"),basisType("Gaussian"),basisName("generic"),  
  Normalized("no"),gridPtr(0)
{
  IonSystem.setName("i");
  IonChargeIndex=IonSystem.getSpeciesSet().addAttribute("charge");
  cout << "Index of ion charge " << IonChargeIndex << endl;
  createGridNode(argc,argv);
}

void QMCGaussianParserBase::init() {
  //IonName.resize(24);
  IonName[1] = "H"; IonName[2] = "He"; IonName[3] = "Li";
  IonName[4] = "Be"; IonName[5] = "B"; IonName[6] = "C";
  IonName[7] = "N"; IonName[8] = "O"; IonName[9] = "F";
  IonName[10] = "Ne"; IonName[11] = "Na"; IonName[12] = "Mg";
  IonName[13] = "Al"; IonName[14] = "Si"; IonName[15] = "P";
  IonName[16] = "S"; IonName[17] = "Cl"; IonName[18] = "Ar";
  gShellType.resize(7);
  gShellType[1]="s"; gShellType[2]="sp"; gShellType[3]="p"; gShellType[4]="d"; gShellType[5]="f"; gShellType[6]="g";
  gShellID.resize(7);
  gShellID[1]=0; gShellID[2]=0; gShellID[3]=1; gShellID[4]=2; gShellID[5]=3; gShellID[6]=4;
}

void QMCGaussianParserBase::setOccupationNumbers() {

  if(NumberOfAlpha==0) {
    if(SpinRestricted) {
      NumberOfAlpha = NumberOfEls/2;
      NumberOfBeta = NumberOfEls-NumberOfAlpha;
    } else {
      multimap<value_type,int> e;
      for(int i=0; i<SizeOfBasisSet; i++) e.insert(pair<value_type,int>(EigVal_alpha[i],0));
      for(int i=0; i<SizeOfBasisSet; i++) e.insert(pair<value_type,int>(EigVal_beta[i],1));
      NumberOfAlpha=0; NumberOfBeta=0;
      int n=0;
      multimap<value_type,int>::iterator it(e.begin());
      LOGMSG("Unrestricted HF. Sorted eigen values")
      while(n<NumberOfEls && it != e.end()) {
        LOGMSG(n << " " << (*it).first << " " << (*it).second)
        if((*it).second == 0) {NumberOfAlpha++;}
        else {NumberOfBeta++;}
        ++it;++n;
      }
    }
  }

  LOGMSG("Number of alpha electrons " << NumberOfAlpha)
  LOGMSG("Number of beta electrons " << NumberOfBeta)

  Occ_alpha.resize(SizeOfBasisSet,0);
  Occ_beta.resize(SizeOfBasisSet,0);
  for(int i=0; i<NumberOfAlpha; i++) Occ_alpha[i]=1;
  for(int i=0; i<NumberOfBeta; i++) Occ_beta[i]=1;
}

xmlNodePtr QMCGaussianParserBase::createElectronSet() {

  ParticleSet els;
  els.setName("e");
  vector<int> nel(2);
  nel[0]=NumberOfAlpha;
  nel[1]=NumberOfBeta;
  els.create(nel);

  int iu=els.getSpeciesSet().addSpecies("u");
  int id=els.getSpeciesSet().addSpecies("d");
  int ic=els.getSpeciesSet().addAttribute("charge");
  els.getSpeciesSet()(ic,iu)=-1;
  els.getSpeciesSet()(ic,id)=-1;

  //Create InitMolecularSystem to assign random electron positions
  InitMolecularSystem m(0,"test");
  if(IonSystem.getTotalNum()>1) {
    m.initMolecule(&IonSystem,&els);
  } else {
    m.initAtom(&els);
  }

  XMLSaveParticle o(els);
  return o.createNode();
}

xmlNodePtr QMCGaussianParserBase::createIonSet() {
  const double ang_to_bohr=1.0/0.529177e0;
  if(!BohrUnit) IonSystem.R *= ang_to_bohr;

  SpeciesSet& ionSpecies(IonSystem.getSpeciesSet());
  for(int i=0; i<NumberOfAtoms; i++) {
    ionSpecies.addSpecies(GroupName[i]);
  }

  for(int i=0; i<NumberOfAtoms; i++) {
    ionSpecies(IonChargeIndex,IonSystem.GroupID[i])=Qv[i];
  }

  XMLSaveParticle o(IonSystem);
  return o.createNode();
}
xmlNodePtr QMCGaussianParserBase::createBasisSet() {

  xmlNodePtr bset = xmlNewNode(NULL,(const xmlChar*)"basisset");
  /*
  xmlNewProp(bset,(const xmlChar*)"ref",(const xmlChar*)"i");
  xmlNodePtr cur = xmlAddChild(bset,xmlNewNode(NULL,(const xmlChar*)"distancetable"));
  xmlNewProp(cur,(const xmlChar*)"source",(const xmlChar*)"i");
  xmlNewProp(cur,(const xmlChar*)"target",(const xmlChar*)"e");
  */

  xmlNodePtr cur=NULL;
  std::map<int,int> species;
  int gtot = 0;
  for(int iat=0; iat<NumberOfAtoms; iat++) {
    int itype = IonSystem.GroupID[iat];
    int ng = 0;
    std::map<int,int>::iterator it=species.find(itype);
    if(it == species.end()) {
      for(int ig=gBound[iat]; ig<gBound[iat+1]; ig++) {
        ng += gNumber[ig];
      }
      species[itype] = ng;
      if(cur) {
        cur = xmlAddSibling(cur,createCenter(iat,gtot));
      } else {
        cur = xmlAddChild(bset,createCenter(iat,gtot));
      }
    } else {
      ng = (*it).second;
    }
    gtot += ng;
  }

  return bset;
}
xmlNodePtr 
QMCGaussianParserBase::createDeterminantSet() {

  setOccupationNumbers();

  xmlNodePtr slaterdet = xmlNewNode(NULL,(const xmlChar*)"slaterdeterminant");

  //check spin-dependent properties
  //int nup = NumberOfEls/2;
  //int ndown = NumberOfEls-nup;
  std::ostringstream up_size, down_size, b_size, occ;
  up_size <<NumberOfAlpha; down_size << NumberOfBeta; b_size<<SizeOfBasisSet;

  //create a determinant Up
  xmlNodePtr adet = xmlNewNode(NULL,(const xmlChar*)"determinant");
  xmlNewProp(adet,(const xmlChar*)"id",(const xmlChar*)"updet");
  xmlNewProp(adet,(const xmlChar*)"orbitals",(const xmlChar*)up_size.str().c_str());

  occ<<"\n";
  vector<int>::iterator it(Occ_alpha.begin()); 
  int i=0;
  while(i<SizeOfBasisSet) {
    int n = (i+10<SizeOfBasisSet)? 10 : SizeOfBasisSet-i;
    std::copy(it, it+n, ostream_iterator<int>(occ," "));
    occ << "\n"; it += 10; i+=10;
  }

  xmlNodePtr occ_data 
    = xmlNewTextChild(adet,NULL,(const xmlChar*)"occupation",(const xmlChar*)occ.str().c_str());
  xmlNewProp(occ_data,(const xmlChar*)"size",(const xmlChar*)b_size.str().c_str());

  int btot=SizeOfBasisSet*SizeOfBasisSet;
  int n=btot/4, b=0;
  int dn=btot-n*4;

  std::ostringstream eig;
  eig.setf(std::ios::scientific, std::ios::floatfield);
  eig.setf(std::ios::right,std::ios::adjustfield);
  eig.precision(14);
  eig << "\n";
  for(int k=0; k<n; k++) {
    eig << setw(22) << EigVec[b] << setw(22) << EigVec[b+1] << setw(22) << EigVec[b+2] << setw(22) <<  EigVec[b+3] << "\n";
    b += 4;
  }
  for(int k=0; k<dn; k++) { eig << setw(22) << EigVec[b++]; }
  if(dn) eig << endl;
  xmlNodePtr det_data 
    = xmlNewTextChild(adet,NULL,(const xmlChar*)"coefficient",(const xmlChar*)eig.str().c_str());
  xmlNewProp(det_data,(const xmlChar*)"size",(const xmlChar*)b_size.str().c_str());

  xmlNodePtr cur = xmlAddChild(slaterdet,adet);
  adet = xmlNewNode(NULL,(const xmlChar*)"determinant");
  xmlNewProp(adet,(const xmlChar*)"id",(const xmlChar*)"downdet");
  xmlNewProp(adet,(const xmlChar*)"orbitals",(const xmlChar*)down_size.str().c_str());
  if(SpinRestricted)
    xmlNewProp(adet,(const xmlChar*)"ref",(const xmlChar*)"updet");
  else {
    std::ostringstream occ_beta;
    occ_beta<<"\n";
    it=Occ_beta.begin(); 
    int i=0;
    while(i<SizeOfBasisSet) {
      int n = (i+10<SizeOfBasisSet)? 10 : SizeOfBasisSet-i;
      std::copy(it, it+n, ostream_iterator<int>(occ_beta," "));
      occ_beta << "\n"; it += 10; i+=10;
    }
    occ_data=xmlNewTextChild(adet,NULL,(const xmlChar*)"occupation",(const xmlChar*)occ_beta.str().c_str());
    xmlNewProp(occ_data,(const xmlChar*)"size",(const xmlChar*)b_size.str().c_str());

    std::ostringstream eigD;
    eigD.setf(std::ios::scientific, std::ios::floatfield);
    eigD.setf(std::ios::right,std::ios::adjustfield);
    eigD.precision(14);
    eigD << "\n";
    b=SizeOfBasisSet*SizeOfBasisSet;
    for(int k=0; k<n; k++) {
      eigD << setw(22) << EigVec[b] << setw(22) << EigVec[b+1] << setw(22) << EigVec[b+2] << setw(22) <<  EigVec[b+3] << "\n";
      b += 4;
    }
    for(int k=0; k<dn; k++) {
      eigD << setw(22) << EigVec[b++];
    }
    if(dn) eigD << endl;
    det_data 
      = xmlNewTextChild(adet,NULL,(const xmlChar*)"coefficient",(const xmlChar*)eigD.str().c_str());
    xmlNewProp(det_data,(const xmlChar*)"size",(const xmlChar*)b_size.str().c_str());
  }

  cur = xmlAddSibling(cur,adet);
  return slaterdet;
}

xmlNodePtr QMCGaussianParserBase::createCenter(int iat, int off_) {

  //CurrentCenter = IonName[GroupID[iat]];
  //CurrentCenter = IonSystem.Species.speciesName[iat];
  CurrentCenter=GroupName[iat];
  xmlNodePtr abasis = xmlNewNode(NULL,(const xmlChar*)"atomicBasisSet");
  xmlNewProp(abasis,(const xmlChar*)"name",(const xmlChar*)basisName.c_str());
  xmlNewProp(abasis,(const xmlChar*)"angular",(const xmlChar*)"spherical");
  xmlNewProp(abasis,(const xmlChar*)"type",(const xmlChar*)basisType.c_str());
  xmlNewProp(abasis,(const xmlChar*)"elementType",(const xmlChar*)CurrentCenter.c_str());
  xmlNewProp(abasis,(const xmlChar*)"normalized",(const xmlChar*)Normalized.c_str());
  xmlAddChild(abasis,xmlCopyNode(gridPtr,1));
  for(int ig=gBound[iat], n=0; ig< gBound[iat+1]; ig++,n++) {
    createShell(n, ig, off_,abasis);
    off_ += gNumber[ig];
  }

  return abasis;
}

void
QMCGaussianParserBase::createShell(int n, int ig, int off_, xmlNodePtr abasis) {

  int gid(gShell[ig]);
  int ng(gNumber[ig]);

  xmlNodePtr ag = xmlNewNode(NULL,(const xmlChar*)"basisGroup");
  xmlNodePtr ag1 = 0;

  char l_name[4],n_name[4],a_name[32];

  sprintf(a_name,"%s%d%d",CurrentCenter.c_str(),n,gShellID[gid]);
  sprintf(l_name,"%d",gShellID[gid]);
  sprintf(n_name,"%d",n);
  xmlNewProp(ag,(const xmlChar*)"rid", (const xmlChar*)a_name);
  xmlNewProp(ag,(const xmlChar*)"n", (const xmlChar*)n_name);
  xmlNewProp(ag,(const xmlChar*)"l", (const xmlChar*)l_name);
  xmlNewProp(ag,(const xmlChar*)"type", (const xmlChar*)"Gaussian");
  if(gid == 2) {
    sprintf(a_name,"%s%d1",CurrentCenter.c_str(),n);
    ag1 = xmlNewNode(NULL,(const xmlChar*)"basisGroup");
    xmlNewProp(ag1,(const xmlChar*)"rid", (const xmlChar*)a_name);
    xmlNewProp(ag1,(const xmlChar*)"n", (const xmlChar*)n_name);
    xmlNewProp(ag1,(const xmlChar*)"l", (const xmlChar*)"1");
    xmlNewProp(ag1,(const xmlChar*)"type", (const xmlChar*)"Gaussian");
  }

  for(int ig=0, i=off_; ig<ng; ig++, i++) {
    std::ostringstream a,b,c;
    a.setf(std::ios::scientific, std::ios::floatfield);
    b.setf(std::ios::scientific, std::ios::floatfield);
    a.precision(12);
    b.precision(12);

    a<<gExp[i]; b<<gC0[i];

    xmlNodePtr anode = xmlNewNode(NULL,(const xmlChar*)"radfunc");
    xmlNewProp(anode,(const xmlChar*)"exponent", (const xmlChar*)a.str().c_str());
    xmlNewProp(anode,(const xmlChar*)"contraction", (const xmlChar*)b.str().c_str());
    xmlAddChild(ag,anode);
    if(gid ==2) {
      c.setf(std::ios::scientific, std::ios::floatfield);
      c.precision(12);
      c <<gC1[i];
      anode = xmlNewNode(NULL,(const xmlChar*)"radfunc");
      xmlNewProp(anode,(const xmlChar*)"exponent", (const xmlChar*)a.str().c_str());
      xmlNewProp(anode,(const xmlChar*)"contraction", (const xmlChar*)c.str().c_str());
      xmlAddChild(ag1,anode);
    }
  }

  xmlAddChild(abasis,ag);
  if(gid == 2) xmlAddChild(abasis,ag1);
}

void QMCGaussianParserBase::map2GridFunctors(xmlNodePtr cur) {

  using namespace ohmmsqmc;

  xmlNodePtr anchor = cur;
  //xmlNodePtr grid_ptr = 0;

  vector<xmlNodePtr> phi_ptr;
  vector<QuantumNumberType> nlms;

  int Lmax = 0;
  int current = 0;
  std::string acenter("none");

  const xmlChar* aptr = xmlGetProp(cur,(const xmlChar*)"elementType");
  if(aptr) acenter = (const char*)aptr;

  xmlNodePtr grid_ptr=0;

  cur = anchor->children;
  while(cur != NULL) {
    string cname((const char*)(cur->name));
    if(cname == "grid") 
      grid_ptr = cur;
    else if(cname == "basisGroup") {
      int n=1,l=0,m=0;
      const xmlChar* aptr = xmlGetProp(cur,(const xmlChar*)"n");
      if(aptr) n = atoi((const char*)aptr);
      aptr = xmlGetProp(cur,(const xmlChar*)"l");
      if(aptr) l = atoi((const char*)aptr);

      Lmax = std::max(l,Lmax);
      phi_ptr.push_back(cur);
      nlms.push_back(QuantumNumberType());
      nlms[current][0]=n;
      nlms[current][1]=l;
      nlms[current][2]=m;
      ++current;
    }
    cur = cur->next;
  }

  if(grid_ptr == 0) {
    LOGMSG("Grid is not defined: using default")
    //xmlAddChild(anchor,gridPtr);
    grid_ptr = xmlCopyNode(gridPtr,1);
    xmlAddChild(anchor,grid_ptr);
  }

  RGFBuilderBase::CenteredOrbitalType aos(Lmax);
  bool normalized(Normalized=="yes");
  RGFBuilderBase* rbuilder = new GTO2GridBuilder(normalized);
  rbuilder->setOrbitalSet(&aos,acenter);
  rbuilder->addGrid(grid_ptr);
  for(int i=0; i<nlms.size(); i++) {
    rbuilder->addRadialOrbital(phi_ptr[i],nlms[i]);
  }
  rbuilder->print(acenter,1);
}

void QMCGaussianParserBase::createGridNode(int argc, char** argv) {

  gridPtr = xmlNewNode(NULL,(const xmlChar*)"grid");
  string gridType("log");
  string gridFirst("1.e-6");
  string gridLast("1.e2");
  string gridSize("1001");
  int iargc=0;
  while(iargc<argc) {
    string a(argv[iargc]);
    if(a == "-gridtype") {
      gridType=argv[++iargc];
    } else if(a == "-frst") {
      gridFirst=argv[++iargc];
    } else if(a == "-last") {
      gridLast=argv[++iargc];
    } else if(a == "-size") {
      gridSize=argv[++iargc];
    }
    ++iargc;
  }
  xmlNewProp(gridPtr,(const xmlChar*)"type",(const xmlChar*)gridType.c_str());
  xmlNewProp(gridPtr,(const xmlChar*)"ri",(const xmlChar*)gridFirst.c_str());
  xmlNewProp(gridPtr,(const xmlChar*)"rf",(const xmlChar*)gridLast.c_str());
  xmlNewProp(gridPtr,(const xmlChar*)"npts",(const xmlChar*)gridSize.c_str());
}
