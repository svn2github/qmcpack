#include "Message/Communicate.h"
#include "LongRange/KContainer.h"
#include <map>

using namespace qmcplusplus;

//Constructor
KContainer::KContainer(ParticleLayout_t& ref): Lattice(ref),kcutoff(0.0) { }

//Destructor
KContainer::~KContainer() { }

//Overloaded assignment operator
KContainer&
KContainer::operator=(const KContainer& ref) {
  //Lattices should be equal. 
  if(&Lattice != &ref.Lattice){
    LOGMSG("ERROR: tried to copy KContainer with different lattices");
    OHMMS::Controller->abort();
  }
  //Now, if kcutoffs are the same then we can be sure that the lists are identical.
  //otherwise the STL containers must have contents copied.
  if(this!=&ref && kcutoff!=ref.kcutoff){
    //All components have a valid '=' defined
    kcutoff = ref.kcutoff;
    kcut2 = ref.kcut2;
    mmax = ref.mmax;
    kpts = ref.kpts;
    kpts_cart = ref.kpts_cart;
    minusk = ref.minusk;
    numk = ref.numk;
  }

  return *this;
}

//Public Methods
// UpdateKLists - call for new k or when lattice changed.
void
KContainer::UpdateKLists(ParticleLayout_t& ref, RealType kc, bool useSphere) {
  kcutoff = kc;
  kcut2 = kc*kc;
  Lattice = ref;

  if(kcutoff <= 0.0){
    LOGMSG("KContainer initialised with cutoff " << kcutoff);
    OHMMS::Controller->abort();
  }

  FindApproxMMax();
  BuildKLists(useSphere);
}

// UpdateKLists - call for new k or when lattice changed.
void
KContainer::UpdateKLists(RealType kc, bool useSphere) {
  kcutoff = kc;
  kcut2 = kc*kc;

  if(kcutoff <= 0.0){
    LOGMSG("KContainer initialised with cutoff " << kcutoff);
    OHMMS::Controller->abort();
  }

  FindApproxMMax();
  BuildKLists(useSphere);
}

//Private Methods:
// FindApproxMMax - compute approximate parallelpiped that surrounds kcutoff
// BuildKLists - Correct mmax and fill lists of k-vectors.
void 
KContainer::FindApproxMMax() {
  //Estimate the size of the parallelpiped that encompases a sphere of kcutoff.
  //mmax is stored as integer translations of the reciprocal cell vectors.
  //Does not require an orthorhombic cell. 
  /* Old method.
  //2pi is not included in Lattice.b
  Matrix<RealType> mmat;
  mmat.resize(3,3);
  for(int j=0;j<3;j++)
    for(int i=0;i<3;i++){
      mmat[i][j] = 0.0;
      for(int k=0;k<3;k++)
	mmat[i][j] = mmat[i][j] + 4.0*M_PI*M_PI*Lattice.b(k)[i]*Lattice.b(j)[k];
    }

  TinyVector<RealType,3> x,temp;
  RealType tempr;
  for(int idim=0;idim<3;idim++){
    int i = ((idim)%3);
    int j = ((idim+1)%3);
    int k = ((idim+2)%3);
    
    x[i] = 1.0;
    x[j] = (mmat[j][k]*mmat[k][i] - mmat[k][k]*mmat[i][j]);
    x[j]/= (mmat[j][j]*mmat[k][k] - mmat[j][k]*mmat[j][k]);
    x[k] = -(mmat[k][i] + mmat[j][k]*x[j])/mmat[k][k];
    
    for(i=0;i<3;i++){
      temp[i] = 0.0;
	for(j=0;j<3;j++)
	  temp[i] += mmat[i][j]*x[j];
    }

    tempr = dot(x,temp);
    mmax[idim] = static_cast<int>(sqrt(4.0*kcut2/tempr)) + 1;  
  }
  */
  // see rmm, Electronic Structure, p. 85 for details
  for (int i = 0; i < 3; i++) 
    mmax[i] = static_cast<int>(floor(sqrt(dot(Lattice.a(i),Lattice.a(i))) * kcutoff / (2 * M_PI))) + 1;
}
void 
KContainer::BuildKLists(bool useSphere) {

  kpts.clear();
  kpts_cart.clear();

  // reserve the space for memory efficiency
  int numGuess=(2*mmax[0]+1)*(2*mmax[1]+1)*(2*mmax[2]+1);
  kpts.reserve(numGuess);
  kpts_cart.reserve(numGuess);
  ksq.reserve(numGuess);  


  TinyVector<int,4> TempActualMax;
  TinyVector<int,3> kvec;
  TinyVector<RealType,3> kvec_cart;
  RealType modk2;
  for(int i=0; i <4; i++) 
    TempActualMax[i] = 0;

  if(useSphere) {
    //Loop over guesses for valid k-points.
    for(int i=-mmax[0]; i<=mmax[0]; i++){
      kvec[0] = i;
      for(int j=-mmax[1]; j<=mmax[1]; j++){
        kvec[1] = j;
        for(int k=-mmax[2]; k<=mmax[2]; k++){
	  kvec[2] = k;
	      
	  //Do not include k=0 in evaluations.
	  if(i==0 && j==0 && k==0)continue;

	  //Convert kvec to Cartesian
	  /*
      	  for(int idim=0; idim<3; idim++){
	    kvec_cart[idim] = 0.0;
	    for(int idir=0; idir<3; idir++){
	      kvec_cart[idim]+=kvec[idir]*Lattice.b(idir)[idim];
	    }
	    kvec_cart[idim]*=TWOPI;
	  }
	  */
	  kvec_cart = Lattice.k_cart(kvec);
	

	  //Find modk
	  modk2 = dot(kvec_cart,kvec_cart);

	  if(modk2>kcut2)continue; //Inside cutoff?

	  //This k-point should be added to the list
	  kpts.push_back(kvec);
	  kpts_cart.push_back(kvec_cart);
	  ksq.push_back(modk2);
	
	  //Update record of the allowed maximum translation.
	  for(int idim=0; idim<3; idim++)
	    if(abs(kvec[idim]) > TempActualMax[idim])
	      TempActualMax[idim] = abs(kvec[idim]);
        }
      }
    }
  } else {
    // Loop over all k-points in the parallelpiped and add them to kcontainer
    // note layout is for interfacing with fft, so for each dimension, the 
    // positive indexes come first then the negative indexes backwards
    // e.g.    0, 1, .... mmax, -mmax+1, -mmax+2, ... -1
    const int idimsize = mmax[0]*2;
    const int jdimsize = mmax[1]*2;
    const int kdimsize = mmax[2]*2;
    for (int i = 0; i < idimsize; i++) {
      kvec[0] = i;
      if (kvec[0] > mmax[0]) kvec[0] -= idimsize;
      for (int j = 0; j < jdimsize; j++) {
        kvec[1] = j;
        if (kvec[1] > mmax[1]) kvec[1] -= jdimsize;
        for (int k = 0; k < kdimsize; k++) {
          kvec[2] = k;
          if (kvec[2] > mmax[2]) kvec[2] -= kdimsize;

          // get cartesian location and modk2
          kvec_cart = Lattice.k_cart(kvec);
          modk2 = dot(kvec_cart, kvec_cart);

          // add k-point to lists
          kpts.push_back(kvec);
          kpts_cart.push_back(kvec_cart);
          ksq.push_back(modk2);
        }
      }
    }
    // set allowed maximum translation
    TempActualMax[0] = mmax[0];
    TempActualMax[1] = mmax[1];
    TempActualMax[2] = mmax[2];

  }
  
  //Finished searching k-points. Copy list of maximum translations.
  mmax[3] = 0;
  for(int idim=0; idim<3; idim++) {
    mmax[idim] = TempActualMax[idim];
    if(mmax[idim] > mmax[3])
      mmax[3] = mmax[idim];
  }

  //Update a record of the number of k vectors
  numk = kpts.size();
  
  //Now fill the array that returns the index of -k when given the index of k.
  minusk.resize(numk);

  // Create a map from the hash value for each k vector to the index
  std::map<int, int> hashToIndex;
  for (int ki=0; ki<numk; ki++) {
    hashToIndex[GetHashOfVec(kpts[ki], numk)] = ki;
  }

  // Use the map to find the index of -k from the index of k
  for(int ki=0; ki<numk; ki++) {
    minusk[ki] = hashToIndex[ GetHashOfVec(-1 * kpts[ki], numk) ];
  }

  //create the map: use simple integer with resolution of 0.001 in ksq
  for(int ik=0; ik<kpts.size(); ik++) {
    int k_ind=static_cast<int>(ksq[ik]*1000);
    std::map<int,std::vector<int>*>::iterator it(kpts_sorted.find(k_ind));
    if(it == kpts_sorted.end()) {
      std::vector<int>* newSet=new std::vector<int>;
      kpts_sorted[k_ind]=newSet;
      newSet->push_back(ik);
    } else {
      (*it).second->push_back(ik);
    }
  }

 // std::map<int,std::vector<int>*>::iterator it(kpts_sorted.begin());
 // cout << "<<<<< sorted kpts " << endl;
 // while(it != kpts_sorted.end()) {
 //   cout << (*it).first << " " << (*it).second->size() << endl;
 //   std::vector<int>::iterator vit((*it).second->begin());
 //   while(vit != (*it).second->end()) {
 //     int ik=(*vit);
 //     cout << "   " << ik <<  " " << kpts[ik] << " " << ksq[ik] << endl;
 //     ++vit;
 //   }
 //   ++it;
 // }
}

