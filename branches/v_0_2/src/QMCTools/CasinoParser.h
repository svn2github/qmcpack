#ifndef OHMMS_TOOLS_CASINOPARSER_H
#define OHMMS_TOOLS_CASINOPARSER_H
#include "QMCTools/QMCGaussianParserBase.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include "OhmmsPETE/TinyVector.h"
#include "OhmmsData/OhmmsElementBase.h"

class CasinoParser: public QMCGaussianParserBase, 
                    public OhmmsAsciiParser {

  std::vector<double> BasisCorrection;

public:

  CasinoParser();

  CasinoParser(int argc, char** argv);

  void parse(const std::string& fname);

  void getGeometry(std::istream& is);

  void getGaussianCenters(std::istream& is);

  //Specialized functions
  void getNumberOfAtoms(std::istream& is);

  void getAtomicPositions(std::istream& is);

  void getAtomicNumbers(std::istream& is);

  void getValenceCharges(std::istream& is);

  double contractionCorrection(int shell_id, double alpha);

  void makeCorrections();
};
#endif
