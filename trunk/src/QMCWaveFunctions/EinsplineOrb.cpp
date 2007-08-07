#include "EinsplineOrb.h"
#include "Numerics/HDFNumericAttrib.h"

namespace qmcplusplus
{
  
  void
  EinsplineOrb<double,2>::read (hid_t h5file, string groupPath)
  {
    cerr << "2D orbital reads not yet implemented.\n";
    abort();
  }
  
  void
  EinsplineOrb<double,3>::read (hid_t h5file, string groupPath)
  {
    string centerName = groupPath + "center";
    string vectorName = groupPath + "eigenvector";
    string  valueName = groupPath + "eigenvalue";
    string radiusName = groupPath + "radius";
    HDFAttribIO<PosType> h_Center(Center);
    HDFAttribIO<double>  h_Radius(Radius);
    HDFAttribIO<double>  h_Energy(Energy);
    h_Center.read(h5file, centerName.c_str());
    h_Radius.read(h5file, radiusName.c_str());
    h_Energy.read(h5file,  valueName.c_str());

    Array<complex<double>,3> rawData;
    Array<double,3> realData;
    HDFAttribIO<Array<complex<double>,3> > h_rawData(rawData);
    h_rawData.read(h5file, vectorName.c_str());
    int nx, ny, nz;
    nx = rawData.size(0); ny=rawData.size(1); nz=rawData.size(2);
    realData.resize(nx-1,ny-1,nz-1);
    for (int ix=0; ix<nx-1; ix++)
      for (int iy=0; iy<ny-1; iy++)
	for (int iz=0; iz<nz-1; iz++)
	  realData(ix,iy,iz) = rawData(ix,iy,iz).real();

    Ugrid x_grid, y_grid, z_grid;
    BCtype_d xBC, yBC, zBC;

    xBC.lCode = PERIODIC;    xBC.rCode = PERIODIC;
    yBC.lCode = PERIODIC;    yBC.rCode = PERIODIC;
    zBC.lCode = PERIODIC;    zBC.rCode = PERIODIC;
    x_grid.start = 0.0;  x_grid.end = 1.0;  x_grid.num = nx-1;
    y_grid.start = 0.0;  y_grid.end = 1.0;  y_grid.num = ny-1;
    z_grid.start = 0.0;  z_grid.end = 1.0;  z_grid.num = nz-1;
    
    fprintf (stderr, "  Center = (%8.5f, %8.5f %8.5f)    Mesh = %dx%dx%d\n", 
	     Center[0], Center[1], Center[2], nx, ny, nz);

    Spline = create_UBspline_3d_d (x_grid, y_grid, z_grid,
				   xBC, yBC, zBC, &realData(0,0,0));
  }
  
  void
  EinsplineOrb<complex<double>,2>::read (hid_t h5file, string groupPath)
  {
    cerr << "2D orbital reads not yet implemented.\n";
    abort();
  }
  
  void
  EinsplineOrb<complex<double>,3>::read (hid_t h5file, string groupPath)
  {
    string centerName = groupPath + "center";
    string vectorName = groupPath + "eigenvector";
    string  valueName = groupPath + "eigenvalue";
    string radiusName = groupPath + "radius";
    HDFAttribIO<PosType> h_Center(Center);
    HDFAttribIO<double>  h_Radius(Radius);
    HDFAttribIO<double>  h_Energy(Energy);
    h_Center.read(h5file, centerName.c_str());
    h_Radius.read(h5file, radiusName.c_str());
    h_Energy.read(h5file,  valueName.c_str());
    fprintf (stderr, "Center = (%8.5f, %8.5f %8.5f)\n", 
	     Center[0], Center[1], Center[2]);

    Array<complex<double>,3> rawData, splineData;
    HDFAttribIO<Array<complex<double>,3> > h_rawData(rawData);
    h_rawData.read(h5file, vectorName.c_str());
    int nx, ny, nz;
    nx = rawData.size(0); ny=rawData.size(1); nz=rawData.size(2);
    splineData.resize(nx-1,ny-1,nz-1);
    for (int ix=0; ix<nx-1; ix++)
      for (int iy=0; iy<ny-1; iy++)
	for (int iz=0; iz<nz-1; iz++)
	  splineData(ix,iy,iz) = rawData(ix,iy,iz);

    Ugrid x_grid, y_grid, z_grid;
    BCtype_z xBC, yBC, zBC;

    xBC.lCode = PERIODIC;    xBC.rCode = PERIODIC;
    yBC.lCode = PERIODIC;    yBC.rCode = PERIODIC;
    zBC.lCode = PERIODIC;    zBC.rCode = PERIODIC;
    x_grid.start = 0.0;  x_grid.end = 1.0;  x_grid.num = nx-1;
    y_grid.start = 0.0;  y_grid.end = 1.0;  y_grid.num = ny-1;
    z_grid.start = 0.0;  z_grid.end = 1.0;  z_grid.num = nz-1;
    
    fprintf (stderr, "  Center = (%8.5f, %8.5f %8.5f)    Mesh = %dx%dx%d\n", 
	     Center[0], Center[1], Center[2], nx, ny, nz);

    Spline = create_UBspline_3d_z (x_grid, y_grid, z_grid,
				   xBC, yBC, zBC, &splineData(0,0,0));
  }
}
