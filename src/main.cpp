#include "NonLinearParabolic3D.hpp"

#include <fstream>
#include <iostream>
#include <deal.II/base/convergence_table.h>
#include <vector>

// Main function.
int
main(int argc, char * argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 1;
  const double T      = 30;
  const double deltat = 0.01;
  // const double theta  = 0.5;

  // std::string mesh = "../mesh/brain_coarse.msh";
  // const Point<3> axonal_center = {0.0, 0.0, 0.0};
  
  // std::string mesh = "../mesh/brain/half/finer-normalized.msh";
  // const Point<3> axonal_center = {0.5, 500, 0.5};
  
  // std::string mesh = "../mesh/mesh-cube-40.msh";
  // const Point<3> axonal_center = {0.5, 500, 0.5};
  
  std::string mesh = "../mesh/sagittal_mesh.msh";
  const Point<2> axonal_center = {190.0, 140.0};
  
  
  NonLinearParabolic3D problem(mesh, axonal_center, r, T, deltat, 5);

  problem.setup();
  problem.solve();
  
  return 0;
}



