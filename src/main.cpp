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
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  
  const unsigned int r = 1;
  const double T      = 25;
  const double deltat = 3;
  // const double theta  = 0.5;

  std::string mesh = "../mesh/mshs/brain_coarse.msh";
  const Point<3> mesh_center = {45.0, 78.0, 62.0};
  
  // std::string mesh = "../mesh/brain/half/finer-normalized.msh";
  // const Point<3> mesh_center = {0.5, 500, 0.5};
  
  
  NonLinearParabolic3D problem(mesh, mesh_center, r, T, deltat, 1);

  problem.setup();
  problem.solve();
  
  return 0;
}



