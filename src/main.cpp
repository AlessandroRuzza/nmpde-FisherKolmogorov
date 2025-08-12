#include "NonLinearParabolic3D.hpp"
#include "mesh_data.hpp"

#include <fstream>
#include <iostream>
#include <vector>

// Main function.
int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 1;
  const double T      = 40;
  const double deltat = 1.0/12.0;
  
  // std::string mesh = "../mesh/ernie_brain_dealii.msh";
  // const Point<3> axonal_center = {0.0, 0.0, 0.0};
  
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
    std::cout << "Note: axonal_vector field will be written only at the first time step (time_step=0), to save disk space." << std::endl;
  }

  NonLinearParabolic3D problem(sagittal, r, T, deltat, 5);

  problem.setup();
  problem.solve();
  
  return 0;
}



