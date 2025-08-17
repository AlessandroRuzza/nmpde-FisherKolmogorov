#include "NonLinearParabolic3D.hpp"
#include "mesh_data.hpp"

#include <fstream>
#include <iostream>
#include <vector>

// Available meshes in mesh_data.hpp:
// 3D: MNI, Ernie, BrainCoarse, Cube40
// 2D: Sagittal, Sagittal_whiteGrayDiff (the second one has much larger differences in coefficient values on the 2 materials)

int main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 1;
  const double T      = 30;
  const double deltat = 1.0/12.0;
  
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
    std::cout << "Note: axonal_vector field will be written only at the first time step (time_step=0), to save disk space." << std::endl;
  }

  NonLinearParabolic3D problem(Sagittal_whiteGrayDiff, r, T, deltat, 2);

  problem.setup();
  problem.solve();
  
  return 0;
}



