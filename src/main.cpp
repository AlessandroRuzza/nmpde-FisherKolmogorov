#include "FisherKolmogorov.hpp"
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
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const unsigned int r = 1;
  const double T      = 30;
  const double deltat = 1.0/12.0;

  if(argc < 2){
    std::cerr << "Usage: " << argv[0] << " <mesh_preset>" << std::endl;
    return 1;
  }

  constexpr unsigned int dim = 3;

  const std::string mesh_preset = argv[1];
  MeshData<dim> mesh = get_mesh_data<dim>(mesh_preset, mpi_rank);

  if(mpi_rank == 0){
    std::cout << "Note: axonal_vector field and material_id will be written only at the first time step (time_step=0), to save disk space." << std::endl;
  }

  FisherKolmogorov problem(mesh, r, T, deltat, 2);

  problem.setup();
  problem.solve();
  
  return 0;
}



