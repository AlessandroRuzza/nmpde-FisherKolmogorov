#include "NonLinearParabolic3D.hpp"


#include <fstream>
#include <iostream>
#include <deal.II/base/convergence_table.h>
#include <vector>

//----------------------------------------------
//---------------------2D-----------------------
//----------------------------------------------

using SolverClass = NonLinearParabolic3D;





void single_execution_file(const bool silent,
                           const unsigned int r,
                           const double T,
                           const double theta,
                           const double deltat,
                           const std::string mesh)
{
  
  SolverClass problem(mesh, r, T, deltat);
  
  std::streambuf* originalBuffer = std::cout.rdbuf();
  std::ofstream nullStream("dev/null");
  
  if (silent){
    std::cout.rdbuf(nullStream.rdbuf());
  }
  
  problem.setup();
  problem.solve();
  
  if (silent){
    std::cout.rdbuf(originalBuffer);
  }
  
}

// Main function.
int
main(int argc, char * argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  
  const bool silent = false;

  const unsigned int r = 1;
  const double T      = 1.0;
  const double theta  = 0.5;

  std::string mesh = "../mesh/brain/bra.msh";
  const double deltat = 0.1;
  
  single_execution_file(silent, r, T, theta, deltat, mesh);
  
  return 0;
}



