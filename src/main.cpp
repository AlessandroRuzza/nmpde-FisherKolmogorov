#include "NonLinearParabolic3D.hpp"


#include <fstream>
#include <iostream>
#include <deal.II/base/convergence_table.h>
#include <vector>

//----------------------------------------------
//---------------------2D-----------------------
//----------------------------------------------

using SolverClass = NonLinearParabolic3D;

void time_file_function(const bool silent,
                        const unsigned int r,
                        const double T,
                        const double theta,
                        const std::string mesh,
                        const std::vector<double> deltat_vector,
                        const unsigned int mpi_rank)

{
  ConvergenceTable table;

  std::vector<double> errors_L2;
  std::vector<double> errors_H1;

  for (const auto &deltat : deltat_vector)
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

    errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
    errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
  }

  // Print the errors and estimate the convergence order.
  if (mpi_rank == 0)
  {
    std::cout << "==============================================="
              << std::endl;

    std::ofstream convergence_file("convergence.csv");
    convergence_file << "dt,eL2,eH1" << std::endl;

    for (unsigned int i = 0; i < deltat_vector.size(); ++i)
    {
      convergence_file << deltat_vector[i] << "," << errors_L2[i] << ","
                       << errors_H1[i] << std::endl;

      std::cout << std::scientific << "dt = " << std::setw(4)
                << std::setprecision(2) << deltat_vector[i];

      std::cout << std::scientific << " | eL2 = " << errors_L2[i];

      // Estimate the convergence order.
      if (i > 0)
      {
        const double p =
                std::log(errors_L2[i] / errors_L2[i - 1]) /
                std::log(deltat_vector[i] / deltat_vector[i - 1]);

        std::cout << " (" << std::fixed << std::setprecision(2)
                  << std::setw(4) << p << ")";
      }
      else
        std::cout << " (  - )";

      std::cout << std::scientific << " | eH1 = " << errors_H1[i];

      // Estimate the convergence order.
      if (i > 0)
      {
        const double p =
                std::log(errors_H1[i] / errors_H1[i - 1]) /
                std::log(deltat_vector[i] / deltat_vector[i - 1]);

        std::cout << " (" << std::fixed << std::setprecision(2)
                  << std::setw(4) << p << ")";
      }
      else
        std::cout << " (  - )";

      std::cout << "\n";
    }
  }
}



void mesh_file_function(const bool silent,
                        const unsigned int r,
                        const double T,
                        const double theta,
                        const double deltat,
                        const std::vector<std::string> meshes,
                        const std::vector<double> h_vals,
                        const unsigned int mpi_rank)


{
  ConvergenceTable table;

  std::vector<double> errors_L2;
  std::vector<double> errors_H1;

  for (const auto &mesh : meshes)
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
    errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
    errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
  }

  // Print the errors and estimate the convergence order.
  if (mpi_rank == 0)
  {
    std::cout << "==============================================="
              << std::endl;


    std::ofstream convergence_file("convergence.csv");
    convergence_file << "h,eL2,eH1" << std::endl;

    for (unsigned int i = 0; i < meshes.size(); ++i)
    {
      const double h = h_vals[i];
      convergence_file << h << "," << errors_L2[i] << ","
                       << errors_H1[i] << std::endl;

      std::cout << std::scientific << "h = " << std::setw(4)
                << std::setprecision(2) << h;

      std::cout << std::scientific << " | eL2 = " << errors_L2[i];

      // Estimate the convergence order.
      if (i > 0)
      {
        const double p =
                std::log(errors_L2[i] / errors_L2[i - 1]) /
                std::log(h / h_vals[i-1]);

        std::cout << " (" << std::fixed << std::setprecision(2)
                  << std::setw(4) << p << ")";
      }
      else
        std::cout << " (  - )";

      std::cout << std::scientific << " | eH1 = " << errors_H1[i];

      // Estimate the convergence order.
      if (i > 0)
      {
        const double p =
                std::log(errors_H1[i] / errors_H1[i - 1]) /
                std::log(h / h_vals[i-1]);

        std::cout << " (" << std::fixed << std::setprecision(2)
                  << std::setw(4) << p << ")";
      }
      else
        std::cout << " (  - )";

      std::cout << "\n";
    }
  }
}





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


  const unsigned int r = 2;
  const double T      = 1.0;
  const double theta  = 0.5;

  const bool silent = true;

  //-------2D--------

  std::string mesh = "../mesh/mesh-cube-20.msh";
  const double deltat_initial = T/2;
  const std::vector<double> deltat_vector = {deltat_initial, deltat_initial/2, deltat_initial/4};
  time_file_function(silent, r, T, theta, mesh, deltat_vector, mpi_rank);


  const std::vector<double> h_vals = {1.0 / 5.0,
                                      1.0 / 10.0,
                                      1.0 / 20.0,
                                      /*1.0 / 40.0*/};

  const double deltat = 0.05;
  const std::vector<std::string> meshes = {"../mesh/mesh-cube-5.msh",
                                           "../mesh/mesh-cube-10.msh",
                                           "../mesh/mesh-cube-20.msh",
                                           /*"../mesh/mesh-cube-40.msh"*/};
            //NOTE: using r=1 for better convergence
  mesh_file_function(silent, 1, T, theta, deltat, meshes, h_vals,  mpi_rank);
  
  
  //single_execution_file(silent, r, T, theta, deltat, mesh);
  
  return 0;
}



