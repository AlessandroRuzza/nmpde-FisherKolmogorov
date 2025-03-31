#ifndef NON_LINEAR_PARABOLIC_3D_HPP
#define NON_LINEAR_PARABOLIC_3D_HPP
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <algorithm>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class NonLinearParabolic3D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;
  
  static constexpr double dext = 8e-6;
  static constexpr double daxn = 8e-5;
  static constexpr double k0 = 6e-1;
  static constexpr double k1 = 5e-1;
  static constexpr double k12 = 1;
  static constexpr double k_tilde1 = 3e-1;

  static constexpr Point<3> center = Point<3>(44.947, 95.2539, 33.1461);
  static constexpr double radius = 10.0;

  // Function for the mu_0 coefficient.
  class FunctionD : public Function<dim>
  {
  public:
    virtual Tensor<2,dim>
    tensor_value(const Point<dim> &p, const Tensor<1,dim> normal_vector) const
    {
      // double x = p[0], y = p[1], z = p[2];
      Tensor<2,dim> identity;
      for(unsigned int i=0; i<dim; i++) identity[i,i] = 1;
      
      //Tensor<2,dim> tensor_normal = normal_vector * normal_vector;
      
      dealii::Tensor<2, dim> tensor_product;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          tensor_product[i,j] = normal_vector[i] * normal_vector[j];
      
      return dext*identity + daxn * tensor_product;
    }
  };

  // Function for the reaction coefficient.
  class FunctionReaction : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return k12 * k0/k1 - k_tilde1;
    }
  };

  // Function for initial conditions.
  class FunctionC0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      double x = p[0], y = p[1], z = p[2];
      double x0 = center[0], y0 = center[1], z0=center[2];
      double r = radius;

      return std::max(0.0, 1.0 - ((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)) / r );
      
      //return abs(sin(x) + sin(y) + sin(z)) / 15.0; //TODO: insert real data for c0
    }
  };
  
  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  NonLinearParabolic3D(const std::string  &mesh_file_name_,
                const unsigned int &r_,
                const double       &T_,
                const double       &deltat_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

protected:
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the linear system associated to the tangent problem.
  void
  solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // mu_0 coefficient.
  FunctionD d;
  
  FunctionReaction alpha;

  // Initial conditions.
  FunctionC0 c_0;
  
  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;
  
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;
  
  
  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;
};

#endif