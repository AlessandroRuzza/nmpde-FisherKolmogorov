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

using namespace dealii;

// Class representing the non-linear diffusion problem.
class NonLinearParabolic3D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  static constexpr double mu0 = 0.1;
  static constexpr double mu1 = 1.0;
  static constexpr double sigma = 2.0;
  static constexpr double b1 = 0.5;
  static constexpr double b2 = 0.5;
  static constexpr double b3 = 0.5;
  static constexpr double dirichlet = 0.0;
  static constexpr double initial = 0.0;

  // Function for the mu_0 coefficient.
  class FunctionMu0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return mu0;
    }
  };

  // Function for the mu_1 coefficient.
  class FunctionMu1 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return mu1;
    }
  };
  
  
  // Function for the transport coefficient.
  class FunctionTransport : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if(component == 0) return b1;
      if(component == 1) return b2;
      else return b3;
    }
    
    virtual void
    vector_value(const Point<dim> & p,
                 Vector<double> &values) const override{
      for (unsigned short i = 0; i < dim; i++)
      {
        values[i] = value(p, i);
      }
    }
  };
  
  // Function for the reaction coefficient.
  class FunctionReaction : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return sigma;
    }
  };


// Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      
      double t = get_time();
      double x = p[0], y = p[1], z = p[2];
      
      double exp_term = exp(-t);
      double sin_x = sin(M_PI * x), cos_x = cos(M_PI * x);
      double sin_y = sin(M_PI * y), cos_y = cos(M_PI * y);
      double sin_z = sin(M_PI * z), cos_z = cos(M_PI * z);
      
      double u = exp(-t) * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
      
      double du_dt = -exp(-t) * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
      double du_dx = M_PI * exp_term * cos_x * sin_y * sin_z;
      double du_dy = M_PI * exp_term * sin_x * cos_y * sin_z;
      double du_dz = M_PI * exp_term * sin_x * sin_y * cos_z;
      
      double laplacian_u = -3 * M_PI * M_PI * exp(-t) * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
      
      double squared_gradient = M_PI * M_PI * exp(-2 * t) *
                                (pow(cos(M_PI * x), 2) * pow(sin(M_PI * y), 2) * pow(sin(M_PI * z), 2) +
                                 pow(sin(M_PI * x), 2) * pow(cos(M_PI * y), 2) * pow(sin(M_PI * z), 2) +
                                 pow(sin(M_PI * x), 2) * pow(sin(M_PI * y), 2) * pow(cos(M_PI * z), 2));
      
      double advection = b1* du_dx + b2* du_dy + b3* du_dz;
      
      return du_dt - (mu0 + mu1 * u * u) * laplacian_u - 2 * mu1 * u * squared_gradient + advection + sigma*u;
      
    }
  };

  // Function for Dirichlet boundary conditions.
  class DirichletFunction : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return dirichlet;
    }
  };

  // Function for initial conditions.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      double x = p[0], y = p[1], z = p[2];
      
      //return x * (1 - x) * y * (1 - y) * z * (1 - z);
      return sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
    }
  };
  
  
  
  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(){}
    
    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      double t = get_time();
      double x = p[0], y = p[1], z = p[2];
      
      return exp(-t) * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
      
    }
    
    // Gradient evaluation.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      double t = get_time();
      double x = p[0], y = p[1], z = p[2];
      
      
      double exp_term = exp(-t);
      double sin_x = sin(M_PI * x), cos_x = cos(M_PI * x);
      double sin_y = sin(M_PI * y), cos_y = cos(M_PI * y);
      double sin_z = sin(M_PI * z), cos_z = cos(M_PI * z);
      
      result[0] = M_PI * exp_term * cos_x * sin_y * sin_z;
      result[1] = M_PI * exp_term * sin_x * cos_y * sin_z;
      result[2] = M_PI * exp_term * sin_x * sin_y * cos_z;
      
      
      return result;
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
  
  double
  compute_error(const VectorTools::NormType &norm_type);

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
  FunctionMu0 mu_0;

  // mu_1 coefficient.
  FunctionMu1 mu_1;
  
  FunctionTransport transport_coeff;
  
  FunctionReaction reaction_coeff;
  
  
  // Forcing term.
  ForcingTerm forcing_term;

  // Dirichlet boundary conditions.
  DirichletFunction dirichlet_func;

  // Initial conditions.
  FunctionU0 u_0;
  
  ExactSolution exact_solution;

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