#ifndef NON_LINEAR_PARABOLIC_3D_HPP
#define NON_LINEAR_PARABOLIC_3D_HPP
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

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
#include <deal.II/lac/solver_bicgstab.h>
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
  static constexpr unsigned int dim = 2;
  
  static constexpr double dext = 6;
  static constexpr double daxn = 20;
  // static constexpr double k0 = 5;
  // static constexpr double k1 = 5;
  // static constexpr double k12 = 1;
  // static constexpr double k_tilde1 = 3e-1;

  static constexpr double alpha_coeff = 2;

  // Misfolded protein start sphere center and radius
  // static constexpr double x0 = 0, y0 = 0, z0=25; // BRAIN_COARSE
  // static constexpr double radius = 20;

  // static constexpr double x0 = 0, y0 = 0, z0=0; // MNI_mesh
  // static constexpr double radius = 20;
  
  static constexpr double x0 = 230, y0 = 100, z0=0; // SAGITTAL
  static constexpr double radius = 15;

  // static constexpr double x0 = 0.5, y0 = 0.5, z0=0.5; // CUBE
  // static constexpr double radius = 0.2;

  // Function for the mu_0 coefficient.
  class FunctionD : public Function<dim>
  {
  private:  
    static constexpr bool override_radial_axon = false;
    static constexpr double center_threshold = 10;

    const Point<dim> axonal_center;

    // Attributes
    static constexpr double a = 60; // Major axis
    static constexpr double b = 30; // Minor axis
    static constexpr double c = 30; // Z axis (for 3D)
    Tensor<1, 3, double> n; // Normal to plane
    Tensor<1, 3, double> u; // Major axis direction (in plane, unit)
    Tensor<1, 3> v;       // Minor axis direction (in plane, unit)

    Tensor<1,2> get_axon_at(const Point<2> &p) const {
        Tensor<1, 2> tangent;
        // Shifted coordinates
        double x = p[0] - axonal_center[0];
        double y = p[1] - axonal_center[1];

        Tensor<1, 2> dist_center;
        dist_center[0] = x; dist_center[1] = y;
        if(dist_center.norm() < center_threshold){ // If too close to the axonal_center, use isotropic diffusion
          Tensor<1,2> zero;
          zero[0] = 0; zero[1] = 0;
          return zero;
        }

        // If (x, y) is on the ellipse, tangent direction:
        // dx/dt = -a * sin(t), dy/dt = b * cos(t)
        // But as a general formula, tangent at (x, y):
        tangent[0] = -b*b * y;
        tangent[1] = a*a * x;

        // Normalize the tangent
        const double norm = tangent.norm();
        if (norm > 0)
            tangent /= norm;

        const bool is_inside_ellipse = ((x*x)/(a*a) + (y*y)/(b*b)) <= 1.0 ;
        if (!is_inside_ellipse){ // Outside ellipse perimeter
            Tensor<1,2> normal;
            normal[0] = -tangent[1];
            normal[1] = tangent[0];
            return normal;
        }
        else  // Inside ellipse perimeter
            return tangent;
    }
    Tensor<1,3> get_axon_at(const dealii::Point<3> &p) const {
        // Shifted coordinates
        double x = p[0] - axonal_center[0];
        double y = p[1] - axonal_center[1];
        double z = p[2] - axonal_center[2];

        // Normal vector to the ellipsoid at (x, y, z) (gradient of implicit equation)
        Tensor<1,3> normal;
        normal[0] = 2.0 * x / (a * a);
        normal[1] = 2.0 * y / (b * b);
        normal[2] = 2.0 * z / (c * c);

        // Normalize the normal
        if (normal.norm() > 0)
            normal /= normal.norm();

        // Test if the point is inside or on the ellipsoid
        bool is_inside_ellipsoid = ( (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c) ) <= 1.0 ;
        if (!is_inside_ellipsoid) {
            return normal;
        } else {
            // Compute a deterministic tangent vector orthogonal to the normal
            // (cross product with a fixed vector, e.g., (0,0,1); if normal is parallel to (0,0,1), use (0,1,0) )
            Tensor<1,3> ref;
            if (std::abs(normal[2]) < 0.99){
                ref[0] = 0.0; ref[1] = 0.0; ref[2] = 1.0;
            }else{
                ref[0] = 0.0; ref[1] = 1.0; ref[2] = 0.0;
            }

            // Tangent = cross(normal, ref)
            Tensor<1,3> tangent;
            tangent[0] = normal[1]*ref[2] - normal[2]*ref[1];
            tangent[1] = normal[2]*ref[0] - normal[0]*ref[2];
            tangent[2] = normal[0]*ref[1] - normal[1]*ref[0];

            if (tangent.norm() > 0)
                tangent /= tangent.norm();

            return tangent;
        }
    }
  public:
    FunctionD(const Point<dim> axonal_center_) : axonal_center{axonal_center_}
    {
      n[0] = 0.0; n[1] = 0.0; n[2] = 1.0;
      u[0] = 1.0; u[1] = 0.0; u[2] = 0.0;
      v = cross_product_3d(n, u);
      v /= v.norm();
    }

    virtual void
    tensor_value(const Point<dim> &p, Tensor<2,dim> &retVal) const
    {
      // double x = p[0], y = p[1], z = p[2];

      Tensor<2,dim> identity;
      for(unsigned int i=0; i<dim; i++){
         identity[i][i] = 1;
      }

      Tensor<1, dim> axonal_vector;
      if(override_radial_axon){
          for (unsigned int i = 0; i < dim; i++)
          {
            axonal_vector[i] = p[i] - axonal_center[i];
          }
          if(axonal_vector.norm() > 0)
              axonal_vector /= axonal_vector.norm();
      }
      else axonal_vector = get_axon_at(p);

      Tensor<2, dim> tensor_product = outer_product(axonal_vector, axonal_vector);

      // for (unsigned int i = 0; i < dim; ++i)
      //   for (unsigned int j = 0; j < dim; ++j){
      //     tensor_product[i][j] = normal_vector[i] * normal_vector[j];
      //   }

      retVal = dext*identity + daxn * tensor_product;
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
      // return k12 * k0/k1 - k_tilde1;
      return alpha_coeff;
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
      double x = p[0], y = p[1], z;
      if(dim > 2)
        z = p[2];
      else
        z = z0;

      return std::max(0.0, 0.3 - ((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)) / radius );
    }
  };
  
  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  NonLinearParabolic3D(const std::string  &mesh_file_name_, 
                const Point<dim> mesh_center_,
                const unsigned int &r_,
                const double       &T_,
                const double       &deltat_,
                const unsigned int &outputPeriod_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , d(mesh_center_)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , axonal_center(mesh_center_)
    , r(r_)
    , deltat(deltat_)
    , outputPeriod(outputPeriod_)
    , mesh(MPI_COMM_WORLD)
  {
    pcout << "MPI size = " << mpi_size << "\n";
  }

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
  const Point<dim> axonal_center;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;
  const unsigned int outputPeriod;

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