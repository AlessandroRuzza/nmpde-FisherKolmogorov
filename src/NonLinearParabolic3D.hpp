#ifndef NON_LINEAR_PARABOLIC_3D_HPP
#define NON_LINEAR_PARABOLIC_3D_HPP
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_function.h>
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


#include <deal.II/base/data_out_base.h>   // DataOutBase::DataOutFilter, XDMFEntry
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>

#include <fstream>
#include <iostream>
#include <algorithm>

using namespace dealii;

template<unsigned int dim>
struct MeshData{
public:
  const std::string mesh_file_name;
  const double dext;
  const double daxn;

  const std::map<unsigned int, std::string> material_names;
  const std::map<std::string, double> alpha_coeffs;

  // Misfolded protein start sphere center and radius
  const double x0, y0, z0, radius; 

  // Min distance from center to enable axonal diffusion
  const double center_threshold;

  const Point<dim> axonal_center; // Elliptical and radial axons center
  const double a; // X axis
  const double b; // Y axis
  const double c; // Z axis (for 3D)
};

// Class representing the non-linear diffusion problem.
class NonLinearParabolic3D
{
  public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;
  using Mesh = MeshData<dim>;

  // Function for the mu_0 coefficient.
  class FunctionD : public Function<dim>
  {
  private:  
    static constexpr bool override_radial_axon = false;

    const Mesh mesh; 
    const double a,b,c;
  public:
    Tensor<1,2> get_axon_at(const Point<2> &p) const {
        Tensor<1,2> normal;
        // Shifted coordinates
        double x = p[0] - mesh.axonal_center[0];
        double y = p[1] - mesh.axonal_center[1];

        normal[0] = x / (a*a);
        normal[1] = y / (b*b);
        double norm = normal.norm();
        if (norm > 0)
            normal /= norm;

        const bool is_inside_ellipse = ((x*x)/(a*a) + (y*y)/(b*b)) <= 1.0 ;
        if (!is_inside_ellipse){ // Outside ellipse perimeter, axon aligns with normal of ellipsoid surface
            return normal;
        }
        else{  // Inside ellipse perimeter, axon aligns with tangent of ellipsoid
            Tensor<1,2> tangent;
            tangent[0] = -normal[1];
            tangent[1] = normal[0];
            return tangent;
        }
    }
    Tensor<1,3> get_axon_at(const Point<3> &p) const {
        // Shifted coordinates
        double x = p[0] - mesh.axonal_center[0];
        double y = p[1] - mesh.axonal_center[1];
        double z = p[2] - mesh.axonal_center[2];

        // Normal vector to the ellipsoid at (x, y, z) (gradient of implicit equation)
        Tensor<1,3> normal;
        normal[0] = x / (a * a);
        normal[1] = y / (b * b);
        normal[2] = z / (c * c);

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
            if (std::abs(normal[2]) < 0.9999){
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

    FunctionD(const Mesh& mesh_) : mesh{mesh_}, a{mesh_.a}, b{mesh_.b}, c{mesh_.c}
    {}

    virtual void
    tensor_value(const Point<dim> &p, Tensor<2,dim> &retVal) const
    {
      Tensor<2,dim> identity = unit_symmetric_tensor<dim>();
      Tensor<1, dim> axonal_vector;

      Tensor<1, dim> dist_center;
      for (unsigned int i = 0; i < dim; i++)
        dist_center[i] = p[i] - mesh.axonal_center[i];

      if(dist_center.norm() < mesh.center_threshold) // If too close to the axonal_center, use isotropic diffusion
          axonal_vector.clear(); // set to zero
      else{
          if(override_radial_axon)
            axonal_vector = dist_center / dist_center.norm(); // Assuming center_threshold > 0, norm is not 0
          else 
            axonal_vector = get_axon_at(p);
      }

      Tensor<2, dim> tensor_product = outer_product(axonal_vector, axonal_vector);

      retVal = mesh.dext*identity + mesh.daxn*tensor_product;
    }
  };

  // Function for the reaction coefficient.
  class FunctionReaction : public Function<dim>
  {
    const Mesh mesh;

  public:
    FunctionReaction(const Mesh& mesh_) : mesh(mesh_)
    {}

    virtual double
    value(const unsigned int material_id,
          const Point<dim> &/*p*/,
          const unsigned int /*component*/ = 0) //cannot override
    {
      const std::string matter = mesh.material_names.at(material_id);
      return mesh.alpha_coeffs.at(matter);
    }
  };

  // Function for initial conditions.
  class FunctionC0 : public Function<dim>
  {
    const double x0,y0,z0;
    const double radius;
  public:
    FunctionC0(const Mesh& mesh_) : x0{mesh_.x0}, y0{mesh_.y0}, z0{mesh_.z0}, radius(mesh_.radius)
    {}
    
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
  NonLinearParabolic3D(const Mesh &mesh_, 
                const unsigned int &r_,
                const double       &T_,
                const double       &deltat_,
                const unsigned int &outputPeriod_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , d(mesh_)
    , alpha(mesh_)
    , c_0(mesh_)
    , T(T_)
    , mesh_file_name(mesh_.mesh_file_name)
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
  
  // White/Gray matter visualization
  Vector<double> material_ids;

  // Axon visualization 
  std::unique_ptr<FESystem<dim>> axon_fe;
  DoFHandler<dim> axon_dof_handler;
  Vector<double> axonal_vector;
  struct AxonTensorFunction : public TensorFunction<1, dim, double>
  {
      const NonLinearParabolic3D &parent;
      AxonTensorFunction(const NonLinearParabolic3D &p) : TensorFunction<1, dim, double>(), parent(p) {}

      virtual Tensor<1, dim, double> value(const Point<dim, double> &p) const override
      {
          return parent.d.get_axon_at(p);
      }
  };
};

#endif