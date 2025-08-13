#include "NonLinearParabolic3D.hpp"

MeshData<3> MNI{
    "../mesh/MNI_mesh_ARuzza.msh", // mesh_file_name
    3,   // dext
    50,  // daxn
    1.2, // alpha_coeff_white
    0.6, // alpha_coeff_gray
    {    // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    },
    -40, -18.0, -10.0, // x0, y0, z0 (initial condition center coords)
    15, // radius
    10, // center_threshold
    {0.0, 0.0, 40.0}, // axonal_center
    60, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};

MeshData<3> Ernie{
    "../mesh/ernie_brain_dealii.msh", // mesh_file_name
    3,   // dext
    50,  // daxn
    1.2, // alpha_coeff_white
    0.6, // alpha_coeff_gray
    {    // material_names (id -> name)
        {1, "white matter"},
        {2, "gray matter"}
    },
    0, 0, 25.0, // x0, y0, z0 (initial condition center coords)
    15, // radius
    10, // center_threshold
    {0.0, 0.0, 0.0}, // axonal_center
    60, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};

MeshData<3> BrainCoarse{
    "../mesh/brain_coarse.msh", // mesh_file_name
    3,   // dext
    40,  // daxn
    1.2, // alpha_coeff_white
    0.6, // alpha_coeff_gray
    {    // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    },
    0.0, 0.0, 25.0, // x0, y0, z0 (initial condition center coords)
    20, // radius
    10, // center_threshold
    {0.0, 0.0, 0.0}, // axonal_center
    60, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};

MeshData<3> Cube40{
    "../mesh/mesh-cube-40.msh", // mesh_file_name
    1,  // dext
    10, // daxn
    1,  // alpha_coeff_white
    1,  // alpha_coeff_gray
    {   // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    },
    0.5, 0.0, 0.1, // x0, y0, z0 (initial condition center coords)
    0.3, // radius
    0.1,  // center_threshold
    {0.5, 0.5, 0.5}, // axonal_center
    0.25, // a (X axis)
    0.25, // b (Y axis)
    0.25  // c (Z axis)
};

MeshData<2> Sagittal{
    "../mesh/sagittal_mesh.msh", // mesh_file_name
    5,  // dext
    50, // daxn
    2,  // alpha_coeff_white
    1,  // alpha_coeff_gray
    {   // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    }, 
    230, 100, 0, // x0, y0, z0 (initial condition center coords)
    20, // radius
    5,  // center_threshold
    {190.0, 140.0}, // axonal_center
    70, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};

MeshData<2> Sagittal_whiteGrayDiff{
    "../mesh/sagittal_mesh.msh", // mesh_file_name
    5,  // dext
    50, // daxn
    0.2,  // alpha_coeff_white
    7,  // alpha_coeff_gray
    {   // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    }, 
    230, 100, 0, // x0, y0, z0 (initial condition center coords)
    20, // radius
    5,  // center_threshold
    {190.0, 140.0}, // axonal_center
    70, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};