#include "NonLinearParabolic3D.hpp"

MeshData<3> brain_coarse{
"../mesh/brain_coarse.msh",
// dext, daxn, 
3, 40,
// alpha white, gray
1.2, 0.6,   

// material map (id -> name)
{{0, "gray matter"}, {1, "white matter"}},

// x0,  y0,  z0,  radius
0.0, 0.0, 25.0, 20, 

10, // center_threshold 
{0.0, 0.0, 0.0}, //axonal_center

// a,  b,  c
60, 40, 30        
};

MeshData<3> cube40{
"../mesh/mesh-cube-40.msh",
// dext, daxn, 
1, 10,
// alpha white, gray
1, 1,   

// material map (id -> name)
{{0, "gray matter"}, {1, "white matter"}},

// x0,  y0,  z0,  radius
0.5, 0.0, 0.1, 0.3, 

0.1, // center_threshold 
{0.5, 0.5, 0.5}, //axonal_center

// a,  b,  c
0.25, 0.25, 0.25        
};

MeshData<2> sagittal{
"../mesh/sagittal_mesh.msh",
// dext, daxn, 
5, 50,
// alpha white, gray
2, 1,   

// material map (id -> name)
{{0, "gray matter"}, {1, "white matter"}},

// x0,  y0,  z0,  radius
230, 100, 0, 20, 

5, // center_threshold 
{190.0, 140.0}, //axonal_center

// a,  b,  c
70, 50, 30   
};


MeshData<3> MNI{
"../mesh/MNI_mesh_ARuzza.msh",
// dext, daxn, 
3, 50,
// alpha white, gray
1.2, 0.6,   

// material map (id -> name)
{{0, "gray matter"}, {1, "white matter"}},

// x0,  y0,  z0,  radius
-40.0, -18.0, -10.0, 15, 

10, // center_threshold 
{0.0, 0.0, 40.0}, //axonal_center

// a,  b,  c
60, 40, 30        
};