#include "FisherKolmogorov.hpp"

template<unsigned int dim>
MeshData<dim> get_mesh_data(const std::string &mesh_preset, unsigned int mpiRank);

MeshData<3> MNI{
    "../mesh/MNI_with_phys.msh", // mesh_file_name
    {    // material_names (id -> name)
        {0, "gray matter"},
        {1, "CSF"},
        {2, "white matter"},
        {3, "ventricule"}
    },
    {    // isotropic diffusion (name -> value)
        {"gray matter", 3},
        {"CSF", 0},
        {"white matter", 3},
        {"ventricule", 0}
    },
    {    // axonal diffusion (name -> value)
        {"gray matter", 50},
        {"CSF", 0},
        {"white matter", 50},
        {"ventricule", 0}
    },
    {    // alpha coefficients (name -> value)
        {"gray matter", 0.6},
        {"CSF", 0},
        {"white matter", 1.8},
        {"ventricule", 0}
    },
    -40, -18.0, -10.0, // x0, y0, z0 (initial condition center coords)
    15, // radius
    10, // center_threshold
    {0.0, 0.0, 40.0}, // axonal_center
    40, // a (X axis)
    60, // b (Y axis)
    30  // c (Z axis)
};

MeshData<3> Ernie{
    "../mesh/ernie_brain_dealii.msh", // mesh_file_name
    {    // material_names (id -> name)
        {1, "white matter"},
        {2, "gray matter"}
    },
    {    // isotropic diffusion (name -> value)
        {"white matter", 3},
        {"gray matter", 3}
    },
    {    // axonal diffusion (name -> value)
        {"white matter", 50},
        {"gray matter", 50}
    },
    {    // alpha coefficients (name -> value)
        {"white matter", 1.2},
        {"gray matter", 0.6}
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
    {    // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    },
    {    // isotropic diffusion (name -> value)
        {"white matter", 3},
        {"gray matter", 3}
    },
    {    // axonal diffusion (name -> value)
        {"white matter", 40},
        {"gray matter", 40}
    },
    {    // alpha coefficients (name -> value)
        {"gray matter", 0.6},
        {"white matter", 1.2}
    },
    0.0, 0.0, 25.0, // x0, y0, z0 (initial condition center coords)
    20, // radius
    10, // center_threshold
    {0.0, 0.0, 0.0}, // axonal_center
    40, // a (X axis)
    60, // b (Y axis)
    30  // c (Z axis)
};

MeshData<3> Cube40{
    "../mesh/mesh-cube-40.msh", // mesh_file_name
    {   // material_names (id -> name)
        {0, "matter"},
        {10, "matter"}
    },
    {    // isotropic diffusion (name -> value)
        {"matter", 0.003}
    },
    {    // axonal diffusion (name -> value)
        {"matter", 0.01}
    },
    {    // alpha coefficients (name -> value)
        {"matter", 5}
    },
    0.5, 0.0, 0.1, // x0, y0, z0 (initial condition center coords)
    0.05, // radius
    0.05,  // center_threshold
    {0.5, 0.5, 0.5}, // axonal_center
    0.25, // a (X axis)
    0.25, // b (Y axis)
    0.25  // c (Z axis)
};

MeshData<2> Sagittal{
    "../mesh/sagittal.msh", // mesh_file_name
    {   // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    }, 
    {    // isotropic diffusion (name -> value)
        {"white matter", 5},
        {"gray matter", 5}
    },
    {    // axonal diffusion (name -> value)
        {"white matter", 50},
        {"gray matter", 50}
    },
    {    // alpha coefficients (name -> value)
        {"gray matter", 1},
        {"white matter", 2}
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
    "../mesh/sagittal.msh", // mesh_file_name
    {   // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    }, 
    {    // isotropic diffusion (name -> value)
        {"white matter", 20},
        {"gray matter", 0.05}
    },
    {    // axonal diffusion (name -> value)
        {"white matter", 0.05},
        {"gray matter", 30}
    },
    {    // alpha coefficients (name -> value)
        {"gray matter", 1},
        {"white matter", 2}
    },
    230, 100, 0, // x0, y0, z0 (initial condition center coords)
    20, // radius
    5,  // center_threshold
    {190.0, 140.0}, // axonal_center
    70, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};

void printErrValidOptions(const std::string& mesh_preset, unsigned int mpiRank){
    if (mpiRank != 0) return;
    std::cerr << "Unknown mesh preset: " << mesh_preset << std::endl;
    std::cerr << "Valid options are: " << std::endl;

    std::cerr << "2D: " << std::endl;
    std::cerr << " - Sagittal" << std::endl;
    std::cerr << " - Sagittal_whiteGrayDiff" << std::endl;

    std::cerr << "3D: " << std::endl;
    std::cerr << " - MNI" << std::endl;
    std::cerr << " - Ernie" << std::endl;
    std::cerr << " - BrainCoarse" << std::endl;
    std::cerr << " - Cube40" << std::endl;
}

inline void printRankZero(const std::string& msg, unsigned int mpiRank){
    if (mpiRank == 0) std::cout << msg << std::endl;
}

unsigned int get_mesh_dimension(const std::string& mesh_preset, unsigned int mpiRank){
    if(mesh_preset == "MNI" || mesh_preset == "Ernie" || mesh_preset == "BrainCoarse" || mesh_preset == "Cube40"){
        return 3;
    } else if(mesh_preset == "Sagittal" || mesh_preset == "Sagittal_whiteGrayDiff"){
        return 2;
    }
    else{
        printErrValidOptions(mesh_preset, mpiRank);
        exit(-1);
    }
}

template<>
MeshData<3> get_mesh_data(const std::string& mesh_preset, unsigned int mpiRank){
    if(mesh_preset == "MNI"){
        printRankZero("Using MNI mesh.", mpiRank);
        return MNI;
    } else if(mesh_preset == "Ernie"){
        printRankZero("Using Ernie mesh.", mpiRank);
        return Ernie;
    } else if(mesh_preset == "BrainCoarse"){
        printRankZero("Using BrainCoarse mesh.", mpiRank);
        return BrainCoarse;
    } else if(mesh_preset == "Cube40"){
        printRankZero("Using Cube40 mesh.", mpiRank);
        return Cube40;
    } else {
        printErrValidOptions(mesh_preset, mpiRank);
        exit(-1);
    }
}
template<>
MeshData<2> get_mesh_data(const std::string& mesh_preset, unsigned int mpiRank){
    if(mesh_preset == "Sagittal"){
        printRankZero("Using Sagittal mesh.", mpiRank);
        return Sagittal;
    } else if(mesh_preset == "Sagittal_whiteGrayDiff"){
        printRankZero("Using Sagittal mesh with more pronounced differences between white and gray matter.", mpiRank);
        return Sagittal_whiteGrayDiff;
    } else {
        printErrValidOptions(mesh_preset, mpiRank);
        exit(-1);
    }
}
