import meshio

mesh = meshio.read("MNI_mesh_ARuzza.inp", file_format="abaqus")
print(type(mesh))
if isinstance(mesh, meshio._mesh.Mesh):
    print(mesh.cell_data)
    print(mesh.cell_sets)
    print(mesh.cells)

    mesh.field_data = {
        "PT_GM": [0, 3],
        "PT_CSF": [1, 3],
        "PT_WM": [2, 3],
        "PT_VENT": [3, 3],
    }

    mesh.cell_sets_to_data()
    print(mesh.cell_data)
    mesh.cell_data["gmsh:physical"] = mesh.cell_data["PT_GM-PT_CSF-PT_WM-PT_VENT"]
    del mesh.cell_data["PT_GM-PT_CSF-PT_WM-PT_VENT"]
    print(mesh.cell_data)

print("Starting write.")
meshio.write("MNI_mesh_ARuzza_with_phys.msh", mesh, file_format="gmsh22", binary=False)

print("Done. New physical groups assigned and written to MNI_mesh_ARuzza_with_phys.msh")

# Check correct conversion
meshG = meshio.read("MNI_mesh_ARuzza_with_phys.msh", file_format="gmsh")
print("\nAfter reading back:")
print("cell_data_dict keys:", meshG.cell_data_dict.keys())
for cell_type, cell_data in meshG.cell_data_dict.items():
    print(f"Cell type: {cell_type}")
    for key, arr in cell_data.items():
        print(f"  {key}: {arr[:10]}")  # Print first 10 for brevity