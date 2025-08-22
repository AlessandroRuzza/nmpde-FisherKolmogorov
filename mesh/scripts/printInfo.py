import meshio
import sys

if len(sys.argv) < 2:
    print("Usage: python printInfo.py <mesh_file>")
    sys.exit(1)

mesh = meshio.read(sys.argv[1], file_format="gmsh")

# Numero totale di punti (nodi)
n_points = mesh.points.shape[0]
print(f"Number of points: {n_points}")

# Numero totale di celle (elementi), per tipo
print("Number of cells by type:")
for cell_block in mesh.cells:
    cell_type = cell_block.type
    n_cells = len(cell_block.data)
    print(f"  {cell_type}: {n_cells}")