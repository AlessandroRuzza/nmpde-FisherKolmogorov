import sys
from collections import Counter

def read_nodes(filename):
    nodes = []
    node_categories = []
    with open(filename) as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"Skipping line {line_num} in nodes file: not enough values", file=sys.stderr)
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                cat = int(parts[2]) # third value is category, not z
                nodes.append((x, y))
                node_categories.append(cat)
            except ValueError:
                print(f"Skipping line {line_num} in nodes file: could not parse as floats/ints", file=sys.stderr)
                continue
    return nodes, node_categories

def read_elements(filename):
    elements = []
    with open(filename) as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"Skipping line {line_num} in elements file: not enough node indices", file=sys.stderr)
                continue
            try:
                node_ids = [int(n) for n in parts]
                if len(node_ids) not in (3, 4):
                    print(f"Skipping line {line_num} in elements file: unsupported element type ({len(node_ids)} nodes)", file=sys.stderr)
                    continue
                elements.append(node_ids)
            except ValueError:
                print(f"Skipping line {line_num} in elements file: could not parse indices as integers", file=sys.stderr)
                continue
    return elements

def get_majority_category(node_indices, node_categories):
    cats = [node_categories[idx-1] for idx in node_indices]  # node indices start at 1
    c = Counter(cats)
    majority = c.most_common(1)[0][0]
    return majority

def main():
    node_file = sys.argv[1] if len(sys.argv) > 1 else "data_sagittal_node.dat"
    elem_file = sys.argv[2] if len(sys.argv) > 2 else "data_sagittal_elem.dat"
    msh_file = sys.argv[3] if len(sys.argv) > 3 else "sagittal_mesh.msh"
    cat_file = sys.argv[4] if len(sys.argv) > 4 else "sagittal_mesh.elem_categories.txt"

    nodes, node_categories = read_nodes(node_file)
    elements = read_elements(elem_file)

    # Write mesh file
    with open(msh_file, "w") as msh:
        msh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Add $PhysicalNames section
        msh.write("$PhysicalNames\n2\n")
        msh.write("2 0 \"White_matter\"\n")
        msh.write("2 1 \"Grey_matter\"\n")
        msh.write("$EndPhysicalNames\n")

        msh.write(f"$Nodes\n{len(nodes)}\n")
        for i, n in enumerate(nodes, start=1):
            msh.write(f"{i} {n[0]} {n[1]} 0.0\n")  # 2D mesh: z=0.0
        msh.write("$EndNodes\n")

        msh.write(f"$Elements\n{len(elements)}\n")
        for i, node_ids in enumerate(elements, start=1):
            if len(node_ids) == 3:
                gmsh_type = 2  # triangle
            elif len(node_ids) == 4:
                gmsh_type = 3  # quadrilateral
            else:
                print(f"Skipping element {i}: unsupported element type ({len(node_ids)} nodes)", file=sys.stderr)
                continue
            majority_cat = get_majority_category(node_ids, node_categories)
            msh.write(f"{i} {gmsh_type} 2 {majority_cat} 0 " + " ".join(str(nid) for nid in node_ids) + "\n")
        msh.write("$EndElements\n")

if __name__ == "__main__":
    main()