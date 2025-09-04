import trimesh


mesh = trimesh.load_mesh("cube.stl")


components = mesh.split(only_watertight=False)

brain_component = min(components, key=lambda c: c.volume)

brain_component.export("brain_only.stl")

