import meshio
import numpy as np

SRC = "m2m_ernie/ernie.msh"
DST = "ernie_brain.msh"

m = meshio.read(SRC)

# --- prendi il blocco dei tetra (volumi) ---
tet_idx = [i for i, b in enumerate(m.cells) if b.type == "tetra"][0]
tet = m.cells[tet_idx].data
tags = m.cell_data["gmsh:physical"][tet_idx]

# --- maschere per WM(1) e GM(2) ---
wm_mask = tags == 1
gm_mask = tags == 2

assert wm_mask.any() and gm_mask.any(), "Nel file non trovo entrambi i tag 1 e 2!"

tet_wm = tet[wm_mask]
tet_gm = tet[gm_mask]

# --- tieni solo i nodi usati e rinumera ---
used = np.unique(np.concatenate([tet_wm, tet_gm]))
old2new = -np.ones(len(m.points), dtype=int)
old2new[used] = np.arange(len(used))

pts = m.points[used]
tet_wm = old2new[tet_wm]
tet_gm = old2new[tet_gm]

# --- due blocchi separati (entrambi 'tetra') ---
cells_out = [
    ("tetra", tet_wm),   # blocco 0 = WM
    ("tetra", tet_gm),   # blocco 1 = GM
]

# --- scrivi anche i tag fisici e (importante) i 'geometrical' diversi ---
cell_data_out = {
    "gmsh:physical": [
        np.full(len(tet_wm), 1, dtype=int),   # WM
        np.full(len(tet_gm), 2, dtype=int),   # GM
    ],
    "gmsh:geometrical": [
        np.full(len(tet_wm), 101, dtype=int), # entity diversa per WM
        np.full(len(tet_gm), 102, dtype=int), # entity diversa per GM
    ],
}

# --- PhysicalNames per i volumi (dim = 3) ---
field_out = {
    "WhiteMatter": (1, 3),
    "GrayMatter":  (2, 3),
}

brain = meshio.Mesh(
    points=pts,
    cells=cells_out,
    cell_data=cell_data_out,
    field_data=field_out,
)

# MSH 2 ASCII: ParaView lo digerisce bene
meshio.write(DST, brain, file_format="gmsh22")
print("✓ Salvato", DST)
