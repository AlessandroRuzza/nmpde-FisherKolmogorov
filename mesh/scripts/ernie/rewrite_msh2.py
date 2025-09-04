import sys
import meshio
import numpy as np
from pathlib import Path

def fail(msg):
    print("ERROR:", msg)
    sys.exit(1)

if len(sys.argv) != 3:
    print("Uso: python3 to_msh22_minimal.py IN.msh OUT.msh")
    sys.exit(1)

IN, OUT = sys.argv[1], sys.argv[2]

# 1) Leggi con meshio
try:
    m = meshio.read(IN)
except Exception as e:
    fail(f"Impossibile leggere {IN}: {e}")

# 2) Prendi i tetra e (se ci sono) i tag fisici
tet = None
tags = None

# trova il blocco 'tetra' in modo robusto
tet_idx = None
for i, cb in enumerate(m.cells):
    if getattr(cb, "type", getattr(cb, "cell_type", None)) == "tetra":
        tet_idx = i
        break
if tet_idx is None:
    fail("Nessun blocco 'tetra' trovato.")

# estrai con compatibilità diverse versioni meshio
cb = m.cells[tet_idx]
tet = getattr(cb, "data", None)
if tet is None:
    fail("Blocco 'tetra' senza data.")

# prova a prendere i tag fisici per i tetra
try:
    # meshio >=5
    tags = m.cell_data_dict.get("gmsh:physical", {}).get("tetra", None)
except Exception:
    tags = None

if tags is None:
    # fallback per meshio <5
    try:
        tags = m.cell_data["gmsh:physical"][tet_idx]
    except Exception:
        # se non ci sono tag, setta tutto a 1
        tags = np.ones(len(tet), dtype=int)

# 3) Teniamo solo i nodi usati dai tetra, rinumeriamo 1-based
used = np.unique(tet.ravel())
old2new = -np.ones(len(m.points), dtype=np.int64)
old2new[used] = np.arange(1, len(used) + 1, dtype=np.int64)  # 1-based per MSH2
pts = m.points[used]
tet_new = old2new[tet]

# 4) Scrivi MSH 2.2 ASCII minimale SENZA BOM e con \n Unix
N = len(pts)
E = len(tet_new)

with open(OUT, "w", newline="\n") as f:
    # Header
    f.write("$MeshFormat\n")
    f.write("2.2 0 8\n")
    f.write("$EndMeshFormat\n")
    # Nodes
    f.write("$Nodes\n")
    f.write(f"{N}\n")
    # id x y z (id 1-based)
    for i, (x, y, z) in enumerate(pts, start=1):
        # stampa in modo pulito (nessun sci-not, nessun CRLF)
        f.write(f"{i} {x:.16g} {y:.16g} {z:.16g}\n")
    f.write("$EndNodes\n")
    # Elements
    f.write("$Elements\n")
    f.write(f"{E}\n")
    # riga elemento: <id> <type=4> <numTags=2> <physTag> <geomTag> n1 n2 n3 n4
    # usiamo geomTag=1 fisso (deal.II lo ignora)
    for eid, (n1, n2, n3, n4), t in zip(range(1, E + 1), tet_new, tags):
        f.write(f"{eid} 4 2 {int(t)} 1 {n1} {n2} {n3} {n4}\n")
    f.write("$EndElements\n")

print(f"✓ Scritto {OUT} — Nnodes={N}, Nelems={E}")
