module purge
module load gcc/11.2.0
module load openmpi/4.1.x
module load hdf5/1.14-parallel
module load openmpi/4.1.x
module load hdf5/1.14.x-parallel
module load cmake/3.24+       # >=3.18 va bene
module load boost/1.8x        # opzionale ma utile

mkdir -p $HOME/src && cd $HOME/src




wget https://zlib.net/zlib-1.3.1.tar.gz    # SHA256: 9a93b2b7dfdac77c...b72df23
tar xf zlib-1.3.1.tar.gz
cd zlib-1.3.1
./configure --prefix=$HOME/.local/zlib-1.3.1
make -j"$(nproc)" && make install


cd $HOME/src
wget https://github.com/dealii/dealii/releases/download/v9.6.0/dealii-9.6.0.tar.gz
tar xf dealii-9.6.0.tar.gz
mkdir -p dealii-9.6.0/build && cd dealii-9.6.0/build


cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/dealii-9.6 \
  -DDEAL_II_WITH_MPI=ON \
  -DDEAL_II_WITH_HDF5=ON \
  -DDEAL_II_WITH_ZLIB=ON \
  -DZLIB_INCLUDE_DIR=$HOME/.local/zlib-1.3.1/include \
  -DZLIB_LIBRARY=$HOME/.local/zlib-1.3.1/lib/libz.so \
  -DDEAL_II_COMPONENT_EXAMPLES=OFF \
  -DDEAL_II_COMPONENT_DOCUMENTATION=OFF \
  -DDEAL_II_WITH_TRILINOS=OFF -DDEAL_II_WITH_PETSC=OFF \
  -DDEAL_II_WITH_P4EST=OFF -DDEAL_II_WITH_METIS=OFF \
  -DDEAL_II_WITH_GSL=OFF -DDEAL_II_WITH_NETCDF=OFF

# Compila con pochi job e salva log
ninja -j2 2>&1 | tee build.log
ninja install




#!/usr/bin/env bash
set -euo pipefail

# --- parametri tuoi (se cambiano, modifica qui) ---
DEAL_VER="9.6"
DEAL_ROOT="$HOME/.local/dealii-${DEAL_VER}"
ZLIB_ROOT="$HOME/.local/zlib-1.3.1"   # se non esiste, il modulo lo ignora

# 0) hook per inizializzare "module" se non c'è
if ! command -v module >/dev/null 2>&1; then
  # prova i punti tipici di Lmod
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    [ -r "$f" ] && source "$f" && break
  done
fi
command -v module >/dev/null 2>&1 || { echo "Errore: comando 'module' non trovato"; exit 1; }

# 1) check install dealii
if [ ! -d "$DEAL_ROOT" ]; then
  echo "⚠️  Non trovo $DEAL_ROOT. Assicurati di aver installato deal.II qui."
  exit 1
fi

# 2) crea modulefile Lua
MOD_DIR="$HOME/.modules/dealii"
MOD_FILE="${MOD_DIR}/${DEAL_VER}.lua"
mkdir -p "$MOD_DIR"

cat > "$MOD_FILE" <<'LUA'
help([[
deal.II 9.6 (user install). Include HDF5 se la tua build lo ha abilitato.
]])
whatis("deal.II 9.6 (user)")

-- dipendenze (scommenta se servono i moduli di sistema)
-- load("gcc/11.2.0")
-- load("openmpi/4.1.6")
-- load("hdf5/1.14.2-parallel")

local home = os.getenv("HOME")
local root = pathJoin(home, ".local/dealii-9.6")   -- cambia qui se installi altrove

-- espone a CMake/compilatore
prepend_path("PATH",              pathJoin(root, "bin"))
prepend_path("LD_LIBRARY_PATH",   pathJoin(root, "lib"))
prepend_path("CMAKE_PREFIX_PATH", root)
setenv("DEAL_II_DIR", root)

-- opzionale: zlib user-local (se presente)
local zroot = pathJoin(home, ".local/zlib-1.3.1")
if (isDir(zroot)) then
  prepend_path("LD_LIBRARY_PATH", pathJoin(zroot, "lib"))
  prepend_path("CPATH",           pathJoin(zroot, "include"))
end
LUA

echo "✓ Creato modulefile: $MOD_FILE"

# 3) attiva il percorso moduli utente e carica dealii/9.6
module use "$HOME/.modules"
module --ignore_cache avail dealii || true
module --ignore_cache load "dealii/${DEAL_VER}"

# 4) verifica
if ! command -v dealii_info >/dev/null 2>&1; then
  echo "⚠️  'dealii_info' non nel PATH dopo il load. Controlla il modulefile."
  exit 1
fi

echo "---- dealii_info ----"
dealii_info | grep -E 'deal.II|HDF5|MPI|ZLIB' || dealii_info || true
echo "---------------------"

echo
echo "✅ Modulo 'dealii/${DEAL_VER}' caricato. Per usarlo nelle build:"
echo "  cmake -S . -B build"
echo "  # nel tuo CMakeLists.txt:  find_package(deal.II 9.6 EXACT CONFIG REQUIRED)"
echo "  cmake --build build -j"
