#!/bin/bash

# Exit on error
set -e

# Install dependencies for Apptainer (Singularity)
sudo apt-get update
sudo apt-get install -y \
    uuid-dev \
    build-essential \
    libseccomp-dev \
    pkg-config \
    squashfs-tools \
    cryptsetup \
    curl wget git \
    libglib2.0-dev \
    libssl-dev \
    libgpgme-dev \
    libseccomp-dev \
    libselinux1-dev
    
# Install latest Go (required for Apptainer)
GO_VERSION=$(curl -s https://go.dev/VERSION?m=text | head -n 1)
wget https://go.dev/dl/${GO_VERSION}.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf ${GO_VERSION}.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

go version

# Install latest Apptainer
APPTAINER_VERSION=$(curl -s https://api.github.com/repos/apptainer/apptainer/releases/latest | grep 'tag_name' | cut -d\" -f4)
APPTAINER_VERSION_STRIPPED=${APPTAINER_VERSION#v}
wget https://github.com/apptainer/apptainer/releases/download/${APPTAINER_VERSION}/apptainer-${APPTAINER_VERSION_STRIPPED}.tar.gz

tar -xzf apptainer-${APPTAINER_VERSION_STRIPPED}.tar.gz
cd apptainer-${APPTAINER_VERSION_STRIPPED}
./mconfig --with-suid
make -C builddir
sudo make -C builddir install
cd ..

# Verify Apptainer version
apptainer --version

# Clean up build directory
rm -rf apptainer-${APPTAINER_VERSION_STRIPPED} apptainer-${APPTAINER_VERSION_STRIPPED}.tar.gz ${GO_VERSION}.linux-amd64.tar.gz

# Build Apptainer Image (uncomment to use)
# sudo apptainer build pde_Fisher.sif Singularity.def