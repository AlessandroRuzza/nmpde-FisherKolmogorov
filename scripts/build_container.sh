#!/bin/bash

# Exit on error
set -e

# Verify Apptainer version
apptainer --version

# Build Apptainer Image
sudo apptainer build pde_Fisher.sif apptainer.def