# Numerical Methods for PDEs: Fisher-Kolmogorov Equation for neurodegenerative diseases

This project implements a numerical solver for the Fisher-Kolmogorov equation using the deal.II finite element library. The solver supports both sequential and parallel (MPI) execution. The `speedup-analysis` folder contains performance analysis tools.


## Download the meshes
Mesh files should be placed in the `mesh/` folder. Download our meshes from [here](https://drive.google.com/drive/folders/1VbYBIZoS3r0KPoPtelDaFatVYBiAmmC3?usp=sharing)

## Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The executable will be created into `build`, and can be executed through
```bash
$ # Run sequentially
$ ./main <mesh_preset>
$ # Or run with mpi
$ mpirun -host localhost:$(nproc) -np <nThreads> ./main <mesh_preset>
```
Passing an unknown preset will print the list of valid meshes. \
Try `./main help` to see. 

The output .vtu and .pvtu files will be in `build/output`

## Speedup analysis
To perform speedup analysis and benchmarking:

```bash
$ # Run the analysis script
$ cd speedup-analysis
$ bash run_speedup.sh [output_csv]

$ # Example
$ bash run_speedup.sh speedup_results.csv

$ # Example with env variables
$ REPEAT=3 THREADS="8 16" MESH_PRESETS="Sagittal MNI" bash run_speedup.sh speedup_results.csv
```
This will run both sequential and MPI parallel versions with 2, 4, 8, 12, and 16 processes.

Optional environment variables (default values shown here):
- `MESH_PRESETS="Sagittal MNI"` - specify which mesh presets to test
- `THREADS="2 4 8 16"` - specify which process counts to test
- `REPEAT=1` - number of repetitions per configuration

```bash
$ # Parse and analyze results
$ python parsecsv.py

$ # Generate plot
$ python speedup_plot.py
```

The speedup analysis will generate:
- Sequential vs parallel execution times
- Speedup curves showing scaling efficiency
- Log files in `speedup-analysis/logs/`

