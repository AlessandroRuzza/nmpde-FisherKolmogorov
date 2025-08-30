## Organizing the source code
Please place all your sources into the `src` folder.

Binary files must not be uploaded to the repository (including executables).

Mesh files should not be uploaded to the repository. If applicable, upload `gmsh` scripts with suitable instructions to generate the meshes (and ideally a Makefile that runs those instructions). If not applicable, consider uploading the meshes to a different file sharing service, and providing a download link as part of the building and running instructions.

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

### Download the meshes
Mesh files should be placed in the `mesh/` folder. Download our meshes from [here]()

## Speedup analysis
To perform speedup analysis and benchmarking:

```bash
$ # Run the analysis script
$ cd speedup-analysis
$ bash run_speedup.sh [mesh_preset] [output_csv]

$ # Example
$ bash run_speedup.sh Sagittal speedup_results.csv

$ # Example with env variable
$ REPEAT=3 THREADS="8 16" bash run_speedup.sh Sagittal speedup_results.csv
```
This will run both sequential and MPI parallel versions with 2, 4, 8, 12, and 16 processes.

Optional environment variables (default values shown here):
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

