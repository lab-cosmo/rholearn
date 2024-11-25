# Part 1: Generate DFT data with `FHI-aims`

## 1.0: TLDR of requried commands

After modifying the user options in [dft-options.yaml](dft-options.yaml) and [hpc-options.yaml](hpc-options.yaml), the commands needed to generate data for training a model are below. For a full explanation of each, read on to the following sections.

```bash
# Modify dft-options.yaml and hpc-options.yaml as appropriate
# ...

# Run SCF
doslearn_run_scf

# Process outputs
doslearn_process_scf
```

## 1.1: Specify DFT and HPC options

Inspect the file [dft-options.yaml](dft-options.yaml) and edit the variables found there, specific for your set up.

You can also inspect the default DFT options, which can be printed with:
```python
import pprint
from rholearn.options import get_defaults

pprint.pprint(get_defaults("dft", "doslearn"))
```
Any of these can be modified by specification in the local file [dft-options.yaml](dft-options.yaml).

**Note**: the options in [hpc-options.yaml](hpc-options.yaml) will also need to be changed, depending on your cluster. The way that `rholearn.aims_interface` creates run scripts for HPC resources has only been tested on a handful of clusters, all with slurm schedulers. It is not completely general and may require some hacking if not compatible with your systems. The `"LOAD_MODULES"` and `"EXPORT_VARIABLES"` attempt to allows generic loading of modules and exporting of environment variables, respectively, but something may be missing.

## 1.2: Converge SCF

Run the SCF procedure. This submits a parallel array of SCF calculations for each structure in the dataset.

```python
from rholearn.aims_interface import scf

scf.doslearn_run_scf()
```
Alternatively, from the command line: `doslearn_run_scf`

After the calculation has finished, the run directory for each structure contains the following files:

```bash
raw/                                # Raw data directory
└── 0/                              # Directory for the first structure (index 0)
    ├── aims.out                    # Output file from FHI-aims SCF calculation
    ├── control.in                  # Input control file for FHI-aims SCF step
    ├── dft-options.yaml            # Copy of DFT options
    ├── Final_KS_eigenvalues.dat    # Output file of eigenvalues
    ├── geometry.in                 # Input file with atomic coordinates and species
    ├── hpc-options.yaml            # Copy of HPC options
    └── slurm_*.out                 # Output file from SLURM job scheduler

└── 1/
    ...
```

The calculation has (hopefully) converged to the SCF solution for the given input options.

Now process the SCF outputs. First, the "aims.out" file is parsed to extract various information and pickle it to file "calc_info.pickle" in the `FHI-aims` run directory.

Next, the eigenvalues contained in "Final_KS_eigenvalues.dat" are parsed and are used to construct the DOS via gaussian smearing of the eigenvalues. To facilitate shifting of the energy reference, the DOS are represented using Cubic Hermite Splines, which are constructed according to the splines settings `DOS_SPLINES` specified in [dft-options.yaml](dft-options.yaml). In the splines settings, `min_energy` and `max_energy` are used to define the energy window in which the splines will be computed for. Endpoints for each piece of the spline are distributed uniformly on the energy grid, with a spacing defined by `interval`. Then, `sigma` is used to denote the width of gaussian smearing used to define the DOS. These splines are then stored in `TensorMap` format and saved to a series of processed data directories at path (i.e. for structure index 0) `data/processed/0/dos/dos_spline.npz`. 

Splines generated with different parameters can be saved to different processed data directories by changing the default option for `RUN_DIR` in [dft-options.yaml](dft-options.yaml). 

```python
from rholearn.aims_interface import scf

scf.doslearn_process_scf()
```

Alternatively, from the command line: `doslearn_process_scf`

In the supporting notebook [part-1-dft](./part-1-dft.ipynb), SCF convergence can be checked.
