# Part 1: Generate DFT data with `FHI-aims`

## 1.0: TLDR of requried commands

After modifying the user options in `dft-options.yaml` and `hpc-options.yaml`, the commands needed to generate data for training a model are below. For a full explanation of each, read on to the following sections.

```bash
# Modify dft-options.yaml and hpc-options.yaml as appropriate
# ...

# Run SCF
rholearn_run_scf
```

## 1.1: Specify DFT and HPC options

Inspect the file `dft-options.yaml` and edit the variables found there, specific for your set up.
Note that the option `DOS` must be set to `true` for generating an output file of eigenvalues necessary
for DOS-learning.

You can also inspect the default DFT options, which can be printed with:
```python
import pprint
from rholearn.options import get_defaults

pprint.pprint(get_defaults("dft", "doslearn"))
```
Any of these can be modified by specification in the local file `dft-options.yaml`.

**Note**: the options in `hpc-options.yaml` will also need to be changed, depending on your cluster. The way that `rholearn.aims_interface` creates run scripts for HPC resources has only been tested on a handful of clusters, all with slurm schedulers. It is not completely general and may require some hacking if not compatible with your systems. The `"LOAD_MODULES"` and `"EXPORT_VARIABLES"` attempt to allows generic loading of modules and exporting of environment variables, respectively, but something may be missing.

## 1.2: Converge SCF

Run the SCF procedure. This submits a parallel array of SCF calculations for each structure in the dataset.

```python
from rholearn.aims_interface import scf

scf.run_scf()
```
Alternatively, from the command line: `rholearn_run_scf`

After the calculation has finished, the run directory for each structure contains the following files:

```bash
raw/                                # Raw data directory
└── 0/                              # Directory for the first structure (index 0)
    ├── aims.out                    # Output file from FHI-aims SCF calculation
    ├── control.in                  # Input control file for FHI-aims SCF step
    ├── cube_001_total_density.cube # Cube file containing total electron density
    ├── D_spin_01_kpt_000001.csc    # Density matrix restart file
    ├── dft-options.yaml            # Copy of DFT options
    ├── Final_KS_eigenvalues.dat    # Output file of eigenvalues
    ├── geometry.in                 # Input file with atomic coordinates and species
    ├── hpc-options.yaml            # Copy of HPC options
    └── slurm_*.out                 # Output file from SLURM job scheduler

└── 1/
    ...
```

The calculation has (hopefully) converged to the SCF solution for the given input options, and saved the converged solution to the checkpoint density matrix file `D_*.csc`.

Now process the SCF outputs - this essentially just parses `aims.out` to extracts various information and pickles it to file `calc_info.pickle`.
```python
from rholearn.aims_interface import scf

scf.process_scf()
```

Alternatively, from the command line: `rholearn_process_scf`

In the supporting notebook [part-1-dft](./part-1-dft.ipynb), SCF convergence can be checked.
