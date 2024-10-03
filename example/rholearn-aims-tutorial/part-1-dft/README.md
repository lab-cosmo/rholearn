# Part 1: Generate DFT data with `FHI-aims`

## 1.0: TLDR of requried commands

After modifying the user-setting in `dft_settings.py` and `hpc_settings.py`, the commands needed to generate data for training a model are below. For a full explanation of each, read on to the following sections.

```bash
# Modify dft_settings.py and hpc_settings.py as appropriate
# ...

# Run SCF
python -c 'from rholearn.aims_interface import scf; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; scf.run_scf(DFT_SETTINGS, HPC_SETTINGS);'

# Process SCF
python -c 'from rholearn.aims_interface import scf; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; scf.process_scf(DFT_SETTINGS, HPC_SETTINGS);'

# Setup RI
python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.set_up_ri_fit_sbatch(DFT_SETTINGS, HPC_SETTINGS);'

# Run RI
python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.run_ri_fit(DFT_SETTINGS, HPC_SETTINGS);'

# Process RI
python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.process_ri_fit(DFT_SETTINGS, HPC_SETTINGS)'
```

## 1.1: Specify DFT and HPC settings

Inspect the file `dft_settings.py` and edit the variables found there specific for your set up. `FRAME_IDXS` can be edited, though in the interest of brevity of the demonstration this can left alone. 

You can also inspect the default DFT settings, which can be printed with:
```python
import pprint
from rholearn.settings.defaults import dft_defaults

pprint.pprint(dft_defaults.DFT_DEFAULTS)
```
Any of these can be modified by specification in the local file `dft_settings.py`.

**Note**: the settings in `hpc_settings.py` will also need to be changed, depending on your cluster. The way that `rholearn.aims_interface` creates run scripts for HPC resources has only been tested on a handful of clusters, all with slurm schedulers. It is certainly not general and may require some hacking if not compatible with your systems. The `"load_modules"` and `"export_vars"` attempt to allows generic loading of modules and exporting of environment variables, respectively, but something may be missing.

## 1.2: Converge SCF

Run the SCF procedure. This submits a parallel array of SCF calculations for each structure in the dataset.

```python
from rholearn.aims_interface import scf
from dft_settings import DFT_SETTINGS
from hpc_settings import HPC_SETTINGS

scf.run_scf(DFT_SETTINGS, HPC_SETTINGS)

# Alternatively: a one-liner for the command line
python -c 'from rholearn.aims_interface import scf; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; scf.run_scf(DFT_SETTINGS, HPC_SETTINGS);'
```
After the calculation has finished, the run directory for each structure contains the following files:

```bash
raw/                                # Raw data directory
└── 0/                              # Directory for the first structure (index 0)
    ├── aims.out                    # Output file from FHI-aims SCF calculation
    ├── control.in                  # Input control file for FHI-aims SCF step
    ├── cube_001_total_density.cube # Cube file containing total electron density
    ├── D_spin_01_kpt_000001.csc    # Density matrix restart file
    ├── dft_settings.py             # Copy of python script with DFT settings
    ├── geometry.in                 # Input file with atomic coordinates and species
    └── slurm_*.out                 # Output file from SLURM job scheduler

└── 1/
    ...
```

The calculation has (hopefully) converged to the SCF solution for the given input settings, and saved the converged solution to the checkpoint density matrix file `D_*.csc`.

Now process the SCF outputs - this essentially just parses `aims.out` to extracts various information and pickles it to file `calc_info.pickle`.
```python
from rholearn.aims_interface import scf
from dft_settings import DFT_SETTINGS
from hpc_settings import HPC_SETTINGS

scf.process_scf(DFT_SETTINGS)

# Alternatively: a one-liner for the command line
python -c 'from rholearn.aims_interface import scf; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; scf.process_scf(DFT_SETTINGS, HPC_SETTINGS);'
```

In the supporting notebook [part-1-dft](./part-1-dft.ipynb), SCF convergence can be checked and reference SCF electron densitites visualised.


## 1.3: Perform RI decomposition

Now RI fitting can be performed. In `FHI-aims`, the following steps are executed:
* the density matrix restart files are read in as the SCF checkpoint,
* the real-space electron density is reconstructed from this density matrix,
* and then the scalar field is decomposed onto the RI basis and fitting coefficients calculated.

First, **create the input files** for the RI calculation.
```python
from rholearn.aims_interface import ri_fit
from dft_settings import DFT_SETTINGS
from hpc_settings import HPC_SETTINGS

ri_fit.set_up_ri_fit_sbatch(DFT_SETTINGS, HPC_SETTINGS)

# Alternatively: a one-liner for the command line
python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.set_up_ri_fit_sbatch(DFT_SETTINGS, HPC_SETTINGS);'
```

Next, **run the RI fitting** procedure.
```python
from rholearn.aims_interface import ri_fit
from dft_settings import DFT_SETTINGS
from hpc_settings import HPC_SETTINGS

ri_fit.run_ri_fit(DFT_SETTINGS, HPC_SETTINGS)

# Alternatively: a one-liner for the command line
python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.run_ri_fit(DFT_SETTINGS, HPC_SETTINGS);'
```

After the calculation has completed, the directory structure looks like:
```bash
raw/                                # Raw data directory
└── 0/                              # Directory for the first structure (index 0)
    └── edensity/                   # Run directory for the RI step
        ├── aims.out                # Output file from FHI-aims RI calculation
        ├── basis_info.out          # The RI basis set definition
        ├── control.in              # Input control file for FHI-aims RI step
        ├── D_*.csc                 # Symlink to the density matrix restart file
        ├── dft_settings.py         # Copy of python script with DFT settings
        ├── geometry.in             # Input file with atomic coordinates and species
        ├── partition_tab.out       # Output file with partitioning information
        ├── rho_rebuilt_ri.out      # Reconstructed electron density from RI fitting
        ├── rho_scf.out             # Electron density from SCF calculation
        ├── ri_ovlp.out             # Overlap matrix for RI basis
        ├── ri_projections.out      # Electron density projections on RI basis
        ├── ri_restart_ceoffs.out   # RI fitting coefficients
        └── slurm_*.out             # Output file from SLURM job scheduler
    ├── aims.out                    # Previous outputs from SCF step
    ├── ...                         # ...

└── 1/
    ...
```

Finally, **process the RI outputs**.

```python
from rholearn.aims_interface import ri_fit
from dft_settings import DFT_SETTINGS
from hpc_settings import HPC_SETTINGS

ri_fit.process_ri_fit(DFT_SETTINGS, HPC_SETTINGS)

# Alternatively: a one-liner for the command line
python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.process_ri_fit(DFT_SETTINGS, HPC_SETTINGS)'
```

This creates a set of subdirectories, one for each frame, containing the following processed data:
```bash
processed/                            # Processed data directory
└── 0/                                # Directory for the first structure (index 0)
    └── edensity/                     # Run directory for the RI step
        ├── basis_set.pickle          # RI basis definition, parsed from basis_info.out
        ├── calc_info.pickle          # Info parsed from aims.out for the RI step
        ├── df_error.pickle           # Calculated density fitting error metrics
        ├── ri_coeffs.npz             # metatensor TensorMap of the RI fitting coefficient vector
        ├── ri_ovlp.npz               # metatensor TensorMap of the RI overlap matrix
        └── ri_projs.npz              # metatensor TensorMap of the RI projection vector

└── 1/
    ...
```

The processed data contained in `processed/`, along with the `.xyz` file in `data/`, will be used as the reference data to train a surrogate model in the next step, the instructions for which can be found in [the next README](../part-2-ml/README.md).
