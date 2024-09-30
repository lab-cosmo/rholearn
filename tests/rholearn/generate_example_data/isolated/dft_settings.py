"""
User settings for running DFT and RI procedures in FHI-aims. Use this file to overwrite
settings found in :py:mod:`rholearn.settings.defaults.dft_defaults`. 

For global variables that are dictionaries, the user settings are used to 'update' the
defaults - thus only overwriting the keys that are present in the user settings.

Any variables defined here must be included in the final dictionary `DFT_SETTINGS` at
the end of this file.
"""

# Absolute path to the top directory where FHI-aims output data will be stored
_ISOLATED_DIR = "path/to/rholearn/tests/rholearn/generate_example_data/isolated/"
DATA_DIR = _ISOLATED_DIR + "data/"

# Absolute path to .xyz file containing dataset
XYZ = DATA_DIR + "isolated.xyz"

# List of frame indices for the above dataset to compute reference data for
FRAME_IDXS = [0]

# The FHI-aims run command, including absolute path to aims executable
AIMS_COMMAND = "srun /path/to/aims.x < control.in > aims.out"

# Absolute path to species defaults directory
SPECIES_DEFAULTS = _ISOLATED_DIR + "aims_species/edensity/tight/default"

# Other user settings here:
# ...

# Physical settings for running FHI-aims
BASE_AIMS = {
    "xc": "pbe",
    "spin": "none",
    "charge": 0,
    "relativistic": "atomic_zora scalar",
}

# For SCF procedure
SCF = {
    "elsi_restart": "write 1000",
    # "output": "cube total_density",
}

# Final dictionary of DFT settings
DFT_SETTINGS = {
    "DATA_DIR": DATA_DIR,
    "XYZ": XYZ,
    "FRAME_IDXS": FRAME_IDXS,
    "AIMS_COMMAND": AIMS_COMMAND,
    "SPECIES_DEFAULTS": SPECIES_DEFAULTS,
    "BASE_AIMS": BASE_AIMS,
    "SCF": SCF,
}