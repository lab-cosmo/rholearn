"""
User settings for running DFT and RI procedures in FHI-aims. Use this file to overwrite
settings found in :py:mod:`rholearn.settings.defaults.dft_defaults`. 

For global variables that are dictionaries, the user settings are used to 'update' the
defaults - thus only overwriting the keys that are present in the user settings.

Any variables defined here must be included in the final dictionary `DFT_SETTINGS` at
the end of this file.
"""
from os.path import join

# Absolute path to the top directory where FHI-aims output data will be stored
TUTORIAL_DIR = "/path/to/rholearn/example/rholearn-aims-tutorial"
DATA_DIR = join(TUTORIAL_DIR, "part-1-dft/data/")

# Absolute path to .xyz file containing dataset
XYZ = join(DATA_DIR, "qm7.xyz")

# List of frame indices for the above dataset to compute reference data for
FRAME_IDXS = list(range(32))

# The FHI-aims run command, including absolute path to aims executable
AIMS_COMMAND = "srun /path/to/aims.x < control.in > aims.out"

# Absolute path to species defaults directory
SPECIES_DEFAULTS = join(TUTORIAL_DIR, "part-1-dft/species_defaults/light/default")

# Other user settings here:
# ...

# Final dictionary of DFT settings
DFT_SETTINGS = {
    "DATA_DIR": DATA_DIR,
    "XYZ": XYZ,
    "FRAME_IDXS": FRAME_IDXS,
    "AIMS_COMMAND": AIMS_COMMAND,
    "SPECIES_DEFAULTS": SPECIES_DEFAULTS,
}