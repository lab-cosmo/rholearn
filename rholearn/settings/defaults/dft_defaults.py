"""
Default settings for running DFT and RI procedures in FHI-aims.
"""

# Physical settings for running FHI-aims
BASE_AIMS = {
    "xc": "pbe0",
    "spin": "none",
    "charge": 0,
    "relativistic": "atomic_zora scalar",
}

# For SCF procedure
SCF = {
    "elsi_restart": "write 1000",
    "output": "cube total_density",
}

# For RI-fitting procedure
RI = {
    "ri_full_output": True,
    "ri_density_restart": "write",
    "default_prodbas_acc": 1e-5,
    "elsi_restart": "read",
    "ri_skip_scf": True,
    "ri_ovlp_write_triu": True,
}

# For rebuilding fields
REBUILD = {
    "ri_density_restart": "read 0",
    "ri_full_output": True,
    "default_max_l_prodbas": RI.get("default_max_l_prodbas"),
    "default_prodbas_acc": RI.get("default_prodbas_acc"),
    "output": "cube total_density",
}

# Cube grid settings
CUBE = {
    "n_points": (100, 100, 100),
}

# A unique ID for the RI fit procedure
RI_FIT_ID = "edensity"

# Final dictionary of DFT settings
DFT_DEFAULTS = {
    "BASE_AIMS": BASE_AIMS,
    "SCF": SCF,
    "RI": RI,
    "REBUILD": REBUILD,
    "CUBE": CUBE,
    "RI_FIT_ID": RI_FIT_ID,
    "FIELD_NAME": "edensity",
    "MASK": None,
}