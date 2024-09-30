"""
Settings for running calculations on HPC clusters.

Any variables defined here must be included in the final dictionary `HPC_SETTINGS` at
the end of this file.
"""

# Slurm scheduler parameters
SLURM_PARAMS = {
    "job-name": "qm7",
    "nodes": 1,
    "time": "00:15:00",
    "mem-per-cpu": 0,
    "partition": "jobs",
    "ntasks-per-node": 18,
}

# Cluster settings to include in run scripts
HPC = {
    "load_modules": [  # modules to load, i.e. "module load X"
        "intel", "intel-mpi", "intel-mkl"
    ],
    "export_vars": [  # environment variables to export, i.e. "export X"
        "OMP_NUM_THREADS=1",
        "MKL_DYNAMIC=FALSE",
        "MKL_NUM_THREADS=1",
        "I_MPI_FABRICS=shm",
    ],
}

# Final dictionary of HPC settings
HPC_SETTINGS = {
    "SLURM_PARAMS": SLURM_PARAMS,
    "HPC": HPC,
}