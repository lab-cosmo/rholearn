"""
Module for defining and running HPC scripts for model training.
"""

import os
import shutil
from os.path import exists, join


def run_training_sbatch(run_dir: str, python_command: str, **kwargs) -> None:
    """
    Writes a bash script to `fname` that allows running of model training.
    `run_dir` must contain two files; "run_training.py" and "settings.py".
    """
    top_dir = os.getcwd()

    # Copy training script and settings
    shutil.copy(join(top_dir, "ml_settings.py"), join(run_dir, "ml_settings.py"))

    os.chdir(run_dir)
    fname = "run_training.sh"

    with open(join(run_dir, fname), "w+") as f:
        # Make a dir for the slurm outputs
        if not exists(join(run_dir, "slurm_out")):
            os.mkdir(join(run_dir, "slurm_out"))

        f.write("#!/bin/bash\n")  # Write the header
        for tag, val in kwargs.items():  # Write the sbatch parameters
            f.write(f"#SBATCH --{tag}={val}\n")
        f.write(f"#SBATCH --output={join(run_dir, 'slurm_out', 'slurm_train.out')}\n")
        f.write("#SBATCH --get-user-env\n\n")

        # Define the run directory, cd to it, run command
        f.write(f"RUNDIR={run_dir}\n")
        f.write("cd $RUNDIR\n\n")
        f.write(f"{python_command}\n")

    os.system(f"sbatch {fname}")
    os.chdir(top_dir)
