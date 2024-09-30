"""
For running FHI-aims calculations on an HPC cluster.
"""

import datetime
import os
from os.path import join
from typing import Callable, Dict, List, Optional

import numpy as np


def write_aims_sbatch_array(
    fname: str,
    aims_command: str,
    array_idxs: List[int],
    run_dir: Callable,
    dm_restart_dir: Optional[Callable] = None,
    load_modules: Optional[List[str]] = None,
    export_vars: Optional[List[str]] = None,
    slurm_params: Optional[Dict[str, str]] = None,
) -> None:
    """
    Writes a slurm sbatch script to run FHI-aims.

    Assumes control.in and geometry.in files are already where they need to be.
    """
    with open(fname, "wt") as f:

        # Write the header
        f.write("#!/bin/bash\n")

        # Write the slurm parameters
        for tag, val in slurm_params.items():
            f.write(f"#SBATCH --{tag}={val}\n")

        # Write the array indices
        array_idxs = np.sort(array_idxs)
        array_range = np.arange(array_idxs[0], array_idxs[-1] + 1)
        if len(array_idxs) == len(array_range):
            if np.all(array_idxs == array_range):
                f.write(f"#SBATCH --array={array_idxs[0]}-{array_idxs[-1]}\n")
        else:
            f.write(f"#SBATCH --array={','.join(map(str, array_idxs))}\n")

        # slurm_out to the run dir, with a timestamp
        if "output" not in slurm_params:
            f.write(
                f"#SBATCH --output={join(run_dir('%a'), f'slurm_{timestamp()}.out')}\n"
            )

        # Load modules
        if load_modules is not None:
            f.write("\n# Load modules, set env variables, increase stack size\n")
            f.write(f"module load {' '.join(load_modules)}\n\n")

        # Some environment varibales that need to be set
        if export_vars is not None:
            f.write("# Set environment variables\n")
            for var in export_vars:
                f.write(f"export {var}\n")
            f.write("\n")

        # Increase stack size to unlim
        f.write("ulimit -s unlimited\n")
        f.write("\n")

        # Get the structure idx in the sbatch array
        f.write("# Get the structure idx from the SLURM job ID\n")
        f.write("ARRAY_IDX=${SLURM_ARRAY_TASK_ID}\n\n")

        # Define the run directory and cd to it
        f.write("# Define the run directory and cd into it\n")
        f.write(f"RUNDIR={run_dir('${ARRAY_IDX}')}\n")
        f.write("cd $RUNDIR\n\n")

        # Create a symbolic link to the density matrix restart directory
        if dm_restart_dir is not None:
            f.write("# Create symlink to DM restart dir\n")
            f.write(f"RESTART_FILE_DIR={dm_restart_dir('${ARRAY_IDX}')}\n")
            f.write("for rfile in $(ls $RESTART_FILE_DIR/D_spin_*); do ln -s $rfile $(basename $rfile); done\n")

        # Write the run AIMS command
        f.write("# Run AIMS\n")
        f.write(aims_command)

    return

def write_python_sbatch_array(
    fname: str,
    array_idxs: List[int],
    run_dir: Callable,
    python_command: str,
    slurm_params: dict,
) -> None:
    """
    Writes a slurm sbatch script to run python code.
    """
    with open(fname, "wt") as f:

        # Write the header
        f.write("#!/bin/bash\n")

        # Write the slurm parameters
        for tag, val in slurm_params.items():
            f.write(f"#SBATCH --{tag}={val}\n")

        # Write the array indices
        array_idxs = np.sort(array_idxs)
        array_range = np.arange(array_idxs[0], array_idxs[-1] + 1)
        if len(array_idxs) == len(array_range):
            if np.all(array_idxs == array_range):
                f.write(f"#SBATCH --array={array_idxs[0]}-{array_idxs[-1]}\n")
        else:
            f.write(f"#SBATCH --array={','.join(map(str, array_idxs))}\n")

        # slurm_out to the run dir, with a timestamp
        if "output" not in slurm_params:
            f.write(
                f"#SBATCH --output={join(run_dir('%a'), f'slurm_{timestamp()}.out')}\n"
            )

       # Get the structure idx in the sbatch array
        f.write("\n# Get the structure idx from the SLURM job ID\n")
        f.write("ARRAY_IDX=${SLURM_ARRAY_TASK_ID}\n\n")

        # Define the run directory and cd to it
        f.write("# Define the run directory and cd into it\n")
        f.write(f"RUNDIR={run_dir('${ARRAY_IDX}')}\n")
        f.write("cd $RUNDIR\n\n")

        # Write the run python command
        f.write("# Run Python\n")
        f.write(f"{python_command}\n")

    return     


def run_script(run_dir: str, command: str):
    """
    Runs the ``command`` in directory ``run_dir``.
    """
    curr_dir = os.getcwd()
    os.chdir(path=run_dir)
    os.system(command)
    os.chdir(path=curr_dir)


def aims_is_finished(run_dir: str) -> bool:
    """
    If 'Leaving FHI-aims.' is found in aims.out, returns True. Returns false otherwise.
    """
    aims_out = join(run_dir, "aims.out")
    if os.path.exists(aims_out):
        with open(aims_out, "r") as f:
            return "Leaving FHI-aims." in f.read()
    return False


def timestamp() -> str:
    """Return a timestamp string in format YYYYMMDDHHMMSS."""
    return datetime.datetime.today().strftime("%Y%m%d%H%M%S")
