"""
Module for HPC utilities.
"""
import os
from os.path import join
from rholearn.utils.utils import timestamp

def write_python_sbatch(
    fname: str,
    run_dir: str,
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

        # slurm_out to the run dir, with a timestamp
        if "output" not in slurm_params:
            f.write(
                f"#SBATCH --output={join(run_dir, f'slurm_{timestamp()}.out')}\n"
            )

        # Define the run directory and cd to it
        f.write("# Define the run directory and cd into it\n")
        f.write(f"RUNDIR={run_dir}\n")
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