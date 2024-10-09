"""
For converging SCF calculations for a given system in FHI-aims.
"""

import os
import shutil
from os.path import exists, join

from rholearn.aims_interface import hpc, io
from rholearn.options import get_options
from rholearn.utils import system


def run_scf() -> None:
    """
    Runs a FHI-aims SCF calculation according to the settings in `dft_settings`.
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options()

    # Get the frames and indices
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = None
    frames = system.read_frames_from_xyz(dft_options["XYZ"], frame_idxs)
    if frame_idxs is None:
        frame_idxs = list(range(len(frames)))

    # Add virtual nodes if applicable
    if dft_options["VIRTUAL_NODES"] is True:
        frames = system.add_virtual_nodes_in_bonds(frames)

    # Build the calculation settings for each frame in turn
    for A, frame in enumerate(frames):

        # Make RI dir and copy settings file
        if not exists(dft_options["SCF_DIR"](A)):
            os.makedirs(dft_options["SCF_DIR"](A))
        shutil.copy("dft-options.yaml", dft_options["SCF_DIR"](A))
        shutil.copy("hpc-options.yaml", dft_options["SCF_DIR"](A))

        # Get control parameters
        control_params = io.get_control_parameters_for_frame(
            frame, dft_options["BASE_AIMS"], dft_options["SCF"], dft_options["CUBE"]
        )

        # Write input files
        io.write_geometry(frame, dft_options["SCF_DIR"](A))
        io.write_control(
            frame,
            dft_options["SCF_DIR"](A),
            parameters=control_params,
            species_defaults=dft_options["SPECIES_DEFAULTS"],
        )

    # Write submission script and run FHI-aims via sbatch array
    fname = f"run-aims-scf-{hpc.timestamp()}.sh"
    hpc.write_aims_sbatch_array(
        fname=fname,
        aims_command=dft_options["AIMS_COMMAND"],
        array_idxs=frame_idxs,
        run_dir=dft_options["SCF_DIR"],  # callable to each structure dir
        dm_restart_dir=None,
        load_modules=hpc_options["LOAD_MODULES"],
        export_vars=hpc_options["EXPORT_VARIABLES"],
        slurm_params=hpc_options["SLURM_PARAMS"],
    )
    hpc.run_script(".", f"sbatch {fname}")

    return


def process_scf() -> None:
    """
    Parses aims.out files from SCF runs and saves them to the SCF directory.
    """

    # Set the DFT settings globally
    dft_options, hpc_options = _get_options()

    # Get the frame indices
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(
            range(len(system.read_frames_from_xyz(dft_options["XYZ"], frame_idxs)))
        )

    python_command = (
        "python3 -c"
        "'from rholearn.aims_interface import parser;"
        " from rholearn.utils.io import pickle_dict;"
        ' calc_info = parser.parse_aims_out(aims_output_dir=".");'
        ' pickle_dict("calc_info.pickle", calc_info);'
        "'"
    )

    # Process the RI fit output for each frame
    fname = f"run-process-scf-{hpc.timestamp()}.sh"
    hpc.write_python_sbatch_array(
        fname=fname,
        array_idxs=frame_idxs,
        run_dir=dft_options["SCF_DIR"],
        python_command=python_command,
        slurm_params=hpc_options["SLURM_PARAMS"],
    )
    hpc.run_script(".", "sbatch " + fname)


def _get_options() -> None:
    """
    Gets the DFT and HPC options. Ensures the defaults are set first and then
    overwritten with user settings.
    """
    dft_options = get_options("dft")
    hpc_options = get_options("hpc")

    # Set some extra directories
    dft_options["SCF_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "raw", f"{frame_idx}"
    )
    dft_options["RI_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "raw", f"{frame_idx}", dft_options["RI_FIT_ID"]
    )
    dft_options["PROCESSED_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "processed", f"{frame_idx}", dft_options["RI_FIT_ID"]
    )

    return dft_options, hpc_options
