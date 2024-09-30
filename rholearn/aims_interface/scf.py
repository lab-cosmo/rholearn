"""
For converging SCF calculations for a given system in FHI-aims.
"""

import shutil
import os
from os.path import exists, join

from rholearn.aims_interface import io, hpc
from rholearn.utils import system
from rholearn.settings.defaults import dft_defaults


def run_scf(dft_settings: dict, hpc_settings: dict) -> None:
    """
    Runs a FHI-aims SCF calculation according to the settings in `dft_settings`.
    """

    # Set the DFT settings globally
    _set_settings_globally(dft_settings, hpc_settings)

    # Get the frames
    all_frames = system.read_frames_from_xyz(XYZ)
    frames = [all_frames[A] for A in FRAME_IDXS]

    # Build the calculation settings for each frame in turn
    for A, frame in zip(FRAME_IDXS, frames):

        # Make RI dir and copy settings file
        if not exists(SCF_DIR(A)):
            os.makedirs(SCF_DIR(A))
        shutil.copy("dft_settings.py", SCF_DIR(A))

        # Get control parameters
        control_params = io.get_control_parameters_for_frame(
            frame, BASE_AIMS, SCF, CUBE
        )

        # Write input files
        io.write_geometry(frame, SCF_DIR(A))
        io.write_control(
            frame,
            SCF_DIR(A),
            parameters=control_params,
            species_defaults=SPECIES_DEFAULTS,
        )

    # Write submission script and run FHI-aims via sbatch array
    fname = "run-aims-scf.sh"
    hpc.write_aims_sbatch_array(
        fname=fname,
        aims_command=AIMS_COMMAND,
        array_idxs=FRAME_IDXS,
        run_dir=SCF_DIR,  # callable to each structure dir
        dm_restart_dir=None,
        load_modules=HPC["load_modules"],
        export_vars=HPC["export_vars"],
        slurm_params=SLURM_PARAMS,
    )
    hpc.run_script(".", f"sbatch {fname}")

    return


def process_scf(dft_settings: dict, hpc_settings: dict) -> None:
    """
    Parses aims.out files from SCF runs and saves them to the SCF directory.
    """
    
    # Set the DFT settings globally
    _set_settings_globally(dft_settings, hpc_settings)

    python_command = (
        "python3 -c"
        "'from rholearn.aims_interface import parser;"
        " from rholearn.utils.io import pickle_dict;"
        " calc_info = parser.parse_aims_out(aims_output_dir=\".\");"
        " pickle_dict(\"calc_info.pickle\", calc_info);"
        "'"
    )

    # Process the RI fit output for each frame
    fname = "run-process-scf.sh"
    hpc.write_python_sbatch_array(
        fname=fname,
        array_idxs=FRAME_IDXS,
        run_dir=SCF_DIR,
        python_command=python_command,
        slurm_params=SLURM_PARAMS,
    )
    hpc.run_script(".", "sbatch " + fname)



def _set_settings_globally(dft_settings: dict, hpc_settings: dict) -> None:
    """
    Sets the settings globally. Ensures the defaults are set first and then
    overwritten with user settings.
    """
    # Update DFT and ML defaults with user settings
    dft_settings_ = dft_defaults.DFT_DEFAULTS
    dft_settings_.update(dft_settings)

    # Set them globally
    for settings_dict in [dft_settings_, hpc_settings]:
        for key, value in settings_dict.items():
            globals()[key] = value


    # Set some directories
    globals()["SCF_DIR"] = lambda frame_idx: join(DATA_DIR, "raw", f"{frame_idx}")
    globals()["RI_DIR"] = lambda frame_idx: join(DATA_DIR, "raw", f"{frame_idx}", RI_FIT_ID)
    globals()["PROCESSED_DIR"] = lambda frame_idx: join(DATA_DIR, "processed", f"{frame_idx}", RI_FIT_ID)
