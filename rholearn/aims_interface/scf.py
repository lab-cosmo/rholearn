"""
For converging SCF calculations for a given system in FHI-aims.
"""

import os
import shutil
from os.path import exists, join

from rholearn.aims_interface import hpc, io, parser
from rholearn.options import get_options
from rholearn.utils import system


def rholearn_run_scf() -> None:
    """Runs a FHI-aims SCF calculation to generate data for rholearn"""
    _run_scf("rholearn")


def doslearn_run_scf() -> None:
    """Runs a FHI-aims SCF calculation to generate data for doslearn"""
    _run_scf("doslearn")


def rholearn_process_scf() -> None:
    """Processes FHI-aims SCF outputs for rholearn"""
    _process_scf("rholearn")


def doslearn_process_scf() -> None:
    """Processes FHI-aims SCF outputs for doslearn"""
    _process_scf("doslearn")


def _run_scf(model: str) -> None:
    """
    Runs a FHI-aims SCF calculation.

    ``model`` is either "rholearn" or "doslearn".
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options(model)

    # Get the frames and indices
    frames = system.read_frames_from_xyz(dft_options["XYZ"])
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(range(len(frames)))

    # Exclude some structures if specified
    if dft_options["IDX_EXCLUDE"] is not None:
        frame_idxs = [A for A in frame_idxs if A not in dft_options["IDX_EXCLUDE"]]

    assert len(frame_idxs) > 0, "No frames in the selection."

    frames = [frames[A] for A in frame_idxs]

    # Build the calculation settings for each frame in turn
    for A, frame in zip(frame_idxs, frames):

        # Make RI dir and copy settings file
        if not exists(dft_options["SCF_DIR"](A)):
            os.makedirs(dft_options["SCF_DIR"](A))
        shutil.copy("dft-options.yaml", dft_options["SCF_DIR"](A))
        shutil.copy("hpc-options.yaml", dft_options["SCF_DIR"](A))

        # Get control parameters
        control_params = io.get_control_parameters_for_frame(
            frame, dft_options["BASE_AIMS"], dft_options["SCF"], dft_options.get("CUBE")
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


def _process_scf(model: str) -> None:
    """
    Parses aims.out files from SCF runs and saves them to the SCF directory.

    ``model`` is either "rholearn" or "doslearn".
    """

    # Set the DFT settings globally
    dft_options, hpc_options = _get_options(model)

    # Get the frames and indices
    frames = system.read_frames_from_xyz(dft_options["XYZ"])
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(range(len(frames)))

    # Exclude some structures if specified
    if dft_options["IDX_EXCLUDE"] is not None:
        frame_idxs = [A for A in frame_idxs if A not in dft_options["IDX_EXCLUDE"]]

    assert len(frame_idxs) > 0, "No frames in the selection."

    python_command = (
        "from rholearn.aims_interface import parser;"
        " from rholearn.utils.io import pickle_dict;"
        " calc_info = parser.parse_aims_out(aims_output_dir='.');"
        " pickle_dict('calc_info.pickle', calc_info);"
    )
    python_command = 'python3 -c "' + python_command + '"'

    # Process the SCF outputs for each frame
    fname = f"run-process-scf-{hpc.timestamp()}.sh"
    hpc.write_python_sbatch_array(
        fname=fname,
        array_idxs=frame_idxs,
        run_dir=dft_options["SCF_DIR"],
        python_command=python_command,
        slurm_params=hpc_options["SLURM_PARAMS"],
    )
    hpc.run_script(".", "sbatch " + fname)

    if dft_options.get("DOS_SPLINES") is not None:

        for A in frame_idxs:
            os.makedirs(dft_options["PROCESSED_DIR"](A), exist_ok=True)
            shutil.copy("dft-options.yaml", dft_options["PROCESSED_DIR"](A))
            shutil.copy("hpc-options.yaml", dft_options["PROCESSED_DIR"](A))

        python_command = (
            'python3 -c "from rholearn.aims_interface import scf;'
            ' scf._spline_eigenvalues_for_frame(frame_idx="${ARRAY_IDX}");"'
        )
        # Process the RI fit output for each frame
        fname = f"process-scf-doslearn-{hpc.timestamp()}.sh"
        hpc.write_python_sbatch_array(
            fname=fname,
            array_idxs=frame_idxs,
            run_dir=dft_options["PROCESSED_DIR"],
            python_command=python_command,
            slurm_params=hpc_options["SLURM_PARAMS"],
        )
        hpc.run_script(".", "sbatch " + fname)


def _spline_eigenvalues_for_frame(frame_idx: int) -> None:
    """
    Parses the files "Final_KS_eigenvalues.dat" and splines them. Saves the resulting
    TensorMap objects to the processed data directory.
    """

    # Set the DFT settings globally
    dft_options, _ = _get_options("doslearn")

    # Spline eigenenergies
    parser.spline_eigenenergies(
        aims_output_dir=dft_options["SCF_DIR"](frame_idx),
        frame_idx=frame_idx,
        sigma=dft_options["DOS_SPLINES"]["sigma"],
        min_energy=dft_options["DOS_SPLINES"]["min_energy"],
        max_energy=dft_options["DOS_SPLINES"]["max_energy"],
        interval=dft_options["DOS_SPLINES"]["interval"],
        save_dir=dft_options["PROCESSED_DIR"](frame_idx),
    )


def _get_options(model: str) -> None:
    """
    Gets the DFT and HPC options. Ensures the defaults are set first and then
    overwritten with user settings.
    """
    dft_options = get_options("dft", model)
    hpc_options = get_options("hpc")

    # Set some extra directories
    dft_options["SCF_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "raw", f"{frame_idx}"
    )
    dft_options["RI_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "raw", f"{frame_idx}", dft_options["RUN_ID"]
    )
    dft_options["PROCESSED_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "processed", f"{frame_idx}", dft_options["RUN_ID"]
    )

    return dft_options, hpc_options
