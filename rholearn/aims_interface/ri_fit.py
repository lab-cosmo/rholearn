"""
For generating RI coefficients to fit a real-space scalar field in FHI-aims, and
rebuilding real-space scalar fields from RI coefficients.
"""

import os
import shutil
from os.path import exists, join

import numpy as np
from chemfiles import Frame

from rholearn.aims_interface import hpc, io, orbitals, parser
from rholearn.options import get_options
from rholearn.rholearn import mask
from rholearn.utils import system
from rholearn.utils.io import pickle_dict, unpickle_dict


def run_ri_fit() -> None:
    """
    Runs a FHI-aims RI fitting calculation.
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options()

    # Get the frames and indices
    frames = system.read_frames_from_xyz(dft_options["XYZ"])
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(range(len(frames)))

    # Exclude some structures if specified
    if dft_options["IDX_EXCLUDE"] is not None:
        frame_idxs = [A for A in frame_idxs if A not in dft_options["IDX_EXCLUDE"]]

    frames = [frames[A] for A in frame_idxs]

    # Write submission script and run FHI-aims via sbatch array
    fname = f"run-aims-ri-{hpc.timestamp()}.sh"
    hpc.write_aims_sbatch_array(
        fname=fname,
        aims_command=dft_options["AIMS_COMMAND"],
        array_idxs=frame_idxs,
        run_dir=dft_options["RI_DIR"],
        dm_restart_dir=dft_options["SCF_DIR"],  # use DM restart files
        load_modules=hpc_options["LOAD_MODULES"],
        export_vars=hpc_options["EXPORT_VARIABLES"],
        slurm_params=hpc_options["SLURM_PARAMS"],
    )
    hpc.run_script(".", f"sbatch {fname}")

    return


def setup_ri_fit() -> None:
    """
    Runs the RI fitting set up as an sbatch array job for each frame.
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options()

    # Get the frames and indices
    frames = system.read_frames_from_xyz(dft_options["XYZ"])
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(range(len(frames)))

    # Exclude some structures if specified
    if dft_options["IDX_EXCLUDE"] is not None:
        frame_idxs = [A for A in frame_idxs if A not in dft_options["IDX_EXCLUDE"]]

    # Define the python command to run for the given frame
    python_command = (
        'python3 -c "from rholearn.aims_interface import ri_fit; '
        'ri_fit.setup_ri_fit_for_frame("${ARRAY_IDX}");'
        '"'
    )

    # Write submission script and run FHI-aims via sbatch array
    fname = f"set-up-ri-{hpc.timestamp()}.sh"
    hpc.write_python_sbatch_array(
        fname=fname,
        array_idxs=frame_idxs,
        run_dir=lambda _: ".",
        python_command=python_command,
        slurm_params=hpc_options["SLURM_PARAMS"],
    )
    hpc.run_script(".", f"sbatch {fname}")

    return


def setup_ri_fit_for_frame(frame_idx: int) -> None:
    """
    Prepares the input files for an RI fitting calculation for a single indexed frame.
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options()

    # Get the frames and indices
    frame = system.read_frames_from_xyz(dft_options["XYZ"])[frame_idx]

    # Retype masked atoms for the RI calculation
    if len(dft_options["MASK"]) != 0:
        frame = mask.retype_frame(frame, **dft_options["MASK"])

    # Make RI dir and copy settings file
    if not exists(dft_options["RI_DIR"](frame_idx)):
        os.makedirs(dft_options["RI_DIR"](frame_idx))
    shutil.copy("dft-options.yaml", dft_options["RI_DIR"](frame_idx))
    shutil.copy("hpc-options.yaml", dft_options["RI_DIR"](frame_idx))

    # Write the eigenstate occupations to file (if applicable)
    _write_eigenstate_occs_to_file(dft_options, hpc_options, frame_idx, frame)

    # Get control parameters
    control_params = io.get_control_parameters_for_frame(
        frame, dft_options["BASE_AIMS"], dft_options["RI"], dft_options["CUBE"]
    )

    # Write control.in and geometry.in to the RI dir
    io.write_geometry(frame, dft_options["RI_DIR"](frame_idx))
    io.write_control(
        frame,
        dft_options["RI_DIR"](frame_idx),
        parameters=control_params,
        species_defaults=dft_options["SPECIES_DEFAULTS"],
    )

    return


def process_ri_fit() -> None:
    """
    Processes the outputs of the RI fitting calculation(s)
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options()

    # Get the frames and indices
    frames = system.read_frames_from_xyz(dft_options["XYZ"])
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(range(len(frames)))

    # Exclude some structures if specified
    if dft_options["IDX_EXCLUDE"] is not None:
        frame_idxs = [A for A in frame_idxs if A not in dft_options["IDX_EXCLUDE"]]

    for A in frame_idxs:
        os.makedirs(dft_options["PROCESSED_DIR"](A), exist_ok=True)
        shutil.copy("dft-options.yaml", dft_options["PROCESSED_DIR"](A))
        shutil.copy("hpc-options.yaml", dft_options["PROCESSED_DIR"](A))

    python_command = (
        'python3 -c "from rholearn.aims_interface import ri_fit;'
        ' ri_fit.process_ri_fit_for_frame(frame_idx="${ARRAY_IDX}");"'
    )

    # Process the RI fit output for each frame
    fname = f"process-ri-{hpc.timestamp()}.sh"
    hpc.write_python_sbatch_array(
        fname=fname,
        array_idxs=frame_idxs,
        run_dir=dft_options["PROCESSED_DIR"],
        python_command=python_command,
        slurm_params=hpc_options["SLURM_PARAMS"],
    )
    hpc.run_script(".", "sbatch " + fname)

def process_ri_fit_for_frame(frame_idx: int) -> None:
    """
    Process the RI outputs. Function should be called from the FHI-aims output
    directory. 
    """

    dft_options, hpc_options = _get_options()
    parser.process_ri_outputs(
        aims_output_dir=dft_options['RI_DIR'](frame_idx),
        structure_idx=frame_idx,
        ovlp_cond_num=dft_options['PROCESS_RI']['overlap_cond_num'],
        cutoff_ovlp=dft_options['PROCESS_RI']['cutoff_overlap'],
        ovlp_sparsity_threshold=float(dft_options['PROCESS_RI']['sparsity_threshold']),
        save_dir=dft_options['PROCESSED_DIR'](frame_idx)
    )
    parser.process_df_error(
        aims_output_dir=dft_options['RI_DIR'](frame_idx),
        save_dir=dft_options['PROCESSED_DIR'](frame_idx),
        **dft_options["MASK"],
    )


def _write_eigenstate_occs_to_file(
    dft_options: dict, hpc_options: dict, A: int, frame: Frame
) -> None:
    """
    Identifies the field to be constructed and writes the eigenstate
    occupations to file using the appropriate settings.
    """
    if dft_options["FIELD_NAME"] == "edensity":  # no occupations required
        return
    
    # Calculate KSO occupations
    calc_info = unpickle_dict(join(dft_options["SCF_DIR"](A), "calc_info.pickle"))

    if dft_options["FIELD_NAME"] == "edensity_from_occs":

        eigenstate_occs = orbitals.get_eigenstate_occs_e_density(
            dft_options["SCF_DIR"](A)
        )

    elif dft_options["FIELD_NAME"] == "ildos":

        if dft_options.get("ILDOS") is None:
            raise ValueError(
                "Options field `ILDOS` must be provided in dft-options.yaml"
            )

        # Save ILDOS settings
        ildos_kwargs = {k: v for k, v in dft_options["ILDOS"].items()}
        ildos_kwargs["target_energy"] = calc_info[ildos_kwargs["target_energy"]]

        # Get KSO occupations
        eigenstate_occs = orbitals.get_eigenstate_occs_ildos(
            aims_output_dir=dft_options["SCF_DIR"](A),
            gaussian_width=ildos_kwargs["gaussian_width"],
            energy_window=ildos_kwargs["energy_window"],
            target_energy=ildos_kwargs["target_energy"],
            method=ildos_kwargs["method"],
        )

    elif dft_options["FIELD_NAME"] == "homo":
        eigenstate_occs = orbitals.get_eigenstate_occs_homo(dft_options["SCF_DIR"](A))

    elif dft_options["FIELD_NAME"] == "lumo":
        eigenstate_occs = orbitals.get_eigenstate_occs_lumo(dft_options["SCF_DIR"](A))

    else:
        raise ValueError(f"Unknown named field: {dft_options['FIELD_NAME']}")

    # Write to file
    np.savetxt(join(dft_options["RI_DIR"](A), "eigenstate_occs.in"), eigenstate_occs)


def _get_options() -> None:
    """
    Sets the settings globally. Ensures the defaults are set first and then
    overwritten with user settings.
    """
    dft_options = get_options("dft", "rholearn")
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
