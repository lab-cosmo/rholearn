"""
For generating RI coefficients to fit a real-space scalar field in FHI-aims, and
rebuilding real-space scalar fields from RI coefficients.
"""

import os
from os.path import exists, join
import shutil
from typing import List

from chemfiles import Frame
import numpy as np

from rholearn import mask
from rholearn.aims_interface import io, hpc, orbitals
from rholearn.settings.defaults import dft_defaults
from rholearn.utils.io import pickle_dict, unpickle_dict


def run_ri_fit(dft_settings: dict, hpc_settings: dict) -> None:
    """
    Runs a FHI-aims RI fit calculation according to the settings in `dft_settings`.
    """

    # Set the DFT settings globally
    _set_settings_globally(dft_settings, hpc_settings)

    # Write submission script and run FHI-aims via sbatch array
    fname = "run-aims-ri.sh"
    hpc.write_aims_sbatch_array(
        fname=fname,
        aims_command=AIMS_COMMAND,
        array_idxs=FRAME_IDXS,
        run_dir=RI_DIR,
        dm_restart_dir=SCF_DIR,  # use DM restart files
        load_modules=HPC["load_modules"],
        export_vars=HPC["export_vars"],
        slurm_params=SLURM_PARAMS,
    )
    hpc.run_script(".", f"sbatch {fname}")

    return


def set_up_ri_fit_sbatch(dft_settings: dict, hpc_settings: dict) -> None:
    """
    Runs the RI fitting set up as an sbatch array job for each frame.
    """

    # Set the DFT settings globally
    _set_settings_globally(dft_settings, hpc_settings)

    # Define the python command to run for the given frame
    python_command = (
        'python3 -c "from rholearn.aims_interface import ri_fit; '
        "from rholearn.utils import system;"
        "from dft_settings import DFT_SETTINGS; "
        "from hpc_settings import HPC_SETTINGS; "
        "all_frames = system.read_frames_from_xyz(DFT_SETTINGS['XYZ']);"
        'ri_fit.set_up_ri_fit(DFT_SETTINGS, HPC_SETTINGS, "${ARRAY_IDX}", all_frames["${ARRAY_IDX}"]);'
        '"'
    )

    # Write submission script and run FHI-aims via sbatch array
    fname = "set-up-ri.sh"
    hpc.write_python_sbatch_array(
        fname=fname,
        array_idxs=FRAME_IDXS,
        run_dir=lambda _: ".",
        python_command=python_command,
        slurm_params=SLURM_PARAMS,
    )
    hpc.run_script(".", f"sbatch {fname}")

    return


def set_up_ri_fit(dft_settings: dict, hpc_settings: dict, frame_idx: int, frame: Frame) -> None:
    """
    Prepares the input files for an RI fit calculation for a single frame.
    """
    
    # Set the DFT settings globally
    _set_settings_globally(dft_settings, hpc_settings)

    # Retype masked atoms for the RI calculation
    if MASK is not None:
        # Check that there are some active atoms in each frame
        assert (
            len(MASK["get_active_coords"](frame.positions)) > 0
        ), "No active atoms found in frame. Check the MASK settings."
        frame = mask.retype_masked_atoms(
            frame,
            get_masked_coords=MASK["get_masked_coords"],
        )

    # Make RI dir and copy settings file
    if not exists(RI_DIR(frame_idx)):
        os.makedirs(RI_DIR(frame_idx))
    shutil.copy("dft_settings.py", RI_DIR(frame_idx))

    # Write the KS-orbital weights to file (if applicable)
    _write_kso_weights_to_file(dft_settings, hpc_settings, frame_idx, frame)

    # Get control parameters
    control_params = io.get_control_parameters_for_frame(frame, BASE_AIMS, RI, CUBE)

    # Write control.in and geometry.in to the RI dir
    io.write_geometry(frame, RI_DIR(frame_idx))
    io.write_control(
        frame,
        RI_DIR(frame_idx),
        parameters=control_params,
        species_defaults=SPECIES_DEFAULTS,
    )

    return


def process_ri_fit(dft_settings: dict, hpc_settings: dict) -> None:
    """
    Processes the outputs of the RI fit calculations according to the settings in
    """
    
    # Set the DFT settings globally
    _set_settings_globally(dft_settings, hpc_settings)

    python_command = (
        'python3 -c "from rholearn.aims_interface import parser; '
        "parser.process_ri_outputs("
        "aims_output_dir='.',"
        ' structure_idx="${ARRAY_IDX}",'
        ' ovlp_cond_num=False,'
        f" save_dir='{PROCESSED_DIR('${ARRAY_IDX}')}',"
        ");"
        "parser.process_df_error("
        f"aims_output_dir='{RI_DIR('${ARRAY_IDX}')}',"
        f" save_dir='{PROCESSED_DIR('${ARRAY_IDX}')}',"
        ')"'
    )

    # Process the RI fit output for each frame
    fname = "process-ri.sh"
    hpc.write_python_sbatch_array(
        fname=fname,
        array_idxs=FRAME_IDXS,
        run_dir=RI_DIR,
        python_command=python_command,
        slurm_params=SLURM_PARAMS,
    )
    hpc.run_script(".", "sbatch " + fname)


def _write_kso_weights_to_file(dft_settings: dict, hpc_settings: dict, A: int, frame: Frame) -> None:
    """
    Identifies the field to be constructed and writes the KS-orbital
    weights to file using the appropriate settings.
    """
    # Set the DFT settings globally
    _set_settings_globally(dft_settings, hpc_settings)

    if FIELD_NAME == "edensity":  # no weights required
        # assert RI.get("ri_fit_total_density") is not None, (
        #     "FHI-aims tag `ri_fit_total_density` must be set to `true`"
        #     f" for fitting to the `{FIELD_NAME}` field."
        # )
        # assert RI.get("ri_fit_field_from_kso_weights") is None, (
        #     "FHI-aims tag `ri_fit_field_from_kso_weights` must not be specified"
        #     f" if fitting to the `{FIELD_NAME}` field."
        # )
        pass
    else:  # calculate KSO weights

        assert RI.get("ri_fit_total_density") is None, (
            "FHI-aims tag `ri_fit_total_density` must not be specified"
            f" if fitting to the `{FIELD_NAME}` field."
        )
        assert RI.get("ri_fit_field_from_kso_weights") is not None, (
            "FHI-aims tag `ri_fit_field_from_kso_weights` must be set to `true`"
            f" if fitting to the `{FIELD_NAME}` field."
        )

        # Get SCF calculation info and path to KS-orbital info
        calc_info = unpickle_dict(join(SCF_DIR(A), "calc_info.pickle"))

        if FIELD_NAME == "edensity_from_weights":

            kso_weights = orbitals.get_kso_weight_vector_e_density(SCF_DIR(A))

        elif FIELD_NAME == "ildos":

            # Save LDOS settings
            ldos_kwargs = {k: v for k, v in LDOS.items()}
            ldos_kwargs["target_energy"] = calc_info[ldos_kwargs["target_energy"]]
            pickle_dict(join(RI_DIR(A), "ldos_settings"), ldos_kwargs)

            # Get KSO weights
            kso_weights = orbitals.get_kso_weight_vector_ildos(
                aims_output_dir=SCF_DIR(A), **ldos_kwargs
            )

        elif FIELD_NAME == "homo":
            kso_weights = get_kso_weight_vector_homo(SCF_DIR(A))

        elif FIELD_NAME == "lumo":
            kso_weights = get_kso_weight_vector_lumo(SCF_DIR(A))

        else:
            raise ValueError(f"Unknown named field: {FIELD_NAME}")

        # Write to file
        np.savetxt(join(RI_DIR(A), "ks_orbital_weights.in"), kso_weights)


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
