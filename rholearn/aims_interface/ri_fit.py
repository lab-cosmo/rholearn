"""
For generating RI coefficients to fit a real-space scalar field in FHI-aims, and
rebuilding real-space scalar fields from RI coefficients.
"""

import os
import shutil
from os.path import exists, join

import numpy as np
from chemfiles import Frame

from rholearn.aims_interface import hpc, io, orbitals
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
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = None
    frames = system.read_frames_from_xyz(dft_options["XYZ"], frame_idxs)
    if frame_idxs is None:
        frame_idxs = list(range(len(frames)))

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

    # Get the frame indices
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(range(len(system.read_frames_from_xyz(dft_options["XYZ"]))))

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
    if dft_options["MASK"] is not None:
        # Check that there are some active atoms in each frame
        assert (
            len(dft_options["MASK"]["get_active_coords"](frame.positions)) > 0
        ), "No active atoms found in frame. Check the MASK settings."
        frame = mask.retype_masked_atoms(
            frame,
            get_masked_coords=dft_options["MASK"]["get_masked_coords"],
        )

    # Make RI dir and copy settings file
    if not exists(dft_options["RI_DIR"](frame_idx)):
        os.makedirs(dft_options["RI_DIR"](frame_idx))
    shutil.copy("dft-options.yaml", dft_options["RI_DIR"](frame_idx))
    shutil.copy("hpc-options.yaml", dft_options["RI_DIR"](frame_idx))

    # Write the KS-orbital weights to file (if applicable)
    _write_kso_weights_to_file(dft_options, hpc_options, frame_idx, frame)

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


def process_ri_fit(get_ovlp_cond_num: bool = False) -> None:
    """
    Processes the outputs of the RI fitting calculation(s)
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options()

    # Get the frame indices
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(range(len(system.read_frames_from_xyz(dft_options["XYZ"]))))

    for A in frame_idxs:
        shutil.copy("dft-options.yaml", dft_options["RI_DIR"](A))
        shutil.copy("hpc-options.yaml", dft_options["RI_DIR"](A))

    python_command = (
        'python3 -c "from rholearn.aims_interface import parser; '
        "from rholearn.aims_interface import ri_fit; "
        "dft_options, hpc_options = ri_fit._get_options(); "
        "parser.process_ri_outputs("
        "aims_output_dir='.',"
        ' structure_idx="${ARRAY_IDX}",'
        f" ovlp_cond_num={'True' if get_ovlp_cond_num else 'False'},"
        f" save_dir='{dft_options['PROCESSED_DIR']('${ARRAY_IDX}')}',"
        ");"
        "parser.process_df_error("
        f"aims_output_dir='{dft_options['RI_DIR']('${ARRAY_IDX}')}',"
        f" save_dir='{dft_options['PROCESSED_DIR']('${ARRAY_IDX}')}',"
        ')"'
    )

    # Process the RI fit output for each frame
    fname = f"process-ri-{hpc.timestamp()}.sh"
    hpc.write_python_sbatch_array(
        fname=fname,
        array_idxs=frame_idxs,
        run_dir=dft_options["RI_DIR"],
        python_command=python_command,
        slurm_params=hpc_options["SLURM_PARAMS"],
    )
    hpc.run_script(".", "sbatch " + fname)


def _write_kso_weights_to_file(
    dft_options: dict, hpc_options: dict, A: int, frame: Frame
) -> None:
    """
    Identifies the field to be constructed and writes the KS-orbital
    weights to file using the appropriate settings.
    """
    if dft_options["FIELD_NAME"] == "edensity":  # no weights required
        # assert dft_options["RI"].get("ri_fit_total_density") is not None, (
        #     "FHI-aims tag `ri_fit_total_density` must be set to `true`"
        #     f" for fitting to the `{dft_options["FIELD_NAME"]}` field."
        # )
        # assert dft_options["RI"].get("ri_fit_field_from_kso_weights") is None, (
        #     "FHI-aims tag `ri_fit_field_from_kso_weights` must not be specified"
        #     f" if fitting to the `{dft_options["FIELD_NAME"]}` field."
        # )
        pass
    else:  # calculate KSO weights

        assert dft_options["RI"].get("ri_fit_total_density") is None, (
            "FHI-aims tag `ri_fit_total_density` must not be specified"
            f" if fitting to the `{dft_options['FIELD_NAME']}` field."
        )
        assert dft_options["RI"].get("ri_fit_field_from_kso_weights") is not None, (
            "FHI-aims tag `ri_fit_field_from_kso_weights` must be set to `true`"
            f" if fitting to the `{dft_options['FIELD_NAME']}` field."
        )

        # Get SCF calculation info and path to KS-orbital info
        calc_info = unpickle_dict(join(dft_options["SCF_DIR"](A), "calc_info.pickle"))

        if dft_options["FIELD_NAME"] == "edensity_from_weights":

            kso_weights = orbitals.get_kso_weight_vector_e_density(
                dft_options["SCF_DIR"](A)
            )

        elif dft_options["FIELD_NAME"] == "ildos":

            # Save LDOS settings
            ldos_kwargs = {k: v for k, v in dft_options["LDOS"].items()}
            ldos_kwargs["target_energy"] = calc_info[ldos_kwargs["target_energy"]]
            pickle_dict(join(dft_options["RI_DIR"](A), "ldos_settings"), ldos_kwargs)

            # Get KSO weights
            kso_weights = orbitals.get_kso_weight_vector_ildos(
                aims_output_dir=dft_options["SCF_DIR"](A), **ldos_kwargs
            )

        elif dft_options["FIELD_NAME"] == "homo":
            kso_weights = orbitals.get_kso_weight_vector_homo(dft_options["SCF_DIR"](A))

        elif dft_options["FIELD_NAME"] == "lumo":
            kso_weights = orbitals.get_kso_weight_vector_lumo(dft_options["SCF_DIR"](A))

        else:
            raise ValueError(f"Unknown named field: {dft_options['FIELD_NAME']}")

        # Write to file
        np.savetxt(join(dft_options["RI_DIR"](A), "ks_orbital_weights.in"), kso_weights)


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
