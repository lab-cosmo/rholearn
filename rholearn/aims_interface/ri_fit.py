"""
For generating RI coefficients to fit a real-space scalar field in FHI-aims, and
rebuilding real-space scalar fields from RI coefficients.
"""

import os
import shutil
from functools import partial
from os.path import exists, join
import sys

import numpy as np
from chemfiles import Frame

from rholearn.aims_interface import hpc, io, orbitals, parser
from rholearn.options import get_options
from rholearn.rholearn import mask
from rholearn.utils import cube, system
from rholearn.utils.io import unpickle_dict


def run_ri_fit() -> None:
    """
    Runs a FHI-aims RI fitting calculation.
    """

    # Get the DFT and HPC options
    if len(sys.argv) == 1:
        run_id = None
    else:
        run_id = sys.argv[1]
    dft_options, hpc_options = _get_options(run_id)

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

    # Get the energy bins, if any
    if len(dft_options["ILDOS"]) == 0:
        energy_bins = [0]
    else:
        energy_bins = orbitals.get_energy_bins(
            dft_options["ILDOS"]["min_energy"],
            dft_options["ILDOS"]["max_energy"],
            dft_options["ILDOS"]["interval"],
        )
        energy_bins = list(range(len(energy_bins)))

    # Run RI calculation for each energy bin in the range
    for energy_bin in energy_bins:

        # Write submission script and run FHI-aims via sbatch array
        fname = f"run-aims-ri-{hpc.timestamp()}.sh"
        hpc.write_aims_sbatch_array(
            fname=fname,
            aims_command=dft_options["AIMS_COMMAND"],
            array_idxs=frame_idxs,
            run_dir=partial(dft_options["RI_DIR"], energy_bin=energy_bin),
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
    if len(sys.argv) == 1:
        run_id = None
    else:
        run_id = sys.argv[1]
    dft_options, hpc_options = _get_options(run_id)

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

    # Get the energy bins, if any
    if len(dft_options["ILDOS"]) == 0:
        energy_bins = [0]
    else:
        energy_bins = orbitals.get_energy_bins(
            dft_options["ILDOS"]["min_energy"],
            dft_options["ILDOS"]["max_energy"],
            dft_options["ILDOS"]["interval"],
        )
        energy_bins = list(range(len(energy_bins)))

    # Set up RI calculation for each energy bin in the range
    for energy_bin in energy_bins:

        # Define the python command to run for the given frame
        python_command = (
            'python3 -c "from rholearn.aims_interface import ri_fit; '
            'ri_fit.setup_ri_fit_for_frame_and_energy('
            '"${ARRAY_IDX}",'
            f'{energy_bin},'
        )
        if run_id is not None:
            python_command += f'\'{run_id}\''
        python_command += ');"'

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


def setup_ri_fit_for_frame_and_energy(
    frame_idx: int, energy_bin: int, run_id: str = None
) -> None:
    """
    Prepares the input files for an RI fitting calculation for a single indexed frame.
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options(run_id)

    # Get the frames and indices
    frame = system.read_frames_from_xyz(dft_options["XYZ"])[frame_idx]

    # Retype masked atoms for the RI calculation
    if len(dft_options["MASK"]) != 0:
        frame = mask.retype_frame(frame, **dft_options["MASK"])

    # Only write the overlap matrix for the first energy bin
    if energy_bin == 0:
        dft_options["RI"]["ri_ovlp_no_write"] = False
    else:
        dft_options["RI"]["ri_ovlp_no_write"] = True

    # Make RI dir and copy settings file
    if run_id is None:
        run_id = ""
    else:
        run_id = f"-{run_id}"
    if not exists(dft_options["RI_DIR"](frame_idx, energy_bin)):
        os.makedirs(dft_options["RI_DIR"](frame_idx, energy_bin))
    shutil.copy(f"dft-options{run_id}.yaml", dft_options["RI_DIR"](frame_idx, energy_bin))
    shutil.copy("hpc-options.yaml", dft_options["RI_DIR"](frame_idx, energy_bin))

    # Write the eigenstate occupations to file (if applicable)
    _write_eigenstate_occs_to_file(
        dft_options, hpc_options, frame_idx, frame, energy_bin
    )

    # Get control parameters
    control_params = io.get_control_parameters_for_frame(
        frame, dft_options["BASE_AIMS"], dft_options["RI"], dft_options["CUBE"]
    )

    # Write control.in and geometry.in to the RI dir
    io.write_geometry(frame, dft_options["RI_DIR"](frame_idx, energy_bin))
    io.write_control(
        frame,
        dft_options["RI_DIR"](frame_idx, energy_bin),
        parameters=control_params,
        species_defaults=dft_options["SPECIES_DEFAULTS"],
    )

    return


def process_ri_fit() -> None:
    """
    Processes the outputs of the RI fitting calculation(s)
    """

    # Get the DFT and HPC options
    if len(sys.argv) == 1:
        run_id = None
    else:
        run_id = sys.argv[1]
    dft_options, hpc_options = _get_options(run_id)

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

    # Process RI calculations for each energy bin in the range
    if run_id is None:
        run_id_ = ""
    else:
        run_id_ = f"-{run_id}"

    # Get the energy bins, if any
    if len(dft_options["ILDOS"]) == 0:
        energy_bins = [0]
    else:
        energy_bins = orbitals.get_energy_bins(
            dft_options["ILDOS"]["min_energy"],
            dft_options["ILDOS"]["max_energy"],
            dft_options["ILDOS"]["interval"],
        )
        energy_bins = list(range(len(energy_bins)))
    for energy_bin in energy_bins:
        for A in frame_idxs:
            os.makedirs(dft_options["PROCESSED_DIR"](A, energy_bin), exist_ok=True)
            shutil.copy(f"dft-options{run_id_}.yaml", dft_options["PROCESSED_DIR"](A, energy_bin))
            shutil.copy("hpc-options.yaml", dft_options["PROCESSED_DIR"](A, energy_bin))

        python_command = (
            'python3 -c "from rholearn.aims_interface import ri_fit;'
            ' ri_fit.process_ri_fit_for_frame_and_energy('
            '"${ARRAY_IDX}",'
            f'{energy_bin},'
        )
        if run_id is not None:
            python_command += f'\'{run_id}\''
        python_command += ');"'

        # Process the RI fit output for each frame
        fname = f"process-ri-{hpc.timestamp()}.sh"
        hpc.write_python_sbatch_array(
            fname=fname,
            array_idxs=frame_idxs,
            run_dir=lambda _: ".",
            python_command=python_command,
            slurm_params=hpc_options["SLURM_PARAMS"],
        )
        hpc.run_script(".", "sbatch " + fname)

    return


def process_ri_fit_for_frame_and_energy(
    frame_idx: int, energy_bin: int, run_id: str = None,
) -> None:
    """
    Process the RI outputs. Function should be called from the FHI-aims output
    directory.
    """

    # Get the DFT and HPC options
    dft_options, hpc_options = _get_options(run_id)

    # Process the RI outputs
    if dft_options["PROCESS_RI"].get("sparsity_threshold") is None:
        sparsity_threshold = None
    else:
        sparsity_threshold = float(dft_options["PROCESS_RI"].get("sparsity_threshold"))
    parser.process_ri_outputs(
        aims_output_dir=dft_options["RI_DIR"](frame_idx, energy_bin),
        structure_idx=frame_idx,
        energy_bin=energy_bin,
        ovlp_cond_num=dft_options["PROCESS_RI"]["overlap_cond_num"],
        cutoff_ovlp=dft_options["PROCESS_RI"]["cutoff_overlap"],
        ovlp_sparsity_threshold=sparsity_threshold,
        save_dir=dft_options["PROCESSED_DIR"](frame_idx, energy_bin),
    )
    parser.process_df_error(
        aims_output_dir=dft_options["RI_DIR"](frame_idx, energy_bin),
        save_dir=dft_options["PROCESSED_DIR"](frame_idx, energy_bin),
        **dft_options["MASK"],
    )

    # Plot STM images, if applicable
    if dft_options["STM"].get("mode") is not None:
        q_scf = cube.RhoCube(
            join(dft_options["RI_DIR"](frame_idx, energy_bin), "cube_001_total_density.cube")
        )
        q_ri = cube.RhoCube(
            join(dft_options["RI_DIR"](frame_idx, energy_bin), "cube_002_ri_density.cube")
        )

        # Plot the STM scatter
        if dft_options["STM"].get("mode") == "ccm":
            fig, ax = cube.plot_contour_ccm(
                cubes=[q_scf, q_ri],
                save_dir=dft_options["PROCESSED_DIR"](frame_idx, energy_bin),
                **dft_options["STM"]["options"],
            )

        else:
            assert dft_options["STM"].get("mode") == "chm"
            fig, ax = cube.plot_contour_chm(
                cubes=[q_scf, q_ri],
                save_dir=dft_options["PROCESSED_DIR"](frame_idx, energy_bin),
                **dft_options["STM"]["options"],
            )


def _write_eigenstate_occs_to_file(
    dft_options: dict, hpc_options: dict, A: int, frame: Frame, energy_bin: int,
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

        # Get the energy window for this bin
        bin_center = orbitals.get_energy_bins(
            dft_options["ILDOS"]["min_energy"],
            dft_options["ILDOS"]["max_energy"],
            dft_options["ILDOS"]["interval"],
        )[energy_bin]
        energy_window = [
            bin_center - 0.5 * dft_options["ILDOS"]["interval"], 
            bin_center + 0.5 * dft_options["ILDOS"]["interval"],
        ]

        # Save ILDOS settings
        ildos_kwargs = {k: v for k, v in dft_options["ILDOS"].items()}

        # Get the energy reference
        if ildos_kwargs["energy_reference"] == "Fermi":
            ildos_kwargs["target_energy"] = calc_info["fermi_eV"]
        elif ildos_kwargs["energy_reference"] == "Hartree":
            ildos_kwargs["target_energy"] = 0.0
        elif ildos_kwargs["energy_reference"] == "Custom":
            ildos_kwargs["target_energy"] = unpickle_dict(
                ildos_kwargs["read_energy_references_from"]
            )[A]
        else:
            raise ValueError(
                "ILDOS['energy_reference'] must be one of "
                "'Fermi', 'Hartree', or 'Custom'"
            )

        # Get KSO occupations
        eigenstate_occs = orbitals.get_eigenstate_occs_ildos(
            aims_output_dir=dft_options["SCF_DIR"](A),
            gaussian_width=ildos_kwargs["sigma"],
            energy_window=energy_window,
            target_energy=ildos_kwargs["target_energy"],
            method="gaussian_analytical",
        )

        # Scale them by the orbital occupation if specified
        if dft_options["ILDOS"].get("occupation") is not None:
            eigenstate_occs *= dft_options["ILDOS"]["occupation"]

    elif dft_options["FIELD_NAME"] == "homo":
        eigenstate_occs = orbitals.get_eigenstate_occs_homo(dft_options["SCF_DIR"](A))

    elif dft_options["FIELD_NAME"] == "lumo":
        eigenstate_occs = orbitals.get_eigenstate_occs_lumo(dft_options["SCF_DIR"](A))

    else:
        raise ValueError(f"Unknown named field: {dft_options['FIELD_NAME']}")

    # Write to file
    np.savetxt(join(dft_options["RI_DIR"](A, energy_bin), "eigenstate_occs.in"), eigenstate_occs)


def _get_options(run_id: str = None) -> None:
    """
    Sets the options globally. Ensures the defaults are set first and then
    overwritten with user options.
    """
    dft_options = get_options("dft", "rholearn", run_id)
    hpc_options = get_options("hpc")


    # Perform some checks
    if run_id is not None:
        assert run_id == dft_options["RUN_ID"], (
            "`run_id` passed via CLI not consistent with 'RUN_ID' in dft-options"
        )

    # Set some extra directories
    dft_options["SCF_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "raw", f"{frame_idx}"
    )
    dft_options["RI_DIR"] = lambda frame_idx, energy_bin: join(
        dft_options["DATA_DIR"], "raw", f"{frame_idx}", dft_options["RUN_ID"], f"e_{energy_bin}"
    )
    dft_options["PROCESSED_DIR"] = lambda frame_idx, energy_bin: join(
        dft_options["DATA_DIR"], "processed", f"{frame_idx}", dft_options["RUN_ID"], f"e_{energy_bin}"
    )

    return dft_options, hpc_options
