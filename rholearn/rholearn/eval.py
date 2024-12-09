import os
import time
from os.path import exists, join
from typing import List

import matplotlib.pyplot
import numpy as np
import torch

from rholearn.aims_interface import fields, ri_rebuild
from rholearn.options import get_options
from rholearn.rholearn import mask, train_utils
from rholearn.utils import convert, cube, io, system


def eval():
    """
    Runs model evaluation in the following steps:
        1. Load model
        2. Perform inference
        3. Rebuild fields by calling FHI-aims
        4. Evaluate MAE against reference fields
        5. Generates STM images by parsing FHI-aims cube output files

    Steps 2 and 3 only occur if rebuild output files don't already exist. Steps 4 and 5
    only occur if the respective `EVAL` options are specified in ml-options.yaml.
    """
    t0_eval = time.time()
    dft_options, hpc_options, ml_options = _get_options()

    # ===== Setup =====
    os.makedirs(join(ml_options["ML_DIR"], "outputs"), exist_ok=True)
    log_path = join(ml_options["ML_DIR"], "outputs/eval.log")
    io.log(log_path, "===== BEGIN =====")
    io.log(log_path, f"Working directory: {ml_options['ML_DIR']}")

    # Random seed and dtype
    torch.manual_seed(ml_options["SEED"])
    torch.set_default_dtype(getattr(torch, ml_options["TRAIN"]["dtype"]))

    # Get a callable to the evaluation subdirectory, parametrized by frame idx
    rebuild_dir = lambda frame_idx: join(  # noqa: E731
        ml_options["EVAL_DIR"](ml_options["EVAL"]["eval_epoch"]), f"{frame_idx}"
    )

    # Load all the frames
    io.log(log_path, "Loading frames")
    all_frames = system.read_frames_from_xyz(dft_options["XYZ"])

    # Get frame indices we have data for (or a subset if specified)
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = list(range(len(all_frames)))

    # Exclude some structures if specified
    if dft_options["IDX_EXCLUDE"] is not None:
        frame_idxs = [A for A in frame_idxs if A not in dft_options["IDX_EXCLUDE"]]

    _check_input_settings(dft_options, ml_options, frame_idxs)

    # ===== Get the test data =====

    io.log(log_path, "Get the test IDs and frames")
    _, _, test_id = train_utils.crossval_idx_split(  # cross-validation split of idxs
        frame_idxs=frame_idxs,
        n_train=ml_options["N_TRAIN"],
        n_val=ml_options["N_VAL"],
        n_test=ml_options["N_TEST"],
        seed=ml_options["SEED"],
    )

    # Now take a subset of the test IDs, if applicable
    if ml_options["EVAL"]["subset_size"] is not None:
        test_id = test_id[: ml_options["EVAL"]["subset_size"]]

    io.log(log_path, f"    Test system ID: {test_id}")
    test_frames = [all_frames[A] for A in test_id]

    # ===== Load model and perform inference =====

    io.log(
        log_path,
        f"Load model from checkpoint at epoch {ml_options['EVAL']['eval_epoch']}",
    )
    model = torch.load(
        join(ml_options["CHKPT_DIR"](ml_options["EVAL"]["eval_epoch"]), "model.pt")
    )

    # Check that evaluation hasn't already happened
    all_aims_outs = [join(rebuild_dir(A), "aims.out") for A in test_id]
    if all([exists(aims_out) for aims_out in all_aims_outs]):
        msg = (
            "Rebuild calculations have already been run for this model."
            " Re-running evaluation on pre-existing FHI-aims rebuild files."
            " To re-run the rebuild, remove the existing output files."
        )
        io.log(log_path, msg)

    else:

        # Perform inference
        io.log(log_path, "Perform inference")
        t0_infer = time.time()
        test_preds_mts = model.predict(frames=test_frames, frame_idxs=test_id)
        dt_infer = time.time() - t0_infer
        io.log(
            log_path,
            train_utils.report_dt(dt_infer, "Model inference (on basis) complete"),
        )
        io.log(
            log_path, train_utils.report_dt(dt_infer / len(test_id), "    or per frame")
        )

        # ===== Rebuild fields =====
        io.log(log_path, "Rebuild real-space field(s) in FHI-aims")

        # Convert predicted coefficients to flat numpy arrays
        if model._descriptor_calculator._masked_system_type is not None:
            test_frames = mask.retype_frame(
                test_frames,
                model._descriptor_calculator._masked_system_type,
                **model._descriptor_calculator._mask_kwargs,
            )
        test_preds_numpy = [
            convert.coeff_vector_blocks_to_flat(
                frame, coeff_vector, basis_set=model._target_basis
            )
            for frame, coeff_vector in zip(test_frames, test_preds_mts)
        ]

        # Run rebuild routine
        t0_build = time.time()
        ri_rebuild.rebuild_field(
            frame_idxs=test_id,
            frames=test_frames,
            coefficients=test_preds_numpy,
            rebuild_dir=rebuild_dir,
            aims_command=dft_options["AIMS_COMMAND"],
            base_settings=dft_options["BASE_AIMS"],
            rebuild_settings=dft_options["REBUILD"],
            cube_settings=dft_options["CUBE"],
            species_defaults=dft_options["SPECIES_DEFAULTS"],
            slurm_params=hpc_options["SLURM_PARAMS"],
            load_modules=hpc_options["LOAD_MODULES"],
            export_vars=hpc_options["EXPORT_VARIABLES"],
        )

        # Wait for FHI-aims rebuild(s) to finish
        io.log(log_path, "Waiting for FHI-aims rebuild calculation(s) to finish...")
        while len(all_aims_outs) > 0:
            for aims_out in all_aims_outs:
                if exists(aims_out):
                    with open(aims_out, "r") as f:
                        # Basic check to see if AIMS calc has finished
                        if "Leaving FHI-aims." in f.read():
                            all_aims_outs.remove(aims_out)

        dt_build = time.time() - t0_build
        io.log(
            log_path,
            train_utils.report_dt(
                dt_build, "Build predicted field(s) in FHI-aims complete"
            ),
        )

    # ===== Evaluate NMAE =====
    if len(ml_options["EVAL"]["target_type"]) > 0:
        if isinstance(ml_options["EVAL"]["target_type"], str):
            target_types = [ml_options["EVAL"]["target_type"]]
        else:
            target_types = ml_options["EVAL"]["target_type"]

        io.log(
            log_path,
            f"Evaluate MAE versus reference field type(s): {target_types}",
        )
        if model._descriptor_calculator._masked_system_type is not None:
            io.log(log_path, "Evaluting errors on active region coordinates only")

        nmaes = {t: [] for t in target_types}
        for A in test_id:
            # Load grid and predicted field
            grid = np.load(join(dft_options["RI_DIR"](A), "partition_tab.npy"))
            rho_ml = np.loadtxt(join(rebuild_dir(A), "rho_rebuilt_ri.out"))

            # Check grid in ML dir is consistent
            _tmp_grid = np.loadtxt(join(rebuild_dir(A), "partition_tab.out"))
            assert np.all(grid == _tmp_grid)
            assert np.all(grid[:, :3] == rho_ml[:, :3])
            rho_ml = rho_ml[:, 3]

            # Build a coordinate mask, if applicable
            if model._descriptor_calculator._masked_system_type is not None:
                grid_mask = mask.get_point_indices_by_region(
                    points=grid[:, :3],
                    masked_system_type=(
                        model._descriptor_calculator._masked_system_type
                    ),
                    region="active",
                    **model._descriptor_calculator._mask_kwargs,
                )
                rho_ml = rho_ml[grid_mask]
                grid = grid[grid_mask]

            # Load each reference field - either SCF or RI
            for target_type in target_types:
                if target_type == "scf":
                    rho_ref = np.load(join(dft_options["RI_DIR"](A), "rho_scf.npy"))
                else:
                    assert target_type == "ri"
                    rho_ref = np.load(
                        join(dft_options["RI_DIR"](A), "rho_rebuilt_ri.npy")
                    )

                # Build a coordinate mask, if applicable
                if model._descriptor_calculator._masked_system_type is not None:
                    rho_ref = rho_ref[grid_mask]

                # Get the MAE and normalization
                abs_error, norm = fields.field_absolute_error(
                    input=rho_ml,
                    target=rho_ref,
                    grid=grid,
                )
                nmae = 100 * abs_error / norm
                nmaes[target_type].append(nmae)

                # Also compute the squared error
                squared_error, norm = fields.field_squared_error(
                    input=rho_ml,
                    target=rho_ref,
                    grid=grid,
                )

                # Log and save the results
                io.log(
                    log_path,
                    f"system {A} target {target_type} abs_error {abs_error:.5f}"
                    f" norm {norm:.5f} nmae {nmae:.5f}"
                    f" squared_error {squared_error:.5f}",
                )
                np.savez(
                    join(rebuild_dir(A), f"error_{target_type}.npz"),
                    abs_error=abs_error,
                    norm=norm,
                    nmae=nmae,
                    squared_error=squared_error,
                )

        io.log(log_path, "Mean % NMAE per structure:")
        for target_type in nmaes.keys():
            io.log(
                log_path,
                f"    {target_type}:"
                f" {torch.mean(torch.tensor(nmaes[target_type])):.5f}",
            )

    # ===== Generate STM images =====
    if ml_options["EVAL"]["generate_stm"] is True:

        t0_stm = time.time()
        io.log(log_path, "Generating STM images")

        for A in test_id:
            io.log(log_path, f"    Structure: {A}")
            # Load the SCF, RI, and ML cube files
            q_scf = cube.RhoCube(
                join(dft_options["RI_DIR"](A), "cube_001_total_density.cube")
            )
            q_ri = cube.RhoCube(
                join(dft_options["RI_DIR"](A), "cube_002_ri_density.cube")
            )
            q_ml = cube.RhoCube(join(rebuild_dir(A), "cube_001_ri_density.cube"))

            # Plot the STM scatter
            if dft_options["STM"]["mode"] == "ccm":

                fig, ax = cube.plot_contour_ccm(
                    cubes=[q_scf, q_ri, q_ml],
                    save_dir=rebuild_dir(A),
                    **dft_options["STM"]["options"],
                )

            else:

                assert dft_options["STM"]["mode"] == "chm"
                fig, ax = cube.plot_contour_chm(
                    cubes=[q_scf, q_ri, q_ml],
                    save_dir=rebuild_dir(A),
                    **dft_options["STM"]["options"],
                )
            matplotlib.pyplot.close()

        dt_stm = time.time() - t0_stm
        io.log(log_path, train_utils.report_dt(dt_stm, "STM image generation complete"))
        io.log(
            log_path, train_utils.report_dt(dt_stm / len(test_id), "    or per frame")
        )

    # ===== Report eval timings =====
    dt_eval = time.time() - t0_eval
    io.log(log_path, train_utils.report_dt(dt_eval, "Evaluation complete (total time)"))


def _get_options():
    """
    Gets the DFT and ML options. Ensures the defaults are set first and then overwritten
    with user settings.
    """

    dft_options = get_options("dft", "rholearn")
    hpc_options = get_options("hpc")
    ml_options = get_options("ml", "rholearn")

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
    ml_options["ML_DIR"] = os.getcwd()
    ml_options["CHKPT_DIR"] = train_utils.create_subdir(os.getcwd(), "checkpoint")
    ml_options["EVAL_DIR"] = train_utils.create_subdir(os.getcwd(), "evaluation")

    return dft_options, hpc_options, ml_options


def _check_input_settings(dft_options: dict, ml_options: dict, frame_idxs: List[int]):
    """
    Checks input settings for validity. Assumes they have already been set
    globally, i.e. by :py:fun:`set_settings_globally`.
    """
    if not os.path.exists(dft_options["XYZ"]):
        raise FileNotFoundError(f"XYZ file not found at path: {dft_options['XYZ']}")
    if ml_options["N_TRAIN"] <= 0:
        raise ValueError("must have size non-zero training set")
    if ml_options["N_VAL"] <= 0:
        raise ValueError("must have size non-zero validation set")
    if (
        len(frame_idxs)
        < ml_options["N_TRAIN"] + ml_options["N_VAL"] + ml_options["N_TEST"]
    ):
        raise ValueError(
            "the sum of sizes of training, validation, and test"
            " sets must be <= the number of frames"
        )

    for target_type in ml_options["EVAL"]["target_type"]:
        if target_type not in ["scf", "ri"]:
            raise ValueError("EVAL['target_type'] must one or both of ['scf', 'ri']")
