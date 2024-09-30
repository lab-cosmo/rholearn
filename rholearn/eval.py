import os
import time
from functools import partial
from os.path import exists, join
from typing import List, Union

import numpy as np
import torch

from rholearn import train_utils
from rholearn.aims_interface import fields, ri_rebuild
from rholearn.settings.defaults import dft_defaults, ml_defaults
from rholearn.utils import convert, io, system


def eval(dft_settings: dict, ml_settings: dict, hpc_settings: dict):
    """
    Runs model evaluation in the following steps:
        1. Load model
        2. Perform inference
        3. Rebuild fields by calling FHI-aims
        4. Evaluate MAE against reference fields
    """
    t0_eval = time.time()

    # ===== Set global settings =====
    _set_settings_globally(
        dft_settings, ml_settings, hpc_settings
    )  # must be in this order
    _check_input_settings()

    # ===== Setup =====
    log_path = join(ML_DIR, "logs/eval.log")
    io.log(log_path, f"===== BEGIN =====")
    io.log(log_path, f"Working directory: {ML_DIR}")

    # Random seed and dtype
    torch.manual_seed(SEED)
    torch.set_default_dtype(TORCH["dtype"])

    # Get a callable to the evaluation subdirectory, parametrized by frame idx
    rebuild_dir = lambda frame_idx: join(EVAL_DIR(EVAL["eval_epoch"]), f"{frame_idx}")

    # ===== Get the test data =====

    io.log(log_path, f"Get the test IDs and frames")
    _, _, test_id = train_utils.crossval_idx_split(  # cross-validation split of idxs
        frame_idxs=FRAME_IDXS,
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        seed=SEED,
    )
    all_frames = system.read_frames_from_xyz(XYZ)
    test_frames = [all_frames[A] for A in test_id]

    # ===== Load model and perform inference =====

    # Check that evaluation hasn't already happened
    all_aims_outs = [join(rebuild_dir(A), "aims.out") for A in test_id]
    if any([exists(aims_out) for aims_out in all_aims_outs]):
        msg = (
            "Rebuild calculations have already been run for this model."
            " Move or remove these and try again. Exiting."
        )
        io.log(log_path, msg)
        raise SystemExit(msg)

    io.log(log_path, f"Load model from checkpoint at epoch {EVAL['eval_epoch']}")
    model = torch.load(join(CHKPT_DIR(EVAL["eval_epoch"]), "model.pt"))

    # Perform inference
    io.log(log_path, f"Perform inference")
    t0_infer = time.time()
    test_preds_mts = model.predict(frames=test_frames, frame_idxs=test_id)
    dt_infer = time.time() - t0_infer
    io.log(
        log_path, train_utils.report_dt(dt_infer, "Model inference (on basis) complete")
    )
    io.log(log_path, train_utils.report_dt(dt_infer / len(test_id), "   or per frame"))

    # ===== Rebuild fields =====
    io.log(log_path, f"Rebuild real-space field(s) in FHI-aims")

    # Convert predicted coefficients to flat numpy arrays
    test_preds_numpy = [
        convert.coeff_vector_blocks_to_flat(
            frame, coeff_vector, basis_set=model._target_basis
        )
        for frame, coeff_vector in zip(test_frames, test_preds_mts)
    ]

    # Run rebuild routine
    # rebuild_settings = {k: v for k, v in BASE_AIMS.items()}
    # rebuild_settings.update({k: v for k, v in REBUILD.items()})
    t0_build = time.time()
    ri_rebuild.rebuild_field(
        frame_idxs=test_id,
        frames=test_frames,
        coefficients=test_preds_numpy,
        rebuild_dir=rebuild_dir,
        aims_command=AIMS_COMMAND,
        base_settings=BASE_AIMS,
        rebuild_settings=REBUILD,
        cube_settings=CUBE,
        species_defaults=SPECIES_DEFAULTS,
        hpc_settings=HPC,
        slurm_params=SLURM_PARAMS,
    )

    # Wait for FHI-aims rebuild(s) to finish
    io.log(log_path, f"Waiting for FHI-aims rebuild calculation(s) to finish...")
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
        train_utils.report_dt(dt_build, "Build predcted field(s) in FHI-aims complete"),
    )

    # ===== Evaluate NMAE =====
    io.log(log_path, f"Evaluate MAE versus reference field type: {EVAL['target_type']}")
    nmaes = []
    for A in test_id:
        # Load grid and predicted field
        grid = np.loadtxt(join(RI_DIR(A), "partition_tab.out"))
        rho_ml = np.loadtxt(join(rebuild_dir(A), "rho_rebuilt_ri.out"))

        # Load reference field - either SCF or RI
        if EVAL["target_type"] == "scf":
            rho_ref = np.loadtxt(join(RI_DIR(A), f"rho_scf.out"))
        else:
            assert EVAL["target_type"] == "ri"
            rho_ref = np.loadtxt(join(RI_DIR(A), f"rho_rebuilt_ri.out"))

        # Get the MAE and normalization
        abs_error, norm = fields.field_absolute_error(
            input=rho_ml,
            target=rho_ref,
            grid=grid,
        )
        nmae = 100 * abs_error / norm
        nmaes.append(nmae)

        # Also compute the squared error
        squared_error, norm = fields.field_squared_error(
            input=rho_ml,
            target=rho_ref,
            grid=grid,
        )

        # Log and save the results
        io.log(
            log_path,
            f"system {A} abs_error {abs_error:.5f} norm {norm:.5f}"
            f" nmae {nmae:.5f} squared_error {squared_error:.5f}",
        )
        np.savez(
            join(rebuild_dir(A), "mae.npz"),
            abs_error=abs_error,
            norm=norm,
            nmae=nmae,
            squared_error=squared_error,
        )

    io.log(
        log_path, f"Mean % NMAE per structure: {torch.mean(torch.tensor(nmaes)):.5f}"
    )

    # ===== Finish =====
    dt_eval = time.time() - t0_eval
    io.log(log_path, train_utils.report_dt(dt_eval, "Evaluation complete (total time)"))


def _set_settings_globally(
    dft_settings: dict,
    ml_settings: dict,
    hpc_settings: dict,
) -> None:
    """
    Sets the settings globally. Ensures the defaults are set first and then
    overwritten with user settings.
    """
    # Update DFT and ML defaults with user settings
    dft_settings_ = dft_defaults.DFT_DEFAULTS
    ml_settings_ = ml_defaults.ML_DEFAULTS
    dft_settings_.update(dft_settings)
    ml_settings_.update(ml_settings)

    # Set them globally
    for settings_dict in [dft_settings_, ml_settings_, hpc_settings]:
        for key, value in settings_dict.items():
            globals()[key] = value

    # Set some directories
    globals()["SCF_DIR"] = lambda frame_idx: join(DATA_DIR, "raw", f"{frame_idx}")
    globals()["RI_DIR"] = lambda frame_idx: join(
        DATA_DIR, "raw", f"{frame_idx}", RI_FIT_ID
    )
    globals()["PROCESSED_DIR"] = lambda frame_idx: join(
        DATA_DIR, "processed", f"{frame_idx}", RI_FIT_ID
    )

    # Run directory is just the current directory
    globals()["ML_DIR"] = os.getcwd()

    # Callable directory path to checkpoint. parametrized by epoch number
    globals()["CHKPT_DIR"] = train_utils.create_subdir(os.getcwd(), "checkpoint")
    globals()["EVAL_DIR"] = train_utils.create_subdir(os.getcwd(), "evaluation")

    # Directory for the log file(s)
    os.makedirs(join(ML_DIR, "logs"), exist_ok=True)


def _check_input_settings():
    """
    Checks input settings for validatity. Assumes they have already been set
    globally, i.e. by :py:fun:`set_settings_gloablly`.
    """
    if not os.path.exists(XYZ):
        raise FileNotFoundError(f"XYZ file not found at path: {XYZ}")
    if N_TRAIN <= 0:
        raise ValueError("must have size non-zero training set")
    if N_VAL <= 0:
        raise ValueError("must have size non-zero validation set")
    if len(FRAME_IDXS) < N_TRAIN + N_VAL + N_TEST:
        raise ValueError(
            "the sum of sizes of training, validation, and test"
            " sets must be <= the number of frames"
        )

    if EVAL["target_type"] not in ["scf", "ri"]:
        raise ValueError("EVAL['target_type'] must be 'scf' or 'ri'")
