import os
import time
from os.path import exists, join
from typing import List

import numpy as np
import torch

from rholearn import train_utils
from rholearn.aims_interface import fields, ri_rebuild
from rholearn.options import get_options
from rholearn.utils import convert, io, system


def eval():
    """
    Runs model evaluation in the following steps:
        1. Load model
        2. Perform inference
        3. Rebuild fields by calling FHI-aims
        4. Evaluate MAE against reference fields
    """
    t0_eval = time.time()
    dft_options, hpc_options, ml_options = _get_options()

    # Read frames and frame indices
    frames = system.read_frames_from_xyz(dft_options["XYZ"])
    frame_idxs = list(range(len(frames)))

    _check_input_settings(dft_options, ml_options, frame_idxs)

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

    # ===== Get the test data =====

    io.log(log_path, "Get the test IDs and frames")
    _, _, test_id = train_utils.crossval_idx_split(  # cross-validation split of idxs
        frame_idxs=frame_idxs,
        n_train=ml_options["N_TRAIN"],
        n_val=ml_options["N_VAL"],
        n_test=ml_options["N_TEST"],
        seed=ml_options["SEED"],
    )
    all_frames = system.read_frames_from_xyz(dft_options["XYZ"])
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

    io.log(
        log_path,
        f"Load model from checkpoint at epoch {ml_options['EVAL']['eval_epoch']}",
    )
    model = torch.load(
        join(ml_options["CHKPT_DIR"](ml_options["EVAL"]["eval_epoch"]), "model.pt")
    )

    # Perform inference
    io.log(log_path, "Perform inference")
    t0_infer = time.time()
    test_preds_mts = model.predict(frames=test_frames, frame_idxs=test_id)
    dt_infer = time.time() - t0_infer
    io.log(
        log_path, train_utils.report_dt(dt_infer, "Model inference (on basis) complete")
    )
    io.log(log_path, train_utils.report_dt(dt_infer / len(test_id), "   or per frame"))

    # ===== Rebuild fields =====
    io.log(log_path, "Rebuild real-space field(s) in FHI-aims")

    # Convert predicted coefficients to flat numpy arrays
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
        train_utils.report_dt(dt_build, "Build predcted field(s) in FHI-aims complete"),
    )

    # ===== Evaluate NMAE =====
    io.log(
        log_path,
        "Evaluate MAE versus reference field type:"
        f" {ml_options['EVAL']['target_type']}",
    )
    nmaes = []
    for A in test_id:
        # Load grid and predicted field
        grid = np.loadtxt(join(dft_options["RI_DIR"](A), "partition_tab.out"))
        rho_ml = np.loadtxt(join(rebuild_dir(A), "rho_rebuilt_ri.out"))

        # Load reference field - either SCF or RI
        if ml_options["EVAL"]["target_type"] == "scf":
            rho_ref = np.loadtxt(join(dft_options["RI_DIR"](A), "rho_scf.out"))
        else:
            assert ml_options["EVAL"]["target_type"] == "ri"
            rho_ref = np.loadtxt(join(dft_options["RI_DIR"](A), "rho_rebuilt_ri.out"))

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


def _get_options():
    """
    Gets the DFT and ML options. Ensures the defaults are set first and then overwritten
    with user settings.
    """

    dft_options = get_options("dft")
    hpc_options = get_options("hpc")
    ml_options = get_options("ml")

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

    if ml_options["EVAL"]["target_type"] not in ["scf", "ri"]:
        raise ValueError("EVAL['target_type'] must be 'scf' or 'ri'")
