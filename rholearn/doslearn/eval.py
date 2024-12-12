import os
import time
from os.path import join

import metatensor.torch as mts
import numpy as np
import torch

from rholearn.doslearn import train_utils
from rholearn.options import get_options
from rholearn.rholearn import train_utils as rho_train_utils
from rholearn.utils import io, system


def eval():
    """
    Runs model evaluation in the following steps:
        1. Load model
        2. Perform inference
        3. Rebuild fields by calling FHI-aims
        4. Evaluate MAE against reference fields
    """
    t0_eval = time.time()
    dft_options, ml_options = _get_options()

    # Get frame indices we have data for (or a subset if specified)
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = None
    # Load all the frames
    all_frames = system.read_frames_from_xyz(dft_options["XYZ"], frame_idxs)

    if frame_idxs is None:
        frame_idxs = list(range(len(all_frames)))

    # ===== Setup =====
    os.makedirs(join(ml_options["ML_DIR"], "outputs"), exist_ok=True)
    log_path = join(ml_options["ML_DIR"], "outputs/eval.log")
    io.log(log_path, "===== BEGIN =====")
    io.log(log_path, f"Working directory: {ml_options['ML_DIR']}")

    # Load Spline Positions
    spline_positions = train_utils.get_spline_positions(
        min_energy=dft_options["DOS_SPLINES"]["min_energy"],
        max_energy=dft_options["DOS_SPLINES"]["max_energy"]
        - ml_options["TARGET_DOS"]["max_energy_buffer"],
        interval=dft_options["DOS_SPLINES"]["interval"],
    )

    # Random seed and dtype
    torch.manual_seed(ml_options["SEED"])
    torch.set_default_dtype(getattr(torch, ml_options["TRAIN"]["dtype"]))

    # Get a callable to the evaluation subdirectory, parametrized by frame idx
    rebuild_dir = lambda frame_idx: join(  # noqa: E731
        ml_options["EVAL_DIR"](ml_options["EVAL"]["eval_epoch"]), f"{frame_idx}"
    )

    # ===== Get the test data =====

    io.log(log_path, "Get the test IDs and frames")
    _, _, test_id = (
        rho_train_utils.crossval_idx_split(  # cross-validation split of idxs
            frame_idxs=frame_idxs,
            n_train=ml_options["N_TRAIN"],
            n_val=ml_options["N_VAL"],
            n_test=ml_options["N_TEST"],
            seed=ml_options["SEED"],
        )
    )
    dtype = getattr(torch, ml_options["TRAIN"]["dtype"])
    device = torch.device(ml_options["TRAIN"]["device"])
    test_splines_mts = [
        mts.load(join(dft_options["PROCESSED_DIR"](A), "dos_spline.npz")).to(
            dtype=dtype, device=device
        )
        for A in test_id
    ]
    test_splines = torch.vstack([i.blocks(0)[0].values for i in test_splines_mts])
    test_frames = [all_frames[A] for A in test_id]

    # ===== Load model and perform inference =====

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
    test_preds_mts = model(frames=test_frames, frame_idxs=test_id)
    test_preds = mts.sum_over_samples(test_preds_mts, "atom")
    test_preds = test_preds[0].values
    dt_infer = time.time() - t0_infer
    test_preds_numpy = test_preds.detach().numpy()
    for index_A, A in enumerate(test_id):
        os.makedirs(rebuild_dir(A), exist_ok=True)
        np.save(
            join(rebuild_dir(A), "prediction.npy"), test_preds_numpy[index_A]
        )  # Save in each directory
    io.log(log_path, rho_train_utils.report_dt(dt_infer, "Model inference complete"))
    io.log(
        log_path, rho_train_utils.report_dt(dt_infer / len(test_id), "   or per frame")
    )

    # Evaluate Test DOS RMSE
    # Normalize DOS wrt system size for evaluation
    test_preds_eval = mts.mean_over_samples(test_preds_mts, "atom")
    test_preds_eval = test_preds_eval[0].values
    # Obtain shift invariant MSE
    test_MSE, shited_test_targets, test_shifts = train_utils.opt_mse_spline(
        test_preds_eval,
        model._x_dos,
        test_splines,
        spline_positions,
        n_epochs=200,
    )
    errors = (test_preds_eval - shited_test_targets) ** 2
    samplewise_error = torch.trapezoid(errors, model._x_dos, dim=1).detach().numpy()
    energywise_error = torch.mean(errors, dim=0).detach().numpy()
    os.makedirs(rebuild_dir("Test"), exist_ok=True)
    np.save(join(rebuild_dir("Test"), "test_MSEs.npy"), samplewise_error)
    np.save(join(rebuild_dir("Test"), "test_eMSEs.npy"), energywise_error)
    # Save Predictions and targets in each folder
    for index_A, A in enumerate(test_id):
        np.save(
            join(rebuild_dir(A), "normalized_prediction.npy"),
            test_preds_eval[index_A].detach().numpy(),
        )
        np.save(
            join(rebuild_dir(A), "aligned_target.npy"),
            shited_test_targets[index_A].detach().numpy(),
        )
        np.save(
            join(rebuild_dir(A), "target_shift.npy"),
            test_shifts[index_A].detach().numpy(),
        )

    io.log(log_path, f"Test RMSE: {torch.sqrt(test_MSE):.5f}")

    # ===== Finish =====
    dt_eval = time.time() - t0_eval
    io.log(
        log_path, rho_train_utils.report_dt(dt_eval, "Evaluation complete (total time)")
    )


def _get_options():
    """
    Gets the DFT and ML options. Ensures the defaults are set first and then overwritten
    with user settings.
    """

    dft_options = get_options("dft", "doslearn")
    ml_options = get_options("ml", "doslearn")

    # Set some extra directories
    dft_options["SCF_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "raw", f"{frame_idx}"
    )
    dft_options["PROCESSED_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "processed", f"{frame_idx}", dft_options["RUN_ID"]
    )
    ml_options["ML_DIR"] = os.getcwd()
    ml_options["CHKPT_DIR"] = rho_train_utils.create_subdir(os.getcwd(), "checkpoint")
    ml_options["EVAL_DIR"] = rho_train_utils.create_subdir(os.getcwd(), "evaluation")

    return dft_options, ml_options
