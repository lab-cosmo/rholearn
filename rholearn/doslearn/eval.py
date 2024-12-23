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

    # ===== Setup =====
    os.makedirs(join(ml_options["ML_DIR"], "outputs"), exist_ok=True)
    log_path = join(ml_options["ML_DIR"], "outputs/eval.log")
    io.log(log_path, "===== BEGIN =====")
    io.log(log_path, f"Working directory: {ml_options['ML_DIR']}")

    # Random seed and dtype
    torch.manual_seed(ml_options["SEED"])
    torch.set_default_dtype(getattr(torch, ml_options["TRAIN"]["dtype"]))

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
    test_splines = torch.vstack([tensor[0].values for tensor in test_splines_mts])
    test_frames = [all_frames[A] for A in test_id]

    # ===== Load model and perform inference =====

    io.log(
        log_path,
        f"Load model from checkpoint at epoch {ml_options['EVAL']['eval_epoch']}",
    )
    model = torch.load(
        join(ml_options["CHKPT_DIR"](ml_options["EVAL"]["eval_epoch"]), "model.pt"),
        weights_only=False,
    )

    # Perform inference to predict the DOS for each structure. Using model.predict gives
    # a non-normalized DOS, i.e. where atomic contributions are summed, not averaged.
    io.log(log_path, "Perform inference")
    t0_infer = time.time()
    test_preds_mts = model.predict(frames=test_frames, frame_idxs=test_id)
    dt_infer = time.time() - t0_infer
    io.log(log_path, rho_train_utils.report_dt(dt_infer, "Model inference complete"))
    io.log(
        log_path, rho_train_utils.report_dt(dt_infer / len(test_id), "   or per frame")
    )

    # Save predictions
    for A, test_pred in zip(test_id, test_preds_mts):
        os.makedirs(rebuild_dir(A), exist_ok=True)
        np.save(
            join(rebuild_dir(A), "prediction.npy"),
            test_pred[0].values.detach().numpy()
        )

    # ===== Evaluate test DOS RMSE

    # Normalize DOS wrt system size for evaluation
    test_preds_eval = model(frames=test_frames, frame_idxs=test_id)
    test_preds_eval = mts.sum_over_samples(test_preds_eval, "center_type")
    test_preds_eval = mts.mean_over_samples(test_preds_eval, "atom")
    test_preds_eval = test_preds_eval[0].values
    
    # Compute the spline positions
    spline_positions = train_utils.get_spline_positions(
        min_energy=dft_options["DOS_SPLINES"]["min_energy"],
        max_energy=dft_options["DOS_SPLINES"]["max_energy"],
        interval=dft_options["DOS_SPLINES"]["interval"],
    )

    # Obtain shift invariant MSE
    if ml_options["TARGET_DOS"]["adaptive_reference"]:
        test_mse, shited_test_targets, test_shifts = train_utils.opt_mse_spline(
            test_preds_eval,
            model._x_dos,
            test_splines,
            spline_positions,
            n_epochs=200,
        )
        for index_A, A in enumerate(test_id):
            np.save(
                join(rebuild_dir(A), "target_shift.npy"),
                test_shifts[index_A].detach().numpy(),
            )

    # Evaluate the target DOS from splines without optimizing the shift
    else:
        shited_test_targets = train_utils.evaluate_spline(
            test_splines,
            spline_positions,
            model._x_dos + torch.zeros(len(test_id)).view(-1, 1),
        )
        test_mse = train_utils.t_get_rmse(
            test_preds_eval, shited_test_targets, model._x_dos
        )
    io.log(log_path, f"Test RMSE: {torch.sqrt(test_mse):.5f}")
    
    
    # Compute the MSE by sample
    test_mse_samplewise = train_utils.t_get_rmse(
        test_preds_eval, shited_test_targets, model._x_dos, samplewise=True,
    )

    # Save normalized predictions, targets, and MSE for each structure
    for index_A, A in enumerate(test_id):
        np.save(
            join(rebuild_dir(A), "mse.npy"),
            test_mse_samplewise[index_A].detach().numpy(),
        )
        np.save(
            join(rebuild_dir(A), "x_dos.npy"),
            model._x_dos.detach().numpy(),
        )
        np.save(
            join(rebuild_dir(A), "normalized_prediction.npy"),
            test_preds_eval[index_A].detach().numpy(),
        )
        np.save(
            join(rebuild_dir(A), "aligned_target.npy"),
            shited_test_targets[index_A].detach().numpy(),
        )

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
