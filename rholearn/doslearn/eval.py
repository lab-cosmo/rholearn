import os
import time
from os.path import join

import torch

from rholearn.doslearn import train_utils
from rholearn.options import get_options
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
    dft_options, _, ml_options = _get_options()

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
    test_frames = [all_frames[A] for A in test_id]

    # TODO: complete


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
    ml_options["ML_DIR"] = os.getcwd()
    ml_options["CHKPT_DIR"] = train_utils.create_subdir(os.getcwd(), "checkpoint")
    ml_options["EVAL_DIR"] = train_utils.create_subdir(os.getcwd(), "evaluation")

    return dft_options, ml_options
