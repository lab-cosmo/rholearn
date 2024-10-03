import os
import time
from os.path import exists, join
from typing import Callable

import torch

from rholearn import train_utils
from rholearn.model import RhoModel
from rholearn.settings.defaults import dft_defaults, ml_defaults, net_default
from rholearn.utils import convert, io, system
from rholearn.utils.io import pickle_dict
from rholearn.utils.utils import timestamp


def train(dft_settings: dict, ml_settings: dict, net: Callable):
    """
    Runs model training in the following steps:
        1. Build/load model, optimizer, and scheduler (if applicable)
        2. Create corss-validation splits of frames
        3. Build training and validation datasets and dataloaders
        4. For each training epoch:
            a. Peform training step, at every epoch
            b. Perform validation step, at certain intervals
            c. Log results, at certain intervals
            d. Checkpoint model, at certain intervals
    """
    t0_setup = time.time()

    # ===== Set global settings =====
    _set_settings_globally(dft_settings, ml_settings, net)  # must be in this order
    _check_input_settings()

    # ===== Setup =====
    log_path = join(ML_DIR, "logs/train.log")
    io.log(log_path, f"===== BEGIN =====")
    io.log(log_path, f"Working directory: {ML_DIR}")

    # Random seed and dtype
    torch.manual_seed(SEED)
    torch.set_default_dtype(TORCH["dtype"])

    # ===== Build model, optimizer, and scheduler if applicable =====

    # Initialize or load pre-trained model, initialize new optimizer and scheduler
    if TRAIN.get("restart_epoch") is None:

        epochs = torch.arange(TRAIN["n_epochs"] + 1)

        if PRETRAINED_MODEL is None:  # initialize model from scratch
            io.log(log_path, "Initializing model")
            atom_types = None  # TODO: generalise to subsets
            target_basis = convert.get_global_basis_set(
                [
                    io.unpickle_dict(join(PROCESSED_DIR(A), "basis_set.pickle"))
                    for A in FRAME_IDXS
                ],
                center_types=atom_types,
            )
            model = RhoModel(
                target_basis=target_basis,
                spherical_expansion_hypers=SPHERICAL_EXPANSION_HYPERS,
                n_correlations=N_CORRELATIONS,
                net=NET,
                get_selected_atoms=GET_SELECTED_ATOMS,
                **TORCH,
                angular_cutoff=ANGULAR_CUTOFF,
            )

        else:  # Use pre-trained model
            io.log(log_path, "Using pre-trained model")
            model = PRETRAINED_MODEL

        # Initialize optimizer
        io.log(log_path, "Initializing optimizer")
        optimizer = OPTIMIZER(filter(lambda p: p.requires_grad, model.parameters()))

        # Initialize scheduler
        scheduler = None
        if SCHEDULER is not None:
            io.log(log_path, "Using LR scheduler")
            scheduler = SCHEDULER(optimizer)

    # Load model, optimizer, scheduler from checkpoint for restarting training
    else:

        epochs = torch.arange(TRAIN["restart_epoch"] + 1, TRAIN["n_epochs"] + 1)

        # Load model
        io.log(log_path, "Loading model from checkpoint")
        model = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "model.pt"))

        # Load optimizer
        io.log(log_path, "Loading optimizing from checkpoint")
        optimizer = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt"))

        # Load scheduler
        scheduler = None
        if exists(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt")):
            io.log(log_path, "Loading scheduler from checkpoint")
            scheduler = torch.load(
                join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt")
            )

    # Try a model save/load
    torch.save(model, join(ML_DIR, "model.pt"))
    torch.load(join(ML_DIR, "model.pt"))
    os.remove(join(ML_DIR, "model.pt"))
    io.log(log_path, str(model).replace("\n", f"\n# {timestamp()}"))

    # ===== Create datasets and dataloaders =====

    # First define subsets of system IDs for crossval
    io.log(log_path, f"Split system IDs into subsets")
    all_subset_id = train_utils.crossval_idx_split(  # cross-validation split of idxs
        frame_idxs=FRAME_IDXS,
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        seed=SEED,
    )
    pickle_dict(
        join(ML_DIR, "crossval_idxs.pickle"),
        {
            "train": all_subset_id[0],
            "val": all_subset_id[1],
            "test": all_subset_id[2],
        }
    )
    all_frames = system.read_frames_from_xyz(XYZ)

    # Train dataset
    io.log(log_path, f"Training system ID: {all_subset_id[0]}")
    io.log(log_path, "Build training dataset")
    train_dataset = train_utils.get_dataset(
        frames=[all_frames[A] for A in all_subset_id[0]],
        frame_idxs=list(all_subset_id[0]),
        model=model,
        data_names=TRAIN_DATA_NAMES,
        load_dir=PROCESSED_DIR,
        overlap_cutoff=OVERLAP_CUTOFF,
        **TORCH,
    )

    # Train dataloader
    io.log(log_path, "Build training dataloader")
    train_dloader = train_utils.get_dataloader(
        dataset=train_dataset,
        join_kwargs={
            "remove_tensor_name": True,
            "different_keys": "union",
        },
        dloader_kwargs={
            "batch_size": TRAIN["batch_size"],
            "shuffle": True,
        },
    )

    # Validation dataset
    io.log(log_path, f"Validation system ID: {all_subset_id[1]}")
    io.log(log_path, "Build validation dataset")
    val_dataset = train_utils.get_dataset(
        frames=[all_frames[A] for A in all_subset_id[1]],
        frame_idxs=list(all_subset_id[1]),
        model=model,
        data_names=VAL_DATA_NAMES,
        load_dir=PROCESSED_DIR,
        overlap_cutoff=None,  # no cutoff for validation
        **TORCH,
    )

    # Validation dataloader
    io.log(log_path, "Build validation dataloader")
    val_dloader = train_utils.get_dataloader(
        dataset=val_dataset,
        join_kwargs={
            "remove_tensor_name": True,
            "different_keys": "union",
        },
        dloader_kwargs={
            "batch_size": TRAIN["batch_size"],
            "shuffle": True,
        },
    )

    dt_setup = time.time() - t0_setup
    io.log(log_path, train_utils.report_dt(dt_setup, "Setup complete"))
    
    # ===== Training loop =====

    io.log(log_path, f"Start training over epochs {epochs[0]} -> {epochs[-1]}")

    t0_training = time.time()
    best_val_loss = torch.tensor(float("inf"))
    for epoch in epochs:

        # Run training step at every epoch
        t0_train = time.time()
        train_loss = train_utils.epoch_step(
            dataloader=train_dloader,
            model=model,
            loss_fn=TRAIN_LOSS_FN,
            optimizer=optimizer,
            check_metadata=epoch == 0,
        )
        dt_train = time.time() - t0_train

        # Run validation step on this epoch
        val_loss_via_c = torch.nan
        # val_loss_via_w = torch.nan
        dt_val_via_c = torch.nan
        # dt_val_via_w = torch.nan
        lr = torch.nan
        if epoch % TRAIN["val_interval"] == 0:
            # a) using target_c only
            t0_val_via_c = time.time()
            val_loss_via_c = train_utils.epoch_step(
                dataloader=val_dloader,
                model=model,
                loss_fn=VAL_LOSS_FN,
                optimizer=None,
                use_target_w=False,
                check_metadata=epoch == 0,
            )
            dt_val_via_c = time.time() - t0_val_via_c

            # # b) using target_c and target_w
            # t0_val_via_w = time.time()
            # val_loss_via_w = train_utils.epoch_step(
            #     dataloader=val_dloader,
            #     model=model,
            #     loss_fn=VAL_LOSS_FN,
            #     optimizer=None,
            #     use_target_w=True,
            #     check_metadata=epoch == 0,
            # )
            # dt_val_via_w = time.time() - t0_val_via_w

            # Step scheduler based on validation loss via c
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss_via_c)
                else:
                    scheduler.step()
                lr = scheduler._last_lr[0]

        # Log results on this epoch
        if epoch % TRAIN["log_interval"] == 0:
            log_msg = (
                f"epoch {epoch}"
                f" train_loss {train_loss}"
                f" val_loss {val_loss_via_c}"
                # f" val_loss_via_w {val_loss_via_w}"
                f" dt_train {dt_train:.3f}"
                f" dt_val {dt_val_via_c:.3f}"
                # f" dt_val_via_w {dt_val_via_w:.3f}"
                f" lr {lr}"
            )
            io.log(log_path, log_msg)

        # Checkpoint on this epoch
        if epoch % TRAIN["checkpoint_interval"] == 0 and epoch != 0:
            train_utils.save_checkpoint(
                model, optimizer, scheduler, chkpt_dir=CHKPT_DIR(epoch)
            )

        # Save checkpoint if best validation loss
        if val_loss_via_c < best_val_loss:
            best_val_loss = val_loss_via_c
            train_utils.save_checkpoint(
                model, optimizer, scheduler, chkpt_dir=CHKPT_DIR("best")
            )

    # Finish
    dt_train = time.time() - t0_training
    io.log(log_path, train_utils.report_dt(dt_train, "Training complete"))
    io.log(log_path, f"Best validation loss: {best_val_loss:.5f}")


def _set_settings_globally(
    dft_settings: dict, ml_settings: dict, net: Callable
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
    for settings_dict in [dft_settings_, ml_settings_]:
        for key, value in settings_dict.items():
            globals()[key] = value

    # Then set the NN architecture
    if net is None:
        globals()["NET"] = net_default.NET
    else:
        globals()["NET"] = net

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
