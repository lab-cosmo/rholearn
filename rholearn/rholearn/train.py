import os
import time
from os.path import exists, join
from typing import List

import torch
from rholearn.rholearn.loss import RhoLoss
from rholearn.rholearn.model import RhoModel

from rholearn.rholearn import train_utils
from rholearn.options import get_options
from rholearn.utils import convert, io, system, utils


def train():
    """
    Runs model training in the following steps:
        1. Build/load model, optimizer, and scheduler (if applicable)
        2. Create cross-validation splits of frames
        3. Build training and validation datasets and dataloaders
        4. For each training epoch:
            a. Perform training step, at every epoch
            b. Perform validation step, at certain intervals
            c. Log results, at certain intervals
            d. Checkpoint model, at certain intervals
    """
    t0_setup = time.time()
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

    _check_input_settings(dft_options, ml_options, frame_idxs)

    # ===== Setup =====
    os.makedirs(join(ml_options["ML_DIR"], "outputs"), exist_ok=True)
    log_path = join(ml_options["ML_DIR"], "outputs/train.log")
    io.log(log_path, "===== BEGIN =====")
    io.log(log_path, f"Working directory: {ml_options['ML_DIR']}")

    # Random seed and dtype
    torch.manual_seed(ml_options["SEED"])
    torch.set_default_dtype(getattr(torch, ml_options["TRAIN"]["dtype"]))

    # ===== Build model, optimizer, and scheduler if applicable =====

    # Initialize or load pre-trained model, initialize new optimizer and scheduler
    if ml_options["TRAIN"]["restart_epoch"] is None:

        epochs = torch.arange(1, ml_options["TRAIN"]["n_epochs"] + 1)

        if ml_options["PRETRAINED_MODEL"] is None:  # initialize model from scratch
            io.log(log_path, "Initializing model")
            atom_types = None  # TODO: generalise to subsets
            target_basis = convert.get_global_basis_set(
                [
                    io.unpickle_dict(
                        join(dft_options["PROCESSED_DIR"](A), "basis_set.pickle")
                    )
                    for A in frame_idxs
                ],
                center_types=atom_types,
            )
            model = RhoModel(
                target_basis=target_basis,
                spherical_expansion_hypers=ml_options["SPHERICAL_EXPANSION_HYPERS"],
                n_correlations=ml_options["N_CORRELATIONS"],
                layer_norm=ml_options["DESCRIPTOR_LAYER_NORM"],
                nn_layers=ml_options["NN_LAYERS"],
                get_selected_atoms=ml_options["GET_SELECTED_ATOMS"],
                device=torch.device(ml_options["TRAIN"]["device"]),
                dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
                angular_cutoff=ml_options["ANGULAR_CUTOFF"],
            )

        else:  # Use pre-trained model
            io.log(
                log_path,
                f"Using pre-trained model from path {ml_options['PRETRAINED_MODEL']}",
            )
            model = torch.load(ml_options["PRETRAINED_MODEL"])

        # Initialize optimizer
        io.log(log_path, "Initializing optimizer")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            **ml_options["OPTIMIZER_ARGS"],
        )

        # Initialize scheduler
        scheduler = None
        if ml_options["SCHEDULER_ARGS"] is not None:
            io.log(log_path, "Using LR scheduler")
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **ml_options["SCHEDULER_ARGS"]
            )

        # Initialize the best validation loss
        best_val_loss = torch.tensor(float("inf"))

    # Load model, optimizer, scheduler from checkpoint for restarting training
    else:

        epochs = torch.arange(
            ml_options["TRAIN"]["restart_epoch"] + 1,
            ml_options["TRAIN"]["n_epochs"] + 1,
        )

        # Load model
        io.log(log_path, "Loading model from checkpoint")
        model = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "model.pt",
            )
        )

        # Load optimizer
        io.log(log_path, "Loading optimizer from checkpoint")
        optimizer = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "optimizer.pt",
            )
        )

        # Load scheduler
        scheduler = None
        if exists(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "scheduler.pt",
            )
        ):
            io.log(log_path, "Loading scheduler from checkpoint")
            scheduler = torch.load(
                join(
                    ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                    "scheduler.pt",
                )
            )

        # Load the validation loss
        best_val_loss = torch.load(
            join(
                    ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                    "val_loss.pt",
                )
        )

    # Try a model save/load
    torch.save(model, join(ml_options["ML_DIR"], "model.pt"))
    torch.load(join(ml_options["ML_DIR"], "model.pt"))
    os.remove(join(ml_options["ML_DIR"], "model.pt"))
    io.log(log_path, str(model).replace("\n", f"\n# {utils.timestamp()}"))

    # Initialize loss functions
    train_loss_fn = RhoLoss(**ml_options["TRAIN_LOSS_FN_ARGS"])
    val_loss_fn = RhoLoss(**ml_options["VAL_LOSS_FN_ARGS"])

    # ===== Create datasets and dataloaders =====

    # First define subsets of system IDs for crossval
    io.log(log_path, "Split system IDs into subsets")
    all_subset_id = train_utils.crossval_idx_split(  # cross-validation split of idxs
        frame_idxs=frame_idxs,
        n_train=ml_options["N_TRAIN"],
        n_val=ml_options["N_VAL"],
        n_test=ml_options["N_TEST"],
        seed=ml_options["SEED"],
    )
    io.pickle_dict(
        join(ml_options["ML_DIR"], "outputs", "crossval_idxs.pickle"),
        {
            "train": all_subset_id[0],
            "val": all_subset_id[1],
            "test": all_subset_id[2],
        },
    )

    # Train dataset
    io.log(log_path, f"Training system ID: {all_subset_id[0]}")
    io.log(log_path, "Build training dataset")
    train_dataset = train_utils.get_dataset(
        frames=[all_frames[A] for A in all_subset_id[0]],
        frame_idxs=list(all_subset_id[0]),
        model=model,
        data_names=ml_options["TRAIN_DATA_NAMES"],
        load_dir=dft_options["PROCESSED_DIR"],
        overlap_cutoff=ml_options["OVERLAP_CUTOFF"],
        device=torch.device(ml_options["TRAIN"]["device"]),
        dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
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
            "batch_size": ml_options["TRAIN"]["batch_size"],
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
        data_names=ml_options["VAL_DATA_NAMES"],
        load_dir=dft_options["PROCESSED_DIR"],
        overlap_cutoff=None,  # no cutoff for validation
        device=torch.device(ml_options["TRAIN"]["device"]),
        dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
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
            "batch_size": ml_options["TRAIN"]["batch_size"],
            "shuffle": True,
        },
    )

    # Finish setup
    dt_setup = time.time() - t0_setup
    io.log(log_path, train_utils.report_dt(dt_setup, "Setup complete"))

    # ===== Training loop =====

    io.log(log_path, f"Start training over epochs {epochs[0]} -> {epochs[-1]}")

    t0_training = time.time()
    for epoch in epochs:

        # Run training step at every epoch
        t0_train = time.time()
        train_loss = train_utils.epoch_step(
            dataloader=train_dloader,
            model=model,
            loss_fn=train_loss_fn,
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
        if epoch % ml_options["TRAIN"]["val_interval"] == 0:
            # a) using target_c only
            t0_val_via_c = time.time()
            val_loss_via_c = train_utils.epoch_step(
                dataloader=val_dloader,
                model=model,
                loss_fn=val_loss_fn,
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
            #     loss_fn=val_loss_fn,
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
        if epoch % ml_options["TRAIN"]["log_interval"] == 0:
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
        if epoch % ml_options["TRAIN"]["checkpoint_interval"] == 0 and epoch != 0:
            train_utils.save_checkpoint(
                model, optimizer, scheduler, chkpt_dir=ml_options["CHKPT_DIR"](epoch)
            )

        # Save checkpoint if best validation loss
        if val_loss_via_c < best_val_loss:
            best_val_loss = val_loss_via_c
            train_utils.save_checkpoint(
                model, optimizer, scheduler, chkpt_dir=ml_options["CHKPT_DIR"]("best")
            )

        # TODO: Early stopping!
        # if scheduler.get_last_lr() <= ml_options["SCHEDULER_ARGS"]["min_lr"]:
        #     break

    # Finish
    dt_train = time.time() - t0_training
    io.log(log_path, train_utils.report_dt(dt_train, "Training complete"))
    io.log(log_path, f"Best validation loss: {best_val_loss:.5f}")


def _get_options():
    """
    Gets the DFT and ML options. Ensures the defaults are set first and then overwritten
    with user settings.
    """

    dft_options = get_options("dft", "rholearn")
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

    return dft_options, ml_options


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
