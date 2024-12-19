import os
import time
from functools import partial
from os.path import join
from typing import List

import metatensor.torch as mts
import torch
from metatensor.torch.learn.data import DataLoader, group_and_join

from rholearn.aims_interface import parser
from rholearn.doslearn import train_utils
from rholearn.doslearn.model import SoapDosNet
from rholearn.options import get_options
from rholearn.rholearn import train_utils as rho_train_utils
from rholearn.utils import io, system, utils


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

    # ===== Setup =====
    os.makedirs(join(ml_options["ML_DIR"], "outputs"), exist_ok=True)
    log_path = join(ml_options["ML_DIR"], "outputs/train.log")
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

    _check_input_settings(dft_options, ml_options, frame_idxs)  # TODO: basic checks

    # ===== Crossval split of IDs =====

    # First define subsets of system IDs for crossval
    io.log(log_path, "Split system IDs into subsets")
    all_subset_id = rho_train_utils.crossval_idx_split(
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

    # ===== Build model, optimizer, and scheduler if applicable =====

    # Initialize or load pre-trained model, initialize new optimizer and scheduler
    if ml_options["TRAIN"]["restart_epoch"] is None:

        epochs = torch.arange(1, ml_options["TRAIN"]["n_epochs"] + 1)

        if ml_options["PRETRAINED_MODEL"] is None:  # initialize model from scratch
            io.log(log_path, "Initializing model")

            # Get atom types
            atom_types = set()
            for frame in all_frames:
                for type_ in system.get_types(frame):
                    atom_types.update({type_})
            atom_types = list(atom_types)

            # Init model
            model = SoapDosNet(
                ml_options["SPHERICAL_EXPANSION_HYPERS"],
                atom_types=atom_types,
                min_energy=dft_options["DOS_SPLINES"]["min_energy"],
                max_energy=dft_options["DOS_SPLINES"]["max_energy"]
                - ml_options["TARGET_DOS"]["max_energy_buffer"],
                interval=dft_options["DOS_SPLINES"]["interval"],
                energy_reference=ml_options["TARGET_DOS"]["reference"],
                hidden_layer_widths=ml_options["HIDDEN_LAYER_WIDTHS"],
                dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
                device=ml_options["TRAIN"]["device"],
            )

        else:  # Use pre-trained model
            io.log(
                log_path,
                f"Using pre-trained model from path {ml_options['PRETRAINED_MODEL']}",
            )
            model = torch.load(
                ml_options["PRETRAINED_MODEL"],
                weights_only=False,
            )

        # Define the learnable alignment for the training structures that is used for
        # the adaptive energy reference
        train_alignment = torch.nn.Parameter(
            torch.zeros(
                len(all_subset_id[0]),
                dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
                device=ml_options["TRAIN"]["device"],
            )
        )

        # Initialize optimizer and scheduler
        io.log(log_path, "Initializing optimizer")
        if ml_options["USE_ADAPTIVE_REFERENCE"]:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + [train_alignment],
                lr=ml_options["OPTIMIZER_ARGS"]["lr"],
            )
        else:
            optimizer = torch.optim.Adam(
                list(model.parameters()),
                lr=ml_options["OPTIMIZER_ARGS"]["lr"],
            )

        # Initialize scheduler
        io.log(log_path, "Initializing scheduler")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=ml_options["SCHEDULER_ARGS"]["gamma"],
            patience=ml_options["SCHEDULER_ARGS"]["patience"],
            threshold=ml_options["SCHEDULER_ARGS"]["threshold"],
            min_lr=ml_options["SCHEDULER_ARGS"]["min_lr"],
        )

        # Initialize the best validation loss
        best_val_loss = torch.tensor(float("inf"))

    # Load model, optimizer, scheduler from checkpoint for restarting training
    else:

        epochs = torch.arange(
            ml_options["TRAIN"]["restart_epoch"] + 1,
            ml_options["TRAIN"]["restart_epoch"] + ml_options["TRAIN"]["n_epochs"] + 1,
        )

        # Load model
        io.log(log_path, "Loading model from checkpoint")
        model = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "model.pt",
            ),
            weights_only=False,
        )

        # Load optimizer
        io.log(log_path, "Loading optimizer from checkpoint")
        optimizer = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "optimizer.pt",
            ),
            weights_only=False,
        )

        # Load scheduler
        io.log(log_path, "Loading scheduler from checkpoint")
        scheduler = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "scheduler.pt",
            ),
            weights_only=False,
        )

        # Load the validation loss
        best_val_loss = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "val_loss.pt",
            ),
            weights_only=False,
        )

        # Load alignment
        train_alignment = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "train_alignment.pt",
            ),
            weights_only=False,
        )

    # Try a model save/load
    torch.save(model, join(ml_options["ML_DIR"], "model.pt"))
    torch.load(
        join(ml_options["ML_DIR"], "model.pt"),
        weights_only=False,
    )
    os.remove(join(ml_options["ML_DIR"], "model.pt"))
    io.log(log_path, str(model).replace("\n", f"\n# {utils.timestamp()}"))

    # ===== Create datasets and dataloaders =====

    # Train dataset
    io.log(log_path, f"Training system ID: {all_subset_id[0]}")
    io.log(log_path, "Build training dataset")
    train_dset = train_utils.get_dataset(
        frames=[all_frames[A] for A in all_subset_id[0]],
        frame_idxs=list(all_subset_id[0]),
        model=model,
        load_dir=dft_options["PROCESSED_DIR"],
        precomputed_descriptors=ml_options["PRECOMPUTED_DESCRIPTORS"],
        dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
        device=torch.device(ml_options["TRAIN"]["device"]),
    )

    # Build train dataloader
    io.log(log_path, "Build training dataloader")
    join_kwargs = {"remove_tensor_name": True, "different_keys": "union"}
    train_loader = DataLoader(
        train_dset,
        batch_size=ml_options["TRAIN"]["batch_size"],
        shuffle=False,
        collate_fn=partial(group_and_join, join_kwargs=join_kwargs),
    )

    # Val dataset
    io.log(log_path, f"Validation system ID: {all_subset_id[1]}")
    io.log(log_path, "Build validation dataset")
    val_dset = train_utils.get_dataset(
        frames=[all_frames[A] for A in all_subset_id[1]],
        frame_idxs=list(all_subset_id[1]),
        model=model,
        load_dir=dft_options["PROCESSED_DIR"],
        precomputed_descriptors=ml_options["PRECOMPUTED_DESCRIPTORS"],
        dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
        device=torch.device(ml_options["TRAIN"]["device"]),
    )

    # Build val dataloader
    io.log(log_path, "Build validation dataloader")
    if ml_options["TRAIN"]["val_batch_size"] is None:
        val_batch_size = len(all_subset_id[1])
    else:
        val_batch_size = ml_options["TRAIN"]["val_batch_size"]
    val_loader = DataLoader(
        val_dset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=partial(group_and_join, join_kwargs=join_kwargs),
    )

    # Get the spline positions
    spline_positions = train_utils.get_spline_positions(
        min_energy=dft_options["DOS_SPLINES"]["min_energy"],
        max_energy=dft_options["DOS_SPLINES"]["max_energy"],
        # - ml_options["TARGET_DOS"]["max_energy_buffer"],
        interval=dft_options["DOS_SPLINES"]["interval"],
    )

    # Finish setup
    dt_setup = time.time() - t0_setup
    io.log(log_path, rho_train_utils.report_dt(dt_setup, "Setup complete"))

    # ===== Training loop =====

    io.log(log_path, f"Start training over epochs {epochs[0]} -> {epochs[-1]}")

    t0_training = time.time()
    for epoch in epochs:

        # Training step
        train_loss = 0
        for batch in train_loader:

            optimizer.zero_grad()  # zero gradients

            # Compute prediction
            dos_pred_train = model(frames=batch.frames, descriptor=batch.descriptor)

            # Sum over "center_type" and mean over "atom"
            dos_pred_train = mts.sum_over_samples(dos_pred_train, "center_type")
            dos_pred_train = mts.mean_over_samples(dos_pred_train, "atom")
            dos_pred_train = dos_pred_train[0].values

            # Align the targets. Enforce that alignment has a mean of 0 to eliminate
            # systematic shifts across the entire dataset
            batch_train_alignment = train_alignment[
                [
                    (torch.tensor(all_subset_id[0]) == sample_id).nonzero(as_tuple=True)[0].item()
                    for sample_id in batch.sample_id
                ]
            ]
            normalized_alignment = batch.energy_reference.squeeze() + (
                batch_train_alignment - torch.mean(train_alignment)
            )

            # Compute the target DOS
            dos_target_train = train_utils.evaluate_spline(
                batch.splines[0].values,
                spline_positions,
                model._x_dos + normalized_alignment.view(-1, 1),
            )
            # Compute loss
            train_loss_batch = train_utils.t_get_mse(
                dos_pred_train, dos_target_train, model._x_dos
            )
            train_loss_batch *= len(list(batch.sample_id)) / ml_options["N_TRAIN"]

            # Update parameters
            train_loss_batch.backward()
            optimizer.step()

            train_loss += train_loss_batch.item()

        # Validation step
        val_loss = torch.nan
        if epoch % ml_options["TRAIN"]["val_interval"] == 0:
            dos_pred_val = []
            dos_splines_val = []
            energy_reference_val = []
            for batch in val_loader:
                # Make a prediction
                dos_pred_val_batch = model(frames=batch.frames, descriptor=batch.descriptor)
                
                # Reduce over "center_type" and "atom"
                dos_pred_val_batch = mts.sum_over_samples(dos_pred_val_batch, "center_type")
                dos_pred_val_batch = mts.mean_over_samples(dos_pred_val_batch, "atom")
                dos_pred_val_batch = dos_pred_val_batch[0].values

                # Store various quantities from this batch
                dos_pred_val.append(dos_pred_val_batch)
                dos_splines_val.append(batch.splines[0].values)
                energy_reference_val.append(batch.energy_reference)

            # Optimize the adaptive energy reference
            if ml_options["USE_ADAPTIVE_REFERENCE"]:
                val_loss, _ = train_utils.opt_mse_spline(
                    torch.vstack(dos_pred_val),
                    model._x_dos,
                    torch.vstack(dos_splines_val),
                    spline_positions,
                    n_epochs=50,
                )

            # Or used the fixed energy reference (either "Fermi" or "Hartree")
            else:
                with torch.no_grad():
                    # Compute the target DOS via splines. TODO: for a fixed energy
                    # reference, splines are not needed here. Instead the unshifted
                    # target DOS can just be stored and used in validation loss
                    # computation. For now, we use the slower version of evaluating the
                    # spline for consistency with the adaptive energy reference
                    # approach.
                    dos_target_val = train_utils.evaluate_spline(
                        torch.vstack(dos_splines_val),
                        spline_positions,
                        model._x_dos 
                        + torch.concatenate(energy_reference_val).view(-1, 1),
                    )

                    # Compute the validation loss
                    val_loss = train_utils.t_get_mse(
                        torch.vstack(dos_pred_val),
                        dos_target_val,
                        model._x_dos,
                    )
            scheduler.step(val_loss)

        # Log results on this epoch
        lr = scheduler._last_lr[0]
        if epoch % ml_options["TRAIN"]["log_interval"] == 0 or epoch == 1:
            log_msg = (
                f"epoch {epoch}"
                f" train_loss {train_loss}"
                f" val_loss {val_loss}"
                f" lr {lr}"
            )
            io.log(log_path, log_msg)

        # Checkpoint on this epoch, including alignment
        if epoch % ml_options["TRAIN"]["checkpoint_interval"] == 0:
            rho_train_utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                best_val_loss,
                chkpt_dir=ml_options["CHKPT_DIR"](epoch),
            )
            torch.save(train_alignment, join(ml_options["CHKPT_DIR"](epoch), "alignment.pt"))

        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            rho_train_utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                val_loss,
                chkpt_dir=ml_options["CHKPT_DIR"]("best"),
            )
            torch.save(train_alignment, join(ml_options["CHKPT_DIR"]("best"), "alignment.pt"))

        # Early stopping
        if lr <= ml_options["SCHEDULER_ARGS"]["min_lr"]:
            io.log(log_path, "Early stopping")
            break

    # Finish
    dt_train = time.time() - t0_training
    io.log(log_path, rho_train_utils.report_dt(dt_train, "Training complete"))
    io.log(log_path, f"Best validation loss: {best_val_loss:.5f}")


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
