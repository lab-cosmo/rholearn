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

    # Get frame indices we have data for (or a subset if specified)
    if dft_options.get("IDX_SUBSET") is not None:
        frame_idxs = dft_options.get("IDX_SUBSET")
    else:
        frame_idxs = None
    # Load all the frames
    all_frames = system.read_frames_from_xyz(dft_options["XYZ"], frame_idxs)

    if frame_idxs is None:
        frame_idxs = list(range(len(all_frames)))

    _check_input_settings(dft_options, ml_options, frame_idxs)  # TODO: basic checks

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

            # Get atom types
            atom_types = set()
            for frame in all_frames:
                for type_ in system.get_types(frame):
                    atom_types.update({type_})
            atom_types = list(atom_types)

            # Initialize model
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
            print(model._x_dos)

        else:  # Use pre-trained model
            io.log(
                log_path,
                f"Using pre-trained model from path {ml_options['PRETRAINED_MODEL']}",
            )
            model = torch.load(ml_options["PRETRAINED_MODEL"])

        # Load the Fermi levels
        if model._energy_reference == "Fermi":
            # TODO: make this part of the data parsing - i.e. save a torch tensor
            # "fermi.pt" along with "dos_spline.npz"?
            energy_reference = [
                parser.parse_fermi_energy(dft_options["SCF_DIR"](A)) for A in frame_idxs
            ]
        else:
            assert ml_options["TARGET_DOS"]["reference"] == "Hartree"
            energy_reference = [0.0] * len(frame_idxs)
        energy_reference = torch.tensor(energy_reference)
        # Define the learnable alignment that is used for the adaptive energy reference
        alignment = torch.nn.Parameter(
            torch.zeros_like(
                energy_reference,
                dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
                device=ml_options["TRAIN"]["device"],
            )
        )

        # Initialize optimizer and scheduler
        io.log(log_path, "Initializing optimizer")
        optimizer = torch.optim.Adam(
            list(model.parameters()) + [alignment],
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

        # Load alignment
        alignment = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "alignment.pt",
            )
        )

    # Try a model save/load
    torch.save(model, join(ml_options["ML_DIR"], "model.pt"))
    torch.load(join(ml_options["ML_DIR"], "model.pt"))
    os.remove(join(ml_options["ML_DIR"], "model.pt"))
    io.log(log_path, str(model).replace("\n", f"\n# {utils.timestamp()}"))

    # ===== Create datasets and dataloaders =====

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
        max_energy=dft_options["DOS_SPLINES"]["max_energy"]
        - ml_options["TARGET_DOS"]["max_energy_buffer"],
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

            optimizer.zero_grad()

            # Compute prediction
            prediction = model(frames=batch.frames, descriptor=batch.descriptor)
            prediction = mts.mean_over_samples(prediction, "atom")
            prediction = prediction[0].values

            # Align the targets with respect to the original energy reference.
            # Enforce that alignment has a mean of 0 to eliminate
            # systematic shifts across the entire dataset
            normalized_alignment = energy_reference + (
                alignment - torch.mean(alignment)
            )
            target = train_utils.evaluate_spline(
                batch.splines[0].values,
                spline_positions,
                model._x_dos + normalized_alignment[list(batch.sample_id)].view(-1, 1),
            )

            # Compute loss
            batch_loss = train_utils.t_get_mse(prediction, target, model._x_dos)
            batch_loss *= len(list(batch.sample_id)) / ml_options["N_TRAIN"]

            # Update parameters
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        # Validation step
        val_loss = torch.nan
        if epoch % ml_options["TRAIN"]["val_interval"] == 0:
            val_predictions = []
            val_splines = []
            for batch in val_loader:
                prediction = model(frames=batch.frames, descriptor=batch.descriptor)
                prediction = mts.mean_over_samples(prediction, "atom")
                prediction = prediction[0].values
                val_predictions.append(prediction)
                val_splines.append(batch.splines[0].values)

            val_loss, _, _ = train_utils.opt_mse_spline(
                torch.vstack(val_predictions),
                model._x_dos,
                torch.vstack(val_splines),
                spline_positions,
                n_epochs=50,
            )
            scheduler.step(val_loss)

        lr = scheduler._last_lr[0]

        # Log results on this epoch
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
            torch.save(alignment, join(ml_options["CHKPT_DIR"](epoch), "alignment.pt"))

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
            torch.save(alignment, join(ml_options["CHKPT_DIR"]("best"), "alignment.pt"))

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
