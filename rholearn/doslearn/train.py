import os
import time
from functools import partial
from os.path import join

import metatensor.torch as mts
import torch
from metatensor.torch.learn.data import (DataLoader, IndexedDataset,
                                         group_and_join)

from rholearn.aims_interface import io, parser
from rholearn.doslearn import train_utils
from rholearn.doslearn.model import SoapDosNet
from rholearn.options import get_options
from rholearn.utils import system


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

    # Read frames and frame indices
    frames = system.read_frames_from_xyz(dft_options["XYZ"])
    frame_idxs = list(range(len(frames)))

    # ===== Setup =====
    os.makedirs(join(ml_options["ML_DIR"], "outputs"), exist_ok=True)
    log_path = join(ml_options["ML_DIR"], "outputs/train.log")
    io.log(log_path, "===== BEGIN =====")
    io.log(log_path, f"Working directory: {ml_options['ML_DIR']}")

    # Random seed and dtype
    torch.manual_seed(ml_options["SEED"])
    torch.set_default_dtype(getattr(torch, ml_options["TRAIN"]["dtype"]))
    # Define prediction window
    n_grid_points = int(
        torch.ceil(
            torch.tensor(
                ml_options["PREDICTED_DOS"]["max_energy"]
                - ml_options["PREDICTED_DOS"]["min_energy"]
            )
            / ml_options["PREDICTED_DOS"]["interval"]
        )
    )
    x_dos = (
        ml_options["PREDICTED_DOS"]["min_energy"]
        + torch.arange(n_grid_points) * ml_options["PREDICTED_DOS"]["interval"]
    )
    # Define spline window
    n_spline_points = int(
        torch.ceil(
            torch.tensor(
                ml_options["DOS_SPLINES"]["max_energy"]
                - ml_options["DOS_SPLINES"]["min_energy"]
            )
            / ml_options["DOS_SPLINES"]["interval"]
        )
    )
    spline_positions = (
        ml_options["DOS_SPLINES"]["min_energy"]
        + torch.arange(n_grid_points) * ml_options["DOS_SPLINES"]["interval"]
    )

    # Get atom types
    atom_types = set()
    for frame in frames:
        for type_ in system.get_types(frame):
            atom_types.update({type_})
    atom_types = list(atom_types)

    # Define out properties
    out_properties = [
        mts.Labels(
            ["point"], torch.arange(n_grid_points, dtype=torch.int64).reshape(-1, 1)
        )
        for _ in atom_types
    ]

    # Initialize model
    io.log(log_path, "Initializing model")
    model = SoapDosNet(
        ml_options["SPHERICAL_EXPANSION_HYPERS"],
        atom_types=atom_types,
        out_properties=out_properties,
        hidden_layer_widths=ml_options["HIDDEN_LAYER_WIDTHS"],
        dtype=ml_options["TRAIN"]["dtype"],
        device=ml_options["TRAIN"]["device"],
    )

    # Try a model save/load
    torch.save(model, join(ml_options["ML_DIR"], "model.pt"))
    torch.load(join(ml_options["ML_DIR"], "model.pt"))
    os.remove(join(ml_options["ML_DIR"], "model.pt"))
    io.log(log_path, str(model).replace("\n", f"\n# {utils.timestamp()}"))

    # Compute descriptor
    io.log(log_path, "Computing SOAP power spectrum descriptors")
    descriptor = model.compute_descriptor(frames).to(
        dtype=ml_options["TRAIN"]["dtype"], device=ml_options["TRAIN"]["device"]
    )
    descriptor = [
        mts.slice(
            descriptor,
            "samples",
            mts.Labels(["system"], torch.tensor([A]).reshape(-1, 1)),
        )
        for A in frame_idxs
    ]

    # Check if splines have been computed
    splines_precomputed = os.path.exists(join(ml_options["ML_DIR"], "splines.pt"))
    if splines_precomputed:
        splines = torch.load(join(ml_options["ML_DIR"], "splines.pt"))
    else:
        # Parse eigenvalues
        io.log(log_path, "Parsing raw eigenvalues from FHI-aims output directories")
        eigvals = [
            parser.parse_eigenvalues(ml_options["SCF_DIR"](A)) for A in frame_idxs
        ]
        eigvals = [i for i in eigvals]
        # Set Energy Reference
        if ml_options["PREDICTED_DOS"]["reference"] == "Fermi":
            energy_reference = [
                parser.parse_fermi_energy(ml_options["SCF_DIR"](A)) for A in frame_idxs
            ]
        elif ml_options["PREDICTED_DOS"]["reference"] == "Hartree":
            energy_reference = [0.0] * len(eigvals)

        # Compute splines
        io.log(log_path, "Computing splines of target eigenenergies")
        splines = [
            train_utils.spline_eigenenergies(
                frame,
                frame_idx,
                energies,
                reference,
                sigma=ml_options["PREDICTED_DOS"]["sigma"],
                min_energy=ml_options["DOS_SPLINES"]["min_energy"],
                max_energy=ml_options["DOS_SPLINES"]["max_energy"],
                interval=ml_options["DOS_SPLINES"]["interval"],
            )
            for frame, frame_idx, energies, reference in zip(
                frames, frame_idxs, eigvals, energy_reference
            )
        ]
        torch.save(splines, join(ml_options["ML_DIR"], "splines.pt"))

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

    # Build datasets
    train_dset = IndexedDataset(
        sample_id=all_subset_id[0],
        frames=[frames[i] for i in all_subset_id[0]],
        descriptor=[descriptor[i] for i in all_subset_id[0]],
        splines=[splines[i] for i in all_subset_id[0]],
    )
    val_dset = IndexedDataset(
        sample_id=all_subset_id[1],
        frames=[frames[i] for i in all_subset_id[1]],
        descriptor=[descriptor[i] for i in all_subset_id[1]],
        splines=[splines[i] for i in all_subset_id[1]],
    )

    # Build dataloaders
    train_loader = DataLoader(
        train_dset,
        batch_size=ml_options["TRAIN"]["batch_size"],
        shuffle=True,
        collate_fn=partial(group_and_join, join_kwargs={"remove_tensor_name": True}),
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=ml_options["TRAIN"]["val_batch_size"],
        shuffle=False,
        collate_fn=partial(group_and_join, join_kwargs={"remove_tensor_name": True}),
    )

    # Define the learnable alignment that is used for the adaptive energy reference
    alignment = torch.nn.Parameter(
        torch.zeros(
            len(train_dset),
            dtype=ml_options["TRAIN"]["dtype"],
            device=ml_options["TRAIN"]["device"],
        )
    )

    if ml_options["TRAIN"]["restart_epoch"] is None:
        # Initialize necessary variables, optimizer and scheduler
        io.log(log_path, "Initializing Training Variables")
        epochs = torch.arange(1, ml_options["TRAIN"]["n_epochs"] + 1)
        best_state = copy.deepcopy(model.state_dict())
        best_alignment = copy.deepcopy(alignment.detach())
        best_train_loss = torch.tensor(100.0)
        best_val_loss = torch.tensor(100.0)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + [alignment],
            lr=ml_options["OPTIMIZER_ARGS"]["lr"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=ml_options["SCHEDULER_ARGS"]["gamma"],
            patience=ml_options["SCHEDULER_ARGS"]["patience"],
            threshold=ml_options["SCHEDULER_ARGS"]["threshold"],
            min_lr=ml_options["SCHEDULER_ARGS"]["min_lr"],
        )
    else:
        io.log(
            log_path,
            f"Loading Training Variables and Model from Epoch: {ml_options['TRAIN']['restart_epoch']}",
        )
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
        # Load best model state dict
        io.log(log_path, "Loading best state from checkpoint")
        model = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "best_model_state.pt",
            )
        )
        # Load alignment
        io.log(log_path, "Loading alignment from checkpoint")
        alignment = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "alignment.pt",
            )
        )
        io.log(log_path, "Loading best alignment from checkpoint")
        best_alignment = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "best_alignment.pt",
            )
        )
        # Load optimizer
        io.log(log_path, "Loading Optimizer from checkpoint")
        optimizer = torch.optim.Adam(
            list(model.parameters()) + [alignment],
            lr=ml_options["OPTIMIZER_ARGS"]["lr"],
        )
        optimizer.load_state_dict(
            torch.load(
                join(
                    ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                    "optimizer_state_dict.pt",
                )
            )
        )
        # Load Scheduler
        io.log(log_path, "Loading Scheduler from checkpoint")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=ml_options["SCHEDULER_ARGS"]["gamma"],
            patience=ml_options["SCHEDULER_ARGS"]["patience"],
            threshold=ml_options["SCHEDULER_ARGS"]["threshold"],
            min_lr=ml_options["SCHEDULER_ARGS"]["min_lr"],
        )
        scheduler.load_state_dict(
            torch.load(
                join(
                    ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                    "scheduler_state_dict.pt",
                )
            )
        )
        # Load best training performances
        io.log(log_path, "Loading Best Training Performance from checkpoint")
        best_train_loss, best_val_loss = torch.load(
            join(
                ml_options["CHKPT_DIR"](
                    ml_options["TRAIN"]["restart_epoch"], "parameters.pt"
                )
            )
        )

    dt_setup = time.time() - t0_setup
    io.log(log_path, train_utils.report_dt(dt_setup, "Setup complete"))

    # Start training loop
    for epoch in epochs:

        # Training step
        epoch_train_loss = 0
        for batch in train_loader:

            optimizer.zero_grad()
            # Compute prediction
            prediction = model(frames=batch.frames, descriptor=batch.descriptor)
            prediction = prediction.keys_to_samples(["center_type"])
            prediction = mts.mean_over_samples(prediction, "atom")
            prediction = prediction[0].values

            # Align the targets
            ## Enforce that alignment has a mean of 0 to eliminate systematic shifts across the entire dataset
            normalized_alignment = alignment - torch.mean(alignment)
            target = train_utils.evaluate_spline(
                batch.splines[0].values,
                x_dos,
                spline_positions
                + normalized_alignment[list(batch.sample_id)].view(-1, 1),
            )

            # Compute loss
            batch_loss = train_utils.t_get_mse(prediction, target, x_dos)
            batch_loss *= len(list(batch.sample_id)) / ml_options["N_TRAIN"]

            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()
        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss

        # Validation step
        if epoch % ml_options["TRAIN"]["val_interval"] == 0:
            # Performs validation step
            val_predictions = []
            for batch in val_loader:
                prediction = model(frames=batch.frames, descriptor=batch.descriptor)
                prediction = prediction.keys_to_samples(["center_type"])
                prediction = mts.mean_over_samples(prediction, "atom")
                prediction = prediction[0].values
                val_predictions.append(prediction)

            epoch_val_loss, _ = train_utils.Opt_MSE_spline(
                torch.vstack(val_predictions),
                x_dos,
                torch.vstack(val_dset.splines),
                spline_positions,
                n_epochs=50,
            )
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_alignment = alignment.detach()

        scheduler.step(epoch_val_loss)
        if epoch % ml_options["TRAIN"]["log_interval"] == 0:
            # Logs the current and best losses
            io.log(
                log_path,
                (
                    f"Epoch {epoch}: Current loss (Train, Val): {epoch_train_loss.item():.4}, {epoch_val_loss.item():.4}"
                    f", Best loss (Train, Val): {best_train_loss.item():.4}, {best_val_loss.item():.4}"
                ),
            )

        if epoch % ml_options["TRAIN"]["checkpoint_interval"] == 0:
            # Save relevant parameters
            train_utils.save_checkpoint(
                model,
                best_state,
                alignment,
                best_alignment,
                [best_train_loss, best_val_loss],
                optimizer,
                scheduler,
                chkpt_dir=ml_options["CHKPT_DIR"](epoch),
            )

        if scheduler.get_last_lr() <= ml_options["SCHEDULER_ARGS"]["min_lr"]:
            break

    # After training ends
    train_utils.save_checkpoint(
        model,
        best_state,
        alignment,
        best_alignment,
        [best_train_loss, best_val_loss],
        optimizer,
        scheduler,
        chkpt_dir=ml_options["CHKPT_DIR"](epoch),
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
    ml_options["ML_DIR"] = os.getcwd()
    ml_options["CHKPT_DIR"] = train_utils.create_subdir(os.getcwd(), "checkpoint")

    return dft_options, ml_options
