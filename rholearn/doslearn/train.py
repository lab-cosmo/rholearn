import os
from os.path import join
from functools import partial
import time

import torch

import metatensor.torch as mts
from metatensor.torch.learn.data import IndexedDataset, DataLoader, group_and_join

from rholearn.utils import system

from rholearn.aims_interface import io
from rholearn.doslearn.model import SoapDosNet

from rholearn.aims_interface import parser
from rholearn.options import get_options

from rholearn.doslearn import train_utils


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

    n_grid_points = int(torch.ceil(torch.tensor(ml_options["DOS"]["max_energy"] - ml_options["DOS"]["min_energy"]) / ml_options["DOS"]["interval"]))
    x_dos = ml_options["DOS"]["min_energy"] + torch.arange(n_grid_points) * ml_options["DOS"]["interval"]

    # Get atom types
    atom_types = set()
    for frame in frames:
        for type_ in system.get_types(frame):
            atom_types.update({type_})
    atom_types = list(atom_types)

    # Define out properties
    out_properties = [
        mts.Labels(
            ["point"],
            torch.arange(n_grid_points, dtype=torch.int64).reshape(-1, 1)
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
    descriptor = model.compute_descriptor(frames).to(dtype=ml_options["TRAIN"]["dtype"], device=ml_options["TRAIN"]["device"])
    descriptor = [
        mts.slice(descriptor, "samples", mts.Labels(["system"], torch.tensor([A]).reshape(-1, 1)))
        for A in frame_idxs
    ]

    # Parse eigenvalues
    io.log(log_path, "Parsing raw eigenvalues from FHI-aims output directories")
    eigvals = [
        parser.parse_eigenvalues(ml_options["SCF_DIR"](A)) for A in frame_idxs
    ]
    eigvals = [i[:2] for i in eigvals]

    # Compute splines
    io.log(log_path, "Computing splines of target eigenenergies")
    splines = [
        parser.spline_eigenenergies(
            frame, frame_idx, energies, sigma=ml_options["DOS"]["sigma"], min_energy=ml_options["DOS"]["min_energy"], max_energy=ml_options["DOS"]["max_energy"], interval=ml_options["DOS"]["interval"]
        )
        for frame, frame_idx, energies in zip(frames, frame_idxs, eigvals)
    ]

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
        shuffle=ml_options["TRAIN"]["batch_size"], 
        collate_fn=partial(group_and_join, join_kwargs={"remove_tensor_name": True})
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=ml_options["TRAIN"]["batch_size"],
        shuffle=True, 
        collate_fn=partial(group_and_join, join_kwargs={"remove_tensor_name": True})
    )

    # Define the learnable alignment
    alignment = torch.nn.Parameter(
        torch.zeros(len(train_dset), dtype=ml_options["TRAIN"]["dtype"], device=ml_options["TRAIN"]["device"])
    )

    # Get optimizer and scheduler
    optimizer = torch.optim.Adam(list(model.parameters()) + [alignment], lr=lr)
    # TODO!
    # scheduler = ... 

    # Start training loop
    for epoch in range(ml_options["TRAIN"]["n_epochs"]):

        # Training step
        epoch_train_loss = 0
        for batch in train_loader:

            optimizer.zero_grad()
            # Compute prediction
            prediction = model(frames=batch.frames, descriptor=batch.descriptor)
            prediction = prediction.keys_to_samples(["center_type"])
            prediction = mts.sum_over_samples(prediction, "atom")
            prediction = prediction[0].values

            # Align the targets
            target = train_utils.evaluate_spline(
                batch.splines[0].values,
                x_dos,
                x_dos + alignment[list(batch.sample_id)].view(-1,1)
            )

            # Compute loss
            batch_loss = torch.nn.MSELoss()(prediction, target)

            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()

        # Validation step
        # TODO!

        if epoch % 10 == 0:
            print(epoch_train_loss)


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
    dft_options["RI_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "raw", f"{frame_idx}", dft_options["RI_FIT_ID"]
    )
    dft_options["PROCESSED_DIR"] = lambda frame_idx: join(
        dft_options["DATA_DIR"], "processed", f"{frame_idx}", dft_options["RI_FIT_ID"]
    )
    ml_options["ML_DIR"] = os.getcwd()
    ml_options["CHKPT_DIR"] = train_utils.create_subdir(os.getcwd(), "checkpoint")

    return dft_options, ml_options