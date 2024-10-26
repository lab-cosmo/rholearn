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
    # Define prediction window
    n_grid_points = int(torch.ceil(torch.tensor(ml_options["PREDICTED_DOS"]["max_energy"] - ml_options["PREDICTED_DOS"]["min_energy"]) / ml_options["PREDICTED_DOS"]["interval"]))
    x_dos = ml_options["PREDICTED_DOS"]["min_energy"] + torch.arange(n_grid_points) * ml_options["PREDICTED_DOS"]["interval"]
    # Define spline window
    n_spline_points = int(torch.ceil(torch.tensor(ml_options["DOS_SPLINES"]["max_energy"] - ml_options["DOS_SPLINES"]["min_energy"]) / ml_options["DOS_SPLINES"]["interval"]))
    spline_positions = ml_options["DOS_SPLINES"]["min_energy"] + torch.arange(n_grid_points) * ml_options["DOS_SPLINES"]["interval"]

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
    # Should we remove these?
    # Try a model save/load
    # torch.save(model, join(ml_options["ML_DIR"], "model.pt"))
    # torch.load(join(ml_options["ML_DIR"], "model.pt"))
    # os.remove(join(ml_options["ML_DIR"], "model.pt"))
    # io.log(log_path, str(model).replace("\n", f"\n# {utils.timestamp()}"))

    # Check if descriptors have been computed
    descriptor_saved = os.path.exists(join(ml_options["ML_DIR"], "descriptors.pt"))
    if descriptor_saved:
        descriptor = torch.load(join(ml_options["ML_DIR"], "descriptors.pt"))
    else:
        # Compute descriptor
        io.log(log_path, "Computing SOAP power spectrum descriptors")
        descriptor = model.compute_descriptor(frames).to(dtype=ml_options["TRAIN"]["dtype"], device=ml_options["TRAIN"]["device"])
        descriptor = [
            mts.slice(descriptor, "samples", mts.Labels(["system"], torch.tensor([A]).reshape(-1, 1)))
            for A in frame_idxs
        ]
        torch.save(descriptor, join(ml_options["ML_DIR"], "descriptors.pt")) # Is this the best way to save it?
    # Check if splines have been computed
    splines_saved = os.path.exists(join(ml_options["ML_DIR"], "splines.pt"))
    if descriptor_saved:
        descriptor = torch.load(join(ml_options["ML_DIR"], "splines.pt"))
    else:
    # Parse eigenvalues
        io.log(log_path, "Parsing raw eigenvalues from FHI-aims output directories")
        eigvals = [
            parser.parse_eigenvalues(ml_options["SCF_DIR"](A)) for A in frame_idxs
        ]
        eigvals = [i[:2] for i in eigvals] # For debugging, remove :2 during final run
        # Set Energy Reference
        if ml_options["PREDICTED_DOS"]["reference"] == 'Fermi':
            energy_reference = [
                parser.parse_fermi_energy(ml_options["SCF_DIR"](A)) for A in frame_idxs
            ]
        elif ml_options["PREDICTED_DOS"]["reference"] == 'Hartree':
            energy_reference = [0.] * len(eigvals)

        # Compute splines
        io.log(log_path, "Computing splines of target eigenenergies")
        splines = [
            train_utils.spline_eigenenergies( #Should spline_eigenenergies be in parser or trainutisl?
                frame, frame_idx, energies, reference, sigma=ml_options["DOS"]["sigma"], min_energy=ml_options["DOS_SPLINES"]["min_energy"], max_energy=ml_options["DOS_SPLINES"]["max_energy"], interval=ml_options["DOS_SPLINES"]["interval"]
            )
            for frame, frame_idx, energies, reference in zip(frames, frame_idxs, eigvals, energy_reference)
        ]
        torch.save(splines, join(ml_options["ML_DIR"], "splines.pt"))

    # First define subsets of system IDs for crossval
    io.log(log_path, "Split system IDs into subsets")
    all_subset_id = train_utils.crossval_idx_split(  # cross-validation split of idxs
        frame_idxs=frame_idxs,
        n_train=ml_options["N_TRAIN"], # Would it make more sense to use ratios for them/ provide support for ratios?
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
        shuffle=False, # Validation set should be computed all in one go for computational efficiency
        collate_fn=partial(group_and_join, join_kwargs={"remove_tensor_name": True})
    )
    # Define the learnable alignment that is used for the adaptive energy reference
    alignment = torch.nn.Parameter(
        torch.zeros(len(train_dset), dtype=ml_options["TRAIN"]["dtype"], device=ml_options["TRAIN"]["device"])
    )

    # Get optimizer and scheduler
    optimizer = torch.optim.Adam(list(model.parameters()) + [alignment], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = ml_options['SCHEDULER_ARGS']['gamma'],
                                                             patience = ml_options['SCHEDULER_ARGS']['patience'], 
                                                             threshold = ml_options['SCHEDULER_ARGS']['threshold'],
                                                             min_lr = ml_options['SCHEDULER_ARGS']['min_lr'])

    # Check for a previous checkpoint
    alignment_saved = os.path.exists(join(ml_options["CHKPT_DIR"], "alignment.pt"))
    optimizer_saved = os.path.exists(join(ml_options["CHKPT_DIR"], "optimizer.pt"))
    scheduler_saved = os.path.exists(join(ml_options["CHKPT_DIR"], "scheduler.pt"))
    model_saved = os.path.exists(join(ml_options["CHKPT_DIR"], "latest_model.pt"))
    best_state_saved = os.path.exists(join(ml_options["CHKPT_DIR"], "best_model.pt"))
    parameters_saved = os.path.exists(join(ml_options["CHKPT_DIR"], "parameters.pt"))
    

    if alignment_saved & optimizer_saved & scheduler_saved & model_saved & best_state_saved & parameters_saved:
        alignment = torch.load(os.path.exists(join(ml_options["CHKPT_DIR"], "alignment.pt")))
        optimizer.load_state_dict(torch.load(join(ml_options["CHKPT_DIR"], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(join(ml_options["CHKPT_DIR"], "scheduler.pt")))
        model.load_state_dict(torch.load(join(ml_options["CHKPT_DIR"], "latest_model.pt")))
        best_state = torch.load(ml_options["CHKPT_DIR"], "best_model.pt")
        best_train_loss, best_val_loss = torch.load(join(ml_options["CHKPT_DIR"], "parameters.pt"))
    else:
        best_state = copy.deepcopy(model.state_dict())
        best_train_loss = torch.tensor(100.0)
        best_val_loss = torch.tensor(100.0)

    dt_setup = time.time() - t0_setup
    io.log(log_path, train_utils.report_dt(dt_setup, "Setup complete"))

    # Initialize variables to track model performance during training
    

    # Start training loop
    for epoch in range(ml_options["TRAIN"]["n_epochs"]):

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
                spline_positions + normalized_alignment[list(batch.sample_id)].view(-1,1)
            )

            # Compute loss
            batch_loss = train_utils.t_get_mse(prediction, target, x_dos)
            batch_loss *= len(list(batch.sample_id))/ml_options["N_TRAIN"] # Is this the best way to find the size of the batch, this line is to normalize the mean so 
                                                                     # that the epoch loss is the train MSE

            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()
        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss

        # Validation step
        if epoch % ml_options['TRAIN']['val_interval'] == 0:
            # Performs validation step
            val_predictions = []
            for batch in val_loader:
                prediction = model(frames=batch.frames, descriptor=batch.descriptor)
                prediction = prediction.keys_to_samples(["center_type"])
                prediction = mts.mean_over_samples(prediction, "atom")
                prediction = prediction[0].values
                val_predictions.append(prediction)

            epoch_val_loss, _ = Opt_MSE_spline(torch.vstack(val_predictions), x_dos, torch.vstack(val_dset.splines), #Does val_dset work this way?
                                            spline_positions, n_epochs = 50)/ml_options["N_VAL"]
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state = copy.deepcopy(model.state_dict())


        scheduler.step(epoch_val_loss)
        if epoch % ml_options['TRAIN']['log_interval'] == 0:
            # Logs the current and best losses
            io.log(log_path, (f"Epoch {epoch}: Current loss (Train, Val): {epoch_train_loss.item():.4}, {epoch_val_loss.item():.4}"
                    f", Best loss (Train, Val): {best_train_loss.item():.4}, {best_val_loss.item():.4}")) #Does this way of logging make sense for training outputs

        if epoch % ml_options['TRAIN']['checkpoint_interval'] == 0:
            # Save relevant parameters
            torch.save(model.state_dict(), join(ml_options["CHKPT_DIR"], "latest_model.pt"))
            torch.save(alignment, join(ml_options["CHKPT_DIR"], "alignment.pt"))
            torch.save(opt.state_dict(), join(ml_options["CHKPT_DIR"], "optimizer.pt"))
            torch.save(scheduler.state_dict(), join(ml_options["CHKPT_DIR"], "scheduler.pt"))
            torch.save([best_train_loss, best_val_loss], join(ml_options["CHKPT_DIR"], "parameters.pt"))
            torch.save(best_state, join(ml_options["CHKPT_DIR"], "best_model.pt"))

        if scheduler.get_last_lr() <= ml_options['SCHEDULER_ARGS']['min_lr']:
            break
    
    # After training ends

    torch.save(model.state_dict(), join(ml_options["CHKPT_DIR"], "latest_model.pt"))
    torch.save(alignment, join(ml_options["CHKPT_DIR"], "alignment.pt"))
    torch.save(opt.state_dict(), join(ml_options["CHKPT_DIR"], "optimizer.pt"))
    torch.save(scheduler.state_dict(), join(ml_options["CHKPT_DIR"], "scheduler.pt"))
    torch.save([best_train_loss, best_val_loss], join(ml_options["CHKPT_DIR"], "parameters.pt"))
    torch.save(best_state, join(ml_options["CHKPT_DIR"], "best_model.pt"))

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