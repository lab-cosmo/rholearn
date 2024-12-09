import os
import time
from os.path import join
from typing import List

import metatensor.torch as mts
import torch

from rholearn.options import get_options
from rholearn.rholearn import train_utils
from rholearn.rholearn.descriptor import DescriptorCalculator
from rholearn.rholearn.loss import RhoLoss
from rholearn.rholearn.model import RhoModel
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

    _check_input_settings(dft_options, ml_options, frame_idxs)

    # ===== Split indices into train, val, test =====

    io.log(log_path, "Split system IDs into subsets")
    all_subset_id = train_utils.crossval_idx_split(  # cross-validation split of idxs
        frame_idxs=frame_idxs,
        n_train=ml_options["N_TRAIN"],
        n_val=ml_options["N_VAL"],
        n_test=ml_options["N_TEST"],
        seed=ml_options["SEED"],
    )
    if ml_options["OVERFIT"] is True:
        overfit_idx = all_subset_id[0][0]
        io.log(
            log_path,
            f"WARNING: performing overfitting experiment on structure {overfit_idx}",
        )
        all_subset_id[0] = [overfit_idx]
        all_subset_id[1] = [overfit_idx]
        all_subset_id[2] = [overfit_idx]
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

        if ml_options["LOAD_MODEL"]["path"] is None:  # initialize model from scratch

            io.log(log_path, "Initializing descriptor calculator")
            target_basis = convert.get_global_basis_set(
                [
                    io.unpickle_dict(
                        join(dft_options["PROCESSED_DIR"](A), "basis_set.pickle")
                    )
                    for A in frame_idxs
                ],
            )
            center_types = [
                i.item()
                for i in torch.unique(
                    torch.tensor(target_basis.column("center_type")), sorted=True
                )
            ]
            in_keys = train_utils.target_basis_set_to_in_keys(target_basis)
            descriptor_calculator = DescriptorCalculator(
                in_keys=in_keys,
                descriptor_hypers=ml_options["DESCRIPTOR_HYPERS"],
                center_types=center_types,
                device=torch.device(ml_options["TRAIN"]["device"]),
                dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
                **dft_options["MASK"],
            )

        else:  # Load model
            io.log(
                log_path,
                f"Loading existing model from path {ml_options['LOAD_MODEL']['path']}",
            )
            model = torch.load(ml_options["LOAD_MODEL"]["path"], weights_only=False)
            descriptor_calculator = model._descriptor_calculator
            if len(ml_options["LOAD_MODEL"]["add_modules"]) > 0:
                io.log(log_path, "Adding modules to the architecture")
                model._add_modules(ml_options["LOAD_MODEL"]["add_modules"])

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
            ),
            weights_only=False,
        )
        descriptor_calculator = model._descriptor_calculator

        # Load optimizer
        io.log(log_path, "Loading optimizer from checkpoint")
        optimizer = train_utils.get_optimizer(
            model, ml_options["OPTIMIZER"], ml_options["OPTIMIZER_ARGS"]
        )
        optimizer.load_state_dict(
            torch.load(
                join(
                    ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                    "optimizer_state_dict.pt",
                ),
                weights_only=False,
            )
        )

        # Load scheduler
        scheduler = None
        if ml_options["SCHEDULER_ARGS"] is not None:
            io.log(
                log_path,
                f"Loading scheduler from checkpoint: {ml_options['SCHEDULER']}",
            )
            scheduler = train_utils.get_scheduler(
                optimizer, ml_options["SCHEDULER"], ml_options["SCHEDULER_ARGS"]
            )
            scheduler.load_state_dict(
                torch.load(
                    join(
                        ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                        "scheduler_state_dict.pt",
                    ),
                    weights_only=False,
                )
            )

        # Load the validation loss
        best_val_loss = torch.load(
            join(
                ml_options["CHKPT_DIR"](ml_options["TRAIN"]["restart_epoch"]),
                "val_loss.pt",
            ),
            weights_only=False,
        )

    # ===== Initialize loss functions =====
    io.log(
        log_path,
        f"Initializing train loss fn with args: {ml_options['LOSS_FN']['train']}",
    )
    train_loss_fn = RhoLoss(**ml_options["LOSS_FN"]["train"])

    io.log(
        log_path, f"Initializing val loss fn with args: {ml_options['LOSS_FN']['val']}"
    )
    val_loss_fn = RhoLoss(**ml_options["LOSS_FN"]["val"])

    # ===== Create datasets and dataloaders =====

    # Dataset - train
    io.log(log_path, "Build training dataset")
    io.log(log_path, f"    Training system ID: {all_subset_id[0]}")
    with torch.no_grad():
        # if ml_options["DESCRIPTOR_HYPERS"]["use_pca"] is True:
        # Check fitting of each PCA module
        for attr in descriptor_calculator.__dir__():
            if attr.startswith("_epca"):
                _epca_attr = getattr(descriptor_calculator, attr)
                if _epca_attr is None:
                    continue
                if (
                    ml_options["TRAIN"]["restart_epoch"] is None
                    and ml_options["LOAD_MODEL"] is None
                ):
                    # If the model is newly initialized (i.e. not loaded) and PCA is
                    # to be used, ensure they are currently not fitted
                    assert _epca_attr._is_fitted is False, (
                        "PCAs should not already be fitted"
                    )
                else:
                    # The PCA should already be fitted
                    assert _epca_attr._is_fitted is True, (
                        "PCAs should not already be fitted"
                    )

        # Build training dataset
        train_dataset = train_utils.get_dataset(
            frames=[all_frames[A] for A in all_subset_id[0]],
            frame_idxs=list(all_subset_id[0]),
            descriptor_calculator=descriptor_calculator,
            loss_fn=train_loss_fn,
            load_dir=dft_options["PROCESSED_DIR"],
            overlap_cutoff=ml_options["OVERLAP_CUTOFF"],
            overlap_threshold=ml_options["OVERLAP_THRESHOLD"],
            device=torch.device(ml_options["TRAIN"]["device"]),
            dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
            log_path=log_path,
            vectors_to_sparse_by_center_type=False,
        )

        if (
            ml_options["TRAIN"]["restart_epoch"] is None
            and ml_options["LOAD_MODEL"] is None
            # and ml_options["DESCRIPTOR_HYPERS"]["use_pca"] is True
        ):
            # If newly initializing a model and using PCAs, check that they are now
            # fitted.
            for attr in descriptor_calculator.__dir__():
                if attr.startswith("_epca"):
                    _epca_attr = getattr(descriptor_calculator, attr)
                    if _epca_attr is None:
                        continue
                    assert _epca_attr._is_fitted is True, (
                        "PCAs should now be fitted"
                    )

    # Dataset - val
    io.log(log_path, "Build validation dataset")
    io.log(log_path, f"    Validation system ID: {all_subset_id[1]}")
    with torch.no_grad():
        val_dataset = train_utils.get_dataset(
            frames=[all_frames[A] for A in all_subset_id[1]],
            frame_idxs=list(all_subset_id[1]),
            descriptor_calculator=descriptor_calculator,
            loss_fn=val_loss_fn,
            load_dir=dft_options["PROCESSED_DIR"],
            overlap_cutoff=None,  # no cutoff for validation
            overlap_threshold=None,  # no threshold for validation
            device=torch.device(ml_options["TRAIN"]["device"]),
            dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
            log_path=log_path,
            vectors_to_sparse_by_center_type=False,
        )

    # Report mean sizes - train
    io.log(log_path, "Training data sizes (MB):")
    train_sizes = train_utils.get_mean_data_sizes(train_dataset)
    assert len(train_sizes) != 0
    for name, size in train_sizes.items():
        io.log(log_path, f"    {name}: {size:.3f}")

    # Report mean sizes - val
    io.log(log_path, "Validation data sizes (MB):")
    val_sizes = train_utils.get_mean_data_sizes(val_dataset)
    assert len(val_sizes) != 0
    for name, size in val_sizes.items():
        io.log(log_path, f"    {name}: {size:.3f}")

    # Dataloader - train
    join_kwargs = {
        "remove_tensor_name": True,
        "different_keys": "union",
    }
    io.log(log_path, "Build training dataloader")
    train_dloader = train_utils.get_dataloader(
        dataset=train_dataset,
        join_kwargs=join_kwargs,
        dloader_kwargs={
            "batch_size": (
                len(train_dataset)
                if ml_options["OPTIMIZER"] == "LBFGS"
                else ml_options["TRAIN"].get("batch_size")
            ),
            "shuffle": True,
        },
    )

    # Dataloader - val
    io.log(log_path, "Build validation dataloader")
    val_dloader = train_utils.get_dataloader(
        dataset=val_dataset,
        join_kwargs=join_kwargs,
        dloader_kwargs={
            "batch_size": len(val_dataset),
            "shuffle": False,
        },
    )

    # Initialize the model (if not loading it). This needs to happen after construction
    # of the training dataset in case PCA fitting occurred.
    if ml_options["TRAIN"]["restart_epoch"] is None:
        if ml_options["LOAD_MODEL"]["path"] is None:
            io.log(log_path, "Initializing model")
            in_properties = train_utils.center_types_to_descriptor_basis_in_properties(
                in_keys, descriptor_calculator
            )
            out_properties = train_utils.target_basis_set_to_out_properties(
                in_keys,
                target_basis,
            )
            model = RhoModel(
                in_keys=in_keys,
                in_properties=in_properties,
                out_properties=out_properties,
                descriptor_calculator=descriptor_calculator,
                architecture=ml_options["ARCHITECTURE"],
                target_basis=target_basis,
                dtype=getattr(torch, ml_options["TRAIN"]["dtype"]),
                device=torch.device(ml_options["TRAIN"]["device"]),
                pretrain=ml_options["PRETRAIN"],
                pretrain_args=ml_options["PRETRAIN_ARGS"],
                **dft_options["MASK"],
            )

        # Initialize the best validation loss
        best_val_loss = torch.tensor(float("inf"))

    # Log model architecture
    io.log(log_path, str(model).replace("\n", f"\n# {utils.timestamp()}"))

    # Collate and store all of the training descriptors and targets for use in
    # pre-training and target standardization.
    all_train_descriptor = mts.join(
        [sample.descriptor for sample in train_dataset], "samples", **join_kwargs
    )
    all_train_target_c = mts.join(
        [batch.target_c for batch in train_dataset], "samples", **join_kwargs
    )
    all_val_descriptor = mts.join(
        [batch.descriptor for batch in val_dataset], "samples", **join_kwargs
    )
    all_val_target_c = mts.join(
        [batch.target_c for batch in val_dataset], "samples", **join_kwargs
    )

    # For loss evaluation, the target coefficients need to be converted from being block
    # sparse in angular order and species type to just block sparse in species type.
    # Modify the data in the training and validation datasets
    train_dataset._data["target_c"] = [
        convert.coeff_vector_to_sparse_by_center_type(c, "torch")
        for c in train_dataset._data["target_c"]
    ]
    val_dataset._data["target_c"] = [
        convert.coeff_vector_to_sparse_by_center_type(c, "torch")
        for c in val_dataset._data["target_c"]
    ]

    # Compute and store a target standardizer, if applicable
    if ml_options["STANDARDIZE_TARGETS"]:
        io.log(log_path, "Computing target standardizer using training data")
        standardizer = train_utils.get_tensor_std(all_train_target_c)
        model._set_standardizer(standardizer)

    # Compute the validation loss before (pre)training
    with torch.no_grad():
        loaded_val_loss = train_utils.epoch_step(
            dataloader=val_dloader,
            model=model,
            loss_fn=val_loss_fn,
            optimizer=None,
            check_metadata=True,
        )
        if loaded_val_loss < best_val_loss:
            best_val_loss = loaded_val_loss
    io.log(
        log_path, f"Validation loss of model before (pre)training: {loaded_val_loss}"
    )

    # Finish setup
    dt_setup = time.time() - t0_setup
    io.log(log_path, train_utils.report_dt(dt_setup, "Setup complete"))

    # ===== PRETRAINING =====

    # Pretrain the model weights with an analytical model, if applicable
    if ml_options["PRETRAIN"]:
        io.log(log_path, "Pretraining linear layer with analytical model")
        io.log(log_path, f"  Pretrain args: {ml_options['PRETRAIN_ARGS']}")
        t0_pretrain = time.time()

        # Standardize the targets, if applicable
        if ml_options["STANDARDIZE_TARGETS"]:
            all_train_target_c = train_utils.standardize_tensor(
                all_train_target_c,
                standardizer,
            )
            all_val_target_c = train_utils.standardize_tensor(
                all_val_target_c,
                standardizer,
            )

        # Fit the pretrainer on the training subset
        io.log(log_path, "    Fitting pretrainer")
        model._pretrainer.fit(X=all_train_descriptor, Y=all_train_target_c)

        # Validate the pretrainer on the validation data
        io.log(log_path, "    Validating pretrainer")
        model._pretrainer.validate(X=all_val_descriptor, Y=all_val_target_c)

        # Write validation block losses to file (and regularizer if using RidgeCV)
        io.log(log_path, "Pretrainer validation losses:")
        if ml_options["PRETRAIN_ARGS"].get("alphas") is None:
            for key, pretrain_val_loss in zip(
                model._in_keys,
                model._pretrainer._best_val_losses,
            ):
                io.log(log_path, f"  key {key} pretrain_val_loss {pretrain_val_loss}")

        else:
            for key, pretrain_val_loss, alpha in zip(
                model._in_keys,
                model._pretrainer._best_val_losses,
                model._pretrainer._best_alphas,
            ):
                io.log(
                    log_path,
                    f"  key {key} pretrain_val_loss {pretrain_val_loss} alpha {alpha}",
                )

        # Initialize the weights using the pretrainer
        model._initialize_weights_from_pretrainer()

        # Do a single validation step to sanity check the validation loss
        with torch.no_grad():
            pretrained_val_loss = train_utils.epoch_step(
                dataloader=val_dloader,
                model=model,
                loss_fn=val_loss_fn,
                optimizer=None,
                check_metadata=True,
            )
            if pretrained_val_loss < best_val_loss:
                best_val_loss = pretrained_val_loss
        io.log(log_path, f"Validation loss of pretrained model: {pretrained_val_loss}")

        # Save a checkpoint of the pretrained model
        train_utils.save_checkpoint(
            model,
            optimizer=None,
            scheduler=None,
            val_loss=pretrained_val_loss,
            chkpt_dir=ml_options["CHKPT_DIR"]("pretrain"),
        )

        # Also save the results
        if ml_options["PRETRAIN_ARGS"].get("alphas") is None:
            torch.save(
                torch.tensor(model._pretrainer._best_val_losses),
                join(ml_options["ML_DIR"], "outputs", "pretrainer_val_losses.pt"),
            )
        else:
            torch.save(
                torch.tensor(model._pretrainer._all_val_losses),
                join(ml_options["ML_DIR"], "outputs", "pretrainer_val_losses.pt"),
            )
            torch.save(
                torch.tensor(model._pretrainer._alphas),
                join(ml_options["ML_DIR"], "outputs", "pretrainer_alphas.pt"),
            )

        # Also save prediction and target
        # mts.save("d.npz", all_train_descriptor)
        pretrain_val_input_c = model._pretrainer.predict(all_val_descriptor)
        if ml_options["STANDARDIZE_TARGETS"]:
            mts.save(
                "pretrain_input_c.npz",
                train_utils.unstandardize_tensor(pretrain_val_input_c, standardizer),
            )
            mts.save(
                "target_c.npz",
                train_utils.unstandardize_tensor(all_val_target_c, standardizer),
            )
        else:
            mts.save("pretrain_input_c.npz", pretrain_val_input_c)
            mts.save("target_c.npz", all_val_target_c)

        # Finish pretrain
        dt_pretrain = time.time() - t0_pretrain
        io.log(
            log_path,
            train_utils.report_dt(dt_pretrain, "Analytical pretraining complete"),
        )

    # Try a model save/load
    torch.save(model, join(ml_options["ML_DIR"], "model.pt"))
    torch.load(join(ml_options["ML_DIR"], "model.pt"), weights_only=False)
    os.remove(join(ml_options["ML_DIR"], "model.pt"))

    if len(epochs) == 0:
        io.log(log_path, "No gradient descent to run - training finished.")
        return

    # ===== Training loop =====

    # Initialize the optimizer and scheduler (if not loading them)
    if ml_options["TRAIN"]["restart_epoch"] is None:

        # Initialize optimizer
        io.log(log_path, "Initializing optimizer")
        optimizer = train_utils.get_optimizer(
            model, ml_options["OPTIMIZER"], ml_options["OPTIMIZER_ARGS"]
        )

        # Initialize scheduler
        scheduler = None
        if ml_options["SCHEDULER_ARGS"] is not None:
            io.log(log_path, f"Using LR scheduler: {ml_options['SCHEDULER']}")
            scheduler = train_utils.get_scheduler(
                optimizer, ml_options["SCHEDULER"], ml_options["SCHEDULER_ARGS"]
            )

    # Freeze some parts of the architecture, if applicable
    if len(ml_options["FREEZE_MODULES"]) > 0:
        io.log(log_path, "Frezzing specified parameter groups")
        for map_name, freeze in ml_options["FREEZE_MODULES"].items():
            if isinstance(freeze, bool):
                model._set_requires_grad(map_name, not freeze)
            else:
                # Create a labels object for freezing parameters that match a specific
                # key selection
                assert "names" in freeze.keys() and "values" in freeze.keys(), (
                    "Must pass key selection 'names' and 'values'"
                    " for freezing parameters."
                )
                selected_keys = mts.Labels(
                    names=freeze["names"],
                    values=torch.tensor(freeze["values"], dtype=torch.int64).reshape(
                        -1, len(freeze["names"])
                    ),
                )
                model._set_requires_grad(map_name, False, selected_keys)

    io.log(
        log_path,
        "Start training by gradient descent over"
        f" epochs {epochs[0]} -> {epochs[-1]} (inclusive)",
    )

    t0_training = time.time()
    for epoch in epochs:

        # Run training step at every epoch
        t0_train = time.time()
        train_loss = train_utils.epoch_step(
            dataloader=train_dloader,
            model=model,
            loss_fn=train_loss_fn,
            optimizer=optimizer,
            check_metadata=epoch == 1,
        )
        dt_train = time.time() - t0_train

        # Run validation step on this epoch
        val_loss = torch.nan
        dt_val = torch.nan
        lr = torch.nan
        if epoch % ml_options["TRAIN"]["val_interval"] == 0 or epoch == 1:
            # a) using target_c only
            t0_val = time.time()
            with torch.no_grad():
                val_loss = train_utils.epoch_step(
                    dataloader=val_dloader,
                    model=model,
                    loss_fn=val_loss_fn,
                    optimizer=None,
                    check_metadata=epoch == 1,
                )
            dt_val = time.time() - t0_val

            # Step scheduler based on validation loss via c
            if scheduler is not None:
                if ml_options["SCHEDULER"] == "ReduceLROnPlateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                lr = scheduler._last_lr[0]

        # Log results on this epoch
        if epoch % ml_options["TRAIN"]["log_interval"] == 0 or epoch == 1:
            log_msg = (
                f"epoch {epoch}"
                f" train_loss {train_loss}"
                f" val_loss {val_loss}"
                f" dt_train {dt_train:.3f}"
                f" dt_val {dt_val:.3f}"
                f" lr {lr}"
            )
            io.log(log_path, log_msg)
            # Log parameter gradient norms if applicable
            if ml_options["TRAIN"]["log_grad_norms"] is not None:
                if epoch % ml_options["TRAIN"]["log_grad_norms"] == 0 or epoch == 1:
                    io.log(log_path, "Gradient norms:")
                    grad_norms = model._get_grad_norms()
                    for map_name, vals in grad_norms.items():
                        io.log(log_path, f"  {map_name}:")
                        for key, norms in vals.items():
                            io.log(log_path, f"    {key}: {norms}")

        # Checkpoint on this epoch
        if epoch % ml_options["TRAIN"]["checkpoint_interval"] == 0:
            train_utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                val_loss=val_loss,
                chkpt_dir=ml_options["CHKPT_DIR"](epoch),
            )

        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            train_utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                val_loss=val_loss,
                chkpt_dir=ml_options["CHKPT_DIR"]("best"),
            )

        # Early stopping
        if lr <= ml_options["MIN_LR"]:
            io.log(log_path, "Early stopping")
            break

    # Finish
    dt_train = time.time() - t0_training
    io.log(log_path, train_utils.report_dt(dt_train, "Training complete"))
    io.log(log_path, train_utils.report_dt(dt_train / epoch, "   or per epoch"))
    io.log(log_path, f"Best validation loss: {best_val_loss:.10f}")


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
    if ml_options["OPTIMIZER"] == "LBFGS":
        assert (
            ml_options["TRAIN"].get("batch_size") is None
        ), "Cannot use minibatching with LBFGS"
    else:
        assert (
            ml_options["TRAIN"].get("batch_size") is not None
        ), "Must supply a TRAIN['batch_size'] option."
