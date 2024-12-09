"""
Module containing functions to perform model training and evaluation steps.
"""

import os
import random
from functools import partial
from os.path import exists, join
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import metatensor.torch as mts
import numpy as np
import torch
from chemfiles import Atom, Frame, UnitCell
from metatensor.torch.learn.data import DataLoader, IndexedDataset
from metatensor.torch.learn.data._namedtuple import namedtuple

from rholearn.rholearn import mask
from rholearn.utils import convert, io, system


def create_subdir(ml_dir: str, name: str):
    """
    Creates a subdirectory at relative path:and returns path to training subdirectory at
    relative path:

        f"{`ml_dir`}/{`name`}"

    and returns a callable that points to further subdirectories indexed by the epoch
    number, i.e.:

        f"{`ml_dir`}/{`name`}/epoch_{`epoch`}"

    where the callable is parametrized by the variable `epoch`. This is used for
    creating checkpoint and evaluation directories.
    """

    def subdir(epoch):
        return join(ml_dir, name, f"epoch_{epoch}")

    if not exists(join(ml_dir, name)):
        os.makedirs(join(ml_dir, name))

    return subdir


def load_tensormap_to_torch(
    path: str,
    dtype: Optional[torch.dtype],
    device: Optional[str],
) -> torch.ScriptObject:
    """Loads a TensorMap from file and converts its backend to torch."""
    return mts.load(path).to(dtype=dtype, device=device)


def load_and_process_overlap(
    frame_idx: int,
    frame: Frame,
    load_dir: callable,
    descriptor_calculator: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dtype: torch.dtype,
    device: str,
    log_path: str,
    overlap_cutoff: Optional[float],
    overlap_threshold: Optional[float],
    save_dir: Optional[callable] = None,
) -> None:
    """
    Loads the overlap from disk and pre-processes it in various ways, according to input
    settings:

        - Neighbor list cutoff
        - Conditioning
        - Sparsifying by removing overlaps smaller than a threshold
        - Masking by atom types
    """
    # Load overlap from disk
    overlap = load_tensormap_to_torch(
        join(load_dir(frame_idx), "ri_ovlp.npz"), dtype=dtype, device=device
    )

    # Drop empty blocks that may be present
    overlap = mask._drop_empty_blocks(overlap, "torch")

    # Apply a cutoff to the overlap matrices if applicable
    if overlap_cutoff is not None:
        overlap = mask.cutoff_overlap_matrix(
            [frame],
            [frame_idx],
            overlap_matrix=overlap,
            cutoff=overlap_cutoff,
            drop_empty_blocks=True,
            backend="torch",
        )

    # Condition overlaps if applicable
    if loss_fn._conditioner is not None:
        overlap = condition_overlap(overlap, loss_fn._conditioner)

    # Remove overlaps that are below a certain threshold value, if applicable
    if overlap_threshold is not None:
        overlap = mask.sparsify_overlap_matrix(
            overlap_matrix=overlap,
            sparsity_threshold=overlap_threshold,
            drop_empty_blocks=True,
            backend="torch",
        )

    # Mask overlaps
    if descriptor_calculator._masked_system_type is not None:
        atom_idxs_to_keep = list(
            mask.get_point_indices_by_region(
                points=frame.positions,
                masked_system_type=descriptor_calculator._masked_system_type,
                region="active",
                **descriptor_calculator._mask_kwargs,
            )
        ) + list(
            mask.get_point_indices_by_region(
                points=frame.positions,
                masked_system_type=descriptor_calculator._masked_system_type,
                region="buffer",
                **descriptor_calculator._mask_kwargs,
            )
        )
        overlap = mask.mask_overlap_matrix(
            overlap,
            atom_idxs_to_keep=atom_idxs_to_keep,
            drop_empty_blocks=True,
            backend="torch",
        )
        # And remove buffer-buffer overlaps
        # overlap = [
        #     mask._drop_non_active_overlaps(
        #         frame,
        #         descriptor_calculator._masked_system_type,
        #         ovlp,
        #         "torch",
        #         **descriptor_calculator._mask_kwargs,
        #     )
        #     for frame, ovlp in zip(frames, overlap)
        # ]

    # Return a copy so that any sliced dimensions are released from memory
    return overlap.copy()


def condition_overlap(
    overlap_matrix: mts.TensorMap, conditioner: float
) -> mts.TensorMap:
    """
    Conditions the overlap matrix by adding the identity matirx, scaled by the specified
    ``conditioner``.
    """
    for (a1, a2), block in overlap_matrix.items():
        if a1 != a2:
            continue

        assert block.values.shape[1] == block.values.shape[2]
        dim = block.values.shape[1]

        for sample_i, sample in enumerate(block.samples):
            if sample["atom_1"] != sample["atom_2"]:
                continue

            block.values[sample_i] += torch.eye(dim) * conditioner

    return overlap_matrix


def get_tensor_std(tensor: mts.TensorMap) -> mts.TensorMap:
    """
    For each block, computes the norm over components and then a standard deviation over
    samples. Scales this by the length of the ISC (i.e. 2l + 1). Stores the resulting
    vector of length `len(block.properties)` in a TensorBlock, and returns the total
    result in a TensorMap.
    """
    std_blocks = []
    for key, block in tensor.items():

        std_values = torch.std(torch.norm(block.values, dim=1), dim=0) * (
            2 * key["o3_lambda"] + 1
        )
        std_blocks.append(
            mts.TensorBlock(
                samples=mts.Labels.single(),
                components=[],
                properties=block.properties,
                values=std_values.reshape(1, -1),
            )
        )

    return mts.TensorMap(tensor.keys, std_blocks)


def standardize_tensor(
    tensor: mts.TensorMap, standardizer: mts.TensorMap
) -> mts.TensorMap:
    """
    Standardizes the input ``tensor`` using the ``standardizer`` layer.
    """
    for key, block in tensor.items():
        block.values[:] /= standardizer.block(key).values

    return tensor


def unstandardize_tensor(
    tensor: mts.TensorMap, standardizer: mts.TensorMap
) -> mts.TensorMap:
    """
    Standardizes the input ``tensor`` using the ``standardizer`` layer.
    """
    for key, block in tensor.items():
        block.values[:] *= standardizer.block(key).values

    return tensor


def target_basis_set_to_in_keys(target_basis: mts.Labels) -> mts.Labels:
    """
    Converts the basis set definition to the set of in_keys on which the model is
    defined.

    Returned is a Labels object with the names "o3_lambda" and "center_type" and values,
    extracted from `target_basis`
    """
    assert target_basis.names[:2] == ["o3_lambda", "center_type"]
    return mts.Labels(
        names=["o3_lambda", "o3_sigma", "center_type"],
        values=torch.tensor(
            [
                [o3_lambda, 1, center_type]
                for o3_lambda, center_type in target_basis.values[:, :2]
            ],
            dtype=torch.int32,
        ).reshape(-1, 3),
    )


def center_types_to_descriptor_basis_in_properties(
    in_keys: mts.Labels,
    descriptor_calculator: torch.nn.Module,
) -> List[mts.Labels]:
    """
    Builds a dummy :py:class:`chemfiles.Frame` object from the global atom types in
    `descriptor_calculator._center_types`, computes the descriptor for it, and extracts
    the properties for each block indexed by the keys in `in_keys`.

    For masked systems, the neighbor types are just the standard atom types, i.e. 1 and
    6 for H and C, not 1006 for pseudo-C.
    """
    dummy_frames = []
    for center_type in descriptor_calculator._center_types:
        dummy_frame = Frame()
        dummy_frame.add_atom(
            Atom(
                name=system.atomic_number_to_atomic_symbol(int(center_type)),
                type=system.atomic_number_to_atomic_symbol(int(center_type)),
            ),
            position=[0, 0, 0],
        )
        dummy_frame.cell = UnitCell([100, 100, 100])
        dummy_frames.append(dummy_frame)
    descriptor = descriptor_calculator(
        frames=dummy_frames,
        mask_system=False,
    )

    return [descriptor[key].properties for key in in_keys]


def target_basis_set_to_out_properties(
    in_keys: mts.Labels,
    target_basis: mts.Labels,
) -> List[mts.Labels]:
    """
    Converts the basis set definition to a list of Labels objects corresponding to the
    out properties for each key in `in_keys`.

    Returned is a list of Labels, each of which enumerate the radial channels "n" for
    each combination of o3_lambda and center_type in `in_keys`, extracted from
    `target_basis`
    """
    out_properties = []
    for key in in_keys:
        o3_lambda, _, center_type = key

        nmax_idxs = torch.all(
            target_basis.values[:, :2] == torch.tensor([[o3_lambda, center_type]]),
            dim=1,
        )
        assert torch.sum(nmax_idxs) == 1
        nmax = target_basis.values[nmax_idxs, 2].item()

        out_properties.append(
            mts.Labels(
                names=["n"],
                values=torch.arange(nmax).reshape(-1, 1),
            )
        )
    return out_properties


def get_dataset(
    frames: List[Frame],
    frame_idxs: List[int],
    descriptor_calculator: torch.nn.Module,
    loss_fn: torch.nn.Module,
    load_dir: Callable,
    overlap_cutoff: Optional[float],
    overlap_threshold: Optional[float],
    dtype: Optional[torch.dtype],
    device: Optional[str],
    log_path: str,
    vectors_to_sparse_by_center_type: bool = False,
) -> torch.nn.Module:
    """
    Builds a dataset for the given ``frames``, using the ``model`` to pre-compute and
    store decriptors. Loads all the required target data.
    """
    if not isinstance(frame_idxs, list):
        frame_idxs = list(frame_idxs)

    if isinstance(overlap_cutoff, str):
        overlap_cutoff = float(overlap_cutoff)

    if isinstance(overlap_threshold, str):
        overlap_threshold = float(overlap_threshold)

    # Descriptors
    io.log(log_path, "    Computing descriptors")
    descriptor = descriptor_calculator(
        frames=frames,
        frame_idxs=frame_idxs,
        mask_system=descriptor_calculator._masked_system_type is not None,
        reindex=True,
        split_by_frame=True,
    )

    # Coefficients
    if "target_c" in loss_fn._required_data:
        # Load
        io.log(log_path, "    Loading target coeffs")
        target_c = [
            load_tensormap_to_torch(
                join(load_dir(A), "ri_coeffs.npz"), dtype=dtype, device=device
            )
            for A in frame_idxs
        ]
        # Convert to block sparse only in "center_type" for loss evaluation
        if vectors_to_sparse_by_center_type:
            target_c = [
                convert.coeff_vector_to_sparse_by_center_type(c, "torch")
                for c in target_c
            ]
    else:
        target_c = [None] * len(frame_idxs)

    # Projections
    if "target_w" in loss_fn._required_data:
        # Load
        io.log(log_path, "    Loading target projs")
        target_w = [
            load_tensormap_to_torch(
                join(load_dir(A), "ri_projs.npz"), dtype=dtype, device=device
            )
            for A in frame_idxs
        ]
        # Convert to block sparse only in "center_type" for loss evaluation
        target_w = [
            convert.coeff_vector_to_sparse_by_center_type(w, "torch") for w in target_w
        ]
    else:
        target_w = [None] * len(frame_idxs)

    # Overlaps
    if "overlap" in loss_fn._required_data:
        io.log(log_path, "    Loading and processing overlaps")
        if overlap_cutoff is not None:
            io.log(log_path, f"    Cutting off overlaps at {overlap_cutoff} Ang")
        if loss_fn._conditioner is not None:
            io.log(
                log_path,
                f"    Conditioning overlaps with conditioner: {loss_fn._conditioner}",
            )
        if overlap_threshold is not None:
            io.log(
                log_path,
                f"    Sparsifying overlaps with threshold: {overlap_threshold}",
            )
        if descriptor_calculator._masked_system_type is not None:
            io.log(log_path, "    Masking overlaps")
        overlap = [
            load_and_process_overlap(
                A,
                frame,
                load_dir,
                descriptor_calculator,
                loss_fn,
                dtype,
                device,
                log_path,
                overlap_cutoff,
                overlap_threshold,
                save_dir=None,
            )
            for A, frame in zip(frame_idxs, frames)
        ]

    else:
        overlap = [None] * len(frame_idxs)

    # Now mask all tensors if required, keeping active and buffer region atoms.
    # Descriptors will already be masked. Overlaps are masked in function
    # `load_and_process_overlap`.
    if descriptor_calculator._masked_system_type is not None:
        atom_idxs_to_keep = [
            list(
                mask.get_point_indices_by_region(
                    points=frame.positions,
                    masked_system_type=descriptor_calculator._masked_system_type,
                    region="active",
                    **descriptor_calculator._mask_kwargs,
                )
            )
            + list(
                mask.get_point_indices_by_region(
                    points=frame.positions,
                    masked_system_type=descriptor_calculator._masked_system_type,
                    region="buffer",
                    **descriptor_calculator._mask_kwargs,
                )
            )
            for frame in frames
        ]
        # Target coefficients
        if target_c[0] is not None:
            io.log(log_path, "    Masking target coeffs")
            target_c = [
                mask.mask_coeff_vector(
                    coeff_vector=targ_c,
                    atom_idxs_to_keep=idxs,
                    drop_empty_blocks=False,
                    backend="torch",
                )
                for targ_c, idxs in zip(target_c, atom_idxs_to_keep)
            ]

        # Target projections
        if target_w[0] is not None:
            io.log(log_path, "    Masking target projs")
            target_w = [
                mask.mask_coeff_vector(
                    coeff_vector=targ_w,
                    atom_idxs_to_keep=idxs,
                    drop_empty_blocks=False,
                    backend="torch",
                )
                for targ_w, idxs in zip(target_w, atom_idxs_to_keep)
            ]

    # Finally construct the dataset
    dataset = IndexedDataset(
        sample_id=frame_idxs,
        frame=frames,
        descriptor=descriptor,
        target_c=target_c,
        target_w=target_w,
        overlap=overlap,
    )

    return dataset


def get_mean_data_sizes(
    dataset: IndexedDataset,
    n_samples: int = 20,
) -> Tuple[float]:
    """
    Returns the mean size in memory of each data field present in ``dataset``, in MB.
    """
    if n_samples > len(dataset):
        n_samples = len(dataset)

    sizes = {}
    for i in random.sample(range(len(dataset)), n_samples):
        sample = dataset[i]
        for field_name in dataset._field_names:

            # Only get sizes of TensorMaps
            tensor = getattr(sample, field_name)
            if not hasattr(tensor, "keys_to_properties"):  # not a TensorMap!
                continue

            if field_name not in sizes:
                sizes[field_name] = []

            sizes[field_name].append(get_size_of_tensormap(tensor))

    # Mean results
    for field_name in sizes:
        sizes[field_name] = np.mean(sizes[field_name])

    return sizes


def get_size_of_tensormap(tensor: mts.TensorMap) -> float:
    """
    Returns the size (in MB) of a TensorMap, as a sum of the size in memory of each
    block values tensor.

    Note: assumes no gradients.
    """
    size = 0
    for block in tensor:
        size += block.values.element_size() * block.values.nelement() * 1e-6

    return size


def group_and_join_nonetypes(
    batch: List[NamedTuple],
    fields_to_join: Optional[List[str]] = None,
    join_kwargs: Optional[dict] = None,
) -> NamedTuple:
    """
    A modified form of :py:meth:`metatensor.torch.learn.data.group_and_join` that
    handles data fields that are NoneType. Any fields that are a list of ``None`` are
    'joined' to a single ``None``. All other functionality is the same, but

    This is useful for passing data straight to the :py:class:`rholearn.loss.RhoLoss`
    class.
    """
    data: List[Union[mts.TensorMap, torch.Tensor]] = []
    names = batch[0]._fields
    if fields_to_join is None:
        fields_to_join = names
    if join_kwargs is None:
        join_kwargs = {}
    for name, field in zip(names, list(zip(*batch))):

        if name == "sample_id":  # special case, keep as is
            data.append(field)
            continue

        if name in fields_to_join:  # Join tensors if requested
            if isinstance(field[0], torch.ScriptObject) and field[0]._has_method(
                "keys_to_properties"
            ):  # inferred metatensor.torch.TensorMap type
                data.append(mts.join(field, axis="samples", **join_kwargs))
            elif isinstance(field[0], torch.Tensor):  # torch.Tensor type
                data.append(torch.vstack(field))
            elif isinstance(field[0], type(None)):  # NoneType
                data.append(None)
            else:
                data.append(field)

        else:  # otherwise just keep as a list
            data.append(field)

    return namedtuple("Batch", names)(*data)


def get_dataloader(
    dataset: torch.nn.Module,
    join_kwargs: Optional[dict] = None,
    dloader_kwargs: Optional[dict] = None,
) -> torch.nn.Module:
    """
    Builds the training dataloader.

    ``join_kwargs`` are used in the collate fn for joining TensorMaps, i.e.
    "remove_tensor_name" or "different_keys". ``dloader_kwargs`` are used in the
    DataLoader, i.e. "batch_size".
    """
    if join_kwargs is None:
        join_kwargs = {"remove_tensor_name": True, "different_keys": "union"}
    if dloader_kwargs is None:
        dloader_kwargs = {}
    collate_fn = partial(
        group_and_join_nonetypes,
        join_kwargs=join_kwargs,
    )

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        **dloader_kwargs,  # i.e. batch_size
    )


def get_optimizer(model: torch.nn.Module, name: str, args: dict):
    """Gets the optimizer by name"""
    if name == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            **args,
        )
    elif name == "LBFGS":
        optimizer = torch.optim.LBFGS(
            filter(lambda p: p.requires_grad, model.parameters()),
            **args,
        )
    else:
        raise ValueError(f"unsupported optimizer: {name}")

    return optimizer


def get_scheduler(optimizer, name: str, args: dict):
    """Gets the scheduler by name"""
    if name is None:
        return None
    if name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **args)
    elif name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **args)
    else:
        raise ValueError(f"unsupported scheduler: {name}")

    return scheduler


def epoch_step(
    dataloader,
    model,
    loss_fn,
    optimizer=None,
    check_metadata: bool = True,
) -> Tuple[torch.Tensor]:
    """
    Performs a single epoch of training / validation by minibatching. Returns the loss
    for the epoch, normalized by the total number of samples across all minibatches.

    Assumes the model is already in training mode, or within a torch.no_grad() context
    if not training. If ``optimizer`` is None, this function does not zero optimizer
    gradiants, perform a backward pass, or update the model parameters.
    """
    loss_epoch = 0
    n_samples_epoch = 0
    for batch in dataloader:

        if optimizer is None:  # i.e. for validation

            input_c = model(  # forward pass
                frames=list(batch.frame),
                frame_idxs=batch.sample_id,
                descriptor=batch.descriptor,
                split_by_frame=False,
                check_metadata=check_metadata,
            )
            input_c = convert.coeff_vector_to_sparse_by_center_type(input_c, "torch")
            batch_loss = loss_fn(  # compute loss
                input_c=mask._drop_empty_blocks(input_c, "torch"),
                target_c=mask._drop_empty_blocks(batch.target_c, "torch"),
                target_w=mask._drop_empty_blocks(batch.target_w, "torch"),
                overlap=batch.overlap,
                check_metadata=check_metadata,
            )
            loss_epoch += batch_loss  # store loss
            n_samples_epoch += len(batch.sample_id)  # store num frames

        else:

            # If using LBFGS, the epoch step must be treated specially
            if isinstance(optimizer, torch.optim.LBFGS):

                def closure():
                    optimizer.zero_grad()
                    input_c = model(  # forward pass
                        frames=list(batch.frame),  # noqa: B023
                        frame_idxs=batch.sample_id,  # noqa: B023
                        descriptor=batch.descriptor,  # noqa: B023
                        split_by_frame=False,
                        check_metadata=check_metadata,
                    )
                    input_c = convert.coeff_vector_to_sparse_by_center_type(
                        input_c, "torch"
                    )
                    batch_loss = loss_fn(  # compute loss
                        input_c=mask._drop_empty_blocks(input_c, "torch"),
                        target_c=mask._drop_empty_blocks(batch.target_c, "torch"),  # noqa: B023
                        target_w=mask._drop_empty_blocks(batch.target_w, "torch"),  # noqa: B023
                        overlap=batch.overlap,  # noqa: B023
                        check_metadata=check_metadata,
                    )
                    batch_loss.backward()  # backward pass
                    return batch_loss

                batch_loss = optimizer.step(closure)
                loss_epoch += batch_loss  # store loss
                n_samples_epoch += len(batch.sample_id)  # store num frames

            # Standard torch GD-based optimizer
            else:
                optimizer.zero_grad()  # zero grads
                input_c = model(  # forward pass
                    frames=list(batch.frame),
                    frame_idxs=batch.sample_id,
                    descriptor=batch.descriptor,
                    split_by_frame=False,
                    check_metadata=check_metadata,
                )
                input_c = convert.coeff_vector_to_sparse_by_center_type(
                    input_c, "torch"
                )
                batch_loss = loss_fn(  # compute loss
                    input_c=mask._drop_empty_blocks(input_c, "torch"),
                    target_c=mask._drop_empty_blocks(batch.target_c, "torch"),
                    target_w=mask._drop_empty_blocks(batch.target_w, "torch"),
                    overlap=batch.overlap,
                    check_metadata=check_metadata,
                )
                batch_loss.backward()  # backward pass
                optimizer.step()  # update parameters
                loss_epoch += batch_loss  # store loss
                n_samples_epoch += len(batch.sample_id)  # store num frames

    return loss_epoch / n_samples_epoch


def save_checkpoint(
    model: torch.nn.Module, optimizer, scheduler, val_loss: torch.Tensor, chkpt_dir: str
):
    """
    Saves model object, model state dict, optimizer state dict, scheduler state dict,
    to file.
    """
    if not exists(chkpt_dir):  # create chkpoint dir
        os.makedirs(chkpt_dir)

    torch.save(model, join(chkpt_dir, "model.pt"))  # model obj
    torch.save(  # model state dict
        model.state_dict(),
        join(chkpt_dir, "model_state_dict.pt"),
    )

    # Optimizer and scheduler
    if optimizer is not None:
        torch.save(optimizer.state_dict(), join(chkpt_dir, "optimizer_state_dict.pt"))
    if scheduler is not None:
        torch.save(
            scheduler.state_dict(),
            join(chkpt_dir, "scheduler_state_dict.pt"),
        )

    # Save the validation loss
    torch.save(val_loss, join(chkpt_dir, "val_loss.pt"))


def report_dt(dt: float, message: str):
    """
    Returns a `message` and time delta `dt` in seconds, minutes or hours.
    """
    if dt <= 60:
        dt = dt
        time_unit = "seconds"
    elif 60 < dt <= 3600:
        dt = dt / 60
        time_unit = "minutes"
    else:
        dt = dt / 3600
        time_unit = "hours"

    return f"{message} in {dt} {time_unit}."


# ===== Fxns for creating groups of indices for train/test/validation splits


def crossval_idx_split(
    frame_idxs: List[int], n_train: int, n_val: int, n_test: int, seed: int = 42
) -> Tuple[np.ndarray]:
    """Shuffles and splits ``frame_idxs``."""
    frame_idxs_ = frame_idxs.copy()
    np.random.default_rng(seed=seed).shuffle(frame_idxs_)

    # Take the test set as the first ``n_test`` idxs. This will be consistent regardless
    # of ``n_train`` and ``n_val``.
    test_id = frame_idxs_[:n_test]

    # Now shuffle the remaining idxs and draw the train and val idxs
    frame_idxs_ = frame_idxs_[n_test:]

    val_id = frame_idxs_[:n_val]
    train_id = frame_idxs_[n_val : n_train + n_val]

    assert len(np.intersect1d(train_id, val_id)) == 0
    assert len(np.intersect1d(train_id, test_id)) == 0
    assert len(np.intersect1d(val_id, test_id)) == 0

    return [train_id, val_id, test_id]


def _group_idxs(
    idxs: List[int],
    n_groups: int,
    group_sizes: Optional[Union[List[float], List[int]]] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Returns the indices in `idxs` in `n_groups` groups of indices, according
    to the relative or absolute sizes in `group_sizes`.

    For instance, if `n_groups` is 2 (i.e. for a train/test split), 2 arrays are
    returned. If `n_groups` is 3 (i.e. for a train/test/validation split), 3
    arrays are returned.

    If `group_sizes` is None, the group sizes returned are (to the nearest
    integer) equally sized for each group. If `group_sizes` is specified as a
    List of floats (i.e. relative sizes, whose sum is <= 1), the group sizes
    returned are converted to absolute sizes, i.e. multiplied by `n_indices`. If
    `group_sizes` is specified as a List of int, the group sizes returned
    are the absolute sizes specified.

    If `shuffle` is False, no shuffling of `idxs` is performed. If true, and
    `seed` is not None, `idxs` is shuffled using `seed` as the seed for the
    random number generator. If `seed` is None, the random number generator is
    not manually seeded.
    """
    # Check that group sizes are valid
    if group_sizes is not None:
        if len(group_sizes) != n_groups:
            raise ValueError(
                f"Length of group_sizes ({len(group_sizes)})"
                f" must match n_groups ({n_groups})."
            )

    # Create a copy of the indices so that shuffling doesn't affect the original
    idxs = np.array(idxs).copy()

    # Shuffle indices if seed is specified
    if shuffle:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(idxs)

    # Get absolute group sizes for train/test/validation split
    abs_group_sizes = _get_group_sizes(n_groups, len(idxs), group_sizes)

    if np.sum(abs_group_sizes) != len(idxs):
        raise ValueError("sum of group sizes not equal to len of `idxs` passed")

    # Grouped the indices
    grouped_idxs = []
    prev_size = 0
    for size in abs_group_sizes:
        grouped_idxs.append(idxs[prev_size : size + prev_size])
        prev_size += size

    # Check that there are no intersections between the groups
    ref_group = grouped_idxs[0]
    for group in grouped_idxs[1:]:
        assert len(np.intersect1d(ref_group, group)) == 0

    return grouped_idxs


def _get_group_sizes(
    n_groups: int,
    n_indices: int,
    group_sizes: Optional[Union[List[float], List[int]]] = None,
) -> np.ndarray:
    """
    Parses the `group_sizes` arg and returns an array of group sizes in absolute
    terms. If `group_sizes` is None, the group sizes returned are (to the
    nearest integer) evenly distributed across the number of unique indices;
    i.e. if there are 12 unique indices (`n_indices=10`), and `n_groups` is 3,
    the group sizes returned will be np.array([4, 4, 4]).

    If `group_sizes` is specified as a List of floats (i.e. relative sizes,
    whose sum is <= 1), the group sizes returned are converted to absolute
    sizes, i.e. multiplied by `n_indices`. If `group_sizes` is specified as a
    List of int, no conversion is performed. A cascade round is used to make
    sure that the group sizes are integers, with the sum of the List
    preserved and the rounding error minimized.

    :param n_groups: an int, the number of groups to split the data into :param
        n_indices: an int, the number of unique indices present in the data by
        which the data should be grouped.
    :param n_indices: a :py:class:`int` for the number of unique indices present
        in the input data for the specified `axis` and `names`.
    :param group_sizes: a sequence of :py:class:`float` or :py:class:`int`
        indicating the absolute or relative group sizes, respectively.

    :return: a :py:class:`numpy.ndarray` of :py:class:`int` indicating the
        absolute group sizes.
    """
    if group_sizes is None:  # equally sized groups
        group_sizes = np.array([1 / n_groups] * n_groups) * n_indices
    elif np.all([isinstance(size, int) for size in group_sizes]):  # absolute
        group_sizes = np.array(group_sizes)
    else:  # relative; List of float
        group_sizes = np.array(group_sizes) * n_indices

    # The group sizes may not be integers. Use cascade rounding to round them
    # all to integers whilst attempting to minimize rounding error.
    group_sizes = _cascade_round(group_sizes)

    return group_sizes


def _cascade_round(array: np.ndarray) -> np.ndarray:
    """
    Given an array of floats that sum to an integer, this rounds the floats
    and returns an array of integers with the same sum.
    Adapted from https://jsfiddle.net/cd8xqy6e/.
    """
    # Check type
    if not isinstance(array, np.ndarray):
        raise TypeError("must pass `array` as a numpy array.")
    # Check sum
    mod = np.sum(array) % 1
    if not np.isclose(round(mod) - mod, 0):
        raise ValueError("elements of `array` must sum to an integer.")

    float_tot, integer_tot = 0, 0
    rounded_array = []
    for element in array:
        new_int = round(element + float_tot) - integer_tot
        float_tot += element
        integer_tot += new_int
        rounded_array.append(new_int)

    # Check that the sum is preserved
    assert round(np.sum(array)) == round(np.sum(rounded_array))

    return np.array(rounded_array)


def _get_log_subset_sizes(
    n_max: int,
    n_subsets: int,
    base: Optional[float] = 10.0,
) -> np.array:
    """
    Returns an ``n_subsets`` length array of subset sizes equally spaced along a
    log of specified ``base`` (default base 10) scale from 0 up to ``n_max``.
    Elements of the returned array are rounded to integer values. The final
    element of the returned array may be less than ``n_max``.
    """
    # Generate subset sizes evenly spaced on a log scale, custom base
    subset_sizes = np.logspace(
        np.log(n_max / n_subsets) / np.log(base),
        np.log(n_max) / np.log(base),
        num=n_subsets,
        base=base,
        endpoint=True,
        dtype=int,
    )
    return subset_sizes


def reindex_tensormap(
    tensor: mts.TensorMap,
    system_ids: List[int],
) -> mts.TensorMap:
    """
    Takes a single TensorMap `tensor` containing data on multiple systems and re-indexes
    the "system" dimension of the samples. Assumes input has numeric system indices from
    {0, ..., N_system - 1} (inclusive), and maps these indices one-to-one with those
    passed in ``system_ids``.
    """
    assert tensor.sample_names[0] == "system"

    index_mapping = {i: A for i, A in enumerate(system_ids)}

    def new_row(row):
        return [index_mapping[row[0].item()]] + [i for i in row[1:]]

    new_blocks = []
    for block in tensor.blocks():
        new_samples = mts.Labels(
            names=block.samples.names,
            values=torch.tensor(
                [new_row(row) for row in block.samples.values],
                dtype=torch.int32,
            ),
        )
        new_block = mts.TensorBlock(
            values=block.values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return mts.TensorMap(tensor.keys, new_blocks)


def split_tensormap_by_system(
    tensor: mts.TensorMap, system_ids: List[int]
) -> List[mts.TensorMap]:
    """
    Splits a single TensorMap `tensor` into per-system TensorMaps along the samples
    axis.
    """
    return [
        mts.slice(
            tensor,
            "samples",
            selection=mts.Labels(
                names="system", values=torch.tensor([A]).reshape(-1, 1)
            ),
        )
        for A in system_ids
    ]


# def drop_blocks_for_nonpresent_types(
#     frames: Union[Frame, List[Frame]],
#     tensor: mts.TensorMap,
#     masked_system_type: Optional[str] = None,
#     **mask_kwargs,
# ) -> mts.TensorMap:
#     """
#     Drops blocks from a TensorMap `tensor` that correspond to atom
#     types not present in the given ``frame``.
#     """
#     # Get the unique atom types in `frames`
#     if isinstance(frames, Frame):
#         frames = [frames]

#     atom_types = []
#     for frame in frames:
#         if masked_system_type is not None:  # retype frames if applicable
#             frame = mask.retype_frame(
#                 frame, masked_system_type, **mask_kwargs
#             )
#         for _type in system.get_types(frame):
#             if _type not in atom_types:
#                 atom_types.append(_type)

#     # Build the new keys and TensorMap
#     new_key_vals = []
#     for key in tensor.keys:
#         if key["center_type"] in atom_types:
#             new_key_vals.append(list(key.values))

#     new_keys = mts.Labels(
#         names=tensor.keys.names,
#         values=_dispatch.int_array(new_key_vals, "torch"),
#     )
#     return mts.TensorMap(
#         keys=new_keys,
#         blocks=[tensor[key] for key in new_keys],
#     )
