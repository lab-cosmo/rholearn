"""
Module for masking :py:class:`chemfile.Frame` objects
"""

from typing import List, Optional, Tuple, Union

import metatensor
import metatensor.torch
import numpy as np
from chemfiles import Atom, Frame, UnitCell

from rholearn.utils import _dispatch, geometry, system

MASKED_SYSTEM_TYPES = ["slab"]


def get_point_indices_by_region(
    points: np.ndarray,
    masked_system_type: str,
    region: str,
    **kwargs,
) -> Tuple[callable]:
    """
    For a given set of points (i.e. a 3 x npts shaped array), returns the indices of
    points belonging to the "active", "buffer", or "masked" ``region`` for a specified
    ``masked_system_type``.

    The following system types are supported:

    - Slab:
        - Active atoms are those at:
            z >= (- surface_depth)

        - Buffer atoms are those at:
            (- surface_depth) > z >= (- surface_depth - buffer_depth)

        - Masked atoms are those at:
            z < (- surface_depth - buffer_depth)

    """
    assert masked_system_type in MASKED_SYSTEM_TYPES, (
        f"``masked_system_type`` must be one of {MASKED_SYSTEM_TYPES},"
        f" got: {masked_system_type}"
    )

    # Slab
    if masked_system_type == "slab":
        assert (
            kwargs.get("surface_depth") is not None
        ), "kwarg ``surface_depth`` must be passed"
        assert (
            kwargs.get("buffer_depth") is not None
        ), "kwarg ``buffer_depth`` must be passed"
        if region == "active":
            return np.where(points[:, 2] >= -kwargs.get("surface_depth"))[0]
        elif region == "buffer":
            return np.where(
                (points[:, 2] < -kwargs.get("surface_depth"))
                & (
                    points[:, 2]
                    >= -kwargs.get("surface_depth") - kwargs.get("buffer_depth")
                )
            )[0]
        else:
            assert region == "masked"
            return np.where(
                points[:, 2] < -kwargs.get("surface_depth") - kwargs.get("buffer_depth")
            )[0]

    raise ValueError(
        f"invalid `masked_system_type`: {masked_system_type}."
        f" Must be one of {MASKED_SYSTEM_TYPES}"
    )


def retype_frame(frame: Frame, masked_system_type: str, **kwargs) -> Frame:
    """
    Retypes a Frame. Supported system types are:

        - "slab"

    Active atoms are not retyped. Buffer atoms are retyped to carry a "_1" suffix.
    Masked atoms are retyped to carry a "_2" suffix.

    How atoms are categorised by region is controlled by ``**kwargs``.
    """
    if isinstance(frame, list):
        return [retype_frame(f, masked_system_type, **kwargs) for f in frame]

    # Re-type the buffer atoms
    frame = retype_atoms(
        frame,
        indices=get_point_indices_by_region(
            frame.positions, masked_system_type, "buffer", **kwargs
        ),
        type_suffix="_1",
    )

    # Re-type masked atoms
    frame = retype_atoms(
        frame,
        indices=get_point_indices_by_region(
            frame.positions, masked_system_type, "masked", **kwargs
        ),
        type_suffix="_2",
    )

    return frame


def retype_atoms(
    frame: Frame,
    indices: List[int],
    type_suffix: Optional[str] = None,
) -> Frame:
    """
    For each frame passed in ``frame``, retypes the atoms whose indices are passed in
    `indices`. Retyping involves modifying the `name` and `type` attributes of each
    :py:class:`chemfiles.Atom` object to be retyped by suffixing with `type_suffix`, or
    "_1" if not specified.
    """
    # Use "_1" by default for the type suffix
    if type_suffix is None:
        type_suffix = "_1"

    retyped_frame = Frame()
    retyped_frame.cell = UnitCell(frame.cell.matrix)
    for i, atom in enumerate(frame.atoms):
        atom_name = atom.name
        atom_type = atom.type
        if i in indices:
            atom_name += type_suffix
            atom_type += type_suffix

        retyped_frame.add_atom(Atom(name=atom_name, type=atom_type), frame.positions[i])

    return retyped_frame


def mask_coeff_vector(
    coeff_vector: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    atom_idxs_to_keep: List[int],
    drop_empty_blocks: bool = False,
    backend: Optional[str] = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Takes a TensorMap corresponding to a vector of coefficients and slices it to only
    contain sample indices in `atom_idxs_to_keep`, essentially masking those indices not
    present.

    Any blocks that have been sliced to zero samples can be dropped if
    ``drop_empty_blocks`` is set to true.

    Assumes that the atomic centers are in the samples and named "atom".
    """
    # Assign backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Slice samples axis
    coeff_vector_masked = mts.slice(
        coeff_vector,
        axis="samples",
        selection=mts.Labels(
            names=["atom"],
            values=_dispatch.array(atom_idxs_to_keep, backend).reshape(-1, 1),
        ),
    )

    if drop_empty_blocks:  # identify empty blocks and drop
        coeff_vector_masked = _drop_empty_blocks(coeff_vector_masked, backend)

    return coeff_vector_masked


def unmask_coeff_vector(
    coeff_vector: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    frame: Frame,
    frame_idx: int,
    in_keys: Union[metatensor.Labels, metatensor.torch.Labels],
    properties: List[Union[metatensor.Labels, metatensor.torch.Labels]],
    backend: Optional[str] = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Builds a zeros tensor with the correct dimensions according to the atomic samples in
    `frame` and the specified `in_keys` and `properties`. Fills in corresponding samples
    from `coeff_vector` and returns.
    """
    # First slice the keys and properties to only include atom types present in
    # ``frame``
    atom_types = system.get_types(frame)
    system_key_vals = []
    system_props = []
    for key, props in zip(in_keys, properties):
        if key["center_type"] in atom_types:
            system_key_vals.append([v for v in key.values])
            system_props.append(props)

    in_keys = _dispatch.labels(in_keys.names, system_key_vals, backend)

    # Build the zeros tensor with the correct dimensions
    coeff_vector_unmasked = _build_zeros_tensor(
        frame=frame,
        frame_idx=frame_idx,
        in_keys=in_keys,
        properties=system_props,
        backend=backend,
    )

    # Modify in-place the zeros tensor, filling in values from the
    # masked tensor for each of the samples present
    for masked_key, masked_block in coeff_vector.items():
        for masked_sample_i, masked_sample in enumerate(masked_block.samples):
            # Find the corresponding sample in the unmasked tensor
            unmasked_sample_i = coeff_vector_unmasked[masked_key].samples.position(
                masked_sample
            )
            coeff_vector_unmasked[masked_key].values[unmasked_sample_i] = (
                masked_block.values[masked_sample_i]
            )

    return coeff_vector_unmasked


def mask_overlap_matrix(
    overlap_matrix: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    atom_idxs_to_keep: List[int],
    drop_empty_blocks: bool = False,
    backend: Optional[str] = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Takes a TensorMap corresponding to a matrix of overlaps and slices it to only
    contain sample indices in `atom_idxs_to_keep`, essentially masking those indices not
    present. Any blocks that have been sliced to zero samples are dropped. Returns the
    masked TensorMap.

    Assumes the pairs of atomic centers are in the samples and named as "atom_1" and
    "atom_2".
    """
    # Assign backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Mask the tensor. First slice "atom_1" in the samples, then slice "atom_2"
    # in the samples.
    overlap_matrix_masked = mts.slice(
        overlap_matrix,
        axis="samples",
        selection=mts.Labels(
            names=["atom_1"],
            values=_dispatch.array(atom_idxs_to_keep, backend).reshape(-1, 1),
        ),
    )
    overlap_matrix_masked = mts.slice(
        overlap_matrix_masked,
        axis="samples",
        selection=mts.Labels(
            names=["atom_2"],
            values=_dispatch.array(atom_idxs_to_keep, backend).reshape(-1, 1),
        ),
    )

    if drop_empty_blocks:
        overlap_matrix_masked = _drop_empty_blocks(overlap_matrix_masked, backend)

    return overlap_matrix_masked


def cutoff_overlap_matrix(
    frames: List[Frame],
    frame_idxs: List[int],
    overlap_matrix: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    cutoff: Optional[float] = None,
    radii: Optional[dict] = None,
    drop_empty_blocks: bool = False,
    backend: Optional[str] = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Takes a TensorMap corresponding to a matrix of overlaps for each atom pair (where i1
    < i2) and slices it according to various criteria:

        - if ``cutoff`` is specified, removes overlaps between atoms further than this
          away from each other
        - alternatively (only defined in absense of ``cutoff``), ``radii`` can be passed
          to specify the actual basis function radii of each atomic type in ``frame``,
          and uses this as atomic type dependent cutoff. Must only be passed if data for
          one structure is passed.

    Any blocks that have been sliced to zero samples can be dropped if
    ``drop_empty_blocks`` is true.

    Returns the masked TensorMap.
    """
    # Assign backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # First compute the neighbor list, within the specified `cutoff` or the
    # twice the max radius in `radii`.
    if cutoff is None:
        assert radii is not None, "must specify either `cutoff` or `radii`"
        assert (
            len(frames) == 1 and len(frame_idxs) == 1
        ), "If passing `radii`, only one frame should be passed"
        cutoff = 2 * max(list(radii.values()))
    else:
        assert radii is None, "cannot specify both `cutoff` and `radii`"

    neighborlist = geometry.get_neighbor_list(
        frames, frame_idxs, cutoff=cutoff, backend=backend
    )

    # If `radii` is passed, slice the neighborlist to account for
    # max radii of each chemical species
    if radii is not None:
        assert len(frames) == 1
        frame = frames[0]

        keep_idxs = []
        atom_types = system.get_types(frame)
        for label_idx, (_, i, j) in enumerate(neighborlist):
            # Check that this atom type is reported in `radii`. If not, this atom may
            # have an empty basis set definition, thus having no overlap.
            if atom_types[i] not in radii or atom_types[j] not in radii:
                continue
            if frame.distance(i, j) <= radii[atom_types[i]] + radii[atom_types[j]]:
                keep_idxs.append(label_idx)

        neighborlist = mts.Labels(
            names=neighborlist.names,
            values=_dispatch.int_array(
                [
                    [elem for elem in neighborlist.values[label_idx]]
                    for label_idx in keep_idxs
                ],
                backend=backend,
            ),
        )

    # Now slice the overlap matrix based on the neighbor list
    overlap_matrix_masked = mts.slice(
        overlap_matrix,
        axis="samples",
        selection=neighborlist,
    )

    if drop_empty_blocks:
        overlap_matrix_masked = _drop_empty_blocks(overlap_matrix_masked, backend)

    return overlap_matrix_masked


def sparsify_overlap_matrix(
    overlap_matrix: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    sparsity_threshold: float,
    drop_empty_blocks: bool = False,
    backend: str = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    For each sample in the input ``overlap`` matrix (i.e. atom pair), drops the sample
    if the absolute values of all overlaps between basis functions on these atoms is
    less than ``sparsity_threshold``.

    Any empty blocks that remain after this can be dropped by setting
    ``drop_empty_blocks`` to true.
    """
    # Assign backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"Unknown backend: {backend}")

    new_blocks = []
    for block in overlap_matrix:

        # Discard any samples where all the absolute overlap values are less than
        # the specified threshold.
        samples_mask = [
            bool(
                _dispatch.any(
                    _dispatch.abs(sample, backend) >= sparsity_threshold, backend
                )
            )
            for sample in block.values
        ]
        new_blocks.append(
            mts.TensorBlock(
                samples=mts.Labels(
                    names=block.samples.names,
                    values=_dispatch.int_array(
                        block.samples.values[samples_mask],
                        backend,
                    ),
                ),
                components=block.components,
                properties=block.properties,
                values=block.values[samples_mask],
            )
        )

    return mts.TensorMap(overlap_matrix.keys, new_blocks)


def _drop_non_active_overlaps(
    frame: Frame,
    masked_system_type: str,
    overlap_matrix: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    backend: Optional[str] = "numpy",
    **kwargs,
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    For the given `frame` and `masked_system_type`, drops the overlaps between atom
    pairs that are both atoms in non-active regions, i.e. buffer or masked atoms.
    """
    # Assign backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Find the indices of atoms in the active region
    active_atom_idxs = get_point_indices_by_region(
        frame.positions, masked_system_type, "active", **kwargs
    )

    new_blocks = []
    for block in overlap_matrix:
        # Keep only the samples where either atom i or j is in the active atoms
        samples_mask = [
            sample_i
            for sample_i, (A, i, j) in enumerate(block.samples)
            if i in active_atom_idxs or j in active_atom_idxs
        ]
        new_blocks.append(
            mts.TensorBlock(
                samples=mts.Labels(
                    block.samples.names, block.samples.values[samples_mask]
                ),
                components=block.components,
                properties=block.properties,
                values=block.values[samples_mask],
            )
        )

    overlap_matrix = mts.TensorMap(overlap_matrix.keys, new_blocks)

    return _drop_empty_blocks(overlap_matrix, backend)


def _build_zeros_tensor(
    frame: Frame,
    frame_idx: int,
    in_keys: Union[metatensor.Labels, metatensor.torch.Labels],
    properties: List[Union[metatensor.Labels, metatensor.torch.Labels]],
    backend: Optional[str] = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Builds a zeros-like TensorMap with the correct metadata corresponding to the target
    property, based on the atoms in input system and the target basis set definition.
    """
    # Assign backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Build the zeros tensor
    blocks = []
    for key, props in zip(in_keys, properties):
        o3_lambda, _, center_type = key
        ref_atom_sym = system.atomic_number_to_atomic_symbol(center_type)
        samples = mts.Labels(
            names=["system", "atom"],
            values=_dispatch.int_array(
                [
                    [frame_idx, atom_id]
                    for atom_id, atom_sym in enumerate(system.get_symbols(frame))
                    if atom_sym == ref_atom_sym
                ],
                backend,
            ).reshape(-1, 2),
        )
        components = [
            mts.Labels(
                names=["o3_mu"],
                values=_dispatch.arange(
                    -o3_lambda, o3_lambda + 1, backend=backend
                ).reshape(-1, 1),
            )
        ]
        blocks.append(
            mts.TensorBlock(
                values=_dispatch.zeros(
                    (
                        len(samples),
                        *(len(c) for c in components),
                        len(props),
                    ),
                    backend=backend,
                ),
                samples=samples,
                components=components,
                properties=props,
            )
        )
    return mts.TensorMap(in_keys, blocks)


def _drop_empty_blocks(
    tensor: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    backend: str = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Drops blocks from a TensorMap that have been sliced to zero samples.
    """
    # Return None if None
    if tensor is None:
        return tensor

    # Assign backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Find keys to drop
    keys_to_drop = []
    for key, block in tensor.items():
        if any([dim == 0 for dim in block.values.shape]):
            keys_to_drop.append(key)

    if len(keys_to_drop) == 0:
        return tensor

    # Drop blocks
    tensor_dropped = mts.drop_blocks(
        tensor,
        keys=mts.Labels(
            names=keys_to_drop[0].names,
            values=_dispatch.array(
                [[i for i in k.values] for k in keys_to_drop], backend
            ),
        ),
    )

    return tensor_dropped
