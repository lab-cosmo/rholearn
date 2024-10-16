"""
Module for masking :py:class:`chemfile.Frame` objects
"""

from typing import Callable, List, Optional, Tuple, Union

import metatensor
import metatensor.torch
import numpy as np
from chemfiles import Frame

from rholearn.utils import _dispatch, geometry, system


def get_atom_idxs_by_region(
    frame: Frame,
    get_active_coords: Callable,
    get_buffer_coords: Callable,
    get_masked_coords: Callable,
) -> Tuple[np.ndarray]:
    """
    Using the callable functions ``get_active_coords``, ``get_buffer_coords``, and
    ``get_masked_coords`` determines the indices in each of 'active', 'buffer', and
    'masked' regions of the input system in ``frame``. Returns these grouped indices,
    respectively.

    The callable functions should take a :py:class:`chemfiles.Frame` object as input and
    return a boolean array indicating whether each atom is in the active or buffer

    Masked region atoms are assumed to be those that are neither in the active nor
    buffer regions.
    """
    # Get active and buffer atom indices
    idxs_active = get_active_coords(frame.positions)
    idxs_buffer = get_buffer_coords(frame.positions)
    idxs_masked = get_masked_coords(frame.positions)

    # Check zero intersections
    assert len(np.intersect1d(idxs_active, idxs_buffer)) == 0
    assert len(np.intersect1d(idxs_active, idxs_masked)) == 0

    return idxs_active, idxs_buffer, idxs_masked


def retype_masked_atoms(
    frame: Union[Frame, List[Frame]],
    get_masked_coords: Callable,
    type_suffix: Optional[str] = None,
) -> Frame:
    """
    For each frame passed in ``frame``, identifies the atoms to be masked and retypes
    them.

    The idxs of the masked atoms are returned by the callable ``get_masked_coords``.
    Retyping involves modifying the `name` and `type` attributes of each
    :py:class:`chemfiles.Atom` object to be masked by suffixing with a "_1".
    """
    # Use "_1" by default for the type suffix
    if type_suffix is None:
        type_suffix = "_1"

    if isinstance(frame, list):
        return [retype_masked_atoms(f, get_masked_coords, type_suffix) for f in frame]

    # Get the indices of the various species
    idxs_masked = get_masked_coords(frame.positions)

    for i, atom in enumerate(frame.atoms):
        if i in idxs_masked:
            atom.name += type_suffix
            atom.type += type_suffix

    return frame


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
        labels=mts.Labels(
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
        labels=mts.Labels(
            names=["atom_1"],
            values=_dispatch.array(atom_idxs_to_keep, backend).reshape(-1, 1),
        ),
    )
    overlap_matrix_masked = mts.slice(
        overlap_matrix_masked,
        axis="samples",
        labels=mts.Labels(
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
    Takes a TensorMap corresponding to a matrix of overlaps and slices it to only
    contain sample indices for pairs of atoms that are within the specified ``cutoff``.
    Any blocks that have been sliced to zero samples can be dropped. Returns the masked
    TensorMap.

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

    # First compute the neighbor list, within the specified `cutoff` or the
    # twice the max radius in `radii`.
    if cutoff is None:
        assert radii is not None, "must specify either `cutoff` or `radii`"
        assert len(frames) == 1 and len(frame_idxs) == 1, (
            "If passing `radii`, only one frame should be passed"
        )
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
            if (
                frame.distance(i, j) <= radii[atom_types[i]] + radii[atom_types[j]]
            ):
                keep_idxs.append(label_idx)

        neighborlist = mts.Labels(
            names=neighborlist.names,
            values=_dispatch.int_array(
                [
                    [elem for elem in neighborlist.values[label_idx]]
                    for label_idx in keep_idxs
                ],
                backend=backend,
            )
        )

    # Now slice the overlap matrix based on the neighbor list
    overlap_matrix_masked = mts.slice(
        overlap_matrix,
        axis="samples",
        labels=neighborlist,
    )

    if drop_empty_blocks:
        overlap_matrix_masked = _drop_empty_blocks(overlap_matrix_masked, backend)

    return overlap_matrix_masked


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
        o3_lambda, center_type = key
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
