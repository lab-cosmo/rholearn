"""
Module for converting coefficient (or projection) vectors and overlap matrices
between numpy ndarrays and metatensor TensorMap formats.
"""

from typing import List, Optional, Tuple, Union

import metatensor
import metatensor.torch
import numpy as np
import torch
from chemfiles import Frame

from rholearn.utils import _dispatch, system, utils


def get_global_basis_set(
    basis_sets: dict,
    center_types: Optional[List[int]] = None,
    backend: str = "numpy",
) -> Union[metatensor.Labels, metatensor.torch.Labels]:
    """
    Takes a list of basis set definiitions in the form:

    basis_sets = [
        {
            "lmax": {"H": 1, "C": 2},
            "nmax": {("H", 0): 2, ("H", 1): 3, ("C", 0): 4, ("C", 1): 5, ("C", 2): 6},
        }
    ]

    checks for consistency between all the basis sets, and returns a :py:class:`Labels`
    object with dimensions ["o3_lambda", "center_type", "nmax"]
    """
    # Set backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"invalid backend: {backend}")

    # Extarct global basis
    global_basis = {}
    for basis in basis_sets:
        for key, val in basis["nmax"].items():
            center_symbol, o3_lambda = key
            center_type = system.atomic_symbol_to_atomic_number(center_symbol)
            if center_types is not None:
                if center_type not in center_types:
                    continue
            new_key = (o3_lambda, center_type)
            if new_key not in global_basis:
                global_basis[new_key] = val
            else:
                assert global_basis[new_key] == val

    return mts.Labels(
        ["o3_lambda", "center_type", "nmax"],
        values=_dispatch.int_array(
            [
                [o3_lambda, center_type, nmax]
                for (o3_lambda, center_type), nmax in global_basis.items()
            ],
            backend=backend,
        ),
    )


def _get_flat_index(
    symbol_list: list,
    lmax: dict,
    nmax: dict,
    atom_i: int,
    o3_lambda: int,
    n: int,
    o3_mu: int,
) -> int:
    """
    Get the flat index of the coefficient pointed to by the basis function indices
    ``atom_i``, ``o3_lambda``, ``n``, ``o3_mu``.

    Given the basis set definition specified by ``lmax`` and ``nmax``, the assumed
    ordering of basis function coefficients follows the following hierarchy, which can
    be read as nested loops over the various indices. Be mindful that some indices range
    are from 0 to x (exclusive) and others from 0 to x + 1 (exclusive). The ranges
    reported below are ordered.

    1. Loop over atoms (index ``atom_i``, of chemical species ``a``) in the structure.
       ``atom_i`` takes values 0 to N (** exclusive **), where N is the number of atoms
       in the structure.
    2. Loop over spherical harmonics channel (index ``o3_lambda``) for each atom.
       ``o3_lambda`` takes values from 0 to ``lmax[a] + 1`` (** exclusive **), where
       ``a`` is the chemical species of atom ``atom_i``, given by the chemical symbol at
       the ``atom_i``th position of ``symbol_list``.
    3. Loop over radial channel (index ``n``) for each atom ``atom_i`` and
       spherical harmonics channel ``o3_lambda`` combination. ``n`` takes values
       from 0 to ``nmax[(a, o3_lambda)]`` (** exclusive **).
    4. Loop over spherical harmonics component (index ``o3_mu``) for each atom.
       ``o3_mu`` takes values from ``-o3_lambda`` to ``o3_lambda`` (** inclusive **).

    Note that basis function coefficient vectors, projection vectors, and overlap
    matrices outputted by quantum chemistry packages such as PySCF and AIMS may follow
    different conventions. ``rholearn`` provides parsing functions to standardize these
    outputs to the convention described above. Once standardized, the functions in this
    module can be used to convert to metatensor format.

    :param lmax : dict containing the maximum spherical harmonics (o3_lambda) value for
        each atom type.
    :param nmax: dict containing the maximum radial channel (n) value for each
        combination of atom type and o3_lambda.
    :param atom_i: int, the atom index.
    :param o3_lambda: int, the spherical harmonics index.
    :param n: int, the radial channel index.
    :param o3_mu: int, the spherical harmonics component index.

    :return int: the flat index of the coefficient pointed to by the input indices.
    """
    # Check atom index is valid
    if atom_i not in np.arange(0, len(symbol_list)):
        raise ValueError(
            f"invalid atom index, atom_i={atom_i} is "
            f"not in the range [0, {len(symbol_list)})."
            f"Passed symbol list: {symbol_list}"
        )
    # Check o3_lambda value is valid
    if o3_lambda not in np.arange(0, lmax[symbol_list[atom_i]] + 1):
        raise ValueError(
            "invalid spherical harmonics index, "
            f"o3_lambda={o3_lambda} is not in the range "
            f"of valid values for species {symbol_list[atom_i]}: "
            f"[0, {lmax[symbol_list[atom_i]]}] (inclusive)."
        )
    # Check n value is valid
    if n not in np.arange(0, nmax[(symbol_list[atom_i], o3_lambda)]):
        raise ValueError(
            f"invalid radial index, n={n}"
            " is not in the range of valid values for species "
            f"{symbol_list[atom_i]}, o3_lambda={o3_lambda}:"
            f" [0, {nmax[(symbol_list[atom_i], o3_lambda)]}) (exclusive)."
        )
    # Check o3_mu value is valid
    if o3_mu not in np.arange(-o3_lambda, o3_lambda + 1):
        raise ValueError(
            f"invalid azimuthal index, o3_mu={o3_mu} is not"
            " in the o3_lambda range [-o3_lambda, o3_lambda] = "
            f"[{-o3_lambda}, +{o3_lambda}] (inclusive)."
        )
    # Define the atom offset
    atom_offset = 0
    for atom_index in range(atom_i):
        symbol = symbol_list[atom_index]
        for l_tmp in np.arange(lmax[symbol] + 1):
            atom_offset += (2 * l_tmp + 1) * nmax[(symbol, l_tmp)]

    # Define the o3_lambda offset
    l_offset = 0
    symbol = symbol_list[atom_i]
    for l_tmp in range(o3_lambda):
        l_offset += (2 * l_tmp + 1) * nmax[(symbol, l_tmp)]

    # Define the n offset
    n_offset = (2 * o3_lambda + 1) * n

    # Define the o3_mu offset
    m_offset = o3_mu + o3_lambda

    return atom_offset + l_offset + n_offset + m_offset


# ===== convert numpy to metatensor format =====


def coeff_vector_ndarray_to_dict(
    frame: Frame,
    coeff_vector: np.ndarray,
    lmax: dict,
    nmax: dict,
) -> dict:
    """
    Convert a flat array of basis function coefficients (or projections) to a dictionary
    of the form:

    Keys: tuples of (o3_lambda, symbol)
    Blocks: dict of the form {atom_i: array} where array is the coefficient array for
        the atom ``atom_i``.
    """
    symbols = system.get_symbols(frame)

    # First, confirm the length of the flat array is as expected
    num_coeffs_by_uniq_symbol = {}
    for symbol in np.unique(symbols):
        n_coeffs = 0
        for l_tmp in range(lmax[symbol] + 1):
            for _n_tmp in range(nmax[(symbol, l_tmp)]):
                n_coeffs += 2 * l_tmp + 1
        num_coeffs_by_uniq_symbol[symbol] = n_coeffs

    num_coeffs_by_symbol = np.array(
        [num_coeffs_by_uniq_symbol[symbol] for symbol in symbols]
    )
    assert np.sum(num_coeffs_by_symbol) == len(coeff_vector)

    # Split the flat array by atomic index
    split_by_atom = np.split(coeff_vector, np.cumsum(num_coeffs_by_symbol), axis=0)[:-1]
    assert len(split_by_atom) == len(symbols)
    assert np.sum([len(x) for x in split_by_atom]) == len(coeff_vector)

    num_coeffs_by_l = {
        symbol: np.array(
            [
                (2 * l_tmp + 1) * nmax[(symbol, l_tmp)]
                for l_tmp in range(lmax[symbol] + 1)
            ]
        )
        for symbol in np.unique(symbols)
    }
    for symbol in np.unique(symbols):
        assert np.sum(num_coeffs_by_l[symbol]) == num_coeffs_by_uniq_symbol[symbol]

    # Split each atom array by angular momentum channel
    new_split_by_atom = []
    for symbol, atom_arr in zip(symbols, split_by_atom):
        split_by_l = np.split(atom_arr, np.cumsum(num_coeffs_by_l[symbol]), axis=0)[:-1]
        assert len(split_by_l) == lmax[symbol] + 1

        new_split_by_l = []
        for l_tmp, l_arr in enumerate(split_by_l):
            assert len(l_arr) == nmax[(symbol, l_tmp)] * (2 * l_tmp + 1)

            # Reshape to have components and properties on the 2nd and 3rd axes.
            # IMPORTANT: Fortran order!
            l_arr = l_arr.reshape((1, 2 * l_tmp + 1, nmax[(symbol, l_tmp)]), order="F")
            new_split_by_l.append(l_arr)

        new_split_by_atom.append(new_split_by_l)

    # Create a dict to store the arrays by o3_lambda and species
    coeff_dict = {}
    for symbol in np.unique(symbols):
        for l_tmp in range(lmax[symbol] + 1):
            coeff_dict[(l_tmp, symbol)] = {}

    for i_tmp, (symbol, atom_arr) in enumerate(zip(symbols, new_split_by_atom)):
        for l_tmp, l_array in zip(range(lmax[symbol] + 1), atom_arr):
            coeff_dict[(l_tmp, symbol)][i_tmp] = l_array

    return coeff_dict


def coeff_vector_ndarray_to_tensormap(
    frame: Frame,
    coeff_vector: np.ndarray,
    lmax: dict,
    nmax: dict,
    structure_idx: int,
    tests: Optional[int] = 0,
    backend: str = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Convert a vector of basis function coefficients (or projections) to metatensor
    TensorMap format.

    :param frame: :py:class:`chemfiles.Frame` object containing the atomic structure for
        which the coefficients (or projections) were calculated.
    :param coeff_vector: np.ndarray of shape (N,), where N is the number of basis
        functions the electron density is expanded onto.
    :param lmax: dict containing the maximum spherical harmonics (o3_lambda) value for
        each atomic species.
    :param nmax: dict containing the maximum radial channel (n) value for each
        combination of atomic species and o3_lambda value.
    :param structure_idx: int, the index of the structure in the overall dataset. If
        None (default), the samples metadata of each block in the output TensorMap will
        not contain an index for the structure, atom_i.e. the sample names will just be
        ["atom"]. If an integer, the sample names will be ["system", "atom"] and the
        index for "system" will be ``structure_idx``.
    :param tests: int, the number of coefficients to randomly compare between the raw
        input array and processed TensorMap to check for correct conversion.

    :return TensorMap: the TensorMap containing the coefficients data and metadata.
    """
    # Parse the coefficient vector into blocks
    coeff_dict = coeff_vector_ndarray_to_dict(frame, coeff_vector, lmax, nmax)

    # Convert to TensorMap
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"invalid backend: {backend}")

    # Build the TensorMap keys
    keys = mts.Labels(
        names=["o3_lambda", "o3_sigma", "center_type"],
        values=np.array(
            [
                [o3_lambda, 1, system.atomic_symbol_to_atomic_number(symbol)]
                for o3_lambda, symbol in coeff_dict.keys()
            ]
        ),
    )

    # Build the TensorMap blocks
    blocks = []
    for o3_lambda, _, center_type in keys:
        symbol = system.atomic_number_to_atomic_symbol(center_type)
        raw_block = coeff_dict[(o3_lambda, symbol)]
        atom_idxs = np.sort(list(raw_block.keys()))

        # Define the sample values
        sample_values = np.array([[structure_idx, atom_i] for atom_i in atom_idxs])
        block = mts.TensorBlock(
            samples=mts.Labels(names=["system", "atom"], values=sample_values),
            components=[
                mts.Labels(
                    names=["o3_mu"],
                    values=np.arange(-o3_lambda, +o3_lambda + 1).reshape(-1, 1),
                ),
            ],
            properties=mts.Labels(
                names=["n"],
                values=np.arange(nmax[(symbol, o3_lambda)]).reshape(-1, 1),
            ),
            values=np.ascontiguousarray(
                np.concatenate([raw_block[atom_i] for atom_i in atom_idxs], axis=0)
            ),
        )
        assert block.values.shape == (
            len(atom_idxs),
            2 * o3_lambda + 1,
            nmax[(symbol, o3_lambda)],
        )
        blocks.append(block)

    # Construct TensorMap
    tensor = mts.TensorMap(keys=keys, blocks=blocks)

    # Check number of elements
    assert utils.num_elements_tensormap(tensor) == len(coeff_vector)

    # Check values of the coefficients, repeating the test `tests` number of times.
    for _ in range(tests):
        if not test_coeff_vector_conversion(
            frame,
            lmax,
            nmax,
            coeff_vector,
            tensor,
            structure_idx=structure_idx,
        ):
            raise ValueError("Conversion test failed.")

    return tensor


def overlap_matrix_ndarray_to_dict(
    frame: Frame,
    overlap_matrix: np.ndarray,
    lmax: dict,
    nmax: dict,
) -> dict:
    """
    Converts a full 2D overlap matrix to dictionary format, where the keys are pairs of
    atomic symbols and the values are dictionaries of the form {(i_1, i_2): matrix}
    where matrix is the overlap matrix for the atom pair (i_1, i_2).

    Note that only the diagonal blocks and the upper triangle off-diagonal blocks along
    the atomic index axes are stored. To preserve the outcome of a matrix multiplication
    with this upper-triagnular matrix, the values of the stored off-diagonal blocks are
    scaled by a factor of 2.
    """

    # Get the number of coefficients for each atom type
    symbols = system.get_symbols(frame)
    num_coeffs_by_uniq_symbol = {}
    for symbol in np.unique(symbols):
        n_coeffs = 0
        for l_tmp in range(lmax[symbol] + 1):
            for _n_tmp in range(nmax[(symbol, l_tmp)]):
                n_coeffs += 2 * l_tmp + 1
        num_coeffs_by_uniq_symbol[symbol] = n_coeffs

    num_coeffs_by_symbol = np.array(
        [num_coeffs_by_uniq_symbol[symbol] for symbol in symbols]
    )

    # Split the overlap into a list of matrices along axis 0, one for each atom
    split_by_i1 = np.split(overlap_matrix, np.cumsum(num_coeffs_by_symbol), axis=0)[:-1]

    ovlp_dict = {}
    for i_1, i1_matrix in enumerate(split_by_i1):

        # Split the overlap into a list of matrices along axis 1, one for each atom
        split_by_i2 = np.split(i1_matrix, np.cumsum(num_coeffs_by_symbol), axis=1)[:-1]
        for i_2, i2_matrix in enumerate(split_by_i2):

            # Only store diagonal and upper triagnle oof-diagonal blocks by atomic index
            if i_1 == i_2:
                i2_matrix_ = i2_matrix
            elif i_1 < i_2:
                i2_matrix_ = i2_matrix * 2
            else:
                continue

            if (symbols[i_1], symbols[i_2]) not in ovlp_dict:
                ovlp_dict[(symbols[i_1], symbols[i_2])] = {}

            if (i_1, i_2) not in ovlp_dict[(symbols[i_1], symbols[i_2])]:
                ovlp_dict[(symbols[i_1], symbols[i_2])][(i_1, i_2)] = i2_matrix_

    return ovlp_dict


def overlap_matrix_ndarray_to_tensormap(
    frame: Frame,
    overlap_matrix: np.ndarray,
    lmax: dict,
    nmax: dict,
    structure_idx: int,
    backend: str = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Converts the 2D array ``overlap_matrix`` into TensorMap format.

    The output TensorMap has the following structure:

        - Keys: ["center_1_type", "center_2_type"]
        - Samples: ["system", "atom_1", "atom_2"]
        - Components: ["l1_n1_m1"]
        - Properties: ["l2_n2_m2"]

    where the components and properties are composite indices for, respectively and
    hierarchically, ["o3_lambda", "n", "o3_mu"] for each in a pair of basis
    functions.
    """
    # Parse ndarray -> dict
    ovlp_dict = overlap_matrix_ndarray_to_dict(frame, overlap_matrix, lmax, nmax)

    # Convert to TensorMap
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"invalid backend: {backend}")

    key_vals = []
    blocks = []
    for (sym1, sym2), val in ovlp_dict.items():

        a1 = system.atomic_symbol_to_atomic_number(sym1)
        a2 = system.atomic_symbol_to_atomic_number(sym2)

        samples_vals = []
        block_vals = []
        for (i1, i2), ovlp in val.items():
            samples_vals.append([structure_idx, i1, i2])
            block_vals.append(ovlp)

        block_vals = _dispatch.stack(block_vals, axis=0, backend=backend)

        if block_vals.shape[1] == 0 or block_vals.shape[2] == 0:
            # empty basis functions for these atom types
            continue

        key_vals.append([a1, a2])

        block = mts.TensorBlock(
            values=block_vals,
            samples=mts.Labels(
                names=["system", "atom_1", "atom_2"],
                values=_dispatch.int_array(samples_vals, backend=backend),
            ),
            components=[
                mts.Labels(
                    names=["l1_n1_m1"],
                    values=_dispatch.arange(
                        0, block_vals.shape[1], backend=backend
                    ).reshape(-1, 1),
                ),
            ],
            properties=mts.Labels(
                names=["l2_n2_m2"],
                values=_dispatch.arange(
                    0, block_vals.shape[2], backend=backend
                ).reshape(-1, 1),
            ),
        )

        blocks.append(block)

    return mts.TensorMap(
        keys=mts.Labels(
            names=["center_1_type", "center_2_type"],
            values=_dispatch.int_array(key_vals, backend=backend),
        ),
        blocks=blocks,
    )


def coeff_vector_to_sparse_by_center_type(
    coeff_vector: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    backend: str = "numpy",
) -> Union[metatensor.TensorMap, metatensor.torch.TensorMap]:
    """
    Converts a TensorMap of spherical basis set coefficients, in the standard block
    sparse format of:

        - Keys: ["o3_lambda", "o3_sigma", "center_type"]
        - Samples: ["system", "atom"]
        - Components: ["o3_mu"]
        - Properties: ["n"]

    To a different block sparse format suitable for multiplication with on-site only
    overlaps, atom_i.e.:

        - Keys: ["center_type"]
        - Samples: ["system", "atom"]
        - Components: []
        - Properties: ["o3_lambda", "n", "o3_mu"]

    where the properties dimensions are ordered to following the respective hierarchy of
    indices, atom_i.e. with "o3_lambda" being the outer loop and "o3_mu" being the inner
    loop.
    """
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"invalid backend: {backend}")

    coeff_vector = mts.remove_dimension(coeff_vector, "keys", "o3_sigma")
    coeff_vector = coeff_vector.components_to_properties("o3_mu")
    coeff_vector = coeff_vector.keys_to_properties("o3_lambda", sort_samples=False)
    coeff_vector = mts.permute_dimensions(coeff_vector, "properties", (2, 0, 3, 1))
    coeff_vector = mts.sort(coeff_vector, "properties")

    return coeff_vector


# ===== convert metatensor to numpy format =====


def coeff_vector_blocks_to_flat(
    frame: Frame,
    coeff_vector: Union[metatensor.TensorMap, metatensor.torch.TensorMap],
    lmax: Optional[dict] = None,
    nmax: Optional[dict] = None,
    basis_set: Optional[metatensor.torch.Labels] = None,
) -> np.ndarray:
    """
    Convert a metatensor TensorMap of basis function coefficients (or projections) in
    block sparse format to a flat vector in numpy ndarray format.

    :param frame: :py:class:`chemfiles.Frame` object containing the atomic structure for
        which the coefficients (or projections) were calculated.
    :param coeff_vector: the TensorMap containing the basis function coefficients data
        and metadata.
    :param lmax: dict containing the maximum spherical harmonics (o3_lambda) value for
        each atomic species.
    :param nmax: dict containing the maximum radial channel (n) value for each
        combination of atomic species and o3_lambda value.
    :param basis_set: metatensor Labels object containing the basis set definition. This
        can be passed as an alternative to the `lmax` and `nmax` arguments.

    :return array: :py:class:`np.ndarray` vector of coefficients converted from
        TensorMap format, of shape (N,), where N is the number of basis functions the
        electron density is expanded onto.
    """
    if basis_set is not None:
        assert (
            lmax is None and nmax is None
        ), "Must either specify `basis_set` or `lmax` and `nmax`, not both."
        lmax, nmax = labels_basis_def_to_lmax_nmax_dicts(basis_set)

    assert (
        lmax is not None and nmax is not None
    ), "Must specify `lmax` and `nmax`, or just `basis_set`."

    # Check the samples names and get the structure index
    assert coeff_vector.sample_names == ["system", "atom"]

    # Loop over the blocks and split up the values tensors
    coeff_dict = {}
    for key, block in coeff_vector.items():
        o3_lambda, _, a = key
        symbol = system.atomic_number_to_atomic_symbol(a)
        tmp_dict = {}

        # Store the block values in a dict by atom index
        for atom_idx in np.unique(block.samples["atom"]):
            atom_idx_mask = block.samples["atom"] == atom_idx
            # Get the array of values for this atom, of species `symbol` and
            # `o3_lambda`` value The shape of this array is (1, 2*o3_lambda+1,
            # nmax[(symbol, o3_lambda)]
            atom_arr = block.values[atom_idx_mask]
            assert atom_arr.shape == (1, 2 * o3_lambda + 1, nmax[(symbol, o3_lambda)])
            # Reshape to a flatten array and store. IMPORTANT: Fortran order
            atom_arr = np.reshape(atom_arr, (-1,), order="F")
            tmp_dict[atom_idx] = atom_arr
        coeff_dict[(o3_lambda, symbol)] = tmp_dict

    # Combine the individual arrays into a single flat vector
    # Loop over the atomic species in the order given in `frame`
    coeffs = np.array([])
    for atom_i, symbol in enumerate(system.get_symbols(frame)):
        if symbol not in lmax:
            lmax[symbol] = -1
        for o3_lambda in range(lmax[symbol] + 1):
            coeffs = np.append(coeffs, coeff_dict[(o3_lambda, symbol)][atom_i])

    return coeffs


def labels_basis_def_to_lmax_nmax_dicts(
    basis_set: Union[metatensor.Labels, metatensor.torch.Labels],
) -> Tuple[dict]:
    """
    Converts a metatensor Labels object containing the basis set definition, with
    dimensions ["o3_lambda", "center_type", "nmax"]:

        Labels(
            o3_lambda   center_type     nmax
                0            6           11
                1            6           9
                2            6           7
                            ...
        )

    to 2 dictionaries of the form:

        lmax = {"C": 2, "H": 3}
        nmax = {(0, "C"): 11, (1, "C"): 11, ...}
    """
    lmax = {}
    nmax = {}
    for o3_lambda, center_type, nmax_val in basis_set.values:
        center_symbol = system.atomic_number_to_atomic_symbol(center_type)
        if center_symbol not in lmax:
            lmax[center_symbol] = int(o3_lambda)
        else:
            if lmax[center_symbol] < int(o3_lambda):
                lmax[center_symbol] = int(o3_lambda)
        nmax[(center_symbol, int(o3_lambda))] = int(nmax_val)

    return lmax, nmax


# ===== Functions to test conversions =====


def test_coeff_vector_conversion(
    frame: Frame,
    lmax: dict,
    nmax: dict,
    coeffs_flat: np.ndarray,
    coeffs_tm: metatensor.TensorMap,
    structure_idx: Optional[int] = None,
    print_level: int = 0,
) -> bool:
    """
    Tests that the TensorMap has been constructed correctly from the raw coefficients
    vector.
    """
    # Define some useful variables
    n_atoms = len(frame.atoms)
    species_symbols = np.array(system.get_symbols(frame))

    # Pick a random atom index and find its chemical symbol
    rng = np.random.default_rng()
    atom_i = rng.integers(n_atoms)
    symbol = species_symbols[atom_i]

    # Pick a random o3_lambda, n, and o3_mu
    o3_lambda = rng.integers(lmax[symbol] + 1)
    n = rng.integers(nmax[(symbol, o3_lambda)])
    o3_mu = rng.integers(-o3_lambda, o3_lambda + 1)
    if print_level > 0:
        print(
            "Atom:",
            atom_i,
            symbol,
            "o3_lambda:",
            o3_lambda,
            "n:",
            n,
            "o3_mu:",
            o3_mu,
        )

    # Get the flat index + value of this basis function coefficient in the flat array
    flat_index = _get_flat_index(
        species_symbols, lmax, nmax, atom_i, o3_lambda, n, o3_mu
    )
    raw_elem = coeffs_flat[flat_index]
    if print_level > 0:
        print("Raw array: idx", flat_index, "coeff", raw_elem)

    # Pick out this element from the TensorMap
    tm_block = coeffs_tm.block(
        o3_lambda=o3_lambda,
        center_type=system.atomic_symbol_to_atomic_number(symbol),
    )
    if structure_idx is None:
        s_idx = tm_block.samples.position(
            metatensor.Labels(names=["atom"], values=np.array([[atom_i]]))[0]
        )
    else:
        s_idx = tm_block.samples.position(
            metatensor.Labels(
                names=["system", "atom"], values=np.array([[structure_idx, atom_i]])
            )[0]
        )
    c_idx = tm_block.components[0].position(
        metatensor.Labels(names=["o3_mu"], values=np.array([[o3_mu]]))[0]
    )
    p_idx = tm_block.properties.position(
        metatensor.Labels(names=["n"], values=np.array([[n]]))[0]
    )

    tm_elem = tm_block.values[s_idx][c_idx][p_idx]
    if print_level > 0:
        print("TensorMap: idx", (s_idx, c_idx, p_idx), "coeff", tm_elem)

    return np.isclose(raw_elem, tm_elem)


def test_overlap_matrix_conversion(
    frame: Frame,
    lmax: dict,
    nmax: dict,
    overlap_numpy: np.ndarray,
    overlap_mts: metatensor.TensorMap,
    structure_idx: Optional[int] = None,
    off_diags_dropped: bool = False,
    print_level: int = 0,
    onsite: bool = False,
) -> bool:
    """
    Tests that the TensorMap has been constructed correctly from the raw overlap matrix.
    """
    # Define some useful variables
    n_atoms = len(frame.atoms)
    species_symbols = np.array(system.get_symbols(frame))

    # Pick 2 random atom indices and find their chemical symbols
    # and define their species symbols
    rng = np.random.default_rng()

    # Pick atom idxs
    i_1, i_2 = rng.integers(n_atoms), rng.integers(n_atoms)
    if onsite:
        i_2 = i_1

    if off_diags_dropped:  # a_1 <= a_2 must hold
        atomic_nums = [
            system.atomic_symbol_to_atomic_number(sym) for sym in species_symbols
        ]
        a_1, a_2 = atomic_nums[i_1], atomic_nums[i_2]

        if a_1 > a_2:  # swap
            a_1, a_2 = a_2, a_1
            i_1, i_2 = i_2, i_1

    symbol_1, symbol_2 = species_symbols[i_1], species_symbols[i_2]

    # Pick pairs of random o3_lambda
    l_1 = rng.integers(lmax[symbol_1] + 1)
    if off_diags_dropped:  # ensure l_1 <= l_2
        l_2 = rng.integers(l_1, lmax[symbol_2] + 1)
    else:
        l_2 = rng.integers(lmax[symbol_2] + 1)

    # Pick random pairs of n and o3_mu based on the o3_lambda values
    n_1, n_2 = rng.integers(nmax[(symbol_1, l_1)]), rng.integers(nmax[(symbol_2, l_2)])
    m_1, m_2 = rng.integers(-l_1, l_1 + 1), rng.integers(-l_2, l_2 + 1)

    if print_level > 0:
        print("Atom 1:", i_1, symbol_1, "l_1:", l_1, "n_1:", n_1, "m_1:", m_1)
        print("Atom 2:", i_2, symbol_2, "l_2:", l_2, "n_2:", n_2, "m_2:", m_2)

    # Get the flat row and column indices for this matrix element
    row_idx = _get_flat_index(species_symbols, lmax, nmax, i_1, l_1, n_1, m_1)
    col_idx = _get_flat_index(species_symbols, lmax, nmax, i_2, l_2, n_2, m_2)
    raw_elem = overlap_numpy[row_idx][col_idx]

    # Check that the matrix element is symmetric
    assert np.isclose(raw_elem, overlap_numpy[col_idx][row_idx])

    if print_level > 0:
        print("Raw matrix: idx", (row_idx, col_idx), "coeff", raw_elem)

    # Pick out this matrix element from the TensorMap
    # Start by extracting the block.
    if onsite:
        tm_block = overlap_mts.block(
            o3_lambda_1=l_1,
            o3_lambda_2=l_2,
            center_type=system.atomic_symbol_to_atomic_number(symbol_1),
        )
    else:
        tm_block = overlap_mts.block(
            o3_lambda_1=l_1,
            o3_lambda_2=l_2,
            center_1_type=system.atomic_symbol_to_atomic_number(symbol_1),
            center_2_type=system.atomic_symbol_to_atomic_number(symbol_2),
        )

    # Define the samples, components, and properties indices for the TensorBlock
    if structure_idx is None:
        s_idx = tm_block.samples.position((i_1,))
    else:
        s_idx = tm_block.samples.position((structure_idx, i_1))
    c_idx_1 = tm_block.components[0].position((m_1,))
    c_idx_2 = tm_block.components[1].position((n_1,))

    # Full and onsite overlaps differ in that the latter has no atom_2 index
    if onsite:
        c_idx_3 = tm_block.components[2].position((n_2,))
        p_idx = tm_block.properties.position((m_2,))
        tm_elem = tm_block.values[s_idx][c_idx_1][c_idx_2][c_idx_3][p_idx]
        if print_level > 0:
            print(
                "TensorMap: idx",
                (s_idx, c_idx_1, c_idx_2, c_idx_3, p_idx),
                "coeff",
                tm_elem,
            )
    else:
        c_idx_3 = tm_block.components[2].position((n_2,))
        c_idx_4 = tm_block.components[3].position((m_2,))
        p_idx = tm_block.properties.position((i_2,))
        tm_elem = tm_block.values[s_idx][c_idx_1][c_idx_2][c_idx_3][c_idx_4][p_idx]
        if print_level > 0:
            print(
                "TensorMap: idx",
                (s_idx, c_idx_1, c_idx_2, c_idx_3, c_idx_4, p_idx),
                "coeff",
                tm_elem,
            )

    return np.isclose(raw_elem, tm_elem)


# ===== Converting mts.core <--> mts.torch backends


def mts_tensormap_torch_to_core(tensor: torch.ScriptObject) -> metatensor.TensorMap:
    return metatensor.TensorMap(
        keys=mts_labels_torch_to_core(tensor.keys),
        blocks=[mts_tensorblock_torch_to_core(block) for block in tensor],
    )


def mts_tensorblock_torch_to_core(block: torch.ScriptObject) -> metatensor.TensorBlock:
    return metatensor.TensorBlock(
        values=np.array(block.values),
        samples=mts_labels_torch_to_core(block.samples),
        components=[mts_labels_torch_to_core(c) for c in block.components],
        properties=mts_labels_torch_to_core(block.properties),
    )


def mts_labels_torch_to_core(labels: torch.ScriptObject) -> metatensor.Labels:
    return metatensor.Labels(labels.names, values=np.array(labels.values))


def mts_tensormap_core_to_torch(
    tensor: metatensor.TensorMap,
) -> metatensor.torch.TensorMap:
    return metatensor.torch.TensorMap(
        keys=mts_labels_core_to_torch(tensor.keys),
        blocks=[mts_tensorblock_core_to_torch(block) for block in tensor],
    )


def mts_tensorblock_core_to_torch(
    block: metatensor.TensorBlock,
) -> metatensor.torch.TensorBlock:
    return metatensor.torch.TensorBlock(
        values=torch.tensor(block.values),
        samples=mts_labels_core_to_torch(block.samples),
        components=[mts_labels_core_to_torch(c) for c in block.components],
        properties=mts_labels_core_to_torch(block.properties),
    )


def mts_labels_core_to_torch(labels: metatensor.Labels) -> metatensor.torch.Labels:
    return metatensor.torch.Labels(labels.names, values=torch.tensor(labels.values))
