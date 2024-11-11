"""
Module to handle XYZ files and their conversion to :py:class:`chemfiles.Frame` objects.
"""

from os.path import exists
from typing import List, Optional, Union

import ase
import metatensor
import metatensor.torch
import numpy as np
import torch
import vesin
from ase.geometry.analysis import Analysis
from chemfiles import Atom, Frame, Trajectory
from metatensor.torch.atomistic import System

from rholearn.utils import ATOMIC_NUMBERS_TO_SYMBOLS, ATOMIC_SYMBOLS_TO_NUMBERS
from rholearn.utils._dispatch import int_array


def read_frames_from_xyz(
    xyz_path: str, idxs: Optional[List[int]] = None
) -> List[Frame]:
    """
    Reads a .xyz file and returns a list of chemfiles.Frames.

    If ``index`` is passed, only the frames indexed by the list will be returned.
    """
    assert isinstance(xyz_path, str), f"Invalid path: {xyz_path}. Must be a string."
    if not exists(xyz_path):
        raise FileNotFoundError(f"File not found: {xyz_path}")
    with Trajectory(xyz_path, "r") as trajectory:
        if idxs is None:
            idxs = list(range(trajectory.nsteps))
        if isinstance(idxs, int):
            idxs = [idxs]
        assert isinstance(idxs, list)
        frames = [trajectory.read_step(i) for i in idxs]

    return frames


def atomic_symbol_to_atomic_number(symbol: str) -> int:
    """
    Converts the atomic symbol string to atomic number integer.

    If ``symbol`` is a standard chemical symbol, the atomic (proton) number Z is
    returned, according to the periodic table.

    If ``symbol`` is a chemical symbol suffixed with a positive integer, of the form
    "X_n", where X is the chemical symbol and n is the integer, the atomic number
    returned is given by ``(1000 * n) + Z, where Z is the atomic number of X.
    """
    if symbol in ATOMIC_SYMBOLS_TO_NUMBERS:
        return ATOMIC_SYMBOLS_TO_NUMBERS[symbol]

    standard_symbol, n = symbol.split("_")
    n = int(n)
    assert n >= 0, (
        f"Invalid atomic symbol: {symbol}. Must be of the form"
        " 'X_n', where X is the chemical symbol and n is a"
        " positive integer."
    )
    return (1000 * n) + ATOMIC_SYMBOLS_TO_NUMBERS[standard_symbol]


def atomic_number_to_atomic_symbol(number: int) -> str:
    """
    Converts the atomic number integer to atomic symbol string.

    If ``number`` corresponds to a proton number on the periodic table, the standard
    chemical symbol is returned.

    If ``number`` is greater than 1000, it is assumed to be a pseudo-species. Returned
    is a string of the form "X_n", where X is the standard chemical species of the
    element with proton number Z ``number % 1000``, and n is given by ``number //
    1000``.
    """
    assert number >= 0, f"Invalid atomic number: {number}. Must be a positive integer."
    if number in ATOMIC_NUMBERS_TO_SYMBOLS:
        return ATOMIC_NUMBERS_TO_SYMBOLS[number]

    standard_symbol = ATOMIC_NUMBERS_TO_SYMBOLS[number % 1000]
    n = number // 1000

    return standard_symbol + "_" + str(n)


def get_symbols(frame: Frame) -> List[str]:
    """
    Returns the list of atomic types in the frame.
    """
    return [atom.type for atom in frame.atoms]


def get_types(frame: Frame) -> List[int]:
    """
    Returns the list of atomic numbers in the frame.
    """
    return [atomic_symbol_to_atomic_number(atom.type) for atom in frame.atoms]


def frame_to_atomistic_system(
    frame: Union[Frame, List[Frame]],
    dtype: torch.dtype = torch.float64,
    device: torch.device = "cpu",
) -> System:
    """
    Converts a :py:class:`chemfiles.Frame` object (or list of them) to a
    :py:class:`metatensor.torch.atomistic.System` object.

    ``dtype`` and ``device`` are used to set the data type and device of the torch
    tensors in the resulting System object.
    """

    if isinstance(frame, list):
        return [frame_to_atomistic_system(f, dtype, device) for f in frame]

    return System(
        types=torch.tensor(get_types(frame), dtype=torch.int32, device=device),
        positions=torch.tensor(frame.positions, dtype=dtype, device=device),
        cell=torch.tensor(frame.cell.matrix, dtype=dtype, device=device),
        pbc=torch.tensor([not all(row == 0) for row in frame.cell.matrix])
    )


def chemfiles_frame_to_ase_frame(frame: Union[Frame, List[Frame]]) -> ase.Atoms:
    """
    Converts a :py:class:`chemfiles.Frame` to an :py:class:`ase.Atoms` object.
    """
    if isinstance(frame, list):
        return [chemfiles_frame_to_ase_frame(f) for f in frame]

    if all([d == 0 for d in frame.cell.lengths]):
        cell = None
        pbc = False
    else:
        cell = frame.cell.matrix
        pbc = True

    return ase.Atoms(
        symbols=get_symbols(frame),
        positions=frame.positions,
        cell=cell,
        pbc=pbc,
    )


def get_neighbor_list(
    frames: List[Frame],
    frame_idxs: List[int],
    cutoff: float,
    backend: str = "numpy",
) -> Union[metatensor.Labels, metatensor.torch.Labels]:
    """
    Computes the neighbour list for each frame in ``frames`` and returns a
    :py:class:`metatensor.Labels` object with dimensions "system", "atom_1",
    and "atom_2". Atom indices are traingular, such that "atom_1" <= "atom_2".

    Self terms are included by default.
    """
    # Assign backend
    if backend == "numpy":
        mts = metatensor
    elif backend == "torch":
        mts = metatensor.torch
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'numpy' or 'torch'.")

    # Initialise the neighbor list calculator
    nl = vesin.NeighborList(cutoff=cutoff, full_list=False)

    labels_values = []
    for A, frame in zip(frame_idxs, frames):

        # Compute the neighbor list
        if any([d == 0 for d in frame.cell.lengths]):
            box = np.zeros((3, 3))
            periodic = False
        else:
            box = frame.cell.matrix
            periodic = True

        i_list, j_list = nl.compute(
            points=frame.positions,
            box=box,
            periodic=periodic,
            quantities="ij",
        )

        # Now add in the self terms as vesin does not include them
        i_list = np.concatenate([i_list, np.arange(len(frame.positions), dtype=int)])
        j_list = np.concatenate([j_list, np.arange(len(frame.positions), dtype=int)])

        # Ensure i <= j
        new_i_list = []
        new_j_list = []
        for i, j in zip(i_list, j_list):
            if i < j:
                new_i_list.append(i)
                new_j_list.append(j)
            else:
                new_i_list.append(j)
                new_j_list.append(i)

        # Sort by i
        sort_idxs = np.argsort(new_i_list)
        new_i_list = np.array(new_i_list)[sort_idxs]
        new_j_list = np.array(new_j_list)[sort_idxs]

        # Add dimension for system index
        for i, j in zip(new_i_list, new_j_list):
            labels_values.append([A, i, j] if i <= j else [A, j, i])

    return mts.Labels(
        names=["system", "atom_1", "atom_2"],
        values=int_array(labels_values, backend=backend),
    )
