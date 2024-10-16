"""
Module to handle XYZ files and their conversion to :py:class:`chemfiles.Frame` objects.
"""

from os.path import exists
from typing import List, Optional, Union

import ase
import torch

from chemfiles import Frame, Trajectory
from metatensor.torch.atomistic import System

from rholearn.utils import ATOMIC_NUMBERS_TO_SYMBOLS, ATOMIC_SYMBOLS_TO_NUMBERS


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

