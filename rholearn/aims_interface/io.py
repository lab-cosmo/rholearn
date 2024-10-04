"""
For reading and writing files in the AIMS interface.
"""

import datetime
from os.path import exists, join
from typing import Dict, Optional

import numpy as np
from chemfiles import Atom, Frame, UnitCell

from rholearn.utils import system


def write_geometry(frame: Frame, write_dir: str) -> None:
    """
    Writes a FHI-aims geometry.in file to ``write_dir``.

    For any atom with type "X", "empty" is written to the start of the corresponding
    line in geometry.in. For all other atom types, the corresponding line in geometry.in
    starts with 'atom'.

    For instance, a molecule with types: ["H", "O", "O_1", "X"] will be written as:

        atom  x y z  H
        atom  x y z  O
        atom  x y z  O_1
        empty x y z  X

    Note: atom constraints are not yet implemented.
    """
    with open(join(write_dir, "geometry.in"), "w") as f:
        # Write header
        f.write(
            "#=======================================================\n"
            f"# Created using rholearn at {_timestamp()}\n"
            "#=======================================================\n"
        )
        # Write lattice vectors
        if hasattr(frame, "cell"):
            if not all([length == 0 for length in frame.cell.lengths]):
                for lattice_vector in frame.cell.matrix:
                    f.write(
                        f"lattice_vector"
                        f" {lattice_vector[0]:.8f}"
                        f" {lattice_vector[1]:.8f}"
                        f" {lattice_vector[2]:.8f}"
                        "\n"
                    )

        # Write atoms
        for i, atom in enumerate(frame.atoms):
            atom_or_empty = "empty" if atom.type == "X" else "atom"
            f.write(
                f"{atom_or_empty}"
                f" {frame.positions[i, 0]:.8f}"
                f" {frame.positions[i, 1]:.8f}"
                f" {frame.positions[i, 2]:.8f}"
                f" {atom.type}"
                "\n"
            )


def read_geometry(read_dir: str, fname: Optional[str] = "geometry.in") -> Frame:
    """
    Reads a FHI-aims "geometry.in" file (or alternatively ``fname``) from ``read_dir``
    and parses it into a :py:class:`chemfiles.Frame` object.

    Note: this function currently only parses cell parameters and atomic positions and
    symbols from "geometry.in".
    """
    with open(join(read_dir, fname), "r") as f:
        lines = f.readlines()

    # Parse cell and atom information
    cell = []
    positions = []
    symbols = []
    for line in lines:
        if line.startswith("#"):  # comment line
            continue

        if line.startswith("lattice_vector"):
            cell.append([float(x) for x in line.split()[1:4]])

        if line.startswith("atom") or line.startswith("empty"):
            positions.append([float(x) for x in line.split()[1:4]])
            symbols.append(line.split()[4])

    # Build the Frame object
    frame = Frame()
    frame.cell = UnitCell(cell)
    for symbol, position in zip(symbols, positions):
        frame.add_atom(Atom(name=symbol, type=symbol), position=position)

    return frame


def write_control(
    frame: Frame,
    write_dir: str,
    parameters: dict,
    species_defaults: str,
) -> None:
    """
    Writes a FHI-aims control.in file.

    ``species_defaults`` can be a path to a directory containing species defaults, or a
    dictionary of paths for each unique atomic number present in ``frame``.

    If the former, the species default files are assumed to have the following name
    structure:

        - "01_H_default" for atomic number 1, atomic symbol H
        - "02_He_default" for atomic number 2, atomic symbol He
        - ...
        - "1079_Au_1_default" for pseudo-atomic number 1079, pseudo-atomic symbol Au_1
    """
    with open(join(write_dir, "control.in"), "w") as f:
        # Write header
        f.write(
            "#=======================================================\n"
            f"# Created using rholearn at {_timestamp()}\n"
            "#=======================================================\n"
        )

        # Write parameters
        for key, value in parameters.items():
            if value is None:
                continue
            # Convert types of dict values to str, and put into
            # list for easier handling
            if not isinstance(value, list):
                if isinstance(value, bool):
                    value = ".true." if value else ".false."
                else:
                    value = str(value)
                value = [value]

            for val in value:
                if key == "output" and val in ["cube ri_fit", "cube total_density"]:
                    f.write(f"{key.ljust(40)} {val}\n")
                    if parameters.get("cubes") is not None:
                        f.write(f"{parameters['cubes']}\n")
                elif key == "cubes":
                    continue
                else:
                    f.write(f"{key.ljust(40)} {val}\n")

    # Parse the species defaults, accounting for masked atom types
    species_default_strings = {}
    unique_atomic_numbers = np.unique(system.get_types(frame))
    for atomic_number in unique_atomic_numbers:

        # Read the species default from directory. Filename should be, i.e.
        # "01_H_default" or "79_Au_default" for standard atomic types or
        # "1079_Au_1_default" for pseudo atomic types
        symbol = system.atomic_number_to_atomic_symbol(atomic_number)
        fname = str(atomic_number).zfill(2) + "_" + symbol + "_default"

        # Case 1: we have a species default for this species. Read and use.
        if exists(join(species_defaults, fname)):
            with open(join(species_defaults, fname), "r") as species_file:
                species_default_str = species_file.read()

        # Case 2: no species default - assume a pseudo atom type and create an empty
        # basis set definition
        else:
            std_atomic_number = atomic_number % 1000
            std_symbol = system.atomic_number_to_atomic_symbol(std_atomic_number)
            fname = str(std_atomic_number).zfill(2) + "_" + std_symbol + "_default"

            with open(join(species_defaults, fname), "r") as species_file:
                species_default_str = species_file.read()

            # Find the index of the line that says, i.e. "species     H"
            lines = species_default_str.split("\n")
            index = None
            for i, line in enumerate(lines):
                if line.strip().startswith("species") and std_symbol in line:
                    index = i
                    break

            # Replace the line with "species X_1 \n max_n_prodbas 0"
            if index is None:
                raise ValueError("cannot find species line in species default")

            lines[index] = f"species {std_symbol}_1 \n    max_n_prodbas    0"

            # Join the lines back into a single string
            species_default_str = "\n".join(lines)

        # Store the species default
        species_default_strings[atomic_number] = species_default_str

    # Now write each species default into control.in
    with open(join(write_dir, "control.in"), "a") as f:
        for atomic_number in unique_atomic_numbers:
            f.write(species_default_strings[atomic_number])
            f.write("\n")

    return


def get_control_parameters_for_frame(
    frame: Frame,
    base_settings: dict,
    extra_settings: dict,
    cube_settings: Optional[dict] = None,
) -> dict:
    """
    Returns the parameters needed to write the 'control.in' file for ``frame``, by
    combining the ``base_settings`` with the ``extra_settings``.

    The latter is for example, RI fit or RI rebuild settings.

    Cube edges if {"output": "cube ri_fit"} present in ``extra_settings``.
    """

    control_params = base_settings.copy()
    control_params.update({**extra_settings})

    # Add tailored cube edges if needed
    if ("cube ri_fit" in extra_settings.get("output", [])) or (
        "cube total_density" in extra_settings.get("output", [])
    ):
        if cube_settings is None:
            raise ValueError("Must pass `cube_settings` if 'cube' in `extra_settings`")
        if cube_settings.get("slab", False) is True:
            control_params.update(
                _get_aims_cube_edges_slab(
                    frame,
                    cube_settings.get("n_points"),
                    z_min=None,  # want cube file generated on full system
                    z_max=None,
                )
            )
        else:
            control_params.update(
                _get_aims_cube_edges(frame, cube_settings.get("n_points"))
            )

    return control_params


def _timestamp() -> str:
    """Return a timestamp string in format YYYY-MM-DD-HH:MM:SS."""
    return datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")


def _get_aims_cube_edges_slab(
    slab: Frame, n_points: tuple, z_min: float = None, z_max: float = None
) -> Dict[str, str]:
    """
    Returns FHI-aims keywords for specifying the cube file edges in control.in. The x
    and y edges are taken to be the lattice vectors of the slab.

    The edges of the total cube are calculated as follows, assuming that the slab is
    oriented along the z-axis. The x and y edges are given by the cell lengths along
    these axes.

    As the slab is also assumed to have a large vacuum region along the z-axis, the
    z-edge is calculated as the maximum bounding length of atomic positions, plus 5
    Angstrom either side. This is unless `z_min` and/or `z_max` are passed, which
    override the max/min z-coords of the atomic positions.

    Returned is a dict like:
        {"cubes": "cube origin 1.59 9.85 12.80 \n"
            + "cube edge 101 0.15 0.0 1.0 \n",
            + "cube edge 101 0.0 0.15 0.0 \n",
            + "cube edge 101 0.0 0.0 0.15 \n",
        }
    as required for writing a control.in using the ASE interface.
    """
    if np.all(slab.cell.matrix == 0.0):
        raise ValueError("Slab must have periodic boundary conditions")

    # Check cell is cubic
    if not np.all(
        [
            param == 0
            for param in [
                slab.cell.matrix[0, 1],
                slab.cell.matrix[0, 2],
                slab.cell.matrix[1, 0],
                slab.cell.matrix[1, 2],
                slab.cell.matrix[2, 0],
                slab.cell.matrix[2, 1],
            ]
        ]
    ):
        raise ValueError(f"Cell not cubic: {slab.cell.matrix}")

    x_min = np.min(slab.positions[:, 0])
    x_max = np.max(slab.positions[:, 0])
    y_min = np.min(slab.positions[:, 1])
    y_max = np.max(slab.positions[:, 1])

    if z_min is None:
        z_min = np.min(slab.positions[:, 2]) - 5
    if z_max is None:
        z_max = np.max(slab.positions[:, 2]) + 5

    min_coord = np.array([x_min, y_min, z_min])
    max_coord = np.array([x_max, y_max, z_max])
    max_lengths = max_coord - min_coord

    # Take the x and y lattice vectors to be the length of the cube in these directions.
    # For the z-direction, take the bounding box of nuclear positions.
    max_lengths = [slab.cell.matrix[0, 0], slab.cell.matrix[1, 1], max_lengths[2]]
    center = (min_coord + max_coord) / 2

    return {
        "cubes": f"cube origin {np.round(center[0], 3)} "
        + f"{np.round(center[1], 3)} {np.round(center[2], 3)}"
        + "\n"
        + f"cube edge {n_points[0]} "
        + f"{np.round(max_lengths[0] / (n_points[0] - 1), 3)} 0.0 0.0"
        + "\n"
        + f"cube edge {n_points[1]} 0.0 "
        + f"{np.round(max_lengths[1] / (n_points[1] - 1), 3)} 0.0"
        + "\n"
        + f"cube edge {n_points[2]} 0.0 0.0 "
        + f"{np.round(max_lengths[2] / (n_points[2] - 1), 3)}"
        + "\n",
    }


def _get_aims_cube_edges(frame: Frame, n_points: tuple) -> Dict[str, str]:
    """
    Returns FHI-aims keywords for specifying the cube file edges in control.in.
    The x, y, z edges are taken to be the lattice vectors of the frame.

    Returned is a dict like:
        {"cubes": "cube origin 1.59 9.85 12.80 \n"
            + "cube edge 101 0.15 0.0 1.0 \n",
            + "cube edge 101 0.0 0.15 0.0 \n",
            + "cube edge 101 0.0 0.0 0.15 \n",
        }
    as required for writing a control.in using the ASE interface.
    """
    # Find the bounding box and center of the frame
    x_min = np.min(frame.positions[:, 0])
    x_max = np.max(frame.positions[:, 0])
    y_min = np.min(frame.positions[:, 1])
    y_max = np.max(frame.positions[:, 1])
    z_min = np.min(frame.positions[:, 2])
    z_max = np.max(frame.positions[:, 2])

    min_coord = np.array([x_min, y_min, z_min])
    max_coord = np.array([x_max, y_max, z_max])
    center = (min_coord + max_coord) / 2

    if np.all([length > 0 for length in frame.cell.lengths]):
        # check square cell
        if not np.all(
            [
                param == 0
                for param in [
                    frame.cell.matrix[0, 1],
                    frame.cell.matrix[0, 2],
                    frame.cell.matrix[1, 0],
                    frame.cell.matrix[1, 2],
                    frame.cell.matrix[2, 0],
                    frame.cell.matrix[2, 1],
                ]
            ]
        ):
            raise ValueError(f"Cell not square: {frame.cell.matrix}")
        # take lattice vectors as cube edges
        max_lengths = [
            frame.cell.matrix[0, 0],
            frame.cell.matrix[1, 1],
            frame.cell.matrix[2, 2],
        ]
    else:
        # take bounding box as cube edges, plus 5 Angstrom
        max_lengths = (max_coord - min_coord) + np.array([5, 5, 5])

    return {
        "cubes": f"cube origin {np.round(center[0], 3)} "
        + f"{np.round(center[1], 3)} {np.round(center[2], 3)}"
        + "\n"
        + f"cube edge {n_points[0]} "
        + f"{np.round(max_lengths[0] / (n_points[0] - 1), 3)} 0.0 0.0"
        + "\n"
        + f"cube edge {n_points[1]} 0.0 "
        + f"{np.round(max_lengths[1] / (n_points[1] - 1), 3)} 0.0"
        + "\n"
        + f"cube edge {n_points[2]} 0.0 0.0 "
        + f"{np.round(max_lengths[2] / (n_points[2] - 1), 3)}"
        + "\n",
    }
