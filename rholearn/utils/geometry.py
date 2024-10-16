from typing import List, Optional, Union

import ase
import ase.build
import ase.io
import numpy as np
import vesin

from ase.geometry.analysis import Analysis

import metatensor
import metatensor.torch

import chemfiles
from chemfiles import Atom, Frame, UnitCell

from rholearn.utils import system
from rholearn.utils._dispatch import int_array



def perturb_slab(slab, noise_scale_factor: float = 0.25):
    """
    Applies Gaussian noise to all lattice positions of the input `slab`, except the
    bottom layer and its passivating Hydrogens. Assumes that the only Hydrogen present
    in the slab are the ones passivating the bottom layer.

    Returns the perturbed slab, leaving the positions of the input `slab` unchanged.

    `noise_scale_factor` scales the Gaussian noise applied to the unconstrained atoms.
    """
    perturbed_slab = slab.copy()

    # Identify the indices of the Si atoms on the bottom layer
    z_min = min(
        [
            position
            for position, symbol in zip(
                slab.positions[:, 2], perturbed_slab.get_chemical_symbols()
            )
            if symbol == "Si"
        ]
    )
    idxs_layer0 = np.where(np.abs(perturbed_slab.positions[:, 2] - z_min) < 0.1)[0]
    idxs_hydrogen = np.array(
        [i for i, sym in enumerate(perturbed_slab.get_chemical_symbols()) if sym == "H"]
    )

    idxs_to_perturb = [
        i
        for i in range(perturbed_slab.get_global_number_of_atoms())
        if i not in idxs_layer0 and i not in idxs_hydrogen
    ]

    for idx in idxs_to_perturb:
        perturbed_slab.positions[idx] += (
            np.random.rand(*perturbed_slab.positions[idx].shape) * noise_scale_factor
        )

    return perturbed_slab


def perturb_slab_constrain_bottom_layers(
    slab, constrained_slice_distance: float = None, noise_scale_factor: float = 0.25
):
    """
    Applies Gaussian noise to all lattice positions of the input `slab`, except those
    with a z-coordinate equal in the range [`z_min`, `z_min +
    constrained_slice_distance`], where `z_min` is the minimum z-coordinate in the
    system, and `constrained_slice_distance` is the thickness of the slab in the
    z-direction that should be constrained.

    Returns the perturbed slab, leaving the positions of the input `slab` unchanged.

    `noise_scale_factor` scales the Gaussian noise applied to the unconstrained atoms.
    """
    perturbed_slab = slab.copy()

    if constrained_slice_distance is None:
        idxs_constrained = []
    else:
        # Identify the indices of the Si atoms on the bottom layer
        z_min = np.min(perturbed_slab.positions[:, 2])
        idxs_constrained = np.where(
            perturbed_slab.positions[:, 2] <= z_min + constrained_slice_distance
        )[0]
        idxs_constrained = np.array(idxs_constrained)

    # Perturb the unconstrained atoms
    idxs_to_perturb = [
        i
        for i in range(perturbed_slab.get_global_number_of_atoms())
        if i not in idxs_constrained
    ]

    for idx in idxs_to_perturb:
        perturbed_slab.positions[idx] += (
            np.random.rand(*perturbed_slab.positions[idx].shape) * noise_scale_factor
        )

    return perturbed_slab


def adsorb_h2_on_slab(slab, height: float, lattice_param: float = 5.431020511):
    """
    Randomly places a H2 molecule above the top layer of the `slab` at  height
    that is a random pertubation of the chosen `height`.
    """
    # Build the H2 molecule: bond length is 0.74 Angstrom
    h2_bond = 0.74
    h2 = ase.Atoms(
        "H2", positions=[[0, 0, 0], [h2_bond + 2 * np.random.rand(1)[0], 0, 0]]
    )

    ase.build.add_adsorbate(
        slab,
        h2,
        position=np.random.rand(2) * lattice_param,  # randomnly places on cell surrface
        height=1.5
        + np.random.rand(1)[0],  # height randomly perturbed from 1.5 above surface
    )

    return slab


def build_pristine_bulk_si(
    size: tuple,
    lattice_param: float = 5.431020511,
):
    bulk = ase.build.bulk("Si", crystalstructure="diamond", a=lattice_param)
    bulk = bulk.repeat((2, 2, 2))
    bulk.info["size"] = size
    bulk.info["lattice_param"] = lattice_param
    bulk.info["category"] = "pristine"

    return bulk


def build_pristine_passivated_si_slab(
    size: tuple, lattice_param: float = 5.431020511, passivate: bool = True
):
    slab = ase.build.diamond100("Si", size=size, vacuum=None, a=lattice_param)
    if passivate:
        slab = h_passivate_slab_underside(slab)

    slab.cell[2] = [0, 0, 100]

    slab.info["size"] = size
    slab.info["lattice_param"] = lattice_param
    slab.info["passivated"] = passivate
    slab.info["category"] = "pristine"
    slab.info["supercell"] = False

    return slab


def h_passivate_slab_underside(slab: ase.Atoms) -> ase.Atoms:

    # Identify the indices of the atoms on the bottom layer
    z_min = min(slab.positions[:, 2])
    idxs_layer0 = np.where(np.abs(slab.positions[:, 2] - z_min) < 0.1)[0]

    si_h_bond = 1.476
    h_si_h_angle = 170
    y_displacement = si_h_bond * np.arcsin(np.pi * h_si_h_angle / (2 * 360))
    z_displacement = si_h_bond * np.arccos(np.pi * h_si_h_angle / (2 * 360))

    # Add 2 Hydrogens to passivate each Si atom on the bottom layer
    for idx in idxs_layer0:
        # Calculate the positions of the Hydrogens
        h1_position = slab[idx].position - [0, +y_displacement, z_displacement]
        h2_position = slab[idx].position - [0, -y_displacement, z_displacement]

        # Add adatoms
        slab.append(ase.Atom("H", position=h1_position))
        slab.append(ase.Atom("H", position=h2_position))

    return slab


def make_slab_topside_dimer(
    slab: ase.Atoms,
    desired_dimer_distance: float = 2.24,
    buckled=None,
    aligned=True,
) -> ase.Atoms:
    # Identify the indices of the atoms on the top layer
    z_max = max(slab.positions[:, 2])
    idxs_top_layer = np.where(np.abs(slab.positions[:, 2] - z_max) < 0.1)[0]
    assert len(idxs_top_layer) == 2

    # There should be a 'square' of atoms on this layer. Move the y-coodinates closer
    # together.
    idx1, idx2 = idxs_top_layer[0], idxs_top_layer[1]

    current_dimer_distance = np.abs(slab[idx1].position[1] - slab[idx2].position[1])
    slab[idx1].position[1] += (current_dimer_distance - desired_dimer_distance) / 2
    slab[idx2].position[1] -= (current_dimer_distance - desired_dimer_distance) / 2

    if buckled:
        slab[idx1].position[1] += buckled
        slab[idx2].position[1] -= buckled
        slab[idx1].position[2] += buckled
        slab[idx2].position[2] -= buckled

    return slab


def make_slab_topside_dimer_pair(
    slab: ase.Atoms,
    desired_dimer_distance: float = 2.24,
    buckled=None,
    aligned=True,
) -> ase.Atoms:
    # Identify the indices of the atoms on the top layer
    z_max = max(slab.positions[:, 2])
    idxs_top_layer = np.where(np.abs(slab.positions[:, 2] - z_max) < 0.1)[0]
    assert len(idxs_top_layer) == 4

    # There should be a 'square' of atoms on this layer. Move the y-coodinates closer
    # together.
    dimer_pair_idxs = [
        (idxs_top_layer[0], idxs_top_layer[2]),
        (idxs_top_layer[1], idxs_top_layer[3]),
    ]
    if buckled is not None:
        if aligned:
            tilt_factors = [1, 1]
        else:
            tilt_factors = [1, -1]
    else:
        tilt_factors = [0, 0]
    for (idx1, idx2), tilt_factor in zip(dimer_pair_idxs, tilt_factors):
        current_dimer_distance = np.abs(slab[idx1].position[1] - slab[idx2].position[1])
        slab[idx1].position[1] += (current_dimer_distance - desired_dimer_distance) / 2
        slab[idx2].position[1] -= (current_dimer_distance - desired_dimer_distance) / 2

        if buckled:
            slab[idx1].position[1] += buckled
            slab[idx2].position[1] -= buckled
            slab[idx1].position[2] += buckled * tilt_factor
            slab[idx2].position[2] -= buckled * tilt_factor

    return slab


def add_virtual_nodes(frame: Union[Frame, List[Frame]], method: str) -> Frame:
    """
    Add virtual nodes to the frame using the specified method.

    ``method`` must be one of the following:

        - ``"molecule"``: add a virtual node in the middle of each bond in the frame.
          Bonds are calculated with the ASE geometry analysis tool.
        - ``"slab+adsorbate"``: add virtual nodes in the bonds of the adsorbate. Then,
          with a bounding box of the adsorbate excluded, a 3D grid of virtual nodes are
          added between the surface of the slab and the top of the adsorbate.
    """
    valid_methods = ["molecule", "slab+adsorbate"]
    assert method in valid_methods, f"Invalid method: {method}. Must be one of {valid_methods}."

    if method == "molecule":
        return add_virtual_nodes_in_bonds(frame)
    
    assert method == "slab+adsorbate"

    # First add virtual nodes in the bonds of the adsorbate
    adsorbate_atom_idxs = [i for i, position in enumerate(frame.positions) if position[2] > 0]
    frame = add_virtual_nodes_in_bonds(frame, adsorbate_atom_idxs)

    # Calculate the min and max coordinates of the adsorbate bounding box
    bbox_min_coords = np.min(frame.positions[adsorbate_atom_idxs], axis=0)
    bbox_max_coords = np.max(frame.positions[adsorbate_atom_idxs], axis=0)
    
    # Find the ounding box of the volume above the slab
    slab_atom_idxs = [i for i, position in enumerate(frame.positions) if position[2] <= 0]
    min_x = np.min(frame.positions[slab_atom_idxs], axis=0)[0]
    max_x = np.max(frame.positions[slab_atom_idxs], axis=0)[0]
    min_y = np.min(frame.positions[slab_atom_idxs], axis=0)[1]
    max_y = np.max(frame.positions[slab_atom_idxs], axis=0)[1]
    min_z = np.max(frame.positions[slab_atom_idxs], axis=0)[2]
    max_z = np.max(frame.positions[adsorbate_atom_idxs], axis=0)[2]

    # Assign nodes uniformly to the empty space around the adsorbate, above the slab
    # surface
    n_points = [5, 5, 5]
    new_symbols = list(system.get_symbols(frame))
    new_positions = list(frame.positions)
    for z in np.linspace(min_z, max_z, n_points[2]):
        for y in np.linspace(min_y, max_y, n_points[1]):
            for x in np.linspace(min_x, max_x, n_points[0]):

                # # Check this coordinate isn't inside the adsorbate bounding box
                # if all(np.array([x, y, z]) > bbox_min_coords) and all(
                #     np.array([x, y, z]) < bbox_max_coords
                # ):
                #     continue

                
                new_positions.append([x, y, z])
                new_symbols.append("X")

    new_frame = Frame()
    new_frame.cell = UnitCell(frame.cell.matrix)
    for symbol, position in zip(new_symbols, new_positions):
        new_frame.add_atom(Atom(name=symbol, type=symbol), position)

    return new_frame


    # Now add a mesh of virtual nodes over the surface of the slab
    # adsorbate_bounding_box = np.array(
    #     [
    #         [np.min(frame.positions[:, 0]), np.min(frame.positions[:, 1]), 0],
    #         [np.max(frame.positions[:, 0]), np.max(frame.positions[:, 1]), 0],
    #     ]
    # )
    


def add_virtual_nodes_in_bonds_adsorbate(frame: Union[Frame, List[Frame]]) -> Frame:

    pass

def add_virtual_nodes_in_bonds(
    frame: Union[Frame, List[Frame]], selected_atoms: Optional[List[int]] = None
) -> Frame:
    """
    Add a virtual node in the middle of each bond in the frame.

    Uses the ASE geometry analysis to get the bonds between atoms.

    ``selected_atoms`` can be used to specify a subset of atoms to consider when
    computing bonds.

    TODO: currently, this may not be deterministic as when virtual nodes are placed very
    close to each other, the one placed first (determined by the order of the atoms in
    the input frame) will be kept and the other discarded.
    """
    if isinstance(frame, list):
        return [add_virtual_nodes_in_bonds(f) for f in frame]
    
    assert isinstance(frame, chemfiles.frame.Frame), (
        f"Invalid frame type: {type(frame)}. Must be chemfiles.frame.Frame."
    )

    if selected_atoms is None:
        selected_atoms = list(range(len(frame.positions)))

    new_symbols = list(system.get_symbols(frame))
    new_positions = list(frame.positions)

    unique_symbols = list(set(system.get_symbols(frame)))

    analysis = Analysis(system.chemfiles_frame_to_ase_frame(frame))
    for symbol_i, symbol_1 in enumerate(unique_symbols):
        for symbol_2 in unique_symbols[symbol_i:]:
            # Get the bonds between the two chemical types and add an atom there
            for i, j in analysis.get_bonds(symbol_1, symbol_2, unique=True)[0]:

                if i not in selected_atoms or j not in selected_atoms:
                    continue

                new_position = (frame.positions[i] + frame.positions[j]) / 2

                # Check we're not placing virtual nodes on top of each other
                if np.any(np.linalg.norm(new_positions - new_position, axis=1) < 0.1):
                    continue

                new_positions.append(new_position)
                new_symbols.append("X")

    new_frame = Frame()
    new_frame.cell = UnitCell(frame.cell.matrix)
    for symbol, position in zip(new_symbols, new_positions):
        new_frame.add_atom(Atom(name=symbol, type=symbol), position)

    return new_frame


def get_neighbor_list(
    frames: List[Frame],
    frame_idxs: List[int],
    cutoff: float,
    backend: str = "numpy",
) -> Union[metatensor.Labels, metatensor.torch.Labels]:
    """
    Computes the neighbour list for each frame in ``frames`` and returns a
    :py:class:`metatensor.Labels` object with dimensions "system", "atom_1",
    and "atom_2". Atom indices are triangular, such that "atom_1" <= "atom_2".

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
