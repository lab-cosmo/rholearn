from os.path import join 
from typing import List, Optional
from chemfiles import Frame

import matplotlib.pyplot as plt
import numpy as np

import metatensor

from rholearn.utils import system


def plot_ovlp_by_separation(
    frames: List[Frame],
    overlaps: List[metatensor.TensorMap],
    save_dir: Optional[str] = None,
) -> None:
    """
    Plots the max and mean value of the overlap matrix between pairs of atoms as a
    function of their interatomic distance.

    Generates subplots for each pair of atom types in the system.
    """
    atom_types = set()
    for frame in frames:
        atom_types.update(system.get_types(frame))

    atom_types = {atom_type: i for i, atom_type in enumerate(sorted(atom_types))}
    
    fig, axes = plt.subplots(
        len(atom_types), len(atom_types), figsize=(2 * len(atom_types), 2 * len(atom_types)), sharex=True, sharey=True
    )

    for frame, overlap in zip(frames, overlaps):
        for (a1, a2), block in overlap.items():

            if a1 < a2:
                a1, a2 = a2, a1

            a1, a2 = int(a1), int(a2)

            ax = axes[atom_types[a1], atom_types[a2]]

            distances = [
                frame.distance(i1, i2)
                for (_, i1, i2) in block.samples
            ]
            ax.scatter(
                distances,
                [np.max(np.abs(matrix)) for matrix in block.values],
                marker=".",
                c="red",
            )
            ax.scatter(
                distances,
                [np.mean(np.abs(matrix)) for matrix in block.values],
                marker=".",
                c="blue",
            )
            ax.set_yscale("log")

    # Set the row and column labels
    for atom_type, i in atom_types.items():
        axes[i, 0].set_ylabel(f"r_ij, alpha = {atom_type}")
        axes[-1, i].set_xlabel(f"r_ij, alpha = {atom_type}")

    if save_dir is not None:
        plt.savefig(join(save_dir, "ovlp_by_separation.png"))

    return