from os.path import join 
from typing import List, Optional, Tuple, Union
from chemfiles import Frame

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter

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
        plt.savefig(join(save_dir, "ovlp_by_separation.png"), dpi=300, bbox_inches="tight")

    return


def plot_training(
    train_dir: callable, labels: List[str], save_dir: str, xlim: Optional[Tuple[float]] = None
) -> None:
    """
    Makes 3-panel subplots of the training loss, validation loss, and LR, parsed from
    train.log.

    ``train_dir`` must be a callable that points to the training directory when
    parametrized with each element in ``labels``. These labels are also used in the plot
    legend.
    """
    # Parse data from `train_dir`/outputs/train.log
    data_fields = [
        "epoch", "train_loss", "val_loss", "dt_train", "dt_val", "lr"
    ]
    results = {
        label: {field: [] for field in data_fields} 
        for label in labels
    }
    for label in labels:
        with open(join(train_dir(label), "outputs/train.log"), "r") as f:
            lines = f.readlines()

        for line in lines:
        
            line = line.split()
            if len(line) < 3:
                continue

            if line[2] != "epoch":
                continue

            # Parse and store data
            fields = line[2:][::2]
            data = line[3:][::2]
            for field, value in zip(fields, data):
                results[label][field].append(float(value))

    # Plot data
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    for label in labels:
        epochs = np.array(results[label]["epoch"])
        train_losses = np.array(results[label]["train_loss"])
        val_losses = np.array(results[label]["val_loss"])
        lrs = np.array(results[label]["lr"])
        if xlim is not None:
            epoch_mask = [i for i, epoch in enumerate(results[label]["epoch"]) if epoch >= xlim[0]]
            epochs = epochs[epoch_mask]
            train_losses = train_losses[epoch_mask]
            val_losses = val_losses[epoch_mask]
            lrs = lrs[epoch_mask]
        # Train loss
        axes[0].plot(
            epochs,
            train_losses,
            label=f"{label}",
        )
        # Val loss
        axes[1].plot(
            epochs,
            val_losses,
            label=f"{label}",
        )
        # LR
        axes[2].plot(
            epochs,
            lrs,
            label=f"{label}",
        )

    # Formatting
    [ax.set_xscale("log") for ax in axes];
    [ax.set_yscale("log") for ax in axes];
    axes[0].legend()
    axes[-1].set_xlabel("epoch")
    axes[0].set_ylabel("train_loss")
    axes[1].set_ylabel("val_loss")
    axes[2].set_ylabel("lr")

    # Save
    plt.savefig(join(save_dir, "training_curve.png"), dpi=300, bbox_inches="tight")

    return

def plot_val_loss_curve(
    train_dir: callable, labels: List[str], save_dir: str, x_axis: List[int],
) -> None:
    """
    Makes 3-panel subplots of the training loss, validation loss, and LR, parsed from
    train.log.

    ``train_dir`` must be a callable that points to the training directory when
    parametrized with each element in ``labels``. These labels are also used in the plot
    legend.
    """
    # Parse data from `train_dir`/outputs/train.log
    results = {
        label: {"best": None, "last": None}
        for label in labels
    }
    for label in labels:
        with open(join(train_dir(label), "outputs/train.log"), "r") as f:
            lines = f.readlines()

        val_losses = []
        for line in lines:
        
            line = line.split()
            if len(line) < 3:
                continue

            if line[2] != "epoch":
                continue

            # Parse and store data
            fields = line[2:][::2]
            data = line[3:][::2]
            for field, value in zip(fields, data):
                if field == "val_loss":
                    val_losses.append(float(value))

        results[label]["best"] = np.min(val_losses)
        results[label]["last"] = val_losses[-1]

    # Plot data
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(  # Best val loss
        x_axis,
        [results[label]["best"] for label in labels],
        label="best",
        marker=".",
    )
    ax.plot(  # Best val loss
        x_axis,
        [results[label]["last"] for label in labels],
        label="last",
        marker=".",
    )

    # Formatting
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N_train")
    ax.set_ylabel("val_loss")

    # Apply the custom locator and formatter to the x axis
    ax.xaxis.set_major_locator(LogLocator(base=2.0, numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(_log_formatter))

    # Save
    plt.savefig(join(save_dir, "val_loss_curve.png"), dpi=300, bbox_inches="tight")

    return


def plot_test_error_learning_curve(
    train_dir: callable, labels: List[str], save_dir: str, x_axis: List[int],
) -> None:
    """
    Makes 3-panel subplots of the training loss, validation loss, and LR, parsed from
    train.log.

    ``train_dir`` must be a callable that points to the training directory when
    parametrized with each element in ``labels``. These labels are also used in the plot
    legend.
    """
    # Parse data from `train_dir`/outputs/train.log
    field_names = ["abs_error", "norm", "nmae", "squared_error"]
    results = {
        label: {
            "ri": {field: [] for field in field_names},
            "scf": {field: [] for field in field_names}
        }
        for label in labels
    }
    for label in labels:
        with open(join(train_dir(label), "outputs/eval.log"), "r") as f:
            lines = f.readlines()

        for line in lines:
        
            line = line.split()
            if len(line) < 3:
                continue

            if line[2] != "system":
                continue

            # Parse and store data
            target = line[5]
            fields = line[6:][::2]
            data = line[7:][::2]
            for field, value in zip(fields, data):
                results[label][target][field].append(float(value))

    # Plot data
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(  # NMAE - RI
        x_axis,
        [np.mean(results[label]["ri"]["nmae"]) for label in labels],
        label="ML vs RI",
        marker=".",
    )
    ax.plot(  # NMAE - SCF
        x_axis,
        [np.mean(results[label]["scf"]["nmae"]) for label in labels],
        label="ML vs SCF",
        marker=".",
    )

    # Formatting
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N train")
    ax.set_ylabel("NMAE %")

    # Apply the custom locator and formatter to the x axis
    ax.xaxis.set_major_locator(LogLocator(base=2.0, numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(_log_formatter))

    # Save
    plt.savefig(join(save_dir, "test_error_curve.png"), dpi=300, bbox_inches="tight")

    return


def plot_pretrainer_val_losses(
    train_dir: callable,
    labels: Union[List[str], List[List[str]]],
    save_dir: str,
    x_axis: Union[List[int], List[List[int]]], 
    legend: List[str],
):
    """
    Extracts the validation loss of the pretrainer and plots them for each label in
    ``labels``.
    """
    fig, ax = plt.subplots()

    if isinstance(labels[0], str):
        labels = [labels]
        x_axis = [x_axis]

    for labels_, x_axis_, legend_ in zip(labels, x_axis, legend):

        val_losses = []
        for label in labels_:
            with open(join(train_dir(label), "outputs/train.log"), "r") as f:
                lines = f.readlines()

            for line in lines:

                line = line.split()
                if len(line) < 5:
                    continue

                if line[2:7] == "Validation loss of pretrained model:".split():

                    val_loss = float(line[7])
                    val_losses.append(val_loss)

    
        ax.plot(x_axis_, val_losses, marker=".", label=legend_)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N train")
    ax.set_ylabel("Pretrainer validation loss")
    ax.legend()

    # Apply the custom locator and formatter to the x axis
    ax.xaxis.set_major_locator(LogLocator(base=2.0, numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(_log_formatter))

    plt.savefig(
        join(save_dir, "pretrainer_val_losses.png"),
        dpi=300, 
        bbox_inches="tight"
    )

    return

def _log_formatter(x, pos):
    """Custom formatter function to display integers on x axis"""
    return f'{int(x)}'


def plot_slab_density_errors(
    aims_output_dir: str,
    surface_depth: Optional[float] = None,
    buffer_depth: Optional[float] = None,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    save_dir: Optional[str] = None,
    error: str = "nmae",
):
    """
    Plots the binned SCF and RI densities as a function of z-coordinate, and the error.
    """
    dir = "light_global"
    scf = np.loadtxt(join(aims_output_dir, "rho_scf.out"))
    ri = np.loadtxt(join(aims_output_dir, "rho_rebuilt_ri.out"))
    grid = np.loadtxt(join(aims_output_dir, "partition_tab.out"))
    assert np.all(scf[:, :2] == ri[:, :2])
    assert np.all(scf[:, :2] == grid[:, :2])

    # Define the midpoints of `nbins` bins across the z range
    nbins = 100
    if z_min is None:
        z_min = np.min(grid[:, 2])
    if z_max is None:
        z_max = np.min(grid[:, 2])
    bins = np.linspace(z_min, z_max, nbins)

    scfs = []
    ris = []
    errors = []
    for i in range(1, len(bins)):
        mask = (grid[:, 2] >= bins[i-1]) & (grid[:, 2] < bins[i])
        scf_density = scf[mask, 3]
        ri_density = ri[mask, 3]
        
        scfs.append((scf[mask, 3] * grid[mask, 3]).mean())
        ris.append((ri[mask, 3] * grid[mask, 3]).mean())

        abs_error = np.dot(np.abs(ri[mask, 3] - scf[mask, 3]), grid[mask, 3])
        if error == "nmae":
            normalization = np.dot(scf[mask, 3], grid[mask, 3])
            errors.append(100 * abs_error / normalization)
        else:
            assert error == "abs"
            errors.append(abs_error)

    # Plot
    fig, axes = plt.subplots(2, 1)
    axes[0].scatter(bins[1:], scfs, label="SCF density", marker=".")
    axes[0].scatter(bins[1:], ris, label="RI density", marker=".")
    axes[1].scatter(bins[1:], errors, label=f"{error}", marker=".")
    [ax.set_yscale("log") for ax in axes];
    [ax.legend() for ax in axes];

    # Region marking
    if surface_depth is not None:
        [ax.axvline(- surface_depth, color="green", linestyle="--") for ax in axes];
    if buffer_depth is not None:
        [ax.axvline(-surface_depth - buffer_depth, color="orange", linestyle="--") for ax in axes];

    if save_dir is not None:
        plt.savefig(join(save_dir, "slab_density_errors.png"))

    return    
