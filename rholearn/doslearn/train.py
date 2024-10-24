from functools import partial

import torch

import metatensor.torch as mts
from metatensor.torch.learn.data import IndexedDataset, DataLoader, group_and_join

from rholearn.utils import system

from rholearn.aims_interface import io
from doslearn.model import SoapDosNet
import parser


def train():

    # Start training

    n_grid_points = int(torch.ceil(torch.tensor(max_energy - min_energy) / interval))
    x_dos = min_energy + torch.arange(n_grid_points) * interval

    # Define idxs
    frame_idxs = list(range(10))

    # Get frames
    frames = [
        read_geometry(aims_dir(A))
        for A in range(10)
    ]

    # Get atom types
    atom_types = set()
    for frame in frames:
        for type_ in system.get_types(frame):
            atom_types.update({type_})
    atom_types = list(atom_types)

    # Define out properties
    out_properties = [
        mts.Labels(
            ["point"],
            torch.arange(n_grid_points, dtype=torch.int64).reshape(-1, 1)
        )
        for _ in atom_types
    ]

    # Initialize model
    model = SoapDosNet(
        hypers,
        atom_types=atom_types,
        out_properties=out_properties,
        hidden_layer_widths=hidden_layer_widths,
        dtype=dtype,
        device=device,
    )

    # Compute descriptor
    descriptor = model.compute_descriptor(frames).to(dtype=dtype, device=device)
    descriptor = [
        mts.slice(descriptor, "samples", mts.Labels(["system"], torch.tensor([A]).reshape(-1, 1)))
        for A in frame_idxs
    ]

    # Parse eigenvalues
    eigvals = [
        parser.parse_eigenvalues(aims_dir(A)) for A in frame_idxs
    ]
    eigvals = [i[:2] for i in eigvals]

    # Compute splines
    splines = [
        parser.spline_eigenenergies(
            frame, frame_idx, energies, sigma=sigma, min_energy=min_energy, max_energy=max_energy, interval=interval
        )
        for frame, frame_idx, energies in zip(frames, frame_idxs, eigvals)
    ]

    # Crossval split
    train_idxs = list(range(7))
    test_idxs = list(range(7, 10))

    # Build datasets
    train_dset = IndexedDataset(
        sample_id=train_idxs,
        frames=[frames[i] for i in train_idxs],
        descriptor=[descriptor[i] for i in train_idxs],
        splines=[splines[i] for i in train_idxs],
    )

    test_dset = IndexedDataset(
        sample_id=test_idxs,
        frames=[frames[i] for i in test_idxs],
        descriptor=[descriptor[i] for i in test_idxs],
        splines=[splines[i] for i in test_idxs],
    )

    # Build dataloaders
    train_loader = DataLoader(
        train_dset,
        batch_size=2,
        shuffle=True, 
        collate_fn=partial(group_and_join, join_kwargs={"remove_tensor_name": True})
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=2,
        shuffle=True, 
        collate_fn=partial(group_and_join, join_kwargs={"remove_tensor_name": True})
    )
