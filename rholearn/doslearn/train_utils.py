from os.path import exists, join
from typing import List, Optional, Tuple

import metatensor.torch as mts
import torch
from chemfiles import Frame
from metatensor.torch.learn.data import IndexedDataset

from rholearn.aims_interface import parser


def get_dataset(
    frames: List[Frame],
    frame_idxs: List[int],
    model: torch.nn.Module,
    load_dir: callable,
    precomputed_descriptors: bool,
    dtype: Optional[torch.dtype],
    device: Optional[str],
) -> torch.nn.Module:
    """
    Builds a dataset for the given ``frames``, using the ``model`` to pre-compute and
    store descriptors.
    """
    if not isinstance(frame_idxs, list):
        frame_idxs = list(frame_idxs)

    # Descriptors
    if precomputed_descriptors:
        if not exists(load_dir(frame_idxs[0])):
            raise FileNotFoundError(
                f"Precomputed descriptors not found in {load_dir(frame_idxs[0])}"
            )
        descriptors = [
            mts.load(join(load_dir(A), "descriptor.npz")) for A in frame_idxs
        ]
    else:
        descriptors = model.compute_descriptor(
            frames=frames,
            frame_idxs=frame_idxs,
        )
        descriptors = [
            mts.slice(
                descriptors,
                axis="samples",
                selection=mts.Labels(["system"], torch.tensor([A]).reshape(-1, 1)),
            )
            for A in frame_idxs
        ]

    # Load splines
    splines = [
        mts.load(join(load_dir(A), "dos_spline.npz")).to(dtype=dtype, device=device)
        for A in frame_idxs
    ]

    # Load the energy references
    if model._energy_reference == "Fermi":
        energy_reference = [
            torch.tensor(
                [torch.load(join(load_dir(A), "e_fermi.pt"), weights_only=False)],
                dtype=dtype,
                device=device,
            )
            for A in frame_idxs
        ]
    else:
        assert model._energy_reference == "Hartree"
        energy_reference = [torch.tensor([0.0], dtype=dtype, device=device)] * len(frame_idxs)

    return IndexedDataset(
        sample_id=frame_idxs,
        frames=frames,
        energy_reference=energy_reference,
        descriptor=descriptors,
        splines=splines,
    )


def get_spline_positions(
    min_energy: float, max_energy: float, interval: float
) -> torch.Tensor:
    """
    Get spline positions for the given energy range and interval.
    """
    n_spline_points = int(torch.ceil(torch.tensor(max_energy - min_energy) / interval))
    return min_energy + torch.arange(n_spline_points) * interval


def evaluate_spline(
    spline_coefs: torch.Tensor, spline_positions: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate splines on selected points.

    :param spline_coefs: :py:class:`torch.Tensor` corresponding to the Cubic Hermite
        Spline Coefficients
    :param spline_positions: :py:class:`torch.Tensor` corresponding to the Spline
        Positions
    :param x: :py:class:`torch.Tensor` corresponding to the points to evaluate splines
        on.

    :return: :py:class:`torch.Tensor` evaluated spline values.
    """

    interval = torch.round(spline_positions[1] - spline_positions[0], decimals=4)
    x = torch.clamp(x, min=spline_positions[0], max=spline_positions[-1] - 0.0005)
    indexes = torch.floor((x - spline_positions[0]) / interval).long()
    expanded_index = indexes.unsqueeze(dim=1).expand(-1, 4, -1)
    x_1 = x - spline_positions[indexes]
    x_2 = x_1 * x_1
    x_3 = x_2 * x_1
    x_0 = torch.ones_like(x_1)
    x_powers = torch.stack([x_3, x_2, x_1, x_0]).permute(1, 0, 2)
    value = torch.sum(
        torch.mul(x_powers, torch.gather(spline_coefs, 2, expanded_index)), axis=1
    )
    return value


def t_get_mse(
    predicted_dos: torch.Tensor, true_dos: torch.Tensor, x_dos: torch.Tensor
) -> torch.Tensor:
    """Compute mean squared error between two Density of States ."""
    # Check if it contains one DOS sample or a collection of samples
    if len(predicted_dos.size()) > 1:
        mse = (torch.trapezoid((predicted_dos - true_dos) ** 2, x_dos, axis=1)).mean()
    else:
        mse = (torch.trapezoid((predicted_dos - true_dos) ** 2, x_dos, axis=0)).mean()
    return mse


def t_get_rmse(
    predicted_dos: torch.Tensor, true_dos: torch.Tensor, x_dos: torch.Tensor
) -> torch.Tensor:
    """Compute root mean squared error between two Density of States ."""
    # Check if it contains one DOS sample or a collection of samples
    if len(predicted_dos.size()) > 1:
        rmse = torch.sqrt(
            (torch.trapezoid((predicted_dos - true_dos) ** 2, x_dos, axis=1)).mean()
        )
    else:
        rmse = torch.sqrt(
            (torch.trapezoid((predicted_dos - true_dos) ** 2, x_dos, axis=0)).mean()
        )
    return rmse


def opt_rmse_spline(
    predicted_dos: torch.Tensor,
    x_dos: torch.Tensor,
    target_splines: torch.Tensor,
    spline_positions: torch.Tensor,
    n_epochs: int,
) -> Tuple[torch.Tensor]:
    """
    RMSE on optimal shift of energy axis. The optimal shift is obtained via grid search
    followed by gradient descent.
    """
    # Perform an initial grid search to reduce the number of epochs needed for gradient
    # descent
    optim_search_mse = []
    offsets = torch.arange(-2, 2, 0.1)
    with torch.no_grad():
        for offset in offsets:
            shifts = torch.zeros(predicted_dos.shape[0]) + offset
            shifted_target = evaluate_spline(
                target_splines, spline_positions, x_dos + shifts.view(-1, 1)
            )
            loss_i = ((predicted_dos - shifted_target) ** 2).mean(dim=1)
            optim_search_mse.append(loss_i)
        optim_search_mse = torch.vstack(optim_search_mse)
        min_index = torch.argmin(optim_search_mse, dim=0)
        optimal_offset = offsets[min_index]

    # Fine tuning with gradient descent
    offset = optimal_offset
    shifts = torch.nn.parameter.Parameter(offset.float())
    opt_adam = torch.optim.Adam([shifts], lr=1e-2)
    best_shifts = shifts.clone()

    shifted_target = evaluate_spline(
        target_splines, spline_positions, x_dos + shifts.view(-1, 1)
    ).detach()
    each_loss = ((predicted_dos - shifted_target) ** 2).mean(dim=1).float()
    best_error = each_loss

    for _ in range(n_epochs):
        shifted_target = evaluate_spline(
            target_splines, spline_positions, x_dos + shifts.view(-1, 1)
        ).detach()

        def closure():
            opt_adam.zero_grad()
            shifted_target = evaluate_spline(
                target_splines, spline_positions, x_dos + shifts.view(-1, 1)
            )
            loss_i = ((predicted_dos - shifted_target) ** 2).mean()
            loss_i.backward(gradient=torch.tensor(1), inputs=shifts)
            return loss_i

        opt_adam.step(closure)

        with torch.no_grad():
            shifted_target = evaluate_spline(
                target_splines, spline_positions, x_dos + shifts.view(-1, 1)
            ).detach()
            each_loss = ((predicted_dos - shifted_target) ** 2).mean(dim=1).float()
            index = each_loss < best_error
            best_error[index] = each_loss[index].clone()
            best_shifts[index] = shifts[index].clone()

    # Evaluate
    optimal_shift = best_shifts
    shifted_target = evaluate_spline(
        target_splines, spline_positions, x_dos + optimal_shift.view(-1, 1)
    )
    rmse = t_get_rmse(predicted_dos, shifted_target, x_dos)
    return rmse, optimal_shift


def opt_mse_spline(
    predicted_dos: torch.Tensor,
    x_dos: torch.Tensor,
    target_splines: torch.Tensor,
    spline_positions: torch.Tensor,
    n_epochs: int,
) -> Tuple[torch.Tensor]:
    """
    MSE on optimal shift of energy axis. The optimal shift is obtained via grid search
    followed by gradient descent.
    """
    # Perform an initial grid search to reduce the number of epochs needed for gradient
    # descent
    optim_search_mse = []
    offsets = torch.arange(-2, 2, 0.1)
    with torch.no_grad():
        for offset in offsets:
            shifts = torch.zeros(predicted_dos.shape[0]) + offset
            shifted_target = evaluate_spline(
                target_splines, spline_positions, x_dos + shifts.view(-1, 1)
            )
            loss_i = ((predicted_dos - shifted_target) ** 2).mean(dim=1)
            optim_search_mse.append(loss_i)
        optim_search_mse = torch.vstack(optim_search_mse)
        min_index = torch.argmin(optim_search_mse, dim=0)
        optimal_offset = offsets[min_index]

    # Fine tuning with gradient descent
    offset = optimal_offset
    shifts = torch.nn.parameter.Parameter(offset.float())
    opt_adam = torch.optim.Adam([shifts], lr=1e-2)
    best_shifts = shifts.clone()

    shifted_target = evaluate_spline(
        target_splines, spline_positions, x_dos + shifts.view(-1, 1)
    ).detach()
    each_loss = ((predicted_dos - shifted_target) ** 2).mean(dim=1).float()
    best_error = each_loss

    for _ in range(n_epochs):
        shifted_target = evaluate_spline(
            target_splines, spline_positions, x_dos + shifts.view(-1, 1)
        ).detach()

        def closure():
            opt_adam.zero_grad()
            shifted_target = evaluate_spline(
                target_splines, spline_positions, x_dos + shifts.view(-1, 1)
            )
            loss_i = ((predicted_dos - shifted_target) ** 2).mean()
            loss_i.backward(gradient=torch.tensor(1), inputs=shifts)
            return loss_i

        opt_adam.step(closure)

        with torch.no_grad():
            shifted_target = evaluate_spline(
                target_splines, spline_positions, x_dos + shifts.view(-1, 1)
            ).detach()
            each_loss = ((predicted_dos - shifted_target) ** 2).mean(dim=1).float()
            index = each_loss < best_error
            best_error[index] = each_loss[index].clone()
            best_shifts[index] = shifts[index].clone()
    # Evaluate
    optimal_shift = best_shifts
    shifted_target = evaluate_spline(
        target_splines, spline_positions, x_dos + optimal_shift.view(-1, 1)
    )
    mse = t_get_mse(predicted_dos, shifted_target, x_dos)
    return mse, shifted_target, optimal_shift
