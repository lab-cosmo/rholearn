import os
from os.path import exists, join
from typing import List, Optional, Tuple

import metatensor.torch as mts
import numpy as np
import torch
from chemfiles import Frame
from scipy.interpolate import CubicHermiteSpline


def evaluate_spline(
    spline_coefs: torch.Tensor, spline_positions: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Evaluate splines on selected points .

    Args:
        spline_coefs ([tensor]): [Cubic Hermite Spline Coefficients]
        spline_positions ([tensor]): [Spline Positions]
        x ([tensor]): [Points to evaluate splines on]

    Returns:
        [tensor]: [Evaluated spline values]
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


def spline_eigenenergies(
    frame: Frame,
    frame_idx: int,
    energies: List[float],
    reference: float,
    sigma: float,
    min_energy: float,
    max_energy: float,
    interval: float,
    dtype: Optional[torch.dtype] = torch.float64,
) -> torch.Tensor:
    """
    Splines a list of list of eigenenergies for each k-point to a common grid.
    """
    # Store number of k-points, flatten eigenenergies and apply energy reference adjustment
    n_kpts = len(energies)
    energies = torch.tensor(energies, dtype=dtype).flatten() - reference

    # Create energy grid
    n_grid_points = int(torch.ceil(torch.tensor(max_energy - min_energy) / interval))
    x_dos = min_energy + torch.arange(n_grid_points) * interval

    # Compute normalization factors
    normalization = (
        1
        / torch.sqrt(2 * torch.tensor(np.pi) * sigma**2)
        / len(frame.positions)
        / n_kpts
    ).to(dtype)

    # Compute DOS at each energy grid point
    l_dos_E = (
        torch.sum(
            torch.exp(-0.5 * ((x_dos - energies.view(-1, 1)) / sigma) ** 2), dim=0
        )
        * 2
        * normalization
    )

    # Compute derivative of DOS at each energy grid point
    l_dos_E_deriv = (
        torch.sum(
            torch.exp(-0.5 * ((x_dos - energies.view(-1, 1)) / sigma) ** 2)
            * (-1 * ((x_dos - energies.view(-1, 1)) / sigma) ** 2),
            dim=0,
        )
        * 2
        * normalization
    )

    # Compute spline interpolation and return
    splines = torch.tensor(CubicHermiteSpline(x_dos, l_dos_E, l_dos_E_deriv).c)
    splines = splines.reshape(1, *splines.shape)

    return mts.TensorMap(
        keys=mts.Labels.single(),
        blocks=[
            mts.TensorBlock(
                samples=mts.Labels(
                    ["system"],
                    torch.tensor([frame_idx], dtype=torch.int64).reshape(-1, 1),
                ),
                components=[
                    mts.Labels(["coeffs"], torch.arange(4).reshape(-1, 1)),
                ],
                properties=mts.Labels(
                    ["point"],
                    torch.arange(n_grid_points - 1, dtype=torch.int64).reshape(-1, 1),
                ),
                values=splines,
            )
        ],
    )


def create_subdir(ml_dir: str, name: str):
    """
    Creates a subdirectory at relative path:and returns path to training subdirectory at
    relative path:

        f"{`ml_dir`}/{`name`}"

    and returns a callable that points to further subdirectories indexed by the epoch
    number, i.e.:

        f"{`ml_dir`}/{`name`}/epoch_{`epoch`}"

    where the callable is parametrized by the variable `epoch`. This is used for
    creating checkpoint and evaluation directories.
    """

    def subdir(epoch):
        return join(ml_dir, name, f"epoch_{epoch}")

    if not exists(join(ml_dir, name)):
        os.makedirs(join(ml_dir, name))

    return subdir


def crossval_idx_split(
    frame_idxs: List[int], n_train: int, n_val: int, n_test: int, seed: int = 42
) -> Tuple[np.ndarray]:
    """Shuffles and splits ``frame_idxs``."""
    # Shuffle idxs using the standard seed (42)
    frame_idxs_ = frame_idxs.copy()
    np.random.default_rng(seed=seed).shuffle(frame_idxs_)

    # Take the test set as the first ``n_test`` idxs. This will be consistent regardless
    # of ``n_train`` and ``n_val``.
    test_id = frame_idxs_[:n_test]

    # Now shuffle the remaining idxs and draw the train and val idxs
    frame_idxs_ = frame_idxs_[n_test:]

    train_id = frame_idxs_[:n_train]
    val_id = frame_idxs_[n_train : n_train + n_val]

    assert len(np.intersect1d(train_id, val_id)) == 0
    assert len(np.intersect1d(train_id, test_id)) == 0
    assert len(np.intersect1d(val_id, test_id)) == 0

    return [train_id, val_id, test_id]


def t_get_mse(
    predicted_dos: torch.Tensor, true_dos: torch.Tensor, x_dos: torch.Tensor
) -> torch.Tensor:
    """Compute mean squared error between two Density of States ."""
    # Check if it contains one DOS sample or a collection of samples
    if len(a.size()) > 1:
        mse = (torch.trapezoid((predicted_dos - true_dos) ** 2, x_dos, axis=1)).mean()
    else:
        mse = (torch.trapezoid((predicted_dos - true_dos) ** 2, x_dos, axis=0)).mean()
    return mse


def t_get_rmse(
    predicted_dos: torch.Tensor, true_dos: torch.Tensor, x_dos: torch.Tensor
) -> torch.Tensor:
    """Compute root mean squared error between two Density of States ."""
    # Check if it contains one DOS sample or a collection of samples
    if len(a.size()) > 1:
        rmse = torch.sqrt(
            (torch.trapezoid((predicted_dos - true_dos) ** 2, x_dos, axis=1)).mean()
        )
    else:
        rmse = torch.sqrt(
            (torch.trapezoid((predicted_dos - true_dos) ** 2, x_dos, axis=0)).mean()
        )
    return rmse


def Opt_RMSE_spline(
    predicted_dos: torch.Tensor,
    x_dos: torch.Tensor,
    target_splines: torch.Tensor,
    spline_positions: torch.Tensor,
    n_epochs: int,
) -> Tuple[torch.Tensor]:
    """RMSE on optimal shift of energy axis. The optimal shift is obtained via grid search followed by gradient descent."""

    # Performing an initial Grid search to reduce the number of epochs needed for gradient descent
    all_shifts = []
    all_mse = []
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

    for i in range(n_epochs):
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

        mse = opt_adam.step(closure)

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


def Opt_MSE_spline(
    predicted_dos: torch.Tensor,
    x_dos: torch.Tensor,
    target_splines: torch.Tensor,
    spline_positions: torch.Tensor,
    n_epochs: int,
) -> Tuple[torch.Tensor]:
    """MSE on optimal shift of energy axis. The optimal shift is obtained via grid search followed by gradient descent."""

    # Performing an initial Grid search to reduce the number of epochs needed for gradient descent
    all_shifts = []
    all_mse = []
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

    for i in range(n_epochs):
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

        mse = opt_adam.step(closure)

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
    rmse = t_get_mse(predicted_dos, shifted_target, x_dos)
    return rmse, optimal_shift


def save_checkpoint(
    model: torch.nn.Module,
    best_state,
    alignment,
    best_alignment,
    parameters,
    optimizer,
    scheduler,
    chkpt_dir: str,
):
    """
    Saves model object, model state dict, best state dict, alignment, best alignment, training parameters, optimizer state dict, scheduler state dict,
    to file.
    """
    if not exists(chkpt_dir):  # create chkpoint dir
        os.makedirs(chkpt_dir)

    torch.save(model, join(chkpt_dir, "model.pt"))  # model obj
    torch.save(  # model state dict
        model.state_dict(),
        join(chkpt_dir, "model_state_dict.pt"),
    )

    torch.save(best_state, join(chkpt_dir, "best_model_state.pt"))  # best_state

    torch.save(best_state, join(chkpt_dir, "alignment.pt"))  # alignment

    torch.save(best_state, join(chkpt_dir, "best_alignment.pt"))  # best_alignment

    torch.save(parameters, join(chkpt_dir, "parameters.pt"))

    # Optimizer and scheduler
    torch.save(optimizer.state_dict(), join(chkpt_dir, "optimizer_state_dict.pt"))
    if scheduler is not None:
        torch.save(
            scheduler.state_dict(),
            join(chkpt_dir, "scheduler_state_dict.pt"),
        )
