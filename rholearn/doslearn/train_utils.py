from typing import List, Optional
from chemfiles import Frame
import torch

from scipy.interpolate import CubicHermiteSpline

import metatensor.torch as mts


def evaluate_spline(spline_coefs: torch.Tensor, spline_positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ Evaluate splines on selected points .

    Args:
        spline_coefs ([tensor]): [Cubic Hermite Spline Coefficients] 
        spline_positions ([tensor]): [Spline Positions] 
        x ([tensor]): [Points to evaluate splines on]

    Returns:
        [tensor]: [Evaluated spline values]
    """

    interval = torch.round(spline_positions[1] - spline_positions[0], decimals = 4)
    x = torch.clamp(x, min = spline_positions[0], max = spline_positions[-1]- 0.0005)
    indexes = torch.floor((x - spline_positions[0])/interval).long()
    expanded_index = indexes.unsqueeze(dim=1).expand(-1,4,-1)
    x_1 = x - spline_positions[indexes]
    x_2 = x_1 * x_1
    x_3 = x_2 * x_1
    x_0 = torch.ones_like(x_1)
    x_powers = torch.stack([x_3, x_2, x_1, x_0]).permute(1,0,2)
    value = torch.sum(torch.mul(x_powers, torch.gather(spline_coefs, 2, expanded_index)), axis = 1) 
    return value


def spline_eigenenergies(
    frame: Frame,
    frame_idx: int,
    energies: List[float],
    sigma: float,
    min_energy: float,
    max_energy: float,
    interval: float,
    dtype: Optional[torch.dtype] = torch.float64,
) -> torch.Tensor:
    """
    Splines a list of list of eigenenergies for each k-point to a common grid.
    """
    # Store number of k-points and flatten eigenenergies
    n_kpts = len(energies)
    energies = torch.tensor(energies, dtype=dtype).flatten()

    # Create energy grid
    n_grid_points = int(torch.ceil(torch.tensor(max_energy - min_energy) / interval))
    x_dos = min_energy + torch.arange(n_grid_points) * interval

    # Compute normalization factors
    normalization = (1 / torch.sqrt(2 * torch.tensor(np.pi) * sigma ** 2) / len(frame.positions) / n_kpts).to(dtype)

    # Compute DOS at each energy grid point
    l_dos_E = torch.sum(torch.exp(-0.5*((x_dos - energies.view(-1, 1)) / sigma) ** 2), dim = 0) * 2 * normalization

    # Compute derivative of DOS at each energy grid point
    l_dos_E_deriv = torch.sum(torch.exp(-0.5 * ((x_dos - energies.view(-1, 1)) / sigma) ** 2) *
                                (-1 * ((x_dos - energies.view(-1, 1)) / sigma) ** 2), dim =0) * 2 * normalization
    
    # Compute spline interpolation and return
    splines = torch.tensor(CubicHermiteSpline(x_dos, l_dos_E, l_dos_E_deriv).c)
    splines = splines.reshape(1, *splines.shape)

    return mts.TensorMap(
        keys=mts.Labels.single(),
        blocks=[
            mts.TensorBlock(
                samples=mts.Labels(
                    ["system"], torch.tensor([frame_idx], dtype=torch.int64).reshape(-1, 1)
                ),
                components=[
                    mts.Labels(["coeffs"], torch.arange(4).reshape(-1, 1)),
                ],
                properties=mts.Labels(
                    ["point"],
                    torch.arange(n_grid_points - 1, dtype=torch.int64).reshape(-1, 1)
                ),
                values=splines,
            )
        ]
    )