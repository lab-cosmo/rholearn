from typing import Optional
import torch


class L1Loss(torch.nn.Module):
    """
    Computes the absolute error between two tensors, either point-wise or integrated
    along the energy range.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, x_dos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the absolute error between the ``input`` and ``target`` DOS.

        The tensors ``input`` and ``target`` must have the same 2D shape correspondind
        to (n_samples, n_grid).
        
        If ``x_dos=None``, a tensor with shape (n_samples, n_grid) is returned.

        If ``x_dos`` is passed, the error is integrated along the energy range. A tensor
        of shape (n_samples,) is returned.
        """
        loss = torch.abs(input - target)
        loss = torch.trapezoid(loss, x_dos, axis=1)
        return loss


class L2Loss(torch.nn.Module):
    """
    Computes the squared error between two tensors, either point-wise or integrated
    along the energy range.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, x_dos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the squared error between the ``input`` and ``target`` DOS.

        The tensors ``input`` and ``target`` must have the same 2D shape correspondind
        to (n_samples, n_grid).
        
        If ``x_dos=None``, a tensor with shape (n_samples, n_grid) is returned.

        If ``x_dos`` is passed, the error is integrated along the energy range. A tensor
        of shape (n_samples,) is returned.
        """
        loss = (input - target) ** 2
        loss = torch.trapezoid(loss, x_dos, axis=1)
        return loss
