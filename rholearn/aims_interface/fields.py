"""
Module for computing quantities from scalar fields output by FHI-aims, i.e. 2D arrays of
grid coordinates and scalar values.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np


def field_absolute_error(
    input: np.ndarray,
    target: np.ndarray,
    grid: np.ndarray,
    already_sorted: bool = False,
) -> Tuple[float]:
    """
    Calculates and returns:

    - the integrated absolute error between the between the ``input`` and ``target``
        fields. This is calculated as the absolute error at each grid point, integrated
        over all space. Integration is performed numerically by taking the dot product
        of the error field at each grid point with the integration weight (tabulated
        partition function) in ``grid``.
    - the normalization factor: the ``target`` field, integrated over all space as
        above.

    All of ``input``, ``target``, and ``grid`` are arrays of shape (N_points, 4), where
    the first 3 columns at the xyz coordinates of each grid point, and final column is
    the value of the field at that point, or the integration weight in the case of
    ``grid``.
    """
    if not (
        np.all(input[:, :3] == target[:, :3]) and np.all(target[:, :3] == grid[:, :3])
    ):

        if already_sorted:
            raise ValueError(
                "grid point coordinates are not equivalent between"
                " input, target, and grid scalar fields"
            )

        else:  # sort the grid points and re-call the function
            input = sort_field_by_grid_points(input)
            target = sort_field_by_grid_points(target)
            grid = sort_field_by_grid_points(grid)

            return field_absolute_error(
                input, target, grid, already_sorted=True
            )

    abs_error = np.dot(np.abs(input[:, 3] - target[:, 3]), grid[:, 3])
    normalization = np.dot(target[:, 3], grid[:, 3])

    return abs_error, normalization


def field_squared_error(
    input: np.ndarray,
    target: np.ndarray,
    grid: np.ndarray,
    already_sorted: bool = False,
) -> Tuple[float]:
    """
    Calculates and returns:

        - the integrated squared error between the between the ``input`` and ``target``
          fields. This is calculated as the squared error at each grid point, integrated
          over all space. Integration is performed numerically by taking the dot product
          of the error field at each grid point with the integration weight (tabulated
          partition function) in ``grid``.
        - the normalization factor: the ``target`` field, integrated over all space as
          above.

    All of ``input``, ``target``, and ``grid`` are arrays of shape (N_points, 4), where
    the first 3 columns at the xyz coordinates of each grid point, and final column is
    the value of the field at that point, or the integration weight in the case of
    ``grid``.
    """
    if not (
        np.all(input[:, :3] == target[:, :3]) and np.all(target[:, :3] == grid[:, :3])
    ):

        if already_sorted:
            raise ValueError(
                "grid point coordinates are not equivalent between"
                " input, target, and grid scalar fields"
            )

        else:  # sort the grid points and re-call the function
            input = sort_field_by_grid_points(input)
            target = sort_field_by_grid_points(target)
            grid = sort_field_by_grid_points(grid)

            return field_squared_error(
                input, target, grid, already_sorted=True
            )

    squared_error = np.dot((input[:, 3] - target[:, 3]) ** 2, grid[:, 3])
    normalization = np.dot(target[:, 3], grid[:, 3])

    return squared_error, normalization


def sort_field_by_grid_points(field: Union[str, np.ndarray]) -> np.ndarray:
    """
    Loads the scalar field from file and returns a 2D array sorted by the norm of the
    grid points.

    Assumes the file at `field_path` has four columns, corresponding to the x, y, z,
    coordinates of the grid points and the scalar field value at that grid point.
    """
    if isinstance(field, str):  # load from file
        field = np.loadtxt(
            field,
            dtype=[
                ("x", np.float64),
                ("y", np.float64),
                ("z", np.float64),
                ("w", np.float64),
            ],
        )
    else:  # create a structured array
        field = field.ravel().view(
            dtype=[
                ("x", np.float64),
                ("y", np.float64),
                ("z", np.float64),
                ("w", np.float64),
            ]
        )
    field = np.sort(field, order=["x", "y", "z"])

    return np.array([[x, y, z, w] for x, y, z, w in field], dtype=np.float64)


def calculate_electrostatic_potential(
    rho: np.ndarray,
    grid: np.ndarray,
    eval_coords: Optional[np.ndarray] = None,
    eval_coords_size: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray]:
    """
    Calculate the electrostatic potential for a given charge density `rho` and grid
    points on which it is evaluated `grid` (including integration weights).

    A vector of points `eval_coords` can be provided as the points at which the
    potential is evaluated.

    If `eval_coords` is None, the grid is constructed by uniformly discretizing over the
    min and max coordinates in `grid` along each axis, according to the
    `eval_coord_size` parameter.

    Returned is a tuple of the Z evaluation coordinates, and the electrostatic potential
    V_z at the Z coordinate, averaged over the evaluation points in that Z plane.
    """
    # Check grid points match
    assert np.all(rho[:, :3] == grid[:, :3])

    # Weight the charge density by the grid weights
    rho = rho[:, 3] * grid[:, 3]
    coords = grid[:, :3]

    if eval_coords is None:
        if eval_coords_size is None:
            raise ValueError(
                "If `eval_coords` is None, `eval_coords_size` must be provided"
            )
        min_x, max_x = grid[:, 0].min(), grid[:, 0].max()
        min_y, max_y = grid[:, 1].min(), grid[:, 1].max()
        min_z, max_z = grid[:, 2].min(), grid[:, 2].max()
        X, Y, Z = [
            np.linspace(min_, max_, size)
            for min_, max_, size in zip(
                [min_x, min_y, min_z], [max_x, max_y, max_z], eval_coords_size
            )
        ]
    else:
        if eval_coords.shape[1] != 3:
            raise ValueError("eval_coords must have shape (N_pts, 3)")
        Z = eval_coords[:, 2]

    V = []
    for z in Z:
        # Get evaluation coordinates in the current z plane
        eval_coords = np.array([np.array([x, y, z]) for x in X for y in Y])

        # Calculate |r - r'| for all r in the `eval_coords` and all r' in `coords`
        norm_length = np.linalg.norm(
            np.abs(
                coords.reshape(coords.shape[0], 1, coords.shape[1]).repeat(
                    eval_coords.shape[0], axis=1
                )
                - eval_coords
            ),
            axis=2,
        )

        # Calculate the integrand for all r in `coords` and all r' in `eval_coords`
        integrand = rho.reshape(-1, 1) / norm_length

        # Calculate the integral, y summing over `coords`
        integral = integrand.sum(axis=0)

        # Mean over points in the Z plane
        V_z = integral.mean()
        V.append(V_z)

    return Z, np.array(V)
