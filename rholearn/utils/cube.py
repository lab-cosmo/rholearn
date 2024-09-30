"""
Module containing the RhoCube class, which wraps the cube_tools.cube class from
https://github.com/funkymunkycool/Cube-Toolz. 

Allows reading and manipulation of cube files, with added functionality for generating
contour plots (i.e. for use in STM image generation).
"""

from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
from scipy.interpolate import CubicSpline

from chemfiles import Atom, Frame
import cube_tools


class RhoCube(cube_tools.cube):

    def __init__(self, file_path: str):
        super(RhoCube, self).__init__(file_path)
        self.file_path = file_path
        self.frame = self.frame()

    def frame(self) -> Frame:
        """
        Builds an ASE atoms object from the atomic positions and chemical
        symbols in the cube file.
        """
        frame = Frame()
        for symbol, position in zip(self.atoms, self.atomsXYZ):
            frame.add_atom(Atom(name=symbol, type=symbol), position=position)
        return frame

    def get_slab_slice(
        self, axis: int = 2, center_coord: float = None, thickness: float = 1.0
    ) -> np.ndarray:
        """
        Get a 2D grid of the cube data, sliced at the specified center
        coordinate of the specified axis (i.e. 0 =
        x, 1 = y, 2 = z), and summed over the thickness of the slab.

        For instance, passing `axis=2`, `center_coord=7.5`, and `thickness=1.0`,
        a 2D grid of cube data will be returned by summing over XY arrays of Z
        coordinates that are in the range 7.0 to 8.0.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Invalid axis")
        if not (
            (self.X[1] == self.X[2] == 0)
            and (self.Y[0] == self.Y[2] == 0)
            and (self.Z[0] == self.Z[1] == 0)
        ):
            raise ValueError("Can only handle X, Y, Z axis aligned cubes")

        # Define the min and max Z coordinate of the surface slab to be summed over
        slab_min = center_coord - (thickness / 2)
        slab_max = center_coord + (thickness / 2)

        keep_axes = [i for i in range(3) if i != axis]
        cube_sizes = [self.X, self.Y, self.Z]
        n_cubes = [self.NX, self.NY, self.NZ]

        # Return grid coordinates of the axes that are kept
        axis_a_coords = (
            np.arange(n_cubes[keep_axes[0]]) * cube_sizes[keep_axes[0]][keep_axes[0]]
        ) + self.origin[keep_axes[0]]
        axis_b_coords = (
            np.arange(n_cubes[keep_axes[1]]) * cube_sizes[keep_axes[1]][keep_axes[1]]
        ) + self.origin[keep_axes[0]]

        # Initialize the 2D grid of cube data to return
        axis_c_vals = np.zeros((n_cubes[keep_axes[0]], n_cubes[keep_axes[1]]))
        for i_cube in range(n_cubes[axis]):
            coord_of_curr_slice = i_cube * cube_sizes[axis][axis] + self.origin[axis]

            # If within the Z coords of the slab, accumulate density of this XY slice
            if slab_min <= coord_of_curr_slice <= slab_max:
                if axis == 0:
                    axis_c_vals += self.data[i_cube, :, :]
                elif axis == 1:
                    axis_c_vals += self.data[:, i_cube, :]
                else:
                    assert axis == 2
                    axis_c_vals += self.data[:, :, i_cube]

        return axis_a_coords, axis_b_coords, axis_c_vals

    def get_height_profile_map(
        self,
        isovalue: float,
        tolerance: float,
        grid_multiplier: int,
        z_min: float = None,
        z_max: float = None,
        xy_tiling: List[int] = None,
    ) -> np.ndarray:
        """
        Calculates the height profile of the density in `self.data` at the target
        `isovalue`, within the specified `tolerance`.

        For each XY point, the Z-coordinates of the cube array are densified by a factor
        of `grid_multiplier`, then interpolated with splines to find the Z-coordinates
        that match the taregt `isovalue` within the specified `tolerance`.

        The intended use of this function is for visualizing slab surface densities. As
        such, the surface is assumed to be aligned at Z=0, with the slab at negative
        Z-coordinates. Any Z-coordinate that is outside the range of `min_z` and
        `max_z` is ignored and returned as NaN.
        """

        height_map = np.ones((self.NX, self.NY)) * np.nan
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):

                # Spline this 1D array of values, build a cubic spline
                z_grid = (np.arange(self.NZ) * self.Z[2]) + self.origin[2]
                spliner = CubicSpline(z_grid, self.data[i, j, :])

                # Build a finer grid for interpolation
                z_grid_fine = np.linspace(
                    z_grid.min(), z_grid.max(), self.NZ * grid_multiplier
                )

                # Find the idxs of the values that match the isovalue, within the tolerance
                match_idxs = np.where(
                    np.abs(spliner(z_grid_fine) - isovalue) < tolerance
                )[0]
                if len(match_idxs) == 0:
                    continue

                # Find the idx of the matching value with the greatest z-coordinate
                match_idx = match_idxs[np.argmax(z_grid_fine[match_idxs])]
                z_height = z_grid_fine[match_idx]
                if (z_min is not None and z_height < z_min) or (
                    z_max is not None and z_height > z_max
                ):
                    continue
                height_map[i, j] = z_grid_fine[match_idx]

        if xy_tiling is None:
            xy_tiling = [1, 1]
        assert len(xy_tiling) == 2

        X = (np.arange(self.NX * xy_tiling[0]) * self.X[0]) + self.origin[0]
        Y = (np.arange(self.NY * xy_tiling[1]) * self.Y[1]) + self.origin[1]
        Z = np.tile(height_map, reps=xy_tiling)

        return X, Y, Z.T

    def show_volumetric(self, isovalue: float = 0.01):
        """
        Uses py3Dmol to display the volumetric data in the cube file.
        """
        v = py3Dmol.view()
        v.addModelsAsFrames(open(self.file_path, "r").read(), "cube")
        v.setStyle({"stick": {}})
        v.addVolumetricData(
            open(self.file_path, "r").read(),
            "cube",
            {"isoval": isovalue, "color": "blue", "opacity": 0.8},
        )
        v.show()


# ===== For generating contour plots from cube files


def contour_scatter_matrix(
    settings: dict,
    cube_paths: List[str] = None,
    cubes: Optional[List[RhoCube]] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """
    Returns a scatter matrix of contour plots from the list of cube files paths.

    The diagonals correspond to the contour plots, while the off-diagonals correspond to
    the delta (error) contour plots. For instance, for 3 cube files, A, B, and C,the
    returned scatter matrix has the following structure:

            +---------+---------+---------+
            |    A    |  B - A  |  C - A  |
            +---------+---------+---------+
            | [blank] |    B    |  C - B  |
            +---------+---------+---------+
            | [blank] | [blank] |    C    |
            +---------+---------+---------+

    Returned are the `fig, axes` tuple of objects returned when calling
    `plt.subplots()`.

    Passed can either be the paths to the cube files ``cube_paths`` or the
    :py:class:`RhoCube` objects of the pre-parsed cube files ``cubes``.
    """
    # Parse cube files
    if cube_paths is None:
        assert cubes is not None, "must specify either ``cube_paths`` or ``cubes``"
    else:
        assert cubes is None, "cannot specify both ``cube_paths`` and ``cubes``"
        cubes = [RhoCube(path) for path in cube_paths]

    # Create a scatter matrix
    fig, axes = plt.subplots(
        len(cubes),
        len(cubes),
        figsize=(5 * len(cubes), 5 * len(cubes)),
        sharey=True,
        sharex=True,
    )

    # Generate the contour data
    X, Y, Z = [], [], []
    for q in cubes:
        if settings["mode"] == "chm":
            x, y, z = q.get_slab_slice(
                axis=2,
                center_coord=q.frame.positions[:, 2].max() + settings["center_coord"],
                thickness=settings["thickness"],
            )

        elif settings["mode"] == "ccm":
            x, y, z = q.get_height_profile_map(
                isovalue=settings["isovalue"],
                tolerance=settings["tolerance"],
                grid_multiplier=settings.get("grid_multiplier"),
                z_min=settings.get("z_min"),
                z_max=settings.get("z_max"),
                xy_tiling=settings.get("xy_tiling"),
            )
        else:
            raise ValueError("Invalid STM mode")
        X.append(x)
        Y.append(y)
        Z.append(z)

    # Calculate the min and max of the fields to be plotted. This ensures consistent
    # color bar scales for all scatter plots
    vmin, vmax = [], []
    for row, row_ax in enumerate(axes):
        for col, ax in enumerate(row_ax):
            if row == col:
                z_ = Z[row].flatten()
            elif row < col:  # upper triangle
                z_ = (Z[col] - Z[row]).flatten()
            else:
                continue
            vmin.append(np.min(z_[~np.isnan(z_)]))
            vmax.append(np.max(z_[~np.isnan(z_)]))

    # Make the contour plots
    for row, row_ax in enumerate(axes):
        for col, ax in enumerate(row_ax):

            if row == col:
                x, y, z = X[row], Y[row], Z[row]
            elif row < col:  # upper triangle
                x, y, z = X[row], Y[col], Z[col] - Z[row]
            else:
                continue

            cs = ax.contourf(
                x,
                y,
                z,
                vmin=np.min(vmin),
                vmax=np.max(vmax),
                cmap="viridis",
                levels=settings["levels"],
            )

            fig.colorbar(cs)
            ax.set_aspect("equal")
            ax.set_xlabel("x / Ang")
            ax.set_ylabel("y / Ang")

    return fig, axes
