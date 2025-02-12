"""
Module containing the RhoCube class, which wraps the cube_tools.cube class from
https://github.com/funkymunkycool/Cube-Toolz.

Allows reading and manipulation of cube files, with added functionality for generating
contour plots (i.e. for use in STM image generation).
"""

from os.path import join
from typing import List, Optional, Tuple, Union

import cube_tools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
from chemfiles import Atom, Frame, UnitCell

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error

from rholearn.utils import ATOMIC_NUMBERS_TO_SYMBOLS
from rholearn.utils.io import pickle_dict


class RhoCube(cube_tools.cube):

    def __init__(
        self,
        file_path: str,
        expect_lengths_in_bohr: Optional[bool] = True,
    ) -> None:
        super(RhoCube, self).__init__(file_path)
        self.file_path = file_path

        # Expects that the cube lengths and origin are given in atomic units (i.e.
        # Bohr). Convert to SI units (i.e. Angstrom)
        if expect_lengths_in_bohr:
            bohr_to_ang = 0.529177249
            self.X *= bohr_to_ang
            self.Y *= bohr_to_ang
            self.Z *= bohr_to_ang
            self.origin *= bohr_to_ang

        self.frame = self.frame()

    def frame(self) -> Frame:
        """
        Builds an ASE atoms object from the atomic positions and chemical
        symbols in the cube file.
        """
        frame = Frame()
        frame.cell = UnitCell(
            np.vstack([self.X * self.NX, self.Y * self.NY, self.Z * self.NZ])
        )
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
        Calculates the height profile of the density at the target `isovalue`, within
        the specified `tolerance`. Tiles the grid if specified and then transforms the
        coordinates to the physical space.
        """
        # Assuming cell_matrix is defined with each column as a vector for x, y, z
        cell_matrix = np.array([self.X, self.Y, self.Z]).T  # Columns as x, y, z vectors
        height_map = np.full((self.NX, self.NY), np.nan)

        # Compute height map
        for i in range(self.NX):
            for j in range(self.NY):
                z_grid = (np.arange(self.NZ) * self.Z[2]) + self.origin[2]
                spliner = CubicSpline(z_grid, self.data[i, j, :])
                z_grid_fine = np.linspace(z_grid.min(), z_grid.max(), self.NZ * grid_multiplier)
                match_idxs = np.where(np.abs(spliner(z_grid_fine) - isovalue) < tolerance)[0]
                if match_idxs.size > 0:
                    match_idx = match_idxs[np.argmax(z_grid_fine[match_idxs])]
                    z_height = z_grid_fine[match_idx]
                    if (z_min is None or z_height >= z_min) and (z_max is None or z_height <= z_max):
                        height_map[i, j] = z_height

        if xy_tiling is None:
            xy_tiling = [1, 1]

        height_map_tiled = np.tile(height_map, xy_tiling)

        # Create indices for tiled map
        tiled_NX, tiled_NY = self.NX * xy_tiling[0], self.NY * xy_tiling[1]
        X_indices_tiled, Y_indices_tiled = np.meshgrid(np.arange(tiled_NX), np.arange(tiled_NY), indexing="ij")
        # No Z indices used, set all to zero (flat)

        # Manually apply transformations
        # tiled_X = cell_matrix[0, 0] * X_indices_tiled + cell_matrix[0, 1] * Y_indices_tiled
        # tiled_Y = cell_matrix[1, 0] * X_indices_tiled + cell_matrix[1, 1] * Y_indices_tiled

        # Swapping matrix application to indices:
        # tiled_X = cell_matrix[1, 0] * X_indices_tiled + cell_matrix[1, 1] * Y_indices_tiled
        # tiled_Y = cell_matrix[0, 0] * X_indices_tiled + cell_matrix[0, 1] * Y_indices_tiled

        # Another way of swapping indices application:
        tiled_X = cell_matrix[0, 0] * Y_indices_tiled + cell_matrix[0, 1] * X_indices_tiled
        tiled_Y = cell_matrix[1, 0] * Y_indices_tiled + cell_matrix[1, 1] * X_indices_tiled

        return tiled_X, tiled_Y, height_map_tiled

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


def plot_contour_ccm(
    cubes: List[RhoCube],
    isovalues: Union[float, List[float]],
    tolerances: Union[float, List[float]],
    grid_multiplier: int,
    levels: int,
    z_min: float = None,
    z_max: float = None,
    xy_tiling: List[int] = None,
    cmap: str = "viridis",
    plot_atoms: bool = False,
    orthogonalize_image: bool = False,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    save_dir: str = None,
):
    """
    Plots a contour plot of the height profile map of the cube file.
    """
    if isinstance(cubes, RhoCube):
        cubes = [cubes]

    if isinstance(isovalues, float):
        isovalues = [isovalues]
    if isinstance(tolerances, float):
        tolerances = [tolerances]
    assert len(isovalues) == len(tolerances), (
        "`isovalues` and `tolerances` must have the same length"
    )
    if xy_tiling is None:
        xy_tiling = [1, 1]

    if orthogonalize_image:
        pad_tiling = 2
        xy_tiling = [i + pad_tiling for i in xy_tiling]

    fig, axes = plt.subplots(
        len(isovalues) + 1 if plot_atoms else len(isovalues),
        len(cubes),
        # figsize=(10 * len(cubes), 5 * len(isovalues)),
        sharey=True,
        sharex=True,
    )

    if plot_atoms:  # plot the nuclear positions
        for ax in axes[0, :]:
            for atom_i, atom in enumerate(cubes[0].frame.atoms):

                atom_type = int(atom.type)
                position = cubes[0].frame.positions[atom_i]

                # Only plot atoms in the z range
                if position[2] < z_min or position[2] > z_max:
                    continue

                # Inverse of the lattice vectors for converting Cartesian to fractional
                lattice_vectors = cubes[0].frame.cell.matrix
                lattice_inv = np.linalg.inv(lattice_vectors)

                # Convert Cartesian to fractional coordinates
                fractional_positions = np.dot(
                    cubes[0].frame.cell.wrap(position) + np.sum(cubes[0].frame.cell.matrix / 2, axis=0),
                    lattice_inv,
                )

                # Map fractional coordinates into the unit cell [0, 1)
                fractional_positions %= 1.0

                for x_tile in range(xy_tiling[0]):
                    for y_tile in range(xy_tiling[1]):

                        # Translate the fractional coordinates into the tiling
                        frac_pos_tiled = np.copy(fractional_positions)
                        frac_pos_tiled[0] += x_tile
                        frac_pos_tiled[1] += y_tile
            
                        # Convert fractional coordinates back to Cartesian
                        position = np.dot(frac_pos_tiled, lattice_vectors)
                        
                        # Write the atomic type annotation
                        colors = {1: "white", 6: "gray", 7: "blue", 8: "red", 29: "green"}
                        ax.annotate(
                            ATOMIC_NUMBERS_TO_SYMBOLS[atom_type],
                            xy=(position[0], position[1]),
                            fontsize=5,
                            color='black',
                            ha='center',
                            va='center',
                            bbox=dict(boxstyle="circle,pad=0.2", edgecolor="black", facecolor=colors[atom_type])
                        )

    isovalue_range = range(1, len(isovalues) + 1) if plot_atoms else range(len(isovalues))
    ssims = {
        isovalue_i: {
            Z_input_i: {} 
            for Z_input_i in range(1, len(cubes))
        } 
        for isovalue_i in isovalue_range
    }
    mses = {
        isovalue_i: {
            Z_input_i: {} 
            for Z_input_i in range(1, len(cubes))
        } 
        for isovalue_i in isovalue_range
    }
    for isovalue_i, isovalue, tolerance in zip(isovalue_range, isovalues, tolerances):

        X, Y, Z = [], [], []
        for q in cubes:
            x, y, z = q.get_height_profile_map(
                isovalue=isovalue,
                tolerance=tolerance,
                grid_multiplier=grid_multiplier,
                z_min=z_min,
                z_max=z_max,
                xy_tiling=xy_tiling,
            )
            X.append(x)
            Y.append(y)
            Z.append(z)

        # Set the min and max contour values for consistent scale bar
        _Z = np.array(Z).flatten()
        _Z = _Z[~np.isnan(_Z)]
        if z_min is None:
            z_min = np.min(_Z)
        if z_max is None:
            z_max = np.max(_Z)

        # Define a normalization object with the desired z_min and z_max
        norm = Normalize(vmin=z_min, vmax=z_max)

        # Define levels explicitly to span the full range [z_min, z_max]
        levels_grid = np.linspace(z_min, z_max, levels)

        # Plot the contour maps
        for ax_i, (x, y, z, ax) in enumerate(zip(X, Y, Z, axes[isovalue_i])):

            # Reflect in x and y axis planes to correct for matplotlib's way of plotting
            # contour maps
            x = -x + np.abs(np.min(-x))
            y = -y + np.abs(np.min(-y))

            if orthogonalize_image:
                # Ensure there is no empty space by moving the origin and setting axes limit
                # to 'cut out' a non-empty rectangle in the xy plane
                x -= cubes[0].frame.cell.matrix[1][0] * (xy_tiling[1] - pad_tiling)
                y -= cubes[0].frame.cell.matrix[0][1] * (xy_tiling[0] - pad_tiling)

            cs = ax.contourf(
                x,
                y,
                z,
                vmin=z_min,
                vmax=z_max,
                cmap=cmap,
                levels=levels_grid,
                norm=norm,
            )

            if orthogonalize_image:
                ax.set_xlim(
                    [
                        0,
                        cubes[0].frame.cell.matrix[0][0] * (xy_tiling[0] - pad_tiling),
                    ]
                )
                ax.set_ylim(
                    [
                        0,
                        cubes[0].frame.cell.matrix[1][1] * (xy_tiling[1] - pad_tiling),
                    ]
                )

            # Adjust colorbar size and position
            if ax_i == len(cubes) - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(
                    cs,
                    cax=cax,
                    boundaries=np.linspace(z_min, z_max, levels),
                    ticks=np.linspace(z_min, z_max, int((z_max - z_min) // 1) + 1),
                    norm=norm,
                )
            
            # Format
            ax.set_facecolor("black")
            if ax_i == 0:
                ax.set_title(f"iso: {isovalue}")

        # Compute structural similarity index
        if len(cubes) > 1:
            for Z_target_i in range(len(Z) - 1):
                Z_target = np.nan_to_num(Z[Z_target_i], nan=0.0, posinf=0.0, neginf=0.0)

                for Z_input_i in range(Z_target_i + 1, len(Z)):

                    Z_input = np.nan_to_num(Z[Z_input_i], nan=0.0, posinf=0.0, neginf=0.0)
                    ssims[isovalue_i][Z_input_i][Z_target_i] = structural_similarity(
                        Z_target, Z_input, data_range=Z_target.max() - Z_target.min(),
                    )
                    mses[isovalue_i][Z_input_i][Z_target_i] = mean_squared_error(Z_target, Z_input)
            
            # Annotate the axes titles with the errors
            for input_i in range(1, len(Z)):
                ssim_i = ""
                for target_i in range(input_i):
                    ssim_i += f"{target_i}: {ssims[isovalue_i][input_i][target_i]:.2f}, "
                axes[isovalue_i][input_i].set_title(f"SSIM: {ssim_i}")

    [ax.set_xlabel("x / Å") for ax in axes[-1, :]]
    [ax.set_ylabel("y / Å") for ax in axes[:, 0]]
    [ax.set_aspect("equal") for ax in axes.flatten()]

    # Save plot and errors
    if save_dir is not None:
        plt.savefig(join(save_dir, "cube_scatter_ccm.png"), bbox_inches="tight", dpi=300)
        pickle_dict(
            join(save_dir, f"image_errors.pickle"),
            {"ssim": ssims, "mse": mses},
        )

    return fig, axes


def plot_contour_chm(
    cubes: List[RhoCube],
    center_coord: float,
    thickness: float,
    levels: int,
    cmap: str = "viridis",
    save_dir: str = None,
):
    """
    Plots a contour plot of the height profile map of the cube file.
    """
    if isinstance(cubes, RhoCube):
        cubes = [cubes]

    fig, axes = plt.subplots(
        1,
        len(cubes),
        figsize=(15 * len(cubes), 10 * len(cubes)),
        sharey=True,
        sharex=True,
    )

    for q, ax in zip(cubes, axes):
        x, y, z = q.get_slab_slice(
            axis=2,
            center_coord=q.frame.positions[:, 2].max() + center_coord,
            thickness=thickness,
        )

        cs = ax.contourf(
            x,
            y,
            z,
            cmap=cmap,
            levels=levels,
        )
        ax.colorbar(cs)
        ax.set_aspect("equal")
        ax.set_xlabel("x / Å")
        ax.set_ylabel("y / Å")
        ax.set_facecolor("black")

    if save_dir is not None:
        plt.savefig(join(save_dir, "cube_scatter_chm.png"), bbox_inches="tight", dpi=300)

    return fig, axes


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
        for col in range(len(row_ax)):
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
            ax.set_xlabel("x / Å")
            ax.set_ylabel("y / Å")

    return fig, axes
