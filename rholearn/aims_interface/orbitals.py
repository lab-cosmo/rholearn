"""
Module to calculate Kohn-Sham orbital occs for constructing scalar fields in
FHI-aims, and to handle outputs in terms of orbitals.
"""

from typing import List, Optional, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.special import erf

from rholearn.aims_interface import parser

# ===== HOMO / LUMO =====


def get_eigenstate_occs_homo(aims_output_dir: str) -> np.ndarray:
    """
    Returns the eiegenstate occupations for constructing the HOMO. This identifies the
    highest-occupied eigenstate a value of 1, and all other eigenstates get a weight of
    zero.
    """
    kso_info = parser.get_eigenstate_info(aims_output_dir)

    # Find the indices of the HOMO states
    homo_kso_idx = get_homo_kso_idx(kso_info, max_occ=2)

    # Fill in occs
    occs = np.zeros(kso_info.shape[0])
    homo_kso = kso_info[homo_kso_idx - 1]
    assert homo_kso["kso_i"] == homo_kso_idx
    occs[homo_kso_idx - 1] = homo_kso["k_weight"]

    return occs


def get_eigenstate_occs_lumo(aims_output_dir: str) -> np.ndarray:
    """
    Returns the eiegenstate occupations for constructing the LUMO. This assigns the
    lowest-unoccupied eigenstate a value of 1, and all other eigenstates get a weight of
    zero.
    """
    kso_info = parser.get_eigenstate_info(aims_output_dir)

    # Find the indices of the LUMO states
    lumo_kso_idx = get_lumo_kso_idx(kso_info, max_occ=2)

    # Fill in occs
    occs = np.zeros(kso_info.shape[0])
    lumo_kso = kso_info[lumo_kso_idx - 1]
    assert lumo_kso["kso_i"] == lumo_kso_idx
    occs[lumo_kso_idx - 1] = lumo_kso["k_weight"]

    return occs


def get_homo_kso_idx(kso_info: Union[str, np.ndarray], max_occ: int = 2) -> np.ndarray:
    """
    Returns the KSO index that corresponds to the HOMO state. This is defined as the
    highest energy state with an occupation greater than `max_occ` / 2.

    Note that the returned index correpsonds to the FHI-aims KSO index, which is
    1-indexed.
    """
    if isinstance(kso_info, str):
        kso_info = parser.get_eigenstate_info(kso_info, as_array=True)
    kso_info = np.sort(kso_info, order="energy_eV")

    occ_states = kso_info[np.where(kso_info["occ"] > (max_occ / 2))[0]]
    return np.sort(occ_states, order="energy_eV")[-1]["kso_i"]


def get_lumo_kso_idx(kso_info: Union[str, np.ndarray], max_occ: int = 2) -> np.ndarray:
    """
    Returns the KSO index that corresponds to the LUMO state. This is defined as the
    lowest energy state with an occupation less than `max_occ` / 2.

    Note that the returned index correpsonds to the FHI-aims KSO index, which is
    1-indexed.
    """
    if isinstance(kso_info, str):
        kso_info = parser.get_eigenstate_info(kso_info, as_array=True)
    kso_info = np.sort(kso_info, order="energy_eV")

    unocc_states = kso_info[np.where(kso_info["occ"] < (max_occ / 2))[0]]
    return np.sort(unocc_states, order="energy_eV")[0]["kso_i"]


# ===== Electron density, LDOS, ILDOS =====


def get_eigenstate_occs_e_density(aims_output_dir: str) -> np.ndarray:
    """
    Returns the eiegenstate occupations for constructing the electron density.

    Each eigenstate occupation is given by the product of the k-point weight and the
    electronic occupation for each eigenstate.
    """
    kso_info = parser.get_eigenstate_info(aims_output_dir)

    return np.array([kso["occ"] for kso in kso_info])


def get_eigenstate_occs_ldos(
    aims_output_dir: str, gaussian_width: float, target_energy: float
) -> np.ndarray:
    """
    Returns the eiegenstate occupations for constructing the Local Density of States
    (LDOS).

    Each eigenstate occupation is given by the evaluation of a Gaussian function, of
    width `gaussian_width`, centered on the eigenstate eigenvalue and evaluated at the
    `target_energy`.

    Where `target_energy` is not passed, the eigenvalue of the highest occupied
    eigenstate is used.

    `gaussian_width`, and `target_energy`  must be passed in units of eV.
    """
    kso_info = parser.get_eigenstate_info(aims_output_dir)
    W_vect = []
    for kso in kso_info:
        W_a = evaluate_gaussian(
            target=target_energy, center=kso["energy_eV"], width=gaussian_width
        )
        W_vect.append(W_a)

    return np.array(W_vect)


def get_eigenstate_occs_ildos(
    aims_output_dir: str,
    gaussian_width: float,
    energy_window: List[float],
    target_energy: float,
    method: str = "gaussian_analytical",
) -> np.ndarray:
    """
    Returns the eiegenstate occupations for constructing the Local Density of States
    (LDOS).

    Each eigenstate occupation is given by the sum of Gaussian functions, each of width
    `gaussian_width`, centered on the eigenstate eigenvalue and evaluated on a discrete
    energy grid from `(target_energy + energy_window[0])` to `(target_energy +
    energy_window[1])`

    Where `target_energy` is not passed, the eigenvalue of the highest occupied
    eigenstate is used.

    `energy_window`, `gaussian_width`, and `target_energy`  must be passed in units of
    eV.

    `method` is the method of computing the integral, either assuming a Gaussian
    integrated analytically (``method="gaussian_analytical"``) or numerically
    (``method="gaussian_numerical"``), or a delta function (``method="delta"``) centered
    on each eigenvalue.
    """
    if method == "gaussian_analytical":
        return _get_eigenstate_occs_ildos_analytical(
            aims_output_dir=aims_output_dir,
            gaussian_width=gaussian_width,
            energy_window=energy_window,
            target_energy=target_energy,
        )
    elif method == "gaussian_numerical":
        return _get_eigenstate_occs_ildos_numerical(
            aims_output_dir=aims_output_dir,
            gaussian_width=gaussian_width,
            energy_window=energy_window,
            target_energy=target_energy,
            energy_grid_points=1000,
        )
    else:
        raise ValueError(
            "Invalid option for `method`. must be one of ['gaussian_analytical',"
            " 'gaussian_numerical']"
        )


def _get_eigenstate_occs_ildos_analytical(
    aims_output_dir: str,
    gaussian_width: float,
    energy_window: List[float],
    target_energy: float,
) -> np.ndarray:
    """
    For each eigenstate, the weight is given by the analytical integral of a Gaussian
    centered on the energy eigenvalue, evaluated at the `target_energy`:

    W(a) = 0.5 * (
        erf( (\\epsilon_a - \\epsilon - \\epsilon_1) / (\\sigma * \\sqrt(2)) )
        - erf( (\\epsilon_a - \\epsilon - \\epsilon_2) / (\\sigma * \\sqrt(2))
    )

    where \\epsilon is the `target_energy`, \\epsilon_1 is the lower and \\epsilon_2 the
    higher of the `energy_window`, g is the Gaussian function of width \\sigma.
    """
    if not np.abs(energy_window[0] - energy_window[1]) > 0:
        raise ValueError("energy window size must be non-zero")

    # Define the integration limits. If biasing voltage is negative, limits should be
    # switched
    if energy_window[1] > energy_window[0]:
        lim_lo, lim_hi = (
            target_energy + energy_window[0],
            target_energy + energy_window[1],
        )
    else:
        lim_lo, lim_hi = (
            target_energy + energy_window[1],
            target_energy + energy_window[0],
        )

    kso_info = parser.get_eigenstate_info(aims_output_dir)

    W_vect = []
    for kso in kso_info:
        W_a = 0.5 * (
            erf((kso["energy_eV"] - lim_lo) / (np.sqrt(2) * gaussian_width))
            - erf((kso["energy_eV"] - lim_hi) / (np.sqrt(2) * gaussian_width))
        )
        W_vect.append(W_a)

    return np.array(W_vect)


def _get_eigenstate_occs_ildos_numerical(
    aims_output_dir: str,
    gaussian_width: float,
    energy_window: List[float],
    target_energy: float,
    energy_grid_points: int = 1000,
) -> np.ndarray:
    """
    For each eigenstate, the occupation is given by the numerical integral of a Gaussian
    centered on the energy eigenvalue \\epsilon_a, evaluated at the `target_energy`:

    W(a) = (V / n)
        * \\sum_{\\epsilon'=\\epsilon + \\epsilon_1}^{\\epsilon + \\epsilon_2}
        * g(\\epsilon' - \\epsilon_a, \\sigma)

    where \\epsilon is the target energy, \\epsilon_1 is the lower and \\epsilon_2 the
    higher of the `energy_window`, n is the number of energy grid points, g is the
    Gaussian function of width \\sigma.
    """
    kso_info = parser.get_eigenstate_info(aims_output_dir)

    if target_energy is None:  # Find the energy of the HOMO
        homo_idx = parser.find_homo_kso_idxs(kso_info)[0]
        target_energy = kso_info[homo_idx - 1]["energy_eV"]  # 1-indexing!

    W_vect = []
    for kso in kso_info:
        W_a = (
            np.abs(energy_window[0] - energy_window[1]) / energy_grid_points
        ) * np.sum(
            [
                evaluate_gaussian(
                    target=tmp_target_energy,
                    center=kso["energy_eV"],
                    width=gaussian_width,
                )
                for tmp_target_energy in np.linspace(
                    target_energy + energy_window[0],
                    target_energy + energy_window[1],
                    energy_grid_points,
                )
            ]
        )
        W_vect.append(W_a)

    return np.array(W_vect)


def evaluate_gaussian(target: float, center: float, width: float) -> float:
    """
    Evaluates a Gaussian function with the specified parameters at the target value
    """
    return (1.0 / (width * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((target - center) / width) ** 2
    )


def calculate_dos(
    aims_output_dir: str,
    gaussian_width: float,
    e_grid: np.ndarray = None,
):
    """
    Centers a Gaussian of width `gaussian_width` on each energy eigenvalue read from
    file "eigenstate_info.out" at dir `aims_output_dir` and plots the sum of these as
    the Density of States (DOS).
    """
    # Load KSO info
    kso_info = parser.get_eigenstate_info(aims_output_dir)

    if e_grid is None:
        e_grid = np.linspace(
            np.min(kso_info["energy_eV"]) - 20,
            np.max(kso_info["energy_eV"]) + 20,
            10000,
        )

    dos = np.zeros(len(e_grid))
    for kso in kso_info:
        dos += evaluate_gaussian(
            target=e_grid, center=kso["energy_eV"], width=gaussian_width
        )

    return e_grid, dos


def calculate_fermi_energy(
    aims_output_dir: str,
    n_electrons: int,
    e_grid: Optional[np.ndarray] = None,
    gaussian_width: float = 0.3,
    interpolation_truncation: Optional[float] = 0.1,
) -> float:
    """
    Calculates the Fermi energy by integrating the cumulative density of states.

    If `e_grid` is specified, the DOS will be calculated on this grid. Otherwise, the
    `e_grid` is taken across the range of energies in the file "eigenstate_info.out" in
    dir `aims_output_dir` file.

    The number of electrons `n_electrons` should correspond to the number of expected
    electrons to be integrated over in the DOS for the given energy range.

    If the `e_grid` range is specified as a subset, i.e. excluding core states,
    `n_electrons` should beadjusted accordingly.
    """
    # Load KSO info
    kso_info = parser.get_eigenstate_info(aims_output_dir)

    # Calculate the DOS, k-weighted and accounting for orbital occupancy
    e_grid, dos = calculate_dos(
        kso_info,
        gaussian_width=gaussian_width,
        e_grid=e_grid,
    )

    # Compute the cumulative DOS and interpolate
    cumulative_dos = cumulative_trapezoid(dos, e_grid, axis=0) - n_electrons
    interpolated = interp1d(
        e_grid[:-1], cumulative_dos, kind="cubic", copy=True, assume_sorted=True
    )
    fermi_energy = brentq(
        interpolated,
        e_grid[0] + interpolation_truncation,
        e_grid[-1] - interpolation_truncation,
    )

    return fermi_energy
