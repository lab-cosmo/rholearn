"""
Module for parsing outputs from FHI-aims calculations.
"""

import os
from os.path import exists, join
from typing import Callable, List, Optional, Tuple, Union

import metatensor
import metatensor.torch
import numpy as np
import torch
from chemfiles import Frame
from scipy.interpolate import CubicHermiteSpline

from rholearn.aims_interface import fields, io, parser
from rholearn.rholearn import mask
from rholearn.utils import convert, system, utils
from rholearn.utils.io import pickle_dict


def get_eigenstate_info(
    aims_output_dir: str,
    fname: Optional[str] = "eigenstate_info.out",
    as_array: bool = True,
) -> dict:
    """
    Parses the AIMS output file "eigenstate_info.out" in directory ``aims_output_dir``
    into a numpy array containing all the Kohn-Sham orbital information.

    The number of rows of this file is equal to the number of KS-orbitals, i.e. the
    product of the number of KS states, spin states, and k-points.

    Each column corresponds to, respectively: KS-orbital index, KS state index, spin
    state, k-point index, k-point weight, occupation, and energy eigenvalue.

    If `as_array` is True, the data is returned as a structured numpy array. Otherwise,
    it is returned as a dict.

    The file "ks_orbital_info.out" is produced by setting the keyword
    `ri_fit_write_orbital_info` to True in the AIMS control.in file.
    """
    # Load file
    file = np.loadtxt(join(aims_output_dir, fname))

    # Check number of columns
    if len(file[0]) == 8:
        with_weights = True
    elif len(file[0]) == 7:
        with_weights = False
    else:
        raise ValueError(
            "expected 7 (without KSO weights) or 8 (with weights)"
            f" columns. Got: {len(file[0])}"
        )
    if as_array:  # returned structured array
        if with_weights:
            w_col = [
                ("kso_weight", float),
            ]
        else:
            w_col = []
        info = np.array(
            [tuple(row) for row in file],
            dtype=[
                ("kso_i", int),
                ("state_i", int),
                ("spin_i", int),
                ("kpt_i", int),
                ("k_weight", float),
                ("occ", float),
                ("energy_eV", float),
            ]
            + w_col,
        )

    else:  # as dict
        info = {}
        for row in file:
            if with_weights:
                i_kso, i_state, i_spin, i_kpt, k_weight, occ, energy, weight = row
            else:
                i_kso, i_state, i_spin, i_kpt, k_weight, occ, energy = row
            info[i_kso] = {
                "i_kso": i_kso,
                "i_state": i_state,
                "i_spin": i_spin,
                "i_kpt": i_kpt,
                "k_weight": k_weight,
                "occ": occ,
                "energy": energy,
            }
            if with_weights:
                info[i_kso].update({"kso_weight": weight})

    return info


def extract_basis_set_info(
    frame: Frame,
    aims_output_dir: str,
    fname: Optional[str] = "basis_info.out",
) -> Tuple[dict, dict]:
    """
    For the given ``frame``, parses the basis set information from file
    "basis_info.out" in the directory ``aims_output_dir``.

    Returns the lmax and nmax definitions as dictionaries.

    :param frame: an :py:class:`chemfiles.Frame` object corresponding to the structure
        for which the AIMS basis set info should be extracted.
    :param aims_output_dir: a `str` of the absolute path to the directory containing
        AIMS output files. In particular, this directory must contain a file called
        "basis_info.out". This contains the information of the constructed
        RI basis set for the structure passed in ``frame``.
    :param fname: a `str` of the name of the file containing the basis set information.
        Default is "basis_info.out".

    :return lmax: a `dict` of the maximum angular momentum for each chemical species in
        ``frame``.
    :return nmax: a `dict` of the maximum radial channel index for each chemical species
        and angular channel in ``frame``.
    """
    # Read the basis set information
    basis_info_file = join(aims_output_dir, fname)
    with open(basis_info_file, "r") as f:
        lines = f.readlines()

    # Get the species symbols for the atoms in the frame
    symbols = system.get_symbols(frame)

    # Parse the file to extract the line number intervals for each atom block
    intervals = []
    for line_i, line in enumerate(lines):
        line_split = line.split()
        if len(line_split) == 0:
            continue

        if line_split[0] == "atom":
            block_start = line_i
            continue
        elif line_split[:2] == ["For", "atom"]:
            block_end = line_i + 1
            intervals.append((block_start, block_end))
            continue

    # Group the lines of the file into atom blocks
    atom_blocks = [lines[start:end] for start, end in intervals]

    # Parse the lmax and nmax values for each chemical species
    # This assumes that the basis set parameters is the same for every atom of
    # the same chemical species
    lmax, nmax = {}, {}
    for block in atom_blocks:
        # Get the atom index (zero indexed)
        atom_idx = int(block[0].split()[1]) - 1

        # Skip if we already have an entry for this chemical species
        symbol = symbols[atom_idx]
        if symbol in lmax.keys():
            continue

        # Get the max l value and store
        assert int(block[-1].split()[2]) - 1 == atom_idx
        species_lmax = int(block[-1].split()[6])
        lmax[symbol] = species_lmax

        # Parse the nmax values and store. There are (lmax + 1) angular channels
        for o3_lambda in range(species_lmax + 1):
            line = block[o3_lambda + 1]
            assert o3_lambda == int(line.split()[3])
            species_nmax = int(line.split()[6])
            nmax[(symbol, o3_lambda)] = species_nmax

    return lmax, nmax


def get_prodbas_radii(aims_output_dir: str, fname: Optional[str] = "aims.out") -> dict:
    """
    Parses the charge and field radii, and the multipole moments for each product basis
    function (combination of atomic symbol and angular channel) from the AIMS output
    file.
    """

    with open(join(aims_output_dir, fname), "r") as f:
        lines = f.readlines()

    for _i, line in enumerate(lines):
        if "Constructing auxiliary basis (full product) ..." in line:
            break

    assert (
        lines[_i + 5].split()
        == "  | Species   l  charge radius    field radius  multipol momen".split()
    )

    prodbas_data = {}
    for line in lines[_i + 6 :]:

        line = line.split()
        if len(line) != 9:
            break

        if line[0] == "|" and line[4] == "A" and line[6] == "A" and line[8] == "a.u.":
            symbol, l, charge_radius, field_radius, mulitpole_mom = (
                line[1],
                int(line[2]),
                float(line[3]),
                float(line[5]),
                float(line[7]),
            )
            tmp_dict = {
                "charge_radius": charge_radius,
                "field_radius": field_radius,
                "mulitpole_mom": mulitpole_mom,
            }
            if symbol in prodbas_data:
                if l in prodbas_data[symbol]:
                    prodbas_data[symbol][l].append(tmp_dict)
                else:
                    prodbas_data[symbol][l] = [tmp_dict]
            else:
                prodbas_data[symbol] = {l: [tmp_dict]}

        else:
            break

    return prodbas_data


def get_max_overlap_radius_by_type(aims_output_dir: str) -> float:
    """
    Returns the maximum overlap radius by species type.
    Reads the product basis function radii by parsing "aims.out" file in the directory
    ``aims_output_dir``, using the function :py:func:`get_prodbas_radii`.
    """
    # Get the product basis function radii
    radii = get_prodbas_radii(aims_output_dir)

    max_radii = {}
    for symbol, l_dict in radii.items():
        species_type = system.atomic_symbol_to_atomic_number(symbol)
        max_radii[species_type] = max(
            [item["charge_radius"] for l, l_list in radii[symbol].items() for item in l_list]
        )

    return max_radii


def parse_aims_out(aims_output_dir: str, fname: str = "aims.out") -> dict:
    """
    Extracts relevant information from the main AIMS output file "aims.out",
    stored in the directory at absolute path ``aims_output_dir``.

    :param aims_output_dir: a `str` of the absolute path to the directory
        containing AIMS output files. In particular, this directory must contain
        a file called "aims.out".

    :returns: a `dict` of the relevant information extracted from the AIMS
        output.
    """
    # Read aims.out
    with open(os.path.join(aims_output_dir, fname), "r") as f:
        lines = f.readlines()

    # Initialize a dict to store the extracted information
    calc_info = {
        "aims": {
            "run_dir": aims_output_dir,
        },
        "scf": {
            "num_cycles": 0,
            "converged": False,
            "d_charge_density": [],
            "d_tot_energy_eV": [],
        },
        "ks_states": {},
        "prodbas_acc": {},
        "prodbas_radial_fn_radii_ang": {},
    }

    # Parse each line for relevant information
    for line_i, line in enumerate(lines):
        split = line.split()

        # AIMS unique identifier for the run
        if split[:2] == "aims_uuid :".split():
            calc_info["aims"]["run_id"] = split[2]

        # AIMS version
        if split[:3] == "FHI-aims version      :".split():
            calc_info["aims"]["version"] = split[3]

        # AIMS commit version
        if split[:3] == "Commit number         :".split():
            calc_info["aims"]["commit"] = split[3]

        # Number of atoms
        # Example:
        # "| Number of atoms                   :       64"
        if split[:5] == "| Number of atoms                   :".split():
            calc_info["num_atoms"] = int(split[5])

        # Net and non-zero number of real-space integration points
        # Example: "| Net number of integration points:    49038"
        if split[:6] == "| Net number of integration points:".split():
            calc_info["num_int_points"] = {
                "net": int(split[6]),
                "non-zero": int(lines[line_i + 1].split()[7]),  # on next line
            }

        # Number of spin states
        # Example:
        # "| Number of spin channels           :        1"
        if split[:6] == "| Number of spin channels           :".split():
            calc_info["num_spin_states"] = int(split[6])

        # Requested and actually used number of k points
        # Example: "| k-points reduced from:        8 to        8"
        if split[:4] == "| k-points reduced from:".split():
            calc_info["num_k_points"] = {
                "requested": int(split[4]),
                "actual": int(split[6]),
            }

        # Number of auxiliary basis functions after RI fitting.
        # Example: "| Shrink_full_auxil_basis : there are totally 1001
        # partial auxiliary wave functions."
        if split[:6] == "| Shrink_full_auxil_basis : there are totally".split():
            calc_info["num_abfs"] = int(split[6])

        # For the following quantities, every time a new SCF loop is
        # encountered in aims.out, the values are overwritten such that only
        # the values from the final SCF loop are returned.

        # SCF convergence criteria
        # Example:
        # Self-consistency convergence accuracy:
        # | Change of charge density      :  0.9587E-07
        # | Change of unmixed KS density  :  0.2906E-06
        # | Change of sum of eigenvalues  : -0.2053E-05 eV
        # | Change of total energy        : -0.1160E-11 eV
        if split[:6] == "| Change of charge density :".split():
            calc_info["scf"]["d_charge_density"] = float(split[6])
        if split[:6] == "| Change of total energy :".split():
            calc_info["scf"]["d_tot_energy_eV"] = float(split[6])

        # Number of Kohn-Sham states
        # Example: " Number of Kohn-Sham states (occupied + empty):       11"
        if split[:8] == "| Number of Kohn-Sham states (occupied + empty):".split():
            calc_info["num_ks_states"] = int(split[8])

        # Highest occupied state
        # Example:
        # "Highest occupied state (VBM) at     -9.04639836 eV"
        if split[:5] == "Highest occupied state (VBM) at".split():
            calc_info["homo_eV"] = float(split[5])

        # Lowest unoccupied state
        # Example:
        # "Lowest unoccupied state (CBM) at    -0.05213986 eV"
        if split[:5] == "Lowest unoccupied state (CBM) at".split():
            calc_info["lumo_eV"] = float(split[5])

        # HOMO-LUMO gap
        # Example:
        # "Overall HOMO-LUMO gap:      8.99425850 eV."
        if split[:4] == "Overall HOMO-LUMO gap:".split():
            calc_info["homo_lumo_gap_eV"] = float(split[4])

        # Fermi level / chemical potential
        # Example:
        # "| Chemical potential (Fermi level):    -9.07068018 eV"
        if split[:5] == "| Chemical potential (Fermi level):".split():
            calc_info["fermi_eV"] = float(split[5])

        # SCF converged?
        if "Self-consistency cycle converged." in line:
            calc_info["scf"]["converged"] = True

        # Number of SCF cycles run
        # Example:
        # Computational steps:
        # | Number of self-consistency cycles          :           21
        # | Number of SCF (re)initializations          :            1
        if split[:6] == "| Number of self-consistency cycles          :".split():
            calc_info["scf"]["num_cycles"] = int(split[6])

        # Kohn-Sham states, occs, and eigenvalues (eV)
        # Example:
        # State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]
        # 1       2.00000         -19.211485         -522.77110
        # 2       2.00000          -1.038600          -28.26175
        # 3       2.00000          -0.545802          -14.85203
        # ...
        if (
            split[:6]
            == "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]".split()
        ):
            for state_i in range(1, calc_info["num_ks_states"] + 1):
                state, occ, eig_ha, eig_eV = lines[line_i + state_i].split()
                assert int(state) == state_i
                calc_info["ks_states"][int(state)] = {
                    "occ": float(occ),
                    "eig_eV": float(eig_eV),
                }

        # Final total energy
        # Example: | Total energy of the DFT ...
        # ... / Hartree-Fock s.c.f. calculation : -2078.592149198 eV
        if (
            split[:11]
            == "| Total energy of the DFT / Hartree-Fock s.c.f. calculation :".split()
        ):
            calc_info["tot_energy_eV"] = float(split[11])

        # Extract the total time for the calculation
        # Example:
        # Detailed time accounting                  :  max(cpu_time)    wall_clock(cpu1)
        # | Total time                               :       28.746 s          28.901 s
        # | Preparation time                         :        0.090 s           0.173 s
        # | Boundary condition initalization         :        0.031 s           0.031 s
        if split[:4] == "| Total time :".split():
            calc_info["time"] = {
                "max(cpu_time)": float(split[4]),
                "wall_clock(cpu1)": float(split[6]),
            }

        # Extract the default prodbas accuracy
        # Example:
        # "Species H: Using default value for prodbas_acc =   1.000000E-04."
        if split[2:8] == "Using default value for prodbas_acc =".split():
            calc_info["prodbas_acc"][split[1][:-1]] = float(split[8][:-1])

        # Cutoff radius for evaluating overlap matrix
        # Example:
        # "ri_fit: Found cutoff radius for calculating ovlp matrix:   2.00000"
        if (
            split[:8]
            == "ri_fit: Found cutoff radius for calculating ovlp matrix:".split()
        ):
            calc_info["ri_fit_cutoff_radius"] = float(split[8])

        # Extract the charge radii of the product basis radial functions
        if split[:2] == "Product basis:".split():
            assert lines[line_i + 1].split()[:3] == "| charge radius:".split()
            assert lines[line_i + 2].split()[:3] == "| field radius:".split()
            assert lines[line_i + 3].split()[:9] == (
                "| Species   l  charge radius    "
                "field radius  multipol momen".split()
            )

            tmp_line_i = line_i + 4
            keep_reading = True
            while keep_reading:
                tmp_split = lines[tmp_line_i].split()

                # Break if not valid lines
                if len(tmp_split) == 0:
                    keep_reading = False
                    break
                if tmp_split[-1] != "a.u.":
                    keep_reading = False
                    break

                # This is one of the charge radius lines we want to read from
                if calc_info["prodbas_radial_fn_radii_ang"].get(tmp_split[1]) is None:
                    calc_info["prodbas_radial_fn_radii_ang"][tmp_split[1]] = [
                        float(tmp_split[3])
                    ]
                else:
                    calc_info["prodbas_radial_fn_radii_ang"][tmp_split[1]].append(
                        float(tmp_split[3])
                    )

                tmp_line_i += 1

        # Extract ri_fit info
        # Example:
        # ri_fit: Finished.
        if split[:2] == "ri_fit: Finished.".split():
            calc_info["ri_fit_finished"] = True

    return calc_info


def extract_input_file_from_control(
    aims_output_dir: str,
    section_type: str,
    save_dir_path: str,
    fname: Optional[str] = "aims.out",
):
    """
    Parse "aims.out" file (or optionally ``fname``) in directory ``aims_output_dir`` and
    extracts either the "geometry.in" or "control.in" section, depending on whether
    `section_type="geometry"` or `section_type="control"`, respectively.

    Writes it to ``save_dir_path``/{geometry,control}.in.

    Assumes these sections are contained between "----" separators in the aims.out file.
    For parsing the geometry, keywords "trust_radius" and "hessian_block" are ignored.
    """
    assert section_type in [
        "control",
        "geometry",
    ], "section_type must be 'control' or 'geometry'"
    with open(join(aims_output_dir, fname), "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if f"Parsing {section_type}.in" in line:
                for j, next_line in enumerate(lines[i:], i):
                    if "-----" in next_line:
                        start = j + 1
                        break

        geom_lines = []
        for next_line in lines[start:]:
            split = next_line.split()
            if len(split) == 0:
                continue
            if section_type == "geometry" and split[0] in [
                "trust_radius",
                "hessian_block",
            ]:
                continue
            if "---" in next_line:
                break

            geom_lines.append(next_line)

    with open(os.path.join(save_dir_path, section_type, ".in"), "w") as f:
        for line in geom_lines:
            f.write(line)


def load_ovlp_matrix_to_square_matrix_numpy(
    aims_output_dir: str, fname: Optional[str] = "ri_ovlp.out"
) -> np.ndarray:
    """
    Loads the overlap matrix from the file "ri_ovlp.out" (alternatively ``fname``) in
    directory ``aims_output_dir`` and converts it to a full square matrix.

    Assumes the overlap matrix is stored in a flat vector format, representing the upper
    triangular part of the full matrix.
    """

    # Read the overlap and ensure a correct dimension. Currently it is a flat vector
    # representing the triangular matrix, of length ((n^2 + n) / 2), where `n` is the
    # length of the corresponding coefficient vector.
    ovlp_numpy = np.loadtxt(join(aims_output_dir, fname))
    coeffs_numpy = np.loadtxt(join(aims_output_dir, "ri_restart_coeffs.out"))
    ovlp_dim = int(coeffs_numpy.shape[0])
    assert len(ovlp_numpy) == ((ovlp_dim**2) + ovlp_dim) / 2

    # Convert to full square matrix and check symmetric
    s_full = np.empty((ovlp_dim, ovlp_dim))
    tri_idxs = np.triu_indices(ovlp_dim)
    s_full[tri_idxs] = ovlp_numpy
    s_full.T[tri_idxs] = ovlp_numpy
    ovlp_numpy = s_full
    assert np.allclose(ovlp_numpy, ovlp_numpy.T)

    return ovlp_numpy


def process_ri_outputs(
    aims_output_dir: str,
    structure_idx: int,
    ovlp_cond_num: bool,
    cutoff_ovlp: bool,
    ovlp_sparsity_threshold: Optional[float],
    save_dir: str,
) -> None:
    """
    Processes the RI coefficients, projections, and overlap matrices into
    :py:class:`TensorMap` format and writes them to ``save_dir``.

    Processing of each file can be switched on or off with the boolean flags.
    """
    if not exists(save_dir):
        os.makedirs(save_dir)

    # Read "geometry.in" as the Frame
    frame = io.read_geometry(aims_output_dir)

    # Load the basis set definition
    lmax, nmax = parser.extract_basis_set_info(frame, aims_output_dir)

    # ===== RI coefficients

    # Load to numpy
    coeffs_numpy = np.loadtxt(join(aims_output_dir, "ri_restart_coeffs.out"))
    
    # Save to numpy to compress, then remove the original text file
    np.save(join(aims_output_dir, "ri_restart_coeffs.npy"), coeffs_numpy)
    os.remove(join(aims_output_dir, "ri_restart_coeffs.out"))

    # Standard block sparse format
    coeffs_mts = convert.coeff_vector_ndarray_to_tensormap(
        frame,
        coeff_vector=coeffs_numpy,
        lmax=lmax,
        nmax=nmax,
        structure_idx=structure_idx,
        backend="numpy",
    )
    metatensor.save(join(save_dir, "ri_coeffs.npz"), coeffs_mts)

    # ===== RI projections

    # Load to numpy
    projs_numpy = np.loadtxt(join(aims_output_dir, "ri_projections.out"))

    # Save to numpy to compress, then remove the original text file
    np.save(join(aims_output_dir, "ri_projections.npy"), projs_numpy)
    os.remove(join(aims_output_dir, "ri_projections.out"))

    # Standard block sparse format
    projs_mts = convert.coeff_vector_ndarray_to_tensormap(
        frame,
        coeff_vector=projs_numpy,
        lmax=lmax,
        nmax=nmax,
        structure_idx=structure_idx,
        backend="numpy",
    )
    metatensor.save(join(save_dir, "ri_projs.npz"), projs_mts)

    # ===== RI overlap

    # Load the overlap matrix to a full square matrix
    ovlp_numpy = load_ovlp_matrix_to_square_matrix_numpy(aims_output_dir, "ri_ovlp.out")

    # Save to numpy to compress, then remove the original text file
    np.save(join(aims_output_dir, "ri_ovlp.npy"), ovlp_numpy)
    os.remove(join(aims_output_dir, "ri_ovlp.out"))

    # Convert to TensorMap and save
    ovlp = convert.overlap_matrix_ndarray_to_tensormap(
        frame,
        overlap_matrix=ovlp_numpy,
        lmax=lmax,
        nmax=nmax,
        structure_idx=structure_idx,
        backend="numpy",
    )

    # Parse basis function radii
    max_radii = parser.get_max_overlap_radius_by_type(aims_output_dir)
    pickle_dict(
        join(save_dir, "max_bf_radii.pickle"),
        max_radii,
    )

    # Cutoff overlap if applicable
    if cutoff_ovlp:
        ovlp = mask.cutoff_overlap_matrix(
            frames=[frame],
            frame_idxs=[structure_idx],
            overlap_matrix=ovlp,
            radii=max_radii,
            drop_empty_blocks=True,
            backend="numpy",
        )

    # Remove overlaps that are below a threshold
    if ovlp_sparsity_threshold is not None:
        ovlp = mask.sparsify_overlap_matrix(
            overlap_matrix=ovlp,
            sparsity_threshold=ovlp_sparsity_threshold,
            drop_empty_blocks=True,
            backend="numpy",
        )
    
    # Save
    metatensor.save(
        join(save_dir, "ri_ovlp.npz"),
        utils.make_contiguous_numpy(ovlp),
    )

    # Calculate the condition number of the overlap matrix and save
    if ovlp_cond_num:
        cond_num = np.linalg.cond(ovlp_numpy)
        pickle_dict(
            join(save_dir, "ri_ovlp_cond_num.pickle"),
            {"cond": cond_num},
        )

    # 2) Parse aims.out
    aims_out_parsed = parse_aims_out(aims_output_dir)
    pickle_dict(join(save_dir, "calc_info.pickle"), aims_out_parsed)

    # 3) Save basis set definition
    lmax, nmax = extract_basis_set_info(frame, aims_output_dir)
    pickle_dict(
        join(save_dir, "basis_set.pickle"),
        {"lmax": lmax, "nmax": nmax},
    )

    # ===== Scalar fields =====

    # Load the scalar field data
    partition_tab = np.loadtxt(join(aims_output_dir, "partition_tab.out"))
    rho_scf = np.loadtxt(join(aims_output_dir, "rho_scf.out"))
    rho_rebuilt_ri = np.loadtxt(join(aims_output_dir, "rho_rebuilt_ri.out"))

    # Check grid consistency
    assert np.all(partition_tab[:, :3] == rho_scf[:, :3])
    assert np.all(partition_tab[:, :3] == rho_rebuilt_ri[:, :3])
    assert np.all(rho_scf[:, :3] == rho_rebuilt_ri[:, :3])

    # Compress with numpy. For the densities, just store the field values
    np.save(join(aims_output_dir, "partition_tab.npy"), partition_tab)
    np.save(join(aims_output_dir, "rho_scf.npy"), rho_scf[:, 3])
    np.save(join(aims_output_dir, "rho_rebuilt_ri.npy"), rho_rebuilt_ri[:, :3])

    # Remove original text files
    os.remove(join(aims_output_dir, "partition_tab.out"))
    os.remove(join(aims_output_dir, "rho_scf.out"))
    os.remove(join(aims_output_dir, "rho_rebuilt_ri.out"))
    
    return

def process_df_error(
    aims_output_dir: str,
    save_dir: str,
    masked_system_type: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Processes the density fitting error of the RI density reconstruction versus the SCF
    density. If ``masked_system_type`` and the corresponding mask kwargs are passed, the
    scalar field coordinates are masked to only evaluate the error on the active region.

    Saves this to "df_error.pickle", or "df_error_masked.pickle" if
    ``masked_system_type`` is specified.
    """
    # Load fields
    input = np.load(join(aims_output_dir, "rho_rebuilt_ri.npy"))
    target = np.load(join(aims_output_dir, "rho_scf.npy"))
    grid = np.load(join(aims_output_dir, "partition_tab.npy"))

    # Build a coordinate mask, if applicable
    if masked_system_type is not None:
        grid_mask = mask.get_point_indices_by_region(
                points=grid[:, :3],
                masked_system_type=masked_system_type,
                region="active",
                **kwargs,
            )
        input = input[grid_mask]
        target = target[grid_mask]
        grid = grid[grid_mask]

    # Calc errors
    abs_error, norm = fields.field_absolute_error(
        input,
        target,
        grid,
    )
    squared_error, norm = fields.field_squared_error(
        input,
        target,
        grid,
    )
    pickle_dict(
        join(
            save_dir,
            "df_error.pickle" if masked_system_type is None else f"df_error_masked.pickle",
        ),
        {
            "abs_error": abs_error,
            "squared_error": squared_error,
            "norm": norm,
            "mae_percent": 100 * abs_error / norm,
            "mse_percent": 100 * squared_error / norm,
        },
    )

    return

def check_converged(aims_output_dir: str) -> bool:
    """
    Checks aims calculation is converged.
    """
    with open(join(aims_output_dir, "aims.out"), "r") as f:
        if "Have a nice day." in f.read():
            return True
        return False


def parse_fermi_energy(aims_output_dir: str) -> float:
    """
    Extracts the Fermi energy (chemical potential) from "aims.out".
    """
    with open(join(aims_output_dir, "aims.out"), "r") as f:
        lines = f.readlines()

    for line in lines[::-1]:  # read from bottom up
        if "| Chemical potential (Fermi level):" in line:
            return float(line.split()[-2])
    raise ValueError("Fermi energy not found in aims.out")


def parse_eigenvalues(aims_output_dir: str) -> List[List[float]]:
    """
    Parses the eigenvalues from the output file of an AIMS calculation.
    """

    echunk = False  # Determines if we are in the output chunk with the eigenenergies
    first = True
    energies = []
    k_energy = []
    with open(join(aims_output_dir, "Final_KS_eigenvalues.dat"), "r") as f:
        while True:
            line = f.readline()
            if "k-point number:" in line:
                echunk = False  # We have reached the start of the next k-point
                # Save the stored eigenenergies for each k-point, unless its the first
                # one
                if first:
                    first = False
                else:
                    energies.append(k_energy)
            if echunk:
                try:
                    energy = float(line.split()[-1])
                    k_energy.append(energy)
                except:  # noqa: B001, E722
                    # TODO: remove bare except here
                    pass

            if "k-point in cartesian units" in line:
                echunk = True
                k_energy = []
            if line == "":
                energies.append(k_energy)
                break

    return energies


def spline_eigenenergies(
    aims_output_dir: str,
    frame_idx: int,
    sigma: float,
    min_energy: float,
    max_energy: float,
    interval: float,
    dtype: Optional[torch.dtype] = torch.float64,
    save_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Splines a list of list of eigenenergies for each k-point to a common grid.
    """
    # Parse eigenenergies
    frame = io.read_geometry(aims_output_dir)
    energies = parser.parse_eigenvalues(aims_output_dir)

    # Store number of k-points and flatten eigenenergies
    n_kpts = len(energies)
    energies = torch.tensor(energies, dtype=dtype).flatten()

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

    splines_mts = metatensor.torch.TensorMap(
        keys=metatensor.torch.Labels.single(),
        blocks=[
            metatensor.torch.TensorBlock(
                samples=metatensor.torch.Labels(
                    ["system"],
                    torch.tensor([frame_idx], dtype=torch.int64).reshape(-1, 1),
                ),
                components=[
                    metatensor.torch.Labels(["coeffs"], torch.arange(4).reshape(-1, 1)),
                ],
                properties=metatensor.torch.Labels(
                    ["point"],
                    torch.arange(n_grid_points - 1, dtype=torch.int64).reshape(-1, 1),
                ),
                values=splines,
            )
        ],
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        metatensor.torch.save(join(save_dir, "dos_spline.npz"), splines_mts)

    return splines_mts


def parse_angle_resolved_pdos(aims_output_dir: str, frame: Optional[Frame] = None):
    """
    Parses the angle resolved PDOS data in files named, for instance:

        "atom_proj_dos_tetrahedron_Au0001.dat"

    and produced by setting the FHI-aims tag `output` to, for instance:

        "atom_proj_dos_tetrahedron -30 5 1000 0.3",

    where the arguments are the energy range, number of points, and Gaussian width.

    Returns a dictionary of the parsed data, where each key is a (symbol, l) tuple, and
    the values are a dictionary of PDOS arrays indexed by atom index (NOTE: 1-indexing).

    ``frame`` can be optionally passed. If none, it is read from "geometry.in" in
    ``aims_output_dir``.
    """

    if frame is None:
        frame = io.read_geometry(aims_output_dir)

    pdos_data = {}
    for i, sym in enumerate(system.get_symbols(frame), start=1):

        # Read atom PDOS file
        pdos_atom = np.loadtxt(
            join(aims_output_dir, f"atom_proj_dos_tetrahedron_{sym}{str(i).zfill(4)}.dat")
        )

        # Store the energy array only once
        if i == 1:
            energy = pdos_atom[:, 0]

        # Store total PDOS (non-angle resolved)
        if ("total", sym) not in pdos_data:
            pdos_data[("total", sym)] = {}
        pdos_data[("total", sym)][i] = pdos_atom[:, 1]

        for o3_lambda in range(pdos_atom.shape[1] - 2):
            if (o3_lambda, sym) not in pdos_data:
                pdos_data[(o3_lambda, sym)] = {}
            pdos_data[(o3_lambda, sym)][i] = pdos_atom[:, o3_lambda + 2]

    return energy, pdos_data