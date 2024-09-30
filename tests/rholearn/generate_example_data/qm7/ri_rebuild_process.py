from os.path import join
import shutil

import numpy as np
import torch

import metatensor as mts

from rholearn.aims_interface import fields, io
from rholearn.utils import convert
from rholearn.utils.io import unpickle_dict

from dft_settings import *

ri_dir = lambda A: join(DATA_DIR, "raw", f"{A}", "edensity")
rebuild_dir = lambda A: join(ri_dir(A), "rebuild")
processed_dir = lambda A: join(DATA_DIR, "processed", f"{A}", "edensity")

for A in FRAME_IDXS:
    # Convert the ML coeffs to metatensor and save
    ml_coeffs_numpy = np.loadtxt(join(rebuild_dir(A), "ri_restart_coeffs.out"))
    basis_set = unpickle_dict(join(processed_dir(A), "basis_set.pickle"))
    ml_coeffs_mts = convert.coeff_vector_ndarray_to_tensormap(
        frame=io.read_geometry(ri_dir(A)),
        coeff_vector=ml_coeffs_numpy,
        structure_idx=A,
        lmax=basis_set["lmax"],
        nmax=basis_set["nmax"],
    )
    mts.save(join(processed_dir(A), "ml_coeffs.npz"), ml_coeffs_mts)

    # Compute the squared error
    squared_error, norm = fields.field_squared_error(
        input=np.loadtxt(join(rebuild_dir(A), "rho_rebuilt_ri.out")),
        target=np.loadtxt(join(ri_dir(A), "rho_rebuilt_ri.out")),
        grid=np.loadtxt(join(ri_dir(A), "partition_tab.out")),
    )
    torch.save(torch.tensor(squared_error), join(processed_dir(A), "squared_error.pt"))
