from os.path import join
import shutil

import numpy as np

from dft_settings import *

ri_dir = lambda A: join(DATA_DIR, "raw", f"{A}", "edensity")

for A in FRAME_IDXS:
    shutil.move(
        join(ri_dir(A), "ri_restart_coeffs.out"),
        join(ri_dir(A), "ri_restart_coeffs.out.copy"),
    )

    coeffs = np.loadtxt(join(ri_dir(A), "ri_restart_coeffs.out.copy"))
    coeffs += np.random.normal(0, 1e-2, coeffs.shape)

    np.savetxt(join(ri_dir(A), "ri_restart_coeffs.out"), coeffs)