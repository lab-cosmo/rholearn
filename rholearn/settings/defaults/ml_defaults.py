"""
Defaults for running model training.
"""

from functools import partial

import torch
from rholearn.loss import RhoLoss

# Define a seed. This is used to shuffle frame indices for cross-validation splits and
# intialize model parameters
SEED = 42

# Rascaline SphericalExpansion hyperparameters
SPHERICAL_EXPANSION_HYPERS = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 8,  # Exclusive
    "max_angular": 5,  # Inclusive
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
    "center_atom_weight": 1.0,
    "radial_scaling": {"Willatt2018": {"exponent": 4, "rate": 1, "scale": 3.5}},
}

# Number of CG tensor products to correlate spherical expansion
N_CORRELATIONS = 1

# Max angular momentum to compute in CG products. Anything <
# SPHERICAL_EXPANSION_HYPERS["max_angular"] * (N_CORRELATIONS + 1) loses information but
# speeds up computation.
ANGULAR_CUTOFF = None

# Value to cutoff the overlap matrix (Angstrom)
OVERLAP_CUTOFF = None

# For training the model
TRAIN = {
    "batch_size": 1,
    "n_epochs": 501,  # number of total epochs to run. End in a 1, i.e. 20001.
    "restart_epoch": None,  # The epoch of the last saved checkpoint, or None for no restart
    "val_interval": 50,  # validate every x intervals
    "log_interval": 50,  # epoch interval at which to log metrics
    "checkpoint_interval": 50,  # save model and optimizer state every x intervals
}

# For evaluating a model checkpoint
EVAL = {
    "eval_epoch": "best",
    "target_type": "ri",  # evaluate MAE against SCF ("scf") or RI ("ri") scalar field
}

# Torch factory settings
TORCH = {
    "dtype": torch.float64,  # important for model accuracy
    "device": torch.device(type="cpu"),
}

# Optimizer, partially initialized
OPTIMIZER = partial(torch.optim.Adam, **{"lr": 1e-1})

# Scheduler, partially initialized
SCHEDULER = partial(
    torch.optim.lr_scheduler.StepLR,
    **{"step_size": 10, "gamma": 0.1}  # NOTE: this is per validation step
)

# Training and validation loss functions
TRAIN_LOSS_FN = RhoLoss(  # full, non-truncated loss
    orthogonal=False,
    truncated=False,
)
VAL_LOSS_FN = RhoLoss(  # validation is *always* the full, non-truncated loss
    orthogonal=False,
    truncated=False,
)

# Data names, must be compatible with the data expected by the loss functions above
TRAIN_DATA_NAMES = {
    "target_c": "ri_coeffs.npz",
    "target_w": "ri_projs.npz",
    "overlap": "ri_ovlp.npz",
}
VAL_DATA_NAMES = {
    "target_c": "ri_coeffs.npz",
    "target_w": "ri_projs.npz",
    "overlap": "ri_ovlp.npz",
}

# Final dict of all global variables defined above
ML_DEFAULTS = {
    "ANGULAR_CUTOFF": ANGULAR_CUTOFF,
    "EVAL": EVAL,
    "GET_SELECTED_ATOMS": None,
    "N_CORRELATIONS": N_CORRELATIONS,
    "OPTIMIZER": OPTIMIZER,
    "OVERLAP_CUTOFF": OVERLAP_CUTOFF,
    "PRETRAINED_MODEL": None,
    "SCHEDULER": SCHEDULER,
    "SPHERICAL_EXPANSION_HYPERS": SPHERICAL_EXPANSION_HYPERS,
    "TORCH": TORCH,
    "TRAIN": TRAIN,
    "TRAIN_DATA_NAMES": TRAIN_DATA_NAMES,
    "TRAIN_LOSS_FN": TRAIN_LOSS_FN,
    "SEED": SEED,
    "VAL_DATA_NAMES": VAL_DATA_NAMES,
    "VAL_LOSS_FN": VAL_LOSS_FN,
}
