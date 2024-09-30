"""
User settings for running model training. Use this file to overwrite settings found in
:py:mod:`rholearn.settings.defaults.ml_defaults`. 

For global variables that are dictionaries, the user settings are used to 'update' the
defaults - thus only overwriting the keys that are present in the user settings.

Any variables defined here must be included in the final dictionary `ML_SETTINGS` at
the end of this file.
"""
from functools import partial
import torch

# Cross-validation splits (number of frames)
N_TRAIN, N_VAL, N_TEST = 16, 8, 8

# Other user settings here:
# ...
# for instance, to override the following:

# For training the model
TRAIN = {
    "batch_size": 4,
    "n_epochs": 501,  # number of total epochs to run. End in a 1, i.e. 1001.
    "restart_epoch": None,  # The epoch of the last saved checkpoint, or None for no restart
    "val_interval": 100,  # validate every x intervals
    "log_interval": 100,  # epoch interval at which to log metrics
    "checkpoint_interval": 100,  # save model and optimizer state every x intervals
}

# Optimizer, partially initialized to allow for parametrization with the model later
OPTIMIZER = partial(torch.optim.Adam, **{"lr": 1e-1})

# Scheduler, partially initialized to allow for parametrization with the model later
SCHEDULER = partial(
    torch.optim.lr_scheduler.StepLR,
    **{"step_size": 2, "gamma": 0.1}  # NOTE: this is per validation step
)

# Final dictionary of ML settings
ML_SETTINGS = {
    "N_TRAIN": N_TRAIN,
    "N_VAL": N_VAL,
    "N_TEST": N_TEST,
    "TRAIN": TRAIN,
    "OPTIMIZER": OPTIMIZER,
    "SCHEDULER": SCHEDULER,
}