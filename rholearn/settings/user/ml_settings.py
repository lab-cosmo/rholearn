"""
User settings for running model training. Use this file to overwrite settings found in
:py:mod:`rholearn.settings.defaults.ml_defaults`. 

For global variables that are dictionaries, the user settings are used to 'update' the
defaults - thus only overwriting the keys that are present in the user settings.

Any variables defined here must be included in the final dictionary `ML_SETTINGS` at
the end of this file.
"""

# Cross-validation splits (number of frames)
N_TRAIN, N_VAL, N_TEST = 0, 0, 0

# Other user settings here:
# ...

# Final dictionary of ML settings
ML_SETTINGS = {
    "N_TRAIN": N_TRAIN,
    "N_VAL": N_VAL,
    "N_TEST": N_TEST,
}