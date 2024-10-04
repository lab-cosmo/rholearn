from os.path import dirname, join, realpath

import yaml

DIR_PATH = dirname(realpath(__file__))


def get_options(options_type: str):
    """
    Load the DFT user options from the dft-options.yaml file and override the
    corresponding DFT defaults.

    The user options are read from a yaml in the directory in which this function is
    called from.

    By contrast the defaults are read from the dft-defaults.yaml file in the
    :py:mod:`rholearn.options` module.
    """
    assert options_type in [
        "dft",
        "ml",
        "hpc",
    ], f"Invalid options_type {options_type}. Must be 'dft' or 'ml'."

    # First get the defaults
    if options_type == "hpc":
        defaults = {}
    else:
        defaults = get_defaults(options_type)

    # Now update with user settings
    with open(f"{options_type}-options.yaml", "r") as f:
        user = yaml.safe_load(f)

    return update_options(defaults, user)


def get_defaults(options_type: str):
    """
    Load either the DFT or ML defaults from the dft-defaults.yaml or ml-defaults.yaml
    file, if ``options_type`` is "dft" or "ml" respectively.
    """
    assert options_type in [
        "dft",
        "ml",
    ], f"Invalid options_type {options_type}. Must be 'dft' or 'ml'."

    with open(join(DIR_PATH, f"{options_type}-defaults.yaml"), "r") as f:
        return yaml.safe_load(f)


def update_options(default, user):
    """
    Update the default configuration with the user configuration.
    """
    for key, value in user.items():
        if isinstance(value, dict):
            default[key] = update_options(default.get(key, {}), value)
        else:
            default[key] = value
    return default
