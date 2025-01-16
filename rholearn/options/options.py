from os.path import dirname, exists, join, realpath

import yaml

DIR_PATH = dirname(realpath(__file__))


def get_options(options_type: str, model: str = None, run_id: str = None):
    """
    Load the user options of type ``options_type`` "dft", "ml", "hpc" from the
    {options_type}-options.yaml file and override the corresponding DFT defaults.

    The user options are read from a yaml in the directory in which this function is
    called from.

    By contrast the defaults are read from the dft-defaults.yaml file in the
    :py:mod:`rholearn.options` module, from the ``model`` directory either "rholearn" or
    "doslearn".
    """
    assert options_type in [
        "dft",
        "ml",
        "hpc",
    ], f"Invalid options_type {options_type}. Must be 'dft' or 'ml'."

    if options_type in ["dft", "ml"]:
        assert model in [
            "rholearn",
            "doslearn",
        ], f"Invalid model {model}. Must be 'rholearn' or 'doslearn'."

    # First get the defaults
    if options_type == "hpc":
        defaults = {}
    else:
        defaults = get_defaults(options_type, model)

    # Load the user options
    if run_id is None:
        run_id = ""
    else:
        run_id = f"-{run_id}"
    file_path = f"{options_type}-options{run_id}.yaml"

    if not exists(file_path):
        raise FileNotFoundError(f"User options file {file_path} not found.")
    with open(file_path, "r") as f:
        user = yaml.safe_load(f)

    # Now update options with user options
    return update_options(defaults, user)


def get_defaults(options_type: str, model: str):
    """
    Load either the DFT or ML defaults from the dft-defaults.yaml or ml-defaults.yaml
    file, if ``options_type`` is "dft" or "ml" respectively.
    """
    assert options_type in [
        "dft",
        "ml",
    ], f"Invalid options_type {options_type}. Must be 'dft' or 'ml'."

    assert model in [
        "rholearn",
        "doslearn",
    ], f"Invalid model {model}. Must be 'rholearn' or 'doslearn'."

    with open(join(DIR_PATH, f"{model}-{options_type}-defaults.yaml"), "r") as f:
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
