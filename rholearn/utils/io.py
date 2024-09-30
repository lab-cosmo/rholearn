"""
Helper functions for file I/O operations and logging.
"""
import os
import pickle
from rholearn.utils.utils import timestamp


def log(
    log_path: str, line: str, comment: bool = True, use_timestamp: bool = True
) -> None:
    """
    Writes the string in `line` to the file at `log_path`, inserting a newline
    character at the end. By default a '#' is prefixed to every line, followed
    optionally by a timestamp.
    """
    log_line = ""
    if comment:
        log_line = "#"
    if use_timestamp:
        log_line += " " + timestamp() + " "
    log_line += line
    if os.path.exists(log_path):
        with open(log_path, "a") as f:
            f.write(log_line + "\n")
    else:
        with open(log_path, "w") as f:
            f.write(log_line + "\n")

    return


def pickle_dict(path: str, dict: dict) -> None:
    """
    Pickles a dict at the specified absolute path. Add a .pickle suffix if
    not given in the path.
    """
    if not path.endswith(".pickle"):
        path += ".pickle"
    with open(path, "wb") as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_dict(path: str) -> dict:
    """
    Unpickles a dict object from the specified absolute path and returns
    it.
    """
    with open(path, "rb") as handle:
        d = pickle.load(handle)
    return d