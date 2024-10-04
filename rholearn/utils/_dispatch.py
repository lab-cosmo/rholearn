import metatensor
import metatensor.torch
import numpy as np
import torch


def array(array, backend: str):

    if backend == "torch":
        return torch.tensor(array)

    elif backend == "numpy":
        return np.array(array)

    raise ValueError(f"Unknown backend: {backend}")


def arange(start, stop, backend: str = None):

    if backend == "torch":
        return torch.arange(start, stop)

    elif backend == "numpy":
        return np.arange(start, stop)

    raise ValueError(f"Unknown backend: {backend}")


def int_array(array, backend: str):

    if backend == "torch":
        return torch.tensor(array, dtype=torch.int32)

    elif backend == "numpy":
        return np.array(array, dtype=np.int32)

    raise ValueError(f"Unknown backend: {backend}")


def zeros(*shape, backend: str):

    if backend == "torch":
        return torch.zeros(*shape)

    elif backend == "numpy":
        return np.zeros(*shape)

    raise ValueError(f"Unknown backend: {backend}")


def labels(names, values, backend: str):

    if backend == "torch":
        return metatensor.torch.Labels(names, torch.tensor(values, dtype=torch.int32))

    elif backend == "numpy":
        return metatensor.Labels(names, np.array(values, dtype=np.int32))

    raise ValueError(f"Unknown backend: {backend}")


def stack(arrays, axis, backend: str):

    if backend == "torch":
        return torch.stack(arrays, dim=axis)

    elif backend == "numpy":
        return np.stack(arrays, axis=axis)

    raise ValueError(f"Unknown backend: {backend}")


# def sort(array, axis, backend: str):

#     if backend == "torch":
#         return torch.sort(array, dim=axis)

#     elif backend == "numpy":
#         return np.sort(array, axis=axis)

#     raise ValueError(f"Unknown backend: {backend}")
