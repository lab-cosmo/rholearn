import metatensor
import metatensor.torch
import numpy as np
import torch


def abs(array, backend: str):

    if backend == "torch":
        return torch.abs(array)

    elif backend == "numpy":
        return np.abs(array)

    raise ValueError(f"Unknown backend: {backend}")

def all(array, backend: str):

    if backend == "torch":
        return torch.all(array)

    elif backend == "numpy":
        return np.all(array)

    raise ValueError(f"Unknown backend: {backend}")
    
def any(array, backend: str):

    if backend == "torch":
        return torch.any(array)

    elif backend == "numpy":
        return np.any(array)

    raise ValueError(f"Unknown backend: {backend}")


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


def labels(names, values, backend: str):

    if backend == "torch":
        return metatensor.torch.Labels(names, torch.tensor(values, dtype=torch.int32))

    elif backend == "numpy":
        return metatensor.Labels(names, np.array(values, dtype=np.int32))

    raise ValueError(f"Unknown backend: {backend}")


def max(array, axis, backend: str):

    if backend == "torch":
        return torch.max(array, dim=axis)

    elif backend == "numpy":
        return np.max(array, axis=axis)

    raise ValueError(f"Unknown backend: {backend}")

def mean(array, axis, backend: str):

    if backend == "torch":
        return torch.mean(array, dim=axis)

    elif backend == "numpy":
        return np.mean(array, axis=axis)

    raise ValueError(f"Unknown backend: {backend}")

def min(array, axis, backend: str):

    if backend == "torch":
        return torch.min(array, dim=axis)

    elif backend == "numpy":
        return np.min(array, axis=axis)

    raise ValueError(f"Unknown backend: {backend}")

# def sort(array, axis, backend: str):

#     if backend == "torch":
#         return torch.sort(array, dim=axis)

#     elif backend == "numpy":
#         return np.sort(array, axis=axis)

#     raise ValueError(f"Unknown backend: {backend}")


def stack(arrays, axis, backend: str):

    if backend == "torch":
        return torch.stack(arrays, dim=axis)

    elif backend == "numpy":
        return np.stack(arrays, axis=axis)

    raise ValueError(f"Unknown backend: {backend}")


def zeros(*shape, backend: str):

    if backend == "torch":
        return torch.zeros(*shape)

    elif backend == "numpy":
        return np.zeros(*shape)

    raise ValueError(f"Unknown backend: {backend}")
