import ctypes
import datetime
import gc
import os
import pickle
from typing import List, Union, Optional

import numpy as np
import torch

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap


def timestamp() -> str:
    """Return a timestamp string in format YYYY-MM-DD-HH:MM:SS."""
    return datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")


def make_contiguous_numpy(tensor: TensorMap) -> TensorMap:
    """
    Takes a TensorMap of numpy backend and makes the ndarray block and
    gradient values contiguous in memory.
    """
    new_blocks = []
    for key, block in tensor.items():
        new_block = TensorBlock(
            values=np.ascontiguousarray(block.values),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        for parameter, gradient in block.gradients():
            new_gradient = TensorBlock(
                values=np.ascontiguousarray(gradient.values),
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            )
            new_block.add_gradient(parameter, new_gradient)
        new_blocks.append(new_block)

    return TensorMap(
        keys=tensor.keys,
        blocks=new_blocks,
    )


def flatten_tensormap(
    tensor: TensorMap, backend: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Sorts the TensorMap and then flattens all block values and returns as a 1D numpy
    array.
    """
    try:
        tensor = mts.sort(tensor)
    except ValueError:
        tensor = mts.torch.sort(tensor)
    if backend == "numpy":
        flattened = np.array([])
        for block in tensor.blocks():
            flattened = np.concatenate((flattened, block.values.flatten()))
    elif backend == "torch":
        flattened = torch.tensor([])
        for block in tensor.blocks():
            flattened = torch.cat((flattened, block.values.flatten()))

    else:
        raise ValueError(f"Unknown backend {backend}, must be 'numpy' or 'torch'")

    return flattened


def num_elements_tensormap(tensor: TensorMap) -> int:
    """
    Returns the total number of elements in the input tensor.

    If the input tensor is a TensorMap the number of elements is given by the
    sum of the product of the dimensions for each block.

    If the input tensor is a TensorBlock or a torch.Tensor, the number of
    elements is just given by the product of the dimensions.
    """
    n_elems = 0
    if isinstance(tensor.block(0).values, np.ndarray):
        for block in tensor.blocks():
            n_elems += np.prod(block.values.shape)
    elif isinstance(tensor.block(0).values, torch.Tensor):
        for block in tensor.blocks():
            n_elems += torch.prod(torch.tensor(block.values.shape))

    return int(n_elems)
