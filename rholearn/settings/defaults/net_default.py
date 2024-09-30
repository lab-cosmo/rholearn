"""
Module for defining a neural network architecture based on `metatensor.torch.learn.nn`
modules. This should be defined as a function that takes the following input metadata as
(lists of) :py:class:`Labels` objects:

- in_keys: the sparse keys, with named dimensions ["o3_lambda", "o3_sigma",
  "center_type"], of the target scalar field decomposed onto an angular basis. These
  defined the keys for which the descriptor is computed.

- in_properties: a list of Labels, one for each key in `in_keys`, defining the
  properties the descriptor that is to be input to the NN.

- out_properties: a list of Labels, one for each key in `in_keys`, defining the
  properties of the output of the NN. These correspond to the radial basis indices of
  the basis-decomposed target scalar field.

such that it can be initialized on runtime according to the tuned basis set
decomposition of the target scalar field.
"""
from typing import List

import torch
import metatensor.torch as mts
from metatensor.torch.learn import nn


def net(
    in_keys: mts.Labels,
    in_properties: List[mts.Labels],
    out_properties: List[mts.Labels],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    """Builds a NN sequential ModuleMap. This default is just a linear layer."""
    all_invariant_key_idxs = [
        i for i, key in enumerate(in_keys) if key["o3_lambda"] == 0
    ]
    sequential = nn.Sequential(
        in_keys,
        nn.EquivariantLinear(
            in_keys=in_keys,
            invariant_key_idxs=all_invariant_key_idxs,
            in_features=[len(in_props) for in_props in in_properties],
            out_properties=out_properties,
            bias=True,  # bias is only applied to invariants
            dtype=dtype,
            device=device,
        ),
    )
    return sequential

# Final global variable to be used, as a callable
NET = net
