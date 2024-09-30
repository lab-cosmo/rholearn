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

This function is called when setting up a training calculation according to the specific
basis set decomposition of the target scalar field.

The sequential defined in this module is used to override the default architecture
present in :py:mod:`rholearn.settings.defaults.net`.
"""
# from typing import List

# import torch
# import metatensor.torch as mts
# from metatensor.torch.learn import nn

# def net(
#     in_keys: mts.Labels,
#     in_properties: List[mts.Labels],
#     out_properties: List[mts.Labels],
#     dtype: torch.dtype,
#     device: torch.device,
# ) -> torch.nn.Module:
#     """
#     Builds a NN sequential ModuleMap.
#     """
#     all_invariant_key_idxs = [
#         i for i, key in enumerate(in_keys) if key["o3_lambda"] == 0
#     ]
#     sequential = nn.Sequential(
#         in_keys,
#         ...,  # add layers here
#     )
#     return sequential


# Or, unless above function is defined:
net = None  


# Final global variable to be used, as a callable or None
NET = net