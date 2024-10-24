from typing import List, Optional, Union
import torch

from chemfiles import Frame, Atom

import rascaline.torch
from rascaline.torch import SoapPowerSpectrum

import metatensor.torch as mts
from metatensor.torch.learn import ModuleMap

from rholearn.utils import system


class SoapDosNet(torch.nn.Module):
    """
    Constructs a SOAP-based model for the Density of States (DOS) of a system.
    """

    def __init__(
        self,
        soap_hypers: dict,
        atom_types: List[int],
        out_properties: List[mts.Labels],
        hidden_layer_widths: Optional[List[int]] = None,
        # n_train,
        # adaptive
        dtype: Optional[torch.dtype] = torch.float64,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        
        super(SoapDosNet,self).__init__()

        # Construct the descriptor calculator
        self._spherical_expansion_calc = SoapPowerSpectrum(**soap_hypers).to(dtype=dtype, device=device)
        self._atom_types = atom_types
        self._dtype = dtype
        self._device = device

        # Infer feature dimension
        dummy_descriptor = self._get_dummy_descriptor()

        # Get metadata
        in_keys = dummy_descriptor.keys
        if not isinstance(out_properties, list):
            raise TypeError("out_properties must be a list of mts.Labels")
        out_features = [len(out_props) for out_props in out_properties]
        
        # Initialize NNs for each block 
        if hidden_layer_widths is None:
            nets = [
                torch.nn.Linear(
                    in_features=len(dummy_descriptor[key].properties),
                    out_features=out_feats,
                )
                for key, out_feats in zip(in_keys, out_features)
            ]
        else:
            nets = []
            for key_i, key in enumerate(in_keys):

                net = torch.nn.Sequential()
                prev_width = len(dummy_descriptor[key].properties)
                block_hidden_layer_widths = hidden_layer_widths + [out_features[key_i]]
                for layer_i, width in enumerate(block_hidden_layer_widths):
                    net.append(torch.nn.Linear(
                        in_features=prev_width,
                        out_features=width,
                    ))
                    if layer_i < len(block_hidden_layer_widths) - 1:
                        net.append(torch.nn.SiLU())

                    prev_width = width
                nets.append(net)

        # Build a ModuleMap
        self._nn = ModuleMap(
            in_keys=in_keys,
            modules=nets,
            out_properties=out_properties,
        )
        self._nn.to(dtype=dtype, device=device)


    def compute_descriptor(
        self,
        frames: List[Frame],
        frame_idxs: Optional[List[int]] = None
    ) -> mts.TensorMap:
        """
        Compute the SOAP descriptor for a list of frames.
        """
        if frame_idxs is None:
            frame_idxs = list(range(len(frames)))
        systems = rascaline.torch.systems_to_torch(frames)
        soap = self._spherical_expansion_calc(systems)
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

        return soap

    def forward(
        self, frames: List[Frame], descriptor: Optional[mts.TensorMap] = None
    ) -> mts.TensorMap:
        """
        Computes a descriptor if not provided, and passes it through the neural network.
        """

        if descriptor is None:
            descriptor = self.compute_descriptor(frames).to(self._dtype)
        return self._nn.forward(descriptor)


    def _get_dummy_descriptor(self) -> mts.TensorMap:
        """
        Builds a dummy :py:class:`chemfiles.Frame` object from the global atom types in
        `descriptor_calculator._atom_types`, computes the descriptor for it, and extracts
        the properties for each block indexed by the keys in `in_keys`.
        """
        dummy_frame = Frame()
        for i, atom_type in enumerate(self._atom_types):
            dummy_frame.add_atom(
                Atom(
                    name=system.atomic_number_to_atomic_symbol(int(atom_type)),
                    type=system.atomic_number_to_atomic_symbol(int(atom_type)),
                ),
                position=[0, 0, 10 * i],
            )
        descriptor = self.compute_descriptor([dummy_frame])

        return descriptor
    