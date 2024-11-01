from typing import List, Optional

import metatensor.torch as mts
import rascaline.torch
import torch
from chemfiles import Atom, Frame
from metatensor.torch.learn import ModuleMap
from rascaline.torch import SoapPowerSpectrum

from rholearn.rholearn import train_utils
from rholearn.utils import system


class SoapDosNet(torch.nn.Module):
    """
    Constructs a SOAP-based model for the Density of States (DOS) of a system.
    """

    def __init__(
        self,
        soap_hypers: dict,
        atom_types: List[int],
        min_energy: float,
        max_energy: float,
        interval: float,
        energy_reference: str,
        hidden_layer_widths: Optional[List[int]] = None,
        dtype: Optional[torch.dtype] = torch.float64,
        device: Optional[torch.device] = "cpu",
    ) -> None:

        super(SoapDosNet, self).__init__()

        # Construct the descriptor calculator
        self._spherical_expansion_calc = SoapPowerSpectrum(**soap_hypers)
        self._atom_types = atom_types
        self._min_energy = min_energy
        self._max_energy = max_energy
        self._interval = interval
        self._dtype = dtype
        self._device = device

        assert energy_reference in [
            "Fermi",
            "Hartree",
        ], "Energy reference must be either 'Fermi' or 'Hartree'."
        self._energy_reference = energy_reference

        # Define the target DOS energy grid
        n_grid_points = int(
            torch.ceil(torch.tensor(max_energy - min_energy) / interval)
        )
        self._x_dos = min_energy + torch.arange(n_grid_points) * interval

        # Infer feature dimension
        dummy_descriptor = self._get_dummy_descriptor()

        # Get metadata
        in_keys = dummy_descriptor.keys
        out_features = [n_grid_points for _ in in_keys]
        out_properties = [
            mts.Labels(
                ["point"], torch.arange(n_grid_points, dtype=torch.int64).reshape(-1, 1)
            )
            for _ in atom_types
        ]

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
                    net.append(
                        torch.nn.Linear(
                            in_features=prev_width,
                            out_features=width,
                        )
                    )
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
        self._nn

    def compute_descriptor(
        self, frames: List[Frame], frame_idxs: Optional[List[int]] = None
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
        self,
        *,
        frames: List[Frame],
        frame_idxs: Optional[List[int]] = None,
        descriptor: Optional[mts.TensorMap] = None,
        split_by_frame: bool = False,
    ) -> mts.TensorMap:
        """
        Computes a descriptor if not provided, and passes it through the neural network
        to predict the local DOS per atom.
        """
        # Compute descriptor if not provided
        if descriptor is None:
            descriptor = self.compute_descriptor(frames).to(self._dtype)

        # Make prediction
        prediction = self._nn.forward(descriptor)

        # Move center_type to samples
        prediction = prediction.keys_to_samples(["center_type"])

        # Re-index "system" metadata along samples axis
        if frame_idxs is not None:
            prediction = train_utils.reindex_tensormap(prediction, frame_idxs)

        # Split by frame
        if split_by_frame:
            prediction = train_utils.split_tensormap_by_system(prediction, frame_idxs)

        return prediction

    def predict(
        self, frames: List[Frame], frame_idxs: Optional[List[int]] = None
    ) -> mts.TensorMap:
        """
        Computes a descriptor if not provided, and passes it through the neural network
        to predict the global DOS per structure.
        """
        # Make prediction
        prediction = self.forward(
            frames=frames,
            frame_idxs=None,
            descriptor=None,
            split_by_frame=False,
        )

        # Sum atomic contributions to get global DOS
        prediction = mts.sum_over_samples(prediction, "atom")

        # Re-index "system" metadata along samples axis and split by frame
        prediction = train_utils.reindex_tensormap(prediction, frame_idxs)
        prediction = train_utils.split_tensormap_by_system(prediction, frame_idxs)

        return prediction

    def _get_dummy_descriptor(self) -> mts.TensorMap:
        """
        Builds a dummy :py:class:`chemfiles.Frame` object from the global atom types in
        `descriptor_calculator._atom_types`, computes the descriptor for it, and
        extracts the properties for each block indexed by the keys in `in_keys`.
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
