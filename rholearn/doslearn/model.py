from typing import List, Optional

import featomic.torch
import metatensor.torch as mts
import torch
from chemfiles import Atom, Frame
from featomic.torch import SoapPowerSpectrum
from metatensor.torch.learn import ModuleMap

from rholearn.doslearn import train_utils
from rholearn.rholearn import train_utils as rho_train_utils
from rholearn.utils import system

ALLOWED_ENERGY_REFS = [
    "Fermi-fixed",
    "Hartree-fixed",
    "Fermi-adaptive",
    "Hartree-adaptive",
]

class ExponentialLayer(torch.nn.Module):
    """
    Custom module to apply torch.exp
    """
    def forward(self, x):
        return torch.exp(x)

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
        self._soap_power_spectrum_calc = SoapPowerSpectrum(**soap_hypers)
        self._atom_types = atom_types
        self._min_energy = min_energy
        self._max_energy = max_energy
        self._interval = interval
        self._dtype = dtype
        self._device = device

        assert energy_reference in ALLOWED_ENERGY_REFS, (
            f"Energy reference must be in {ALLOWED_ENERGY_REFS}. Got: {energy_reference}"
        )
        self._energy_reference = energy_reference

        # Define the target DOS energy grid
        self._x_dos = train_utils.get_spline_positions(
            min_energy=self._min_energy,
            max_energy=self._max_energy,
            interval=self._interval,
        )

        # Infer feature dimension
        dummy_descriptor = self._get_dummy_descriptor()

        # Get metadata
        in_keys = dummy_descriptor.keys
        out_features = [len(self._x_dos) for _ in in_keys]
        out_properties = [
            mts.Labels(
                ["point"], torch.arange(len(self._x_dos), dtype=torch.int64).reshape(-1, 1)
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
                net.append(ExponentialLayer())
                nets.append(net)

        # Build a ModuleMap
        self._nn = ModuleMap(
            in_keys=in_keys,
            modules=nets,
            out_properties=out_properties,
        )

    def compute_descriptor(
        self,
        frames: List[Frame],
        frame_idxs: Optional[List[int]] = None,
    ) -> mts.TensorMap:
        """
        Compute the SOAP descriptor for a list of frames.
        """
        systems = featomic.torch.systems_to_torch(frames)
        soap = self._soap_power_spectrum_calc(systems)
        soap = soap.keys_to_properties(
            mts.Labels(
                names=["neighbor_1_type", "neighbor_2_type"],
                values=torch.tensor(
                    [
                        [j, k]
                        for j in self._atom_types
                        for k in self._atom_types
                    ]
                ),
            )
        )

        # Re-index "system" metadata along samples axis
        if frame_idxs is not None:
            soap = rho_train_utils.reindex_tensormap(soap, frame_idxs)

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
            prediction = rho_train_utils.reindex_tensormap(prediction, frame_idxs)

        # Split by frame
        if split_by_frame:
            prediction = rho_train_utils.split_tensormap_by_system(prediction, frame_idxs)

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

        # In the case of prediction we do not want a normalized DOS prediction, so sum
        # atomic contributions. This is in contrast to what would typically be done in
        # traning, whereby reduction over atoms is done by a mean, not a sum.
        prediction = mts.sum_over_samples(prediction, "center_type")
        prediction = mts.sum_over_samples(prediction, "atom")

        # Re-index "system" metadata along samples axis and split by frame
        prediction = rho_train_utils.reindex_tensormap(prediction, frame_idxs)
        prediction = rho_train_utils.split_tensormap_by_system(prediction, frame_idxs)

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
