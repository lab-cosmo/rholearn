"""
Module containing the global net class `RhoModel`.
"""
from typing import Callable, List, Optional, Tuple, Union

import metatensor.torch as mts
import torch
from chemfiles import Atom, Frame
from metatensor.torch.learn import nn
from rascaline.torch import SphericalExpansion
from rascaline.torch.utils import DensityCorrelations

from rholearn.rholearn import mask, train_utils
from rholearn.utils import system


class _DescriptorCalculator(torch.nn.Module):
    """
    Transforms a :py:class:`chemfiles.Frame` into an equivariant power spectrum
    descriptor, according to the specified hypers.
    """

    def __init__(
        self,
        center_types: List[int],
        spherical_expansion_hypers: dict,
        n_correlations: int,
        dtype: torch.dtype,
        device: torch.device,
        angular_cutoff: Optional[int] = None,
        masked_system_type: Optional[str] = None,
        **mask_kwargs,
    ) -> None:

        super().__init__()

        self._center_types = center_types
        self._spherical_expansion_hypers = spherical_expansion_hypers

        self._spherical_expansion = SphericalExpansion(**spherical_expansion_hypers)
        self._n_correlations = n_correlations

        if angular_cutoff is None:
            max_angular = spherical_expansion_hypers["max_angular"] * (
                n_correlations + 1
            )
        else:
            max_angular = angular_cutoff
        self._max_angular = max_angular
        self._angular_cutoff = angular_cutoff
        self._dtype = dtype
        self._device = device

        self._density_correlations = DensityCorrelations(
            n_correlations=n_correlations,
            max_angular=max_angular,
            skip_redundant=True,
            dtype=dtype,
            device=device,
        )
        self._masked_system_type = masked_system_type
        self._mask_kwargs = mask_kwargs

    def compute_density(
        self,
        frames: List[Frame],
        mask_system: bool,
    ) -> mts.TensorMap:
        """
        Computes the density for the given frames.

        If using a masked system type, only active and buffer region atoms are computed.
        """
        # If using a masked system, compute separate densities for active and buffer
        # region atoms and combine the descriptors. This ensure the neighbor types are
        # the 'normal' ones, i.e. those in self._center_types, but that any buffer atoms
        # present are re-typed and exist as their own central species.
        if mask_system:

            # Get selected samples for the active and buffer atoms
            active_atom_samples = []
            buffer_atom_samples = []
            for A, frame in enumerate(frames):
                active_atom_idxs = mask.get_point_indices_by_region(
                    points=frame.positions,
                    masked_system_type=self._masked_system_type,
                    region="active",
                    **self._mask_kwargs,
                )
                buffer_atom_idxs = mask.get_point_indices_by_region(
                    points=frame.positions,
                    masked_system_type=self._masked_system_type,
                    region="buffer",
                    **self._mask_kwargs,
                )
                for i in active_atom_idxs:
                    active_atom_samples.extend([A, i])
                for i in buffer_atom_idxs:
                    buffer_atom_samples.extend([A, i])
            
            # Compute Spherical Expansion for atoms in the active region
            density_active = self._spherical_expansion.compute(
                system.frame_to_atomistic_system(frames),
                selected_samples=mts.Labels(
                    names=["system", "atom"],
                    values=torch.tensor(active_atom_samples, dtype=torch.int64).reshape(-1, 2),
                ),
            )
            # Compute Spherical Expansion for atoms in the buffer region
            density_buffer = self._spherical_expansion.compute(
                system.frame_to_atomistic_system(frames),
                selected_samples=mts.Labels(
                    names=["system", "atom"],
                    values=torch.tensor(buffer_atom_samples, dtype=torch.int64).reshape(-1, 2),
                ),
            )
            # Re-type the center species, i.e. for carbon 6 -> 1006, for gold 79 -> 1079
            new_key_vals = density_buffer.keys.values
            new_key_vals[:, density_buffer.keys.names.index("center_type")] += 1000
            density_buffer = mts.TensorMap(
                keys=mts.Labels(
                    names=density_buffer.keys.names,
                    values=new_key_vals,
                ),
                blocks=density_buffer.blocks(),
            )

            # Join the TensorMaps for the active and buffer regions
            density = mts.join(
                [density_active, density_buffer],
                "samples",
                remove_tensor_name=True,
                different_keys="union",
            )
            density = mask._drop_empty_blocks(density, "torch")

        else:
            # Compute Spherical Expansion for the normal system
            density = self._spherical_expansion.compute(
                system.frame_to_atomistic_system(frames),
            )

        # Move 'neighbor_type' to properties, accounting for neighbor types not
        # necessarily present in the frames but present globally. In the case of a
        # masked system, these should still only be the standard atom types, not the
        # buffer region atom types.
        density = density.keys_to_properties(
            keys_to_move=mts.Labels(
                names=["neighbor_type"],
                values=torch.tensor(
                    [a for a in self._center_types if a // 1000 == 0],
                    dtype=torch.int64,
                ).reshape(-1, 1),
            )
        )

        return density

    def compute(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        compute_metadata: bool = False,
        mask_system: bool = False,
    ) -> mts.TensorMap:
        """
        Takes as input a :py:class:`chemfiles.Frame` frames. Computes an equivariant
        power spectrum descriptor and returns a :py:class:`TensorMap` object.

        In the returned descriptor, the "system" dimension of the samples is indexed
        numerically for each frame passed in ``frames``.

        The steps are as follows:

        1) Computes a density using the SphericalExpansion calculator. In the case of a
           masked system type, active and (retyped) buffer region atoms are computed.
        2) Moves the 'neighbor_type' key to properties, ensuring consistent properties
           for the global atom types stored in the class' ``_center_types`` attribute.
        3) Computes the lambda-SOAP descriptor using the DensityCorrelations calculator.
        4) Removes the redundant 'o3_sigma' key name from the descriptor.
        5) Drops any blocks whose keys are not found in ``selected_keys``.

        :param frames: a list of :py:class:`chemfiles.Frame` object to compute
            descriptors for.
        :param selected_keys: a :py:class:`mts.Labels` object containing the keys to
            select from the descriptor. If None, all keys are computed.
        """
        if mask_system is True:
            if self._masked_system_type is None:
                raise ValueError(
                    "cannot compute descriptor for a masked system if ``masked_system_type``"
                    " not passed to the constructor"
                )

        # Compute density
        density = self.compute_density(frames, mask_system=mask_system)

        # Compute lambda-SOAP
        if compute_metadata:
            descriptor = self._density_correlations.compute_metadata(
                density,
                selected_keys=selected_keys,
                angular_cutoff=self._angular_cutoff,
            )
        else:
            descriptor = self._density_correlations.compute(
                density,
                selected_keys=selected_keys,
                angular_cutoff=self._angular_cutoff,
            )

        cg_key_names = []
        for i in range(1, self._n_correlations + 2):
            cg_key_names += [f"l_{i}"]
            if i > 2:
                cg_key_names += [f"k_{i-1}"]
        descriptor = descriptor.keys_to_properties(cg_key_names)

        return descriptor

    def forward(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        compute_metadata: bool = False,
        mask_system: bool = None,
    ) -> Union[mts.TensorMap, List[mts.TensorMap]]:
        """
        Calls the :py:meth:`compute` method.
        """
        return self.compute(
            frames=frames,
            selected_keys=selected_keys,
            compute_metadata=compute_metadata,
            mask_system=mask_system,
        )

    def __repr__(self) -> str:
        return (
            "DescriptorCalculator("
            f"\n\tcenter_types={self._center_types},"
            f"\n\tspherical_expansion_hypers={self._spherical_expansion_hypers},"
            f"\n\tn_correlations={self._n_correlations},"
            f"\n\tangular_cutoff={self._angular_cutoff},"
            f"\n\tdtype={self._dtype},"
            f"\n\tdevice={self._device},"
            f"\n\tmasked_system_type={self._masked_system_type},"
            f"\n\tmask_kwargs={self._mask_kwargs},"
            "\n)"
        )


class RhoModel(torch.nn.Module):
    """
    Global model class for predicting a target field on an equivariant basis.

    :param spherical_expansion_hypers: `dict`, the hypers for the spherical expansion
        calculator.
    :param n_correlations: `int`, the number of Clebsch Gordan tensor products to take
        of the spherical expansion. This builds the body order of the equivariant
        descriptor. ``n_correlations=1`` forms an equivariant power spectrum
        ("lambda-SOAP)", ``n_correlations=2`` forms an equivariant bispectrum, etc.
    :param target_basis: :py:class:`Labels` object containing the basis set definition.
        This must contain 3 dimensions, respectively named: ["o3_lambda", "center_type",
        "nmax"]. This indicates the number of radial functions for each combination of
        "o3_lambda" and "center_type". The target basis must be defines the basis for
        each atom type that the model can predict on. Any systems or systems passed to
        :py:meth:`forward` or :py:meth:`predict` respectively must contain only atoms of
        these types.
    :param layer_norm: 
    :param nn_layers:
    :param dtype: `torch.dtype`, the data type for the model.
    :param device: `torch.device`, the device for the model.
    :param angular_cutoff: `int`, the maximum angular momentum to compute in
        intermediate Clebsch Gordan tensor products. If None, the maximum angular
        momentum is set to `spherical_expansion_hypers["max_angular"] * (n_correlations
        + 1)`.
    :param masked_system_type: 
    """

    def __init__(
        self,
        *,
        spherical_expansion_hypers: dict,
        n_correlations: int,
        target_basis: mts.Labels,
        layer_norm: bool = False,
        nn_layers: Optional[List[dict]] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device = "cpu",
        angular_cutoff: Optional[int] = None,
        masked_system_type: Optional[str] = None,
        **mask_kwargs,
    ) -> None:

        super().__init__()

        # Torch settings
        self._dtype = dtype
        self._device = device

        # Set target basis
        self._target_basis = target_basis

        # Set the atom types
        self._center_types = [
            i.item()
            for i in torch.unique(
                torch.tensor(target_basis.column("center_type")), sorted=True
            )
        ]

        # Set descriptor calculator
        self._descriptor_calculator = _DescriptorCalculator(
            center_types=self._center_types,
            spherical_expansion_hypers=spherical_expansion_hypers,
            n_correlations=n_correlations,
            dtype=dtype,
            device=device,
            angular_cutoff=angular_cutoff,
            masked_system_type=masked_system_type,
            **mask_kwargs,
        )
        self._masked_system_type = masked_system_type
        self._mask_kwargs = mask_kwargs

        # Set metadata
        self._in_keys = _target_basis_set_to_in_keys(self._target_basis)
        self._in_properties = _center_types_to_descriptor_basis_in_properties(
            self._in_keys,
            self._descriptor_calculator,
        )
        self._out_properties = _target_basis_set_to_out_properties(
            self._in_keys,
            self._target_basis,
        )

        # Set NN
        self._set_net(layer_norm, nn_layers)

    def _set_net(
        self, layer_norm: bool = False, nn_layers: Optional[List[callable]] = None
    ) -> None:
        """
        Initializes the NN by calling the `net` Callable passed to the constructor. The
        NN is initialized with the model metadata, i.e. the attributes ``_in_keys``,
        ``_in_properties``, and ``_out_properties``, and the torch settings ``_dtype``
        and ``_device``.
        """
        if nn_layers is None:
            nn_layers = []

        init_layers = []

        # Start with a layer norm if requested
        if layer_norm:
            init_layers.append(
                nn.InvariantLayerNorm(
                    in_keys=self._in_keys,
                    in_features=[
                        len(in_props)
                        for key, in_props in zip(self._in_keys, self._in_properties)
                        if key["o3_lambda"] == 0
                    ],
                )
            )

        # If no layers passed, just use a linear layer
        if len(nn_layers) == 0:
            init_layers.append(
                nn.EquivariantLinear(
                    in_keys=self._in_keys,
                    in_features=[len(in_props) for in_props in self._in_properties],
                    out_properties=self._out_properties,
                    invariant_keys=mts.Labels(
                        names=["o3_lambda", "o3_sigma"],
                        values=torch.tensor([0, 1], dtype=torch.int64).reshape(-1, 2),
                    ),
                    bias=True,
                    dtype=self._dtype,
                    device=self._device,
                )
            )
        else:
            for layer_i, layer in enumerate(nn_layers):
                # Update layer-specific args with ones required by all modules
                assert len(layer) == 1, "Each layer must be a dict with a single key"
                module_name, args = layer.popitem()
                args.update(
                    dict(
                        in_keys=self._in_keys,
                    )
                )
                if module_name == "EquivariantLinear":
                    args.update(dict(dtype=self._dtype, device=self._device))

                # If the first layer, the in_features need setting dynamically
                if layer_i == 0:
                    assert (
                        "out_features" in args
                    ), "'out_features' must be passed for the first layer"
                    args.update(
                        dict(
                            in_features=[
                                len(in_props) for in_props in self._in_properties
                            ]
                        )
                    )

                # If the last layer, the out_properties need setting dynamically instead
                # of out_features
                if layer_i == len(nn_layers) - 1:
                    assert (
                        "in_features" in args
                    ), "'in_features' must be passed for the last layer"
                    assert (
                        "out_features" not in args
                    ), "'out_features' must not be passed for the last layer"
                    assert (
                        "out_properties" not in args
                    ), "'out_properties' must be passed for the last layer"
                    args.update(dict(out_properties=self._out_properties))

                # Initialize the layer
                init_layers.append(getattr(nn, module_name)(**args))

        # Build the sequential NN
        self._net = nn.Sequential(self._in_keys, *init_layers)


    def _check_descriptor_metadata(self, descriptor: mts.TensorMap) -> None:
        """
        Checks the metadata of the ``descriptor`` TensorMap against the model metadata
        """
        for key, in_props in zip(self._in_keys, self._in_properties):
            if key not in descriptor.keys:
                continue
            if not descriptor[key].properties == in_props:
                raise ValueError(
                    "properties not consistent between model and"
                    f" descriptor at key {key}."
                    f" model: {in_props}\n"
                    f" descriptor: {descriptor[key].properties}"
                )

    def compute_descriptor(
        self,
        frames: List[Frame],
        frame_idxs: List[int] = None,
        reindex: bool = False,
        split_by_frame: bool = False,
    ) -> Union[mts.TensorMap, List[mts.TensorMap]]:
        """
        Computes the descriptor for the given frames.

        If ``frame_idxs`` is passed, the descriptor :py:class:`TensorMap` is re-indexed
        along the "system" dimension to match the indices in ``frame_idxs``.

        If ``split_by_frame`` is True, returns a list of TensorMaps, one for each frame.
        Note that when split into per-system TensorMaps, each TensorMap will still have
        the sparse keys for the set of global atom types.
        """
        if reindex or split_by_frame:
            assert frame_idxs is not None, "`frame_idxs` must be passed to reindex"

        # Compute descriptor
        descriptor = self._descriptor_calculator(
            frames=frames,
            selected_keys=self._in_keys,
            compute_metadata=False,
            mask_system=self._masked_system_type is not None,
        )

        # Re-index "system" metadata along samples axis
        if reindex:
            descriptor = train_utils.reindex_tensormap(descriptor, frame_idxs)

        # Split by frame
        if split_by_frame:
            descriptor = train_utils.split_tensormap_by_system(descriptor, frame_idxs)

        return descriptor

    def apply_nn(
        self,
        descriptor: mts.TensorMap,
        frame_idxs: List[int] = None,
        reindex: bool = False,
        split_by_frame: bool = False,
        check_metadata: bool = True,
    ) -> mts.TensorMap:
        """
        Applies the NN to the given descriptor.

        If ``frame_idxs`` is passed, the prediction :py:class:`TensorMap` is re-indexed
        along the "system" dimension to match the indices in ``frame_idxs``.

        If ``split_by_frame`` is True, returns a list of TensorMaps, one for each
        frame.
        """
        if reindex or split_by_frame:
            assert frame_idxs is not None, "`frame_idxs` must be passed to reindex"

        # Check descriptor
        assert isinstance(
            descriptor, torch.ScriptObject
        ), f"Expected `descriptor` to be TensorMap, got {type(descriptor)}"
        if check_metadata:
            self._check_descriptor_metadata(descriptor)

        # Apply NN
        prediction = self._net(descriptor)

        # Re-index "system" metadata along samples axis
        if reindex:
            prediction = train_utils.reindex_tensormap(prediction, frame_idxs)

        # Split by frame
        if split_by_frame:
            prediction = train_utils.split_tensormap_by_system(prediction, frame_idxs)

        return prediction

    def forward(
        self,
        *,
        frames: List[Frame],
        frame_idxs: List[int],
        descriptor: List[mts.TensorMap] = None,
        split_by_frame: bool = False,
        check_metadata: bool = True,
    ) -> List:
        """
        Predicts basis coefficients for the given ``frames`` or ``descriptor``.

        If ``descriptor`` is None, it is first computed, and then passed through the NN.
        Otherwise if ``descriptor`` is passed, it is passed through the NN as is. It is
        assumed to have the desired metadata - i.e. the correct "system" sample
        dimension metadata to match the indices in ``frame_idxs``.

        If ``split_by_frame`` is True, returns a list of TensorMaps, one for each frame.

        Note that in this case, unlike the :py:meth:`compute_descriptor` and
        :py:meth:`apply_nn` methods, after splitting into per-frame TensorMaps, only
        blocks in each prediction that correspond to atom types in the respective
        ``frame`` will be kept.

        If ``check_metadata`` is True, the metadata of the descriptor is checked against
        the model metadata. This can be turned off for speed reasons.
        """
        # Calculate descriptor if not passed. Keep as a single TensorMap for pasing
        # through the NN. Ensure "system" dimension is re-indexed to match the
        # ``frame_idxs``.
        if descriptor is None:
            descriptor = self.compute_descriptor(
                frames=frames,
                frame_idxs=frame_idxs,
                reindex=True,
                split_by_frame=False,
            )

        # Pass descriptor through NN. Don't reindex the tensor (as assumed to have the
        # correct "system" sample dimension metadata), but do split if requested.
        prediction = self.apply_nn(
            descriptor=descriptor,
            frame_idxs=frame_idxs,
            reindex=False,
            split_by_frame=split_by_frame,
            check_metadata=check_metadata,
        )

        return prediction

        # # Drop blocks that don't correspond to the atom types in the respective frames
        # if split_by_frame:
        #     return [
        #         train_utils.drop_blocks_for_nonpresent_types(
        #             frame, pred, self._masked_system_type, **self._mask_kwargs
        #         )
        #         for frame, pred in zip(frames, prediction)
        #     ]

        # return train_utils.drop_blocks_for_nonpresent_types(
        #     frames, prediction, self._masked_system_type, **self._mask_kwargs
        # )

    def predict(
        self,
        frames: List[Frame],
        frame_idxs: List[int] = None,
    ) -> List[mts.TensorMap]:
        """
        Makes a prediction on a list of :py:class:`chemfiles.Frame` objects.

        If descriptors are computed with for a certain subset of selected atoms, the
        returned :py:class:`TensorMap` predictions are padded with zeros for samples
        that have been masked by this sample selection.

        ``frame_idxs`` can be passed to ensure returned predictions contain the correct
        frame IDs for each frame passed in ``frames``.
        """
        if not isinstance(frames, list):
            raise ValueError(
                f"Expected `frames` to be a list of Frames, got {type(frames)}"
            )

        if not all([isinstance(frame, Frame) for frame in frames]):
            raise ValueError("Expected `frames` to be a List[Frame]")

        if frame_idxs is None:
            frame_idxs = torch.arange(len(frames))

        self.eval()
        with torch.no_grad():  # make predictions
            predictions = self(
                frames=frames,
                frame_idxs=frame_idxs,
                split_by_frame=True,
                check_metadata=True,
            )

            # Return if no un-masking is necessary
            if self._masked_system_type is None:
                return predictions

            # Unmask the predictions if necessary
            predictions_unmasked = []
            for frame, frame_idx, pred in zip(frames, frame_idxs, predictions):
                pred_unmasked = mask.unmask_coeff_vector(
                    coeff_vector=pred,
                    frame=mask.retype_frame(
                        frame, self._masked_system_type, **self._mask_kwargs
                    ),
                    frame_idx=frame_idx,
                    in_keys=self._in_keys,
                    properties=self._out_properties,
                    backend="torch",
                )
                pred_unmasked = pred_unmasked.to(device=self._device, dtype=self._dtype)
                predictions_unmasked.append(pred_unmasked)

            return predictions_unmasked

    def __getitem__(self, i: int) -> nn.ModuleMap:
        """
        Gets the i-th module (i.e. corresponding to the i-th key/block) of the NN.
        """
        return self._net.module_map[i]

    def __iter__(self) -> Tuple[mts.LabelsEntry, nn.ModuleMap]:
        """
        Iterates over the model's NN modules, returning the key and block NN in a tuple.
        """
        return iter(zip(self._in_keys, self._net.module_map))

    def __repr__(self) -> str:
        representation = (
            "RhoModel("
            + "\n  target_basis="
            + str(self._target_basis).replace("\t", "\t").replace("\n", "\n  ")
            + "\n  descriptor_calculator="
            + str(self._descriptor_calculator).replace("\t", "\t").replace("\n", "\n  ")
        )
        representation += "\n  net="
        for key, block_nn in self:
            representation += f"\n    {key}:"
            representation += "\n\t" + str(block_nn).replace("\n", "\n\t")

        representation += f"\n  dtype={self._dtype},"
        representation += f"\n  device={self._device},"
        representation += "\n)"

        return representation


# ===== Helper functions =====


def _target_basis_set_to_in_keys(target_basis: mts.Labels) -> mts.Labels:
    """
    Converts the basis set definition to the set of in_keys on which the model is
    defined.

    Returned is a Labels object with the names "o3_lambda" and "center_type" and values,
    extracted from `target_basis`
    """
    assert target_basis.names[:2] == ["o3_lambda", "center_type"]
    return mts.Labels(
        names=["o3_lambda", "o3_sigma", "center_type"],
        values=torch.tensor(
            [
                [o3_lambda, 1, center_type]
                for o3_lambda, center_type in target_basis.values[:, :2]
            ],
            dtype=torch.int32,
        ).reshape(-1, 3),
    )


def _center_types_to_descriptor_basis_in_properties(
    in_keys: mts.Labels,
    descriptor_calculator: torch.nn.Module,
) -> List[mts.Labels]:
    """
    Builds a dummy :py:class:`chemfiles.Frame` object from the global atom types in
    `descriptor_calculator._center_types`, computes the descriptor for it, and extracts
    the properties for each block indexed by the keys in `in_keys`.

    For masked systems, the neighbor types are just the standard atom types, i.e. 1 and
    6 for H and C, not 1006 for pseudo-C.
    """
    dummy_frames = []
    for center_type in descriptor_calculator._center_types:
        dummy_frame = Frame()
        dummy_frame.add_atom(
            Atom(
                name=system.atomic_number_to_atomic_symbol(int(center_type)),
                type=system.atomic_number_to_atomic_symbol(int(center_type)),
            ),
            position=[0, 0, 0],
        )
        dummy_frames.append(dummy_frame)
    descriptor = descriptor_calculator(
        frames=dummy_frames,
        selected_keys=in_keys,
        compute_metadata=True,
        mask_system=False,
    )

    return [descriptor[key].properties for key in in_keys]


def _target_basis_set_to_out_properties(
    in_keys: mts.Labels,
    target_basis: mts.Labels,
) -> List[mts.Labels]:
    """
    Converts the basis set definition to a list of Labels objects corresponding to the
    out properties for each key in `in_keys`.

    Returned is a list of Labels, each of which enumerate the radial channels "n" for
    each combination of o3_lambda and center_type in `in_keys`, extracted from
    `target_basis`
    """
    out_properties = []
    for key in in_keys:
        o3_lambda, _, center_type = key

        nmax_idxs = torch.all(
            target_basis.values[:, :2] == torch.tensor([[o3_lambda, center_type]]),
            dim=1,
        )
        assert torch.sum(nmax_idxs) == 1
        nmax = target_basis.values[nmax_idxs, 2].item()

        out_properties.append(
            mts.Labels(
                names=["n"],
                values=torch.arange(nmax).reshape(-1, 1),
            )
        )
    return out_properties
