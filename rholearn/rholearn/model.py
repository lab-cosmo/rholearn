"""
Module containing the global net class `RhoModel`.
"""

from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from chemfiles import Frame
from metatensor.torch.learn import nn
from metatensor.torch.learn.nn.equivariant_transformation import \
    _CovariantTransform
from metatensor.torch.learn.nn.layer_norm import _LayerNorm

from rholearn.rholearn import mask, pretrainer, train_utils

SUPPORTED_NN_LAYERS = [
    "LayerNorm",
    "Tanh",
    "ReLU",
    "SiLU",
    "Linear",
    "CovariantTransform",
    "Identity",
    "Dropout",
]


class _Net(torch.nn.Module):
    """Wraps a :py:class:`ModuleMap` in a native :py:class:`torch.nn.Module`."""

    def __init__(self, module_map: nn.ModuleMap):
        super().__init__()
        self.module_map = module_map

    def forward(self, tensor: mts.TensorMap) -> mts.TensorMap:
        """Apply the transformation to the input tensor map `tensor`."""
        # Currently not supporting gradients
        if len(tensor[0].gradients_list()) != 0:
            raise ValueError(
                "Gradients not supported. Please use metatensor.remove_gradients()"
                " before using this module"
            )
        return self.module_map(tensor)


class RhoModel(torch.nn.Module):
    """
    Global model class for predicting a target field on an equivariant basis.

    :param descriptor_hypers: `dict`
    :param target_basis: :py:class:`Labels` object containing the basis set definition.
        This must contain 3 dimensions, respectively named: ["o3_lambda", "center_type",
        "nmax"]. This indicates the number of radial functions for each combination of
        "o3_lambda" and "center_type". The target basis must be defines the basis for
        each atom type that the model can predict on. Any systems or systems passed to
        :py:meth:`forward` or :py:meth:`predict` respectively must contain only atoms of
        these types.
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
        in_keys: mts.Labels,
        in_properties: List[mts.Labels],
        out_properties: List[mts.Labels],
        descriptor_calculator: torch.nn.Module,
        architecture: Dict[str, List[dict]],
        target_basis: mts.Labels,
        dtype: torch.dtype = torch.float64,
        device: torch.device = "cpu",
        pretrain: bool = False,
        pretrain_args: Optional[dict] = None,
        masked_system_type: Optional[str] = None,
        **mask_kwargs,
    ) -> None:

        super().__init__()

        # Torch settings
        self._dtype = dtype
        self._device = device

        # Set metadata
        self._in_keys = in_keys
        self._in_properties = in_properties
        self._out_properties = out_properties

        # Set descriptor calculator
        self._descriptor_calculator = descriptor_calculator

        # Set target basis
        self._target_basis = target_basis

        # Set architecture
        self._architecture = torch.nn.ModuleDict(
            {
                map_name: self._get_module_map(layers)
                for map_name, layers in architecture.items()
            }
        )

        # Initialize weights using RidgeCV - only applicable when the model is a single
        # linear layer
        self._pretrain = pretrain
        if self._pretrain:
            assert "linear" in self._architecture, (
                "Pretrainer only applicable when a 'linear' module is"
                " included in `architecture`"
            )
            self._pretrain_args = pretrain_args
            self._set_pretrainer(self._pretrain_args)

        # Create an attribute for storing the target standardizer
        self._standardizer = None

    def _get_module_map(self, layers: List[dict]) -> None:
        """
        Creates a ModuleMap for the NN transformation.
        """

        if len(layers) == 0:
            raise

        for layer in layers:
            assert len(layer) == 1
            layer_name = list(layer.keys())[0]
            assert layer_name in SUPPORTED_NN_LAYERS, (
                f"module name {layer_name} not in the list of supported layers:"
                f" {SUPPORTED_NN_LAYERS}"
            )

        # Build each block module
        block_modules = []
        for key_i, key in enumerate(self._in_keys):

            block_module = []
            prev_out_features = None
            for layer in layers:
                assert len(layer) == 1
                layer_name = list(layer.keys())[0]
                args = layer[layer_name]
                if isinstance(args, dict):
                    args = {**args}

                # If applying an identity, just append it
                if layer_name == "Identity":
                    block_module.append(torch.nn.Identity())
                    continue

                if layer_name == "Dropout":
                    block_module.append(torch.nn.Dropout(args["p"]))
                    continue

                # Do not apply these modules to covariants
                if (
                    layer_name in ["LayerNorm", "Tanh", "ReLU", "SiLU"]
                    and key["o3_lambda"] != 0
                ):
                    block_module.append(torch.nn.Identity())
                    continue

                # Only apply a linear bias to invariants
                if layer_name == "Linear":
                    if key["o3_lambda"] == 0:
                        args.update(dict(bias=True))
                    else:
                        args.update(dict(bias=False))

                    if "in_features" not in args.keys():
                        if prev_out_features is None:  # infer from descriptor size
                            args.update(
                                dict(in_features=len(self._in_properties[key_i]))
                            )
                        else:
                            args.update(dict(in_features=prev_out_features))

                    if (
                        "out_features" not in args.keys()
                    ):  # assume the output dimension is the target dimension
                        args.update(dict(out_features=len(self._out_properties[key_i])))

                    prev_out_features = args["out_features"]
                    block_module.append(torch.nn.Linear(**args))

                # If using a CovariantTransform, parse the sub-layers that comprise it
                elif layer_name == "CovariantTransform":
                    if prev_out_features is None:  # infer from descriptor size
                        in_features = len(self._in_properties[key_i])
                    else:
                        in_features = prev_out_features

                    transform_module = []
                    for sublayer in args:
                        assert len(sublayer) == 1, f"{sublayer}"
                        sublayer_name = list(sublayer.keys())[0]
                        sublayer_args = {**sublayer[sublayer_name]}
                        if sublayer_name == "Linear":
                            if "in_features" not in sublayer_args:
                                sublayer_args.update(dict(in_features=in_features))
                            if "out_features" not in sublayer_args:
                                sublayer_args.update(dict(out_features=in_features))
                        transform_module.append(
                            getattr(torch.nn, sublayer_name)(**sublayer_args)
                        )

                    block_module.append(
                        _CovariantTransform(
                            module=torch.nn.Sequential(*transform_module)
                        )
                    )

                elif layer_name == "LayerNorm":
                    if prev_out_features is None:  # infer from descriptor size
                        args.update(dict(in_features=len(self._in_properties[key_i])))
                    else:
                        args.update(dict(in_features=prev_out_features))
                    prev_out_features = args["in_features"]
                    block_module.append(_LayerNorm(**args))

                elif layer_name in ["Tanh", "ReLU", "SiLU"]:

                    block_module.append(getattr(torch.nn, layer_name)(**args))

                else:
                    raise ValueError(
                        f"layer name not supported. Must be one"
                        f" of: {SUPPORTED_NN_LAYERS}"
                    )

            block_modules.append(
                torch.nn.Sequential(*block_module).to(
                    dtype=self._dtype, device=self._device
                )
            )

        # Store the net as a ModuleMap
        return _Net(
            nn.ModuleMap(
                in_keys=self._in_keys,
                modules=block_modules,
                out_properties=self._out_properties,
            )
        )

    def _set_pretrainer(self, pretrain_args: Optional[dict]) -> None:
        """
        Sets the pretrainer model.
        """
        # Standard linear regression
        if pretrain_args.get("alphas") is None:
            self._pretrainer = pretrainer.RhoLinearRegression(
                in_keys=self._in_keys,
                in_properties=self._in_properties,
                out_properties=self._out_properties,
                dtype=self._dtype,
                device=self._device,
            )

        # RidgeCV
        else:
            self._pretrainer = pretrainer.RhoRidgeCV(
                in_keys=self._in_keys,
                in_properties=self._in_properties,
                out_properties=self._out_properties,
                dtype=self._dtype,
                device=self._device,
                alphas=[i.item() for i in torch.logspace(*pretrain_args.get("alphas"))],
            )

    def _initialize_weights_from_pretrainer(self) -> None:
        """
        Initializes the weights of the named module map "linear" using those fitted by
        the pretrainer.
        """
        for key_i, key in enumerate(self._in_keys):

            block_model = self._architecture["linear"].module_map[key_i]
            best_model = self._pretrainer._best_models[key_i]

            for i, param in enumerate(block_model.parameters()):
                if i == 0:  # weights
                    param.data = torch.nn.Parameter(
                        torch.tensor(
                            best_model.coef_,
                            dtype=self._dtype,
                            device=self._device,
                        )
                    )
                elif i == 1:  # biases
                    param.data = torch.nn.Parameter(
                        torch.tensor(
                            best_model.intercept_,
                            dtype=self._dtype,
                            device=self._device,
                        )
                    )
                else:
                    raise ValueError(
                        "expecting 2 parameter tensors in the model for block"
                        f" at key {key}, got {len(block_model.parameters())}"
                    )

        return

    def _set_standardizer(self, standardizer: mts.TensorMap) -> None:
        """
        Stores the passed ``standardizer``. This is used as a multiplier layer
        to un-standardize predicted targets in :py:meth:`forward`.
        """
        # Check metadata
        for key in self._in_keys:
            assert (
                key in standardizer.keys
            ), f"block indexed by key {key} missing from `standardizer`"
        for key, block in standardizer.items():
            assert (
                key in self._in_keys
            ), f"unexpected block at key {key} found in `standardizer`"
            assert block.properties == self._out_properties[self._in_keys.position(key)]

            # Turn off gradients
            block.values.requires_grad = False

        self._standardizer = standardizer

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

    def _add_modules(self, architecture) -> None:
        self._architecture.update(
            {
                map_name: self._get_module_map(layers)
                for map_name, layers in architecture.items()
            }
        )

    def _set_requires_grad(
        self,
        map_name: str,
        requires_grad: bool,
        selected_keys: Optional[mts.Labels] = None,
    ) -> None:
        """
        For the architecture ModuleMap of name ``map_name``, sets the ``requires_grad``
        attribute of all its parameters.
        """
        # If not using a key selection, just update `requires_grad` for all parameters
        # in the named module
        if selected_keys is None:
            for param in self._architecture[map_name].module_map.parameters():
                param.requires_grad = requires_grad

        # Otherwise, update the `requires_grad` attribute of parameters that match the
        # key selection
        else:
            _selection = self._in_keys.select(selected_keys)
            _selection = mts.Labels(
                self._in_keys.names, self._in_keys.values[_selection]
            )
            for key, block_nn in zip(
                self._in_keys, self._architecture[map_name].module_map
            ):
                if key in _selection:
                    for param in block_nn.parameters():
                        param.requires_grad = requires_grad

        return

    def _get_grad_norms(self) -> Dict[str, Dict[str, list]]:
        """
        Returns a list (one for each parameter) of gradient norms for each block model
        in each architecture module.
        """
        grad_norms = {}
        with torch.no_grad():
            for map_name in self._architecture.keys():
                grad_norms[map_name] = {}
                for key, block_nn in zip(
                    self._in_keys, self._architecture[map_name].module_map
                ):
                    grad_norms[map_name][key] = []
                    for param in block_nn.parameters():
                        grad = param.grad
                        if grad is not None:
                            grad_norms[map_name][key].append(torch.norm(grad).item())

        return grad_norms

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

        # Apply each architecture and sum the predictions
        prediction = list(self._architecture.values())[0](descriptor)
        if len(self._architecture) > 1:
            for mmap in list(self._architecture.values())[1:]:
                prediction = mts.add(prediction, mmap(descriptor))

        # Un-standardize the predictions, if applicable
        if self._standardizer is not None:
            prediction = train_utils.unstandardize_tensor(
                prediction, self._standardizer
            )

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
            descriptor = self._descriptor_calculator(
                frames=frames,
                frame_idxs=frame_idxs,
                mask_system=self._descriptor_calculator._masked_system_type is not None,
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
            if self._descriptor_calculator._masked_system_type is None:
                return predictions

            # Unmask the predictions if necessary
            predictions_unmasked = []
            for frame, frame_idx, pred in zip(frames, frame_idxs, predictions):
                pred_unmasked = mask.unmask_coeff_vector(
                    coeff_vector=pred,
                    frame=mask.retype_frame(
                        frame,
                        self._descriptor_calculator._masked_system_type,
                        **self._descriptor_calculator._mask_kwargs,
                    ),
                    frame_idx=frame_idx,
                    in_keys=self._in_keys,
                    properties=self._out_properties,
                    backend="torch",
                )
                pred_unmasked = pred_unmasked.to(device=self._device, dtype=self._dtype)
                predictions_unmasked.append(pred_unmasked)

            return predictions_unmasked

    def __getitem__(self, map_name: str, i: int) -> nn.ModuleMap:
        """
        Gets the i-th module (i.e. corresponding to the i-th key/block) of the NN.
        """
        return self._architecture[map_name].module_map[i]

    def __iter__(self, map_name: str) -> Tuple[mts.LabelsEntry, nn.ModuleMap]:
        """
        Iterates over the model's NN modules, returning the key and block NN in a tuple.
        """
        return iter(zip(self._in_keys, self._architecture[map_name].module_map))

    def __repr__(self) -> str:
        representation = (
            "RhoModel("
            + "\n  target_basis = "
            + str(self._target_basis).replace("\t", "\t").replace("\n", "\n  ")
            + "\n  descriptor_calculator = "
            + str(self._descriptor_calculator).replace("\t", "\t").replace("\n", "\n  ")
        )
        representation += "\n  architecture = "
        for map_name in self._architecture:
            representation += f"\n    {map_name}:"
            for key, block_nn in zip(
                self._in_keys, self._architecture[map_name].module_map
            ):
                representation += f"\n      {str(key).replace("LabelsEntry", "")}: "
                representation += "\n        " + str(block_nn).replace(
                    "\n", "\n          "
                )

        representation += f"\n  dtype = {self._dtype},"
        representation += f"\n  device = {self._device},"
        representation += "\n )"

        return representation
