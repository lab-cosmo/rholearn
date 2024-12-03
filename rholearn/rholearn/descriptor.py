"""
Module containing classes to compute descriptors.
"""
import warnings
from typing import Callable, List, Optional, Tuple, Union

import metatensor.torch as mts
import torch
from chemfiles import Atom, Frame
from metatensor.torch.learn import nn
from metatensor.torch.learn.nn.equivariant_transformation import _CovariantTransform
from metatensor.torch.learn.nn.layer_norm import _LayerNorm
from rascaline.torch import SphericalExpansion
from rascaline.torch.utils import ClebschGordanProduct, DensityCorrelations
from rascaline.torch.utils.clebsch_gordan._density_correlations import (
    _filter_redundant_keys,
    _increment_property_names,
)

from rholearn.rholearn import mask, pretrainer, train_utils
from rholearn.utils import _dispatch, system

SUPPORTED_DESCRIPTORS = [
    "rho1",
    "rho1_x_rho1",
    "rho1_x_rho1_x_rho1",
    "rho2",
    "rho3",
    "rho2_p_rho1",
    "rho3_p_rho2_p_rho1",
    "rho_x_V",
]


class DescriptorCalculator(torch.nn.Module):
    """
    Transforms a :py:class:`chemfiles.Frame` into an atom centered density decriptor
    according to the specified hypers.
    """

    def __init__(
        self,
        in_keys: mts.Labels,
        descriptor_hypers: dict,
        center_types: List[int],
        dtype: torch.dtype,
        device: torch.device,
        masked_system_type: Optional[str] = None,
        **mask_kwargs,
    ) -> None:

        super().__init__()

        self._in_keys = in_keys
        self._descriptor_hypers = descriptor_hypers
        self._center_types = center_types
        self._dtype = dtype
        self._device = device
        self._masked_system_type = masked_system_type
        self._mask_kwargs = mask_kwargs

        # Initialize the specific calculators
        assert self._descriptor_hypers["descriptor_type"] in SUPPORTED_DESCRIPTORS, (
            f"self._descriptor_hypers['descriptor_type'] must be"
            f" one of: {SUPPORTED_DESCRIPTORS},"
            f" got: {self._descriptor_hypers['descriptor_type']}"
        )

        if self._descriptor_hypers["descriptor_type"] == "rho1":

            self._spherical_expansion_hypers = descriptor_hypers["spherical_expansion_hypers"]
            self._spherical_expansion = SphericalExpansion(
                **self._spherical_expansion_hypers
            )

        elif self._descriptor_hypers["descriptor_type"] == "rho2":

            self._spherical_expansion_hypers = descriptor_hypers["spherical_expansion_hypers"]
            if descriptor_hypers["angular_cutoff"] is None:
                max_angular = self._spherical_expansion_hypers["max_angular"] * 2
            else:
                max_angular = descriptor_hypers["angular_cutoff"]
            self._angular_cutoff = descriptor_hypers["angular_cutoff"]
            self._spherical_expansion = SphericalExpansion(
                **self._spherical_expansion_hypers
            )
            self._density_correlations = DensityCorrelations(
                n_correlations=1,
                max_angular=max_angular,
                skip_redundant=True,
                dtype=dtype,
                device=device,
            )

            if descriptor_hypers.get("epca_components") is None:
                self._epca = None
            else:
                self._epca = EquivariantPCA(descriptor_hypers["epca_components"])

        elif self._descriptor_hypers["descriptor_type"] == "rho1_x_rho1":

            self._spherical_expansion_1_hypers = descriptor_hypers["spherical_expansion_1_hypers"]
            self._spherical_expansion_2_hypers = descriptor_hypers["spherical_expansion_2_hypers"]
            if descriptor_hypers["angular_cutoff"] is None:
                max_angular = self._spherical_expansion_1_hypers["max_angular"] + self._spherical_expansion_2_hypers["max_angular"]
            else:
                max_angular = descriptor_hypers["angular_cutoff"]
            self._angular_cutoff = descriptor_hypers["angular_cutoff"]
            self._spherical_expansion_1 = SphericalExpansion(
                **self._spherical_expansion_1_hypers
            )
            self._spherical_expansion_2 = SphericalExpansion(
                **self._spherical_expansion_2_hypers
            )
            self._cg_product = ClebschGordanProduct(
                max_angular=max_angular,
                dtype=dtype,
                device=device,
            )

            if descriptor_hypers.get("epca_components") is None:
                self._epca = None
            else:
                self._epca = EquivariantPCA(descriptor_hypers["epca_components"])

        elif self._descriptor_hypers["descriptor_type"] == "rho1_x_rho1_x_rho1":

            self._spherical_expansion_1_hypers = descriptor_hypers["spherical_expansion_1_hypers"]
            self._spherical_expansion_2_hypers = descriptor_hypers["spherical_expansion_2_hypers"]
            self._spherical_expansion_3_hypers = descriptor_hypers["spherical_expansion_3_hypers"]
            if descriptor_hypers["angular_cutoff"] is None:
                max_angular = (
                    self._spherical_expansion_1_hypers["max_angular"] 
                    + self._spherical_expansion_2_hypers["max_angular"]
                    + self._spherical_expansion_3_hypers["max_angular"]
                )
            else:
                max_angular = descriptor_hypers["angular_cutoff"]
            self._angular_cutoff = descriptor_hypers["angular_cutoff"]
            self._spherical_expansion_1 = SphericalExpansion(
                **self._spherical_expansion_1_hypers
            )
            self._spherical_expansion_2 = SphericalExpansion(
                **self._spherical_expansion_2_hypers
            )
            self._spherical_expansion_3 = SphericalExpansion(
                **self._spherical_expansion_3_hypers
            )
            self._cg_product = ClebschGordanProduct(
                max_angular=max_angular,
                dtype=dtype,
                device=device,
            )

            if descriptor_hypers.get("epca_components") is None:
                self._epca_1 = None
                self._epca_2 = None
            else:
                self._epca_1 = EquivariantPCA(descriptor_hypers["epca_components"][0])
                self._epca_2 = EquivariantPCA(descriptor_hypers["epca_components"][1])

        elif self._descriptor_hypers["descriptor_type"] == "rho_x_V":

            self._spherical_expansion_hypers = descriptor_hypers["spherical_expansion_hypers"]
            self._lode_spherical_expansion_hypers = descriptor_hypers["lode_spherical_expansion_hypers"]
            if descriptor_hypers["angular_cutoff"] is None:
                max_angular = self._spherical_expansion_hypers["max_angular"] + self._lode_spherical_expansion_hypers["max_angular"]
            else:
                max_angular = descriptor_hypers["angular_cutoff"]
            self._angular_cutoff = descriptor_hypers["angular_cutoff"]
            self._spherical_expansion = SphericalExpansion(
                **self._spherical_expansion_hypers
            )
            self._lode_spherical_expansion = LodeSphericalExpansion(
                **self._lode_spherical_expansion_hypers
            )
            self._cg_product = ClebschGordanProduct(
                max_angular=max_angular,
                dtype=dtype,
                device=device,
            )

            if descriptor_hypers.get("epca_components") is None:
                self._epca = None
            else:
                self._epca = EquivariantPCA(descriptor_hypers["epca_components"])
        
        elif self._descriptor_hypers["descriptor_type"] == "rho3":

            self._spherical_expansion_hypers = descriptor_hypers["spherical_expansion_hypers"]
            if descriptor_hypers["angular_cutoff"] is None:
                max_angular = self._spherical_expansion_hypers["max_angular"] * 3
            else:
                max_angular = descriptor_hypers["angular_cutoff"]
            self._angular_cutoff = descriptor_hypers["angular_cutoff"]
            self._spherical_expansion = SphericalExpansion(
                **self._spherical_expansion_hypers
            )
            self._cg_product = ClebschGordanProduct(
                max_angular=max_angular,
                keys_filter=_filter_redundant_keys,
                dtype=dtype,
                device=device,
            )

            if descriptor_hypers.get("epca_components") is None:
                self._epca_1 = None
                self._epca_2 = None
            else:
                self._epca_1 = EquivariantPCA(descriptor_hypers["epca_components"])
                self._epca_2 = EquivariantPCA(descriptor_hypers["epca_components"])

        elif self._descriptor_hypers["descriptor_type"] == "rho2_p_rho1":
            self._spherical_expansion_1_hypers = descriptor_hypers["spherical_expansion_1_hypers"]
            self._spherical_expansion_2_hypers = descriptor_hypers["spherical_expansion_2_hypers"]
            if descriptor_hypers["angular_cutoff"] is None:
                max_angular = self._spherical_expansion_1_hypers["max_angular"] * 2
            else:
                max_angular = descriptor_hypers["angular_cutoff"]
            self._angular_cutoff = descriptor_hypers["angular_cutoff"]
            self._spherical_expansion_1 = SphericalExpansion(
                **self._spherical_expansion_1_hypers
            )
            self._spherical_expansion_2 = SphericalExpansion(
                **self._spherical_expansion_2_hypers
            )
            self._density_correlations = DensityCorrelations(
                n_correlations=1,
                max_angular=max_angular,
                skip_redundant=True,
                dtype=dtype,
                device=device,
            )

        elif self._descriptor_hypers["descriptor_type"] == "rho3_p_rho2_p_rho1":

            self._spherical_expansion_1_hypers = descriptor_hypers["spherical_expansion_1_hypers"]
            self._spherical_expansion_2_hypers = descriptor_hypers["spherical_expansion_2_hypers"]
            self._spherical_expansion_3_hypers = descriptor_hypers["spherical_expansion_3_hypers"]
            self._angular_cutoff = descriptor_hypers["angular_cutoff"]
            # Initialize spherical expansion calculators
            self._spherical_expansion_1 = SphericalExpansion(
                **self._spherical_expansion_1_hypers
            )
            self._spherical_expansion_2 = SphericalExpansion(
                **self._spherical_expansion_2_hypers
            )
            self._spherical_expansion_3 = SphericalExpansion(
                **self._spherical_expansion_3_hypers
            )
            # Initialize DensityCorrelations calculators
            if descriptor_hypers["angular_cutoff"] is None:
                max_angular_1 = self._spherical_expansion_1_hypers["max_angular"] * 3
                max_angular_2 = self._spherical_expansion_2_hypers["max_angular"] * 2
            else:
                max_angular_1 = descriptor_hypers["angular_cutoff"]
                max_angular_2 = descriptor_hypers["angular_cutoff"]
            self._density_correlations_1 = DensityCorrelations(
                n_correlations=2,
                max_angular=max_angular_1,
                skip_redundant=True,
                dtype=dtype,
                device=device,
            )
            self._density_correlations_2 = DensityCorrelations(
                n_correlations=1,
                max_angular=max_angular_2,
                skip_redundant=True,
                dtype=dtype,
                device=device,
            )

        else:
            raise ValueError(
                "``descriptor_hypers['descriptor_type']`` must be"
                f" one of {SUPPORTED_DESCRIPTORS}"
            )

    def compute(
        self,
        frames: List[Frame],
        frame_idxs: List[int] = None,
        mask_system: bool = False,
        reindex: bool = False,
        split_by_frame: bool = False,
    ) -> Union[mts.TensorMap, List[mts.TensorMap]]:
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
        """
        if mask_system is True:
            if self._masked_system_type is None:
                raise ValueError(
                    "cannot compute descriptor for a masked system if ``masked_system_type``"
                    " not passed to the constructor"
                )

        # Compute descriptor according to the specific algorithm
        descriptor = getattr(self, "_" + self._descriptor_hypers["descriptor_type"])(
            frames=frames, selected_keys=self._in_keys, mask_system=mask_system,
        )

        # Re-index "system" metadata along samples axis
        if reindex:
            descriptor = train_utils.reindex_tensormap(descriptor, frame_idxs)

        # Split by frame
        if split_by_frame:
            descriptor = train_utils.split_tensormap_by_system(descriptor, frame_idxs)

        return descriptor

    def forward(
        self,
        frames: List[Frame],
        frame_idxs: List[int] = None,
        mask_system: bool = None,
        reindex: bool = False,
        split_by_frame: bool = False,
    ) -> Union[mts.TensorMap, List[mts.TensorMap]]:
        """
        Calls the :py:meth:`compute` method.
        """
        return self.compute(
            frames=frames,
            frame_idxs=frame_idxs,
            mask_system=mask_system,
            reindex=reindex,
            split_by_frame=split_by_frame,
        )

    def _compute_density(
        self,
        frames: List[Frame],
        calculator: SphericalExpansion,
        mask_system: bool,
        selected_keys: Optional[mts.Labels] = None,
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
            density_active = calculator.compute(
                system.frame_to_atomistic_system(frames),
                selected_samples=mts.Labels(
                    names=["system", "atom"],
                    values=torch.tensor(active_atom_samples, dtype=torch.int64).reshape(-1, 2),
                ),
            )
            # Compute Spherical Expansion for atoms in the buffer region
            density_buffer = calculator.compute(
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

            # Make sure the density matches the target basis key selection
            if selected_keys is not None:
                key_idxs = density.keys.select(selected_keys)
                density = mts.TensorMap(
                    keys=mts.Labels(density.keys.names, density.keys.values[key_idxs]),
                    blocks=[block for i, block in enumerate(density) if i in key_idxs]
                )

        else:
            # Compute Spherical Expansion for the normal system
            density = calculator.compute(
                system.frame_to_atomistic_system(frames),
                selected_keys=selected_keys,
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

    def _rho1(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        mask_system: bool = None,
    ) -> mts.TensorMap:
        """
        Computes a spherical expansion atomic density
        """
        return self._compute_density(
            frames,
            calculator=self._spherical_expansion,
            selected_keys=selected_keys,
            mask_system=mask_system,
        )

    def _rho1_x_rho1(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        mask_system: bool = None,
    ) -> mts.TensorMap:
        """
        Computes an equivariant power spectrum descriptor.
        """
        # Compute the first density
        density_1 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_1,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Compute the second density
        density_2 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_2,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Rename properties
        density_1 = mts.rename_dimension(
            density_1, "properties", "n", "n_1"
        )
        density_1 = mts.rename_dimension(
            density_1, "properties", "neighbor_type", "neighbor_1_type"
        )

        density_2 = mts.rename_dimension(
            density_2, "properties", "n", "n_2"
        )
        density_2 = mts.rename_dimension(
            density_2, "properties", "neighbor_type", "neighbor_2_type"
        )

        # Take the CG product
        rho2 = self._cg_product.compute(
            density_1,
            density_2,
            o3_lambda_1_new_name="l_1",
            o3_lambda_2_new_name="l_2",
            selected_keys=selected_keys,
        )
        rho2 = rho2.keys_to_properties(["l_1", "l_2"])

        # Do PCA contraction
        if self._epca is not None:
            if not self._epca._is_fitted:
                self._epca.fit(rho2)

            rho2 = self._epca.transform(rho2)

        return rho2

    def _rho1_x_rho1_x_rho1(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        mask_system: bool = None,
    ) -> mts.TensorMap:
        """
        Computes an equivariant power spectrum descriptor.
        """
        # Compute the first density
        density_1 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_1,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Compute the second density
        density_2 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_2,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Compute the second density
        density_3 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_3,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Rename properties
        density_1 = mts.rename_dimension(
            density_1, "properties", "n", "n_1"
        )
        density_1 = mts.rename_dimension(
            density_1, "properties", "neighbor_type", "neighbor_1_type"
        )
        density_2 = mts.rename_dimension(
            density_2, "properties", "n", "n_2"
        )
        density_2 = mts.rename_dimension(
            density_2, "properties", "neighbor_type", "neighbor_2_type"
        )
        density_3 = mts.rename_dimension(
            density_3, "properties", "n", "n_3"
        )
        density_3 = mts.rename_dimension(
            density_3, "properties", "neighbor_type", "neighbor_3_type"
        )

        # Take the CG product
        rho2 = self._cg_product.compute(
            density_1,
            density_2,
            o3_lambda_1_new_name="l_1",
            o3_lambda_2_new_name="l_2",
            selected_keys=None,
        )

        # Do a PCA contraction
        if self._epca_1 is not None:
            if not self._epca_1._is_fitted:
                self._epca_1.fit(rho2)

            rho2 = self._epca_1.transform(rho2)

        # Take another CG product
        rho3 = self._cg_product.compute(
            rho2,
            density_3,
            o3_lambda_1_new_name="k_2",
            o3_lambda_2_new_name="l_3",
            selected_keys=selected_keys,
        )
        rho3 = rho3.keys_to_properties(["l_1", "l_2", "k_2", "l_3"])

        # Do another PCA contraction
        if self._epca_2 is not None:
            if not self._epca_2._is_fitted:
                self._epca_2.fit(rho3)

            rho3 = self._epca_2.transform(rho3)

        return rho3

    def _rho_x_V(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        mask_system: bool = None,
    ) -> mts.TensorMap:
        """
        Computes an equivariant power spectrum descriptor.
        """
        density_sr = self._compute_density(
            frames,
            calculator=self._spherical_expansion,
            selected_keys=None,
            mask_system=mask_system,
        )

    def _rho2(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        mask_system: bool = None,
    ) -> mts.TensorMap:
        """
        Computes an equivariant power spectrum descriptor.
        """
        # Compute density
        density = self._compute_density(
            frames,
            calculator=self._spherical_expansion,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Compute power spectrum
        rho2 = self._density_correlations.compute(
            density,
            selected_keys=selected_keys,
            angular_cutoff=self._angular_cutoff,
        )

        # Move CG keys to properties
        cg_key_names = []
        for i in range(1, 3):
            cg_key_names += [f"l_{i}"]
            if i > 2:
                cg_key_names += [f"k_{i-1}"]
        rho2 = rho2.keys_to_properties(cg_key_names)

        # Drop redundant properties
        rho2 = acdc_drop_redundant_properties(rho2, 2)

        # Do PCA contraction
        if self._epca is not None:
            if not self._epca._is_fitted:
                self._epca.fit(rho2)

            rho2 = self._epca.transform(rho2)

        return rho2

    def _rho3(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        mask_system: bool = None,
    ) -> mts.TensorMap:
        """
        Computes an equivariant power spectrum descriptor.
        """
        # Compute density
        density = self._compute_density(
            frames,
            calculator=self._spherical_expansion,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Compute power spectrum
        if self._angular_cutoff is None:
            angular_cutoff = None
        else:
            angular_cutoff = mts.Labels(
                names=["o3_lambda"],
                values=_dispatch.arange(
                    0, self._angular_cutoff + 1, "torch"
                ).reshape(-1, 1),
            )
        rho2 = self._cg_product.compute(
            _increment_property_names(density, 1),
            _increment_property_names(density, 2),
            o3_lambda_1_new_name="l_1",
            o3_lambda_2_new_name="l_2",
            selected_keys=angular_cutoff,
        )

        # Drop redundant properties
        rho2 = acdc_drop_redundant_properties(rho2, 2)

        # Do a PCA contraction, if applicable
        if self._epca_1 is not None:
            if not self._epca_1._is_fitted():
                self._epca_1.fit(rho2)

            rho2 = self._epca_1.transform(rho2)

        # Compute bispectrum
        rho3 = self._cg_product.compute(
            rho2,
            _increment_property_names(density, 3),
            o3_lambda_1_new_name="k_2",
            o3_lambda_2_new_name="l_3",
            selected_keys=selected_keys,
        )

        # Move CG keys to properties
        cg_key_names = []
        for i in range(1, 4):
            cg_key_names += [f"l_{i}"]
            if i > 2:
                cg_key_names += [f"k_{i-1}"]
        rho3 = rho3.keys_to_properties(cg_key_names)

        # # Drop redundant properties
        # TODO: is this valid?
        # rho3 = acdc_drop_redundant_properties(rho3, 3)

        # Do a PCA contraction, if applicable
        if self._epca_2 is not None:
            if not self._epca_2._is_fitted():
                self._epca_2.fit(rho3)

            rho3 = self._epca_2.transform(rho3)

        return rho3

    def _rho2_p_rho1(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        mask_system: bool = None,
    ) -> mts.TensorMap:
        """
        Computes:
            1. equivariant power spectrum
            2. spherical expansion

        and concatenates the features.
        """
        # Compute density for making the power spectrum
        density = self._compute_density(
            frames,
            calculator=self._spherical_expansion_1,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Compute the power spectrum
        rho2 = self._density_correlations.compute(
            density,
            selected_keys=selected_keys,
            angular_cutoff=self._angular_cutoff,
        )

        # Move CG keys to properties
        cg_key_names = []
        for i in range(1, 3):
            cg_key_names += [f"l_{i}"]
            if i > 2:
                cg_key_names += [f"k_{i-1}"]
        rho2 = rho2.keys_to_properties(cg_key_names)

        # Compute the density for feature concatenation
        rho1 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_2,
            selected_keys=selected_keys,
            mask_system=mask_system,
        )

        # Concatenate features
        def _rename_properties(tensor, nu: int):
            new_blocks = []
            for key, block in tensor.items():
                new_blocks.append(
                    mts.TensorBlock(
                        samples=block.samples,
                        components=block.components,
                        properties=mts.Labels(
                            ["nu", "q"],
                            torch.tensor(
                                [[nu, i] for i in range(len(block.properties))],
                                dtype=torch.int64,
                            ),
                        ),
                        values=block.values,
                    )
                )
            return mts.TensorMap(tensor.keys, new_blocks)

        descriptor = mts.join(
            [_rename_properties(rho2, 2), _rename_properties(rho1, 1)],
            axis="properties",
            remove_tensor_name=True,
        )

        return descriptor

    def _rho3_p_rho2_p_rho1(
        self,
        frames: List[Frame],
        selected_keys: Optional[mts.Labels] = None,
        mask_system: bool = None,
    ) -> mts.TensorMap:
        """
        Computes:
            1. equivariant bispectrum
            2. equivariant power spectrum
            3. spherical expansion

        and concatenates the features.
        """
        # Compute density for making the bispectrum
        density_1 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_1,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Compute the bispectrum
        rho3 = self._density_correlations_1.compute(
            density_1,
            selected_keys=selected_keys,
            angular_cutoff=self._angular_cutoff,
        )

        # Move CG keys to properties
        cg_key_names = []
        for i in range(1, 4):
            cg_key_names += [f"l_{i}"]
            if i > 2:
                cg_key_names += [f"k_{i-1}"]
        rho3 = rho3.keys_to_properties(cg_key_names)

        # Compute density for making the power spectrum
        density_2 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_2,
            selected_keys=None,
            mask_system=mask_system,
        )

        # Compute the power spectrum
        rho2 = self._density_correlations_2.compute(
            density_2,
            selected_keys=selected_keys,
            angular_cutoff=self._angular_cutoff,
        )

        # Move CG keys to properties
        cg_key_names = []
        for i in range(1, 3):
            cg_key_names += [f"l_{i}"]
            if i > 2:
                cg_key_names += [f"k_{i-1}"]
        rho2 = rho2.keys_to_properties(cg_key_names)

        # Compute the density for feature concatenation
        rho1 = self._compute_density(
            frames,
            calculator=self._spherical_expansion_3,
            selected_keys=selected_keys,
            mask_system=mask_system,
        )

        # Concatenate features
        def _rename_properties(tensor, nu: int):
            new_blocks = []
            for key, block in tensor.items():
                new_blocks.append(
                    mts.TensorBlock(
                        samples=block.samples,
                        components=block.components,
                        properties=mts.Labels(
                            ["nu", "q"],
                            torch.tensor(
                                [[nu, i] for i in range(len(block.properties))],
                                dtype=torch.int64,
                            ),
                        ),
                        values=block.values,
                    )
                )
            return mts.TensorMap(tensor.keys, new_blocks)

        descriptor = mts.join(
            [
                _rename_properties(rho3, 3),
                _rename_properties(rho2, 2),
                _rename_properties(rho1, 1)
            ],
            axis="properties",
            remove_tensor_name=True,
            different_keys="union",
        )

        return descriptor

    def __repr__(self) -> str:
        return (
            "DescriptorCalculator("
            f"\n\tdescriptor_hypers={self._descriptor_hypers},"
            f"\n\tcenter_types={self._center_types},"
            f"\n\tdtype={self._dtype},"
            f"\n\tdevice={self._device},"
            f"\n\tmasked_system_type={self._masked_system_type},"
            f"\n\tmask_kwargs={self._mask_kwargs},"
            "\n)"
        )


class EquivariantPCA(torch.nn.Module):
    """
    Scikit-learn-like Principal Component Analysis for TensorMaps
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        verbose: Optional[bool] = True,
        key_lambda_name: Optional[str] = "o3_lambda",
    ) -> None:

        super().__init__()

        self.n_components = n_components
        self.verbose = verbose
        self.key_lambda_name = key_lambda_name
        self._is_fitted = False

    @staticmethod
    def _get_mean(values: torch.tensor, lambda_key: int) -> float:
        # TODO: use mean for invariants. For now, zero mean for all.
        return 0.0
        # if l == 0:
        #     sums = np.sum(values.detach().numpy(), axis=1)
        #     signs = torch.from_numpy(((sums <= 0) - 0.5) * 2.0)
        #     values = signs[:, None] * values
        #     mean = torch.mean(values, dim=0)
        #     return mean
        # else:
        #     return 0.0

    def _svdsolve(self, X: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = self._svd_flip(U, Vt)
        eigs = S**2 / (X.shape[0] - 1)
        return eigs, Vt.T

    @staticmethod
    def _svd_flip(u: torch.tensor, v: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """translated from sklearn implementation"""
        max_abs_cols = torch.argmax(abs(u), axis=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
        return u, v

    def _fit(self, values: torch.tensor, lambda_key: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        nsamples, ncomps, nprops = values.shape
        values = values.reshape(nsamples * ncomps, nprops)

        mean = self._get_mean(values, lambda_key)
        eigs, components = self._svdsolve(values - mean)

        return mean, eigs, components

    def fit(self, tensor: mts.TensorMap) -> None:
        """Fit the EPCA"""
        if self._is_fitted:
            raise RuntimeError(f"EquivariantPCA is already fitted.")

        values_: List[torch.tensor] = []
        explained_variance_: List[torch.tensor] = []
        explained_variance_ratio_: List[torch.tensor] = []
        components_: List[torch.tensor] = []
        retained_components_: List[torch.tensor] = []

        for key, block in tensor.items():
            lambda_key = key[self.key_lambda_name]

            values = block.values.clone()
            nsamples, ncomps, nprops = values.shape

            if nsamples <= 1:
                retained = _dispatch.int_array(range(0), "torch")
                eigs = None
                eigs_ratio = None
                components = None
            else:
                # Perform SVD
                mean, eigs, components = self._fit(values, lambda_key=lambda_key)
                eigs_ratio = eigs / sum(eigs)
                eigs_csum = torch.cumsum(eigs_ratio, dim=0)
                if self.n_components is None:
                    max_comp = components.shape[1]
                elif 0 < self.n_components < 1:
                    max_comp = (eigs_csum > self.retain_variance).nonzero()[1, 0]
                elif self.n_components < min(nsamples * ncomps, nprops):
                    max_comp = self.n_components
                else:
                    # Use all the available components
                    warnings.warn(
                        (
                            f"n_components={self.n_components} too big: "
                            "retaining everything"
                        ),
                        stacklevel=3,
                    )
                    max_comp = min(nsamples * ncomps, nprops)
                retained = _dispatch.int_array(range(max_comp), "torch")
                eigs = eigs[retained]
                eigs_ratio = eigs_ratio[retained]
                components = components[:, retained]

                values = values.reshape(nsamples * ncomps, nprops)
                values = (values - mean) @ components
                values = values.reshape(nsamples, ncomps, len(retained))

            # Append values for new TensorMap
            values_.append(values)

            # Append PCA information
            explained_variance_.append(eigs)
            explained_variance_ratio_.append(eigs_ratio)
            components_.append(components)
            retained_components_.append(retained)

        # Update class attributes
        self._in_keys = tensor.keys
        self.values_ = values_
        self.explained_variance_ = explained_variance_
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.components_ = components_
        self.retained_components_ = retained_components_

        # Flag that the PCA is fitted
        self._is_fitted = True

    def transform(self, tensor: mts.TensorMap) -> mts.TensorMap:
        """Transform a new TensorMap using the stored fit"""
        if not self._is_fitted:
            raise RuntimeError(f"EquivariantPCA is not fitted.")

        blocks: List[mts.TensorBlock] = []

        for idx, block in enumerate(tensor.blocks()):
            # Retrieve components and mean for this block from the stored fit
            retained = self.retained_components_[idx]
            components = self.components_[idx]
            
            if components is None:
                raise RuntimeError(
                    f"No PCA components found for block {idx}. "
                    "Ensure that the TensorMap matches the fitted structure."
                )

            # Prepare the values to be transformed
            values = block.values.clone()
            nsamples, ncomps, nprops = values.shape
            values = values.reshape(nsamples * ncomps, nprops)

            # Perform the transformation using the stored components
            projected_values = values @ components
            projected_values = projected_values[:, retained]
            projected_values = projected_values.reshape(nsamples, ncomps, len(retained))

            # Construct a new block with the transformed values
            properties = mts.Labels(
                names=["pc"],
                values=_dispatch.int_array(
                    [[i] for i in range(len(retained))], "torch"
                ),
            )

            block = mts.TensorBlock(
                values=projected_values,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            blocks.append(block)

        return mts.TensorMap(tensor.keys, blocks)

    def fit_transform(self, tensor: mts.TensorMap) -> mts.TensorMap:
        return self.fit(tensor).transform(tensor)


# ===== Helper functions =====

def acdc_drop_redundant_properties(
    tensor: mts.TensorMap, correlation_order: int
) -> mts.TensorMap:
    """
    For the invariant blocks of the input atom-centered density correlation ``tensor``,
    drops the redundant properties.

    The properties that are kept are the ones that satisfy the inequality:

        "neighbor_1_type" <= "neighbor_2_type" <= ... <= "neighbor__{correlation_order}_type"
        and
        "n_1" <= "n_2" <= ... <= "n_{correlation_order}"
    """
    
    # Drop the redundant properties
    l_in_keys = "l_1" in tensor.keys.names
    new_blocks = []
    for key, block in tensor.items():

        # Keep covariant blocks as they are
        if key["o3_lambda"] != 0:
            new_blocks.append(block)
            continue

        # Check the l list is uniform if it's in the keys
        if l_in_keys:
            l_list = [key[f"l_{i}"] for i in range(1, correlation_order + 1)]
            assert all([l == l_list[0] for l in l_list])


        # Remove redundant properties from invariant blocks
        keep = []
        for i, p in enumerate(block.properties):
            
            if not l_in_keys:
                l_list = [p[f"l_{i}"] for i in range(1, correlation_order + 1)]
                assert all([l == l_list[0] for l in l_list])
            
            # Now filter based on "n" and "neighbor_type"
            n_list = [p[f"n_{i}"] for i in range(1, correlation_order + 1)]
            neigh_list = [p[f"neighbor_{i}_type"] for i in range(1, correlation_order + 1)]

            if sorted(n_list) == n_list and sorted(neigh_list) == neigh_list:
                keep.append(i)

        # Construct new block
        new_blocks.append(
            mts.TensorBlock(
                samples=block.samples,
                components=block.components,
                properties=mts.Labels(
                    block.properties.names,
                    block.properties.values[keep],
                ),
                values=block.values[..., keep],
            )
        )
    
    return mts.TensorMap(tensor.keys, new_blocks)

