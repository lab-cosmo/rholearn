"""
Module for evaluating the loss between input and target scalar fields expanded on a
spherical basis.
"""

from typing import Optional

import metatensor.torch as mts
import torch

# Define the supported loss solvers and the required parameters for RhoLoss.forward
SOLVERS = {
    "orthogonal": ["input_c", "target_c"],
    "nonorthogonal_via_c": ["input_c", "target_c", "overlap"],
    "nonorthogonal_via_w": ["input_c", "target_c", "overlap", "target_w"],
}

class RhoLoss(torch.nn.Module):
    """
    Implements different solvers for L2 loss between two real space scalar fields:

    .. math::

        L = \\int_{\\mathcal{R}} d\\textbf{r} | ...
                \\rho^{input} (\\textbf{r}) - \\rho^{target} (\\textbf{r}) | ^ 2

    where both input and target scalar fields are expanded on the same spherical basis
    set.

    :param solver: a :py:class:`str` indicating the solver to use. Each solver requires
        different input data.
    :param truncated: :py:class:`bool`, whether or not to drop terms from the loss that
        have zero-derivative with respect to model parameters.
    :param conditioner: the overlap conditioner to use in loss evaluation. Only
        applicable for solvers "nonorthogonal_via_c" and "nonorthogonal_via_w". Assumes
        that ``overlap`` matrices passed to :py:meth:`forward` have already been
        conditioned by summation with the ``conditioner``-scaled identity matrix.
    """
    def __init__(
        self,
        solver: str,
        truncated: bool = False,
        conditioner: Optional[float] = None
    ) -> None:

        super().__init__()
        assert solver in SOLVERS, f"``solver`` must be one of: {SOLVERS}"
        self._solver = solver
        self._required_data = SOLVERS[solver]
        if truncated:
            if solver in ["orthogonal", "nonorthogonal_via_c"]:
                raise ValueError("``truncated`` can only be passed with ")
        self._truncated = truncated
        if conditioner is not None:
            if solver == "orthogonal":
                raise ValueError("``conditioner`` cannot be passed with solver='orthogonal'")
            assert truncated, "``truncated`` must be true if passing ``conditioner``"
            assert conditioner > 0, "conditioner must be > 0"
        self._conditioner = conditioner

    def forward(
        self,
        input_c: mts.TensorMap,
        target_c: Optional[mts.TensorMap] = None,
        target_w: Optional[mts.TensorMap] = None,
        overlap: Optional[mts.TensorMap] = None,
        check_metadata: bool = True,
    ) -> torch.Tensor:
        """
        Computes the loss with the specified ``solver``.

        :param truncated: :py:class:`bool`, whether to compute a truncated form of the
            loss whose gradient with respect to model parameters is equivalent to the
            non-truncated form. If true, terms with zero gradient wrt model parameters
            are not computed. Default false.
        :param conditioned: :py:class:`bool`, whether the input ``overlap``, if passed,
            should be assumed to be conditioned - i.e. relative to the true overlap if
            the idenitity matrix has been subtracted from it.
        """
        for req_data in self._required_data:
            assert locals().get(req_data) is not None, (
                f"``{req_data}`` data is required for solver {self._solver}"
            )

        # Orthogonal basis
        # L = (∆c) ^ 2
        if self._solver == "orthogonal":

            if check_metadata:
                _check_metadata_coefficients(input_c)
                _equal_metadata_raise(input_c, target_c, "input_c", "target_c")
                
            delta_c = mts.subtract(input_c, target_c)
            loss = _dot(delta_c, delta_c)

        # Non-orthogonal basis, via target coefficients
        #     L = ∆c S ∆c
        # or if conditioned:
        #     L = ∆c.(S + bI).∆c - b ∆cT.∆c
        elif self._solver == "nonorthogonal_via_c":            
            if check_metadata:
                _check_metadata_coefficients(input_c)
                _equal_metadata_raise(input_c, target_c, "input_c", "target_c")
                _check_metadata_overlap(overlap)

            delta_c = mts.subtract(input_c, target_c)
            loss = _cSc(delta_c, overlap)
            if self._conditioner is not None:
                loss -= self._conditioner * _dot(delta_c, delta_c)

        # Non-orthogonal basis, via target projections
        else:
            assert self._solver == "nonorthogonal_via_w"

            if check_metadata:
                _check_metadata_coefficients(input_c)
                _equal_metadata_raise(input_c, target_c, "input_c", "target_c")
                _equal_metadata_raise(input_c, target_w, "input_c", "target_w")
                _check_metadata_overlap(overlap)

            if self._conditioner is None:
                # Non-orthogonal basis, via target projections
                #     L = c^{inp}T.S.c^{inp} - 2 c^{inp}T.w^{tar}
                # if not truncated, the term is added:
                #     L += c^{tar}T.w^{tar}
                loss = _cSc(input_c, overlap) - 2 * _dot(input_c, target_w)
                if not self._truncated:
                    loss += _cSc(target_c, overlap)

            else:
                # Non-orthogonal basis, via target projections (conditioned)
                #     L = c^{inp}T.(S + bI).c^{inp} 
                #         - b ( c^{inp} + (1/b)w^{tar} )T.( c^{inp} + (1/b)w^{tar} )
                input_c_plus_target_w = mts.add(
                    input_c, mts.multiply(target_w, 1 / self._conditioner)
                )
                loss = (
                    _cSc(input_c, overlap) 
                    - self._conditioner 
                    * _dot(input_c_plus_target_w, input_c_plus_target_w)
                )
            
        return loss


# =========================== 
# ===== Base operations =====
# ===========================


def _cSc(
    coefficients: mts.TensorMap, overlap: mts.TensorMap,
) -> torch.Tensor:
    """
    Evaluates the double matrix multiplcation and reduces to a scalar:

    .. math::

        c . \\hat{S} . c

    where ``coefficients`` is the vector "c" and ``overlap`` is the matrix "S". Returns
    a scalar.

    ``overlap`` is a :py:class:`TensorMap` with the following metadata dimensions:

        - keys: ["center_1_type", "center_2_type"]
        - samples: ["system", "atom_1", "atom_2"]
        - components: ["l1_n1_m1"]
        - properties: ["l2_n2_m2"]

    the samples can be sliced to contain only a subset of overlaps between pairs of
    atoms, but sample indices must be symmetrized such that "atom_1" < "atom_2". As only
    the upper triangle of the overlap matrix (by atom index) is stored, the matrices
    stored for cross-center overlaps are assumed to be scaled by a factor of 2, where
    self-pairs are not scaled.
    """
    loss = 0
    for (a1, a2), s_block in overlap.items():

        # Get the pair of coefficient blocks
        c1_block = coefficients.block(dict(center_type=a1))
        c2_block = coefficients.block(dict(center_type=a2))

        # Broadcast along samples axis
        samples_mask_1 = [
            c1_block.samples.position(s) for s in s_block.samples.values[:, :2]
        ]

        projection_block_by_pair = mts.TensorBlock(
            values=torch.bmm(
                c1_block.values[samples_mask_1].unsqueeze(1),
                s_block.values,
            ).squeeze(1),
            samples=s_block.samples,
            components=[],
            properties=c2_block.properties,
        )

        projection_block = mts.mean_over_samples_block(
            projection_block_by_pair, "atom_1"
        )

        # Now mask c2_block
        samples_mask_2 = [
            c2_block.samples.position(s) for s in projection_block.samples.values
        ]
        loss += torch.tensordot(
            projection_block.values, c2_block.values[samples_mask_2]
        )

    return loss


def _Sc(
    coefficients: mts.TensorMap,
    overlap: mts.TensorMap,
) -> mts.TensorMap:
    """
    Performs the matrix multiplication:

    .. math::

        w = S . c

    where ``coefficients`` is the vector "c" and ``overlap`` is the matrix "S". Returns
    the projection vector as a :py:class:`TensorMap` with the same metadata as the input
    ``coefficients``.
    """
    blocks = []
    for (a1, a2), s_block in overlap.items():

        # Get the coeff blocks
        c1_block = coefficients.block(dict(center_type=a1))
        c2_block = coefficients.block(dict(center_type=a2))

        # Broadcast along samples axis
        samples_mask = [
            c2_block.samples.position(s) for s in s_block.samples.values[:, ::2]
        ]

        projection_block_by_pair = mts.TensorBlock(
            values=torch.bmm(
                s_block.values,
                c2_block.values[samples_mask].unsqueeze(2),
            ).squeeze(2),
            samples=s_block.samples,
            components=[],
            properties=c1_block.properties,
        )

        projection_block = mts.mean_over_samples_block(
            projection_block_by_pair, "atom_2"
        )

        blocks.append(projection_block)

    # Create a TensorMap from the projection blocks, sum over center_2_type
    projections = mts.TensorMap(keys=overlap.keys, blocks=blocks)
    projections = projections.keys_to_samples("center_2_type")
    projections = mts.sum_over_samples(projections, "center_2_type")
    projections = mts.rename_dimension(
        projections, "keys", "center_1_type", "center_type"
    )

    return projections


def _dot(coefficients_1: mts.TensorMap, coefficients_2: mts.TensorMap) -> torch.Tensor:
    """
    Performs the dot product between two vectors:

    .. math::

        c_1 . c_2

    where ``coefficients_1`` is the vector "c_1" and ``coefficients_2`` is the vector
    "c_2". Both are assumed to have equivalent metadata. Returns a scalar.
    """
    dot_product = 0
    for key in coefficients_1.keys:
        dot_product += torch.tensordot(coefficients_1[key].values, coefficients_2[key].values)
    return dot_product


# ===========================
# ===== Metadata checks =====
# ===========================


def _equal_metadata_raise(
    vector_1: mts.TensorMap,
    vector_2: mts.TensorMap,
    name_1: str,
    name_2: str,
) -> None:
    """
    Raises an ValueError if the metadata of ``vector_1`` and ``vector_2`` are not equal.
    Returns None otherwise.
    """
    try:
        mts.equal_metadata_raise(vector_1, vector_2)
    except ValueError as e:
        raise ValueError(f"Metadata of `{name_1}` and `{name_2}` must match.") from e


def _check_metadata_coefficients(coefficients: mts.TensorMap) -> None:
    """
    Raises an ValueError if the metadata of ``coefficients`` is not valid. Returns None
    otherwise.
    """
    assert coefficients.keys.names == ["center_type"], (
        f"Invalid keys. Must be {['center_type']}," f" found {coefficients.keys.names}."
    )
    assert coefficients.sample_names == ["system", "atom"], (
        f"Invalid sample names. Must be {['system', 'atom']},"
        f" found {coefficients.sample_names}."
    )
    assert coefficients.property_names == ["o3_lambda", "n", "o3_mu"], (
        f"Invalid property names. Must be {['o3_lambda', 'n', 'o3_mu']},"
        f" found {coefficients.property_names}."
    )


def _check_metadata_overlap(overlap: mts.TensorMap) -> None:
    """
    Raises an ValueError if the metadata of ``overlap`` is not valid or compatible with
    the metadata of ``input_c``. Returns None otherwise.
    """
    # Check the Labels names
    assert overlap.keys.names == [
        "center_1_type",
        "center_2_type",
    ], (
        f"Invalid keys. Must be {['center_1_type', 'center_2_type']},"
        f" found {overlap.keys.names}."
    )
    assert overlap.sample_names == ["system", "atom_1", "atom_2"], (
        f"Invalid sample names. Must be {['system', 'atom_1', 'atom_2']},"
        f" found {overlap.sample_names}."
    )
    assert overlap.component_names == ["l1_n1_m1"], (
        f"Invalid component names. Must be {['l1_n1_m1']},"
        f" found {overlap.component_names}."
    )
    assert overlap.property_names == ["l2_n2_m2"], (
        f"Invalid component names. Must be {['l2_n2_m2']},"
        f" found {overlap.component_names}."
    )
