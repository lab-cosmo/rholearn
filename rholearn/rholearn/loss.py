"""
Module for evaluating the loss between input and target scalar fields expanded on a
spherical basis.
"""

from typing import Optional

import metatensor.torch as mts
import torch


class RhoLoss(torch.nn.Module):
    """
    Implements the general L2 loss function:

    .. math::

        L = ( c^{inp} - c^{tar} ) \\hat{S}^{tar} ( c^{inp} - c^{tar} )

    and derived forms for evaluating the loss between an input and target scalar field,
    both expanded on the same spherical basis set.

    :param orthogonal: bool, whether the basis set is orthogonal. Default false. If true
        and ``normalized=False``, the overlap passed to :py:meth:`forward` must be
        diagonal. If true and ``normalized=True``, the overlap need not be passed.
    :param normalized: optional bool, whether the basis set is normalized. This
        parameter is only relevant if ``orthogonal=True``. If ``orthogonal=False``, the
        basis is assumed to be non-normalized and the overlap matrix must be passed to
        :py:meth:`forward`. Default false when not set and applicable.
    :param truncated: optional bool, whether to compute a truncated form of the loss
        whose gradient with respect to model parameters is equivalent to the
        non-truncated form. Not applicable when ``orthogonal=True`` and
        ``normalized=True``, or when only ``target_c`` (and not ``target_w``) are passed
        to :py:meth:`forward`. Default false when not set and applicable.
    """

    def __init__(
        self,
        orthogonal: bool = False,
        normalized: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:

        super().__init__()

        # Check parameters
        if orthogonal:
            if normalized is None:
                normalized = False
            assert (
                truncated is None
            ), "``truncated`` only applies to nonorthogonal basis sets."

        else:
            if truncated is None:
                truncated = False
            assert normalized is None, (
                "``normalized`` must be None, as non-normalized and normalized"
                " basis sets are treated equivalently by this class for"
                " non-orthogonal basis sets"
            )

        # Set attributes
        self._orthogonal = orthogonal
        self._normalized = normalized
        self._truncated = truncated

    def _error_msg(self, variable_name: str, should_be_passed: bool) -> str:
        return (
            f"`{variable_name}` must {'' if should_be_passed else 'not'} be passed"
            f" for the combination of `orthogonal={self._orthogonal}`"
            f" and `normalized={self._normalized}`"
            f" and `truncated={self._truncated}`."
        )

    def forward(
        self,
        input_c: mts.TensorMap,
        target_c: Optional[mts.TensorMap] = None,
        target_w: Optional[mts.TensorMap] = None,
        overlap: Optional[mts.TensorMap] = None,
        check_metadata: bool = True,
    ) -> torch.Tensor:
        """
        Computes the loss between the input and target scalar fields, expanded on a
        spherical basis.
        """
        if self._orthogonal:
            return self._forward_orthogonal(
                input_c=input_c,
                target_c=target_c,
                target_w=target_w,
                overlap=overlap,
                check_metadata=check_metadata,
            )

        return self._forward_nonorthogonal(
            input_c=input_c,
            target_c=target_c,
            target_w=target_w,
            overlap=overlap,
            check_metadata=check_metadata,
        )

    def _forward_orthogonal(
        self,
        input_c: mts.TensorMap,
        target_c: Optional[mts.TensorMap],
        target_w: Optional[mts.TensorMap],
        overlap: Optional[mts.TensorMap],
        check_metadata: bool = True,
    ) -> torch.Tensor:
        """
        Evaluates the loss for an orthogonal basis, either normalized or not.

        Assumes that ``overlap`` is a vector corresponding to the diagonal of the
        overlap matrix, in :py:class:`TensorMap` format, with the same metadata as the
        input and target TensorMaps.
        """
        # Normalized basis
        if self._normalized is True:
            # Overlap need not be passed as it is the identity matrix in this case
            assert overlap is None, self._error_msg("overlap", False)

            # For an orthonormal basis, `c` and `w` are equivalent. Either can be
            # passed.
            if target_w is not None:
                assert target_c is None, self._error_msg("target_c", False)
                target_c = target_w
            assert target_c is not None, self._error_msg("overlap", True)

            return _orthonormal_basis(
                input_c=input_c,
                target_c=target_c,
                check_metadata=check_metadata,
            )

        # Non-normalized basis. Overlap is diagonal.
        assert overlap is not None, self._error_msg("overlap", True)

        if target_w is None:
            # Evaluate via `c` only. The attribute ``truncated`` is irrelevant in this
            # case.
            assert target_c is not None, self._error_msg("target_c", True)
            return _orthogonal_basis_via_c(
                overlap=overlap,
                input_c=input_c,
                target_c=target_c,
                check_metadata=check_metadata,
            )

        # if target_c is None:
        #     target_c = [None] * len(input_c)

        return _orthogonal_basis_via_w(
            overlap=overlap,
            input_c=input_c,
            target_c=target_c,
            target_w=target_w,
            truncated=self._truncated,
            check_metadata=check_metadata,
        )

    def _forward_nonorthogonal(
        self,
        input_c: mts.TensorMap,
        target_c: Optional[mts.TensorMap],
        target_w: Optional[mts.TensorMap],
        overlap: Optional[mts.TensorMap],
        check_metadata: bool = True,
    ) -> torch.Tensor:
        """
        Evaluates the loss for a nonorthogonal basis.

        Assumes that ``overlap`` is the overlap matrix in :py:class:`TensorMap` format.
        """
        assert overlap is not None, self._error_msg("overlap", True)

        if target_w is None:
            # Evaluate via `c` only. The attribute ``truncated`` is irrelevant in this
            # case.
            assert target_c is not None, self._error_msg("target_c", True)
            return _nonorthogonal_via_c(
                overlap=overlap,
                input_c=input_c,
                target_c=target_c,
                check_metadata=check_metadata,
            )

        if not self._truncated:
            assert target_c is not None, (
                "if evaluating the non-truncated loss with the overlap matrix,"
                " ``target_c`` must be passed."
            )

        return _nonorthogonal_via_w(
            overlap=overlap,
            input_c=input_c,
            target_c=target_c,
            target_w=target_w,
            truncated=self._truncated,
            check_metadata=check_metadata,
        )


# =============================================
# ===== Loss evaluation: orthogonal basis =====
# =============================================


def _orthonormal_basis(
    input_c: mts.TensorMap, target_c: mts.TensorMap, check_metadata: bool = True
) -> torch.Tensor:
    """
    Evaluates the L2 loss between `input` and `target` coefficients from an orthonormal
    basis set The overlap matrix is implicitly treated as the identity.

    .. math::

        L  = ( c^{inp} - c^{tar} ) ^ 2

    Note: this is the formal definition of the loss for an orthogonal basis.

    As for an orthonormal basis the coefficients and projections are equivalent, either
    can be passed for target and input.

    The passed TensorMaps can correspond to multiple structures, but must have equal
    metadata.
    """
    # Check metadata
    if check_metadata:
        _equal_metadata_raise(input_c, target_c, "input_c", "target_c")

    # Evaluate loss
    loss = 0
    for key in input_c.keys:
        delta_c_block = input_c.block(key).values - target_c.block(key).values
        block_loss = delta_c_block * delta_c_block
        loss += block_loss.sum()

    return loss


def _orthogonal_basis_via_c(
    overlap: mts.TensorMap,
    input_c: mts.TensorMap,
    target_c: mts.TensorMap,
    check_metadata: bool = True,
) -> torch.Tensor:
    """
    Evaluates the L2 loss between `input` and `target` coefficients fromm an orthogonal
    basis set, i.e. as if the overlap matrix was diagonal but not necessarily the
    identity. Evaluation occurs via the target coefficients, and not the projections.

    .. math::

        L = \\sum_i
            ( c^{inp}_i - c^{tar}_i ) \\hat{S}^{tar}_{ii} ( c^{inp}_i - c^{tar}_i)

    Note: this is the formal definition of the loss for an orthogonal basis.

    The TensorMaps passed can correspond to multiple structures.
    """
    # Check metadata
    if check_metadata:
        _equal_metadata_raise(input_c, overlap, "input_c", "overlap")
        _equal_metadata_raise(input_c, target_c, "input_c", "target_c")

    # Evaluate loss
    loss = 0
    for key in input_c.keys:
        delta_c_block = input_c.block(key).values - target_c.block(key).values
        s_block = overlap.block(key).values
        block_loss = delta_c_block * s_block * delta_c_block
        loss += block_loss.sum()

    return loss


def _orthogonal_basis_via_w(
    overlap: mts.TensorMap,
    input_c: mts.TensorMap,
    target_c: Optional[mts.TensorMap] = None,
    target_w: Optional[mts.TensorMap] = None,
    truncated: bool = False,
    check_metadata: bool = True,
) -> torch.Tensor:
    """
    Evaluates the L2 loss between `input` and `target` coefficients as if they are from
    an orthogonal basis set, i.e. as if the overlap matrix was diagonal, but not the
    identity (or a scalar multiple of it).

    .. math::

        L = \\sum_i
            c^{inp}_i \\hat{S}_{ii} c^{inp}_i - 2 * c^{inp}_i w^{tar}_i

    Note: this is the formal definition of the loss for an orthogonal basis if
    ``truncated`` is set to false. If true, this expression does not match the formal
    definition of the loss, as only terms with nonzero-derivative with respect to model
    parameters are computed.

    The TensorMaps passed can correspond to multiple structures.
    """
    # Check metadata
    if check_metadata:
        _equal_metadata_raise(input_c, overlap, "input_c", "overlap")
        if target_c is not None:
            _equal_metadata_raise(input_c, target_c, "input_c", "target_c")
        if target_w is not None:
            _equal_metadata_raise(input_c, target_w, "input_c", "target_w")

    # Evaluate loss
    loss = 0
    for key in input_c.keys:
        in_c_block = input_c.block(key).values
        tar_w_block = target_w.block(key).values
        s_block = overlap.block(key).values

        # c^{in} S c^{in} - 2 c^{in} w^{tar}
        block_loss = (in_c_block * s_block * in_c_block) - (
            2 * in_c_block * tar_w_block
        )
        if not truncated:  # + c^{tar} w^{tar}
            tar_c_block = target_c.block(key).values
            block_loss += tar_c_block * tar_w_block

        loss += block_loss.sum()

    return loss


# =================================================
# ===== Loss evaluation: non-orthogonal basis =====
# =================================================


def _nonorthogonal_via_c(
    overlap: mts.TensorMap,
    input_c: mts.TensorMap,
    target_c: mts.TensorMap,
    check_metadata: bool = True,
) -> torch.Tensor:
    """
    Evaluates the L2 loss between `input` and `target` fields expanded on a
    nonorthogonal basis. Evaluation occurs via the target coefficients, and not the
    projections.

    The expression evaluated in this function is:

    .. math::

        L = ( c^{inp} - c^{tar} ) \\hat{S} ( c^{inp} - c^{tar} )

    which corresponds to the formal definition of the L2 loss on the scalar field when
    overlaps between all basis functions are included in ``overlap``.
    """
    # Check metadata
    if check_metadata:
        _check_metadata_coefficients(input_c)
        _equal_metadata_raise(input_c, target_c, "input_c", "target_c")
        _check_metadata_overlap(overlap=overlap)

    # Evaluate loss
    return _cSc(coefficients=mts.subtract(input_c, target_c), overlap=overlap)


def _nonorthogonal_via_w(
    overlap: mts.TensorMap,
    input_c: mts.TensorMap,
    target_c: Optional[mts.TensorMap] = None,
    target_w: Optional[mts.TensorMap] = None,
    truncated: bool = False,
    check_metadata: bool = True,
) -> torch.Tensor:
    """
    Evaluates the L2 loss between `input` and `target` fields expanded on a
    nonorthogonal basis. Evaluation occurs via the target projections where possible.

    For ``truncated=True``, the following expression is evaluated:

    .. math::

        L = c^{inp} \\hat{S} c^{inp} - 2 c^{inp} w^{tar}

    or for ``truncated=False``:

    .. math::

        L = c^{inp} \\hat{S} c^{inp} - 2 c^{inp} w^{tar} + c^{tar} w^{tar}

    This expression is the formal definition of the loss only when the full overlap is
    passed and ``truncated=False``.
    """
    # Check metadata
    if check_metadata:
        _check_metadata_coefficients(input_c)
        _check_metadata_overlap(overlap=overlap)

        if target_c is not None:
            _equal_metadata_raise(input_c, target_c, "input_c", "target_c")

        if target_w is not None:
            _equal_metadata_raise(input_c, target_w, "input_c", "target_w")

    # Evaluate loss
    loss = _cSc(coefficients=input_c, overlap=overlap)
    loss -= 2 * _dot(input_c, target_w)
    if not truncated:
        loss += _dot(target_c, target_w)

    return loss


# ===========================
# ===== Base operations =====
# ===========================


def _cSc(
    coefficients: mts.TensorMap,
    overlap: mts.TensorMap,
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


def _dot(vector_1: mts.TensorMap, vector_2: mts.TensorMap) -> torch.Tensor:
    """
    Performs the dot product between two vectors:

    .. math::

        v_1 . v_2

    where ``vector_1`` is the vector "v_1" and ``vector_2`` is the vector "v_2". The
    Both are assumed to have equivalent metadata. Returns a scalar.
    """
    return torch.sum(
        torch.stack(
            [
                (vector_1[key].values * vector_2[key].values).sum()
                for key in vector_1.keys
            ]
        )
    )


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
