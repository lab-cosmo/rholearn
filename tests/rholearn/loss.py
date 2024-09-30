"""
Tests :py:mod:`rholearn.loss`.
"""

from os.path import join
from typing import List, Union

import pytest
import torch
import metatensor.torch as mts
from metatensor.torch.learn.nn import Linear

from rholearn import mask
from rholearn.loss import RhoLoss
from rholearn.aims_interface import io
from rholearn.utils import convert, system
from rholearn.utils.io import unpickle_dict

FRAME_IDXS = [0, 1, 2]
XYZ_1 = join(
    "tests",
    "rholearn",
    "generate_example_data",
    "qm7",
    "data",
    "qm7.xyz",
)
DATA_DIR_1 = lambda A: join(
    "tests",
    "rholearn",
    "generate_example_data",
    "qm7",
    "data",
    "processed",
    f"{A}",
    "edensity",
)
DATA_DIR_2 = lambda A, processed: join(
    "tests",
    "rholearn",
    "generate_example_data",
    "isolated",
    "data",
    "processed" if processed else "raw",
    f"{A}",
    "edensity",
)

# ===== Unit tests


@pytest.mark.parametrize("A", FRAME_IDXS)
def test_full_loss_via_c_via_w_equiv(A: int):
    """
    Tests that:

        ( c^{inp} - c^{tar} ) \\hat{S}^{tar} ( c^{inp} - c^{tar} )

    is equivalent to

        c^{inp} \\hat{S}^{tar} c^{inp} - 2 c^{inp} w^{tar} + c^{tar} w^{tar}

    when using the full overlap matrix
    """
    # Load dummy predictions.
    ml_coeffs = convert.coeff_vector_to_sparse_by_center_type(
        mts.load(join(DATA_DIR_1(A), "ml_coeffs.npz")), "torch"
    )

    # Load reference data
    ri_coeffs = convert.coeff_vector_to_sparse_by_center_type(
        mts.load(join(DATA_DIR_1(A), "ri_coeffs.npz")), "torch"
    )
    ri_projs = convert.coeff_vector_to_sparse_by_center_type(
        mts.load(join(DATA_DIR_1(A), "ri_projs.npz")), "torch"
    )
    ovlps_full = mts.load(join(DATA_DIR_1(A), "ri_ovlp.npz"))

    # Init loss fn
    loss_fn = RhoLoss(
        orthogonal=False,
        truncated=False,
    )

    # Calculate losses via c and w
    via_c = loss_fn(
        input_c=ml_coeffs,
        target_c=ri_coeffs,
        target_w=None,
        overlap=ovlps_full,
    )
    via_w = loss_fn(
        input_c=ml_coeffs,
        target_c=ri_coeffs,
        target_w=ri_projs,
        overlap=ovlps_full,
    )
    assert torch.allclose(via_c, via_w)


def test_full_loss_and_grid_mse_equivalent():
    """
    Tests that:

        ( c^{inp} - c^{tar} ) \\hat{S}^{tar} ( c^{inp} - c^{tar} )

    is equivalent to the MSE error on the grid:

        \\int_{R} dr | \\rho^{inp}(r) - \\rho^{tar}(r) |^2

    By joining all frames into a single tensor.
    """
    # Load dummy predictions.
    ml_coeffs = mts.join(
        [
            convert.coeff_vector_to_sparse_by_center_type(
                mts.load(join(DATA_DIR_1(A), "ml_coeffs.npz")), "torch"
            )
            for A in FRAME_IDXS
        ],
        axis="samples",
        remove_tensor_name=True,
        different_keys="union",
    )

    # Load reference data
    ri_coeffs = mts.join(
        [
            convert.coeff_vector_to_sparse_by_center_type(
                mts.load(join(DATA_DIR_1(A), "ri_coeffs.npz")), "torch"
            )
            for A in FRAME_IDXS
        ],
        axis="samples",
        remove_tensor_name=True,
        different_keys="union",
    )
    ri_projs = mts.join(
        [
            convert.coeff_vector_to_sparse_by_center_type(
                mts.load(join(DATA_DIR_1(A), "ri_projs.npz")), "torch"
            )
            for A in FRAME_IDXS
        ],
        axis="samples",
        remove_tensor_name=True,
        different_keys="union",
    )
    ovlps_full = mts.join(
        [mts.load(join(DATA_DIR_1(A), "ri_ovlp.npz")) for A in FRAME_IDXS],
        axis="samples",
        remove_tensor_name=True,
        different_keys="union",
    )

    # Init loss fn
    loss_fn = RhoLoss(
        orthogonal=False,
        truncated=False,
    )

    # Calculate losses via c
    loss_via_c = loss_fn(
        input_c=ml_coeffs,
        target_c=ri_coeffs,
        overlap=ovlps_full,
    )
    loss_via_w = loss_fn(
        input_c=ml_coeffs,
        target_c=ri_coeffs,
        target_w=ri_projs,
        overlap=ovlps_full,
    )

    # Load the pre-calculated MSE error on the grid
    squared_error = torch.sum(
        torch.stack(
            [torch.load(join(DATA_DIR_1(A), "squared_error.pt")) for A in FRAME_IDXS]
        )
    )
    assert torch.allclose(loss_via_c, squared_error)
    assert torch.allclose(loss_via_w, squared_error)


@pytest.mark.parametrize("A", FRAME_IDXS)
def test_weight_update_truncated_loss(A: int):
    """
    Constructs a simple linear model with random initial weights, and checks that
    perfoming a forward and backward pass using the truncated and non-truncated loss
    function gives equivalent updated weights.
    """
    # Load data
    ri_coeffs = mts.load(join(DATA_DIR_1(A), "ri_coeffs.npz"))

    keys = ri_coeffs.keys
    out_props = [ri_coeffs.block(key).properties for key in ri_coeffs.keys]
    linear_1 = Linear(
        in_keys=keys,
        in_features=10,
        out_properties=out_props,
        dtype=torch.float64,
    )
    linear_2 = Linear(
        in_keys=keys,
        in_features=10,
        out_properties=out_props,
        dtype=torch.float64,
    )
    # Copy parameters from linear_1 to linear_2
    for i, param in enumerate(linear_2.parameters()):
        param.data = list(linear_1.parameters())[i].data

    # Check equal parameters
    for i, _ in enumerate(linear_2.parameters()):
        assert torch.equal(
            list(linear_1.parameters())[i].data,
            list(linear_2.parameters())[i].data,
        )

    # Generate a random descriptor
    random_descriptor = mts.TensorMap(
        keys=keys,
        blocks=[
            mts.TensorBlock(
                values=torch.randn(
                    (*ri_coeffs.block(key).values.shape[:-1], 10), dtype=torch.float64
                ),
                samples=ri_coeffs.block(key).samples,
                components=ri_coeffs.block(key).components,
                properties=mts.Labels(
                    names=[f"_"],
                    values=torch.arange(10).reshape(-1, 1),
                ),
            )
            for key in keys
        ],
    )

    # Load reference data
    ri_coeffs = convert.coeff_vector_to_sparse_by_center_type(ri_coeffs, "torch")
    ri_projs = convert.coeff_vector_to_sparse_by_center_type(
        mts.load(join(DATA_DIR_1(A), "ri_projs.npz")), "torch"
    )
    ovlp = mts.load(join(DATA_DIR_1(A), "ri_ovlp.npz"))

    # Initialize the loss functions
    loss_fn_1 = RhoLoss(
        orthogonal=False,
        truncated=False,
    )
    loss_fn_2 = RhoLoss(
        orthogonal=False,
        truncated=True,
    )

    # Perform a single training step for each model/loss_fn pair
    for model, loss_fn in zip([linear_1, linear_2], [loss_fn_1, loss_fn_2]):

        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)  # init optimizer
        optim.zero_grad()  # zero the gradients
        ml_coeffs = model(random_descriptor)  # forward pass
        ml_coeffs = convert.coeff_vector_to_sparse_by_center_type(ml_coeffs, "torch")
        loss_val = loss_fn(
            input_c=ml_coeffs,
            target_c=ri_coeffs,
            target_w=ri_projs,
            overlap=ovlp,
        )
        loss_val.backward()
        optim.step()

    # Check equal parameters
    for i, _ in enumerate(linear_2.parameters()):
        assert torch.allclose(
            list(linear_1.parameters())[i].data,
            list(linear_2.parameters())[i].data,
        )


@pytest.mark.parametrize("A", FRAME_IDXS)
def test_batchable_loss_list_and_joined_equivalent(A: int):
    """
    Tests equivalence is loss evaluation between: a) a list of TensorMaps corresponding
    to different frames, and b) a single TensorMap with all frames joined together.

    This tests all batchable loss functions, i.e. ones where overlap matrices can be
    joined in a minibatch of frames: coefficient, diagonal, and onsite.
    """
    # Load data
    ml_coeffs = [
        convert.coeff_vector_to_sparse_by_center_type(
            mts.load(join(DATA_DIR_1(A), "ml_coeffs.npz")), "torch"
        )
        for A in FRAME_IDXS
    ]
    ri_coeffs = [
        convert.coeff_vector_to_sparse_by_center_type(
            mts.load(join(DATA_DIR_1(A), "ri_coeffs.npz")), "torch"
        )
        for A in FRAME_IDXS
    ]
    ri_projs = [
        convert.coeff_vector_to_sparse_by_center_type(
            mts.load(join(DATA_DIR_1(A), "ri_projs.npz")), "torch"
        )
        for A in FRAME_IDXS
    ]
    overlaps_full = [mts.load(join(DATA_DIR_1(A), "ri_ovlp.npz")) for A in FRAME_IDXS]
    overlaps_cutoff = [
        mask.cutoff_overlap_matrix(
            frames=system.read_frames_from_xyz(XYZ_1),
            frame_idxs=[A],
            overlap_matrix=overlaps_full[A],
            cutoff=3,
            drop_empty_blocks=True,
            backend="torch",
        )
        for A in FRAME_IDXS
    ]

    # Init loss fn
    loss_fn_coeff = RhoLoss(
        orthogonal=True,
        normalized=True,
    )
    loss_fn_full = RhoLoss(
        orthogonal=False,
        truncated=False,
    )
    loss_fn_cutoff = RhoLoss(
        orthogonal=False,
        truncated=False,
    )

    # 1) Evaluate by iteratively calling loss fn
    loss_by_iteration = {
        "coeff": 0,
        "cutoff_via_c": 0,
        "cutoff_via_w": 0,
        "full_via_c": 0,
        "full_via_w": 0,
    }
    for c_ri, c_ml, w_ri, ovlp_cutoff, ovlp_full in zip(
        ri_coeffs, ml_coeffs, ri_projs, overlaps_cutoff, overlaps_full
    ):
        loss_by_iteration["coeff"] += loss_fn_coeff(input_c=c_ml, target_c=c_ri)
        loss_by_iteration["cutoff_via_c"] += loss_fn_cutoff(
            input_c=c_ml,
            target_c=c_ri,
            overlap=ovlp_cutoff,
        )
        loss_by_iteration["cutoff_via_w"] += loss_fn_cutoff(
            input_c=c_ml,
            target_c=c_ri,
            target_w=w_ri,
            overlap=ovlp_cutoff,
        )
        loss_by_iteration["full_via_c"] += loss_fn_full(
            input_c=c_ml,
            target_c=c_ri,
            overlap=ovlp_full,
        )
        loss_by_iteration["full_via_w"] += loss_fn_full(
            input_c=c_ml,
            target_c=c_ri,
            target_w=w_ri,
            overlap=ovlp_full,
        )

    # 2) Evaluate by joined TensorMaps
    c_ml = _join_tensors(ml_coeffs, "mts")
    c_ri = _join_tensors(ri_coeffs, "mts")
    w_ri = _join_tensors(ri_projs, "mts")
    ovlp_cutoff = _join_tensors(overlaps_cutoff, "mts")
    ovlp_full = _join_tensors(overlaps_full, "mts")

    loss_by_joined = {}
    loss_by_joined["coeff"] = loss_fn_coeff(input_c=c_ml, target_c=c_ri)
    loss_by_joined["cutoff_via_c"] = loss_fn_cutoff(
        input_c=c_ml,
        target_c=c_ri,
        overlap=ovlp_cutoff,
    )
    loss_by_joined["cutoff_via_w"] = loss_fn_cutoff(
        input_c=c_ml,
        target_c=c_ri,
        target_w=w_ri,
        overlap=ovlp_cutoff,
    )
    loss_by_joined["full_via_c"] = loss_fn_full(
        input_c=c_ml,
        target_c=c_ri,
        overlap=ovlp_full,
    )
    loss_by_joined["full_via_w"] = loss_fn_full(
        input_c=c_ml,
        target_c=c_ri,
        target_w=w_ri,
        overlap=ovlp_full,
    )

    for loss_type in loss_by_iteration.keys():
        assert torch.allclose(loss_by_iteration[loss_type], loss_by_joined[loss_type])


@pytest.mark.parametrize("A", [0])
def test_loss_onsite_equals_full_isolated_atoms(A: int):
    """
    Tests that for a dimer of isolated atoms (i.e. 100 Ang separation), that the onsite
    loss is equivalent to the full loss, both via projections and coefficients.
    """
    # Load reference data
    ri_coeffs = convert.coeff_vector_to_sparse_by_center_type(
        mts.load(join(DATA_DIR_2(A, True), "ri_coeffs.npz")), "torch"
    )
    ri_projs = convert.coeff_vector_to_sparse_by_center_type(
        mts.load(join(DATA_DIR_2(A, True), "ri_projs.npz")), "torch"
    )
    overlap_full = mts.load(join(DATA_DIR_2(A, True), "ri_ovlp.npz"))

    # Mask the overlap based on a cutoff that reduces to just onsite overlap terms
    frame = io.read_geometry(DATA_DIR_2(A, False))
    overlap_onsite = mask.cutoff_overlap_matrix(
        [frame], [A], overlap_full, cutoff=0.01, drop_empty_blocks=True, backend="torch"
    )

    # Generate some random predicted coefficients by adding noise to the reference
    ml_coeffs = mts.add(
        ri_coeffs, mts.multiply(mts.random_uniform_like(ri_coeffs), 0.1)
    )

    loss_fn_full = RhoLoss(orthogonal=False, truncated=False)
    loss_fn_onsite = RhoLoss(orthogonal=False, truncated=False)

    # Calculate losses
    l_full_via_c = loss_fn_full(
        input_c=ml_coeffs,
        target_c=ri_coeffs,
        overlap=overlap_full,
    )
    l_full_via_w = loss_fn_full(
        input_c=ml_coeffs,
        target_c=ri_coeffs,
        target_w=ri_projs,
        overlap=overlap_full,
    )
    l_on_via_c = loss_fn_onsite(
        input_c=ml_coeffs,
        target_c=ri_coeffs,
        overlap=overlap_onsite,
    )
    l_on_via_w = loss_fn_onsite(
        input_c=ml_coeffs,
        target_c=ri_coeffs,
        target_w=ri_projs,
        overlap=overlap_onsite,
    )

    # All should be equal
    assert torch.allclose(l_full_via_c, l_on_via_c)
    assert torch.allclose(l_full_via_w, l_on_via_w)
    assert torch.allclose(l_full_via_c, l_full_via_w)
    assert torch.allclose(l_on_via_c, l_on_via_w)


def _join_tensors(tensors: List[mts.TensorMap], method: str = "mts") -> mts.TensorMap:
    """Joins a list of tensors into a single tensor."""
    if method == "mts":
        return mts.join(tensors, axis="samples", remove_tensor_name=True)

    blocks = []
    for key in tensors[0].keys:
        blocks.append(
            mts.TensorBlock(
                values=torch.vstack([t.block(key).values for t in tensors]),
                samples=mts.Labels(
                    names=tensors[0].sample_names,
                    values=torch.vstack([t.block(key).samples.values for t in tensors]),
                ),
                components=tensors[0].block(key).components,
                properties=tensors[0].block(key).properties,
            )
        )

    return mts.TensorMap(keys=tensors[0].keys, blocks=blocks)
