"""
Module for fitting basic linear algebra models and predicting with them.
"""
from typing import List 

import matplotlib.pyplot as plt
from sklearn import linear_model
import torch

import metatensor
import metatensor.torch
from metatensor import Labels, TensorBlock, TensorMap


def linalg_fit(
    keys: Labels,
    X: TensorMap,
    Y: TensorMap,
    bias_invariants: bool = True,
    components_in: str = "samples",
    model_type: str = "linear",
    **kwargs,
) -> dict:
    """Fit a ``sklearn.linear_model.LinearRegression model`` on ``X`` and ``Y``."""
    models = []
    for key in keys:
        if bias_invariants:
            if key["o3_lambda"] == 0:
                fit_intercept = True
            else:
                fit_intercept = False
        else:
            fit_intercept = False

        X_block = X.block(key)
        Y_block = Y.block(key)
        X_vals = X_block.values
        Y_vals = Y_block.values

        if len(X_block.values.shape) == 3:
            assert len(Y_block.values.shape) == 3

            s1, c1, p1 = X_vals.shape
            s2, c2, p2 = Y_vals.shape

            if components_in == "samples":
                X_vals = X_vals.reshape((s1 * c1, p1))
                Y_vals = Y_vals.reshape((s2 * c2, p2))

            elif components_in == "properties":

                X_vals = X_vals.reshape((s1, c1 * p1))
                Y_vals = Y_vals.reshape((s2, c2 * p2))

            else:
                raise
        
        else:
            assert len(X_vals.shape) == 2
            assert len(Y_vals.shape) == 2

        if model_type == "linear":
            model = linear_model.LinearRegression(fit_intercept=fit_intercept)
        elif model_type == "ridge":
            model = linear_model.Ridge(
                alpha=kwargs.get("alpha"), fit_intercept=fit_intercept
            )
        elif model_type == "ridgecv":
            model = linear_model.RidgeCV(
                alphas=kwargs.get("alphas"), fit_intercept=fit_intercept, cv=kwargs.get("cv"),
            )
        else:
            raise ValueError("invalid `model_type`")

        model.fit(X_vals, Y_vals)
        models.append(model)

    return models


def linalg_predict(
    keys: Labels,
    models: list,
    out_features: List[Labels],
    X: TensorMap,
    components_in: str = "samples",
    use_torch: bool = True,
) -> TensorMap:
    """
    Predict the output of the linalg model
    """
    predicted_blocks = []
    for key, model, properties in zip(keys, models, out_features):
        X_block = X.block(key)
        X_vals = X_block.values
        if len(X_vals.shape) == 3:
            s, c, p = X_block.values.shape
            # Reshape and predict
            if components_in == "samples":
                X_vals = X_vals.reshape((s * c, p))
            elif components_in == "properties":
                X_vals = X_vals.reshape((s, c * p))
            else:
                raise
        Y_pred_vals = model.predict(X_vals)

        # Reshape and store
        if len(X.block(key).components) != 0:
            Y_pred_vals = Y_pred_vals.reshape((s, c, -1))
        if use_torch:
            samps = metatensor.torch.Labels(
                names=X_block.samples.names, values=torch.tensor(X_block.samples.values)
            )
            comps = [
                metatensor.torch.Labels(names=c.names, values=torch.tensor(c.values))
                for c in X_block.components
            ]
            props = metatensor.torch.Labels(
                names=properties.names, values=torch.tensor(properties.values)
            )
            predicted_blocks.append(
                metatensor.torch.TensorBlock(
                    values=torch.Tensor(Y_pred_vals),
                    samples=samps,
                    components=comps,
                    properties=props,
                )
            )
        else:
            predicted_blocks.append(
                TensorBlock(
                    values=Y_pred_vals,
                    samples=X_block.samples,
                    components=X_block.components,
                    properties=properties,
                )
            )

    if use_torch:
        return metatensor.torch.TensorMap(
            metatensor.torch.Labels(names=keys.names, values=torch.tensor(keys.values)),
            predicted_blocks,
        )
    return TensorMap(keys, predicted_blocks)


def plot_parities(keys, target: TensorMap, prediction: TensorMap):

    # Figure setup
    fig, axes = plt.subplots(2, len(keys) // 2, figsize=(15, 8))

    # m component map
    m_to_index = lambda l: {i: -l + i for i in range(2 * l + 1)}
    index_to_m = lambda l: {v: k for k, v in m_to_index(l).items()}

    # Unique colors for radial channels and markers for m components
    colors = ["black", "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    markers = {-3: 'o', -2: 'v', -1: '^', 0: '<', 1: '>', 2: 's', 3: 'p',}

    # Legends
    for n, color in enumerate(colors):
        axes[0][0].scatter([], [], color=color, marker="o", label=f"n={n}")
    for m, marker in markers.items():
        axes[0][0].scatter([], [], color="k", marker=marker, label=f"m={m}")

    # Parity plots
    for key in keys:
        l, a = key["o3_lambda"], key["center_type"]
        ax = axes[{1: 0, 14: 1}[a], l]

        metatensor.equal_metadata_block_raise(target.block(key), prediction.block(key))
        Y_vals = target.block(key).values
        Y_pred_vals = prediction.block(key).values

        for n in target.block(key).properties.column("n"):
            for m in target.block(key).components[0].column("o3_mu"):
                m_idx = index_to_m(l)[m]
                ax.scatter(
                    Y_vals[:, :, n][:, m_idx], 
                    Y_pred_vals[:, :, n][:, m_idx], 
                    color=colors[n], 
                    marker=markers[m]
                )

        ax.axline([0, 0], [0.001, 0.001], color="gray", linestyle="--")
        ax.set_title(f"l = {key[0]}, a = {key[1]}")
        ax.set_xlabel("DFT")
        ax.set_ylabel("ML")
        # ax.set_aspect("equal")

    # Fig formatting
    # fig.suptitle(f"Field: {ri_restart_idx}, Structure = {A}, Loss: {train_loss:.3f}, Rascal Cutoff = {rascal_cutoff} Ang")
    fig.tight_layout()
    fig.legend()
    # plt.savefig(f"{ri_restart_idx}_A={A}_cut={rascal_cutoff}.png")

    return fig
