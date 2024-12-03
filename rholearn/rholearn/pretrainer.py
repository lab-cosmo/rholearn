from typing import List

import metatensor.torch as mts
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge


class RhoLinearRegression(torch.nn.Module):

    def __init__(
        self,
        in_keys: mts.Labels,
        in_properties: List[mts.Labels],
        out_properties: List[mts.Labels],
        dtype: torch.dtype = torch.float64,
        device: torch.device = "cpu",
    ) -> None:

        super().__init__()

        self._in_keys = in_keys
        self._in_properties = in_properties
        self._out_properties = out_properties
        self._dtype = dtype
        self._device = device
        self._models = [
            LinearRegression(
                fit_intercept=key["o3_lambda"] == 0,
            )
            for key in self._in_keys
        ]
        self._best_val_losses = None

    def fit(self, X: mts.TensorMap, Y: mts.TensorMap) -> None:
        """
        Fits a Ridge model for each block in the TensorMap and each alpha.
        """
        # Check metadata
        mts.equal_metadata_raise(X, Y, check=["samples", "components"])
        for key_i, key in enumerate(self._in_keys):
            assert X.block(key).properties == self._in_properties[key_i]
            assert Y.block(key).properties == self._out_properties[key_i]

        # Iterate over blocks
        for key, model in zip(self._in_keys, self._models):

            X_values = _block_mts_to_numpy(X.block(key))
            Y_values = _block_mts_to_numpy(Y.block(key))

            # Fit a separate model for each alpha
            model.fit(X_values, Y_values)

        self._fitted = True

    def validate(self, X: mts.TensorMap, Y: mts.TensorMap) -> None:
        """
        For each block, finds the best alpha. Stores these in `self._best_alphas`, and
        the corresponding model in `self._best_models`
        """
        if not self._fitted:
            raise ValueError("models not yet fitted. Call the `fit` method.")

        # Check metadata
        mts.equal_metadata_raise(X, Y, check=["samples", "components"])
        for key_i, key in enumerate(self._in_keys):
            assert X.block(key).properties == self._in_properties[key_i]
            assert Y.block(key).properties == self._out_properties[key_i]

        val_losses = []
        for key, model in zip(self._in_keys, self._models):

            X_values = _block_mts_to_numpy(X.block(key))
            Y_values = _block_mts_to_numpy(Y.block(key))

            Y_pred_values = model.predict(X_values)

            loss = ((Y_pred_values - Y_values) ** 2).sum()
            val_losses.append(loss)

        self._best_val_losses = val_losses
        self._best_models = self._models

    def predict(self, X: mts.TensorMap) -> mts.TensorMap:
        """
        Makes a prediction on the input TensorMap using the best
        """
        if not self._fitted:
            raise ValueError("models not yet fitted. Call the `fit` method.")

        # Check metadata
        for key_i, key in enumerate(self._in_keys):
            assert X.block(key).properties == self._in_properties[key_i]

        # Make predictions on each block
        predicted_blocks = []
        for key, model, out_props in zip(
            self._in_keys, self._models, self._out_properties
        ):
            X_values = _block_mts_to_numpy(X.block(key))
            Y_pred_values = model.predict(X_values)
            Y_pred_block = _block_numpy_to_mts(
                Y_pred_values,
                samples=X.block(key).samples,
                components=X.block(key).components,
                properties=out_props,
                dtype=self._dtype,
                device=self._device,
            )
            predicted_blocks.append(Y_pred_block)

        return mts.TensorMap(self._in_keys, predicted_blocks)


class RhoRidgeCV(torch.nn.Module):
    """
    Sets up a manual RidgeCV model that fits one model per block and per regularization
    strength, then selects and stores the best model for each block and its
    corresponding alpha value based on the validation loss.
    """

    def __init__(
        self,
        in_keys: mts.Labels,
        in_properties: List[mts.Labels],
        out_properties: List[mts.Labels],
        alphas: List[float],
        dtype: torch.dtype = torch.float64,
        device: torch.device = "cpu",
    ) -> None:

        super().__init__()

        self._in_keys = in_keys
        self._in_properties = in_properties
        self._out_properties = out_properties
        self._alphas = alphas
        self._dtype = dtype
        self._device = device
        self._models = [
            [
                Ridge(
                    alpha=alpha,
                    fit_intercept=key["o3_lambda"] == 0,
                )
                for alpha in self._alphas
            ]
            for key in self._in_keys
        ]
        self._best_alphas = None
        self._best_models = None
        self._best_val_losses = None
        self._all_val_losses = None

    def fit(self, X: mts.TensorMap, Y: mts.TensorMap) -> None:
        """
        Fits a Ridge model for each block in the TensorMap and each alpha.
        """
        # Check metadata
        mts.equal_metadata_raise(X, Y, check=["samples", "components"])
        for key_i, key in enumerate(self._in_keys):
            assert X.block(key).properties == self._in_properties[key_i]
            assert Y.block(key).properties == self._out_properties[key_i]

        # Iterate over blocks
        for key, models in zip(self._in_keys, self._models):

            X_values = _block_mts_to_numpy(X.block(key))
            Y_values = _block_mts_to_numpy(Y.block(key))

            # Fit a separate model for each alpha
            for model in models:
                model.fit(X_values, Y_values)

        self._fitted = True

    def validate(self, X: mts.TensorMap, Y: mts.TensorMap) -> None:
        """
        For each block, finds the best alpha. Stores these in `self._best_alphas`, and
        the corresponding model in `self._best_models`
        """
        if not self._fitted:
            raise ValueError("models not yet fitted. Call the `fit` method.")

        # Check metadata
        mts.equal_metadata_raise(X, Y, check=["samples", "components"])
        for key_i, key in enumerate(self._in_keys):
            assert X.block(key).properties == self._in_properties[key_i]
            assert Y.block(key).properties == self._out_properties[key_i]

        best_alphas = []
        best_models = []
        best_losses = []
        val_losses = []
        for key, models in zip(self._in_keys, self._models):

            X_values = _block_mts_to_numpy(X.block(key))
            Y_values = _block_mts_to_numpy(Y.block(key))

            block_losses = []
            for model in models:

                Y_pred_values = model.predict(X_values)

                loss = ((Y_pred_values - Y_values) ** 2).sum()
                block_losses.append(loss)

            best_alpha_idx = np.argmin(block_losses)
            best_alphas.append(self._alphas[best_alpha_idx])
            best_models.append(models[best_alpha_idx])
            best_losses.append(block_losses[best_alpha_idx])
            val_losses.append(block_losses)

        self._best_alphas = best_alphas
        self._best_models = best_models
        self._best_val_losses = best_losses
        self._all_val_losses = val_losses

    def predict(self, X: mts.TensorMap) -> mts.TensorMap:
        """
        Makes a prediction on the input TensorMap using the best
        """
        if not self._fitted:
            raise ValueError("models not yet fitted. Call the `fit` method.")

        if self._best_alphas is None:
            raise ValueError(
                "best alphas not yet determined. Call the `get_best_alpha` method."
            )
        if self._best_models is None:
            raise ValueError(
                "best alphas not yet determined. Call the `get_best_alpha` method."
            )

        # Check metadata
        for key_i, key in enumerate(self._in_keys):
            assert X.block(key).properties == self._in_properties[key_i]

        # Make predictions on each block
        predicted_blocks = []
        for key, model, out_props in zip(
            self._in_keys, self._best_models, self._out_properties
        ):
            X_values = _block_mts_to_numpy(X.block(key))
            Y_pred_values = model.predict(X_values)
            Y_pred_block = _block_numpy_to_mts(
                Y_pred_values,
                samples=X.block(key).samples,
                components=X.block(key).components,
                properties=out_props,
                dtype=self._dtype,
                device=self._device,
            )
            predicted_blocks.append(Y_pred_block)

        return mts.TensorMap(self._in_keys, predicted_blocks)


def _block_mts_to_numpy(block: mts.TensorBlock) -> np.ndarray:

    values = block.values
    s, c, p = values.shape
    return np.array(values.reshape(s * c, p), dtype=np.float64)


def _block_numpy_to_mts(
    block: np.ndarray,
    samples: mts.Labels,
    components: mts.Labels,
    properties: mts.Labels,
    dtype: torch.dtype,
    device: torch.device,
) -> mts.TensorBlock:

    values = torch.tensor(
        block.reshape(len(samples), *[len(c) for c in components], len(properties)),
        dtype=dtype,
        device=device,
    )
    return mts.TensorBlock(
        samples=samples,
        components=components,
        properties=properties,
        values=values,
    )
