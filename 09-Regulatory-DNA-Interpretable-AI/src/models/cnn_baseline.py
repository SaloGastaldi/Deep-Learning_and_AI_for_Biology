"""Compact 1D CNN baseline for promoter sequence classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CNNBaselineConfig:
    """Configuration of the compact promoter CNN."""

    input_channels: int = 4
    conv_channels: tuple[int, int, int] = (32, 64, 128)
    kernel_sizes: tuple[int, int, int] = (15, 11, 7)
    pool_size: int = 4
    dense_units: int = 64
    dropout: float = 0.40

    def validate(self) -> None:
        """Validate architecture configuration."""

        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")

        if len(self.conv_channels) != 3:
            raise ValueError("Exactly three convolutional channel values are required.")

        if len(self.kernel_sizes) != 3:
            raise ValueError("Exactly three convolutional kernel sizes are required.")

        if any(channel <= 0 for channel in self.conv_channels):
            raise ValueError("All convolutional channel values must be positive.")

        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError(
                "Kernel sizes must be positive odd integers."
            )

        if self.pool_size <= 1:
            raise ValueError("pool_size must be greater than 1.")

        if self.dense_units <= 0:
            raise ValueError("dense_units must be positive.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

    def to_dict(self) -> dict:
        """Return a serializable representation."""

        return asdict(self)


class ConvBlock(nn.Module):
    """Convolution, normalization, activation, pooling and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=pool_size,
                stride=pool_size,
            ),
            nn.Dropout1d(p=dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block."""

        return self.block(inputs)


class PromoterCNNBaseline(nn.Module):
    """Compact sequence-only CNN for binary promoter classification.

    Input shape
    -----------
    batch_size, 4, sequence_length

    Output shape
    ------------
    batch_size

    The model returns raw logits. Sigmoid must only be applied during
    inference or metric calculation, not inside the model.
    """

    def __init__(
        self,
        config: CNNBaselineConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or CNNBaselineConfig()
        self.config.validate()

        channels_1, channels_2, channels_3 = self.config.conv_channels
        kernel_1, kernel_2, kernel_3 = self.config.kernel_sizes

        self.features = nn.Sequential(
            ConvBlock(
                in_channels=self.config.input_channels,
                out_channels=channels_1,
                kernel_size=kernel_1,
                pool_size=self.config.pool_size,
                dropout=self.config.dropout,
            ),
            ConvBlock(
                in_channels=channels_1,
                out_channels=channels_2,
                kernel_size=kernel_2,
                pool_size=self.config.pool_size,
                dropout=self.config.dropout,
            ),
            ConvBlock(
                in_channels=channels_2,
                out_channels=channels_3,
                kernel_size=kernel_3,
                pool_size=self.config.pool_size,
                dropout=self.config.dropout,
            ),
        )

        self.global_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=channels_3,
                out_features=self.config.dense_units,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(
                in_features=self.config.dense_units,
                out_features=1,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize convolutional and dense layers."""

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one raw classification logit per sequence."""

        if inputs.ndim != 3:
            raise ValueError(
                "Expected input with shape (batch, channels, length), "
                f"observed {tuple(inputs.shape)}"
            )

        if inputs.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} input channels, "
                f"observed {inputs.shape[1]}"
            )

        features = self.features(inputs)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)

        return logits.squeeze(dim=1)

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return positive-class probabilities."""

        return torch.sigmoid(self.forward(inputs))


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters optimized during training."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
