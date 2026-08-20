#!/usr/bin/env python3
"""Validate the CNN baseline before model training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.promoter_dataset import PromoterDataset
from src.models.cnn_baseline import (
    CNNBaselineConfig,
    PromoterCNNBaseline,
    count_trainable_parameters,
)


TRAIN_PATH = Path("data/splits/train.csv")
SUMMARY_PATH = Path("data/metadata/model_validation_summary.json")

BATCH_SIZE = 16
EXPECTED_SEQUENCE_LENGTH = 2200
RANDOM_SEED = 123


def set_seed(seed: int) -> None:
    """Set deterministic random seeds used in this validation."""

    torch.manual_seed(seed)


def validate_forward_pass(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Validate model output shape and numerical values."""

    logits = model(features)

    if logits.shape != labels.shape:
        raise ValueError(
            f"Logit shape {tuple(logits.shape)} does not match "
            f"label shape {tuple(labels.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise ValueError("The forward pass produced non-finite logits.")

    probabilities = torch.sigmoid(logits)

    if not torch.all(
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ):
        raise ValueError("Predicted probabilities are outside [0, 1].")

    return logits


def validate_backward_pass(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: float,
) -> float:
    """Validate loss calculation and gradient propagation."""

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
        )
    )

    loss = criterion(logits, labels)

    if not torch.isfinite(loss):
        raise ValueError("Loss is not finite.")

    model.zero_grad(set_to_none=True)
    loss.backward()

    parameters_with_gradients = 0
    nonfinite_gradients = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameters_with_gradients += 1

        if not torch.isfinite(parameter.grad).all():
            nonfinite_gradients += 1

    if parameters_with_gradients == 0:
        raise ValueError("No model parameters received gradients.")

    if nonfinite_gradients > 0:
        raise ValueError(
            f"{nonfinite_gradients} parameters contain non-finite gradients."
        )

    return float(loss.detach().cpu())


def inspect_intermediate_shapes(
    model: PromoterCNNBaseline,
    features: torch.Tensor,
) -> dict[str, list[int]]:
    """Record tensor shapes through the architecture."""

    shapes: dict[str, list[int]] = {
        "input": list(features.shape),
    }

    intermediate = features

    for index, block in enumerate(model.features, start=1):
        intermediate = block(intermediate)
        shapes[f"conv_block_{index}"] = list(intermediate.shape)

    pooled = model.global_pool(intermediate)
    shapes["global_pool"] = list(pooled.shape)

    logits = model.classifier(pooled)
    shapes["classifier_output_before_squeeze"] = list(logits.shape)

    return shapes


def main() -> int:
    """Run model architecture validation."""

    try:
        set_seed(RANDOM_SEED)

        dataset = PromoterDataset(
            TRAIN_PATH,
            expected_length=EXPECTED_SEQUENCE_LENGTH,
            return_metadata=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        features, labels = next(iter(loader))

        config = CNNBaselineConfig(
            input_channels=4,
            conv_channels=(32, 64, 128),
            kernel_sizes=(15, 11, 7),
            pool_size=4,
            dense_units=64,
            dropout=0.40,
        )

        model = PromoterCNNBaseline(config=config)
        model.train()

        shapes = inspect_intermediate_shapes(
            model=model,
            features=features,
        )

        logits = validate_forward_pass(
            model=model,
            features=features,
            labels=labels,
        )

        loss = validate_backward_pass(
            model=model,
            logits=logits,
            labels=labels,
            positive_weight=dataset.positive_weight,
        )

        trainable_parameters = count_trainable_parameters(model)

        if trainable_parameters <= 0:
            raise ValueError("Model contains no trainable parameters.")

        model.eval()

        with torch.no_grad():
            probabilities = model.predict_proba(features)

        summary = {
            "model_name": "PromoterCNNBaseline",
            "random_seed": RANDOM_SEED,
            "configuration": config.to_dict(),
            "input_batch_shape": list(features.shape),
            "label_batch_shape": list(labels.shape),
            "output_logit_shape": list(logits.shape),
            "output_probability_shape": list(probabilities.shape),
            "intermediate_shapes": shapes,
            "trainable_parameters": trainable_parameters,
            "positive_weight_from_train": dataset.positive_weight,
            "validation_loss": loss,
            "forward_pass": "PASS",
            "backward_pass": "PASS",
            "finite_logits": True,
            "finite_gradients": True,
            "output_policy": (
                "raw_logit_during_training_sigmoid_only_for_inference"
            ),
        }

        SUMMARY_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with SUMMARY_PATH.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print("=" * 72)
        print("CNN BASELINE VALIDATION PASSED")
        print("=" * 72)
        print(f"Input batch:          {tuple(features.shape)}")
        print(f"Output logits:        {tuple(logits.shape)}")
        print(f"Trainable parameters: {trainable_parameters:,}")
        print(f"Initial loss:         {loss:.6f}")
        print(f"Train pos_weight:     {dataset.positive_weight:.6f}")

        print("\nIntermediate shapes:")

        for layer_name, shape in shapes.items():
            print(f"  {layer_name:35s} {tuple(shape)}")

        print(f"\nSummary: {SUMMARY_PATH}")
        print("=" * 72)

        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        RuntimeError,
        IndexError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
