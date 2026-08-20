"""One-hot encoding utilities for DNA promoter sequences."""

from __future__ import annotations

import numpy as np
import torch


DNA_ALPHABET = ("A", "C", "G", "T")
BASE_TO_CHANNEL = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def one_hot_encode_numpy(
    sequence: str,
    expected_length: int | None = 2200,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Encode a DNA sequence as a channels-first NumPy array.

    Parameters
    ----------
    sequence
        DNA sequence containing A, C, G, T or N.
    expected_length
        Required sequence length. Set to None to disable length validation.
    dtype
        NumPy output data type.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(4, sequence_length)``.

    Notes
    -----
    Channel order is A, C, G, T. Ambiguous bases represented by N are
    encoded as zeros in every channel.
    """

    if not isinstance(sequence, str):
        raise TypeError(
            f"sequence must be a string, observed {type(sequence).__name__}"
        )

    sequence = sequence.upper()

    if expected_length is not None and len(sequence) != expected_length:
        raise ValueError(
            f"Expected sequence length {expected_length}, "
            f"observed {len(sequence)}"
        )

    invalid_bases = set(sequence) - set("ACGTN")

    if invalid_bases:
        raise ValueError(
            f"Invalid nucleotide symbols detected: {sorted(invalid_bases)}"
        )

    encoded = np.zeros(
        (len(DNA_ALPHABET), len(sequence)),
        dtype=dtype,
    )

    for position, base in enumerate(sequence):
        channel = BASE_TO_CHANNEL.get(base)

        if channel is not None:
            encoded[channel, position] = 1.0

    return encoded


def one_hot_encode_tensor(
    sequence: str,
    expected_length: int | None = 2200,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode a DNA sequence as a channels-first PyTorch tensor."""

    array = one_hot_encode_numpy(
        sequence=sequence,
        expected_length=expected_length,
        dtype=np.float32,
    )

    return torch.as_tensor(array, dtype=dtype)


def decode_one_hot(
    encoded: np.ndarray | torch.Tensor,
) -> str:
    """Decode a channels-first one-hot representation.

    Positions containing zeros in all four channels are decoded as N.
    """

    if isinstance(encoded, torch.Tensor):
        array = encoded.detach().cpu().numpy()
    else:
        array = np.asarray(encoded)

    if array.ndim != 2 or array.shape[0] != 4:
        raise ValueError(
            "Expected a two-dimensional array with shape (4, length), "
            f"observed {array.shape}"
        )

    decoded: list[str] = []

    for position in range(array.shape[1]):
        column = array[:, position]

        if np.allclose(column, 0):
            decoded.append("N")
            continue

        maximum = float(column.max())

        if not np.isclose(maximum, 1.0):
            raise ValueError(
                f"Invalid one-hot values at position {position}: {column}"
            )

        maximum_indices = np.flatnonzero(np.isclose(column, maximum))

        if len(maximum_indices) != 1:
            raise ValueError(
                f"Ambiguous one-hot encoding at position {position}: {column}"
            )

        decoded.append(DNA_ALPHABET[int(maximum_indices[0])])

    return "".join(decoded)
