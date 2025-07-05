# sequence_padding_module.py

import numpy as np
import imghdr
from typing import List, Union


def is_valid_image(image_path: str) -> bool:
    """
    Validates whether the file at the specified path is a valid image.

    Uses the `imghdr` module to determine the file type.

    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if the file is a valid image, False otherwise.
    """
    return imghdr.what(image_path) is not None


def pad_sequences_manual(
    sequences: List[np.ndarray],
    maxlen: int,
    dtype: str = "float32",
    padding: str = "post",
    truncating: str = "post",
    value: Union[int, float] = 0.0
) -> np.ndarray:
    """
    Pads a list of 2D arrays (sequences) to a uniform length.

    The padding is applied based on the specified parameters.

    Args:
        sequences (List[np.ndarray]): List of 2D numpy arrays (e.g., list of tensors).
        maxlen (int): Maximum length of the padded sequences.
        dtype (str): Desired data type of the output array. Default is 'float32'.
        padding (str): 'post' (default) or 'pre', specifies where to pad.
        truncating (str): 'post' (default) or 'pre', specifies where to truncate.
        value (Union[int, float]): Padding value. Default is 0.0.

    Returns:
        np.ndarray: Numpy array of shape (len(sequences), maxlen, feature_dim).

    Raises:
        ValueError: If the input `sequences` is empty or sequences have inconsistent feature dimensions.
    """
    if not sequences:
        raise ValueError("Input 'sequences' cannot be empty.")

    # Determine feature dimensions
    feature_dim = len(sequences[0][0]) if sequences[0] else 0
    if any(len(seq[0]) != feature_dim for seq in sequences if seq):
        raise ValueError("All sequences must have the same feature dimension.")

    # Initialize the padded array
    padded_sequences = np.full((len(sequences), maxlen, feature_dim), value, dtype=dtype)

    for idx, seq in enumerate(sequences):
        if len(seq) == 0:  # Handle empty sequences
            continue

        # Apply truncation
        if truncating == "post":
            truncated_seq = seq[:maxlen]
        else:  # "pre"
            truncated_seq = seq[-maxlen:]

        # Apply padding
        if padding == "post":
            padded_sequences[idx, :len(truncated_seq)] = truncated_seq
        else:  # "pre"
            padded_sequences[idx, -len(truncated_seq):] = truncated_seq

    return padded_sequences
