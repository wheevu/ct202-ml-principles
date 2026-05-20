"""
features.py — Feature extraction functions for Pokémon sprite images.

Transforms raw pixel arrays into compact feature vectors for KNN classification.
Each function takes an RGB image array and returns a 1D numpy feature vector.
"""

import numpy as np


def average_rgb(image):
    """
    Compute mean R, G, B values normalized to [0, 1].

    This is the simplest possible feature — just 3 numbers representing
    the average color of the image. Fast but loses all spatial information.

    Args:
        image (numpy.ndarray): Shape (H, W, 3), dtype uint8.

    Returns:
        numpy.ndarray: Shape (3,), values in [0, 1].
    """
    # Average over height and width dimensions
    means = image.mean(axis=(0, 1))
    # Normalize from 0–255 to 0–1
    return means / 255.0


def color_histogram(image, bins=8):
    """
    Compute a color histogram per RGB channel, concatenated and normalized.

    A histogram captures the *distribution* of colors, not just the average.
    This gives more information than avg_rgb: e.g., a red-and-white sprite
    has a different histogram than a solid red sprite.

    Args:
        image (numpy.ndarray): Shape (H, W, 3), dtype uint8.
        bins (int): Number of bins per channel (total = bins * 3).

    Returns:
        numpy.ndarray: Shape (bins * 3,), values in [0, 1].
    """
    histograms = []
    for channel in range(3):
        # Compute histogram for this channel
        channel_data = image[:, :, channel].flatten()
        hist, _ = np.histogram(channel_data, bins=bins, range=(0, 256))
        histograms.append(hist)

    # Concatenate all channel histograms
    combined = np.concatenate(histograms).astype(np.float64)
    # Normalize so the vector sums to 1 (ignoring if all zeros)
    total = combined.sum()
    if total > 0:
        combined = combined / total
    return combined


def downsampled_pixels(image, size=(16, 16)):
    """
    Downsample the image and flatten RGB values into a 1D vector.

    This preserves spatial structure (unlike avg_rgb or histogram) but at
    a much lower resolution than the original image. Useful for seeing if
    rough shape information helps classification.

    Args:
        image (numpy.ndarray): Shape (H, W, 3), dtype uint8.
        size (tuple): (width, height) to downsample to.

    Returns:
        numpy.ndarray: Shape (size[0] * size[1] * 3,), values in [0, 1].
    """
    # Use PIL for resizing, then convert back to numpy
    from PIL import Image

    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize(size, Image.Resampling.LANCZOS)
    resized = np.array(pil_img, dtype=np.float64)
    # Flatten and normalize
    return resized.flatten() / 255.0


def extract_features(image, method="avg_rgb", **kwargs):
    """
    Extract a feature vector from an image using the specified method.

    Args:
        image (numpy.ndarray): Shape (H, W, 3), dtype uint8.
        method (str): One of "avg_rgb", "histogram", "pixels", "combined".
        **kwargs: Additional arguments passed to the feature function.

    Returns:
        numpy.ndarray: 1D feature vector.

    Raises:
        ValueError: If method is unknown.
    """
    if method == "avg_rgb":
        return average_rgb(image)
    elif method == "histogram":
        bins = kwargs.get("bins", 8)
        return color_histogram(image, bins=bins)
    elif method == "pixels":
        size = kwargs.get("size", (16, 16))
        return downsampled_pixels(image, size=size)
    elif method == "combined":
        # Concatenate avg_rgb and color_histogram
        avg = average_rgb(image)
        hist = color_histogram(image, bins=kwargs.get("bins", 8))
        return np.concatenate([avg, hist])
    else:
        raise ValueError(
            f"Unknown feature method: '{method}'. "
            f"Choose from: avg_rgb, histogram, pixels, combined"
        )


def get_feature_dim(method, image_size=(64, 64), bins=8, pixel_size=(16, 16)):
    """
    Return the dimensionality of a feature vector for the given method.

    Useful for logging and understanding feature space size.

    Args:
        method (str): Feature method name.
        image_size (tuple): Original image dimensions (unused here, for future use).
        bins (int): Number of histogram bins.
        pixel_size (tuple): Downsample size for pixel method.

    Returns:
        int: Length of feature vector.
    """
    if method == "avg_rgb":
        return 3
    elif method == "histogram":
        return bins * 3
    elif method == "pixels":
        return pixel_size[0] * pixel_size[1] * 3
    elif method == "combined":
        return 3 + bins * 3
    else:
        return 0
