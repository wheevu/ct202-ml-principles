"""
visualize.py — Visualization utilities for KNN experiment outputs.

Generates:
  - Sample grid: thumbnail grid of images from the dataset
  - Confusion matrix: heatmap of true vs. predicted labels
  - Nearest neighbors: a test image with its k closest neighbors
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_sample_grid(samples, output_path, cols=8):
    """
    Save a grid of sample images with their labels.

    Args:
        samples (list[dict]): List of sample dicts with "image" and "label".
        output_path (str or Path): Where to save the image.
        cols (int): Number of columns in the grid.
    """
    n = len(samples)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()

    for i in range(rows * cols):
        if i < n:
            ax = axes[i]
            ax.imshow(samples[i]["image"])
            ax.set_title(samples[i]["label"], fontsize=8)
            ax.axis("off")
        else:
            axes[i].axis("off")

    plt.suptitle("PokéKNN — Sample Images", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sample grid: {output_path}")


def save_confusion_matrix(matrix, labels, output_path):
    """
    Save a confusion matrix heatmap.

    Args:
        matrix (dict): Nested dict matrix[true][pred] = count.
        labels (list): Sorted class labels.
        output_path (str or Path): Where to save the image.
    """
    n = len(labels)
    data = np.zeros((n, n), dtype=int)
    for i, true in enumerate(labels):
        for j, pred in enumerate(labels):
            data[i, j] = matrix[true][pred]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="Blues", interpolation="nearest")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(data[i, j]),
                    ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    plt.colorbar(im, ax=ax, shrink=0.75)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix: {output_path}")


def save_nearest_neighbors(query_sample, neighbor_samples, distances, prediction, output_path):
    """
    Save a visualization showing a test image and its k nearest neighbors.

    The query image is displayed on the left, neighbors to the right,
    with distances and labels shown below each neighbor.

    Args:
        query_sample (dict): The test sample dict with "image" and "label".
        neighbor_samples (list[dict]): The k nearest neighbor sample dicts.
        distances (list[float]): Distances corresponding to each neighbor.
        prediction (str): Predicted label for the query.
        output_path (str or Path): Where to save the image.
    """
    k = len(neighbor_samples)
    fig, axes = plt.subplots(1, k + 1, figsize=((k + 1) * 2.5, 3))

    # Query image
    axes[0].imshow(query_sample["image"])
    axes[0].set_title(f"Query\nTrue: {query_sample['label']}\nPred: {prediction}", fontsize=9)
    axes[0].axis("off")

    # Neighbors
    for i, (neighbor, dist) in enumerate(zip(neighbor_samples, distances)):
        ax = axes[i + 1]
        ax.imshow(neighbor["image"])
        ax.set_title(f"#{i + 1}\n{neighbor['label']}\nd={dist:.3f}", fontsize=8)
        ax.axis("off")

    plt.suptitle("K-Nearest Neighbors", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved nearest neighbors: {output_path}")
