"""
knn.py — K-Nearest Neighbors classifier implemented from scratch.

Core logic:
  1. Compute Euclidean distance between query and all training samples.
  2. Find the k closest training samples.
  3. Let those neighbors vote on the class label.
"""

import numpy as np


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors.

    d(a, b) = sqrt(sum((a_i - b_i)^2))

    Args:
        a (numpy.ndarray): 1D vector.
        b (numpy.ndarray): 1D vector of same length.

    Returns:
        float: Euclidean distance.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def get_neighbors(train_features, train_labels, query_feature, k):
    """
    Find the k nearest neighbors to a query feature vector.

    Args:
        train_features (numpy.ndarray): Shape (n_train, n_features).
        train_labels (list): Length n_train, aligned with train_features.
        query_feature (numpy.ndarray): 1D feature vector to classify.
        k (int): Number of neighbors.

    Returns:
        tuple: (neighbor_labels, neighbor_distances)
            - neighbor_labels (list): Labels of the k nearest neighbors.
            - neighbor_distances (list): Corresponding distances.
    """
    # Compute distance from query to every training sample
    distances = []
    for i, train_vec in enumerate(train_features):
        dist = euclidean_distance(query_feature, train_vec)
        distances.append((dist, train_labels[i]))

    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[0])

    # Take the k nearest
    k_nearest = distances[:k]
    neighbor_labels = [item[1] for item in k_nearest]
    neighbor_distances = [item[0] for item in k_nearest]

    return neighbor_labels, neighbor_distances


def majority_vote(labels):
    """
    Determine the majority class from a list of labels.

    In case of a tie, pick alphabetically (deterministic and simple).

    Args:
        labels (list): List of class labels (strings).

    Returns:
        str: The winning label.
    """
    # Count occurrences of each label
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    # Find the max count; break ties alphabetically
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    # Sort alphabetically for deterministic tie-breaking
    candidates.sort()
    return candidates[0]


def predict_one(train_features, train_labels, query_feature, k):
    """
    Predict the class label for a single query sample.

    Args:
        train_features (numpy.ndarray): Training feature matrix.
        train_labels (list): Training labels.
        query_feature (numpy.ndarray): Query feature vector.
        k (int): Number of neighbors.

    Returns:
        tuple: (predicted_label, neighbor_labels, neighbor_distances)
    """
    neighbor_labels, neighbor_distances = get_neighbors(
        train_features, train_labels, query_feature, k
    )
    prediction = majority_vote(neighbor_labels)
    return prediction, neighbor_labels, neighbor_distances


def predict_batch(train_features, train_labels, test_features, k):
    """
    Predict class labels for multiple test samples.

    Args:
        train_features (numpy.ndarray): Training feature matrix.
        train_labels (list): Training labels.
        test_features (numpy.ndarray): Test feature matrix.
        k (int): Number of neighbors.

    Returns:
        tuple: (predictions, all_neighbor_labels, all_neighbor_distances)
            - predictions (list): Predicted labels for each test sample.
            - all_neighbor_labels (list of list): Neighbor labels per sample.
            - all_neighbor_distances (list of list): Neighbor distances per sample.
    """
    predictions = []
    all_neighbor_labels = []
    all_neighbor_distances = []

    for query in test_features:
        pred, n_labels, n_dists = predict_one(train_features, train_labels, query, k)
        predictions.append(pred)
        all_neighbor_labels.append(n_labels)
        all_neighbor_distances.append(n_dists)

    return predictions, all_neighbor_labels, all_neighbor_distances
