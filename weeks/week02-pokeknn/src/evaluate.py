"""
evaluate.py — Model evaluation utilities.

Functions for computing accuracy, confusion matrices, and basic classification
reports without relying on scikit-learn.
"""


def accuracy_score(y_true, y_pred):
    """
    Compute the fraction of correct predictions.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.

    Returns:
        float: Accuracy as a value between 0.0 and 1.0.
    """
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, labels):
    """
    Compute a confusion matrix from scratch.

    Rows = true labels, Columns = predicted labels.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        labels (list): Sorted list of all unique class labels.

    Returns:
        dict: Nested dict matrix[true_label][pred_label] = count.
    """
    # Initialize matrix with zeros
    matrix = {true: {pred: 0 for pred in labels} for true in labels}

    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1

    return matrix


def classification_report_basic(y_true, y_pred, labels):
    """
    Generate a basic per-class performance summary.

    Reports correct / total for each class.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        labels (list): All unique class labels.

    Returns:
        str: A human-readable report string.
    """
    lines = []
    lines.append(f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append("-" * 40)

    for label in labels:
        total = sum(1 for t in y_true if t == label)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        acc = correct / total if total > 0 else 0.0
        lines.append(f"{label:<12} {correct:>8} {total:>8} {acc:>10.2%}")

    return "\n".join(lines)


def print_evaluation_summary(y_true, y_pred, labels, feature_name="", k=3):
    """
    Print a complete evaluation summary to the console.

    Includes accuracy, per-class breakdown, and confusion matrix.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        labels (list): All unique class labels.
        feature_name (str): Name of the feature method used.
        k (int): Number of neighbors used.
    """
    acc = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels)

    print("\n" + "=" * 55)
    print(f"  Evaluation — Feature: {feature_name}, k={k}")
    print("=" * 55)
    print(f"  Accuracy: {acc:.2%}  ({int(acc * len(y_true))}/{len(y_true)})")
    print()

    print(classification_report_basic(y_true, y_pred, labels))
    print()

    print("  Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {'':<12}", end="")
    for pred_label in labels:
        print(f"{pred_label:<12}", end="")
    print()
    for true_label in labels:
        print(f"  {true_label:<12}", end="")
        for pred_label in labels:
            print(f"{matrix[true_label][pred_label]:<12}", end="")
        print()
    print("=" * 55)
