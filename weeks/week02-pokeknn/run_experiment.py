"""
run_experiment.py — Main entry point for the PokéKNN experiment.

Loads Pokémon sprite images, extracts features, runs KNN classification
for multiple feature methods and k values, evaluates results, compares
against baselines, and saves visualizations.

Usage:
    python run_experiment.py
    python run_experiment.py --k 5 --feature combined
    python run_experiment.py --feature all --k-values 1,3,5,7
    python run_experiment.py --data-dir data/raw --output-dir outputs
"""

import argparse
import sys
from pathlib import Path

# Ensure the script can import from src/ whether run from the week02 folder
# or from the repo root
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from src.dataset import load_samples, train_test_split, print_split_distribution, get_labels
from src.features import extract_features, get_feature_dim
from src.knn import predict_batch
from src.evaluate import (
    accuracy_score,
    confusion_matrix,
    print_evaluation_summary,
    majority_class_baseline,
    random_baseline,
)
from src.visualize import save_sample_grid, save_confusion_matrix, save_nearest_neighbors


def check_dataset(data_dir):
    """
    Verify that the dataset directory exists and contains class subfolders with images.

    Returns True if the dataset looks usable, False otherwise.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_path}")
        return False

    class_folders = sorted([f for f in data_path.iterdir() if f.is_dir()])
    if not class_folders:
        print(f"[ERROR] No class folders found in {data_path}.")
        return False

    # Check that at least one folder has images
    any_images = False
    for class_folder in class_folders:
        image_count = 0
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            image_count += len(list(class_folder.glob(f"*{ext}")))
        print(f"  Found {image_count} images in '{class_folder.name}'")
        if image_count > 0:
            any_images = True

    if not any_images:
        print(f"\n[ERROR] No image files found in any class folder.")
        return False

    return True


def print_setup_instructions(data_dir):
    """Print instructions for setting up the dataset when none is found."""
    data_path = Path(data_dir)
    print("\n" + "=" * 55)
    print("  Dataset not found or incomplete.")
    print("=" * 55)
    print(f"\nTo set up the PokéKNN dataset:")
    print(f"\n  1. Create class folders:")
    for cls in ["fire", "water", "grass", "electric"]:
        folder = data_path / cls
        print(f"     mkdir -p {folder}")
    print(f"\n  2. Download Pokémon sprite PNG/JPG images and place them")
    print(f"     into the corresponding folders.")
    print(f"     Sources: https://pokemondb.net/sprites")
    print(f"              https://veekun.com/dex/downloads")
    print(f"\n  3. Rerun this script:")
    print(f"     python run_experiment.py")
    print(f"\n  Tip: ~10 images per class is enough to start.")
    print("=" * 55)


def parse_k_values(k_values_str):
    """Parse a comma-separated string like '1,3,5,7' into a list of ints."""
    return [int(k.strip()) for k in k_values_str.split(",")]


def run_experiment(args):
    """Run the full PokéKNN experiment."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    image_size = (args.image_size, args.image_size)
    test_size = args.test_size
    seed = args.seed
    feature_choice = args.feature

    # Determine k values to try
    if args.k_values:
        k_values = parse_k_values(args.k_values)
    else:
        k_values = [args.k]

    # Determine which feature methods to run
    if feature_choice == "all":
        feature_methods = ["avg_rgb", "histogram", "pixels", "combined"]
    else:
        feature_methods = [feature_choice]

    print("\n" + "=" * 60)
    print("  PokéKNN — KNN Pokémon Type Classifier")
    print("=" * 60)
    print(f"  Data directory:   {data_dir}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Image size:        {image_size[0]}x{image_size[1]}")
    print(f"  k values:          {', '.join(str(k) for k in k_values)}")
    print(f"  Test split:        {test_size:.0%}")
    print(f"  Seed:              {seed}")
    print(f"  Feature methods:   {', '.join(feature_methods)}")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    try:
        samples = load_samples(data_dir, image_size=image_size)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print_setup_instructions(data_dir)
        return

    print(f"  Loaded {len(samples)} samples total.")
    labels = get_labels(samples)
    print(f"  Classes: {', '.join(labels)}")

    # ── Step 2: Stratified split ──────────────────────────────────────
    print("\n[2/5] Stratified train/test split...")
    train_samples, test_samples = train_test_split(samples, test_size=test_size, seed=seed)
    print_split_distribution(train_samples, test_samples, labels)

    # ── Step 3: Save sample grid ──────────────────────────────────────
    print("\n[3/5] Saving sample grid...")
    grid_samples = train_samples[:min(16, len(train_samples))]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_sample_grid(grid_samples, output_dir / "sample_grid.png")

    # Extract labels for reuse
    train_labels = [s["label"] for s in train_samples]
    test_labels = [s["label"] for s in test_samples]

    # ── Step 4: Baselines ─────────────────────────────────────────────
    print("\n[4/5] Computing baselines...")

    # Majority-class baseline: always predict the most common training label
    majority_preds, majority_acc = majority_class_baseline(train_labels, len(test_labels))
    # Evaluate majority baseline on the actual test labels
    majority_true_acc = accuracy_score(test_labels, majority_preds)
    print(f"  Majority-class baseline:  {majority_true_acc:.2%} "
          f"(always predict '{majority_preds[0]}')")

    # Random baseline: randomly pick a label per test sample
    random_preds, random_expected = random_baseline(labels, len(test_labels), seed=seed)
    random_true_acc = accuracy_score(test_labels, random_preds)
    print(f"  Random baseline:         {random_true_acc:.2%} "
          f"(expected ~{random_expected:.0%} with {len(labels)} classes)")

    # ── Step 5: Run experiments (feature × k) ─────────────────────────
    print("\n[5/5] Running KNN experiments...")

    # Store results as: results[feature_method][k] = accuracy
    results = {}

    for method in feature_methods:
        feat_dim = get_feature_dim(method, image_size=image_size)
        print(f"\n  --- Feature: '{method}' (dim={feat_dim}) ---")

        # Extract features once per method (same features, different k)
        train_features = [extract_features(s["image"], method) for s in train_samples]
        test_features = [extract_features(s["image"], method) for s in test_samples]

        results[method] = {}

        for k in k_values:
            # Predict
            predictions, neighbor_labels, neighbor_dists = predict_batch(
                train_features, train_labels, test_features, k
            )
            acc = accuracy_score(test_labels, predictions)
            results[method][k] = {
                "accuracy": acc,
                "predictions": predictions,
                "neighbor_labels_list": neighbor_labels,
                "neighbor_distances_list": neighbor_dists,
            }
            print(f"    k={k}: {acc:.2%}")

    # ── Print compact k-sweep table ───────────────────────────────────
    if len(k_values) > 1 or len(feature_methods) > 1:
        print("\n" + "-" * 50)
        print("  Accuracy summary:")
        print(f"  {'Feature':<12}", end="")
        for k in k_values:
            print(f"  {'k=' + str(k):>8}", end="")
        print()
        print("  " + "-" * (12 + len(k_values) * 10))

        for method in feature_methods:
            print(f"  {method:<12}", end="")
            for k in k_values:
                print(f"  {results[method][k]['accuracy']:>7.1%}", end="")
            print()
        print("-" * 50)

    # ── Find the single best (feature, k) combination ─────────────────
    best_feature = None
    best_k = None
    best_acc = -1.0

    for method in feature_methods:
        for k in k_values:
            acc = results[method][k]["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_feature = method
                best_k = k

    print(f"\n  Best combination: {best_feature}, k={best_k} ({best_acc:.2%})")

    # ── Print full evaluation for the best combination ────────────────
    best_result = results[best_feature][best_k]
    print_evaluation_summary(
        test_labels,
        best_result["predictions"],
        labels,
        feature_name=best_feature,
        k=best_k,
    )

    # ── Save visualizations for the best combination ──────────────────
    print("  Saving visualizations...")

    # Confusion matrix
    cm = confusion_matrix(test_labels, best_result["predictions"], labels)
    save_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")

    # Nearest neighbors (use first test sample)
    if test_samples:
        query_sample = test_samples[0]
        pred_label = best_result["predictions"][0]
        neighbor_dists = best_result["neighbor_distances_list"][0]
        neighbor_labels = best_result["neighbor_labels_list"][0]

        # Find the corresponding training images for the neighbors
        neighbor_samples = []
        for nlabel in neighbor_labels:
            match = None
            for s in train_samples:
                if s["label"] == nlabel:
                    match = s
                    break
            if match:
                neighbor_samples.append(match)
            else:
                neighbor_samples.append(query_sample)

        save_nearest_neighbors(
            query_sample,
            neighbor_samples[:best_k],
            neighbor_dists[:best_k],
            pred_label,
            output_dir / "nearest_neighbors.png",
        )

    # ── Final summary ─────────────────────────────────────────────────
    beats_baseline = "yes" if best_acc > majority_true_acc else "no (tie or below)"

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    print(f"  Dataset size:              {len(samples)}")
    print(f"  Classes:                   {', '.join(labels)}")
    print(f"  Train/test split:          {len(train_samples)}/{len(test_samples)}")
    print(f"  Majority-class baseline:   {majority_true_acc:.2%}")
    print(f"  Random baseline:           {random_true_acc:.2%}")
    print(f"  Best KNN feature method:   {best_feature}")
    print(f"  Best k:                    {best_k}")
    print(f"  Best KNN accuracy:         {best_acc:.2%}")
    print(f"  Beats majority baseline?   {beats_baseline}")
    print(f"\n  Outputs saved to: {output_dir}/")
    print(f"    - sample_grid.png")
    print(f"    - confusion_matrix.png")
    print(f"    - nearest_neighbors.png")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PokéKNN — Classification with KNN from scratch"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw data directory with class subfolders (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to output directory for visualizations (default: outputs)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of neighbors for KNN (default: 3). Ignored if --k-values is set.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Comma-separated k values to sweep, e.g. '1,3,5,7' (default: use --k)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resize images to image-size x image-size (default: 64)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="all",
        choices=["avg_rgb", "histogram", "pixels", "combined", "all"],
        help="Feature method to use, or 'all' to compare all methods (default: all)",
    )

    args = parser.parse_args()

    # Resolve data directory relative to the script's location
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir
    args.data_dir = str(data_dir)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    args.output_dir = str(output_dir)

    # Check dataset first
    if not check_dataset(args.data_dir):
        print_setup_instructions(args.data_dir)
        return

    run_experiment(args)


if __name__ == "__main__":
    main()
