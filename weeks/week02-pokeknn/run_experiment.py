"""
run_experiment.py — Main entry point for the PokéKNN experiment.

Loads Pokémon sprite images, extracts features, runs KNN classification
for multiple feature methods, evaluates results, and saves visualizations.

Usage:
    python run_experiment.py
    python run_experiment.py --k 5 --feature combined
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

from src.dataset import load_samples, train_test_split, get_labels
from src.features import extract_features, get_feature_dim
from src.knn import predict_batch
from src.evaluate import print_evaluation_summary
from src.visualize import save_sample_grid, save_confusion_matrix, save_nearest_neighbors

# Import evaluate helpers directly for clarity
from src.evaluate import accuracy_score, confusion_matrix


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
    """
    Print instructions for setting up the dataset when none is found.
    """
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


def run_experiment(args):
    """
    Run the full PokéKNN experiment.
    """
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    k = args.k
    image_size = (args.image_size, args.image_size)
    test_size = args.test_size
    seed = args.seed
    feature_choice = args.feature

    # Determine which feature methods to run
    if feature_choice == "all":
        feature_methods = ["avg_rgb", "histogram", "pixels", "combined"]
    else:
        feature_methods = [feature_choice]

    print("\n" + "=" * 55)
    print("  PokéKNN — KNN Pokémon Type Classifier")
    print("=" * 55)
    print(f"  Data directory:  {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Image size:       {image_size[0]}x{image_size[1]}")
    print(f"  k (neighbors):    {k}")
    print(f"  Test split:       {test_size:.0%}")
    print(f"  Seed:             {seed}")
    print(f"  Feature methods:  {', '.join(feature_methods)}")
    print("=" * 55)

    # --- Step 1: Load data ---
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

    # --- Step 2: Split into train/test ---
    print("\n[2/5] Splitting dataset...")
    train_samples, test_samples = train_test_split(samples, test_size=test_size, seed=seed)
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

    # --- Step 3: Save sample grid ---
    print("\n[3/5] Saving sample grid...")
    # Show up to 16 samples from the training set
    grid_samples = train_samples[:min(16, len(train_samples))]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_sample_grid(grid_samples, output_dir / "sample_grid.png")

    # --- Step 4: Run experiments for each feature method ---
    print("\n[4/5] Running experiments...")

    results = {}

    for method in feature_methods:
        print(f"\n  --- Feature method: '{method}' ---")
        feat_dim = get_feature_dim(method, image_size=image_size)
        print(f"  Feature vector size: {feat_dim}")

        # Extract features
        train_features = [extract_features(s["image"], method) for s in train_samples]
        test_features = [extract_features(s["image"], method) for s in test_samples]

        train_features_arr = [f.tolist() if hasattr(f, 'tolist') else f for f in train_features]
        test_features_arr = [f.tolist() if hasattr(f, 'tolist') else f for f in test_features]

        train_labels = [s["label"] for s in train_samples]
        test_labels = [s["label"] for s in test_samples]

        # Predict
        predictions, neighbor_labels_list, neighbor_distances_list = predict_batch(
            train_features, train_labels, test_features, k
        )

        # Evaluate
        acc = accuracy_score(test_labels, predictions)
        results[method] = {
            "accuracy": acc,
            "predictions": predictions,
            "test_labels": test_labels,
            "neighbor_labels_list": neighbor_labels_list,
            "neighbor_distances_list": neighbor_distances_list,
            "test_samples": test_samples,
        }

        print_evaluation_summary(test_labels, predictions, labels, feature_name=method, k=k)

    # --- Step 5: Pick best method and save outputs ---
    print("\n[5/5] Saving visualizations...")

    # Find the best feature method by accuracy
    best_method = max(results, key=lambda m: results[m]["accuracy"])
    best_result = results[best_method]
    print(f"  Best feature method: '{best_method}' "
          f"(accuracy: {best_result['accuracy']:.2%})")

    # Save confusion matrix for the best method
    cm = confusion_matrix(
        best_result["test_labels"],
        best_result["predictions"],
        labels
    )
    save_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")

    # Save nearest neighbors for a test sample (first test sample)
    if best_result["test_samples"]:
        query_sample = best_result["test_samples"][0]
        pred_label = best_result["predictions"][0]
        neighbor_dists = best_result["neighbor_distances_list"][0]
        neighbor_labels = best_result["neighbor_labels_list"][0]

        # Find the corresponding training samples for the neighbors
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
                # Fallback: use a placeholder (shouldn't happen in practice)
                neighbor_samples.append(query_sample)

        save_nearest_neighbors(
            query_sample,
            neighbor_samples[:k],  # Only k neighbors
            neighbor_dists[:k],
            pred_label,
            output_dir / "nearest_neighbors.png",
        )

    # --- Final summary ---
    print("\n" + "=" * 55)
    print("  Experiment complete!")
    print("=" * 55)
    print(f"\n  Results summary:")
    for method in feature_methods:
        acc = results[method]["accuracy"]
        print(f"    {method:<12}  {acc:.2%}")
    print(f"\n  Best method: {best_method} ({results[best_method]['accuracy']:.2%})")
    print(f"\n  Outputs saved to: {output_dir}/")
    print(f"    - sample_grid.png")
    print(f"    - confusion_matrix.png")
    print(f"    - nearest_neighbors.png")
    print("=" * 55)


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
        help="Number of neighbors for KNN (default: 3)",
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
