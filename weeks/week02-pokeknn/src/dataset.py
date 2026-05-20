"""
dataset.py — Image loading, preprocessing, and train/test splitting utilities.

Handles loading Pokémon sprite images from class folders, compositing RGBA
images onto a white background, resizing, and splitting into train/test sets.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image


def composited_to_rgb(image):
    """
    Convert an image to RGB, handling RGBA (transparent) backgrounds.

    Pillow's Image.convert('RGB') leaves transparent pixels black, which
    would bias color features. Instead, composite the sprite onto a white
    background first so transparency doesn't add dark artifacts.

    Args:
        image (PIL.Image): Input image (any mode).

    Returns:
        PIL.Image: RGB image composited on white background.
    """
    # Palette (P) mode can have transparency — expand to RGBA first
    if image.mode == "P":
        image = image.convert("RGBA")

    if image.mode == "RGBA":
        # Create a white background the same size
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        # Composite the sprite onto the background
        background.paste(image, (0, 0), image)
        return background.convert("RGB")
    else:
        return image.convert("RGB")


def load_samples(data_dir, image_size=(64, 64), allowed_extensions=(".png", ".jpg", ".jpeg")):
    """
    Load all images from class subfolders into a list of samples.

    Expected directory structure:
        data_dir/
            fire/
                charmander.png
                ...
            water/
                squirtle.png
                ...
            grass/
                bulbasaur.png
                ...
            electric/
                pikachu.png
                ...

    Each sample is a dict:
        {
            "image": numpy array of shape (H, W, 3), dtype uint8,
            "label": str (folder name),
            "path":  str (original file path)
        }

    Args:
        data_dir (str or Path): Path to the raw data directory.
        image_size (tuple): (width, height) to resize images to.
        allowed_extensions (tuple): Image file extensions to load.

    Returns:
        list[dict]: List of sample dictionaries.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    samples = []
    # Each subfolder is treated as a class label
    class_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])

    if not class_folders:
        raise FileNotFoundError(
            f"No class folders found in {data_dir}. "
            f"Expected subfolders like 'fire/', 'water/', etc."
        )

    for class_folder in class_folders:
        label = class_folder.name
        image_files = []
        for ext in allowed_extensions:
            image_files.extend(class_folder.glob(f"*{ext}"))
            image_files.extend(class_folder.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"  Warning: No images found in {class_folder}")
            continue

        for img_path in sorted(image_files):
            try:
                with Image.open(img_path) as img:
                    img = composited_to_rgb(img)
                    img = img.resize(image_size, Image.Resampling.LANCZOS)
                    image_array = np.array(img, dtype=np.uint8)

                samples.append({
                    "image": image_array,
                    "label": label,
                    "path": str(img_path),
                })
            except Exception as e:
                print(f"  Warning: Could not load {img_path}: {e}")

    return samples


def train_test_split(samples, test_size=0.2, seed=42):
    """
    Split samples into training and test sets with stratification.

    Stratified = split each class separately so every class appears in
    both train and test when possible. This avoids accidentally leaving
    a class out of the test set when the dataset is small.

    Done from scratch — no scikit-learn dependency.

    Args:
        samples (list): List of sample dicts.
        test_size (float): Fraction of samples to use for testing (0.0–1.0).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_samples, test_samples)
    """
    rng = random.Random(seed)
    train = []
    test = []

    # Group samples by class label
    by_label = {}
    for s in samples:
        label = s["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(s)

    # Split each class separately
    for label, class_samples in sorted(by_label.items()):
        # Shuffle within the class
        shuffled = list(class_samples)
        rng.shuffle(shuffled)

        # At least 1 test sample per class if there's more than 1 sample
        n_test = max(1, int(len(shuffled) * test_size))
        # If the class has very few samples, still keep at least 1 for training
        if len(shuffled) - n_test < 1:
            n_test = len(shuffled) - 1 if len(shuffled) > 1 else 0

        split_idx = len(shuffled) - n_test
        train.extend(shuffled[:split_idx])
        test.extend(shuffled[split_idx:])

    return train, test


def print_split_distribution(train_samples, test_samples, labels):
    """
    Print class distribution for train and test sets.

    Args:
        train_samples (list): Training samples.
        test_samples (list): Test samples.
        labels (list): Sorted class label names.
    """
    def count_by_label(samples):
        counts = {l: 0 for l in labels}
        for s in samples:
            counts[s["label"]] += 1
        return counts

    train_counts = count_by_label(train_samples)
    test_counts = count_by_label(test_samples)

    print(f"  {'Class':<12} {'Train':>8} {'Test':>8} {'Total':>8}")
    print(f"  " + "-" * 40)
    for label in labels:
        tc = train_counts[label]
        tsc = test_counts[label]
        print(f"  {label:<12} {tc:>8} {tsc:>8} {tc + tsc:>8}")
    print(f"  " + "-" * 40)
    print(f"  {'Total':<12} {len(train_samples):>8} {len(test_samples):>8} {len(train_samples) + len(test_samples):>8}")


def get_labels(samples):
    """
    Return sorted list of unique labels from samples.

    Args:
        samples (list): List of sample dicts.

    Returns:
        list[str]: Sorted unique labels.
    """
    return sorted(set(s["label"] for s in samples))
