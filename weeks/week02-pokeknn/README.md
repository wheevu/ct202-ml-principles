# Week 02 — PokéKNN

Source code and experiment runner for the PokéKNN project.

📖 **Full documentation is in the [main README](../../README.md#weekly-builds).**

```bash
# Quick start
cd weeks/week02-pokeknn
python run_experiment.py --help
```

See the main README for dataset setup, feature explanations, and expected outputs.

### Evaluation notes

- **Stratified splitting** — the train/test split preserves class proportions so every class appears in both train and test when possible. This avoids accidentally leaving a class out of the test set (common with small datasets and random shuffling).

- **Baseline comparison** — majority-class and random baselines are printed alongside KNN results. If KNN doesn't beat "always predict the most common class", the features aren't capturing useful signal.

- **Small dataset caveat** — with ~12 images per class, raw accuracy can shift noticeably depending on which samples end up in the test set. Stratified splitting helps, but results should be read as rough indicators, not definitive benchmarks.

- **Why this is still useful** — because the dataset is small, raw accuracy can change a lot depending on the split. Stratified splitting makes the test set more balanced, and baseline classifiers show whether KNN is learning useful patterns beyond simple guessing. The value is in understanding the full pipeline and seeing how each design choice (features, k, evaluation method) affects results.
