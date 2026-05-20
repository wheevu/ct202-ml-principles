# CT202 Machine Learning Lab

This repo tracks my learning artifacts for CT202 — Principles of Machine Learning.
Each week contains a small build that turns a course concept into code, notes, and visual outputs. All that.

## Weekly builds

| Week | Project | Concepts |
|------|---------|----------|
| Week 02 | PokéKNN | classification, image features, Euclidean distance, KNN, evaluation |

<details>
<summary><strong>Week 02 — PokéKNN</strong></summary>
<br>

K-nearest neighbors image classifier that guesses a Pokémon's broad type group from sprite features
(average color, color histograms, downsampled pixels) using Euclidean distance.

> A KNN classifier can only compare samples through the features we give it.
> If the features capture mostly color, the model learns color similarity, not true Pokémon lore.

| Feature method | Description | Length |
|---|---|---|
| `avg_rgb` | Mean R, G, B (normalized) | 3 |
| `histogram` | 8-bin histogram per RGB channel | 24 |
| `pixels` | Downsampled to 16×16, flattened | 768 |
| `combined` | avg_rgb + histogram | 27 |

**Run it:**

```bash
cd weeks/week02-pokeknn
python run_experiment.py --k 3 --feature all
```

All options: `--data-dir`, `--output-dir`, `--k`, `--image-size`, `--test-size`, `--seed`, `--feature`.

Outputs go to `outputs/`: sample grid, confusion matrix, nearest-neighbor visualization,
plus per-method accuracy in the console.

**Dataset:** create `data/raw/{fire,water,grass,electric}/` with sprite PNGs/JPGs (~10 per class).
Sources: [PokémonDB](https://pokemondb.net/sprites) or [Veekun](https://veekun.com/dex/downloads).

**Limitations:** color-based features confuse similar-toned types (water vs. electric).
No shape/lore understanding. Small datasets limit generalization.
Transparent sprites get composited onto white, which shifts color stats.

**Source:** everything in `src/` — KNN, train/test split, evaluation, visualizations all from scratch.
Only helpers are numpy, Pillow, matplotlib.

</details>
