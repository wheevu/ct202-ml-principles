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

What, you think Pokemons aren't cool? Don't even start, bro.

K-nearest neighbors image classifier that guesses a Pokémon's broad type group from sprite features
(average color, color histograms, downsampled pixels) using Euclidean distance. Caveman style.

> A KNN classifier can only compare samples through the features we give it.
> If the features capture mostly color, the model learns color similarity, not true Pokémon lore; inquire the Pokedex for that.

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

**Dataset:** create `data/raw/{fire,water,grass,electric}/` with sprite PNGs/JPGs (~60 per class).
Sources: [PokémonDB](https://pokemondb.net/sprites) or [Veekun](https://veekun.com/dex/downloads).

**Limitations:** as stated, color-based features confuse similar-toned types (water vs. electric).
No shape/lore understanding. Small datasets limit generalization.
Transparent sprites get composited onto white, which shifts color stats.

### Iteration notes

**First pass** — I started with 12 sprites per class (48 total), a simple random shuffle split, and no baselines. (bare with me, it's a learning process)

Results from a k-sweep (k = 1, 3, 5, 7):

```
Feature            k=1       k=3       k=5       k=7
----------------------------------------------------
avg_rgb         50.0%    50.0%    37.5%    62.5%
histogram       25.0%    37.5%    12.5%    25.0%
pixels          37.5%    50.0%    12.5%    62.5%
combined        62.5%    25.0%    37.5%    50.0%
```

Everything was all over the place. `avg_rgb` jumped from 37.5% to 62.5% just by changing k.
`combined` went from 62.5% (k=1) to 25.0% (k=3). That's not it bro.

**The issue** — with only 2 test samples per class after splitting, a single correct/incorrect flip swung accuracy by 12.5 points.
A lucky split made a bad feature method look good and vice versa. Red flag.
Without baselines, there was no reference point for whether KNN was actually learning anything (not the point?). In comes the improved version:

**Second pass** — three changes:
- Stratified splitting so each class keeps proportional representation in train and test (Disney, hit me up)
- Majority-class and random baselines printed alongside KNN results
- Data pool bumped to 60 sprites per class (240 total, 48 test samples) (ran out of disk space)

Results from the same k-sweep:

```
Feature            k=1       k=3       k=5       k=7
----------------------------------------------------
avg_rgb         45.8%    52.1%    50.0%    54.2%
histogram       43.8%    50.0%    56.2%    54.2%
pixels          33.3%    41.7%    45.8%    50.0%
combined        43.8%    52.1%    54.2%    56.2%
```

Gap between `min` and `max` for each method is now ~10 points instead of ~40.
`Histogram` at k=5 scored 56.2% — and every method beat the 25% majority baseline. **Now** we're talking.

**What changed**

- `Accuracy` settled into a consistent 50–56% range across all methods.
- The confusion matrix stabilized: `electric`↔`water` confusion (both blue-toned) and `fire`↔`electric` confusion (warm colors) showed up as real patterns.
avg_rgb and histogram consistently beat the 25% majority baseline, confirming the features capture *some* signal.
- The baselines (from the balanced test set) made it easy to see that "always predict electric" would score 25%, so KNN doing 56% is meaningful even if it's not gonna earn me a "I'm proud of you, son" :(

**Source:** everything in `src/` — KNN, train/test split, evaluation, visualizations all from scratch.
Only helpers are numpy, Pillow, matplotlib.

</details>
