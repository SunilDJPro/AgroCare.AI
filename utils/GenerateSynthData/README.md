# AgroCare IoT — Synthetic Dataset Generator

Generates temporally-coherent sliding window datasets **(60 timesteps × 5 sensor channels)** for training the 1D-CNN + Attention plant health classifier.

## Quick Start

```bash
pip install -r requirements.txt

# Generate 10,000 samples from built-in reference snapshots
python utils/generate_synthetic_dataset.py

# Generate from a small CSV (10-15 snapshot rows) + built-in references
python utils/generate_synthetic_dataset.py --csv data/my_snapshots.csv

# Generate from a large CSV (≥60 rows of continuous time-series)
python utils/generate_synthetic_dataset.py --csv data/sensor_log.csv

# Custom size, seed, and output paths
python utils/generate_synthetic_dataset.py --total 15000 --seed 123 --output-dir data/v2 --plot-dir plots/v2
```

## How It Works

### Input Modes

| CSV Size | Behavior |
|---|---|
| **No CSV** | Generates entirely from 3 built-in reference snapshots per class |
| **Small CSV** (< 60 rows) | Each row is treated as a **reference snapshot** — a center point from which full 60-step temporal windows are generated. Merged with built-in references for maximum diversity. |
| **Large CSV** (≥ 60 rows) | Treated as continuous time-series. Segmented into sliding windows directly, then remaining quota filled synthetically. |
| **Multiple CSVs** | `--csv file1.csv file2.csv` — each file is processed independently (can mix small + large). |

### Generation Pipeline

```
Reference Snapshots ──→ Perturb Centers ──→ Temporal Pattern Engine ──→ Base Windows
                                                                            │
                                          ┌─────────────────────────────────┘
                                          ▼
                                    Augmentation Pool
                                    ├─ SMOTE interpolation (30%)
                                    ├─ Time warping (15%)
                                    ├─ Magnitude warping (15%)
                                    ├─ Gaussian jitter (15%)
                                    └─ Combined aug (25%)
                                          │
                                          ▼
                                  Validation & Relabeling ──→ Final Dataset (.npz)
```

1. **Base generation** — Each reference snapshot is randomly perturbed within class-safe bounds, then the Temporal Pattern Engine generates a realistic 60-step window with class-specific behaviors (stable signals for Normal, declining moisture for Watering Required, extreme values/combinations for Risky).

2. **Augmentation** — Base windows are expanded to the target count using SMOTE-style time-series interpolation, time warping, magnitude warping, and jitter.

3. **Validation** — Every generated window is checked against threshold rules. Samples that don't match their intended class (e.g., a "Watering Required" sample that drifted into Risky territory) are relabeled to the correct class.

### Temporal Patterns by Class

| Class | Soil Moisture | Temperature | UV | Humidity |
|---|---|---|---|---|
| **Normal** | Stable (35-90%) | Stable (18-34°C) | Diurnal, low | Moderate |
| **Watering Required** | Declining trend | Slightly elevated | Moderate | Variable |
| **Risky** | Very low / dropping | High (>35°C) | High (>7.0) | Often low |

### CSV Format

```csv
soil_moisture,soil_temp,amb_temp,amb_humidity,uv_index,label
70,30,32,55,3.0,0
20,33,33,65,4.0,1
10,38,40,35,7.0,2
```

- **Label values:** `0` = Normal, `1` = Watering Required, `2` = Risky
- Column names are flexible — if headers don't match, positional mapping is used (first 5 = sensors, last = label).

## CLI Reference

```
usage: generate_synthetic_dataset.py [-h] [--total N] [--csv [FILE ...]]
                                     [--output-dir DIR] [--plot-dir DIR]
                                     [--seed S] [--no-validate] [--no-plots]

Options:
  --total N          Total samples to generate (default: 10000)
  --csv FILE [...]   CSV file(s) with real-world data or reference snapshots
  --output-dir DIR   Output directory for .npz dataset (default: data/)
  --plot-dir DIR     Output directory for analytics plots (default: plots/)
  --seed S           Random seed for reproducibility (default: 42)
  --no-validate      Skip validation/relabeling step
  --no-plots         Skip analytics plot generation
```

## Outputs

| File | Description |
|---|---|
| `data/agrocare_dataset.npz` | Dataset with `X` (N, 60, 5) float32 and `y` (N,) int64 arrays |
| `data/dataset_metadata.json` | Generation stats, class distribution, validation info |
| `plots/01_class_distribution.png` | Bar chart of class balance |
| `plots/02_sensor_distributions.png` | Violin plots of per-sensor distributions by class |
| `plots/03_temporal_examples.png` | Example 60-step windows for each class × sensor |
| `plots/04_sensor_means_by_class.png` | Grouped bar chart of mean sensor values ±1σ |
| `plots/05_correlation_matrices.png` | Per-class inter-sensor correlation heatmaps |
| `plots/06_pca_projection.png` | 2D PCA projection of the full dataset |

## Loading the Dataset

```python
import numpy as np

data = np.load("data/agrocare_dataset.npz")
X = data["X"]  # shape: (N, 60, 5)
y = data["y"]  # shape: (N,)

print(f"Samples: {len(X)}, Window: {X.shape[1]}×{X.shape[2]}")
```

## Thresholds & Class Rules

These are the decision boundaries enforced by the validator:

- **Watering Required:** Soil moisture < 30% (mean across window)
- **Risky — single extremes:** Soil moisture < 10%, any temp > 40°C, UV > 8.5
- **Risky — combinations:** Moisture < 20% AND temp > 35°C, or temp > 35°C AND humidity < 25%
- **MCU safety:** The MCU triggers watering independently if moisture < 10% (hardware-level fallback, not modeled here)

Thresholds can be modified by editing the `Thresholds` dataclass in the script.