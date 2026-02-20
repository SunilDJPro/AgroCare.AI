# AgroCare.AI

IoT-powered plant care with an AI model that reads sensor windows and decides if a plant is **Normal**, **Watering Required**, or **Risky**.

## What’s inside
- `models/agrocare_net.py`: 1D CNN backbone plus temporal and channel attention heads, fused for 3-class classification.
- `train.py`: Full training pipeline with class weighting, checkpoints, LR scheduling, and plots.
- `inference.py`: CLI for interactive entry, single-shot values, or CSV batch predictions using saved normalization stats.
- `checkpoints/`: Latest trained weights (best model and epoch snapshots).
- `plots/`: Training curves and confusion matrix from the latest run.

## Architecture at a glance
- **Input window**: 60 timesteps × 5 channels (soil moisture, soil temp, ambient temp, humidity, UV).
- **CNN backbone**: Three Conv1d blocks (Conv → BatchNorm → GELU → Pool) expanding channels to 128 while reducing time steps to 15.
- **Temporal attention**: Multi-head self-attention learns which timesteps matter most (e.g., recent drops in moisture).
- **Channel attention**: Squeeze-and-Excitation scales feature channels to emphasize influential sensors.
- **Fusion head**: Concatenate temporal + channel features → FC(256→64) with GELU + dropout → FC to 3 logits.
- **Outputs**: Class probabilities for `Normal`, `Watering Required`, `Risky` plus per-class metrics during evaluation.

## Data & preprocessing
- Expected dataset: `.npz` with `X` shaped `(N, 60, 5)` and integer labels `y` in `{0,1,2}`.
- Per-channel z-score normalization is computed on the dataset and stored in checkpoints for inference reuse.
- Train/val/test split defaults to 70/15/15 with deterministic seeding.
- Class weights are derived from label frequencies and applied to the CrossEntropy loss to handle imbalance.

## Quickstart (training)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train with defaults
python train.py --dataset data/agrocare_dataset.npz --epochs 50

# Resume or adjust hyperparameters
python train.py --dataset data/agrocare_dataset.npz --resume checkpoints/best_model.pt --lr 5e-4 --batch-size 64
```
Artifacts land in `checkpoints/` and `plots/` (training curves + confusion matrix).

## Quickstart (inference)
```bash
# Interactive: type sensor values when prompted
python inference.py --model checkpoints/best_model.pt

# Single-shot values (SM ST AT AH UV)
python inference.py --model checkpoints/best_model.pt --values 70 30 32 55 3.0

# CSV batch prediction
# Columns: soil_moisture, soil_temp, amb_temp, amb_humidity, uv_index
python inference.py --model checkpoints/best_model.pt --csv data/test.csv --output data/test_predictions.csv
```
Inference automatically applies the saved normalization stats from the checkpoint.

## Training pipeline highlights
- **Optimizer**: AdamW with weight decay; **Scheduler**: ReduceLROnPlateau on validation loss.
- **Checkpointing**: Every `--checkpoint-every` epochs plus best-model overwrite; includes optimizer/scheduler states and normalization stats.
- **Metrics & plots**: Tracks train/val loss and accuracy; saves `plots/training_curves.png` and `plots/confusion_matrix.png` after training.
- **Device selection**: CUDA → MPS → CPU fallback.

## Notes
- Keep `.npz` datasets small in the repo; regenerate larger sets as needed.
- Checkpoint files are large; avoid duplicating them in commits.
