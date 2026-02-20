![AAI-530 Course Banner](AAI-530_Course-Banner3.png)

# AgroCare.AI — IoT Plant Health Monitoring with Deep Learning

**Data Analytics and Internet of Things (AAI-530-IN1)**
University of San Diego
Triton — Group 2

| Team Member | Contribution |
|-------------|-------------|
| Sunil Prasath | Model architecture & training pipeline |
| Zohair Sabunwala | IoT system design & data generation |
| Balaji Rao | Inference system & evaluation |

---

## Overview

AgroCare.AI is an end-to-end IoT plant health monitoring system that classifies plant condition in real time using five onboard sensor channels. A microcontroller collects soil and environmental readings at 1 Hz; each 60-second window is passed through a trained 1D convolutional neural network to produce one of three actionable states:

| Label | Meaning | System Response |
|-------|---------|-----------------|
| **Normal** | Optimal soil and environmental conditions | No action |
| **Watering Required** | Soil moisture declining toward threshold | Activate irrigation relay |
| **Risky** | Extreme temperature, UV, or moisture deficit | Alert + irrigation |

The deep model (AgroCareNet) is compared against a logistic regression baseline that operates on handcrafted statistical features, demonstrating the advantage of learning directly from raw sensor waveforms.

---

## Repository Structure

```
agrocare/
├── models/
│   ├── agrocare_net.py          # 1D-CNN + dual attention architecture
│   └── baseline.py              # Logistic regression baseline
├── utils/
│   └── GenerateSynthData/
│       ├── generate_synthetic_dataset.py
│       └── data/agrocare_dataset.npz   # 9,999-sample training dataset
├── train.py                     # Full training pipeline
├── inference.py                 # CLI inference (interactive / single-shot / CSV)
├── checkpoints/                 # Trained weights and normalization stats
├── plots/                       # Training curves and confusion matrices
│   └── report/                  # Publication-ready figures
└── agrocare_model_walkthrough.ipynb  # End-to-end walkthrough notebook
```

---

## Model Architecture — AgroCareNet

AgroCareNet is a 1D convolutional network with a dual-attention fusion head trained to classify 60-step, 5-channel sensor windows.

```
Input  (batch, 60, 5)
  │
  ├─ CNN Block 1 ── Conv1d(5→32) × 2  +  MaxPool  →  (batch, 32, 30)
  ├─ CNN Block 2 ── Conv1d(32→64) × 2 +  MaxPool  →  (batch, 64, 15)
  └─ CNN Block 3 ── Conv1d(64→128) × 2             →  (batch, 128, 15)
            │
     ┌──────┴──────┐
     ▼             ▼
Temporal Attn   Channel Attn
(heads = 4)     (SE-Block, r=8)
(batch, 128)    (batch, 128)
     └──────┬──────┘
            ▼
     Concat  →  FC(256→64)  →  GELU  →  Dropout  →  FC(64→3)
            ▼
     Softmax  →  {Normal, Watering Required, Risky}

Total trainable parameters: 188,787
```

**Sensor channels (in order):**

| Ch | Sensor | Unit | Nominal Range |
|----|--------|------|---------------|
| 0 | Soil Moisture | % | 0 – 100 |
| 1 | Soil Temperature | °C | 0 – 60 |
| 2 | Ambient Temperature | °C | 10 – 60 |
| 3 | Relative Humidity | % | 10 – 90 |
| 4 | UV Index | — | 0 – 15 |

---

## Dataset

The training dataset was synthesized using a physics-informed generator that builds class-specific temporal patterns from reference sensor snapshots, applies Gaussian noise and time-series augmentations, then validates each window against labeling rules.

| Split | Samples | Normal | Watering Required | Risky |
|-------|--------:|-------:|------------------:|------:|
| Train (70%) | 6,999 | 2,329 | 1,843 | 2,827 |
| Validation (15%) | 1,500 | 499 | 395 | 606 |
| Test (15%) | 1,500 | 499 | 395 | 606 |
| **Total** | **9,999** | **3,327** | **2,633** | **4,039** |

Per-channel z-score normalization statistics (computed on the full dataset) are saved inside each checkpoint and automatically applied at inference time.

---

## Results

| Metric | Logistic Regression | AgroCareNet |
|--------|:-------------------:|:-----------:|
| Test Accuracy | 95.6% | **97.7%** |
| Macro avg F1 | 0.955 | **0.976** |
| Weighted avg F1 | 0.957 | **0.977** |
| Parameters | ~few hundred | 188,787 |

**Per-class F1:**

| Class | LR | AgroCareNet |
|-------|----|-------------|
| Normal | 0.997 | **0.999** |
| Watering Required | 0.923 | **0.959** |
| Risky | 0.945 | **0.972** |

---

## Quickstart

**Environment setup:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Training:**
```bash
# Train with defaults (50 epochs, batch 64, AdamW + ReduceLROnPlateau)
python train.py --dataset utils/GenerateSynthData/data/agrocare_dataset.npz --epochs 50

# Resume from a checkpoint
python train.py --dataset utils/GenerateSynthData/data/agrocare_dataset.npz \
                --resume checkpoints/best_model.pt --lr 5e-4
```
Outputs are written to `checkpoints/` (weights + normalization stats) and `plots/` (training curves, confusion matrix).

**Inference:**
```bash
# Interactive — enter sensor values at the prompt
python inference.py --model checkpoints/best_model.pt

# Single snapshot (soil_moisture soil_temp amb_temp humidity uv_index)
python inference.py --model checkpoints/best_model.pt --values 70 30 32 55 3.0

# CSV batch — columns: soil_moisture, soil_temp, amb_temp, amb_humidity, uv_index
python inference.py --model checkpoints/best_model.pt \
                    --csv data/readings.csv --output data/predictions.csv
```

**Walkthrough notebook:**
```bash
jupyter notebook agrocare_model_walkthrough.ipynb
```
The notebook reproduces data inspection, baseline training, AgroCareNet evaluation, all report figures, and an interactive inference demo from a single top-to-bottom run.

---

## Training Pipeline

| Component | Choice |
|-----------|--------|
| Optimizer | AdamW (weight decay 1e-4) |
| Scheduler | ReduceLROnPlateau on validation loss |
| Loss | CrossEntropyLoss with inverse-frequency class weights |
| Initial LR | 1e-3 |
| Batch size | 64 |
| Epochs | 50 |
| Random seed | 42 |
| Device | CUDA → MPS → CPU (auto-detected) |

Checkpoints include the full optimizer and scheduler state, normalization statistics, and model configuration, enabling exact training resumption and zero-configuration inference.

---

## License

This project was developed for academic purposes as part of the AAI-530 course at the University of San Diego. All synthetic data was generated programmatically; no real plant sensor data was used.
