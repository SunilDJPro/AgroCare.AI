"""
AgroCare IoT — Inference & Interactive Prediction
===================================================
Load a trained AgroCareNet model and predict plant health status from
manually entered sensor values or from a CSV file.

Modes:
    1. Interactive CLI  — Enter 5 sensor values, get instant prediction
    2. CSV Batch        — Predict for all rows in a CSV file
    3. Single-shot      — Pass values as CLI arguments

Usage:
    python inference.py --model checkpoints/best_model.pt              # Interactive mode
    python inference.py --model checkpoints/best_model.pt --csv data/test.csv  # Batch mode
    python inference.py --model checkpoints/best_model.pt --values 70 30 32 55 3.0  # Single-shot

Notes:
    - For interactive/single-shot mode, the 5 sensor values are repeated across
      60 timesteps to form a static window (matching the training input shape).
    - The model checkpoint contains normalization stats (mean/std per channel)
      which are automatically applied to raw inputs.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.agrocare_net import AgroCareNet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Normal", "Watering Required", "Risky"]
CLASS_ACTIONS = [
    "✓ No action needed. Plant is healthy.",
    "💧 Watering cycle triggered (5 seconds via relay pump).",
    "⚠ ALERT: Manual care required! Check plant immediately.",
]
CLASS_COLORS = ["\033[92m", "\033[93m", "\033[91m"]  # green, yellow, red
RESET_COLOR = "\033[0m"

SENSOR_NAMES = [
    "Soil Moisture (%)",
    "Soil Temperature (°C)",
    "Ambient Temperature (°C)",
    "Ambient Humidity (%)",
    "UV Index",
]
SENSOR_RANGES = [
    (0, 100),    # Soil Moisture
    (-40, 80),   # Soil Temperature
    (-40, 80),   # Ambient Temperature
    (0, 100),    # Ambient Humidity
    (0, 20),     # UV Index
]

WINDOW_SIZE = 60
NUM_CHANNELS = 5


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model and normalization stats from checkpoint.

    Returns:
        model: AgroCareNet in eval mode
        norm_stats: dict with 'means' and 'stds' tensors
    """
    print(f"[MODEL] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model from saved config
    config = checkpoint.get("model_config", {})
    model = AgroCareNet(
        in_channels=config.get("in_channels", 5),
        num_classes=config.get("num_classes", 3),
        feature_dim=config.get("feature_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load normalization stats
    norm_stats = {
        "means": checkpoint["norm_means"].to(device),
        "stds": checkpoint["norm_stds"].to(device),
    }

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("best_val_loss", "?")
    print(f"         Epoch: {epoch}, Best val_loss: {val_loss}")
    print(f"         Parameters: {model.count_parameters():,}")
    print(f"         Norm means: {norm_stats['means'].cpu().numpy().round(2)}")
    print(f"         Norm stds:  {norm_stats['stds'].cpu().numpy().round(2)}")

    return model, norm_stats


# ---------------------------------------------------------------------------
# Prediction Core
# ---------------------------------------------------------------------------


def predict_from_snapshot(
    model: AgroCareNet,
    norm_stats: dict,
    sensor_values: List[float],
    device: torch.device,
) -> dict:
    """Predict plant health from a single 5-value sensor snapshot.

    The snapshot is repeated 60 times to create a static window matching
    the model's expected input shape of (1, 60, 5).

    Args:
        model: Loaded AgroCareNet model (in eval mode)
        norm_stats: Dict with 'means' and 'stds' tensors for normalization
        sensor_values: List of 5 floats [soil_moist, soil_temp, amb_temp, amb_humid, uv]
        device: Torch device

    Returns:
        Dict with prediction details
    """
    # Build static window: (1, 60, 5)
    values = torch.tensor(sensor_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 5)
    window = values.repeat(1, WINDOW_SIZE, 1).to(device)  # (1, 60, 5)

    # Normalize using training stats
    window = (window - norm_stats["means"]) / norm_stats["stds"]

    # Predict
    with torch.no_grad():
        logits = model(window)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (3,)
        class_idx = probs.argmax().item()

    return {
        "class_idx": class_idx,
        "class_name": CLASS_NAMES[class_idx],
        "action": CLASS_ACTIONS[class_idx],
        "confidence": probs[class_idx].item(),
        "probabilities": {
            CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))
        },
        "input_values": sensor_values,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def display_prediction(result: dict):
    """Pretty-print a prediction result to the console."""
    cls = result["class_idx"]
    color = CLASS_COLORS[cls]

    print(f"\n  {'─' * 54}")
    print(f"  │ Input Sensors:                                    │")
    for i, name in enumerate(SENSOR_NAMES):
        val = result["input_values"][i]
        print(f"  │   {name:<30} {val:>8.2f}        │")

    print(f"  ├{'─' * 54}┤")
    print(f"  │ {color}Prediction: {result['class_name']:>20}{RESET_COLOR}              │")
    print(f"  │ Confidence: {result['confidence']*100:>19.1f}%              │")
    print(f"  ├{'─' * 54}┤")
    print(f"  │ Class Probabilities:                              │")
    for cls_name, prob in result["probabilities"].items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  │   {cls_name:<20} {bar} {prob*100:5.1f}% │")

    print(f"  ├{'─' * 54}┤")
    print(f"  │ Action: {result['action']:<45}│")
    print(f"  {'─' * 54}")


# ---------------------------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------------------------


def interactive_mode(model: AgroCareNet, norm_stats: dict, device: torch.device):
    """Run an interactive loop where the user enters sensor values manually."""
    print(f"\n{'='*60}")
    print("  AgroCare IoT — Interactive Prediction Mode")
    print(f"{'='*60}")
    print("  Enter 5 sensor values when prompted, or type 'quit' to exit.")
    print("  You can enter all 5 values on one line (comma or space separated)")
    print("  or enter them one at a time.\n")

    while True:
        print(f"\n  {'─'*40}")
        raw = input("  Enter sensor values (or 'quit'): ").strip()

        if raw.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 🌱")
            break

        # Try parsing all 5 from one line
        values = None
        try:
            # Support comma, space, or mixed separators
            parts = raw.replace(",", " ").split()
            if len(parts) == 5:
                values = [float(p) for p in parts]
        except ValueError:
            pass

        # If single-line didn't work, prompt individually
        if values is None:
            values = []
            try:
                for i, name in enumerate(SENSOR_NAMES):
                    lo, hi = SENSOR_RANGES[i]
                    while True:
                        val_str = input(f"    {name} [{lo}–{hi}]: ").strip()
                        if val_str.lower() in ("quit", "exit", "q"):
                            print("\n  Goodbye! 🌱")
                            return
                        try:
                            val = float(val_str)
                            if lo <= val <= hi:
                                values.append(val)
                                break
                            else:
                                print(f"      ⚠ Value out of range [{lo}, {hi}]. Try again.")
                        except ValueError:
                            print(f"      ⚠ Invalid number. Try again.")
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye! 🌱")
                return

        # Validate ranges
        valid = True
        for i, v in enumerate(values):
            lo, hi = SENSOR_RANGES[i]
            if not (lo <= v <= hi):
                print(f"  ⚠ {SENSOR_NAMES[i]} = {v} is out of range [{lo}, {hi}]")
                valid = False
        if not valid:
            print("  Skipping prediction. Please try again.")
            continue

        # Predict
        result = predict_from_snapshot(model, norm_stats, values, device)
        display_prediction(result)


# ---------------------------------------------------------------------------
# CSV Batch Mode
# ---------------------------------------------------------------------------


def batch_predict_csv(
    model: AgroCareNet,
    norm_stats: dict,
    csv_path: str,
    device: torch.device,
    output_path: Optional[str] = None,
):
    """Predict for every row in a CSV file and optionally save results."""
    import pandas as pd

    print(f"\n[BATCH] Processing {csv_path}")
    df = pd.read_csv(csv_path)

    sensor_cols = ["soil_moisture", "soil_temp", "amb_temp", "amb_humidity", "uv_index"]
    if not all(c in df.columns for c in sensor_cols):
        print(f"[WARN] Columns don't match. Using first 5 columns as sensor values.")
        df.columns = sensor_cols[: len(df.columns)] if len(df.columns) >= 5 else df.columns
        sensor_cols = list(df.columns[:5])

    predictions = []
    for idx, row in df.iterrows():
        values = [float(row[c]) for c in sensor_cols]
        result = predict_from_snapshot(model, norm_stats, values, device)
        predictions.append(result)

        # Print compact result
        cls_color = CLASS_COLORS[result["class_idx"]]
        print(
            f"  Row {idx+1:>4d} │ "
            f"SM={values[0]:5.1f}  ST={values[1]:5.1f}  AT={values[2]:5.1f}  "
            f"AH={values[3]:5.1f}  UV={values[4]:4.1f} │ "
            f"{cls_color}{result['class_name']:>20}{RESET_COLOR} "
            f"({result['confidence']*100:.1f}%)"
        )

    # Save results
    if output_path:
        df_out = df.copy()
        df_out["predicted_class"] = [p["class_name"] for p in predictions]
        df_out["confidence"] = [round(p["confidence"], 4) for p in predictions]
        for cls_name in CLASS_NAMES:
            df_out[f"prob_{cls_name.lower().replace(' ', '_')}"] = [
                p["probabilities"][cls_name] for p in predictions
            ]
        df_out.to_csv(output_path, index=False)
        print(f"\n[BATCH] Results saved to {output_path}")

    # Summary
    print(f"\n  Summary: {len(predictions)} predictions")
    for cls in range(len(CLASS_NAMES)):
        count = sum(1 for p in predictions if p["class_idx"] == cls)
        print(f"    {CLASS_NAMES[cls]:>20}: {count}")

    return predictions


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="AgroCare IoT — Inference & Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--values", nargs=5, type=float, default=None,
        metavar=("SOIL_MOIST", "SOIL_TEMP", "AMB_TEMP", "AMB_HUMID", "UV"),
        help="Single-shot mode: 5 sensor values for one prediction",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Batch mode: CSV file with sensor readings to predict",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path for batch predictions (optional)",
    )

    args = parser.parse_args()

    # Device (inference can typically run on CPU fine)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model, norm_stats = load_model(args.model, device)

    # Route to appropriate mode
    if args.values is not None:
        # Single-shot prediction
        result = predict_from_snapshot(model, norm_stats, args.values, device)
        display_prediction(result)

    elif args.csv is not None:
        # Batch CSV prediction
        output = args.output or args.csv.replace(".csv", "_predictions.csv")
        batch_predict_csv(model, norm_stats, args.csv, device, output_path=output)

    else:
        # Interactive mode (default)
        interactive_mode(model, norm_stats, device)


if __name__ == "__main__":
    main()