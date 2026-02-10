"""
AgroCare IoT - Synthetic Dataset Generator
============================================
Generates temporally-coherent sliding window datasets (60 timesteps × 5 channels)
for training a 1D-CNN + Attention model on plant health classification.

Classes:
    0 = Normal          - Plant is healthy, no action needed
    1 = Watering Required - Soil moisture low, automated watering triggered
    2 = Risky           - Extreme conditions or dangerous combinations, manual care needed

Sensors (channels):
    0 = Soil Moisture (%)
    1 = Soil Temperature (°C)
    2 = Ambient Temperature (°C)
    3 = Ambient Humidity (%)
    4 = UV Index

Generation Pipeline:
    1. Base window generation from reference samples with class-specific temporal patterns
    2. Gaussian jitter + correlated noise injection
    3. SMOTE-style time-series interpolation for diversity
    4. Time warping & magnitude warping augmentation
    5. Validation against class rules (reject mislabeled windows)
    6. Analytics & visualization

Usage:
    python utils/generate_synthetic_dataset.py                          # From references only
    python utils/generate_synthetic_dataset.py --csv data/real_data.csv # Augment real data
    python utils/generate_synthetic_dataset.py --total 15000            # Custom dataset size
"""

import os
import sys
import argparse
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

# Sensor names for labeling plots / logs
SENSOR_NAMES = [
    "Soil Moisture (%)",
    "Soil Temperature (°C)",
    "Ambient Temperature (°C)",
    "Ambient Humidity (%)",
    "UV Index",
]
SENSOR_SHORT = ["soil_moist", "soil_temp", "amb_temp", "amb_humid", "uv_index"]

CLASS_NAMES = ["Normal", "Watering Required", "Risky"]
NUM_CLASSES = 3

# Sliding window parameters (from proposal)
WINDOW_SIZE = 60  # 60 timesteps
SAMPLE_RATE_HZ = 0.5  # 1 reading every 2 seconds  → 120 s window
NUM_CHANNELS = 5


@dataclass
class SensorConfig:
    """Physical limits and operational ranges for each sensor."""

    # Absolute hardware limits [min, max]
    abs_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "soil_moist": (0.0, 100.0),
            "soil_temp": (-40.0, 80.0),
            "amb_temp": (-40.0, 80.0),
            "amb_humid": (0.0, 100.0),
            "uv_index": (0.0, 20.0),
        }
    )

    # Realistic indoor-plant operating ranges (for generation centering)
    realistic_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "soil_moist": (5.0, 95.0),
            "soil_temp": (15.0, 50.0),
            "amb_temp": (18.0, 48.0),
            "amb_humid": (15.0, 90.0),
            "uv_index": (0.0, 12.0),
        }
    )


@dataclass
class Thresholds:
    """Decision boundaries between classes."""

    soil_moisture_low: float = 30.0  # below → watering concern
    soil_moisture_critical: float = 10.0  # MCU safety trigger
    temp_stress: float = 35.0  # above → stress zone
    uv_dangerous: float = 7.0  # above → dangerous exposure

    # Combination thresholds for Risky
    combo_moisture_low: float = 20.0  # moisture < this AND ...
    combo_temp_high: float = 40.0  # ... temp > this → risky
    combo_humidity_low: float = 25.0  # very low humidity + high temp → risky


@dataclass
class ReferenceSnapshots:
    """Human-provided reference sensor values per class.
    Format: [soil_moist, soil_temp, amb_temp, amb_humid, uv_index]
    """

    normal: List[List[float]] = field(
        default_factory=lambda: [
            [70.0, 30.0, 32.0, 55.0, 3.0],
            [60.0, 26.0, 28.0, 60.0, 2.0],
            [80.0, 24.0, 25.0, 50.0, 1.5],
        ]
    )
    watering_required: List[List[float]] = field(
        default_factory=lambda: [
            [20.0, 33.0, 33.0, 65.0, 4.0],
            [25.0, 31.0, 34.0, 50.0, 3.5],
            [15.0, 29.0, 30.0, 55.0, 5.0],
        ]
    )
    risky: List[List[float]] = field(
        default_factory=lambda: [
            [10.0, 38.0, 40.0, 35.0, 7.0],  # combo: low moisture + high temp
            [8.0, 35.0, 42.0, 20.0, 9.0],  # extreme UV + low humidity
            [12.0, 40.0, 43.0, 30.0, 8.5],  # multi-extreme
        ]
    )


# ---------------------------------------------------------------------------
# Temporal Pattern Generators
# ---------------------------------------------------------------------------


class TemporalPatternEngine:
    """Generates realistic 60-step temporal patterns for each sensor channel,
    conditioned on the target class."""

    def __init__(
        self,
        sensor_cfg: SensorConfig,
        thresholds: Thresholds,
        rng: np.random.Generator,
    ):
        self.cfg = sensor_cfg
        self.th = thresholds
        self.rng = rng

    def _smooth_noise(
        self, length: int, sigma: float = 3.0, scale: float = 1.0
    ) -> np.ndarray:
        """Generate temporally smooth noise using Gaussian-filtered white noise."""
        raw = self.rng.normal(0, scale, size=length)
        return gaussian_filter1d(raw, sigma=sigma)

    def _random_walk(
        self,
        center: float,
        length: int,
        drift: float = 0.0,
        volatility: float = 0.3,
        smoothness: float = 3.0,
    ) -> np.ndarray:
        """Brownian-motion-style walk around a center value with optional drift."""
        increments = self.rng.normal(drift, volatility, size=length)
        walk = np.cumsum(increments)
        # Re-center around the target center
        walk = walk - walk.mean() + center
        # Smooth for temporal coherence
        walk = gaussian_filter1d(walk, sigma=smoothness)
        # Apply drift as a linear trend overlaid
        if abs(drift) > 0.01:
            trend = np.linspace(0, drift * length * 0.5, length)
            walk = walk + trend
        return walk

    def _diurnal_uv(
        self, base_uv: float, length: int, phase_offset: float = 0.0
    ) -> np.ndarray:
        """Simulate UV with a sinusoidal diurnal component.
        Over a 120-second window this is subtle, but captures the concept of
        time-of-day variation when composing longer synthetic sequences."""
        t = np.linspace(0, 2 * np.pi, length) + phase_offset
        # Small sinusoidal modulation ± 15% of base
        modulation = 0.15 * base_uv * np.sin(t * self.rng.uniform(0.5, 2.0))
        noise = self._smooth_noise(length, sigma=4.0, scale=base_uv * 0.05)
        return np.clip(base_uv + modulation + noise, 0, 20)

    def _clamp_channel(self, values: np.ndarray, channel_idx: int) -> np.ndarray:
        """Clip values to the absolute sensor range."""
        key = SENSOR_SHORT[channel_idx]
        lo, hi = self.cfg.abs_range[key]
        return np.clip(values, lo, hi)

    def generate_window(
        self,
        center_values: np.ndarray,
        class_label: int,
        scenario_variant: Optional[str] = None,
    ) -> np.ndarray:
        """Generate a single (60, 5) temporal window for a given class.

        Args:
            center_values: (5,) array of sensor center values for this window.
            class_label: 0=Normal, 1=Watering Required, 2=Risky.
            scenario_variant: Optional sub-scenario string for class-specific behavior.

        Returns:
            window: (60, 5) ndarray.
        """
        L = WINDOW_SIZE
        window = np.zeros((L, NUM_CHANNELS))

        if class_label == 0:
            window = self._gen_normal(center_values, L)
        elif class_label == 1:
            window = self._gen_watering(center_values, L)
        elif class_label == 2:
            window = self._gen_risky(center_values, L, scenario_variant)

        # Clamp all channels
        for ch in range(NUM_CHANNELS):
            window[:, ch] = self._clamp_channel(window[:, ch], ch)

        return window

    def _gen_normal(self, center: np.ndarray, L: int) -> np.ndarray:
        """Normal: stable readings with minor fluctuations, no concerning trends."""
        w = np.zeros((L, NUM_CHANNELS))

        # Soil moisture — stable, slight random walk
        w[:, 0] = self._random_walk(center[0], L, drift=0.0, volatility=0.3, smoothness=4.0)

        # Soil temperature — very stable
        w[:, 1] = self._random_walk(center[1], L, drift=0.0, volatility=0.1, smoothness=5.0)

        # Ambient temperature — slight variation
        w[:, 2] = self._random_walk(center[2], L, drift=0.0, volatility=0.15, smoothness=4.0)

        # Ambient humidity — moderate variation
        w[:, 3] = self._random_walk(center[3], L, drift=0.0, volatility=0.25, smoothness=3.5)

        # UV — diurnal-like pattern
        w[:, 4] = self._diurnal_uv(center[4], L, phase_offset=self.rng.uniform(0, 2 * np.pi))

        return w

    def _gen_watering(self, center: np.ndarray, L: int) -> np.ndarray:
        """Watering Required: soil moisture declining trend, other params moderate."""
        w = np.zeros((L, NUM_CHANNELS))

        # Soil moisture — DECLINING trend (key signature of this class)
        decline_rate = self.rng.uniform(-0.15, -0.05)  # negative drift
        w[:, 0] = self._random_walk(
            center[0], L, drift=decline_rate, volatility=0.4, smoothness=3.0
        )

        # Soil temperature — might be slightly elevated (dry soil heats faster)
        slight_rise = self.rng.uniform(0.0, 0.03)
        w[:, 1] = self._random_walk(center[1], L, drift=slight_rise, volatility=0.15, smoothness=4.0)

        # Ambient temperature — possibly warm
        w[:, 2] = self._random_walk(center[2], L, drift=0.0, volatility=0.2, smoothness=4.0)

        # Ambient humidity — can vary
        w[:, 3] = self._random_walk(center[3], L, drift=0.0, volatility=0.3, smoothness=3.5)

        # UV — moderate
        w[:, 4] = self._diurnal_uv(center[4], L, phase_offset=self.rng.uniform(0, 2 * np.pi))

        return w

    def _gen_risky(
        self, center: np.ndarray, L: int, variant: Optional[str] = None
    ) -> np.ndarray:
        """Risky: extreme single values or dangerous multi-sensor combinations."""
        w = np.zeros((L, NUM_CHANNELS))

        if variant is None:
            variant = self.rng.choice(
                [
                    "extreme_moisture",
                    "extreme_temp",
                    "extreme_uv",
                    "combo_moisture_temp",
                    "combo_temp_humidity",
                    "multi_extreme",
                ]
            )

        if variant == "extreme_moisture":
            # Very low moisture, other params may be moderate
            w[:, 0] = self._random_walk(center[0], L, drift=-0.1, volatility=0.3, smoothness=3.0)
            w[:, 1] = self._random_walk(center[1], L, drift=0.0, volatility=0.15, smoothness=4.0)
            w[:, 2] = self._random_walk(center[2], L, drift=0.0, volatility=0.2, smoothness=4.0)
            w[:, 3] = self._random_walk(center[3], L, drift=0.0, volatility=0.3, smoothness=3.5)
            w[:, 4] = self._diurnal_uv(center[4], L)

        elif variant == "extreme_temp":
            # Very high soil + ambient temperature
            w[:, 0] = self._random_walk(center[0], L, drift=-0.05, volatility=0.3, smoothness=3.0)
            w[:, 1] = self._random_walk(center[1], L, drift=0.05, volatility=0.2, smoothness=3.0)
            w[:, 2] = self._random_walk(center[2], L, drift=0.05, volatility=0.2, smoothness=3.0)
            w[:, 3] = self._random_walk(center[3], L, drift=-0.03, volatility=0.25, smoothness=3.5)
            w[:, 4] = self._diurnal_uv(center[4], L)

        elif variant == "extreme_uv":
            # Dangerously high UV exposure
            w[:, 0] = self._random_walk(center[0], L, drift=-0.03, volatility=0.3, smoothness=3.0)
            w[:, 1] = self._random_walk(center[1], L, drift=0.02, volatility=0.15, smoothness=4.0)
            w[:, 2] = self._random_walk(center[2], L, drift=0.02, volatility=0.2, smoothness=4.0)
            w[:, 3] = self._random_walk(center[3], L, drift=-0.02, volatility=0.25, smoothness=3.5)
            w[:, 4] = self._diurnal_uv(
                max(center[4], self.th.uv_dangerous + self.rng.uniform(0.5, 3.0)), L
            )

        elif variant == "combo_moisture_temp":
            # Low moisture + high temperature (the key combination)
            w[:, 0] = self._random_walk(
                min(center[0], self.th.combo_moisture_low - self.rng.uniform(0, 5)),
                L, drift=-0.1, volatility=0.3, smoothness=3.0,
            )
            w[:, 1] = self._random_walk(
                max(center[1], self.th.combo_temp_high + self.rng.uniform(0, 3)),
                L, drift=0.03, volatility=0.2, smoothness=3.0,
            )
            w[:, 2] = self._random_walk(
                max(center[2], self.th.combo_temp_high + self.rng.uniform(0, 5)),
                L, drift=0.03, volatility=0.2, smoothness=3.0,
            )
            w[:, 3] = self._random_walk(center[3], L, drift=0.0, volatility=0.3, smoothness=3.5)
            w[:, 4] = self._diurnal_uv(center[4], L)

        elif variant == "combo_temp_humidity":
            # High temp + very low humidity (heat stress)
            w[:, 0] = self._random_walk(center[0], L, drift=-0.05, volatility=0.3, smoothness=3.0)
            w[:, 1] = self._random_walk(
                max(center[1], self.th.temp_stress + self.rng.uniform(1, 5)),
                L, drift=0.03, volatility=0.2, smoothness=3.0,
            )
            w[:, 2] = self._random_walk(
                max(center[2], self.th.temp_stress + self.rng.uniform(2, 8)),
                L, drift=0.03, volatility=0.2, smoothness=3.0,
            )
            w[:, 3] = self._random_walk(
                min(center[3], self.th.combo_humidity_low),
                L, drift=-0.02, volatility=0.2, smoothness=3.5,
            )
            w[:, 4] = self._diurnal_uv(center[4], L)

        elif variant == "multi_extreme":
            # Multiple sensors in extreme range simultaneously
            w[:, 0] = self._random_walk(
                min(center[0], 12.0), L, drift=-0.1, volatility=0.3, smoothness=3.0
            )
            w[:, 1] = self._random_walk(
                max(center[1], 38.0), L, drift=0.04, volatility=0.2, smoothness=3.0
            )
            w[:, 2] = self._random_walk(
                max(center[2], 40.0), L, drift=0.04, volatility=0.2, smoothness=3.0
            )
            w[:, 3] = self._random_walk(
                min(center[3], 28.0), L, drift=-0.02, volatility=0.2, smoothness=3.5
            )
            w[:, 4] = self._diurnal_uv(max(center[4], 7.5), L)

        return w


# ---------------------------------------------------------------------------
# Augmentation Techniques
# ---------------------------------------------------------------------------


class TimeSeriesAugmentor:
    """Augmentation methods designed for multi-channel temporal windows."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def smote_interpolate(
        self, window_a: np.ndarray, window_b: np.ndarray
    ) -> np.ndarray:
        """SMOTE-style interpolation between two windows of the same class.
        Interpolates element-wise with a random alpha per channel for diversity."""
        alphas = self.rng.uniform(0.2, 0.8, size=(1, NUM_CHANNELS))
        return window_a * alphas + window_b * (1 - alphas)

    def time_warp(self, window: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Smoothly distort the time axis (stretch/compress random segments)."""
        L = window.shape[0]
        # Create a smooth random warping path
        warp_steps = self.rng.normal(1.0, sigma, size=4)
        warp_steps = np.clip(warp_steps, 0.5, 1.5)

        # Build a piecewise warping function
        orig_steps = np.linspace(0, L - 1, num=len(warp_steps))
        cumulative = np.cumsum(warp_steps)
        cumulative = cumulative / cumulative[-1] * (L - 1)

        warp_fn = interp1d(orig_steps, cumulative, kind="linear", fill_value="extrapolate")
        warped_indices = warp_fn(np.linspace(0, L - 1, L))
        warped_indices = np.clip(warped_indices, 0, L - 1)

        # Resample each channel
        warped = np.zeros_like(window)
        for ch in range(NUM_CHANNELS):
            interp_ch = interp1d(np.arange(L), window[:, ch], kind="linear", fill_value="extrapolate")
            warped[:, ch] = interp_ch(warped_indices)

        return warped

    def magnitude_warp(self, window: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Smoothly scale the magnitude of sensor readings over time."""
        L = window.shape[0]
        # Per-channel smooth scaling curve
        warped = np.zeros_like(window)
        for ch in range(NUM_CHANNELS):
            knots = self.rng.normal(1.0, sigma, size=4)
            knots = np.clip(knots, 0.8, 1.2)
            x_knots = np.linspace(0, L - 1, len(knots))
            scale_fn = interp1d(x_knots, knots, kind="quadratic", fill_value="extrapolate")
            scale_curve = scale_fn(np.arange(L))
            warped[:, ch] = window[:, ch] * scale_curve
        return warped

    def jitter(self, window: np.ndarray, noise_scales: Optional[np.ndarray] = None) -> np.ndarray:
        """Add channel-specific Gaussian jitter."""
        if noise_scales is None:
            # Default noise proportional to typical sensor variability
            noise_scales = np.array([1.0, 0.3, 0.4, 0.8, 0.15])
        noise = self.rng.normal(0, 1, size=window.shape) * noise_scales[np.newaxis, :]
        return window + noise

    def channel_dropout(self, window: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
        """Randomly zero-out an entire channel (simulates sensor failure).
        Used sparingly to improve model robustness."""
        result = window.copy()
        for ch in range(NUM_CHANNELS):
            if self.rng.random() < drop_prob:
                result[:, ch] = 0.0
        return result


# ---------------------------------------------------------------------------
# Validation / Labeling
# ---------------------------------------------------------------------------


class SampleValidator:
    """Validates generated windows against class rules and relabels if necessary."""

    def __init__(self, thresholds: Thresholds):
        self.th = thresholds

    def compute_window_stats(self, window: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics for classification validation."""
        means = window.mean(axis=0)
        return {
            "soil_moist_mean": means[0],
            "soil_temp_mean": means[1],
            "amb_temp_mean": means[2],
            "amb_humid_mean": means[3],
            "uv_mean": means[4],
            "soil_moist_min": window[:, 0].min(),
            "soil_moist_trend": window[-10:, 0].mean() - window[:10, 0].mean(),  # end vs start
            "temp_max": max(window[:, 1].max(), window[:, 2].max()),
            "uv_max": window[:, 4].max(),
        }

    def validate_label(self, window: np.ndarray, intended_label: int) -> Tuple[bool, int]:
        """Check if window actually matches the intended class label.

        Returns:
            (is_valid, corrected_label): is_valid=True if intended matches reality,
                                          corrected_label is the "true" label.
        """
        stats = self.compute_window_stats(window)

        # --- Determine ground truth label from sensor statistics ---
        is_risky = False
        is_watering = False

        # Risky: single-sensor extremes
        if stats["soil_moist_mean"] < self.th.soil_moisture_critical:
            is_risky = True
        if stats["temp_max"] > self.th.combo_temp_high:
            is_risky = True
        if stats["uv_max"] > self.th.uv_dangerous + 1.5:
            is_risky = True

        # Risky: dangerous combinations
        if (
            stats["soil_moist_mean"] < self.th.combo_moisture_low
            and stats["temp_max"] > self.th.temp_stress
        ):
            is_risky = True
        if stats["temp_max"] > self.th.temp_stress and stats["amb_humid_mean"] < self.th.combo_humidity_low:
            is_risky = True

        # Watering Required
        if (
            not is_risky
            and stats["soil_moist_mean"] < self.th.soil_moisture_low
        ):
            is_watering = True
        # Also: declining moisture trend even if currently borderline
        if (
            not is_risky
            and stats["soil_moist_mean"] < self.th.soil_moisture_low + 5
            and stats["soil_moist_trend"] < -2.0
        ):
            is_watering = True

        # Assign ground truth
        if is_risky:
            true_label = 2
        elif is_watering:
            true_label = 1
        else:
            true_label = 0

        return (true_label == intended_label, true_label)


# ---------------------------------------------------------------------------
# CSV Ingestion
# ---------------------------------------------------------------------------


def _parse_csv(csv_path: str):
    """Parse a CSV file and return sensor values + labels as arrays.

    Expected CSV format:
        soil_moisture, soil_temp, amb_temp, amb_humidity, uv_index, label

    Returns:
        values: (N, 5) float32 sensor readings
        labels: (N,)  int64 class labels
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    expected_cols = ["soil_moisture", "soil_temp", "amb_temp", "amb_humidity", "uv_index", "label"]

    # Flexible column matching
    if not all(c in df.columns for c in expected_cols):
        print(f"[WARN] CSV columns {list(df.columns)} don't match expected {expected_cols}")
        print("       Attempting positional mapping (first 5 cols = sensors, last col = label)")
        df.columns = expected_cols[: len(df.columns)]

    sensor_cols = expected_cols[:5]
    values = df[sensor_cols].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)
    return values, labels


def load_real_data_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load real-world sensor data from CSV for augmentation.

    For LARGE CSVs (≥60 rows):
        Segments contiguous time-series data into (60, 5) sliding windows.

    For SMALL CSVs (<60 rows):
        Returns empty arrays — these are handled separately as reference
        snapshots via `load_csv_as_references()`.

    Returns:
        X: (N, 60, 5) windows
        y: (N,) labels
    """
    values, labels = _parse_csv(csv_path)

    if len(values) < WINDOW_SIZE:
        print(f"[INFO] CSV {csv_path} has {len(values)} rows (< {WINDOW_SIZE}). "
              f"Will be used as reference snapshots instead of sliding windows.")
        return np.empty((0, WINDOW_SIZE, NUM_CHANNELS)), np.empty((0,), dtype=np.int64)

    # Create sliding windows from long time-series
    windows = []
    window_labels = []
    for i in range(len(values) - WINDOW_SIZE + 1):
        w = values[i : i + WINDOW_SIZE]
        # Majority-vote label for the window
        lbl_window = labels[i : i + WINDOW_SIZE]
        majority = int(np.bincount(lbl_window).argmax())
        windows.append(w)
        window_labels.append(majority)

    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)


def load_csv_as_references(csv_path: str) -> Optional[Dict[int, List[List[float]]]]:
    """Load a small CSV (<60 rows) as per-class reference snapshots.

    Each row is treated as a single sensor snapshot (center point) from which
    full 60-step temporal windows will be generated synthetically.

    Returns:
        Dict mapping class_label → list of [soil_moist, soil_temp, amb_temp, amb_humid, uv] snapshots,
        or None if the CSV is large enough for sliding windows.
    """
    values, labels = _parse_csv(csv_path)

    if len(values) >= WINDOW_SIZE:
        return None  # Large CSV — use sliding window path instead

    refs_by_class: Dict[int, List[List[float]]] = {0: [], 1: [], 2: []}
    for i in range(len(values)):
        lbl = int(labels[i])
        if lbl in refs_by_class:
            refs_by_class[lbl].append(values[i].tolist())

    total = sum(len(v) for v in refs_by_class.values())
    for cls in range(NUM_CLASSES):
        count = len(refs_by_class[cls])
        print(f"       {CLASS_NAMES[cls]}: {count} reference snapshots")

    print(f"[INFO] Loaded {total} snapshot references from {csv_path}")
    return refs_by_class


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------


class SyntheticDatasetGenerator:
    """Orchestrates the full synthetic dataset generation pipeline."""

    def __init__(
        self,
        total_samples: int = 10000,
        seed: int = SEED,
        sensor_cfg: Optional[SensorConfig] = None,
        thresholds: Optional[Thresholds] = None,
        references: Optional[ReferenceSnapshots] = None,
    ):
        self.total_samples = total_samples
        self.rng = np.random.default_rng(seed)
        self.cfg = sensor_cfg or SensorConfig()
        self.th = thresholds or Thresholds()
        self.refs = references or ReferenceSnapshots()

        self.engine = TemporalPatternEngine(self.cfg, self.th, self.rng)
        self.augmentor = TimeSeriesAugmentor(self.rng)
        self.validator = SampleValidator(self.th)

        # Per-class target count (balanced)
        self.per_class = self.total_samples // NUM_CLASSES

    def _perturb_center(
        self, ref: np.ndarray, class_label: int
    ) -> np.ndarray:
        """Create a randomly perturbed version of a reference snapshot while
        keeping it within the valid range for its class."""

        center = ref.copy()

        # Class-specific perturbation ranges
        if class_label == 0:  # Normal
            perturbation = np.array([
                self.rng.uniform(-15, 15),   # soil moisture: 40-95 approx
                self.rng.uniform(-6, 4),     # soil temp: 20-34
                self.rng.uniform(-6, 4),     # amb temp: 20-34
                self.rng.uniform(-10, 10),   # humidity: 40-70
                self.rng.uniform(-1.5, 2.0), # UV: 0.5-6
            ])
            center += perturbation
            # Enforce normal bounds
            center[0] = np.clip(center[0], 35.0, 90.0)  # moisture well above threshold
            center[1] = np.clip(center[1], 18.0, 34.0)
            center[2] = np.clip(center[2], 18.0, 34.0)
            center[3] = np.clip(center[3], 30.0, 80.0)
            center[4] = np.clip(center[4], 0.0, 6.0)

        elif class_label == 1:  # Watering Required
            perturbation = np.array([
                self.rng.uniform(-8, 8),
                self.rng.uniform(-4, 4),
                self.rng.uniform(-4, 4),
                self.rng.uniform(-10, 10),
                self.rng.uniform(-1.5, 1.5),
            ])
            center += perturbation
            # Must have low-ish moisture, but not extreme-risky combos
            center[0] = np.clip(center[0], 12.0, 32.0)
            center[1] = np.clip(center[1], 20.0, 34.0)
            center[2] = np.clip(center[2], 22.0, 34.0)
            center[3] = np.clip(center[3], 30.0, 75.0)
            center[4] = np.clip(center[4], 0.5, 6.5)

        elif class_label == 2:  # Risky
            perturbation = np.array([
                self.rng.uniform(-5, 5),
                self.rng.uniform(-3, 5),
                self.rng.uniform(-3, 5),
                self.rng.uniform(-8, 8),
                self.rng.uniform(-2, 4),
            ])
            center += perturbation
            # At least some values should be extreme
            center[0] = np.clip(center[0], 3.0, 22.0)
            center[1] = np.clip(center[1], 30.0, 50.0)
            center[2] = np.clip(center[2], 33.0, 48.0)
            center[3] = np.clip(center[3], 15.0, 45.0)
            center[4] = np.clip(center[4], 4.0, 14.0)

        return center

    def _generate_base_windows(self, class_label: int, count: int) -> List[np.ndarray]:
        """Generate base windows for a class from reference snapshots."""
        ref_list = [self.refs.normal, self.refs.watering_required, self.refs.risky][class_label]
        ref_array = [np.array(r) for r in ref_list]

        windows = []
        for _ in range(count):
            # Pick a random reference and perturb
            ref = ref_array[self.rng.integers(0, len(ref_array))]
            center = self._perturb_center(ref, class_label)

            # Determine risky sub-variant
            variant = None
            if class_label == 2:
                variant = self.rng.choice([
                    "extreme_moisture", "extreme_temp", "extreme_uv",
                    "combo_moisture_temp", "combo_temp_humidity", "multi_extreme",
                ])

            window = self.engine.generate_window(center, class_label, variant)
            windows.append(window)

        return windows

    def _augment_pool(self, windows: List[np.ndarray], target_count: int) -> List[np.ndarray]:
        """Expand a pool of windows using augmentation until target_count is reached."""
        pool = list(windows)  # start with originals

        while len(pool) < target_count:
            aug_type = self.rng.choice(
                ["smote", "time_warp", "magnitude_warp", "jitter", "combined"],
                p=[0.30, 0.15, 0.15, 0.15, 0.25],
            )

            if aug_type == "smote":
                # Pick two random windows and interpolate
                idx_a, idx_b = self.rng.integers(0, len(windows), size=2)
                new_w = self.augmentor.smote_interpolate(windows[idx_a], windows[idx_b])

            elif aug_type == "time_warp":
                idx = self.rng.integers(0, len(windows))
                new_w = self.augmentor.time_warp(windows[idx], sigma=self.rng.uniform(0.1, 0.3))

            elif aug_type == "magnitude_warp":
                idx = self.rng.integers(0, len(windows))
                new_w = self.augmentor.magnitude_warp(windows[idx], sigma=self.rng.uniform(0.05, 0.15))

            elif aug_type == "jitter":
                idx = self.rng.integers(0, len(windows))
                new_w = self.augmentor.jitter(windows[idx])

            elif aug_type == "combined":
                idx = self.rng.integers(0, len(windows))
                new_w = self.augmentor.time_warp(windows[idx], sigma=0.15)
                new_w = self.augmentor.magnitude_warp(new_w, sigma=0.08)
                new_w = self.augmentor.jitter(new_w)

            pool.append(new_w.astype(np.float32))

        return pool[:target_count]

    def generate(
        self,
        csv_paths: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Run the full generation pipeline.

        Args:
            csv_paths: Optional list of CSV files with real-world data to include.
            validate: Whether to validate/relabel samples against class rules.

        Returns:
            X: (N, 60, 5) float32 tensor
            y: (N,) int64 labels
            metadata: dict with generation statistics
        """
        metadata = {
            "total_requested": self.total_samples,
            "per_class_target": self.per_class,
            "csv_files_used": csv_paths or [],
            "generation_stats": {},
        }

        all_X = []
        all_y = []

        # --- Phase 1: Ingest CSV data if available ---
        csv_counts = {0: 0, 1: 0, 2: 0}
        csv_ref_counts = {0: 0, 1: 0, 2: 0}
        if csv_paths:
            for cp in csv_paths:
                if not os.path.exists(cp):
                    print(f"[WARN] CSV path {cp} not found, skipping.")
                    continue

                print(f"[INFO] Loading data from {cp}")

                # --- Try as sliding windows first (large CSV) ---
                X_real, y_real = load_real_data_csv(cp)
                if len(X_real) > 0:
                    all_X.append(X_real)
                    all_y.append(y_real)
                    for lbl in range(NUM_CLASSES):
                        csv_counts[lbl] += int((y_real == lbl).sum())
                    print(f"       Loaded {len(X_real)} sliding windows from CSV")
                    continue

                # --- Small CSV: use rows as reference snapshots ---
                csv_refs = load_csv_as_references(cp)
                if csv_refs is not None:
                    for cls_lbl, snapshots in csv_refs.items():
                        if len(snapshots) > 0:
                            ref_target = [
                                self.refs.normal,
                                self.refs.watering_required,
                                self.refs.risky,
                            ][cls_lbl]
                            ref_target.extend(snapshots)
                            csv_ref_counts[cls_lbl] += len(snapshots)
                    print(f"       Merged CSV snapshots into reference pool")

        metadata["csv_counts"] = csv_counts
        metadata["csv_reference_counts"] = csv_ref_counts

        # --- Phase 2: Generate synthetic base windows per class ---
        for cls in range(NUM_CLASSES):
            remaining = max(0, self.per_class - csv_counts[cls])
            if remaining == 0:
                print(f"[INFO] Class {CLASS_NAMES[cls]}: already have {csv_counts[cls]} from CSV, skipping generation")
                continue

            # Generate ~40% as base, then augment to fill
            base_count = max(remaining // 3, 50)
            print(f"[INFO] Class {CLASS_NAMES[cls]}: generating {base_count} base windows → augmenting to {remaining}")

            base_windows = self._generate_base_windows(cls, base_count)

            # --- Phase 3: Augment to target count ---
            augmented = self._augment_pool(base_windows, remaining)

            X_cls = np.array(augmented, dtype=np.float32)
            y_cls = np.full(len(augmented), cls, dtype=np.int64)

            all_X.append(X_cls)
            all_y.append(y_cls)

            metadata["generation_stats"][CLASS_NAMES[cls]] = {
                "base_generated": base_count,
                "augmented_total": len(augmented),
            }

        # --- Combine ---
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        # --- Phase 4: Validation and relabeling ---
        if validate:
            corrections = 0
            for i in range(len(X)):
                is_valid, true_label = self.validator.validate_label(X[i], int(y[i]))
                if not is_valid:
                    y[i] = true_label
                    corrections += 1
            metadata["validation_corrections"] = corrections
            print(f"[INFO] Validation: {corrections}/{len(X)} samples relabeled ({corrections/len(X)*100:.1f}%)")

        # --- Phase 5: Shuffle ---
        shuffle_idx = self.rng.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # --- Clamp to absolute sensor ranges ---
        for ch_idx, key in enumerate(SENSOR_SHORT):
            lo, hi = self.cfg.abs_range[key]
            X[:, :, ch_idx] = np.clip(X[:, :, ch_idx], lo, hi)

        metadata["final_total"] = len(X)
        metadata["final_distribution"] = {
            CLASS_NAMES[i]: int((y == i).sum()) for i in range(NUM_CLASSES)
        }
        print(f"[INFO] Final dataset: {len(X)} samples")
        print(f"       Distribution: {metadata['final_distribution']}")

        return X, y, metadata


# ---------------------------------------------------------------------------
# Analytics & Visualization
# ---------------------------------------------------------------------------


class DatasetAnalytics:
    """Generate comprehensive visualizations of the synthetic dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray, output_dir: str = "plots"):
        self.X = X
        self.y = y
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, orange, red
        self.class_names = CLASS_NAMES

    def plot_all(self):
        """Generate all analytics plots."""
        print("[INFO] Generating analytics plots...")
        self.plot_class_distribution()
        self.plot_sensor_distributions()
        self.plot_temporal_examples()
        self.plot_sensor_means_by_class()
        self.plot_correlation_matrices()
        self.plot_feature_space_pca()
        print(f"[INFO] All plots saved to {self.output_dir}/")

    def plot_class_distribution(self):
        """Bar chart of class distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = [int((self.y == i).sum()) for i in range(NUM_CLASSES)]
        bars = ax.bar(self.class_names, counts, color=self.colors, edgecolor="white", linewidth=1.5)

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count}\n({count/len(self.y)*100:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_ylim(0, max(counts) * 1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(self.output_dir / "01_class_distribution.png", dpi=150)
        plt.close(fig)

    def plot_sensor_distributions(self):
        """Violin + box plots of each sensor's distribution, split by class."""
        fig, axes = plt.subplots(1, NUM_CHANNELS, figsize=(22, 5))
        fig.suptitle("Sensor Value Distributions by Class (Window Means)", fontsize=14, fontweight="bold")

        # Compute per-window means
        means = self.X.mean(axis=1)  # (N, 5)

        for ch_idx, ax in enumerate(axes):
            data_by_class = [means[self.y == cls, ch_idx] for cls in range(NUM_CLASSES)]

            vp = ax.violinplot(data_by_class, positions=range(NUM_CLASSES), showmedians=True)
            for i, body in enumerate(vp["bodies"]):
                body.set_facecolor(self.colors[i])
                body.set_alpha(0.6)
            vp["cmedians"].set_color("black")

            ax.set_xticks(range(NUM_CLASSES))
            ax.set_xticklabels(self.class_names, fontsize=9)
            ax.set_title(SENSOR_NAMES[ch_idx], fontsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(self.output_dir / "02_sensor_distributions.png", dpi=150)
        plt.close(fig)

    def plot_temporal_examples(self):
        """Show example temporal windows for each class."""
        fig, axes = plt.subplots(NUM_CLASSES, NUM_CHANNELS, figsize=(22, 10))
        fig.suptitle("Example Temporal Windows per Class", fontsize=14, fontweight="bold")

        for cls in range(NUM_CLASSES):
            # Pick 3 random examples and overlay
            indices = np.where(self.y == cls)[0]
            chosen = np.random.choice(indices, size=min(3, len(indices)), replace=False)

            for ch_idx in range(NUM_CHANNELS):
                ax = axes[cls, ch_idx]
                for idx in chosen:
                    ax.plot(
                        self.X[idx, :, ch_idx],
                        alpha=0.7,
                        linewidth=1.2,
                        color=self.colors[cls],
                    )
                ax.set_title(
                    f"{self.class_names[cls]} — {SENSOR_NAMES[ch_idx]}" if cls == 0 else SENSOR_NAMES[ch_idx],
                    fontsize=9,
                )
                if ch_idx == 0:
                    ax.set_ylabel(self.class_names[cls], fontsize=10, fontweight="bold")
                if cls == NUM_CLASSES - 1:
                    ax.set_xlabel("Timestep", fontsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(self.output_dir / "03_temporal_examples.png", dpi=150)
        plt.close(fig)

    def plot_sensor_means_by_class(self):
        """Grouped bar chart of mean sensor values per class."""
        fig, ax = plt.subplots(figsize=(12, 6))
        means = self.X.mean(axis=1)  # (N, 5)

        x = np.arange(NUM_CHANNELS)
        width = 0.25

        for cls in range(NUM_CLASSES):
            cls_means = means[self.y == cls].mean(axis=0)
            cls_stds = means[self.y == cls].std(axis=0)
            bars = ax.bar(
                x + cls * width, cls_means, width,
                yerr=cls_stds, capsize=3,
                label=self.class_names[cls],
                color=self.colors[cls], alpha=0.85,
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(SENSOR_NAMES, fontsize=10)
        ax.set_ylabel("Mean Value", fontsize=12)
        ax.set_title("Mean Sensor Values by Class (±1σ)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(self.output_dir / "04_sensor_means_by_class.png", dpi=150)
        plt.close(fig)

    def plot_correlation_matrices(self):
        """Per-class correlation matrices between sensor channels."""
        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(18, 5))
        fig.suptitle("Inter-Sensor Correlation by Class", fontsize=14, fontweight="bold")

        means = self.X.mean(axis=1)  # (N, 5)

        for cls, ax in enumerate(axes):
            data = means[self.y == cls]
            corr = np.corrcoef(data.T)

            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(NUM_CHANNELS))
            ax.set_yticks(range(NUM_CHANNELS))
            ax.set_xticklabels(SENSOR_SHORT, fontsize=8, rotation=45, ha="right")
            ax.set_yticklabels(SENSOR_SHORT, fontsize=8)
            ax.set_title(self.class_names[cls], fontsize=11, fontweight="bold")

            # Annotate cells
            for i in range(NUM_CHANNELS):
                for j in range(NUM_CHANNELS):
                    ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Correlation")
        plt.tight_layout(rect=[0, 0, 0.95, 0.93])
        fig.savefig(self.output_dir / "05_correlation_matrices.png", dpi=150)
        plt.close(fig)

    def plot_feature_space_pca(self):
        """PCA projection of flattened windows into 2D for visual inspection."""
        from sklearn.decomposition import PCA

        # Flatten windows: (N, 60*5) = (N, 300)
        X_flat = self.X.reshape(len(self.X), -1)

        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_flat)

        fig, ax = plt.subplots(figsize=(9, 7))
        for cls in range(NUM_CLASSES):
            mask = self.y == cls
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=self.colors[cls], label=self.class_names[cls],
                alpha=0.4, s=8, edgecolors="none",
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
        ax.set_title("PCA Projection of Dataset (Flattened Windows)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, markerscale=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(self.output_dir / "06_pca_projection.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="AgroCare IoT — Synthetic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--total", type=int, default=10000,
        help="Total number of samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--csv", nargs="*", default=None,
        help="Path(s) to real-world CSV files to include and augment",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Directory to save the generated dataset (default: data/)",
    )
    parser.add_argument(
        "--plot-dir", type=str, default="plots",
        help="Directory to save analytics plots (default: plots/)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation/relabeling step",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip analytics plot generation",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  AgroCare IoT — Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  Target samples : {args.total}")
    print(f"  Seed           : {args.seed}")
    print(f"  CSV inputs     : {args.csv or 'None (reference-only mode)'}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Plot dir       : {args.plot_dir}")
    print("=" * 60)

    # --- Generate ---
    generator = SyntheticDatasetGenerator(
        total_samples=args.total,
        seed=args.seed,
    )

    X, y, metadata = generator.generate(
        csv_paths=args.csv,
        validate=not args.no_validate,
    )

    # --- Save dataset ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "agrocare_dataset.npz"
    np.savez_compressed(
        dataset_path,
        X=X,
        y=y,
    )
    print(f"\n[INFO] Dataset saved to {dataset_path}")
    print(f"       X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"       File size: {os.path.getsize(dataset_path) / 1e6:.2f} MB")

    # Save metadata
    meta_path = out_dir / "dataset_metadata.json"
    # Convert numpy types to python native for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=convert)
    print(f"[INFO] Metadata saved to {meta_path}")

    # --- Analytics ---
    if not args.no_plots:
        analytics = DatasetAnalytics(X, y, output_dir=args.plot_dir)
        analytics.plot_all()

    print("\n[DONE] Dataset generation complete!")
    return X, y, metadata


if __name__ == "__main__":
    main()