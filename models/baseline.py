"""
Baseline models for AgroCare sensor classification.

Provides lightweight statistical feature extraction and a Logistic
Regression classifier as an interpretable reference alongside the
AgroCareNet deep model.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["Normal", "Watering Required", "Risky"]


@dataclass
class BaselineMetrics:
    """Container for baseline performance and artifacts."""

    pipeline: Pipeline
    feature_names: List[str]
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    report: Dict[str, Dict[str, float]]


def build_stat_features(X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Derive simple per-channel statistics from windowed sensor data.

    Extracts mean, standard deviation, minimum, and maximum for each of the
    five channels, yielding a compact 20-dimensional feature vector per
    example.
    """

    reducers = [
        ("mean", np.mean),
        ("std", np.std),
        ("min", np.min),
        ("max", np.max),
    ]

    features: List[np.ndarray] = []
    names: List[str] = []

    for reducer_name, func in reducers:
        reduced = func(X, axis=1)  # shape: (N, 5)
        features.append(reduced)
        names.extend([f"{reducer_name}_ch{i}" for i in range(reduced.shape[1])])

    stacked = np.concatenate(features, axis=1)
    return stacked, names


def train_logreg_baseline(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> BaselineMetrics:
    """Train and evaluate a Logistic Regression baseline.

    Splits the dataset into train/val/test partitions with stratification,
    fits a scaled Logistic Regression classifier, and returns accuracy plus
    a classification report.
    """

    features, names = build_stat_features(X)

    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        y,
        test_size=temp_ratio,
        stratify=y,
        random_state=seed,
    )

    relative_test = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=seed,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    def score_split(split_X: np.ndarray, split_y: np.ndarray) -> float:
        preds = pipeline.predict(split_X)
        return accuracy_score(split_y, preds)

    train_acc = score_split(X_train, y_train)
    val_acc = score_split(X_val, y_val)
    test_acc = score_split(X_test, y_test)

    full_preds = pipeline.predict(features)
    report = classification_report(
        y,
        full_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return BaselineMetrics(
        pipeline=pipeline,
        feature_names=names,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        report=report,
    )


def predict_snapshot(
    pipeline: Pipeline,
    sensor_values: List[float],
    noise_std: float = 0.0,
) -> Dict[str, float]:
    """Predict class probabilities for a single 5-value snapshot.

    Uses raw sensor values (no dataset normalization) to match the baseline
    training distribution. Optional noise can be added to better mimic
    real windows.
    """

    window = np.array(sensor_values, dtype=np.float32)[None, None, :]
    window = np.repeat(window, 60, axis=1)

    if noise_std > 0:
        rng = np.random.default_rng(0)
        window = window + rng.normal(0, noise_std, size=window.shape)

    features, _ = build_stat_features(window)
    probs = pipeline.predict_proba(features)[0]
    class_idx = int(np.argmax(probs))

    return {
        "class_idx": class_idx,
        "class_name": CLASS_NAMES[class_idx],
        "probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)},
    }
