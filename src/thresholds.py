from typing import Dict, List, Tuple

import numpy as np

from .metrics import compute_macro_weighted_f1, precision_recall_f1


def optimize_thresholds(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    label_order: List[str],
    grid_step: float = 0.01,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    thresholds: Dict[str, float] = {}
    per_label_metrics: Dict[str, Dict[str, float]] = {}
    pr_curves: Dict[str, List[Dict[str, float]]] = {}
    grid = np.arange(0.0, 1.0 + 1e-6, grid_step)
    for idx, label in enumerate(label_order):
        probs_label = probabilities[:, idx]
        y_label = y_true[:, idx]
        best_f1 = -1.0
        best_thresh = 0.5
        best_prec, best_rec = 0.0, 0.0
        curve_points: List[Dict[str, float]] = []
        for t in grid:
            preds = (probs_label >= t).astype(int)
            prec, rec, f1 = precision_recall_f1(y_label, preds)
            curve_points.append({"threshold": float(t), "precision": prec, "recall": rec})
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)
                best_prec, best_rec = prec, rec
        thresholds[label] = best_thresh
        per_label_metrics[label] = {
            "threshold": best_thresh,
            "precision": best_prec,
            "recall": best_rec,
            "f1": best_f1,
            "support": float(y_label.sum()),
        }
        pr_curves[label] = curve_points
    macro_f1, weighted_f1 = compute_macro_weighted_f1(per_label_metrics)
    summary = {"macro_f1": macro_f1, "weighted_f1": weighted_f1}
    per_label_metrics["summary"] = summary
    return thresholds, per_label_metrics, pr_curves


def apply_thresholds(probabilities: np.ndarray, thresholds: Dict[str, float], label_order: List[str]) -> np.ndarray:
    decisions = np.zeros_like(probabilities, dtype=int)
    for idx, label in enumerate(label_order):
        t = thresholds.get(label, 0.5)
        decisions[:, idx] = (probabilities[:, idx] >= t).astype(int)
    return decisions