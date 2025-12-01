from typing import Dict, List, Tuple

import numpy as np


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def compute_macro_weighted_f1(per_label: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    f1_values = [v.get("f1", 0.0) for v in per_label.values()]
    supports = [v.get("support", 0.0) for v in per_label.values()]
    macro = float(np.mean(f1_values)) if f1_values else 0.0
    total_support = float(np.sum(supports))
    if total_support == 0:
        weighted = macro
    else:
        weighted = float(np.average(f1_values, weights=np.maximum(supports, 1e-9)))
    return macro, weighted