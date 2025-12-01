import numpy as np

from src.thresholds import optimize_thresholds


def test_optimize_thresholds_simple_case():
    probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])
    y_true = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    thresholds, metrics, _ = optimize_thresholds(probs, y_true, ["A", "B"], grid_step=0.1)
    assert 0 <= thresholds["A"] <= 1
    assert 0 <= thresholds["B"] <= 1
    # best threshold near 0.5 should give high f1
    assert metrics["summary"]["macro_f1"] >= 0.9