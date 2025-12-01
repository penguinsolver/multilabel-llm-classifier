import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


COMMON_NOISE = [
    "uhm", "pls", "asap", "ty", "thx", "???", "help!!",
    "maybe later", "idk", "quick", "urgent", "broken",
]


def _inject_typos(text: str, rng: np.random.Generator) -> str:
    if len(text) < 4:
        return text
    chars = list(text)
    idx = rng.integers(0, len(chars))
    chars[idx] = chars[idx].upper() if chars[idx].islower() else chars[idx].lower()
    return "".join(chars)


def _compose_text(selected_labels: List[str], label_defs: Dict[str, Dict[str, str]], rng: np.random.Generator) -> str:
    fragments: List[str] = []
    for label in selected_labels:
        ld = label_defs[label]
        fragments.append(random.choice(ld.get("examples_positive", [ld["definition"]])))
    for label, ld in label_defs.items():
        if label in selected_labels:
            continue
        if rng.random() < 0.2:
            fragments.append(random.choice(ld.get("examples_negative", [ld["not_definition"]])))
    if not fragments:
        fragments.append("General inquiry about the service features.")
    rng.shuffle(fragments)
    text = " ".join(fragments)
    if rng.random() < 0.4:
        text = _inject_typos(text, rng)
    if rng.random() < 0.5:
        text += " " + rng.choice(COMMON_NOISE)
    return text


def generate_synthetic_dataset(labels: List[Dict[str, str]], seed: int, n_train: int = 120, n_val: int = 60, n_test: int = 60) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    label_names = [l["name"] for l in labels]
    label_defs = {l["name"]: l for l in labels}
    weights = np.linspace(0.5, 0.1, num=len(label_names))

    total = n_train + n_val + n_test
    rows = []
    for idx in range(total):
        active = []
        for name, w in zip(label_names, weights):
            if rng.random() < w:
                active.append(name)
        if rng.random() < 0.2 and len(active) > 1:
            active = active[: rng.integers(1, len(active) + 1)]
        text = _compose_text(active, label_defs, rng)
        rows.append({
            "text_id": f"synthetic_{idx}",
            "text": text,
            "y_true": active,
            "y_true_multi": [1 if name in active else 0 for name in label_names],
        })
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    return {"train": train_df, "val": val_df, "test": test_df}