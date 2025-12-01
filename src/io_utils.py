import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_label_config(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    labels = data.get("labels", [])
    if not labels:
        raise ValueError("Label config is empty or missing 'labels' key")
    return labels


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def read_dataset_csv(path: str | Path, label_order: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column")
    if "text_id" not in df.columns:
        df["text_id"] = [f"row_{i}" for i in range(len(df))]
    if "y_true" in df.columns:
        parsed = []
        for val in df["y_true"]:
            if isinstance(val, str):
                val = val.strip()
                if val.startswith("["):
                    try:
                        parsed.append(json.loads(val))
                    except json.JSONDecodeError:
                        parsed.append(val.split(";"))
                elif val:
                    parsed.append([v for v in val.split(";") if v])
                else:
                    parsed.append([])
            elif isinstance(val, list):
                parsed.append(val)
            else:
                parsed.append([])
        df["y_true"] = parsed
    else:
        df["y_true"] = [[] for _ in range(len(df))]
    def to_multi_hot(labels: List[str]) -> List[int]:
        label_set = set(labels)
        return [1 if name in label_set else 0 for name in label_order]

    df["y_true_multi"] = df["y_true"].apply(to_multi_hot)
    return df


def flatten_predictions(predictions: List[Dict[str, Any]], label_order: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in predictions:
        base = {
            "text_id": item["text_id"],
            "text": item["text"],
            "bitstring_raw": item.get("bitstring_raw"),
            "parse_error": item.get("parse_error", False),
        }
        label_info = item.get("labels", [])
        for info in label_info:
            label = info["label"]
            base[f"prob_{label}"] = info.get("probability")
            base[f"raw_{label}"] = info.get("decision_raw")
            base[f"final_{label}"] = info.get("decision_final")
            if info.get("rationale") is not None:
                base[f"rationale_{label}"] = info.get("rationale")
        rows.append(base)
    return pd.DataFrame(rows)


def write_run_config(path: str | Path, args: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "args": args,
        "extra": extra or {},
    }
    save_json(payload, path)