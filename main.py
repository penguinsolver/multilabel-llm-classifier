import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_gen import generate_synthetic_dataset
from src.io_utils import (
    ensure_dir,
    flatten_predictions,
    load_json,
    load_label_config,
    read_dataset_csv,
    save_csv,
    save_json,
    set_seed,
    write_run_config,
)
from src.llm_infer import LLMClassifier
from src.metrics import compute_macro_weighted_f1, precision_recall_f1
from src.parsing import build_prediction_record, safe_validate_prediction
from src.prompting import PromptTemplates, load_prompt_templates
from src.thresholds import apply_thresholds, optimize_thresholds


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_CONFIG = "config/model_config.yaml"
DEFAULT_PROMPT_FILE = "config/prompt_template.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministische multi-label classificatie pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Run end-to-end demo met synthetische data")
    mode.add_argument("--trainval", action="store_true", help="Run train/val threshold zoek")
    mode.add_argument("--predict", action="store_true", help="Run predictie op input dataset")

    parser.add_argument("--config", default="config/labels_demo.yaml", help="Pad naar label config YAML")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Pad naar model config (YAML)")
    parser.add_argument("--model", default=None, help="Modelnaam of pad (override config)")
    parser.add_argument("--device", default=None, help="cpu of cuda (override config)")
    parser.add_argument("--mock", action="store_true", help="Force mock model (override config)")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help="Pad naar gecombineerd promptbestand (.txt)")
    parser.add_argument("--rationale", action="store_true", help="Genereer rationale voor positieve labels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grid-step", type=float, default=0.01, help="Grid stap voor thresholds")
    parser.add_argument("--input", help="Input CSV voor predict/trainval")
    parser.add_argument("--output", help="Output CSV pad voor predict")
    parser.add_argument("--thresholds", help="JSON thresholds pad voor predict of save locatie")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Pad voor artifacts")
    return parser.parse_args()


def load_model_settings(args: argparse.Namespace) -> Tuple[str, str, bool]:
    model = DEFAULT_MODEL
    device = "cpu"
    mock = False
    if args.model_config and Path(args.model_config).exists():
        import yaml

        cfg = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8")) or {}
        model = cfg.get("model", model)
        device = cfg.get("device", device)
        mock = cfg.get("mock", mock)
    if args.model:
        model = args.model
    if args.device:
        device = args.device
    if args.mock:
        mock = True
    return model, device, mock


def load_templates_from_args(args: argparse.Namespace) -> PromptTemplates:
    return load_prompt_templates(prompt_file=args.prompt_file)


def prepare_predictor(args: argparse.Namespace, label_count: int, templates: PromptTemplates) -> LLMClassifier:
    model, device, mock = load_model_settings(args)
    return LLMClassifier(
        model_name=model,
        device=device,
        label_count=label_count,
        seed=args.seed,
        prompt_templates=templates,
        mock=mock,
    )


def run_inference(
    predictor: LLMClassifier,
    df: pd.DataFrame,
    labels: List[Dict[str, str]],
    thresholds: Dict[str, float] | None,
    rationale: bool,
) -> Tuple[List[Dict], np.ndarray]:
    label_order = [l["name"] for l in labels]
    predictions = []
    prob_matrix: List[List[float]] = []
    for _, row in df.iterrows():
        res = predictor.predict(row["text"], labels)
        parse_error = res.bitstring is None
        probs = res.probabilities if res.probabilities is not None else [0.0] * len(labels)
        rationales = None
        if rationale and thresholds is not None and not parse_error:
            positive_labels = [label_order[i] for i, p in enumerate(probs) if p >= thresholds.get(label_order[i], 0.5)]
            rationales = predictor.generate_rationales(row["text"], positive_labels)
        record = build_prediction_record(
            text_id=str(row["text_id"]),
            text=row["text"],
            bitstring=res.bitstring,
            probabilities=probs,
            label_order=label_order,
            thresholds=thresholds,
            rationales=rationales,
            parse_error=parse_error,
        )
        record = safe_validate_prediction(record)
        predictions.append(record.model_dump())
        prob_matrix.append(probs)
    return predictions, np.array(prob_matrix)


def evaluate(probabilities: np.ndarray, decisions: np.ndarray, y_true: np.ndarray, label_order: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(label_order):
        prec, rec, f1 = precision_recall_f1(y_true[:, idx], decisions[:, idx])
        metrics[label] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(y_true[:, idx].sum()),
        }
    macro, weighted = compute_macro_weighted_f1(metrics)
    metrics["summary"] = {"macro_f1": macro, "weighted_f1": weighted}
    return metrics


def save_predictions(predictions: List[Dict], label_order: List[str], path: str) -> None:
    df = flatten_predictions(predictions, label_order)
    save_csv(df, path)


def run_demo(args: argparse.Namespace, labels: List[Dict[str, str]], templates: PromptTemplates) -> None:
    datasets = generate_synthetic_dataset(labels, seed=args.seed)
    label_order = [l["name"] for l in labels]
    predictor = prepare_predictor(args, len(labels), templates)

    print("Running validation inference (mock mode=" + str(predictor.mock) + ")...")
    val_predictions, val_probs = run_inference(predictor, datasets["val"], labels, thresholds=None, rationale=False)
    val_true = np.vstack(datasets["val"]["y_true_multi"].to_numpy())

    thresholds, val_metrics, pr_curves = optimize_thresholds(val_probs, val_true, label_order, grid_step=args.grid_step)
    decisions_val = apply_thresholds(val_probs, thresholds, label_order)
    val_metrics_final = evaluate(val_probs, decisions_val, val_true, label_order)

    val_predictions_final, _ = run_inference(predictor, datasets["val"], labels, thresholds=thresholds, rationale=args.rationale)

    print("Running test inference...")
    test_predictions, test_probs = run_inference(predictor, datasets["test"], labels, thresholds=thresholds, rationale=args.rationale)
    test_true = np.vstack(datasets["test"]["y_true_multi"].to_numpy())
    test_decisions = apply_thresholds(test_probs, thresholds, label_order)
    test_metrics = evaluate(test_probs, test_decisions, test_true, label_order)

    art_dir = ensure_dir(args.artifacts_dir)
    save_predictions(val_predictions_final, label_order, str(art_dir / "predictions_val.csv"))
    save_predictions(test_predictions, label_order, str(art_dir / "predictions_test.csv"))
    save_json(thresholds, art_dir / "thresholds.json")
    save_json(val_metrics_final, art_dir / "metrics_val.json")
    save_json(test_metrics, art_dir / "metrics_test.json")
    save_json(pr_curves, art_dir / "pr_curves.json")
    write_run_config(art_dir / "run_config.json", vars(args), {"label_order": label_order})
    print(f"Artifacts written to {art_dir}")


def run_trainval(args: argparse.Namespace, labels: List[Dict[str, str]], templates: PromptTemplates) -> None:
    label_order = [l["name"] for l in labels]
    if args.input:
        df = read_dataset_csv(args.input, label_order)
        val_df = df
    else:
        datasets = generate_synthetic_dataset(labels, seed=args.seed)
        val_df = datasets["val"]
    predictor = prepare_predictor(args, len(labels), templates)
    val_predictions, val_probs = run_inference(predictor, val_df, labels, thresholds=None, rationale=False)
    val_true = np.vstack(val_df["y_true_multi"].to_numpy())
    thresholds, val_metrics, pr_curves = optimize_thresholds(val_probs, val_true, label_order, grid_step=args.grid_step)
    decisions = apply_thresholds(val_probs, thresholds, label_order)
    final_metrics = evaluate(val_probs, decisions, val_true, label_order)

    art_dir = ensure_dir(args.artifacts_dir)
    save_predictions(val_predictions, label_order, str(art_dir / "predictions_val.csv"))
    save_json(thresholds, args.thresholds or art_dir / "thresholds.json")
    save_json(final_metrics, art_dir / "metrics_val.json")
    save_json(pr_curves, art_dir / "pr_curves.json")
    write_run_config(art_dir / "run_config.json", vars(args), {"label_order": label_order})
    print("Thresholds saved to", args.thresholds or art_dir / "thresholds.json")


def run_predict(args: argparse.Namespace, labels: List[Dict[str, str]], templates: PromptTemplates) -> None:
    if not args.input or not args.output or not args.thresholds:
        raise ValueError("Predict mode requires --input, --output, and --thresholds")
    label_order = [l["name"] for l in labels]
    thresholds = load_json(args.thresholds)
    df = read_dataset_csv(args.input, label_order)
    predictor = prepare_predictor(args, len(labels), templates)
    predictions, _ = run_inference(predictor, df, labels, thresholds=thresholds, rationale=args.rationale)
    save_predictions(predictions, label_order, args.output)
    print("Predictions saved to", args.output)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    templates = load_templates_from_args(args)
    labels = load_label_config(args.config)
    if args.demo:
        run_demo(args, labels, templates)
    elif args.trainval:
        run_trainval(args, labels, templates)
    elif args.predict:
        run_predict(args, labels, templates)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()