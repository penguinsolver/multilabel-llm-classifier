# Deterministische multi-label LLM classifier (Qwen 2.5 7B Instruct)

## Wat is het?
- Lokale, reproduceerbare multi-label classifier met open-source LLM (Qwen 2.5 7B Instruct) of mock backend (geen download nodig).
- Output per tekst: bitstring, probabilities uit logits, thresholds per label, optionele rationale (JSON).
- Eén promptbestand (`config/prompt_template.txt`), één labels-bestand (`config/labels_demo.yaml`), één modelconfig (`config/model_config.yaml`).
- Dummy data + demo run zorgen dat alles direct draait.

## Must-have snelle start
### Native (mock, geen model download)
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --demo --mock
```
Artifacts: `artifacts/` (predictions/metrics/thresholds/pr_curves/run_config).

### Docker (mock, geen model download)
```bash
docker build -t multilabel-llm .
docker run --rm -v $(pwd)/artifacts:/app/artifacts multilabel-llm
```
Default command: `python main.py --demo --mock --artifacts-dir /app/artifacts`.
GPU (optioneel): `docker run --gpus all ... python main.py --demo --device cuda --model Qwen/Qwen2.5-7B-Instruct`.

### Echte data / echt model (kort stappenplan)
1) Labels aanpassen: `config/labels_demo.yaml` (volgorde bepaalt bitstring index/N).
2) Prompt fine-tunen: `config/prompt_template.txt` ([SYSTEM]/[USER]/[REPAIR]/[RATIONALE]).
3) Model kiezen: `config/model_config.yaml` (`model`, `device`, `mock=false`) of CLI `--model`/`--device`.
4) Thresholds trainen op jouw data (CSV met `text_id,text[,y_true]`):
   ```bash
   python main.py --trainval --input your.csv --thresholds artifacts/thresholds.json --prompt-file config/prompt_template.txt
   ```
5) Voorspellen met thresholds:
   ```bash
   python main.py --predict --input your.csv --output preds.csv --thresholds artifacts/thresholds.json --prompt-file config/prompt_template.txt
   ```
6) Optioneel rationale: voeg `--rationale` toe bij predict.

## Belangrijke bestanden (simpel gehouden)
- `config/prompt_template.txt` – gecombineerd promptbestand met secties [SYSTEM]/[USER]/[REPAIR]/[RATIONALE]. Pas in teksteditor aan.
- `config/labels_demo.yaml` – labels/definities/voorbeelden (volgorde = bitstring index).
- `config/model_config.yaml` – standaard model/device/mock.
- `data/sample_input.csv` – voorbeeldinput (text_id,text,y_true).
- `main.py` – CLI/orkestratie.
- `src/prompting.py` – leest promptbestand en bouwt berichten.
- `src/llm_infer.py` – mock/real model, logits?probabilities.
- `src/thresholds.py`/`src/metrics.py`/`src/parsing.py`/`src/data_gen.py` – thresholds/metrics/validatie/dummy data.
- `tests/` – unit + smoke.

## Prompt aanpassen (één file)
- Open `config/prompt_template.txt`.
- Placeholders: `{n_labels}`, `{label_block}`, `{text}`, `{labels_csv}` (rationale).
- Gebruik de blokken [SYSTEM]/[USER]/[REPAIR]/[RATIONALE]; CLI `--prompt-file` kan een ander .txt aanwijzen.

## Model wisselen
- `config/model_config.yaml` (`model`, `device`, `mock`).
- Of CLI: `--model <naam>` `--device cpu|cuda` `--mock`.

## Data aanleveren
- CSV kolommen: `text_id`, `text`, optioneel `y_true` (lijst/JSON/semi-colon). Voorbeeld: `data/sample_input.csv`.
- Train/val eigen data (thresholds): `python main.py --trainval --input your.csv --thresholds artifacts/thresholds.json --prompt-file config/prompt_template.txt`
- Predict eigen data: `python main.py --predict --input your.csv --output preds.csv --thresholds artifacts/thresholds.json --prompt-file config/prompt_template.txt`
- XLSX? Converteer naar CSV of breid `read_dataset_csv` uit.
- HuggingFace dataset? Exporteer naar CSV met dezelfde kolommen.

## Pipeline (kort)
input text -> prompt -> bitstring -> logits -> probabilities -> threshold grid search -> final labels -> artifacts.

## Thresholds opnieuw trainen
`python main.py --trainval --mock --grid-step 0.02` (synthetic) of met eigen CSV `--input your.csv`. Resultaat: `artifacts/thresholds.json`.

## Rationale (optioneel)
`--rationale` geeft per positieve label een korte zin (JSON).

## Dummy data
- Demo genereert synthetische data met overlap/imbalance/ruis.
- Voorbeeldinput `data/sample_input.csv` beschikbaar.

## Outputverwachting
- `predictions_*.csv` kolommen: `text_id,text,bitstring_raw,parse_error,prob_<LABEL>,raw_<LABEL>,final_<LABEL>[,rationale_<LABEL>]`.
- `thresholds.json`: per-label drempels.
- `metrics_*.json`: per-label precision/recall/F1 + macro/weighted.

## Nice to have (later)
- Extra promptvarianten per taal.
- Quantization/4-bit.
- XLSX direct inlezen.
- UI voor prompt/label beheer.

## Troubleshooting
- Invalid bitstring? Check `config/prompt_template.txt` en of digits single-token voor je model.
- Te confident? Pas prompt aan (conservatiever) en regenerate thresholds.
- GPU errors? Gebruik `--mock` of `--device cpu`.
- VC++ redistributable nodig op Windows voor echte PyTorch build.

## Tests
`pytest -q` (mock, geen model download).
