import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover - optional dependency in mock mode
    torch = None  # type: ignore
    TORCH_IMPORT_ERROR = e
else:
    TORCH_IMPORT_ERROR = None

from .parsing import BitstringValidationError, validate_bitstring
from .prompting import (
    PromptTemplates,
    build_messages_for_classification,
    build_rationale_messages,
    build_repair_messages,
)


def _require_torch():
    if torch is None:
        raise ImportError(f"PyTorch is required for real model inference: {TORCH_IMPORT_ERROR}")


def _logits_to_probabilities(logits_pairs: List[Tuple[float, float]]) -> List[float]:
    probs: List[float] = []
    for log0, log1 in logits_pairs:
        max_log = max(log0, log1)
        exp0 = math.exp(log0 - max_log)
        exp1 = math.exp(log1 - max_log)
        prob1 = exp1 / (exp0 + exp1)
        probs.append(prob1)
    return probs


@dataclass
class GenerationResult:
    bitstring: Optional[str]
    probabilities: Optional[List[float]]
    token_logits: Optional[List[Tuple[float, float]]]
    parse_error: Optional[str] = None


class MockLLM:
    def __init__(self, label_count: int, seed: int, templates: PromptTemplates):
        self.label_count = label_count
        self.seed = seed
        self.templates = templates

    def generate_bitstring(self, text: str, labels: List[Dict[str, str]], repair: bool = False) -> GenerationResult:
        base = abs(hash(text)) % (10**6)
        rng = np.random.default_rng(self.seed + base + (1 if repair else 0))
        probs = rng.uniform(0.1, 0.9, size=self.label_count)
        logits_pairs = []
        bits = []
        for p in probs:
            logit1 = float(math.log(p / (1 - p)))
            logits_pairs.append((0.0, logit1))
            bits.append("1" if p >= 0.5 else "0")
        bitstring = "".join(bits)
        return GenerationResult(bitstring=bitstring, probabilities=probs.tolist(), token_logits=logits_pairs)

    def generate_rationales(self, text: str, positive_labels: List[str]) -> Dict[str, str]:
        return {label: f"Mock rationale for {label} given text signal." for label in positive_labels}


class RealLLM:
    def __init__(self, model_name: str, device: str, label_count: int, templates: PromptTemplates):
        _require_torch()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available; use --device cpu or --mock")
        self.model_name = model_name
        self.device = torch.device(device)
        self.label_count = label_count
        self.templates = templates
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if len(self.tokenizer.encode("0", add_special_tokens=False)) != 1 or len(self.tokenizer.encode("1", add_special_tokens=False)) != 1:
            raise ValueError("Tokenizer must encode '0' and '1' as single tokens; adjust model or tokenizer")
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.token_zero = self.tokenizer.convert_tokens_to_ids("0")
        self.token_one = self.tokenizer.convert_tokens_to_ids("1")

    def _messages_to_inputs(self, messages: List[Dict[str, str]]):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return prompt.to(self.device)

    def generate_bitstring(self, text: str, labels: List[Dict[str, str]], repair: bool = False) -> GenerationResult:
        messages = build_repair_messages(text, labels, self.templates) if repair else build_messages_for_classification(text, labels, self.templates)
        inputs = self._messages_to_inputs(messages)
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.label_count,
                min_new_tokens=self.label_count,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = output.sequences[0, inputs.shape[1] :]
        bits: List[str] = []
        logits_pairs: List[Tuple[float, float]] = []
        for token_id, score in zip(gen_ids.tolist(), output.scores):
            token_str = self.tokenizer.decode([token_id]).strip()
            bits.append(token_str)
            logit0 = float(score[0, self.token_zero].item())
            logit1 = float(score[0, self.token_one].item())
            logits_pairs.append((logit0, logit1))
        bitstring = "".join(bits)
        probabilities = _logits_to_probabilities(logits_pairs)
        return GenerationResult(bitstring=bitstring, probabilities=probabilities, token_logits=logits_pairs)

    def generate_rationales(self, text: str, positive_labels: List[str]) -> Dict[str, str]:
        if not positive_labels:
            return {}
        messages = build_rationale_messages(text, positive_labels, self.templates)
        inputs = self._messages_to_inputs(messages)
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = output[0, inputs.shape[1] :]
        text_out = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        try:
            data = json.loads(text_out)
            return {k: str(v) for k, v in data.items() if k in positive_labels}
        except Exception:
            return {label: "" for label in positive_labels}


class LLMClassifier:
    def __init__(self, model_name: str, device: str, label_count: int, seed: int, prompt_templates: PromptTemplates, mock: bool = False):
        self.label_count = label_count
        self.mock = mock
        self.templates = prompt_templates
        self.backend = MockLLM(label_count, seed, prompt_templates) if mock else RealLLM(model_name, device, label_count, prompt_templates)

    def _validate(self, result: GenerationResult) -> GenerationResult:
        bitstring = result.bitstring
        try:
            bitstring = validate_bitstring(bitstring, self.label_count)
            return GenerationResult(bitstring=bitstring, probabilities=result.probabilities, token_logits=result.token_logits)
        except BitstringValidationError as e:
            return GenerationResult(bitstring=None, probabilities=None, token_logits=result.token_logits, parse_error=str(e))

    def predict(self, text: str, labels: List[Dict[str, str]]) -> GenerationResult:
        first = self.backend.generate_bitstring(text, labels, repair=False)
        validated = self._validate(first)
        if validated.bitstring is not None:
            return validated
        repair = self.backend.generate_bitstring(text, labels, repair=True)
        validated_repair = self._validate(repair)
        if validated_repair.bitstring is not None:
            return validated_repair
        return GenerationResult(bitstring=None, probabilities=None, token_logits=repair.token_logits, parse_error=validated_repair.parse_error)

    def generate_rationales(self, text: str, positive_labels: List[str]) -> Dict[str, str]:
        return self.backend.generate_rationales(text, positive_labels)