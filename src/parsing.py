import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

BIT_REGEX_TEMPLATE = "^[01]{%d}$"


class LabelOutput(BaseModel):
    label: str
    decision_raw: Optional[int] = Field(default=None)
    probability: Optional[float] = Field(default=None)
    decision_final: Optional[int] = Field(default=None)
    rationale: Optional[str] = None


class PredictionOutput(BaseModel):
    text_id: str
    text: str
    bitstring_raw: Optional[str] = None
    parse_error: bool = False
    labels: List[LabelOutput] = Field(default_factory=list)


class BitstringValidationError(Exception):
    pass


def validate_bitstring(bitstring: str, length: int) -> str:
    if bitstring is None:
        raise BitstringValidationError("bitstring is None")
    regex = BIT_REGEX_TEMPLATE % length
    if not re.fullmatch(regex, bitstring.strip()):
        raise BitstringValidationError(f"Bitstring must match {regex}")
    return bitstring


def build_label_outputs(
    bitstring: Optional[str],
    probabilities: Optional[List[float]],
    thresholds: Optional[Dict[str, float]],
    label_order: List[str],
    rationales: Optional[Dict[str, str]] = None,
) -> List[LabelOutput]:
    outputs: List[LabelOutput] = []
    if bitstring is None or probabilities is None:
        for label in label_order:
            outputs.append(LabelOutput(label=label))
        return outputs
    for idx, label in enumerate(label_order):
        decision_raw = int(bitstring[idx])
        prob = probabilities[idx]
        decision_final = None
        if thresholds is not None:
            thresh = thresholds.get(label, 0.5)
            decision_final = int(prob >= thresh)
        rationale = None
        if rationales:
            rationale = rationales.get(label)
        outputs.append(
            LabelOutput(
                label=label,
                decision_raw=decision_raw,
                probability=prob,
                decision_final=decision_final,
                rationale=rationale,
            )
        )
    return outputs


def build_prediction_record(
    text_id: str,
    text: str,
    bitstring: Optional[str],
    probabilities: Optional[List[float]],
    label_order: List[str],
    thresholds: Optional[Dict[str, float]] = None,
    rationales: Optional[Dict[str, str]] = None,
    parse_error: bool = False,
) -> PredictionOutput:
    labels = build_label_outputs(bitstring, probabilities, thresholds, label_order, rationales)
    record = PredictionOutput(
        text_id=text_id,
        text=text,
        bitstring_raw=bitstring,
        parse_error=parse_error,
        labels=labels,
    )
    return record


def safe_validate_prediction(record: PredictionOutput) -> PredictionOutput:
    try:
        return PredictionOutput.model_validate(record.model_dump())
    except ValidationError as e:
        raise ValueError(f"Prediction validation failed: {e}")