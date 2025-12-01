import pytest

from src.parsing import BitstringValidationError, build_prediction_record, validate_bitstring


def test_validate_bitstring_success():
    assert validate_bitstring("010", 3) == "010"


def test_validate_bitstring_error():
    with pytest.raises(BitstringValidationError):
        validate_bitstring("012", 3)


def test_build_prediction_record_shapes():
    labels = ["A", "B", "C"]
    record = build_prediction_record(
        text_id="1",
        text="hello",
        bitstring="101",
        probabilities=[0.2, 0.7, 0.8],
        label_order=labels,
        thresholds={"A": 0.5, "B": 0.5, "C": 0.5},
    )
    assert len(record.labels) == 3
    assert record.labels[1].decision_raw == 0
    assert record.labels[2].decision_final == 1