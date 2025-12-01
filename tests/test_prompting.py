from src.prompting import PromptTemplates, build_messages_for_classification, build_repair_messages, load_prompt_templates

TEMPLATES = load_prompt_templates()
SAMPLE_LABELS = [
    {"name": "A", "definition": "alpha", "not_definition": "not alpha", "examples_positive": ["good"], "examples_negative": ["bad"]},
    {"name": "B", "definition": "beta", "not_definition": "not beta", "examples_positive": ["yes"], "examples_negative": ["no"]},
]


def test_build_messages_contains_bitstring_instruction():
    messages = build_messages_for_classification("text", SAMPLE_LABELS, TEMPLATES)
    assert messages[0]["role"] == "system"
    assert "bitstring" in messages[0]["content"].lower()
    assert "N=2" in messages[1]["content"]
    assert "LABEL_ORDER" in messages[1]["content"]


def test_repair_prompt_mentions_invalid():
    messages = build_repair_messages("text", SAMPLE_LABELS, TEMPLATES)
    assert "ongeldig" in messages[1]["content"].lower()
    assert "bitstring" in messages[1]["content"].lower()