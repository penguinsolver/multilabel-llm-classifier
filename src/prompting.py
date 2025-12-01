from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_COMBINED_PROMPT = '''
[SYSTEM]
Je bent een deterministische classificatiemodule. Output is EXACT een bitstring van lengte {n_labels}. Alleen chars 0 of 1. Geen spaties, geen newline, geen uitleg. Multi-label toegestaan. Als je twijfelt: kies 0. Gebruik labelvolgorde exact zoals gegeven.

[USER]
N={n_labels}
{label_block}
TEXT: """{text}"""
Output: alleen bitstring.

[REPAIR]
Vorige output was ongeldig. Geef alleen bitstring exact lengte {n_labels} met 0/1, geen extra tekst. Herhaal de taak.
N={n_labels}
{label_block}
TEXT: """{text}"""
Output: alleen bitstring.

[RATIONALE]
Rationale voor positieve labels. Alleen JSON object teruggeven zonder uitleg.
Labels: {labels_csv}
TEXT: """{text}"""
'''

SECTION_KEYS = ["SYSTEM", "USER", "REPAIR", "RATIONALE"]


@dataclass
class PromptTemplates:
    system_template: str
    user_template: str
    repair_template: str
    rationale_template: str


def _extract_sections(text: str) -> Dict[str, List[str]]:
    current = None
    sections: Dict[str, List[str]] = {k: [] for k in SECTION_KEYS}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            key = stripped.strip("[]").upper()
            current = key if key in SECTION_KEYS else None
            continue
        if current:
            sections[current].append(line)
    return sections


def _sections_to_templates(sections: Dict[str, List[str]], fallback: PromptTemplates) -> PromptTemplates:
    def get_content(key: str, default: str) -> str:
        content = "\n".join(sections.get(key, [])).strip()
        return content if content else default

    return PromptTemplates(
        system_template=get_content("SYSTEM", fallback.system_template),
        user_template=get_content("USER", fallback.user_template),
        repair_template=get_content("REPAIR", fallback.repair_template),
        rationale_template=get_content("RATIONALE", fallback.rationale_template),
    )


def _default_templates() -> PromptTemplates:
    sections = _extract_sections(DEFAULT_COMBINED_PROMPT)
    return PromptTemplates(
        system_template="\n".join(sections["SYSTEM"]).strip(),
        user_template="\n".join(sections["USER"]).strip(),
        repair_template="\n".join(sections["REPAIR"]).strip(),
        rationale_template="\n".join(sections["RATIONALE"]).strip(),
    )


DEFAULT_TEMPLATES = _default_templates()


def load_prompt_templates(prompt_file: Optional[str] = None) -> PromptTemplates:
    if prompt_file is None:
        return DEFAULT_TEMPLATES
    p = Path(prompt_file)
    if not p.exists():
        return DEFAULT_TEMPLATES
    sections = _extract_sections(p.read_text(encoding="utf-8"))
    return _sections_to_templates(sections, DEFAULT_TEMPLATES)


def build_label_block(labels: List[Dict[str, str]]) -> str:
    lines = ["LABEL_ORDER: " + ", ".join([l["name"] for l in labels])]
    for label in labels:
        lines.append(f"- {label['name']}: IS {label['definition']}")
        lines.append(f"  IS_NOT {label['not_definition']}")
        pos_examples = label.get("examples_positive", [])
        neg_examples = label.get("examples_negative", [])
        if pos_examples:
            lines.append("  POS_EX: " + " | ".join(pos_examples))
        if neg_examples:
            lines.append("  NEG_EX: " + " | ".join(neg_examples))
    return "\n".join(lines)


def build_classification_user_prompt(text: str, labels: List[Dict[str, str]], templates: PromptTemplates) -> str:
    label_block = build_label_block(labels)
    return templates.user_template.format(n_labels=len(labels), label_block=label_block, text=text)


def build_messages_for_classification(text: str, labels: List[Dict[str, str]], templates: PromptTemplates) -> List[Dict[str, str]]:
    system_prompt = templates.system_template.format(n_labels=len(labels))
    user_prompt = build_classification_user_prompt(text, labels, templates)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_messages(text: str, labels: List[Dict[str, str]], templates: PromptTemplates) -> List[Dict[str, str]]:
    system_prompt = templates.system_template.format(n_labels=len(labels))
    repair_user = templates.repair_template.format(
        n_labels=len(labels),
        label_block=build_label_block(labels),
        text=text,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": repair_user},
    ]


def build_rationale_messages(text: str, positive_labels: List[str], templates: PromptTemplates) -> List[Dict[str, str]]:
    user_content = templates.rationale_template.format(
        labels_csv=", ".join(positive_labels),
        text=text,
    )
    return [
        {"role": "system", "content": "Geef enkel JSON met rationale per label. Geen extra tekst."},
        {"role": "user", "content": user_content},
    ]