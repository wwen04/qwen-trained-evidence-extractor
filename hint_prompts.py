"""Shared no-attempt hint wording; evidence is the only optional prompt block."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
INITIAL_HINT_TIERS = {
    1: "Invite the student to notice one relevant visual detail",
    2: "Connect a visible detail to a useful science idea",
    3: "Suggest a small reasoning step without solving the question",
}
INITIAL_HINT_MAX_NEW_TOKENS = 160


def load_initial_hint_template(root=ROOT):
    template = (Path(root) / "prompts/initial_hint.txt").read_text(encoding="utf-8").rstrip("\n")
    if set(re.findall(r"\{\{(\w+)\}\}", template)) != {
        "question", "evidence_line", "tier", "tier_instruction"
    }:
        raise ValueError("Unexpected fields in the no-attempt hint template.")
    return template


def render_initial_hint(question, level=1, evidence=None, *, template=None):
    if level not in INITIAL_HINT_TIERS:
        raise ValueError("level must be 1, 2, or 3")
    values = {"question": str(question), "tier": str(level),
              "tier_instruction": INITIAL_HINT_TIERS[level],
              "evidence_line": "" if evidence is None else f"Visual evidence (unverified): {evidence}\n"}
    if template is None:
        template = load_initial_hint_template()
    # Do not interpret placeholder-like text within a question or evidence string.
    return re.sub(r"\{\{(\w+)\}\}", lambda match: values[match[1]], template)
