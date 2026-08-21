"""The Layer 1 gold set: questions with the articles that answer them.

This file defines what "correct" means for retrieval. Everything downstream —
recall@k, the regression gate, the README table, the CV bullet — inherits its
accuracy from here, and no amount of engineering detects an error in it. A
question whose ground truth names the wrong article produces a permanent,
invisible retrieval "miss" that no improvement can ever fix.

Two rules follow, and both are enforced rather than trusted:

1. **Ground truth never comes from our own retrieval.** Deriving it by running
   the search and recording the top hit would make recall@k measure the system
   against itself, which is a tautology dressed as a metric. It comes from the
   source's own citation, or from reading the statute.
2. **Every cited article must exist in the corpus.** A gold answer naming an
   article the corpus does not contain is unachievable by construction, and
   silently caps the score.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

GOLD_PATH = Path("evals/gold_qa.jsonl")

Topic = Literal[
    "wynagrodzenie",
    "czas_pracy",
    "okres_probny",
    "urlop",
    "zakaz_konkurencji",
    "wypowiedzenie",
    "poufnosc",
    "kary_umowne",
    "bhp",
    "dyskryminacja",
    "rodzicielstwo",
    "zatrudnienie",
    "other",
]

# Sources write superscripts in many ways: "151-1", "151(1)", "151¹", "Art. 151 1".
# Canonical storage is "151^1", matching the article ids produced by the parser.
#
# Unicode superscripts need a separator inserted, not merely a character swap:
# translating "151¹" glyph-for-glyph yields "1511", which is a different and real
# article. The run must become "151^1".
_SUPERSCRIPT = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e",
    "ᶠ": "f", "ᵍ": "g", "ʰ": "h", "ⁱ": "i",
}  # fmt: skip


def _expand_superscripts(text: str) -> str:
    out: list[str] = []
    in_run = False
    for char in text:
        if char in _SUPERSCRIPT:
            if not in_run:
                out.append("^")
                in_run = True
            out.append(_SUPERSCRIPT[char])
        else:
            in_run = False
            out.append(char)
    return "".join(out)


def normalise_article(raw: str) -> str:
    """Bring an article reference into the canonical form used by the corpus."""
    text = _expand_superscripts(raw.strip())
    text = re.sub(r"^\s*art\.?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("(", "^").replace(")", "")
    text = text.replace("-", "^").replace(" ", "")
    # Collapse any doubling introduced by the substitutions above.
    return re.sub(r"\^+", "^", text).strip("^")


class GoldQuestion(BaseModel):
    id: str = Field(description="Stable identifier; referenced by misses_at_5 in metrics.")
    question: str = Field(min_length=10)
    ground_truth_articles: list[str] = Field(min_length=1)
    topic: Topic = "other"
    act: str = "kp"

    source_url: str = ""
    evidence: str = Field(default="", description="Quote from the source supporting the citation.")
    reference_answer: str = ""
    notes: str = ""

    @field_validator("ground_truth_articles", mode="after")
    @classmethod
    def _canonicalise(cls, articles: list[str]) -> list[str]:
        return [normalise_article(a) for a in articles]


def load_gold(path: Path = GOLD_PATH) -> list[GoldQuestion]:
    if not path.exists():
        return []
    return [
        GoldQuestion.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_gold(questions: list[GoldQuestion], path: Path = GOLD_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(q.model_dump_json() for q in questions)
    path.write_text(body + "\n", encoding="utf-8", newline="\n")
