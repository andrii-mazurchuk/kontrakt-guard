"""Structural checks on a parsed act.

These fail the build rather than warn. A parser that silently drops thirty
articles still produces a plausible corpus and a beautiful recall@k — computed
over a corpus with holes in it. Retrieval evaluation cannot detect that, because
the gold answers for the missing articles simply never surface and look like
ordinary retrieval misses. The only defence is to check the shape of the parse
before anything downstream trusts it.
"""

from __future__ import annotations

import re
from collections import Counter

from ingestion.parse import Article

# A capitalised "Art. N." inside a body. In-text references are lowercase
# ("w rozumieniu art. 22 § 1"), so the capital is what distinguishes a heading
# that was missed from a citation that was intended.
EMBEDDED_HEADING_RE = re.compile(r"(?<![\w.])Art\.\s*\d+(?:\^[0-9a-ząćęłńóśźż]+)*\.")

# Kodeks pracy runs Art. 1 to Art. 305 with roughly 170 amendment-inserted
# articles alongside. The band is wide enough to absorb ordinary amendment
# churn and narrow enough to catch a parser that lost a chapter.
EXPECTED_ARTICLES = {"kp": (430, 530)}


class ParseValidationError(RuntimeError):
    """The parsed act failed a structural check."""


def validate(articles: list[Article], act: str) -> list[str]:
    """Return report lines, or raise if the parse is structurally unsound."""
    problems: list[str] = []
    report: list[str] = []

    if not articles:
        raise ParseValidationError(f"{act}: parsed zero articles")

    report.append(f"{act}: {len(articles)} articles, {sum(a.repealed for a in articles)} repealed")
    report.append(f"  superscripted: {sum('^' in a.article for a in articles)}")
    report.append(f"  paragraphs:    {sum(len(a.paragraphs) for a in articles)}")
    report.append(f"  range:         {articles[0].display} .. {articles[-1].display}")

    duplicates = sorted(k for k, n in Counter(a.article for a in articles).items() if n > 1)
    if duplicates:
        problems.append(f"duplicate article ids: {duplicates[:10]}")

    out_of_order = [
        f"{articles[i - 1].display} -> {articles[i].display}"
        for i in range(1, len(articles))
        if articles[i].sort_key < articles[i - 1].sort_key
    ]
    if out_of_order:
        problems.append(f"articles out of statutory order: {out_of_order[:10]}")

    low, high = EXPECTED_ARTICLES.get(act, (1, 10_000))
    if not low <= len(articles) <= high:
        problems.append(f"article count {len(articles)} outside expected band {low}-{high}")

    unplaced = [a.display for a in articles if not a.title_path]
    if unplaced:
        problems.append(f"articles with no structural path: {unplaced[:10]}")

    hollow = [a.display for a in articles if not a.repealed and len(a.text) < 20]
    if hollow:
        problems.append(f"articles with suspiciously little text: {hollow[:10]}")

    # An article marked not-in-force should be nothing but the marker. A long one
    # means a paragraph-level repeal was mistaken for an article-level one, which
    # removes a live article from retrieval and caps recall with no visible error.
    substantial = [a.display for a in articles if a.repealed and len(a.text) > 120]
    if substantial:
        problems.append(
            f"articles marked not-in-force but carrying substantial text "
            f"(paragraph-level repeal misread as article-level): {substantial[:10]}"
        )

    # An article body must not contain another article's heading. When it does,
    # the parser failed to recognise a boundary and swallowed the next article
    # whole — the article vanishes from the corpus while its text is served under
    # the wrong citation. Article numbering has legal gaps, so a missing id proves
    # nothing; this is the signal that does.
    swallowed = [
        f"{a.display} contains {m.group(0)!r}"
        for a in articles
        if (m := EMBEDDED_HEADING_RE.search(a.text))
    ]
    if swallowed:
        problems.append(f"articles that swallowed a following article: {swallowed[:10]}")

    if problems:
        raise ParseValidationError(
            f"{act}: parse failed {len(problems)} structural check(s)\n  " + "\n  ".join(problems)
        )

    return report
