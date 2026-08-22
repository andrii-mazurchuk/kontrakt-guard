"""Prompts for the Q&A flow.

Polish, per the repository convention: anything touching legal text stays in the
language of the corpus. A rewrite prompt written in English would be asking the
model to translate as well as to reformulate, and the retrieval index it feeds is
Polish either way.

Kept in one module so a prompt change is visible in a diff as a prompt change,
rather than buried inside a node.
"""

from __future__ import annotations

# Query rewriting. The gap this closes is register, not meaning: users write
# "szef każe mi zostawać po godzinach", the statute says "praca w godzinach
# nadliczbowych", and the lexical leg matches lemmas rather than intent.
REWRITE_SYSTEM = """Jesteś asystentem wyszukiwania w polskim prawie pracy.

Twoim jedynym zadaniem jest przepisanie pytania użytkownika na frazę wyszukiwania
w języku, jakim posługuje się Kodeks pracy.

Zasady:
- Używaj terminologii ustawowej, nie potocznej ("praca w godzinach nadliczbowych",
  nie "nadgodziny"; "rozwiązanie umowy o pracę", nie "zwolnienie").
- Zachowaj wszystkie liczby, okresy i wartości z pytania — to one często decydują
  o tym, który przepis jest właściwy.
- Nie odpowiadaj na pytanie i nie dodawaj przepisów, których użytkownik nie podał.
- Jeśli pytanie już jest sformułowane językiem ustawy, zwróć je bez zmian.
- Zwróć samą frazę wyszukiwania, bez wyjaśnień."""

# Relevance grading. Graded per chunk, independently: a single call ranking all
# candidates at once would let position in the list influence the judgement, and
# the whole purpose of this node is to be a second opinion on the ranking.
GRADE_SYSTEM = """Oceniasz, czy podany fragment Kodeksu pracy jest przydatny do
odpowiedzi na pytanie użytkownika.

Fragment jest PRZYDATNY, jeżeli zawiera przepis, na który trzeba się powołać,
odpowiadając na pytanie — nawet jeśli sam nie wystarcza do pełnej odpowiedzi.

Fragment NIE jest przydatny, jeżeli dotyczy innej instytucji prawnej, a jedynie
używa podobnego słownictwa.

Oceniaj wyłącznie ten jeden fragment. Nie zakładaj, że istnieją inne fragmenty."""

# Answering. Every constraint here exists because its absence is a known failure
# mode of grounded generation, not because it sounds prudent.
ANSWER_SYSTEM = """Jesteś asystentem prawnym odpowiadającym na pytania z zakresu
polskiego prawa pracy wyłącznie na podstawie załączonych fragmentów Kodeksu pracy.

Zasady bezwzględne:
- Korzystaj TYLKO z załączonych fragmentów. Nie dodawaj wiedzy spoza nich, nawet
  jeśli jesteś pewien, że jest poprawna.
- Każde twierdzenie o treści prawa musi wskazywać artykuł, z którego wynika.
- Nie podawaj kwot, terminów ani limitów, których nie ma w załączonych fragmentach.
- Jeżeli fragmenty nie wystarczają, powiedz to wprost zamiast zgadywać.
- Jeżeli fragment jest oznaczony jako uchylony, nie opieraj na nim odpowiedzi —
  możesz jedynie zaznaczyć, że przepis już nie obowiązuje.
- Odpowiadaj zwięźle i po polsku."""

# Appended to every answer the flow returns. Not a legal opinion, and the project
# is a portfolio artefact rather than a service — saying so is cheap and its
# absence would be the single most obviously wrong thing about the output.
DISCLAIMER = (
    "To nie jest porada prawna. Odpowiedź powstała automatycznie na podstawie "
    "tekstu Kodeksu pracy i może być niepełna lub nieaktualna. W sprawie "
    "indywidualnej skonsultuj się z prawnikiem lub Państwową Inspekcją Pracy."
)

# The refusal. Returned when nothing survives grading, and deliberately specific:
# "I could not ground this in the corpus" is a different statement from "the law
# does not say", and conflating them is how a refusal becomes misinformation.
REFUSAL = (
    "Nie znalazłem w Kodeksie pracy przepisów, na których mógłbym oprzeć "
    "odpowiedź na to pytanie. Nie oznacza to, że prawo tej kwestii nie reguluje "
    "— może ona być uregulowana w innej ustawie, której nie mam w korpusie, albo "
    "pytanie wymaga doprecyzowania."
)


def grade_user_prompt(question: str, citation: str, path: str, content: str) -> str:
    """One chunk, presented with the structural path that gives it context.

    The path matters: an article about notice periods reads differently under
    "Rozdział II — Umowa o pracę" than it would in isolation, and the chunk text
    alone often does not repeat what its chapter already established.
    """
    location = f"{citation} ({path})" if path else citation
    return f"Pytanie użytkownika:\n{question}\n\nFragment — {location}:\n{content}"


def answer_user_prompt(question: str, passages: list[str]) -> str:
    joined = "\n\n---\n\n".join(passages)
    return (
        f"Pytanie:\n{question}\n\n"
        f"Fragmenty Kodeksu pracy:\n\n{joined}\n\n"
        "Odpowiedz na pytanie, powołując się na numery artykułów."
    )
