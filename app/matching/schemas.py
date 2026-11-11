"""Matcher data model (PLAN.md §3 [5]).

The CV is cut into numbered **chunks** (one bullet, one job header, one skills
line, ...). Chunk text is what the grader sees (the redacted copy); the span
points into the user's original ``Document.text`` so evidence can be quoted
from their own CV.

Each requirement gets one ``RequirementMatch``: a label on the 0–4 rubric,
the method that decided it, and the chunk IDs that are its evidence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.extraction.schemas import Span

Label = Literal["none", "tangential", "partial", "meets", "exceeds"]
LABELS: tuple[Label, ...] = ("none", "tangential", "partial", "meets", "exceeds")
LABEL_RATING: dict[str, int] = {label: i for i, label in enumerate(LABELS)}

# deterministic: decided in code from dates, degrees, languages, certificates.
# keyword: decided by exact / alias skill matches. llm: the judge decided.
# retrieval: no judge available; a cautious label from retrieval scores.
Method = Literal["deterministic", "keyword", "llm", "retrieval"]
Confidence = Literal["high", "medium", "low"]

ChunkKind = Literal[
    "summary", "experience", "bullet", "education", "skills", "project", "certification", "language",
    "publication", "other",
]


def min_label(a: Label, b: Label | None) -> Label:
    """The lower of two labels; ``None`` means no ceiling."""
    if b is None:
        return a
    return a if LABEL_RATING[a] <= LABEL_RATING[b] else b


class Chunk(BaseModel):
    id: int  # 1-based, stable for one CV
    kind: ChunkKind
    text: str  # as the grader sees it (redacted)
    context: str = ""  # the entry it belongs to, e.g. "Senior Software Engineer, Acme GmbH (2021-01 – present)"
    span: Span | None = None  # in the original Document.text
    section: str = ""  # experience, education, skills, ...
    entry: int | None = None  # index into the Candidate list of that section

    @property
    def search_text(self) -> str:
        """Text used for retrieval. Without the entry context: prefixing "Engineer, Acme (2018 – 2020):"
        to a bullet lowers bge-reranker scores for the bullet's own content (0.66 -> 0.14 in a test)."""
        return self.text

    def quote(self, original_text: str | None = None) -> str:
        """The user's own words when the original text is at hand, else the grader's copy."""
        if original_text is not None and self.span is not None:
            return self.span.of(original_text)
        return self.text


class Evidence(BaseModel):
    chunk_id: int
    score: float | None = None  # retrieval / reranker score in [0, 1], when one was computed
    terms: list[str] = Field(default_factory=list)  # requirement keywords found in this chunk


class KeywordHit(BaseModel):
    keyword: str  # as written in the posting
    canonical: str | None = None
    found: bool = False
    # The posting's exact wording appears in the CV. False with found=True means the CV uses
    # another name ("K8s" for "Kubernetes"): fine for a person, a risk for a literal ATS filter.
    verbatim: bool = False
    chunk_ids: list[int] = Field(default_factory=list)
    cv_forms: list[str] = Field(default_factory=list)  # how the CV writes it


class RequirementMatch(BaseModel):
    requirement_id: str
    label: Label
    method: Method
    confidence: Confidence
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)
    keywords: list[KeywordHit] = Field(default_factory=list)
    # A ceiling set in code (too few years, a lower degree); the judge cannot rate above it.
    cap: Label | None = None
    facts: list[str] = Field(default_factory=list)  # computed facts shown to the judge and the user
    candidates: list[int] = Field(default_factory=list)  # chunk IDs offered to the judge, best first

    @property
    def rating(self) -> int:
        return LABEL_RATING[self.label]


class MatchResult(BaseModel):
    matches: list[RequirementMatch] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    retriever: str = "lexical"
    llm_calls: int = 0
    warnings: list[str] = Field(default_factory=list)

    def match(self, req_id: str) -> RequirementMatch:
        for m in self.matches:
            if m.requirement_id.lower() == req_id.lower():
                return m
        raise KeyError(req_id)

    def chunk(self, chunk_id: int) -> Chunk:
        return self.chunks[chunk_id - 1]

    def by_method(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.matches:
            out[m.method] = out.get(m.method, 0) + 1
        return out
