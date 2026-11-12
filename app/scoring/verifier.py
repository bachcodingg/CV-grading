"""[6] Evidence verifier (PLAN.md §3 [6]).

The matcher's evidence is a list of chunk IDs. Before anything is scored,
this module checks, independently of how the evidence was chosen:

1. every cited chunk ID exists;
2. every cited chunk's text really is in the user's original CV. Chunk text
   is the redacted copy, sometimes rebuilt from parsed fields ("Engineer,
   Acme (2021-01 – present)"), so the check compares words with the chunk's
   source span; chunks without a span are located in the CV by fuzzy search
   (rapidfuzz, ≥ ``min_quote_score``);
3. **no evidence, no points**: a positive label left with no verified
   evidence becomes "none" (except a deterministic fact that never cited a
   chunk, such as "the CV itself is written in English");
4. **sanity rule**: a judge "meets"/"exceeds" whose cited chunks all have a
   very low retrieval / reranker score and contain none of the requirement's
   keywords is downgraded to "partial" with low confidence;
5. no label is above the ceiling computed in code (``cap``).

``find_quote`` does the same check for free text (rewrite suggestions of the
coach): a quote counts only if it fuzzy-matches the CV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.config import VerifierConfig, get_config
from app.extraction.schemas import Span
from app.matching.schemas import LABEL_RATING, Chunk, Label, MatchResult, RequirementMatch, min_label
from app.privacy.redactor import PLACEHOLDER

_PLACEHOLDER_RE = re.compile("|".join(re.escape(p) for p in sorted(set(PLACEHOLDER.values()), key=len, reverse=True)))
_TOKEN = re.compile(r"[0-9a-z]+")
# Dates written by build_chunks as "2021-01"; the CV may say "Jan 2021" or "01/2021". Keep the year.
_YEAR_MONTH = re.compile(r"\b(\d{4})-\d{2}\b")
# Words build_chunks adds that the CV may not contain in that form.
_GLUE = {"in", "of", "the", "and", "at", "grade", "present", "project"}
# Kinds whose text build_chunks assembles from parsed fields (not copied verbatim).
_BUILT = {"experience", "education", "skills", "project", "certification", "language"}
_PREFIXED = {"skills", "language"}  # "Tools: Docker, Git" / "Languages: English (C2)"


@dataclass
class ChunkCheck:
    chunk_id: int
    verified: bool
    how: str  # "span": words found in its source span; "located": found by fuzzy search; "failed"
    score: float  # word coverage or fuzzy score, 0-100
    span: Span | None = None  # where the chunk is in the original CV


@dataclass
class QuoteMatch:
    span: Span
    score: float


@dataclass
class Issue:
    requirement_id: str
    code: str  # INVALID_CHUNK, UNVERIFIED_EVIDENCE, NO_EVIDENCE, LOW_SUPPORT, CAP_EXCEEDED
    message: str
    label_before: Label
    label_after: Label


@dataclass
class Verification:
    result: MatchResult  # a verified copy: unverifiable evidence removed, labels adjusted
    chunks: dict[int, ChunkCheck] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    evidence_checked: int = 0  # evidence items cited by the matcher
    evidence_verified: int = 0  # ... that point at a real chunk found in the CV

    @property
    def faithfulness(self) -> float | None:
        """Share of cited evidence that verified (PLAN.md §7: target > 98%)."""
        return self.evidence_verified / self.evidence_checked if self.evidence_checked else None

    @property
    def changed(self) -> list[Issue]:
        """Issues that changed a label."""
        return [i for i in self.issues if i.label_before != i.label_after]

    def issues_for(self, req_id: str) -> list[Issue]:
        return [i for i in self.issues if i.requirement_id.lower() == req_id.lower()]


# ------------------------------------------------------------------ text helpers


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _content(chunk: Chunk) -> str:
    """The part of a chunk's text that must be found in the CV."""
    text = _PLACEHOLDER_RE.sub(" ", chunk.text)
    if chunk.kind in _BUILT:
        text = _YEAR_MONTH.sub(r"\1", text)
    if chunk.kind in _PREFIXED and ": " in text:
        text = text.split(": ", 1)[1]
    return text


def _normalized(text: str) -> tuple[str, list[int]]:
    """Lowercased text with whitespace runs collapsed, and the original index of every character."""
    out: list[str] = []
    index: list[int] = []
    space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if out and not space:
                out.append(" ")
                index.append(i)
            space = True
            continue
        space = False
        low = ch.lower()
        out.append(low if len(low) == 1 else ch)
        index.append(i)
    return "".join(out), index


def _same_facts(piece: str, window: str, min_coverage: float) -> bool:
    """A fuzzy hit still has to keep the facts: every number exactly, and most of the words."""
    have = set(_tokens(window))
    want = _tokens(piece)
    if any(t not in have for t in want if any(ch.isdigit() for ch in t)):
        return False  # "400 services" is not "40 services"
    words = set(want)
    return not words or len(words & have) / len(words) >= min_coverage


def find_quote(quote: str, original_text: str, min_score: float | None = None,
               cfg: VerifierConfig | None = None) -> QuoteMatch | None:
    """Where ``quote`` appears in the CV (fuzzy, case and whitespace insensitive), or ``None``.

    A typo is tolerated; a changed number or a swapped tool is not. Redaction
    placeholders in the quote are skipped, so text from the grader's redacted
    copy can be found in the user's original CV.
    """
    cfg = cfg or get_config().verifier
    min_score = cfg.min_quote_score if min_score is None else min_score
    pieces = [" ".join(p.split()).lower() for p in _PLACEHOLDER_RE.split(quote)]
    pieces = [p for p in pieces if p]
    if not pieces or not original_text:
        return None
    hay, index = _normalized(original_text)
    found: list[QuoteMatch] = []
    for piece in pieces:
        if len(piece) < 4:  # too short to fuzzy-match meaningfully
            pos = hay.find(piece)
            if pos < 0:
                return None
            found.append(QuoteMatch(Span(start=index[pos], end=index[pos + len(piece) - 1] + 1), 100.0))
            continue
        al = fuzz.partial_ratio_alignment(piece, hay)
        if al is None or al.score < min_score or al.dest_end <= al.dest_start:
            return None
        # Widen to whole words so a hit ending mid-word ("4" of "40") is compared fairly.
        lo, hi = al.dest_start, al.dest_end
        while lo > 0 and hay[lo - 1].isalnum():
            lo -= 1
        while hi < len(hay) and hay[hi].isalnum():
            hi += 1
        if not _same_facts(piece, hay[lo:hi], cfg.min_word_coverage):
            return None
        found.append(QuoteMatch(Span(start=index[al.dest_start], end=index[al.dest_end - 1] + 1), al.score))
    span = Span(start=min(f.span.start for f in found), end=max(f.span.end for f in found))
    return QuoteMatch(span, round(min(f.score for f in found), 1))


def check_chunk(chunk: Chunk, original_text: str, cfg: VerifierConfig | None = None) -> ChunkCheck:
    """Is this chunk's text in the original CV?"""
    cfg = cfg or get_config().verifier
    words = set(_tokens(_content(chunk))) - _GLUE
    if not words:
        return ChunkCheck(chunk.id, False, "failed", 0.0)
    span = chunk.span
    if span is not None and 0 <= span.start < span.end <= len(original_text):
        source = set(_tokens(span.of(original_text)))
        coverage = len(words & source) / len(words)
        if coverage >= cfg.min_word_coverage:
            return ChunkCheck(chunk.id, True, "span", round(coverage * 100, 1), span)
    # No usable span, or the words are not where the span says: search the whole CV.
    if chunk.kind in _BUILT:
        # Rebuilt text is not verbatim; it verifies only if its words cluster in one place in the CV.
        return _locate_words(chunk, words, original_text, cfg)
    match = find_quote(chunk.text, original_text, cfg.min_quote_score, cfg)
    if match is None:
        return ChunkCheck(chunk.id, False, "failed", 0.0)
    return ChunkCheck(chunk.id, True, "located", match.score, match.span)


def _locate_words(chunk: Chunk, words: set[str], text: str, cfg: VerifierConfig) -> ChunkCheck:
    """Find a window of the CV (about the chunk's length, 3× slack) that holds enough of its words."""
    positions = [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text.lower())]
    width = max(len(_content(chunk)) * 3, 80)
    best, best_span = 0.0, None
    for i, (start, _, _) in enumerate(positions):
        seen: set[str] = set()
        end = start
        for s, e, tok in positions[i:]:
            if s - start > width:
                break
            if tok in words:
                seen.add(tok)
                end = e
        cov = len(seen) / len(words)
        if cov > best:
            best, best_span = cov, Span(start=start, end=end)
    if best >= cfg.min_word_coverage and best_span is not None:
        return ChunkCheck(chunk.id, True, "located", round(best * 100, 1), best_span)
    return ChunkCheck(chunk.id, False, "failed", round(best * 100, 1))


# ------------------------------------------------------------------ verification


def _scorer(result: MatchResult) -> str:
    return result.retriever.split("+")[-1]


def _set(m: RequirementMatch, label: Label, note: str) -> None:
    m.label = label
    m.confidence = "low"
    m.reason = f"{m.reason} [verifier: {note}]" if m.reason else f"[verifier: {note}]"


def verify_matches(result: MatchResult, original_text: str, cfg: VerifierConfig | None = None) -> Verification:
    """Check every piece of evidence against the original CV and return a corrected copy of ``result``."""
    cfg = cfg or get_config().verifier
    out = result.model_copy(deep=True)
    ver = Verification(result=out)
    for chunk in out.chunks:
        check = check_chunk(chunk, original_text, cfg)
        ver.chunks[chunk.id] = check
        if check.verified and check.span is not None:
            chunk.span = check.span  # quotes now always come from the user's own text
    low = cfg.low_support.get(_scorer(out), 0.0)

    for m in out.matches:
        before = m.label
        cited = bool(m.evidence)
        kept = []
        for ev in m.evidence:
            ver.evidence_checked += 1
            valid = 1 <= ev.chunk_id <= len(out.chunks) and out.chunks[ev.chunk_id - 1].id == ev.chunk_id
            if not valid:
                ver.issues.append(Issue(m.requirement_id, "INVALID_CHUNK",
                                        f"cited chunk {ev.chunk_id} does not exist", before, before))
            elif not ver.chunks[ev.chunk_id].verified:
                ver.issues.append(Issue(m.requirement_id, "UNVERIFIED_EVIDENCE",
                                        f"chunk {ev.chunk_id} was not found in the CV text; ignored", before, before))
            else:
                ver.evidence_verified += 1
                kept.append(ev)
        m.evidence = kept

        # A deterministic label that cites nothing rests on a fact computed from the whole CV
        # ("the CV itself is written in English"); every other positive label needs verified text.
        needs_text = cited or m.method != "deterministic"
        if m.rating > 0 and not kept and needs_text:
            _set(m, "none", "no verified evidence, no points")
            ver.issues.append(Issue(m.requirement_id, "NO_EVIDENCE",
                                    f"'{before}' had no evidence found in the CV; rated 'none'", before, "none"))
        elif m.method == "llm" and m.rating >= LABEL_RATING["meets"]:
            weak = all((ev.score or 0.0) < low and not ev.terms for ev in kept)
            if weak:
                label = min_label("partial", m.label)
                _set(m, label, "cited excerpts barely relate to the requirement")
                ver.issues.append(Issue(m.requirement_id, "LOW_SUPPORT",
                                        f"judge said '{before}' but every cited excerpt scored below {low} "
                                        f"({_scorer(out)}); downgraded", before, label))
        capped = min_label(m.label, m.cap)
        if capped != m.label:
            prev = m.label
            _set(m, capped, f"capped at '{capped}'")
            ver.issues.append(Issue(m.requirement_id, "CAP_EXCEEDED",
                                    f"'{prev}' is above the ceiling computed in code", prev, capped))
    return ver
