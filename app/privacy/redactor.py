"""Redaction: the copy of the CV that the grader sees (PLAN.md §3 [3]).

Every personal item is replaced by a fixed placeholder (``[NAME]``,
``[DATE OF BIRTH]``...). Placeholders do not depend on the value, so two
CVs that differ only in name, gender, age or origin redact to the same
text; the bias audit ([9]) relies on that.

Offsets change when text is replaced. ``RedactedCV`` keeps the mapping so
evidence found in the redacted text can be shown to the user as the
original CV text (``to_original`` / ``original_text``).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from app.extraction.schemas import Bullet, Candidate, Contact, Span
from app.privacy.detectors import PRIORITY, PersonalData, PersonalItem

PLACEHOLDER: dict[str, str] = {
    "name": "[NAME]", "email": "[EMAIL]", "phone": "[PHONE]", "link": "[LINK]", "address": "[ADDRESS]",
    "location": "[LOCATION]", "date_of_birth": "[DATE OF BIRTH]", "age": "[AGE]",
    "place_of_birth": "[PLACE OF BIRTH]", "gender": "[GENDER]", "pronouns": "[PRONOUNS]",
    "pronoun": "[PRONOUN]", "gender_term": "[GENDER TERM]", "nationality": "[NATIONALITY]",
    "ethnicity": "[ETHNICITY]", "marital_status": "[MARITAL STATUS]", "children": "[CHILDREN]",
    "religion": "[RELIGION]",
}


@dataclass
class Replacement:
    kind: str
    original: Span  # in Document.text
    redacted: Span  # in RedactedCV.text
    placeholder: str


@dataclass
class RedactedCV:
    text: str
    original_text: str
    replacements: list[Replacement] = field(default_factory=list)
    candidate: Candidate | None = None
    photo_removed: bool = False
    warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.replacements:
            out[r.kind] = out.get(r.kind, 0) + 1
        return out

    def to_original(self, pos: int, *, is_end: bool = False) -> int:
        """Map an offset in the redacted text back to ``Document.text``.

        A position inside a placeholder maps to the start of the original
        item, or to its end when ``is_end`` is set, so a span that touches a
        placeholder covers the whole original item.
        """
        starts = [r.redacted.start for r in self.replacements]
        i = bisect.bisect_right(starts, pos) - 1
        if i < 0:
            return pos
        r = self.replacements[i]
        if pos < r.redacted.end or (is_end and pos == r.redacted.end):
            if is_end and pos > r.redacted.start:
                return r.original.end
            return r.original.start
        return r.original.end + (pos - r.redacted.end)

    def original_span(self, span: Span) -> Span:
        return Span(start=self.to_original(span.start), end=self.to_original(span.end, is_end=True))

    def original_of(self, span: Span) -> str:
        """The user's own CV text behind a span of the redacted text."""
        return self.original_span(span).of(self.original_text)


def _choose(items: list[PersonalItem]) -> list[PersonalItem]:
    """Non-overlapping items: longer first, then by kind priority."""
    chosen: list[PersonalItem] = []
    for it in sorted((i for i in items if i.span and i.kind in PLACEHOLDER),
                     key=lambda i: (-(i.span.end - i.span.start), PRIORITY.index(i.kind), i.span.start)):
        if all(it.span.end <= c.span.start or it.span.start >= c.span.end for c in chosen):
            chosen.append(it)
    return sorted(chosen, key=lambda i: i.span.start)


def redact_text(text: str, items: list[PersonalItem]) -> tuple[str, list[Replacement]]:
    out: list[str] = []
    reps: list[Replacement] = []
    pos = 0
    length = 0
    for it in _choose(items):
        chunk = text[pos: it.span.start]
        out.append(chunk)
        length += len(chunk)
        ph = PLACEHOLDER[it.kind]
        out.append(ph)
        reps.append(Replacement(kind=it.kind, original=it.span, redacted=Span(start=length, end=length + len(ph)),
                                placeholder=ph))
        length += len(ph)
        pos = it.span.end
    out.append(text[pos:])
    return "".join(out), reps


# ------------------------------------------------------------------ Candidate


class _SpanRedactor:
    """Redacts strings of the Candidate that point back into the document."""

    def __init__(self, doc_text: str, chosen: list[PersonalItem]):
        self.doc_text = doc_text
        self.chosen = chosen
        # Values found at known spans can be redacted exactly; anything else
        # (e.g. the summary, joined from several lines) by its personal pieces.
        pieces = sorted({i.text for i in chosen if len(i.text) >= 2}, key=len, reverse=True)
        self.by_text = {i.text: PLACEHOLDER[i.kind] for i in sorted(chosen, key=lambda i: PRIORITY.index(i.kind),
                                                                      reverse=True)}
        self.piece_re = re.compile("|".join(rf"(?<!\w){re.escape(p)}(?!\w)" for p in pieces)) if pieces else None

    def at(self, value: str | None, span: Span | None) -> str | None:
        if not value:
            return value
        if span is not None and span.of(self.doc_text) == value:
            inner = [i for i in self.chosen if span.start <= i.span.start and i.span.end <= span.end]
            if not inner:
                return value
            shifted = [PersonalItem(kind=i.kind, text=i.text, span=Span(start=i.span.start - span.start,
                                                                        end=i.span.end - span.start)) for i in inner]
            return redact_text(value, shifted)[0]
        if self.piece_re is None:
            return value
        return self.piece_re.sub(lambda m: self.by_text[m.group()], value)


def redact_candidate(cand: Candidate, doc_text: str, items: list[PersonalItem], *, has_name: bool = False) -> Candidate:
    """A deep copy with contact details replaced and personal data cut out of every text field.

    Spans keep pointing into the *original* ``Document.text`` so evidence can
    still be cited from the user's own CV.
    """
    red = cand.model_copy(deep=True)
    r = _SpanRedactor(doc_text, _choose(items))

    c = red.contact
    red.contact = Contact(
        name=PLACEHOLDER["name"] if c.name or has_name else None,
        email=PLACEHOLDER["email"] if c.email else None,
        phone=PLACEHOLDER["phone"] if c.phone else None,
        location=PLACEHOLDER["location"] if c.location else None,
        links=[PLACEHOLDER["link"]] * len(c.links),
        span=c.span,
    )
    red.summary = r.at(red.summary, None)

    def bullets(bs: list[Bullet]) -> None:
        for b in bs:
            b.text = r.at(b.text, b.span) or b.text

    def fields(entry, names: tuple[str, ...]) -> None:
        for name in names:
            value = getattr(entry, name)
            if isinstance(value, str):
                setattr(entry, name, r.at(value, entry.field_spans.get(name)))

    for e in red.experience:
        fields(e, ("title", "org", "location", "dates_text"))
        bullets(e.bullets)
    for e in red.education:
        fields(e, ("degree", "field", "institution", "grade"))
        bullets(e.details)
    for p in red.projects:
        fields(p, ("name",))
        p.technologies = [r.at(t, None) or t for t in p.technologies]
        p.links = [PLACEHOLDER["link"]] * len(p.links) if p.links else []
        bullets(p.bullets)
    for s in red.skills:
        s.name = r.at(s.name, s.span if s.span and s.span.of(doc_text) == s.name else s.field_spans.get("name")) or s.name
    for x in red.certifications:
        fields(x, ("name", "issuer"))
    for pub in red.publications:
        pub.text = r.at(pub.text, pub.span) or pub.text
    for o in red.other_sections:
        o.text = r.at(o.text, o.span) or o.text
    return red


def redact(doc_text: str, found: PersonalData, cand: Candidate | None = None) -> RedactedCV:
    text, reps = redact_text(doc_text, found.items)
    out = RedactedCV(text=text, original_text=doc_text, replacements=reps,
                     photo_removed=bool(found.of_kind("photo")), warnings=list(found.warnings))
    if cand is not None:
        has_name = any(i.kind == "name" and i.section in ("header", "contact") for i in found.items)
        out.candidate = redact_candidate(cand, doc_text, found.items, has_name=has_name)
    return out
