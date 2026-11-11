"""Cut a structured CV into numbered evidence chunks (PLAN.md §3 [5]).

A chunk is the smallest piece of the CV that can stand as evidence on its
own: one bullet, one job header, one education entry, one skills line, one
certificate. The judge picks chunk IDs instead of writing quotes, so it
cannot invent evidence.

Build chunks from the **redacted** ``Candidate`` (``RedactedCV.candidate``):
its text fields are what the grader may see, and its spans still point into
the original ``Document.text`` for quoting.
"""

from __future__ import annotations

from app.extraction.schemas import Candidate, PartialDate, Span
from app.matching.schemas import Chunk, ChunkKind


def _range(start: PartialDate | None, end: PartialDate | None, current: bool) -> str:
    if start is None and end is None and not current:
        return ""
    return f"{start or '?'} – {'present' if current else (end or '?')}"


def _union(spans) -> Span | None:
    spans = [s for s in spans if s is not None]
    if not spans:
        return None
    return Span(start=min(s.start for s in spans), end=max(s.end for s in spans))


def _join(*parts: str | None, sep: str = ", ") -> str:
    return sep.join(p.strip() for p in parts if p and p.strip())


class _Builder:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, kind: ChunkKind, text: str | None, *, span: Span | None = None, context: str = "",
            section: str = "", entry: int | None = None) -> None:
        text = " ".join((text or "").split())
        if not text:
            return
        self.chunks.append(Chunk(id=len(self.chunks) + 1, kind=kind, text=text, context=context, span=span,
                                 section=section, entry=entry))


def build_chunks(cand: Candidate) -> list[Chunk]:
    b = _Builder()
    b.add("summary", cand.summary, section="summary")

    for i, e in enumerate(cand.experience):
        dates = _range(e.start, e.end, e.current)
        header = _join(e.title, e.org, e.location) + (f" ({dates})" if dates else "")
        fs = e.field_spans
        b.add("experience", header, span=_union([fs.get(k) for k in ("title", "org", "location", "dates")]) or e.span,
              section="experience", entry=i)
        for bullet in e.bullets:
            b.add("bullet", bullet.text, span=bullet.span, context=header, section="experience", entry=i)

    for i, e in enumerate(cand.education):
        dates = _range(e.start, e.end, e.current)
        degree = _join(e.degree, e.field, sep=" in ") if e.degree else e.field
        text = _join(degree, e.institution) + (f" ({dates})" if dates else "")
        if e.grade:
            text += f", grade {e.grade}"
        b.add("education", text, span=e.span, section="education", entry=i)
        for d in e.details:
            b.add("education", d.text, span=d.span, context=text, section="education", entry=i)

    # One chunk per skills line ("Tools: Docker, Kubernetes"). Skills found by NER in bullets are
    # already inside their bullet's chunk.
    groups: dict[str, list] = {}
    for s in cand.skills:
        if s.provenance.get("name") == "ner":
            continue
        groups.setdefault(s.category or "Skills", []).append(s)
    for category, skills in groups.items():
        names = ", ".join(s.name + (f" ({s.level})" if s.level else "") for s in skills)
        b.add("skills", f"{category}: {names}", span=_union([s.span for s in skills]), section="skills")

    for i, p in enumerate(cand.projects):
        dates = _range(p.start, p.end, p.current)
        header = (p.name or "Project") + (f" [{', '.join(p.technologies)}]" if p.technologies else "")
        header += f" ({dates})" if dates else ""
        b.add("project", header, span=p.field_spans.get("name") or p.span, section="projects", entry=i)
        for bullet in p.bullets:
            b.add("project", bullet.text, span=bullet.span, context=header, section="projects", entry=i)

    for i, c in enumerate(cand.certifications):
        text = c.name + (f" — {c.issuer}" if c.issuer and c.issuer.lower() not in c.name.lower() else "")
        text += f" ({c.date})" if c.date else ""
        b.add("certification", text, span=c.span, section="certifications", entry=i)

    if cand.languages:
        text = ", ".join(lang.name + (f" ({lang.level})" if lang.level else "") for lang in cand.languages)
        b.add("language", f"Languages: {text}", span=_union([lang.span for lang in cand.languages]),
              section="languages")

    for i, pub in enumerate(cand.publications):
        b.add("publication", pub.text, span=pub.span, section="publications", entry=i)

    for i, o in enumerate(cand.other_sections):
        if o.kind in ("references", "interests"):
            continue
        # Line spans are exact only while the text has the original length (nothing redacted in it).
        exact = o.span is not None and o.span.end - o.span.start == len(o.text)
        pos = 0
        for line in o.text.split("\n"):
            start = pos
            pos += len(line) + 1
            if not line.strip():
                continue
            lead = len(line) - len(line.lstrip())
            span = (Span(start=o.span.start + start + lead, end=o.span.start + start + len(line.rstrip()))
                    if exact else o.span)
            b.add("other", line, span=span, context=o.heading, section=o.kind, entry=i)
    return b.chunks
