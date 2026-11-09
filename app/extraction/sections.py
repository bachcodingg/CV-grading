"""Rule-based section splitter.

A line is a section heading when it is short and either matches a known
heading alias (after normalisation: case, numbering, decoration,
letter-spacing such as "E X P E R I E N C E", small typos) or looks like a
heading by layout (DOCX heading style, ALL CAPS, or bold and larger than
body text). Lines before the first heading form the ``header`` section.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.ingestion.document import Document, Line

SECTION_ALIASES: dict[str, list[str]] = {
    "summary": ["summary", "profile", "professional summary", "about me", "about", "objective",
                "career objective", "personal statement", "profile summary", "executive summary",
                "personal profile", "professional profile", "career summary", "overview"],
    "experience": ["experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "career history", "relevant experience",
                   "professional background", "internships", "internship experience",
                   "industry experience", "work experience and internships", "career",
                   "positions", "professional history", "experience and internships"],
    "education": ["education", "academic background", "education and training", "academic qualifications",
                  "qualifications", "studies", "academic history", "academic education", "education history"],
    "skills": ["skills", "technical skills", "core competencies", "competencies", "key skills", "expertise",
               "areas of expertise", "tech stack", "technologies", "tools", "tools and technologies",
               "it skills", "computer skills", "hard skills", "skills and tools", "skills and technologies",
               "technical expertise", "programming skills", "skill set", "skillset", "core skills",
               "soft skills", "skills and competencies", "technical proficiencies"],
    "projects": ["projects", "personal projects", "selected projects", "academic projects", "side projects",
                 "key projects", "project experience", "open source", "open source contributions",
                 "relevant projects", "university projects"],
    "certifications": ["certifications", "certificates", "licenses", "licenses and certifications",
                       "certifications and licenses", "courses", "training", "courses and certifications",
                       "professional development", "certifications and courses", "online courses",
                       "trainings", "continuing education"],
    "languages": ["languages", "language skills", "spoken languages", "language proficiency"],
    "publications": ["publications", "papers", "selected publications", "research", "talks",
                     "presentations", "publications and talks", "conference papers", "patents"],
    "awards": ["awards", "honors", "honours", "achievements", "awards and honors", "awards and honours",
               "scholarships", "honors and awards", "awards and achievements", "accomplishments"],
    "volunteering": ["volunteering", "volunteer experience", "volunteer work", "extracurricular activities",
                     "activities", "leadership", "leadership and activities", "extracurriculars",
                     "community involvement", "leadership experience"],
    "interests": ["interests", "hobbies", "hobbies and interests", "personal interests"],
    "references": ["references", "referees"],
    "contact": ["contact", "contact information", "contact details", "personal details",
                "personal information", "personal data", "details", "info", "contact info"],
}

# Single generic words that also occur as normal content; need a layout cue.
_WEAK = {"about", "career", "positions", "studies", "qualifications", "tools", "technologies", "training",
         "courses", "research", "talks", "patents", "activities", "leadership", "details", "info",
         "overview", "employment", "expertise", "competencies", "references", "contact"}

_ALIAS_TO_KIND = {alias: kind for kind, aliases in SECTION_ALIASES.items() for alias in aliases}


@dataclass
class Section:
    kind: str  # one of SECTION_ALIASES keys, "header" or "other"
    heading: str
    heading_line: Line | None
    lines: list[Line] = field(default_factory=list)

    @property
    def start(self) -> int:
        first = self.heading_line or (self.lines[0] if self.lines else None)
        return first.start if first else 0

    @property
    def end(self) -> int:
        last = self.lines[-1] if self.lines else self.heading_line
        return last.end if last else 0

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


def normalize_heading(text: str) -> str:
    t = text.strip()
    # "E X P E R I E N C E" / "W O R K  E X P E R I E N C E" -> words
    if re.fullmatch(r"(?:[A-Za-z] ){3,}[A-Za-z](?:\s{2,}(?:[A-Za-z] )*[A-Za-z])*", t):
        t = " ".join(w.replace(" ", "") for w in re.split(r"\s{2,}", t))
    t = t.lower().replace("&", " and ").replace("+", " and ")
    t = re.sub(r"^[\W\d_]+|[\W\d_]+$", "", t)  # numbering, bullets, dashes, colons
    t = re.sub(r"[^\w\s'-]", " ", t)
    return " ".join(t.split())


def _layout_cue(line: Line, body_size: float | None) -> bool:
    text = line.text.strip()
    if line.style and re.match(r"(heading|title)", line.style, re.IGNORECASE):
        return True
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) >= 4 and letters.isupper():
        return True
    bigger = body_size is not None and line.font_size is not None and line.font_size >= body_size + 1
    return line.bold and (bigger or len(text.split()) <= 3)


def _body_font_size(doc: Document) -> float | None:
    sizes = [ln.font_size for ln in doc.lines if ln.font_size]
    return statistics.median(sizes) if sizes else None


def classify_heading(line: Line, body_size: float | None = None) -> str | None:
    """Return the section kind if ``line`` is a heading, else None."""
    raw = line.text.strip()
    if not raw or len(raw) > 60 or line.list_item:
        return None
    norm = normalize_heading(raw)
    if not norm or len(norm.split()) > 6:
        return None
    if re.search(r"\d", raw.rstrip(".:)").lstrip("0123456789.) ")):
        return None  # dates, phone numbers, "Python 3"
    if raw.endswith((".", ",", ";")) and not raw.endswith("..."):
        return None
    cue = _layout_cue(line, body_size)

    kind = _ALIAS_TO_KIND.get(norm)
    if kind is None and len(norm) >= 6:
        best = max(_ALIAS_TO_KIND, key=lambda a: fuzz.ratio(norm, a))
        if fuzz.ratio(norm, best) >= 90:
            kind = _ALIAS_TO_KIND[best]
            norm = best
    if kind is None and cue and len(norm.split()) <= 5:
        # "Professional Experience & Internships", "Skills Summary"
        for alias in sorted(_ALIAS_TO_KIND, key=len, reverse=True):
            if alias not in _WEAK and re.search(rf"\b{re.escape(alias)}\b", norm):
                kind = _ALIAS_TO_KIND[alias]
                break
    if kind is not None:
        if norm in _WEAK and not cue:
            return None
        return kind
    # Unknown heading: needs a strong layout cue and must look like a title.
    if cue and len(norm.split()) <= 4 and not line.in_table:
        letters = re.sub(r"[^A-Za-z]", "", raw)
        is_caps = len(letters) >= 4 and letters.isupper()
        is_styled = bool(line.style and re.match(r"heading", line.style, re.IGNORECASE))
        bigger = body_size is not None and line.font_size is not None and line.font_size >= body_size + 1.5
        if is_caps or is_styled or (line.bold and bigger):
            return "other"
    return None


def split_sections(doc: Document) -> list[Section]:
    body_size = _body_font_size(doc)
    sections = [Section(kind="header", heading="", heading_line=None)]
    for line in doc.lines:
        kind = classify_heading(line, body_size)
        # The name at the very top is big and bold but is not a section.
        if kind == "other" and sections[-1].kind == "header" and not sections[-1].lines:
            kind = None
        if kind is not None:
            sections.append(Section(kind=kind, heading=line.text.strip(), heading_line=line))
        else:
            sections[-1].lines.append(line)
    return [s for s in sections if s.lines or s.heading_line is not None]


def sections_of(sections: list[Section], kind: str) -> list[Section]:
    return [s for s in sections if s.kind == kind]
