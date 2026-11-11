"""Step 1 of the matcher: checks computed in code (PLAN.md §3 [5]).

- **years**: minimum years against the merged years of experience, or against
  the jobs that mention the requirement's keywords ("3+ years of Python").
- **education**: lowest accepted degree level and field of study.
- **language**: spoken languages and CEFR / wording levels.
- **certification**: certificate names, fuzzy-matched.

A check either decides the requirement (``decisive``) or sets a ceiling
(``cap``) that the judge cannot exceed: someone with 2 of 5 required years
cannot "meet" the requirement however good the bullets read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from app.config import MatchingConfig
from app.extraction.dates import total_years
from app.extraction.schemas import Candidate, Requirement
from app.matching.keywords import keywords_in_chunks
from app.matching.schemas import LABEL_RATING, Chunk, Label

# ------------------------------------------------------------------ degrees

DEGREE_LEVEL = {"secondary": 0, "bachelor": 1, "master": 2, "phd": 3}
_CV_LEVELS: list[tuple[str, re.Pattern]] = [
    ("phd", re.compile(r"\bph\.?\s?d|\bdoctor|\bdr\.|\bd\.?\s?phil", re.IGNORECASE)),
    ("master", re.compile(
        r"\bm\.\s?sc|\bmsc\b|\bmaster|\bm\.\s?a\.|\bm\.\s?s\.|\bm\.\s?eng|\bll\.?\s?m\b|\bm\.?\s?tech\b|\bdiplom"
        r"|\bdipl\.|\bmagister|\bstaatsexamen|\b(?:MS|MA|MEng|MBA|MPhil|MRes)\b", re.IGNORECASE)),
    ("bachelor", re.compile(
        r"\bb\.\s?sc|\bbsc\b|\bbachelor|\bb\.\s?a\.|\bb\.\s?s\.|\bb\.\s?eng|\bll\.?\s?b\b|\bb\.?\s?tech\b"
        r"|\b(?:BS|BA|BEng|BBA)\b", re.IGNORECASE)),
    ("secondary", re.compile(r"\babitur|\ba-levels?\b|\bhigh school|\bged\b|\bassociate", re.IGNORECASE)),
]
_LEVEL_NAME = {"bachelor": "bachelor's", "master": "master's", "phd": "PhD", "secondary": "school-leaving"}

# Small, explicit list of field names that mean the same thing on CVs and postings.
FIELD_SYNONYMS: dict[str, set[str]] = {
    "computer science": {"informatics", "informatik", "computing", "cs", "computer engineering",
                         "software engineering", "information technology", "wirtschaftsinformatik"},
    "electrical engineering": {"electronics", "elektrotechnik", "ee"},
    "mechanical engineering": {"maschinenbau"},
    "mathematics": {"maths", "math", "applied mathematics", "mathematik"},
    "physics": {"physik"},
    "economics": {"volkswirtschaftslehre", "vwl"},
    "business administration": {"business", "management", "bwl", "betriebswirtschaftslehre", "mba"},
    "statistics": {"data science"},
}
_RELATED = re.compile(r"\b(?:related|similar|comparable|equivalent|relevant|quantitative|technical|stem)\b"
                      r"(?:\s+(?:field|discipline|subject|area|degree|qualification))?", re.IGNORECASE)
_EQUIVALENT_EXPERIENCE = re.compile(r"\bor\s+(?:\w+\s+){0,2}(?:equivalent|comparable|relevant)\s+"
                                    r"(?:\w+\s+){0,2}experience\b", re.IGNORECASE)


def degree_level(text: str | None) -> str | None:
    if not text:
        return None
    for level, pat in _CV_LEVELS:
        if pat.search(text):
            return level
    return None


def wanted_fields(text: str) -> tuple[list[str], bool]:
    """Fields of study a requirement names, and whether related fields are accepted."""
    related = bool(_RELATED.search(text))
    m = re.search(r"\b(?:in|of)\s+(.+)$", text)
    if not m:
        return [], related
    rest = _EQUIVALENT_EXPERIENCE.sub("", m.group(1))
    rest = re.sub(r"\(.*?\)", " ", rest)
    rest = _RELATED.sub(" ", rest)
    rest = re.sub(r"\b(?:is|are)\s+(?:required|preferred|a plus|desired)\b.*$", " ", rest, flags=re.IGNORECASE)
    fields = []
    for part in re.split(r",|/|\bor\b|\band\b|;", rest):
        part = re.sub(r"\b(?:a|an|the|field|fields|discipline|degree|similar|area|subject|any)\b", " ", part,
                      flags=re.IGNORECASE)
        part = " ".join(part.split()).strip(" .")
        if len(part) >= 2 and re.search(r"[A-Za-z]{2}", part):
            fields.append(part)
    return fields, related


def _canon_field(name: str) -> str:
    low = " ".join(name.lower().split())
    for canon, alts in FIELD_SYNONYMS.items():
        if low == canon or low in alts:
            return canon
    return low


def field_matches(wanted: str, have: str, min_score: float) -> bool:
    a, b = _canon_field(wanted), _canon_field(have)
    if a == b:
        return True
    if fuzz.token_set_ratio(a, b) >= min_score:
        return True
    # "M.Sc. Computer Science and Engineering" contains a synonym of the wanted field.
    return any(_canon_field(alt) == a for alt in re.split(r"\s+(?:and|&)\s+|,\s*", have.lower()))


# ------------------------------------------------------------------ languages

_LANG_ALIASES = {"deutsch": "german", "englisch": "english", "französisch": "french", "spanisch": "spanish",
                 "mandarin": "chinese", "farsi": "persian", "filipino": "tagalog"}
_CEFR_VALUE = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6}
_WORD_LEVEL: list[tuple[str, float]] = [
    (r"native|mother tongue|muttersprache|bilingual", 7),
    (r"business fluent|full professional|verhandlungssicher|fluent|fließend|fluency", 5),
    (r"professional(?: working)?|proficien\w*|advanced|excellent|very good|strong", 5),
    (r"upper[- ]intermediate", 4),
    (r"good|gut|intermediate|conversational|limited working|working knowledge", 3),
    (r"elementary|basic|grundkenntnisse", 2),
    (r"beginner", 1),
]
_CEFR_NAME = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2", 7: "native"}


def language_level(text: str | None) -> float | None:
    """CEFR-like value: A1=1 ... C2=6, native=7. A range ("B2/C1") counts as its lower end."""
    if not text:
        return None
    cefr = [_CEFR_VALUE[m.lower()] for m in re.findall(r"\b([ABCabc][12])\b", text)]
    if cefr:
        return float(min(cefr))
    for pat, value in _WORD_LEVEL:
        if re.search(rf"\b(?:{pat})\b", text, re.IGNORECASE):
            return value
    return None


def _lang_key(name: str) -> str:
    low = name.lower().strip()
    return _LANG_ALIASES.get(low, low)


def _level_name(v: float) -> str:
    return _CEFR_NAME.get(int(v), str(v))


# ------------------------------------------------------------------ years

_GENERIC_WORDS = re.compile(
    r"\b(?:years?|yrs?|months?|of|professional|experience|experienced|work|working|industry|relevant|practical|"
    r"hands-on|proven|at|least|minimum|min|plus|more|than|or|and|in|a|an|the|with|as|similar|role|roles|position|"
    r"positions|comparable|total|overall|full-time|post-graduate|postgraduate|required|preferred|ideally|must|is|"
    r"are|be|have|you|your|ideally|desired|strong|solid|good|which|we|expect|looking|for|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|over|up|to|some|several|prior|previous|demonstrated|track|record|nice|bonus|"
    r"would|great|job|career)\b",
    re.IGNORECASE,
)


def has_domain(text: str, years_text: str = "") -> bool:
    """True if the requirement names a field beyond 'N years of professional experience'."""
    rest = text.replace(years_text, " ") if years_text else text
    rest = re.sub(r"[\d.,+–\-()/:;]+", " ", rest)
    rest = _GENERIC_WORDS.sub(" ", rest)
    return bool(re.search(r"[A-Za-z]{3,}", rest))


def _periods(entries) -> list:
    return [(e.start, e.end, e.current) for e in entries if e.start is not None]


# ------------------------------------------------------------------ result


@dataclass
class Check:
    label: Label | None = None  # the label (decisive) or a fallback when nothing better decides
    decisive: bool = False
    cap: Label | None = None
    reason: str = ""
    chunk_ids: list[int] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    confidence: str = "high"


def _chunks_of(chunks: list[Chunk], section: str, entries: set[int] | None = None, kind: str | None = None) -> list[int]:
    return [c.id for c in chunks
            if c.section == section and (entries is None or c.entry in entries) and (kind is None or c.kind == kind)]


def check_years(req: Requirement, cand: Candidate, chunks: list[Chunk], today: date) -> Check | None:
    if req.min_years is None:
        return None
    need = req.min_years
    years_span = req.field_spans.get("min_years")
    years_text = req.text[years_span.start - req.span.start: years_span.end - req.span.start] \
        if years_span and req.span else ""
    terms = [k.canonical or k.text for k in req.keywords]
    out = Check()
    if terms:
        # Years in jobs whose header or bullets mention one of the keywords.
        hit_ids = set(keywords_in_chunks([c for c in chunks if c.section == "experience"], terms))
        entries = {c.entry for c in chunks if c.id in hit_ids and c.entry is not None}
        jobs = [e for i, e in enumerate(cand.experience) if i in entries]
        scope = f"in jobs that mention {', '.join(terms)}"
        if not _periods(jobs):
            # Listed (skills, projects) but no dated job mentions it: years cannot be attributed.
            out.facts.append(f"No dated job on the CV mentions {', '.join(terms)}; "
                             f"the requirement asks for {need:g} years.")
            out.cap, out.label, out.confidence = "partial", "tangential", "low"
            out.reason = f"{need:g}+ years required; no dated job mentions {', '.join(terms)}"
            return out
        have = total_years(_periods(jobs), today)
        out.chunk_ids = _chunks_of(chunks, "experience", entries, kind="experience")
    else:
        if cand.years_of_experience is None:
            out.facts.append("The CV gives no dates for its jobs, so years of experience cannot be computed.")
            out.cap, out.confidence = "partial", "low"
            out.reason = f"{need:g}+ years required; job dates missing on the CV"
            return out
        have = cand.years_of_experience
        scope = "in total (overlapping jobs counted once)"
        out.chunk_ids = _chunks_of(chunks, "experience", kind="experience")
    out.facts.append(f"Computed from the CV dates: {have:g} years of work experience {scope}; "
                     f"the requirement asks for {need:g}.")
    # "3+ years of Python" is fully checked here; "5+ years in backend development" needs the judge.
    domain = has_domain(_strip_terms(req.text, terms), years_text)
    if have >= need:
        out.reason = f"{have:g} years {scope} (≥ {need:g} required)"
        if not domain:
            out.label, out.decisive = "meets", True
        return out
    ratio = have / need if need else 1.0
    out.cap = "partial" if ratio >= 0.75 else "tangential" if have > 0 else "none"
    out.label = out.cap
    out.reason = f"{have:g} of {need:g} required years {scope}"
    # Years from keyword-matched jobs are a lower bound (a job may not name the skill): the judge sees them.
    out.decisive = out.cap == "none" or (not domain and not terms)
    return out


def _strip_terms(text: str, terms: list[str]) -> str:
    for t in terms:
        text = re.sub(re.escape(t), " ", text, flags=re.IGNORECASE)
    return text


def check_education(req: Requirement, cand: Candidate, chunks: list[Chunk], cfg: MatchingConfig) -> Check | None:
    if req.category != "education":
        return None
    out = Check()
    if not cand.education:
        out.label, out.decisive = "none", True
        out.reason, out.confidence = "no education entry found on the CV", "medium"
        return out
    need_level = req.degree_level
    need = DEGREE_LEVEL[need_level] if need_level else None
    fields, related = wanted_fields(req.text)
    equivalent = bool(_EQUIVALENT_EXPERIENCE.search(req.text))

    best: tuple | None = None
    for i, e in enumerate(cand.education):
        level = degree_level(e.degree)
        have_text = " ".join(x for x in (e.field, e.degree) if x)
        field_ok = not fields or any(field_matches(f, e.field or have_text, cfg.field_match_score) for f in fields)
        rank = (DEGREE_LEVEL.get(level, -1) if level else -1, field_ok, not e.current)
        if best is None or rank > best[0]:
            best = (rank, i, e, level, field_ok)
    _, i, e, level, field_ok = best
    out.chunk_ids = _chunks_of(chunks, "education", {i})
    degree = " ".join(x for x in (e.degree, ("in " + e.field) if e.field else None) if x)
    if level is None:
        out.reason, out.confidence = "degree level on the CV not recognised", "low"
        return out
    have = DEGREE_LEVEL[level]
    if need is not None:
        out.facts.append(f"Highest degree on the CV: {degree} ({_LEVEL_NAME[level]} level"
                         f"{', in progress' if e.current else ''}); the requirement accepts "
                         f"{_LEVEL_NAME[need_level]} or higher.")
    if need is not None and have < need:
        out.reason = f"{degree}: below the {_LEVEL_NAME[need_level]} degree asked for"
        if equivalent:  # "... or equivalent experience": the judge looks at the jobs instead
            out.reason += " (equivalent experience is accepted)"
            out.chunk_ids += [c.id for c in chunks if c.kind == "experience"]
            out.label, out.confidence = "tangential", "low"
        else:
            out.label = out.cap = "tangential" if have >= 1 else "none"
            out.decisive = True
        return out
    if e.current:
        out.label, out.decisive = "partial", field_ok
        out.reason = f"{degree} in progress" + (f" (expected {e.end})" if e.end else "")
        out.cap = "partial"
        return out
    if field_ok:
        out.label, out.decisive = "meets", True
        out.reason = f"{degree}" + (" matches the field asked for" if fields else "")
        return out
    # Right level, field not recognised: "a related field" is for the judge to decide.
    out.cap = "meets"
    out.label, out.confidence = ("partial" if related else "tangential"), "low"
    out.reason = f"{degree}: level fits, field differs from {', '.join(fields)}"
    return out


def check_language(req: Requirement, cand: Candidate, chunks: list[Chunk], cv_in_english: bool = True) -> Check | None:
    if req.category != "language" or not req.languages:
        return None
    need = language_level(req.language_level)
    have_by = {_lang_key(lang.name): lang for lang in cand.languages}
    lang_chunks = [c.id for c in chunks if c.kind == "language"]
    results: list[tuple[Label, str]] = []
    for name in req.languages:
        lang = have_by.get(_lang_key(name))
        if lang is None:
            if _lang_key(name) == "english" and cv_in_english:
                results.append(("partial", "English not listed, but the CV itself is written in English"))
            else:
                results.append(("none", f"{name} not listed on the CV"))
            continue
        have = language_level(lang.level)
        if need is None:
            results.append(("meets", f"{lang.name}" + (f" ({lang.level})" if lang.level else "") + " on the CV"))
        elif have is None:
            results.append(("partial", f"{lang.name} listed without a level; {_level_name(need)} asked for"))
        elif have >= need:
            results.append(("meets", f"{lang.name} {lang.level} (≥ {_level_name(need)})"))
        else:
            label: Label = "partial" if need - have <= 1 else "tangential"
            results.append((label, f"{lang.name} {lang.level}, below the {_level_name(need)} asked for"))
    pick = max if req.any_of else min
    label, _ = pick(results, key=lambda r: LABEL_RATING[r[0]])
    out = Check(label=label, decisive=True, chunk_ids=lang_chunks if label != "none" else [])
    out.reason = "; ".join(r for _, r in results)
    out.confidence = "medium" if any("without a level" in r or "not listed, but" in r for _, r in results) else "high"
    return out


def _cert_text(text: str) -> str:
    text = re.sub(r"\b(?:certificat(?:e|ion)s?|certified|is|a|an|the|required|preferred|plus|nice to have|"
                  r"ideally|or equivalent|or similar|such as|e\.g\.?|valid|current)\b", " ", text, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[^\w+#&/ -]", " ", text).split()).lower()


def check_certification(req: Requirement, cand: Candidate, chunks: list[Chunk], cfg: MatchingConfig) -> Check | None:
    if req.category != "certification" or not cand.certifications:
        return None
    want = _cert_text(req.text)
    if not want:
        return None
    best = max(((fuzz.token_set_ratio(want, _cert_text(c.name + " " + (c.issuer or ""))), i)
                for i, c in enumerate(cand.certifications)), default=(0, -1))
    if best[0] < cfg.cert_match_score:
        return None
    cert = cand.certifications[best[1]]
    return Check(label="meets", decisive=True, reason=f"certificate on the CV: {cert.name}",
                 chunk_ids=_chunks_of(chunks, "certifications", {best[1]}))


def run_checks(req: Requirement, cand: Candidate, chunks: list[Chunk], cfg: MatchingConfig, today: date) -> Check | None:
    """The first check that applies; years and education/language/certification never overlap in practice."""
    for check in (check_language(req, cand, chunks), check_education(req, cand, chunks, cfg),
                  check_certification(req, cand, chunks, cfg), check_years(req, cand, chunks, today)):
        if check is not None:
            return check
    return None
