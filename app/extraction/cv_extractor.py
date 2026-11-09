"""Rule-based CV structurer (step 1–2 of PLAN.md §3 [2]).

Turns sections into entries using layout and wording cues only. Fields the
rules cannot decide are left empty (or listed in ``uncertain``) for the
optional NER and LLM steps. Every value is a substring of a CV line, so its
span in ``Document.text`` is exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.extraction.dates import find_date_range, find_single_date
from app.extraction.schemas import (
    Bullet,
    Candidate,
    Certification,
    Contact,
    DateRange,
    EducationEntry,
    ExperienceEntry,
    Language,
    OtherSection,
    Project,
    Publication,
    Skill,
    Span,
)
from app.extraction.sections import Section, classify_heading, split_sections
from app.ingestion.document import Document, Line
from app.patterns import EMAIL, PRIVATE_USE, URL, find_phones

# ------------------------------------------------------------------ cues

BULLET_RE = re.compile(r"^\s*(?:[•·▪●◦‣∙○■□➢➤►▶✓✔◆◇\-–—*>]|\d{1,2}[.)])\s+")
# Separators inside an entry header line: tab, pipe, middle dot, spaced dashes, " at ", " @ ".
SEP_RE = re.compile(r"\t+|\s{3,}|\s+[|·•]\s+|\s+[–—-]\s+|\s+@\s+")
# "Data Scientist at Globex" (but not "University of Texas at Austin").
AT_RE = re.compile(r"^(?P<title>.+?),?\s+at\s+(?P<org>[A-Z0-9].*)$")

TITLE_WORDS = re.compile(
    r"\b(?:engineer|developer|programmer|manager|intern|internship|analyst|scientist|researcher|consultant|"
    r"assistant|lead|head|director|designer|specialist|architect|administrator|officer|coordinator|technician|"
    r"tutor|teacher|student|werkstudent|trainee|associate|founder|co-founder|cto|ceo|cfo|coo|owner|devops|sre|"
    r"product owner|scrum master|fellow|professor|lecturer|postdoc|phd candidate|doctoral|editor|writer|"
    r"accountant|advisor|adviser|representative|agent|operator|supervisor|executive|partner|apprentice|mentor|"
    r"instructor|president|vice|chair|member|volunteer|freelancer|freelance|contractor|staff|principal|"
    r"strategist|marketer|recruiter|nurse|physician|lawyer|paralegal|auditor|controller|planner|"
    r"hiwi|research assistant|teaching assistant)s?\b",
    re.IGNORECASE,
)
ORG_STRONG = re.compile(
    r"\b(?:gmbh|ggmbh|e\.\s?v\.|inc\.?|ltd\.?|llc|llp|plc|corp\.?|corporation|co\.|company|s\.a\.|b\.v\.|"
    r"n\.v\.|university|universit[äa]t|institute|institut|college|school|hochschule|foundation|ministry|"
    r"hospital|clinic|association|society|fraunhofer|max planck|helmholtz|bank)(?!\w)",
    re.IGNORECASE,
)
ORG_SUFFIX = re.compile(r"\b(?:AG|SE|KG|UG|OHG|GbR|KGaA|AB|AS|ASA|Oy|SpA|SAS|SARL|Pty|BV|NV)\b")
ORG_WEAK = re.compile(
    r"\b(?:group|labs?|technologies|technology|solutions|systems|software|agency|studio|consulting|partners|"
    r"ventures|holding|capital|council|bureau|office)\b",
    re.IGNORECASE,
)


def org_score(text: str) -> int:
    return 2 * bool(ORG_STRONG.search(text) or ORG_SUFFIX.search(text)) + bool(ORG_WEAK.search(text))


def title_score(text: str) -> int:
    return 2 * bool(TITLE_WORDS.search(text)) - 2 * bool(ORG_STRONG.search(text) or ORG_SUFFIX.search(text))


INSTITUTION_WORDS = re.compile(
    r"\b(?:universit(?:y|ät|at|é|e|à|ad)|college|institute|institut|school|academy|akademie|hochschule|"
    r"polytechnic|politecnico|gymnasium|lyc[ée]e|eth|epfl|tum|lmu|rwth|kit|mit|tu\s+\w+|fh\s+\w+)\b",
    re.IGNORECASE,
)
DEGREE_RE = re.compile(
    r"(?:\bPh\.?\s?D\.?(?=\s|$|,)|\bDoctor(?:ate)?(?: of [A-Z][a-z]+)?|\bDr\.\s?(?:rer\.\s?nat\.|-Ing\.|phil\.)?"
    r"|\bM\.\s?Sc\.?|\bB\.\s?Sc\.?|\bMSc\b|\bBSc\b|\bM\.\s?A\.(?=\s|$|,)|\bB\.\s?A\.(?=\s|$|,)"
    r"|\bM\.\s?Eng\.?|\bB\.\s?Eng\.?|\bMEng\b|\bBEng\b|\bM\.\s?S\.(?=\s|$|,)|\bB\.\s?S\.(?=\s|$|,)"
    r"|\b(?:MS|BS|MA|BA)(?=\s+(?:in|of)\s)|\bMBA\b|\bLL\.?\s?M\.?|\bLL\.?\s?B\.?|\bB\.?\s?Tech\b|\bM\.?\s?Tech\b"
    r"|\bMaster(?:'s)?(?: of [A-Z][a-z]+(?: [A-Z][a-z]+)?| degree)?|\bBachelor(?:'s)?(?: of [A-Z][a-z]+(?: [A-Z][a-z]+)?| degree)?"
    r"|\bDiplom[a-z-]*|\bAbitur\b|\bA-Levels?\b|\bHigh School Diploma\b|\bAssociate(?:'s)? (?:Degree|of [A-Z][a-z]+)"
    r"|\bStaatsexamen\b|\bMagister\b|\bGED\b)",
)
GRADE_RE = re.compile(
    r"(?:(?:c?gpa|grade|final grade|note|abschlussnote|average|score)\s*[:\-]?\s*"
    r"(?P<num>\d(?:[.,]\d{1,2})?(?:\s*/\s*\d+(?:[.,]\d{1,2})?)?))"
    r"|(?P<words>first[- ]class(?: honou?rs)?|upper second|2:1|summa cum laude|magna cum laude|cum laude|"
    r"with (?:high )?distinction|with honou?rs|mit auszeichnung|sehr gut)",
    re.IGNORECASE,
)
COUNTRIES = {
    "germany", "deutschland", "austria", "österreich", "switzerland", "schweiz", "france", "italy", "spain",
    "portugal", "netherlands", "belgium", "luxembourg", "denmark", "sweden", "norway", "finland", "iceland",
    "ireland", "uk", "united kingdom", "england", "scotland", "wales", "poland", "czechia", "czech republic",
    "slovakia", "hungary", "romania", "bulgaria", "greece", "turkey", "türkiye", "ukraine", "russia", "estonia",
    "latvia", "lithuania", "croatia", "serbia", "slovenia", "usa", "us", "united states", "canada", "mexico",
    "brazil", "argentina", "chile", "colombia", "peru", "india", "china", "japan", "korea", "south korea",
    "vietnam", "thailand", "singapore", "malaysia", "indonesia", "philippines", "australia", "new zealand",
    "israel", "uae", "united arab emirates", "saudi arabia", "qatar", "egypt", "morocco", "nigeria", "kenya",
    "south africa", "pakistan", "bangladesh", "sri lanka", "nepal", "iran", "taiwan", "hong kong",
    # US states / provinces / German states often used instead of a country
    "california", "new york", "texas", "washington", "massachusetts", "illinois", "florida", "ontario",
    "quebec", "british columbia", "bavaria", "bayern", "berlin", "hamburg", "hesse", "hessen", "saxony",
    "sachsen", "nrw", "baden-württemberg",
}
_LOC_HEAD = r"[A-ZÄÖÜ][\wäöüß.'-]+(?:[ -][A-ZÄÖÜ][\wäöüß.'-]+){0,2}"
_LOC_RE = re.compile(rf"^(?P<city>{_LOC_HEAD}),\s*(?P<region>[A-Za-zÄÖÜäöüß. -]+?)(?:\s*\((?:remote|hybrid)\))?$")


class _Location:
    """Matches "Remote", "Munich, Germany", "Austin, TX" (not "Engineer, Beta Corp")."""

    @staticmethod
    def match(text: str) -> bool:
        t = text.strip()
        if re.fullmatch(r"(?:fully\s+)?(?:remote|hybrid|on-?site)", t, re.IGNORECASE):
            return True
        m = _LOC_RE.match(t)
        if not m or TITLE_WORDS.search(t) or org_score(t):
            return False
        region = m.group("region").strip()
        return region.lower() in COUNTRIES or bool(re.fullmatch(r"[A-Z]{2}", region))


LOCATION_RE = _Location()
LANGUAGE_NAMES = {
    "english", "german", "french", "spanish", "italian", "portuguese", "dutch", "polish", "russian",
    "ukrainian", "turkish", "arabic", "hebrew", "persian", "farsi", "hindi", "urdu", "bengali", "punjabi",
    "tamil", "telugu", "chinese", "mandarin", "cantonese", "japanese", "korean", "vietnamese", "thai",
    "indonesian", "malay", "swedish", "norwegian", "danish", "finnish", "icelandic", "greek", "czech",
    "slovak", "hungarian", "romanian", "bulgarian", "serbian", "croatian", "bosnian", "slovenian",
    "albanian", "lithuanian", "latvian", "estonian", "catalan", "basque", "galician", "swahili", "tagalog",
    "filipino", "latin", "afrikaans", "deutsch", "englisch", "französisch", "spanisch",
}
LANG_LEVEL_RE = re.compile(
    r"\b(?:[ABC][12](?:\+|/[ABC][12])?|native(?: speaker)?|mother tongue|bilingual|fluent|business fluent|"
    r"full professional(?: proficiency)?|professional(?: working)?(?: proficiency)?|proficient|advanced|"
    r"upper[- ]intermediate|intermediate|conversational|elementary|basic|beginner|limited working(?: proficiency)?|"
    r"muttersprache|fließend|verhandlungssicher|gut|grundkenntnisse)\b",
    re.IGNORECASE,
)
CERT_ISSUERS = re.compile(
    r"\b(?:AWS|Amazon Web Services|Amazon|Google(?: Cloud)?|Microsoft|Azure|Coursera|Udemy|edX|Udacity|"
    r"Oracle|Cisco|CompTIA|PMI|Scrum\.org|Scrum Alliance|Linux Foundation|CNCF|IBM|Meta|HashiCorp|"
    r"Salesforce|SAP|ISC2|\(ISC\)²|ISACA|DataCamp|LinkedIn Learning|Red Hat|Databricks|Snowflake|"
    r"TÜV|IHK|Goethe(?:-Institut)?|Cambridge|IELTS|TOEFL|DeepLearning\.AI|Stanford)\b",
)
TECH_LABEL_RE = re.compile(r"^\s*(?:tech(?:nologies)?|stack|tools|built with|skills used)\s*[:\-]\s*", re.IGNORECASE)


def _clean(text: str) -> str:
    text = PRIVATE_USE.sub(" ", text)
    return " ".join(text.split()).strip(" ,;|·•–—-:")


def is_bullet(line: Line) -> bool:
    return line.list_item or bool(BULLET_RE.match(line.text))


def bullet_text(line: Line) -> str:
    return BULLET_RE.sub("", line.text, count=1).strip()


def span_in(line: Line, value: str | None, start_at: int = 0) -> Span | None:
    """Span of ``value`` inside ``line`` (in Document.text coordinates)."""
    if not value:
        return None
    idx = line.text.find(value, start_at)
    if idx < 0:
        idx = line.text.lower().find(value.lower(), start_at)
    if idx < 0:
        return None
    return Span(start=line.start + idx, end=line.start + idx + len(value))


def block_span(lines: list[Line]) -> Span | None:
    return Span(start=lines[0].start, end=lines[-1].end) if lines else None


@dataclass
class Piece:
    text: str
    line: Line


def pieces_of(line: Line, remove: Span | None = None) -> list[Piece]:
    """Split a header line at separators, optionally cutting out a date span."""
    text = line.text
    if remove is not None and line.start <= remove.start < line.end + 1:
        a, b = remove.start - line.start, remove.end - line.start
        text = text[:a] + "\t" + text[b:]
    text = re.sub(r"[()\[\]]\s*\t\s*[()\[\]]", "\t", text)  # "(Jan 2020 – 2021)" leftovers
    parts = [_clean(p) for p in SEP_RE.split(text)]
    out = []
    for p in parts:
        p = re.sub(r"^\(|\)$", "", p).strip()
        if p and not re.fullmatch(r"[\W_]+", p):
            out.append(Piece(p, line))
    return out


# --------------------------------------------------------------- entries


def split_entries(lines: list[Line]) -> list[list[Line]]:
    """Group a section's lines into entries (one job / degree / project each).

    A new entry starts at a header line after body text (bullets or
    sentences), or at a second date when no body was seen. When dates sit on
    their own line *below* the title, the number of header lines before the
    first date tells how many trailing lines belong to the next entry.
    """
    entries: list[list[Line]] = []
    cur: list[Line] = []
    has_body = has_date = False
    lead = None  # header lines before the date in the first entry
    for ln in lines:
        if is_bullet(ln):
            cur.append(ln)
            has_body = True
            continue
        date = find_date_range(ln.text) is not None
        if cur and _is_description(ln):
            cur.append(ln)
            has_body = True
            continue
        if date and not has_date and lead is None:
            lead = len(cur)
        if cur and has_body:
            entries.append(cur)
            cur, has_body, has_date = [], False, False
        elif cur and date and has_date:
            carry: list[Line] = []
            if lead:
                last_date = max(i for i, l in enumerate(cur) if find_date_range(l.text))
                tail = cur[last_date + 1:]
                carry = tail[-lead:] if len(tail) >= lead else []
            entries.append(cur[: len(cur) - len(carry)])
            cur, has_body, has_date = carry, False, False
        cur.append(ln)
        has_date = has_date or date
    if cur:
        entries.append(cur)
    return [e for e in entries if e]


def _is_description(line: Line) -> bool:
    """A sentence-like line (long or ending with a full stop) that is not a date line."""
    text = line.text.rstrip()
    return find_date_range(text) is None and (len(text.split()) > 12 or text.endswith("."))


def _header_and_body(lines: list[Line]) -> tuple[list[Line], list[Line]]:
    header: list[Line] = []
    body: list[Line] = []
    for ln in lines:
        if body or is_bullet(ln) or (header and _is_description(ln)):
            body.append(ln)
        else:
            header.append(ln)
    return header, body


def _dates_in(header: list[Line]) -> DateRange | None:
    for ln in header:
        r = find_date_range(ln.text, offset=ln.start)
        if r:
            return r
    return None


def _bullets(lines: list[Line]) -> list[Bullet]:
    out = []
    for ln in lines:
        text = bullet_text(ln) if is_bullet(ln) else ln.text.strip()
        if text:
            out.append(Bullet(text=text, span=span_in(ln, text)))
    return out


def _set(entry, name: str, piece: Piece | None, value: str | None = None, uncertain: bool = False) -> None:
    if piece is None:
        return
    value = value or piece.text
    setattr(entry, name, value)
    span = span_in(piece.line, value)
    if span:
        entry.field_spans[name] = span
    entry.provenance[name] = "rules"
    if uncertain:
        entry.uncertain.append(name)


def _apply_dates(entry, dates: DateRange | None) -> None:
    if dates is None:
        return
    entry.start, entry.end, entry.current = dates.start, dates.end, dates.current
    if hasattr(entry, "dates_text"):
        entry.dates_text = dates.text
    if dates.span:
        entry.field_spans["dates"] = dates.span
    entry.provenance["dates"] = "rules"


def _split_comma(pieces: list[Piece]) -> list[Piece]:
    """"Software Engineer, Beta Corp" -> two pieces, unless the tail is a location."""
    out: list[Piece] = []
    for p in pieces:
        if "," in p.text and not LOCATION_RE.match(p.text):
            head, tail = (s.strip() for s in p.text.split(",", 1))
            if head and tail:
                out += [Piece(head, p.line), Piece(tail, p.line)]
                continue
        out.append(p)
    return out


def parse_experience(lines: list[Line]) -> ExperienceEntry:
    header, body = _header_and_body(lines)
    entry = ExperienceEntry(span=block_span(lines))
    dates = _dates_in(header)
    _apply_dates(entry, dates)

    pieces: list[Piece] = []
    explicit_at = False
    for ln in header:
        if re.search(r"\s@\s", ln.text):
            explicit_at = True
        for p in pieces_of(ln, dates.span if dates else None):
            m = AT_RE.match(p.text)
            if m and title_score(m.group("title")) > 0:
                explicit_at = True
                pieces += [Piece(m.group("title").strip(), ln), Piece(m.group("org").strip(), ln)]
            else:
                pieces.append(p)
    loc = next((p for p in pieces if LOCATION_RE.match(p.text)), None)
    if loc:
        _set(entry, "location", loc)
        pieces.remove(loc)
    pieces = _split_comma(pieces)
    title = max((p for p in pieces if title_score(p.text) > 0), key=lambda p: title_score(p.text), default=None)
    org = max((p for p in pieces if p is not title and org_score(p.text) > 0),
              key=lambda p: org_score(p.text), default=None)
    rest = [p for p in pieces if p is not title and p is not org]
    guessed = []
    if title is None and rest:
        title = rest.pop(0)
        guessed.append("title")
    if org is None and rest:
        org = rest.pop(0)
        if not explicit_at:
            guessed.append("org")
    if title is not None and org is not None and not explicit_at and guessed == ["title", "org"]:
        # Neither keyword matched: bold usually marks the job title.
        if org.line.bold and not title.line.bold:
            title, org = org, title
    _set(entry, "title", title, uncertain="title" in guessed)
    _set(entry, "org", org, uncertain="org" in guessed)
    if entry.location is None and rest and len(rest[0].text.split()) <= 3 and rest[0].text[:1].isupper():
        _set(entry, "location", rest[0], uncertain=True)
    entry.bullets = _bullets(body)
    return entry


def parse_education(lines: list[Line]) -> EducationEntry:
    header, body = _header_and_body(lines)
    entry = EducationEntry(span=block_span(lines))
    dates = _dates_in(header)
    if dates is None:
        for ln in header:
            single = find_single_date(ln.text, offset=ln.start)
            if single:
                entry.end, span = single
                entry.field_spans["dates"] = span
                entry.provenance["dates"] = "rules"
                dates = DateRange(end=single[0], span=span)
                break
    else:
        _apply_dates(entry, dates)

    for ln in lines:
        m = GRADE_RE.search(ln.text)
        if m:
            value = (m.group("num") or m.group("words")).strip()
            entry.grade = value
            entry.field_spans["grade"] = Span(start=ln.start + m.start(), end=ln.start + m.end())
            entry.provenance["grade"] = "rules"
            break

    pieces: list[Piece] = []
    for ln in header:
        if GRADE_RE.fullmatch(ln.text.strip()) or re.match(r"^\s*(?:c?gpa|grade|note)\b", ln.text, re.I):
            continue
        pieces += pieces_of(ln, dates.span if dates else None)
    pieces = [p for p in pieces if not LOCATION_RE.match(p.text)]
    pieces = _split_comma(pieces)

    deg_piece = next((p for p in pieces if DEGREE_RE.search(p.text)), None)
    if deg_piece:
        m = DEGREE_RE.search(deg_piece.text)
        degree = m.group().strip()
        after = deg_piece.text[m.end():].strip()
        fm = re.match(r"^(?:\(?(?:in|of)\s+|[:\-–]\s*)?(?P<f>[A-Za-zÄÖÜäöüß][\w&/ .'-]*?)\)?$", after)
        if fm and fm.group("f") and not INSTITUTION_WORDS.search(fm.group("f")):
            _set(entry, "field", deg_piece, fm.group("f").strip())
        _set(entry, "degree", deg_piece, degree)
    inst = next((p for p in pieces if p is not deg_piece and INSTITUTION_WORDS.search(p.text)), None)
    rest = [p for p in pieces if p is not deg_piece and p is not inst]
    if inst is None and rest:
        inst = rest.pop(0)
        _set(entry, "institution", inst, uncertain=True)
    else:
        _set(entry, "institution", inst)
    if entry.field is None and entry.degree is not None and rest:
        cand = rest[0]
        if not INSTITUTION_WORDS.search(cand.text) and len(cand.text.split()) <= 5:
            _set(entry, "field", cand, uncertain=True)
    entry.details = [b for b in _bullets(body)]
    return entry


def parse_project(lines: list[Line]) -> Project:
    header, body = _header_and_body(lines)
    entry = Project(span=block_span(lines))
    dates = _dates_in(header)
    _apply_dates(entry, dates)
    if dates is None:
        for ln in header:
            if single := find_single_date(ln.text, offset=ln.start):
                entry.end, span = single
                entry.field_spans["dates"] = span
                entry.provenance["dates"] = "rules"
                dates = DateRange(end=single[0], span=span)
                break
    for ln in lines:
        entry.links += [m.group() for m in URL.finditer(ln.text) if not EMAIL.search(m.group())]
    pieces = []
    for ln in header:
        pieces += pieces_of(ln, dates.span if dates else None)
    pieces = [p for p in pieces if not URL.fullmatch(p.text)]
    if pieces:
        _set(entry, "name", pieces[0])
        tech = re.search(r"\(([^)]*,[^)]*)\)", pieces[0].line.text)
        if tech:
            entry.technologies += [t.strip() for t in tech.group(1).split(",") if t.strip()]
            entry.name = _clean(entry.name.split("(")[0])
    kept = []
    for ln in body:
        text = bullet_text(ln) if is_bullet(ln) else ln.text
        if TECH_LABEL_RE.match(text):
            entry.technologies += [t for t in (_clean(x) for x in re.split(r"[,;|•·]", TECH_LABEL_RE.sub("", text))) if t]
        else:
            kept.append(ln)
    entry.bullets = _bullets(kept)
    return entry


# ----------------------------------------------------------- list sections


def _items(line: Line) -> tuple[str | None, list[tuple[str, int]]]:
    """Split "Category: a, b (x), c" into (category, [(item, offset_in_line)])."""
    text = bullet_text(line) if is_bullet(line) else line.text
    base = line.text.find(text) if text else 0
    category = None
    m = re.match(r"^\s*([A-Za-z][\w &/+.-]{1,40}?)\s*[:–—]\s+(?=\S)", text)
    if m and len(m.group(1).split()) <= 4:
        category = m.group(1).strip()
        base += m.end()
        text = text[m.end():]
    items = []
    depth = 0
    start = 0
    for i, ch in enumerate(text + ","):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        elif depth == 0 and (ch in ",;|•·" or (ch == "\t")):
            raw = text[start:i]
            stripped = raw.strip()
            if stripped:
                items.append((stripped, base + start + raw.find(stripped)))
            start = i + 1
    return category, items


def parse_skills(lines: list[Line], cand: Candidate) -> None:
    seen = {s.name.lower() for s in cand.skills}
    for ln in lines:
        category, items = _items(ln)
        names = [t for t, _ in items]
        is_lang_list = bool(names) and sum(
            1 for t in names if t.split()[0].lower().strip("():") in LANGUAGE_NAMES
        ) >= max(1, len(names) // 2 + 1)
        if is_lang_list:
            parse_language_line(ln, cand)
            continue
        for text, off in items:
            level = None
            name = text
            m = re.match(r"^(.*?)\s*[(\[]([^)\]]+)[)\]]\s*$", text)
            if m and m.group(1):
                name, level = m.group(1).strip(), m.group(2).strip()
            name = _clean(name)
            if not name or len(name.split()) > 6 or name.endswith(".") and len(name.split()) > 3:
                continue
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            skill = Skill(name=name, category=category, level=level,
                          span=Span(start=ln.start + off, end=ln.start + off + len(text)))
            skill.provenance["name"] = "rules"
            cand.skills.append(skill)


def parse_language_line(line: Line, cand: Candidate) -> None:
    _, items = _items(line)
    if not items:
        return
    for text, off in items:
        m = re.match(r"^([A-Za-zÄÖÜäöüß]+(?:\s+\([A-Za-z]+\))?)\s*(?:[(\[:–—-]\s*|\s+)?(.*?)[)\]]?$", text)
        if not m:
            continue
        name, level = m.group(1).strip(), (m.group(2) or "").strip(" ()-–:")
        if "(" in name:  # "English (C2)" captured as name by the optional group
            name, extra = name.split("(", 1)
            name, level = name.strip(), extra.strip(" )") or level
        if name.lower() not in LANGUAGE_NAMES and not LANG_LEVEL_RE.search(level or ""):
            continue
        lang = Language(name=name, level=level or None,
                        span=Span(start=line.start + off, end=line.start + off + len(text)))
        lang.provenance["name"] = "rules"
        if not any(l.name.lower() == name.lower() for l in cand.languages):
            cand.languages.append(lang)


def parse_certifications(lines: list[Line], cand: Candidate) -> None:
    for ln in lines:
        text = bullet_text(ln) if is_bullet(ln) else ln.text.strip()
        if not text:
            continue
        cert = Certification(name=text, span=Span(start=ln.start, end=ln.end))
        single = find_single_date(text)
        name = text
        if single:
            cert.date = single[0]
            off = ln.text.find(text)
            cert.field_spans["date"] = Span(start=ln.start + off + single[1].start, end=ln.start + off + single[1].end)
            name = (text[: single[1].start] + text[single[1].end:])
        name = _clean(re.sub(r"\(\s*\)", "", name))
        parts = re.split(r"\s+[|·•]\s+|\s+[–—-]\s+|\t|\s{3,}|\s+(?:by|from|issued by)\s+", name)
        pieces = [p for p in (_clean(x) for x in parts) if p]
        if len(pieces) == 1 and (m := re.match(r"^(.*\S),\s*([^,]+)$", pieces[0])) and CERT_ISSUERS.fullmatch(m.group(2)):
            pieces = [m.group(1), m.group(2)]
        if not pieces:
            continue
        cert.name = pieces[0]
        issuer = pieces[1] if len(pieces) > 1 else None
        if issuer is None and (m := CERT_ISSUERS.search(cert.name)):
            issuer = m.group()
        cert.issuer = issuer
        for key in ("name", "issuer"):
            if (value := getattr(cert, key)) and (span := span_in(ln, value)):
                cert.field_spans[key] = span
            if value:
                cert.provenance[key] = "rules"
        cand.certifications.append(cert)


def parse_publications(lines: list[Line], cand: Candidate) -> None:
    for group in split_entries(lines) if lines else []:
        text = " ".join((bullet_text(l) if is_bullet(l) else l.text.strip()) for l in group)
        year = find_single_date(text)
        pub = Publication(text=text, year=year[0].year if year else None, span=block_span(group))
        pub.provenance["text"] = "rules"
        cand.publications.append(pub)


# ---------------------------------------------------------------- contact


def parse_contact(doc: Document, sections: list[Section]) -> Contact:
    contact = Contact()
    preferred = [ln for s in sections if s.kind in ("header", "contact") for ln in s.lines]
    top = doc.lines[:25]
    pools = [preferred, top, doc.lines]

    def first(pattern_fn, name: str):
        for pool in pools:
            for ln in pool:
                for m in pattern_fn(ln.text):
                    return ln, m
        return None

    if hit := first(lambda t: list(EMAIL.finditer(t)), "email"):
        ln, m = hit
        contact.email = m.group()
        contact.field_spans["email"] = Span(start=ln.start + m.start(), end=ln.start + m.end())
    if hit := first(find_phones, "phone"):
        ln, m = hit
        contact.phone = m.group().strip()
        contact.field_spans["phone"] = Span(start=ln.start + m.start(), end=ln.start + m.end())
    for ln in preferred or top:
        for m in URL.finditer(ln.text):
            if EMAIL.search(ln.text[max(0, m.start() - 40): m.end() + 1]) and "@" in ln.text[max(0, m.start() - 1):m.start() + 1]:
                continue
            link = m.group()
            if "@" not in link and link not in contact.links and (contact.email is None or link not in contact.email):
                contact.links.append(link)
    # Location: a "City, Country" piece in the header/contact lines.
    for ln in preferred or top:
        for p in pieces_of(ln):
            txt = re.sub(r"^(?:address|location|wohnort|adresse)\s*:\s*", "", p.text, flags=re.I)
            if LOCATION_RE.match(txt) and not EMAIL.search(txt):
                contact.location = txt
                if span := span_in(ln, txt):
                    contact.field_spans["location"] = span
                break
        if contact.location:
            break

    # Name: short, capitalised, no digits/@, not a heading or job title; prefer the biggest font.
    candidates = []
    for i, ln in enumerate((preferred or top)[:8]):
        t = _clean(re.sub(r"^(?:name\s*:\s*)", "", ln.text, flags=re.I))
        words = t.split()
        if not 2 <= len(words) <= 4 or re.search(r"[\d@/:|]", t):
            continue
        if not all(w[:1].isupper() for w in words if w.lower() not in {"von", "van", "de", "der", "da", "di", "le", "la", "bin", "al"}):
            continue
        if classify_heading(ln) is not None or TITLE_WORDS.search(t) or org_score(t) or LOCATION_RE.match(t):
            continue
        candidates.append((-(ln.font_size or 0), i, ln, t))
    if candidates:
        _, _, ln, name = min(candidates, key=lambda c: (c[0], c[1]))
        contact.name = name
        if span := span_in(ln, name):
            contact.field_spans["name"] = span
    for key in ("name", "email", "phone", "location"):
        if getattr(contact, key):
            contact.provenance[key] = "rules"
    lines = preferred or top
    contact.span = block_span(lines) if lines else None
    return contact


# ------------------------------------------------------------------ main


def extract_candidate(doc: Document, sections: list[Section] | None = None) -> Candidate:
    """Rule-based extraction. Years of experience are added by the pipeline."""
    sections = sections if sections is not None else split_sections(doc)
    cand = Candidate()
    cand.contact = parse_contact(doc, sections)

    for sec in sections:
        if not sec.lines:
            continue
        if sec.kind == "summary":
            text = " ".join(ln.text.strip() for ln in sec.lines)
            cand.summary = f"{cand.summary} {text}".strip() if cand.summary else text
        elif sec.kind == "experience":
            cand.experience += [parse_experience(g) for g in split_entries(sec.lines)]
        elif sec.kind == "education":
            cand.education += [parse_education(g) for g in split_entries(sec.lines)]
        elif sec.kind == "projects":
            cand.projects += [parse_project(g) for g in split_entries(sec.lines)]
        elif sec.kind == "skills":
            parse_skills(sec.lines, cand)
        elif sec.kind == "languages":
            for ln in sec.lines:
                parse_language_line(ln, cand)
        elif sec.kind == "certifications":
            parse_certifications(sec.lines, cand)
        elif sec.kind == "publications":
            parse_publications(sec.lines, cand)
        elif sec.kind in ("header", "contact"):
            # Long sentences in the header are usually an unlabelled summary.
            if cand.summary is None and sec.kind == "header":
                long = [ln.text.strip() for ln in sec.lines if len(ln.text.split()) >= 12
                        and not EMAIL.search(ln.text)]
                if long:
                    cand.summary = " ".join(long)
        else:
            other = OtherSection(kind=sec.kind, heading=sec.heading, text=sec.text,
                                 span=Span(start=sec.start, end=sec.end))
            cand.other_sections.append(other)

    kinds = {s.kind for s in sections}
    if "experience" not in kinds:
        cand.warnings.append("No work-experience section heading was found.")
    if "education" not in kinds:
        cand.warnings.append("No education section heading was found.")
    if "skills" not in kinds:
        cand.warnings.append("No skills section heading was found.")
    for i, e in enumerate(cand.experience, 1):
        if e.start is None:
            cand.warnings.append(f"Experience entry {i} ({e.title or e.org or '?'}) has no recognisable dates.")
    if not cand.contact.email:
        cand.warnings.append("No email address found.")
    return cand

