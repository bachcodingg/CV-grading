"""[4] JD Parser: a pasted job posting -> atomic, prioritised requirements (PLAN.md §3 [4]).

Rules first, like the CV structurer:

1. Clean the text and split it into sections by heading ("Requirements",
   "What you'll bring", "Nice to have", "Responsibilities", "Benefits", ...).
   Each requirement-type section sets a default priority.
2. Each bullet (or sentence of a paragraph) is an item; an item is cut into
   atomic requirements where its clauses differ in kind or in priority
   ("BSc in CS and 3+ years of Python" -> two requirements).
3. Per requirement, in code: category, priority from wording cues (they
   override the section default), minimum years, degree level, spoken
   languages, and keywords from the skill vocabulary plus tech-looking terms.
4. Optional: GLiNER adds skills the vocabulary does not know; the local LLM
   classifies requirements the rules could not, and splits long compound
   ones. LLM output is only accepted when every part is verbatim posting text.

Every requirement keeps its span in ``JobPosting.text``.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.config import Config, LLMConfig, get_config
from app.extraction.cv_extractor import LANG_LEVEL_RE, LANGUAGE_NAMES
from app.extraction.schemas import (
    Bullet,
    Category,
    DegreeLevel,
    JDKeyword,
    JobPosting,
    Priority,
    Requirement,
    Span,
)
from app.ingestion.document import clean_text
from app.matching.skills import find_skills

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ sections

_REQUIRED = "required"
_SHOULD = "preferred_should"
_NICE = "preferred_nice"
_RESP = "responsibilities"
_STACK = "stack"
_BENEFITS = "benefits"
_ABOUT = "about"
_APPLY = "apply"
_OTHER = "other"
_HEADER = "header"

# (kind, pattern). Order matters: "Preferred qualifications" is checked before "qualifications",
# "About the role" before "About <company>". With a heading signal (colon, markdown, ALL CAPS) the
# pattern may appear anywhere in the line; without one the whole line must match.
HEADING_PATTERNS: list[tuple[str, str]] = [
    (_NICE, r"nice[- ]to[- ]haves?|bonus(?: points)?|good[- ]to[- ]haves?|(?:it'?s |would be )?(?:a )?plus(?:es)?"
            r"|extra (?:credit|points)|even better(?: if)?|what (?:would )?makes? you stand out|optional"
            r"|(?:it would be )?great if you(?: also)?(?: have)?"),
    (_SHOULD, r"preferred(?: qualifications| skills| experience| requirements)?|desirable(?: skills)?"
              r"|desired(?: skills| qualifications| experience)?|ideally(?: you(?: also)? have)?|preferably"
              r"|additional (?:qualifications|skills)"),
    (_RESP, r"(?:key |main |your |core |primary )?(?:responsibilit(?:y|ies)|tasks|duties)"
            r"|what you(?:'ll| will)? (?:do|be doing|work on)|your (?:role|mission|impact|day[- ]to[- ]day|job|challenges?)"
            r"|(?:the|this) (?:role|job|position|opportunity)|about (?:the|this) (?:role|job|position)"
            r"|role (?:description|overview)|job description|in this role(?: you will)?|day[- ]to[- ]day"
            r"|what the job involves|the challenge|your contribution"),
    (_REQUIRED, r"(?:basic |minimum |required |key |essential |core |job |technical |must[- ]have )?"
                r"(?:requirements?|qualifications?|skills?(?: (?:and|&) (?:experience|qualifications|requirements))?"
                r"|must[- ]haves?|competenc(?:e|es|ies)|essentials)"
                r"|(?:your|candidate|ideal candidate|the ideal candidate|applicant)? ?profile|about you|who you are"
                r"|you are|you have|you bring|you'?ll bring|what you(?:'ll| will)? (?:bring|need|have|offer)"
                r"|what (?:we(?:'re| are)|we) (?:look|are looking|re looking) for|what we(?:'re| are) looking for"
                r"|we(?:'re| are) looking for|what you need(?: to succeed)?|your (?:skills|experience|background|"
                r"qualifications|expertise|toolbox|knowledge)|experience(?: (?:and|&) (?:skills|qualifications))?"
                r"|(?:the )?ideal candidate|you should have|you must have|what it takes|who we(?:'re| are) looking for"),
    (_STACK, r"(?:our |the )?(?:tech(?:nology)? stack|stack|tools (?:we use|and technologies)|technologies(?: we use)?)"),
    (_BENEFITS, r"(?:our |your )?(?:benefits|perks|offer)|what we offer|we offer|why (?:join us|you'?ll love [\w ]+|"
                r"work (?:with|for|at) us)|what'?s in it for you|compensation(?: and benefits)?|salary|what you get"),
    (_ABOUT, r"about (?:us|the company|the team|[\w&.-]+(?: [\w&.-]+)?)|who we are|our (?:company|team|story|"
             r"mission|culture|values)|company (?:description|overview)|the team|meet the team"),
    (_APPLY, r"how to apply|application(?: process)?|apply(?: now)?|interested|next steps|(?:our )?hiring process"
             r"|equal opportunit(?:y|ies)(?: employer)?|diversity(?: (?:and|&) inclusion)?|contact(?: person)?"),
]
_COMPILED = [(k, re.compile(rf"\b(?:{p})\b"), re.compile(rf"(?:{p})")) for k, p in HEADING_PATTERNS]

SECTION_PRIORITY: dict[str, Priority] = {_REQUIRED: "must", _SHOULD: "should", _NICE: "nice"}
_REQ_KINDS = set(SECTION_PRIORITY)
_NO_KEYWORDS = {_BENEFITS, _ABOUT, _APPLY}

BULLET_RE = re.compile(r"^\s*(?:[•·▪●◦‣∙○■□➢➤►▶✓✔✅☑◆◇→\-–—*+>]|\d{1,2}[.)]|[a-z][.)](?=\s))\s*")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+|^\s*\*\*(.+)\*\*:?\s*$|^\s*__(.+)__:?\s*$")


@dataclass
class _Line:
    text: str
    start: int
    end: int

    @property
    def bullet(self) -> bool:
        return bool(BULLET_RE.match(self.text)) and len(BULLET_RE.sub("", self.text, count=1).strip()) > 0


@dataclass
class _Section:
    kind: str
    heading: str
    lines: list[_Line] = field(default_factory=list)


def clean_posting(text: str) -> str:
    text = html.unescape(clean_text(text)).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip("\n")
    return re.sub(r"\n{3,}", "\n\n", text)


def _lines(text: str) -> list[_Line]:
    out, pos = [], 0
    for raw in text.split("\n"):
        if raw.strip():
            lead = len(raw) - len(raw.lstrip())
            out.append(_Line(text=raw.strip(), start=pos + lead, end=pos + len(raw)))
        pos += len(raw) + 1
    return out


def normalize_heading(text: str) -> str:
    t = _MD_HEADING.sub(lambda m: m.group(1) or m.group(2) or "", text.strip())
    t = t.lower().replace("&", " and ").replace("’", "'")
    t = re.sub(r"[^\w\s'/-]", " ", t)
    return " ".join(t.split()).strip(" -/")


def heading_kind(line: _Line, next_line: _Line | None) -> tuple[str | None, bool]:
    """(kind, strong) if ``line`` is a section heading, else (None, False)."""
    raw = line.text
    if line.bullet or len(raw) > 80:
        return None, False
    md = bool(_MD_HEADING.match(raw))
    body = _MD_HEADING.sub(lambda m: m.group(1) or m.group(2) or "", raw).strip()
    if body.endswith((".", ",", ";")) and not body.endswith("..."):
        return None, False
    norm = normalize_heading(raw)
    words = norm.split()
    if not words or len(words) > 10:
        return None, False
    letters = re.sub(r"[^A-Za-z]", "", body)
    caps = len(letters) >= 4 and letters.isupper()
    colon = body.endswith(":")
    strong = md or caps
    signal = strong or colon or (next_line is not None and next_line.bullet and len(words) <= 6)
    for kind, search, full in _COMPILED:
        if full.fullmatch(norm) or (signal and search.search(norm)):
            if not strong and len(words) > 6 and kind not in _REQ_KINDS:
                # "To succeed in this role you will need:" introduces a list; it is not a heading.
                return None, False
            return kind, True
    if strong and len(words) <= 6:
        return _OTHER, True
    return None, False


def split_jd_sections(text: str) -> list[_Section]:
    lines = _lines(text)
    sections = [_Section(kind=_HEADER, heading="")]
    for i, ln in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        kind, _ = heading_kind(ln, nxt)
        if kind is not None and len(sections) == 1 and not sections[0].lines and kind == _OTHER:
            kind = None  # the job title at the very top, often in caps or markdown
        if kind is not None:
            sections.append(_Section(kind=kind, heading=ln.text.strip().strip("#*_ ").rstrip(":")))
        else:
            sections[-1].lines.append(ln)
    return [s for s in sections if s.lines or s.kind != _HEADER]


# ------------------------------------------------------------------ items


def _items(text: str, section: _Section) -> list[tuple[int, int]]:
    """Spans of items: bullets (with wrapped continuation lines) or paragraph sentences."""
    groups: list[list[_Line]] = []
    for ln in section.lines:
        cont = (groups and not ln.bullet and groups[-1][0].bullet and ln.text[:1].islower())
        if cont:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    out = []
    for g in groups:
        first = g[0]
        start = first.start
        if first.bullet:
            start += BULLET_RE.match(first.text).end()
        end = g[-1].end
        if first.bullet:
            out.append((start, end))
        else:
            out.extend(_sentences(text, start, end))
    return out


_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|etc|incl|approx|vs|min|max|esp|resp|ca|no|dr|prof|mr|ms|mrs|st|"
                     r"[A-Za-z]\.[A-Za-z]+|[A-Z])\.)$", re.IGNORECASE)
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[\"(A-Z0-9])")


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    out, s = [], start
    for m in _SENT_END.finditer(text, start, end):
        if _ABBREV.search(text[s:m.start()]):
            continue
        out.append((s, m.start()))
        s = m.end()
    out.append((s, end))
    return out


# ------------------------------------------------------------------ requirement analysis

CUE_PATTERNS: list[tuple[Priority, re.Pattern]] = [
    ("should", re.compile(r"\b(?:a |is a |would be a )?(?:strong|big|huge|major|great|definite|real|significant) plus\b",
                          re.IGNORECASE)),
    ("nice", re.compile(r"\bnice[- ]to[- ]have\b|\bbonus\b|\b(?:is|are|would be|as) an? (?:additional )?plus\b"
                        r"|\ba plus\b|\bplus\b(?=\s*[.!)]*\s*$)|\bgood[- ]to[- ]have\b|\ban advantage\b"
                        r"|\badvantageous\b|\bbeneficial\b|\boptional\b|\bnot (?:required|a must|mandatory)\b"
                        r"|\bwould be (?:great|nice|helpful|appreciated|cool)\b|\bappreciated\b|\bnice\b",
                        re.IGNORECASE)),
    ("should", re.compile(r"\bpreferred\b|\bpreferabl[ey]\b|\bideally\b|\bdesir(?:ed|able)\b|\bshould\b"
                          r"|\b(?:would be |is )?(?:highly )?valu(?:ed|able)\b|\bwelcome\b", re.IGNORECASE)),
    ("must", re.compile(r"\bmust\b(?!\s+not)|\b(?:is |are )?required\b|\brequirement\b|\bmandatory\b|\bessential\b"
                        r"|\bindispensable\b|\bneeds? to\b|\bhave to\b|\bobligatory\b|\bmust[- ]have\b",
                        re.IGNORECASE)),
]


def priority_cue(text: str) -> tuple[Priority, str] | None:
    """The earliest priority wording in ``text`` ("nice to have" -> nice)."""
    best: tuple[int, Priority, str] | None = None
    for prio, pat in CUE_PATTERNS:
        m = pat.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), prio, m.group().strip())
    return (best[1], best[2]) if best else None


_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "a couple of": 2, "a few": 2}
_NUM = r"(\d{1,2}(?:[.,]5)?|" + "|".join(sorted(_WORDNUM, key=len, reverse=True)) + r")"
_YRS = r"(?:years?|yrs?\.?)"
_YEARS_PATTERNS = [
    re.compile(rf"\b{_NUM}\s*\+?\s*(?:-|–|—|to)\s*{_NUM}\s*\+?\s*{_YRS}", re.IGNORECASE),
    re.compile(rf"(?:at least|minimum(?: of)?|min\.|over|more than|>|≥)\s*{_NUM}\s*\+?\s*{_YRS}", re.IGNORECASE),
    re.compile(rf"\b{_NUM}\s*\+\s*{_YRS}", re.IGNORECASE),
    re.compile(rf"\b{_NUM}\s*{_YRS}(?:'s?)?\s+(?:of\s+)?(?:[\w-]+\s+){{0,4}}?(?:experience|exp\.|background|work)",
               re.IGNORECASE),
    re.compile(rf"\b{_NUM}\s*{_YRS}\s+(?:in|with|as|of|working)\b", re.IGNORECASE),
]
_MONTHS_RE = re.compile(rf"\b{_NUM}\s*\+?\s*months?\b", re.IGNORECASE)


def _num(token: str) -> float:
    t = token.lower()
    return float(_WORDNUM[t]) if t in _WORDNUM else float(t.replace(",", "."))


def min_years(text: str) -> tuple[float, Span] | None:
    if re.search(r"\byears? old\b", text, re.IGNORECASE):
        return None
    for pat in _YEARS_PATTERNS:
        m = pat.search(text)
        if m:
            return _num(m.group(1)), Span(start=m.start(), end=m.end())
    m = _MONTHS_RE.search(text)
    if m and re.search(r"experience|internship|working", text, re.IGNORECASE):
        return round(_num(m.group(1)) / 12, 2), Span(start=m.start(), end=m.end())
    return None


CERT_RE = re.compile(
    r"\bcertif(?:ied|ication|ications|icate|icates)\b|\blicen[cs]e[ds]?\b|\b(?:PMP|CISSP|CISM|CISA|CPA|ACCA|CFA|ITIL|"
    r"PRINCE2|CCNA|CCNP|OSCP|CEH|CKA|CKAD|CKS|PSM|CSM|ISTQB|TOGAF|Six Sigma)\b",
    re.IGNORECASE,
)
EDU_RE = re.compile(
    r"\bdegree\b(?!\s+of\b)|\bbachelor'?s?\b|\bmaster'?s\b|\bmaster(?: of| degree)\b|\bph\.?\s?d\.?|\bdoctorate\b"
    r"|\bb\.\s?sc\.?|\bm\.\s?sc\.?|\bbsc\b|\bmsc\b|\bmba\b|\bdiplom\w*|\buniversity\b|\bgraduate[ds]?\b"
    r"|\benrolled\b|\bstudent\b|\bstud(?:ies|ying) (?:in|of)\b|\bstudy program\w*|\b(?:BS|MS|BA|MA|BEng|MEng)\b"
    r"(?=\s+(?:in|or|degree|/)|\s*[,/)])|\bcomputer science or (?:a )?(?:similar|related)\b",
    re.IGNORECASE,
)
_LEVELS: list[tuple[DegreeLevel, re.Pattern]] = [
    ("bachelor", re.compile(r"\bbachelor|\bb\.\s?sc|\bbsc\b|\bBS\b|\bBA\b|\bBEng\b|\bundergraduate\b|\bdegree\b(?!\s+of)",
                            re.IGNORECASE)),
    ("master", re.compile(r"\bmaster'?s?\b|\bm\.\s?sc|\bmsc\b|\bMS\b|\bMA\b|\bMEng\b|\bmba\b|\bdiplom", re.IGNORECASE)),
    ("phd", re.compile(r"\bph\.?\s?d|\bdoctora(?:te|l)\b", re.IGNORECASE)),
]
_LANG_NAMES_RE = re.compile(
    r"\b(" + "|".join(sorted((n.capitalize() for n in LANGUAGE_NAMES if n.isalpha()), key=len, reverse=True)) + r")\b"
)
_LANG_CONTEXT = re.compile(
    r"\b(?:language|languages|speak|speaking|spoken|written|verbal|oral|command of|skills|proficiency|fluency|"
    r"knowledge|level|communicat\w*)\b|\b[ABC][12]\b",
    re.IGNORECASE,
)
_CEFR = re.compile(r"\b[ABC][12]\b(?:\s*[/+-]\s*[ABC]?[12])?")
SOFT_RE = re.compile(
    r"\bcommunicat\w*|\bteam ?player\b|\bteam ?work\b|\bcollaborat\w*|\bself[- ]?(?:motivated|starter|driven|organi[sz]ed)"
    r"|\bproactiv\w*|\bownership\b|\binitiative\b|\bproblem[- ]solv\w*|\banalytical (?:thinking|skills|mindset|mind)\b"
    r"|\battention to detail\b|\bdetail[- ]oriented\b|\borgani[sz]ational skills\b|\btime management\b|\binterpersonal\b"
    r"|\bleadership\b|\bstakeholder management\b|\badaptab\w*|\bflexib\w*|\bcurio(?:us|sity)\b|\beager to learn\b"
    r"|\bwillingness to (?:learn|travel)\b|\bhands-on mentality\b|\bstructured (?:way of working|approach|working|mindset)\b"
    r"|\bindependent(?:ly)?\b|\bautonomous(?:ly)?\b|\bcreativ\w*|\bempath\w*|\bresilien\w*|\bpresentation skills\b"
    r"|\bcustomer[- ](?:focus\w*|orient\w*|centric)\b|\bmindset\b|\bpassion(?:ate)?\b|\bmotivated\b|\breliab\w*"
    r"|\bunder pressure\b|\bmulti-?task\w*|\bcritical thinking\b|\bnegotiation\b|\bteam[- ]oriented\b|\bcan-do\b"
    r"|\bopen[- ]minded\b|\bsense of humou?r\b|\bstrong work ethic\b|\bdrive\b|\bgrowth mindset\b|\bcuriosity\b",
    re.IGNORECASE,
)
SKILL_CUE = re.compile(
    r"\b(?:knowledge (?:of|in)|proficien\w*|familiar(?:ity)?|experience (?:with|in|using|of|building|developing|"
    r"working)|hands-on|expertise (?:in|with)|skills? (?:in|with)|understanding of|working knowledge|command of|"
    r"competen\w* in|fluent in|good at|comfortable (?:with|in)|exposure to|experienced (?:in|with))\b",
    re.IGNORECASE,
)
EXPERIENCE_CUE = re.compile(r"\b(?:experience|background|track record|history of|worked|working in)\b", re.IGNORECASE)
REQUIREMENT_CUE = re.compile(
    r"\b(?:experience|years?|degree|bachelor|master'?s|ph\.?d|knowledge of|familiar\w*|proficien\w*|fluent|fluency|"
    r"understanding of|ability to|skills?|expertise|you have|you bring|you are|you'?ll need|must|required|"
    r"certif\w*|background in|track record|hands-on|we expect|we require|ideally|preferred|plus)\b",
    re.IGNORECASE,
)
ANY_OF_RE = re.compile(r"\b(?:e\.g\.?|eg\.|such as|for example|for instance|one of|any of|at least one|or similar|"
                       r"or equivalent|or comparable|or related|like)\b|\be\.g\.", re.IGNORECASE)

_TECH_TOKEN = re.compile(
    r"(?<![\w.+#/-])(?:[A-Za-z]*[a-z][A-Z][\w+#]*|[A-Za-z]+[0-9][A-Za-z0-9]*|[A-Za-z]+\.(?:js|io|ai)|[A-Z]{2,6}s?)"
    r"(?![\w+#-])"
)
_TECH_STOP = {
    "US", "USA", "EU", "UK", "IT", "HR", "CV", "CVs", "PhD", "MSc", "BSc", "MBA", "BA", "BS", "MS", "MA", "MEng",
    "BEng", "CEO", "CTO", "CFO", "COO", "CPO", "VP", "FTE", "PTO", "EUR", "USD", "GBP", "CHF", "OR", "AND", "YOU",
    "WE", "OUR", "THE", "NOT", "ALL", "NEW", "ASAP", "FAQ", "GmbH", "AG", "SE", "LLC", "INC", "LTD", "CET", "CEST",
    "ID", "OK", "TV", "PC", "AM", "PM", "DE", "EN", "FR", "WFH", "IPO", "DACH", "EMEA", "APAC", "NYC", "SF", "LA",
    "TBD", "YoE", "iPhone", "iPad", "LinkedIn", "YouTube", "WhatsApp", "McKinsey", "PhDs", "MAs", "BAs", "FAQs",
    "WHO", "WHAT", "OF", "TO", "IN", "IS", "ARE", "BE", "A1", "A2", "B1", "B2", "C1", "C2", "Q1", "Q2", "Q3", "Q4",
    "H1", "H2", "COVID", "COVID19", "DEI", "LGBTQ", "LGBTQIA", "BIPOC",
}


@dataclass
class _Analysis:
    category: Category
    strong: bool
    uncertain: bool = False
    years: tuple[float, Span] | None = None
    cue: tuple[Priority, str] | None = None


def _keywords(text: str, base: int) -> list[JDKeyword]:
    found = [JDKeyword(text=m.text, canonical=m.canonical, span=Span(start=m.start, end=m.end), source="vocab")
             for m in find_skills(text, base=base)]
    taken = [(k.span.start, k.span.end) for k in found]
    for m in _TECH_TOKEN.finditer(text):
        tok = m.group()
        s, e = base + m.start(), base + m.end()
        if tok in _TECH_STOP or tok.rstrip("s") in _TECH_STOP or re.fullmatch(r"\d+", tok):
            continue
        if any(a < e and s < b for a, b in taken):
            continue
        found.append(JDKeyword(text=tok, canonical=None, span=Span(start=s, end=e), source="pattern"))
        taken.append((s, e))
    return sorted(found, key=lambda k: k.span.start)


def _strip_cue(text: str) -> str:
    for _, pat in CUE_PATTERNS:
        text = pat.sub(" ", text)
    return re.sub(r"\b(?:is|are|would be|also|a|an)\b", " ", text, flags=re.IGNORECASE)


# A clause that opens with "experience with" / "solid knowledge of" starts a new requirement.
_OPENS_REQUIREMENT = re.compile(
    r"^(?:(?:strong|solid|good|deep|excellent|proven|basic|some|practical|first|professional|working|very good|"
    r"in-depth|sound|extensive)\s+)*" + SKILL_CUE.pattern,
    re.IGNORECASE,
)


def analyze(text: str, keywords: list[JDKeyword] | None = None) -> _Analysis:
    """Category of one atomic requirement, plus its years and priority cue."""
    has_kw = bool(keywords if keywords is not None else find_skills(text))
    years = min_years(text)
    cue = priority_cue(text)
    if CERT_RE.search(text):
        cat, strong = "certification", True
    elif EDU_RE.search(text):
        cat, strong = "education", True
    elif _LANG_NAMES_RE.search(text) and (_LANG_CONTEXT.search(text) or LANG_LEVEL_RE.search(text)
                                          or len(_strip_cue(text).split()) <= 3):
        cat, strong = "language", True
    elif years:
        cat, strong = "experience", True
    elif SOFT_RE.search(text) and not has_kw:
        cat, strong = "soft_skill", True
    elif SKILL_CUE.search(text):
        if has_kw:
            cat, strong = "skill", True
        elif EXPERIENCE_CUE.search(text):
            cat, strong = "experience", True
        else:
            return _Analysis("skill", True, uncertain=True, years=years, cue=cue)
    elif has_kw:
        cat, strong = "skill", False
    elif EXPERIENCE_CUE.search(text):
        cat, strong = "experience", True
    else:
        return _Analysis("other", False, uncertain=True, years=years, cue=cue)
    return _Analysis(cat, strong, years=years, cue=cue)


# Clause boundaries inside one item: ";", ", and", ", ", " and ", " as well as " (outside brackets).
_BOUNDARY = re.compile(r";\s*|,\s+(?:and\s+|as well as\s+|plus\s+)?|\s+(?:and|as well as)\s+")


def _clauses(text: str, start: int, end: int) -> list[tuple[int, int]]:
    depth, cuts = 0, []
    i = start
    while i < end:
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif depth == 0:
            m = _BOUNDARY.match(text, i, end)
            if m and m.end() > m.start():
                cuts.append((m.start(), m.end()))
                i = m.end()
                continue
        i += 1
    out, s = [], start
    for a, b in cuts:
        out.append((s, a))
        s = b
    out.append((s, end))
    return [(a, b) for a, b in out if text[a:b].strip()]


def atomic_parts(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Merge clauses back together unless a clause starts a different kind of requirement."""
    groups: list[list[tuple[int, int]]] = []
    group_info: list[_Analysis] = []
    for a, b in _clauses(text, start, end):
        clause = text[a:b]
        info = analyze(clause)
        if groups:
            cur = group_info[-1]
            new_kind = info.strong and info.category != cur.category and cur.category != "other"
            # "Strong Python skills and experience with FastAPI": two skill requirements.
            new_kind |= (info.category == cur.category == "skill" and bool(_OPENS_REQUIREMENT.match(clause))
                         and bool(find_skills(clause)) and bool(find_skills(text[groups[-1][0][0]:groups[-1][-1][1]])))
            new_prio = info.cue is not None and cur.cue is not None and info.cue[0] != cur.cue[0]
            new_prio |= info.cue is not None and cur.cue is None and info.strong and cur.strong
            both_years = info.years is not None and cur.years is not None
            if not (new_kind or new_prio or both_years):
                groups[-1].append((a, b))
                merged = analyze(text[groups[-1][0][0]:b])
                group_info[-1] = merged
                continue
        groups.append([(a, b)])
        group_info.append(info)
    return [(g[0][0], g[-1][1]) for g in groups]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in " \t\n,;:-–—•*":
        start += 1
    while end > start and text[end - 1] in " \t\n,;:.!":
        end -= 1
    return start, end


def _degree_level(text: str) -> DegreeLevel | None:
    for level, pat in _LEVELS:
        if pat.search(text):
            return level
    return None


def _languages(text: str) -> tuple[list[str], str | None]:
    names = []
    for m in _LANG_NAMES_RE.finditer(text):
        if m.group(1) not in names:
            names.append(m.group(1))
    level = None
    if m := _CEFR.search(text):
        level = m.group().replace(" ", "")
    elif m := LANG_LEVEL_RE.search(text):
        level = m.group().lower()
    return names, level


def _any_of(text: str, keywords: list[JDKeyword], base: int) -> bool:
    if len(keywords) < 2:
        return False
    if ANY_OF_RE.search(text):
        return True
    for a, b in zip(keywords, keywords[1:]):
        gap = text[a.span.end - base: b.span.start - base]
        if re.fullmatch(r"\s*(?:/|\||,?\s*or|,?\s*and/or)\s*", gap, re.IGNORECASE):
            return True
    return False


def build_requirement(text: str, start: int, end: int, section: _Section) -> Requirement | None:
    start, end = _trim(text, start, end)
    item = text[start:end]
    if len(item) < 2 or not re.search(r"[A-Za-z]{2}", item):
        return None
    kws = _keywords(item, start)
    info = analyze(item, kws)
    req = Requirement(id="", text=item, category=info.category, priority="should",
                      section=section.heading, span=Span(start=start, end=end), keywords=kws)
    req.provenance["category"] = "rules"
    if info.uncertain:
        req.uncertain.append("category")
    if info.cue:
        req.priority, req.priority_cue, req.priority_source = info.cue[0], info.cue[1], "cue"
    elif section.kind in SECTION_PRIORITY:
        req.priority, req.priority_source = SECTION_PRIORITY[section.kind], "section"
    if info.years:
        req.min_years = info.years[0]
        req.field_spans["min_years"] = Span(start=start + info.years[1].start, end=start + info.years[1].end)
    if req.category == "education":
        req.degree_level = _degree_level(item)
    if req.category == "language":
        req.languages, req.language_level = _languages(item)
    req.any_of = _any_of(item, kws, start)
    return req


def _is_lead_in(text: str) -> bool:
    """'You have:' / 'Technical skills:' introduce a list; they are not requirements."""
    return text.rstrip().endswith(":") and len(text.split()) <= 12


def _guess_title(posting: str, sections: list[_Section]) -> str | None:
    m = re.search(r"^(?:job title|position|role|title)\s*[:\-–]\s*(.+)$", posting, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    if sections and sections[0].kind == _HEADER and sections[0].lines:
        first = sections[0].lines[0].text.strip("#*_ ").strip()
        if len(first.split()) <= 12 and not first.endswith((".", "!", "?")) and not BULLET_RE.match(first):
            return first
    # Prose postings: "We are hiring a DevOps Engineer to join ..."
    m = re.search(r"\b(?:hiring|looking for|seeking|searching for|recruiting)\s+(?:an?\s+|our\s+(?:next|new)\s+)?"
                  r"((?:(?:Senior|Junior|Lead|Staff|Principal)\s+)?[A-Z][\w/+#.-]*(?:\s+[A-Z][\w/+#.-]*){0,5}"
                  r"(?:\s+\([mfwdx/]+\))?)", posting)
    return m.group(1).strip() if m else None


# ------------------------------------------------------------------ optional models


class RequirementRefinement(BaseModel):
    category: Category
    parts: list[str] = Field(default_factory=list)


def _refine_with_llm(posting: JobPosting, llm, cfg: LLMConfig) -> None:
    from app.extraction.gap_fill import ground
    from app.llm.client import LLMError
    from app.llm.prompts import JD_REFINE, JD_SYSTEM

    out: list[Requirement] = []
    for req in posting.requirements:
        compound = len(req.text.split()) > 28
        if "category" not in req.uncertain and not compound:
            out.append(req)
            continue
        try:
            res = llm.generate_json(JD_REFINE.format(requirement=req.text), RequirementRefinement, system=JD_SYSTEM)
        except LLMError as exc:
            posting.warnings.append(f"LLM refinement stopped: {exc}")
            out.extend(posting.requirements[len(out):])
            break
        parts = []
        for p in res.parts:
            g = ground(p, req.text, req.span.start, cfg.min_grounding_score)
            if g is None:
                parts = []
                break
            parts.append(g[1])
        if compound and len(parts) >= 2:
            section = _Section(kind=_OTHER, heading=req.section)
            for span in sorted(parts, key=lambda s: s.start):
                sub = build_requirement(posting.text, span.start, span.end, section)
                if sub is None:
                    continue
                if sub.priority_source != "cue":  # a part without its own wording keeps the item's priority
                    sub.priority, sub.priority_source, sub.priority_cue = (
                        req.priority, req.priority_source, req.priority_cue)
                sub.provenance["span"] = "llm"
                out.append(sub)
            continue
        if "category" in req.uncertain:
            req.category = res.category
            req.provenance["category"] = "llm"
            req.uncertain = [f for f in req.uncertain if f != "category"]
        out.append(req)
    posting.requirements = out


def _enrich_with_ner(posting: JobPosting, model, threshold: float) -> None:
    from app.extraction.ner import SKILL_LABELS, find_entities, plausible_skill

    for req in posting.requirements:
        taken = [(k.span.start, k.span.end) for k in req.keywords]
        for ent in find_entities(model, req.text, req.span.start, list(SKILL_LABELS), threshold):
            s, e = ent.span.start, ent.span.end
            if any(a < e and s < b for a, b in taken) or not plausible_skill(ent.text.strip()):
                continue
            req.keywords.append(JDKeyword(text=ent.text.strip(), span=ent.span, source="ner"))
            taken.append((s, e))
        req.keywords.sort(key=lambda k: k.span.start)
        if req.category == "other" and req.keywords:
            req.category = "skill"
            req.provenance["category"] = "ner"
            req.uncertain = [f for f in req.uncertain if f != "category"]


# ------------------------------------------------------------------ pipeline


def extract_requirements(text: str) -> JobPosting:
    """Rule-based steps 1–3. ``text`` is the raw pasted posting."""
    posting_text = clean_posting(text)
    posting = JobPosting(text=posting_text)
    sections = split_jd_sections(posting_text)
    posting.title = _guess_title(posting_text, sections)
    has_req_section = any(s.kind in _REQ_KINDS for s in sections)

    stack_keywords: list[JDKeyword] = []
    header_keywords: list[JDKeyword] = []
    for sec in sections:
        for a, b in _items(posting_text, sec):
            lead_in = _is_lead_in(posting_text[a:b])
            a, b = _trim(posting_text, a, b)
            item = posting_text[a:b]
            if not item.strip():
                continue
            if sec.kind in _REQ_KINDS:
                if lead_in:
                    continue
                for pa, pb in atomic_parts(posting_text, a, b):
                    req = build_requirement(posting_text, pa, pb, sec)
                    if req is not None:
                        posting.requirements.append(req)
            elif sec.kind == _RESP:
                posting.responsibilities.append(Bullet(text=item.strip(), span=Span(start=a, end=b)))
            elif sec.kind == _STACK:
                stack_keywords += _keywords(item, a)
            elif sec.kind in (_HEADER, _OTHER):
                if sec.kind == _HEADER:
                    header_keywords += [k for k in _keywords(item, a) if k.source == "vocab"]
                # Requirements written as prose ("You have 3+ years of ...") outside a requirements list.
                if not lead_in and REQUIREMENT_CUE.search(item) and (
                    not has_req_section or sec.kind == _OTHER
                ):
                    for pa, pb in atomic_parts(posting_text, a, b):
                        req = build_requirement(posting_text, pa, pb, sec)
                        if req is not None and (req.category != "other" or req.priority_source == "cue"):
                            posting.requirements.append(req)

    if not has_req_section:
        if posting.requirements:
            posting.warnings.append(
                "No requirements list found; requirements were taken from sentences that read like requirements. "
                "Check them and adjust priorities."
            )
        else:
            posting.warnings.append("No requirements found in the posting.")

    resp_keywords = [k for r in posting.responsibilities for k in _keywords(r.text, r.span.start)]
    seen: set[str] = set()
    for kw in sorted([*header_keywords, *(k for r in posting.requirements for k in r.keywords),
                      *resp_keywords, *stack_keywords], key=lambda k: k.span.start):
        if kw.key not in seen:
            seen.add(kw.key)
            posting.keywords.append(kw)
    _number(posting)
    return posting


def _number(posting: JobPosting) -> None:
    for i, req in enumerate(posting.requirements, 1):
        req.id = f"R{i}"


def _refresh_keywords(posting: JobPosting) -> None:
    seen = {k.key for k in posting.keywords}
    for req in posting.requirements:
        for kw in req.keywords:
            if kw.key not in seen:
                seen.add(kw.key)
                posting.keywords.append(kw)
    posting.keywords.sort(key=lambda k: k.span.start)


def parse_job_posting(
    text: str,
    *,
    use_ner: bool | None = None,
    use_llm: bool | None = None,
    ner_model=None,
    llm=None,
    cfg: Config | None = None,
) -> JobPosting:
    """Turn a pasted job posting into a ``JobPosting`` with numbered requirements.

    Like ``structure_cv``: ``use_ner`` / ``use_llm`` default to the config, models can
    be injected, and missing extras or an unreachable Ollama become warnings.
    """
    cfg = cfg or get_config()
    use_ner = cfg.ner.enabled if use_ner is None else use_ner
    use_llm = cfg.llm.enabled if use_llm is None else use_llm
    posting = extract_requirements(text)

    if use_ner and posting.requirements:
        from app.extraction.ner import load_gliner

        model = ner_model
        if model is None:
            try:
                model = load_gliner(cfg.ner)
            except ImportError:
                posting.warnings.append("NER skipped: install the 'ner' extra (pip install -e .[ner]).")
            except Exception as exc:
                posting.warnings.append(f"NER skipped: could not load {cfg.ner.model} ({exc}).")
        if model is not None:
            _enrich_with_ner(posting, model, cfg.ner.threshold)

    needs_llm = any("category" in r.uncertain or len(r.text.split()) > 28 for r in posting.requirements)
    if use_llm and needs_llm:
        from app.llm.client import LLMError, OllamaClient

        client = llm
        if client is None:
            try:
                client = OllamaClient(cfg.llm)
                if not client.available():
                    posting.warnings.append(
                        f"LLM refinement skipped: Ollama is not running at {cfg.llm.host} "
                        f"or model {cfg.llm.model!r} is not pulled."
                    )
                    client = None
            except ImportError:
                posting.warnings.append("LLM refinement skipped: install the 'llm' extra (pip install -e .[llm]).")
                client = None
            except LLMError as exc:
                posting.warnings.append(f"LLM refinement skipped: {exc}")
                client = None
        if client is not None:
            _refine_with_llm(posting, client, cfg.llm)
            _number(posting)

    _refresh_keywords(posting)
    if posting.requirements and not posting.by_priority("must"):
        posting.warnings.append("No must-have requirements detected; mark the essential ones yourself.")
    return posting


def format_posting(p: JobPosting) -> str:
    out = [f"Job posting: {p.title or '(no title found)'}"]
    labels = {"must": "Must have", "should": "Should have", "nice": "Nice to have"}
    for prio in ("must", "should", "nice"):
        reqs = p.by_priority(prio)
        if not reqs:
            continue
        out.append(f"\n{labels[prio]} ({len(reqs)}):")
        for r in reqs:
            extra = []
            if r.min_years is not None:
                extra.append(f"≥{r.min_years:g} yrs")
            if r.degree_level:
                extra.append(f"≥{r.degree_level}")
            if r.languages:
                extra.append("/".join(r.languages) + (f" {r.language_level}" if r.language_level else ""))
            if r.keywords:
                extra.append(("any of: " if r.any_of else "") + ", ".join(k.canonical or k.text for k in r.keywords))
            why = f' ← "{r.priority_cue}"' if r.priority_source == "cue" else ""
            flag = "?" if "category" in r.uncertain else ""
            src = f"[{r.provenance['category']}]" if r.provenance.get("category") not in (None, "rules") else ""
            out.append(f"  {r.id:>4}  [{r.category}{flag}{src}] {r.text}{why}")
            if extra:
                out.append(f"        {' · '.join(extra)}")
    if p.responsibilities:
        out.append(f"\nResponsibilities: {len(p.responsibilities)} listed")
    if p.keywords:
        out.append("\nATS keywords: " + ", ".join(k.text for k in p.keywords))
    if p.warnings:
        out.append("\nWarnings:")
        out += [f"  ! {w}" for w in p.warnings]
    out.append("\n(priority from wording ← \"cue\", otherwise from the section; ? = category guessed. "
               "Change a priority with --priority R3=nice)")
    return "\n".join(out)
