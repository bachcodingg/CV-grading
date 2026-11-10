"""Rule-based detection of personal data in a CV (PLAN.md §3 [3]).

Finds what identifies the candidate (name, email, phone, links, address) and
what can trigger bias in a human or automated screener (photo, date of birth,
age, gender markers, nationality, marital status, children, religion). Every
item keeps its span in ``Document.text`` so the redactor can cut it out
exactly and the bias-exposure scan can show the user where it is.

Context decides what counts: "single" is a marital status in a
personal-details block but not in "single-page app", and "German" is a
nationality after "Nationality:" but a skill in the languages section. The
detectors are therefore scoped to sections where needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.config import PrivacyConfig
from app.extraction.schemas import Contact, Span
from app.extraction.sections import Section
from app.ingestion.document import Document, Line
from app.patterns import EMAIL, URL, find_phones

Kind = Literal[
    "name", "email", "phone", "link", "address", "location", "photo",
    "date_of_birth", "age", "place_of_birth", "gender", "pronouns", "pronoun", "gender_term",
    "nationality", "ethnicity", "marital_status", "children", "religion",
]
Source = Literal["contact", "rules", "presidio"]

# Identity data a real CV needs (the user is not told to remove it) vs.
# characteristics that are irrelevant to the job and can trigger bias.
IDENTITY_KINDS: frozenset[str] = frozenset({"name", "email", "phone", "link", "location"})

# When two items overlap, the redactor keeps the longer one; on a tie, the
# kind listed first here wins (a name token beats "Christian" as a religion).
PRIORITY: tuple[str, ...] = (
    "name", "email", "link", "phone", "date_of_birth", "place_of_birth", "age", "address", "location",
    "nationality", "ethnicity", "marital_status", "children", "religion", "gender", "pronouns", "pronoun",
    "gender_term", "photo",
)


@dataclass
class PersonalItem:
    kind: Kind
    text: str
    span: Span | None  # None for photos (no text)
    source: Source = "rules"
    section: str = ""  # section kind the item was found in ("header", "experience", ...)
    page: int | None = None  # for photos
    # Surrounding phrase shown to the user when only part of it is redacted
    # ("Vietnamese" redacted, "Vietnamese Students' Association" shown).
    context: str = ""

    @property
    def key(self) -> tuple:
        return (self.kind, self.span.start if self.span else None, self.span.end if self.span else None, self.page)


@dataclass
class PersonalData:
    items: list[PersonalItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def of_kind(self, *kinds: str) -> list[PersonalItem]:
        return [i for i in self.items if i.kind in kinds]

    def kinds(self) -> set[str]:
        return {i.kind for i in self.items}

    def add(self, item: PersonalItem) -> None:
        if all(item.key != other.key for other in self.items):
            self.items.append(item)

    def sort(self) -> None:
        """Order by position and drop items nested in a longer item of the same kind."""
        def inside(a: PersonalItem, b: PersonalItem) -> bool:
            return (a is not b and a.kind == b.kind and a.span is not None and b.span is not None
                    and b.span.start <= a.span.start and a.span.end <= b.span.end
                    and (b.span.end - b.span.start) > (a.span.end - a.span.start))
        self.items = [a for a in self.items if not any(inside(a, b) for b in self.items)]
        self.items.sort(key=lambda i: (i.span.start if i.span else -1, PRIORITY.index(i.kind)))


# ------------------------------------------------------------------ section scope

PERSONAL_SECTIONS = {"header", "contact"}
# Where words like "married" or a religious affiliation are about the person, not the job.
PRIVATE_LIFE_SECTIONS = PERSONAL_SECTIONS | {"summary", "interests", "other", "references", "volunteering", "awards"}
# Language names here are skills, never nationalities or religions.
SKILL_SECTIONS = {"languages", "skills"}


def line_sections(doc: Document, sections: list[Section]) -> list[tuple[Line, str]]:
    """Every line with the kind of the section it belongs to (headings included)."""
    out: list[tuple[Line, str]] = []
    seen: set[int] = set()
    for sec in sections:
        for ln in ([sec.heading_line] if sec.heading_line else []) + sec.lines:
            out.append((ln, sec.kind))
            seen.add(id(ln))
    out += [(ln, "other") for ln in doc.lines if id(ln) not in seen]
    out.sort(key=lambda p: p[0].start)
    return out


# ------------------------------------------------------------------ patterns

_I = re.IGNORECASE
_MONTHS = (r"(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|januar|februar|märz|mai|juni|juli|"
           r"oktober|dezember)\.?")
_YEAR = r"(?:19|20)\d{2}"
_NUM_DATE = r"\d{1,2}\s?[./-]\s?\d{1,2}\s?[./-]\s?(?:19|20)?\d{2}"
_TEXT_DATE = (rf"\d{{1,2}}(?:st|nd|rd|th)?\.?\s+{_MONTHS}\s+{_YEAR}"
              rf"|{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+{_YEAR}")
_BIRTH_DATE = rf"(?:{_NUM_DATE}|{_TEXT_DATE}|{_MONTHS}\s+{_YEAR}|{_YEAR})"
_PLACE = r"[A-ZÄÖÜ][\wäöüß'-]+(?:(?:,\s*|\s+)[A-ZÄÖÜ][\wäöüß'-]+){0,2}"
_DATE_LIKE = re.compile(rf"{_NUM_DATE}|\d{{1,2}}[./-]{_YEAR}")

# "Label: value" fields, typical of German-style personal-details blocks.
FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "date_of_birth": ("date of birth", "date & place of birth", "date and place of birth", "birth date",
                      "birthdate", "birthday", "born", "d.o.b.", "dob", "geburtsdatum", "geboren"),
    "place_of_birth": ("place of birth", "birthplace", "birth place", "geburtsort"),
    "age": ("age", "alter"),
    "gender": ("gender", "sex", "geschlecht", "salutation", "anrede"),
    "pronouns": ("pronouns",),
    "nationality": ("nationality", "nationalities", "citizenship", "citizen of", "staatsangehörigkeit",
                    "staatsbürgerschaft", "nationalität"),
    "ethnicity": ("ethnicity", "ethnic origin", "race"),
    "marital_status": ("marital status", "family status", "civil status", "relationship status", "familienstand"),
    "children": ("children", "kids", "dependents", "dependants", "kinder"),
    "religion": ("religion", "religious affiliation", "confession", "denomination", "faith", "konfession"),
    "address": ("address", "home address", "postal address", "street", "adresse", "anschrift", "wohnort"),
}
_LABEL_TO_KIND = {label: kind for kind, labels in FIELD_LABELS.items() for label in labels}
_LABELS = "|".join(sorted((re.escape(l) for l in _LABEL_TO_KIND), key=len, reverse=True))
FIELD_RE = re.compile(rf"(?<![\w])(?P<label>{_LABELS})\s*(?::|\t)\s*", _I)
# Where a field value ends: a separator, or the next label.
_VALUE_END = re.compile(r"\s+[|·•]\s+|\t|;|\s{3,}")

BORN_RE = re.compile(
    rf"(?<![\w-])(?i:born|b\.)\s+(?:(?i:on)\s+)?(?P<date>{_BIRTH_DATE})(?:\s+(?i:in)\s+(?P<place>{_PLACE}))?"
)
BORN_IN_RE = re.compile(rf"(?<![\w-])(?i:born\s+in)\s+(?P<place>{_PLACE})(?:,?\s+(?i:in|on)\s+(?P<date>{_BIRTH_DATE}))?")
AGE_RE = re.compile(r"\b(?:aged\s+[1-9]\d\b|[1-9]\d\s*(?:-\s*)?(?:years?|yrs?)(?:\s*-\s*|\s+)old\b|[1-9]\d\s*y/o\b)", _I)

PRONOUNS_RE = re.compile(
    r"\(?\b(?:she|he|they|ze|xe|sie|er)\s*/\s*(?:her|him|them|hir|zir|xem|ihr|ihm)"
    r"(?:\s*/\s*(?:hers|his|theirs|hirs|zirs|xyrs))?\b\)?", _I)
PRONOUN_RE = re.compile(r"\b(?:he|she|him|her|his|hers|himself|herself)\b", _I)
SALUTATION_RE = re.compile(r"(?<![\w])(?:Mr|Mrs|Ms|Miss|Mx|Herr|Frau)\b\.?(?=\s+[A-ZÄÖÜ])")
GENDER_TERM_RE = re.compile(
    r"\b(?:wom[ae]n(?:'s)?|men(?:'s)?|female|male|girls?(?:')?|boys?(?:')?|ladies|gentlemen|sorority|fraternity|chairwoman|chairman|"
    r"spokeswoman|spokesman|businesswoman|businessman|sisterhood|brotherhood|mother|father|"
    r"maternity|paternity)\b", _I)

_MARITAL = (r"married|single|divorced|widowed|widow(?:er)?|engaged|separated|in a (?:civil )?partnership|"
            r"in a relationship|verheiratet|ledig|geschieden|verwitwet")
_COUNT = r"(?:\d|one|two|three|four|five|six)"
MARITAL_RE = re.compile(rf"\b(?:happily\s+)?(?:{_MARITAL})\b(?:,?\s*(?:with\s+)?{_COUNT}\s+(?:children|kids))?", _I)
CHILDREN_RE = re.compile(
    rf"\b(?:(?:mother|father|mom|mum|dad|parent)\s+of\s+{_COUNT}(?:\s+(?:children|kids|sons?|daughters?))?"
    rf"|{_COUNT}\s+(?:children|kids|sons?|daughters?)\b(?:\s*\([^)]*\))?)", _I)

DEMONYMS = {
    "afghan", "albanian", "algerian", "american", "argentinian", "argentine", "armenian", "australian",
    "austrian", "azerbaijani", "bangladeshi", "belarusian", "belgian", "bolivian", "bosnian", "brazilian",
    "british", "bulgarian", "cameroonian", "canadian", "chilean", "chinese", "colombian", "croatian",
    "cuban", "cypriot", "czech", "danish", "dutch", "ecuadorian", "egyptian", "emirati", "english",
    "eritrean", "estonian", "ethiopian", "filipino", "finnish", "french", "georgian", "german", "ghanaian",
    "greek", "guatemalan", "hungarian", "icelandic", "indian", "indonesian", "iranian", "iraqi", "irish",
    "israeli", "italian", "jamaican", "japanese", "jordanian", "kazakh", "kenyan", "korean", "kosovar",
    "kurdish", "kuwaiti", "latvian", "lebanese", "libyan", "lithuanian", "luxembourgish", "macedonian",
    "malaysian", "maltese", "mexican", "moldovan", "mongolian", "montenegrin", "moroccan", "nepalese",
    "nepali", "nigerian", "norwegian", "pakistani", "palestinian", "peruvian", "polish", "portuguese",
    "qatari", "romanian", "russian", "saudi", "scottish", "senegalese", "serbian", "singaporean",
    "slovak", "slovenian", "somali", "south african", "spanish", "sri lankan", "sudanese", "swedish",
    "swiss", "syrian", "taiwanese", "tajik", "tanzanian", "thai", "tunisian", "turkish", "ugandan",
    "ukrainian", "uruguayan", "uzbek", "venezuelan", "vietnamese", "welsh", "yemeni", "zimbabwean",
    "african", "asian", "european", "latino", "latina", "latinx", "hispanic", "arab", "roma",
}
_DEMONYM = "|".join(sorted((re.escape(d) for d in DEMONYMS), key=len, reverse=True))
CITIZEN_RE = re.compile(
    rf"(?<![\w-])(?:(?i:{_DEMONYM})|EU|E\.U\.|US|U\.S\.|USA|UK)\s+(?i:citizen(?:ship)?|national|passport(?:\s+holder)?)\b"
    rf"|\b(?i:citizen(?:ship)?\s+of)\s+(?:the\s+)?{_PLACE}"
    rf"|\b(?i:(?:dual|double)\s+(?:citizen(?:ship)?|nationality))\b(?:\s*\([^)]*\))?")
# Community groups named after an origin ("Vietnamese Students' Association").
ORIGIN_GROUP_RE = re.compile(
    rf"(?<![\w-])(?P<origin>(?i:{_DEMONYM})(?:[- ](?i:american))?)(?:\s+[\w'’-]+){{0,2}}?\s+"
    rf"(?i:association|society|community|club|network|union|council|alliance|circle|diaspora)\b")
RELIGION_RE = re.compile(
    r"\b(?:Christian(?!\s+[A-ZÄÖÜ])|Catholic|Protestant|Evangelical|Lutheran|Baptist|Methodist|Anglican|"
    r"Mormon|Muslim|Islamic|Jewish|Hindu|Buddhist|Sikh|Jain|church|Church(?!\s+[A-Z][a-z]+\s+(?:Street|Road))|"
    r"mosque|Mosque|synagogue|Synagogue|parish|Parish|Bible|bible|Quran|Koran|Torah|"
    r"Hillel|Chabad|YMCA|YWCA)\b")

_STREET_DE = (r"[A-ZÄÖÜ][\wäöüß.-]*(?:straße|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer|steig|chaussee)"
              r"\s+\d+\s?[a-zA-Z]?\b")
_STREET_EN = (r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][\w.'-]*\s+){1,4}"
              r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|"
              r"Terrace|Close|Crescent|Parkway|Pkwy|Highway|Hwy|Square|Sq)\b\.?"
              r"(?:,?\s*(?:Apt|Apartment|Suite|Unit|Flat)\.?\s*#?\w+)?")
_POSTAL = (r"\b\d{5}\s+[A-ZÄÖÜ][\wäöüß-]+(?:\s+(?:am|an der|im|bei)?\s*[A-ZÄÖÜ][\wäöüß-]+)?"  # 80331 München
           r"|\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"  # TX 78701
           r"|\b[A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2}\b")  # SW1A 1AA
ADDRESS_RE = re.compile(rf"{_STREET_DE}|{_STREET_EN}|{_POSTAL}")
_ADDRESS_GLUE = re.compile(r"[\s,]*")

# Only in the CV header; elsewhere ".net" in "ASP.NET" is a skill, not a link.
STRONG_URL = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin\.com/in|github\.com|gitlab\.com|xing\.com/profile)/[\w./-]+", _I)
_NAME_PARTICLES = {"von", "van", "de", "der", "den", "da", "di", "du", "le", "la", "bin", "binti", "al", "el", "del",
                   "dos", "das", "ter", "ten", "zu", "y"}


# ------------------------------------------------------------------ helpers


def _item(kind: str, line: Line, a: int, b: int, section: str, source: Source = "rules") -> PersonalItem:
    text = line.text[a:b]
    # Trim decoration that the regexes swallow at the edges.
    left = len(text) - len(text.lstrip(" ,;|·•–—-"))
    right = len(text) - len(text.rstrip(" ,;|·•–—-"))
    a, b = a + left, b - right
    return PersonalItem(kind=kind, text=line.text[a:b], span=Span(start=line.start + a, end=line.start + b),
                        source=source, section=section)


def _fields(line: Line, section: str) -> list[PersonalItem]:
    """"Nationality: German | Marital status: married" → one item per field (label included)."""
    out = []
    matches = list(FIELD_RE.finditer(line.text))
    for i, m in enumerate(matches):
        # A field label has to start the line or follow a separator ("Age: 31" not "in the age: of").
        before = line.text[: m.start()]
        if before.strip() and not re.search(r"(?:[|·•;,]|\t|\s{2,})\s*$", before):
            continue
        limit = matches[i + 1].start() if i + 1 < len(matches) else len(line.text)
        end = limit
        if sep := _VALUE_END.search(line.text, m.end(), limit):
            end = sep.start()
        if not line.text[m.end():end].strip(" ,"):
            continue  # a label with no value
        kind = _LABEL_TO_KIND[m.group("label").lower()]
        out.append(_item(kind, line, m.start(), end, section))
    return out


def _name_variants(name: str) -> tuple[list[str], list[str]]:
    """Full-name spellings and the individual name tokens worth redacting alone."""
    name = " ".join(name.split())
    full = {name, name.title(), name.upper()}
    tokens = [t for t in re.split(r"[\s-]+", name) if t and t.lower().strip(".") not in _NAME_PARTICLES]
    singles: set[str] = set()
    for t in tokens:
        t = t.strip(".,")
        if len(t) < 2 or (len(t) == 2 and t.endswith(".")):
            continue  # initials
        singles |= {t, t.title(), t.upper()}
    return sorted(full, key=len, reverse=True), sorted(singles, key=len, reverse=True)


def guess_name(lines: list[tuple[Line, str]]) -> str | None:
    """Fallback when the structurer found no name: "Ms. Anna Schmidt (she/her)", "Name: Jane Doe"."""
    from app.extraction.cv_extractor import LOCATION_RE, TITLE_WORDS, org_score
    from app.extraction.sections import classify_heading

    for ln, sec in lines[:10]:
        if sec not in PERSONAL_SECTIONS:
            continue
        t = re.sub(r"^\s*(?:full\s+)?name\s*:\s*", "", ln.text, flags=_I)
        t = re.sub(r"\([^)]*\)", " ", SALUTATION_RE.sub(" ", PRONOUNS_RE.sub(" ", t)))
        t = " ".join(re.split(r"\s+[|·•]\s+|\t|,|\s{3,}", t)[0].split()).strip(" ;|·•–—-")
        words = t.split()
        if not 2 <= len(words) <= 4 or re.search(r"[\d@/:]", t):
            continue
        if not all(w[:1].isupper() or w.lower() in _NAME_PARTICLES for w in words):
            continue
        if classify_heading(ln) is not None or TITLE_WORDS.search(t) or org_score(t) or LOCATION_RE.match(t):
            continue
        return t
    return None


def _find_all(pattern: re.Pattern, line: Line, kind: str, section: str) -> list[PersonalItem]:
    return [_item(kind, line, m.start(), m.end(), section) for m in pattern.finditer(line.text)]


def _photos(doc: Document, cfg: PrivacyConfig) -> list[PersonalItem]:
    out = []
    pages = {p.number: p for p in doc.pages}
    for img in doc.images:
        if img.page != 0 or not img.width or not img.height:
            continue
        w, h = img.width, img.height
        if not (cfg.photo_min_side <= min(w, h) and max(w, h) <= cfg.photo_max_side and 0.7 <= h / w <= 1.6):
            continue
        page = pages.get(img.page)
        if img.bbox and page and page.height and img.bbox[1] > page.height * 0.5:
            continue  # portraits sit near the top of page 1
        out.append(PersonalItem(kind="photo", text=f"{w:.0f}×{h:.0f} pt image on page 1", span=None,
                                section="header", page=img.page))
    return out


# ------------------------------------------------------------------ main


def detect_personal_data(
    doc: Document,
    sections: list[Section],
    contact: Contact | None = None,
    cfg: PrivacyConfig | None = None,
) -> PersonalData:
    """Rule-based personal-data scan. ``contact`` (from the structurer) supplies the name and location."""
    cfg = cfg or PrivacyConfig()
    found = PersonalData()
    add = found.add
    lines = line_sections(doc, sections)

    # --- the candidate's name, everywhere it appears ----------------------
    name = (contact.name if contact else None) or guess_name(lines)
    if name:
        full, singles = _name_variants(name)
        full_re = re.compile("|".join(rf"(?<![\w@./]){re.escape(v)}(?:'s)?(?![\w@])" for v in full))
        single_re = re.compile("|".join(rf"(?<![\w@./-]){re.escape(v)}(?:'s)?(?![\w@-])" for v in singles)) if singles else None
        for ln, sec in lines:
            items = _find_all(full_re, ln, "name", sec)
            add_items = items[:]
            if single_re:
                for m in single_re.finditer(ln.text):
                    if not any(it.span.start <= ln.start + m.start() < it.span.end for it in items):
                        add_items.append(_item("name", ln, m.start(), m.end(), sec))
            for it in add_items:
                it.source = "contact"
                add(it)
    else:
        found.warnings.append("No candidate name was recognised; the name may not be redacted.")

    if contact and contact.location and (span := contact.field_spans.get("location")):
        add(PersonalItem(kind="location", text=span.of(doc.text), span=span, source="contact",
                         section=section_of_pos(lines, span.start)))

    # --- line by line ------------------------------------------------------
    for ln, sec in lines:
        text = ln.text
        personal = sec in PERSONAL_SECTIONS
        for m in EMAIL.finditer(text):
            add(_item("email", ln, m.start(), m.end(), sec))
        for m in find_phones(text):
            if not _DATE_LIKE.fullmatch(m.group().strip()):  # "12.03.1995" is a date, not a phone
                add(_item("phone", ln, m.start(), m.end(), sec))
        emails = [(e.start(), e.end()) for e in EMAIL.finditer(text)]
        for m in (URL if personal else STRONG_URL).finditer(text):
            if any(a <= m.start() < b for a, b in emails):
                continue  # the domain part of an email address
            add(_item("link", ln, m.start(), m.end(), sec))

        for it in _fields(ln, sec):
            add(it)

        for m in BORN_RE.finditer(text):
            add(_item("date_of_birth", ln, m.start(), m.end("date"), sec))
            if m.group("place"):
                add(_item("place_of_birth", ln, m.start("place"), m.end("place"), sec))
        for m in BORN_IN_RE.finditer(text):
            add(_item("place_of_birth", ln, m.start(), m.end("place"), sec))
            if m.group("date"):
                add(_item("date_of_birth", ln, m.start("date"), m.end("date"), sec))
        for it in _find_all(AGE_RE, ln, "age", sec):
            add(it)

        for it in _find_all(PRONOUNS_RE, ln, "pronouns", sec):
            add(it)
        declared = [it.span for it in found.items if it.kind == "pronouns" and ln.start <= it.span.start < ln.end]
        for m in PRONOUN_RE.finditer(text):
            if not any(s.start <= ln.start + m.start() < s.end for s in declared):
                add(_item("pronoun", ln, m.start(), m.end(), sec))
        if personal:
            for it in _find_all(SALUTATION_RE, ln, "gender", sec):
                add(it)
        for it in _find_all(GENDER_TERM_RE, ln, "gender_term", sec):
            add(it)

        if sec not in SKILL_SECTIONS:
            for it in _find_all(CITIZEN_RE, ln, "nationality", sec):
                add(it)
            for m in ORIGIN_GROUP_RE.finditer(text):
                # Redact only the origin; "Students' Association" is still evidence of a role.
                it = _item("nationality", ln, m.start("origin"), m.end("origin"), sec)
                it.context = m.group()
                add(it)
            for it in _find_all(RELIGION_RE, ln, "religion", sec):
                add(it)
            for it in _find_all(CHILDREN_RE, ln, "children", sec):
                add(it)
        if sec in PRIVATE_LIFE_SECTIONS:
            for m in MARITAL_RE.finditer(text):
                # "single" / "engaged" / "separated" also occur in job text ("single-page app"):
                # count them only as a standalone piece of a personal-details line.
                if m.group().lower().startswith(("single", "engaged", "separated")) and not (
                        personal and _standalone(text, m.start(), m.end())):
                    continue
                add(_item("marital_status", ln, m.start(), m.end(), sec))
        if personal:
            _addresses(ln, sec, add)

    found.items.extend(_photos(doc, cfg))
    found.sort()
    return found


def _standalone(text: str, a: int, b: int) -> bool:
    """True when text[a:b] is a whole piece of the line, between separators."""
    sep = r"(?:[|·•,;:]|\t|\s{2,})"
    return bool(re.fullmatch(rf"(?:.*{sep})?\s*", text[:a])) and bool(re.fullmatch(rf"\s*(?:{sep}.*)?", text[b:]))


def _addresses(line: Line, section: str, add) -> None:
    """Street + postcode pieces on one line become a single address item."""
    matches = list(ADDRESS_RE.finditer(line.text))
    i = 0
    while i < len(matches):
        a, b = matches[i].start(), matches[i].end()
        while i + 1 < len(matches) and _ADDRESS_GLUE.fullmatch(line.text[b: matches[i + 1].start()]):
            i += 1
            b = matches[i].end()
        add(_item("address", line, a, b, section))
        i += 1


def section_of_pos(lines: list[tuple[Line, str]], pos: int) -> str:
    for ln, kind in lines:
        if ln.start <= pos <= ln.end:
            return kind
    return "other"
