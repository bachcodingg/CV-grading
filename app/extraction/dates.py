"""CV date ranges and years of experience, computed in code (never by the LLM).

Handles "Jan 2021 – Present", "01/2018 - 12/2020", "2016–2018",
"Sept. 2019 to June 2020", "Summer 2019", "since 2021", "2020-03 – now",
"Jan '20 - Mar '21" and German month names (CVs from Germany often mix).
Odd single dates fall back to ``dateparser``.
"""

from __future__ import annotations

import re
from datetime import date

from app.extraction.schemas import DateRange, PartialDate, Span

_MONTHS = {
    "jan": 1, "january": 1, "januar": 1, "jän": 1, "feb": 2, "february": 2, "februar": 2,
    "mar": 3, "march": 3, "mär": 3, "märz": 3, "maerz": 3, "apr": 4, "april": 4,
    "may": 5, "mai": 5, "jun": 6, "june": 6, "juni": 6, "jul": 7, "july": 7, "juli": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "okt": 10, "oktober": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12, "dez": 12, "dezember": 12,
}
_SEASONS = {"spring": 4, "summer": 7, "fall": 10, "autumn": 10, "winter": 1}

_MONTH_RE = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))
_SEASON_RE = "|".join(_SEASONS)
_YEAR = r"(?:19[5-9]\d|20\d{2})"

TOKEN = (
    rf"(?:(?:{_MONTH_RE})\.?,?\s*(?:{_YEAR}|'\d{{2}})"  # Jan 2020, Sept. 2019, Jan '20
    rf"|(?:{_SEASON_RE})\s+{_YEAR}"  # Summer 2019
    rf"|(?:0?[1-9]|1[0-2])\s*[/.]\s*{_YEAR}"  # 01/2020, 1.2020
    rf"|{_YEAR}\s*[-/.]\s*(?:0[1-9]|1[0-2])(?!\d)"  # 2020-01, 2020/01
    rf"|{_YEAR})"  # 2020
)
PRESENT = (
    r"(?:present|current(?:ly)?|now|today|ongoing|to date|date|heute|jetzt|aktuell|dato)"
)
_DASH = r"(?:\s*(?:-|–|—|−|~|→|to|until|till|bis)\s*)"

RANGE_RE = re.compile(
    rf"(?<![\w/])(?P<start>{TOKEN}){_DASH}(?P<end>(?:(?:expected|exp\.|anticipated|voraussichtlich)\s*)?{TOKEN}|{PRESENT})"
    rf"(?![\w/])",
    re.IGNORECASE,
)
SINCE_RE = re.compile(rf"\b(?:since|seit|from)\s+(?P<start>{TOKEN})(?![\w/])", re.IGNORECASE)
OPEN_RE = re.compile(rf"(?<![\w/])(?P<start>{TOKEN})\s*(?:-|–|—)\s*$", re.IGNORECASE)
SINGLE_RE = re.compile(rf"(?<![\w/.]){TOKEN}(?![\w/]|\.\d)", re.IGNORECASE)


def parse_date(token: str) -> PartialDate | None:
    """Parse one date token ("Jan 2020", "01/2020", "2020", "Summer 2019")."""
    t = token.strip().lower().rstrip(".,")
    if m := re.fullmatch(rf"({_MONTH_RE})\.?,?\s*({_YEAR}|'\d{{2}})", t):
        year = m.group(2)
        y = int(year) if not year.startswith("'") else 2000 + int(year[1:])
        if year.startswith("'") and y > date.today().year + 6:
            y -= 100
        return PartialDate(year=y, month=_MONTHS[m.group(1)])
    if m := re.fullmatch(rf"({_SEASON_RE})\s+({_YEAR})", t):
        return PartialDate(year=int(m.group(2)), month=_SEASONS[m.group(1)])
    if m := re.fullmatch(rf"(0?[1-9]|1[0-2])\s*[/.]\s*({_YEAR})", t):
        return PartialDate(year=int(m.group(2)), month=int(m.group(1)))
    if m := re.fullmatch(rf"({_YEAR})\s*[-/.]\s*(0[1-9]|1[0-2])", t):
        return PartialDate(year=int(m.group(1)), month=int(m.group(2)))
    if re.fullmatch(_YEAR, t):
        return PartialDate(year=int(t))
    return _dateparser_fallback(token)


def _dateparser_fallback(token: str) -> PartialDate | None:
    if not re.search(r"\d{4}", token):
        return None
    import dateparser

    parsed = dateparser.parse(
        token, languages=["en", "de"], settings={"PREFER_DAY_OF_MONTH": "first", "REQUIRE_PARTS": ["year"]}
    )
    if parsed is None:
        return None
    has_month = bool(re.search(r"[a-z]{3}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", token.lower()))
    return PartialDate(year=parsed.year, month=parsed.month if has_month else None)


def _valid(start: PartialDate | None, end: PartialDate | None) -> bool:
    max_year = date.today().year + 8  # expected graduation dates
    for d in (start, end):
        if d is not None and not 1950 <= d.year <= max_year:
            return False
    if start and end and start.ordinal(is_end=False) > end.ordinal(is_end=True) + 11:
        return False
    return True


def find_date_range(text: str, offset: int = 0) -> DateRange | None:
    """Find the first date range (or open-ended "since X" / "X –") in ``text``."""
    candidates: list[tuple[int, DateRange]] = []
    for m in RANGE_RE.finditer(text):
        start = parse_date(m.group("start"))
        end_txt = m.group("end")
        current = re.fullmatch(PRESENT, end_txt.strip(), re.IGNORECASE) is not None
        end_txt = re.sub(r"^(?:expected|exp\.|anticipated|voraussichtlich)\s*", "", end_txt, flags=re.I)
        end = None if current else parse_date(end_txt)
        if start and (end or current) and _valid(start, end):
            candidates.append((m.start(), DateRange(start=start, end=end, current=current, text=m.group(),
                                                    span=Span(start=offset + m.start(), end=offset + m.end()))))
            break
    for rx in (SINCE_RE, OPEN_RE):
        m = rx.search(text)
        if m and (start := parse_date(m.group("start"))) and _valid(start, None):
            candidates.append((m.start(), DateRange(start=start, current=True, text=m.group().strip(),
                                                    span=Span(start=offset + m.start(), end=offset + m.end()))))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])[1]


def find_single_date(text: str, offset: int = 0) -> tuple[PartialDate, Span] | None:
    """Find a lone date such as a graduation year or certification date."""
    for m in SINGLE_RE.finditer(text):
        d = parse_date(m.group())
        if d and _valid(d, None):
            return d, Span(start=offset + m.start(), end=offset + m.end())
    return None


def months_between(start: PartialDate, end: PartialDate | None, current: bool, today: date) -> int:
    """Inclusive length in months; open ranges run until ``today``."""
    s = start.ordinal(is_end=False)
    e = (today.year * 12 + today.month - 1) if (current or end is None) else end.ordinal(is_end=True)
    if end is not None and end.month is None and start.month is None and end.year == start.year:
        e = s + 5  # "2020 – 2020": count half a year
    return max(e - s + 1, 0)


def total_years(periods: list[tuple[PartialDate, PartialDate | None, bool]], today: date | None = None) -> float:
    """Total years covered by ``periods``, merging overlaps (two parallel jobs count once)."""
    today = today or date.today()
    now = today.year * 12 + today.month - 1
    intervals = []
    for start, end, current in periods:
        s = start.ordinal(is_end=False)
        n = months_between(start, end, current, today)
        if n > 0:
            intervals.append((s, min(s + n - 1, now)))
    intervals.sort()
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    months = sum(e - s + 1 for s, e in merged if e >= s)
    return round(months / 12, 1)
