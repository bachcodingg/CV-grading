"""Bias-exposure scan: what on the CV could trigger bias in a screener (PLAN.md §3 [3]).

This is shown to the job seeker as information only. The tool's own grade is
always computed on the redacted copy, so nothing here changes the score; the
point is that real recruiters and ATS see the original. Each finding says
what was found, why it can matter, and a neutral alternative where one
exists. The user decides what to keep.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal

from app.config import PrivacyConfig
from app.extraction.schemas import Candidate
from app.privacy.detectors import FIELD_RE, ORIGIN_GROUP_RE, PersonalData, PersonalItem

Severity = Literal["warning", "info"]
_ORDER = {"warning": 0, "info": 1}
_SCHOOL = re.compile(r"\b(?:abitur|high school|secondary school|a-levels?|gcse|ged|gymnasium|realschule|"
                     r"mittlere reife|baccalaur[ée]at|matura)\b", re.IGNORECASE)


@dataclass
class ExposureFinding:
    code: str
    severity: Severity
    message: str
    advice: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class BiasExposureReport:
    findings: list[ExposureFinding]
    # How many personal items the grader's copy had removed, by kind.
    redacted: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def codes(self) -> set[str]:
        return {f.code for f in self.findings}

    def format(self) -> str:
        icon = {"warning": "!", "info": "i"}
        out = ["Bias-exposure scan (information only: you decide what to keep)"]
        if not self.findings:
            out.append("  ✓ Nothing on the CV is a common trigger for bias (photo, age, origin, family, religion).")
        for f in self.findings:
            out.append(f"  [{icon[f.severity]}] {f.message}")
            if f.advice:
                out.append(f"      → {f.advice}")
            for ex in f.examples[:3]:
                out.append(f"      · {ex}")
        out.append("  Your grade is always computed on a redacted copy, so none of this affects your score.")
        return "\n".join(out)


def _examples(items: list[PersonalItem]) -> list[str]:
    seen: list[str] = []
    for it in items:
        t = " ".join((it.context or it.text).split())
        if t not in seen:
            seen.append(t)
    return seen


def _labelled(item: PersonalItem) -> bool:
    return bool(FIELD_RE.match(item.text))


def scan_bias_exposure(
    found: PersonalData,
    cand: Candidate | None = None,
    *,
    today: date | None = None,
    cfg: PrivacyConfig | None = None,
    redacted: dict[str, int] | None = None,
) -> BiasExposureReport:
    cfg = cfg or PrivacyConfig()
    today = today or date.today()
    findings: list[ExposureFinding] = []

    def add(code: str, severity: Severity, items: list[PersonalItem], message: str, advice: str) -> None:
        if items:
            findings.append(ExposureFinding(code, severity, message, advice, _examples(items)))

    of = found.of_kind
    if photos := of("photo"):
        findings.append(ExposureFinding(
            "PHOTO", "warning", "A photo was detected on page 1.",
            "In the US, UK, Canada and Ireland photos are generally left off, and some recruiters reject CVs "
            "with one to avoid discrimination claims. A photo invites judgments on appearance, age, gender and "
            "ethnicity. In Germany a photo is still common but optional, and employers may not demand one.",
            [p.text for p in photos]))
    add("DATE_OF_BIRTH", "warning", of("date_of_birth"), "Your date of birth is on the CV.",
        "It reveals your age, which is irrelevant to most jobs and a well-documented source of bias. "
        "It is not expected on English-language CVs and is optional in Germany.")
    add("AGE", "warning", of("age"), "Your age is stated on the CV.",
        "Age is irrelevant to most jobs and a common source of bias. Leaving it out is standard.")
    add("PLACE_OF_BIRTH", "warning", of("place_of_birth"), "Your place of birth is on the CV.",
        "It can signal national or ethnic origin and says nothing about your qualifications.")

    nationality = of("nationality")
    groups = [i for i in nationality if i.context and ORIGIN_GROUP_RE.fullmatch(i.context)]
    add("NATIONALITY", "warning", [i for i in nationality if i not in groups],
        "Your nationality or citizenship is on the CV.",
        "If it is there to show you can work in the country, a neutral line such as "
        "\"Eligible to work in the EU, no sponsorship needed\" says the same without revealing your origin.")
    add("ORIGIN_GROUP", "info", groups,
        "A community group named after a national or ethnic origin is mentioned.",
        "Membership and leadership roles are strengths, so keep them if they matter to you. Be aware that "
        "screeners can infer your origin from them.")
    add("ETHNICITY", "warning", of("ethnicity"), "Your ethnicity is stated on the CV.",
        "It is not relevant to your qualifications and is not expected on a CV.")
    add("MARITAL_STATUS", "warning", of("marital_status"), "Your marital status is on the CV.",
        "It is irrelevant to the job and can trigger bias; studies find that married women and mothers in "
        "particular are rated lower for the same CV.")
    add("CHILDREN", "warning", of("children"), "The CV mentions children.",
        "Studies of identical CVs find a \"motherhood penalty\" (mothers get fewer callbacks). "
        "Family details are not expected on a CV.")

    religion = of("religion")
    add("RELIGION", "warning", [i for i in religion if _labelled(i)], "Your religion is stated on the CV.",
        "It is irrelevant to the job and not expected on a CV.")
    add("RELIGIOUS_AFFILIATION", "info", [i for i in religion if not _labelled(i)],
        "A religious affiliation can be inferred from the CV.",
        "If the activity shows relevant skills, keep it; describing it neutrally (e.g. \"community choir\", "
        "\"youth mentoring programme\") keeps the skill and leaves out the faith.")

    gender = of("gender")
    add("GENDER", "warning", [i for i in gender if _labelled(i)], "Your gender is stated on the CV.",
        "It is not needed on a CV.")
    add("SALUTATION", "info", [i for i in gender if not _labelled(i)],
        "A gendered title (Mr/Ms/Mrs/Herr/Frau) is used with your name.",
        "Titles are not needed on a CV; your name on its own is standard.")
    add("PRONOUNS", "info", of("pronouns"), "You list your pronouns.",
        "This is your choice. Some readers welcome it and some may react to it; the grade here is unaffected.")
    add("THIRD_PERSON", "info", [i for i in of("pronoun") if i.section in ("header", "summary", "contact")],
        "Your summary refers to you in the third person.",
        "CVs are usually written without pronouns (\"Backend engineer with 5 years…\"), which also keeps "
        "gender out of the text.")
    children_spans = [i.span for i in of("children", "marital_status") if i.span]
    terms = [i for i in of("gender_term")
             if not any(s.start <= i.span.start < s.end for s in children_spans)]
    add("GENDER_CODED_WORDS", "info", terms, "The CV contains gender-coded words.",
        "Some automated screeners have learned to penalise such words (Amazon scrapped a recruiting tool that "
        "downgraded CVs containing \"women's\"). Memberships and roles are strengths; keep them if they matter "
        "to you.")
    add("FULL_ADDRESS", "info", of("address"), "Your full street address is on the CV.",
        "City and country are enough. A full address is a privacy risk if the CV is forwarded and can "
        "invite bias about the neighbourhood or commute.")

    # --- age signals from dates ---------------------------------------------
    if cand is not None:
        school = [e for e in cand.education
                  if (e.start or e.end) and _SCHOOL.search(" ".join(filter(None, (e.degree, e.institution, e.field))))]
        if school:
            findings.append(ExposureFinding(
                "SCHOOL_DATES", "info", "School-leaving dates are listed, which lets a reader estimate your age.",
                "Once you have a degree, many people drop school entries or at least their dates.",
                [" ".join(filter(None, (e.degree, e.institution, str(e.end or e.start)))) for e in school]))
        cutoff = today.year - cfg.old_experience_years
        old = [e for e in cand.experience if e.start and e.start.year <= cutoff]
        if old:
            findings.append(ExposureFinding(
                "OLD_EXPERIENCE", "info", f"Roles starting {cfg.old_experience_years}+ years ago are dated.",
                "Long dated histories reveal age. Common advice: detail the last 10–15 years and summarise "
                "earlier roles in one line without dates.",
                [f"{e.title or '?'} @ {e.org or '?'} ({e.start})" for e in old]))

    findings.sort(key=lambda f: _ORDER[f.severity])
    return BiasExposureReport(findings=findings, redacted=dict(redacted or {}))
