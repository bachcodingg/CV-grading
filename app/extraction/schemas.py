"""Structured CV schema (PLAN.md §3 [2]).

Every entry keeps the character span of its source block in
``Document.text`` plus per-field spans, so later stages (matcher,
evidence verifier, coach) can always cite the exact CV text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Provenance = Literal["rules", "ner", "llm"]


class Span(BaseModel):
    start: int
    end: int

    def of(self, text: str) -> str:
        return text[self.start : self.end]


class PartialDate(BaseModel):
    """A CV date: the year is required, the month is often missing."""

    year: int
    month: int | None = Field(default=None, ge=1, le=12)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}" if self.month else f"{self.year:04d}"

    def ordinal(self, *, is_end: bool) -> int:
        """Months since year 0. A missing month counts as mid-year."""
        month = self.month if self.month else (6 if is_end else 7)
        return self.year * 12 + month - 1


class DateRange(BaseModel):
    start: PartialDate | None = None
    end: PartialDate | None = None
    current: bool = False  # "– Present"
    text: str = ""
    span: Span | None = None


class Sourced(BaseModel):
    """Base for anything extracted from the CV."""

    span: Span | None = None  # the whole source block
    field_spans: dict[str, Span] = Field(default_factory=dict)
    provenance: dict[str, Provenance] = Field(default_factory=dict)
    # Fields filled by a fallback guess (e.g. "first piece is the title");
    # the LLM gap-filler may re-check these.
    uncertain: list[str] = Field(default_factory=list)


class Bullet(BaseModel):
    text: str
    span: Span | None = None


class Contact(Sourced):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[str] = Field(default_factory=list)


class ExperienceEntry(Sourced):
    title: str | None = None
    org: str | None = None
    location: str | None = None
    start: PartialDate | None = None
    end: PartialDate | None = None
    current: bool = False
    dates_text: str | None = None
    bullets: list[Bullet] = Field(default_factory=list)
    duration_months: int | None = None


class EducationEntry(Sourced):
    degree: str | None = None
    field: str | None = None
    institution: str | None = None
    start: PartialDate | None = None
    end: PartialDate | None = None
    current: bool = False
    grade: str | None = None
    details: list[Bullet] = Field(default_factory=list)


class Skill(Sourced):
    name: str
    category: str | None = None
    level: str | None = None


class Project(Sourced):
    name: str | None = None
    start: PartialDate | None = None
    end: PartialDate | None = None
    current: bool = False
    technologies: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    bullets: list[Bullet] = Field(default_factory=list)


class Certification(Sourced):
    name: str
    issuer: str | None = None
    date: PartialDate | None = None


class Language(Sourced):
    name: str
    level: str | None = None


class Publication(Sourced):
    text: str
    year: int | None = None


class OtherSection(Sourced):
    kind: str  # awards, volunteering, interests, references, other...
    heading: str
    text: str


class Candidate(BaseModel):
    contact: Contact = Field(default_factory=Contact)
    summary: str | None = None
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)
    other_sections: list[OtherSection] = Field(default_factory=list)
    # Computed in code from experience dates, overlapping periods merged.
    years_of_experience: float | None = None
    # Problems the structurer noticed (missing sections, undated jobs, ...).
    warnings: list[str] = Field(default_factory=list)

    def skill_names(self) -> list[str]:
        return [s.name for s in self.skills]


# ---------------------------------------------------------------- job posting (PLAN.md §3 [4])

Category = Literal["skill", "experience", "education", "certification", "soft_skill", "language", "other"]
Priority = Literal["must", "should", "nice"]
PrioritySource = Literal["cue", "section", "default", "user"]
DegreeLevel = Literal["bachelor", "master", "phd"]


class JDKeyword(BaseModel):
    """A term from the posting, used for matching and the ATS coverage check."""

    text: str  # verbatim, as written in the posting
    canonical: str | None = None  # vocabulary name ("K8s" -> "Kubernetes"); None = not in the vocabulary
    span: Span
    source: Literal["vocab", "pattern", "ner"] = "vocab"

    @property
    def key(self) -> str:
        return (self.canonical or self.text).lower()


class Requirement(Sourced):
    id: str  # "R1", "R2", ... stable within one posting
    text: str
    category: Category
    priority: Priority
    priority_source: PrioritySource = "default"
    priority_cue: str | None = None  # the wording that decided the priority ("nice to have")
    min_years: float | None = None
    degree_level: DegreeLevel | None = None  # lowest degree the requirement accepts
    languages: list[str] = Field(default_factory=list)  # spoken languages, e.g. ["German"]
    language_level: str | None = None  # "C1", "fluent", "native", ...
    keywords: list[JDKeyword] = Field(default_factory=list)
    any_of: bool = False  # keywords are alternatives ("Python or Go", "e.g. AWS, GCP")
    section: str = ""  # heading the requirement appeared under


class JobPosting(BaseModel):
    text: str  # the cleaned posting text; every span refers to it
    title: str | None = None
    requirements: list[Requirement] = Field(default_factory=list)
    responsibilities: list[Bullet] = Field(default_factory=list)
    # Distinct terms from requirements, responsibilities and tech-stack sections (first mention each).
    keywords: list[JDKeyword] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def requirement(self, req_id: str) -> Requirement:
        for r in self.requirements:
            if r.id.lower() == req_id.lower():
                return r
        raise KeyError(req_id)

    def set_priority(self, req_id: str, priority: Priority) -> Requirement:
        """The user overrides a priority (PLAN.md §3 [4]: 'user can edit priority')."""
        req = self.requirement(req_id)
        req.priority = priority
        req.priority_source = "user"
        req.priority_cue = None
        return req

    def by_priority(self, priority: Priority) -> list[Requirement]:
        return [r for r in self.requirements if r.priority == priority]
