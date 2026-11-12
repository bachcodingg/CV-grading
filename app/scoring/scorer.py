"""[7] Scoring engine (PLAN.md §3 [7]): the grade is computed in code, never by the LLM.

    overall = Σ w_i · (rating_i / 4) / Σ w_i × 100        w: must=3, should=2, nice=1
    must-have gate: any must-have rated ≤ 1 ("tangential") → overall capped at 49
    grades: A ≥ 85 · B 70–84 · C 55–69 · D 40–54 · E < 40

Ratings come from the **verified** match result (``verify_matches``). Two more
numbers are reported next to the fit grade, never mixed into it:

- **ATS keyword coverage**: posting keywords found word-for-word in the CV
  (a literal ATS filter does not know that "K8s" is "Kubernetes");
- **CV quality**: quantified achievements, action verbs, length, sections,
  contact details, dated jobs.

A confidence level says how much the grade can be trusted: low when parsing
lost text, when much of the weight rests on low-confidence ratings (no judge,
keywords only), or when many labels came from the small LLM.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.config import ScoringConfig, get_config
from app.extraction.schemas import Bullet, Candidate, JobPosting
from app.ingestion.parse_check import ParseReport
from app.matching.keywords import CVIndex
from app.matching.schemas import Chunk, MatchResult
from app.scoring.verifier import Verification

Level = str  # "high" | "medium" | "low"
_RANK = {"high": 0, "medium": 1, "low": 2}
# Parse-check findings that mean text may be missing or scrambled for the matcher too.
_LOSSY = {"OCR_USED", "GARBLED_TEXT", "TABLES", "TEXT_BOXES", "ICON_GLYPHS", "LARGE_IMAGE", "ODD_FONTS"}


@dataclass
class RequirementScore:
    requirement_id: str
    text: str
    priority: str
    weight: float
    label: str
    rating: int
    method: str
    confidence: str
    points: float  # weight × rating / 4
    max_points: float  # weight
    gate: bool = False  # this must-have triggered the must-have gate

    @property
    def lost(self) -> float:
        return self.max_points - self.points


@dataclass
class PriorityScore:
    priority: str
    count: int
    points: float
    max_points: float

    @property
    def percent(self) -> float | None:
        return round(100 * self.points / self.max_points, 1) if self.max_points else None


@dataclass
class KeywordCoverage:
    """Posting keywords in the CV, as a literal ATS filter would count them."""

    total: int = 0
    verbatim: list[str] = field(default_factory=list)  # posting wording found as written
    renamed: list[tuple[str, list[str]]] = field(default_factory=list)  # (posting term, how the CV writes it)
    missing: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float | None:
        return len(self.verbatim) / self.total if self.total else None


@dataclass
class QualityCheck:
    code: str
    passed: bool
    points: float
    max_points: float
    detail: str


@dataclass
class CVQuality:
    checks: list[QualityCheck] = field(default_factory=list)

    @property
    def score(self) -> int | None:
        total = sum(c.max_points for c in self.checks)
        return round(100 * sum(c.points for c in self.checks) / total) if total else None


@dataclass
class Grade:
    score: float | None  # 0-100 after the must-have gate; None without requirements
    raw_score: float | None  # before the gate
    grade: str | None  # A-E
    gated: bool
    requirements: list[RequirementScore]
    by_priority: list[PriorityScore]
    confidence: Level
    confidence_reasons: list[str]
    ats: KeywordCoverage
    quality: CVQuality
    faithfulness: float | None = None  # share of cited evidence that verified
    labels_changed: int = 0  # labels the verifier corrected
    warnings: list[str] = field(default_factory=list)

    @property
    def gate_failures(self) -> list[RequirementScore]:
        return [r for r in self.requirements if r.gate]

    def gaps(self, n: int | None = None) -> list[RequirementScore]:
        """Requirements that lost the most points, biggest first."""
        lost = sorted((r for r in self.requirements if r.lost > 0), key=lambda r: (-r.lost, r.requirement_id))
        return lost[:n] if n else lost

    def to_dict(self) -> dict:
        data = asdict(self)
        data["gate_failures"] = [r.requirement_id for r in self.gate_failures]
        data["ats"]["ratio"] = self.ats.ratio
        data["quality"]["score"] = self.quality.score
        for p, d in zip(self.by_priority, data["by_priority"]):
            d["percent"] = p.percent
        return data

    def format(self) -> str:
        return format_grade(self)


def band(score: float, cfg: ScoringConfig) -> str:
    for letter, lowest in cfg.bands:
        if score >= lowest:
            return letter
    return "E"


# ------------------------------------------------------------------ ATS keyword coverage


def keyword_coverage(job: JobPosting, cv_text: str | None = None, chunks: list[Chunk] | None = None) -> KeywordCoverage:
    """Posting keywords looked up in the whole CV text (what an ATS scans), else in the chunks."""
    if cv_text is not None:
        chunks = [Chunk(id=1, kind="other", text=cv_text)]
    index = CVIndex.build(chunks or [])
    out = KeywordCoverage(total=len(job.keywords))
    for kw in job.keywords:
        hit = index.find(kw)
        name = kw.text
        if hit.verbatim:
            out.verbatim.append(name)
        elif hit.found:
            out.renamed.append((name, hit.cv_forms))
        else:
            out.missing.append(name)
    return out


# ------------------------------------------------------------------ CV quality

_ACTION_VERBS = set("""
achieve accelerate administer advise analyse analyze architect assess audit automate build coach collaborate
conduct configure consolidate coordinate create cut decrease define deliver deploy design develop devise direct
drive enable engineer establish evaluate expand facilitate founded generate grow guide handle identify implement
improve increase initiate integrate introduce launch lead led maintain manage mentor migrate model modernise
modernize monitor negotiate optimise optimize orchestrate organise organize own oversee partner pioneer plan
present prioritise prioritize produce program prototype publish raise rebuild redesign reduce refactor resolve
restructure revamp run save scale secure ship simplify spearhead standardise standardize streamline strengthen
supervise support teach test train transform troubleshoot unify upgrade wrote write
built ran drove grew made won taught brought began oversaw undertook set rewrote rebuilt
""".split())
_FIRST_WORD = re.compile(r"[A-Za-z][a-z]+")
_NUMBER = re.compile(r"[$€£]\s?\d|\d+(?:[.,]\d+)?\s?(?:%|x\b|k\b|m\b|\+)|\b\d+(?:[.,]\d+)?\b")
_YEAR = re.compile(r"^(?:19[5-9]\d|20[0-4]\d)$")


def _base_verb(word: str) -> str:
    w = word.lower()
    for suffix, repl in (("ied", "y"), ("ed", ""), ("ed", "e"), ("s", "")):
        if w.endswith(suffix):
            cand = w[: -len(suffix)] + repl
            if cand in _ACTION_VERBS:
                return cand
    return w


def starts_with_action_verb(text: str) -> bool:
    m = _FIRST_WORD.match(text.strip().lstrip("-•*–· "))
    if not m:
        return False
    word = m.group()
    if word.lower() in _ACTION_VERBS or _base_verb(word) in _ACTION_VERBS:
        return True
    return len(word) > 4 and word.lower().endswith("ed")  # "Containerised", "Onboarded"


def is_quantified(text: str) -> bool:
    for m in _NUMBER.finditer(text):
        token = m.group().strip()
        if not _YEAR.match(token):
            return True
    return False


def cv_quality(cand: Candidate, cv_text: str = "", pages: int | None = None,
               cfg: ScoringConfig | None = None) -> CVQuality:
    cfg = cfg or get_config().scoring
    q = CVQuality()

    def add(code: str, passed: bool, max_points: float, detail: str, partial: float | None = None) -> None:
        pts = max_points if passed else (max_points * partial if partial else 0.0)
        q.checks.append(QualityCheck(code, passed, round(pts, 1), max_points, detail))

    bullets: list[Bullet] = [b for e in cand.experience for b in e.bullets] + [b for p in cand.projects for b in p.bullets]
    if bullets:
        share = sum(is_quantified(b.text) for b in bullets) / len(bullets)
        add("QUANTIFIED", share >= cfg.quantified_share, 20,
            f"{share:.0%} of {len(bullets)} bullets contain a number (target ≥ {cfg.quantified_share:.0%})",
            partial=min(share / cfg.quantified_share, 1.0) if cfg.quantified_share else None)
        share = sum(starts_with_action_verb(b.text) for b in bullets) / len(bullets)
        add("ACTION_VERBS", share >= cfg.action_verb_share, 15,
            f"{share:.0%} of bullets start with an action verb (target ≥ {cfg.action_verb_share:.0%})",
            partial=min(share / cfg.action_verb_share, 1.0) if cfg.action_verb_share else None)
        long = [b for b in bullets if len(b.text.split()) > cfg.max_bullet_words]
        add("BULLET_LENGTH", not long, 5,
            "bullets are concise" if not long else f"{len(long)} bullet(s) longer than {cfg.max_bullet_words} words")
    else:
        add("BULLETS", False, 40, "no bullet points under experience or projects: describe what you did and achieved")

    words = len(cv_text.split())
    if cv_text:
        ok = cfg.min_words <= words <= cfg.max_words
        add("LENGTH", ok, 10, f"{words} words" + ("" if ok else f" (aim for {cfg.min_words}–{cfg.max_words})"))
    if pages is not None:
        add("PAGES", pages <= cfg.max_pages, 5, f"{pages} page(s)" + ("" if pages <= cfg.max_pages
                                                                    else f" (aim for ≤ {cfg.max_pages})"))
    missing = [name for name, items in (("experience", cand.experience), ("education", cand.education),
                                        ("skills", cand.skills)) if not items]
    add("SECTIONS", not missing, 20, "experience, education and skills sections found" if not missing
        else f"missing section(s): {', '.join(missing)}", partial=(3 - len(missing)) / 3)
    contact = [n for n, v in (("email", cand.contact.email), ("phone", cand.contact.phone)) if not v]
    add("CONTACT", not contact, 10, "email and phone found" if not contact
        else f"no {' or '.join(contact)} found by the parser", partial=0.5 if len(contact) == 1 else None)
    if cand.experience:
        undated = [e for e in cand.experience if e.start is None]
        add("DATES", not undated, 10, "every job has dates" if not undated
            else f"{len(undated)} job(s) without a start date", partial=1 - len(undated) / len(cand.experience))
    add("SUMMARY", bool(cand.summary), 5, "summary/profile found" if cand.summary
        else "no summary: two lines on who you are and what you look for help screeners")
    return q


# ------------------------------------------------------------------ the grade


def _confidence(scores: list[RequirementScore], result: MatchResult, parse: ParseReport | None,
                verification: Verification | None, cfg: ScoringConfig) -> tuple[Level, list[str]]:
    level, reasons = "high", []

    def lower(to: Level, why: str) -> None:
        nonlocal level
        reasons.append(why)
        if _RANK[to] > _RANK[level]:
            level = to

    if not scores:
        lower("low", "the posting has no requirements to grade against")
        return level, reasons
    if parse is not None:
        errors = [f.code for f in parse.findings if f.severity == "error"]
        lossy = [f.code for f in parse.findings if f.code in _LOSSY and f.severity != "info"]
        if errors:
            lower("low", f"parsing problems may have hidden CV text ({', '.join(errors)})")
        elif lossy:
            lower("medium", f"parse-check warnings ({', '.join(lossy)}): some CV text may be missed")
    weight = sum(s.weight for s in scores)
    low = sum(s.weight for s in scores if s.confidence == "low") / weight
    if low >= cfg.low_conf_low:
        lower("low", f"{low:.0%} of the weight rests on low-confidence ratings")
    elif low >= cfg.low_conf_medium:
        lower("medium", f"{low:.0%} of the weight rests on low-confidence ratings")
    if any(s.method == "retrieval" for s in scores) and result.llm_calls == 0:
        lower("medium", "no LLM judge ran: requirements that need judgment were rated from keywords and retrieval")
    llm = sum(s.method == "llm" for s in scores) / len(scores)
    if llm > cfg.llm_share_medium:
        lower("medium", f"{llm:.0%} of the requirements were rated by the small local LLM")
    if verification is not None and verification.changed:
        lower("medium", f"the evidence check corrected {len(verification.changed)} rating(s)")
    return level, reasons


def score_match(
    job: JobPosting,
    result: MatchResult,
    *,
    verification: Verification | None = None,
    candidate: Candidate | None = None,
    cv_text: str | None = None,
    parse: ParseReport | None = None,
    cfg: ScoringConfig | None = None,
) -> Grade:
    """Grade from a (verified) match result. ``candidate``/``cv_text``/``parse`` feed CV quality, ATS coverage
    and confidence; pass the user's **original** candidate and text here (no LLM sees them)."""
    cfg = cfg or get_config().scoring
    if verification is not None:
        result = verification.result
    scores: list[RequirementScore] = []
    for req in job.requirements:
        m = result.match(req.id)
        w = cfg.weights[req.priority]
        scores.append(RequirementScore(
            requirement_id=req.id, text=req.text, priority=req.priority, weight=w, label=m.label,
            rating=m.rating, method=m.method, confidence=m.confidence, points=round(w * m.rating / 4, 2),
            max_points=w, gate=req.priority == "must" and m.rating <= cfg.gate_max_rating))

    by_priority = []
    for prio in ("must", "should", "nice"):
        group = [s for s in scores if s.priority == prio]
        if group:
            by_priority.append(PriorityScore(prio, len(group), round(sum(s.points for s in group), 2),
                                             sum(s.max_points for s in group)))
    warnings = list(result.warnings)
    if scores:
        raw = round(100 * sum(s.points for s in scores) / sum(s.max_points for s in scores), 1)
        gated = any(s.gate for s in scores) and raw > cfg.gate_cap
        score = min(raw, cfg.gate_cap) if any(s.gate for s in scores) else raw
        grade = band(score, cfg)
    else:
        raw = score = grade = None
        gated = False
        warnings.append("No requirements were found in the posting, so there is nothing to grade.")
    level, reasons = _confidence(scores, result, parse, verification, cfg)
    return Grade(
        score=score, raw_score=raw, grade=grade, gated=gated, requirements=scores, by_priority=by_priority,
        confidence=level, confidence_reasons=reasons,
        ats=keyword_coverage(job, cv_text, None if cv_text is not None else result.chunks),
        quality=cv_quality(candidate, cv_text or "", parse.pages if parse else None, cfg) if candidate else CVQuality(),
        faithfulness=verification.faithfulness if verification else None,
        labels_changed=len(verification.changed) if verification else 0,
        warnings=warnings,
    )


# ------------------------------------------------------------------ text report

_PRIO = {"must": "Must have", "should": "Should have", "nice": "Nice to have"}


def format_grade(g: Grade) -> str:
    if g.score is None:
        return "Grade: —  (no requirements to grade against)"
    out = [f"Grade: {g.grade}  ({g.score:g}/100)   confidence: {g.confidence}"]
    gate = g.gate_failures
    if gate:
        ids = ", ".join(f"{r.requirement_id} ({r.label})" for r in gate)
        tail = f"score capped at {g.score:g} (raw {g.raw_score:g})" if g.gated else "already below the cap"
        out.append(f"  Must-have gate: {ids} → {tail}")
    for p in g.by_priority:
        out.append(f"  {_PRIO[p.priority]:<13} {p.points:g} / {p.max_points:g} points ({p.percent:g}%), "
                   f"{p.count} requirement(s)")
    gaps = g.gaps(5)
    if gaps:
        out.append("\nBiggest gaps (points lost):")
        for r in gaps:
            out.append(f"  {r.requirement_id:>4} {r.priority:<6} {r.label:<10} −{r.lost:g}  {r.text}")
    a = g.ats
    if a.total:
        extra = f", {len(a.renamed)} more under another name" if a.renamed else ""
        out.append(f"\nATS keyword coverage: {len(a.verbatim)}/{a.total} word-for-word ({a.ratio:.0%}){extra}")
        if a.renamed:
            out.append("  Posting term vs your wording: " + "; ".join(
                f"\"{k}\" (CV: \"{'/'.join(forms)}\")" for k, forms in a.renamed))
        if a.missing:
            out.append("  Not on the CV: " + ", ".join(a.missing))
    if g.quality.checks:
        out.append(f"\nCV quality: {g.quality.score}/100")
        for c in g.quality.checks:
            out.append(f"  {'✔' if c.passed else '✘'} {c.detail}")
    if g.faithfulness is not None:
        changed = f"; {g.labels_changed} rating(s) corrected" if g.labels_changed else ""
        out.append(f"\nEvidence check: {g.faithfulness:.0%} of cited CV excerpts verified{changed}")
    if g.confidence_reasons:
        out.append(f"Confidence {g.confidence}:")
        out += [f"  - {r}" for r in g.confidence_reasons]
    return "\n".join(out)
