"""[6] Evidence verifier and [7] scoring engine (PLAN.md §3 [6]–[7]).

``grade_cv`` runs the whole pipeline for one CV and one posting:

    parse-check → structure → redact → match (redacted CV) → verify evidence → score

The LLM (if any) only ever sees the redacted CV; CV quality and ATS keyword
coverage are computed in code from the user's own text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.config import Config, get_config
from app.extraction import structure_cv
from app.extraction.schemas import Candidate, JobPosting
from app.ingestion.document import Document
from app.ingestion.parse_check import ParseReport, check_document
from app.matching import match_requirements
from app.matching.schemas import MatchResult
from app.privacy import PrivacyResult, analyze_privacy
from app.scoring.scorer import Grade, format_grade, score_match
from app.scoring.verifier import Verification, check_chunk, find_quote, verify_matches

__all__ = [
    "Grade", "GradeRun", "Verification", "check_chunk", "find_quote", "format_grade", "grade_cv", "score_match",
    "verify_matches",
]


@dataclass
class GradeRun:
    parse: ParseReport
    candidate: Candidate  # the user's own structured CV (not redacted)
    privacy: PrivacyResult
    matches: MatchResult  # as the matcher returned it
    verification: Verification  # verification.result is what was scored
    grade: Grade


def grade_cv(
    doc: Document,
    job: JobPosting,
    *,
    use_llm: bool | None = None,
    llm=None,
    use_models: bool | None = None,
    use_ner: bool | None = None,
    use_presidio: bool | None = None,
    cfg: Config | None = None,
    today: date | None = None,
) -> GradeRun:
    """Grade ``doc`` for ``job``. Optional models default to the config; missing ones become warnings."""
    cfg = cfg or get_config()
    parse = check_document(doc, cfg.ingestion)
    cand = structure_cv(doc, use_ner=use_ner, use_llm=use_llm, llm=llm, cfg=cfg, today=today)
    priv = analyze_privacy(doc, cand, use_presidio=use_presidio, cfg=cfg, today=today)
    matches = match_requirements(job, priv.redacted.candidate, use_llm=use_llm, llm=llm, use_models=use_models,
                                 cfg=cfg, today=today)
    verification = verify_matches(matches, doc.text, cfg.verifier)
    grade = score_match(job, matches, verification=verification, candidate=cand, cv_text=doc.text, parse=parse,
                        cfg=cfg.scoring)
    return GradeRun(parse, cand, priv, matches, verification, grade)
