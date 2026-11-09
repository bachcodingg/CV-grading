"""[2] CV Structurer: sections, dates and a structured ``Candidate``.

Pipeline (rules first, models only for what is left):

1. rule-based section splitter and entry parser      (always)
2. GLiNER zero-shot NER on CPU                        (optional, ``ner`` extra)
3. local LLM gap-filling, one entry per call          (optional, needs Ollama)
4. durations and years of experience, computed in code
"""

from __future__ import annotations

import logging
from datetime import date

from app.config import Config, get_config
from app.extraction.cv_extractor import extract_candidate
from app.extraction.dates import months_between, total_years
from app.extraction.schemas import Candidate
from app.extraction.sections import split_sections
from app.ingestion.document import Document

log = logging.getLogger(__name__)

__all__ = ["Candidate", "structure_cv"]


def _add_durations(cand: Candidate, today: date) -> None:
    periods = []
    for e in cand.experience:
        if e.start is None:
            continue
        e.duration_months = months_between(e.start, e.end, e.current, today)
        periods.append((e.start, e.end, e.current))
    cand.years_of_experience = total_years(periods, today) if periods else None


def structure_cv(
    doc: Document,
    *,
    use_ner: bool | None = None,
    use_llm: bool | None = None,
    ner_model=None,
    llm=None,
    cfg: Config | None = None,
    today: date | None = None,
) -> Candidate:
    """Turn an ingested ``Document`` into a ``Candidate``.

    ``use_ner`` / ``use_llm`` default to the config. Passing ``ner_model`` or
    ``llm`` injects a ready model (tests, or reuse across many CVs). Missing
    optional dependencies or an unreachable Ollama never fail the run; they
    are reported in ``Candidate.warnings``.
    """
    cfg = cfg or get_config()
    today = today or date.today()
    use_ner = cfg.ner.enabled if use_ner is None else use_ner
    use_llm = cfg.llm.enabled if use_llm is None else use_llm

    cand = extract_candidate(doc, split_sections(doc))

    if use_ner:
        from app.extraction.ner import enrich_with_ner, load_gliner

        try:
            model = ner_model or load_gliner(cfg.ner)
        except ImportError:
            cand.warnings.append("NER skipped: install the 'ner' extra (pip install -e .[ner]).")
            model = None
        except Exception as exc:
            cand.warnings.append(f"NER skipped: could not load {cfg.ner.model} ({exc}).")
            model = None
        if model is not None:
            enrich_with_ner(cand, doc.text, model, cfg.ner)

    if use_llm:
        from app.extraction.gap_fill import fill_gaps
        from app.llm.client import LLMError, OllamaClient

        client = llm
        if client is None:
            try:
                client = OllamaClient(cfg.llm)
                if not client.available():
                    cand.warnings.append(
                        f"LLM gap-filling skipped: Ollama is not running at {cfg.llm.host} "
                        f"or model {cfg.llm.model!r} is not pulled."
                    )
                    client = None
            except ImportError:
                cand.warnings.append("LLM gap-filling skipped: install the 'llm' extra (pip install -e .[llm]).")
                client = None
            except LLMError as exc:
                cand.warnings.append(f"LLM gap-filling skipped: {exc}")
                client = None
        if client is not None:
            fill_gaps(cand, doc.text, client, cfg.llm)

    _add_durations(cand, today)
    return cand
