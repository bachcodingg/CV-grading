"""LLM gap-filling for fields the rules left empty or uncertain (PLAN.md §3 [2] step 4).

One entry per call with a tiny schema, which suits 1–2B models and keeps the
KV cache small. Every value the model returns must be found in the entry's
own text (fuzzy match ≥ ``LLMConfig.min_grounding_score``); the stored value
is the verbatim CV substring, so the LLM can never add facts.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel
from rapidfuzz import fuzz

from app.config import LLMConfig
from app.extraction.dates import find_date_range, parse_date
from app.extraction.schemas import Candidate, EducationEntry, ExperienceEntry, Span
from app.llm.client import LLMClient, LLMError
from app.llm.prompts import EDUCATION_FILL, EXPERIENCE_FILL, EXTRACT_SYSTEM

log = logging.getLogger(__name__)

_MAX_BLOCK = 700  # characters of the entry sent to the model


class ExperienceFill(BaseModel):
    title: str | None = None
    org: str | None = None
    start: str | None = None
    end: str | None = None


class EducationFill(BaseModel):
    degree: str | None = None
    field: str | None = None
    institution: str | None = None
    start: str | None = None
    end: str | None = None


def ground(value: str | None, block: str, base: int, min_score: float) -> tuple[str, Span] | None:
    """Locate ``value`` in ``block``; return the verbatim substring and its span."""
    if not value or not value.strip():
        return None
    value = value.strip()
    idx = block.find(value)
    if idx < 0:
        idx = block.lower().find(value.lower())
    if idx >= 0:
        return block[idx: idx + len(value)], Span(start=base + idx, end=base + idx + len(value))
    if len(value) < 4:
        return None  # too short for a meaningful fuzzy match
    al = fuzz.partial_ratio_alignment(value.lower(), block.lower())
    if al is None or al.score < min_score:
        return None
    text = block[al.dest_start: al.dest_end].strip()
    if not text:
        return None
    start = base + block.find(text, al.dest_start)
    return text, Span(start=start, end=start + len(text))


def _needed(entry, fields: tuple[str, ...]) -> list[str]:
    return [f for f in fields if getattr(entry, f) in (None, "") or f in entry.uncertain]


def _header_block(entry, doc_text: str) -> tuple[str, int] | None:
    if entry.span is None:
        return None
    end = entry.span.end
    bullets = getattr(entry, "bullets", None) or getattr(entry, "details", None) or []
    if bullets and bullets[0].span:
        end = min(end, max(bullets[0].span.end, entry.span.start + 200))
    end = min(end, entry.span.start + _MAX_BLOCK)
    return doc_text[entry.span.start:end], entry.span.start


def _accept(entry, name: str, grounded: tuple[str, Span] | None) -> bool:
    if grounded is None:
        return False
    value, span = grounded
    setattr(entry, name, value)
    entry.field_spans[name] = span
    entry.provenance[name] = "llm"
    entry.uncertain = [f for f in entry.uncertain if f != name]
    return True


def _fill_dates(entry, result, block: str, base: int, min_score: float) -> None:
    if entry.start is not None or not result.start:
        return
    start_g = ground(result.start, block, base, min_score)
    end_g = ground(result.end, block, base, min_score) if result.end else None
    if start_g is None:
        return
    joined = f"{start_g[0]} – {end_g[0]}" if end_g else start_g[0]
    rng = find_date_range(joined)
    start = rng.start if rng else parse_date(start_g[0])
    if start is None:
        return
    entry.start = start
    entry.end = rng.end if rng else (parse_date(end_g[0]) if end_g else None)
    entry.current = bool(rng and rng.current)
    entry.field_spans["dates"] = Span(start=start_g[1].start, end=(end_g or start_g)[1].end)
    entry.provenance["dates"] = "llm"


def fill_entry(entry: ExperienceEntry | EducationEntry, doc_text: str, llm: LLMClient, cfg: LLMConfig) -> bool:
    """Ask the LLM about one entry. Returns True if anything was filled."""
    if isinstance(entry, ExperienceEntry):
        fields, schema, prompt = ("title", "org"), ExperienceFill, EXPERIENCE_FILL
    else:
        fields, schema, prompt = ("degree", "institution"), EducationFill, EDUCATION_FILL
    needed = _needed(entry, fields)
    if isinstance(entry, EducationEntry) and entry.degree and entry.field is None:
        needed.append("field")
    if entry.start is None and entry.end is None:
        needed.append("dates")
    if not needed:
        return False
    blk = _header_block(entry, doc_text)
    if blk is None:
        return False
    block, base = blk
    result = llm.generate_json(prompt.format(block=block), schema, system=EXTRACT_SYSTEM)
    changed = False
    for name in needed:
        if name == "dates":
            before = entry.start
            _fill_dates(entry, result, block, base, cfg.min_grounding_score)
            changed |= entry.start is not before
            continue
        changed |= _accept(entry, name, ground(getattr(result, name, None), block, base, cfg.min_grounding_score))
    return changed


def fill_gaps(cand: Candidate, doc_text: str, llm: LLMClient, cfg: LLMConfig | None = None) -> Candidate:
    cfg = cfg or LLMConfig()
    for entry in [*cand.experience, *cand.education]:
        try:
            fill_entry(entry, doc_text, llm, cfg)
        except LLMError as exc:
            cand.warnings.append(f"LLM gap-filling stopped: {exc}")
            log.warning("LLM gap-filling stopped: %s", exc)
            break
    return cand
