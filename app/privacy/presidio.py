"""Optional Presidio + spaCy pass on top of the rule-based detectors (CPU only).

Rules know the CV's own name and the usual "Label: value" fields. A model adds
what they cannot see: other people's names (referees), and nationality,
religious or political groups mentioned in free text (Presidio's ``NRP``).

Models are only run where their false positives cannot damage grading:
person names are never looked for in experience/skills text (spaCy tags
"Spring Boot" or "Kafka" as people often enough), and ``NRP`` is skipped
in the languages and skills sections, where "German" is a skill.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.config import PrivacyConfig
from app.extraction.schemas import Span
from app.privacy.detectors import PERSONAL_SECTIONS, PRIVATE_LIFE_SECTIONS, SKILL_SECTIONS, PersonalData, PersonalItem

log = logging.getLogger(__name__)

ENTITY_KIND = {"PERSON": "name", "NRP": "nationality", "LOCATION": "address"}
_PERSON_SECTIONS = PRIVATE_LIFE_SECTIONS | {"publications"}


class Analyzer(Protocol):
    """The part of ``presidio_analyzer.AnalyzerEngine`` used here."""

    def analyze(self, text: str, language: str, entities: list[str] | None = None, **kwargs: Any) -> list[Any]: ...


def load_presidio(cfg: PrivacyConfig) -> Analyzer:
    """Presidio with a local spaCy model. Raises ImportError / OSError if not installed."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": cfg.spacy_model}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


def _wanted(entity: str, section: str) -> bool:
    if entity == "PERSON":
        return section in _PERSON_SECTIONS
    if entity == "NRP":
        return section not in SKILL_SECTIONS
    if entity == "LOCATION":
        return section in PERSONAL_SECTIONS
    return False


def enrich_with_presidio(
    found: PersonalData,
    text: str,
    line_kinds: list[tuple[Any, str]],
    analyzer: Analyzer,
    cfg: PrivacyConfig | None = None,
) -> PersonalData:
    """Add model-found items. Rule items win: model hits inside them are dropped."""
    cfg = cfg or PrivacyConfig()
    # Run per section block, so each hit is checked against its own section.
    blocks: list[tuple[int, int, str]] = []
    for ln, kind in line_kinds:
        if blocks and blocks[-1][2] == kind:
            blocks[-1] = (blocks[-1][0], ln.end, kind)
        else:
            blocks.append((ln.start, ln.end, kind))
    covered = [i.span for i in found.items if i.span]
    for start, end, section in blocks:
        entities = [e for e in ENTITY_KIND if _wanted(e, section)]
        if not entities:
            continue
        for r in analyzer.analyze(text=text[start:end], language="en", entities=entities):
            if r.score < cfg.presidio_threshold:
                continue
            a, b = start + r.start, start + r.end
            if any(s.start <= a and b <= s.end for s in covered):
                continue
            if r.entity_type == "PERSON" and len(text[a:b].strip(" .")) <= 1:
                continue  # a bare initial ("J.")
            # A location that is only a city is fine on a CV; keep it for redaction, not as an address.
            kind = ENTITY_KIND[r.entity_type]
            if kind == "address":
                kind = "location"
            found.add(PersonalItem(kind=kind, text=text[a:b], span=Span(start=a, end=b),
                                   source="presidio", section=section))
    found.sort()
    return found
