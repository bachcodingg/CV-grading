"""[3] Redaction (for grading) + bias-exposure scan (for the user).

One detection pass feeds both outputs:

1. rule-based detectors: the CV's own name everywhere, contact data, address,
   "Label: value" personal fields, free-text age / birth / origin / family /
   religion / gender cues, and a photo heuristic                (always)
2. Presidio + spaCy on CPU: other people's names, origin groups (optional, ``privacy`` extra)

``redact`` builds the grader's copy with value-independent placeholders;
``scan_bias_exposure`` explains to the user what a screener could react to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from app.config import Config, get_config
from app.extraction.cv_extractor import parse_contact
from app.extraction.schemas import Candidate
from app.extraction.sections import Section, split_sections
from app.ingestion.document import Document
from app.privacy.bias_exposure import BiasExposureReport, scan_bias_exposure
from app.privacy.detectors import PersonalData, PersonalItem, detect_personal_data, line_sections
from app.privacy.redactor import RedactedCV, redact

log = logging.getLogger(__name__)

__all__ = [
    "BiasExposureReport", "PersonalData", "PersonalItem", "PrivacyResult", "RedactedCV",
    "analyze_privacy", "find_personal_data", "redact", "scan_bias_exposure",
]


@dataclass
class PrivacyResult:
    personal: PersonalData
    redacted: RedactedCV
    exposure: BiasExposureReport


def find_personal_data(
    doc: Document,
    cand: Candidate | None = None,
    *,
    sections: list[Section] | None = None,
    use_presidio: bool | None = None,
    presidio=None,
    cfg: Config | None = None,
) -> PersonalData:
    """Detect personal data. ``presidio`` injects a ready analyzer (tests, or reuse across CVs).

    A missing ``privacy`` extra or spaCy model never fails the run; it is
    reported in ``PersonalData.warnings``.
    """
    cfg = cfg or get_config()
    sections = sections if sections is not None else split_sections(doc)
    contact = cand.contact if cand is not None else parse_contact(doc, sections)
    found = detect_personal_data(doc, sections, contact, cfg.privacy)

    use_presidio = cfg.privacy.presidio_enabled if use_presidio is None else use_presidio
    if use_presidio or presidio is not None:
        from app.privacy.presidio import enrich_with_presidio, load_presidio

        analyzer = presidio
        if analyzer is None:
            try:
                analyzer = load_presidio(cfg.privacy)
            except ImportError:
                found.warnings.append("Presidio skipped: install the 'privacy' extra (pip install -e .[privacy]).")
            except OSError as exc:  # spaCy model not downloaded
                found.warnings.append(
                    f"Presidio skipped: spaCy model {cfg.privacy.spacy_model!r} is not installed "
                    f"(python -m spacy download {cfg.privacy.spacy_model}). ({exc.__class__.__name__})")
        if analyzer is not None:
            enrich_with_presidio(found, doc.text, line_sections(doc, sections), analyzer, cfg.privacy)
    return found


def analyze_privacy(
    doc: Document,
    cand: Candidate | None = None,
    *,
    use_presidio: bool | None = None,
    presidio=None,
    cfg: Config | None = None,
    today: date | None = None,
) -> PrivacyResult:
    """Redacted copy for the grader plus the user's bias-exposure report, from one detection pass."""
    cfg = cfg or get_config()
    found = find_personal_data(doc, cand, use_presidio=use_presidio, presidio=presidio, cfg=cfg)
    redacted = redact(doc.text, found, cand)
    exposure = scan_bias_exposure(found, cand, today=today, cfg=cfg.privacy, redacted=redacted.counts())
    return PrivacyResult(personal=found, redacted=redacted, exposure=exposure)
