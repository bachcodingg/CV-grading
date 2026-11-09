"""Optional zero-shot NER with GLiNER (step 3 of PLAN.md §3 [2]).

Runs on the CPU only, so it never touches the 2 GB VRAM budget. It adds what
rules cannot see:

- skills/tools mentioned in bullets, projects and the summary (not only in
  the skills section)
- organisations for experience entries whose ``org`` is empty or uncertain
- degrees for education entries the regex missed

Entities are located by offsets, so every value keeps its source span.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from app.config import NERConfig
from app.extraction.schemas import Candidate, Skill, Span

log = logging.getLogger(__name__)

SKILL_LABELS = ("skill", "programming language", "software tool")
_MAX_CHARS = 1200  # GLiNER's context is ~384 tokens; keep chunks well below it


class EntityModel(Protocol):
    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5) -> list[dict]: ...


@dataclass
class Entity:
    text: str
    label: str
    score: float
    span: Span


def load_gliner(cfg: NERConfig) -> EntityModel:
    """Load GLiNER on the CPU. Raises ImportError if the ``ner`` extra is missing."""
    from gliner import GLiNER

    model = GLiNER.from_pretrained(cfg.model, map_location="cpu")
    model.to("cpu")
    model.eval()
    return model


def _chunks(text: str, base: int) -> list[tuple[str, int]]:
    """Split at line boundaries into pieces GLiNER can take in one pass."""
    out, start = [], 0
    while start < len(text):
        end = min(start + _MAX_CHARS, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut > start:
                end = cut
        out.append((text[start:end], base + start))
        start = end
    return out


def find_entities(model: EntityModel, text: str, base: int, labels: list[str], threshold: float) -> list[Entity]:
    entities = []
    for chunk, offset in _chunks(text, base):
        if not chunk.strip():
            continue
        for e in model.predict_entities(chunk, labels, threshold=threshold):
            entities.append(Entity(text=e["text"], label=e["label"], score=float(e["score"]),
                                   span=Span(start=offset + e["start"], end=offset + e["end"])))
    return entities


_GENERIC = {
    "app", "apps", "web app", "application", "applications", "framework", "frameworks", "tool", "tools",
    "system", "systems", "platform", "platforms", "pipeline", "pipelines", "model", "models", "dashboard",
    "dashboards", "service", "services", "api", "apis", "website", "software", "code", "data", "database",
}


def plausible_skill(text: str) -> bool:
    """Reject phrases like "Designed A/B testing framework" or "Web app"."""
    words = text.split()
    if not words or len(words) > 4 or text.lower() in _GENERIC:
        return False
    first = words[0].lower()
    if len(words) > 1 and len(first) > 4 and (first.endswith("ed") or first.endswith("ing")):
        return False  # starts with an action verb: a bullet fragment, not a skill
    return True


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", name.lower())


def enrich_with_ner(cand: Candidate, doc_text: str, model: EntityModel, cfg: NERConfig | None = None) -> Candidate:
    cfg = cfg or NERConfig()
    known = {_norm(s.name) for s in cand.skills}

    # Skills mentioned in context.
    regions: list[Span] = [b.span for e in cand.experience for b in e.bullets if b.span]
    regions += [b.span for p in cand.projects for b in p.bullets if b.span]
    for region in regions:
        for ent in find_entities(model, region.of(doc_text), region.start, list(SKILL_LABELS), cfg.threshold):
            key = _norm(ent.text)
            if not key or key in known or not plausible_skill(ent.text.strip()):
                continue
            known.add(key)
            skill = Skill(name=ent.text.strip(), span=ent.span, category="mentioned")
            skill.provenance["name"] = "ner"
            cand.skills.append(skill)

    # Organisations for jobs without a confident org.
    for entry in cand.experience:
        if entry.span is None or (entry.org and "org" not in entry.uncertain):
            continue
        header_end = entry.bullets[0].span.start if entry.bullets and entry.bullets[0].span else entry.span.end
        header = doc_text[entry.span.start:header_end]
        orgs = [e for e in find_entities(model, header, entry.span.start, ["organization"], cfg.threshold)
                if entry.title is None or e.text not in entry.title]
        if orgs:
            best = max(orgs, key=lambda e: e.score)
            entry.org = best.text
            entry.field_spans["org"] = best.span
            entry.provenance["org"] = "ner"
            entry.uncertain = [f for f in entry.uncertain if f != "org"]

    # Degrees the regex missed.
    for edu in cand.education:
        if edu.degree or edu.span is None:
            continue
        degrees = find_entities(model, edu.span.of(doc_text), edu.span.start, ["degree"], cfg.threshold)
        if degrees:
            best = max(degrees, key=lambda e: e.score)
            edu.degree = best.text
            edu.field_spans["degree"] = best.span
            edu.provenance["degree"] = "ner"
    return cand
