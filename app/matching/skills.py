"""Skill vocabulary: canonical names, aliases and boundary-aware matching.

The vocabulary lives in ``skills_aliases.yaml``. ``find_skills`` returns every
non-overlapping mention in a text (longest match first) with its exact span,
so the JD parser can list verbatim keywords and the matcher can map "K8s" in a
CV to "Kubernetes" in a posting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_VOCAB_FILE = Path(__file__).with_name("skills_aliases.yaml")

# A skill must not be glued to other word characters: "C" is not in "C-level" or
# "R&D", "Java" is not in "JavaScript", ".NET" is not in "ASP.NETs".
_BEFORE = r"(?<![\w+#&@.-])"
_AFTER = r"(?![\w+#&]|[.-]\w)"


@dataclass(frozen=True)
class SkillMention:
    text: str  # verbatim, as written in the source text
    canonical: str
    start: int
    end: int


@dataclass(frozen=True)
class Vocabulary:
    canonical: dict[str, str]  # lowercased surface form -> canonical name
    pattern: re.Pattern
    sensitive: frozenset[str]  # surface forms that only match as written

    def lookup(self, name: str) -> str | None:
        """Canonical name for a surface form ("k8s" -> "Kubernetes"), else None."""
        key = " ".join(name.split()).lower()
        canon = self.canonical.get(key)
        if canon is None:
            return None
        if any(s.lower() == key for s in self.sensitive) and " ".join(name.split()) not in self.sensitive:
            return None
        return canon


def _surface_pattern(form: str) -> str:
    # Flexible whitespace and optional space around "/" ("CI / CD", "A/B").
    parts = [re.escape(p) for p in form.split()]
    body = r"\s+".join(parts)
    return body.replace(r"/", r"\s?/\s?")


@lru_cache(maxsize=4)
def load_vocabulary(path: str | None = None) -> Vocabulary:
    data = yaml.safe_load(Path(path or _VOCAB_FILE).read_text(encoding="utf-8"))
    sensitive = frozenset(data.get("case_sensitive") or [])
    canonical: dict[str, str] = {}
    forms: list[str] = []
    for name, aliases in (data.get("skills") or {}).items():
        name = str(name)
        for form in [name, *(str(a) for a in aliases or [])]:
            key = " ".join(form.split()).lower()
            canonical.setdefault(key, name)
            forms.append(form)
    alts = []
    for form in sorted(set(forms), key=len, reverse=True):
        pat = _surface_pattern(form)
        alts.append(pat if form in sensitive else f"(?i:{pat})")
    pattern = re.compile(_BEFORE + "(?:" + "|".join(alts) + ")" + _AFTER)
    return Vocabulary(canonical=canonical, pattern=pattern, sensitive=sensitive)


def find_skills(text: str, vocab: Vocabulary | None = None, base: int = 0) -> list[SkillMention]:
    vocab = vocab or load_vocabulary()
    out = []
    for m in vocab.pattern.finditer(text):
        canon = vocab.lookup(m.group())
        if canon is None:
            continue
        out.append(SkillMention(text=m.group(), canonical=canon, start=base + m.start(), end=base + m.end()))
    return out


def canonical_skill(name: str, vocab: Vocabulary | None = None) -> str | None:
    return (vocab or load_vocabulary()).lookup(name)
