"""Step 2 of the matcher: exact and alias keyword matching (PLAN.md §3 [5]).

Every posting keyword is looked up in the CV chunks: vocabulary skills by
canonical name (the CV's "K8s" satisfies the posting's "Kubernetes"), other
terms by a case-insensitive, boundary-aware search. Each hit records whether
the posting's own wording appears verbatim, which the ATS coverage check of
the coach ([8]) needs.

A plain skill requirement whose keywords are all found (one of them for
"AWS, GCP or Azure") is decided here and skips the LLM. Requirements that
ask for strength ("strong", "expert", "5+ years") or whose keywords are only
partly found go on to retrieval and the judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.extraction.schemas import JDKeyword, Requirement
from app.matching.schemas import Chunk, KeywordHit, Label
from app.matching.skills import find_skills

# Chunks that show a skill in use (not just a claim in a skills list).
USAGE_KINDS = {"bullet", "experience", "project", "summary"}
# Kinds that count as clear evidence for a plain skill requirement.
CLEAR_KINDS = USAGE_KINDS | {"skills"}

# Wording that asks for a strength a keyword alone cannot show.
QUALIFIER_RE = re.compile(
    r"\b(?:strong|solid|deep|expert\w*|extensive|advanced|proven|excellent|in-depth|very good|outstanding|"
    r"thorough|senior|production[- ]grade|production experience|mastery|significant|substantial|profound|"
    r"hands-on experience|track record)\b",
    re.IGNORECASE,
)


def _term_re(term: str) -> re.Pattern:
    body = r"\s+".join(re.escape(p) for p in term.split())
    return re.compile(rf"(?<![\w+#.-]){body}(?![\w+#]|[.-]\w)", re.IGNORECASE)


@dataclass
class CVIndex:
    """Vocabulary skills found in each chunk, by lowercased canonical name."""

    chunks: list[Chunk]
    by_canonical: dict[str, list[tuple[int, str]]] = field(default_factory=dict)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> CVIndex:
        index = cls(chunks=chunks)
        for c in chunks:
            for m in find_skills(c.text):
                index.by_canonical.setdefault(m.canonical.lower(), []).append((c.id, m.text))
        return index

    def find(self, kw: JDKeyword) -> KeywordHit:
        hit = KeywordHit(keyword=kw.text, canonical=kw.canonical)
        wanted = " ".join(kw.text.split()).lower()
        if kw.canonical:
            # The vocabulary already knows case rules ("Go" vs "go"); compare its surface forms.
            pairs = list(self.by_canonical.get(kw.canonical.lower(), []))
            hit.verbatim = any(" ".join(form.split()).lower() == wanted for _, form in pairs)
        else:  # a term outside the vocabulary is only found as written
            term = _term_re(kw.text)
            pairs = [(c.id, m.group()) for c in self.chunks for m in term.finditer(c.text)]
            hit.verbatim = bool(pairs)
        for chunk_id, form in pairs:
            if chunk_id not in hit.chunk_ids:
                hit.chunk_ids.append(chunk_id)
            if form not in hit.cv_forms:
                hit.cv_forms.append(form)
        hit.chunk_ids.sort()
        hit.found = bool(hit.chunk_ids)
        return hit


def keywords_in_chunks(chunks: list[Chunk], terms: list[str]) -> list[int]:
    """Chunk IDs that mention any of ``terms`` (canonical names or plain words)."""
    pats = [_term_re(t) for t in terms]
    out = []
    for c in chunks:
        canon = {m.canonical.lower() for m in find_skills(c.text)}
        if any(t.lower() in canon for t in terms) or any(p.search(c.text) for p in pats):
            out.append(c.id)
    return out


@dataclass
class KeywordDecision:
    label: Label | None  # fallback label when nothing better decides
    decisive: bool
    reason: str
    chunk_ids: list[int]  # evidence, usage first
    confidence: str = "medium"


def decide(req: Requirement, hits: list[KeywordHit], chunks: list[Chunk]) -> KeywordDecision | None:
    if not hits:
        return None
    by_id = {c.id: c for c in chunks}
    found = [h for h in hits if h.found]
    needed = 1 if req.any_of else len(hits)

    def rank(cid: int) -> tuple[int, int]:
        return (0 if by_id[cid].kind in USAGE_KINDS else 1 if by_id[cid].kind == "skills" else 2, cid)

    ids = sorted({cid for h in found for cid in h.chunk_ids}, key=rank)
    names = [h.canonical or h.keyword for h in found]
    missing = [h.canonical or h.keyword for h in hits if not h.found]

    if not found:
        listed = ", ".join(h.canonical or h.keyword for h in hits)
        reason = f"none of {listed} found on the CV" if len(hits) > 1 else f"{listed} not found on the CV"
        return KeywordDecision("none", False, reason, [], "low")
    if len(found) < needed:
        label: Label = "partial" if len(found) * 2 >= len(hits) else "tangential"
        return KeywordDecision(label, False, f"found {', '.join(names)}; missing {', '.join(missing)}", ids, "low")

    # Every needed keyword is there. For "all of" requirements each keyword needs a clear chunk.
    def clear(h: KeywordHit) -> bool:
        return any(by_id[cid].kind in CLEAR_KINDS for cid in h.chunk_ids)

    used = [h for h in found if any(by_id[cid].kind in USAGE_KINDS for cid in h.chunk_ids)]
    is_clear = any(clear(h) for h in found) if req.any_of else all(clear(h) for h in found)
    plain = req.category == "skill" and req.min_years is None and not QUALIFIER_RE.search(req.text)
    where = "used in experience/projects" if len(used) == len(found) or (req.any_of and used) else "listed on the CV"
    reason = f"{', '.join(names)} {where}"
    if plain and is_clear:
        return KeywordDecision("meets", True, reason, ids, "high" if where.startswith("used") else "medium")
    if not is_clear:
        reason = f"{', '.join(names)} only mentioned under {by_id[ids[0]].kind}"
        return KeywordDecision("partial", False, reason, ids, "low")
    return KeywordDecision("meets", False, reason + " (strength not judged)", ids, "low")
