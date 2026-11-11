"""[5] Matcher pipeline: cheapest step first (PLAN.md §3 [5]).

Per requirement:

1. **deterministic** checks in code (years, degree, languages, certificates):
   decide the requirement or set a ceiling (``cap``);
2. **keyword + alias** hits: a plain skill requirement with clear evidence is
   decided here;
3. **retrieval** (lexical, or CPU embeddings + reranker) picks up to three
   evidence candidates, together with the chunks steps 1–2 pointed at;
4. **LLM judge**, one requirement per call, labels the requirement and cites
   excerpt numbers. The code-computed cap always wins.

Without a judge (no Ollama, ``use_llm=False``), undecided requirements get a
cautious label from steps 1–3 with ``confidence="low"``; retrieval alone never
earns more than "partial".
"""

from __future__ import annotations

import logging
from datetime import date

from app.config import Config, get_config
from app.extraction.schemas import Candidate, JobPosting, Requirement
from app.matching.chunks import build_chunks
from app.matching.deterministic import Check, run_checks
from app.matching.keywords import CVIndex, KeywordDecision, decide
from app.matching.schemas import (
    LABEL_RATING,
    Chunk,
    Evidence,
    Label,
    MatchResult,
    Method,
    RequirementMatch,
    min_label,
)
from app.matching.retrieval import LexicalRetriever, Reranker, Retriever

log = logging.getLogger(__name__)


class _LazyJudge:
    """Connects to Ollama only when a requirement actually needs the judge."""

    def __init__(self, enabled: bool, client, cfg: Config, result: MatchResult) -> None:
        self.enabled = enabled
        self.client = client
        self.cfg = cfg
        self.result = result
        self.tried = client is not None

    def get(self):
        if not self.enabled:
            return None
        if self.client is None and not self.tried:
            self.tried = True
            from app.llm.client import LLMError, OllamaClient

            try:
                client = OllamaClient(self.cfg.llm)
                if client.available():
                    self.client = client
                else:
                    self.result.warnings.append(
                        f"LLM judge skipped: Ollama is not running at {self.cfg.llm.host} "
                        f"or model {self.cfg.llm.model!r} is not pulled; undecided requirements got "
                        "low-confidence labels from keywords and retrieval.")
            except ImportError:
                self.result.warnings.append("LLM judge skipped: install the 'llm' extra (pip install -e .[llm]).")
            except LLMError as exc:
                self.result.warnings.append(f"LLM judge skipped: {exc}")
        return self.client

    def fail(self, exc: Exception) -> None:
        self.result.warnings.append(f"LLM judge stopped: {exc}. Remaining requirements were not judged.")
        self.client, self.enabled = None, False


def _load_retrievers(use_models: bool, cfg: Config, result: MatchResult) -> tuple[Retriever, Reranker | None]:
    if not use_models:
        return LexicalRetriever(), None
    from app.matching.retrieval import load_embedding_retriever, load_reranker

    try:
        retriever: Retriever = load_embedding_retriever(cfg.matching)
    except ImportError:
        result.warnings.append("Embeddings skipped: install the 'matching' extra (pip install -e .[matching]).")
        return LexicalRetriever(), None
    except Exception as exc:
        result.warnings.append(f"Embeddings skipped: could not load {cfg.matching.embedding_model} ({exc}).")
        return LexicalRetriever(), None
    reranker = None
    if cfg.matching.reranker_enabled:
        try:
            reranker = load_reranker(cfg.matching)
        except Exception as exc:
            result.warnings.append(f"Reranker skipped: could not load {cfg.matching.reranker_model} ({exc}).")
    return retriever, reranker


def _candidates(req: Requirement, chunks: list[Chunk], pointed: list[int], retriever: Retriever,
                reranker: Reranker | None, cfg: Config) -> tuple[list[int], dict[int, float], str]:
    """Up to ``top_n`` chunk IDs for the judge, and a relevance score for each."""
    m = cfg.matching
    ranked = retriever.rank(req.text, chunks, len(chunks))
    scores = dict(ranked)
    scorer = retriever.name
    top = [cid for cid, _ in ranked[: m.top_k]]
    pool = list(dict.fromkeys(pointed + top))
    if reranker is not None and pool:
        by_id = {c.id: c for c in chunks}
        reranked = reranker.rerank(req.text, [by_id[cid] for cid in pool])
        scores, scorer = dict(reranked), reranker.name
        top = [cid for cid, _ in reranked]
    # Chunks the code pointed at come first (at most top_n - 1 of them, so retrieval always gets a say).
    head = pointed[: max(m.top_n - 1, 1)]
    order = list(dict.fromkeys(head + [cid for cid in top if scores.get(cid, 0) > 0] + pointed))
    return order[: m.top_n], scores, scorer


def _retrieval_label(top_score: float, scorer: str, cfg: Config) -> Label:
    partial, tangential = cfg.matching.fallback_thresholds.get(scorer, (0.75, 0.4))
    return "partial" if top_score >= partial else "tangential" if top_score >= tangential else "none"


def _evidence(ids: list[int], scores: dict[int, float], hits_by_chunk: dict[int, list[str]]) -> list[Evidence]:
    return [Evidence(chunk_id=cid, score=scores.get(cid), terms=hits_by_chunk.get(cid, [])) for cid in ids]


def match_requirement(req: Requirement, cand: Candidate, chunks: list[Chunk], index: CVIndex, *,
                      retriever: Retriever, reranker: Reranker | None, judge: _LazyJudge, cfg: Config,
                      today: date) -> RequirementMatch:
    hits = [index.find(k) for k in req.keywords]
    hits_by_chunk: dict[int, list[str]] = {}
    for h in hits:
        for cid in h.chunk_ids:
            hits_by_chunk.setdefault(cid, []).append(h.canonical or h.keyword)

    check: Check | None = run_checks(req, cand, chunks, cfg.matching, today)
    kw: KeywordDecision | None = decide(req, hits, chunks)
    cap = check.cap if check else None
    facts = list(check.facts) if check else []
    base = dict(requirement_id=req.id, keywords=hits, cap=cap, facts=facts)

    if check is not None and check.decisive and check.label is not None:
        return RequirementMatch(**base, label=min_label(check.label, cap), method="deterministic",
                                confidence=check.confidence, reason=check.reason,
                                evidence=_evidence(check.chunk_ids, {}, hits_by_chunk))
    if kw is not None and kw.decisive and kw.label is not None:
        label = min_label(kw.label, cap)
        reason = kw.reason + (f"; capped: {check.reason}" if cap and label != kw.label else "")
        return RequirementMatch(**base, label=label, method="keyword", confidence=kw.confidence, reason=reason,
                                evidence=_evidence(kw.chunk_ids, {}, hits_by_chunk))

    pointed = list(dict.fromkeys((check.chunk_ids if check else []) + (kw.chunk_ids if kw else [])))
    cand_ids, scores, scorer = _candidates(req, chunks, pointed, retriever, reranker, cfg)

    # The cautious label used when there is no judge (or the judge cites nothing).
    fallbacks: list[tuple[Label, Method, str]] = []
    if check is not None and check.label is not None:
        fallbacks.append((check.label, "deterministic", check.reason))
    if kw is not None and kw.label is not None:
        fallbacks.append((kw.label, "keyword", kw.reason))
    if fallbacks:
        fb_label, fb_method, fb_reason = min(fallbacks, key=lambda f: LABEL_RATING[f[0]])
    else:
        top = max((scores.get(cid, 0.0) for cid in cand_ids), default=0.0)
        fb_label, fb_method = _retrieval_label(top, scorer, cfg), "retrieval"
        fb_reason = (f"closest CV text scores {top:.2f} ({scorer}); not judged" if cand_ids
                     else "nothing related found on the CV")
    fb_label = min_label(fb_label, cap)
    fb_ids = cand_ids if fb_label != "none" else []
    if fb_method == "retrieval":  # only chunks that cleared the threshold count as evidence
        floor = cfg.matching.fallback_thresholds.get(scorer, (0.75, 0.4))[1]
        fb_ids = [cid for cid in fb_ids if scores.get(cid, 0.0) >= floor]

    by_id = {c.id: c for c in chunks}
    client = judge.get() if cand_ids else None
    if client is not None:
        from app.llm.client import LLMError
        from app.matching.judge import judge as run_judge

        try:
            verdict = run_judge(req, [by_id[cid] for cid in cand_ids], client, facts)
        except LLMError as exc:
            judge.fail(exc)
            verdict = None
        if verdict is not None:
            judge.result.llm_calls += 1
            if verdict.label != "none" and not verdict.chunk_ids:
                # No evidence, no points: the judge's label is not used.
                return RequirementMatch(**base, label=fb_label, method=fb_method, confidence="low",
                                        reason=f"{fb_reason} (judge cited no excerpt)", candidates=cand_ids,
                                        evidence=_evidence(fb_ids, scores, hits_by_chunk))
            label = min_label(verdict.label, cap)
            reason = verdict.reason
            if label != verdict.label:
                reason += f" Capped at {label!r}: {check.reason}."
            confidence = "low" if verdict.dropped else "medium"
            return RequirementMatch(**base, label=label, method="llm", confidence=confidence, reason=reason,
                                    candidates=cand_ids, evidence=_evidence(verdict.chunk_ids, scores, hits_by_chunk))

    return RequirementMatch(**base, label=fb_label, method=fb_method, confidence="low", reason=fb_reason,
                            candidates=cand_ids, evidence=_evidence(fb_ids, scores, hits_by_chunk))


def match_requirements(
    job: JobPosting,
    cand: Candidate,
    *,
    use_llm: bool | None = None,
    llm=None,
    use_models: bool | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    cfg: Config | None = None,
    today: date | None = None,
) -> MatchResult:
    """Match every requirement of ``job`` against ``cand``.

    Pass the **redacted** candidate (``analyze_privacy(...).redacted.candidate``) so the
    judge never sees name, age, origin or other personal data. ``use_llm`` / ``use_models``
    default to the config; ``llm``, ``retriever`` and ``reranker`` inject ready models.
    Missing extras or an unreachable Ollama become warnings, never errors.
    """
    cfg = cfg or get_config()
    today = today or date.today()
    use_llm = (llm is not None or cfg.llm.enabled) if use_llm is None else use_llm
    use_models = cfg.matching.models_enabled if use_models is None else use_models

    chunks = build_chunks(cand)
    result = MatchResult(chunks=chunks)
    if retriever is None:
        retriever, loaded_reranker = _load_retrievers(use_models, cfg, result)
        reranker = reranker or loaded_reranker
    result.retriever = retriever.name + (f"+{reranker.name}" if reranker is not None else "")
    judge = _LazyJudge(use_llm, llm if use_llm else None, cfg, result)
    index = CVIndex.build(chunks)

    if not chunks:
        result.warnings.append("No CV content to match against (empty or unparsed CV).")
    for req in job.requirements:
        result.matches.append(match_requirement(req, cand, chunks, index, retriever=retriever, reranker=reranker,
                                                judge=judge, cfg=cfg, today=today))
    return result


# ------------------------------------------------------------------ text report

_MARK = {"exceeds": "★", "meets": "✔", "partial": "◐", "tangential": "·", "none": "✘"}


def format_matches(result: MatchResult, job: JobPosting, original_text: str | None = None) -> str:
    """Per-requirement report. With ``original_text`` the evidence is quoted from the user's own CV."""
    counts = ", ".join(f"{n} {m}" for m, n in sorted(result.by_method().items()))
    out = [f"Requirement match ({counts or 'no requirements'}; retriever: {result.retriever}, "
           f"LLM calls: {result.llm_calls})"]
    labels = {"must": "Must have", "should": "Should have", "nice": "Nice to have"}
    for prio in ("must", "should", "nice"):
        reqs = job.by_priority(prio)
        if not reqs:
            continue
        out.append(f"\n{labels[prio]}:")
        for r in reqs:
            m = result.match(r.id)
            conf = "" if m.confidence == "high" else f", {m.confidence} confidence"
            out.append(f"  {r.id:>4} {_MARK[m.label]} {m.label:<10} [{m.method}{conf}] {r.text}")
            if m.reason:
                out.append(f"        {m.reason}")
            for ev in m.evidence[:3]:
                quote = " ".join(result.chunk(ev.chunk_id).quote(original_text).split())
                quote = quote if len(quote) <= 110 else quote[:107] + "…"
                out.append(f"        ↳ \"{quote}\"")
            renamed = [h for h in m.keywords if h.found and not h.verbatim]
            if renamed:
                out.append("        ATS: posting says " + ", ".join(
                    f"\"{h.keyword}\", CV says \"{'/'.join(h.cv_forms)}\"" for h in renamed))
    if result.warnings:
        out.append("\nWarnings:")
        out += [f"  ! {w}" for w in result.warnings]
    out.append(f"\n({_MARK['exceeds']} exceeds  {_MARK['meets']} meets  {_MARK['partial']} partial  "
               f"{_MARK['tangential']} tangential  {_MARK['none']} none. Grade and coaching come in later steps.)")
    return "\n".join(out)
