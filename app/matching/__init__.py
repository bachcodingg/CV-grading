"""[5] Matcher: requirement-by-requirement evidence, cheapest step first (PLAN.md §3 [5]).

deterministic checks → keyword + alias hits → retrieval (lexical or CPU
embeddings + reranker) → small LLM judge, one requirement per call.
"""

from app.matching.chunks import build_chunks
from app.matching.matcher import format_matches, match_requirement, match_requirements
from app.matching.schemas import LABEL_RATING, LABELS, Chunk, Evidence, KeywordHit, MatchResult, RequirementMatch

__all__ = [
    "LABELS", "LABEL_RATING", "Chunk", "Evidence", "KeywordHit", "MatchResult", "RequirementMatch",
    "build_chunks", "format_matches", "match_requirement", "match_requirements",
]
