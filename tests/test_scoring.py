import json
from datetime import date

import pytest

from app.cli import main
from app.config import Config, LLMConfig, ScoringConfig, VerifierConfig
from app.extraction.jd_parser import extract_requirements
from app.extraction.schemas import (
    Bullet,
    Candidate,
    Contact,
    EducationEntry,
    ExperienceEntry,
    JDKeyword,
    JobPosting,
    PartialDate,
    Requirement,
    Skill,
    Span,
)
from app.ingestion import load_document
from app.llm.client import OllamaClient
from app.matching.schemas import Chunk, Evidence, MatchResult, RequirementMatch
from app.scoring import grade_cv, score_match, verify_matches
from app.scoring.scorer import band, cv_quality, is_quantified, keyword_coverage, starts_with_action_verb
from app.scoring.verifier import check_chunk, find_quote
from fakes import FakeOllama
from test_jd_parser import POSTING

TODAY = date(2026, 11, 9)
OFFLINE = Config(llm=LLMConfig(enabled=False))
VCFG = VerifierConfig()
SCFG = ScoringConfig()

CV = """Jane Doe
Profile
Backend engineer who likes data platforms.
Experience
Senior Software Engineer – Acme GmbH
Jan 2021 – Present
Led migration of 40 services to K8s, cutting infra cost by 30%
Skills
Python, Docker
"""


def _at(text: str) -> Span:
    start = CV.index(text)
    return Span(start=start, end=start + len(text))


def _chunks() -> list[Chunk]:
    return [
        Chunk(id=1, kind="summary", text="Backend engineer who likes data platforms."),  # no span
        Chunk(id=2, kind="experience", text="Senior Software Engineer, Acme GmbH (2021-01 – present)",
              span=_at("Senior Software Engineer – Acme GmbH\nJan 2021 – Present")),
        Chunk(id=3, kind="bullet", text="Led migration of 40 services to K8s, cutting infra cost by 30%",
              span=_at("Led migration of 40 services to K8s, cutting infra cost by 30%")),
        Chunk(id=4, kind="skills", text="Skills: Python, Docker", span=_at("Python, Docker")),
    ]


# ------------------------------------------------------------------ [6] verifier


def test_chunks_verify_against_the_original_cv():
    checks = {c.id: check_chunk(c, CV, VCFG) for c in _chunks()}
    assert all(c.verified for c in checks.values())
    assert checks[3].how == "span" and checks[2].how == "span"  # rebuilt header: words, dates by year
    assert checks[1].how == "located" and checks[1].span.of(CV) == "Backend engineer who likes data platforms."


def test_invented_chunk_text_fails():
    fake = Chunk(id=9, kind="bullet", text="Founded a robotics start-up with 200 employees")
    assert not check_chunk(fake, CV, VCFG).verified
    wrong_span = Chunk(id=9, kind="bullet", text="Founded a robotics start-up", span=_at("Python, Docker"))
    assert not check_chunk(wrong_span, CV, VCFG).verified


def test_chunk_with_a_stale_span_is_located_elsewhere():
    moved = Chunk(id=3, kind="bullet", text="Led migration of 40 services to K8s", span=_at("Python, Docker"))
    check = check_chunk(moved, CV, VCFG)
    assert check.verified and check.how == "located"
    assert check.span.of(CV) == "Led migration of 40 services to K8s"


def test_find_quote():
    assert find_quote("led migration of 40 services", CV).span.of(CV) == "Led migration of 40 services"
    assert find_quote("Led migraton of 40 services to K8s", CV).score >= 90  # small typo still matches
    assert find_quote("[NAME] Profile", CV) is not None  # placeholders from the redacted copy are skipped
    assert find_quote("Led migration of 400 services to AWS", CV) is None
    assert find_quote("", CV) is None


def _result(*matches: RequirementMatch, retriever="lexical") -> MatchResult:
    return MatchResult(matches=list(matches), chunks=_chunks(), retriever=retriever)


def _m(rid, label, method="llm", ids=(), scores=None, cap=None, terms=None, confidence="medium"):
    scores = scores or {}
    ev = [Evidence(chunk_id=i, score=scores.get(i), terms=(terms or {}).get(i, [])) for i in ids]
    return RequirementMatch(requirement_id=rid, label=label, method=method, confidence=confidence, reason="r",
                            evidence=ev, cap=cap)


def test_no_evidence_no_points():
    res = _result(_m("R1", "meets", ids=[7]), _m("R2", "partial", method="keyword", ids=[3, 99]))
    ver = verify_matches(res, CV, VCFG)
    r1, r2 = ver.result.match("R1"), ver.result.match("R2")
    assert r1.label == "none" and r1.evidence == [] and r1.confidence == "low" and "verifier" in r1.reason
    assert r2.label == "partial" and [e.chunk_id for e in r2.evidence] == [3]
    assert [i.code for i in ver.issues] == ["INVALID_CHUNK", "NO_EVIDENCE", "INVALID_CHUNK"]
    assert ver.evidence_checked == 3 and ver.evidence_verified == 1
    assert res.match("R1").label == "meets"  # the matcher's result is not modified


def test_unverifiable_chunk_is_dropped():
    res = _result(_m("R1", "meets", ids=[3]))
    res.chunks[2].text = "Led migration of 400 services to AWS Lambda for NASA"
    ver = verify_matches(res, CV, VCFG)
    assert ver.result.match("R1").label == "none"
    assert {i.code for i in ver.issues} == {"UNVERIFIED_EVIDENCE", "NO_EVIDENCE"}
    assert ver.faithfulness == 0.0


def test_deterministic_fact_without_a_cited_chunk_is_kept():
    res = _result(_m("R1", "partial", method="deterministic", ids=[]))
    ver = verify_matches(res, CV, VCFG)
    assert ver.result.match("R1").label == "partial" and not ver.issues and ver.faithfulness is None


def test_low_support_downgrades_a_judge_meets():
    weak = _m("R1", "exceeds", ids=[4], scores={4: 0.05})
    keyword = _m("R2", "meets", ids=[4], scores={4: 0.05}, terms={4: ["Python"]})
    strong = _m("R3", "meets", ids=[3], scores={3: 0.6})
    ver = verify_matches(_result(weak, keyword, strong), CV, VCFG)
    assert ver.result.match("R1").label == "partial" and ver.issues_for("R1")[0].code == "LOW_SUPPORT"
    assert ver.result.match("R2").label == "meets"  # a requirement keyword is in the chunk
    assert ver.result.match("R3").label == "meets"
    # Reranker scale: 0.05 is fine there.
    ver = verify_matches(_result(_m("R1", "meets", ids=[4], scores={4: 0.05}), retriever="embeddings+reranker"),
                         CV, VCFG)
    assert ver.result.match("R1").label == "meets"


def test_cap_is_enforced_again():
    ver = verify_matches(_result(_m("R1", "meets", method="keyword", ids=[2], cap="tangential")), CV, VCFG)
    assert ver.result.match("R1").label == "tangential" and ver.changed[0].code == "CAP_EXCEEDED"


def test_verified_chunks_get_a_span_for_quoting():
    ver = verify_matches(_result(_m("R1", "meets", ids=[1])), CV, VCFG)
    assert ver.result.chunk(1).quote(CV) == "Backend engineer who likes data platforms."


# ------------------------------------------------------------------ [7] scorer


def _job(*reqs: tuple[str, str], keywords=()) -> JobPosting:
    rs = [Requirement(id=f"R{i}", text=t, category="skill", priority=p, span=Span(start=0, end=1))
          for i, (t, p) in enumerate(reqs, 1)]
    return JobPosting(text="", requirements=rs, keywords=list(keywords))


def _labels(*labels: str) -> MatchResult:
    return MatchResult(matches=[_m(f"R{i}", lab, method="keyword", ids=[3], confidence="high")
                                for i, lab in enumerate(labels, 1)], chunks=_chunks())


def test_weighted_score_and_bands():
    job = _job(("a", "must"), ("b", "should"), ("c", "nice"))
    g = score_match(job, _labels("meets", "partial", "none"), cfg=SCFG)
    # (3·3/4 + 2·2/4 + 1·0) / 6 × 100 = 54.2
    assert g.score == g.raw_score == 54.2 and g.grade == "D" and not g.gated
    assert [(p.priority, p.points, p.max_points) for p in g.by_priority] == [
        ("must", 2.25, 3.0), ("should", 1.0, 2.0), ("nice", 0.0, 1.0)]
    assert [r.requirement_id for r in g.gaps()] == ["R2", "R3", "R1"]
    assert score_match(job, _labels("exceeds", "exceeds", "exceeds"), cfg=SCFG).grade == "A"
    assert [band(s, SCFG) for s in (85, 84.9, 70, 55, 40, 39.9)] == ["A", "B", "B", "C", "D", "E"]


def test_must_have_gate_caps_the_score():
    job = _job(("a", "must"), ("b", "must"), ("c", "should"), ("d", "nice"))
    g = score_match(job, _labels("tangential", "exceeds", "exceeds", "exceeds"), cfg=SCFG)
    # (3·1/4 + 3 + 2 + 1) / 9 × 100 = 75, capped by the gate
    assert g.raw_score == 75.0 and g.score == 49 and g.gated and g.grade == "D"
    assert [r.requirement_id for r in g.gate_failures] == ["R1"]
    assert "Must-have gate: R1 (tangential) → score capped at 49 (raw 75)" in g.format()
    # A failed nice-to-have does not gate.
    g = score_match(job, _labels("meets", "meets", "meets", "none"), cfg=SCFG)
    assert not g.gate_failures and g.score == 66.7


def test_no_requirements():
    g = score_match(_job(), MatchResult(), cfg=SCFG)
    assert g.score is None and g.grade is None and g.confidence == "low"
    assert "nothing to grade" in g.warnings[0] and "no requirements" in g.format()


def test_confidence_reasons():
    job = _job(("a", "must"), ("b", "must"))
    res = _labels("meets", "meets")
    assert score_match(job, res, cfg=SCFG).confidence == "high"
    res.matches[1].confidence = "low"
    res.matches[1].method = "retrieval"
    g = score_match(job, res, cfg=SCFG)
    assert g.confidence == "low" and any("50% of the weight" in r for r in g.confidence_reasons)
    assert any("no LLM judge" in r for r in g.confidence_reasons)


def test_keyword_coverage_is_literal():
    def kw(text, canonical):
        return JDKeyword(text=text, canonical=canonical, span=Span(start=0, end=len(text)))

    job = _job(("a", "must"), keywords=[kw("Kubernetes", "Kubernetes"), kw("Python", "Python"),
                                         kw("Redis", "Redis")])
    cov = keyword_coverage(job, CV)
    assert cov.verbatim == ["Python"] and cov.renamed == [("Kubernetes", ["K8s"])] and cov.missing == ["Redis"]
    assert cov.ratio == pytest.approx(1 / 3)


def test_bullet_helpers():
    assert starts_with_action_verb("Led migration of 40 services")
    assert starts_with_action_verb("• Containerised the build")
    assert starts_with_action_verb("Optimizes queries")
    assert not starts_with_action_verb("Responsible for the database")
    assert is_quantified("cut cost by 30%") and is_quantified("served 2 TB per day")
    assert not is_quantified("Since 2019 in the platform team")


def test_cv_quality_checks():
    good = Candidate(
        contact=Contact(email="a@b.c", phone="+49 1"), summary="Engineer.",
        experience=[ExperienceEntry(title="Eng", start=PartialDate(year=2020), bullets=[
            Bullet(text="Cut build time by 40%"), Bullet(text="Built the Python API")])],
        education=[EducationEntry(degree="B.Sc.")], skills=[Skill(name="Python")])
    q = cv_quality(good, "word " * 400, pages=1, cfg=SCFG)
    assert q.score == 100 and all(c.passed for c in q.checks)
    bare = Candidate(experience=[ExperienceEntry(title="Eng")])
    q = cv_quality(bare, "word " * 50, pages=3, cfg=SCFG)
    failed = {c.code for c in q.checks if not c.passed}
    assert {"BULLETS", "LENGTH", "PAGES", "SECTIONS", "CONTACT", "DATES", "SUMMARY"} <= failed
    assert q.score < 20


# ------------------------------------------------------------------ pipeline


@pytest.fixture(scope="module")
def run(single_column_pdf):
    doc = load_document(single_column_pdf)
    return doc, grade_cv(doc, extract_requirements(POSTING), use_llm=False, use_ner=False, use_presidio=False,
                         use_models=False, cfg=OFFLINE, today=TODAY)


def test_grade_without_a_judge(run):
    _, r = run
    g = r.grade
    assert g.grade == "E" and g.score == 37.5  # FastAPI/Django, PostgreSQL/Redis, communication missing
    assert {x.requirement_id for x in g.gate_failures} == {"R1", "R3", "R5", "R7"}
    assert g.confidence == "low" and r.verification.faithfulness == 1.0
    assert g.ats.verbatim == ["Python", "AWS"] and "PostgreSQL" in g.ats.missing
    assert g.quality.score is not None


def test_grade_with_a_judge_uses_only_redacted_text(single_column_pdf):
    from test_matching import _judge_reply

    doc = load_document(single_column_pdf)
    job = extract_requirements(POSTING)
    labels = {r.text: "meets" for r in job.requirements}
    fake = FakeOllama(replies=[_judge_reply(labels)] * 20)
    run = grade_cv(doc, job, llm=OllamaClient(LLMConfig(), client=fake), use_ner=False, use_presidio=False,
                   use_models=False, cfg=OFFLINE, today=TODAY)
    prompts = [q["messages"][-1]["content"] for q in fake.requests]
    assert prompts and not any("Jane" in p for p in prompts)
    judged = [m for m in run.verification.result.matches if m.method == "llm"]
    assert judged and all(m.evidence for m in judged if m.label != "none")
    # Judged requirements gain points over the rules-only run, and the grade stays explainable.
    assert run.grade.score > 37.5 and run.grade.requirements[0].method == "llm"


def test_cli_grade(single_column_pdf, tmp_path, capsys):
    posting = tmp_path / "posting.txt"
    posting.write_text(POSTING, encoding="utf-8")
    args = ["grade", str(single_column_pdf), str(posting), "--no-llm", "--today", "2026-11-09"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Grade: E  (37.5/100)   confidence: low" in out
    assert "ATS keyword coverage: 2/13 word-for-word" in out and "R4 ✔ meets" in out
    assert main(args + ["--brief"]) == 0
    assert "R4 ✔ meets" not in capsys.readouterr().out
    assert main(args + ["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["grade"]["grade"] == "E" and data["grade"]["gate_failures"] == ["R1", "R3", "R5", "R7"]
    assert data["verification"]["faithfulness"] == 1.0
    assert not any("Jane" in c["text"] for c in data["matches"]["chunks"])
