from datetime import date

import pytest

from app.config import Config, LLMConfig, MatchingConfig
from app.extraction import structure_cv
from app.extraction.jd_parser import extract_requirements
from app.extraction.schemas import (
    Candidate,
    EducationEntry,
    ExperienceEntry,
    JDKeyword,
    Language,
    PartialDate,
    Requirement,
    Skill,
    Span,
)
from app.ingestion import load_document
from app.llm.client import OllamaClient
from app.matching import build_chunks, format_matches, match_requirements
from app.matching.deterministic import (
    check_education,
    check_language,
    check_years,
    degree_level,
    has_domain,
    language_level,
    wanted_fields,
)
from app.matching.judge import build_prompt
from app.matching.keywords import CVIndex, decide
from app.matching.retrieval import CrossEncoderReranker, EmbeddingRetriever, LexicalRetriever
from app.privacy import analyze_privacy
from fakes import FakeOllama
from test_jd_parser import POSTING

TODAY = date(2026, 11, 9)
MCFG = MatchingConfig()
OFFLINE = Config(llm=LLMConfig(enabled=False), matching=MCFG)


@pytest.fixture(scope="module")
def sample(single_column_pdf):
    doc = load_document(single_column_pdf)
    cand = structure_cv(doc, use_llm=False, use_ner=False, today=TODAY)
    priv = analyze_privacy(doc, cand, use_presidio=False, today=TODAY)
    return doc, priv.redacted.candidate


@pytest.fixture(scope="module")
def job():
    return extract_requirements(POSTING)


def req(text, category="skill", **kw):
    r = Requirement(id="R1", text=text, category=category, priority="must", span=Span(start=0, end=len(text)), **kw)
    return r


def kw(text, canonical=None, start=0):
    return JDKeyword(text=text, canonical=canonical, span=Span(start=start, end=start + len(text)))


# ------------------------------------------------------------------ chunks


def test_chunks_use_redacted_text_and_original_spans(sample):
    doc, cand = sample
    chunks = build_chunks(cand)
    assert [c.id for c in chunks] == list(range(1, len(chunks) + 1))
    kinds = [c.kind for c in chunks]
    assert kinds[:4] == ["summary", "experience", "bullet", "bullet"]
    assert {"education", "skills", "certification", "language"} <= set(kinds)
    bullet = next(c for c in chunks if c.text.startswith("Built Python ETL"))
    assert bullet.span.of(doc.text) == bullet.text
    assert bullet.context == "Senior Software Engineer, Acme GmbH (2021-01 – present)"
    assert "Jane" not in " ".join(c.text + c.context for c in chunks)
    tools = next(c for c in chunks if c.text.startswith("Tools:"))
    assert tools.quote(doc.text) == "Docker, Kubernetes, Terraform, Git"


def test_chunk_quote_maps_redacted_text_back_to_the_cv():
    from samples import doc_from_text

    doc = doc_from_text("""
Jane Doe
jane.doe@example.com

EXPERIENCE
Engineer at Acme GmbH, Jan 2020 – Dec 2021
- Presented results with Jane's manager to the board
""")
    cand = structure_cv(doc, use_llm=False, use_ner=False, today=TODAY)
    red = analyze_privacy(doc, cand, use_presidio=False).redacted.candidate
    bullet = next(c for c in build_chunks(red) if c.kind == "bullet")
    assert "[NAME]" in bullet.text and "Jane" not in bullet.text
    assert bullet.quote(doc.text) == "Presented results with Jane's manager to the board"


# ------------------------------------------------------------------ deterministic


@pytest.mark.parametrize("text,level", [
    ("M.Sc.", "master"), ("Master of Science", "master"), ("Dipl.-Ing.", "master"), ("MBA", "master"),
    ("B.Sc.", "bachelor"), ("Bachelor of Arts", "bachelor"), ("BEng", "bachelor"), ("PhD", "phd"),
    ("Dr. rer. nat.", "phd"), ("Abitur", "secondary"), ("Certificate", None),
])
def test_degree_levels(text, level):
    assert degree_level(text) == level


def test_wanted_fields():
    assert wanted_fields("Bachelor's or Master's degree in Computer Science, or a related field") == (
        ["Computer Science"], True)
    assert wanted_fields("Degree in Mathematics, Physics or Statistics") == (["Mathematics", "Physics", "Statistics"],
                                                                            False)
    assert wanted_fields("University degree") == ([], False)


def _edu(*entries):
    return Candidate(education=[EducationEntry(degree=d, field=f, current=cur) for d, f, cur in entries])


def test_education_check():
    r = req("Master's degree in Computer Science", "education", degree_level="master")
    ok = check_education(r, _edu(("B.Sc.", "Physics", False), ("M.Sc.", "Informatics", False)), [], MCFG)
    assert (ok.label, ok.decisive) == ("meets", True)  # Informatics is Computer Science
    low = check_education(r, _edu(("B.Sc.", "Computer Science", False)), [], MCFG)
    assert (low.label, low.cap, low.decisive) == ("tangential", "tangential", True)
    busy = check_education(r, _edu(("M.Sc.", "Computer Science", True)), [], MCFG)
    assert (busy.label, busy.decisive) == ("partial", True)
    other = check_education(r, _edu(("M.Sc.", "Chemistry", False)), [], MCFG)
    assert other.decisive is False and other.cap == "meets"
    none = check_education(r, Candidate(), [], MCFG)
    assert (none.label, none.decisive) == ("none", True)
    eq = req("Bachelor's degree in CS or equivalent practical experience", "education", degree_level="bachelor")
    school = check_education(eq, _edu(("Abitur", None, False)), [], MCFG)
    assert school.decisive is False and school.cap is None  # the judge looks at the jobs instead


def test_language_levels():
    assert language_level("C1") == 5 and language_level("B2/C1") == 4 and language_level("native") == 7
    assert language_level("business fluent") == 5 and language_level("Grundkenntnisse") == 2
    assert language_level(None) is None and language_level("sometimes") is None


def test_language_check():
    cand = Candidate(languages=[Language(name="Englisch", level="B2"), Language(name="German")])
    c1 = check_language(req("English C1", "language", languages=["English"], language_level="C1"), cand, [])
    assert (c1.label, c1.decisive) == ("partial", True)  # one step below
    de = check_language(req("German B2", "language", languages=["German"], language_level="B2"), cand, [])
    assert de.label == "partial" and "without a level" in de.reason
    fr = check_language(req("French", "language", languages=["French"]), cand, [])
    assert fr.label == "none"
    either = req("German or French", "language", languages=["German", "French"], any_of=True)
    assert check_language(either, cand, []).label == "meets"
    implicit = check_language(req("English", "language", languages=["English"]), Candidate(), [])
    assert implicit.label == "partial" and "written in English" in implicit.reason


def _jobs(*periods):
    return Candidate(experience=[
        ExperienceEntry(title=t, start=PartialDate(year=s), end=PartialDate(year=e), bullets=[])
        for t, s, e in periods])


def test_years_check_total_and_domain():
    cand = _jobs(("Engineer", 2018, 2023))
    cand.years_of_experience = 5.5
    plain = req("3+ years of professional experience", "experience", min_years=3)
    plain.field_spans["min_years"] = Span(start=0, end=8)
    c = check_years(plain, cand, build_chunks(cand), TODAY)
    assert (c.label, c.decisive) == ("meets", True)
    domain = req("7+ years of experience in backend development", "experience", min_years=7)
    c = check_years(domain, cand, build_chunks(cand), TODAY)
    assert c.cap == "partial" and not c.decisive and "5.5 of 7" in c.reason
    assert has_domain("5+ years of experience in backend development")
    assert not has_domain("at least 3 years of relevant professional experience")


def test_years_check_scoped_to_keywords():
    cand = Candidate(experience=[
        ExperienceEntry(title="Python Developer", start=PartialDate(year=2023, month=1),
                        end=PartialDate(year=2024, month=12)),
        ExperienceEntry(title="Accountant", start=PartialDate(year=2015, month=1), end=PartialDate(year=2022, month=12)),
    ], skills=[Skill(name="Rust")])
    chunks = build_chunks(cand)
    r = req("5+ years of Python", min_years=5, keywords=[kw("Python", "Python", 15)])
    c = check_years(r, cand, chunks, TODAY)
    assert c.cap == "tangential" and not c.decisive and "2 of 5" in c.reason  # 2 years, not 10
    rust = req("3+ years of Rust", min_years=3, keywords=[kw("Rust", "Rust", 15)])
    c = check_years(rust, cand, chunks, TODAY)
    assert c.cap == "partial" and not c.decisive and "no dated job" in c.reason


# ------------------------------------------------------------------ keywords


def _skill_cv():
    return Candidate(
        experience=[ExperienceEntry(title="Engineer", org="Acme",
                                    bullets=[{"text": "Moved 40 services to K8s"}])],
        skills=[Skill(name="Python", category="Languages"), Skill(name="PostgreSQL", category="Databases")],
    )


def test_alias_hit_is_found_but_not_verbatim():
    chunks = build_chunks(_skill_cv())
    hit = CVIndex.build(chunks).find(kw("Kubernetes", "Kubernetes"))
    assert hit.found and not hit.verbatim and hit.cv_forms == ["K8s"]
    assert CVIndex.build(chunks).find(kw("gRPC")).found is False


def test_keyword_decisions():
    chunks = build_chunks(_skill_cv())
    index = CVIndex.build(chunks)

    def run(text, kws, **extra):
        r = req(text, keywords=kws, **extra)
        return decide(r, [index.find(k) for k in kws], chunks)

    d = run("Experience with Kubernetes", [kw("Kubernetes", "Kubernetes")])
    assert (d.label, d.decisive, d.confidence) == ("meets", True, "high")
    d = run("Knowledge of PostgreSQL", [kw("PostgreSQL", "PostgreSQL")])
    assert (d.label, d.decisive, d.confidence) == ("meets", True, "medium")  # only listed
    d = run("Strong Python skills", [kw("Python", "Python")])
    assert d.label == "meets" and not d.decisive  # strength goes to the judge
    d = run("PostgreSQL and Redis", [kw("PostgreSQL", "PostgreSQL"), kw("Redis", "Redis")])
    assert (d.label, d.decisive) == ("partial", False)
    d = run("Redis or PostgreSQL", [kw("Redis", "Redis"), kw("PostgreSQL", "PostgreSQL")], any_of=True)
    assert (d.label, d.decisive) == ("meets", True)
    d = run("Redis", [kw("Redis", "Redis")])
    assert (d.label, d.decisive) == ("none", False) and d.reason == "Redis not found on the CV"


# ------------------------------------------------------------------ retrieval


def test_lexical_retriever_folds_aliases():
    chunks = build_chunks(_skill_cv())
    ranked = LexicalRetriever().rank("Experience running Kubernetes clusters", chunks, 3)
    assert ranked[0][0] == next(c.id for c in chunks if "K8s" in c.text)
    assert all(0 < s <= 1 for _, s in ranked)
    assert LexicalRetriever().rank("Excellent communication", chunks, 3) == []


class FakeEmbedder:
    """Bag-of-words vectors over a tiny vocabulary."""

    VOCAB = ["kubernetes", "k8s", "python", "postgresql", "team", "services"]

    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.calls += 1
        return [[float(w in t.lower()) + (float("k8s" in t.lower()) if w == "kubernetes" else 0) for w in self.VOCAB]
                for t in texts]


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        return [3.0 if "PostgreSQL" in passage else -3.0 for _, passage in pairs]  # logits


def test_embedding_retriever_and_reranker():
    chunks = build_chunks(_skill_cv())
    model = FakeEmbedder()
    retriever = EmbeddingRetriever(model, query_prefix="q: ")
    top = retriever.rank("kubernetes", chunks, 2)
    assert "K8s" in next(c.text for c in chunks if c.id == top[0][0])
    retriever.rank("python", chunks, 2)
    assert model.calls == 3  # chunk vectors computed once, then one call per query
    reranked = CrossEncoderReranker(FakeCrossEncoder()).rerank("databases", chunks)
    best = next(c for c in chunks if c.id == reranked[0][0])
    assert "PostgreSQL" in best.text and 0.9 < reranked[0][1] < 1 and reranked[-1][1] < 0.1


# ------------------------------------------------------------------ pipeline


def test_rules_only_matching(sample, job):
    doc, cand = sample
    res = match_requirements(job, cand, use_llm=False, cfg=OFFLINE, today=TODAY)
    got = {m.requirement_id: (m.label, m.method) for m in res.matches}
    assert got["R4"] == ("meets", "deterministic")  # M.Sc. Computer Science
    assert got["R8"] == ("meets", "deterministic")  # English C2 >= C1
    assert got["R9"] == ("meets", "deterministic")  # German native
    assert got["R11"] == ("meets", "deterministic")  # AWS cert, "– Associate" suffix
    assert got["R3"] == ("none", "keyword")  # FastAPI or Django
    assert got["R5"] == ("none", "keyword")  # neither PostgreSQL nor Redis
    assert got["R2"] == ("meets", "keyword")  # "Strong" not judged: low confidence
    assert res.match("R2").confidence == "low"
    assert got["R6"] == ("partial", "keyword")  # AWS only as a certificate
    assert res.llm_calls == 0 and res.retriever == "lexical"
    # Years are enough (8.5 >= 5) but "backend development" needs the judge.
    r1 = res.match("R1")
    assert r1.method == "retrieval" and r1.confidence == "low" and r1.label in ("tangential", "partial")
    assert any("8.5 years" in f for f in r1.facts)
    for m in res.matches:
        for ev in m.evidence:
            assert res.chunk(ev.chunk_id)  # every evidence ID is a real chunk


def test_report_quotes_the_original_cv(sample, job):
    doc, cand = sample
    res = match_requirements(job, cand, use_llm=False, cfg=OFFLINE, today=TODAY)
    text = format_matches(res, job, doc.text)
    assert "R4 ✔ meets" in text and "Must have:" in text and "Nice to have:" in text
    assert '↳ "AWS Certified Solutions Architect – Associate (2022)"' in text


def _judge_reply(labels):
    """Answer by requirement text; cite excerpt 1."""

    def reply(kwargs):
        prompt = kwargs["messages"][-1]["content"]
        requirement = prompt.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
        label = labels.get(requirement, "none")
        return {"label": label, "evidence": [] if label == "none" else [1, 7], "reason": f"judged {label}"}

    return reply


def test_llm_judge_with_caps_and_evidence(sample, job):
    doc, cand = sample
    labels = {"5+ years of professional experience in backend development": "exceeds",
              "Strong Python skills": "meets",
              "Experience with cloud platforms (e.g. AWS, GCP or Azure) is required": "partial"}
    fake = FakeOllama(replies=[_judge_reply(labels)] * 20)
    client = OllamaClient(LLMConfig(), client=fake)
    res = match_requirements(job, cand, llm=client, cfg=OFFLINE, today=TODAY)
    judged = {m.requirement_id for m in res.matches if m.method == "llm"}
    assert {"R1", "R2", "R6"} <= judged
    assert "R4" not in judged and "R11" not in judged  # decided in code, no LLM call
    assert res.match("R7").method == "retrieval"  # nothing related retrieved: no call
    assert res.llm_calls == len(fake.requests)
    r1, r6 = res.match("R1"), res.match("R6")
    assert r1.label == "exceeds" and r1.cap is None
    assert r6.label == "partial" and r6.confidence == "low"  # excerpt 7 does not exist: dropped
    assert [e.chunk_id for e in r6.evidence] == [r6.candidates[0]]
    # Prompts carry the redacted text and the computed facts, never the name.
    prompts = [r["messages"][-1]["content"] for r in fake.requests]
    assert not any("Jane" in p or "jane.doe" in p for p in prompts)
    assert any("Fact: Computed from the CV dates: 8.5 years" in p for p in prompts)
    assert all(r["format"]["properties"]["label"]["enum"][0] == "none" for r in fake.requests)


def test_judge_cannot_exceed_the_cap():
    cand = _jobs(("Backend Engineer", 2023, 2024))
    cand.years_of_experience = 2.0
    job = extract_requirements("Requirements:\n- 5+ years of experience in backend development\n")
    client = OllamaClient(LLMConfig(), client=FakeOllama(
        replies=[{"label": "exceeds", "evidence": [1], "reason": "Long backend career."}]))
    m = match_requirements(job, cand, llm=client, cfg=OFFLINE, today=TODAY).matches[0]
    assert m.method == "llm" and m.cap == "tangential" and m.label == "tangential"
    assert "Capped at 'tangential'" in m.reason


def test_positive_label_without_evidence_falls_back():
    cand = _skill_cv()
    job = extract_requirements("Requirements:\n- Excellent communication skills\n- Experience leading a team\n")
    client = OllamaClient(LLMConfig(), client=FakeOllama(
        replies=[{"label": "meets", "evidence": [], "reason": "Seems fine."}] * 2))
    res = match_requirements(job, cand, llm=client, cfg=OFFLINE, today=TODAY)
    for m in res.matches:
        assert m.method != "llm" and m.confidence == "low"


def test_unreachable_ollama_is_a_warning(sample, job):
    _, cand = sample
    client = OllamaClient(LLMConfig(), client=FakeOllama(fail_with=ConnectionError("refused")))
    res = match_requirements(job, cand, llm=client, cfg=OFFLINE, today=TODAY)
    assert any("LLM judge stopped" in w for w in res.warnings)
    assert res.llm_calls == 0 and all(m.method != "llm" for m in res.matches)


def test_judge_prompt_lists_numbered_excerpts():
    chunks = build_chunks(_skill_cv())
    prompt = build_prompt(req("Experience with Kubernetes"), chunks[:2], ["Some fact."])
    assert '[1] "Engineer, Acme"' in prompt
    assert '[2] "Moved 40 services to K8s" (from: Engineer, Acme)' in prompt
    assert "Fact: Some fact." in prompt


def test_missing_matching_extra_falls_back_to_lexical(monkeypatch, sample, job):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _, cand = sample
    res = match_requirements(job, cand, use_llm=False, use_models=True, cfg=OFFLINE, today=TODAY)
    assert res.retriever == "lexical" and any("'matching' extra" in w for w in res.warnings)


@pytest.mark.slow
def test_real_embeddings_and_reranker(sample, job):
    pytest.importorskip("sentence_transformers")
    _, cand = sample
    res = match_requirements(job, cand, use_llm=False, use_models=True, cfg=OFFLINE, today=TODAY)
    assert res.retriever == "embeddings+reranker"
    r1 = res.match("R1")
    assert r1.candidates and all(e.score is not None for e in r1.evidence)
