from datetime import date

from app.config import Config, LLMConfig, NERConfig
from app.extraction import structure_cv
from app.ingestion import load_document
from app.llm.client import OllamaClient
from fakes import FakeOllama
from samples import doc_from_text
from test_ner import FakeGLiNER

TODAY = date(2026, 11, 9)
OFFLINE = Config(llm=LLMConfig(enabled=False), ner=NERConfig(enabled=False))


def test_rules_only_with_years_of_experience(single_column_pdf):
    cand = structure_cv(load_document(single_column_pdf), cfg=OFFLINE, today=TODAY)
    # Jan 2021 – Nov 2026 (71 months) + Jun 2018 – Dec 2020 (31 months)
    assert [e.duration_months for e in cand.experience] == [71, 31]
    assert cand.years_of_experience == 8.5
    assert cand.warnings == []


def test_overlapping_jobs_are_not_double_counted():
    doc = doc_from_text("""
EXPERIENCE
Engineer at Acme GmbH, Jan 2020 – Dec 2021
- Built X
Teaching Assistant at Some University, Jan 2021 – Dec 2021
- Taught Y
""")
    cand = structure_cv(doc, cfg=OFFLINE, today=TODAY)
    assert [e.duration_months for e in cand.experience] == [24, 12]
    assert cand.years_of_experience == 2.0


def test_unreachable_ollama_is_a_warning_not_a_crash(single_column_pdf):
    cfg = Config(llm=LLMConfig(enabled=True, host="http://127.0.0.1:9"), ner=NERConfig(enabled=False))
    cand = structure_cv(load_document(single_column_pdf), cfg=cfg, today=TODAY)
    assert any("LLM gap-filling skipped" in w for w in cand.warnings)
    assert cand.experience[0].title == "Senior Software Engineer"


def test_injected_models_run_in_order():
    doc = doc_from_text("""
EXPERIENCE
Contoso
Data wrangler
2020 – 2023
- Built streaming pipelines with Apache Kafka
""")
    fake_llm = FakeOllama([{"title": "Data wrangler", "org": "Contoso"}])
    cand = structure_cv(doc, use_ner=True, use_llm=True, ner_model=FakeGLiNER(),
                        llm=OllamaClient(LLMConfig(), client=fake_llm), cfg=OFFLINE, today=TODAY)
    job = cand.experience[0]
    # NER fixed the swapped guess first; the LLM then confirmed the uncertain title.
    assert job.org == "Contoso" and job.provenance["org"] == "ner"
    assert job.title == "Data wrangler" and job.provenance["title"] == "llm"
    assert "Apache Kafka" in cand.skill_names()
    assert cand.years_of_experience == 3.0
