import re

import pytest

from app.config import NERConfig
from app.extraction.cv_extractor import extract_candidate
from app.extraction.ner import enrich_with_ner, load_gliner
from samples import doc_from_text

CV = """
EXPERIENCE
Data Engineer
Contoso
2020 – 2023
- Built streaming pipelines with Apache Kafka and Spark
- Automated deployments using Terraform

EDUCATION
Diplom-Wirtschaftsinformatik, Some Uni
2014 – 2019

SKILLS
Python, SQL, Spark
"""


class FakeGLiNER:
    """Pretends to be GLiNER: tags a fixed vocabulary by regex."""

    VOCAB = {"Apache Kafka": "software tool", "Spark": "software tool", "Terraform": "software tool",
             "Contoso": "organization", "Diplom-Wirtschaftsinformatik": "degree"}

    def __init__(self):
        self.calls = []

    def predict_entities(self, text, labels, threshold=0.5):
        self.calls.append((text, tuple(labels)))
        out = []
        for word, label in self.VOCAB.items():
            if label in labels:
                for m in re.finditer(re.escape(word), text):
                    out.append({"start": m.start(), "end": m.end(), "text": word, "label": label, "score": 0.9})
        return out


def test_enrich_adds_mentioned_skills_and_fills_gaps():
    doc = doc_from_text(CV)
    cand = extract_candidate(doc)
    job = cand.experience[0]
    job.org, job.uncertain = None, []  # pretend the rules found nothing
    cand.education[0].degree = None
    enrich_with_ner(cand, doc.text, FakeGLiNER(), NERConfig())

    mentioned = {s.name: s for s in cand.skills if s.provenance.get("name") == "ner"}
    assert set(mentioned) == {"Apache Kafka", "Terraform"}  # Spark is already listed
    assert mentioned["Apache Kafka"].span.of(doc.text) == "Apache Kafka"
    assert job.org == "Contoso" and job.provenance["org"] == "ner"
    assert job.field_spans["org"].of(doc.text) == "Contoso"
    assert cand.education[0].degree == "Diplom-Wirtschaftsinformatik"


def test_confident_rule_values_are_not_overwritten():
    doc = doc_from_text(CV.replace("Contoso", "Contoso GmbH"))
    cand = extract_candidate(doc)
    assert cand.experience[0].org == "Contoso GmbH" and not cand.experience[0].uncertain
    model = FakeGLiNER()
    enrich_with_ner(cand, doc.text, model, NERConfig())
    assert cand.experience[0].org == "Contoso GmbH"
    assert not any("organization" in labels for _, labels in model.calls)


@pytest.mark.slow
def test_real_gliner_on_cpu():
    pytest.importorskip("gliner")
    try:
        model = load_gliner(NERConfig())
    except Exception as exc:  # offline / not downloaded
        pytest.skip(f"GLiNER model unavailable: {exc}")
    doc = doc_from_text(CV)
    cand = enrich_with_ner(extract_candidate(doc), doc.text, model, NERConfig())
    names = {s.name.lower() for s in cand.skills}
    assert "terraform" in names or "apache kafka" in names


@pytest.mark.parametrize("text,ok", [
    ("LightGBM", True), ("scikit-learn", True), ("A/B testing", True), ("Apache Kafka", True),
    ("Designed A/B testing framework", False), ("Web app", False), ("pipelines", False),
    ("Building scalable data platforms quickly now", False),
])
def test_plausible_skill(text, ok):
    from app.extraction.ner import plausible_skill

    assert plausible_skill(text) is ok


def test_ner_fixes_title_org_guessed_backwards():
    doc = doc_from_text("EXPERIENCE\nContoso\nData wrangler\n2020 – 2023\n- Did things")
    cand = extract_candidate(doc)
    job = cand.experience[0]
    assert (job.title, job.org) == ("Contoso", "Data wrangler") and set(job.uncertain) == {"title", "org"}
    enrich_with_ner(cand, doc.text, FakeGLiNER(), NERConfig())
    assert (job.title, job.org) == ("Data wrangler", "Contoso")
    assert job.uncertain == ["title"]  # still a guess, the LLM may confirm it
    assert job.field_spans["title"].of(doc.text) == "Data wrangler"
