from app.config import LLMConfig
from app.extraction.cv_extractor import extract_candidate
from app.extraction.gap_fill import fill_gaps, ground
from app.extraction.schemas import PartialDate as D
from app.llm.client import OllamaClient
from fakes import FakeOllama
from samples import doc_from_text

CV = """
EXPERIENCE
Initech
Robotics wizard
Spring '21 → Fall '22
- Built robots

EDUCATION
Diplom-Ingenieur Maschinenbau
Some Place Of Learning
"""


def client(replies):
    return OllamaClient(LLMConfig(), client=FakeOllama(replies))


def test_ground_returns_verbatim_substring():
    block = "Initech\nRobotics Wizard\n"
    value, span = ground("robotics wizard", block, 100, 90)
    assert value == "Robotics Wizard" and (span.start, span.end) == (108, 123)
    assert ground("Senior Robotics Wizard of Initech Labs Inc", block, 0, 90) is None
    assert ground("CEO", block, 0, 90) is None


def test_fills_only_missing_or_uncertain_fields_with_grounded_values():
    doc = doc_from_text(CV)
    cand = extract_candidate(doc)
    job = cand.experience[0]
    assert job.start is None  # "Spring '21 → Fall '22" is beyond the rules
    fake = FakeOllama([
        {"title": "Robotics wizard", "org": "Initech", "start": "Spring '21", "end": "Fall '22"},
        {"degree": "Diplom-Ingenieur", "field": "Maschinenbau", "institution": "Some Place Of Learning"},
    ])
    fill_gaps(cand, doc.text, OllamaClient(LLMConfig(), client=fake))

    assert (job.title, job.org) == ("Robotics wizard", "Initech")
    assert job.provenance["org"] == "llm" and "org" not in job.uncertain
    assert job.field_spans["org"].of(doc.text) == "Initech"
    edu = cand.education[0]
    assert edu.institution == "Some Place Of Learning"
    assert edu.field_spans["institution"].of(doc.text) == "Some Place Of Learning"
    # The prompt holds one entry only, not the whole CV.
    prompt = fake.requests[0]["messages"][-1]["content"]
    assert "Initech" in prompt and "Maschinenbau" not in prompt


def test_hallucinated_values_are_rejected():
    doc = doc_from_text(CV)
    cand = extract_candidate(doc)
    job = cand.experience[0]
    before = (job.title, job.org)
    fill_gaps(cand, doc.text, client([
        {"title": "Chief Technology Officer", "org": "Google", "start": "2015", "end": "2020"},
        {"degree": "PhD", "institution": "MIT"},
    ]))
    assert (job.title, job.org) == before
    assert job.start is None
    assert cand.education[0].institution != "MIT"
    assert all(v != "llm" for v in job.provenance.values())


def test_llm_dates_are_parsed_in_code():
    doc = doc_from_text("EXPERIENCE\nInitech\nWizard\nMarch of 2019 through July of 2021\n- Did things")
    cand = extract_candidate(doc)
    assert cand.experience[0].start is None
    fill_gaps(cand, doc.text, client([{"title": "Wizard", "org": "Initech",
                                       "start": "March of 2019", "end": "July of 2021"}]))
    job = cand.experience[0]
    assert (job.start, job.end) == (D(year=2019, month=3), D(year=2021, month=7))
    assert job.provenance["dates"] == "llm"


def test_complete_entries_skip_the_llm():
    doc = doc_from_text("EXPERIENCE\nSoftware Engineer at Acme GmbH, Jan 2020 – Dec 2022\n- Built X")
    cand = extract_candidate(doc)
    fake = FakeOllama()
    fill_gaps(cand, doc.text, OllamaClient(LLMConfig(), client=fake))
    assert fake.requests == []


def test_llm_failure_becomes_a_warning():
    doc = doc_from_text(CV)
    cand = extract_candidate(doc)
    fill_gaps(cand, doc.text, client(["broken", "broken", "broken"]))
    assert any("LLM gap-filling stopped" in w for w in cand.warnings)
