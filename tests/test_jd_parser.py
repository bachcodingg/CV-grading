from app.config import LLMConfig
from app.extraction.jd_parser import (
    extract_requirements,
    format_posting,
    min_years,
    parse_job_posting,
    priority_cue,
    split_jd_sections,
)
from app.llm.client import OllamaClient
from fakes import FakeOllama

POSTING = """Senior Backend Engineer (m/f/d)
Acme Robotics GmbH · Munich · Hybrid

We're building the operating system for warehouse robots.

What you'll do:
• Design and build Python microservices for fleet management
• Own our CI/CD pipelines on GitHub Actions

What you'll bring:
• 5+ years of professional experience in backend development
• Strong Python skills and experience with FastAPI or Django
• Bachelor's or Master's degree in Computer Science, or a related field
• Solid knowledge of PostgreSQL and Redis
• Experience with cloud platforms (e.g. AWS, GCP or Azure) is required
• Excellent communication skills and fluency in English (C1); German is a plus

Nice to have:
- Experience with ROS or embedded systems
- AWS Certified Solutions Architect
- Contributions to open-source projects

What we offer:
- 30 days of vacation
- Conference budget for Rust and Scala events
"""


def req(posting, text):
    return next(r for r in posting.requirements if r.text == text)


def test_sections_and_their_kinds():
    kinds = [(s.kind, s.heading) for s in split_jd_sections(POSTING)]
    assert kinds == [
        ("header", ""),
        ("responsibilities", "What you'll do"),
        ("required", "What you'll bring"),
        ("preferred_nice", "Nice to have"),
        ("benefits", "What we offer"),
    ]


def test_title_and_responsibilities():
    p = extract_requirements(POSTING)
    assert p.title == "Senior Backend Engineer (m/f/d)"
    assert [b.text for b in p.responsibilities][0] == "Design and build Python microservices for fleet management"


def test_atomic_requirements_with_section_priorities():
    p = extract_requirements(POSTING)
    texts = [(r.id, r.text, r.category, r.priority) for r in p.requirements]
    assert texts == [
        ("R1", "5+ years of professional experience in backend development", "experience", "must"),
        ("R2", "Strong Python skills", "skill", "must"),
        ("R3", "experience with FastAPI or Django", "skill", "must"),
        ("R4", "Bachelor's or Master's degree in Computer Science, or a related field", "education", "must"),
        ("R5", "Solid knowledge of PostgreSQL and Redis", "skill", "must"),
        ("R6", "Experience with cloud platforms (e.g. AWS, GCP or Azure) is required", "skill", "must"),
        ("R7", "Excellent communication skills", "soft_skill", "must"),
        ("R8", "fluency in English (C1)", "language", "must"),
        ("R9", "German is a plus", "language", "nice"),
        ("R10", "Experience with ROS or embedded systems", "skill", "nice"),
        ("R11", "AWS Certified Solutions Architect", "certification", "nice"),
        ("R12", "Contributions to open-source projects", "other", "nice"),
    ]
    for r in p.requirements:
        assert p.text[r.span.start:r.span.end] == r.text


def test_requirement_details():
    p = extract_requirements(POSTING)
    assert req(p, "5+ years of professional experience in backend development").min_years == 5
    edu = req(p, "Bachelor's or Master's degree in Computer Science, or a related field")
    assert edu.degree_level == "bachelor"
    eng = req(p, "fluency in English (C1)")
    assert (eng.languages, eng.language_level) == (["English"], "C1")
    cloud = req(p, "Experience with cloud platforms (e.g. AWS, GCP or Azure) is required")
    assert cloud.any_of and [k.canonical for k in cloud.keywords] == ["AWS", "Google Cloud", "Microsoft Azure"]
    assert cloud.priority_source == "cue" and cloud.priority_cue == "is required"
    db = req(p, "Solid knowledge of PostgreSQL and Redis")
    assert not db.any_of and db.priority_source == "section"
    assert "category" in req(p, "Contributions to open-source projects").uncertain


def test_posting_keywords_skip_benefits_and_keep_verbatim_text():
    p = extract_requirements(POSTING)
    words = [k.text for k in p.keywords]
    assert "GCP" in words and "Kubernetes" not in words
    assert "Rust" not in words and "Scala" not in words  # benefits section
    assert words.index("microservices") < words.index("FastAPI")  # posting order
    assert all(p.text[k.span.start:k.span.end] == k.text for k in p.keywords)


def test_priority_cues():
    assert priority_cue("Docker is a plus") == ("nice", "is a plus")
    assert priority_cue("Knowledge of Go is a strong plus") == ("should", "is a strong plus")
    assert priority_cue("A degree is preferred but not required") == ("should", "preferred")
    assert priority_cue("Not required, but nice: Rust") == ("nice", "Not required")
    assert priority_cue("Python is a must") == ("must", "must")
    assert priority_cue("Build Python services") is None


def test_min_years():
    assert min_years("5+ years of Python")[0] == 5
    assert min_years("3-5 years of experience")[0] == 3
    assert min_years("at least two years of experience with SQL")[0] == 2
    assert min_years("minimum of 4 yrs in sales")[0] == 4
    assert min_years("6 months of internship experience")[0] == 0.5
    assert min_years("Experience with Python") is None


def test_prose_posting_without_headings():
    p = extract_requirements(
        "We are hiring a DevOps Engineer to join our platform team in Berlin. You have 3-5 years of experience "
        "running Kubernetes clusters in production. Experience with Terraform and Ansible is required. "
        "Knowledge of Go is a strong plus. A degree in computer science is preferred but not required. "
        "We offer a competitive salary and 28 days of holiday."
    )
    assert p.title == "DevOps Engineer"
    got = [(r.category, r.priority, r.min_years) for r in p.requirements]
    assert got == [("experience", "should", 3), ("skill", "must", None), ("skill", "should", None),
                   ("education", "should", None)]
    assert any("No requirements list found" in w for w in p.warnings)


def test_lead_ins_wrapped_bullets_and_soft_skills():
    p = extract_requirements(
        "Requirements\n"
        "To succeed in this role you will need:\n"
        "- At least two years of experience with SQL and\n"
        "  dashboarding tools such as Tableau or Power BI\n"
        "- Hands-on mentality and a structured way of working\n"
        "Ideally you also have:\n"
        "- Familiarity with A/B testing; knowledge of Airflow would be great\n"
    )
    got = [(r.text, r.category, r.priority) for r in p.requirements]
    assert got == [
        ("At least two years of experience with SQL and\n  dashboarding tools such as Tableau or Power BI",
         "experience", "must"),
        ("Hands-on mentality and a structured way of working", "soft_skill", "must"),
        ("Familiarity with A/B testing", "skill", "should"),
        ("knowledge of Airflow would be great", "skill", "nice"),
    ]
    assert p.requirements[0].any_of


def test_degree_of_autonomy_is_not_education():
    p = extract_requirements("Requirements\n- A high degree of autonomy and ownership")
    assert p.requirements[0].category == "soft_skill"


def test_user_can_override_priority():
    p = extract_requirements(POSTING)
    r = p.set_priority("r9", "must")
    assert (r.priority, r.priority_source, r.priority_cue) == ("must", "user", None)


def test_llm_classifies_uncertain_and_splits_long_requirements():
    text = ("Requirements\n"
            "- Contributions to open-source projects\n"
            "- A proven track record shipping mobile apps to the App Store together with a portfolio of "
            "published work and the willingness to travel to customer sites across the whole of Europe\n")
    long_req = ("A proven track record shipping mobile apps to the App Store together with a portfolio of "
                "published work and the willingness to travel to customer sites across the whole of Europe")
    fake = FakeOllama([
        {"category": "experience", "parts": []},
        {"category": "experience", "parts": [
            "A proven track record shipping mobile apps to the App Store together with a portfolio of published work",
            "the willingness to travel to customer sites across the whole of Europe"]},
    ])
    p = parse_job_posting(text, use_llm=True, llm=OllamaClient(LLMConfig(), client=fake))
    assert [r.id for r in p.requirements] == ["R1", "R2", "R3"]
    first = p.requirements[0]
    assert first.category == "experience" and first.provenance["category"] == "llm" and not first.uncertain
    assert p.requirements[1].text.endswith("portfolio of published work")
    assert p.requirements[2].text == "the willingness to travel to customer sites across the whole of Europe"
    assert p.requirements[2].priority == "must"  # inherits the section priority
    assert long_req not in fake.requests[0]["messages"][-1]["content"]  # one requirement per call


def test_llm_parts_that_are_not_in_the_posting_are_rejected():
    long_req = ("Experience building distributed systems at scale in a fast moving environment with many teams "
                "and stakeholders, ideally in fintech or another regulated industry with audits")
    fake = FakeOllama([{"category": "experience", "parts": ["Experience with Kafka", "PhD in physics"]}])
    p = parse_job_posting(f"Requirements\n- {long_req}\n", use_llm=True,
                          llm=OllamaClient(LLMConfig(), client=fake))
    assert [r.text for r in p.requirements] == [long_req]


def test_llm_failure_becomes_a_warning():
    fake = FakeOllama(["broken", "broken", "broken"])
    p = parse_job_posting("Requirements\n- Contributions to open-source projects\n", use_llm=True,
                          llm=OllamaClient(LLMConfig(), client=fake))
    assert len(p.requirements) == 1
    assert any("LLM refinement stopped" in w for w in p.warnings)


def test_no_llm_call_when_rules_are_sure():
    fake = FakeOllama()
    parse_job_posting("Requirements\n- 3+ years of Python\n", use_llm=True, llm=OllamaClient(LLMConfig(), client=fake))
    assert fake.requests == []


def test_format_posting_groups_by_priority():
    out = format_posting(extract_requirements(POSTING))
    assert out.index("Must have (8)") < out.index("Nice to have (4)")
    assert '← "is a plus"' in out
