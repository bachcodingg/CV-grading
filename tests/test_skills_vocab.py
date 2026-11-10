from app.matching.skills import canonical_skill, find_skills


def names(text):
    return [(m.text, m.canonical) for m in find_skills(text)]


def test_aliases_map_to_canonical_names():
    assert names("K8s, golang and CI / CD") == [("K8s", "Kubernetes"), ("golang", "Go"), ("CI / CD", "CI/CD")]
    assert canonical_skill("postgres") == "PostgreSQL"
    assert canonical_skill("Microsoft Azure") == "Microsoft Azure"
    assert canonical_skill("Underwater basket weaving") is None


def test_token_boundaries():
    assert names("C-level stakeholders and R&D") == []
    assert names("C/C++ and C#") == [("C", "C"), ("C++", "C++"), ("C#", "C#")]
    assert names("JavaScript, not Java") == [("JavaScript", "JavaScript"), ("Java", "Java")]
    assert names("ASP.NET and .NET Core") == [("ASP.NET", ".NET"), (".NET Core", ".NET")]


def test_common_words_only_match_as_written():
    assert names("react quickly, go the extra mile, the rest of the team, excel at") == []
    assert names("React, Go, REST APIs, Excel") == [
        ("React", "React"), ("Go", "Go"), ("REST APIs", "REST"), ("Excel", "Excel")]
    assert canonical_skill("go") is None and canonical_skill("Go") == "Go"


def test_longest_match_wins():
    assert names("Spring Boot and React Native") == [("Spring Boot", "Spring Boot"), ("React Native", "React Native")]


def test_spans_are_offsets_into_the_text():
    text = "We use Kubernetes"
    m = find_skills(text, base=100)[0]
    assert (m.start, m.end) == (107, 117)
