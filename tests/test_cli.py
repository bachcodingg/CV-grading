import json

from app.cli import main


def test_analyze_text_output(single_column_pdf, capsys):
    assert main(["analyze", str(single_column_pdf), "--no-llm", "--today", "2026-11-09"]) == 0
    out = capsys.readouterr().out
    assert "No parsing problems found" in out
    assert "What a parser extracts" in out
    assert "Senior Software Engineer @ Acme GmbH  (2021-01 – present)" in out
    assert "Years of experience (overlaps merged): 8.5" in out


def test_check_json(tricky_docx, capsys):
    assert main(["check", str(tricky_docx), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {f["code"] for f in data["findings"]} >= {"TABLES", "TEXT_BOXES", "CONTACT_IN_HEADER"}


def test_structure_json(plain_docx, capsys):
    assert main(["structure", str(plain_docx), "--json", "--no-llm"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["contact"]["email"] == "jane.doe@example.com"
    assert data["experience"][0]["org"] == "Acme GmbH"


def test_privacy_json(tricky_docx, capsys):
    assert main(["privacy", str(tricky_docx), "--json", "--no-llm"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "PHOTO" in {f["code"] for f in data["bias_exposure"]["findings"]}
    assert "jane.doe@example.com" not in data["redacted_text"]
    assert "[EMAIL]" in data["redacted_text"]


def test_analyze_includes_bias_scan(single_column_pdf, capsys):
    assert main(["analyze", str(single_column_pdf), "--no-llm", "--no-text"]) == 0
    out = capsys.readouterr().out
    assert "Bias-exposure scan" in out
    assert "What the grader sees" not in out


def test_bad_input_returns_error_code(tmp_path, capsys):
    bad = tmp_path / "cv.txt"
    bad.write_text("hello")
    assert main(["check", str(bad)]) == 2
    assert "unsupported" in capsys.readouterr().err


POSTING = """Data Engineer
Requirements:
- 3+ years of experience with Python
- Knowledge of Airflow is a plus
"""


def test_jd_text_output(tmp_path, capsys):
    f = tmp_path / "jd.txt"
    f.write_text(POSTING, encoding="utf-8")
    assert main(["jd", str(f), "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert "Job posting: Data Engineer" in out
    assert "R1  [experience] 3+ years of experience with Python" in out
    assert '← "is a plus"' in out


def test_jd_json_with_priority_override(tmp_path, capsys):
    f = tmp_path / "jd.txt"
    f.write_text(POSTING, encoding="utf-8")
    assert main(["jd", str(f), "--no-llm", "--json", "--priority", "R2=must"]) == 0
    data = json.loads(capsys.readouterr().out)
    r2 = data["requirements"][1]
    assert (r2["priority"], r2["priority_source"]) == ("must", "user")
    assert [k["canonical"] for k in data["keywords"]] == ["Python", "Apache Airflow"]


def test_jd_from_stdin_and_bad_override(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(POSTING))
    assert main(["jd", "-", "--no-llm", "--priority", "R9=must"]) == 2
    assert "no requirement 'R9'" in capsys.readouterr().err


def test_match_text_and_json(single_column_pdf, tmp_path, capsys):
    from test_jd_parser import POSTING

    posting = tmp_path / "posting.txt"
    posting.write_text(POSTING, encoding="utf-8")
    args = ["match", str(single_column_pdf), str(posting), "--no-llm", "--today", "2026-11-09"]
    assert main(args + ["--priority", "R9=should"]) == 0
    out = capsys.readouterr().out
    assert "Job posting: Senior Backend Engineer (m/f/d)" in out
    assert "R4 ✔ meets" in out and "Should have:" in out
    assert main(args + ["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {m["requirement_id"] for m in data["matches"]} == {r["id"] for r in data["requirements"]}
    assert not any("Jane" in c["text"] for c in data["chunks"])  # the matcher saw the redacted copy
    assert any(c["original_text"] == "Docker, Kubernetes, Terraform, Git" for c in data["chunks"])
