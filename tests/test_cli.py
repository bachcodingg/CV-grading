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
