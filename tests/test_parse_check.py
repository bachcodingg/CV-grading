from app.ingestion import load_document
from app.ingestion.document import Document, FontInfo, Line, PageInfo
from app.ingestion.parse_check import check_document


def codes(report):
    return {f.code for f in report.findings}


def _doc(*texts, **kw) -> Document:
    doc = Document(source="x", file_type="pdf", extractor="test", pages=[PageInfo(0, char_count=500)], **kw)
    doc.lines = [Line(t, block=i) for i, t in enumerate(texts)]
    return doc.finalize()


BODY = ("Jane Doe", "jane.doe@example.com · +49 151 2345 6789",
        "EXPERIENCE", "Software Engineer at Acme GmbH, 2020 – 2023, built many useful things for customers")


def test_clean_cv_has_no_errors_or_warnings(single_column_pdf):
    report = check_document(load_document(single_column_pdf))
    assert report.ok
    assert not report.by_severity("warning"), report.format(show_text=False)
    assert "Senior Software Engineer" in report.parser_view


def test_scanned_cv_is_an_error(scanned_pdf):
    report = check_document(load_document(scanned_pdf))
    assert not report.ok
    assert "NO_TEXT" in codes(report)


def test_table_and_multicolumn(table_pdf, two_column_pdf):
    assert "TABLES" in codes(check_document(load_document(table_pdf)))
    assert "MULTI_COLUMN" in codes(check_document(load_document(two_column_pdf)))


def test_running_header_with_contact(multipage_pdf):
    assert "CONTACT_IN_HEADER" in codes(check_document(load_document(multipage_pdf)))


def test_icon_images(icon_pdf):
    assert "ICON_IMAGES" in codes(check_document(load_document(icon_pdf)))


def test_tricky_docx(tricky_docx, symbol_docx):
    found = codes(check_document(load_document(tricky_docx)))
    assert {"TABLES", "TEXT_BOXES", "CONTACT_IN_HEADER"} <= found
    assert "ICON_GLYPHS" in codes(check_document(load_document(symbol_docx)))


def test_private_use_icon_glyphs():
    report = check_document(_doc(*BODY, " jane@example.com"))
    finding = next(f for f in report.findings if f.code == "ICON_GLYPHS")
    assert finding.severity == "warning" and finding.examples


def test_garbled_text_and_type3_fonts():
    doc = _doc(*BODY, "(cid:12)(cid:15)(cid:3) garbled", fonts=[FontInfo("T3Font", kind="Type3")])
    found = codes(check_document(doc))
    assert {"GARBLED_TEXT", "ODD_FONTS"} <= found


def test_missing_email():
    report = check_document(_doc("Jane Doe", "EXPERIENCE", "x" * 200))
    assert "NO_EMAIL" in codes(report)


def test_report_formats_and_serialises(tricky_docx):
    report = check_document(load_document(tricky_docx))
    text = report.format()
    assert "What a parser extracts" in text and "[!]" in text
    assert report.to_dict()["findings"][0]["code"]
    # Errors sort before warnings before infos.
    order = [f.severity for f in report.findings]
    assert order == sorted(order, key=["error", "warning", "info"].index)
