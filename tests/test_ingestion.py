import pytest

from app.ingestion import detect_type, load_document


def test_load_document_dispatches_on_content(plain_docx, single_column_pdf):
    assert load_document(plain_docx).file_type == "docx"
    # Wrong extension: magic bytes win.
    assert load_document(single_column_pdf.read_bytes(), filename="cv.docx").file_type == "pdf"


def test_detect_type_rejects_unsupported():
    with pytest.raises(ValueError, match="legacy .doc"):
        detect_type(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100, "cv.doc")
    with pytest.raises(ValueError, match="unsupported"):
        detect_type(b"hello", "cv.txt")
