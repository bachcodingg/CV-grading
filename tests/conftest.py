import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import samples  # noqa: E402


@pytest.fixture(scope="session")
def cv_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cvs")


@pytest.fixture(scope="session")
def single_column_pdf(cv_dir) -> Path:
    return samples.build_single_column_pdf(cv_dir / "single.pdf")


@pytest.fixture(scope="session")
def two_column_pdf(cv_dir) -> Path:
    return samples.build_two_column_pdf(cv_dir / "two_col.pdf")


@pytest.fixture(scope="session")
def table_pdf(cv_dir) -> Path:
    return samples.build_table_pdf(cv_dir / "table.pdf")


@pytest.fixture(scope="session")
def scanned_pdf(cv_dir) -> Path:
    return samples.build_scanned_pdf(cv_dir / "scanned.pdf")


@pytest.fixture(scope="session")
def multipage_pdf(cv_dir) -> Path:
    return samples.build_multipage_header_pdf(cv_dir / "multipage.pdf")


@pytest.fixture(scope="session")
def icon_pdf(cv_dir) -> Path:
    return samples.build_icon_pdf(cv_dir / "icons.pdf")


@pytest.fixture(scope="session")
def plain_docx(cv_dir) -> Path:
    return samples.build_docx(cv_dir / "plain.docx")


@pytest.fixture(scope="session")
def tricky_docx(cv_dir) -> Path:
    """Header contact, table layout, text box, symbol glyph and a photo."""
    return samples.build_docx(
        cv_dir / "tricky.docx",
        contact_in_header=True,
        experience_as_table=True,
        textbox=True,
        photo=True,
    )


@pytest.fixture(scope="session")
def symbol_docx(cv_dir) -> Path:
    return samples.build_docx(cv_dir / "symbols.docx", symbols=True)
