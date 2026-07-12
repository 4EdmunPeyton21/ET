from pathlib import Path
from fpdf import FPDF

from ingestion.parser import parse_document, doc_type_for


def _make_pdf(path: Path, lines: list[str]) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 12)
    for line in lines:
        pdf.multi_cell(0, 8, line)
    pdf.output(str(path))


def test_doc_type_for_pdf_and_spreadsheet():
    assert doc_type_for(Path("a.pdf")) == "pdf"
    assert doc_type_for(Path("a.csv")) == "spreadsheet"
    assert doc_type_for(Path("a.xlsx")) == "spreadsheet"


def test_parse_pdf_extracts_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Equipment P-101 preventive maintenance."])

    doc_type, pages = parse_document(pdf_path)

    assert doc_type == "pdf"
    assert len(pages) == 1
    assert "P-101" in pages[0].text
    assert pages[0].page_number == 1


def test_parse_csv_extracts_rows(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("equipment_tag,description\nP-101,Feed Pump\n")

    doc_type, pages = parse_document(csv_path)

    assert doc_type == "spreadsheet"
    assert len(pages) == 1
    assert "P-101" in pages[0].text
    assert "Feed Pump" in pages[0].text
