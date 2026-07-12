from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pdfplumber


@dataclass
class ParsedPage:
    page_number: int
    text: str


def doc_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".csv", ".xlsx", ".xls"):
        return "spreadsheet"
    raise ValueError(f"Unsupported file type: {suffix}")


def parse_pdf(path: Path) -> list[ParsedPage]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(ParsedPage(page_number=i, text=text))
    return pages


def parse_spreadsheet(path: Path) -> list[ParsedPage]:
    if path.suffix.lower() == ".csv":
        sheets = {path.stem: pd.read_csv(path)}
    else:
        sheets = pd.read_excel(path, sheet_name=None)

    pages = []
    for i, (sheet_name, df) in enumerate(sheets.items(), start=1):
        lines = [f"Sheet: {sheet_name}", ", ".join(str(c) for c in df.columns)]
        for _, row in df.iterrows():
            lines.append(", ".join(str(v) for v in row.values))
        pages.append(ParsedPage(page_number=i, text="\n".join(lines)))
    return pages


def parse_document(path: Path) -> tuple[str, list[ParsedPage]]:
    doc_type = doc_type_for(path)
    if doc_type == "pdf":
        return doc_type, parse_pdf(path)
    return doc_type, parse_spreadsheet(path)
