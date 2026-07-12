import csv
from pathlib import Path
from fpdf import FPDF

import config


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.set_font("Helvetica", "", 12)
    pdf.ln(4)
    for para in paragraphs:
        pdf.multi_cell(0, 8, para)
        pdf.ln(2)
    pdf.output(str(path))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_sop_pump_maintenance(out_dir: Path) -> None:
    write_pdf(
        out_dir / "sop_pump_maintenance.pdf",
        "SOP-014: Preventive Maintenance for Feed Pump P-101",
        [
            "Document ID: SOP-014",
            "Equipment: P-101 (Feed Pump)",
            "Approved by: Rajesh Kumar",
            "Regulatory Reference: OISD-STD-118",
            "This procedure covers routine preventive maintenance of pump P-101, "
            "including seal inspection and lubrication checks, to be performed quarterly.",
            "Related work order: WO-2044 documents the most recent seal replacement "
            "performed on P-101.",
            "Any deviation from this procedure must be logged and reported per "
            "OISD-STD-118 section 4.2.",
        ],
    )


def generate_inspection_report_hx204(out_dir: Path) -> None:
    write_pdf(
        out_dir / "inspection_report_hx204.pdf",
        "Inspection Report: Heat Exchanger HX-204",
        [
            "Document ID: INSP-2026-031",
            "Equipment: HX-204 (Shell and Tube Heat Exchanger)",
            "Inspection Date: 2026-03-25",
            "Inspector: Anita Sharma",
            "Findings: Tube bundle fouling within acceptable limits after cleaning. "
            "This inspection follows the cleaning work performed under WO-2045.",
            "Regulatory Reference: PESO-CCE-2019",
            "Recommendation: Re-inspect in 6 months or after any process upset.",
        ],
    )


def generate_regulatory_reference_sheet(out_dir: Path) -> None:
    write_pdf(
        out_dir / "regulatory_reference_sheet.pdf",
        "Regulatory Reference Sheet",
        [
            "OISD-STD-118: Covers preventive maintenance practices for rotating "
            "equipment including pumps and compressors in the process industry.",
            "PESO-CCE-2019: Petroleum and Explosives Safety Organisation guidelines "
            "for pressure vessel and heat exchanger inspection intervals.",
            "Factory-Act-1948-Sec-87: Statutory requirement for reporting dangerous "
            "occurrences involving pressure plant, including vessels such as V-301.",
        ],
    )


def generate_work_orders_csv(out_dir: Path) -> None:
    rows = [
        {"work_order_id": "WO-2044", "equipment_tag": "P-101", "date": "2026-03-14",
         "performed_by": "Rajesh Kumar", "description": "Replaced pump mechanical seal"},
        {"work_order_id": "WO-2045", "equipment_tag": "HX-204", "date": "2026-03-20",
         "performed_by": "Anita Sharma", "description": "Cleaned tube bundle, removed fouling"},
        {"work_order_id": "WO-2046", "equipment_tag": "V-301", "date": "2026-04-02",
         "performed_by": "Mohammed Iqbal", "description": "Inspected pressure relief valve"},
    ]
    _write_csv(out_dir / "work_orders.csv", rows)


def generate_equipment_master_csv(out_dir: Path) -> None:
    rows = [
        {"equipment_tag": "P-101", "description": "Feed Pump", "location": "Unit 1", "install_date": "2018-06-01"},
        {"equipment_tag": "HX-204", "description": "Shell and Tube Heat Exchanger", "location": "Unit 2", "install_date": "2015-11-12"},
        {"equipment_tag": "V-301", "description": "Separator Vessel", "location": "Unit 3", "install_date": "2012-02-20"},
        {"equipment_tag": "C-102", "description": "Feed Gas Compressor", "location": "Unit 1", "install_date": "2019-09-05"},
    ]
    _write_csv(out_dir / "equipment_master.csv", rows)


def generate_all(out_dir: Path = config.INCOMING_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_sop_pump_maintenance(out_dir)
    generate_inspection_report_hx204(out_dir)
    generate_regulatory_reference_sheet(out_dir)
    generate_work_orders_csv(out_dir)
    generate_equipment_master_csv(out_dir)
    print(f"Generated synthetic corpus in {out_dir}")


if __name__ == "__main__":
    generate_all()
