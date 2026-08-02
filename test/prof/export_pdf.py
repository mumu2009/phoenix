#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize API and benchmark outputs into a PDF report")
    parser.add_argument("--api-report", default="", help="API regression JSON report path")
    parser.add_argument("--benchmark-json", nargs="*", default=[], help="One or more benchmark JSON report paths")
    parser.add_argument("--backend-report", default="", help="Optional backend matrix JSON report path")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--title", default="Odin Validation Summary", help="PDF title")
    return parser.parse_args()


def load_json(path_str: str) -> Dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"json report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def result_lookup(api_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in api_report.get("results", []):
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            out[item["name"]] = item
    return out


def module_rows(api_report: Dict[str, Any]) -> List[List[str]]:
    lookup = result_lookup(api_report)
    mapping = [
        ("Gateway", ["root_html", "api_chat_proxy_ollama"]),
        ("Context", ["context_lifecycle"]),
        ("NLP", ["v51_chat_text"]),
        ("Multimodal", ["vision_analyze", "speech_analyze", "speech_synthesize", "speech_ingest"]),
    ]
    rows = [["Module", "Status", "Evidence"]]
    for module_name, case_names in mapping:
        cases = [lookup.get(name, {}) for name in case_names]
        ok = all(bool(item.get("ok")) for item in cases if item)
        detail = "; ".join(str(item.get("detail", "")) for item in cases if item.get("detail"))
        status = "PASS" if ok and len(cases) == len(case_names) else "CHECK"
        rows.append([module_name, status, detail or "n/a"])
    return rows


def backend_rows(backend_report: Dict[str, Any]) -> List[List[str]]:
    rows = [["Backend", "Status", "Detail"]]
    for item in backend_report.get("results", []):
        if not isinstance(item, dict):
            continue
        rows.append([
            str(item.get("name") or item.get("mode") or "N/A"),
            str(item.get("status", "N/A")),
            str(item.get("detail", ""))[:160],
        ])
    return rows


def benchmark_rows(doc: Dict[str, Any]) -> List[List[str]]:
    meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
    routes = doc.get("routes", {}) if isinstance(doc.get("routes"), dict) else {}
    if routes:
        system_summary = routes.get("system", {}).get("summary", {}) if isinstance(routes.get("system"), dict) else {}
        ollama_summary = routes.get("ollama", {}).get("summary", {}) if isinstance(routes.get("ollama"), dict) else {}
    else:
        system_summary = doc.get("system", {}).get("summary", {}) if isinstance(doc.get("system"), dict) else {}
        ollama_summary = doc.get("ollama", {}).get("summary", {}) if isinstance(doc.get("ollama"), dict) else {}
    rows = [["Metric", "System API", "Direct Ollama"]]
    rows.append(["Model", str(meta.get("ollama_model", "")), str(meta.get("ollama_model", ""))])
    rows.append(["Success Rate", f"{float(system_summary.get('success_rate', 0.0)):.2f}%", f"{float(ollama_summary.get('success_rate', 0.0)):.2f}%"])
    rows.append(["Avg Latency (ms)", f"{float(system_summary.get('latency_avg_ms', 0.0)):.2f}", f"{float(ollama_summary.get('latency_avg_ms', 0.0)):.2f}"])
    rows.append(["P95 Latency (ms)", f"{float(system_summary.get('latency_p95_ms', 0.0)):.2f}", f"{float(ollama_summary.get('latency_p95_ms', 0.0)):.2f}"])
    if "speed_score_avg" in system_summary or "speed_score_avg" in ollama_summary:
        rows.append(["Speed Score Avg", f"{float(system_summary.get('speed_score_avg', 0.0)):.2f}", f"{float(ollama_summary.get('speed_score_avg', 0.0)):.2f}"])
    if float(system_summary.get("quality_avg", -1.0)) >= 0.0 or float(ollama_summary.get("quality_avg", -1.0)) >= 0.0:
        rows.append(["Intelligence Avg", f"{float(system_summary.get('quality_avg', -1.0)):.2f}", f"{float(ollama_summary.get('quality_avg', -1.0)):.2f}"])
    if "balanced_score" in system_summary or "balanced_score" in ollama_summary:
        rows.append(["Balanced Score", f"{float(system_summary.get('balanced_score', -1.0)):.2f}", f"{float(ollama_summary.get('balanced_score', -1.0)):.2f}"])
    return rows


def add_table(story: List[Any], rows: List[List[str]], col_widths: List[float]) -> None:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 5 * mm))


def main() -> int:
    args = parse_args()
    api_report = load_json(args.api_report) if args.api_report else {}
    backend_report = load_json(args.backend_report) if args.backend_report else {}
    benchmark_docs = [load_json(path) for path in args.benchmark_json]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "OdinTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#111827"),
        leading=28,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "OdinHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "OdinBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),
    )

    story: List[Any] = []
    story.append(Paragraph(args.title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 4 * mm))

    if api_report:
        inventory = api_report.get("inventory", {}) if isinstance(api_report.get("inventory"), dict) else {}
        story.append(Paragraph("Module Validation", heading_style))
        story.append(Paragraph(
            f"API endpoints: {inventory.get('total', 'n/a')} | byMethod={inventory.get('byMethod', {})} | byKind={inventory.get('byKind', {})}",
            body_style,
        ))
        story.append(Spacer(1, 2 * mm))
        add_table(story, module_rows(api_report), [35 * mm, 22 * mm, 120 * mm])

    if backend_report:
        story.append(Paragraph("Backend Matrix", heading_style))
        add_table(story, backend_rows(backend_report), [28 * mm, 24 * mm, 125 * mm])

    for index, doc in enumerate(benchmark_docs, start=1):
        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        story.append(Paragraph(f"Benchmark {index}", heading_style))
        story.append(Paragraph(
            f"System: {meta.get('system_url', 'n/a')}<br/>Ollama: {meta.get('ollama_url', 'n/a')}<br/>Model: {meta.get('ollama_model', 'n/a')}<br/>Quality Cases: {meta.get('quality_case_count', meta.get('prompt_count', 'n/a'))} | Questionnaire: {meta.get('questionnaire_count', 'n/a')} | Rounds: {meta.get('rounds', meta.get('repeats', 'n/a'))} | Concurrency: {meta.get('concurrency', 'n/a')}",
            body_style,
        ))
        story.append(Spacer(1, 2 * mm))
        add_table(story, benchmark_rows(doc), [42 * mm, 65 * mm, 65 * mm])

    pdf = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    pdf.build(story)
    print(f"[OK] pdf written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())