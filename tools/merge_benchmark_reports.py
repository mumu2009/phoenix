"""Merge multiple memory-tier benchmark JSON reports into a single summary.

Usage:
    python tools/merge_benchmark_reports.py build/memory_tier_benchmark_v1_no_ollama_tui_<run_id>/
    python tools/merge_benchmark_reports.py build/memory_tier_benchmark_v1_no_ollama_tui_*/
    python tools/merge_benchmark_reports.py path/to/report1.json path/to/report2.json ...

Outputs:
    <out-dir>/merged_summary.json
    <out-dir>/merged_summary.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


SCENARIO_LABELS: dict[str, str] = {
    "short_dialogue": "Short",
    "long_dialogue_5_15": "Long(5-15)",
    "ultra_long_dialogue_15_plus": "Ultra(15+)",
    "cross_session": "CrossSession",
}


def collect_json_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(p.glob("memory_tier_benchmark_v1_*.json"))
            files.extend(f for f in found if "cache" not in f.name)
        elif p.suffix == ".json" and p.exists():
            files.append(p)
    return files


def load_reports(files: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict) and "providers" in data:
                data["_source_file"] = str(f)
                reports.append(data)
            else:
                print(f"[skip] {f}: not a valid benchmark report", flush=True)
        except Exception as exc:
            print(f"[skip] {f}: {exc}", flush=True)
    return reports


def _safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0}
    return {
        "avg": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": round(statistics.quantiles(values, n=20)[18], 2) if len(values) >= 20 else round(max(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "stddev": round(statistics.pstdev(values), 2),
    }


def merge(reports: list[dict[str, Any]]) -> dict[str, Any]:
    provider_scenario_data: dict[str, dict[str, dict[str, list[float]]]] = {}

    for report in reports:
        for provider, pdata in report.get("providers", {}).items():
            if pdata.get("skipped"):
                continue
            if provider not in provider_scenario_data:
                provider_scenario_data[provider] = {}
            for scenario, sdata in pdata.get("scenarios", {}).items():
                if sdata.get("skipped"):
                    continue
                if scenario not in provider_scenario_data[provider]:
                    provider_scenario_data[provider][scenario] = {
                        "successRates": [],
                        "avgLatencies": [],
                        "medianLatencies": [],
                        "p95Latencies": [],
                        "avgSimilarities": [],
                        "sampleCounts": [],
                        "requestCounts": [],
                    }
                d = provider_scenario_data[provider][scenario]
                d["successRates"].append(sdata.get("successRateSemanticGe70", 0.0))
                d["avgLatencies"].append(sdata.get("latencyMs", {}).get("avg", 0.0))
                d["medianLatencies"].append(sdata.get("latencyMs", {}).get("median", 0.0))
                d["p95Latencies"].append(sdata.get("latencyMs", {}).get("p95", 0.0))
                d["avgSimilarities"].append(sdata.get("avgSemanticSimilarity", 0.0))
                d["sampleCounts"].append(sdata.get("samples", 0))
                d["requestCounts"].append(sdata.get("requestsSent", 0))

    merged_providers: dict[str, Any] = {}
    for provider, scenarios in provider_scenario_data.items():
        merged_providers[provider] = {}
        for scenario, d in scenarios.items():
            merged_providers[provider][scenario] = {
                "rounds": len(d["successRates"]),
                "successRate": _safe_stats(d["successRates"]),
                "latencyAvgMs": _safe_stats(d["avgLatencies"]),
                "latencyMedianMs": _safe_stats(d["medianLatencies"]),
                "latencyP95Ms": _safe_stats(d["p95Latencies"]),
                "avgSimilarity": _safe_stats(d["avgSimilarities"]),
                "totalSamples": sum(d["sampleCounts"]),
                "totalRequests": sum(d["requestCounts"]),
            }

    return {
        "mergedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sourceFiles": [r.get("_source_file", "") for r in reports],
        "roundsIngested": len(reports),
        "providers": merged_providers,
    }


def _bar(value: float, width: int = 20, char: str = "#") -> str:
    filled = max(0, min(width, round(value / 100.0 * width)))
    return char * filled + "-" * (width - filled)


def build_markdown(summary: dict[str, Any]) -> str:
    providers = list(summary["providers"].keys())
    all_scenarios: list[str] = []
    for pdata in summary["providers"].values():
        for s in pdata:
            if s not in all_scenarios:
                all_scenarios.append(s)

    lines: list[str] = [
        "# Memory Tier Benchmark — Merged Summary Report",
        "",
        f"**Merged at:** {summary['mergedAt']}  ",
        f"**Rounds ingested:** {summary['roundsIngested']}  ",
        f"**Providers:** {', '.join(providers)}  ",
        "",
    ]

    x_labels = [SCENARIO_LABELS.get(s, s) for s in all_scenarios]

    # ── 1. Success rate across rounds (Mermaid) ──────────────────────────────
    lines.append("## Mean Success Rate by Scenario (%)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append('    title "Mean Success Rate (semantic >= 0.70) % across rounds"')
    lines.append(f'    x-axis {json.dumps(x_labels)}')
    lines.append('    y-axis "Success Rate (%)" 0 --> 100')
    for provider in providers:
        pdata = summary["providers"][provider]
        values = [pdata.get(s, {}).get("successRate", {}).get("avg", 0.0) for s in all_scenarios]
        lines.append(f'    bar [{", ".join(str(round(v, 1)) for v in values)}]')
    lines.append("```")
    lines.append("")

    # ── 2. Mean avg latency (Mermaid) ────────────────────────────────────────
    lines.append("## Mean Average Latency by Scenario (ms)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append('    title "Mean Avg Latency (ms) across rounds"')
    lines.append(f'    x-axis {json.dumps(x_labels)}')
    lines.append('    y-axis "Latency (ms)"')
    for provider in providers:
        pdata = summary["providers"][provider]
        values = [pdata.get(s, {}).get("latencyAvgMs", {}).get("avg", 0.0) for s in all_scenarios]
        lines.append(f'    bar [{", ".join(str(round(v, 1)) for v in values)}]')
    lines.append("```")
    lines.append("")

    # ── 3. Summary table ─────────────────────────────────────────────────────
    lines.append("## Aggregated Summary Table")
    lines.append("")
    header_scenarios = [SCENARIO_LABELS.get(s, s) for s in all_scenarios]
    col_w = max(22, max((len(h) for h in header_scenarios), default=22))
    header_parts = ["Provider".ljust(16)] + [h.ljust(col_w) for h in header_scenarios]
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("|" + "|".join(["-" * (len(p) + 2) for p in header_parts]) + "|")
    for provider in providers:
        pdata = summary["providers"][provider]
        row_parts = [provider.ljust(16)]
        for s in all_scenarios:
            sd = pdata.get(s)
            if sd:
                sr_avg = sd["successRate"]["avg"]
                sr_std = sd["successRate"]["stddev"]
                lat_avg = sd["latencyAvgMs"]["avg"]
                rounds = sd["rounds"]
                cell = f"{sr_avg:.1f}±{sr_std:.1f}% / {lat_avg:.0f}ms (n={rounds})"
            else:
                cell = "N/A"
            row_parts.append(cell.ljust(col_w))
        lines.append("| " + " | ".join(row_parts) + " |")
    lines.append("")
    lines.append("_Format: `mean_success% ± stddev% / mean_avg_latency_ms (n=rounds)`_")
    lines.append("")

    # ── 4. Detailed per-provider breakdown ───────────────────────────────────
    lines.append("## Detailed Per-Provider Breakdown")
    lines.append("")
    for provider in providers:
        pdata = summary["providers"][provider]
        lines.append(f"### `{provider}`")
        lines.append("")
        for scenario in all_scenarios:
            sd = pdata.get(scenario)
            if not sd:
                continue
            label = SCENARIO_LABELS.get(scenario, scenario)
            sr = sd["successRate"]
            lat = sd["latencyAvgMs"]
            sim = sd["avgSimilarity"]
            bar_vis = _bar(sr["avg"], 20)
            lines.append(f"#### {label}")
            lines.append("")
            lines.append("```")
            lines.append(f"Success rate : [{bar_vis}] {sr['avg']:.1f}%  (stddev={sr['stddev']:.1f}  min={sr['min']:.1f}  max={sr['max']:.1f})")
            lines.append(f"Latency avg  : avg={lat['avg']:.1f}ms  stddev={lat['stddev']:.1f}ms  min={lat['min']:.1f}ms  max={lat['max']:.1f}ms")
            lines.append(f"Similarity   : avg={sim['avg']:.4f}  stddev={sim['stddev']:.4f}")
            lines.append(f"Total samples: {sd['totalSamples']}  total_requests={sd['totalRequests']}  rounds={sd['rounds']}")
            lines.append("```")
            lines.append("")

    # ── 5. Quadrant: success vs latency ──────────────────────────────────────
    if len(providers) >= 2 and len(all_scenarios) >= 2:
        lines.append("## Success Rate vs Latency (Quadrant)")
        lines.append("")
        lines.append("```mermaid")
        lines.append("quadrantChart")
        lines.append('    title "Mean Success Rate vs Mean Latency (merged)"')
        lines.append('    x-axis "Low Success" --> "High Success"')
        lines.append('    y-axis "Fast" --> "Slow"')
        lines.append('    quadrant-1 "High-success, Slow"')
        lines.append('    quadrant-2 "High-success, Fast"')
        lines.append('    quadrant-3 "Low-success, Fast"')
        lines.append('    quadrant-4 "Low-success, Slow"')
        all_lats = [
            sd["latencyAvgMs"]["avg"]
            for pdata in summary["providers"].values()
            for sd in pdata.values()
        ]
        max_lat = max(all_lats) if all_lats else 1.0
        for provider in providers:
            pdata = summary["providers"][provider]
            for scenario, sd in pdata.items():
                sr = sd["successRate"]["avg"]
                lat = sd["latencyAvgMs"]["avg"]
                x = round(sr / 100.0, 3)
                y = round(lat / max(max_lat, 1.0), 3)
                label = f"{provider[:6]}-{SCENARIO_LABELS.get(scenario, scenario)[:6]}"
                lines.append(f"    {label}: [{x}, {y}]")
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append(f"_Source files: {len(summary['sourceFiles'])} reports merged_")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge memory-tier benchmark JSON reports into a summary")
    parser.add_argument("inputs", nargs="+", help="JSON files or directories containing benchmark reports")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: first input dir or cwd)")
    args = parser.parse_args()

    files = collect_json_files(args.inputs)
    if not files:
        print("[ERROR] no valid benchmark JSON files found", flush=True)
        return 1

    print(f"[INFO] merging {len(files)} report file(s):", flush=True)
    for f in files:
        print(f"  {f}", flush=True)

    reports = load_reports(files)
    if not reports:
        print("[ERROR] no reports could be loaded", flush=True)
        return 1

    summary = merge(reports)

    out_dir = Path(args.out_dir) if args.out_dir else Path(files[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "merged_summary.json"
    md_out = out_dir / "merged_summary.md"

    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.write_text(build_markdown(summary), encoding="utf-8")

    print(f"[OK] wrote {json_out}", flush=True)
    print(f"[OK] wrote {md_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
