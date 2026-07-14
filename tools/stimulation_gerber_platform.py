#!/usr/bin/env python3
"""Gerber + probe + ngspice 仿真平台。

设计目标：
1. 输入 Gerber 文件夹或 zip；
2. 根据锚点和探针坐标，把 PCB 上的物理位置映射回导体岛；
3. 结合 eext JSON 中仍然可用的 R/C/L/D 数值信息，生成 ngspice deck；
4. 调用 ngspice 执行工作点或瞬态分析；
5. 在指定探针位置输出数值。

重要边界：Gerber 只包含几何与铜层信息，不包含“这个封装上的电阻值是多少”之类的原理图语义。
因此，本平台要求同时提供 componentNetlists 或等价元件清单，以恢复器件电气模型。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from gerbonara.rs274x import GerberFile


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "build" / "stimulation_gerber"
DEFAULT_INI_PATH = WORKSPACE_ROOT / "catastrophe" / "outsides" / "ngspice_start" / "DuSpiceStart.ini"
GROUND_CANONICAL = "0"
SUPPORTED_COMPONENTS = {"R", "C", "L", "D"}
VALUE_SUFFIXES = {
    "T": 1e12,
    "G": 1e9,
    "MEG": 1e6,
    "K": 1e3,
    "M": 1e-3,
    "U": 1e-6,
    "N": 1e-9,
    "P": 1e-12,
    "F": 1e-15,
}


class GerberSimulationError(RuntimeError):
    pass


@dataclass(slots=True)
class PassiveComponent:
    designator: str
    kind: str
    value_text: str
    value: float | None
    nodes: tuple[str, str]
    source_file: str


@dataclass(slots=True)
class SourceSpec:
    name: str
    kind: str
    positive: str
    negative: str
    waveform: dict[str, Any]


@dataclass(slots=True)
class AnalysisSpec:
    kind: str
    observe: list[str]
    step: float | None = None
    stop: float | None = None


@dataclass(slots=True)
class AnchorSpec:
    net: str
    layer: str
    x_mm: float
    y_mm: float
    tolerance_mm: float


@dataclass(slots=True)
class ProbeSpec:
    name: str
    layer: str
    x_mm: float
    y_mm: float
    tolerance_mm: float
    expected_net: str | None = None


@dataclass(slots=True)
class CopperObject:
    layer: str
    index: int
    kind: str
    bbox: tuple[float, float, float, float]
    obj: Any


@dataclass(slots=True)
class GerberScenario:
    name: str
    gerber_root: Path
    layer_files: dict[str, Path]
    component_netlists: list[Path]
    aliases: dict[str, str]
    anchors: list[AnchorSpec]
    probes: list[ProbeSpec]
    sources: list[SourceSpec]
    analyses: list[AnalysisSpec]
    ngspice_exe: Path | None
    ngspice_ini: Path | None
    prepare_only: bool


@dataclass(slots=True)
class ProbeMapping:
    name: str
    layer: str
    x_mm: float
    y_mm: float
    island_id: int
    net: str
    board_net: str | None
    expected_net: str | None


@dataclass(slots=True)
class FlyingProbeLocation:
    designator: str
    pin_number: str
    layer: str
    net: str
    x_mm: float
    y_mm: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gerber + probe + ngspice PCB 仿真平台")
    parser.add_argument("--gerber-scenario", help="Gerber 仿真场景 JSON")
    parser.add_argument("--gerber-root", help="直接指定 Gerber zip 或目录，而不必先写场景 JSON")
    parser.add_argument("--scenario-name", help="直接导入 Gerber 根路径时使用的场景名称")
    parser.add_argument("--layer", action="append", help="直接导入 Gerber 根路径时的层映射，格式如 top=camera_top.gtl")
    parser.add_argument("--component-netlists", nargs="*", help="直接导入 Gerber 根路径时使用的 eext/EasyEDA JSON 网表列表")
    parser.add_argument("--anchors-file", help="直接导入 Gerber 根路径时使用的 anchors JSON 文件")
    parser.add_argument("--probes-file", help="直接导入 Gerber 根路径时使用的 probes JSON 文件")
    parser.add_argument("--sources-file", help="直接导入 Gerber 根路径时使用的 sources JSON 文件")
    parser.add_argument("--analyses-file", help="直接导入 Gerber 根路径时使用的 analyses JSON 文件")
    parser.add_argument("--aliases-file", help="直接导入 Gerber 根路径时使用的 aliases JSON 文件")
    parser.add_argument("--emit-gerber-template", help="输出一份 Gerber 场景模板 JSON")
    parser.add_argument("--audit-flyingprobe-path", help="对 Gerber 导出的 FlyingProbeTesting.json 做网表一致性审计；可传 zip 或目录")
    parser.add_argument("--audit-netlists", nargs="*", help="用于审计的 eext JSON 网表列表")
    parser.add_argument("--prepare-only", action="store_true", help="只生成映射和 SPICE deck，不执行 ngspice")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_ROOT), help="输出目录根路径")
    parser.add_argument("--ngspice-exe", help="手动指定 ngspice.exe 路径")
    args = parser.parse_args(argv)

    if args.emit_gerber_template:
        template_path = Path(args.emit_gerber_template).resolve()
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with template_path.open("w", encoding="utf-8") as handle:
            json.dump(build_template_scenario(), handle, indent=2, ensure_ascii=False)
        print(f"模板已输出: {template_path}")
        return 0

    if args.audit_flyingprobe_path:
        if not args.audit_netlists:
            raise SystemExit("FlyingProbe 审计模式需要同时提供 --audit-netlists")
        flyingprobe_path = Path(args.audit_flyingprobe_path).resolve()
        netlists = [Path(item).resolve() if Path(item).is_absolute() else resolve_path(WORKSPACE_ROOT, Path(item)) for item in args.audit_netlists]
        audit_name = sanitize_path_part(f"flyingprobe_audit_{flyingprobe_path.stem}")
        output_root = Path(args.out).resolve() / audit_name
        output_root.mkdir(parents=True, exist_ok=True)
        summary = run_flyingprobe_audit(flyingprobe_path, netlists, output_root)
        print(summary)
        return 0

    if not args.gerber_scenario and not args.gerber_root:
        raise SystemExit("Gerber 模式需要提供 --gerber-scenario 或 --gerber-root")

    if args.gerber_scenario:
        scenario_path = Path(args.gerber_scenario).resolve()
        with scenario_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        base_dir = scenario_path.parent
    else:
        raw = build_cli_scenario_payload(args)
        base_dir = Path(args.gerber_root).resolve().parent

    scenario = load_scenario(raw, base_dir, args.prepare_only, args.ngspice_exe)

    output_root = Path(args.out).resolve() / sanitize_path_part(scenario.name)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = run_scenario(scenario, output_root)
    print(summary)
    return 0


def build_cli_scenario_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.anchors_file or not args.probes_file or not args.layer:
        raise SystemExit("直接导入 Gerber 根路径时，至少需要 --layer、--anchors-file 和 --probes-file")

    gerber_root = Path(args.gerber_root).resolve()
    layers = parse_cli_layers(args.layer)
    anchors = load_json_payload(Path(args.anchors_file).resolve(), expect_list=True)
    probes = load_json_payload(Path(args.probes_file).resolve(), expect_list=True)
    aliases = load_json_payload(Path(args.aliases_file).resolve(), expect_dict=True) if args.aliases_file else {}
    sources = load_json_payload(Path(args.sources_file).resolve(), expect_list=True) if args.sources_file else []
    analyses = load_json_payload(Path(args.analyses_file).resolve(), expect_list=True) if args.analyses_file else [
        {"type": "op", "observe": [str(item.get("name", "")) for item in probes if item.get("name")]}
    ]
    component_netlists = [
        str(Path(item).resolve() if Path(item).is_absolute() else resolve_path(WORKSPACE_ROOT, Path(item)))
        for item in (args.component_netlists or [])
    ]
    return {
        "name": args.scenario_name or gerber_root.stem,
        "gerberRoot": str(gerber_root),
        "layers": layers,
        "componentNetlists": component_netlists,
        "aliases": aliases,
        "anchors": anchors,
        "probes": probes,
        "sources": sources,
        "analyses": analyses,
    }


def load_json_payload(path: Path, expect_list: bool = False, expect_dict: bool = False) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if expect_list and not isinstance(payload, list):
        raise SystemExit(f"JSON 不是数组: {path}")
    if expect_dict and not isinstance(payload, dict):
        raise SystemExit(f"JSON 不是对象: {path}")
    return payload


def parse_cli_layers(values: list[str]) -> dict[str, str]:
    layers: dict[str, str] = {}
    for raw in values:
        key, separator, value = str(raw).partition("=")
        if not separator or not key.strip() or not value.strip():
            raise SystemExit(f"--layer 参数格式错误，应为 key=file: {raw}")
        layers[key.strip()] = value.strip()
    return layers


def build_template_scenario() -> dict[str, Any]:
    return {
        "name": "board_probe_example",
        "gerberRoot": "path/to/gerber_folder_or_zip",
        "layers": {
            "top": "board_top.gtl",
            "bottom": "board_bottom.gbl"
        },
        "componentNetlists": [
            "catastrophe/eext_netlist_1.json",
            "catastrophe/eext_netlist_9.json"
        ],
        "aliases": {
            "AGND": "GND"
        },
        "anchors": [
            {
                "net": "CAM_3V3",
                "layer": "top",
                "x_mm": 10.0,
                "y_mm": 10.0,
                "tolerance_mm": 0.35
            },
            {
                "net": "CAM_RESET_N",
                "layer": "top",
                "x_mm": 20.0,
                "y_mm": 10.0,
                "tolerance_mm": 0.35
            }
        ],
        "probes": [
            {
                "name": "probe_cam_vdd",
                "layer": "top",
                "x_mm": 12.0,
                "y_mm": 10.0,
                "tolerance_mm": 0.35
            },
            {
                "name": "probe_cam_reset",
                "layer": "top",
                "x_mm": 22.0,
                "y_mm": 10.0,
                "tolerance_mm": 0.35,
                "expected_net": "CAM_RESET_N"
            }
        ],
        "sources": [
            {
                "name": "VCC",
                "kind": "voltage",
                "positive": "CAM_3V3",
                "negative": "GND",
                "waveform": {"kind": "dc", "value": 3.3}
            }
        ],
        "analyses": [
            {
                "type": "op",
                "observe": ["probe:probe_cam_vdd", "probe:probe_cam_reset", "I(VCC)"]
            },
            {
                "type": "tran",
                "step": 1e-6,
                "stop": 1e-3,
                "observe": ["probe:probe_cam_vdd", "probe:probe_cam_reset"]
            }
        ],
        "ngspice": {
            "exe": "",
            "ini": "catastrophe/outsides/ngspice_start/DuSpiceStart.ini"
        },
        "prepareOnly": False
    }


def run_flyingprobe_audit(flyingprobe_path: Path, netlists: list[Path], output_root: Path) -> str:
    expected = load_expected_pinmaps(netlists)
    actual = load_flyingprobe_pinmaps(flyingprobe_path)

    expected_components = set(expected)
    actual_components = set(actual)
    common_components = sorted(expected_components & actual_components)
    missing_components = sorted(expected_components - actual_components)
    extra_components = sorted(actual_components - expected_components)

    mismatched_pins: list[dict[str, Any]] = []
    missing_pins: list[dict[str, Any]] = []
    extra_pins: list[dict[str, Any]] = []
    ambiguous_pins: list[dict[str, Any]] = []

    for designator in common_components:
        expected_pins = expected[designator]["pins"]
        actual_pins = actual[designator]["pins"]
        for pin_number, expected_net in sorted(expected_pins.items(), key=lambda item: pin_sort_key(item[0])):
            if pin_number not in actual_pins:
                missing_pins.append(
                    {
                        "designator": designator,
                        "pin": pin_number,
                        "expected_net": expected_net,
                        "source_files": expected[designator]["source_files"],
                    }
                )
                continue
            actual_nets = sorted({normalize_net_name(net) for net in actual_pins[pin_number]["nets"]})
            if len(actual_nets) > 1:
                ambiguous_pins.append(
                    {
                        "designator": designator,
                        "pin": pin_number,
                        "expected_net": expected_net,
                        "actual_nets": actual_nets,
                        "locations": actual_pins[pin_number]["locations"],
                    }
                )
            actual_net = actual_nets[0] if actual_nets else ""
            if actual_net != expected_net:
                mismatched_pins.append(
                    {
                        "designator": designator,
                        "pin": pin_number,
                        "expected_net": expected_net,
                        "actual_net": actual_net,
                        "locations": actual_pins[pin_number]["locations"],
                        "source_files": expected[designator]["source_files"],
                    }
                )

        for pin_number, actual_info in sorted(actual_pins.items(), key=lambda item: pin_sort_key(item[0])):
            if pin_number not in expected_pins:
                extra_pins.append(
                    {
                        "designator": designator,
                        "pin": pin_number,
                        "actual_nets": actual_info["nets"],
                        "locations": actual_info["locations"],
                    }
                )

    report = {
        "flyingprobe_path": str(flyingprobe_path),
        "netlists": [str(path) for path in netlists],
        "counts": {
            "expected_components": len(expected_components),
            "actual_components": len(actual_components),
            "common_components": len(common_components),
            "missing_components": len(missing_components),
            "extra_components": len(extra_components),
            "mismatched_pins": len(mismatched_pins),
            "missing_pins": len(missing_pins),
            "extra_pins": len(extra_pins),
            "ambiguous_pins": len(ambiguous_pins),
        },
        "missing_components": missing_components,
        "extra_components": extra_components,
        "mismatched_pins": mismatched_pins,
        "missing_pins": missing_pins,
        "extra_pins": extra_pins,
        "ambiguous_pins": ambiguous_pins,
    }
    with (output_root / "flyingprobe_audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    connector_mismatches = [item for item in mismatched_pins if item["designator"].startswith("J")]
    connector_missing_components = [item for item in missing_components if item.startswith("J")]
    connector_extra_components = [item for item in extra_components if item.startswith("J")]

    markdown = [
        f"# FlyingProbe 审计报告：{flyingprobe_path.name}",
        "",
        "## 总览",
        "",
        f"- 参与比对的 eext 网表数：{len(netlists)}",
        f"- 期望器件数：{len(expected_components)}",
        f"- 实际 FlyingProbe 器件数：{len(actual_components)}",
        f"- 同名器件数：{len(common_components)}",
        f"- 缺失器件数：{len(missing_components)}",
        f"- 额外器件数：{len(extra_components)}",
        f"- 逐脚网络不一致：{len(mismatched_pins)}",
        f"- 实际缺失引脚：{len(missing_pins)}",
        f"- 实际额外引脚：{len(extra_pins)}",
        f"- 单脚多网络歧义：{len(ambiguous_pins)}",
        "",
    ]
    if connector_missing_components:
        markdown.extend([
            "## 连接器缺失",
            "",
            ", ".join(connector_missing_components[:80]),
            "",
        ])
    if connector_extra_components:
        markdown.extend([
            "## 连接器额外出现",
            "",
            ", ".join(connector_extra_components[:80]),
            "",
        ])
    if connector_mismatches:
        markdown.extend([
            "## 连接器关键错网",
            "",
            "| Designator | Pin | Expected | Actual | X(mil) | Y(mil) |",
            "|---|---:|---|---|---:|---:|",
        ])
        for item in connector_mismatches[:120]:
            location = item["locations"][0] if item["locations"] else {"x": "", "y": ""}
            markdown.append(
                f"| {item['designator']} | {item['pin']} | {item['expected_net']} | {item['actual_net']} | {location['x']} | {location['y']} |"
            )
        markdown.append("")
    if ambiguous_pins:
        markdown.extend([
            "## 单脚多网络歧义",
            "",
            "| Designator | Pin | Expected | Actual Nets |",
            "|---|---:|---|---|",
        ])
        for item in ambiguous_pins[:80]:
            markdown.append(
                f"| {item['designator']} | {item['pin']} | {item['expected_net']} | {', '.join(item['actual_nets'])} |"
            )
        markdown.append("")

    summary_path = output_root / "summary.md"
    summary_path.write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")
    return f"FlyingProbe 审计已完成，输出目录: {output_root}"


def load_expected_pinmaps(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for entry in raw.values():
            props = entry.get("props", {})
            designator = str(props.get("Designator", "")).strip().upper()
            if not designator:
                continue
            pins = {str(pin): normalize_net_name(net) for pin, net in entry.get("pins", {}).items()}
            if designator not in result:
                result[designator] = {"pins": {}, "source_files": []}
            if str(path) not in result[designator]["source_files"]:
                result[designator]["source_files"].append(str(path))
            for pin_number, net_name in pins.items():
                if pin_number in result[designator]["pins"] and result[designator]["pins"][pin_number] != net_name:
                    raise GerberSimulationError(
                        f"网表之间存在冲突的同名器件/引脚定义: {designator}.{pin_number} -> {result[designator]['pins'][pin_number]} / {net_name}"
                    )
                result[designator]["pins"][pin_number] = net_name
    return result


def load_flyingprobe_pinmaps(path: Path) -> dict[str, dict[str, Any]]:
    raw = read_flyingprobe_json(path)
    pins = raw.get("pins")
    if not isinstance(pins, dict):
        raise GerberSimulationError("FlyingProbeTesting.json 缺少 pins 段")
    fields = list(pins.get("fields", []))
    rows = list(pins.get("rows", []))
    required_fields = ["PIN_NAME", "NET_NAME", "PIN_X", "PIN_Y", "LAYER"]
    missing = [field for field in required_fields if field not in fields]
    if missing:
        raise GerberSimulationError(f"FlyingProbeTesting.json pins 字段缺失: {missing}")

    index_map = {field: fields.index(field) for field in required_fields}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pin_name = str(row[index_map["PIN_NAME"]]).strip().upper()
        match = re.fullmatch(r"([A-Z]+\d+)_([^\s]+)", pin_name)
        if not match:
            continue
        designator = match.group(1)
        pin_number = match.group(2)
        location = {
            "x": row[index_map["PIN_X"]],
            "y": row[index_map["PIN_Y"]],
            "layer": row[index_map["LAYER"]],
        }
        component = result.setdefault(designator, {"pins": {}})
        pin_entry = component["pins"].setdefault(pin_number, {"nets": [], "locations": []})
        net_name = str(row[index_map["NET_NAME"]]).strip()
        if net_name not in pin_entry["nets"]:
            pin_entry["nets"].append(net_name)
        if location not in pin_entry["locations"]:
            pin_entry["locations"].append(location)
    return result


def try_load_flyingprobe_locations(path: Path, aliases: dict[str, str]) -> list[FlyingProbeLocation]:
    try:
        raw = read_flyingprobe_json(path)
    except GerberSimulationError:
        return []
    pins = raw.get("pins")
    if not isinstance(pins, dict):
        return []
    fields = list(pins.get("fields", []))
    rows = list(pins.get("rows", []))
    required_fields = ["PIN_NAME", "NET_NAME", "PIN_X", "PIN_Y", "LAYER"]
    if any(field not in fields for field in required_fields):
        return []

    index_map = {field: fields.index(field) for field in required_fields}
    result: list[FlyingProbeLocation] = []
    for row in rows:
        pin_name = str(row[index_map["PIN_NAME"]]).strip().upper()
        match = re.fullmatch(r"([A-Z]+\d+)_([^\s]+)", pin_name)
        if not match:
            continue
        result.append(
            FlyingProbeLocation(
                designator=match.group(1),
                pin_number=match.group(2),
                layer=normalize_flyingprobe_layer(row[index_map["LAYER"]]),
                net=canonical_net(str(row[index_map["NET_NAME"]]).strip(), aliases),
                x_mm=coordinate_to_mm(row[index_map["PIN_X"]]),
                y_mm=coordinate_to_mm(row[index_map["PIN_Y"]]),
            )
        )
    return result


def normalize_flyingprobe_layer(value: Any) -> str:
    text = str(value).strip().upper()
    if text.startswith("T"):
        return "top"
    if text.startswith("B"):
        return "bottom"
    return text.lower()


def coordinate_to_mm(value: Any) -> float:
    numeric = float(value)
    if abs(numeric) > 500:
        return numeric * 0.0254
    return numeric


def locate_flyingprobe_point(spec_layer: str, x_mm: float, y_mm: float, tolerance_mm: float, locations: list[FlyingProbeLocation]) -> FlyingProbeLocation:
    candidates: list[tuple[float, FlyingProbeLocation]] = []
    for location in locations:
        if location.layer != spec_layer:
            continue
        distance = math.hypot(location.x_mm - x_mm, location.y_mm - y_mm)
        if distance <= tolerance_mm:
            candidates.append((distance, location))
    if not candidates:
        raise GerberSimulationError(f"在 FlyingProbe 数据中无法把点 ({x_mm}, {y_mm}) 映射到层 {spec_layer} 的引脚")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def resolve_anchor_nets_from_flyingprobe(
    anchors: list[AnchorSpec],
    locations: list[FlyingProbeLocation],
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, str], int]]:
    result: dict[tuple[str, int], str] = {}
    island_lookup: dict[tuple[str, str], int] = {}
    next_island = 1
    for anchor in anchors:
        hit = locate_flyingprobe_point(anchor.layer, anchor.x_mm, anchor.y_mm, anchor.tolerance_mm, locations)
        island_key = (anchor.layer, hit.net)
        if island_key not in island_lookup:
            island_lookup[island_key] = next_island
            next_island += 1
        key = (anchor.layer, island_lookup[island_key])
        if key in result and result[key] != anchor.net:
            raise GerberSimulationError(f"同一 FlyingProbe 网络被锚定到多个网络: {result[key]} 与 {anchor.net}")
        result[key] = anchor.net
    return result, island_lookup


def resolve_probe_nets_from_flyingprobe(
    probes: list[ProbeSpec],
    locations: list[FlyingProbeLocation],
    anchor_map: dict[tuple[str, int], str],
    island_lookup: dict[tuple[str, str], int],
) -> dict[str, ProbeMapping]:
    result: dict[str, ProbeMapping] = {}
    for probe in probes:
        hit = locate_flyingprobe_point(probe.layer, probe.x_mm, probe.y_mm, probe.tolerance_mm, locations)
        island_key = (probe.layer, hit.net)
        if island_key not in island_lookup:
            raise GerberSimulationError(f"探针 {probe.name} 命中的 FlyingProbe 网络没有对应 anchor")
        island_id = island_lookup[island_key]
        key = (probe.layer, island_id)
        if key not in anchor_map:
            raise GerberSimulationError(f"探针 {probe.name} 命中的 FlyingProbe 网络没有 anchor 对应的网络名")
        net = anchor_map[key]
        if probe.expected_net and net != probe.expected_net:
            raise GerberSimulationError(f"探针 {probe.name} 预期网络 {probe.expected_net}，实际映射为 {net}")
        result[probe.name] = ProbeMapping(
            name=probe.name,
            layer=probe.layer,
            x_mm=probe.x_mm,
            y_mm=probe.y_mm,
            island_id=island_id,
            net=net,
            board_net=hit.net,
            expected_net=probe.expected_net,
        )
    return result


def read_flyingprobe_json(path: Path) -> dict[str, Any]:
    if path.is_dir():
        file_path = path / "FlyingProbeTesting.json"
        if not file_path.exists():
            raise GerberSimulationError(f"目录中未找到 FlyingProbeTesting.json: {path}")
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            try:
                with archive.open("FlyingProbeTesting.json", "r") as handle:
                    return json.loads(handle.read().decode("utf-8"))
            except KeyError as exc:
                raise GerberSimulationError(f"Gerber zip 中未找到 FlyingProbeTesting.json: {path}") from exc
    raise GerberSimulationError(f"FlyingProbe 审计路径必须是 zip 或目录: {path}")


def pin_sort_key(text: str) -> tuple[int, str]:
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def load_scenario(raw: dict[str, Any], base_dir: Path, cli_prepare_only: bool, cli_ngspice_exe: str | None) -> GerberScenario:
    aliases = {normalize_net_name(key): normalize_net_name(value) for key, value in raw.get("aliases", {}).items()}
    aliases.setdefault("GND", GROUND_CANONICAL)
    aliases.setdefault("0", GROUND_CANONICAL)
    aliases.setdefault(GROUND_CANONICAL, GROUND_CANONICAL)

    gerber_root = resolve_path(base_dir, Path(raw["gerberRoot"]))
    extracted_root, cleanup_dir = materialize_gerber_root(gerber_root)
    layer_files = resolve_layer_files(extracted_root, raw.get("layers", {}))

    component_netlists = [resolve_path(base_dir, Path(item)) for item in raw.get("componentNetlists", [])]
    anchors = [
        AnchorSpec(
            net=canonical_net(str(item["net"]), aliases),
            layer=str(item["layer"]),
            x_mm=float(item["x_mm"]),
            y_mm=float(item["y_mm"]),
            tolerance_mm=float(item.get("tolerance_mm", 0.25)),
        )
        for item in raw.get("anchors", [])
    ]
    probes = [
        ProbeSpec(
            name=str(item["name"]),
            layer=str(item["layer"]),
            x_mm=float(item["x_mm"]),
            y_mm=float(item["y_mm"]),
            tolerance_mm=float(item.get("tolerance_mm", 0.25)),
            expected_net=canonical_net(str(item["expected_net"]), aliases) if item.get("expected_net") else None,
        )
        for item in raw.get("probes", [])
    ]
    if not anchors:
        raise GerberSimulationError("Gerber 场景至少需要一个 anchor")
    if not probes:
        raise GerberSimulationError("Gerber 场景至少需要一个 probe")

    sources = [
        SourceSpec(
            name=str(item["name"]),
            kind=str(item["kind"]).strip().lower(),
            positive=canonical_net(str(item["positive"]), aliases),
            negative=canonical_net(str(item.get("negative", "GND")), aliases),
            waveform=dict(item.get("waveform", {"kind": "dc", "value": 0.0})),
        )
        for item in raw.get("sources", [])
    ]
    analyses = [
        AnalysisSpec(
            kind=str(item["type"]).strip().lower(),
            observe=[canonical_observation_token(token, aliases) for token in item.get("observe", [])],
            step=float(item["step"]) if "step" in item else None,
            stop=float(item["stop"]) if "stop" in item else None,
        )
        for item in raw.get("analyses", [])
    ]
    if not analyses:
        raise GerberSimulationError("Gerber 场景至少需要一个 analysis")

    ngspice_block = raw.get("ngspice", {})
    ngspice_exe = resolve_optional_path(base_dir, cli_ngspice_exe or ngspice_block.get("exe"))
    ngspice_ini = resolve_optional_path(base_dir, ngspice_block.get("ini")) or (DEFAULT_INI_PATH if DEFAULT_INI_PATH.exists() else None)

    scenario = GerberScenario(
        name=str(raw.get("name", "gerber_probe_scenario")),
        gerber_root=extracted_root,
        layer_files=layer_files,
        component_netlists=component_netlists,
        aliases=aliases,
        anchors=anchors,
        probes=probes,
        sources=sources,
        analyses=analyses,
        ngspice_exe=ngspice_exe,
        ngspice_ini=ngspice_ini,
        prepare_only=bool(raw.get("prepareOnly", False) or cli_prepare_only),
    )
    if cleanup_dir is not None:
        scenario.layer_files["__cleanup__"] = cleanup_dir
    return scenario


def run_scenario(scenario: GerberScenario, output_root: Path) -> str:
    mapping_mode = "geometry"
    copper = load_copper_layers(scenario.layer_files)
    flyingprobe_locations = try_load_flyingprobe_locations(scenario.gerber_root, scenario.aliases)
    if flyingprobe_locations:
        mapping_mode = "flyingprobe"
        anchor_map, island_lookup = resolve_anchor_nets_from_flyingprobe(scenario.anchors, flyingprobe_locations)
        probe_map = resolve_probe_nets_from_flyingprobe(scenario.probes, flyingprobe_locations, anchor_map, island_lookup)
    else:
        islands = build_layer_islands(copper)
        anchor_map = resolve_anchor_nets(scenario.anchors, copper, islands)
        probe_map = resolve_probe_nets(scenario.probes, copper, islands, anchor_map)
    components, unsupported = load_component_netlists(scenario.component_netlists, scenario.aliases)

    serialized_probe_map = [
        {
            "name": probe.name,
            "layer": probe.layer,
            "x_mm": probe.x_mm,
            "y_mm": probe.y_mm,
            "island_id": probe.island_id,
            "net": probe.net,
            "board_net": probe.board_net,
            "expected_net": probe.expected_net,
        }
        for probe in probe_map.values()
    ]
    with (output_root / "probe_map.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized_probe_map, handle, indent=2, ensure_ascii=False)

    with (output_root / "unsupported_components.json").open("w", encoding="utf-8") as handle:
        json.dump(unsupported, handle, indent=2, ensure_ascii=False)

    ngspice_exe = discover_ngspice_executable(scenario.ngspice_exe, scenario.ngspice_ini)
    executed = False
    markdown = [
        f"# Gerber 探针仿真报告：{scenario.name}",
        "",
        "## 输入摘要",
        "",
        f"- Gerber 根路径：{scenario.gerber_root}",
        f"- 层数：{len([key for key in scenario.layer_files if key != '__cleanup__'])}",
        f"- componentNetlists：{len(scenario.component_netlists)}",
        f"- 可生成 SPICE 的器件数：{len(components)}",
        f"- 忽略器件数：{len(unsupported)}",
        f"- anchors：{len(scenario.anchors)}",
        f"- probes：{len(scenario.probes)}",
        f"- 点位映射来源：{mapping_mode}",
        "",
        "## 探针映射",
        "",
        "| Probe | Layer | X(mm) | Y(mm) | Island | Sim Net | Board Net |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for probe in probe_map.values():
        markdown.append(f"| {probe.name} | {probe.layer} | {probe.x_mm:.4f} | {probe.y_mm:.4f} | {probe.island_id} | {probe.net} | {probe.board_net or probe.net} |")
    markdown.append("")

    if unsupported:
        markdown.extend([
            "## 已忽略器件",
            "",
            ", ".join(unsupported[:80]),
            "",
        ])

    if scenario.prepare_only:
        markdown.extend([
            "## 执行状态",
            "",
            "- 本次以 prepare-only 运行，仅生成 probe 映射和 SPICE deck，未执行 ngspice。",
            "",
        ])
    elif ngspice_exe is None:
        markdown.extend([
            "## 执行状态",
            "",
            "- 未找到可用 ngspice.exe；已生成 deck，但尚未实际执行。",
            "",
        ])
    else:
        executed = True

    analysis_results: list[dict[str, Any]] = []
    for index, analysis in enumerate(scenario.analyses):
        deck_path = output_root / f"analysis_{index}_{sanitize_path_part(analysis.kind)}.cir"
        data_path = output_root / f"analysis_{index}_{sanitize_path_part(analysis.kind)}.data"
        log_path = output_root / f"analysis_{index}_{sanitize_path_part(analysis.kind)}.log"
        deck_text, headers = build_spice_deck(components, scenario.sources, analysis, probe_map, data_path)
        deck_path.write_text(deck_text, encoding="utf-8")

        if scenario.prepare_only or ngspice_exe is None:
            analysis_results.append({"analysis": analysis.kind, "headers": headers, "executed": False, "data": None})
            continue

        run_ngspice(ngspice_exe, deck_path, log_path)
        rows = parse_wrdata(data_path)
        summary = summarize_rows(rows)
        analysis_results.append({"analysis": analysis.kind, "headers": headers, "executed": True, "data": rows, "summary": summary, "deck": str(deck_path), "log": str(log_path)})

        markdown.extend([
            f"## {analysis.kind.upper()} 结果",
            "",
            f"- deck: {deck_path}",
            f"- data: {data_path}",
            f"- log: {log_path}",
            "",
            "| 观察点 | 最小值 | 最大值 | 结束值 |",
            "|---|---:|---:|---:|",
        ])
        for key, value in summary.items():
            markdown.append(f"| {key} | {value['min']:.9g} | {value['max']:.9g} | {value['final']:.9g} |")
        markdown.append("")

    summary_path = output_root / "summary.md"
    summary_path.write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")
    with (output_root / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scenario": {
                    "name": scenario.name,
                    "gerber_root": str(scenario.gerber_root),
                    "layer_files": {key: str(value) for key, value in scenario.layer_files.items() if key != "__cleanup__"},
                    "component_netlists": [str(path) for path in scenario.component_netlists],
                },
                "probe_map": serialized_probe_map,
                "unsupported_components": unsupported,
                "analysis_results": analysis_results,
                "executed": executed and not scenario.prepare_only,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    cleanup_dir = scenario.layer_files.get("__cleanup__")
    if cleanup_dir is not None and Path(cleanup_dir).exists():
        shutil.rmtree(Path(cleanup_dir), ignore_errors=True)

    state = "prepare-only" if scenario.prepare_only else ("executed" if ngspice_exe is not None else "deck-only")
    return f"Gerber 仿真平台已完成 {state} 流程，输出目录: {output_root}"


def materialize_gerber_root(path: Path) -> tuple[Path, Path | None]:
    if path.is_dir():
        return path, None
    if path.suffix.lower() == ".zip":
        temp_root = Path(tempfile.mkdtemp(prefix="stimulation_gerber_"))
        with zipfile.ZipFile(path, "r") as archive:
            archive.extractall(temp_root)
        children = [child for child in temp_root.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            return children[0], temp_root
        return temp_root, temp_root
    raise GerberSimulationError(f"Gerber 根路径既不是目录也不是 zip: {path}")


def resolve_layer_files(root: Path, configured: dict[str, Any]) -> dict[str, Path]:
    if configured:
        resolved = {str(layer): resolve_path(root, Path(str(filename))) for layer, filename in configured.items()}
        for layer, file_path in resolved.items():
            if not file_path.exists():
                raise GerberSimulationError(f"层文件不存在: {layer} -> {file_path}")
        return resolved

    detected: dict[str, Path] = {}
    candidates = list(root.rglob("*"))
    patterns = {
        "top": [".gtl", "f.cu", "top", "front"],
        "bottom": [".gbl", "b.cu", "bottom", "back"],
    }
    for layer, tokens in patterns.items():
        for item in candidates:
            if not item.is_file():
                continue
            lower = item.name.lower()
            if any(token in lower for token in tokens):
                detected[layer] = item
                break
    if not detected:
        raise GerberSimulationError("未能自动识别铜层文件，请在场景里显式提供 layers")
    return detected


def load_copper_layers(layer_files: dict[str, Path]) -> dict[str, list[CopperObject]]:
    result: dict[str, list[CopperObject]] = {}
    for layer, path in layer_files.items():
        if layer == "__cleanup__":
            continue
        gerber = GerberFile.open(path)
        objects: list[CopperObject] = []
        for index, obj in enumerate(getattr(gerber, "objects", [])):
            if not getattr(obj, "polarity_dark", True):
                continue
            bbox = normalize_bbox(obj.bounding_box())
            if bbox is None:
                continue
            objects.append(CopperObject(layer=layer, index=index, kind=type(obj).__name__, bbox=bbox, obj=obj))
        result[layer] = objects
    return result


def build_layer_islands(copper: dict[str, list[CopperObject]], tolerance_mm: float = 0.02) -> dict[str, dict[int, int]]:
    mapping: dict[str, dict[int, int]] = {}
    for layer, objects in copper.items():
        parent = list(range(len(objects)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        order = sorted(range(len(objects)), key=lambda idx: objects[idx].bbox[0])
        active: list[int] = []
        for current in order:
            current_bbox = objects[current].bbox
            active = [other for other in active if objects[other].bbox[2] + tolerance_mm >= current_bbox[0]]
            for other in active:
                if bboxes_touch(current_bbox, objects[other].bbox, tolerance_mm):
                    union(current, other)
            active.append(current)

        layer_map: dict[int, int] = {}
        root_to_island: dict[int, int] = {}
        next_island = 1
        for index in range(len(objects)):
            root = find(index)
            if root not in root_to_island:
                root_to_island[root] = next_island
                next_island += 1
            layer_map[index] = root_to_island[root]
        mapping[layer] = layer_map
    return mapping


def resolve_anchor_nets(
    anchors: list[AnchorSpec],
    copper: dict[str, list[CopperObject]],
    islands: dict[str, dict[int, int]],
) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    for anchor in anchors:
        hit = locate_point(anchor.layer, anchor.x_mm, anchor.y_mm, anchor.tolerance_mm, copper)
        island_id = islands[anchor.layer][hit.index]
        key = (anchor.layer, island_id)
        if key in result and result[key] != anchor.net:
            raise GerberSimulationError(f"同一铜岛被锚定到多个网络: {result[key]} 与 {anchor.net}")
        result[key] = anchor.net
    return result


def resolve_probe_nets(
    probes: list[ProbeSpec],
    copper: dict[str, list[CopperObject]],
    islands: dict[str, dict[int, int]],
    anchor_map: dict[tuple[str, int], str],
) -> dict[str, ProbeMapping]:
    result: dict[str, ProbeMapping] = {}
    for probe in probes:
        hit = locate_point(probe.layer, probe.x_mm, probe.y_mm, probe.tolerance_mm, copper)
        island_id = islands[probe.layer][hit.index]
        key = (probe.layer, island_id)
        if key not in anchor_map:
            raise GerberSimulationError(f"探针 {probe.name} 命中的铜岛没有 anchor 对应的网络名")
        net = anchor_map[key]
        if probe.expected_net and net != probe.expected_net:
            raise GerberSimulationError(f"探针 {probe.name} 预期网络 {probe.expected_net}，实际映射为 {net}")
        result[probe.name] = ProbeMapping(
            name=probe.name,
            layer=probe.layer,
            x_mm=probe.x_mm,
            y_mm=probe.y_mm,
            island_id=island_id,
            net=net,
            board_net=net,
            expected_net=probe.expected_net,
        )
    return result


def locate_point(layer: str, x_mm: float, y_mm: float, tolerance_mm: float, copper: dict[str, list[CopperObject]]) -> CopperObject:
    if layer not in copper:
        raise GerberSimulationError(f"场景引用了未知层: {layer}")
    candidates: list[tuple[float, CopperObject]] = []
    for obj in copper[layer]:
        if not bbox_contains_point(obj.bbox, x_mm, y_mm, tolerance_mm):
            continue
        distance = point_to_object_distance(x_mm, y_mm, obj)
        if distance <= tolerance_mm:
            candidates.append((distance, obj))
    if not candidates:
        raise GerberSimulationError(f"在层 {layer} 上无法把点 ({x_mm}, {y_mm}) 映射到任何铜对象")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def point_to_object_distance(x_mm: float, y_mm: float, obj: CopperObject) -> float:
    graphic = obj.obj
    if obj.kind == "Line":
        width = aperture_diameter(graphic) * 0.5
        distance = point_to_segment_distance(x_mm, y_mm, float(graphic.x1), float(graphic.y1), float(graphic.x2), float(graphic.y2))
        return max(0.0, distance - width)
    if obj.kind == "Flash":
        radius = aperture_diameter(graphic) * 0.5
        center_distance = math.hypot(x_mm - float(graphic.x), y_mm - float(graphic.y))
        return max(0.0, center_distance - radius)
    return point_to_bbox_distance(x_mm, y_mm, obj.bbox)


def aperture_diameter(graphic: Any) -> float:
    aperture = getattr(graphic, "aperture", None)
    if aperture is None:
        return 0.0
    for attr in ("diameter", "width", "x_size", "x_dia"):
        if hasattr(aperture, attr):
            try:
                return float(getattr(aperture, attr))
            except (TypeError, ValueError):
                continue
    return 0.0


def bboxes_touch(first: tuple[float, float, float, float], second: tuple[float, float, float, float], tol: float) -> bool:
    return not (
        first[2] < second[0] - tol
        or second[2] < first[0] - tol
        or first[3] < second[1] - tol
        or second[3] < first[1] - tol
    )


def bbox_contains_point(bbox: tuple[float, float, float, float], x_mm: float, y_mm: float, tol: float) -> bool:
    return bbox[0] - tol <= x_mm <= bbox[2] + tol and bbox[1] - tol <= y_mm <= bbox[3] + tol


def point_to_bbox_distance(x_mm: float, y_mm: float, bbox: tuple[float, float, float, float]) -> float:
    dx = max(bbox[0] - x_mm, 0.0, x_mm - bbox[2])
    dy = max(bbox[1] - y_mm, 0.0, y_mm - bbox[3])
    return math.hypot(dx, dy)


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def normalize_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    try:
        (x1, y1), (x2, y2) = bbox
        min_x = min(float(x1), float(x2))
        min_y = min(float(y1), float(y2))
        max_x = max(float(x1), float(x2))
        max_y = max(float(y1), float(y2))
        return (min_x, min_y, max_x, max_y)
    except Exception as exc:  # pragma: no cover - defensive
        raise GerberSimulationError(f"无法规范化 Gerber 边界框: {bbox!r}") from exc


def discover_ngspice_executable(cli_path: Path | None, ini_path: Path | None) -> Path | None:
    if cli_path and cli_path.exists():
        return cli_path

    for candidate in read_ngspice_candidates_from_ini(ini_path):
        if candidate.exists():
            return candidate

    which = shutil.which("ngspice")
    return Path(which) if which else None


def read_ngspice_candidates_from_ini(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    candidates: list[Path] = []
    current_section = ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith(";"):
            continue
        if text.startswith("[") and text.endswith("]"):
            current_section = text[1:-1].strip().lower()
            continue
        if current_section in {"ngspiceexec", "duspiceexec"} and text.lower().startswith("file="):
            candidate = Path(text.split("=", 1)[1].strip())
            candidates.append(candidate)
    return candidates


def load_component_netlists(paths: list[Path], aliases: dict[str, str]) -> tuple[list[PassiveComponent], list[str]]:
    components: list[PassiveComponent] = []
    unsupported: list[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for component_id, entry in raw.items():
            props = entry.get("props", {})
            designator = str(props.get("Designator", component_id)).strip().upper()
            if not designator:
                continue
            kind = designator[0]
            pin_items = sorted_pin_items(entry.get("pins", {}).items())
            if kind not in SUPPORTED_COMPONENTS or len(pin_items) < 2:
                unsupported.append(designator)
                continue
            value_text = str(props.get("value", "")).strip()
            value = parse_component_value(value_text, kind)
            if kind in {"R", "C", "L"} and (value is None or value <= 0.0):
                unsupported.append(designator)
                continue
            components.append(
                PassiveComponent(
                    designator=designator,
                    kind=kind,
                    value_text=value_text,
                    value=value,
                    nodes=(canonical_net(str(pin_items[0][1]), aliases), canonical_net(str(pin_items[1][1]), aliases)),
                    source_file=str(path.relative_to(WORKSPACE_ROOT)),
                )
            )
    return components, sorted(set(unsupported))


def build_spice_deck(
    components: list[PassiveComponent],
    sources: list[SourceSpec],
    analysis: AnalysisSpec,
    probe_map: dict[str, ProbeMapping],
    data_path: Path,
) -> tuple[str, list[str]]:
    lines = [
        "* auto-generated by stimulation_gerber_platform.py",
        ".option TEMP=27",
        ".model DDEFAULT D",
    ]
    for component in components:
        lines.append(component_to_spice_line(component))
    for source in sources:
        lines.append(source_to_spice_line(source))

    expressions: list[str] = []
    headers: list[str] = []
    if analysis.kind == "tran":
        expressions.append("time")
        headers.append("time_s")
    for token in analysis.observe:
        expr, header = observation_to_spice_expression(token, probe_map)
        expressions.append(expr)
        headers.append(header)

    lines.extend([
        ".control",
        "set noaskquit",
        "set wr_vecnames",
        "set wr_singlescale",
    ])
    if analysis.kind == "op":
        lines.append("op")
    elif analysis.kind == "tran":
        if analysis.step is None or analysis.stop is None:
            raise GerberSimulationError("tran 分析必须提供 step 和 stop")
        lines.append(f"tran {analysis.step:.12g} {analysis.stop:.12g}")
    else:
        raise GerberSimulationError(f"不支持的分析类型: {analysis.kind}")

    lines.append(f"wrdata {quote_spice_path(Path(data_path.name))} {' '.join(expressions)}")
    lines.extend(["quit", ".endc", ".end", ""])
    return "\n".join(lines), headers


def component_to_spice_line(component: PassiveComponent) -> str:
    a = spice_node_name(component.nodes[0])
    b = spice_node_name(component.nodes[1])
    if component.kind == "R":
        return f"{component.designator} {a} {b} {component.value:.12g}"
    if component.kind == "C":
        return f"{component.designator} {a} {b} {component.value:.12g}"
    if component.kind == "L":
        return f"{component.designator} {a} {b} {component.value:.12g}"
    if component.kind == "D":
        return f"{component.designator} {a} {b} DDEFAULT"
    raise GerberSimulationError(f"不支持的器件类型: {component.kind}")


def source_to_spice_line(source: SourceSpec) -> str:
    prefix = "V" if source.kind == "voltage" else "I"
    name = sanitize_symbol_name(source.name)
    a = spice_node_name(source.positive)
    b = spice_node_name(source.negative)
    waveform = source.waveform
    kind = str(waveform.get("kind", "dc")).strip().lower()
    if kind == "dc":
        return f"{prefix}{name} {a} {b} DC {float(waveform.get('value', 0.0)):.12g}"
    if kind == "pulse":
        return (
            f"{prefix}{name} {a} {b} PULSE("
            f"{float(waveform.get('low', 0.0)):.12g} {float(waveform.get('high', 1.0)):.12g} "
            f"{float(waveform.get('delay', 0.0)):.12g} {float(waveform.get('rise', 1e-9)):.12g} "
            f"{float(waveform.get('fall', 1e-9)):.12g} {float(waveform.get('width', 1e-6)):.12g} "
            f"{float(waveform.get('period', 1e-3)):.12g})"
        )
    if kind == "sine":
        return (
            f"{prefix}{name} {a} {b} SIN("
            f"{float(waveform.get('offset', 0.0)):.12g} {float(waveform.get('amplitude', 1.0)):.12g} "
            f"{float(waveform.get('frequency', 1.0)):.12g} 0 0 {float(waveform.get('phase_deg', 0.0)):.12g})"
        )
    if kind == "step":
        t_step = float(waveform.get("t_step", 0.0))
        low = float(waveform.get("v1", 0.0))
        high = float(waveform.get("v2", 1.0))
        edge = max(t_step * 1e-6, 1e-12)
        return f"{prefix}{name} {a} {b} PWL(0 {low:.12g} {t_step:.12g} {low:.12g} {t_step + edge:.12g} {high:.12g})"
    raise GerberSimulationError(f"不支持的源波形: {kind}")


def observation_to_spice_expression(token: str, probe_map: dict[str, ProbeMapping]) -> tuple[str, str]:
    text = token.strip()
    diff_match = re.fullmatch(r"V\(([^(),]+),([^(),]+)\)", text, flags=re.IGNORECASE)
    if diff_match:
        left = spice_node_name(diff_match.group(1).strip())
        right = spice_node_name(diff_match.group(2).strip())
        return f"v({left},{right})", text
    single_voltage_match = re.fullmatch(r"V\(([^(),]+)\)", text, flags=re.IGNORECASE)
    if single_voltage_match:
        node = spice_node_name(single_voltage_match.group(1).strip())
        return f"v({node})", text
    if text.lower().startswith("probe:"):
        probe_name = text.split(":", 1)[1]
        if probe_name not in probe_map:
            raise GerberSimulationError(f"未知探针: {probe_name}")
        return f"v({spice_node_name(probe_map[probe_name].net)})", probe_name
    if text in probe_map:
        return f"v({spice_node_name(probe_map[text].net)})", text
    if text.upper().startswith("I(") and text.endswith(")"):
        branch = sanitize_symbol_name(text[2:-1])
        return f"i(V{branch})", text
    return f"v({spice_node_name(text)})", text


def run_ngspice(executable: Path, deck_path: Path, log_path: Path) -> None:
    result = subprocess.run(
        [str(executable), "-b", "-o", str(log_path), str(deck_path)],
        capture_output=True,
        cwd=str(deck_path.parent),
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GerberSimulationError(
            f"ngspice 执行失败: exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )


def parse_wrdata(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise GerberSimulationError(f"ngspice 未生成 wrdata 文件: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return []
    headers = re.split(r"\s+", lines[0])
    rows: list[dict[str, float]] = []
    for line in lines[1:]:
        parts = re.split(r"\s+", line)
        if len(parts) < len(headers):
            continue
        row = {headers[index]: float(parts[index]) for index in range(len(headers))}
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = [key for key in rows[0].keys() if key.lower() != "time" and key.lower() != "time_s"]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        summary[key] = {"min": min(values), "max": max(values), "final": values[-1]}
    return summary


def quote_spice_path(path: Path) -> str:
    text = path.as_posix()
    return f'"{text}"' if any(char.isspace() for char in text) else text


def spice_node_name(net: str) -> str:
    normalized = normalize_net_name(net)
    if normalized == GROUND_CANONICAL:
        return GROUND_CANONICAL
    return "N_" + sanitize_symbol_name(normalized)


def sanitize_symbol_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip().upper())
    cleaned = cleaned.strip("_") or "X"
    return cleaned if cleaned[0].isalpha() else "N" + cleaned


def canonical_observation_token(token: str, aliases: dict[str, str]) -> str:
    text = str(token).strip()
    if text.lower().startswith("probe:"):
        return text
    diff_match = re.fullmatch(r"V\(([^(),]+),([^(),]+)\)", text, flags=re.IGNORECASE)
    if diff_match:
        left = canonical_net(diff_match.group(1).strip(), aliases)
        right = canonical_net(diff_match.group(2).strip(), aliases)
        return f"V({left},{right})"
    single_voltage_match = re.fullmatch(r"V\(([^(),]+)\)", text, flags=re.IGNORECASE)
    if single_voltage_match:
        return f"V({canonical_net(single_voltage_match.group(1).strip(), aliases)})"
    if text.startswith("I(") and text.endswith(")"):
        return f"I({text[2:-1].strip().upper()})"
    return canonical_net(text, aliases)


def canonical_net(net: str, aliases: dict[str, str]) -> str:
    current = normalize_net_name(net)
    seen = set()
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


def normalize_net_name(net: str) -> str:
    text = str(net).strip().upper()
    return GROUND_CANONICAL if text in {"0", "GND", "GROUND"} else text


def parse_component_value(value_text: str, kind: str) -> float | None:
    if kind == "D":
        return 0.0
    cleaned = value_text.strip().replace("Ω", "").replace("ohm", "").replace("Ohm", "")
    cleaned = cleaned.replace("F", "").replace("H", "") if kind in {"C", "L"} else cleaned
    upper_cleaned = cleaned.upper()
    if kind == "R" and re.fullmatch(r"\d+R\d+", upper_cleaned):
        cleaned = upper_cleaned.replace("R", ".", 1)
        upper_cleaned = cleaned.upper()
    if kind == "R" and upper_cleaned.endswith("R"):
        cleaned = cleaned[:-1]
        upper_cleaned = cleaned.upper()
    if upper_cleaned in {"0R", "0"}:
        return 1e-9
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z]+)?\s*", cleaned)
    if not match:
        return None
    magnitude = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    if not suffix:
        return magnitude
    if suffix in VALUE_SUFFIXES:
        return magnitude * VALUE_SUFFIXES[suffix]
    if suffix.endswith("OHM") and suffix[:-3] in VALUE_SUFFIXES:
        return magnitude * VALUE_SUFFIXES[suffix[:-3]]
    return None


def sorted_pin_items(items: Iterable[tuple[Any, Any]]) -> list[tuple[str, Any]]:
    def sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
        key = str(item[0])
        return (0, f"{int(key):08d}") if key.isdigit() else (1, key)

    return [(str(key), value) for key, value in sorted(items, key=sort_key)]


def resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    fallback = (WORKSPACE_ROOT / path).resolve()
    return fallback


def resolve_optional_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    return resolve_path(base_dir, Path(str(value)))


def sanitize_path_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "scenario"
