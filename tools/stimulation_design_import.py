#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


GROUND_NAMES = {"0", "GND", "AGND", "PGND", "DGND", "EARTH", "GROUND"}
PASSIVE_KIND_MAP = {"R": "resistor", "C": "capacitor", "L": "inductor", "D": "diode"}
TRACE_THICKNESS_MM = 0.035
TRACE_CAPACITANCE_PF_PER_MM = 0.18
TRACE_INDUCTANCE_NH_PER_MM = 0.85
ACTIVE_LIBRARY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stm32f103c8_interface", ("STM32F103", "BLUEPILL")),
    ("stm32f407_interface", ("STM32F407", "STM32F4", "GD32F427", "GD32F4")),
    ("at89c51_interface", ("AT89C51", "8051", "89C51")),
    ("raspberry_pi_40pin_interface", ("RASPBERRY_PI", "RPI", "BCM2711", "40PIN")),
    ("arduino_uno_interface", ("ARDUINO_UNO", "ATMEGA328P", "UNO")),
    ("arduino_mega_interface", ("ARDUINO_MEGA", "ATMEGA2560", "MEGA_2560")),
    ("esp32_devkit_interface", ("ESP32", "ESP_WROOM")),
    ("esp8266_nodemcu_interface", ("ESP8266", "NODEMCU", "ESP_12")),
    ("can_transceiver_terminated", ("CAN", "SN65HVD", "TJA1050", "MCP2551")),
    ("rs485_transceiver_terminated", ("RS485", "MAX485", "SP3485", "SN75176")),
    ("logic_buffer_multi_io", ("SN74LVC244", "74LVC244", "LVC244APWR")),
    ("analog_switch_spst_load", ("SN74LVC1G66", "74LVC1G66", "1G66DCKR", "ANALOG_SWITCH")),
    ("analog_mux_8ch_load", ("74VHC4051", "74HC4051", "CD4051", "4051")),
    ("comparator_dual_open_collector", ("LM393", "LM2903")),
    ("comparator_single_fast", ("LMV7219",)),
    ("segment_display_4digit_common_anode", ("SLR0394DWA5BD", "4_DIGIT_7_SEGMENT", "COMMON_ANODE")),
    ("spi_peripheral_load", ("74LVC595", "74HC595", "74VHC595", "MCP3208", "MCP3008", "W25Q", "EEPROM", "FLASH")),
    ("ldo_basic", ("AMS1117", "LM1117", "XC6206", "ME6211", "MIC5205", "BQ24650", "HLK_", "REGULATOR", "LDO")),
)


class DesignImportError(RuntimeError):
    pass


def detect_design_input_kind(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_file():
        suffix = resolved.suffix.lower()
        if suffix == ".zip":
            return "gerber-zip"
        if suffix == ".kicad_pcb":
            raise DesignImportError(f"KiCad 导入入口已经停用；当前主线只接受 EasyEDA/JLCEDA netlist 或 scenario: {resolved}")
        if suffix == ".json":
            payload = _try_load_json(resolved)
            if _looks_like_easyeda_netlist(payload):
                return "easyeda-netlist-json"
            if _looks_like_multiphysics_scenario(payload):
                return "scenario-json"
    if resolved.is_dir():
        if any(_is_gerber_filename(item.name) for item in resolved.rglob("*") if item.is_file()):
            return "gerber-folder"
        if any(item.suffix.lower() == ".kicad_pcb" for item in resolved.rglob("*.kicad_pcb")):
            raise DesignImportError(f"KiCad 工程目录已经停用；请改用当前 EasyEDA/JLCEDA netlist 导入: {resolved}")
        if any(item.suffix.lower() == ".kicad_sch" for item in resolved.rglob("*.kicad_sch")):
            raise DesignImportError(f"KiCad 工程目录已经停用；请改用当前 EasyEDA/JLCEDA netlist 导入: {resolved}")
        if any(_looks_like_easyeda_netlist(_try_load_json(item)) for item in resolved.rglob("*.json") if item.is_file()):
            return "easyeda-folder"
    raise DesignImportError(f"无法识别输入格式: {resolved}")


def import_design_input(path: Path, input_kind: str, consider_line_effects: bool) -> dict[str, Any]:
    resolved = path.resolve()
    kind = detect_design_input_kind(resolved) if input_kind == "auto" else input_kind
    if kind in {"gerber-zip", "gerber-folder"}:
        raise DesignImportError("Gerber zip/目录请走 Gerber 入口：tools/stimulation --gerber-root ...，或继续使用 --gerber-scenario。")
    if kind == "scenario-json":
        payload = _try_load_json(resolved)
        if not isinstance(payload, dict):
            raise DesignImportError(f"场景 JSON 不是对象: {resolved}")
        return payload
    if kind == "easyeda-netlist-json":
        components, libraries, unsupported = _parse_easyeda_netlists([resolved])
        connectivity_report = _build_easyeda_pin_connectivity_report([resolved])
        return _build_imported_scenario(resolved.stem, resolved, kind, components, libraries, unsupported, consider_line_effects, [], connectivity_report)
    if kind == "easyeda-folder":
        json_paths = [item for item in sorted(resolved.rglob("*.json")) if _looks_like_easyeda_netlist(_try_load_json(item))]
        if not json_paths:
            raise DesignImportError(f"EasyEDA 目录中未找到可识别的 netlist JSON: {resolved}")
        components, libraries, unsupported = _parse_easyeda_netlists(json_paths)
        connectivity_report = _build_easyeda_pin_connectivity_report(json_paths)
        return _build_imported_scenario(resolved.name, resolved, kind, components, libraries, unsupported, consider_line_effects, [], connectivity_report)
    if kind in {"kicad-folder", "kicad-pcb"}:
        pcb_path, schematic_paths = _locate_kicad_project_files(resolved)
        if pcb_path is None:
            raise DesignImportError(f"KiCad 工程缺少 .kicad_pcb，当前导入器还不能只靠 .kicad_sch 恢复连线: {resolved}")
        schematic_meta = _parse_kicad_symbol_metadata(schematic_paths)
        components, libraries, unsupported, segments = _parse_kicad_pcb(pcb_path, schematic_meta)
        return _build_imported_scenario(pcb_path.stem, resolved, kind, components, libraries, unsupported, consider_line_effects, segments)
    raise DesignImportError(f"当前不支持的输入类型: {kind}")


def _build_imported_scenario(
    name: str,
    source_path: Path,
    input_kind: str,
    components: list[dict[str, Any]],
    libraries: list[dict[str, Any]],
    unsupported: list[str],
    consider_line_effects: bool,
    segments: list[dict[str, Any]],
    connectivity_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    imported_components = [dict(item) for item in components]
    imported_libraries = [
        {
            **dict(item),
            "nets": {str(key): str(value) for key, value in dict(item.get("nets", {})).items()},
        }
        for item in libraries
    ]
    line_summary: list[dict[str, Any]] = []
    if consider_line_effects:
        imported_components, imported_libraries, line_summary = _inject_line_effect_models(imported_components, imported_libraries, segments)

    connectivity_report = connectivity_report or _build_connectivity_report(imported_components, imported_libraries)
    _raise_for_connectivity_risks(connectivity_report, source_path)

    nets = sorted(
        {str(node) for component in imported_components for node in component.get("nodes", [])}
        | {str(node) for library in imported_libraries for node in dict(library.get("nets", {})).values()}
    )
    sources = _build_auto_sources(nets)
    observe = _pick_observe_nets(nets)
    if not observe:
        observe = [net for net in nets if _normalize_net_name(net) not in GROUND_NAMES][:6]
    regions = _build_import_regions(nets, line_summary)
    metadata = {
        "importSummary": {
            "inputKind": input_kind,
            "inputPath": str(source_path),
            "componentCount": len(imported_components),
            "libraryInstanceCount": len(imported_libraries),
            "unsupportedComponentCount": len(set(unsupported)),
            "lineEffectsEnabled": consider_line_effects,
            "lineModelCount": len(line_summary),
        },
        "connectivityReport": connectivity_report,
        "importNotes": [
            f"导入源: {source_path}",
            f"输入类型: {input_kind}",
			f"自动识别到 {len(imported_components)} 个离散可仿真器件、{len(imported_libraries)} 个接口级库实例，忽略 {len(set(unsupported))} 个暂不支持的器件。",
        ],
        "unsupportedPreview": sorted(set(unsupported))[:80],
        "lineEffects": line_summary,
    }
    if consider_line_effects:
        metadata["importNotes"].append(f"已启用线路寄生近似，共注入 {len(line_summary)} 组 trace R/L/C 与热模型。")
    elif segments:
        metadata["importNotes"].append("当前未启用线路寄生近似；可加 --consider-line-effects 让导入器注入 trace R/L/C/热模型。")
    metadata["importNotes"].append(
        f"连通性检查通过：{connectivity_report['netCount']} 个网络、{connectivity_report['endpointCount']} 个端点、"
        f"{connectivity_report['singletonNetCount']} 个单端风险、{connectivity_report['orphanDesignatorCount']} 个孤立器件。"
    )

    scenario = {
        "name": _sanitize_identifier(name) or "imported_design",
        "domain": "multiphysics",
        "libraries": imported_libraries,
        "components": imported_components,
        "sources": sources,
        "analyses": [
            {"type": "op", "observe": observe[:10]},
            {"type": "tran", "step": 5e-5, "stop": 5e-3, "observe": observe[:8]},
        ],
        "regions": regions,
        "partition": {"enabled": True, "workers": 4, "maxComponentsPerPartition": 180, "highFanoutBoundaryThreshold": 18},
        "targetRuntimeSeconds": 60.0,
        "metadata": metadata,
    }
    if sources:
        scenario["seed_nets"] = sorted({source["positive"] for source in sources if _normalize_net_name(source["positive"]) not in GROUND_NAMES})[:12]
        scenario["hops"] = 5
    return scenario


def _try_load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _looks_like_multiphysics_scenario(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("analyses"), list) and (
        isinstance(payload.get("components"), list)
        or isinstance(payload.get("netlists"), list)
        or isinstance(payload.get("libraries"), list)
    )


def _looks_like_easyeda_netlist(payload: Any) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    sample = next(iter(payload.values()))
    return isinstance(sample, dict) and isinstance(sample.get("props"), dict) and isinstance(sample.get("pins"), dict)


def _summarize_connectivity_records(records: list[tuple[str, str, str]]) -> dict[str, Any]:
    endpoint_counts: Counter[str] = Counter()
    power_nets: set[str] = set()
    net_members: dict[str, list[str]] = {}
    designator_nets: dict[str, list[str]] = {}

    for designator, pin_name, net_name in records:
        canonical_net = _canonical_net(net_name)
        if not canonical_net:
            continue
        endpoint_counts[canonical_net] += 1
        net_members.setdefault(canonical_net, []).append(f"{designator}:{pin_name}")
        designator_nets.setdefault(designator, []).append(canonical_net)
        if _guess_voltage_for_net(canonical_net) is not None or canonical_net in _power_net_candidates([canonical_net]):
            power_nets.add(canonical_net)

    singleton_nets = sorted(
        net
        for net, count in endpoint_counts.items()
        if count < 2 and _normalize_net_name(net) not in GROUND_NAMES and net not in power_nets
    )
    singleton_set = set(singleton_nets)
    orphan_designators: list[str] = []
    for designator, nets in designator_nets.items():
        signal_nets = [net for net in _dedupe_nets(nets) if _normalize_net_name(net) not in GROUND_NAMES]
        if signal_nets and all(net in singleton_set for net in signal_nets):
            orphan_designators.append(designator)
    orphan_designators.sort()
    return {
        "netCount": len(endpoint_counts),
        "endpointCount": sum(endpoint_counts.values()),
        "singletonNetCount": len(singleton_nets),
        "singletonNetPreview": singleton_nets[:80],
        "singletonNetMembers": {net: net_members.get(net, [])[:16] for net in singleton_nets[:24]},
        "orphanDesignatorCount": len(orphan_designators),
        "orphanDesignatorPreview": orphan_designators[:80],
    }


def _build_easyeda_pin_connectivity_report(paths: list[Path]) -> dict[str, Any]:
    records: list[tuple[str, str, str]] = []
    for path in paths:
        payload = _try_load_json(path)
        if not _looks_like_easyeda_netlist(payload):
            continue
        assert isinstance(payload, dict)
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            props = dict(entry.get("props", {}))
            designator = str(props.get("Designator", "")).strip().upper()
            if not designator:
                continue
            for pin_name, net_name in dict(entry.get("pins", {})).items():
                if str(net_name).strip():
                    records.append((designator, str(pin_name), str(net_name)))
    report = _summarize_connectivity_records(records)

    endpoint_counts: Counter[str] = Counter()
    for _, _, net_name in records:
        canonical_net = _canonical_net(net_name)
        if canonical_net:
            endpoint_counts[canonical_net] += 1

    drc_log_path = _find_latest_sch_drc_log(paths)
    if drc_log_path is not None:
        drc_singletons = _parse_singleton_nets_from_jlc_drc_log(drc_log_path)
        missing_in_netlist = sorted(net for net in drc_singletons if endpoint_counts.get(net, 0) == 0)
        singleton_in_netlist = sorted(net for net in drc_singletons if endpoint_counts.get(net, 0) == 1)
        resolved_in_netlist = sorted(net for net in drc_singletons if endpoint_counts.get(net, 0) >= 2)
        report["drcCrossCheck"] = {
            "logPath": str(drc_log_path),
            "drcSingletonNetCount": len(drc_singletons),
            "missingInNetlistCount": len(missing_in_netlist),
            "missingInNetlistPreview": missing_in_netlist[:80],
            "singletonInNetlistCount": len(singleton_in_netlist),
            "singletonInNetlistPreview": singleton_in_netlist[:80],
            "resolvedInNetlistCount": len(resolved_in_netlist),
            "resolvedInNetlistPreview": resolved_in_netlist[:80],
        }
    return report


def _find_latest_sch_drc_log(paths: list[Path]) -> Path | None:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        parent = path.parent.resolve()
        if parent in seen:
            continue
        seen.add(parent)
        candidates.extend(item.resolve() for item in parent.glob("schDrcLog_*.txt") if item.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _parse_singleton_nets_from_jlc_drc_log(log_path: Path) -> list[str]:
    singleton_nets: set[str] = set()
    pattern = re.compile(r"导线\s+([^\s]+)\s+\$\d+N\d+\s+是单网络")
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    for line in text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        net_name = _canonical_net(match.group(1))
        if not net_name or _normalize_net_name(net_name) in GROUND_NAMES:
            continue
        singleton_nets.add(net_name)
    return sorted(singleton_nets)


def _build_connectivity_report(components: list[dict[str, Any]], libraries: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[tuple[str, str, str]] = []
    for component in components:
        designator = str(component.get("designator", "")).strip().upper() or "UNKNOWN"
        for index, node in enumerate(component.get("nodes", []), start=1):
            records.append((designator, str(index), str(node)))

    for library in libraries:
        designator = str(library.get("name", "")).strip().upper() or str(library.get("model", "LIBRARY")).strip().upper()
        for pin_name, net_name in dict(library.get("nets", {})).items():
            records.append((designator, str(pin_name), str(net_name)))
    return _summarize_connectivity_records(records)


def _raise_for_connectivity_risks(report: dict[str, Any], source_path: Path) -> None:
    singleton_preview = list(report.get("singletonNetPreview", []))
    orphan_preview = list(report.get("orphanDesignatorPreview", []))
    drc_cross_check = dict(report.get("drcCrossCheck", {}))
    drc_missing_preview = list(drc_cross_check.get("missingInNetlistPreview", []))
    drc_singleton_preview = list(drc_cross_check.get("singletonInNetlistPreview", []))
    enforce_drc_cross_check = _should_enforce_drc_cross_check()
    if not singleton_preview and not orphan_preview and (not enforce_drc_cross_check or (not drc_missing_preview and not drc_singleton_preview)):
        return
    fragments: list[str] = [f"导入源 {source_path} 存在未闭合连接，已停止场景生成"]
    if singleton_preview:
        fragments.append(
            f"singleton nets={report.get('singletonNetCount', 0)}，示例: {', '.join(singleton_preview[:12])}"
        )
    if orphan_preview:
        fragments.append(
            f"orphan designators={report.get('orphanDesignatorCount', 0)}，示例: {', '.join(orphan_preview[:12])}"
        )
    if enforce_drc_cross_check and drc_missing_preview:
        fragments.append(
            f"DRC 单网络在网表缺失={drc_cross_check.get('missingInNetlistCount', 0)}，示例: {', '.join(drc_missing_preview[:12])}"
        )
    if enforce_drc_cross_check and drc_singleton_preview:
        fragments.append(
            f"DRC 单网络在网表仍是单端={drc_cross_check.get('singletonInNetlistCount', 0)}，示例: {', '.join(drc_singleton_preview[:12])}"
        )
    raise DesignImportError("；".join(fragments))


def _should_enforce_drc_cross_check() -> bool:
    return os.environ.get("STIM_ENFORCE_DRC_CROSS_CHECK", "0").strip().lower() in {"1", "true", "yes", "on"}


def _parse_easyeda_netlists(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    components: list[dict[str, Any]] = []
    libraries: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for path in paths:
        payload = _try_load_json(path)
        if not _looks_like_easyeda_netlist(payload):
            continue
        assert isinstance(payload, dict)
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            props = dict(entry.get("props", {}))
            designator = str(props.get("Designator", "")).strip().upper()
            if not designator:
                continue
            kind_letter = designator[0]
            pin_items = sorted(entry.get("pins", {}).items(), key=lambda item: _pin_sort_key(str(item[0])))
            nets = [_canonical_net(str(net)) for _, net in pin_items if str(net).strip()]
            if kind_letter not in PASSIVE_KIND_MAP or len(nets) < 2:
                library_instance = _build_active_library_instance(
                    designator,
                    nets,
                    {
                        "value": str(props.get("value", "")).strip(),
                        "deviceName": str(props.get("device_name", "")).strip(),
                        "supplierPart": str(props.get("Supplier Part", "")).strip(),
                        "searchKeyword": str(props.get("search_keyword", "")).strip(),
                    },
                    pin_items=pin_items,
                )
                if library_instance:
                    libraries.append(library_instance)
                else:
                    unsupported.append(designator)
                continue
            value = str(props.get("value", "")).strip()
            if not value:
                unsupported.append(designator)
                continue
            components.append(
                {
                    "designator": designator,
                    "kind": PASSIVE_KIND_MAP[kind_letter],
                    "value": value,
                    "nodes": nets[:2],
                    "source": str(path),
                    "physics": _default_import_physics(kind_letter),
                    "metadata": {
                        "deviceName": str(props.get("device_name", "")).strip(),
                        "supplierPart": str(props.get("Supplier Part", "")).strip(),
                        "sourceFile": str(path),
                    },
                }
            )
    return components, libraries, unsupported


def _locate_kicad_project_files(path: Path) -> tuple[Path | None, list[Path]]:
    resolved = path.resolve()
    if resolved.is_file() and resolved.suffix.lower() == ".kicad_pcb":
        return resolved, sorted(resolved.parent.glob("*.kicad_sch"))
    if not resolved.is_dir():
        return None, []
    pcb_candidates = sorted(resolved.rglob("*.kicad_pcb"), key=lambda item: (len(item.parts), len(item.name)))
    schematic_paths = sorted(resolved.rglob("*.kicad_sch"), key=lambda item: (len(item.parts), item.name))
    return (pcb_candidates[0] if pcb_candidates else None), schematic_paths


def _parse_kicad_symbol_metadata(paths: list[Path]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for block in _extract_sexpr_blocks(text, "symbol"):
            if "(lib_id " not in block:
                continue
            reference = _match_property(block, "Reference").strip().upper()
            if not reference or not any(char.isdigit() for char in reference):
                continue
            metadata[reference] = {
                "value": _match_property(block, "Value").strip(),
                "footprint": _match_property(block, "Footprint").strip(),
                "libId": _match_first(block, r"\(lib_id\s+\"([^\"]+)\"") or "",
                "manufacturerPart": _match_property(block, "Manufacturer Part").strip(),
                "supplierPart": _match_property(block, "Supplier Part").strip(),
                "sourceFile": str(path),
            }
    return metadata


def _parse_kicad_pcb(pcb_path: Path, schematic_meta: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    text = pcb_path.read_text(encoding="utf-8", errors="ignore")
    net_lookup = {
        int(match.group(1)): _canonical_net(match.group(2))
        for match in re.finditer(r'(?m)^\s*\(net\s+(\d+)\s+\"([^\"]*)\"\)\s*$', text)
    }
    components: list[dict[str, Any]] = []
    libraries: list[dict[str, Any]] = []
    unsupported: list[str] = []

    for block in _extract_sexpr_blocks(text, "footprint"):
        reference = _match_property(block, "Reference").strip().upper()
        if not reference:
            continue
        kind_letter = reference[0]
        ordered_nets = _parse_kicad_pad_nets(block, net_lookup)
        symbol_info = schematic_meta.get(reference, {})
        library_instance = _build_active_library_instance(
            reference,
            ordered_nets,
            {
                "value": symbol_info.get("value", "") or _match_property(block, "Value").strip(),
                "footprint": _match_first(block, r'\(footprint\s+"([^"]+)"') or "",
                "libId": symbol_info.get("libId", ""),
                "manufacturerPart": symbol_info.get("manufacturerPart", ""),
                "supplierPart": symbol_info.get("supplierPart", ""),
            },
            pin_items=None,
        )
        if kind_letter not in PASSIVE_KIND_MAP or len(ordered_nets) < 2:
            if library_instance:
                libraries.append(library_instance)
            else:
                unsupported.append(reference)
            continue
        value = symbol_info.get("value") or _match_property(block, "Value").strip()
        if not value:
            unsupported.append(reference)
            continue
        components.append(
            {
                "designator": reference,
                "kind": PASSIVE_KIND_MAP[kind_letter],
                "value": value,
                "nodes": ordered_nets[:2],
                "source": str(pcb_path),
                "physics": _default_import_physics(kind_letter),
                "metadata": {
                    "footprint": _match_first(block, r'\(footprint\s+\"([^\"]+)\"') or "",
                    "schematicValue": symbol_info.get("value", ""),
                    "libId": symbol_info.get("libId", ""),
                    "manufacturerPart": symbol_info.get("manufacturerPart", ""),
                    "supplierPart": symbol_info.get("supplierPart", ""),
                    "sourceFile": str(pcb_path),
                },
            }
        )

    segments: list[dict[str, Any]] = []
    for block in _extract_sexpr_blocks(text, "segment"):
        net_name = _match_first(block, r'\(net\s+\"([^\"]+)\"\)')
        if net_name is None:
            net_name = _match_first(block, r"\(net\s+\d+\s+\"([^\"]+)\"\)")
        if net_name is None:
            net_code = _match_first(block, r"\(net\s+(\d+)\)")
            if net_code is None:
                continue
            net_name = net_lookup.get(int(net_code), "")
        net_name = _canonical_net(str(net_name))
        if not net_name:
            continue
        start = _match_first(block, r"\(start\s+([-0-9.]+)\s+([-0-9.]+)\)", tuple_result=True)
        end = _match_first(block, r"\(end\s+([-0-9.]+)\s+([-0-9.]+)\)", tuple_result=True)
        width = _match_first(block, r"\(width\s+([-0-9.]+)\)")
        layer = _match_first(block, r'\(layer\s+\"([^\"]+)\"') or ""
        if start is None or end is None or width is None:
            continue
        x1, y1 = (float(start[0]), float(start[1]))
        x2, y2 = (float(end[0]), float(end[1]))
        segments.append(
            {
                "net": net_name,
                "length_mm": math.hypot(x2 - x1, y2 - y1),
                "width_mm": max(float(width), 0.01),
                "layer": layer,
            }
        )
    return components, libraries, unsupported, segments


def _parse_kicad_pad_nets(block: str, net_lookup: dict[int, str]) -> list[str]:
    ordered: list[tuple[str, str]] = []
    for pad_block in _extract_sexpr_blocks(block, "pad"):
        pad_number = _match_first(pad_block, r'\(pad\s+\"?([^\"\s]+)\"?') or ""
        simple_name = _match_first(pad_block, r'\(net\s+\"([^\"]+)\"\)')
        if pad_number and simple_name:
            ordered.append((str(pad_number), _canonical_net(str(simple_name))))
            continue
        match = re.search(r"\(net\s+(\d+)\s+\"([^\"]+)\"\)", pad_block)
        if match:
            ordered.append((str(pad_number), _canonical_net(match.group(2))))
            continue
        net_code = _match_first(pad_block, r"\(net\s+(\d+)\)")
        if pad_number and net_code is not None and int(net_code) in net_lookup:
            ordered.append((str(pad_number), net_lookup[int(net_code)]))
    unique: list[str] = []
    seen: set[str] = set()
    for _, net in sorted(ordered, key=lambda item: _pin_sort_key(item[0])):
        if net not in seen:
            seen.add(net)
            unique.append(net)
    return unique


def _inject_line_effect_models(
    components: list[dict[str, Any]],
    libraries: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not segments:
        return components, libraries, []

    usage_count: Counter[str] = Counter()
    for component in components:
        for node in component.get("nodes", []):
            usage_count[str(node)] += 1
    for library in libraries:
        for node in dict(library.get("nets", {})).values():
            usage_count[str(node)] += 1

    line_by_net: dict[str, dict[str, float]] = {}
    for segment in segments:
        net = str(segment["net"])
        stats = line_by_net.setdefault(net, {"length_mm": 0.0, "width_area": 0.0})
        length_mm = float(segment.get("length_mm", 0.0))
        width_mm = max(float(segment.get("width_mm", 0.0)), 0.01)
        stats["length_mm"] += length_mm
        stats["width_area"] += width_mm * length_mm

    candidates: dict[str, dict[str, float]] = {}
    for net, stats in line_by_net.items():
        net_norm = _normalize_net_name(net)
        if net_norm in GROUND_NAMES:
            continue
        length_mm = stats["length_mm"]
        avg_width = stats["width_area"] / max(length_mm, 1e-9)
        if length_mm < 8.0:
            continue
        if usage_count.get(net, 0) < 2 and _guess_voltage_for_net(net) is None and not _looks_like_signal_net(net):
            continue
        candidates[net] = {"length_mm": length_mm, "avg_width_mm": avg_width}

    if not candidates:
        return components, libraries, []

    remapped_components = [dict(component) for component in components]
    remapped_libraries = [{**dict(library), "nets": dict(library.get("nets", {}))} for library in libraries]
    line_summary: list[dict[str, Any]] = []
    rewritten_nets = {net: f"{_sanitize_identifier(net)}__LOAD" for net in candidates}
    for component in remapped_components:
        component["nodes"] = [rewritten_nets.get(str(node), str(node)) for node in component.get("nodes", [])]
    for library in remapped_libraries:
        library["nets"] = {str(key): rewritten_nets.get(str(value), str(value)) for key, value in dict(library.get("nets", {})).items()}

    line_components: list[dict[str, Any]] = []
    for net, stats in sorted(candidates.items()):
        safe = _sanitize_identifier(net)
        line_mid = f"{safe}__TRACE_MID"
        load_node = rewritten_nets[net]
        heat_node = f"{safe}__TRACE_HEAT"
        length_mm = float(stats["length_mm"])
        avg_width_mm = max(float(stats["avg_width_mm"]), 0.10)
        resistance = _estimate_trace_resistance_ohm(length_mm, avg_width_mm)
        inductance = max(length_mm * TRACE_INDUCTANCE_NH_PER_MM * 1e-9, 1e-12)
        capacitance = max(length_mm * TRACE_CAPACITANCE_PF_PER_MM * 1e-12, 0.0)
        thermal_r = max(18.0 / max(length_mm, 1.0), 0.8)
        thermal_c = max(length_mm * 0.09, 0.6)

        line_components.extend(
            [
                {
                    "designator": f"TLR_{safe}",
                    "kind": "resistor",
                    "value": max(resistance, 1e-5),
                    "nodes": [net, line_mid],
                    "physics": {
                        "temperatureCoefficientPerC": 0.0039,
                        "thermal": {"positive": heat_node, "negative": "AMBIENT", "powerScale": 1.0},
                        "tags": ["line-effect", "trace-resistance"],
                    },
                },
                {
                    "designator": f"TLL_{safe}",
                    "kind": "inductor",
                    "value": inductance,
                    "nodes": [line_mid, load_node],
                    "physics": {
                        "lossResistance": max(resistance * 0.35, 1e-6),
                        "saturationCurrent": max(0.3, 12.0 / (1.0 + length_mm / 60.0)),
                        "thermal": {"positive": heat_node, "negative": "AMBIENT", "powerScale": 0.35},
                        "tags": ["line-effect", "trace-inductance"],
                    },
                },
                {"designator": f"TLRTH_{safe}", "kind": "thermal_resistor", "value": thermal_r, "nodes": [heat_node, "AMBIENT"]},
                {"designator": f"TLCTH_{safe}", "kind": "thermal_capacitor", "value": thermal_c, "nodes": [heat_node, "AMBIENT"]},
            ]
        )
        if capacitance > 0.0:
            line_components.append({"designator": f"TLC_{safe}", "kind": "capacitor", "value": capacitance, "nodes": [load_node, "GND"]})

        line_summary.append(
            {
                "net": net,
                "lengthMm": length_mm,
                "averageWidthMm": avg_width_mm,
                "resistanceOhm": resistance,
                "inductanceH": inductance,
                "capacitanceF": capacitance,
            }
        )

    remapped_components.extend(line_components)
    return remapped_components, remapped_libraries, line_summary


def _build_auto_sources(nets: list[str]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for net in sorted(nets):
        voltage = _guess_voltage_for_net(net)
        if voltage is None:
            continue
        sources.append(
            {
                "name": f"AUTO_{_sanitize_identifier(net)}",
                "kind": "voltage",
                "positive": net,
                "negative": "GND",
                "waveform": {"kind": "dc", "value": voltage},
            }
        )
    return sources


def _pick_observe_nets(nets: list[str]) -> list[str]:
    ranked = sorted(nets, key=lambda item: (_observe_rank(item), item))
    result: list[str] = []
    seen: set[str] = set()
    for net in ranked:
        normalized = _normalize_net_name(net)
        if normalized in GROUND_NAMES or normalized in seen:
            continue
        seen.add(normalized)
        result.append(net)
        if len(result) >= 12:
            break
    return result


def _build_import_regions(nets: list[str], line_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    power_nets = [net for net in nets if _guess_voltage_for_net(net) is not None][:8]
    signal_nets = [net for net in nets if _looks_like_signal_net(net)][:8]
    thermal_nets = [f"{_sanitize_identifier(item['net'])}__TRACE_HEAT" for item in line_summary[:6]]
    regions: list[dict[str, Any]] = []
    if power_nets:
        regions.append({"name": "import_power", "seedNets": power_nets, "hops": 4})
    if signal_nets:
        regions.append({"name": "import_signal", "seedNets": signal_nets, "hops": 4})
    if thermal_nets:
        regions.append({"name": "import_trace_thermal", "seedNets": thermal_nets + ["AMBIENT"], "hops": 3})
    return regions


def _build_active_library_instance(
    designator: str,
    nets: list[str],
    metadata: dict[str, str],
    pin_items: list[tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    unique_nets = _dedupe_nets(nets)
    if len(unique_nets) < 2:
        return None
    model = _detect_active_library_model(metadata)
    if not model:
        return None
    if model == "logic_buffer_multi_io":
        library_instance = _build_logic_buffer_instance(designator, unique_nets, pin_items)
        if library_instance:
            return library_instance
    inferred_nets = _infer_library_nets(model, unique_nets)
    if not inferred_nets:
        return None
    return {"model": model, "name": designator, "nets": inferred_nets}


def _build_logic_buffer_instance(
    designator: str,
    nets: list[str],
    pin_items: list[tuple[str, str]] | None,
) -> dict[str, Any] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    if gnd is None or vdd is None:
        return None
    ordered = list(pin_items or [])
    if not ordered:
        inferred = _infer_multi_io_nets(nets)
        if not inferred:
            return None
        return {"model": "logic_buffer_multi_io", "name": designator, "nets": inferred}

    canonical_pin_map = {
        str(pin).strip(): _canonical_net(str(net))
        for pin, net in ordered
        if str(net).strip()
    }
    pair_map = (("2", "18"), ("3", "17"), ("4", "15"), ("5", "13"), ("6", "11"), ("8", "12"), ("9", "14"), ("7", "16"))
    mapped = {"vdd": vdd, "gnd": gnd}
    pair_pins: list[dict[str, str]] = []
    pair_index = 0
    for input_pin, output_pin in pair_map:
        input_net = canonical_pin_map.get(input_pin)
        output_net = canonical_pin_map.get(output_pin)
        if not input_net or not output_net:
            continue
        if _normalize_net_name(input_net) in GROUND_NAMES or _normalize_net_name(output_net) in GROUND_NAMES:
            continue
        if input_net == output_net:
            continue
        mapped[f"in{pair_index}"] = input_net
        mapped[f"out{pair_index}"] = output_net
        pair_pins.append(
            {
                "inputPin": input_pin,
                "outputPin": output_pin,
                "inputNet": input_net,
                "outputNet": output_net,
            }
        )
        pair_index += 1
    if not pair_pins:
        inferred = _infer_multi_io_nets(nets)
        if not inferred:
            return None
        return {"model": "logic_buffer_multi_io", "name": designator, "nets": inferred}
    return {"model": "logic_buffer_multi_io", "name": designator, "nets": mapped, "params": {"pairPins": pair_pins}}


def _detect_active_library_model(metadata: dict[str, str]) -> str | None:
    blob = " ".join(_normalize_net_name(value) for value in metadata.values() if value).strip()
    if not blob:
        return None
    for model, tokens in ACTIVE_LIBRARY_RULES:
        if any(token in blob for token in tokens):
            return model
    return None


def _infer_library_nets(model: str, nets: list[str]) -> dict[str, str] | None:
    if model == "ldo_basic":
        return _infer_ldo_nets(nets)
    if model in {"stm32f103c8_interface", "stm32f407_interface", "at89c51_interface", "raspberry_pi_40pin_interface", "arduino_uno_interface", "arduino_mega_interface", "esp32_devkit_interface", "esp8266_nodemcu_interface"}:
        return _infer_mcu_nets(nets)
    if model == "spi_peripheral_load":
        return _infer_spi_peripheral_nets(nets)
    if model == "logic_buffer_multi_io":
        return _infer_multi_io_nets(nets)
    if model == "analog_switch_spst_load":
        return _infer_analog_switch_nets(nets)
    if model == "analog_mux_8ch_load":
        return _infer_analog_mux_nets(nets)
    if model in {"comparator_dual_open_collector", "comparator_single_fast"}:
        return _infer_comparator_nets(nets)
    if model == "segment_display_4digit_common_anode":
        return _infer_display_nets(nets)
    if model == "can_transceiver_terminated":
        return _infer_can_nets(nets)
    if model == "rs485_transceiver_terminated":
        return _infer_rs485_nets(nets)
    return None


def _infer_ldo_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    power_nets = _power_net_candidates(nets)
    if gnd is None or len(power_nets) < 2:
        return None
    vin = _choose_highest_power_net(power_nets)
    vout = _choose_lowest_power_net(power_nets, exclude={vin} if vin else set())
    if vin is None or vout is None or vin == vout:
        return None
    return {"vin": vin, "vout": vout, "gnd": gnd}


def _infer_mcu_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    if gnd is None or vdd is None:
        return None
    mapped = {"vdd": vdd, "gnd": gnd}
    patterns = {
        "nrst": ("NRST", "RESET", "RST"),
        "boot0": ("BOOT0", "GPIO0", "IO0"),
        "en": ("CHIP_EN", "GLOBAL_EN", "ENABLE", "EN"),
        "uart_tx": ("UART_TX", "TXD", "TX"),
        "uart_rx": ("UART_RX", "RXD", "RX"),
        "spi_clk": ("SPI_SCLK", "SPI_CLK", "SCLK", "CLK"),
        "spi_mosi": ("SPI_MOSI", "MOSI", "DIN", "SDI"),
        "spi_miso": ("SPI_MISO", "MISO", "DOUT", "SDO"),
        "spi_cs": ("SPI_CS", "NSS", "CE", "CS", "SS"),
        "i2c_scl": ("I2C_SCL", "SCL"),
        "i2c_sda": ("I2C_SDA", "SDA"),
        "pwm0": ("PWM", "DRV", "GATE"),
    }
    for key, tokens in patterns.items():
        net = _find_net_by_patterns(nets, tokens, exclude=set(mapped.values()))
        if net is not None:
            mapped[key] = net
    return mapped


def _infer_spi_peripheral_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    if gnd is None or vdd is None:
        return None
    mapped = {"vdd": vdd, "gnd": gnd}
    patterns = {
        "spi_clk": ("SPI_SCLK", "SPI_CLK", "SCLK", "CLK"),
        "spi_mosi": ("SPI_MOSI", "MOSI", "DIN", "SDI"),
        "spi_miso": ("SPI_MISO", "MISO", "DOUT", "SDO"),
        "spi_cs": ("SPI_CS", "NSS", "CE", "CS", "SS"),
    }
    for key, tokens in patterns.items():
        net = _find_net_by_patterns(nets, tokens, exclude=set(mapped.values()))
        if net is not None:
            mapped[key] = net
    extras = [net for net in _signal_net_candidates(nets) if net not in set(mapped.values())]
    for index, net in enumerate(extras[:8]):
        mapped[f"io{index}"] = net
    return mapped if len(mapped) >= 3 else None


def _infer_multi_io_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    if gnd is None or vdd is None:
        return None
    mapped = {"vdd": vdd, "gnd": gnd}
    extras = [net for net in _signal_net_candidates(nets) if net not in set(mapped.values())]
    for index, net in enumerate(extras[:16]):
        mapped[f"io{index}"] = net
    return mapped if any(key.startswith("io") for key in mapped) else None


def _infer_analog_mux_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    if gnd is None or vdd is None:
        return None
    mapped = {"vdd": vdd, "gnd": gnd}
    for key, tokens in {
        "addr0": ("ADDR0", "SEL0", "S0", "A0"),
        "addr1": ("ADDR1", "SEL1", "S1", "A1"),
        "addr2": ("ADDR2", "SEL2", "S2", "A2"),
        "enable": ("ENABLE", "OE", "INH", "EN"),
    }.items():
        net = _find_net_by_patterns(nets, tokens, exclude=set(mapped.values()))
        if net is not None:
            mapped[key] = net
    remaining = [net for net in _signal_net_candidates(nets) if net not in set(mapped.values())]
    if not remaining:
        return None
    mapped["common"] = _find_net_by_patterns(remaining, ("COM", "SIG", "ADC", "READ", "SENSE", "OUT", "MUX")) or remaining[0]
    remaining = [net for net in remaining if net != mapped["common"]]
    for index, net in enumerate(remaining[:8]):
        mapped[f"ch{index}"] = net
    return mapped if any(key.startswith("ch") for key in mapped) else None


def _infer_analog_switch_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    if gnd is None or vdd is None:
        return None
    mapped = {"vdd": vdd, "gnd": gnd}
    remaining = [net for net in _signal_net_candidates(nets) if net not in {gnd, vdd}]
    if len(remaining) < 3:
        return None
    control = _find_net_by_patterns(remaining, ("CTRL", "ENABLE", "EN", "OE", "SEL", "WCFG_COL", "COL"))
    if control is None:
        control = remaining[-1]
    mapped["ctrl"] = control
    remaining = [net for net in remaining if net != control]
    analog_a = _find_net_by_patterns(remaining, ("WCFG_ROW", "ROW", "COM", "COMMON", "SRC", "IN")) or remaining[0]
    mapped["a"] = analog_a
    remaining = [net for net in remaining if net != analog_a]
    analog_b = _find_net_by_patterns(remaining, ("ACCUM", "OUT", "SIG", "SENSE", "LOAD", "READ")) or remaining[0]
    mapped["b"] = analog_b
    return mapped


def _infer_comparator_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    if gnd is None or vdd is None:
        return None
    mapped = {"vdd": vdd, "gnd": gnd}
    remaining = [net for net in _signal_net_candidates(nets) if net not in {gnd, vdd}]
    outputs: list[str] = []
    for token in ("OUT", "CMP", "ALERT", "READ", "SDO"):
        net = _find_net_by_patterns(remaining, (token,), exclude=set(outputs))
        if net is not None:
            outputs.append(net)
    if not outputs and remaining:
        outputs = remaining[:2]
    for index, net in enumerate(outputs[:2]):
        mapped[f"out{index}"] = net
    inputs = [net for net in remaining if net not in outputs]
    for key, net in zip(("in0p", "in0n", "in1p", "in1n"), inputs[:4], strict=False):
        mapped[key] = net
    return mapped if outputs else None


def _infer_display_nets(nets: list[str]) -> dict[str, str] | None:
    mapped: dict[str, str] = {}
    patterns = {
        "seg_a": ("DISP_DRV_A", "SEG_A"),
        "seg_b": ("DISP_DRV_B", "SEG_B"),
        "seg_c": ("DISP_DRV_C", "SEG_C"),
        "seg_d": ("DISP_DRV_D", "SEG_D"),
        "seg_e": ("DISP_DRV_E", "SEG_E"),
        "seg_f": ("DISP_DRV_F", "SEG_F"),
        "seg_g": ("DISP_DRV_G", "SEG_G"),
        "seg_dp": ("DISP_DRV_DP", "SEG_DP", "DECIMAL"),
        "dig_0": ("DISP_DIG0", "DIG0", "COM0"),
        "dig_1": ("DISP_DIG1", "DIG1", "COM1"),
        "dig_2": ("DISP_DIG2", "DIG2", "COM2"),
        "dig_3": ("DISP_DIG3", "DIG3", "COM3"),
    }
    for key, tokens in patterns.items():
        net = _find_net_by_patterns(nets, tokens, exclude=set(mapped.values()))
        if net is not None:
            mapped[key] = net
    has_segment = any(key.startswith("seg_") for key in mapped)
    has_digit = any(key.startswith("dig_") for key in mapped)
    return mapped if has_segment and has_digit else None


def _infer_can_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    canh = _find_net_by_patterns(nets, ("CANH",))
    canl = _find_net_by_patterns(nets, ("CANL",))
    if None in {gnd, vdd, canh, canl}:
        return None
    return {"vdd": vdd, "gnd": gnd, "canh": canh, "canl": canl}


def _infer_rs485_nets(nets: list[str]) -> dict[str, str] | None:
    gnd = _find_ground_net(nets)
    vdd = _choose_logic_supply(nets)
    line_a = _find_net_by_patterns(nets, ("RS485_A", "DIFF_A", "BUS_A", "A_P"))
    line_b = _find_net_by_patterns(nets, ("RS485_B", "DIFF_B", "BUS_B", "B_P"))
    if None in {gnd, vdd, line_a, line_b}:
        return None
    return {"vdd": vdd, "gnd": gnd, "a": line_a, "b": line_b}


def _dedupe_nets(nets: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for net in nets:
        canonical = _canonical_net(str(net))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        result.append(canonical)
    return result


def _find_ground_net(nets: list[str]) -> str | None:
    for net in nets:
        if _normalize_net_name(net) in GROUND_NAMES:
            return net
    return None


def _find_net_by_patterns(nets: list[str], patterns: tuple[str, ...], exclude: set[str] | None = None) -> str | None:
    excluded = exclude or set()
    for pattern in patterns:
        for net in nets:
            if net in excluded:
                continue
            if pattern in _normalize_net_name(net):
                return net
    return None


def _power_net_candidates(nets: list[str]) -> list[str]:
    result: list[str] = []
    for net in _dedupe_nets(nets):
        upper = _normalize_net_name(net)
        if upper in GROUND_NAMES:
            continue
        if _guess_voltage_for_net(net) is not None or any(token in upper for token in ("VDD", "VCC", "VIN", "VBAT", "VSYS", "VOUT", "POWER", "SUPPLY", "AVDD", "DVDD")):
            result.append(net)
    return result


def _signal_net_candidates(nets: list[str]) -> list[str]:
    power_nets = set(_power_net_candidates(nets))
    return [net for net in _dedupe_nets(nets) if net not in power_nets and _normalize_net_name(net) not in GROUND_NAMES]


def _choose_logic_supply(nets: list[str]) -> str | None:
    candidates = _power_net_candidates(nets)
    if not candidates:
        return None
    return min(candidates, key=_logic_supply_rank)


def _choose_highest_power_net(nets: list[str]) -> str | None:
    candidates = _power_net_candidates(nets)
    if not candidates:
        return None
    return max(candidates, key=_power_voltage_rank)


def _choose_lowest_power_net(nets: list[str], exclude: set[str] | None = None) -> str | None:
    candidates = [net for net in _power_net_candidates(nets) if net not in (exclude or set())]
    if not candidates:
        return None
    return min(candidates, key=_power_voltage_rank)


def _logic_supply_rank(net: str) -> tuple[float, float, str]:
    voltage = _guess_voltage_for_net(net)
    if voltage is None:
        return (3.0, 999.0, net)
    if abs(voltage - 3.3) < 0.25:
        return (0.0, abs(voltage - 3.3), net)
    if abs(voltage - 5.0) < 0.35:
        return (1.0, abs(voltage - 5.0), net)
    return (2.0, voltage, net)


def _power_voltage_rank(net: str) -> tuple[float, str]:
    voltage = _guess_voltage_for_net(net)
    return (voltage if voltage is not None else 999.0, net)


def _default_import_physics(kind_letter: str) -> dict[str, Any]:
    if kind_letter == "R":
        return {"temperatureCoefficientPerC": 0.0039}
    if kind_letter == "L":
        return {"lossResistance": 0.05, "saturationCurrent": 3.0}
    if kind_letter == "D":
        return {"junctionTempCoefficientPerC": -0.002}
    return {}


def _estimate_trace_resistance_ohm(length_mm: float, width_mm: float) -> float:
    resistivity = 1.724e-8
    length_m = max(length_mm, 1e-6) * 1e-3
    area_m2 = max(width_mm, 0.01) * 1e-3 * TRACE_THICKNESS_MM * 1e-3
    return resistivity * length_m / max(area_m2, 1e-15)


def _observe_rank(net: str) -> tuple[int, int]:
    voltage = _guess_voltage_for_net(net)
    if voltage is not None:
        return (0, -int(voltage * 1000.0))
    if _looks_like_signal_net(net):
        return (1, 0)
    if "THERM" in _normalize_net_name(net) or "TEMP" in _normalize_net_name(net):
        return (2, 0)
    return (3, 0)


def _guess_voltage_for_net(net: str) -> float | None:
    upper = _normalize_net_name(net)
    direct = {
        "3V3": 3.3,
        "+3V3": 3.3,
        "5V": 5.0,
        "+5V": 5.0,
        "12V": 12.0,
        "+12V": 12.0,
        "15V": 15.0,
        "+15V": 15.0,
        "24V": 24.0,
        "+24V": 24.0,
        "VBAT": 12.6,
        "VBUS": 5.0,
        "VIN": 12.0,
        "VCC": 5.0,
        "AVDD": 3.3,
        "DVDD": 3.3,
    }
    if upper in direct:
        return direct[upper]
    match = re.search(r"(?<![A-Z0-9])(\d{1,2})V(\d+)?(?![A-Z0-9])", upper)
    if match:
        integer = int(match.group(1))
        frac = match.group(2)
        return float(f"{integer}.{frac}") if frac else float(integer)
    return None


def _looks_like_signal_net(net: str) -> bool:
    upper = _normalize_net_name(net)
    tokens = ("SPI", "UART", "I2C", "PWM", "CAN", "RS485", "RX", "TX", "CLK", "CS", "EN", "BOOT", "GPIO")
    return any(token in upper for token in tokens)


def _extract_sexpr_blocks(text: str, token: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(rf"(?m)^\s*\({re.escape(token)}(?:\s|$)")
    for match in pattern.finditer(text):
        start = match.start()
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(text[start:index + 1])
                    break
    return blocks


def _match_property(block: str, name: str) -> str:
    return _match_first(block, rf'\(property\s+\"{re.escape(name)}\"\s+\"([^\"]*)\"') or ""


def _match_first(block: str, pattern: str, group: int = 1, tuple_result: bool = False) -> str | tuple[str, ...] | None:
    match = re.search(pattern, block, flags=re.MULTILINE)
    if not match:
        return None
    if tuple_result:
        return match.groups()
    return match.group(group)


def _pin_sort_key(text: str) -> tuple[int, str]:
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def _normalize_net_name(text: str) -> str:
    return re.sub(r"[^A-Z0-9_+]", "_", text.strip().upper())


def _canonical_net(text: str) -> str:
    normalized = _normalize_net_name(text)
    return "GND" if normalized in GROUND_NAMES else normalized


def _sanitize_identifier(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").upper()


def _is_gerber_filename(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".gtl", ".gbl", ".gto", ".gbo", ".gbr", ".ger", ".gm1", ".gko", ".pho", ".art"))