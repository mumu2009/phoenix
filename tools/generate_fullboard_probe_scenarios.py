#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import deque
from pathlib import Path
from typing import Any
import re
import os

from stimulation_design_import import _build_easyeda_pin_connectivity_report, _build_imported_scenario, _parse_easyeda_netlists, detect_design_input_kind, import_design_input
from stimulation_library_catalog import expand_library_instances


GROUND_NAMES = {"0", "GND", "AGND", "PGND", "DGND", "EARTH", "GROUND", "AMBIENT"}
CONDUCTIVE_COMPONENT_KINDS = {"resistor", "inductor", "diode"}
POWER_NET_MARKERS = (
    "VCC",
    "VDD",
    "VIN",
    "VBUS",
    "VREF",
    "VDDA",
    "VDDS",
    "DVDD",
    "AVDD",
    "VAA",
    "VEE",
    "VPP",
    "BAT",
    "PWR",
    "SYS",
    "12V",
    "5V",
    "3V3",
    "3V0",
    "2V8",
    "2V5",
    "1V8",
    "1V2",
    "24V",
)
PATH_DRIVE_LEVELS = (1.2, 2.4, 3.0)
PATH_SINK_CURRENT_A = 2e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline and fault full-board probe scenarios with distributed observe sets.")
    parser.add_argument("--input", required=True, help="Design input path, typically an EasyEDA netlist JSON.")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated scenario JSON files.")
    parser.add_argument("--samples", type=int, default=50, help="Total sample count across baseline and fault scenarios.")
    parser.add_argument("--focus-designators", nargs="+", default=["J1", "U30"], help="Designators whose full pin maps must be included in probe metadata.")
    parser.add_argument("--fault-designator", default="U30", help="Library instance name to degrade in the fault scenario.")
    parser.add_argument("--drc-log", default="", help="Optional JLC schematic DRC log path. When provided, singleton-net lines are cross-checked against the imported netlist.")
    parser.add_argument("--enforce-drc-consistency", action="store_true", help="Fail scenario generation when DRC singleton nets are still missing/singleton in imported netlist.")
    return parser.parse_args()


def load_easyeda_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"EasyEDA netlist must be a JSON object: {path}")
    return payload


def select_easyeda_netlist_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    selected = sorted(path.resolve() for path in input_path.glob("eext_netlist_*.json") if path.is_file())
    if selected:
        return selected
    fallback = sorted(path.resolve() for path in input_path.glob("*.json") if path.is_file())
    if not fallback:
        raise SystemExit(f"No EasyEDA netlist JSON files were found under: {input_path}")
    return fallback


def load_design_scenario(input_path: Path) -> tuple[dict[str, Any], str, list[Path]]:
    if input_path.is_dir():
        selected_paths = select_easyeda_netlist_paths(input_path)
        components, libraries, unsupported = _parse_easyeda_netlists(selected_paths)
        connectivity_report = _build_easyeda_pin_connectivity_report(selected_paths)
        scenario = _build_imported_scenario(input_path.name, input_path, "easyeda-folder-selected", components, libraries, unsupported, False, [], connectivity_report)
        return scenario, "easyeda-folder-selected", selected_paths
    input_kind = detect_design_input_kind(input_path)
    return import_design_input(input_path, input_kind, consider_line_effects=False), input_kind, [input_path]


def load_easyeda_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    return [load_easyeda_payload(path) for path in paths]


def pin_sort_key(text: str) -> tuple[int, Any]:
    text = str(text)
    return (0, int(text)) if text.isdigit() else (1, text)


def normalize_net_name(text: str) -> str:
    return str(text).strip().upper()


def unique_preserving(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def collect_pin_map(payloads: list[dict[str, Any]], designator: str) -> list[dict[str, str]]:
    target = designator.strip().upper()
    for payload in payloads:
        for block in payload.values():
            if not isinstance(block, dict):
                continue
            props = dict(block.get("props", {}))
            if str(props.get("Designator", "")).strip().upper() != target:
                continue
            pins = dict(block.get("pins", {}))
            return [
                {"pin": str(pin), "net": str(net)}
                for pin, net in sorted(pins.items(), key=lambda item: pin_sort_key(item[0]))
            ]
    raise SystemExit(f"Designator not found in EasyEDA netlist: {designator}")


def collect_all_pin_maps(payloads: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    merged: dict[str, dict[str, str]] = {}
    for payload in payloads:
        for block in payload.values():
            if not isinstance(block, dict):
                continue
            props = dict(block.get("props", {}))
            designator = str(props.get("Designator", "")).strip().upper()
            if not designator:
                continue
            pins = dict(block.get("pins", {}))
            target = merged.setdefault(designator, {})
            for pin, net in pins.items():
                pin_name = str(pin)
                net_name = str(net)
                if not net_name.strip():
                    continue
                existing = target.get(pin_name)
                if existing is not None and existing != net_name:
                    raise SystemExit(f"Conflicting pin map for {designator}.{pin_name}: {existing} / {net_name}")
                target[pin_name] = net_name
    return {
        designator: [
            {"pin": str(pin), "net": str(net)}
            for pin, net in sorted(pin_map.items(), key=lambda item: pin_sort_key(item[0]))
        ]
        for designator, pin_map in sorted(merged.items())
    }


def collect_net_endpoint_counts(pin_maps: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pin_map in pin_maps.values():
        for item in pin_map:
            net_name = str(item.get("net", "")).strip()
            if not net_name:
                continue
            counts[net_name] = counts.get(net_name, 0) + 1
    return counts


def parse_drc_singleton_nets(drc_log_path: Path) -> list[str]:
    text = drc_log_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"导线\s+([^\s]+)\s+\$\d+N\d+\s+是单网络")
    nets = sorted({match.group(1).strip() for match in pattern.finditer(text) if match.group(1).strip()})
    return nets


def build_drc_comparison(drc_nets: list[str], net_counts: dict[str, int]) -> dict[str, Any]:
    missing = [net for net in drc_nets if net not in net_counts]
    singleton = [net for net in drc_nets if net_counts.get(net, 0) == 1]
    multi = [net for net in drc_nets if net_counts.get(net, 0) >= 2]
    return {
        "drcSingletonNetCount": len(drc_nets),
        "resolvedInNetlistCount": len(multi),
        "missingInNetlistCount": len(missing),
        "singletonInNetlistCount": len(singleton),
        "missingInNetlistPreview": missing[:80],
        "singletonInNetlistPreview": singleton[:80],
        "resolvedInNetlistPreview": multi[:80],
    }


def require_pin_map(pin_maps: dict[str, list[dict[str, str]]], designator: str) -> list[dict[str, str]]:
    target = designator.strip().upper()
    if target not in pin_maps:
        raise SystemExit(f"Designator not found in EasyEDA netlist: {designator}")
    return list(pin_maps[target])


def preferred_observe_token(nodes: list[str]) -> str | None:
    cleaned = [str(node) for node in nodes if str(node).strip()]
    if not cleaned:
        return None
    preferred = [node for node in cleaned if normalize_net_name(node) not in GROUND_NAMES]
    return preferred[0] if preferred else cleaned[0]


def split_round_robin(tokens: list[str], bucket_count: int) -> list[list[str]]:
    groups = [[] for _ in range(bucket_count)]
    for index, token in enumerate(tokens):
        groups[index % bucket_count].append(token)
    return groups


def build_coverage_tokens(base_scenario: dict[str, Any], mandatory_tokens: list[str]) -> tuple[list[str], int]:
    expanded = expand_library_instances(list(base_scenario.get("libraries", []))) if base_scenario.get("libraries") else {"components": []}
    all_components = list(base_scenario.get("components", [])) + list(expanded.get("components", []))
    component_tokens = [
        token
        for token in (preferred_observe_token(list(component.get("nodes", []))) for component in all_components)
        if token is not None
    ]
    coverage_tokens = unique_preserving(list(mandatory_tokens) + component_tokens)
    return coverage_tokens, len(all_components)


def is_probably_power_net(net_name: str) -> bool:
    canonical = normalize_net_name(net_name)
    if not canonical:
        return False
    if canonical in GROUND_NAMES:
        return True
    if canonical.startswith("+"):
        return True
    if canonical.endswith("V") and canonical[:-1].replace(".", "").isdigit():
        return True
    return any(marker in canonical for marker in POWER_NET_MARKERS)


def endpoint_label(endpoint: dict[str, Any]) -> str:
    return f"{endpoint['designator']}.{endpoint['pin']}"


def endpoint_sort_key(endpoint: dict[str, Any]) -> tuple[int, str, tuple[int, Any]]:
    return (0 if endpoint.get("connectorLike") else 1, str(endpoint["designator"]), pin_sort_key(str(endpoint["pin"])))


def collect_probe_endpoints(pin_maps: dict[str, list[dict[str, str]]], required_designators: set[str] | None = None) -> list[dict[str, Any]]:
    required = {item.strip().upper() for item in (required_designators or set()) if item and str(item).strip()}
    endpoints: list[dict[str, Any]] = []
    for designator, pin_map in pin_maps.items():
        connector_like = designator.startswith("J")
        force_include = designator in required
        for item in pin_map:
            net_name = str(item["net"]).strip()
            if not net_name or normalize_net_name(net_name) in GROUND_NAMES or is_probably_power_net(net_name):
                continue
            endpoints.append(
                {
                    "designator": designator,
                    "pin": str(item["pin"]),
                    "net": net_name,
                    "connectorLike": connector_like,
                    "required": force_include,
                }
            )
    connector_endpoints = [item for item in endpoints if item["connectorLike"]]
    required_endpoints = [item for item in endpoints if item.get("required")]
    if len(connector_endpoints) >= 2:
        selected = {endpoint_label(item): item for item in connector_endpoints}
        for item in required_endpoints:
            selected[endpoint_label(item)] = item
        return sorted(selected.values(), key=endpoint_sort_key)
    return sorted(endpoints, key=endpoint_sort_key)


def build_conductive_graph(base_scenario: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    expanded = expand_library_instances(list(base_scenario.get("libraries", []))) if base_scenario.get("libraries") else {"components": []}
    all_components = list(base_scenario.get("components", [])) + list(expanded.get("components", []))
    graph: dict[str, list[dict[str, Any]]] = {}
    for component in all_components:
        kind = str(component.get("kind", "")).strip().lower()
        if kind not in CONDUCTIVE_COMPONENT_KINDS:
            continue
        nodes = [str(node).strip() for node in component.get("nodes", []) if str(node).strip()]
        if len(nodes) < 2:
            continue
        node_a, node_b = nodes[:2]
        designator = str(component.get("designator") or component.get("name") or "UNNAMED")
        edge = {
            "designator": designator,
            "kind": kind,
            "value": component.get("value"),
            "nodeA": node_a,
            "nodeB": node_b,
        }
        graph.setdefault(node_a, []).append({**edge, "neighbor": node_b})
        graph.setdefault(node_b, []).append({**edge, "neighbor": node_a})
    return graph, all_components


def path_pair_key(start: dict[str, Any], end: dict[str, Any]) -> tuple[str, str]:
    labels = sorted((endpoint_label(start), endpoint_label(end)))
    return labels[0], labels[1]


def reconstruct_component_path(
    start_net: str,
    target_net: str,
    predecessors: dict[str, tuple[str, dict[str, Any]] | None],
) -> tuple[list[str], list[dict[str, Any]]]:
    nets = [target_net]
    components: list[dict[str, Any]] = []
    current = target_net
    while current != start_net:
        step = predecessors.get(current)
        if step is None:
            raise SystemExit(f"Broken path reconstruction: {start_net} -> {target_net}")
        previous_net, edge = step
        components.append(
            {
                "designator": str(edge["designator"]),
                "kind": str(edge["kind"]),
                "value": edge.get("value"),
                "from": previous_net,
                "to": current,
            }
        )
        current = previous_net
        nets.append(current)
    nets.reverse()
    components.reverse()
    return nets, components


def path_touches_fault_designator(components: list[dict[str, Any]], fault_designator: str) -> bool:
    target = fault_designator.strip().upper()
    for component in components:
        designator = str(component.get("designator", "")).strip().upper()
        if designator == target or designator.startswith(f"{target}_"):
            return True
    return False


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return (
        0 if candidate.get("touchesFaultDesignator") else 1,
        int(candidate.get("edgeCount", 0)),
        len(candidate.get("nets", [])),
        endpoint_label(candidate["start"]),
        endpoint_label(candidate["end"]),
    )


def should_replace_candidate(current: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
    if current is None:
        return True
    return candidate_sort_key(candidate) < candidate_sort_key(current)


def record_candidate_path(
    best: dict[tuple[str, str], dict[str, Any]],
    start: dict[str, Any],
    end: dict[str, Any],
    nets: list[str],
    components: list[dict[str, Any]],
    fault_designator: str,
    path_type: str,
) -> None:
    if endpoint_label(end) <= endpoint_label(start):
        return
    if start["designator"] == end["designator"] and start["pin"] == end["pin"]:
        return
    candidate = {
        "start": dict(start),
        "end": dict(end),
        "nets": list(nets),
        "components": list(components),
        "edgeCount": len(components),
        "pathType": path_type,
        "touchesFaultDesignator": path_touches_fault_designator(components, fault_designator),
    }
    key = path_pair_key(start, end)
    if should_replace_candidate(best.get(key), candidate):
        best[key] = candidate


def discover_physical_pin_paths(
    base_scenario: dict[str, Any],
    pin_maps: dict[str, list[dict[str, str]]],
    fault_designator: str,
    required_designators: set[str] | None = None,
) -> list[dict[str, Any]]:
    endpoints = collect_probe_endpoints(pin_maps, required_designators)
    if len(endpoints) < 2:
        raise SystemExit("Not enough signal endpoints were discovered to build pin-to-pin physical scenarios.")

    endpoints_by_net: dict[str, list[dict[str, Any]]] = {}
    for endpoint in endpoints:
        endpoints_by_net.setdefault(endpoint["net"], []).append(endpoint)

    best_candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for net_name, members in sorted(endpoints_by_net.items()):
        ordered = sorted(members, key=endpoint_sort_key)
        for first_index, start in enumerate(ordered):
            for end in ordered[first_index + 1 :]:
                if start["designator"] == end["designator"]:
                    continue
                record_candidate_path(best_candidates, start, end, [net_name], [], fault_designator, "shared-net")

    graph, _ = build_conductive_graph(base_scenario)
    for start in endpoints:
        start_net = str(start["net"])
        if start_net not in graph:
            continue
        queue: deque[str] = deque([start_net])
        predecessors: dict[str, tuple[str, dict[str, Any]] | None] = {start_net: None}
        target_hits = 0
        while queue and target_hits < 12:
            current_net = queue.popleft()
            if current_net != start_net:
                for end in sorted(endpoints_by_net.get(current_net, []), key=endpoint_sort_key):
                    if end["designator"] == start["designator"]:
                        continue
                    nets, components = reconstruct_component_path(start_net, current_net, predecessors)
                    if not components:
                        continue
                    record_candidate_path(best_candidates, start, end, nets, components, fault_designator, "component-path")
                    target_hits += 1
                    if target_hits >= 12:
                        break
            for edge in sorted(graph.get(current_net, []), key=lambda item: (str(item["designator"]), str(item["neighbor"]))):
                neighbor = str(edge["neighbor"])
                if neighbor in predecessors:
                    continue
                predecessors[neighbor] = (current_net, edge)
                queue.append(neighbor)

    paths = sorted(best_candidates.values(), key=candidate_sort_key)
    if not paths:
        raise SystemExit("No physical pin-to-pin paths were discovered from the imported netlist graph.")
    return paths


def select_sample_paths(
    paths: list[dict[str, Any]],
    count: int,
    start_index: int,
    require_fault_designator: bool,
) -> list[dict[str, Any]]:
    pool = [item for item in paths if item.get("touchesFaultDesignator")] if require_fault_designator else list(paths)
    if not pool:
        pool = list(paths)
    if not pool:
        return []
    return [pool[(start_index + index) % len(pool)] for index in range(count)]


def build_path_drive_value(sample_index: int) -> float:
    return PATH_DRIVE_LEVELS[sample_index % len(PATH_DRIVE_LEVELS)]


def build_path_sample_analysis(sample_index: int, observe_group: list[str], path: dict[str, Any]) -> dict[str, Any]:
    drive_value = build_path_drive_value(sample_index)
    source_name = f"PATHDRV_{sample_index + 1:02d}"
    load_name = f"PATHLOAD_{sample_index + 1:02d}"
    input_net = str(path["start"]["net"])
    output_net = str(path["end"]["net"])
    observe_tokens = unique_preserving(
        list(observe_group) + [input_net, output_net, f"I({source_name})"] + [str(token) for token in path.get("nets", [])[:16]]
    )
    component_sequence = [str(item["designator"]) for item in path.get("components", [])]
    path_description = f"{endpoint_label(path['start'])} -> {endpoint_label(path['end'])}"
    if component_sequence:
        path_description = f"{path_description} via {' -> '.join(component_sequence[:8])}"
    elif path.get("nets"):
        path_description = f"{path_description} on shared net {path['nets'][0]}"
    return {
        "type": "op",
        "observe": observe_tokens,
        "sources": [
            {
                "name": source_name,
                "kind": "voltage",
                "positive": input_net,
                "negative": "GND",
                "waveform": {"kind": "dc", "value": drive_value},
            },
            {
                "name": load_name,
                "kind": "current",
                "positive": output_net,
                "negative": "GND",
                "waveform": {"kind": "dc", "value": PATH_SINK_CURRENT_A},
            },
        ],
        "assertions": [
            {
                "actual": output_net,
                "expected": drive_value,
                "tolerance": 0.6,
                "description": f"{path_description}: output pin remains driven through the imported physical path under load",
            }
        ],
        "sample": {
            "sampleId": sample_index + 1,
            "pathModel": "pin-net-pin",
            "pathType": str(path.get("pathType", "component-path")),
            "inputPin": dict(path["start"]),
            "outputPin": dict(path["end"]),
            "netSequence": list(path.get("nets", [])),
            "componentSequence": component_sequence,
            "driveValue": drive_value,
            "sinkCurrentA": PATH_SINK_CURRENT_A,
            "touchesFaultDesignator": bool(path.get("touchesFaultDesignator")),
        },
    }


def build_path_analyses(groups: list[list[str]], paths: list[dict[str, Any]], start_index: int = 0) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not group:
            continue
        path = paths[index % len(paths)]
        analyses.append(build_path_sample_analysis(start_index + index, group, path))
    return analyses


def attach_probe_metadata(
    metadata: dict[str, Any],
    focus_pin_maps: dict[str, list[dict[str, str]]],
    total_samples: int,
    scenario_samples: int,
    coverage_token_count: int,
    simulated_component_count: int,
    scenario_kind: str,
    physical_path_count: int,
) -> dict[str, Any]:
    updated = copy.deepcopy(metadata)
    updated["probePlan"] = {
        "scenarioKind": scenario_kind,
        "totalSampleCount": total_samples,
        "scenarioSampleCount": scenario_samples,
        "focusPinMaps": focus_pin_maps,
        "focusUniqueNetCount": {
            designator: len(unique_preserving([item["net"] for item in pin_map]))
            for designator, pin_map in focus_pin_maps.items()
        },
        "coverageTokenCount": coverage_token_count,
        "simulatedComponentCount": simulated_component_count,
        "physicalPathCount": physical_path_count,
        "pathModel": "pin-net-pin",
    }
    return updated


def degraded_libraries(libraries: list[dict[str, Any]], fault_designator: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    target = fault_designator.strip().upper()
    for item in libraries:
        cloned = copy.deepcopy(item)
        if str(cloned.get("name", "")).strip().upper() == target:
            params = dict(cloned.get("params", {}))
            params.update({"core_load_ohm": 1e12, "series_ohm": 1e9, "core_heat_scale": 0.01})
            cloned["params"] = params
        result.append(cloned)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")

    if args.enforce_drc_consistency:
        os.environ["STIM_ENFORCE_DRC_CROSS_CHECK"] = "1"
    else:
        os.environ.pop("STIM_ENFORCE_DRC_CROSS_CHECK", None)

    base_scenario, input_kind, source_paths = load_design_scenario(input_path)
    payloads = load_easyeda_payloads(source_paths)
    all_pin_maps = collect_all_pin_maps(payloads)
    net_counts = collect_net_endpoint_counts(all_pin_maps)

    drc_comparison: dict[str, Any] | None = None
    drc_log_path: Path | None = None
    if args.drc_log:
        drc_log_path = Path(args.drc_log).resolve()
        if not drc_log_path.is_file():
            raise SystemExit(f"DRC log file not found: {drc_log_path}")
        drc_nets = parse_drc_singleton_nets(drc_log_path)
        drc_comparison = build_drc_comparison(drc_nets, net_counts)
        unresolved = drc_comparison["missingInNetlistCount"] + drc_comparison["singletonInNetlistCount"]
        if args.enforce_drc_consistency and unresolved > 0:
            raise SystemExit(
                "DRC/netlist consistency check failed: "
                f"missing={drc_comparison['missingInNetlistCount']}, singleton={drc_comparison['singletonInNetlistCount']}"
            )

    focus_pin_maps = {designator: require_pin_map(all_pin_maps, designator) for designator in args.focus_designators}
    mandatory_tokens = unique_preserving(
        [item["net"] for pin_map in focus_pin_maps.values() for item in pin_map if normalize_net_name(item["net"]) not in GROUND_NAMES]
    )
    coverage_tokens, simulated_component_count = build_coverage_tokens(base_scenario, mandatory_tokens)
    required_path_designators = {str(item).strip().upper() for item in list(args.focus_designators) + [args.fault_designator] if str(item).strip()}
    physical_paths = discover_physical_pin_paths(base_scenario, all_pin_maps, args.fault_designator, required_path_designators)

    baseline_samples = args.samples // 2
    fault_samples = args.samples - baseline_samples
    token_groups = split_round_robin(coverage_tokens, args.samples)
    baseline_groups = token_groups[:baseline_samples]
    fault_groups = token_groups[baseline_samples:]
    baseline_path_samples = select_sample_paths(physical_paths, baseline_samples, 0, require_fault_designator=False)
    fault_path_samples = select_sample_paths(physical_paths, fault_samples, baseline_samples, require_fault_designator=True)

    baseline = copy.deepcopy(base_scenario)
    baseline["name"] = f"{base_scenario.get('name', 'fullboard')}_probe_baseline_{baseline_samples:02d}"
    baseline["analyses"] = build_path_analyses(baseline_groups, baseline_path_samples, start_index=0)
    baseline["metadata"] = attach_probe_metadata(
        dict(base_scenario.get("metadata", {})),
        focus_pin_maps,
        args.samples,
        baseline_samples,
        len(coverage_tokens),
        simulated_component_count,
        "baseline",
        len(physical_paths),
    )
    if drc_comparison is not None:
        baseline["metadata"]["drcConsistency"] = {
            **drc_comparison,
            "drcLog": str(drc_log_path),
            "enforced": bool(args.enforce_drc_consistency),
        }

    fault = copy.deepcopy(base_scenario)
    fault["name"] = f"{base_scenario.get('name', 'fullboard')}_probe_fault_{fault_samples:02d}"
    fault["libraries"] = degraded_libraries(list(base_scenario.get("libraries", [])), args.fault_designator)
    fault["analyses"] = build_path_analyses(fault_groups, fault_path_samples, start_index=baseline_samples)
    fault["metadata"] = attach_probe_metadata(
        dict(base_scenario.get("metadata", {})),
        focus_pin_maps,
        args.samples,
        fault_samples,
        len(coverage_tokens),
        simulated_component_count,
        "fault-open-circuit",
        len(physical_paths),
    )
    fault["metadata"]["faultMode"] = {"designator": args.fault_designator, "mode": "open-circuit-equivalent"}
    fault["metadata"]["faultReachablePhysicalPathCount"] = sum(1 for item in physical_paths if item.get("touchesFaultDesignator"))
    if drc_comparison is not None:
        fault["metadata"]["drcConsistency"] = {
            **drc_comparison,
            "drcLog": str(drc_log_path),
            "enforced": bool(args.enforce_drc_consistency),
        }

    baseline_path = out_dir / f"{baseline['name']}.json"
    fault_path = out_dir / f"{fault['name']}.json"
    write_json(baseline_path, baseline)
    write_json(fault_path, fault)

    manifest = {
        "input": str(input_path),
        "inputKind": input_kind,
        "inputFileCount": len(source_paths),
        "totalSamples": args.samples,
        "baselineSamples": baseline_samples,
        "faultSamples": fault_samples,
        "coverageTokenCount": len(coverage_tokens),
        "simulatedComponentCount": simulated_component_count,
        "physicalPathCount": len(physical_paths),
        "faultReachablePhysicalPathCount": sum(1 for item in physical_paths if item.get("touchesFaultDesignator")),
        "pathModel": "pin-net-pin",
        "focusDesignators": list(args.focus_designators),
        "focusPinCounts": {designator: len(pin_map) for designator, pin_map in focus_pin_maps.items()},
        "files": {
            "baseline": str(baseline_path),
            "fault": str(fault_path),
        },
    }
    if drc_comparison is not None:
        manifest["drcConsistency"] = {
            **drc_comparison,
            "drcLog": str(drc_log_path),
            "enforced": bool(args.enforce_drc_consistency),
        }
    write_json(out_dir / "manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())