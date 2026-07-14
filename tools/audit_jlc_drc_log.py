from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


SINGLETON_PATTERN = re.compile(r"导线\s+([^\s]+)\s+\$\d+N\d+\s+是单网络")
MULTI_NAME_PATTERN = re.compile(r"导线\s+\$\d+N\d+\s+有多个网络名:\s*(.+)")
DEFAULT_NETLISTS = [
    Path("catastrophe/eext_netlist.json"),
    Path("catastrophe/partitionByFunction_netlist1.json"),
    Path("catastrophe/partitionByFunction_netlist2.json"),
    Path("catastrophe/partitionByFunction_netlist3.json"),
    Path("catastrophe/partitionByFunction_netlist4.json"),
]


def canonical_net(name: str) -> str:
    return str(name).strip()


def parse_drc_log(path: Path) -> tuple[list[str], list[list[str]]]:
    singletons: list[str] = []
    multi_name_groups: list[list[str]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    seen_singletons: set[str] = set()
    seen_groups: set[tuple[str, ...]] = set()
    for line in text.splitlines():
        singleton_match = SINGLETON_PATTERN.search(line)
        if singleton_match is not None:
            net_name = canonical_net(singleton_match.group(1))
            if net_name and net_name not in seen_singletons:
                seen_singletons.add(net_name)
                singletons.append(net_name)
            continue
        multi_name_match = MULTI_NAME_PATTERN.search(line)
        if multi_name_match is None:
            continue
        names = [canonical_net(part) for part in re.split(r"[、,，]\s*", multi_name_match.group(1)) if canonical_net(part)]
        key = tuple(names)
        if names and key not in seen_groups:
            seen_groups.add(key)
            multi_name_groups.append(names)
    return singletons, multi_name_groups


def load_net_endpoint_index(netlist_paths: list[Path]) -> dict[str, dict[str, list[str]]]:
    index: dict[str, dict[str, list[str]]] = {}
    for path in netlist_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        endpoints: dict[str, list[str]] = defaultdict(list)
        for key, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            props = dict(entry.get("props", {}))
            designator = str(props.get("Designator", key)).strip() or str(key)
            for pin_name, net_name in dict(entry.get("pins", {})).items():
                net = canonical_net(str(net_name))
                if not net:
                    continue
                endpoints[net].append(f"{designator}.{pin_name}")
        index[path.name] = {net: sorted(items) for net, items in endpoints.items()}
    return index


def diagnose_singleton(net: str, endpoint_index: dict[str, dict[str, list[str]]]) -> dict[str, object]:
    locations: dict[str, list[str]] = {}
    for label, endpoints in endpoint_index.items():
        hits = endpoints.get(net, [])
        if hits:
            locations[label] = hits
    counts = {label: len(items) for label, items in locations.items()}
    any_multi = any(count >= 2 for count in counts.values())
    if any_multi:
        diagnosis = "netlists show this net is already connected; current JLC schematic likely missed a component instance or drew the wire broken."
    elif locations:
        diagnosis = "net exists in netlists but is still single-ended; verify whether the source partition still carries a stale draft net."
    else:
        diagnosis = "net does not exist in the current netlists; this warning likely comes from an old or manually introduced schematic-only net."
    return {"net": net, "locations": locations, "diagnosis": diagnosis}


def diagnose_multi_name_group(names: list[str], endpoint_index: dict[str, dict[str, list[str]]]) -> dict[str, object]:
    details: dict[str, dict[str, list[str]]] = {}
    multi_endpoint_flags: dict[str, bool] = {}
    for name in names:
        per_netlist: dict[str, list[str]] = {}
        for label, endpoints in endpoint_index.items():
            hits = endpoints.get(name, [])
            if hits:
                per_netlist[label] = hits
        details[name] = per_netlist
        multi_endpoint_flags[name] = any(len(items) >= 2 for items in per_netlist.values())
    if any(multi_endpoint_flags.values()):
        diagnosis = "these nets are distinct in the netlists; the JLC schematic has physically merged separate wires or labels into one conductor."
    else:
        diagnosis = "these names are not resolved as distinct connected nets in the current netlists; verify whether they belong to an old schematic variant."
    return {"names": names, "details": details, "diagnosis": diagnosis}


def build_report(drc_log: Path, netlist_paths: list[Path], extra_singletons: list[str] | None = None) -> dict[str, object]:
    singleton_nets, multi_name_groups = parse_drc_log(drc_log)
    if extra_singletons:
        seen = set(singleton_nets)
        for net in extra_singletons:
            canonical = canonical_net(net)
            if canonical and canonical not in seen:
                seen.add(canonical)
                singleton_nets.append(canonical)
    endpoint_index = load_net_endpoint_index(netlist_paths)
    return {
        "drcLog": str(drc_log),
        "netlists": [str(path) for path in netlist_paths],
        "singletonCount": len(singleton_nets),
        "multiNameGroupCount": len(multi_name_groups),
        "singletons": [diagnose_singleton(net, endpoint_index) for net in singleton_nets],
        "multiNameGroups": [diagnose_multi_name_group(group, endpoint_index) for group in multi_name_groups],
    }


def print_text_report(report: dict[str, object], filter_tokens: list[str]) -> None:
    filter_tokens_normalized = [token.strip() for token in filter_tokens if token.strip()]

    def matches(text: str) -> bool:
        if not filter_tokens_normalized:
            return True
        return any(token in text for token in filter_tokens_normalized)

    print(f"DRC log: {report['drcLog']}")
    print(f"Singleton nets: {report['singletonCount']}")
    print(f"Multi-name groups: {report['multiNameGroupCount']}")

    print("\n[Singleton diagnostics]")
    for entry in report["singletons"]:
        net = str(entry["net"])
        if not matches(net):
            continue
        print(f"- {net}")
        print(f"  diagnosis: {entry['diagnosis']}")
        locations = dict(entry["locations"])
        if not locations:
            print("  netlists: missing from all current netlists")
            continue
        for label, hits in locations.items():
            print(f"  {label}: {len(hits)} -> {', '.join(hits)}")

    print("\n[Multi-name diagnostics]")
    for entry in report["multiNameGroups"]:
        names = [str(item) for item in entry["names"]]
        joined = ", ".join(names)
        if not matches(joined):
            continue
        print(f"- {joined}")
        print(f"  diagnosis: {entry['diagnosis']}")
        details = dict(entry["details"])
        for name in names:
            per_netlist = dict(details.get(name, {}))
            if not per_netlist:
                print(f"  {name}: missing from all current netlists")
                continue
            for label, hits in per_netlist.items():
                print(f"  {name} @ {label}: {len(hits)} -> {', '.join(hits)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit JLC DRC singleton/multi-name warnings against current netlists.")
    parser.add_argument("--drc-log", type=Path, required=True, help="Path to the JLC schDrcLog_*.txt file.")
    parser.add_argument("--netlist", type=Path, action="append", dest="netlists", help="Netlist JSON paths to audit. Defaults to eext + four partition netlists.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    parser.add_argument("--filter", action="append", default=[], help="Only print diagnostics whose net names contain the given token.")
    parser.add_argument("--net", action="append", default=[], help="Add one or more explicit net names to inspect even if the current DRC log file does not contain them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    netlist_paths = args.netlists or DEFAULT_NETLISTS
    report = build_report(args.drc_log, netlist_paths, extra_singletons=list(args.net))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report, list(args.filter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())