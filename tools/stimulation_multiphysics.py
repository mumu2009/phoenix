#!/usr/bin/env python3
"""多物理域硬件仿真核心。

目标：
1. 保留现有 eext JSON 被动网络仿真能力；
2. 增加机械网络等效仿真；
3. 固定为整板单体求解，不再依赖区域截取或分块近似；
4. 输出工业化报告数据与静态图形化 dashboard。

机械域采用 mobility analogy：
- 节点变量: 速度
- 力源: 电流源等效
- 速度源: 电压源等效
- 质量 M: 电容等效
- 阻尼 B: 电导等效
- 弹簧 K: 电感等效，L = 1 / K
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import webbrowser
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable

from stimulation_design_import import DesignImportError, detect_design_input_kind, import_design_input
from stimulation_library_catalog import build_template_scenario, expand_library_instances, get_library_catalog

try:
	import numpy as np
except Exception:  # pragma: no cover - fallback path is intentional
	np = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "build" / "stimulation"
GROUND_CANONICAL = "0"
SUPPORTED_DESIGNATORS = {"R", "C", "L", "D", "M", "K", "B"}
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
PREVIEW_POINT_LIMIT = 240
PCB_NO_FLYWIRE_HARD_CONSTRAINT = True
NO_FLYWIRE_KEYWORDS = ("flywire", "flying-wire", "flying wire", "jumper", "bodge", "飞线", "跳线")


class SimulationError(RuntimeError):
	pass


def has_glob_magic(text: str) -> bool:
	return any(char in text for char in "*?[]")


@dataclass(slots=True)
class Component:
	designator: str
	kind: str
	value_text: str
	value: float | None
	nodes: tuple[str, str]
	source_file: str
	raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceSpec:
	name: str
	kind: str
	domain_kind: str
	positive: str
	negative: str
	waveform: dict[str, Any]

	def value_at(self, time_s: float) -> float:
		kind = str(self.waveform.get("kind", "dc")).strip().lower()
		if kind == "dc":
			return float(self.waveform.get("value", 0.0))

		if kind == "step":
			low = float(self.waveform.get("v1", 0.0))
			high = float(self.waveform.get("v2", 0.0))
			t_step = float(self.waveform.get("t_step", 0.0))
			return low if time_s < t_step else high

		if kind == "pulse":
			low = float(self.waveform.get("low", 0.0))
			high = float(self.waveform.get("high", 1.0))
			delay = float(self.waveform.get("delay", 0.0))
			rise = max(float(self.waveform.get("rise", 1e-9)), 1e-12)
			fall = max(float(self.waveform.get("fall", 1e-9)), 1e-12)
			width = max(float(self.waveform.get("width", 1e-6)), 0.0)
			period = float(self.waveform.get("period", 0.0))
			if time_s < delay:
				return low
			phase = time_s - delay
			if period > 0.0:
				phase %= period
			if phase < rise:
				return low + (high - low) * (phase / rise)
			if phase < rise + width:
				return high
			if phase < rise + width + fall:
				ramp = (phase - rise - width) / fall
				return high - (high - low) * ramp
			return low

		if kind == "sine":
			offset = float(self.waveform.get("offset", 0.0))
			amplitude = float(self.waveform.get("amplitude", 1.0))
			frequency = float(self.waveform.get("frequency", 1.0))
			phase = float(self.waveform.get("phase_deg", 0.0)) * math.pi / 180.0
			return offset + amplitude * math.sin(2.0 * math.pi * frequency * time_s + phase)

		raise SimulationError(f"不支持的波形类型: {kind}")


@dataclass(slots=True)
class AnalysisSpec:
	kind: str
	observe: list[str]
	step: float | None = None
	stop: float | None = None
	sources: list[SourceSpec] = field(default_factory=list)
	assertions: list["AssertionSpec"] = field(default_factory=list)
	sample: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssertionSpec:
	actual: str
	equals: str | None = None
	expected: float | None = None
	tolerance: float = 0.05
	description: str = ""


@dataclass(slots=True)
class Scenario:
	name: str
	domain: str
	netlists: list[Path]
	inline_components: list[dict[str, Any]]
	library_instances: list[dict[str, Any]]
	library_summary: list[dict[str, Any]]
	seed_nets: list[str]
	hops: int
	aliases: dict[str, str]
	sources: list[SourceSpec]
	analyses: list[AnalysisSpec]
	target_runtime_seconds: float | None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedCircuit:
	components: list[Component]
	unsupported_components: list[str]
	nodes: list[str]
	node_index: dict[str, int]
	voltage_sources: list[SourceSpec]
	current_sources: list[SourceSpec]
	inductors: list[Component]
	voltage_source_index: dict[str, int]
	inductor_index: dict[str, int]

	@property
	def variable_count(self) -> int:
		return len(self.nodes) + len(self.voltage_sources) + len(self.inductors)

	def node_variable(self, net: str) -> int | None:
		return self.node_index.get(net)

	def voltage_source_variable(self, name: str) -> int | None:
		base = len(self.nodes)
		index = self.voltage_source_index.get(name.upper())
		return None if index is None else base + index

	def inductor_variable(self, name: str) -> int | None:
		base = len(self.nodes) + len(self.voltage_sources)
		index = self.inductor_index.get(name.upper())
		return None if index is None else base + index


@dataclass(slots=True)
class SolveUnitPlan:
	index: int
	label: str
	components: list[Component]
	local_nets: list[str]


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="多物理域硬件仿真平台")
	parser.add_argument("--preset", help="使用内置预设场景")
	parser.add_argument("--scenario", help="使用外部 JSON 场景文件")
	parser.add_argument("--input", help="直接导入设计输入；支持 KiCad 工程目录、EasyEDA/eext JSON 网表目录等")
	parser.add_argument("--input-kind", default="auto", help="输入类型：auto/kicad-folder/kicad-pcb/easyeda-folder/easyeda-netlist-json/scenario-json")
	parser.add_argument("--consider-line-effects", action="store_true", help="导入 KiCad/EasyEDA 时，把线路寄生 R/L/C 与 trace 热模型并入仿真")
	parser.add_argument("--list-presets", action="store_true", help="列出所有内置预设")
	parser.add_argument("--list-libraries", action="store_true", help="列出所有内置零件库/接口库模型")
	parser.add_argument("--emit-scenario-template", help="输出一个场景模板 JSON")
	parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
	parser.add_argument("--open-dashboard", action="store_true", help="完成后打开 dashboard.html")
	args = parser.parse_args(argv)

	presets = build_builtin_presets()
	if args.list_libraries:
		catalog = get_library_catalog()
		print("可用零件库/接口库:")
		for model in sorted(catalog):
			meta = catalog[model]
			print(f"- {model}: [{meta['category']}] {meta['description']} (fidelity={meta['fidelity']})")
		return 0

	if args.emit_scenario_template:
		path = Path(args.emit_scenario_template).resolve()
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("w", encoding="utf-8") as handle:
			json.dump(build_template_scenario(), handle, indent=2, ensure_ascii=False)
		print(f"已输出模板: {path}")
		return 0

	if args.list_presets:
		print("可用预设:")
		for name, definition in presets.items():
			print(f"- {name}: {definition['description']}")
		return 0

	selected_inputs = sum(bool(item) for item in (args.preset, args.scenario, args.input))
	if selected_inputs != 1:
		raise SystemExit("必须且只能提供 --preset、--scenario 或 --input 其中之一")

	if args.preset:
		if args.preset not in presets:
			raise SystemExit(f"未知预设: {args.preset}")
		raw = json.loads(json.dumps(presets[args.preset]["scenario"], ensure_ascii=False))
		base_dir = WORKSPACE_ROOT
	elif args.scenario:
		scenario_path = Path(args.scenario).resolve()
		with scenario_path.open("r", encoding="utf-8") as handle:
			raw = json.load(handle)
		base_dir = scenario_path.parent
	else:
		input_path = Path(args.input).resolve()
		try:
			raw = import_design_input(input_path, args.input_kind, args.consider_line_effects)
		except DesignImportError as exc:
			detected = ""
			try:
				detected = detect_design_input_kind(input_path)
			except Exception:
				pass
			if detected in {"gerber-zip", "gerber-folder"}:
				raise SystemExit("检测到 Gerber 输入。请改用 tools/stimulation --gerber-root <zip|folder>，并通过 --anchors-file/--probes-file 提供探针定义。") from exc
			raise SystemExit(str(exc)) from exc
		base_dir = input_path if input_path.is_dir() else input_path.parent

	scenario = load_scenario_dict(raw, base_dir)
	output_root = Path(args.out).resolve() / sanitize_path_part(scenario.name)
	output_root.mkdir(parents=True, exist_ok=True)
	report = run_scenario(scenario, output_root)
	print(report["console_summary"])

	if args.open_dashboard and report.get("dashboard_path"):
		webbrowser.open(Path(report["dashboard_path"]).as_uri())

	return 0


def build_builtin_presets() -> dict[str, dict[str, Any]]:
	return {
		"servo_stage_impulse": {
			"description": "机械网络预设：质量-弹簧-阻尼平台的脉冲驱动响应",
			"scenario": {
				"name": "servo_stage_impulse",
				"domain": "mechanical",
				"components": [
					{"designator": "M_STAGE", "kind": "mass", "value": 0.35, "nodes": ["STAGE", "FRAME"]},
					{"designator": "B_GUIDE", "kind": "damper", "value": 9.5, "nodes": ["STAGE", "FRAME"]},
					{"designator": "K_RETURN", "kind": "spring", "value": 180.0, "nodes": ["STAGE", "FRAME"]},
				],
				"seed_nets": ["STAGE", "FRAME"],
				"aliases": {},
				"sources": [
					{
						"name": "FDRIVE",
						"kind": "force",
						"positive": "STAGE",
						"negative": "FRAME",
						"waveform": {"kind": "pulse", "low": 0.0, "high": 15.0, "delay": 0.02, "rise": 0.002, "fall": 0.002, "width": 0.07, "period": 0.20},
					},
				],
				"analyses": [
					{
						"type": "tran",
						"step": 0.001,
						"stop": 0.25,
						"observe": ["STAGE", "X(STAGE)", "A(STAGE)"],
					}
				],
			},
		},
		"hybrid_controller_cell": {
			"description": "多物理域控制单元示例，包含 MCU 接口、常见 IC 接口和机械执行部件。",
			"scenario": {
				"name": "hybrid_controller_cell",
				"domain": "multiphysics",
				"libraries": [
					{
						"model": "stm32f103c8_interface",
						"name": "MCU_MAIN",
						"nets": {
							"vdd": "3V3",
							"gnd": "GND",
							"nrst": "NRST",
							"boot0": "BOOT0",
							"uart_tx": "UART_TX",
							"uart_rx": "UART_RX",
							"spi_clk": "SPI_CLK",
							"spi_mosi": "SPI_MOSI",
							"spi_miso": "SPI_MISO",
							"spi_cs": "SPI_CS",
							"i2c_scl": "I2C_SCL",
							"i2c_sda": "I2C_SDA",
							"pwm0": "PWM_MOTOR",
						},
					},
					{"model": "ldo_basic", "name": "REG_3V3", "nets": {"vin": "12V", "vout": "3V3", "gnd": "GND"}},
					{"model": "level_shifter_bidirectional", "name": "I2C_SHIFT", "nets": {"low": "I2C_SCL", "high": "I2C_SCL_5V", "vcca": "3V3", "vccb": "5V"}},
					{"model": "spi_eeprom_load", "name": "FLASH_A", "nets": {"vdd": "3V3", "gnd": "GND", "spi_clk": "SPI_CLK", "spi_mosi": "SPI_MOSI", "spi_miso": "SPI_MISO", "spi_cs": "SPI_CS"}},
					{"model": "hbridge_dc_motor_interface", "name": "DRV_A", "nets": {"vm": "12V", "gnd": "GND", "out_a": "MOTOR_A", "out_b": "MOTOR_B"}},
					{"model": "linear_actuator_stage", "name": "ACT_A", "nets": {"stage": "STAGE", "frame": "FRAME", "heat": "STAGE_HEAT", "ambient": "AMBIENT", "flux": "ACT_FLUX", "return": "MAG_RETURN"}},
					{"model": "thermal_plate", "name": "HEATSINK", "nets": {"node": "STAGE_HEAT", "ambient": "AMBIENT"}},
				],
				"sources": [
					{"name": "VIN_12V", "kind": "voltage", "positive": "12V", "negative": "GND", "waveform": {"kind": "dc", "value": 12.0}},
					{"name": "VIN_5V", "kind": "voltage", "positive": "5V", "negative": "GND", "waveform": {"kind": "dc", "value": 5.0}},
					{"name": "HEAT_LOAD", "kind": "heat", "positive": "STAGE_HEAT", "negative": "AMBIENT", "waveform": {"kind": "dc", "value": 5.5}},
				],
				"analyses": [
					{"type": "op", "observe": ["12V", "3V3", "5V", "PWM_MOTOR", "STAGE_HEAT", "ACT_FLUX"]},
					{"type": "tran", "step": 5e-4, "stop": 0.03, "observe": ["PWM_MOTOR", "STAGE", "X(STAGE)", "STAGE_HEAT"]},
				],
				"targetRuntimeSeconds": 60.0,
			},
		},
		"benchmark_3000_multiphysics": {
			"description": "约 3000 个 RLC 被动件的多区域多物理域性能基准。",
			"scenario": {
				"name": "benchmark_3000_multiphysics",
				"domain": "multiphysics",
				"libraries": [
					{"model": "passive_mesh_region", "name": "REGION_A", "nets": {"supply": "PWR_A_IN", "gnd": "GND"}, "params": {"count": 1002}},
					{"model": "passive_mesh_region", "name": "REGION_B", "nets": {"supply": "PWR_B_IN", "gnd": "GND"}, "params": {"count": 1002}},
					{"model": "passive_mesh_region", "name": "REGION_C", "nets": {"supply": "PWR_C_IN", "gnd": "GND"}, "params": {"count": 1002}},
					{"model": "solenoid_actuator", "name": "ACT_BENCH", "nets": {"coil_plus": "DRV_PLUS", "coil_minus": "DRV_MINUS", "stage": "ACT_STAGE", "frame": "FRAME", "heat": "ACT_HEAT", "ambient": "AMBIENT", "flux": "ACT_FLUX", "return": "MAG_RETURN"}},
					{"model": "thermal_plate", "name": "SINK_BENCH", "nets": {"node": "ACT_HEAT", "ambient": "AMBIENT"}},
					{"model": "stm32f407_interface", "name": "CTRL_BENCH", "nets": {"vdd": "3V3", "gnd": "GND", "nrst": "NRST", "boot0": "BOOT0", "spi_clk": "SPI_CLK", "spi_mosi": "SPI_MOSI", "spi_miso": "SPI_MISO", "spi_cs": "SPI_CS", "uart_tx": "UART_TX", "uart_rx": "UART_RX", "pwm0": "PWM_BENCH"}},
				],
				"sources": [
					{"name": "VIN_3V3", "kind": "voltage", "positive": "3V3", "negative": "GND", "waveform": {"kind": "dc", "value": 3.3}},
					{"name": "VIN_A", "kind": "voltage", "positive": "PWR_A_IN", "negative": "GND", "waveform": {"kind": "dc", "value": 3.3}},
					{"name": "VIN_B", "kind": "voltage", "positive": "PWR_B_IN", "negative": "GND", "waveform": {"kind": "dc", "value": 5.0}},
					{"name": "VIN_C", "kind": "voltage", "positive": "PWR_C_IN", "negative": "GND", "waveform": {"kind": "dc", "value": 12.0}},
					{"name": "HEAT_BENCH", "kind": "heat", "positive": "ACT_HEAT", "negative": "AMBIENT", "waveform": {"kind": "dc", "value": 6.0}},
				],
				"analyses": [
					{"type": "op", "observe": ["PWR_A_IN", "PWR_B_IN", "PWR_C_IN", "3V3", "ACT_HEAT", "ACT_FLUX"]},
				],
				"targetRuntimeSeconds": 60.0,
			},
		},
	}


def build_full_board_unit(components: list[Component]) -> SolveUnitPlan:
	local_nets = sorted({net for component in components for net in component.nodes})
	return SolveUnitPlan(index=1, label="full_board", components=list(components), local_nets=local_nets)


def assert_no_flywire_hard_constraint(scenario: Scenario) -> None:
	if not PCB_NO_FLYWIRE_HARD_CONSTRAINT:
		return
	text_candidates: list[str] = [scenario.name]
	text_candidates.extend(scenario.seed_nets)
	if scenario.metadata:
		for key, value in scenario.metadata.items():
			key_text = str(key).strip().lower()
			if key_text in {"pcbnoflywire", "no_flywire", "noflywire", "physicalrouterequired"}:
				continue
			if isinstance(value, str):
				text_candidates.append(value)
	for summary in scenario.library_summary:
		if isinstance(summary, dict):
			text_candidates.append(str(summary.get("name", "")))
			text_candidates.append(str(summary.get("description", "")))
	joined = "\n".join(item.lower() for item in text_candidates if item)
	for keyword in NO_FLYWIRE_KEYWORDS:
		if keyword in joined:
			raise SimulationError(
				"硬约束触发: PCB 场景禁止 flywire/jumper/bodge 桥接语义，请改用真实网表与物理审计路径。"
			)


def load_scenario_dict(raw: dict[str, Any], base_dir: Path) -> Scenario:
	domain = str(raw.get("domain", "electrical")).strip().lower()
	if domain not in {"electrical", "mechanical", "thermal", "magnetic", "hybrid", "multiphysics"}:
		raise SimulationError(f"不支持的 domain: {domain}")

	aliases = {normalize_net_name(key): normalize_net_name(value) for key, value in raw.get("aliases", {}).items()}
	aliases.setdefault(normalize_net_name("GND"), GROUND_CANONICAL)
	aliases.setdefault(normalize_net_name("0"), GROUND_CANONICAL)
	aliases.setdefault(normalize_net_name(GROUND_CANONICAL), GROUND_CANONICAL)

	netlists = expand_netlist_paths(raw.get("netlists", []), base_dir)
	library_instances = list(raw.get("libraries", []))
	library_payload = expand_library_instances(library_instances) if library_instances else {"components": [], "sources": [], "notes": [], "summary": []}
	inline_components = list(library_payload.get("components", [])) + list(raw.get("components", []))
	if not netlists and not inline_components:
		raise SimulationError("场景必须至少提供 netlists 或 components 其中之一")

	sources = [load_source_spec(item, aliases, domain) for item in list(library_payload.get("sources", [])) + list(raw.get("sources", []))]
	analyses: list[AnalysisSpec] = []
	for item in raw.get("analyses", []):
		analyses.append(
			AnalysisSpec(
				kind=str(item["type"]).strip().lower(),
				observe=[canonical_observation_token(token, aliases) for token in item.get("observe", [])],
				step=float(item["step"]) if "step" in item else None,
				stop=float(item["stop"]) if "stop" in item else None,
				sources=[load_source_spec(source, aliases, domain) for source in item.get("sources", [])],
				assertions=[load_assertion_spec(assertion, aliases) for assertion in item.get("assertions", [])],
				sample=dict(item.get("sample", {})),
			)
		)
	if not analyses:
		raise SimulationError("场景必须至少提供一个 analyses 条目")

	metadata = dict(raw.get("metadata", {}))
	metadata.setdefault("libraryNotes", list(library_payload.get("notes", [])))
	target_runtime_seconds = float(raw["targetRuntimeSeconds"]) if "targetRuntimeSeconds" in raw else None

	return Scenario(
		name=str(raw.get("name", "unnamed_scenario")),
		domain=domain,
		netlists=netlists,
		inline_components=inline_components,
		library_instances=library_instances,
		library_summary=list(library_payload.get("summary", [])),
		seed_nets=[canonical_net(item, aliases) for item in raw.get("seed_nets", [])],
		hops=max(int(raw.get("hops", 0)), 0),
		aliases=aliases,
		sources=sources,
		analyses=analyses,
		target_runtime_seconds=target_runtime_seconds,
		metadata=metadata,
	)


def load_source_spec(raw: dict[str, Any], aliases: dict[str, str], domain: str) -> SourceSpec:
	domain_kind = str(raw.get("kind", "voltage")).strip().lower()
	kind = normalize_source_kind(domain_kind, domain)
	return SourceSpec(
		name=str(raw["name"]),
		kind=kind,
		domain_kind=domain_kind,
		positive=canonical_net(str(raw["positive"]), aliases),
		negative=canonical_net(str(raw.get("negative", "GND")), aliases),
		waveform=dict(raw.get("waveform", {"kind": "dc", "value": 0.0})),
	)


def load_assertion_spec(raw: dict[str, Any], aliases: dict[str, str]) -> AssertionSpec:
	actual = canonical_observation_token(str(raw["actual"]), aliases)
	equals = canonical_observation_token(str(raw["equals"]), aliases) if "equals" in raw else None
	expected = float(raw["expected"]) if "expected" in raw else None
	if equals is None and expected is None:
		raise SimulationError("断言必须至少提供 equals 或 expected")
	return AssertionSpec(
		actual=actual,
		equals=equals,
		expected=expected,
		tolerance=float(raw.get("tolerance", 0.05)),
		description=str(raw.get("description", "")).strip(),
	)


def normalize_source_kind(kind: str, domain: str) -> str:
	text = kind.strip().lower()
	if text in {"voltage", "v", "velocity", "temperature", "flux-potential"}:
		return "voltage"
	if text in {"current", "i", "force", "heat", "mmf", "torque"}:
		return "current"
	raise SimulationError(f"不支持的源类型: {kind}")


def resolve_path(base_dir: Path, path: Path) -> Path:
	if path.is_absolute():
		return path
	candidate = (base_dir / path).resolve()
	if candidate.exists():
		return candidate
	return (WORKSPACE_ROOT / path).resolve()


def expand_netlist_paths(raw_netlists: Iterable[Any], base_dir: Path) -> list[Path]:
	resolved: list[Path] = []
	seen: set[Path] = set()
	for item in raw_netlists:
		text = str(item).strip()
		if not text:
			continue

		matches: list[Path] = []
		if has_glob_magic(text):
			roots = [base_dir]
			if WORKSPACE_ROOT not in roots:
				roots.append(WORKSPACE_ROOT)
			for root in roots:
				matches.extend(sorted(path.resolve() for path in root.glob(text) if path.is_file()))
			if not matches:
				raise SimulationError(f"netlist 通配符未匹配到任何文件: {text}")
		else:
			candidate = resolve_path(base_dir, Path(text))
			if candidate.is_dir():
				matches = sorted(path.resolve() for path in candidate.glob("eext_netlist*.json") if path.is_file())
				if not matches:
					raise SimulationError(f"netlist 目录中未找到 eext_netlist*.json: {candidate}")
			else:
				matches = [candidate]

		for match in matches:
			if match not in seen:
				seen.add(match)
				resolved.append(match)
	return resolved


def run_scenario(scenario: Scenario, output_root: Path) -> dict[str, Any]:
	started_at = time.perf_counter()
	assert_no_flywire_hard_constraint(scenario)
	components, unsupported = load_domain_components(scenario)
	selected = list(components)
	solve_units = [build_full_board_unit(selected)]
	warnings: list[str] = []
	if scenario.seed_nets or scenario.hops > 0:
		warnings.append("当前版本强制整板求解；seed_nets/hops 仅保留在场景记录中，不再用于截取局部子网。")
	if scenario.library_summary:
		warnings.extend(scenario.metadata.get("libraryNotes", []))
	if scenario.metadata.get("importNotes"):
		warnings.extend(str(item) for item in scenario.metadata.get("importNotes", []))
	if PCB_NO_FLYWIRE_HARD_CONSTRAINT and scenario.domain in {"electrical", "hybrid", "multiphysics"}:
		warnings.append("硬约束: PCB 板级仿真禁止飞线/跳线桥接假设；任何跨网络连通都必须来自网表与物理审计证据。")

	resolved_scenario_path = output_root / "scenario_resolved.json"
	with resolved_scenario_path.open("w", encoding="utf-8") as handle:
		json.dump(serialize_scenario(scenario), handle, indent=2, ensure_ascii=False)

	selection_path = output_root / "selected_components.json"
	with selection_path.open("w", encoding="utf-8") as handle:
		json.dump([serialize_component(item) for item in selected], handle, indent=2, ensure_ascii=False)

	solve_unit_results = execute_solve_units(scenario, solve_units, output_root)
	solve_unit_summary = build_solve_unit_summary(solve_unit_results)
	circuit_summary = build_circuit_summary(selected, unsupported)
	total_elapsed = time.perf_counter() - started_at
	performance_summary = build_performance_summary(scenario, circuit_summary, solve_unit_summary, total_elapsed)

	markdown_sections = render_report_markdown(
		scenario,
		circuit_summary,
		solve_unit_summary,
		performance_summary,
		warnings,
		solve_unit_results,
		total_elapsed,
	)
	summary_path = output_root / "summary.md"
	with summary_path.open("w", encoding="utf-8") as handle:
		handle.write("\n".join(markdown_sections).rstrip() + "\n")

	report_payload = {
		"generatedAtUtc": datetime.now(timezone.utc).isoformat(),
		"scenario": serialize_scenario(scenario),
		"constraints": {
			"pcbNoFlywire": PCB_NO_FLYWIRE_HARD_CONSTRAINT,
			"fullBoardOnly": True,
		},
		"summaryMarkdown": str(summary_path),
		"solver": {
			"backend": solver_backend_name(),
			"fullBoardOnly": True,
			"totalElapsedSeconds": total_elapsed,
		},
		"circuitSummary": circuit_summary,
		"solveUnitSummary": solve_unit_summary,
		"performance": performance_summary,
		"libraries": scenario.library_summary,
		"warnings": warnings,
		"solveUnits": solve_unit_results,
		"summaries": solve_unit_results[0]["analyses"] if len(solve_unit_results) == 1 else [],
	}

	report_path = output_root / "report.json"
	with report_path.open("w", encoding="utf-8") as handle:
		json.dump(report_payload, handle, indent=2, ensure_ascii=False)

	dashboard_path = output_root / "dashboard.html"
	with dashboard_path.open("w", encoding="utf-8") as handle:
		handle.write(build_dashboard_html(report_payload))

	console_summary = (
		f"仿真完成: {scenario.name}\n"
		f"输出目录: {output_root}\n"
		f"域: {scenario.domain}, 可仿真器件: {circuit_summary['selectedComponentCount']}, 求解单元数: {solve_unit_summary['solveUnitCount']}\n"
		f"报告: {summary_path}\n"
		f"图形化 dashboard: {dashboard_path}"
	)
	return {
		"console_summary": console_summary,
		"summary_path": str(summary_path),
		"report_path": str(report_path),
		"dashboard_path": str(dashboard_path),
	}
def load_domain_components(scenario: Scenario) -> tuple[list[Component], list[str]]:
	components: list[Component] = []
	unsupported: list[str] = []
	if scenario.netlists:
		netlist_components, netlist_unsupported = load_netlists(scenario.netlists, scenario.aliases)
		components.extend(netlist_components)
		unsupported.extend(netlist_unsupported)
	if scenario.inline_components:
		inline_components, inline_unsupported = load_inline_components(scenario.inline_components, scenario.aliases, scenario.domain)
		components.extend(inline_components)
		unsupported.extend(inline_unsupported)
	return components, unsupported


def load_netlists(paths: list[Path], aliases: dict[str, str]) -> tuple[list[Component], list[str]]:
	components: list[Component] = []
	unsupported: list[str] = []

	for path in paths:
		with path.open("r", encoding="utf-8") as handle:
			raw = json.load(handle)

		for component_id, entry in raw.items():
			props = entry.get("props", {})
			designator = str(props.get("Designator", component_id)).strip().upper()
			if not designator:
				unsupported.append(component_id)
				continue

			kind = designator[0]
			pin_items = sorted_pin_items(entry.get("pins", {}).items())
			if kind not in {"R", "C", "L", "D"} or len(pin_items) < 2:
				unsupported.append(designator)
				continue

			node_a = canonical_net(str(pin_items[0][1]), aliases)
			node_b = canonical_net(str(pin_items[1][1]), aliases)
			value_text = str(props.get("value", "")).strip()
			value = parse_component_value(value_text, kind)
			if kind in {"R", "C", "L"} and (value is None or value <= 0.0):
				unsupported.append(designator)
				continue

			components.append(
				Component(
					designator=designator,
					kind=kind,
					value_text=value_text,
					value=value,
					nodes=(node_a, node_b),
					source_file=str(path.relative_to(WORKSPACE_ROOT)),
					raw=entry,
				)
			)

	return components, unsupported


def load_inline_components(raw_components: list[dict[str, Any]], aliases: dict[str, str], domain: str) -> tuple[list[Component], list[str]]:
	components: list[Component] = []
	unsupported: list[str] = []
	for index, entry in enumerate(raw_components, start=1):
		kind = normalize_component_kind(str(entry.get("kind", "")).strip(), domain)
		designator = str(entry.get("designator") or entry.get("name") or f"{kind}{index}").strip().upper()
		nodes = entry.get("nodes")
		if not isinstance(nodes, list) or len(nodes) < 2:
			unsupported.append(designator)
			continue
		value = entry.get("value")
		if isinstance(value, (int, float)):
			parsed_value = float(value)
			value_text = str(value)
		else:
			value_text = str(value or "")
			parsed_value = parse_component_value(value_text, kind)
		if kind != "D" and (parsed_value is None or parsed_value <= 0.0):
			unsupported.append(designator)
			continue
		components.append(
			Component(
				designator=designator,
				kind=kind,
				value_text=value_text,
				value=parsed_value,
				nodes=(canonical_net(str(nodes[0]), aliases), canonical_net(str(nodes[1]), aliases)),
				source_file="inline",
				raw=dict(entry),
			)
		)
	return components, unsupported


def normalize_component_kind(text: str, domain: str) -> str:
	value = text.strip().lower()
	mapping = {
		"r": "R",
		"resistor": "R",
		"c": "C",
		"capacitor": "C",
		"l": "L",
		"inductor": "L",
		"d": "D",
		"diode": "D",
		"m": "M",
		"mass": "M",
		"k": "K",
		"spring": "K",
		"b": "B",
		"damper": "B",
		"damping": "B",
		"thermal_resistor": "R",
		"thermal_capacitor": "C",
		"thermal_mass": "C",
		"magnetic_reluctance": "R",
		"reluctance": "R",
		"magnetic_storage": "L",
		"magnetic_inductor": "L",
		"magnetic_loss": "B",
	}
	if value in mapping:
		kind = mapping[value]
		if domain == "electrical" and kind not in {"R", "C", "L", "D"}:
			raise SimulationError(f"电路场景不支持的器件类型: {text}")
		return kind
	raise SimulationError(f"不支持的器件类型: {text}")
def execute_solve_units(scenario: Scenario, solve_units: list[SolveUnitPlan], output_root: Path) -> list[dict[str, Any]]:
	results = [run_solve_unit_analyses(scenario, unit, output_root) for unit in solve_units]
	return sorted(results, key=lambda item: item["index"])


def run_solve_unit_analyses(scenario: Scenario, solve_unit: SolveUnitPlan, output_root: Path) -> dict[str, Any]:
	solve_root = output_root / "solve_units" / solve_unit.label
	solve_root.mkdir(parents=True, exist_ok=True)
	with (solve_root / "components.json").open("w", encoding="utf-8") as handle:
		json.dump([serialize_component(item) for item in solve_unit.components], handle, indent=2, ensure_ascii=False)

	local_nets = set(solve_unit.local_nets)
	base_sources = [item for item in scenario.sources if item.positive in local_nets or item.negative in local_nets]
	preview_circuit = prepare_circuit(solve_unit.components, [], base_sources)
	op_tolerance = 1e-8
	analyses_payload: list[dict[str, Any]] = []
	markdown_lines = [
		f"## {solve_unit.label}",
		"",
		f"- 器件数：{len(solve_unit.components)}",
		f"- 节点数：{len(preview_circuit.nodes)}",
		f"- 变量数：{preview_circuit.variable_count}",
		"- 边界网络数：0",
		"",
	]

	previous_solution: list[float] | None = None
	solve_started = time.perf_counter()
	for analysis_index, analysis in enumerate(scenario.analyses, start=1):
		analysis_started = time.perf_counter()
		analysis_sources = base_sources + [item for item in analysis.sources if item.positive in local_nets or item.negative in local_nets]
		circuit = prepare_circuit(solve_unit.components, [], analysis_sources)
		if previous_solution is not None and len(previous_solution) != circuit.variable_count:
			previous_solution = None
		observe_tokens = merge_analysis_observe_tokens(analysis.observe, analysis.assertions)
		if analysis.kind == "op":
			result = solve_operating_point(circuit, time_s=0.0, initial_guess=previous_solution, tolerance=op_tolerance)
			previous_solution = result
			summary = build_operating_point_summary(circuit, observe_tokens, result)
			assertions = evaluate_analysis_assertions(circuit, result, analysis.assertions, summary)
			elapsed = time.perf_counter() - analysis_started
			payload: dict[str, Any] = {
				"type": "op",
				"observe": list(observe_tokens),
				"summary": summary,
				"elapsedSeconds": elapsed,
			}
			if analysis.sources:
				payload["stimuli"] = [serialize_source_spec(source) for source in analysis.sources]
			if analysis.assertions:
				payload["assertions"] = assertions
				payload["assertionSummary"] = summarize_assertions(assertions)
			if analysis.sample:
				payload["sample"] = dict(analysis.sample)
			analyses_payload.append(payload)
			markdown_lines.extend(render_operating_point_markdown(summary, elapsed))
			continue

		if analysis.kind == "tran":
			if analysis.step is None or analysis.stop is None:
				raise SimulationError("瞬态分析必须提供 step 和 stop")
			transient = run_transient(circuit, analysis, previous_solution, op_tolerance=op_tolerance)
			previous_solution = transient["last_solution"]
			csv_path = solve_root / f"tran_{analysis_index:02d}.csv"
			write_transient_csv(csv_path, transient["rows"])
			summary = summarize_transient_rows(transient["rows"], observe_tokens)
			preview = build_series_preview(transient["rows"], observe_tokens)
			elapsed = time.perf_counter() - analysis_started
			payload = {
				"type": "tran",
				"observe": list(observe_tokens),
				"summary": summary,
				"csv": str(csv_path),
				"previewRows": preview,
				"elapsedSeconds": elapsed,
			}
			if analysis.sources:
				payload["stimuli"] = [serialize_source_spec(source) for source in analysis.sources]
			if analysis.sample:
				payload["sample"] = dict(analysis.sample)
			analyses_payload.append(payload)
			markdown_lines.extend(render_transient_markdown(summary, csv_path, elapsed))
			continue

		raise SimulationError(f"不支持的分析类型: {analysis.kind}")

	return {
		"index": solve_unit.index,
		"label": solve_unit.label,
		"outputRoot": str(solve_root),
		"componentCount": len(solve_unit.components),
		"nodeCount": len(circuit.nodes),
		"variableCount": circuit.variable_count,
		"boundaryNets": [],
		"componentKinds": dict(sorted(Counter(item.kind for item in solve_unit.components).items())),
		"elapsedSeconds": time.perf_counter() - solve_started,
		"analyses": analyses_payload,
		"markdown": markdown_lines,
	}


def build_performance_summary(scenario: Scenario, circuit_summary: dict[str, Any], solve_unit_summary: dict[str, Any], total_elapsed: float) -> dict[str, Any]:
	passive_count = sum(int(circuit_summary["componentKinds"].get(kind, 0)) for kind in ("R", "C", "L"))
	present_domains = []
	if any(kind in circuit_summary["componentKinds"] for kind in ("R", "C", "L", "D")):
		present_domains.append("electrical")
	if any(kind in circuit_summary["componentKinds"] for kind in ("M", "K", "B")):
		present_domains.append("mechanical")
	if any("thermal" in str(item.get("description", "")).lower() for item in scenario.library_summary):
		present_domains.append("thermal")
	if any("magnetic" in str(item.get("description", "")).lower() for item in scenario.library_summary):
		present_domains.append("magnetic")
	return {
		"targetRuntimeSeconds": scenario.target_runtime_seconds,
		"metRuntimeTarget": scenario.target_runtime_seconds is None or total_elapsed <= scenario.target_runtime_seconds,
		"elapsedSeconds": total_elapsed,
		"passiveEquivalentCount": passive_count,
		"componentsPerSecond": 0.0 if total_elapsed <= 0.0 else circuit_summary["selectedComponentCount"] / total_elapsed,
		"largestSolveUnitComponents": solve_unit_summary["largestSolveUnitComponents"],
		"presentDomains": present_domains,
		"lineModelCount": len(scenario.metadata.get("lineEffects", [])) if isinstance(scenario.metadata.get("lineEffects"), list) else 0,
	}


def build_solve_unit_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
	component_counts = [item["componentCount"] for item in results]
	node_counts = [item["nodeCount"] for item in results]
	return {
		"solveUnitCount": len(results),
		"largestSolveUnitComponents": max(component_counts) if component_counts else 0,
		"largestSolveUnitNodes": max(node_counts) if node_counts else 0,
		"averageSolveUnitComponents": (sum(component_counts) / len(component_counts)) if component_counts else 0.0,
		"averageSolveUnitNodes": (sum(node_counts) / len(node_counts)) if node_counts else 0.0,
		"boundaryNetCount": 0,
	}


def build_circuit_summary(selected: list[Component], unsupported: list[str]) -> dict[str, Any]:
	component_kinds = Counter(component.kind for component in selected)
	nets = {net for component in selected for net in component.nodes}
	return {
		"selectedComponentCount": len(selected),
		"ignoredComponentCount": len(set(unsupported)),
		"netCount": len(nets),
		"componentKinds": dict(sorted(component_kinds.items())),
		"ignoredPreview": sorted(set(unsupported))[:80],
	}


def prepare_circuit(components: list[Component], unsupported: list[str], sources: list[SourceSpec]) -> PreparedCircuit:
	voltage_sources = [item for item in sources if item.kind == "voltage"]
	current_sources = [item for item in sources if item.kind == "current"]
	if len(voltage_sources) + len(current_sources) != len(sources):
		bad = [item.name for item in sources if item.kind not in {"voltage", "current"}]
		raise SimulationError(f"存在不支持的源类型: {', '.join(bad)}")

	nets: set[str] = set()
	inductors: list[Component] = []
	for component in components:
		nets.update(net for net in component.nodes if net != GROUND_CANONICAL)
		if component.kind in {"L", "K"}:
			inductors.append(component)

	for source in sources:
		if source.positive != GROUND_CANONICAL:
			nets.add(source.positive)
		if source.negative != GROUND_CANONICAL:
			nets.add(source.negative)

	nodes = sorted(nets)
	node_index = {net: idx for idx, net in enumerate(nodes)}
	voltage_source_index = {item.name.upper(): idx for idx, item in enumerate(voltage_sources)}
	inductor_index = {item.designator.upper(): idx for idx, item in enumerate(inductors)}

	return PreparedCircuit(
		components=components,
		unsupported_components=sorted(set(unsupported)),
		nodes=nodes,
		node_index=node_index,
		voltage_sources=voltage_sources,
		current_sources=current_sources,
		inductors=inductors,
		voltage_source_index=voltage_source_index,
		inductor_index=inductor_index,
	)


def solve_operating_point(
	circuit: PreparedCircuit,
	time_s: float,
	initial_guess: list[float] | None,
	max_iterations: int = 80,
	tolerance: float = 1e-9,
) -> list[float]:
	if circuit.variable_count == 0:
		return []

	guess = list(initial_guess) if initial_guess is not None else [0.0] * circuit.variable_count
	adaptive_relaxation = any(component.kind == "D" or bool(component_physics(component)) for component in circuit.components)
	previous_raw_delta = float("inf")
	best_guess = list(guess)
	best_delta = float("inf")
	best_threshold = tolerance
	for _ in range(max_iterations):
		matrix, rhs = assemble_system(circuit, mode="op", time_s=time_s, step_s=None, guess=guess, previous_solution=guess)
		solution = solve_linear_system(matrix, rhs)
		raw_delta = max(abs(solution[index] - guess[index]) for index in range(len(solution)))
		solution_scale = max(max((abs(value) for value in solution), default=1.0), 1.0)
		convergence_threshold = tolerance
		if adaptive_relaxation:
			convergence_threshold = max(tolerance, 1e-8)
		if adaptive_relaxation and circuit.variable_count >= 64:
			convergence_threshold = max(convergence_threshold, solution_scale * 1e-4)

		next_guess = solution
		delta = raw_delta
		if adaptive_relaxation:
			relaxation = 0.35 if raw_delta > previous_raw_delta * 1.05 else 0.65
			next_guess = [guess[index] + relaxation * (solution[index] - guess[index]) for index in range(len(solution))]
			delta = max(abs(next_guess[index] - guess[index]) for index in range(len(next_guess)))
			previous_raw_delta = raw_delta

		if delta < best_delta:
			best_delta = delta
			best_guess = list(next_guess)
			best_threshold = convergence_threshold

		guess = next_guess
		if delta < tolerance:
			return guess
		if convergence_threshold > tolerance and delta < convergence_threshold:
			return guess
	if adaptive_relaxation and best_delta <= best_threshold * 1.25:
		return best_guess
	raise SimulationError("工作点求解未收敛")


def run_transient(circuit: PreparedCircuit, analysis: AnalysisSpec, initial_solution: list[float] | None, op_tolerance: float = 1e-9) -> dict[str, Any]:
	step_s = float(analysis.step)
	stop_s = float(analysis.stop)
	if step_s <= 0.0:
		raise SimulationError("瞬态分析要求 step > 0")
	if stop_s < 0.0:
		raise SimulationError("瞬态分析要求 stop >= 0")
	base_observe = expand_base_observation_tokens(analysis.observe)
	current_solution = solve_operating_point(circuit, time_s=0.0, initial_guess=initial_solution, tolerance=op_tolerance)

	rows = [capture_observations(circuit, base_observe, current_solution, 0.0)]
	time_s = 0.0
	step_index = 0
	total_steps = max(int(round(stop_s / step_s)), 0)
	while time_s + step_s <= stop_s + step_s * 0.5:
		time_s += step_s
		step_index += 1
		guess = list(current_solution)
		for _ in range(50):
			matrix, rhs = assemble_system(
				circuit,
				mode="tran",
				time_s=time_s,
				step_s=step_s,
				guess=guess,
				previous_solution=current_solution,
			)
			next_solution = solve_linear_system(matrix, rhs)
			delta = max(abs(next_solution[index] - guess[index]) for index in range(len(next_solution))) if next_solution else 0.0
			guess = next_solution
			if delta < 1e-8:
				break
		else:
			raise SimulationError(f"瞬态分析在 t={time_s:.6e}s 未收敛")
		current_solution = guess
		rows.append(capture_observations(circuit, base_observe, current_solution, time_s))
		if step_index % 25 == 0 or step_index == total_steps:
			print(f"[tran] progress {step_index}/{total_steps} t={time_s:.6e}s", flush=True)

	rows = enrich_derived_observations(rows, analysis.observe)
	return {"rows": rows, "last_solution": current_solution}


def assemble_system(
	circuit: PreparedCircuit,
	mode: str,
	time_s: float,
	step_s: float | None,
	guess: list[float],
	previous_solution: list[float] | None,
) -> tuple[Any, Any]:
	size = circuit.variable_count
	if np is not None:
		matrix = np.zeros((size, size), dtype=np.float64)
		rhs = np.zeros(size, dtype=np.float64)
	else:
		matrix = [[0.0 for _ in range(size)] for _ in range(size)]
		rhs = [0.0 for _ in range(size)]

	for node_index in range(len(circuit.nodes)):
		matrix[node_index][node_index] += 1e-12

	for component in circuit.components:
		component_value = component_effective_value(circuit, component, guess)
		if component.kind in {"R", "B"}:
			assert component_value is not None
			conductance = (1.0 / float(component_value)) if component.kind == "R" else float(component_value)
			stamp_conductance(matrix, circuit, component.nodes[0], component.nodes[1], conductance)
			continue

		if component.kind in {"C", "M"}:
			if mode == "tran":
				assert step_s is not None
				assert component_value is not None
				conductance = float(component_value) / step_s
				stamp_conductance(matrix, circuit, component.nodes[0], component.nodes[1], conductance)
				previous_voltage = voltage_between_nodes(circuit, previous_solution, component.nodes[0], component.nodes[1])
				stamp_history_current(rhs, circuit, component.nodes[0], component.nodes[1], conductance * previous_voltage)
			continue

		if component.kind in {"L", "K"}:
			inductor_variable = circuit.inductor_variable(component.designator)
			if inductor_variable is None:
				raise SimulationError(f"缺少电感支路变量: {component.designator}")
			stamp_branch_link(matrix, circuit, component.nodes[0], component.nodes[1], inductor_variable)
			branch_row = inductor_variable
			if mode == "op":
				continue
			assert step_s is not None
			assert component_value is not None
			dynamic_value = float(component_value) if component.kind == "L" else (1.0 / float(component_value))
			matrix[branch_row][branch_row] += -(dynamic_value / step_s)
			previous_current = 0.0 if previous_solution is None else previous_solution[inductor_variable]
			rhs[branch_row] += -(dynamic_value / step_s) * previous_current
			continue

		if component.kind == "D":
			stamp_diode(matrix, rhs, circuit, component, guess)
			continue

	for component in circuit.components:
		stamp_component_couplings(rhs, circuit, component, guess)

	for source in circuit.current_sources:
		stamp_current_source(rhs, circuit, source.positive, source.negative, source.value_at(time_s))

	for index, source in enumerate(circuit.voltage_sources):
		branch_variable = len(circuit.nodes) + index
		stamp_branch_link(matrix, circuit, source.positive, source.negative, branch_variable)
		rhs[branch_variable] += source.value_at(time_s)

	return matrix, rhs


def stamp_diode(matrix: list[list[float]], rhs: list[float], circuit: PreparedCircuit, component: Component, guess: list[float]) -> None:
	isat = 1e-12
	emission = 1.8
	physics = component_physics(component)
	temp_delta = resolve_component_temperature_delta(circuit, guess, component)
	junction_tc = float(physics.get("junctionTempCoefficientPerC", 0.0))
	if junction_tc:
		emission = max(1.1, emission * (1.0 + junction_tc * temp_delta))
	thermal = 0.025852 * emission
	voltage = voltage_between_nodes(circuit, guess, component.nodes[0], component.nodes[1])
	exponent = clamp(voltage / thermal, -40.0, 40.0)
	exp_value = math.exp(exponent)
	conductance = max(isat * exp_value / thermal, 1e-12)
	current = isat * (exp_value - 1.0)
	ieq = current - conductance * voltage
	stamp_conductance(matrix, circuit, component.nodes[0], component.nodes[1], conductance)
	stamp_current_source(rhs, circuit, component.nodes[0], component.nodes[1], ieq)


def stamp_conductance(matrix: list[list[float]], circuit: PreparedCircuit, node_a: str, node_b: str, conductance: float) -> None:
	index_a = circuit.node_variable(node_a)
	index_b = circuit.node_variable(node_b)
	if index_a is not None:
		matrix[index_a][index_a] += conductance
	if index_b is not None:
		matrix[index_b][index_b] += conductance
	if index_a is not None and index_b is not None:
		matrix[index_a][index_b] -= conductance
		matrix[index_b][index_a] -= conductance


def stamp_current_source(rhs: list[float], circuit: PreparedCircuit, positive: str, negative: str, current: float) -> None:
	index_positive = circuit.node_variable(positive)
	index_negative = circuit.node_variable(negative)
	if index_positive is not None:
		rhs[index_positive] -= current
	if index_negative is not None:
		rhs[index_negative] += current


def stamp_history_current(rhs: list[float], circuit: PreparedCircuit, positive: str, negative: str, current: float) -> None:
	index_positive = circuit.node_variable(positive)
	index_negative = circuit.node_variable(negative)
	if index_positive is not None:
		rhs[index_positive] += current
	if index_negative is not None:
		rhs[index_negative] -= current


def stamp_branch_link(matrix: list[list[float]], circuit: PreparedCircuit, positive: str, negative: str, branch_variable: int) -> None:
	index_positive = circuit.node_variable(positive)
	index_negative = circuit.node_variable(negative)
	if index_positive is not None:
		matrix[index_positive][branch_variable] += 1.0
		matrix[branch_variable][index_positive] += 1.0
	if index_negative is not None:
		matrix[index_negative][branch_variable] -= 1.0
		matrix[branch_variable][index_negative] -= 1.0


def solve_linear_system(matrix: Any, rhs: Any) -> list[float]:
	if len(rhs) == 0:
		return []
	if np is not None:
		if isinstance(matrix, np.ndarray):
			array_matrix = matrix
		else:
			array_matrix = np.asarray(matrix, dtype=np.float64)
		if isinstance(rhs, np.ndarray):
			array_rhs = rhs
		else:
			array_rhs = np.asarray(rhs, dtype=np.float64)
		try:
			return np.linalg.solve(array_matrix, array_rhs).tolist()
		except np.linalg.LinAlgError as exc:
			try:
				solution, _, _, _ = np.linalg.lstsq(array_matrix, array_rhs, rcond=None)
				return solution.tolist()
			except np.linalg.LinAlgError as fallback_exc:
				raise SimulationError(f"矩阵奇异，可能存在未激励的浮空节点: {fallback_exc}") from fallback_exc

	size = len(rhs)
	augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
	for pivot_index in range(size):
		pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
		pivot_value = augmented[pivot_row][pivot_index]
		if abs(pivot_value) < 1e-15:
			raise SimulationError("矩阵奇异，可能存在未激励的浮空节点")
		if pivot_row != pivot_index:
			augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
		pivot_value = augmented[pivot_index][pivot_index]
		for column in range(pivot_index, size + 1):
			augmented[pivot_index][column] /= pivot_value
		for row_index in range(size):
			if row_index == pivot_index:
				continue
			factor = augmented[row_index][pivot_index]
			if abs(factor) < 1e-18:
				continue
			for column in range(pivot_index, size + 1):
				augmented[row_index][column] -= factor * augmented[pivot_index][column]
	return [augmented[index][size] for index in range(size)]


def component_physics(component: Component) -> dict[str, Any]:
	physics = component.raw.get("physics", {}) if isinstance(component.raw, dict) else {}
	return physics if isinstance(physics, dict) else {}


def component_effective_value(circuit: PreparedCircuit, component: Component, guess: list[float]) -> float | None:
	if component.value is None:
		return None
	value = float(component.value)
	physics = component_physics(component)
	temp_delta = resolve_component_temperature_delta(circuit, guess, component)
	tempco = float(physics.get("temperatureCoefficientPerC", physics.get("tempco", 0.0)))
	if tempco and component.kind in {"R", "C", "L", "K"}:
		value *= clamp(1.0 + tempco * temp_delta, 0.05, 50.0)
	if component.kind in {"L", "K"}:
		saturation_current = float(physics.get("saturationCurrent", 0.0))
		if saturation_current > 0.0:
			branch_current = abs(resolve_component_current(circuit, guess, component))
			value /= 1.0 + (branch_current / saturation_current)
	return max(value, 1e-15)


def resolve_component_temperature_delta(circuit: PreparedCircuit, solution: list[float] | None, component: Component) -> float:
	if solution is None:
		return 0.0
	physics = component_physics(component)
	thermal = physics.get("thermal")
	if not isinstance(thermal, dict):
		return 0.0
	positive = str(thermal.get("positive") or thermal.get("node") or "")
	negative = str(thermal.get("negative") or thermal.get("ambient") or GROUND_CANONICAL)
	if not positive:
		return 0.0
	return voltage_between_nodes(circuit, solution, positive, negative)


def resolve_component_current(circuit: PreparedCircuit, solution: list[float] | None, component: Component) -> float:
	if solution is None:
		return 0.0
	voltage = voltage_between_nodes(circuit, solution, component.nodes[0], component.nodes[1])
	if component.kind == "R" and component.value:
		return voltage / float(component.value)
	if component.kind == "B" and component.value:
		return float(component.value) * voltage
	if component.kind in {"L", "K"}:
		variable = circuit.inductor_variable(component.designator)
		return 0.0 if variable is None else solution[variable]
	if component.kind == "D":
		isat = 1e-12
		emission = 1.8
		thermal = 0.025852 * emission
		exponent = clamp(voltage / thermal, -40.0, 40.0)
		return isat * (math.exp(exponent) - 1.0)
	return 0.0


def stamp_component_couplings(rhs: list[float], circuit: PreparedCircuit, component: Component, guess: list[float]) -> None:
	physics = component_physics(component)
	thermal = physics.get("thermal")
	if isinstance(thermal, dict):
		positive = str(thermal.get("positive") or thermal.get("node") or "")
		negative = str(thermal.get("negative") or thermal.get("ambient") or GROUND_CANONICAL)
		if positive:
			power = resolve_component_loss_power(circuit, guess, component)
			power *= float(thermal.get("powerScale", 1.0))
			if power > 0.0:
				stamp_current_source(rhs, circuit, positive, negative, power)


def resolve_component_loss_power(circuit: PreparedCircuit, guess: list[float], component: Component) -> float:
	current = resolve_component_current(circuit, guess, component)
	voltage = voltage_between_nodes(circuit, guess, component.nodes[0], component.nodes[1])
	physics = component_physics(component)
	if component.kind == "R" and component.value:
		return current * current * float(component.value)
	if component.kind == "B" and component.value:
		return float(component.value) * voltage * voltage
	if component.kind in {"L", "K"}:
		loss_resistance = float(physics.get("lossResistance", 0.0))
		magnetic_power = current * current * max(loss_resistance, 0.0)
		return magnetic_power + abs(voltage * current) * float(physics.get("magneticLossFactor", 0.0))
	if component.kind == "D":
		return abs(voltage * current)
	return abs(voltage * current) * float(physics.get("lossFactor", 0.0))


def build_operating_point_summary(circuit: PreparedCircuit, observe: list[str], solution: list[float]) -> dict[str, float]:
	summary: dict[str, float] = {}
	for token in observe:
		if token.startswith("X(") or token.startswith("A("):
			summary[token] = 0.0
			continue
		summary[token] = safe_resolve_observation(circuit, solution, token)
	return summary


def merge_analysis_observe_tokens(observe: list[str], assertions: list[AssertionSpec]) -> list[str]:
	merged: list[str] = []
	seen: set[str] = set()
	for token in observe:
		if token not in seen:
			seen.add(token)
			merged.append(token)
	for assertion in assertions:
		for token in (assertion.actual, assertion.equals):
			if token and token not in seen:
				seen.add(token)
				merged.append(token)
	return merged


def serialize_source_spec(source: SourceSpec) -> dict[str, Any]:
	return {
		"name": source.name,
		"kind": source.domain_kind,
		"positive": source.positive,
		"negative": source.negative,
		"waveform": dict(source.waveform),
	}


def evaluate_analysis_assertions(
	circuit: PreparedCircuit,
	solution: list[float],
	assertions: list[AssertionSpec],
	summary: dict[str, float],
) -> list[dict[str, Any]]:
	results: list[dict[str, Any]] = []
	for assertion in assertions:
		actual_value = summary.get(assertion.actual, safe_resolve_observation(circuit, solution, assertion.actual))
		if assertion.equals is not None:
			expected_value = summary.get(assertion.equals, safe_resolve_observation(circuit, solution, assertion.equals))
			expected_payload: dict[str, Any] = {"expectedToken": assertion.equals}
		else:
			expected_value = float(assertion.expected if assertion.expected is not None else float("nan"))
			expected_payload = {"expectedValue": expected_value}
		finite = math.isfinite(actual_value) and math.isfinite(expected_value)
		delta = actual_value - expected_value if finite else float("nan")
		passed = finite and abs(delta) <= assertion.tolerance
		if not math.isfinite(actual_value):
			reason = "actual-not-finite"
		elif not math.isfinite(expected_value):
			reason = "expected-not-finite"
		elif passed:
			reason = "ok"
		else:
			reason = "delta-exceeds-tolerance"
		result = {
			"actualToken": assertion.actual,
			"actualValue": actual_value,
			"tolerance": assertion.tolerance,
			"passed": passed,
			"delta": delta,
			"reason": reason,
		}
		if assertion.description:
			result["description"] = assertion.description
		result.update(expected_payload)
		if math.isfinite(expected_value):
			result["expectedResolvedValue"] = expected_value
		results.append(result)
	return results


def summarize_assertions(assertions: list[dict[str, Any]]) -> dict[str, Any]:
	passed = sum(1 for item in assertions if item.get("passed"))
	failed = len(assertions) - passed
	return {"count": len(assertions), "passed": passed, "failed": failed, "allPassed": failed == 0}


def capture_observations(circuit: PreparedCircuit, observe: list[str], solution: list[float], time_s: float) -> dict[str, float]:
	row = {"time_s": time_s}
	for token in observe:
		row[token] = safe_resolve_observation(circuit, solution, token)
	return row


def safe_resolve_observation(circuit: PreparedCircuit, solution: list[float], token: str) -> float:
	try:
		return resolve_observation(circuit, solution, token)
	except SimulationError:
		return float("nan")


def resolve_observation(circuit: PreparedCircuit, solution: list[float], token: str) -> float:
	if token.startswith("I(") and token.endswith(")"):
		name = token[2:-1].strip().upper()
		source_variable = circuit.voltage_source_variable(name)
		if source_variable is not None:
			return solution[source_variable]
		inductor_variable = circuit.inductor_variable(name)
		if inductor_variable is not None:
			return solution[inductor_variable]
		raise SimulationError(f"无法解析支路电流观察点: {token}")

	node = circuit.node_variable(token)
	if node is None:
		if token == GROUND_CANONICAL:
			return 0.0
		raise SimulationError(f"无法解析节点观察点: {token}")
	return solution[node]


def expand_base_observation_tokens(observe: list[str]) -> list[str]:
	result: list[str] = []
	seen: set[str] = set()
	for token in observe:
		base = base_observation_token(token)
		if base not in seen:
			seen.add(base)
			result.append(base)
	return result


def base_observation_token(token: str) -> str:
	if token.startswith("X(") and token.endswith(")"):
		return token[2:-1].strip()
	if token.startswith("A(") and token.endswith(")"):
		return token[2:-1].strip()
	return token


def enrich_derived_observations(rows: list[dict[str, float]], observe: list[str]) -> list[dict[str, float]]:
	if not rows:
		return rows
	for token in observe:
		if token.startswith("X(") and token.endswith(")"):
			base = token[2:-1].strip()
			rows[0][token] = 0.0
			for index in range(1, len(rows)):
				previous = rows[index - 1]
				current = rows[index]
				dt = current["time_s"] - previous["time_s"]
				current[token] = previous[token] + 0.5 * (previous.get(base, 0.0) + current.get(base, 0.0)) * dt
		elif token.startswith("A(") and token.endswith(")"):
			base = token[2:-1].strip()
			rows[0][token] = 0.0
			for index in range(1, len(rows)):
				previous = rows[index - 1]
				current = rows[index]
				dt = current["time_s"] - previous["time_s"]
				current[token] = 0.0 if dt <= 0.0 else (current.get(base, 0.0) - previous.get(base, 0.0)) / dt
	return rows


def summarize_transient_rows(rows: list[dict[str, float]], observe: list[str]) -> dict[str, dict[str, float]]:
	times = [float(row["time_s"]) for row in rows]
	summary: dict[str, dict[str, float]] = {}
	for token in observe:
		values = [float(row.get(token, 0.0)) for row in rows]
		summary[token] = compute_signal_metrics(times, values)
	return summary


def compute_signal_metrics(times: list[float], values: list[float]) -> dict[str, float]:
	clean_values = [value for value in values if math.isfinite(value)]
	if not clean_values:
		return {
			"min": float("nan"),
			"max": float("nan"),
			"final": float("nan"),
			"avg": float("nan"),
			"rms": float("nan"),
			"pp": float("nan"),
			"settling_time_s": float("nan"),
			"slew_max": float("nan"),
			"overshoot_ratio": float("nan"),
			"abs_integral": float("nan"),
		}

	final = values[-1]
	if not math.isfinite(final):
		final = clean_values[-1]
	minimum = min(clean_values)
	maximum = max(clean_values)
	peak_to_peak = maximum - minimum
	average = sum(clean_values) / len(clean_values)
	rms = math.sqrt(sum(value * value for value in clean_values) / len(clean_values))
	slew_max = 0.0
	abs_integral = 0.0
	for index in range(1, len(values)):
		dt = max(times[index] - times[index - 1], 0.0)
		if dt > 0.0 and math.isfinite(values[index]) and math.isfinite(values[index - 1]):
			slew_max = max(slew_max, abs(values[index] - values[index - 1]) / dt)
			abs_integral += 0.5 * (abs(values[index]) + abs(values[index - 1])) * dt
	settling_band = max(1e-9, max(abs(final), peak_to_peak, 1.0) * 0.02)
	last_bad = -1
	for index, value in enumerate(values):
		if math.isfinite(value) and abs(value - final) > settling_band:
			last_bad = index
	settling_time = times[min(last_bad + 1, len(times) - 1)] if times else 0.0
	if abs(final) < 1e-12:
		overshoot = max(abs(value) for value in clean_values)
	elif final >= 0.0:
		overshoot = max(0.0, (maximum - final) / abs(final))
	else:
		overshoot = max(0.0, (final - minimum) / abs(final))
	return {
		"min": minimum,
		"max": maximum,
		"final": final,
		"avg": average,
		"rms": rms,
		"pp": peak_to_peak,
		"settling_time_s": settling_time,
		"slew_max": slew_max,
		"overshoot_ratio": overshoot,
		"abs_integral": abs_integral,
	}


def build_series_preview(rows: list[dict[str, float]], observe: list[str], limit: int = PREVIEW_POINT_LIMIT) -> list[dict[str, float]]:
	if len(rows) <= limit:
		return [{"time_s": row["time_s"], **{token: row.get(token, 0.0) for token in observe}} for row in rows]
	result: list[dict[str, float]] = []
	for index in range(limit):
		row_index = round(index * (len(rows) - 1) / max(limit - 1, 1))
		row = rows[row_index]
		result.append({"time_s": row["time_s"], **{token: row.get(token, 0.0) for token in observe}})
	return result


def write_transient_csv(path: Path, rows: list[dict[str, float]]) -> None:
	fieldnames = list(rows[0].keys()) if rows else ["time_s"]
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def render_operating_point_markdown(summary: dict[str, float], elapsed_seconds: float) -> list[str]:
	lines = ["### 工作点分析", "", f"- 计算耗时: {elapsed_seconds:.6f}s", ""]
	if not summary:
		lines.extend(["- 本分析未声明显式观察点，已完成网络收敛验证。", ""])
		return lines
	lines.extend(["| 观察点 | 数值 |", "|---|---:|"])
	for key, value in summary.items():
		lines.append(f"| {key} | {value:.9g} |")
	lines.append("")
	return lines


def render_transient_markdown(summary: dict[str, dict[str, float]], csv_path: Path, elapsed_seconds: float) -> list[str]:
	lines = ["### 瞬态分析", "", f"- CSV 输出: {csv_path}", f"- 计算耗时: {elapsed_seconds:.6f}s", ""]
	if not summary:
		lines.extend(["- 本分析未声明显式观察点，仅完成时域收敛与输出文件生成。", ""])
		return lines
	lines.extend([
		"| 观察点 | 最小值 | 最大值 | 结束值 | RMS | 峰峰值 | 稳定时间(s) | 最大斜率 | 超调比 |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|",
	])
	for key, value in summary.items():
		lines.append(
			"| {key} | {min:.9g} | {max:.9g} | {final:.9g} | {rms:.9g} | {pp:.9g} | {settling:.9g} | {slew:.9g} | {overshoot:.9g} |".format(
				key=key,
				min=value["min"],
				max=value["max"],
				final=value["final"],
				rms=value["rms"],
				pp=value["pp"],
				settling=value["settling_time_s"],
				slew=value["slew_max"],
				overshoot=value["overshoot_ratio"],
			)
		)
	lines.append("")
	return lines


def render_report_markdown(
	scenario: Scenario,
	circuit_summary: dict[str, Any],
	solve_unit_summary: dict[str, Any],
	performance_summary: dict[str, Any],
	warnings: list[str],
	solve_unit_results: list[dict[str, Any]],
	total_elapsed: float,
) -> list[str]:
	lines = [
		f"# 仿真报告：{scenario.name}",
		"",
		"## 总览",
		"",
		f"- 域：{scenario.domain}",
		f"- 求解后端：{solver_backend_name()}",
		f"- 总耗时：{total_elapsed:.6f}s",
		f"- 可仿真器件数：{circuit_summary['selectedComponentCount']}",
		f"- 已忽略器件数：{circuit_summary['ignoredComponentCount']}",
		f"- 网络数：{circuit_summary['netCount']}",
		f"- 求解单元数：{solve_unit_summary['solveUnitCount']}",
		f"- 最大求解单元器件数：{solve_unit_summary['largestSolveUnitComponents']}",
		f"- 最大求解单元节点数：{solve_unit_summary['largestSolveUnitNodes']}",
		f"- 被动等效器件数(R/C/L)：{performance_summary['passiveEquivalentCount']}",
		f"- 线路模型数：{performance_summary['lineModelCount']}",
		"",
	]

	if scenario.metadata.get("importSummary"):
		import_summary = scenario.metadata["importSummary"]
		lines.extend([
			"## 输入导入",
			"",
			f"- 输入类型：{import_summary.get('inputKind', 'scenario')}",
			f"- 输入路径：{import_summary.get('inputPath', '-')}",
			f"- 自动导入器件数：{import_summary.get('componentCount', 0)}",
			f"- 接口级库实例数：{import_summary.get('libraryInstanceCount', 0)}",
			f"- 暂不支持器件数：{import_summary.get('unsupportedComponentCount', 0)}",
			f"- 已启用线路效应：{'yes' if import_summary.get('lineEffectsEnabled') else 'no'}",
			"",
		])

	if performance_summary["targetRuntimeSeconds"] is not None:
		lines.extend([
			"## 性能目标",
			"",
			f"- 目标时间：{performance_summary['targetRuntimeSeconds']:.3f}s",
			f"- 实际时间：{performance_summary['elapsedSeconds']:.3f}s",
			f"- 是否达标：{'PASS' if performance_summary['metRuntimeTarget'] else 'FAIL'}",
			f"- 吞吐：{performance_summary['componentsPerSecond']:.3f} components/s",
			f"- 涉及物理域：{', '.join(performance_summary['presentDomains']) or 'electrical'}",
			f"- 线路模型：{performance_summary['lineModelCount']}",
			"",
		])

	if scenario.library_summary:
		lines.extend(["## 零件库/接口库", "", "| 名称 | 模型 | 分类 | 精度级别 | 组件数 | 接口 |", "|---|---|---|---|---:|---|"])
		for item in scenario.library_summary:
			lines.append(
				"| {name} | {model} | {category} | {fidelity} | {components} | {interfaces} |".format(
					name=item["name"],
					model=item["model"],
					category=item["category"],
					fidelity=item["fidelity"],
					components=item["componentCount"],
					interfaces=", ".join(item.get("interfaces", [])) or "-",
				)
			)
		lines.append("")

	if warnings:
		lines.extend(["## 风险与说明", ""])
		for warning in warnings:
			lines.append(f"- {warning}")
		lines.append("")

	lines.extend(["## 器件类型分布", "", "| 类型 | 数量 |", "|---|---:|"])
	for key, value in circuit_summary["componentKinds"].items():
		lines.append(f"| {key} | {value} |")
	lines.append("")

	if circuit_summary["ignoredPreview"]:
		lines.extend(["## 已忽略器件预览", "", ", ".join(circuit_summary["ignoredPreview"]), ""])

	lines.extend([
		"## 整板求解单元概览",
		"",
		"| 求解单元 | 器件数 | 节点数 | 变量数 | 边界网络数 | 耗时(s) |",
		"|---|---:|---:|---:|---:|---:|",
	])
	for solve_unit in solve_unit_results:
		lines.append(
			"| {label} | {components} | {nodes} | {variables} | {boundaries} | {elapsed:.6f} |".format(
				label=solve_unit["label"],
				components=solve_unit["componentCount"],
				nodes=solve_unit["nodeCount"],
				variables=solve_unit["variableCount"],
				boundaries=len(solve_unit["boundaryNets"]),
				elapsed=solve_unit["elapsedSeconds"],
			)
		)
	lines.append("")

	for solve_unit in solve_unit_results:
		lines.extend(solve_unit["markdown"])

	return lines


def build_dashboard_html(report: dict[str, Any]) -> str:
	circuit = report["circuitSummary"]
	solve_unit_summary = report["solveUnitSummary"]
	performance = report.get("performance", {})
	libraries = report.get("libraries", [])
	solve_units = report["solveUnits"]
	warnings = report.get("warnings", [])

	solve_unit_svg = render_solve_unit_bar_chart(solve_units)
	solve_unit_cards = "\n".join(render_solve_unit_dashboard(solve_unit) for solve_unit in solve_units)
	library_rows = "".join(
		f"<tr><td>{escape(str(item['name']))}</td><td>{escape(str(item['model']))}</td><td>{escape(str(item['category']))}</td><td>{escape(', '.join(item.get('interfaces', [])) or '-')}</td></tr>" for item in libraries
	)
	warning_block = "".join(f"<li>{escape(str(item))}</li>" for item in warnings)
	component_rows = "".join(
		f"<tr><td>{escape(str(kind))}</td><td>{count}</td></tr>" for kind, count in circuit["componentKinds"].items()
	)

	return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(str(report['scenario']['name']))} 仿真工作台</title>
  <style>
    :root {{
      --bg: #f4efe5;
      --ink: #102a43;
      --muted: #4f6b82;
      --panel: rgba(255, 250, 244, 0.88);
      --panel-strong: rgba(255, 248, 237, 0.97);
      --line: rgba(16, 42, 67, 0.12);
      --accent: #b04a2f;
      --accent-2: #1c7c54;
      --accent-3: #1565c0;
      --shadow: 0 24px 60px rgba(32, 48, 64, 0.15);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Aptos, "Segoe UI Variable", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(176, 74, 47, 0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(21, 101, 192, 0.14), transparent 30%),
        linear-gradient(160deg, #f8f5ef 0%, #ede6d9 100%);
      min-height: 100vh;
    }}
    .shell {{ max-width: 1380px; margin: 0 auto; padding: 32px 24px 56px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.70), rgba(255,249,240,0.92));
      border: 1px solid rgba(176, 74, 47, 0.12);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
      backdrop-filter: blur(10px);
      margin-bottom: 22px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0.02em; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    h3 {{ margin: 0 0 12px; font-size: 18px; }}
    p, li, td, th {{ line-height: 1.5; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .grid {{ display: grid; gap: 18px; }}
    .grid.cards {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 18px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px 18px 16px;
      box-shadow: 0 10px 28px rgba(20, 40, 60, 0.08);
    }}
    .metric {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .section {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 14px 36px rgba(20, 40, 60, 0.08);
      padding: 22px;
      margin-bottom: 18px;
    }}
    .two-col {{ display: grid; gap: 18px; grid-template-columns: minmax(280px, 1.2fr) minmax(280px, 1fr); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; font-size: 14px; }}
    th {{ color: var(--muted); font-weight: 600; }}
    ul {{ margin: 0; padding-left: 20px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{ background: rgba(16, 42, 67, 0.08); border-radius: 999px; padding: 6px 10px; font-size: 12px; }}
    .analysis {{ border-top: 1px solid var(--line); padding-top: 14px; margin-top: 14px; }}
    .analysis:first-of-type {{ border-top: 0; padding-top: 0; margin-top: 0; }}
    .chart {{ width: 100%; overflow: hidden; border-radius: 16px; border: 1px solid var(--line); background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(247,242,234,0.95)); padding: 10px; margin-top: 12px; }}
    .chart svg {{ width: 100%; height: auto; display: block; }}
    .empty {{ color: var(--muted); font-style: italic; }}
    @media (max-width: 980px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>{escape(str(report['scenario']['name']))} 多物理域仿真工作台</h1>
      <div class="meta">域: {escape(str(report['scenario']['domain']))} · 求解后端: {escape(str(report['solver']['backend']))} · 生成时间: {escape(str(report['generatedAtUtc']))}</div>
      <div class="grid cards">
        <div class="card"><div class="label">可仿真器件</div><div class="metric">{circuit['selectedComponentCount']}</div></div>
        <div class="card"><div class="label">忽略器件</div><div class="metric">{circuit['ignoredComponentCount']}</div></div>
        <div class="card"><div class="label">网络数</div><div class="metric">{circuit['netCount']}</div></div>
		<div class="card"><div class="label">求解单元数</div><div class="metric">{solve_unit_summary['solveUnitCount']}</div></div>
		<div class="card"><div class="label">单元最大器件</div><div class="metric">{solve_unit_summary['largestSolveUnitComponents']}</div></div>
        <div class="card"><div class="label">总耗时</div><div class="metric">{report['solver']['totalElapsedSeconds']:.3f}s</div></div>
				<div class="card"><div class="label">库实例数</div><div class="metric">{len(libraries)}</div></div>
      </div>
    </section>

		<section class="section two-col">
			<div>
				<h2>性能目标</h2>
				<table>
					<tbody>
						<tr><th>目标时间</th><td>{format_number(performance.get('targetRuntimeSeconds', float('nan')))}</td></tr>
						<tr><th>实际时间</th><td>{format_number(performance.get('elapsedSeconds', float('nan')))}</td></tr>
						<tr><th>达标</th><td>{'PASS' if performance.get('metRuntimeTarget', True) else 'FAIL'}</td></tr>
						<tr><th>被动等效件数</th><td>{performance.get('passiveEquivalentCount', 0)}</td></tr>
						<tr><th>吞吐</th><td>{format_number(performance.get('componentsPerSecond', float('nan')))}</td></tr>
						<tr><th>物理域</th><td>{escape(', '.join(performance.get('presentDomains', [])) or 'electrical')}</td></tr>
					</tbody>
				</table>
			</div>
			<div>
				<h2>求解模式</h2>
				<div class="empty">当前版本仅执行整板单体求解，不做区域求解和结果加和。</div>
			</div>
		</section>

    <section class="section two-col">
      <div>
        <h2>器件分布</h2>
        <table>
          <thead><tr><th>类型</th><th>数量</th></tr></thead>
          <tbody>{component_rows}</tbody>
        </table>
      </div>
      <div>
        <h2>风险与说明</h2>
        {('<ul>' + warning_block + '</ul>') if warnings else '<div class="empty">当前没有额外告警。</div>'}
      </div>
    </section>

		<section class="section">
			<h2>零件库/接口库</h2>
			{('<table><thead><tr><th>名称</th><th>模型</th><th>分类</th><th>接口</th></tr></thead><tbody>' + library_rows + '</tbody></table>') if libraries else '<div class="empty">当前场景未使用预置零件库。</div>'}
		</section>

    <section class="section">
		<h2>整板求解单元图</h2>
		<div class="meta">当前版本固定为整板单体求解；图中仅展示求解单元统计。</div>
	  <div class="chart">{solve_unit_svg}</div>
    </section>

    <section class="section">
		<h2>求解单元结果</h2>
	  {solve_unit_cards}
    </section>
  </div>
</body>
</html>
"""


def render_solve_unit_dashboard(solve_unit: dict[str, Any]) -> str:
	boundary_chips = "".join(f"<span class=\"chip\">{escape(str(item))}</span>" for item in solve_unit.get("boundaryNets", [])[:24])
	component_rows = "".join(
		f"<tr><td>{escape(str(kind))}</td><td>{count}</td></tr>" for kind, count in solve_unit.get("componentKinds", {}).items()
	)
	analyses_html = "".join(render_analysis_dashboard(item) for item in solve_unit.get("analyses", [])) or '<div class="empty">该求解单元没有分析结果。</div>'
	return f"""
    <article class="card" style="margin-bottom: 18px;">
	  <h3>{escape(str(solve_unit['label']))}</h3>
	  <div class="meta">器件 {solve_unit['componentCount']} · 节点 {solve_unit['nodeCount']} · 变量 {solve_unit['variableCount']} · 耗时 {solve_unit['elapsedSeconds']:.6f}s</div>
      <div class="two-col" style="margin-top: 14px;">
        <div>
          <table>
            <thead><tr><th>类型</th><th>数量</th></tr></thead>
            <tbody>{component_rows}</tbody>
          </table>
        </div>
        <div>
          <div class="label" style="margin-bottom: 10px;">边界网络</div>
          <div class="chips">{boundary_chips or '<span class="empty">无</span>'}</div>
        </div>
      </div>
      <div style="margin-top: 16px;">{analyses_html}</div>
    </article>
"""


def render_analysis_dashboard(analysis: dict[str, Any]) -> str:
	analysis_type = str(analysis.get("type", "unknown"))
	observe = analysis.get("observe", [])
	if analysis_type == "op":
		summary = analysis.get("summary", {})
		if not summary:
			content = '<div class="empty">工作点分析已完成，但该分析未声明显式观察点。</div>'
		else:
			rows = "".join(f"<tr><td>{escape(str(key))}</td><td>{format_number(value)}</td></tr>" for key, value in summary.items())
			content = f"<table><thead><tr><th>观察点</th><th>数值</th></tr></thead><tbody>{rows}</tbody></table>"
		return f"<div class=\"analysis\"><div class=\"label\">工作点分析</div><div class=\"meta\">耗时 {analysis.get('elapsedSeconds', 0.0):.6f}s</div>{content}</div>"

	summary = analysis.get("summary", {})
	preview_rows = analysis.get("previewRows", [])
	metric_rows = ""
	charts = ""
	for token in observe:
		if token in summary:
			item = summary[token]
			metric_rows += (
				"<tr>"
				f"<td>{escape(str(token))}</td>"
				f"<td>{format_number(item['min'])}</td>"
				f"<td>{format_number(item['max'])}</td>"
				f"<td>{format_number(item['final'])}</td>"
				f"<td>{format_number(item['rms'])}</td>"
				f"<td>{format_number(item['settling_time_s'])}</td>"
				"</tr>"
			)
		if preview_rows and token in preview_rows[0]:
			charts += f"<div class=\"chart\"><div class=\"meta\">{escape(str(token))}</div>{render_signal_chart(preview_rows, token)}</div>"
		content = f"<table><thead><tr><th>观察点</th><th>最小值</th><th>最大值</th><th>结束值</th><th>RMS</th><th>稳定时间(s)</th></tr></thead><tbody>{metric_rows}</tbody></table>" if metric_rows else '<div class="empty">瞬态分析已完成，但没有显式观察点指标。</div>'
	return f"<div class=\"analysis\"><div class=\"label\">瞬态分析</div><div class=\"meta\">耗时 {analysis.get('elapsedSeconds', 0.0):.6f}s · CSV: {escape(str(analysis.get('csv', '')))}</div>{content}{charts}</div>"


def render_solve_unit_bar_chart(solve_units: list[dict[str, Any]], width: int = 1040, height: int = 260) -> str:
	if not solve_units:
		return '<div class="empty">没有求解单元数据。</div>'
	left = 60
	right = 24
	top = 18
	bottom = 46
	inner_width = width - left - right
	inner_height = height - top - bottom
	bar_count = len(solve_units)
	bar_width = max(18.0, inner_width / max(bar_count * 1.4, 1))
	gap = bar_width * 0.4
	max_components = max(max(solve_unit["componentCount"] for solve_unit in solve_units), 1)
	paths: list[str] = []
	labels: list[str] = []
	for index, solve_unit in enumerate(solve_units):
		x = left + index * (bar_width + gap)
		height_ratio = solve_unit["componentCount"] / max_components
		bar_height = inner_height * height_ratio
		y = top + inner_height - bar_height
		paths.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="8" fill="#b04a2f" opacity="0.82" />')
		labels.append(f'<text x="{x + bar_width / 2:.2f}" y="{height - 18:.2f}" font-size="12" text-anchor="middle" fill="#4f6b82">{escape(solve_unit["label"])}</text>')
		labels.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" font-size="12" text-anchor="middle" fill="#102a43">{solve_unit["componentCount"]}</text>')
	y_axis = ''.join(
		f'<text x="18" y="{top + inner_height - (inner_height * tick / 4):.2f}" font-size="12" fill="#4f6b82">{max_components * tick / 4:.0f}</text>'
		for tick in range(5)
	)
	grid = ''.join(
		f'<line x1="{left}" y1="{top + inner_height - (inner_height * tick / 4):.2f}" x2="{width - right}" y2="{top + inner_height - (inner_height * tick / 4):.2f}" stroke="rgba(16,42,67,0.1)" stroke-width="1" />'
		for tick in range(5)
	)
	return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="solve unit bar chart">{grid}{y_axis}{"".join(paths)}{"".join(labels)}</svg>'


def render_signal_chart(preview_rows: list[dict[str, float]], token: str, width: int = 860, height: int = 240) -> str:
	if len(preview_rows) < 2:
		return '<div class="empty">曲线点数不足。</div>'
	times = [float(row["time_s"]) for row in preview_rows]
	values = [float(row.get(token, 0.0)) for row in preview_rows]
	min_value = min(values)
	max_value = max(values)
	left = 52
	right = 18
	top = 16
	bottom = 30
	inner_width = width - left - right
	inner_height = height - top - bottom
	x_min = min(times)
	x_max = max(times)
	y_span = max(max_value - min_value, 1e-12)
	x_span = max(x_max - x_min, 1e-12)
	points: list[str] = []
	for time_value, signal_value in zip(times, values):
		x = left + ((time_value - x_min) / x_span) * inner_width
		y = top + inner_height - ((signal_value - min_value) / y_span) * inner_height
		points.append(f"{x:.2f},{y:.2f}")
	path = "M " + " L ".join(points)
	grid = ''.join(
		f'<line x1="{left}" y1="{top + inner_height * tick / 4:.2f}" x2="{width - right}" y2="{top + inner_height * tick / 4:.2f}" stroke="rgba(16,42,67,0.09)" stroke-width="1" />'
		for tick in range(5)
	)
	return (
		f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(token)} chart">'
		f'{grid}'
		f'<path d="{path}" fill="none" stroke="#1565c0" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />'
		f'<text x="12" y="{top + 10:.2f}" font-size="12" fill="#4f6b82">{format_number(max_value)}</text>'
		f'<text x="12" y="{top + inner_height:.2f}" font-size="12" fill="#4f6b82">{format_number(min_value)}</text>'
		f'<text x="{left:.2f}" y="{height - 8:.2f}" font-size="12" fill="#4f6b82">{format_number(x_min)}</text>'
		f'<text x="{width - right - 48:.2f}" y="{height - 8:.2f}" font-size="12" fill="#4f6b82">{format_number(x_max)}</text>'
		'</svg>'
	)


def serialize_scenario(scenario: Scenario) -> dict[str, Any]:
	return {
		"name": scenario.name,
		"domain": scenario.domain,
		"netlists": [str(path) for path in scenario.netlists],
		"libraries": list(scenario.library_instances),
		"librarySummary": list(scenario.library_summary),
		"components": list(scenario.inline_components),
		"seed_nets": scenario.seed_nets,
		"hops": scenario.hops,
		"aliases": scenario.aliases,
		"sources": [
			{
				"name": source.name,
				"kind": source.kind,
				"domainKind": source.domain_kind,
				"positive": source.positive,
				"negative": source.negative,
				"waveform": source.waveform,
			}
			for source in scenario.sources
		],
		"analyses": [
			{
				"kind": analysis.kind,
				"observe": analysis.observe,
				"step": analysis.step,
				"stop": analysis.stop,
			}
			for analysis in scenario.analyses
		],
		"targetRuntimeSeconds": scenario.target_runtime_seconds,
		"metadata": scenario.metadata,
	}


def serialize_component(component: Component) -> dict[str, Any]:
	return {
		"designator": component.designator,
		"kind": component.kind,
		"value_text": component.value_text,
		"value": component.value,
		"nodes": list(component.nodes),
		"source_file": component.source_file,
	}


def sorted_pin_items(items: Iterable[tuple[Any, Any]]) -> list[tuple[str, Any]]:
	def sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
		key = str(item[0])
		return (0, f"{int(key):08d}") if key.isdigit() else (1, key)

	return [(str(key), value) for key, value in sorted(items, key=sort_key)]


def parse_component_value(value_text: str, kind: str) -> float | None:
	if kind == "D":
		return 0.0
	if not value_text:
		return None
	cleaned = str(value_text).strip()
	cleaned = cleaned.replace("Ω", "").replace("ohm", "").replace("Ohm", "")
	cleaned = cleaned.replace("N/M", "").replace("N*S/M", "").replace("KG", "")
	if kind in {"C", "L"}:
		cleaned = cleaned.replace("F", "").replace("H", "")
	if kind == "R":
		inline_decimal = re.fullmatch(r"\s*([+-]?\d+)R(\d+)\s*", cleaned, re.IGNORECASE)
		if inline_decimal:
			return float(f"{inline_decimal.group(1)}.{inline_decimal.group(2)}")
	if cleaned.upper() in {"0R", "0"}:
		return 1e-9
	match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z]+)?\s*", cleaned)
	if not match:
		return None
	magnitude = float(match.group(1))
	suffix = (match.group(2) or "").upper()
	if not suffix:
		return magnitude
	if suffix == "R":
		return magnitude
	if suffix in VALUE_SUFFIXES:
		return magnitude * VALUE_SUFFIXES[suffix]
	if suffix.endswith("OHM"):
		suffix = suffix[:-3]
		if suffix == "R":
			return magnitude
		if suffix in VALUE_SUFFIXES:
			return magnitude * VALUE_SUFFIXES[suffix]
	return None


def canonical_net(net: str, aliases: dict[str, str]) -> str:
	current = normalize_net_name(net)
	seen = set()
	while current in aliases and current not in seen:
		seen.add(current)
		current = aliases[current]
	return current


def canonical_observation_token(token: str, aliases: dict[str, str]) -> str:
	text = str(token).strip()
	if text.startswith("I(") and text.endswith(")"):
		return f"I({text[2:-1].strip().upper()})"
	if text.startswith("X(") and text.endswith(")"):
		inner = canonical_net(text[2:-1].strip(), aliases)
		return f"X({inner})"
	if text.startswith("A(") and text.endswith(")"):
		inner = canonical_net(text[2:-1].strip(), aliases)
		return f"A({inner})"
	return canonical_net(text, aliases)


def normalize_net_name(net: str) -> str:
	text = str(net).strip().upper()
	return GROUND_CANONICAL if text in {"0", "GND", "GROUND"} else text


def voltage_between_nodes(circuit: PreparedCircuit, solution: list[float] | None, positive: str, negative: str) -> float:
	return node_voltage(circuit, solution, positive) - node_voltage(circuit, solution, negative)


def node_voltage(circuit: PreparedCircuit, solution: list[float] | None, net: str) -> float:
	if net == GROUND_CANONICAL or solution is None:
		return 0.0 if net == GROUND_CANONICAL else 0.0
	index = circuit.node_variable(net)
	return 0.0 if index is None else solution[index]


def clamp(value: float, minimum: float, maximum: float) -> float:
	return max(minimum, min(value, maximum))


def sanitize_path_part(text: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "scenario"


def format_number(value: float | None) -> str:
	if value is None:
		return "N/A"
	try:
		numeric = float(value)
	except (TypeError, ValueError):
		return "N/A"
	if not math.isfinite(numeric):
		return "N/A"
	return f"{numeric:.6g}"


def solver_backend_name() -> str:
	return "numpy.linalg.solve" if np is not None else "python-gaussian-elimination"


if __name__ == "__main__":
	raise SystemExit(main())