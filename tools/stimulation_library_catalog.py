#!/usr/bin/env python3
from __future__ import annotations

from typing import Any


def get_library_catalog() -> dict[str, dict[str, Any]]:
	return {
		"linear_actuator_stage": {
			"category": "mechanical",
			"description": "线性执行器级，包含质量、阻尼、弹簧、热容和磁路等效。",
			"fidelity": "behavioral-equivalent",
		},
		"belt_drive_stage": {
			"category": "mechanical",
			"description": "皮带传动级，适合滑台/同步带场景。",
			"fidelity": "behavioral-equivalent",
		},
		"gear_train_pair": {
			"category": "mechanical",
			"description": "双惯量齿轮副，适合减速箱/谐波段近似。",
			"fidelity": "behavioral-equivalent",
		},
		"solenoid_actuator": {
			"category": "mechanical",
			"description": "电磁执行器，含线圈、磁路、热和机械返回弹簧。",
			"fidelity": "behavioral-equivalent",
		},
		"thermal_plate": {
			"category": "thermal",
			"description": "热板/散热块热阻热容模型。",
			"fidelity": "behavioral-equivalent",
		},
		"ldo_basic": {
			"category": "common-ic",
			"description": "常见 LDO 的板级输出滤波与负载等效。",
			"fidelity": "interface-level",
		},
		"level_shifter_bidirectional": {
			"category": "common-ic",
			"description": "双向电平转换器的上拉、串阻和寄生容性负载。",
			"fidelity": "interface-level",
		},
		"spi_peripheral_load": {
			"category": "common-ic",
			"description": "SPI 外设/移位寄存器/串行 ADC 的板级 IO 与供电负载等效。",
			"fidelity": "interface-level",
		},
		"logic_buffer_multi_io": {
			"category": "common-ic",
			"description": "多路逻辑缓冲器的供电、输出串阻和 IO 寄生负载等效。",
			"fidelity": "interface-level",
		},
		"analog_switch_spst_load": {
			"category": "common-ic",
			"description": "单通道双向模拟开关的控制脚、漏电和通道寄生等效。",
			"fidelity": "interface-level",
		},
		"analog_mux_8ch_load": {
			"category": "common-ic",
			"description": "8 路模拟多路复用器的地址选择、通道寄生和供电负载等效。",
			"fidelity": "interface-level",
		},
		"comparator_dual_open_collector": {
			"category": "common-ic",
			"description": "双比较器开漏输出、输入寄生和供电负载等效。",
			"fidelity": "interface-level",
		},
		"comparator_single_fast": {
			"category": "common-ic",
			"description": "高速单比较器的输出与输入寄生、供电静态负载等效。",
			"fidelity": "interface-level",
		},
		"segment_display_4digit_common_anode": {
			"category": "common-ic",
			"description": "四位共阳数码管的段线/位选线寄生与泄放负载等效。",
			"fidelity": "interface-level",
		},
		"spi_eeprom_load": {
			"category": "common-ic",
			"description": "SPI EEPROM/Flash 的 IO 负载与去耦网络。",
			"fidelity": "interface-level",
		},
		"adc_frontend_rc": {
			"category": "common-ic",
			"description": "ADC 前端 RC 抗混叠与采样保持板级等效。",
			"fidelity": "interface-level",
		},
		"can_transceiver_terminated": {
			"category": "common-ic",
			"description": "CAN 收发器终端和共模滤波网络。",
			"fidelity": "interface-level",
		},
		"rs485_transceiver_terminated": {
			"category": "common-ic",
			"description": "RS-485 收发器终端、偏置和共模抑制网络。",
			"fidelity": "interface-level",
		},
		"hbridge_dc_motor_interface": {
			"category": "common-ic",
			"description": "H 桥驱动到直流电机的板级等效接口。",
			"fidelity": "interface-level",
		},
		"stm32f103c8_interface": {
			"category": "mcu-interface",
			"description": "STM32F103C8/BluePill 板级接口模型。",
			"fidelity": "interface-level",
		},
		"stm32f407_interface": {
			"category": "mcu-interface",
			"description": "STM32F4 高速接口板级模型。",
			"fidelity": "interface-level",
		},
		"at89c51_interface": {
			"category": "mcu-interface",
			"description": "51 系列单片机最小系统和 IO 接口板级模型。",
			"fidelity": "interface-level",
		},
		"raspberry_pi_40pin_interface": {
			"category": "mcu-interface",
			"description": "Raspberry Pi 40Pin 口的板级接口模型。",
			"fidelity": "interface-level",
		},
		"arduino_uno_interface": {
			"category": "mcu-interface",
			"description": "Arduino UNO 板级接口模型。",
			"fidelity": "interface-level",
		},
		"arduino_mega_interface": {
			"category": "mcu-interface",
			"description": "Arduino Mega 2560 板级接口模型。",
			"fidelity": "interface-level",
		},
		"esp32_devkit_interface": {
			"category": "mcu-interface",
			"description": "ESP32 DevKit 常见外围接口模型。",
			"fidelity": "interface-level",
		},
		"esp8266_nodemcu_interface": {
			"category": "mcu-interface",
			"description": "ESP8266 NodeMCU 常见外围接口模型。",
			"fidelity": "interface-level",
		},
		"passive_mesh_region": {
			"category": "benchmark",
			"description": "用于性能压测的大规模 RLC 被动网格区域。",
			"fidelity": "reduced-order",
		},
	}


def build_template_scenario() -> dict[str, Any]:
	return {
		"name": "custom_multiphysics_template",
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
			{
				"model": "hbridge_dc_motor_interface",
				"name": "DRV_A",
				"nets": {"vm": "12V", "gnd": "GND", "out_a": "MOTOR_A", "out_b": "MOTOR_B"},
			},
			{
				"model": "linear_actuator_stage",
				"name": "ACT_STAGE",
				"nets": {
					"stage": "STAGE",
					"frame": "FRAME",
					"heat": "STAGE_HEAT",
					"ambient": "AMBIENT",
					"flux": "ACT_FLUX",
					"return": "MAG_RETURN",
				},
			},
		],
		"sources": [
			{"name": "VIN_12V", "kind": "voltage", "positive": "12V", "negative": "GND", "waveform": {"kind": "dc", "value": 12.0}},
			{"name": "VIN_3V3", "kind": "voltage", "positive": "3V3", "negative": "GND", "waveform": {"kind": "dc", "value": 3.3}},
			{"name": "HEAT_LOAD", "kind": "heat", "positive": "STAGE_HEAT", "negative": "AMBIENT", "waveform": {"kind": "dc", "value": 4.5}},
		],
		"analyses": [
			{"type": "op", "observe": ["12V", "3V3", "PWM_MOTOR", "STAGE_HEAT", "ACT_FLUX"]},
			{"type": "tran", "step": 5e-4, "stop": 0.05, "observe": ["PWM_MOTOR", "STAGE", "X(STAGE)", "STAGE_HEAT"]},
		],
		"regions": [
			{"name": "controller", "seedNets": ["3V3", "UART_TX", "SPI_CLK", "PWM_MOTOR"], "hops": 3},
			{"name": "power_stage", "seedNets": ["12V", "MOTOR_A", "MOTOR_B"], "hops": 3},
			{"name": "actuator", "seedNets": ["STAGE", "FRAME", "STAGE_HEAT", "ACT_FLUX"], "hops": 4},
		],
		"partition": {"enabled": True, "workers": 4, "maxComponentsPerPartition": 160},
		"targetRuntimeSeconds": 60.0,
	}


def expand_library_instances(instances: list[dict[str, Any]]) -> dict[str, Any]:
	catalog = get_library_catalog()
	components: list[dict[str, Any]] = []
	sources: list[dict[str, Any]] = []
	notes: list[str] = []
	summary: list[dict[str, Any]] = []
	for raw in instances:
		model = str(raw.get("model", "")).strip()
		if model not in catalog:
			raise ValueError(f"unknown library model: {model}")
		name = str(raw.get("name") or model).strip().upper()
		nets = {str(key): str(value) for key, value in dict(raw.get("nets", {})).items()}
		params = dict(raw.get("params", {}))
		fragment = _BUILDERS[model](name, nets, params)
		components.extend(fragment.get("components", []))
		sources.extend(fragment.get("sources", []))
		notes.extend(fragment.get("notes", []))
		summary.append(
			{
				"name": name,
				"model": model,
				"category": catalog[model]["category"],
				"fidelity": catalog[model]["fidelity"],
				"description": catalog[model]["description"],
				"componentCount": len(fragment.get("components", [])),
				"sourceCount": len(fragment.get("sources", [])),
				"interfaces": list(fragment.get("interfaces", [])),
			}
		)
	return {"components": components, "sources": sources, "notes": notes, "summary": summary}


def _component(
	designator: str,
	kind: str,
	value: float,
	node_a: str,
	node_b: str,
	physics: dict[str, Any] | None = None,
	metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
	component = {"designator": designator, "kind": kind, "value": value, "nodes": [node_a, node_b]}
	if physics:
		component["physics"] = dict(physics)
	if metadata:
		component["metadata"] = dict(metadata)
	return component


def _source(name: str, kind: str, positive: str, negative: str, value: float) -> dict[str, Any]:
	return {"name": name, "kind": kind, "positive": positive, "negative": negative, "waveform": {"kind": "dc", "value": value}}


def _net(nets: dict[str, str], key: str, fallback: str) -> str:
	return str(nets.get(key, fallback))


def _series_loaded_io(
	name: str,
	pin_name: str,
	net: str,
	reference: str,
	series_ohm: float = 33.0,
	shunt_cap: float = 8e-12,
	pull_ohm: float = 1e6,
	heat: str | None = None,
	ambient: str = "AMBIENT",
	resistor_power_scale: float = 0.06,
) -> list[dict[str, Any]]:
	internal = f"{name}_{pin_name}_INT"
	resistor_physics = {"thermal": {"positive": heat, "negative": ambient, "powerScale": resistor_power_scale}} if heat else None
	return [
		_component(f"{name}_{pin_name}_SER", "resistor", series_ohm, internal, net, physics=resistor_physics),
		_component(f"{name}_{pin_name}_CAP", "capacitor", shunt_cap, internal, reference),
		_component(f"{name}_{pin_name}_PULL", "resistor", pull_ohm, internal, reference, physics=resistor_physics),
	]


def _package_thermal_network(name: str, heat: str, ambient: str, thermal_resistance: float, thermal_capacity: float) -> list[dict[str, Any]]:
	return [
		_component(f"{name}_HEAT_R", "thermal_resistor", thermal_resistance, heat, ambient),
		_component(f"{name}_HEAT_C", "thermal_capacitor", thermal_capacity, heat, ambient),
	]


def _build_linear_actuator_stage(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	stage = _net(nets, "stage", "STAGE")
	frame = _net(nets, "frame", "FRAME")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	flux = _net(nets, "flux", f"{name}_FLUX")
	mag_return = _net(nets, "return", "MAG_RETURN")
	components = [
		_component(f"{name}_MASS", "mass", float(params.get("mass_kg", 0.35)), stage, frame),
		_component(
			f"{name}_DAMP",
			"damper",
			float(params.get("damping_ns_m", 9.5)),
			stage,
			frame,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("friction_to_heat_scale", 1.0))}},
		),
		_component(f"{name}_SPRING", "spring", float(params.get("spring_n_m", 180.0)), stage, frame),
		_component(f"{name}_THERM_R", "thermal_resistor", float(params.get("thermal_resistance", 2.2)), heat, ambient),
		_component(f"{name}_THERM_C", "thermal_capacitor", float(params.get("thermal_capacity", 22.0)), heat, ambient),
		_component(f"{name}_MAG_R", "magnetic_reluctance", float(params.get("reluctance", 4.0)), flux, mag_return),
		_component(
			f"{name}_MAG_L",
			"magnetic_storage",
			float(params.get("magnetic_storage", 1.8e-3)),
			flux,
			mag_return,
			physics={
				"lossResistance": float(params.get("magnetic_loss_resistance", 0.08)),
				"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("magnetic_to_heat_scale", 0.35))},
			},
		),
	]
	return {"components": components, "notes": [f"{name}: 线性执行器级使用行为等效模型，适合结构级/接口级分析。"], "interfaces": ["mechanical", "thermal", "magnetic"]}


def _build_belt_drive_stage(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	drive = _net(nets, "drive", "DRIVE")
	load = _net(nets, "load", "LOAD")
	frame = _net(nets, "frame", "FRAME")
	components = [
		_component(f"{name}_DRIVE_M", "mass", float(params.get("drive_inertia", 0.08)), drive, frame),
		_component(f"{name}_LOAD_M", "mass", float(params.get("load_inertia", 0.16)), load, frame),
		_component(f"{name}_BELT_K", "spring", float(params.get("belt_stiffness", 120.0)), drive, load),
		_component(f"{name}_BELT_B", "damper", float(params.get("belt_damping", 6.0)), drive, load),
	]
	return {"components": components, "interfaces": ["mechanical"]}


def _build_gear_train_pair(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	input_node = _net(nets, "input", "GEAR_IN")
	output_node = _net(nets, "output", "GEAR_OUT")
	frame = _net(nets, "frame", "FRAME")
	components = [
		_component(f"{name}_IN_M", "mass", float(params.get("input_inertia", 0.03)), input_node, frame),
		_component(f"{name}_OUT_M", "mass", float(params.get("output_inertia", 0.12)), output_node, frame),
		_component(f"{name}_MESH_K", "spring", float(params.get("mesh_stiffness", 260.0)), input_node, output_node),
		_component(f"{name}_MESH_B", "damper", float(params.get("mesh_damping", 8.0)), input_node, output_node),
	]
	return {"components": components, "interfaces": ["mechanical"]}


def _build_solenoid_actuator(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	coil_plus = _net(nets, "coil_plus", "COIL_PLUS")
	coil_minus = _net(nets, "coil_minus", "COIL_MINUS")
	flux = _net(nets, "flux", f"{name}_FLUX")
	mag_return = _net(nets, "return", "MAG_RETURN")
	stage = _net(nets, "stage", "PLUNGER")
	frame = _net(nets, "frame", "FRAME")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(
			f"{name}_COIL_R",
			"resistor",
			float(params.get("coil_resistance", 8.2)),
			coil_plus,
			coil_minus,
			physics={
				"temperatureCoefficientPerC": float(params.get("coil_tempco", 0.0039)),
				"thermal": {"positive": heat, "negative": ambient, "powerScale": 1.0},
			},
		),
		_component(
			f"{name}_COIL_L",
			"inductor",
			float(params.get("coil_inductance", 4.8e-3)),
			coil_plus,
			coil_minus,
			physics={
				"lossResistance": float(params.get("coil_loss_resistance", 0.18)),
				"saturationCurrent": float(params.get("coil_saturation_current", 2.4)),
				"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("coil_magnetic_heat_scale", 0.4))},
			},
		),
		_component(f"{name}_PLUNGER_M", "mass", float(params.get("plunger_mass", 0.09)), stage, frame),
		_component(f"{name}_RETURN_K", "spring", float(params.get("return_spring", 140.0)), stage, frame),
		_component(
			f"{name}_PLUNGER_B",
			"damper",
			float(params.get("plunger_damping", 5.0)),
			stage,
			frame,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("friction_to_heat_scale", 1.0))}},
		),
		_component(f"{name}_MAG_R", "magnetic_reluctance", float(params.get("reluctance", 5.5)), flux, mag_return),
		_component(
			f"{name}_MAG_L",
			"magnetic_storage",
			float(params.get("magnetic_storage", 1.4e-3)),
			flux,
			mag_return,
			physics={
				"lossResistance": float(params.get("core_loss_resistance", 0.12)),
				"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("core_to_heat_scale", 0.3))},
			},
		),
		_component(f"{name}_HEAT_R", "thermal_resistor", float(params.get("thermal_resistance", 3.2)), heat, ambient),
		_component(f"{name}_HEAT_C", "thermal_capacitor", float(params.get("thermal_capacity", 18.0)), heat, ambient),
	]
	return {"components": components, "interfaces": ["electrical", "mechanical", "thermal", "magnetic"]}


def _build_thermal_plate(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	plane = _net(nets, "node", f"{name}_TEMP")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_RTH", "thermal_resistor", float(params.get("thermal_resistance", 1.2)), plane, ambient),
		_component(f"{name}_CTH", "thermal_capacitor", float(params.get("thermal_capacity", 42.0)), plane, ambient),
	]
	return {"components": components, "interfaces": ["thermal"]}


def _build_ldo_basic(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vin = _net(nets, "vin", "VIN")
	vout = _net(nets, "vout", "VOUT")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(
			f"{name}_RSER",
			"resistor",
			float(params.get("dropout_resistance", 0.18)),
			vin,
			vout,
			physics={
				"temperatureCoefficientPerC": float(params.get("tempco", 0.0032)),
				"thermal": {"positive": heat, "negative": ambient, "powerScale": 1.0},
			},
		),
		_component(f"{name}_CIN", "capacitor", float(params.get("cin_f", 4.7e-6)), vin, gnd),
		_component(f"{name}_COUT", "capacitor", float(params.get("cout_f", 22e-6)), vout, gnd),
		_component(
			f"{name}_IQ",
			"resistor",
			float(params.get("quiescent_ohm", 10000.0)),
			vin,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("quiescent_heat_scale", 0.25))}},
		),
		_component(f"{name}_LOAD", "resistor", float(params.get("load_ohm", 220.0)), vout, gnd),
	]
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 6.5)), float(params.get("thermal_capacity", 7.5))))
	return {"components": components, "interfaces": ["power", "thermal"]}


def _build_level_shifter_bidirectional(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	low = _net(nets, "low", "LOW_IO")
	high = _net(nets, "high", "HIGH_IO")
	vcca = _net(nets, "vcca", "3V3")
	vccb = _net(nets, "vccb", "5V")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(
			f"{name}_PULL_A",
			"resistor",
			float(params.get("pull_a", 4700.0)),
			low,
			vcca,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.35}},
		),
		_component(
			f"{name}_PULL_B",
			"resistor",
			float(params.get("pull_b", 4700.0)),
			high,
			vccb,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.35}},
		),
		_component(
			f"{name}_SER_A",
			"resistor",
			float(params.get("series_ohm", 33.0)),
			low,
			high,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.25}},
		),
		_component(f"{name}_CAP_A", "capacitor", float(params.get("io_cap_f", 12e-12)), low, "GND"),
		_component(f"{name}_CAP_B", "capacitor", float(params.get("io_cap_f", 12e-12)), high, "GND"),
		_component(
			f"{name}_LEAK",
			"resistor",
			float(params.get("leakage_ohm", 68000.0)),
			vcca,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.2}},
		),
	]
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 15.0)), float(params.get("thermal_capacity", 3.0))))
	return {"components": components, "interfaces": ["i2c", "gpio", "thermal"]}


def _build_spi_peripheral_load(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vdd = _net(nets, "vdd", "3V3")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_CDEC", "capacitor", float(params.get("decoupling", 0.1e-6)), vdd, gnd),
		_component(
			f"{name}_CORE",
			"resistor",
			float(params.get("core_load_ohm", 1800.0)),
			vdd,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("core_heat_scale", 0.9))}},
		),
	]
	for pin_name in ("spi_clk", "spi_mosi", "spi_miso", "spi_cs"):
		if pin_name in nets:
			components.extend(
				_series_loaded_io(
					name,
					pin_name.upper(),
					nets[pin_name],
					gnd,
					series_ohm=float(params.get("series_ohm", 27.0)),
					shunt_cap=float(params.get("io_cap_f", 10e-12)),
					heat=heat,
					ambient=ambient,
				)
			)
	for key in sorted(nets):
		if not key.startswith("io"):
			continue
		index = key[2:]
		net = nets[key]
		components.extend(
			[
				_component(f"{name}_IO{index}_CAP", "capacitor", float(params.get("output_cap_f", 15e-12)), net, gnd),
				_component(f"{name}_IO{index}_BLEED", "resistor", float(params.get("output_bleed_ohm", 1e6)), net, gnd),
			]
		)
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 14.0)), float(params.get("thermal_capacity", 2.8))))
	return {"components": components, "interfaces": ["spi", "logic-io", "thermal"]}


def _build_logic_buffer_multi_io(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vdd = _net(nets, "vdd", "3V3")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_CDEC", "capacitor", float(params.get("decoupling", 0.047e-6)), vdd, gnd),
		_component(
			f"{name}_CORE",
			"resistor",
			float(params.get("core_load_ohm", 12000.0)),
			vdd,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.55}},
		),
	]
	pair_keys = sorted(key[2:] for key in nets if key.startswith("in") and f"out{key[2:]}" in nets)
	for index in pair_keys:
		input_net = nets[f"in{index}"]
		output_net = nets[f"out{index}"]
		components.extend(
			[
				_component(f"{name}_IN{index}_CAP", "capacitor", float(params.get("input_cap_f", 6e-12)), input_net, gnd),
				_component(
					f"{name}_BUF{index}",
					"resistor",
					float(params.get("drive_ohm", 12.0)),
					input_net,
					output_net,
					physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.08}},
				),
				_component(f"{name}_OUT{index}_CAP", "capacitor", float(params.get("io_cap_f", 10e-12)), output_net, gnd),
				_component(
					f"{name}_OUT{index}_BLEED",
					"resistor",
					float(params.get("io_bleed_ohm", 2e6)),
					output_net,
					gnd,
					physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.04}},
				),
			]
		)
	for key in sorted(nets):
		if key.startswith("in") or key.startswith("out"):
			continue
		if not key.startswith("io"):
			continue
		index = key[2:]
		components.extend(
			_series_loaded_io(
				name,
				f"IO{index}",
				nets[key],
				gnd,
				series_ohm=float(params.get("series_ohm", 22.0)),
				shunt_cap=float(params.get("io_cap_f", 10e-12)),
				pull_ohm=float(params.get("io_bleed_ohm", 2e6)),
				heat=heat,
				ambient=ambient,
				resistor_power_scale=0.04,
			)
		)
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 16.0)), float(params.get("thermal_capacity", 2.2))))
	return {"components": components, "interfaces": ["logic-io", "buffer", "thermal"]}


def _build_analog_mux_8ch_load(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vdd = _net(nets, "vdd", "3V3")
	gnd = _net(nets, "gnd", "GND")
	common = _net(nets, "common", f"{name}_COMMON")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_CDEC", "capacitor", float(params.get("decoupling", 0.1e-6)), vdd, gnd),
		_component(
			f"{name}_CORE",
			"resistor",
			float(params.get("core_load_ohm", 3300.0)),
			vdd,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.8}},
		),
		_component(f"{name}_COMMON_CAP", "capacitor", float(params.get("common_cap_f", 18e-12)), common, gnd),
		_component(f"{name}_COMMON_BLEED", "resistor", float(params.get("common_bleed_ohm", 2e6)), common, gnd),
	]
	for pin_name in ("addr0", "addr1", "addr2", "enable"):
		if pin_name in nets:
			components.extend(
				_series_loaded_io(
					name,
					pin_name.upper(),
					nets[pin_name],
					gnd,
					series_ohm=float(params.get("control_series_ohm", 33.0)),
					shunt_cap=float(params.get("control_cap_f", 10e-12)),
					heat=heat,
					ambient=ambient,
				)
			)
	for key in sorted(nets):
		if not key.startswith("ch"):
			continue
		index = key[2:]
		channel = nets[key]
		components.append(_component(f"{name}_CH{index}_CAP", "capacitor", float(params.get("channel_cap_f", 8e-12)), channel, gnd))
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 18.0)), float(params.get("thermal_capacity", 2.0))))
	return {"components": components, "interfaces": ["mux", "analog", "thermal"]}


def _build_analog_switch_spst_load(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vdd = _net(nets, "vdd", "3V3")
	gnd = _net(nets, "gnd", "GND")
	ctrl = nets.get("ctrl")
	analog_a = nets.get("a")
	analog_b = nets.get("b")
	components = [
		_component(f"{name}_CORE", "resistor", float(params.get("core_load_ohm", 250000.0)), vdd, gnd),
	]
	if ctrl:
		components.extend(
			[
				_component(f"{name}_CTRL_CAP", "capacitor", float(params.get("control_cap_f", 4e-12)), ctrl, gnd),
				_component(f"{name}_CTRL_BLEED", "resistor", float(params.get("control_bleed_ohm", 5e6)), ctrl, gnd),
			]
		)
	if analog_a:
		components.append(_component(f"{name}_A_CAP", "capacitor", float(params.get("channel_cap_f", 6e-12)), analog_a, gnd))
	if analog_b:
		components.append(_component(f"{name}_B_CAP", "capacitor", float(params.get("channel_cap_f", 6e-12)), analog_b, gnd))
	if analog_a and analog_b:
		components.append(_component(f"{name}_AB_LEAK", "resistor", float(params.get("off_leak_ohm", 5e7)), analog_a, analog_b))
	return {"components": components, "interfaces": ["switch", "analog"]}


def _build_comparator_dual_open_collector(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vdd = _net(nets, "vdd", "5V")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_CDEC", "capacitor", float(params.get("decoupling", 0.1e-6)), vdd, gnd),
		_component(
			f"{name}_CORE",
			"resistor",
			float(params.get("core_load_ohm", 4700.0)),
			vdd,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.75}},
		),
	]
	for output_key in ("out0", "out1"):
		if output_key not in nets:
			continue
		output_net = nets[output_key]
		index = output_key[-1]
		components.extend(
			[
				_component(
					f"{name}_OUT{index}_PULL",
					"resistor",
					float(params.get("pullup_ohm", 4700.0)),
					output_net,
					vdd,
					physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.3}},
				),
				_component(f"{name}_OUT{index}_CAP", "capacitor", float(params.get("output_cap_f", 12e-12)), output_net, gnd),
			]
		)
	for input_key in ("in0p", "in0n", "in1p", "in1n"):
		if input_key not in nets:
			continue
		input_net = nets[input_key]
		label = input_key.upper()
		components.extend(
			[
				_component(f"{name}_{label}_CAP", "capacitor", float(params.get("input_cap_f", 6e-12)), input_net, gnd),
				_component(f"{name}_{label}_BLEED", "resistor", float(params.get("input_bleed_ohm", 2e6)), input_net, gnd),
			]
		)
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 20.0)), float(params.get("thermal_capacity", 1.8))))
	return {"components": components, "interfaces": ["comparator", "analog", "thermal"]}


def _build_comparator_single_fast(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vdd = _net(nets, "vdd", "3V3")
	gnd = _net(nets, "gnd", "GND")
	components = [
		_component(f"{name}_CORE", "resistor", float(params.get("core_load_ohm", 18000.0)), vdd, gnd),
	]
	if "out0" in nets:
		components.extend(
			[
				_component(f"{name}_OUT0_CAP", "capacitor", float(params.get("output_cap_f", 10e-12)), nets["out0"], gnd),
				_component(f"{name}_OUT0_BLEED", "resistor", float(params.get("output_bleed_ohm", 2e6)), nets["out0"], gnd),
			]
		)
	for input_key in ("in0p", "in0n", "in1p", "in1n"):
		if input_key not in nets:
			continue
		label = input_key.upper()
		components.extend(
			[
				_component(f"{name}_{label}_CAP", "capacitor", float(params.get("input_cap_f", 5e-12)), nets[input_key], gnd),
				_component(f"{name}_{label}_BLEED", "resistor", float(params.get("input_bleed_ohm", 3e6)), nets[input_key], gnd),
			]
		)
	return {"components": components, "interfaces": ["comparator", "analog"]}


def _build_segment_display_4digit_common_anode(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	gnd = _net(nets, "gnd", "GND")
	components: list[dict[str, Any]] = []
	for key in sorted(nets):
		net = nets[key]
		label = key.upper()
		if key.startswith("seg_"):
			components.extend(
				[
					_component(f"{name}_{label}_CAP", "capacitor", float(params.get("segment_cap_f", 15e-12)), net, gnd),
					_component(f"{name}_{label}_BLEED", "resistor", float(params.get("segment_bleed_ohm", 3e6)), net, gnd),
				]
			)
		elif key.startswith("dig_"):
			components.extend(
				[
					_component(f"{name}_{label}_CAP", "capacitor", float(params.get("digit_cap_f", 18e-12)), net, gnd),
					_component(f"{name}_{label}_BLEED", "resistor", float(params.get("digit_bleed_ohm", 2e6)), net, gnd),
				]
			)
	return {"components": components, "interfaces": ["display", "logic-io"]}


def _build_spi_eeprom_load(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vdd = _net(nets, "vdd", "3V3")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_CDEC", "capacitor", float(params.get("decoupling", 0.1e-6)), vdd, gnd),
		_component(
			f"{name}_CORE",
			"resistor",
			float(params.get("core_load_ohm", 3300.0)),
			vdd,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.8}},
		),
	]
	for pin_name in ("spi_clk", "spi_mosi", "spi_miso", "spi_cs"):
		if pin_name in nets:
			components.extend(
				_series_loaded_io(
					name,
					pin_name.upper(),
					nets[pin_name],
					gnd,
					series_ohm=float(params.get("series_ohm", 27.0)),
					shunt_cap=float(params.get("io_cap_f", 10e-12)),
					heat=heat,
					ambient=ambient,
				)
			)
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 16.0)), float(params.get("thermal_capacity", 2.4))))
	return {"components": components, "interfaces": ["spi", "thermal"]}


def _build_adc_frontend_rc(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	input_net = _net(nets, "input", "ADC_IN")
	sampled = _net(nets, "sampled", f"{name}_SAMPLED")
	gnd = _net(nets, "gnd", "GND")
	components = [
		_component(f"{name}_RIN", "resistor", float(params.get("series_ohm", 100.0)), input_net, sampled),
		_component(f"{name}_CFILT", "capacitor", float(params.get("hold_cap_f", 100e-12)), sampled, gnd),
		_component(f"{name}_RBLEED", "resistor", float(params.get("bleed_ohm", 1e6)), sampled, gnd),
	]
	return {"components": components, "interfaces": ["adc"]}


def _build_can_transceiver_terminated(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	canh = _net(nets, "canh", "CANH")
	canl = _net(nets, "canl", "CANL")
	vdd = _net(nets, "vdd", "5V")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_TERM", "resistor", float(params.get("term_ohm", 120.0)), canh, canl),
		_component(f"{name}_CMH", "capacitor", float(params.get("cm_cap_f", 47e-12)), canh, gnd),
		_component(f"{name}_CML", "capacitor", float(params.get("cm_cap_f", 47e-12)), canl, gnd),
		_component(
			f"{name}_CORE",
			"resistor",
			float(params.get("core_load_ohm", 4700.0)),
			vdd,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.75}},
		),
	]
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 13.0)), float(params.get("thermal_capacity", 2.6))))
	return {"components": components, "interfaces": ["can", "thermal"]}


def _build_rs485_transceiver_terminated(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	a = _net(nets, "a", "RS485_A")
	b = _net(nets, "b", "RS485_B")
	vdd = _net(nets, "vdd", "5V")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_TERM", "resistor", float(params.get("term_ohm", 120.0)), a, b),
		_component(f"{name}_BIAS_A", "resistor", float(params.get("bias_ohm", 680.0)), a, vdd),
		_component(f"{name}_BIAS_B", "resistor", float(params.get("bias_ohm", 680.0)), b, gnd),
		_component(
			f"{name}_CORE",
			"resistor",
			float(params.get("core_load_ohm", 5600.0)),
			vdd,
			gnd,
			physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.7}},
		),
	]
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 13.0)), float(params.get("thermal_capacity", 2.6))))
	return {"components": components, "interfaces": ["rs485", "thermal"]}


def _build_hbridge_dc_motor_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	vm = _net(nets, "vm", "12V")
	gnd = _net(nets, "gnd", "GND")
	out_a = _net(nets, "out_a", "MOTOR_A")
	out_b = _net(nets, "out_b", "MOTOR_B")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components = [
		_component(f"{name}_VM_DEC", "capacitor", float(params.get("bulk_cap_f", 47e-6)), vm, gnd),
		_component(
			f"{name}_WIND_R",
			"resistor",
			float(params.get("winding_ohm", 3.6)),
			out_a,
			out_b,
			physics={
				"temperatureCoefficientPerC": float(params.get("winding_tempco", 0.0037)),
				"thermal": {"positive": heat, "negative": ambient, "powerScale": 1.0},
			},
		),
		_component(
			f"{name}_WIND_L",
			"inductor",
			float(params.get("winding_l_h", 1.2e-3)),
			out_a,
			out_b,
			physics={
				"lossResistance": float(params.get("winding_loss_resistance", 0.09)),
				"saturationCurrent": float(params.get("winding_saturation_current", 4.5)),
				"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("magnetic_heat_scale", 0.25))},
			},
		),
		_component(f"{name}_SNUB_A", "capacitor", float(params.get("snub_cap_f", 1e-9)), out_a, gnd),
		_component(f"{name}_SNUB_B", "capacitor", float(params.get("snub_cap_f", 1e-9)), out_b, gnd),
		_component(f"{name}_DIO_A", "diode", 0.0, out_a, vm),
		_component(f"{name}_DIO_B", "diode", 0.0, gnd, out_b),
		_component(f"{name}_HEAT_R", "thermal_resistor", float(params.get("thermal_resistance", 2.8)), heat, ambient),
		_component(f"{name}_HEAT_C", "thermal_capacitor", float(params.get("thermal_capacity", 14.0)), heat, ambient),
	]
	return {"components": components, "interfaces": ["pwm", "motor-drive", "thermal"]}


def _build_generic_mcu_interface(name: str, family: str, nets: dict[str, str], params: dict[str, Any], decoupling_count: int, extra_bias: list[tuple[str, str, float]]) -> dict[str, Any]:
	vdd = _net(nets, "vdd", "3V3")
	gnd = _net(nets, "gnd", "GND")
	heat = _net(nets, "heat", f"{name}_HEAT")
	ambient = _net(nets, "ambient", "AMBIENT")
	components: list[dict[str, Any]] = []
	for index in range(decoupling_count):
		components.append(_component(f"{name}_DEC{index+1}", "capacitor", float(params.get("decoupling_f", 0.1e-6)), vdd, gnd))
	components.append(
		_component(
			f"{name}_CORE",
			"resistor",
			float(params.get("core_load_ohm", 1500.0)),
			vdd,
			gnd,
			physics={
				"temperatureCoefficientPerC": float(params.get("core_tempco", 0.0015)),
				"thermal": {"positive": heat, "negative": ambient, "powerScale": float(params.get("core_heat_scale", 0.95))},
			},
		)
	)
	for signal, default_net, resistance in extra_bias:
		net = _net(nets, signal, default_net)
		if signal == "boot0":
			components.append(_component(f"{name}_BOOT_PD", "resistor", resistance, net, gnd, physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.08}}))
		elif signal in {"nrst", "en", "reset"}:
			components.append(_component(f"{name}_{signal.upper()}_PU", "resistor", resistance, net, vdd, physics={"thermal": {"positive": heat, "negative": ambient, "powerScale": 0.08}}))
	bus_map = {
		"uart_tx": gnd,
		"uart_rx": gnd,
		"spi_clk": gnd,
		"spi_mosi": gnd,
		"spi_miso": gnd,
		"spi_cs": gnd,
		"i2c_scl": vdd,
		"i2c_sda": vdd,
		"pwm0": gnd,
		"pwm1": gnd,
		"gpio0": gnd,
		"gpio1": gnd,
	}
	interfaces: list[str] = []
	for key, reference in bus_map.items():
		if key in nets:
			components.extend(
				_series_loaded_io(
					name,
					key.upper(),
					nets[key],
					reference,
					series_ohm=float(params.get("series_ohm", 33.0)),
					heat=heat,
					ambient=ambient,
				)
			)
			interface_name = key.split("_")[0] if "_" in key else key
			if interface_name not in interfaces:
				interfaces.append(interface_name)
	components.extend(_package_thermal_network(name, heat, ambient, float(params.get("thermal_resistance", 9.5)), float(params.get("thermal_capacity", 5.2))))
	notes = [f"{name}: {family} 采用接口级模型，覆盖常见供电/复位/串行总线，不做指令级 CPU 内核仿真。"]
	interfaces = interfaces or ["gpio"]
	if "thermal" not in interfaces:
		interfaces.append("thermal")
	return {"components": components, "interfaces": interfaces, "notes": notes}


def _build_stm32f103c8_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	return _build_generic_mcu_interface(name, "STM32F103C8", nets, params, 4, [("nrst", "NRST", 10000.0), ("boot0", "BOOT0", 100000.0)])


def _build_stm32f407_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	return _build_generic_mcu_interface(name, "STM32F407", nets, params, 8, [("nrst", "NRST", 10000.0), ("boot0", "BOOT0", 100000.0)])


def _build_at89c51_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	return _build_generic_mcu_interface(name, "AT89C51", nets, params, 2, [("reset", "RST", 10000.0)])


def _build_raspberry_pi_40pin_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	return _build_generic_mcu_interface(name, "Raspberry Pi 40Pin", nets, params, 6, [("en", "GLOBAL_EN", 10000.0)])


def _build_arduino_uno_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	return _build_generic_mcu_interface(name, "Arduino UNO", nets, params, 3, [("reset", "RESET", 10000.0)])


def _build_arduino_mega_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	return _build_generic_mcu_interface(name, "Arduino Mega 2560", nets, params, 5, [("reset", "RESET", 10000.0)])


def _build_esp32_devkit_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	return _build_generic_mcu_interface(name, "ESP32 DevKit", nets, params, 5, [("en", "EN", 10000.0), ("boot0", "IO0", 10000.0)])


def _build_esp8266_nodemcu_interface(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	return _build_generic_mcu_interface(name, "ESP8266 NodeMCU", nets, params, 3, [("en", "EN", 10000.0), ("boot0", "GPIO0", 10000.0)])


def _build_passive_mesh_region(name: str, nets: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
	count = max(int(params.get("count", 300)), 3)
	supply = _net(nets, "supply", f"{name}_VIN")
	ground = _net(nets, "gnd", "GND")
	r_value = float(params.get("r_ohm", 22.0))
	c_value = float(params.get("c_f", 2.2e-7))
	l_value = float(params.get("l_h", 4.7e-6))
	components: list[dict[str, Any]] = []
	prev = supply
	stage_count = count // 3
	remainder = count % 3
	for index in range(stage_count):
		node = f"{name}_N{index:04d}"
		components.append(_component(f"{name}_R{index:04d}", "resistor", r_value * (1.0 + (index % 7) * 0.03), prev, node))
		components.append(_component(f"{name}_C{index:04d}", "capacitor", c_value * (1.0 + (index % 5) * 0.02), node, ground))
		components.append(_component(f"{name}_L{index:04d}", "inductor", l_value * (1.0 + (index % 3) * 0.05), node, ground))
		prev = node
	if remainder >= 1:
		end_node = f"{name}_TAIL"
		components.append(_component(f"{name}_RTAIL", "resistor", r_value, prev, end_node))
		prev = end_node
	if remainder >= 2:
		components.append(_component(f"{name}_CTAIL", "capacitor", c_value, prev, ground))
	return {"components": components, "interfaces": ["benchmark-region"]}


_BUILDERS = {
	"linear_actuator_stage": _build_linear_actuator_stage,
	"belt_drive_stage": _build_belt_drive_stage,
	"gear_train_pair": _build_gear_train_pair,
	"solenoid_actuator": _build_solenoid_actuator,
	"thermal_plate": _build_thermal_plate,
	"ldo_basic": _build_ldo_basic,
	"level_shifter_bidirectional": _build_level_shifter_bidirectional,
	"spi_peripheral_load": _build_spi_peripheral_load,
	"logic_buffer_multi_io": _build_logic_buffer_multi_io,
	"analog_switch_spst_load": _build_analog_switch_spst_load,
	"analog_mux_8ch_load": _build_analog_mux_8ch_load,
	"comparator_dual_open_collector": _build_comparator_dual_open_collector,
	"comparator_single_fast": _build_comparator_single_fast,
	"segment_display_4digit_common_anode": _build_segment_display_4digit_common_anode,
	"spi_eeprom_load": _build_spi_eeprom_load,
	"adc_frontend_rc": _build_adc_frontend_rc,
	"can_transceiver_terminated": _build_can_transceiver_terminated,
	"rs485_transceiver_terminated": _build_rs485_transceiver_terminated,
	"hbridge_dc_motor_interface": _build_hbridge_dc_motor_interface,
	"stm32f103c8_interface": _build_stm32f103c8_interface,
	"stm32f407_interface": _build_stm32f407_interface,
	"at89c51_interface": _build_at89c51_interface,
	"raspberry_pi_40pin_interface": _build_raspberry_pi_40pin_interface,
	"arduino_uno_interface": _build_arduino_uno_interface,
	"arduino_mega_interface": _build_arduino_mega_interface,
	"esp32_devkit_interface": _build_esp32_devkit_interface,
	"esp8266_nodemcu_interface": _build_esp8266_nodemcu_interface,
	"passive_mesh_region": _build_passive_mesh_region,
}