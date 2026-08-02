import importlib.util
import pathlib
import sys
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "stimulation_catastrophe1_suite" / "run_npu_throughput_budget.py"
SPEC = importlib.util.spec_from_file_location("phoenix_npu_throughput_budget", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample_boundary_report() -> dict:
    return {
        "input_boundary": {
            "simulated": {
                "t63_s": 3.6053319910871504e-05,
                "t90_s": 1.1959207472583697e-04,
                "t99_s": 8.185759629120883e-03,
                "settle_1pct_s": 6.8215e-03,
                "settle_0p1pct_s": 1.43065e-02,
            }
        },
        "board_connectivity": {
            "dac_q0_cs0": {
                "only_passives_and_pads": True,
                "members": ["PAD774_1", "R580_1"],
            },
            "adc_q0_sdo0": {
                "only_passives_and_pads": True,
                "members": ["PAD805_2", "R596_2"],
            },
            "adc_q0_convst": {
                "only_passives_and_pads": True,
                "members": ["PAD752_1", "R560_1"],
            },
        },
    }


def sample_eext_fast_boundary_netlist() -> dict:
    payload: dict[str, dict] = {}
    for index in range(8):
        payload[f"u{90 + index}"] = {
            "props": {"Designator": f"U{90 + index}", "device_name": "74HC595D"},
            "pins": {
                "10": "GPIO7_CE1_SD_CS" if index < 6 else "GPIO22_NPU_CS",
                "11": "GPIO10_MOSI",
                "12": "GPIO8_CE0_NPU_CS0",
                "14": "GPIO11_SCLK",
                "9": "NPU_DAC_SHIFT_TAP",
                "15": f"DAC_Q{index % 4}_CS0" if index < 4 else f"ADC_Q{index - 4}_SDO0",
            },
        }
    payload["u98"] = {
        "props": {"Designator": "U98", "device_name": "MCP3208-CI/SL"},
        "pins": {
            "10": "GPIO22_NPU_CS",
            "11": "GPIO10_MOSI",
            "12": "GPIO9_MISO",
            "13": "GPIO11_SCLK",
            "16": "3V3",
            "1": "ADC_Q0_SDO0",
        },
    }
    payload["r964"] = {
        "props": {"Designator": "R964", "device_name": "", "value": "220"},
        "pins": {"1": "GPIO6_NPU_SAMPLE_SYNC", "2": "ADC_Q0_CONVST"},
    }
    return payload


def sample_stack_carrier_netlist(carrier_name: str, designator_base: int) -> dict:
    payload: dict[str, dict] = {}
    resistor_base = designator_base * 10
    carrier_prefix = f"CARR{carrier_name}"
    for index in range(8):
        payload[f"r{resistor_base + index}"] = {
            "props": {
                "Designator": f"R{resistor_base + index}",
                "value": "4.7k",
                "Supplier Part": "C25804",
            },
            "pins": {
                "1": f"{carrier_prefix}_SLOT{index}_CMP0",
                "2": f"{carrier_prefix}_SUM0",
            },
        }
    for index in range(8):
        payload[f"r{resistor_base + 8 + index}"] = {
            "props": {
                "Designator": f"R{resistor_base + 8 + index}",
                "value": "4.7k",
                "Supplier Part": "C25804",
            },
            "pins": {
                "1": f"{carrier_prefix}_SLOT{index}_CMP1",
                "2": f"{carrier_prefix}_SUM1",
            },
        }
    payload[f"r{resistor_base + 16}"] = {
        "props": {"Designator": f"R{resistor_base + 16}", "value": "10k", "Supplier Part": "C25804"},
        "pins": {"1": f"{carrier_prefix}_SUM0", "2": "AGND"},
    }
    payload[f"r{resistor_base + 17}"] = {
        "props": {"Designator": f"R{resistor_base + 17}", "value": "10k", "Supplier Part": "C25804"},
        "pins": {"1": f"{carrier_prefix}_SUM1", "2": "AGND"},
    }
    payload[f"c{resistor_base}"] = {
        "props": {"Designator": f"C{resistor_base}", "value": "1nF", "Supplier Part": "C15849"},
        "pins": {"1": f"{carrier_prefix}_SUM0", "2": "AGND"},
    }
    payload[f"c{resistor_base + 1}"] = {
        "props": {"Designator": f"C{resistor_base + 1}", "value": "1nF", "Supplier Part": "C15849"},
        "pins": {"1": f"{carrier_prefix}_SUM1", "2": "AGND"},
    }
    return payload


def sample_stack_bridge_netlist() -> dict:
    return {
        "u160": {
            "props": {"Designator": "U160", "device_name": "MCP3208-CI/SL", "Supplier Part": "C16939"},
            "pins": {
                "10": "GPIO22_NPU_CS",
                "11": "GPIO10_MOSI",
                "12": "GPIO9_MISO",
                "13": "GPIO11_SCLK",
                "14": "GPIO18_I2S_BCLK",
                "1": "CARRA_SUM0",
                "2": "CARRB_SUM0",
            },
        },
        "u161": {
            "props": {"Designator": "U161", "device_name": "MCP3208-CI/SL", "Supplier Part": "C16939"},
            "pins": {
                "10": "GPIO22_NPU_CS",
                "11": "GPIO10_MOSI",
                "12": "GPIO9_MISO",
                "13": "GPIO11_SCLK",
                "14": "GPIO19_I2S_LRCLK",
                "1": "CARRA_SUM1",
                "2": "CARRB_SUM1",
            },
        },
        "u162": {
            "props": {"Designator": "U162", "device_name": "LM393BIDR", "Supplier Part": "C2865059"},
            "pins": {
                "1": "GPIO5_EXT0",
                "7": "GPIO12_EXT_PWM0",
                "2": "CARRA_SUM0",
                "6": "CARRB_SUM0",
            },
        },
        "u163": {
            "props": {"Designator": "U163", "device_name": "LM393BIDR", "Supplier Part": "C2865059"},
            "pins": {
                "1": "GPIO13_EXT_PWM1",
                "7": "GPIO16_NPU_DRDY",
                "2": "CARRC_SUM0",
                "6": "CARRD_SUM0",
            },
        },
        "u164": {
            "props": {"Designator": "U164", "device_name": "LM393BIDR", "Supplier Part": "C2865059"},
            "pins": {
                "1": "GPIO20_I2S_DIN",
                "7": "GPIO21_I2S_DOUT",
                "2": "CARRE_SUM0",
                "5": "CARRE_SUM1",
                "6": "CARRF_SUM0",
                "8": "CARRF_SUM1",
            },
        },
    }


def sample_stack_module_control_netlist() -> dict:
    return {
        "u177": {
            "props": {"Designator": "U177", "device_name": "LM393BIDR", "Supplier Part": "C2865059"},
            "pins": {"1": "MOD_CMP0", "2": "SLICE_SUM0", "3": "MOD_REF", "5": "MOD_REF", "6": "SLICE_SUM1", "7": "MOD_CMP1", "8": "3V3"},
        },
        "u178": {
            "props": {"Designator": "U178", "device_name": "74VHC4051AFT", "Supplier Part": "C146324"},
            "pins": {"3": "SLICE_SUM0", "9": "GPIO25_NPU_RST_N", "10": "GPIO27_TILE_SEL", "11": "GPIO17_SYNC_REQ", "12": "SLICE_GUARD1", "13": "SLICE_GUARD0", "16": "3V3"},
        },
        "r1700": {"props": {"Designator": "R1700", "value": "10k", "Supplier Part": "C25804"}, "pins": {"1": "MOD_REF", "2": "AGND"}},
        "r1701": {"props": {"Designator": "R1701", "value": "10k", "Supplier Part": "C25804"}, "pins": {"1": "MOD_CMP0", "2": "AGND"}},
        "r1702": {"props": {"Designator": "R1702", "value": "10k", "Supplier Part": "C25804"}, "pins": {"1": "MOD_CMP1", "2": "AGND"}},
        "r1703": {"props": {"Designator": "R1703", "value": "10k", "Supplier Part": "C25804"}, "pins": {"1": "GPIO26_MOTOR_ESTOP", "2": "3V3"}},
        "c1700": {"props": {"Designator": "C1700", "value": "100nF", "Supplier Part": "C1525"}, "pins": {"1": "3V3", "2": "AGND"}},
        "c1701": {"props": {"Designator": "C1701", "value": "100nF", "Supplier Part": "C1525"}, "pins": {"1": "P5VA", "2": "AGND"}},
        "c1702": {"props": {"Designator": "C1702", "value": "100nF", "Supplier Part": "C1525"}, "pins": {"1": "N5VA", "2": "AGND"}},
    }


def sample_stack_module_slice_netlist() -> dict:
    payload: dict[str, dict] = {}
    for index in range(16):
        payload[f"r18{index:02d}"] = {
            "props": {"Designator": f"R18{index:02d}", "value": "10k", "Supplier Part": "C25804"},
            "pins": {"1": "MOD_REF", "2": f"SLICE_NODE{index:02d}"},
        }
        payload[f"c18{index:02d}"] = {
            "props": {"Designator": f"C18{index:02d}", "value": "1nF", "Supplier Part": "C15849"},
            "pins": {"1": f"SLICE_NODE{index:02d}", "2": "AGND"},
        }
    return payload


class NpuThroughputBudgetTests(unittest.TestCase):
    def test_parse_spi_speed_hz_from_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            header_path = pathlib.Path(temp_dir) / "edge_platform.hpp"
            header_path.write_text("struct RuntimeConfig { int npuSpiSpeedHz{24000000}; };\n", encoding="utf-8")

            speed_hz = MODULE.parse_spi_speed_hz(header_path)

        self.assertEqual(speed_hz, 24000000)

    def test_int8_equivalent_scales_match_current_precision_model(self) -> None:
        scales = MODULE.derive_int8_equivalent_scales(input_bits=4, weight_bits=6, output_bits=1)

        self.assertAlmostEqual(scales["inputScaleVsInt8"], 0.5)
        self.assertAlmostEqual(scales["weightScaleVsInt8"], 0.75)
        self.assertAlmostEqual(scales["outputScaleVsInt8"], 0.125)
        self.assertAlmostEqual(scales["multiplyPrecisionScaleVsInt8"], 0.375)
        self.assertAlmostEqual(scales["endToEndScaleVsInt8"], 0.046875)

    def test_int8_equivalent_scaling_distinguishes_raw_and_precision_normalized_tops(self) -> None:
        scaled = MODULE.apply_int8_equivalent_scaling(
            14.641933963636363,
            MODULE.derive_int8_equivalent_scales(input_bits=4, weight_bits=6, output_bits=1),
        )

        self.assertAlmostEqual(scaled["rawOpCountEquivalentTops"], 14.641933963636363)
        self.assertAlmostEqual(scaled["multiplyPrecisionNormalizedTops"], 5.490725236363637)
        self.assertAlmostEqual(scaled["endToEndNormalizedTops"], 0.6863406545454546)

    def test_fp16_equivalent_scaling_distinguishes_raw_and_bit_budget_values(self) -> None:
        scaled = MODULE.apply_fp16_equivalent_scaling(
            14.641933963636363,
            MODULE.derive_fp16_equivalent_scales(input_bits=4, weight_bits=6, output_bits=1),
        )

        self.assertAlmostEqual(scaled["rawOpCountEquivalentTops"], 14.641933963636363)
        self.assertAlmostEqual(scaled["multiplyPrecisionNormalizedTops"], 1.3726813090909092)
        self.assertAlmostEqual(scaled["endToEndNormalizedTops"], 0.08579258181818182)

    def test_detect_controller_connectivity_flags_passive_only_nets(self) -> None:
        connected, blockers = MODULE.detect_controller_connectivity(sample_boundary_report())

        self.assertFalse(connected)
        self.assertTrue(any("dac_q0_cs0" in blocker for blocker in blockers))
        self.assertTrue(any("adc_q0_sdo0" in blocker for blocker in blockers))

    def test_detect_eext_fast_boundary_connectivity_accepts_low_cost_gpio_netlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            netlist_path = pathlib.Path(temp_dir) / "eext_netlist_9.json"
            netlist_path.write_text(__import__("json").dumps(sample_eext_fast_boundary_netlist()), encoding="utf-8")

            connected, blockers, evidence = MODULE.detect_eext_fast_boundary_connectivity(netlist_path)

        self.assertTrue(connected)
        self.assertEqual(blockers, [])
        self.assertTrue(any("8x74HC595D" in item for item in evidence))

    def test_compute_budget_case_matches_expected_spi_t90_budget(self) -> None:
        scenario = MODULE.Scenario(
            name="flywire_single_read",
            description="test",
            read_passes=1,
            weight_reloads_per_inference=0,
            controller_connected=True,
        )
        transport_model = MODULE.TransportModel(
            name="spi_default",
            description="test",
            kind="spi",
            rate_hz=12000000,
        )

        case = MODULE.compute_budget_case(
            window_name="t90",
            analog_window_s=1.1959207472583697e-04,
            scenario=scenario,
            transport_model=transport_model,
            matrix_rows=32,
            matrix_cols=32,
            ops_per_mac=2,
            input_bits=3,
            output_bits=1,
            weight_bits=6,
        )

        expected_transport_s = ((12 + 4) * 8.0) / 12000000.0
        expected_total_s = 1.1959207472583697e-04 + expected_transport_s
        expected_tops = (32 * 32 * 2) / expected_total_s / 1e12
        self.assertAlmostEqual(case["transportSeconds"], expected_transport_s)
        self.assertAlmostEqual(case["utilizableTops"], expected_tops)

    def test_compute_budget_case_matches_expected_gpio_t90_budget(self) -> None:
        scenario = MODULE.Scenario(
            name="flywire_single_read",
            description="test",
            read_passes=1,
            weight_reloads_per_inference=0,
            controller_connected=True,
        )
        transport_model = MODULE.TransportModel(
            name="gpio_max",
            description="test",
            kind="gpio",
            rate_hz=125000000,
            gpio_write_transactions_per_value=1,
            gpio_read_transactions_per_value=2,
            gpio_weight_transactions_per_cell=2,
        )

        case = MODULE.compute_budget_case(
            window_name="t90",
            analog_window_s=1.1959207472583697e-04,
            scenario=scenario,
            transport_model=transport_model,
            matrix_rows=32,
            matrix_cols=32,
            ops_per_mac=2,
            input_bits=3,
            output_bits=1,
            weight_bits=6,
        )

        expected_transport_s = (32 + 32 * 2) / 125000000.0
        expected_total_s = 1.1959207472583697e-04 + expected_transport_s
        expected_vectors = 1.0 / expected_total_s
        self.assertAlmostEqual(case["transportSeconds"], expected_transport_s)
        self.assertAlmostEqual(case["utilizableVectorsPerSecond"], expected_vectors)
        self.assertAlmostEqual(case["opsPerTransportCycle"], 2048.0 / 96.0)
        self.assertAlmostEqual(case["transportOnlyVectorsPerSecond"], 125000000.0 / 96.0)

    def test_build_report_forces_real_board_to_zero_when_disconnected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            header_path = pathlib.Path(temp_dir) / "edge_platform.hpp"
            header_path.write_text("struct RuntimeConfig { int npuSpiSpeedHz{12000000}; };\n", encoding="utf-8")
            netlist_path = pathlib.Path(temp_dir) / "missing_eext_netlist_9.json"

            report = MODULE.build_report(
                boundary_report=sample_boundary_report(),
                boundary_report_path=pathlib.Path(temp_dir) / "boundary.json",
                edge_platform_header=header_path,
                eext_fast_boundary_netlist=netlist_path,
                matrix_rows=32,
                matrix_cols=32,
                input_bits=3,
                output_bits=1,
                weight_bits=6,
                ops_per_mac=2,
            )

        real_board = next(item for item in report["budgets"] if item["name"] == "real_board")
        self.assertFalse(real_board["controllerConnected"])
        for transport_budget in real_board["transportBudgets"]:
            self.assertTrue(all(float(case["utilizableTops"]) == 0.0 for case in transport_budget["windows"]))
            self.assertTrue(all(float(case["utilizableVectorsPerSecond"]) == 0.0 for case in transport_budget["windows"]))

    def test_detect_stack_cluster_topology_counts_carriers_and_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            for page_index, carrier_name in zip(range(10, 16), ["A", "B", "C", "D", "E", "F"]):
                (root / f"eext_netlist_{page_index}.json").write_text(
                    __import__("json").dumps(sample_stack_carrier_netlist(carrier_name, page_index * 10)),
                    encoding="utf-8",
                )
            (root / "eext_netlist_16.json").write_text(__import__("json").dumps(sample_stack_bridge_netlist()), encoding="utf-8")
            (root / "eext_netlist_17.json").write_text(__import__("json").dumps(sample_stack_module_control_netlist()), encoding="utf-8")
            (root / "eext_netlist_18.json").write_text(__import__("json").dumps(sample_stack_module_slice_netlist()), encoding="utf-8")

            topology = MODULE.detect_stack_cluster_topology(root)

        self.assertTrue(topology["connected"])
        self.assertEqual(topology["carrierCount"], 6)
        self.assertEqual(topology["moduleTileCount"], 48)
        self.assertEqual(topology["moduleTilesPerCarrier"], 8)
        self.assertTrue(topology["moduleTemplate"]["connected"])

    def test_build_report_includes_stack_cluster_budget_above_ten_tops(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            header_path = root / "edge_platform.hpp"
            header_path.write_text("struct RuntimeConfig { int npuSpiSpeedHz{12000000}; };\n", encoding="utf-8")
            netlist_path = root / "eext_netlist_9.json"
            netlist_path.write_text(__import__("json").dumps(sample_eext_fast_boundary_netlist()), encoding="utf-8")
            for page_index, carrier_name in zip(range(10, 16), ["A", "B", "C", "D", "E", "F"]):
                (root / f"eext_netlist_{page_index}.json").write_text(
                    __import__("json").dumps(sample_stack_carrier_netlist(carrier_name, page_index * 10)),
                    encoding="utf-8",
                )
            (root / "eext_netlist_16.json").write_text(__import__("json").dumps(sample_stack_bridge_netlist()), encoding="utf-8")
            (root / "eext_netlist_17.json").write_text(__import__("json").dumps(sample_stack_module_control_netlist()), encoding="utf-8")
            (root / "eext_netlist_18.json").write_text(__import__("json").dumps(sample_stack_module_slice_netlist()), encoding="utf-8")

            report = MODULE.build_report(
                boundary_report=sample_boundary_report(),
                boundary_report_path=root / "boundary.json",
                edge_platform_header=header_path,
                eext_fast_boundary_netlist=netlist_path,
                stack_cluster_netlist_root=root,
                matrix_rows=32,
                matrix_cols=32,
                input_bits=3,
                output_bits=1,
                weight_bits=6,
                ops_per_mac=2,
            )

        stack_budget = report["stackCluster"]["budget"]
        stack_cost = report["stackCluster"]["cost"]
        self.assertTrue(stack_budget["connected"])
        self.assertGreater(float(stack_budget["utilizableTops"]), 1000.0)
        self.assertLess(float(stack_budget["utilizableTops"]), 1200.0)
        self.assertEqual(int(stack_budget["slowCalibrationTransactionsPerWindow"]), 3)
        self.assertFalse(bool(stack_budget["hotPathIncludesSlowCalibration"]))
        self.assertTrue(bool(stack_budget["frequencyScaling"]["transportScalingIsLinear"]))
        self.assertIsNone(stack_budget["frequencyScaling"]["asymptoticTopsAtInfiniteTransportHz"])
        self.assertTrue(stack_budget["fp16Equivalent"]["raw12TopsTargetSatisfied"])
        self.assertTrue(stack_cost["icCostRuleSatisfied"])
        self.assertGreater(float(stack_cost["estimatedClusterCostRmb"]), float(stack_cost["estimatedControlPlaneCostRmb"]))

    def test_markdown_mentions_zero_utilizable_tops_for_real_board(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            header_path = root / "edge_platform.hpp"
            header_path.write_text("struct RuntimeConfig { int npuSpiSpeedHz{12000000}; };\n", encoding="utf-8")
            netlist_path = root / "eext_netlist_9.json"
            netlist_path.write_text(__import__("json").dumps(sample_eext_fast_boundary_netlist()), encoding="utf-8")
            for page_index, carrier_name in zip(range(10, 16), ["A", "B", "C", "D", "E", "F"]):
                (root / f"eext_netlist_{page_index}.json").write_text(
                    __import__("json").dumps(sample_stack_carrier_netlist(carrier_name, page_index * 10)),
                    encoding="utf-8",
                )
            (root / "eext_netlist_16.json").write_text(__import__("json").dumps(sample_stack_bridge_netlist()), encoding="utf-8")
            (root / "eext_netlist_17.json").write_text(__import__("json").dumps(sample_stack_module_control_netlist()), encoding="utf-8")
            (root / "eext_netlist_18.json").write_text(__import__("json").dumps(sample_stack_module_slice_netlist()), encoding="utf-8")

            report = MODULE.build_report(
                boundary_report=sample_boundary_report(),
                boundary_report_path=root / "boundary.json",
                edge_platform_header=header_path,
                eext_fast_boundary_netlist=netlist_path,
                stack_cluster_netlist_root=root,
                matrix_rows=32,
                matrix_cols=32,
                input_bits=3,
                output_bits=1,
                weight_bits=6,
                ops_per_mac=2,
            )

        markdown = MODULE.render_markdown(report)
        self.assertIn("- 连通性判定来源：eext-netlist", markdown)
        self.assertIn("## 时序口径", markdown)
        self.assertIn("## Stack-Cluster 预算", markdown)
        self.assertIn("stack evidence:", markdown)
        self.assertIn("成本守卫状态：通过", markdown)
        self.assertIn("最终有效口径：板级 GPIO 协议上限 120000000 Hz", markdown)
        self.assertIn("edge_platform.hpp 当前 SPI 默认值仍是 12000000 Hz", markdown)
        self.assertIn("FP16 对标：raw op-count 对齐约", markdown)
        self.assertIn("架构扩频：12000000 Hz ->", markdown)
        self.assertIn("算力随外部时钟近似线性增长", markdown)
        self.assertIn("## 传输模型：gpio_max", markdown)
        self.assertIn("### 控制摊销", markdown)
        self.assertIn("eext-netlist evidence:", markdown)
        self.assertIn("| real_board | t63 |", markdown)
        self.assertIn("| real_board | GPIO事务 | 96 |", markdown)


if __name__ == "__main__":
    unittest.main()