# tools

本目录收纳开发辅助脚本与外部后端接入工具。

## 当前文件

- `download_gguf_model.cpp`：将远程模型自动下载到 `./GGUF_models/` 的独立 C++ 工具。
- `external_model_adapter.py`：外部模型适配脚本。
- `split_main_cpp.py`：拆分主文件的辅助脚本。
- `start_079_launcher.py`：079 图形化一键启动器，负责拼装 `phoenix_main.exe` 启动参数，并暴露 addon、memebarrier、mechanical mind 等常用开关。
- `stimulation`：板级自建仿真平台，直接读取 `catastrophe/eext_netlist_*.json`，抽取 R/C/L/D 子电路并执行工作点、瞬态仿真。
- `stimulation_gerber_platform.py`：Gerber + 探针 + ngspice 仿真主平台，结合 Gerber 铜层几何和 eext 元件清单生成 SPICE deck。

## 作用

1. 为大文件维护、外部模型桥接和工程辅助提供工具支持。
2. 降低手工操作成本，提升调试和重构效率。

## 079 GUI 启动器

- 根目录 `build_start_079_oneclick_exe.bat` 会把 `tools/start_079_launcher.py` 打包成 `start_079_oneclick.exe`。
- `start_079_oneclick.bat` 优先查找 exe，再回退到仓库内 Python 版 GUI。
- GUI 当前直接覆盖常见 CLI 参数，不替代 `--component-config`；复杂配置仍建议通过 component preset 文件注入。

## stimulation 用法

- 列出内置预设：`python tools/stimulation --list-presets`
- 仿真内置示例场景：`python tools/stimulation --preset servo_stage_impulse`
- 自定义场景：`python tools/stimulation --scenario your_scenario.json`

说明：当前仿真内核固定执行整板单体求解，不再进行区域截取、分块近似或分区结果加和。

### Gerber + Probe + ngspice 用法

- 输出场景模板：`python tools/stimulation --emit-gerber-template tools/stimulation_gerber_template.json`
- 基于场景准备 Gerber 探针仿真：`python tools/stimulation --gerber-scenario tools/stimulation_gerber_example.json --prepare-only`
- 审计 Gerber 飞针导出是否仍与 eext 网表一致：`python tools/stimulation --audit-flyingprobe-path catastrophe/Gerber_xxx.zip --audit-netlists catastrophe/eext_netlist_1.json catastrophe/eext_netlist_2.json ...`
- 若本机存在 `ngspice.exe`，可加 `--ngspice-exe <path>` 直接执行
- 仓库已附带最小示例：`tools/stimulation_gerber_example.json` + `tools/stimulation_gerber_example/camera_top.gtl`

Gerber 模式的核心输入不是只有 Gerber：

- `gerberRoot`：Gerber 文件夹或 zip
- `layers`：铜层文件映射
- `anchors`：把 PCB 上已知物理坐标锚定到逻辑网络名
- `probes`：探针位置
- `componentNetlists`：恢复 R/C/L/D 等元件电气语义，默认可直接复用 `catastrophe/eext_netlist_*.json`
- `sources` / `analyses`：激励与仿真任务

原因很直接：Gerber 自身不保存“这个封装上的电阻值是多少”，只保存几何，所以必须配合元件清单或 eext 网表才能做真正的电路仿真。

FlyingProbe 审计模式用于回答另一个前置问题：导出的实际 PCB 连线，是否仍然和 `eext_netlist_*.json` 里声明的引脚网络一致。它会输出缺失器件、额外器件和逐脚错网，适合在 DRC 通过之后继续做“网表语义”复核。

当前仓库内 `catastrophe/outsides/ngspice_start/DuSpiceStart.ini` 指向的历史 `ngspice.exe` 路径在本机上并不存在，因此默认建议先用 `--prepare-only` 验证坐标映射和 deck 生成；需要真正执行时，再通过 `--ngspice-exe` 显式指定实际可用的 `ngspice.exe`。

### 目前支持的器件与边界

- 支持：电阻、电容、电感、二极管、场景定义的独立电压源/电流源
- 输入源：直接读取 `catastrophe/eext_netlist_*.json`
- 输出：`build/stimulation/<scenario>/summary.md`、CSV 波形与已选器件 JSON
- 不支持：复杂 IC 的内部行为；这类器件会在报告里标记为“已忽略器件”，用于告诉你当前子电路求解到底基于哪些元件