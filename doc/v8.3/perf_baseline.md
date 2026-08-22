# v8.x 性能基线（诊断报告模板 - 由测试者填数）

**问题**：780 token / 24 min（≈0.54 t/s），不可接受。

## 1. 环境事实（已确认）
- RDK X5：8×Cortex-A55，无 dotprod/i8mm SIMD；8B Q4_K_M decode 实测 ≈1.5-2 s/token（运行脚本注释）。
- 模型 4.7G mmap 自 USB（exFAT），RAM 6.9G、常驻 2.5G → 权重页反复换页。
- deliberator 每 tick 独立 HTTP 请求全量 prefill；llama-server slot 前缀缓存可用但旧 prompt 静态/动态混杂无法命中。

## 2. 测量清单（测试者在 RDK 执行并填表）

### 2.1 tick 时间分解（llama-server 日志）
| 项 | 方法 | 实测值 |
|---|---|---|
| prefill 时间 | llama-server `--verbose` 日志的 prompt eval 计时（`eval time`） | 待填 |
| decode 时间 | 同日志 `eval time` per token | 待填 |
| 排队/串行等待 | gateway 日志 `[chat] llamacpp` 行间隔 | 待填 |
| 单 tick 总耗时 | mission 日志间隔 | 待填 |

### 2.2 USB 换页实测
```bash
# 驻留页 vs 模型大小
cat /proc/$(pgrep -f llama-server | head -1)/smaps_rollup | grep -E 'Rss|Pss'
ls -la GGUF_models/blobs/  # 模型真实字节数
# 试 --mlock 全驻留（注意 OOM 风险）：
LLAMA_MLOCK=1 SKIP_BUILD=1 bash tools/rdk_netboot_build_and_run.sh
# 或小量化：
LLAMA_MODEL=$ROOT/GGUF_models/<q3km路径> SKIP_BUILD=1 bash tools/rdk_netboot_build_and_run.sh
```
| 配置 | 内存 | decode s/token | 结论 |
|---|---|---|---|
| 现状 (mmap USB) | | | |
| --mlock | | | |
| Q3_K_M | | | |

### 2.3 优化后对比（同任务两次运行）
| 版本 | 总耗时 | 说明 |
|---|---|---|
| 优化前（v8.3 基线） | 24 min / 780 tok | |
| 静态前缀+瘦身+L1缓存后 | 待填 | 目标 ≥4x |
| + mlock/Q3 后 | 待填 | |
| + 并行解码（若可用）后 | 待填 | 目标合计 ≥8x |

### 2.4 并行解码接受率（§A5）
| N | 接受率 | 加速比 | 结论 |
|---|---|---|---|
| 2 | | | |
| 4 | | | |
| 8 | | | |

## 3. 判定标准
- prefill 占比 >30% → 静态前缀前置是主收益，重点验证 slot 前缀缓存命中（llama-server 日志 `n_past` 复用）。
- decode 占主导 → 只靠 mlock/量化/并行解码；**单 token 模式在 A55 上无法质变**（硬件级 2s/token）。
- L2 查表基准跑 `tools/layer_lookup_probe.cpp` 并填 §lookup_acceleration.md 的数据表。
