# 封存：v7 四组未接线配置（memory / emotion_influence / async_pool / llama_server_mods）

> 状态：**已封存**（v8.0）。四个配置组从 `config/phoenix.json` 移除；
> 对应的模块代码保留并**以文档化默认值运行**（见 §3）。

## 1. 原始设想

v7.0 "Arthur" 引入 `module_overrides/` 的 ahead/memory/emotion_influence/
async_task_system 四模块，预期由 `v7.*` 配置组驱动：
- `v7.memory.*`：MemoryModule（会话记忆）参数；
- `v7.emotion_influence.*`：EmotionInfluence（情绪→采样调制，baseTemperature 等）参数；
- `v7.async_pool.*`：AsyncTaskSystem（异步任务池）参数；
- `v7.llama_server_mods.*`：llama.cpp 拆分补丁参数。

## 2. 封存原因

1. **配置从未接线**：审计实测四个组没有任何 cfg/cfgOr/resolveConfig 读取——
   模块只以硬编码默认值构造（AheadModule 构造 MemoryModule()、
   EmotionInfluence(EmotionInfluence::Config{}) 等），配置键是死配置，只会误导调参者。
2. **模块本身已接线**（区别关键）：ahead 管线实际消费 MemoryModule 的
   memoryForBackend（[Ahead memory] 块）与 EmotionInfluence 的采样/调制输出
   （inferenceOptions、promptModulation、logitBiases）——封存的是**配置面**，不是模块。
3. v7.llama_server_mods.* 的拆分补丁本身已因生成质量缺陷封存（见
   doc/v8.0/migration_backlog.md B2 节）。

## 3. 保留在代码中的部分（文档化默认值）

| 模块 | 运行默认值 | 来源 |
|---|---|---|
| MemoryModule | 默认构造 | ahead_module.cpp |
| EmotionInfluence | baseTemperature=0.7，clamp [0.1,1.5]，temperature = base + 0.35*arousal - 0.15*fear | emotion_influence.hpp/cpp |
| AsyncTaskSystem | 默认构造 | ahead_module.cpp（当前未被 ahead 管线直接驱动） |

## 4. 重启条件

任一满足即可恢复对应配置组：① 出现明确的多部署调参需求且有人提交接线
（AheadConfig::fromJson 读取 v7.* 子块并传给模块构造）；
② llama_server_mods 拆分补丁修复后恢复 v7.llama_server_mods.*。

## 5. 实现记录

- v8.0：审计（doc/v8.0/config_world_audit.md C2）删除四组共 22 键（零读取）；本文档建档。
