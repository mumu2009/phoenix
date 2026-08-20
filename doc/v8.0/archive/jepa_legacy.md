# 封存：jepa.* 配置组遗留（v7.6 改名 video/audio enc-dec 后的过渡层）

> 状态：**过渡封存**（v8.0）。`jepa.*` 作为 `video_model.cpp`/`audio_model.cpp` 的
> 向后兼容读取路径保留，但不再承载新配置；完全退役的触发条件见 §4。

## 1. 原始设想

- v7.5 前：JEPA-v2 图像/语音世界模型是活跃编码器（`jepa.image.*`、`jepa.speech.*`、
  `jepa.camera.*` 配置组），承担 image/audio ↔ concept 桥。
- v7.6 改名：JEPA-v2 → video/audio encoder-decoder 四组件（`video-encoder`/
  `video-decoder`/`audio-encoder`/`audio-decoder`），上游模型换成 LLaVA + 
  Qwen2-Audio，配置主源迁移到 `model_deployment.vision/speech.*`。

## 2. 封存原因

1. **双源配置**：同一后端同时读 `jepa.*` 与 `model_deployment.*` 两套键，语义重叠、
  容易漂移（本次审计已删 `jepa.speech.{variant,conceptDim,backend}` 三个零读取死叶）。
2. **仅兼容性读者**：`jepa.image.*` 只被 `video_model.cpp` 的遗留包装读
  （`frontend_server.cpp:4508` 的 `createVideoModel` 参数），`jepa.camera.*` 只被
  `/camera/analyze` 读；`jepa.speech.horizonModel/horizonDecoderModel` 只走 BPU 权重路径
  （`audio_model.cpp:142-143`）。
3. **理念**：可选模块默认关闭、配置单一来源；遗留过渡层不应永久驻留。

## 3. 保留在代码中的部分

- `jepa.image.variant`（video-encoder）、`jepa.speech.horizonModel/`
  `horizonDecoderModel`：BPU 权重路径与 `video_model.cpp` 兼容读取仍在用，保留。
- `jepa.camera.*`：`/camera/analyze` 路由依赖，保留。
- 运行时目录名 `ijepa/`、`additive_jepa/`：磁盘布局向后兼容，不属本封存范围。

## 4. 退役触发条件

同时满足：① `model_deployment.vision/speech.*` 完全接管 enc-dec 四组件的变体/维度/
后端选择（`video_model.cpp`/`audio_model.cpp` 不再读任何 `jepa.*` 键）；
② `/camera/analyze` 迁移到 `model_deployment` 命名；③ 部署矩阵 649 配置全绿后，
删除 `jepa.*` 全组并更新本档案。

## 5. 实现记录

- v8.0：审计判定过渡封存（`doc/v8.0/config_world_audit.md` C3）；删除 3 个零读取
  `jepa.speech` 死叶（variant/conceptDim/backend）；本文档建档。
