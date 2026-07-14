# v5.2 API 文档

## 1. 说明

本文件整理 v5.2 已公开或已在仓库中约定的接口，并统一参数、返回值、异常和兼容性策略。接口分为两类：

1. 本地前端服务已直接注册的接口
2. 通过 /api/* 网关透传、但已在 README 中公开的接口

## 2. 通用约定

### 2.1 请求

1. 默认使用 JSON 请求体。
2. 文本字段统一使用 UTF-8。
3. sessionId 若未传入，可由服务端生成。

### 2.2 成功返回

典型格式：

```json
{
  "ok": true,
  "result": {},
  "sessionId": "optional",
  "elapsedMs": 12
}
```

### 2.3 失败返回

典型格式：

```json
{
  "ok": false,
  "error": "missing-field",
  "message": "Missing text"
}
```

### 2.4 异常码约定

1. 400：缺少必需参数或请求体非法
2. 401：鉴权失败
3. 404：资源不存在
4. 429：限流或上游拥塞
5. 500：服务内部异常或上游异常

## 3. 本地已注册接口

### 3.1 POST /v51/chat

用途：上下文写入、v5.1 运行时推理与可选学习。

请求参数：

1. text：必填，输入文本
2. sessionId：选填，会话 id
3. mode：选填，默认 auto
4. feedback：选填，数值型反馈，传入时自动触发 learn
5. learningRate：选填，学习率
6. keywords：选填，关键词数组

成功返回字段：

1. ok
2. sessionId
3. context
4. v51
5. learn，只有传 feedback 时才返回

异常：

1. 缺少 text 返回 400
2. 运行时异常返回 500，error 为 v51 chat exception

兼容性：

1. sessionId 允许服务端生成，兼容旧客户端
2. 未启用 v51 时，v51 字段返回 null

示例请求：

```json
{
  "text": "请解释滑动窗口的作用",
  "mode": "auto",
  "feedback": 0.8,
  "keywords": ["滑动窗口", "内存"]
}
```

示例响应：

```json
{
  "ok": true,
  "sessionId": "ctx-1742470000",
  "context": {
    "windowSize": 1
  },
  "v51": {
    "reply": "滑动窗口通过只保留局部精确状态来控制内存占用。"
  },
  "learn": {
    "ok": true
  }
}
```

### 3.2 POST /v51/process

用途：把请求体直接交给 v51Runtime.process。

请求参数：任意 JSON 对象，由运行时解释。

成功返回：运行时 process 结果。

异常：

1. 缺少 JSON body 返回 400
2. 运行时异常返回 500，error 为 v51 process exception

兼容性：作为低层处理接口，字段兼容性由 v51Runtime 保证。

### 3.3 POST /v51/learn

用途：对指定会话或样本执行学习更新。

请求参数：任意 JSON 对象，由运行时解释；通常应包含 sessionId、feedback、keywords、learningRate 等。

成功返回：运行时 learn 结果。

异常：

1. 缺少 JSON body 返回 400
2. 运行时异常返回 500，error 为 v51 learn exception

示例请求：

```json
{
  "sessionId": "ctx-1742470000",
  "feedback": 0.6,
  "learningRate": 0.01,
  "keywords": ["词元", "词素"]
}
```

### 3.4 GET /v51/status

用途：查询 v51 运行时状态。

请求参数：

1. sessionId：选填

成功返回：状态 JSON。

异常：运行时异常返回 500，error 为 v51 status exception。

### 3.5 POST /context/reset

用途：重置指定会话上下文。

请求参数：

1. sessionId：必填

成功返回字段：

1. sessionId
2. removed

异常：缺少 sessionId 返回 400。

### 3.6 GET /context/status

用途：查询指定会话上下文状态。

请求参数：

1. sessionId：必填，query 参数

成功返回：上下文状态 JSON。

异常：缺少 sessionId 返回 400。

### 3.7 POST /vision/analyze

用途：执行图像分析。

请求参数：

1. imageBase64：与 imagePath 二选一
2. imagePath：与 imageBase64 二选一

成功返回：视觉分析结果 JSON。

异常：

1. 缺少 JSON body 返回 400
2. pipeline 未就绪返回 500
3. 图像为空或不可解码返回 400

### 3.8 POST /speech/analyze

用途：分析音频内容。

请求参数：

1. audioBase64：必填

成功返回：音频分析结果 JSON。

异常：缺少 audioBase64 返回 400。

### 3.9 POST /speech/synthesize

用途：把文本转成语音结果。

请求参数：

1. text：必填
2. sampleRate：选填，默认 16000
3. speed：选填，默认 1.0
4. pitch：选填，默认 1.0

成功返回：合成结果 JSON。

异常：缺少 text 返回 400。

### 3.10 POST /speech/ingest

用途：音频识别、上下文写入和可选 v51 推理一体化入口。

请求参数：

1. audioBase64：必填
2. sessionId：选填
3. mode：选填

成功返回字段：

1. ok
2. speech
3. context
4. sessionId
5. v51

异常：缺少 audioBase64 返回 400。

## 4. README 已公开的网关接口

以下接口当前通过 /api/* 路径公开，参数多由后端或外部后端解释；v5.2 在文档层统一其最小契约。

### 4.1 模型生命周期接口

接口列表：

1. GET /api/model/lifecycle
2. POST /api/model/compress
3. POST /api/model/explain
4. POST /api/model/deploy
5. POST /api/model/update

最小请求约定：

1. modelId：模型标识
2. revision：选填，版本号或快照号
3. options：选填，扩展参数

成功返回：

1. ok
2. modelId
3. stage 或 action
4. result

兼容性：未识别字段必须忽略并透传到下游，避免破坏编排链路。

示例请求：

```json
{
  "modelId": "odin-main",
  "revision": "v5.2",
  "options": {
    "compress": true,
    "quant": "int8"
  }
}
```

示例响应：

```json
{
  "ok": true,
  "modelId": "odin-main",
  "action": "compress",
  "result": {
    "estimatedSizeRatio": 0.42,
    "estimatedSpeedup": 1.3
  }
}
```

### 4.2 集群与自治接口

接口列表：

1. GET /api/cluster/status
2. POST /api/cluster/nodes
3. POST /api/cluster/route
4. POST /api/cluster/feedback
5. GET /api/dataset/catalog
6. POST /api/dataset/register
7. POST /api/dataset/activate
8. GET /api/spider/autonomy/status
9. POST /api/spider/autonomy/adapt
10. GET /api/optimizer/autonomy/status
11. POST /api/optimizer/autonomy/iterate
12. POST /api/perf/profile
13. POST /api/gnn/upgrade
14. POST /api/transformer/upgrade

最小请求约定：

1. target：目标节点、数据集或组件名
2. action：执行动作
3. payload：业务负载
4. traceId：选填，链路跟踪 id

成功返回：

1. ok
2. target
3. applied 或 accepted
4. result

异常：上游自治模块不可达时返回 502 或 500，客户端应允许重试。

### 4.3 外部风格训练步接口

接口： POST /api/external_style/train_step

用途：把前端风格信号发送到 llamacpp 或 bitnet 的 style-adapter。

典型请求字段：

1. provider：必填，llamacpp 或 bitnet
2. text：必填，源文本
3. style：选填，风格标签
4. styleKeywords：选填，风格关键词数组
5. targetAlign：选填，目标对齐度
6. observedAlign：选填，观测对齐度
7. lossScale：选填，损失缩放系数
8. features.keywordDensity：选填，0 到 1
9. features.sentiment：选填，-1 到 1
10. features.punctuationDensity：选填，0 到 1
11. graphEdgeWeight：选填，图边权信号
12. graphResidual：选填，图残差信号

成功返回：

1. ok
2. provider
3. accepted
4. adapterState 或 result

兼容性：

1. 同时接受 features.* 与同名顶层字段，兼容旧适配服务
2. provider 不识别时必须返回明确错误码和错误信息

示例请求：

```json
{
  "provider": "llamacpp",
  "text": "请保持正式、技术型、简洁的回答风格",
  "style": "technical-concise",
  "styleKeywords": ["正式", "技术", "简洁"],
  "targetAlign": 0.9,
  "observedAlign": 0.4,
  "lossScale": 0.5,
  "features": {
    "keywordDensity": 0.33,
    "sentiment": 0.1,
    "punctuationDensity": 0.08
  }
}
```

### 4.4 能力与协议探测接口

接口列表：

1. GET /api/runtime/features
2. GET /api/provider/capabilities

关键返回字段：

1. outsidesLinkMode
2. shmProtocolVersion

兼容性策略：客户端应先读取这两个字段，再决定是否启用外部后端特性和共享内存协议扩展。

## 5. 兼容性总策略

1. 新增字段优先采用向后兼容扩展，不删除旧字段。
2. 已公开接口的成功返回应尽量保留 ok 字段。
3. 对网关透传接口，未识别字段默认透传，不在边缘层强校验。
4. 对实验性接口，应在返回中附带 stage、compat 或 experimental 标记。