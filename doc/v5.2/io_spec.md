# v5.2 输入输出规范

## 1. 目标

本文件定义 v5.2 转换算法的输入输出格式，覆盖 GGUF 权重、词表、分词映射、神经动力学参数表和运行时配置。

## 2. 输入规范

### 2.1 GGUF 模型输入

输入对象名称： ggufModel

必需字段：

1. modelPath：GGUF 文件路径
2. architecture：模型架构标识，例如 transformer
3. tensorIndex：张量索引表
4. metadata：GGUF 元数据

推荐结构：

```json
{
  "modelPath": "GGUF_models/odin.gguf",
  "architecture": "transformer",
  "tensorIndex": {
    "embedding": ["token_embd.weight"],
    "attention": ["blk.0.attn_q.weight", "blk.0.attn_k.weight"],
    "mlp": ["blk.0.ffn_up.weight", "blk.0.ffn_down.weight"],
    "norm": ["blk.0.attn_norm.weight"]
  },
  "metadata": {
    "n_layers": 32,
    "n_embd": 4096,
    "tokenizer": "bpe"
  }
}
```

### 2.2 词表输入

输入对象名称： vocabulary

必需字段：

1. tokens：按 id 排列的词元表
2. specialTokens：特殊词元定义
3. tokenizerType：分词器类型

推荐结构：

```json
{
  "tokenizerType": "bpe",
  "tokens": ["<bos>", "你", "好"],
  "specialTokens": {
    "bos": 0,
    "eos": 1,
    "unk": 2
  }
}
```

### 2.3 语义重分辨率映射输入

输入对象名称： semanticMapping

必需字段：

1. sourceTokenId：原始词元 id
2. sourceToken：原始词元文本
3. semanticUnits：拆分后的语义最小单位数组
4. strategy：拆分或合并策略

推荐结构：

```json
{
  "sourceTokenId": 1532,
  "sourceToken": "正在进行",
  "strategy": "split-by-morpheme",
  "semanticUnits": [
    {"id": "sem:zheng", "text": "正", "role": "aspect"},
    {"id": "sem:zai", "text": "在", "role": "progressive"},
    {"id": "sem:jinxing", "text": "进行", "role": "action"}
  ]
}
```

允许策略：

1. identity
2. split-by-morpheme
3. merge-by-similarity
4. context-conditional-split

### 2.4 拟合数据输入

输入对象名称： fitSample

必需字段：

1. inputText：原始输入文本
2. referenceOutput：原模型输出或标注答案
3. sourceHiddenStats：原模型中间统计量，可选但推荐
4. taskType：任务类型

## 3. 输出规范

### 3.1 动力学参数表输出

输出对象名称： neuroDynamicsTable

必需字段：

1. semanticUnitId：语义最小单位 id
2. modelType：lif 或 hh
3. parameters：参数字典
4. fittedFrom：来源权重或样本范围
5. constraints：参数约束信息

LIF 风格示例：

```json
{
  "semanticUnitId": "sem:jinxing",
  "modelType": "lif",
  "parameters": {
    "I_th": 0.42,
    "I_reset": 0.18,
    "tau": 6.5
  },
  "fittedFrom": ["blk.7.attn_q.weight", "blk.7.ffn_up.weight"],
  "constraints": {
    "tau": "> 0",
    "I_th": "> I_reset"
  }
}
```

HH 风格示例：

```json
{
  "semanticUnitId": "sem:jinxing",
  "modelType": "hh",
  "parameters": {
    "C": 1.0,
    "g_Na": 120.0,
    "g_K": 36.0,
    "g_L": 0.3,
    "E_Na": 50.0,
    "E_K": -77.0,
    "E_L": -54.4
  },
  "fittedFrom": ["blk.7.attn_v.weight"],
  "constraints": {
    "C": "> 0",
    "g_Na": ">= 0",
    "g_K": ">= 0",
    "g_L": ">= 0"
  }
}
```

### 3.2 拟合结果输出

输出对象名称： fitResult

必需字段：

1. loss
2. convergence
3. epochs
4. validationError
5. parameterSnapshot

### 3.3 运行时配置输出

输出对象名称： runtimeConfig

必需字段：

1. windowSize：滑动窗口大小
2. summaryStateDim：摘要状态维度
3. personaPool：人格池配置
4. toolBindings：工具绑定清单

## 4. 约束规则

1. 所有 id 必须稳定且可追溯，禁止仅用自然语言文本做主键。
2. 时间常数、电容、电导等动力学参数必须满足物理可解释约束。
3. semanticMapping 中每个 sourceTokenId 必须能回溯到 vocabulary。
4. 拟合结果必须附带样本集版本和时间戳，保证可复现实验。

## 5. 文件组织建议

推荐采用以下产物布局：

1. gguf/*.json：原始模型结构化导出
2. vocab/*.json：词表与特殊词元
3. mapping/*.json：词元到语义最小单位映射
4. dynamics/*.json：动力学参数表
5. fit/*.json：拟合日志与快照
6. runtime/*.json：运行时配置