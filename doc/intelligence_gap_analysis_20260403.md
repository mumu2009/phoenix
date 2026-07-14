# 同一 llama3.1:8b 在 Ollama、llama.cpp、Phoenix 中智能度差异分析

后续路线与类人评测标准见 [agi_direction_and_humanlike_evaluation_20260403.md](./agi_direction_and_humanlike_evaluation_20260403.md)。

## 结论先行

已知三者使用的是同一基座模型 `llama3.1:8b`，但这并不等于三者把“同一份输入”送进了模型。

从当前仓库代码看，出现 `ollama > llama.cpp > phoenix_main.exe(llamacpp)` 的核心原因不是模型本体发生了变化，而是三层差异叠加：

1. `Ollama` 对 Llama 3.1 的 chat template、上下文截断和默认推理参数适配最完整，最接近模型训练时的消息分布。
2. `llama.cpp` 虽然仍然是同一模型推理引擎，但更依赖调用方把 `messages/template/stop/context` 喂对；它比 Ollama 少了一层“针对模型家族的上层语义适配”。
3. `phoenix_main.exe` 在调用外部后端前后又增加了图上下文、上下文提示、长查询降档、工具契约、二次追问、缓存和后处理，进一步改变了模型实际接收到的 prompt 分布，所以最容易在纯问答基准上掉分。

换句话说，真正变化的是：

`同一模型权重` + `不同的消息模板` + `不同的上下文拼装` + `不同的默认参数` + `不同的系统包装`

而不是只有“模型名称相同”这一项。

## 基准现象

最新综合报告 [build/prof_report_rich_multi_fixed.md](../build/prof_report_rich_multi_fixed.md) 给出的结果是：

- 直连 `Ollama` 智能度平均 `10.64`
- 当前系统 API 智能度平均 `9.83`
- 质量分差 `System - Ollama = -0.80`

这说明差异是真实且稳定的，不是单次偶发波动。

## 第一层原因：Ollama 更贴近 Llama 3.1 的原生对话分布

Ollama 这边至少做了两件对质量很关键的事。

### 1. 它明确使用了 Llama 3.1 的 instruct 模板

文件 [outsides/ollama/template/llama3-instruct.gotmpl](../outsides/ollama/template/llama3-instruct.gotmpl) 中，消息会被渲染成：

```text
<|start_header_id|>role<|end_header_id|>

content<|eot_id|>
```

这说明 Ollama 并不是把用户输入简单拼成一段纯文本，而是在严格维持 Llama 3.1 训练时常见的 header / role / end-of-turn 结构。

### 2. 它在服务端负责消息渲染与上下文裁剪

文件 [outsides/ollama/server/prompt.go](../outsides/ollama/server/prompt.go) 里，`chatPrompt` 会：

- 保留系统消息
- 按模型上下文长度做截断
- 再通过模板执行器渲染成最终 prompt

这意味着 Ollama 的输入不是“用户自己乱拼的一段文本”，而是“经过模型家族适配后的规范 chat prompt”。

### 3. 它还有一套稳定的默认推理参数

文件 [outsides/ollama/api/types.go](../outsides/ollama/api/types.go) 定义了默认值：

- `Temperature = 0.8`
- `TopK = 40`
- `TopP = 0.9`
- `RepeatPenalty = 1.0`
- `NumCtx = envconfig.ContextLength()`

这类默认值未必神奇，但它们是“为 Ollama 这个运行时统一整理过”的，通常比“调用方自行拼请求”更稳。

因此，`Ollama` 的优势本质上是：它把模型输入维持在更接近原始训练分布的位置。

## 第二层原因：llama.cpp 比 Ollama 更裸，质量更依赖调用方

在当前仓库里，Phoenix 对 `llama.cpp` 的调用不是直接复用 Ollama 那套模板系统，而是走自己的外部适配器。

文件 [main_hub_parts/tail_parts/094_section_before_contexthint.inc](../main_hub_parts/tail_parts/094_section_before_contexthint.inc) 里的 `chatWithExternalAdapter` 显示：

- 当 `provider == "llamacpp"` 时，请求路径是 `/v1/chat/completions`
- 请求体只包含一条 `messages = [{ role: "user", content: prompt }]`
- 其中 `prompt` 还是 Phoenix 自己拼出来的文本

也就是说，Phoenix 传给 llama.cpp 的并不是“原生多轮角色消息 + 官方模板控制”，而是：

```text
Context:
...graphContext...

User:
...text...
```

再塞进一条 `user` 消息里。

这一点和 Ollama 的差异非常大：

- Ollama 维护的是“角色化消息语义”
- Phoenix 调 llama.cpp 时维护的是“单条 user 文本 + 手工前缀”

因此，哪怕底层模型权重相同，`llama.cpp` 这一路也比直连 Ollama 更容易偏离最佳对话分布。

## 第三层原因：Phoenix 在 llama.cpp 之外又叠加了一整层系统包装

这是 `phoenix_main.exe(llamacpp)` 比“直接 llama.cpp”再弱一档的关键原因。

### 1. Phoenix 会在进入模型前生成 graphContext

文件 [main_hub_parts/tail_parts/098_section_tail.inc](../main_hub_parts/tail_parts/098_section_tail.inc) 的 `/api/chat` 处理逻辑里，Phoenix 会根据开关：

- 跑 graph selector
- 取 graph reply
- 拼 image context
- 注入 context hint

最终把这些内容合成 `graphContext`，然后交给：

- `chatWithOllama(text, modelGraphContext, maxTokens)`
- `chatWithLlamaCpp(text, modelGraphContext, maxTokens)`

这意味着外部模型收到的不是用户原问题，而是“用户问题 + Phoenix 图系统摘要 + 视觉上下文 + 风格/提示注入”。

如果这些附加信息和当前问答任务高度匹配，它可能帮忙；但在通用 QA 基准里，它也很容易成为噪声。

### 2. Phoenix 会把 graphContext 直接拼进 prompt

文件 [main_hub_parts/tail_parts/093_class_gatewayserver.inc](../main_hub_parts/tail_parts/093_class_gatewayserver.inc) 中：

- `chatWithOllama` 会把 prompt 拼成 `Context:\n...\n\nUser:\n...`
- `chatWithExternalAdapter` 对 llama.cpp 和 bitnet 也是同样的拼法

这一步非常关键，因为它说明 Phoenix 并没有把图上下文作为独立系统消息或独立 tool message 传给模型，而是粗暴塞进正文。

这种做法的直接后果是：

- 模型更难区分“真正用户意图”和“系统附加上下文”
- 模型原本训练好的 role 结构被削弱
- prompt 分布偏离 Ollama 原生模板

### 3. Phoenix 对长问题会主动降档

同一份 `/api/chat` 逻辑里，存在：

- `longQuerySafeMode = tokenCount >= 24 || text.size() >= 360`
- 一旦触发，`maxTokens = min(maxTokens, 32)`

这意味着对稍长一点的问题，Phoenix 会主动把生成上限压到 `32`。

这会直接带来两个副作用：

1. 回答变短，解释不完整
2. 推理链和细节展开不够，主观上看起来就会“更笨”

在问答和解释类 benchmark 里，这种截断尤其容易拉低智能得分。

### 4. Phoenix 还会插入 addon tool contract 和二次追问

同一逻辑里，如果开启 `enableAddonToolContract`，Phoenix 会：

- 给 graphContext 附加工具契约
- 解析模型输出中的工具调用意图
- 真正执行 addon
- 再带着 `followupGraphContext` 对模型发起第二次请求

这是一种“系统能力增强”机制，但它的代价是：

- 首次回答不再只是纯模型回答
- 模型输出可能先被工具协议污染
- 二次追问会进一步偏移原始问答任务

所以它有利于系统功能，不一定有利于“纯智力基准”。

### 5. Phoenix 还有缓存、验证和外部风格训练分发

`/api/chat` 还会做：

- inference cache
- verify 分支
- `dispatchExternalStyleTrainStep` 到外部后端

这些机制不一定直接降低智能，但说明 Phoenix 的调用链明显比“直连模型”更复杂。链路越复杂，输入和输出越不纯，越容易在 benchmark 下损失原生模型质量。

## 第四层原因：如果命中本地图推理路径，Phoenix 根本不是标准 LLM

这部分解释的是另一个更强的降级来源。

文件 [main_hub_parts/tail_parts/071_class_localcontroller.inc](../main_hub_parts/tail_parts/071_class_localcontroller.inc) 表明本地控制器直接调用 `RuntimeState::processInput`。

而 [main_hub_parts/tail_parts/051_class_runtimestate.inc](../main_hub_parts/tail_parts/051_class_runtimestate.inc) 中的 `processInput` 和 `composeReply` 显示：

- 输入先被 token 化并映射到 meme
- 再做 graph propagation
- 再从 `surfaceLexicon` 里取高分短语
- 最多把前几个短语用句号拼起来

此外，它还会优先去 `dialogMemory_->retrieve(...)` 里检索历史回复。

这说明 Phoenix 的本地核心更接近：

- 图检索
- 激活传播
- 记忆匹配
- 短语重组

而不是：

- 自回归 next-token generation
- 全序列注意力建模
- 长程语义依赖

所以如果实验路径命中了这套本地图推理，而不是外部 llama.cpp / Ollama，那么“智能度更低”不是参数问题，而是架构问题。

## 为什么会形成你观察到的排位

### `Ollama > llama.cpp`

因为 Ollama 对 Llama 3.1 的上层语义适配更完整：

- 模板更对
- role 更对
- 截断方式更稳
- 默认参数更统一

它更接近模型训练时预期的 chat 输入格式。

### `llama.cpp > phoenix_main.exe(llamacpp)`

因为 Phoenix 在 llama.cpp 之前和之后又做了额外加工：

- graphContext 拼接
- context hint 注入
- 长查询时 `maxTokens <= 32`
- addon tool contract
- followupGraphContext 二次调用
- 缓存与验证
- 外部运行时适配和 supervisor

这些机制增强了系统控制力，但削弱了“纯模型直出”的原始质量。

### 如果命中本地 Phoenix 路径，则差距会更大

因为那条路径本来就不是标准大模型推理，而是图传播和记忆/短语组合系统。

## 最后一句话总结

你观察到的“智能程度降低”本质上是 `输入分布漂移` 和 `系统包装增厚` 的结果。

同一 `llama3.1:8b`：

- 在 `Ollama` 中，输入最接近模型原生 chat 分布
- 在 `llama.cpp` 中，输入已经更依赖调用方自己喂对格式
- 在 `Phoenix(llamacpp)` 中，输入和输出都被系统层再次改写

所以排位出现 `ollama > llama.cpp > phoenix_main.exe(llamacpp)` 是符合当前代码结构的。

## 可复现实验建议

如果要把结论做成更硬的实证，可以做三组最小对照：

1. 直连 Ollama `/api/chat`
2. 直连 llama.cpp `/v1/chat/completions`，手工提供正确 Llama 3.1 chat template
3. 走 Phoenix `/api/chat`，并逐步关闭 `graph selector`、`context hint`、`addon tools`

预期现象是：

- 第 2 组会比第 1 组略弱
- 第 3 组在开启全部 Phoenix 包装时最弱
- 第 3 组关闭附加包装后，会向第 2 组收敛

如果出现这个结果，就能直接证明：质量损失主要来自外壳和包装层，而不是模型本体。