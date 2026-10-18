# 自我层（selfMind）

聊天窗口继续只送用户原话。自我层是它下面的另一条通道：用对话把回答机器学成一个具体的人，并在没有人提问时自己往下想。

这不是现象意识的证明，也不是请人类来判的图灵测试。镜子和图灵探针只检查这个自我模型能不能把“我说过的话 / 这个人”和“服务窗口 / 别人”分开。

## 三件事

1. **对话强化学习。** 语料里 `human` 是要成为的那个人，`chatbot` 是服务窗口的反例，`other` 是别人，`partner` 只作上下文。特征是 24 维（人称、服务腔、记忆词、情感词、字节二元组）。线性策略有五个言语行为：回忆、偏好、续想、反问、客服。每个行为都当场打分，用一步 TD（γ = 0）更新权重，权重钳在 [−4, 4]。奖励是“像这个人”减去“像客服”。每 `holdoutEvery`（默认 4）句 `human` 留出，不进质心，留给图灵探针。
2. **镜子与图灵探针。** 镜子：自己的念头要对自身质心更近，召回至少 0.75；别人的句子，以及把〔自识〕标记贴到别人句子上的情况，假阳性不超过 0.25。图灵探针：留出的人句得分要成对高于服务窗口箔句，准确率至少 0.75。
3. **自动思考。** `tickThought` 不读提问。心跳或 `iterate` 在自我层开启时都会产生一句念头，写进 `result.selfMind`，不写进 `composedPrompt`。所以问句不会把念头变成回答。

底层提示 `kernelText` 是第一人称续写底（`[self-kernel]`），给状态和审计看，不进聊天因果串。

## 配置

`config/phoenix.example.json` 的 `selfMind`：

| 键 | 默认 | 含义 |
|---|---|---|
| enabled | true | 关闭则不再产生念头 |
| name | self | 写进底层提示的名字 |
| learningRate | 0.35 | TD 步长，0 表示只记语料、不改策略 |
| corpusPath | 空 | 启动时若文件存在就摄入一次 |
| learnFromLiveTurns | false | 为 true 时，非任务的短用户句才当作 `human` 学习 |
| holdoutEvery | 4 | 留出句的间隔 |

环境变量 `AI_SELF_MIND_ENABLED` 覆盖开关。

真对话是 JSON 或 JSONL：每行 `{"role":"human"|"chatbot","text":"..."}`，或单个 `{"turns":[...]}`。单文件上限 32MB，JSONL 最多摄入 8000 行。

仓库里的 `doc/v8.3/self_mind_sample.json` 是**合成校准集**，只为了让单元探针可复现，不能当成采集到的人。

运行时语料是公开数据集，不是合成样例：

- 来源：`shareAI/ShareGPT-Chinese-English-90k`，文件 `shareGPT/computer_zh_26k.jsonl`
- 镜像：`https://hf-mirror.com/datasets/shareAI/ShareGPT-Chinese-English-90k`
- 许可：Apache-2.0
- 内容：网友自愿公开的真实用户提问，以及当时分享出来的助手回复；这个发布是中英平行语料。计算机子集的原文在 `runtime_store/datasets/self_mind/sharegpt_computer_zh_26k.jsonl`。运行时切片 `sharegpt_zh_turns.jsonl` 是其中前 8000 句（空白折叠、每句最多 240 字）。说明见同目录 `PROVENANCE.json`。

## 接口

- `CognitionAutonomyManager::configureSelfMind / ingestSelfCorpus / selfMindStatus / selfProbe / tickSelfThought`
- `GET /api/self/status`
- `POST /api/self/ingest`
- `POST /api/self/probe`
- `POST /api/self/think`

状态随 `exportState` / `importState` 的 `selfMind` 字段保存。

## 验证

从仓库根目录：

- 只编译这一层：`test/self_mind_driver.cpp` + `self_mind.cpp`
- 全量套件里：`gtest_runner.exe --gtest_filter=SelfMindTest.*`

协议见 `doc/v7.0/testing_methodology.md` 第 14 节，公式见 `doc/v7.0/algorithm.md` 第 24 节。

## 边界

- 线性特征和五个模板句不能代替语言模型本身的生成。念头是这一层自己说出的短句，不是 llama 的采样。
- 校准集是同一个人、同一件事（码头和潮汐）。换一个很散的说话人，24 维特征可能分不开，探针就应该失败，而不是把门槛放宽。
- `learnFromLiveTurns` 默认关。随便一句用户话不会覆盖已经学到的人。
- 安全模块没有改。
