# 学习深化、族群层、极小专家

聊天因果串仍然只有用户原话。自我层念头仍在 `selfMind`，不写进 `composedPrompt`。这一层是旁边的学习通道。

## 1. 全过程 unit 学习

`study_unit_learn.hpp`。1 个 BPE token = 1 个 unit。

一条样本按这个顺序走，不把原文拼进因果串：

1. BPE 编码（字节起步，合并上限 64）。
2. 每个 token 导入为一个模因节点（固定基向量，可被 unit 行覆盖）。
3. 相邻和窗宽 2 的边。
4. K 步消息传递，写入节点状态。
5. 用线性头预测下一个 unit 的基向量，按残差更新权重。

`recall` 返回 `{id, score, row}`。`sideChannel` 只留计数和 loss，带 `writesCausalString: false`。

`study.unitLearn.enabled` 默认 **true**。`iterate` 在有用户文本时学习，结果放在 `result.studyLearn`，不改 `composedPrompt`。

## 2. 族群层

`population_layer.hpp`。一个个体是一整层多实例的小参数向量（隐层 4，不是 8B，也不新开会话）。

- 默认 **关**：`study.population.enabled=false`。和 `mission.evolution.enabled=false` 一样，测试里打开。
- 人数 n≥2，默认上限 8，硬上限 16。
- 每次 goal 只扰动 n−1 个个体，当前最好的那个不动。
- 前向输出和目标比，误差不超过 `tolerance` 记为正确。
- 保留误差更小的一半（`n/2`），空位用幸存者的副本填回，人数不变。`spawnedInstances` 恒为 0。
- 使命层里封存的 (1+λ) 竞赛没有重开。

## 3. 极小专家

`tiny_expert.hpp`。一层隐藏单元。参数量在几千以内。不加载 GGUF，不碰 llama-server。

召唤注册表增加种类 `tiny-expert`：可以没有 URL，但必须有本地权重。`upsert` 只增改这一条，不删掉已经登记的模型。`study.expert.enabled` 默认 **false**。`summon.enabled` 仍按原配置；关着的时候拒绝登记。

语料与许可见 `runtime_store/datasets/study_expert/PROVENANCE.json`。

## 验证

在仓库根目录：

```
g++ -std=c++20 -O2 -I. -Ibuild/conan_home/p/nlohm1bed1ddc0a2fa/p/include -o build/study_development_driver.exe test/study_development_driver.cpp study_unit_learn.cpp population_layer.cpp tiny_expert.cpp
build\study_development_driver.exe
```

这条命令不链接 `phoenix_main`。
