# module_overrides 挂载说明

将外部模块 `.cpp` 放到本目录后，`compile.bat` 会自动把 `module_overrides/*.cpp` 编译进主程序。

你可以在外部文件中：

1. 基于 `module_mount.hpp` 的接口实现自定义模块；
2. 在静态初始化阶段注册工厂；
3. 启动时自动替换 `main.cpp` 默认实现。

## 最小示例（替换 RedisSynchronizer）

```cpp
#include "module_mount.hpp"

class MyRedisSynchronizer : public IRedisSynchronizer {
public:
    MyRedisSynchronizer(std::shared_ptr<ControllerPoolBase> /*pool*/,
                        const std::string& /*url*/,
                        const std::string& channel)
        : channel_(channel) {}

    void start() override {}
    void publish(const nlohmann::json& /*snapshot*/) override {}
    const std::string& channel() const override { return channel_; }

private:
    std::string channel_;
};

namespace {
struct AutoRegister {
    AutoRegister() {
        module_mount::registerRedisSynchronizerFactory(
            [](std::shared_ptr<ControllerPoolBase> pool,
               const std::string& url,
               const std::string& channel) -> std::shared_ptr<IRedisSynchronizer> {
                return std::make_shared<MyRedisSynchronizer>(std::move(pool), url, channel);
            }
        );
    }
} g_auto_register;
}
```

## 可替换模块

- `IRedisSynchronizer`
- `IStudyEngine`
- `ISnapshotManager`
- `IPersonaForestAverager`
- `ISparkArray`
- `IReinforcementLearner`
- `IAdversarialLearner`
- `IGnnGaLearner`
- `IGatewayServer`

对应注册函数见 `module_mount.hpp`。

## `GNN.py`（高级后端）

`module_overrides/GNN.py` 已支持多后端：

- `backend=auto`：优先 DGL，其次 Torch 回退；
- `backend=dgl`：强制 DGL（推荐）；
- `backend=graphscope`：启用 GraphScope 增强路径（当前为可扩展骨架，配合 Torch/DGL 主干）；
- `backend=torch`：纯 PyTorch 回退。

### 最小示例

```python
from GNN import build_default_gnn_module

gnn = build_default_gnn_module({
    "backend": {
        "backend": "dgl",
        "graphscope_enable_analytics": True,
        "graphscope_session_kwargs": {}
    },
    "model": {
        "in_dim": 128,
        "hidden_dim": 256,
        "out_dim": 128,
        "num_layers": 3,
        "use_attention": True
    }
})
```

### 说明

- 该文件用于“先验证设计，再迁移 C++ 实现”；
- 依赖缺失（DGL/GraphScope/PyTorch）导致无法运行属预期行为；
- 后续可将 `GraphScopeFeatureEnricher.augment` 中的占位逻辑替换为真实图分析特征注入。
