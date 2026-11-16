# 并发崩溃取证报告 — 2026-11-15

## 1. 现象

把 `autonomyLoop.missionDelibLock` 从 `true` 改为 `false`（允许并行 mission 真正并发调用
deliberator）后，`phoenix_main.exe` 在约 60 秒内反复崩溃：

```
异常代码 0xC0000374 (STATUS_HEAP_CORRUPTION)
崩溃模块 ntdll.dll，偏移 0xff489
```

`0xC0000374` 不是段错误，而是 **NT 堆管理器在校验堆块时发现元数据被写坏**。
它通常不是"崩溃点就是错误点"：真正越界写发生在之前某个时刻，只是到下一次
`HeapAlloc/HeapFree` 才被检测到。这也解释了为什么崩溃位置总在 ntdll 内部。

## 2. 方法：先枚举共享可变状态，再逐个验证

粗锁 `missionDelibMu_` 把所有 mission tick 串行化，因此**它掩盖了所有未加锁的共享状态**。
取证的做法是把 deliberator lambda 里触及的每一个可变共享对象列出来，判断它是否
被 `std::async` 出的多个 mission 线程同时修改。

排查结果（`main_hub_parts/111_class_gatewayserver.inc`）：

| 共享对象 | 位置 | 是否加锁 | 结论 |
|---|---|---|---|
| `gDelibSkipCount` / `gDelibLastDelivAtLlm` / `gDelibCacheSkipsNoGrowth` | 411–414 | `gDelibSkipMu` | 安全 |
| `gnnCursors` | 580–581 | `gnnCursorMu` | 安全 |
| `focusByScope` | 1135–1137 | `focusMu` | 安全 |
| `explorePaths`（`std::vector<std::string>`） | 488 | **无** | **竞争 ①** |
| `namedRealPath`（`std::string`） | 489 | **无** | **竞争 ②** |
| `sectionDropStreak`（`std::map`） | 779 | **无** | **竞争 ③** |
| `cognitionMemory_` | 855 | `HierarchicalMemory::mu_` | 安全 |
| `mission_workspace` L1 缓存 | `mission_workspace.hpp` | `gWsCacheMu` | 安全 |
| `EmotionSystem` | `emotion_system.cpp` | `impl_->mutex` | 安全 |
| `TinyExpert` | `tiny_expert.cpp` | `mu_`（但 `fromJson` 无锁） | **竞争 ④** |
| `RuntimeState::processInput/ingestDocument` | `065_*.inc` | `stateMu_` | 安全（但是粗锁） |

## 3. 三个真凶

### ① `explorePaths` 惰性初始化竞争（use-after-free）

```cpp
static std::vector<std::string> explorePaths;   // lambda 级静态
...
if (explorePaths.empty()) {                     // 无锁判断
  ...
  explorePaths.push_back(rel);                  // 扩容 -> 重新分配
  std::sort(explorePaths.begin(), explorePaths.end());
}
...
for (const auto &rel : explorePaths) { ... }    // 另一线程正在遍历
```

线程 A 在 `push_back` 触发 `realloc` 并释放旧缓冲区，线程 B 还在遍历旧缓冲区里的
`std::string`（每个 string 自己还有堆块）→ 读已释放内存 + 后续写入 → 堆元数据损坏。
**这与 `0xC0000374` 的"延迟检测"特征完全一致。**

修复：改成**不可变**的魔法静态（C++11 保证首次初始化线程安全），并提供
`GatewayServer::gatewayExplorePaths()` 作为唯一构造点。之后所有读取都发生在
一个构造完成后不再变化的 `const std::vector` 上。

### ② `namedRealPath` —— 竞争 + 跨任务串味

它记录"上一次回复点名过的仓库文件路径"，被当作 `focus` 的兜底。原实现是**进程全局**
`std::string`：两个 mission 同时 `namedRealPath = rel;` 会对同一个 string 做赋值
（内部缓冲区释放 + 分配），而另一个线程正在 `find("..")` 读它 → 竞争 ②。

同时它还是个**语义 bug**：mission A 指名的文件会泄漏成 mission B 的 focus。

修复：改成 `namedPathByScope`（按 mission scope 的 map）+ `namedPathMu`，
读写分别走 `namedPathGet()/namedPathSet()`。

### ③ `sectionDropStreak` —— 无锁 `std::map`

```cpp
static std::map<std::string, int> sectionDropStreak;
sectionDropStreak[scope] += 1;   // operator[] 可能插入 -> 红黑树旋转
sectionDropStreak[scope] = 0;
```

多线程同时 `operator[]` 插入会旋转/重挂红黑树节点，遍历/析构线程随即读到已释放节点。
这是最典型的"堆损坏"来源。

修复：包成 `ScopeCounter`（`std::mutex` + `std::map` + `get/set/inc`），所有访问走
加锁接口。

### ④ `TinyExpert::fromJson` 无锁（次要，但同类）

`fromJson` 会整体替换 `W1_/b1_/W2_/b2_`。它在 `importState()`（启动/恢复路径）被调用，
而训练线程/推理线程此时可能正在 `forwardLocked` 读权重；`load()` 里调用它时**已经持有
`mu_`**，所以不能简单在 `fromJson` 里加锁（否则自死锁）。

修复：按 `PopulationLayer`/`UnitLearner` 的既有模式拆成
`fromJson()`（加锁入口）+ `fromJsonLocked()`（调用方已持锁），`load()` 改调后者。

## 4. 为什么"回退粗锁"能立刻止血

粗锁把整个 lambda 串行化，①②③ 的并发访问消失，所以崩溃消失。这既验证了
"粗锁是必要的历史防御"，也说明**它掩盖了真实缺陷**——正是本次修掉的三处。

修复三处之后，粗锁不再是正确性前提，而变成一个可关闭的性能开关
（`autonomyLoop.missionDelibLock`）。

## 5. 验证计划

1. 重新编译，`missionDelibLock=false`、`llama.instanceIsolation=shared`、
   `llama_server.parallel=3`；
2. 观察 ≥2 小时：无 `0xC0000374`；
3. 同时观察 `nvidia-smi dmon -s u`：3 路并发应把 SM 利用率从 1–8% 提升到
   批处理区间（此前粗锁只允许 1 个请求在飞，llama.cpp 无法连续批处理）；
4. 回归 `gtest`：`test_study_config_state_split`（配置/状态分离）、
   `test_logic_engine`、`test_plugin_semantics`。

## 6. 教训（写入工程约定）

- **任何 lambda 级 `static` 可变容器，在可能并发的路径上都是缺陷**，除非有锁或
  不可变化（`static const` + 魔法静态）。
- 粗锁只能作为**定位工具**，不能作为长期正确性方案；本次三处缺陷在粗锁下潜伏了
  多个版本。
- 崩溃地址（ntdll 偏移）没有意义，**崩溃类型**（`0xC0000374` vs `0xC0000005`）才决定
  取证方向：前者查堆写入竞争/越界，后者查空指针/悬垂指针。
