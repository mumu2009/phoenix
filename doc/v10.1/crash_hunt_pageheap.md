# 崩溃追猎：为什么 `_GLIBCXX_DEBUG` 不可用，以及正确的 PageHeap 方案

## 1. 结论（实测）

侧车编译 `-D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS` **链接失败**：

```
ld.exe: ...frontend_server.cpp:(.text+...) undefined reference ...
collect2.exe: error: ld returned 1 exit status
```

原因：`_GLIBCXX_DEBUG` 改变 `std::string/vector/map` 的**对象布局与符号名**，
而 conan 预编译库（drogon / trantor / opencv / sqlite3 …）是用普通 ABI 编的，
两边无法混链。这个项目依赖一堆预编译库，所以这条路**不通**。

（`-D_GLIBCXX_ASSERTIONS` 单独用是 ABI 兼容的，但它只检查 vector/string 边界，
对 `std::map` 节点非法化基本无效；我们的怀疑对象恰恰是 map —— 收益太低。）

## 2. 正确方案：Page Heap（无需重编）

原理：给进程的每次分配加独立页 + 尾部守卫页，**越界写的当下**就触发访问违例，
异常地址和栈都指向真正的写点，而不是 25 分钟后的 ntdll 堆校验。

启用（等价于 `gflags /p /enable phoenix_main.exe /full`）：

```powershell
$k = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options\phoenix_main.exe'
New-Item -Path $k -Force | Out-Null
# FLG_HEAP_PAGE_ALLOCS = 0x02000000
Set-ItemProperty -Path $k -Name GlobalFlag -Value 0x02000000 -Type DWord
# 关闭：
# Remove-ItemProperty -Path $k -Name GlobalFlag
```

**代价**：内存占用会显著上升（页粒度 + 守卫页），进程会变慢。因此：
- 只在**短时复现**时开启，复现完立刻关闭；
- 不要在看守不到的时间段对生产网关开启（1.6 GB 进程可能被撑爆）。

## 3. 隔离复现（不碰正在跑的 5097）

```powershell
# 1) 隔离工作目录（避免两个进程同时写 autonomy_state.json）
$w = "E:\079\v6.0Alixander\phoenix\build\tmp\phx_repro"
New-Item -ItemType Directory -Force $w | Out-Null
Copy-Item config $w -Recurse -Force
New-Item -ItemType Directory -Force "$w\runtime_store\long_run" | Out-Null

# 2) 启用 page heap（针对 exe 名，全局生效）
#    ...见 §2...
# 3) 在该目录用新端口启动（--gateway-port 5101），别用 supervisor
Start-Process phoenix_main.exe -ArgumentList '--gateway-port','5101' -WorkingDirectory $w
# 4) 灌 8 个并行 mission（触发器）
1..8 | ForEach-Object {
  Invoke-RestMethod -Uri 'http://127.0.0.1:5101/api/mission/assign' -Method POST `
    -Headers @{Authorization='Bearer local-probe'} -ContentType 'application/json' `
    -Body (@{ goal = "repro $_ : write a 2000-char tutorial about mission pressure" } | ConvertTo-Json)
}
# 5) 等 10–20 分钟；PageHeap 会把写点变成 0xC0000005 + 生成 dump
# 6) 立刻关闭 page heap，并分析：
python tools\dump_stack_resolve.py build\tmp\dumps\phoenix_main.exe.<pid>.dmp --map phoenix_main.map
```

## 4. 判读要点

- PageHeap 下的异常码是 **0xC0000005**（访问违例）而不是 0xC0000374，这是正常的
  —— 它把"延迟发现"变成"当场发现"；
- 栈顶应当是**我们的代码帧**（不是 ntdll）；这才能定位到具体容器操作；
- 若 20 分钟不复现：把 mission 数提到 12、把 `missionDelibLock` 保持 false，
  再加一个 chat 循环制造 slot 竞争。

## 5. 备选（成本更高，仅在 PageHeap 无效时）

1. 用 `Application Verifier`（`appverif.exe`，需安装 Windows SDK 组件）的
   Heaps 检查，图形界面勾选后同样无需重编；
2. 给可疑容器加"影子副本 + 定时校验"（debug 专用，见下）：在每个 tick 结束后
   校验 `docs_`/`invertedIndex_` 的 size 与关键 key 集合是否自洽，不一致就打印
   —— 只能缩小范围，不能定位写点；
3. 最后手段：把 `OnlineResearcher`/`RuntimeState` 的容器全部换成
   `std::shared_mutex` + 读写包装类（结构性消除竞争），代价是热路径开销上升，
   但可从根上消灭这一类 bug。

## 6. 当前状态

- `_GLIBCXX_DEBUG` 侧车：**已放弃**（ABI 冲突，实测链接失败）；
- 生产二进制：`phoenix_main.exe`（19:56 构建，带 map），窗口从 8:15:50 起
  已经连续 11 小时无崩溃；今天 4 次崩溃集中在 00:55 与 08:12–08:15；
- 下一步按 §3 执行，**开 PageHeap 的窗口必须有人/有自动化看守**（内存风险），
  建议在你在线时开、30 分钟内关。
