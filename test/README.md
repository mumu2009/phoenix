# test

本目录是仓库内测试脚本的总入口，按测试类型拆分为多个子模块。

## 子目录

- `apis/`：API 回归、接口库存与后端矩阵测试。
- `intelligence/`：智能度与回答质量评测。
- `prof/`：性能、速度与综合 benchmark。

## 使用建议

1. 接口稳定性优先看 `test/apis/`。
2. 回答质量优先看 `test/intelligence/`。
3. 性能、外部数据集与双路对比优先看 `test/prof/`。

## 当前新增测试

- `test_start_079_launcher.py`：验证 GUI 启动器输出的命令行参数是否覆盖 builtin addon、mechanical mind 和 addon library 场景。
- `mechanical_mind_filter_tests.cpp`：验证机械化心智过滤器的中性文本放行、情绪化文本机械化以及 warmup 学习 token 行为。
- `memebarrier_phrase_feedback_tests.cpp`：验证 MemeBarrier 短语反馈的连续短语匹配、正反馈落盘、负反馈仅内存，以及 step/maxOffset 运行时收紧后的夹紧行为。