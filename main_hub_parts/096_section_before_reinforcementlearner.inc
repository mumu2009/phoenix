// Auto-generated from main.cpp
// Segment: section_before_ReinforcementLearner
// Original lines: 11766-11774


// ------------------ Learning ------------------

// ReinforcementLearner 执行基于测试语料的强化学习回路。
// 调用方式：外部触发 learn 后返回评估与调参结果。
// 实现思路：评估、聚类、统计后生成参数更新策略。
// 注意事项：测试集质量直接决定学习方向可靠性。
// 注意事项：learn 不保证单次迭代一定提升指标。
// 注意事项：调用时应避免与高峰推理强竞争资源。
