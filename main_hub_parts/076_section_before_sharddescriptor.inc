// Auto-generated from main.cpp
// Segment: section_before_ShardDescriptor
// Original lines: 9923-9929


// ShardDescriptor 保存分片节点的健康与向量状态。
// 调用方式：由分片管理器在请求前后更新统计信息。
// 实现思路：通过 EMA 延迟与失败率估算可用性。
// 注意事项：embedding 维度应与路由向量保持一致。
// 注意事项：离线判定阈值影响流量稳定性。
// 注意事项：history 仅用于观测，不作为强一致日志。
