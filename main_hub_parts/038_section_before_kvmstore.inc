/* Auto-generated from main.cpp
   Segment: section_before_KVMStore
   Original lines: 4082-4088 */

/* KVMStore maintains key-value memory data read/write and retrieval logic */
// 调用方式：在会话和长期记忆路径中保存与提取状态。
// 实现思路：围绕键前缀、命名空间和索引执行操作。
// 注意事项：后端能力会影响一致性与时延表现。
// 注意事项：批量操作应明确事务边界与重试策略。
// 注意事项：该类不负责业务层权限校验。
