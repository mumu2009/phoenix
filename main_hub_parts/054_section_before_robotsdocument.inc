// Auto-generated from main.cpp
// Segment: section_before_RobotsDocument
// Original lines: 5545-5556


// ------------------ Robots / Study / Shards ------------------

class ControllerPoolBase;
class RedisSynchronizer;

// RobotsDocument 表示语料文档的标准化结构。
// 调用方式：语料收集与分发流程统一使用该对象。
// 实现思路：聚合 id、文本、分词与来源元信息。
// 注意事项：text 可能较大，序列化时需关注开销。
// 注意事项：tokens 可为空，上层需支持延迟分词。
// 注意事项：source/file 字段用于追溯，不用于鉴权。
