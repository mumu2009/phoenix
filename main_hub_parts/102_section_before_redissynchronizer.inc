// Auto-generated from main.cpp
// Segment: section_before_RedisSynchronizer
// Original lines: 12627-12635


// ------------------ Redis Sync ------------------

// RedisSynchronizer 负责跨实例快照同步广播。
// 调用方式：启动订阅线程并在切换时发布快照消息。
// 实现思路：基于 Redis Pub/Sub 传递 JSON 快照载荷。
// 注意事项：网络波动时需容忍短时同步延迟。
// 注意事项：订阅线程异常应避免影响主请求路径。
// 注意事项：该模块依赖 Redis 可用性。
