/* Auto-generated from main.cpp
   Segment: section_before_NamespacedStore
   Original lines: 3354-3360 */

/* NamespacedStore adds namespace prefix to underlying storage */
// 调用方式：将同一底座按不同业务域隔离键空间。
// 实现思路：统一在 key 前拼接固定前缀后转发调用。
// 注意事项：命名空间为空会抛异常，调用方需预校验。
// 注意事项：前缀策略变更会影响历史数据可见性。
// 注意事项：该封装不改变底层一致性语义。
