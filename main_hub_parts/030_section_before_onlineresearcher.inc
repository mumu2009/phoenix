/* Auto-generated from main.cpp
   Segment: section_before_OnlineResearcher
   Original lines: 3488-3496 */

inline const ModelDefaults& getModelDefaults() {
  static ModelDefaults instance = loadModelDefaultsFromConfig();
  return instance;
}

/* OnlineResearcher provides online material retrieval and summary assistance */
// 调用方式：在复杂问题处理中由运行时按需调用。
// 实现思路：封装抓取、过滤、聚合和提示增强流程。
// 注意事项：外部网络失败时应支持降级路径。
// 注意事项：结果需接受安全和可信度约束。
// 注意事项：该组件不应阻塞核心请求主链路。
