// Auto-generated from main.cpp
// Segment: section_before_GatewayServer
// Original lines: 13028-13036


// ------------------ Gateway ------------------

// GatewayServer 承载 HTTP 网关路由与系统编排逻辑。
// 调用方式：构造后调用 listen 启动对外服务。
// 实现思路：整合控制器池、学习器、快照与鉴权组件。
// 注意事项：路由较多，异常需统一捕获并标准化返回。
// 注意事项：内部模块初始化失败应尽量降级运行。
// 注意事项：该类是系统入口，变更需谨慎回归验证。
