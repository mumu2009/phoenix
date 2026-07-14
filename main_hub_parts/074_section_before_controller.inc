// Auto-generated from main.cpp
// Segment: section_before_Controller
// Original lines: 9905-9911


// Controller 抽象单个 AI 控制器的行为契约。
// 调用方式：池化层通过该接口进行统一调度与管理。
// 实现思路：覆盖响应、参数、快照与度量等关键能力。
// 注意事项：实现类需保证线程安全与异常隔离。
// 注意事项：snapshot/applySnapshot 需尽量幂等。
// 注意事项：name 必须稳定用于路由与指标汇聚。
