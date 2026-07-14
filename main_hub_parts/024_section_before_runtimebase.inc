/* Auto-generated from main.cpp
   Segment: section_before_RuntimeBase
   Original lines: 3403-3414 */

/* ------------------ Core model ------------------ */

class RuntimeState;
struct RobotsDocument;

/* RuntimeBase defines runtime controller base capability set */
// 调用方式：具体运行时实现继承该基类并重写核心接口。
// 实现思路：抽象公共依赖、配置与通用工具函数。
// 注意事项：析构需保证派生类资源正确释放。
// 注意事项：并发成员访问需遵守线程模型约束。
// 注意事项：该类本身不直接承载完整业务流程。
