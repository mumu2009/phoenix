// Auto-generated from main.cpp
// Segment: section_before_DialogMemory
// Original lines: 5382-5388


// DialogMemory 负责存储和检索对话记忆片段。
// 调用方式：对话完成后写入，召回时按签名或 meme 检索。
// 实现思路：维护主记录与倒排索引并控制容量。
// 注意事项：reply/问题文本长度会被裁剪以控制体积。
// 注意事项：索引键冲突需通过稳定哈希策略规避。
// 注意事项：该组件应与业务隐私策略配套使用。
