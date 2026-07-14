// Auto-generated from main.cpp
// Segment: class_PatternMatrix
// Original lines: 5230-5259

class PatternMatrix {
public:
  // PatternMatrix 构造函数注入图桥接所需依赖。
  // 调用方式：运行时初始化阶段创建一次即可。
  // 实现思路：内部组合 GraphTensorBridge 对象。
  // 注意事项：依赖对象生命周期需长于本实例。
  PatternMatrix(KVMStore *kvm, TensorEngine *tensor) : bridge_(kvm, tensor) {}

  json rebuild(const MemeGraph::WindowInfo &windowInfo) {
    auto emb = bridge_.buildEmbeddings(windowInfo.ids, 512);
    auto multi = bridge_.buildMultiOrderCSR(windowInfo);
    auto projection = reducer_.project2D(emb, "auto");
    state_ = json{{"ids", windowInfo.ids},
                  {"embeddings", emb},
                  {"multi", multi},
                  {"projection", projection}};
    return state_;
  }

  // state 返回最近一次重建后的矩阵状态。
  // 调用方式：用于调试可视化或接口下发。
  // 实现思路：直接返回缓存的 JSON 快照。
  // 注意事项：返回值是副本，修改不会回写内部状态。
  json state() const { return state_; }

private:
  GraphTensorBridge bridge_;
  DimReducer reducer_;
  json state_ = json::object();
};
