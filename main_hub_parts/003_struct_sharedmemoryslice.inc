/* Auto-generated from main.cpp
   Segment: struct_SharedMemorySlice
   Original lines: 1628-1649 */

struct SharedMemorySlice {
  std::shared_ptr<std::vector<uint8_t>> owner;
  const uint8_t *ptr{nullptr};
  size_t bytes{0};

  static SharedMemorySlice
  fromVector(std::shared_ptr<std::vector<uint8_t>> data) {
    SharedMemorySlice s;
    s.owner = std::move(data);
    if (s.owner && !s.owner->empty()) {
      s.ptr = s.owner->data();
      s.bytes = s.owner->size();
    }
    return s;
  }

  // empty 判断当前切片是否为空。
  // 调用方式：在消费共享数据前先做空值短路。
  // 实现思路：检查指针为空或字节数为零。
  // 注意事项：返回 true 不代表 owner 一定为空。
  bool empty() const { return ptr == nullptr || bytes == 0; }
};
