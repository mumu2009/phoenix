/* Auto-generated from main.cpp
   Segment: class_ProcessReservedArena
   Original lines: 1685-1714 */

class ProcessReservedArena {
public:
  void initialize(size_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (initialized_)
      return;
    bytes = std::max<size_t>(8ull * 1024ull * 1024ull, bytes);
    bytes = std::min<size_t>(2ull * 1024ull * 1024ull * 1024ull, bytes);
    try {
      block_.resize(bytes);
      const size_t page = 4096;
      for (size_t i = 0; i < block_.size(); i += page)
        block_[i] = 0;
      reservedBytes_ = block_.size();
      initialized_ = true;
    } catch (...) {
      block_.clear();
      reservedBytes_ = 0;
      initialized_ = true;
    }
  }

  size_t reservedBytes() const { return reservedBytes_; }

private:
  std::vector<uint8_t> block_;
  size_t reservedBytes_{0};
  bool initialized_{false};
  mutable std::mutex mu_;
};
