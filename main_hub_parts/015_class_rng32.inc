/* Auto-generated from main.cpp
   Segment: class_Rng32
   Original lines: 2753-2765 */

class Rng32 {
public:
  explicit Rng32(uint32_t seed) : x_(seed ? seed : 0x9e3779b9u) {}
  double next() {
    x_ ^= x_ << 13;
    x_ ^= x_ >> 17;
    x_ ^= x_ << 5;
    return (double)(x_ & 0xffffffffu) / (double)0xffffffffu;
  }

private:
  uint32_t x_{0x9e3779b9u};
};
