/* Auto-generated from main.cpp
   Segment: struct_CSRMatrix
   Original lines: 3982-3988 */

struct CSRMatrix {
  std::vector<uint32_t> rowPtr;
  std::vector<uint32_t> colIdx;
  std::vector<float> values;
  uint32_t nRows{0};
  uint32_t nCols{0};
};
