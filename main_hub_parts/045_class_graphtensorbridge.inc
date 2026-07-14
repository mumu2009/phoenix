/* Auto-generated from main.cpp
   Segment: class_GraphTensorBridge
   Original lines: 4774-4927
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

class GraphTensorBridge
/* GraphTensorBridge maps graph structure to tensor representation */
// 调用方式：训练或推理前将图特征编码为张量输入。
// 实现思路：抽取节点/边属性并构建统一数值表示。
// 注意事项：映射规则变化会影响模型兼容性。
// 注意事项：输入图缺失字段时需执行容错处理。
// 注意事项：桥接过程应尽量减少重复拷贝。
{
public:
  explicit GraphTensorBridge(KVMStore *kvm, TensorEngine *tensor)
      : kvm_(kvm), tensor_(tensor) {}

  std::vector<std::vector<float>>
  computeNodeEmbeddings(const std::vector<std::string> &ids, int D = 512) {
    rebuildRowIndex(ids);
    std::vector<std::vector<float>> embeddings(ids.size(),
                                               std::vector<float>(D, 0.0f));
    for (size_t i = 0; i < ids.size(); i++) {
      auto words = kvm_->getMemeWords(ids[i]);
      if (!words.empty()) {
        std::string joined;
        for (auto &word : words) {
          if (word.empty())
            continue;
          if (!joined.empty())
            joined.push_back(' ');
          joined += word;
        }
        if (!joined.empty()) {
          auto base = textToMiniEmbedding(joined, D);
          if (base.size() == (size_t)D) {
            embeddings[i] = std::move(base);
            continue;
          }
        }
      }
      uint32_t idx = fnv1a32(ids[i]) % (uint32_t)D;
      embeddings[i][idx] += 1.0f;
    }
    // l2NormalizeRows for std::vector<std::vector<float>>
    for (auto &vec : embeddings) {
      float norm = 0.0f;
      for (float v : vec)
        norm += v * v;
      norm = std::sqrt(norm);
      if (norm > 0.0f) {
        for (float &v : vec)
          v /= norm;
      }
    }
    return embeddings;
  }

  static uint32_t fnv1a32(const std::string &str) {
    uint32_t hash = 0x811c9dc5u;
    for (unsigned char c : str) {
      hash ^= c;
      hash *= 0x01000193u;
    }
    return hash;
  }

  void rebuildRowIndex(const std::vector<std::string> &ids) {
    memeIndex_.clear();
    for (size_t i = 0; i < ids.size(); i++)
      memeIndex_[ids[i]] = (uint32_t)i;
  }

  json buildEmbeddings(const std::vector<std::string> &ids, int D = 512) {
    rebuildRowIndex(ids);
    std::vector<float> mat(ids.size() * (size_t)D, 0.0f);
    for (size_t i = 0; i < ids.size(); i++) {
      auto words = kvm_->getMemeWords(ids[i]);
      if (words.empty())
        continue;
      for (auto &word : words) {
        uint32_t idx = fnv1a32(word) % (uint32_t)D;
        mat[i * (size_t)D + idx] += 1.0f;
      }
    }
    tensor_->l2NormalizeRows(mat, ids.size(), (size_t)D);
    embeddings_ = json{{"data", mat}, {"nRows", (int)ids.size()}, {"nCols", D}};
    return embeddings_;
  }

  json buildMultiOrderCSR(const MemeGraph::WindowInfo &graphWindow) {
    const auto &csr = graphWindow.csr;
    const auto &ids = graphWindow.ids;
    std::vector<uint32_t> biRow(csr.rowPtr.size(), 0);
    std::vector<uint32_t> biCol;
    std::vector<float> biVal;
    std::vector<uint32_t> inRow(csr.rowPtr.size(), 0);
    std::vector<uint32_t> inCol;
    std::vector<float> inVal;

    std::vector<std::unordered_map<uint32_t, float>> adjacency(ids.size());
    for (uint32_t row = 0; row < ids.size(); row++) {
      uint32_t start = csr.rowPtr[row];
      uint32_t end = csr.rowPtr[row + 1];
      for (uint32_t idx = start; idx < end; idx++) {
        uint32_t col = csr.colIdx[idx];
        float weight = csr.values[idx];
        adjacency[row][col] = weight;
      }
    }

    uint32_t biCounter = 0;
    for (uint32_t row = 0; row < ids.size(); row++) {
      biRow[row] = biCounter;
      uint32_t start = csr.rowPtr[row];
      uint32_t end = csr.rowPtr[row + 1];
      for (uint32_t idx = start; idx < end; idx++) {
        uint32_t col = csr.colIdx[idx];
        float weight = csr.values[idx];
        auto it = adjacency[col].find(row);
        if (it != adjacency[col].end() && it->second != 0.0f) {
          biCol.push_back(col);
          biVal.push_back((weight + it->second) * 0.5f);
          biCounter++;
        }
        inCol.push_back(row);
        inVal.push_back(weight);
      }
    }
    biRow[ids.size()] = biCounter;
    inRow[ids.size()] = (uint32_t)inCol.size();

    json all = json{{"rowPtr", csr.rowPtr},
                    {"colIdx", csr.colIdx},
                    {"values", csr.values},
                    {"nRows", csr.nRows},
                    {"nCols", csr.nCols}};
    json bi = json{{"rowPtr", biRow},
                   {"colIdx", biCol},
                   {"values", biVal},
                   {"nRows", (int)ids.size()},
                   {"nCols", (int)ids.size()}};
    json inbound = json{{"rowPtr", inRow},
                        {"colIdx", inCol},
                        {"values", inVal},
                        {"nRows", (int)ids.size()},
                        {"nCols", (int)ids.size()}};
    multi_ = json{
        {"all", all}, {"bi", bi}, {"out", all}, {"in", inbound}, {"ids", ids}};
    return multi_;
  }

private:
  KVMStore *kvm_{nullptr};
  TensorEngine *tensor_{nullptr};
  std::unordered_map<std::string, uint32_t> memeIndex_;
  json embeddings_ = json::object();
  json multi_ = json::object();
};
