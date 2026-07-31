/* Auto-generated from main.cpp
   Segment: class_MemeGraph
   Original lines: 4248-4725
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

#include "../semantic_unit.hpp"

class MemeGraph
/* MemeGraph represents memory graph node relationships and retrieval interface */
// 调用方式：在学习、推理与可视化导出路径中共享。
// 实现思路：维护节点、边和索引并支持增量更新。
// 注意事项：图规模增长时需控制遍历与存储开销。
// 注意事项：并发写入应通过内部锁保证一致性。
// 注意事项：导出前建议执行数据清洗。
{
public:
  // MemeGraph 构造函数绑定底层存储实例。
  // 调用方式：创建图对象时注入 KeyValueStore。
  // 实现思路：移动保存共享指针避免额外拷贝。
  // 注意事项：store 为空会影响后续持久化行为。
  explicit MemeGraph(std::shared_ptr<KeyValueStore> store)
      : store_(std::move(store)) {}

  void ensureNode(const std::string &memeId) {
    std::lock_guard<std::mutex> lock(graphMu_);
    ensureNodeLocked(memeId);
  }
  void ensureNodeLocked(const std::string &memeId) {
    if (!nodes_.count(memeId)) {
      nodes_[memeId] = NodeEdges{};
      nodeOrder_.push_back(memeId);
    }
    if (!meta_.count(memeId)) {
      json meta;
      meta["degree"] = 0;
      meta["lastTouched"] =
          (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
      meta_[memeId] = meta;
      metaOrder_.push_back(memeId);
      store_->put("node:" + memeId, meta);
    }
    if (!store_->get("row:" + memeId)) {
      store_->put("row:" + memeId, json{{"neighbors", json::array()}});
    }
  }

  void link(const std::string &fromId, const std::string &toId,
            double weight = 1, int direction = 0) {
    std::lock_guard<std::mutex> lock(graphMu_);
    if (fromId == toId)
      return;
    ensureNodeLocked(fromId);
    ensureNodeLocked(toId);
    auto &table = nodes_[fromId];
    auto it = table.map.find(toId);
    if (it == table.map.end()) {
      table.order.push_back(toId);
      table.map[toId] = {weight, direction};
    } else {
      it->second.weight += weight;
      it->second.direction = direction;
    }
    meta_[fromId]["degree"] = (int)table.map.size();
    meta_[fromId]["lastTouched"] =
        (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
    persistNode(fromId);
    if (direction == 0) {
      auto &back = nodes_[toId];
      auto it2 = back.map.find(fromId);
      if (it2 == back.map.end()) {
        back.order.push_back(fromId);
        back.map[fromId] = {weight, direction};
      } else {
        it2->second.weight += weight;
        it2->second.direction = direction;
      }
      meta_[toId]["degree"] = (int)back.map.size();
      meta_[toId]["lastTouched"] =
          (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
      persistNode(toId);
    }
  }

  void decayEdge(const std::string &fromId, const std::string &toId,
                 double factor) {
    std::lock_guard<std::mutex> lock(graphMu_);
    auto table = ensureRowLoaded(fromId);
    if (!table)
      return;
    auto it = table->map.find(toId);
    if (it == table->map.end())
      return;
    it->second.weight *= factor;
    if (it->second.weight < 1e-5) {
      table->map.erase(it);
      table->order.erase(
          std::remove(table->order.begin(), table->order.end(), toId),
          table->order.end());
    }
    meta_[fromId]["degree"] = (int)table->map.size();
    meta_[fromId]["lastTouched"] =
        (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
    persistNode(fromId);
  }

  std::vector<std::string> getAllPoints() {
    std::lock_guard<std::mutex> lock(graphMu_);
    loadAll();
    std::vector<std::string> ids;
    ids.reserve(metaOrder_.size());
    for (auto &id : metaOrder_) {
      if (meta_.count(id))
        ids.push_back(id);
    }
    return ids;
  }

  struct WindowInfo {
    CSRMatrix csr;
    std::unordered_map<std::string, uint32_t> index;
    std::vector<std::string> ids;
  };

  WindowInfo buildWindow(const std::vector<std::string> &seedIds,
                         int radius = 1) {
    std::lock_guard<std::mutex> lock(graphMu_);
    std::unordered_set<std::string> visited;
    std::vector<std::string> ids;
    ids.reserve(seedIds.size());
    for (auto &id : seedIds) {
      if (visited.insert(id).second)
        ids.push_back(id);
    }
    std::vector<std::string> border = ids;
    int depth = 0;
    while (!border.empty() && visited.size() < 4096 && depth < radius) {
      std::vector<std::string> next;
      for (const auto &id : border) {
        auto table = ensureRowLoaded(id);
        if (!table)
          continue;
        for (auto &to : table->order) {
          if (visited.insert(to).second) {
            next.push_back(to);
            ids.push_back(to);
          }
        }
      }
      border = next;
      depth++;
    }
    std::unordered_map<std::string, uint32_t> index;
    for (uint32_t i = 0; i < ids.size(); i++)
      index[ids[i]] = i;
    CSRMatrix csr;
    csr.nRows = csr.nCols = (uint32_t)ids.size();
    csr.rowPtr.resize(ids.size() + 1);
    for (uint32_t i = 0; i < ids.size(); i++) {
      auto table = ensureRowLoaded(ids[i]);
      csr.rowPtr[i] = (uint32_t)csr.colIdx.size();
      if (table) {
        for (auto &to : table->order) {
          auto itEdge = table->map.find(to);
          if (itEdge == table->map.end())
            continue;
          auto it = index.find(to);
          if (it == index.end())
            continue;
          csr.colIdx.push_back(it->second);
          csr.values.push_back((float)itEdge->second.weight);
        }
      }
    }
    csr.rowPtr[ids.size()] = (uint32_t)csr.colIdx.size();
    return {csr, index, ids};
  }

  std::vector<std::vector<float>>
  messagePassingEmbeddings(const std::vector<std::string> &ids,
                           const std::vector<std::vector<float>> &baseEmb,
                           int rounds = 1, float selfWeight = 0.6f,
                           float neighborWeight = 0.4f) {
    std::lock_guard<std::mutex> lock(graphMu_);
    if (ids.empty() || baseEmb.empty())
      return baseEmb;
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < ids.size(); i++)
      index[ids[i]] = i;
    std::vector<std::vector<float>> cur = baseEmb;
    for (int r = 0; r < rounds; r++) {
      std::vector<std::vector<float>> next = cur;
      for (size_t i = 0; i < ids.size(); i++) {
        auto table = ensureRowLoaded(ids[i]);
        if (!table)
          continue;
        std::vector<float> agg(cur[i].size(), 0.0f);
        double wsum = 0.0;
        for (auto &to : table->order) {
          auto it = table->map.find(to);
          if (it == table->map.end())
            continue;
          auto itIdx = index.find(to);
          if (itIdx == index.end())
            continue;
          double w = it->second.weight;
          if (it->second.direction == 1)
            w *= 0.85; // inbound
          else if (it->second.direction == 2)
            w *= 1.15; // outbound
          for (size_t d = 0; d < agg.size() && d < cur[itIdx->second].size();
               d++) {
            agg[d] += (float)(w * cur[itIdx->second][d]);
          }
          wsum += std::abs(w);
        }
        if (wsum > 0.0) {
          for (auto &v : agg)
            v = (float)(v / wsum);
        }
        for (size_t d = 0; d < next[i].size() && d < agg.size(); d++) {
          next[i][d] = selfWeight * cur[i][d] + neighborWeight * agg[d];
        }
        float norm = 0.0f;
        for (float v : next[i])
          norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0.0f)
          for (auto &v : next[i])
            v /= norm;
      }
      cur = std::move(next);
    }
    return cur;
  }

  json exportSnapshot() {
    std::lock_guard<std::mutex> lock(graphMu_);
    loadAll();
    json nodes = json::array();
    for (auto &id : nodeOrder_) {
      auto it = nodes_.find(id);
      if (it == nodes_.end())
        continue;
      json neighbors = json::array();
      for (auto &to : it->second.order) {
        auto itEdge = it->second.map.find(to);
        if (itEdge == it->second.map.end())
          continue;
        neighbors.push_back({{"to", to},
                             {"weight", itEdge->second.weight},
                             {"direction", itEdge->second.direction}});
      }
      nodes.push_back({{"id", id}, {"neighbors", neighbors}});
    }
    json meta = json::object();
    for (auto &id : metaOrder_) {
      auto it = meta_.find(id);
      if (it != meta_.end())
        meta[id] = it->second;
    }
    return {{"nodes", nodes}, {"meta", meta}};
  }

  std::vector<std::pair<std::string, std::pair<double, int>>>
  neighborsOf(const std::string &memeId) {
    std::lock_guard<std::mutex> lock(graphMu_);
    auto table = ensureRowLoaded(memeId);
    std::vector<std::pair<std::string, std::pair<double, int>>> out;
    if (!table)
      return out;
    for (auto &to : table->order) {
      auto it = table->map.find(to);
      if (it == table->map.end())
        continue;
      out.push_back({to, {it->second.weight, it->second.direction}});
    }
    return out;
  }

  int degreeOf(const std::string &memeId) {
    std::lock_guard<std::mutex> lock(graphMu_);
    auto table = ensureRowLoaded(memeId);
    return table ? (int)table->map.size() : 0;
  }

  int removeOutgoingEdges(const std::string &memeId) {
    std::lock_guard<std::mutex> lock(graphMu_);
    auto tablePtr = ensureRowLoaded(memeId);
    if (!tablePtr)
      return 0;
    auto &table = nodes_[memeId];
    int cut = 0;
    for (auto it = table.order.begin(); it != table.order.end();) {
      auto itEdge = table.map.find(*it);
      if (itEdge != table.map.end() && itEdge->second.direction == 2) {
        std::string targetId = *it;
        table.map.erase(itEdge);
        it = table.order.erase(it);
        cut++;
        auto targetTable = ensureRowLoaded(targetId);
        if (targetTable) {
          auto &t = nodes_[targetId];
          for (auto it2 = t.order.begin(); it2 != t.order.end();) {
            auto itEdge2 = t.map.find(*it2);
            if (itEdge2 != t.map.end() && *it2 == memeId &&
                itEdge2->second.direction == 1) {
              t.map.erase(itEdge2);
              it2 = t.order.erase(it2);
            } else {
              ++it2;
            }
          }
          meta_[targetId]["degree"] = (int)t.map.size();
          meta_[targetId]["lastTouched"] =
              (int64_t)std::chrono::system_clock::now()
                  .time_since_epoch()
                  .count();
          persistNode(targetId);
        }
      } else {
        ++it;
      }
    }
    meta_[memeId]["degree"] = (int)table.map.size();
    meta_[memeId]["lastTouched"] =
        (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
    persistNode(memeId);
    return cut;
  }
  bool removeNode(const std::string &memeId) {
    std::lock_guard<std::mutex> lock(graphMu_);
    loadAll();
    if (!nodes_.count(memeId))
      return false;
    nodes_.erase(memeId);
    store_->del("row:" + memeId);
    meta_.erase(memeId);
    store_->del("node:" + memeId);
    nodeOrder_.erase(std::remove(nodeOrder_.begin(), nodeOrder_.end(), memeId),
                     nodeOrder_.end());
    metaOrder_.erase(std::remove(metaOrder_.begin(), metaOrder_.end(), memeId),
                     metaOrder_.end());
    for (auto &kv : nodes_) {
      auto &table = kv.second;
      if (table.map.erase(memeId)) {
        table.order.erase(
            std::remove(table.order.begin(), table.order.end(), memeId),
            table.order.end());
        meta_[kv.first]["degree"] = (int)table.map.size();
        meta_[kv.first]["lastTouched"] =
            (int64_t)std::chrono::system_clock::now()
                .time_since_epoch()
                .count();
        persistNode(kv.first);
      }
    }
    return true;
  }

  json getMeta(const std::string &memeId) {
    std::lock_guard<std::mutex> lock(graphMu_);
    loadAll();
    auto it = meta_.find(memeId);
    return it == meta_.end() ? json::object() : it->second;
  }

  void setMeta(const std::string &memeId, const json &meta) {
    std::lock_guard<std::mutex> lock(graphMu_);
    if (!meta_.count(memeId))
      metaOrder_.push_back(memeId);
    meta_[memeId] = meta;
    store_->put("node:" + memeId, meta);
  }

  std::vector<std::pair<std::string, json>> metaEntries() {
    std::lock_guard<std::mutex> lock(graphMu_);
    loadAll();
    std::vector<std::pair<std::string, json>> out;
    out.reserve(meta_.size());
    for (auto &id : metaOrder_) {
      auto it = meta_.find(id);
      if (it != meta_.end())
        out.push_back({id, it->second});
    }
    return out;
  }

  /**
   * @brief Attach semantic units to a graph node (v7.0 true-multimodal memory).
   */
  void setNodeSemanticUnits(const std::string &memeId,
                            const std::vector<phoenix::multimodal::SemanticUnit> &units) {
    std::lock_guard<std::mutex> lock(graphMu_);
    ensureNodeLocked(memeId);
    nodeSemanticUnits_[memeId] = units;
  }

  /**
   * @brief Retrieve semantic units attached to a node.
   */
  std::vector<phoenix::multimodal::SemanticUnit> getNodeSemanticUnits(
      const std::string &memeId) const {
    std::lock_guard<std::mutex> lock(graphMu_);
    auto it = nodeSemanticUnits_.find(memeId);
    if (it != nodeSemanticUnits_.end())
      return it->second;
    return {};
  }

  /**
   * @brief Find top-K nodes whose semantic units are most similar to a query.
   */
  std::vector<std::pair<std::string, float>> semanticSearch(
      const phoenix::multimodal::SemanticUnit &query,
      size_t topK = 5) {
    std::lock_guard<std::mutex> lock(graphMu_);
    loadAll();
    std::vector<std::pair<std::string, float>> scored;
    for (const auto &kv : nodeSemanticUnits_) {
      float best = 0.0f;
      for (const auto &unit : kv.second) {
        float sim = phoenix::multimodal::cosineSimilarity(query.semanticVector, unit.semanticVector);
        if (sim > best)
          best = sim;
      }
      scored.emplace_back(kv.first, best);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    if (scored.size() > topK)
      scored.resize(topK);
    return scored;
  }

  void importSnapshot(const json &snapshot) {
    std::lock_guard<std::mutex> lock(graphMu_);
    nodes_.clear();
    meta_.clear();
    nodeOrder_.clear();
    metaOrder_.clear();
    nodeSemanticUnits_.clear();
    for (auto &item : snapshot.value("nodes", json::array())) {
      std::string id = item.value("id", "");
      if (id.empty())
        continue;
      NodeEdges node;
      for (auto &n : item.value("neighbors", json::array())) {
        std::string to = n.value("to", "");
        if (to.empty())
          continue;
        if (node.map.find(to) == node.map.end())
          node.order.push_back(to);
        node.map[to] = {n.value("weight", 0.0), n.value("direction", 0)};
      }
      nodes_[id] = std::move(node);
      nodeOrder_.push_back(id);
      persistNode(id);
    }
    if (snapshot.contains("semanticUnits") && snapshot["semanticUnits"].is_object()) {
      for (auto it = snapshot["semanticUnits"].begin();
           it != snapshot["semanticUnits"].end(); ++it) {
        std::vector<phoenix::multimodal::SemanticUnit> units;
        for (auto &u : it.value()) {
          units.push_back(phoenix::multimodal::SemanticUnit::fromJson(u));
        }
        nodeSemanticUnits_[it.key()] = std::move(units);
      }
    }
    json meta = snapshot.value("meta", json::object());
    for (auto it = meta.begin(); it != meta.end(); ++it) {
      meta_[it.key()] = it.value();
      metaOrder_.push_back(it.key());
      store_->put("node:" + it.key(), it.value());
    }
  }

private:
  mutable std::mutex graphMu_;
  struct Edge {
    double weight{0};
    int direction{0};
  };
  struct NodeEdges {
    std::vector<std::string> order;
    std::unordered_map<std::string, Edge> map;
  };
  std::shared_ptr<KeyValueStore> store_;
  std::unordered_map<std::string, NodeEdges> nodes_;
  std::vector<std::string> nodeOrder_;
  std::unordered_map<std::string, json> meta_;
  std::vector<std::string> metaOrder_;
  std::unordered_map<std::string, std::vector<phoenix::multimodal::SemanticUnit>> nodeSemanticUnits_;
  bool fullyLoaded_{false};

  void loadAll() {
    if (fullyLoaded_)
      return;
    auto metaEntries = store_->entries("node:");
    for (auto &kv : metaEntries) {
      std::string id = kv.first.substr(5);
      meta_[id] = kv.second;
      if (std::find(metaOrder_.begin(), metaOrder_.end(), id) ==
          metaOrder_.end())
        metaOrder_.push_back(id);
    }
    auto rowEntries = store_->entries("row:");
    for (auto &kv : rowEntries) {
      std::string id = kv.first.substr(4);
      NodeEdges node;
      for (auto &n : kv.second.value("neighbors", json::array())) {
        std::string to = n.value("to", "");
        if (to.empty())
          continue;
        if (node.map.find(to) == node.map.end())
          node.order.push_back(to);
        node.map[to] = {n.value("weight", 0.0), n.value("direction", 0)};
      }
      if (std::find(nodeOrder_.begin(), nodeOrder_.end(), id) ==
          nodeOrder_.end())
        nodeOrder_.push_back(id);
      nodes_[id] = std::move(node);
    }
    fullyLoaded_ = true;
  }

  NodeEdges *ensureRowLoaded(const std::string &id) {
    auto it = nodes_.find(id);
    if (it != nodes_.end())
      return &it->second;
    auto row = store_->get("row:" + id);
    if (row && row->is_object()) {
      NodeEdges node;
      for (auto &n : row->value("neighbors", json::array())) {
        std::string to = n.value("to", "");
        if (to.empty())
          continue;
        if (node.map.find(to) == node.map.end())
          node.order.push_back(to);
        node.map[to] = {n.value("weight", 0.0), n.value("direction", 0)};
      }
      nodes_[id] = std::move(node);
      if (std::find(nodeOrder_.begin(), nodeOrder_.end(), id) ==
          nodeOrder_.end())
        nodeOrder_.push_back(id);
      return &nodes_[id];
    }
    return nullptr;
  }

  void persistNode(const std::string &id) {
    auto &table = nodes_[id];
    json neighbors = json::array();
    for (auto &to : table.order) {
      auto it = table.map.find(to);
      if (it == table.map.end())
        continue;
      neighbors.push_back({{"to", to},
                           {"weight", it->second.weight},
                           {"direction", it->second.direction}});
    }
    store_->put("row:" + id, json{{"neighbors", neighbors}});
  }
};
