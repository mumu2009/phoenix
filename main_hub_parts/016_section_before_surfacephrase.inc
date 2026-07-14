/* Auto-generated from main.cpp
   Segment: section_before_SurfacePhrase
   Original lines: 2766-2949
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

/* normalizeBudget normalizes frontend budget configuration structure */
// 调用方式：传入原始 JSON 并获取可选标准化结果。
// 实现思路：按类型分支做兼容转换与字段清洗。
// 注意事项：非法输入返回空值，调用方应提供默认预算。
static std::optional<json> normalizeBudget(const json &raw) {
  if (raw.is_null() || raw.is_boolean())
    return std::nullopt;
  if (raw.is_string()) {
    auto s = raw.get<std::string>();
    std::string lowered;
    lowered.reserve(s.size());
    for (char c : s)
      lowered.push_back((char)std::tolower((unsigned char)c));
    if (lowered == "default" || lowered == "balanced" || lowered == "medium" ||
        lowered == "none")
      return std::nullopt;
    if (lowered == "low" || lowered == "fast")
      return json{{"iteration", 3},
                  {"reflectionTopMemes", 12},
                  {"reflectionTopWords", 16}};
    if (lowered == "high" || lowered == "slow" || lowered == "quality")
      return json{{"iteration", 7},
                  {"reflectionTopMemes", 22},
                  {"reflectionTopWords", 32}};
    if (!s.empty() && s.front() == '{' && s.back() == '}') {
      try {
        return normalizeBudget(json::parse(s));
      } catch (...) {
        return std::nullopt;
      }
    }
    return std::nullopt;
  }
  if (!raw.is_object())
    return std::nullopt;
  json out = json::object();
  auto pickNum = [&](const std::string &key,
                     const std::vector<std::string> &aliases) {
    if (raw.contains(key) && raw[key].is_number()) {
      out[key] = raw[key].get<double>();
      return;
    }
    for (auto &a : aliases) {
      if (raw.contains(a) && raw[a].is_number()) {
        out[key] = raw[a].get<double>();
        return;
      }
    }
  };
  pickNum("mappingDepth", {"depth"});
  pickNum("iteration", {"iters"});
  pickNum("reflectionTopMemes", {"topMemes"});
  pickNum("reflectionTopWords", {"topWords"});
  pickNum("reflectionMinScore", {"minScore"});
  pickNum("radius", {"windowRadius"});
  return out.empty() ? std::nullopt : std::optional<json>(out);
}

static std::optional<json> mergeBudgets(const std::optional<json> &base,
                                        const std::optional<json> &override) {
  if (!base && !override)
    return std::nullopt;
  if (!base)
    return override;
  if (!override)
    return base;
  json out = *base;
  for (auto it = override->begin(); it != override->end(); ++it)
    out[it.key()] = it.value();
  return out;
}

static std::vector<std::string> splitSentences(const std::string &text) {
  std::string raw = text;
  raw.erase(0, raw.find_first_not_of(" \n\r\t"));
  if (raw.empty())
    return {};
  raw.erase(raw.find_last_not_of(" \n\r\t") + 1);

  std::vector<std::string> rough;
  std::string cur;
  for (size_t i = 0; i < raw.size(); i++) {
    char c = raw[i];
    if (c == '\r' || c == '\n') {
      auto t = cur;
      t.erase(0, t.find_first_not_of(" \n\r\t"));
      if (!t.empty()) {
        t.erase(t.find_last_not_of(" \n\r\t") + 1);
        if (!t.empty())
          rough.push_back(t);
      }
      cur.clear();
      continue;
    }
    cur.push_back(c);
    bool endSentence = (c == '!' || c == '?');
    size_t utf8PunctExtra = 0;
    if (!endSentence) {
      if (i + 2 < raw.size()) {
        unsigned char b1 = static_cast<unsigned char>(raw[i]);
        unsigned char b2 = static_cast<unsigned char>(raw[i + 1]);
        unsigned char b3 = static_cast<unsigned char>(raw[i + 2]);
        bool isCnPeriod = (b1 == 0xE3 && b2 == 0x80 && b3 == 0x82);
        bool isCnExclam = (b1 == 0xEF && b2 == 0xBC && b3 == 0x81);
        bool isCnQuestion = (b1 == 0xEF && b2 == 0xBC && b3 == 0x9F);
        if (isCnPeriod || isCnExclam || isCnQuestion) {
          endSentence = true;
          utf8PunctExtra = 2;
          cur.push_back(raw[i + 1]);
          cur.push_back(raw[i + 2]);
        }
      }
    }
    if (endSentence) {
      i += utf8PunctExtra;
      auto t = cur;
      t.erase(0, t.find_first_not_of(" \n\r\t"));
      if (!t.empty()) {
        t.erase(t.find_last_not_of(" \n\r\t") + 1);
        if (!t.empty())
          rough.push_back(t);
      }
      cur.clear();
      while (i + 1 < raw.size()) {
        char n = raw[i + 1];
        if (n == ' ' || n == '\t' || n == '\r' || n == '\n') {
          i++;
          continue;
        }
        break;
      }
    }
  }
  if (!cur.empty()) {
    auto t = cur;
    t.erase(0, t.find_first_not_of(" \n\r\t"));
    if (!t.empty()) {
      t.erase(t.find_last_not_of(" \n\r\t") + 1);
      if (!t.empty())
        rough.push_back(t);
    }
  }

  const int maxWords = 10;
  const int minWords = 2;
  std::vector<std::string> out;

  for (const auto &unit : rough) {
    auto tokens = tokenize(unit);
    if ((int)tokens.size() < minWords)
      continue;
    for (size_t i = 0; i < tokens.size(); i += maxWords) {
      std::ostringstream oss;
      int count = 0;
      for (size_t j = i; j < tokens.size() && j < i + maxWords; j++) {
        if (count++)
          oss << " ";
        oss << tokens[j];
      }
      if (count < minWords) {
        if (!out.empty() && count > 0) {
          out.back() += " " + oss.str();
        }
        continue;
      }
      out.push_back(oss.str());
      if (out.size() >= 12)
        return out;
    }
    if (out.size() >= 12)
      break;
  }
  if (out.size() > 12)
    out.resize(12);
  return out;
}

// SurfacePhrase 表示候选短语及其权重信息。
// 调用方式：作为短语抽取结果元素参与排序与筛选。
// 实现思路：用 phrase 与 weight 两字段描述语义片段。
// 注意事项：weight 仅表示相对强度，不是概率值。
// 注意事项：短语文本应在上游完成基本清洗。
// 注意事项：该结构不负责去重与阈值过滤策略。
