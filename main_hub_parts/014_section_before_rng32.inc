/* Auto-generated from main.cpp
   Segment: section_before_Rng32
   Original lines: 2622-2752
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

#if 0
static TorchTextEmbedder &getTorchEmbedder() {
  static TorchTextEmbedder inst;
  return inst;
}
#endif
// #endif (disabled - no matching #if)
// #endif (disabled - no matching #if)

// Auto-generated from main.cpp
// Segment: section_tail
// Original lines: 2773-20423

static std::vector<float> textToMiniEmbedding(const std::string &text,
                                              int dim = 64) {
  std::vector<float> vec(std::max(1, dim), 0.0f);
  auto words = tokenize(text);
#ifdef HAVE_TORCH
  if (boolFrom(getEnv("AI_USE_TORCH_EMBED", "false"), false)) {
    auto &embedder = getTorchEmbedder();
    embedder.ensureInit(getProjectRoot() / "robots");
    if (embedder.ready) {
      auto out = embedder.embedTokens(words, dim);
      if (!out.empty())
        return out;
    }
  }
#endif
  for (const auto &w : words) {
    auto idx = hashStrSimple(w) % vec.size();
    vec[idx] += 1.0f;
  }
  if (words.size() > 1) {
    for (size_t i = 0; i + 1 < words.size(); i++) {
      std::string pair = words[i] + "_" + words[i + 1];
      auto idx = hashStrSimple(pair) % vec.size();
      vec[idx] += 0.5f;
    }
  }
  return vec;
}

static std::vector<std::string>
perturbTokens(const std::vector<std::string> &words) {
  if (words.size() <= 1)
    return words;
  size_t idx = (size_t)(std::rand() % words.size());
  std::vector<std::string> out;
  out.reserve(words.size() - 1);
  for (size_t i = 0; i < words.size(); i++) {
    if (i != idx)
      out.push_back(words[i]);
  }
  return out;
}

static std::vector<std::string> buildVariants(const std::string &text,
                                              int count = 0) {
  std::vector<std::string> variants;
  if (count <= 0)
    return variants;
  auto words = tokenize(text);
  for (int i = 0; i < count; i++) {
    words = perturbTokens(words);
    std::ostringstream oss;
    for (size_t j = 0; j < words.size(); j++) {
      if (j)
        oss << " ";
      oss << words[j];
    }
    variants.push_back(oss.str());
    if (words.size() <= 1)
      break;
  }
  return variants;
}

// clamp01 将数值限制在 [0,1] 区间。
// 调用方式：用于概率分数和比率结果归一。
// 实现思路：通过 min/max 双边裁剪实现。
// 注意事项：输入 NaN 会沿标准比较行为传播。
static double clamp01(double x) { return std::max(0.0, std::min(1.0, x)); }
// clamp11 将数值限制在 [-1,1] 区间。
// 调用方式：用于相似度和相关系数截断。
// 实现思路：通过 min/max 双边裁剪实现。
// 注意事项：超界值会被硬截断，可能损失梯度信息。
static double clamp11(double x) { return std::max(-1.0, std::min(1.0, x)); }

// clampInt 从 JSON 中读取整数并执行边界裁剪。
// 调用方式：读取请求参数时传入目标范围与回退值。
// 实现思路：先判断数值类型，再按 min/max 做夹取。
// 注意事项：非数值输入直接返回 fallback。
static int clampInt(const json &value, int minVal, int maxVal, int fallback) {
  if (!value.is_number())
    return fallback;
  int n = value.get<int>();
  if (n < minVal)
    return minVal;
  if (n > maxVal)
    return maxVal;
  return n;
}

// extractFirstUrl 从文本中提取首个可识别 URL。
// 调用方式：在线检索或链接跟踪前对用户文本调用。
// 实现思路：先匹配 http(s) 链接，再回退匹配 www 形式。
// 注意事项：匹配结果是启发式，调用方应做二次校验。
static std::string extractFirstUrl(const std::string &text) {
  std::string s = text;
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
            return !std::isspace(c);
          }));
  if (s.empty())
    return {};
  std::regex r1(R"(https?://[\w\-._~:/?#[\]@!$&'()*+,;=%]+)",
                std::regex::icase);
  std::smatch m;
  if (std::regex_search(s, m, r1) && m.size() > 0)
    return m[0].str();
  std::regex r2(
      R"(\b(?:www\.)[a-z0-9\-]+(?:\.[a-z0-9\-]+)+(?:/[\w\-._~:/?#[\]@!$&'()*+,;=%]*)?)",
      std::regex::icase);
  if (std::regex_search(s, m, r2) && m.size() > 0)
    return std::string("https://") + m[0].str();
  return {};
}

// Rng32 是一个轻量级可复现实用随机数发生器。
// 调用方式：以固定种子初始化后持续调用 next()。
// 实现思路：采用异或移位算法输出 [0,1] 区间随机值。
// 注意事项：该实现用于工程随机，不适用于安全场景。
// 注意事项：相同种子会生成相同序列，便于调试复现。
// 注意事项：多线程共享实例需外部同步。
