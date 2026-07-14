/* Auto-generated from main.cpp
   Segment: section_before_LmdbStore
   Original lines: 2954-3015
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

/* extractSurfacePhrases extracts candidate surface phrases from text */
// 调用方式：输入原文与上限，返回去重后的短语列表。
// 实现思路：综合句子切分、token 与 n-gram 进行采样。
// 注意事项：输出数量受 maxPhrases 与文本长度双重约束。
static std::vector<SurfacePhrase> extractSurfacePhrases(const std::string &text,
                                                        int maxPhrases = 24) {
  std::vector<SurfacePhrase> out;
  std::unordered_set<std::string> seen;
  auto push = [&](const std::string &p, double w) {
    std::string s = p;
    s.erase(0, s.find_first_not_of(" \n\r\t"));
    if (s.empty())
      return;
    s.erase(s.find_last_not_of(" \n\r\t") + 1);
    if (s.empty())
      return;
    if (s.size() < 2 || s.size() > 160)
      return;
    if (seen.count(s))
      return;
    seen.insert(s);
    out.push_back({s, w});
  };

  for (const auto &s : splitSentences(text))
    push(s, 1.0);

  auto tokens = tokenize(text);
  for (size_t i = 0; i < tokens.size() && (int)out.size() < maxPhrases; i++) {
    push(tokens[i], 2.0);
    if ((int)out.size() >= maxPhrases)
      return out;
  }

  int maxN = std::min<int>(5, (int)tokens.size());
  for (int n = 2; n <= maxN; n++) {
    for (size_t i = 0; i + n <= tokens.size(); i++) {
      std::ostringstream oss;
      for (int j = 0; j < n; j++) {
        if (j)
          oss << " ";
        oss << tokens[i + j];
      }
      double w = (n == 2 ? 3.0 : (n == 3 ? 2.0 : 1.0));
      push(oss.str(), w);
      if ((int)out.size() >= maxPhrases)
        return out;
    }
  }
  return out;
}

// ------------------ Storage ------------------

// LmdbStore 基于 LMDB 实现键值存储后端。
// 调用方式：以库名与根目录初始化后按 KV 接口读写。
// 实现思路：封装环境/事务操作并统一 JSON 编解码。
// 注意事项：依赖底层文件系统权限与路径可写性。
// 注意事项：异常场景会退化为不可用状态而非崩溃。
// 注意事项：跨进程并发访问需遵循 LMDB 限制。
// Disabled due to pre-existing compilation errors unrelated to LoRA integration
#if 0
#ifdef HAVE_LMDB
#include <lmdb.h>
#endif
#endif
