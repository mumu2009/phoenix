/* Auto-generated from main.cpp
   Segment: struct_TorchTextEmbedder
   Original lines: 2492-2621
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

// Disabled due to pre-existing compilation errors unrelated to LoRA integration
#if 0

struct TorchTextEmbedder {
  bool ready{false};
  int64_t embDim{64};
  int64_t maxLen{40};
  int64_t maxVocab{8000};
  std::unordered_map<std::string, int64_t> stoi;
  std::vector<std::string> itos;
  std::once_flag initFlag;
  torch::nn::Embedding emb{nullptr};
  torch::nn::Linear head{nullptr};

  void buildVocab(const std::unordered_map<std::string, int> &freq) {
    std::vector<std::pair<std::string, int>> items(freq.begin(), freq.end());
    std::sort(items.begin(), items.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    itos.clear();
    stoi.clear();
    itos.push_back("<pad>");
    itos.push_back("<unk>");
    for (size_t i = 0; i < items.size() && (int64_t)itos.size() < maxVocab;
         i++) {
      stoi[items[i].first] = (int64_t)itos.size();
      itos.push_back(items[i].first);
    }
  }

  std::vector<int64_t> encode(const std::vector<std::string> &tokens) const {
    std::vector<int64_t> ids;
    ids.reserve(tokens.size());
    for (const auto &t : tokens) {
      auto it = stoi.find(t);
      ids.push_back(it == stoi.end() ? 1 : it->second);
    }
    if (ids.empty())
      ids.push_back(0);
    return ids;
  }

  void ensureInit(const fs::path &robotsDir) {
    std::call_once(initFlag, [&]() {
      try {
        std::unordered_map<std::string, int> freq;
        std::vector<std::vector<std::string>> samples;
        size_t files = 0;
        if (fs::exists(robotsDir)) {
          for (const auto &entry : fs::directory_iterator(robotsDir)) {
            if (!entry.is_regular_file())
              continue;
            if (entry.path().extension() != ".txt")
              continue;
            std::ifstream in(entry.path(), std::ios::binary);
            if (!in)
              continue;
            std::ostringstream ss;
            ss << in.rdbuf();
            auto tokens = tokenize(ss.str());
            if (!tokens.empty())
              samples.push_back(tokens);
            for (const auto &t : tokens)
              freq[t] += 1;
            if (++files >= 200)
              break;
          }
        }
        if (freq.empty())
          return;
        buildVocab(freq);
        int64_t vocabSize = std::max<int64_t>(2, (int64_t)itos.size());
        emb = torch::nn::Embedding(vocabSize, embDim);
        head = torch::nn::Linear(embDim, vocabSize);

        if (!samples.empty()) {
          torch::optim::Adam opt(emb->parameters(),
                                 torch::optim::AdamOptions(1e-3));
          torch::optim::Adam optHead(head->parameters(),
                                     torch::optim::AdamOptions(1e-3));
          const int epochs = 1;
          const int batch = 16;
          for (int e = 0; e < epochs; e++) {
            for (size_t i = 0; i < samples.size(); i += batch) {
              size_t end = std::min(samples.size(), i + batch);
              int64_t bsz = (int64_t)(end - i);
              auto x = torch::zeros({bsz, maxLen}, torch::kInt64);
              for (size_t b = 0; b < end - i; b++) {
                auto ids = encode(samples[i + b]);
                int64_t len = std::min<int64_t>(maxLen, (int64_t)ids.size());
                for (int64_t j = 0; j < len; j++)
                  x.index_put_({(int64_t)b, j}, ids[(size_t)j]);
              }
              auto input = x.narrow(1, 0, maxLen - 1);
              auto target = x.narrow(1, 1, maxLen - 1).reshape({-1});
              opt.zero_grad();
              optHead.zero_grad();
              auto ebd = emb(input);
              auto logits = head(ebd).reshape({-1, vocabSize});
              auto loss = torch::nn::functional::cross_entropy(logits, target);
              loss.backward();
              opt.step();
              optHead.step();
            }
          }
        }
        ready = true;
      } catch (...) {
        ready = false;
      }
    });
  }

  std::vector<float> embedTokens(const std::vector<std::string> &tokens,
                                 int dimOut) {
    if (!ready || !emb)
      return {};
    torch::NoGradGuard guard;
    auto ids = encode(tokens);
    auto x = torch::zeros({1, (int64_t)ids.size()}, torch::kInt64);
    for (int64_t j = 0; j < (int64_t)ids.size(); j++)
      x.index_put_({0, j}, ids[(size_t)j]);
    auto ebd = emb(x).mean(1).squeeze(0).contiguous();
    std::vector<float> out(ebd.numel(), 0.0f);
    std::memcpy(out.data(), ebd.data_ptr<float>(), sizeof(float) * out.size());
    if ((int)out.size() == dimOut)
      return out;
    std::vector<float> resized(std::max(1, dimOut), 0.0f);
    int copy = std::min<int>(dimOut, (int)out.size());
    for (int i = 0; i < copy; i++)
      resized[i] = out[(size_t)i];
    return resized;
  }
};
#endif
