// Auto-generated from main.cpp
// Segment: section_before_SessionManager
// Original lines: 5564-5850


// RobotsCorpus 管理 robots 语料加载、切分与标准化。
// 调用方式：在训练热身或重建阶段按选项批量调用。
// 实现思路：遍历文本文件并结合词形还原构建文档。
// 注意事项：目录规模较大时建议配合 limit/offset。
// 注意事项：读取异常不会中断全局流程，需看返回结果。
// 注意事项：分块参数会影响后续学习效果。
class RobotsCorpus {
public:
  RobotsCorpus(fs::path dir, fs::path lemmaCsv, bool lemmaAutoload,
               std::size_t lemmaMaxBytes, bool lemmaForce, int chunkMinWords,
               int chunkMaxWords)
      : dir_(std::move(dir)), lemmaCsv_(std::move(lemmaCsv)),
        lemmaAutoload_(lemmaAutoload), lemmaForce_(lemmaForce),
        lemmaMaxBytes_(lemmaMaxBytes > 0 ? lemmaMaxBytes
                                         : 64ull * 1024ull * 1024ull),
        chunkMinWords_(std::max(1, chunkMinWords)),
        chunkMaxWords_(std::max(chunkMinWords_, chunkMaxWords)) {}

  std::vector<std::string> listFiles() const {
    std::vector<std::string> files;
    if (dir_.empty() || !fs::exists(dir_))
      return files;
    try {
      for (auto &entry : fs::directory_iterator(dir_)) {
        if (!entry.is_regular_file())
          continue;
        auto ext = entry.path().extension().string();
        std::string lowerExt;
        lowerExt.reserve(ext.size());
        for (char c : ext)
          lowerExt.push_back(
              static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        if (lowerExt == ".txt")
          files.push_back(entry.path().filename().string());
      }
      std::sort(files.begin(), files.end());
    } catch (const std::exception &e) {
      std::cerr << "[RobotsCorpus] Failed to list files: " << e.what()
                << std::endl;
    }
    return files;
  }

  std::vector<std::string>
  normalizeWords(const std::vector<std::string> &words) {
    std::vector<std::string> out;
    out.reserve(words.size());
    for (auto &w : words) {
      auto lw = lemmatize(w);
      if (!lw.empty())
        out.push_back(lw);
    }
    return out;
  }

  std::vector<RobotsDocument> collect(const json &options) {
    std::vector<RobotsDocument> docs;
    auto files = listFiles();
    if (options.contains("files") && options["files"].is_array()) {
      std::unordered_set<std::string> allow;
      for (auto &v : options["files"])
        if (v.is_string())
          allow.insert(v.get<std::string>());
      files.erase(
          std::remove_if(files.begin(), files.end(),
                         [&](const std::string &f) { return !allow.count(f); }),
          files.end());
    }
    if (options.value("shuffle", false) && files.size() > 1) {
      std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count());
      std::shuffle(files.begin(), files.end(), rng);
    }
    int offset = options.value("offset", 0);
    if (offset < 0)
      offset = 0;
    int limit = options.contains("limit") ? options.value("limit", 0) : 0;
    int maxDocs = (limit > 0) ? limit : 0;

    for (const auto &file : files) {
      auto content = readFileLimited(dir_ / file, maxArticleSize_);
      if (content.empty())
        continue;
      auto paragraphs = splitParagraphs(content);
      int localIndex = 0;
      for (const auto &paragraph : paragraphs) {
        std::string trimmed = paragraph;
        trimmed.erase(0, trimmed.find_first_not_of(" \n\r\t"));
        trimmed.erase(trimmed.find_last_not_of(" \n\r\t") + 1);
        if ((int)trimmed.size() < minParagraphLength_)
          continue;
        auto units = splitSentences(trimmed);
        for (const auto &unit : units) {
          std::string unitText = unit;
          unitText.erase(0, unitText.find_first_not_of(" \n\r\t"));
          unitText.erase(unitText.find_last_not_of(" \n\r\t") + 1);
          if ((int)unitText.size() < minParagraphLength_)
            continue;
          auto tokens = normalizeWords(tokenize(unitText));
          if (tokens.empty())
            continue;
          if (offset > 0) {
            offset--;
            continue;
          }
          RobotsDocument doc;
          doc.id = "robots:" + file + "#" + std::to_string(localIndex);
          doc.file = file;
          doc.source = "robots:" + file;
          doc.text = unitText;
          doc.tokens = std::move(tokens);
          docs.push_back(std::move(doc));
          localIndex++;
          if (maxDocs > 0 && (int)docs.size() >= maxDocs)
            return docs;
        }
      }
    }
    if (options.value("shuffle", false) && docs.size() > 1) {
      std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count());
      std::shuffle(docs.begin(), docs.end(), rng);
    }
    return docs;
  }

  std::string lemmatize(const std::string &word) {
    ensureLemmaLoaded();
    std::string lower;
    lower.reserve(word.size());
    for (unsigned char c : word) {
      if (c < 0x80)
        lower.push_back(static_cast<char>(std::tolower(c)));
      else
        lower.push_back(static_cast<char>(c));
    }
    auto it = lemmaMap_.find(lower);
    return it == lemmaMap_.end() ? lower : it->second;
  }

private:
  fs::path dir_;
  fs::path lemmaCsv_;
  bool lemmaAutoload_{false};
  bool lemmaForce_{false};
  std::size_t lemmaMaxBytes_{64ull * 1024ull * 1024ull};
  int chunkMinWords_{2};
  int chunkMaxWords_{5};
  bool lemmaLoaded_{false};
  std::unordered_map<std::string, std::string> lemmaMap_;
  std::size_t maxArticleSize_{5000000};
  int minParagraphLength_{12};

  void ensureLemmaLoaded() {
    if (lemmaLoaded_)
      return;
    lemmaLoaded_ = true;
    if (!lemmaAutoload_ && !lemmaForce_)
      return;
    lemmaMap_ = loadLemmaMap();
  }

  std::unordered_map<std::string, std::string> loadLemmaMap() {
    std::unordered_map<std::string, std::string> map;
    if (lemmaCsv_.empty() || !fs::exists(lemmaCsv_))
      return map;
    try {
      if (!lemmaForce_) {
        try {
          auto st = fs::file_size(lemmaCsv_);
          if (st > lemmaMaxBytes_) {
            std::cerr << "[RobotsCorpus] lemma.csv too large ("
                      << (st / 1024 / 1024)
                      << "MB), skip autoload. Set AI_LEMMA_FORCE=true or "
                         "increase AI_LEMMA_MAX_MB to load."
                      << std::endl;
            return map;
          }
        } catch (...) {
        }
      }
      std::ifstream in(lemmaCsv_, std::ios::binary);
      std::string line;
      while (std::getline(in, line)) {
        if (line.empty())
          continue;
        auto cols = splitCsvLine(line);
        if (cols.empty())
          continue;
        std::string base = toLower(cols[0]);
        if (base.empty())
          continue;
        map[base] = base;
        for (auto &c : cols) {
          auto token = toLower(c);
          if (!token.empty())
            map[token] = base;
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "[RobotsCorpus] Failed to load lemma map: " << e.what()
                << std::endl;
    }
    return map;
  }

  static std::string toLower(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
      if (c < 0x80)
        out.push_back(static_cast<char>(std::tolower(c)));
      else
        out.push_back(static_cast<char>(c));
    }
    return out;
  }

  static std::vector<std::string> splitCsvLine(const std::string &line) {
    std::vector<std::string> cols;
    std::string cur;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); i++) {
      char c = line[i];
      if (c == '"') {
        if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
          cur.push_back('"');
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (c == ',' && !inQuotes) {
        cols.push_back(cur);
        cur.clear();
      } else {
        cur.push_back(c);
      }
    }
    cols.push_back(cur);
    return cols;
  }

  static std::string readFileLimited(const fs::path &file,
                                     std::size_t maxBytes) {
    try {
      std::ifstream in(file, std::ios::binary);
      if (!in)
        return {};
      std::string out;
      out.reserve(4096);
      char buf[4096];
      std::size_t total = 0;
      while (in && total < maxBytes) {
        in.read(buf, sizeof(buf));
        auto got = (std::size_t)in.gcount();
        if (!got)
          break;
        if (total + got > maxBytes)
          got = maxBytes - total;
        out.append(buf, got);
        total += got;
      }
      return out;
    } catch (...) {
      return {};
    }
  }

  static std::vector<std::string> splitParagraphs(const std::string &content) {
    std::vector<std::string> out;
    std::regex re(R"(\r?\n\s*\r?\n)");
    std::sregex_token_iterator it(content.begin(), content.end(), re, -1);
    std::sregex_token_iterator end;
    for (; it != end; ++it) {
      std::string s = it->str();
      s.erase(0, s.find_first_not_of(" \n\r\t"));
      s.erase(s.find_last_not_of(" \n\r\t") + 1);
      if (!s.empty())
        out.push_back(s);
    }
    return out;
  }
};

