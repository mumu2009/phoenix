// Auto-generated from main.cpp
// Segment: section_before_LemmaWorkerPool
// Original lines: 6049-6144


static std::mutex gLemmaCacheMu;
static std::unordered_map<std::string, LemmaCacheEntry> gLemmaCache;

static std::string trimString(const std::string &s) {
  auto start = s.find_first_not_of(" \n\r\t");
  if (start == std::string::npos)
    return std::string();
  auto end = s.find_last_not_of(" \n\r\t");
  return s.substr(start, end - start + 1);
}

static std::string lowerAscii(const std::string &s) {
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

static std::vector<std::string> splitCsvLineSimple(const std::string &line) {
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
      cols.push_back(trimString(cur));
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  cols.push_back(trimString(cur));
  return cols;
}

static std::unordered_map<std::string, std::string>
loadLemmaMapCached(const std::string &lemmaCsvPath) {
  std::unordered_map<std::string, std::string> map;
  if (lemmaCsvPath.empty())
    return map;
  fs::path file(lemmaCsvPath);
  try {
    if (!fs::exists(file))
      return map;
    auto mtime = fs::last_write_time(file);
    {
      std::lock_guard<std::mutex> lock(gLemmaCacheMu);
      auto it = gLemmaCache.find(file.string());
      if (it != gLemmaCache.end() && it->second.mtime >= mtime) {
        return it->second.map;
      }
    }
    std::ifstream in(file, std::ios::binary);
    if (!in)
      return map;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty())
        continue;
      auto cols = splitCsvLineSimple(line);
      if (cols.empty())
        continue;
      std::string base = lowerAscii(cols[0]);
      base = trimString(base);
      if (base.empty())
        continue;
      map[base] = base;
      for (auto &c : cols) {
        auto token = trimString(lowerAscii(c));
        if (!token.empty())
          map[token] = base;
      }
    }
    {
      std::lock_guard<std::mutex> lock(gLemmaCacheMu);
      gLemmaCache[file.string()] = LemmaCacheEntry{mtime, map};
    }
  } catch (...) {
    return map;
  }
  return map;
}

