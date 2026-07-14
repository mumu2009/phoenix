// Auto-generated from main.cpp
// Segment: class_SnapshotManager
// Original lines: 12909-12983

class SnapshotManager : public ISnapshotManager {
public:
  SnapshotManager(std::shared_ptr<RuntimeBase> runtime, fs::path dir)
      : runtime_(std::move(runtime)), dir_(std::move(dir)) {
    ensureDir(dir_);
  }

  std::vector<std::string> list() const override {
    std::vector<std::string> out;
    if (!fs::exists(dir_))
      return out;
    for (auto &e : fs::directory_iterator(dir_)) {
      if (e.path().extension() == ".json" && e.is_regular_file())
        out.push_back(e.path().filename().string());
    }
    std::sort(out.begin(), out.end(), std::greater<std::string>());
    return out;
  }

  fs::path create(const std::string &name = "auto") override {
    json snap = runtime_->toSnapshot();
    json doc = json::object();
    doc["meta"] = json{{"version", 1}, {"createdAt", nowIso()}, {"tag", name}};
    doc["snapshot"] = snap;
    auto ts = nowIso();
    std::string file = ts;
    std::replace(file.begin(), file.end(), ':', '-');
    std::replace(file.begin(), file.end(), '.', '-');
    file += "_" + name + ".json";
    fs::path out = dir_ / file;
    fs::path tmp = out;
    tmp += ".tmp";
    {
      std::ofstream o(tmp, std::ios::trunc);
      o << doc.dump(2);
    }
    if (fs::exists(out))
      fs::remove(out);
    fs::rename(tmp, out);

    auto files = list();
    const size_t maxKeep = 200;
    for (size_t i = maxKeep; i < files.size(); i++) {
      try {
        fs::remove(dir_ / files[i]);
      } catch (...) {
      }
    }
    return out;
  }

  json restore(const std::string &id) override {
    fs::path file = dir_ / fs::path(id).filename();
    if (!fs::exists(file))
      throw std::runtime_error("snapshot not found");
    std::ifstream in(file);
    json snap;
    in >> snap;
    if (snap.is_object() && snap.contains("snapshot"))
      snap = snap["snapshot"];
    runtime_->fromSnapshot(snap);
    return snap;
  }

  void remove(const std::string &id) override {
    fs::path file = dir_ / fs::path(id).filename();
    if (fs::exists(file)) {
      fs::remove(file);
    }
  }

private:
  std::shared_ptr<RuntimeBase> runtime_;
  fs::path dir_;
};
