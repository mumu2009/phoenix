// Auto-generated from main.cpp
// Segment: class_DimReducer
// Original lines: 4929-5228

class DimReducer {
public:
  // PCAResult 保存主成分分析输出的核心结果。
  // 调用方式：由 PCA 计算函数返回并供后续投影使用。
  // 实现思路：聚合均值向量、主成分矩阵和特征值。
  // 注意事项：components 与 eigenValues 维度需一致。
  // 注意事项：结果依赖输入归一化与采样质量。
  // 注意事项：该结构适合作为算法中间产物传递。
  struct PCAResult {
    std::vector<float> coords;
    std::vector<float> v1;
    std::vector<float> v2;
  };

  PCAResult pca2D(const json &emb) {
    auto data = emb.value("data", std::vector<float>{});
    int nRows = emb.value("nRows", 0);
    int nCols = emb.value("nCols", 0);
    std::vector<float> mean(nCols, 0.0f);
    for (int row = 0; row < nRows; row++) {
      for (int col = 0; col < nCols; col++) {
        mean[col] += data[row * nCols + col];
      }
    }
    for (int col = 0; col < nCols; col++)
      mean[col] /= (nRows > 0 ? nRows : 1);

    std::vector<float> centered(data.size(), 0.0f);
    for (int row = 0; row < nRows; row++) {
      for (int col = 0; col < nCols; col++) {
        int idx = row * nCols + col;
        centered[idx] = data[idx] - mean[col];
      }
    }

    std::vector<std::vector<float>> cov(nCols, std::vector<float>(nCols, 0.0f));
    for (int row = 0; row < nRows; row++) {
      for (int i = 0; i < nCols; i++) {
        for (int j = i; j < nCols; j++) {
          float v = centered[row * nCols + i] * centered[row * nCols + j];
          cov[i][j] += v;
          if (i != j)
            cov[j][i] += v;
        }
      }
    }

    auto powerIter = [&](const std::vector<float> &vec) {
      std::vector<float> next(nCols, 0.0f);
      for (int i = 0; i < nCols; i++) {
        double acc = 0.0;
        for (int j = 0; j < nCols; j++)
          acc += cov[i][j] * vec[j];
        next[i] = (float)acc;
      }
      double norm = 0.0;
      for (auto &v : next)
        norm += v * v;
      norm = std::sqrt(norm) ? std::sqrt(norm) : 1.0;
      for (auto &v : next)
        v = (float)(v / norm);
      return next;
    };

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v1(nCols, 0.0f);
    for (auto &v : v1)
      v = dist(rng);
    for (int i = 0; i < 8; i++)
      v1 = powerIter(v1);
    std::vector<float> v2(nCols, 0.0f);
    for (auto &v : v2)
      v = dist(rng);
    for (int i = 0; i < 8; i++) {
      double proj = 0.0;
      for (int j = 0; j < nCols; j++)
        proj += v2[j] * v1[j];
      for (int j = 0; j < nCols; j++)
        v2[j] = (float)(v2[j] - proj * v1[j]);
      v2 = powerIter(v2);
    }

    std::vector<float> coords(nRows * 2, 0.0f);
    for (int row = 0; row < nRows; row++) {
      double x = 0.0;
      double y = 0.0;
      for (int col = 0; col < nCols; col++) {
        float v = centered[row * nCols + col];
        x += v * v1[col];
        y += v * v2[col];
      }
      coords[row * 2] = (float)x;
      coords[row * 2 + 1] = (float)y;
    }
    return {coords, v1, v2};
  }

  json project2D(const json &emb, const std::string &method = "auto") {
    if (method == "umap") {
      auto coords2d = umap2D(emb, 15, 200);
      json coords = json::array();
      for (auto &c : coords2d)
        coords.push_back(json::array({c[0], c[1]}));
      return json{{"coords", coords}, {"method", "umap"}};
    }
    auto pca = pca2D(emb);
    int nRows = emb.value("nRows", 0);
    json coords = json::array();
    for (int row = 0; row < nRows; row++) {
      coords.push_back(
          json::array({pca.coords[row * 2], pca.coords[row * 2 + 1]}));
    }
    return json{{"coords", coords}, {"method", "pca"}};
  }

private:
  static std::vector<std::array<float, 2>> umap2D(const json &emb,
                                                  int nNeighbors, int epochs) {
    int nRows = emb.value("nRows", 0);
    int nCols = emb.value("nCols", 0);
    if (nRows <= 0 || nCols <= 0)
      return {};
    std::vector<float> data;
    if (emb.contains("data") && emb["data"].is_array()) {
      data.reserve((size_t)nRows * (size_t)nCols);
      for (auto &v : emb["data"])
        data.push_back(v.get<float>());
    }
    if ((int)data.size() != nRows * nCols)
      return {};

    int k = std::max(2, std::min(nNeighbors, nRows - 1));
    double target = std::log2((double)k);

    std::vector<std::vector<std::pair<int, float>>> knn(nRows);
    for (int i = 0; i < nRows; i++) {
      std::vector<std::pair<float, int>> dists;
      dists.reserve(nRows - 1);
      for (int j = 0; j < nRows; j++) {
        if (i == j)
          continue;
        double acc = 0.0;
        for (int c = 0; c < nCols; c++) {
          double diff = data[i * nCols + c] - data[j * nCols + c];
          acc += diff * diff;
        }
        dists.push_back({(float)std::sqrt(acc), j});
      }
      std::partial_sort(dists.begin(), dists.begin() + k, dists.end(),
                        [](auto &a, auto &b) { return a.first < b.first; });
      for (int t = 0; t < k; t++)
        knn[i].push_back({dists[t].second, dists[t].first});
    }

    std::vector<float> rhos(nRows, 0.0f);
    std::vector<float> sigmas(nRows, 1.0f);
    for (int i = 0; i < nRows; i++) {
      float rho = 0.0f;
      for (auto &p : knn[i]) {
        if (p.second > 0) {
          rho = p.second;
          break;
        }
      }
      rhos[i] = rho;
      double lo = 1e-3, hi = 64.0;
      for (int it = 0; it < 32; it++) {
        double mid = (lo + hi) * 0.5;
        double sum = 0.0;
        for (auto &p : knn[i]) {
          double d = std::max(0.0, (double)p.second - rho);
          sum += std::exp(-d / mid);
        }
        if (sum > target)
          hi = mid;
        else
          lo = mid;
      }
      sigmas[i] = (float)((lo + hi) * 0.5);
      if (sigmas[i] <= 1e-6f)
        sigmas[i] = 1.0f;
    }

    std::vector<std::unordered_map<int, float>> weights(nRows);
    for (int i = 0; i < nRows; i++) {
      for (auto &p : knn[i]) {
        double d = std::max(0.0, (double)p.second - rhos[i]);
        double val = std::exp(-d / sigmas[i]);
        weights[i][p.first] = (float)val;
      }
    }

    struct Edge {
      int i;
      int j;
      float w;
    };
    std::vector<Edge> edges;
    edges.reserve(nRows * k);
    for (int i = 0; i < nRows; i++) {
      for (auto &kv : weights[i]) {
        int j = kv.first;
        if (i >= j)
          continue;
        float w1 = kv.second;
        float w2 = weights[j].count(i) ? weights[j][i] : 0.0f;
        float ws = w1 + w2 - w1 * w2;
        if (ws > 0.0f)
          edges.push_back({i, j, ws});
      }
    }

    auto find_ab = [&](double spread, double minDist) {
      double bestA = 1.0, bestB = 1.0, bestErr = 1e9;
      double x1 = minDist + spread;
      double x2 = minDist + 2.0 * spread;
      double t1 = std::exp(-1.0);
      double t2 = std::exp(-2.0);
      for (double b = 0.5; b <= 2.5; b += 0.01) {
        double a = (1.0 / t1 - 1.0) / std::pow(x1, 2.0 * b);
        if (!std::isfinite(a) || a <= 0)
          continue;
        double f2 = 1.0 / (1.0 + a * std::pow(x2, 2.0 * b));
        double err = std::abs(f2 - t2);
        if (err < bestErr) {
          bestErr = err;
          bestA = a;
          bestB = b;
        }
      }
      return std::pair<double, double>(bestA, bestB);
    };

    double minDist = 0.1;
    double spread = 1.0;
    auto ab = find_ab(spread, minDist);
    double a = ab.first;
    double b = ab.second;
    double gamma = 1.0;

    DimReducer reducer;
    auto pca = reducer.pca2D(emb);
    std::vector<std::array<float, 2>> coords(nRows, {0.0f, 0.0f});
    if ((int)pca.coords.size() >= nRows * 2) {
      for (int i = 0; i < nRows; i++) {
        coords[i][0] = pca.coords[i * 2];
        coords[i][1] = pca.coords[i * 2 + 1];
      }
    } else {
      std::mt19937 rng(0xBADC0DEu);
      std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
      for (int i = 0; i < nRows; i++) {
        coords[i][0] = dist(rng);
        coords[i][1] = dist(rng);
      }
    }

    std::mt19937 rng(0x12345678u);
    std::uniform_int_distribution<int> pick(0, nRows - 1);
    int negSamples = 5;
    for (int epoch = 0; epoch < epochs; epoch++) {
      double lr = 1.0 * (1.0 - (double)epoch / std::max(1, epochs));
      for (const auto &e : edges) {
        auto &pa = coords[e.i];
        auto &pb = coords[e.j];
        double dx = pa[0] - pb[0];
        double dy = pa[1] - pb[1];
        double dist2 = dx * dx + dy * dy + 1e-6;
        double gradCoeff = -2.0 * a * b * std::pow(dist2, b - 1.0) /
                           (1.0 + a * std::pow(dist2, b));
        double step = lr * e.w * gradCoeff;
        pa[0] += (float)(step * dx);
        pa[1] += (float)(step * dy);
        pb[0] -= (float)(step * dx);
        pb[1] -= (float)(step * dy);

        for (int t = 0; t < negSamples; t++) {
          int r = pick(rng);
          if (r == e.i)
            continue;
          auto &pn = coords[r];
          double ndx = pa[0] - pn[0];
          double ndy = pa[1] - pn[1];
          double nd2 = ndx * ndx + ndy * ndy + 1e-6;
          double repCoeff =
              2.0 * gamma * b / ((0.001 + nd2) * (1.0 + a * std::pow(nd2, b)));
          double rep = lr * repCoeff;
          pa[0] += (float)(rep * ndx);
          pa[1] += (float)(rep * ndy);
          pn[0] -= (float)(rep * ndx);
          pn[1] -= (float)(rep * ndy);
        }
      }
    }
    return coords;
  }
};
