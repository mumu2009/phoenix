// x5_online_learner.cpp
//
// A tiny C++ MLP with SGD + momentum, built on Eigen3.  It is intended to run
// on the RDK X5 and train a small task head (speech command, vision label,
// etc.) on top of features produced by an ONNX encoder.  This is the "C++
// version of NumPy" requested for the online learning path.
//
// Build:
//   g++ -std=c++17 -O3 -I/usr/include/eigen3 -o x5_online_learner x5_online_learner.cpp
//
// Train:
//   ./x5_online_learner train \
//       --data features.bin --n 10000 --in 128 --out 10 \
//       --hidden 64 32 --lr 0.001 --epochs 20 --save head.bin
//
// Predict:
//   ./x5_online_learner predict \
//       --data features.bin --n 1000 --in 128 --out 10 \
//       --hidden 64 32 --load head.bin
//
// Data format (binary, little-endian, float32):
//   [N x in_dim] feature matrix, then [N x 1] int32 labels (classification)
//   or [N x out_dim] float32 targets (regression, use --regression).

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Layer {
    MatrixXf W;
    VectorXf b;
    MatrixXf dW;
    VectorXf db;
    MatrixXf mW;  // momentum
    VectorXf mb;
    MatrixXf a;   // pre-activation
    MatrixXf h;   // activation
};

struct MLP {
    std::vector<int> arch;  // input, hidden..., output
    std::vector<Layer> layers;
    bool regression = false;
    float lr = 0.001f;
    float momentum = 0.9f;

    void init(const std::vector<int>& a_arch, bool reg, float learning_rate, float mom) {
        arch = a_arch;
        regression = reg;
        lr = learning_rate;
        momentum = mom;
        layers.clear();
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 1; i < arch.size(); ++i) {
            Layer L;
            int in_d = arch[i - 1];
            int out_d = arch[i];
            L.W = MatrixXf(out_d, in_d);
            L.b = VectorXf::Zero(out_d);
            for (int r = 0; r < out_d; ++r) {
                for (int c = 0; c < in_d; ++c) {
                    L.W(r, c) = dist(rng) * std::sqrt(2.0f / in_d);
                }
            }
            L.dW = MatrixXf::Zero(out_d, in_d);
            L.db = VectorXf::Zero(out_d);
            L.mW = MatrixXf::Zero(out_d, in_d);
            L.mb = VectorXf::Zero(out_d);
            layers.push_back(L);
        }
    }

    static MatrixXf relu(const MatrixXf& x) { return x.cwiseMax(0.0f); }
    static MatrixXf relu_grad(const MatrixXf& x) { return (x.array() > 0.0f).cast<float>(); }

    MatrixXf forward(const MatrixXf& x, bool save = false) {
        MatrixXf cur = x;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].a = layers[i].W * cur.transpose();
            layers[i].a.colwise() += layers[i].b;
            if (i + 1 == layers.size()) {
                // output layer
                if (!regression) {
                    // stable softmax per column
                    MatrixXf out = layers[i].a;
                    for (int c = 0; c < out.cols(); ++c) {
                        float mx = out.col(c).maxCoeff();
                        out.col(c) = (out.col(c).array() - mx).exp();
                        float s = out.col(c).sum();
                        out.col(c) /= s;
                    }
                    if (save) layers[i].h = out;
                    return out;
                } else {
                    if (save) layers[i].h = layers[i].a;
                    return layers[i].a;
                }
            } else {
                layers[i].h = relu(layers[i].a);
                if (!save) {
                    // keep only pre-activation for backprop memory
                }
                cur = layers[i].h.transpose();
            }
        }
        return cur;
    }

    float train_step(const MatrixXf& x, const MatrixXf& y, int task) {
        // task: 0 = classification with integer labels (y is Nx1, values 0..C-1)
        //       1 = classification with one-hot targets (y is NxC)
        //       2 = regression (y is NxC)
        MatrixXf pred = forward(x, true);  // C x N
        int N = x.rows();

        // loss and output gradient
        float loss = 0.0f;
        MatrixXf grad = pred;  // C x N
        if (task == 0) {
            for (int n = 0; n < N; ++n) {
                int label = static_cast<int>(y(n, 0) + 0.5f);
                if (label < 0 || label >= pred.rows()) {
                    std::cerr << "BAD label at n=" << n << " label=" << label << "\n";
                }
                float p = pred(label, n);
                if (p <= 0.0f) p = 1e-8f;
                loss += -std::log(p);
                grad(label, n) -= 1.0f;
            }
            loss /= N;
        } else if (task == 1) {
            // y is N x C
            MatrixXf yt = y.transpose();  // C x N
            grad = pred - yt;
            loss = (0.5f / N) * grad.cwiseProduct(grad).sum();
        } else {
            MatrixXf yt = y.transpose();
            grad = pred - yt;
            loss = (0.5f / N) * grad.cwiseProduct(grad).sum();
        }
        grad /= N;

        // backprop
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            MatrixXf a_grad = grad;
            if (i + 1 != static_cast<int>(layers.size())) {
                a_grad = a_grad.cwiseProduct(relu_grad(layers[i].a));
            }
            // a_grad is out_d x N; h_prev is the layer input: N x in_d
            MatrixXf h_prev = (i == 0) ? x : MatrixXf(layers[i - 1].h.transpose());
            layers[i].dW = a_grad * h_prev;
            layers[i].db = a_grad.rowwise().sum();

            // gradient for previous layer
            if (i > 0) {
                grad = layers[i].W.transpose() * a_grad;
            }
        }

        // update with momentum
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].mW = momentum * layers[i].mW + lr * layers[i].dW;
            layers[i].mb = momentum * layers[i].mb + lr * layers[i].db;
            layers[i].W -= layers[i].mW;
            layers[i].b -= layers[i].mb;
        }

        return loss;
    }

    int predict_class(const VectorXf& x) {
        MatrixXf out = forward(x.transpose(), false);
        int C, N;
        C = out.rows();
        N = out.cols();
        int best = 0;
        float best_p = out(0, 0);
        for (int c = 1; c < C; ++c) {
            if (out(c, 0) > best_p) {
                best_p = out(c, 0);
                best = c;
            }
        }
        return best;
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + path);
        int32_t nl = static_cast<int32_t>(arch.size());
        f.write(reinterpret_cast<const char*>(&nl), sizeof(nl));
        for (int v : arch) {
            int32_t iv = v;
            f.write(reinterpret_cast<const char*>(&iv), sizeof(iv));
        }
        int32_t reg = regression ? 1 : 0;
        f.write(reinterpret_cast<const char*>(&reg), sizeof(reg));
        for (const auto& L : layers) {
            int r = L.W.rows(), c = L.W.cols();
            f.write(reinterpret_cast<const char*>(&r), sizeof(r));
            f.write(reinterpret_cast<const char*>(&c), sizeof(c));
            f.write(reinterpret_cast<const char*>(L.W.data()), r * c * sizeof(float));
            f.write(reinterpret_cast<const char*>(L.b.data()), r * sizeof(float));
        }
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot read " + path);
        int32_t nl;
        f.read(reinterpret_cast<char*>(&nl), sizeof(nl));
        arch.resize(nl);
        for (int i = 0; i < nl; ++i) {
            int32_t iv;
            f.read(reinterpret_cast<char*>(&iv), sizeof(iv));
            arch[i] = iv;
        }
        int32_t reg;
        f.read(reinterpret_cast<char*>(&reg), sizeof(reg));
        regression = reg != 0;
        layers.resize(arch.size() - 1);
        for (size_t i = 1; i < arch.size(); ++i) {
            int r, c;
            f.read(reinterpret_cast<char*>(&r), sizeof(r));
            f.read(reinterpret_cast<char*>(&c), sizeof(c));
            Layer L;
            L.W = MatrixXf(r, c);
            L.b = VectorXf::Zero(r);
            f.read(reinterpret_cast<char*>(L.W.data()), r * c * sizeof(float));
            f.read(reinterpret_cast<char*>(L.b.data()), r * sizeof(float));
            L.dW = MatrixXf::Zero(r, c);
            L.db = VectorXf::Zero(r);
            L.mW = MatrixXf::Zero(r, c);
            L.mb = VectorXf::Zero(r);
            layers[i - 1] = L;
        }
    }
};

static bool arg_bool(int& i, int argc, char** argv, const char* name, bool& out) {
    if (std::strcmp(argv[i], name) == 0) {
        out = true;
        return true;
    }
    return false;
}

static bool arg_str(int& i, int argc, char** argv, const char* name, std::string& out) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        out = argv[++i];
        return true;
    }
    return false;
}

static bool arg_float(int& i, int argc, char** argv, const char* name, float& out) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        out = std::stof(argv[++i]);
        return true;
    }
    return false;
}

static bool arg_int(int& i, int argc, char** argv, const char* name, int& out) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        out = std::stoi(argv[++i]);
        return true;
    }
    return false;
}

static bool arg_ints(int& i, int argc, char** argv, const char* name, std::vector<int>& out) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        ++i;
        out.clear();
        while (i < argc && argv[i][0] != '-') {
            out.push_back(std::stoi(argv[i]));
            ++i;
        }
        --i;
        return true;
    }
    return false;
}

static std::pair<MatrixXf, MatrixXf> load_data(const std::string& path, int N, int in_dim, int out_dim, bool regression, bool labels_are_int) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open data " + path);
    // File is interleaved: each sample = [in_dim floats] [label(s)]
    RowMatrixXf rx(N, in_dim);
    RowMatrixXf ry(N, labels_are_int ? 1 : out_dim);
    std::vector<int32_t> tmp(1);
    for (int i = 0; i < N; ++i) {
        f.read(reinterpret_cast<char*>(rx.row(i).data()), in_dim * sizeof(float));
        if (labels_are_int) {
            f.read(reinterpret_cast<char*>(tmp.data()), sizeof(int32_t));
            ry(i, 0) = static_cast<float>(tmp[0]);
        } else {
            f.read(reinterpret_cast<char*>(ry.row(i).data()), out_dim * sizeof(float));
        }
    }
    if (!f) throw std::runtime_error("failed to read data from " + path);
    MatrixXf x = rx;
    MatrixXf y = ry;
    return {x, y};
}

static void save_data(const std::string& path, const MatrixXf& x, const MatrixXf& y, bool labels_are_int) {
    std::ofstream f(path, std::ios::binary);
    RowMatrixXf rx = x;
    RowMatrixXf ry = y;
    int N = x.rows();
    int in_dim = x.cols();
    int out_dim = y.cols();
    for (int i = 0; i < N; ++i) {
        f.write(reinterpret_cast<const char*>(rx.row(i).data()), in_dim * sizeof(float));
        if (labels_are_int) {
            int32_t label = static_cast<int32_t>(ry(i, 0) + 0.5f);
            f.write(reinterpret_cast<const char*>(&label), sizeof(int32_t));
        } else {
            f.write(reinterpret_cast<const char*>(ry.row(i).data()), out_dim * sizeof(float));
        }
    }
}

static int do_train(int argc, char** argv) {
    std::string data, save_path;
    int N = 0, in_dim = 0, out_dim = 0, epochs = 10, batch_size = 32, seed = 42;
    std::vector<int> hidden;
    float lr = 0.001f, momentum = 0.9f;
    bool regression = false;

    for (int i = 2; i < argc; ++i) {
        if (arg_str(i, argc, argv, "--data", data)) continue;
        if (arg_str(i, argc, argv, "--save", save_path)) continue;
        if (arg_int(i, argc, argv, "--n", N)) continue;
        if (arg_int(i, argc, argv, "--in", in_dim)) continue;
        if (arg_int(i, argc, argv, "--out", out_dim)) continue;
        if (arg_ints(i, argc, argv, "--hidden", hidden)) continue;
        if (arg_int(i, argc, argv, "--epochs", epochs)) continue;
        if (arg_int(i, argc, argv, "--batch-size", batch_size)) continue;
        if (arg_int(i, argc, argv, "--seed", seed)) continue;
        if (arg_float(i, argc, argv, "--lr", lr)) continue;
        if (arg_float(i, argc, argv, "--momentum", momentum)) continue;
        if (arg_bool(i, argc, argv, "--regression", regression)) continue;
    }

    if (data.empty() || save_path.empty() || N == 0 || in_dim == 0 || out_dim == 0) {
        std::cerr << "train: --data --save --n --in --out required\n";
        return 1;
    }

    auto [x, y] = load_data(data, N, in_dim, out_dim, regression, !regression);

    std::vector<int> arch = {in_dim};
    arch.insert(arch.end(), hidden.begin(), hidden.end());
    arch.push_back(out_dim);

    MLP mlp;
    mlp.init(arch, regression, lr, momentum);

    std::mt19937 rng(seed);
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    int task = regression ? 2 : 0;  // integer labels by default
    if (!regression && y.cols() == out_dim && out_dim > 1) {
        // treat y as one-hot float targets if columns match and not exactly class index
        task = 1;
    }

    for (int e = 0; e < epochs; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        float total_loss = 0.0f;
        int batches = 0;
        for (int b = 0; b < N; b += batch_size) {
            int end = std::min(b + batch_size, N);
            int bs = end - b;
            MatrixXf bx(bs, in_dim);
            MatrixXf by(bs, y.cols());
            for (int j = 0; j < bs; ++j) {
                bx.row(j) = x.row(idx[b + j]);
                by.row(j) = y.row(idx[b + j]);
            }
            total_loss += mlp.train_step(bx, by, task);
            ++batches;
        }
        std::cout << "epoch " << (e + 1) << "/" << epochs
                  << " loss=" << (total_loss / batches) << "\n";
    }

    mlp.save(save_path);
    std::cout << "saved " << save_path << "\n";
    return 0;
}

static int do_predict(int argc, char** argv) {
    std::string data, load_path;
    int N = 0, in_dim = 0, out_dim = 0;
    std::vector<int> hidden;
    bool regression = false;

    for (int i = 2; i < argc; ++i) {
        if (arg_str(i, argc, argv, "--data", data)) continue;
        if (arg_str(i, argc, argv, "--load", load_path)) continue;
        if (arg_int(i, argc, argv, "--n", N)) continue;
        if (arg_int(i, argc, argv, "--in", in_dim)) continue;
        if (arg_int(i, argc, argv, "--out", out_dim)) continue;
        if (arg_ints(i, argc, argv, "--hidden", hidden)) continue;
        if (arg_bool(i, argc, argv, "--regression", regression)) continue;
    }

    if (data.empty() || load_path.empty() || N == 0 || in_dim == 0) {
        std::cerr << "predict: --data --load --n --in required\n";
        return 1;
    }

    MLP mlp;
    mlp.load(load_path);

    auto [x, y] = load_data(data, N, in_dim, out_dim, regression, !regression);
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        int pred = mlp.predict_class(x.row(i));
        if (regression) {
            VectorXf out = mlp.forward(x.row(i).transpose(), false).col(0);
            std::cout << "sample " << i << " pred=";
            for (int c = 0; c < out.size(); ++c) std::cout << out[c] << " ";
            std::cout << "\n";
        } else {
            int label = static_cast<int>(y(i, 0) + 0.5f);
            std::cout << "sample " << i << " pred=" << pred << " true=" << label << "\n";
            if (pred == label) ++correct;
        }
    }
    if (!regression) {
        std::cout << "accuracy " << (100.0f * correct / N) << "%\n";
    }
    return 0;
}

static int do_synth(int argc, char** argv) {
    std::string path;
    int N = 1000, in_dim = 64, out_dim = 10;
    for (int i = 2; i < argc; ++i) {
        if (arg_str(i, argc, argv, "--out", path)) continue;
        if (arg_int(i, argc, argv, "--n", N)) continue;
        if (arg_int(i, argc, argv, "--in", in_dim)) continue;
        if (arg_int(i, argc, argv, "--out-dim", out_dim)) continue;
    }
    if (path.empty()) { std::cerr << "synth: --out required\n"; return 1; }
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    MatrixXf x(N, in_dim);
    for (int i = 0; i < N; ++i) for (int j = 0; j < in_dim; ++j) x(i, j) = dist(rng);
    // label = argmax of first few features
    MatrixXf y(N, 1);
    for (int i = 0; i < N; ++i) {
        int label = 0;
        float best = x(i, 0);
        for (int c = 1; c < out_dim; ++c) {
            if (x(i, c) > best) { best = x(i, c); label = c; }
        }
        y(i, 0) = static_cast<float>(label);
    }
    save_data(path, x, y, true);
    std::cout << "wrote " << N << " synth samples to " << path << "\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " {train|predict|synth} [args...]\n";
        return 1;
    }
    std::string cmd = argv[1];
    try {
        if (cmd == "train") return do_train(argc, argv);
        if (cmd == "predict") return do_predict(argc, argv);
        if (cmd == "synth") return do_synth(argc, argv);
        std::cerr << "unknown command " << cmd << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
