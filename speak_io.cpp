/* speak_io.cpp - Speech I/O implementation
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

#include "speak_io.hpp"
#include "loggerCXXH.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <random>

#include <Eigen/Dense>

namespace
{
    constexpr float kPi = 3.14159265358979323846f;

    struct AudioLearner;
    static AudioLearner &audioLearner();
    static Eigen::VectorXf spectralStats(const std::vector<float> &x, int sr, int win, int hop);
    static Eigen::VectorXf prosodyStats(const std::vector<float> &x, int sr);
    static float estimatePitchLocal(const std::vector<float> &x, int sampleRate);
    static float computeRmsLocal(const std::vector<float> &x);

    // SpeakReservedArena 用于预留一段连续内存，减少语音处理阶段的突发分配抖动。
    // 调用方式：通过 speakArena() 获取单例后，仅在初始化阶段调用 init()。
    // 实现思路：根据环境变量给出的 MB 大小做上下限裁剪并触页预热。
    // 注意事项：该对象只负责预留，不承诺业务数据都在此内存内分配。
    // 注意事项：init 具有幂等性，重复调用不会重复扩容。
    // 注意事项：异常时会清空预留块，避免半初始化状态。
    class SpeakReservedArena
    {
    public:
        // 初始化预留内存池。
        // 调用方式：传入字节数，通常由 ensureSpeakArena() 统一触发。
        // 实现思路：加锁后按最小/最大阈值修正并进行页面触达。
        // 注意事项：该函数不是实时路径，建议在系统启动时完成。
        void init(size_t bytes)
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (inited_)
                return;
            bytes = std::max<size_t>(8ull * 1024ull * 1024ull, bytes);
            bytes = std::min<size_t>(512ull * 1024ull * 1024ull, bytes);
            try
            {
                block_.resize(bytes);
                const size_t page = 4096;
                for (size_t i = 0; i < block_.size(); i += page)
                    block_[i] = 0;
            }
            catch (...)
            {
                block_.clear();
            }
            inited_ = true;
        }

    private:
        std::vector<uint8_t> block_;
        std::mutex mu_;
        bool inited_{false};
    };

    // 返回语音模块预留内存单例。
    // 调用方式：任何需要预留资源的路径都可调用此函数。
    // 实现思路：使用函数内静态对象保证线程安全懒加载。
    // 注意事项：返回引用长期有效，不应在外部持有其内部资源地址。
    static SpeakReservedArena &speakArena()
    {
        static SpeakReservedArena arena;
        return arena;
    }

    // 确保语音预留内存仅初始化一次。
    // 调用方式：在音频分析或合成入口最先调用。
    // 实现思路：使用 std::call_once 读取环境变量并触发 init。
    // 注意事项：无效配置会回退默认值，避免异常配置造成崩溃。
    static void ensureSpeakArena()
    {
        static std::once_flag once;
        std::call_once(once, []()
                       {
            const char *raw = std::getenv("AI_SPEAK_RESERVED_MB");
            double mb = 64.0;
            if (raw && *raw)
            {
                try
                {
                    mb = std::stod(raw);
                }
                catch (...)
                {
                    mb = 64.0;
                }
            }
            if (!std::isfinite(mb) || mb < 8.0)
                mb = 64.0;
            speakArena().init((size_t)(mb * 1024.0 * 1024.0)); });
    }

    // 获取语音模块统一日志器。
    // 调用方式：在计算、错误、调试路径中直接调用。
    // 实现思路：返回 LoggerCXX 全局单例引用。
    // 注意事项：日志输出量由外部配置控制。
    static LoggerCXX &speakLogger()
    {
        return LoggerCXX::instance();
    }
}

#ifdef HAVE_VOSK
#include <vosk_api.h>
#endif

// SpeakIO 构造函数，负责完成语音子系统的轻量级初始化。
// 调用方式：由上层服务创建 SpeakIO 实例时自动调用。
// 实现思路：仅确保预留内存可用，不做重计算。
// 注意事项：构造函数应保持快速，避免阻塞请求线程。
SpeakIO::SpeakIO()
{
    ensureSpeakArena();
}

// 从 WAV 二进制字节执行完整语音分析流程。
// 调用方式：传入完整 WAV 文件字节数组，返回 Json 结果。
// 实现思路：先解析 WAV，再复用 analyzePcm 统一处理。
// 注意事项：输入格式错误会返回 ok=false 与 error 字段。
Json::Value SpeakIO::analyzeWavBytes(const std::vector<uint8_t> &bytes)
{
    ensureSpeakArena();
    if (speakLogger().enabled())
        speakLogger().log(LoggerCXX::Type::COMPUTE, std::string("speak analyzeWavBytes bytes=") + std::to_string(bytes.size()));
    Json::Value out;
    AudioData audio;
    std::string err;
    if (!parseWav(bytes, audio, err))
    {
        if (speakLogger().enabled())
            speakLogger().log(LoggerCXX::Type::ERROR, std::string("speak parseWav failed: ") + err);
        out["ok"] = false;
        out["error"] = err;
        return out;
    }
    return analyzePcm(audio);
}

// 对标准化 PCM 音频执行多任务分析。
// 调用方式：传入 AudioData（单声道 float 样本）并返回结构化结果。
// 实现思路：组合 ASR、分离、环境识别、音色情绪分析并汇总。
// 注意事项：空音频会直接返回错误，不进入后续流程。
Json::Value SpeakIO::analyzePcm(const AudioData &audio)
{
    if (speakLogger().enabled())
        speakLogger().log(LoggerCXX::Type::COMPUTE, std::string("speak analyzePcm samples=") + std::to_string(audio.mono.size()) + " sr=" + std::to_string(audio.sampleRate));
    Json::Value out;
    if (audio.mono.empty())
    {
        out["ok"] = false;
        out["error"] = "empty audio";
        return out;
    }

    Json::Value asrMeta;
    std::string text = recognizeSpeech(audio, asrMeta);
    Json::Value separation = separateTracks(audio);
    Json::Value env = analyzeEnvironment(audio);
    Json::Value tone = analyzeTone(audio);

    std::ostringstream stage05;
    stage05 << "speech_stage0.5";
    if (env.isMember("env"))
        stage05 << "|env=" << env["env"].asString();
    if (tone.isMember("emotion"))
        stage05 << "|emotion=" << tone["emotion"].asString();
    if (tone.isMember("pitch"))
        stage05 << "|pitch=" << std::fixed << std::setprecision(1) << tone["pitch"].asDouble();

    out["ok"] = true;
    out["text"] = text;
    out["asr"] = asrMeta;
    out["separation"] = separation;
    out["environment"] = env;
    out["tone"] = tone;
    // 生成可学习语料（观察式描述）
    Json::Value corpus(Json::arrayValue);
    if (env.isMember("env"))
        corpus.append("env=" + env["env"].asString());
    if (tone.isMember("emotion"))
        corpus.append("emotion=" + tone["emotion"].asString());
    if (separation.isMember("relations"))
    {
        const auto &rel = separation["relations"];
        if (rel.isMember("energy_share"))
        {
            for (const auto &name : rel["energy_share"].getMemberNames())
            {
                std::ostringstream ss;
                ss << "energy_share:" << name << "=" << std::fixed << std::setprecision(3)
                   << rel["energy_share"][name].asDouble();
                corpus.append(ss.str());
            }
        }
        if (rel.isMember("activation_correlation"))
        {
            for (const auto &name : rel["activation_correlation"].getMemberNames())
            {
                std::ostringstream ss;
                ss << "track_corr:" << name << "=" << std::fixed << std::setprecision(3)
                   << rel["activation_correlation"][name].asDouble();
                corpus.append(ss.str());
            }
        }
    }
    out["learnableCorpus"] = corpus;
    out["stage05"] = stage05.str();
    return out;
}

// 分析环境声学特征并输出环境聚类标签。
// 调用方式：由 analyzePcm 内部调用，也可独立调用。
// 实现思路：计算 RMS/ZCR/谱质心等特征后进入在线聚类器。
// 注意事项：窗口参数会依据采样率自适应，不需外部传参。
Json::Value SpeakIO::analyzeEnvironment(const AudioData &audio)
{
    if (speakLogger().enabled())
        speakLogger().log(LoggerCXX::Type::COMPUTE, "speak analyzeEnvironment");
    Json::Value out;
    float rms = computeRms(audio.mono);
    float zcr = computeZcr(audio.mono);
    float centroid = computeSpectralCentroid(audio.mono, audio.sampleRate);
    Eigen::VectorXf spec = spectralStats(audio.mono, audio.sampleRate,
                                         std::max(256, std::min(1024, audio.sampleRate / 16)),
                                         std::max(64, std::min(256, audio.sampleRate / 64)));
    out["rms"] = rms;
    out["zcr"] = zcr;
    out["centroid"] = centroid;
    Json::Value specArr(Json::arrayValue);
    for (int i = 0; i < spec.size(); ++i)
        specArr.append(spec(i));
    out["spectralFeatures"] = specArr;
    Eigen::VectorXf feat(9);
    feat << rms, zcr, centroid, spec(0), spec(1), spec(2), spec(3), spec(4), spec(5);
    out["env"] = classifyEnvironment(rms, zcr, centroid);
    return out;
}

// 分析语调和情绪相关特征。
// 调用方式：由 analyzePcm 内部调用，可单独用于情绪标签推断。
// 实现思路：估计基频并提取韵律统计特征后聚类分类。
// 注意事项：极短语音可能导致基频估计为 0。
Json::Value SpeakIO::analyzeTone(const AudioData &audio)
{
    if (speakLogger().enabled())
        speakLogger().log(LoggerCXX::Type::COMPUTE, "speak analyzeTone");
    Json::Value out;
    float pitch = estimatePitch(audio.mono, audio.sampleRate);
    float rms = computeRms(audio.mono);
    Eigen::VectorXf prosody = prosodyStats(audio.mono, audio.sampleRate);
    out["pitch"] = pitch;
    Json::Value prosodyArr(Json::arrayValue);
    for (int i = 0; i < prosody.size(); ++i)
        prosodyArr.append(prosody(i));
    out["prosodyFeatures"] = prosodyArr;
    Eigen::VectorXf feat(8);
    feat << rms, pitch, prosody(0), prosody(1), prosody(2), prosody(3), prosody(4), prosody(5);
    out["emotion"] = classifyEmotion(rms, pitch);
    return out;
}

// 调用 ASR 引擎执行语音识别。
// 调用方式：传入音频样本并通过 asrMeta 返回原始识别信息。
// 实现思路：在 HAVE_VOSK 下分块喂入 PCM 并解析 JSON 结果。
// 注意事项：未启用 vosk 时会返回空串并在元信息中给出错误。
std::string SpeakIO::recognizeSpeech(const AudioData &audio, Json::Value &asrMeta)
{
    if (speakLogger().enabled())
        speakLogger().log(LoggerCXX::Type::COMPUTE, std::string("speak recognizeSpeech samples=") + std::to_string(audio.mono.size()));
#ifdef HAVE_VOSK
    const char *modelPath = std::getenv("VOSK_MODEL");
    if (!modelPath || !std::strlen(modelPath))
    {
        asrMeta["ok"] = false;
        asrMeta["error"] = "VOSK_MODEL not set";
        return "";
    }
    vosk_set_log_level(0);
    VoskModel *model = vosk_model_new(modelPath);
    if (!model)
    {
        asrMeta["ok"] = false;
        asrMeta["error"] = "vosk model load failed";
        return "";
    }
    VoskRecognizer *rec = vosk_recognizer_new(model, (float)audio.sampleRate);
    if (!rec)
    {
        vosk_model_free(model);
        asrMeta["ok"] = false;
        asrMeta["error"] = "vosk recognizer create failed";
        return "";
    }
    thread_local std::vector<int16_t> pcm;
    pcm.resize(audio.mono.size());
    for (size_t i = 0; i < audio.mono.size(); i++)
    {
        float v = std::max(-1.0f, std::min(1.0f, audio.mono[i]));
        pcm[i] = (int16_t)std::lround(v * 32767.0f);
    }
    const int chunk = 4000;
    for (size_t i = 0; i < pcm.size(); i += chunk)
    {
        size_t len = std::min<size_t>(chunk, pcm.size() - i);
        vosk_recognizer_accept_waveform(rec, (const char *)(pcm.data() + i), (int)(len * sizeof(int16_t)));
    }
    const char *result = vosk_recognizer_final_result(rec);
    std::string jsonStr = result ? result : "{}";
    asrMeta["ok"] = true;
    asrMeta["raw"] = jsonStr;
    std::string text;
    try
    {
        Json::CharReaderBuilder builder;
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        Json::Value root;
        std::string errs;
        if (reader->parse(jsonStr.data(), jsonStr.data() + jsonStr.size(), &root, &errs))
        {
            text = root.get("text", "").asString();
        }
    }
    catch (...)
    {
        text = "";
    }
    vosk_recognizer_free(rec);
    vosk_model_free(model);
    return text;
#else
    asrMeta["ok"] = false;
    asrMeta["error"] = "vosk not enabled";
    return "";
#endif
}

// 解析 WAV 文件头和 PCM 数据到 AudioData。
// 调用方式：输入原始字节，成功时填充 out 并返回 true。
// 实现思路：扫描 RIFF chunk，提取 fmt/data 并转为单声道 float。
// 注意事项：当前仅支持 PCM16 格式，其他编码会返回失败。
bool SpeakIO::parseWav(const std::vector<uint8_t> &bytes, AudioData &out, std::string &err)
{
    if (speakLogger().enabled())
        speakLogger().log(LoggerCXX::Type::DEBUG, std::string("speak parseWav bytes=") + std::to_string(bytes.size()));
    if (bytes.size() < 44)
    {
        err = "invalid wav";
        return false;
    }
    auto readU32 = [&](size_t off)
    {
        return (uint32_t)bytes[off] | ((uint32_t)bytes[off + 1] << 8) | ((uint32_t)bytes[off + 2] << 16) | ((uint32_t)bytes[off + 3] << 24);
    };
    auto readU16 = [&](size_t off)
    {
        return (uint16_t)bytes[off] | ((uint16_t)bytes[off + 1] << 8);
    };
    if (std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
    {
        err = "not wav";
        return false;
    }
    size_t pos = 12;
    uint16_t audioFormat = 1;
    uint16_t channels = 1;
    uint32_t sampleRate = 16000;
    uint16_t bitsPerSample = 16;
    size_t dataOffset = 0;
    uint32_t dataSize = 0;
    while (pos + 8 <= bytes.size())
    {
        uint32_t chunkId = readU32(pos);
        uint32_t chunkSize = readU32(pos + 4);
        pos += 8;
        if (pos + chunkSize > bytes.size())
            break;
        if (chunkId == 0x20746d66)
        { // 'fmt '
            audioFormat = readU16(pos);
            channels = readU16(pos + 2);
            sampleRate = readU32(pos + 4);
            bitsPerSample = readU16(pos + 14);
        }
        else if (chunkId == 0x61746164)
        { // 'data'
            dataOffset = pos;
            dataSize = chunkSize;
            break;
        }
        pos += chunkSize + (chunkSize % 2);
    }
    if (!dataOffset || dataSize == 0)
    {
        err = "wav data missing";
        return false;
    }
    if (audioFormat != 1)
    {
        err = "unsupported wav format";
        return false;
    }
    if (bitsPerSample != 16)
    {
        err = "unsupported bits per sample";
        return false;
    }
    size_t samples = dataSize / 2;
    out.sampleRate = (int)sampleRate;
    out.channels = (int)channels;
    out.mono.resize(samples / std::max<uint16_t>(1, channels));
    const int16_t *pcm = reinterpret_cast<const int16_t *>(bytes.data() + dataOffset);
    if (channels == 1)
    {
        for (size_t i = 0; i < out.mono.size(); i++)
        {
            out.mono[i] = pcm[i] / 32768.0f;
        }
    }
    else
    {
        size_t frames = samples / channels;
        out.mono.resize(frames);
        for (size_t f = 0; f < frames; f++)
        {
            float acc = 0.0f;
            for (int c = 0; c < channels; c++)
            {
                acc += pcm[f * channels + c] / 32768.0f;
            }
            out.mono[f] = acc / channels;
        }
    }
    return true;
}

// 计算时域均方根能量。
// 调用方式：传入样本向量，返回 RMS 标量。
// 实现思路：使用 Eigen 数组平方均值后开方。
// 注意事项：空输入返回 0，调用方需自行判定是否有效。
float SpeakIO::computeRms(const std::vector<float> &x)
{
    if (x.empty())
        return 0.0f;
    Eigen::Map<const Eigen::ArrayXf> arr(x.data(), static_cast<Eigen::Index>(x.size()));
    return std::sqrt(arr.square().mean());
}

// 计算过零率（ZCR）。
// 调用方式：传入样本向量，返回每样本平均符号变化率。
// 实现思路：比较相邻符号是否变化并求均值。
// 注意事项：样本少于 2 时返回 0。
float SpeakIO::computeZcr(const std::vector<float> &x)
{
    if (x.size() < 2)
        return 0.0f;
    Eigen::Map<const Eigen::ArrayXf> arr(x.data(), static_cast<Eigen::Index>(x.size()));
    // 过零点计数：符号变化
    Eigen::ArrayXf sign = arr.sign();
    Eigen::ArrayXf sign_prev = sign.head(sign.size() - 1);
    Eigen::ArrayXf sign_next = sign.tail(sign.size() - 1);
    Eigen::ArrayXf zc = (sign_prev != sign_next).cast<float>();
    float zcr = zc.sum() / static_cast<float>(x.size());
    return zcr;
}

// 估计谱质心，近似描述频谱重心位置。
// 调用方式：输入波形和采样率，输出质心频率。
// 实现思路：窗函数加权后计算简化 DFT 幅值并求加权平均。
// 注意事项：该实现是 O(N^2) 近似，适合中小窗口离线分析。
float SpeakIO::computeSpectralCentroid(const std::vector<float> &x, int sampleRate)
{
    if (x.empty())
        return 0.0f;
    size_t N = std::min<size_t>(2048, x.size());
    Eigen::ArrayXf window(N);
    for (size_t i = 0; i < N; i++)
        window(i) = 0.5f - 0.5f * std::cos(2.0f * kPi * i / (N - 1));
    Eigen::Map<const Eigen::ArrayXf> arr(x.data(), static_cast<Eigen::Index>(N));
    Eigen::ArrayXf xw = arr.head(N) * window;
    // 计算DFT（仅幅值谱）
    Eigen::ArrayXf mag = Eigen::ArrayXf::Zero(N / 2);
    for (size_t k = 0; k < N / 2; ++k)
    {
        float re = 0.0f, im = 0.0f;
        for (size_t n = 0; n < N; ++n)
        {
            float angle = 2.0f * kPi * k * n / N;
            re += xw(n) * std::cos(angle);
            im -= xw(n) * std::sin(angle);
        }
        mag(k) = std::sqrt(re * re + im * im);
    }
    Eigen::ArrayXf freqs = Eigen::ArrayXf::LinSpaced(N / 2, 0, float(sampleRate) * (N / 2 - 1) / N);
    float num = (freqs * mag).sum();
    float den = mag.sum();
    return den > 0.0f ? num / den : 0.0f;
}

// 估计基频（Pitch）。
// 调用方式：传入样本和采样率，返回估计频率 Hz。
// 实现思路：在目标延迟范围内做自相关搜索最优峰值。
// 注意事项：输入过短或无明显周期时返回 0。
float SpeakIO::estimatePitch(const std::vector<float> &x, int sampleRate)
{
    if (x.size() < 512)
        return 0.0f;
    int minLag = sampleRate / 400; // 400 Hz
    int maxLag = sampleRate / 50;  // 50 Hz
    if (maxLag <= minLag)
        return 0.0f;
    Eigen::Map<const Eigen::ArrayXf> arr(x.data(), static_cast<Eigen::Index>(x.size()));
    float best = 0.0f;
    int bestLag = 0;
    for (int lag = minLag; lag <= maxLag; ++lag)
    {
        Eigen::ArrayXf x1 = arr.head(arr.size() - lag);
        Eigen::ArrayXf x2 = arr.segment(lag, arr.size() - lag);
        float sum = (x1 * x2).sum();
        if (sum > best)
        {
            best = sum;
            bestLag = lag;
        }
    }
    if (bestLag == 0)
        return 0.0f;
    return float(sampleRate) / float(bestLag);
}

namespace
{
    // 计算两段序列的归一化相关性。
    // 调用方式：用于比较分离轨道激活曲线相似度。
    // 实现思路：计算点积并除以范数乘积。
    // 注意事项：遇到空向量或零范数时返回 0。
    static float corrNorm(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.empty() || b.empty())
            return 0.0f;
        size_t n = std::min(a.size(), b.size());
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        double denom = std::sqrt(std::max(1e-9, na * nb));
        return (float)(dot / denom);
    }

    // 计算 STFT 幅值谱矩阵。
    // 调用方式：输入时域信号、采样率、窗长与步长。
    // 实现思路：逐帧加窗并对每个频点执行离散傅里叶变换。
    // 注意事项：空输入会返回空矩阵。
    static Eigen::MatrixXf stftMag(const std::vector<float> &x, int sr, int win, int hop)
    {
        if (x.empty())
            return Eigen::MatrixXf();
        const int nFrames = (int)((x.size() - win + hop) / hop);
        const int nBins = win / 2 + 1;
        Eigen::VectorXf window(win);
        for (int i = 0; i < win; ++i)
            window(i) = 0.5f - 0.5f * std::cos(2.0f * kPi * i / (win - 1));
        Eigen::MatrixXf mag(nBins, std::max(0, nFrames));
        Eigen::VectorXf frame(win);
        for (int f = 0; f < nFrames; ++f)
        {
            int start = f * hop;
            for (int i = 0; i < win; ++i)
                frame(i) = x[start + i] * window(i);
            for (int k = 0; k < nBins; ++k)
            {
                float re = 0.0f, im = 0.0f;
                for (int n = 0; n < win; ++n)
                {
                    float angle = 2.0f * kPi * k * n / win;
                    re += frame(n) * std::cos(angle);
                    im -= frame(n) * std::sin(angle);
                }
                mag(k, f) = std::sqrt(re * re + im * im) + 1e-9f;
            }
        }
        return mag;
    }

    static float computeRmsLocal(const std::vector<float> &x)
    {
        if (x.empty())
            return 0.0f;
        Eigen::Map<const Eigen::ArrayXf> arr(x.data(), static_cast<Eigen::Index>(x.size()));
        return std::sqrt(arr.square().mean());
    }

    static float estimatePitchLocal(const std::vector<float> &x, int sampleRate)
    {
        if (x.size() < 512)
            return 0.0f;
        int minLag = sampleRate / 400;
        int maxLag = sampleRate / 50;
        if (maxLag <= minLag)
            return 0.0f;
        Eigen::Map<const Eigen::ArrayXf> arr(x.data(), static_cast<Eigen::Index>(x.size()));
        float best = 0.0f;
        int bestLag = 0;
        for (int lag = minLag; lag <= maxLag; ++lag)
        {
            Eigen::ArrayXf x1 = arr.head(arr.size() - lag);
            Eigen::ArrayXf x2 = arr.segment(lag, arr.size() - lag);
            float sum = (x1 * x2).sum();
            if (sum > best)
            {
                best = sum;
                bestLag = lag;
            }
        }
        if (bestLag == 0)
            return 0.0f;
        return float(sampleRate) / float(bestLag);
    }

    // 对非负矩阵执行 NMF 分解。
    // 调用方式：传入 V、秩 r、迭代次数，输出 W/H。
    // 实现思路：采用乘法更新规则并在每轮做列归一化。
    // 注意事项：用于近似分离，结果受随机初始化影响。
    static void nmfDecompose(const Eigen::MatrixXf &V, int r, int iters, Eigen::MatrixXf &W, Eigen::MatrixXf &H)
    {
        const int F = V.rows();
        const int T = V.cols();
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        W = Eigen::MatrixXf::Zero(F, r);
        H = Eigen::MatrixXf::Zero(r, T);
        for (int i = 0; i < F; ++i)
            for (int j = 0; j < r; ++j)
                W(i, j) = dist(rng) + 1e-3f;
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < T; ++j)
                H(i, j) = dist(rng) + 1e-3f;

        for (int it = 0; it < iters; ++it)
        {
            Eigen::MatrixXf WH = (W * H).array() + 1e-6f;
            Eigen::MatrixXf Hnum = W.transpose() * (V.array() / WH.array()).matrix();
            Eigen::MatrixXf Hden = W.transpose().rowwise().sum().replicate(1, T).array() + 1e-6f;
            H = H.array() * (Hnum.array() / Hden.array());

            WH = (W * H).array() + 1e-6f;
            Eigen::MatrixXf Wnum = (V.array() / WH.array()).matrix() * H.transpose();
            Eigen::MatrixXf Wden = H.rowwise().sum().transpose().replicate(F, 1).array() + 1e-6f;
            W = W.array() * (Wnum.array() / Wden.array());

            // 归一化列
            for (int k = 0; k < r; ++k)
            {
                float norm = W.col(k).sum();
                if (norm > 1e-6f)
                {
                    W.col(k) /= norm;
                    H.row(k) *= norm;
                }
            }
        }
    }

    // AudioLearner 维护在线聚类器，用于环境与情绪标签自学习。
    // 调用方式：通过 audioLearner() 获取单例并调用 classify* 接口。
    // 实现思路：对特征向量执行增量式最近中心更新。
    // 注意事项：当前标签为 cluster 编号语义，不是固定情感词典。
    // 注意事项：内部含互斥锁，可被多线程并发调用。
    // 注意事项：聚类数可由环境变量调整。
    struct AudioLearner
    {
        // Cluster 表示一个在线聚类中心及其样本计数。
        // 调用方式：由 AudioLearner 内部维护，不对外暴露。
        // 实现思路：centroid 保存中心向量，count 保存更新次数。
        // 注意事项：count 越大，后续更新步长越小。
        // 注意事项：centroid 维度应与输入特征一致。
        // 注意事项：仅在同一特征空间中比较距离。
        struct Cluster
        {
            Eigen::VectorXf centroid;
            int count{0};
        };

        std::mutex mu;
        int envK{6};
        int emoK{6};
        std::vector<Cluster> envClusters;
        std::vector<Cluster> emoClusters;

        // 从环境变量读取整数配置。
        // 调用方式：传入变量名和回退值。
        // 实现思路：解析失败或缺失时返回 fallback。
        // 注意事项：返回值至少为 1，避免无效聚类数。
        static int getEnvInt(const char *name, int fallback)
        {
            const char *v = std::getenv(name);
            if (!v || !*v)
                return fallback;
            try
            {
                return std::max(1, std::stoi(v));
            }
            catch (...)
            {
                return fallback;
            }
        }

        // 构造在线学习器并加载聚类参数。
        // 调用方式：由静态单例初始化时自动调用。
        // 实现思路：读取环境变量设置环境/情绪聚类个数。
        // 注意事项：构造过程不做重计算，开销较小。
        AudioLearner()
        {
            envK = getEnvInt("SPEAK_ENV_CLUSTERS", 6);
            emoK = getEnvInt("SPEAK_EMO_CLUSTERS", 6);
        }

        // 将特征分配到最近聚类并执行增量更新。
        // 调用方式：传入聚类容器、特征向量与最大聚类数。
        // 实现思路：先最近邻，再在未满时扩容或更新中心。
        // 注意事项：返回值为聚类 id，供上层生成标签。
        int assign(std::vector<Cluster> &clusters, const Eigen::VectorXf &feat, int k)
        {
            if (clusters.empty())
            {
                clusters.push_back({feat, 1});
                return 0;
            }
            int best = 0;
            float bestDist = (clusters[0].centroid - feat).squaredNorm();
            for (int i = 1; i < (int)clusters.size(); ++i)
            {
                float d = (clusters[i].centroid - feat).squaredNorm();
                if (d < bestDist)
                {
                    bestDist = d;
                    best = i;
                }
            }
            if ((int)clusters.size() < k)
            {
                clusters.push_back({feat, 1});
                return (int)clusters.size() - 1;
            }
            Cluster &c = clusters[best];
            c.count += 1;
            float lr = 1.0f / (float)c.count;
            c.centroid = (1.0f - lr) * c.centroid + lr * feat;
            return best;
        }

        // 预测并更新环境聚类标签。
        // 调用方式：传入环境特征向量，返回 env_cluster_x。
        // 实现思路：在互斥保护下调用 assign。
        // 注意事项：该标签为在线聚类结果，随数据分布变化。
        std::string classifyEnv(const Eigen::VectorXf &feat)
        {
            std::lock_guard<std::mutex> lock(mu);
            int id = assign(envClusters, feat, envK);
            return "env_cluster_" + std::to_string(id);
        }

        // 预测并更新情绪聚类标签。
        // 调用方式：传入韵律/能量特征，返回 emo_cluster_x。
        // 实现思路：复用同一增量聚类框架。
        // 注意事项：标签仅用于相对分组，不代表绝对情绪定义。
        std::string classifyEmo(const Eigen::VectorXf &feat)
        {
            std::lock_guard<std::mutex> lock(mu);
            int id = assign(emoClusters, feat, emoK);
            return "emo_cluster_" + std::to_string(id);
        }
    };

    // 返回音频学习器单例。
    // 调用方式：各分析流程统一通过此入口获取。
    // 实现思路：函数内静态对象实现懒加载。
    // 注意事项：单例在进程生命周期内保持状态。
    static AudioLearner &audioLearner()
    {
        static AudioLearner learner;
        return learner;
    }

    // 提取谱域统计特征向量。
    // 调用方式：输入音频与 STFT 参数，返回 6 维特征。
    // 实现思路：从每帧幅值谱汇总质心、带宽、滚降、谱流等统计量。
    // 注意事项：空谱矩阵时返回零向量。
    static Eigen::VectorXf spectralStats(const std::vector<float> &x, int sr, int win, int hop)
    {
        Eigen::MatrixXf V = stftMag(x, sr, win, hop);
        if (V.size() == 0)
            return Eigen::VectorXf::Zero(6);
        const int F = V.rows();
        const int T = V.cols();
        Eigen::VectorXf freqs(F);
        for (int f = 0; f < F; ++f)
            freqs(f) = (float)f * sr / win;

        float centroid = 0.0f, bandwidth = 0.0f, rolloff = 0.0f, flux = 0.0f;
        float magSumAll = 0.0f;
        Eigen::VectorXf prev = V.col(0);
        for (int t = 0; t < T; ++t)
        {
            Eigen::VectorXf col = V.col(t);
            float magSum = col.sum() + 1e-9f;
            magSumAll += magSum;
            centroid += (freqs.dot(col) / magSum);
            float meanFreq = freqs.dot(col) / magSum;
            Eigen::VectorXf diff = freqs.array() - meanFreq;
            bandwidth += std::sqrt((diff.array().square() * col.array()).sum() / magSum);

            float cum = 0.0f;
            float target = magSum * 0.85f;
            for (int f = 0; f < F; ++f)
            {
                cum += col(f);
                if (cum >= target)
                {
                    rolloff += freqs(f);
                    break;
                }
            }
            if (t > 0)
            {
                Eigen::VectorXf d = col - prev;
                flux += std::sqrt(std::max(0.0f, d.array().square().sum()));
            }
            prev = col;
        }
        float denom = std::max(1.0f, (float)T);
        centroid /= denom;
        bandwidth /= denom;
        rolloff /= denom;
        flux /= denom;

        Eigen::VectorXf out(6);
        out << centroid, bandwidth, rolloff, flux, (magSumAll / (float)(T * F)), (float)T;
        return out;
    }

    // 提取韵律统计特征。
    // 调用方式：输入波形与采样率，返回 6 维 prosody 特征。
    // 实现思路：分帧后统计 pitch/rms 均值方差并估计 jitter/shimmer。
    // 注意事项：帧数不足时返回零向量。
    static Eigen::VectorXf prosodyStats(const std::vector<float> &x, int sr)
    {
        const int win = std::max(256, std::min(1024, sr / 16));
        const int hop = win / 4;
        int frames = (int)((x.size() - win + hop) / hop);
        if (frames <= 0)
            return Eigen::VectorXf::Zero(6);
        std::vector<float> pitchSeries;
        std::vector<float> rmsSeries;
        pitchSeries.reserve(frames);
        rmsSeries.reserve(frames);
        for (int i = 0; i < frames; ++i)
        {
            int start = i * hop;
            std::vector<float> seg(x.begin() + start, x.begin() + start + win);
            pitchSeries.push_back(estimatePitchLocal(seg, sr));
            rmsSeries.push_back(computeRmsLocal(seg));
        }
        auto meanVar = [](const std::vector<float> &v)
        {
            double m = 0.0;
            for (float x : v)
                m += x;
            m /= std::max<size_t>(1, v.size());
            double var = 0.0;
            for (float x : v)
            {
                double d = x - m;
                var += d * d;
            }
            var /= std::max<size_t>(1, v.size());
            return std::pair<float, float>((float)m, (float)var);
        };
        auto [pMean, pVar] = meanVar(pitchSeries);
        auto [rMean, rVar] = meanVar(rmsSeries);

        // jitter/shimmer 近似
        float jitter = 0.0f, shimmer = 0.0f;
        for (size_t i = 1; i < pitchSeries.size(); ++i)
        {
            jitter += std::abs(pitchSeries[i] - pitchSeries[i - 1]);
            shimmer += std::abs(rmsSeries[i] - rmsSeries[i - 1]);
        }
        jitter /= std::max<size_t>(1, pitchSeries.size());
        shimmer /= std::max<size_t>(1, rmsSeries.size());

        Eigen::VectorXf out(6);
        out << pMean, std::sqrt(std::max(0.0f, pVar)), rMean, std::sqrt(std::max(0.0f, rVar)), jitter, shimmer;
        return out;
    }
}

// 基于 NMF 进行粗粒度声源分离并产出关系特征。
// 调用方式：传入 AudioData，返回 tracks/relations 等字段。
// 实现思路：STFT→NMF→每分量统计能量、连续性与互相关。
// 注意事项：该结果用于分析和学习，不是高保真分轨重建。
Json::Value SpeakIO::separateTracks(const AudioData &audio)
{
    if (speakLogger().enabled())
        speakLogger().log(LoggerCXX::Type::COMPUTE, std::string("speak separateTracks samples=") + std::to_string(audio.mono.size()));
    Json::Value out;
    if (audio.mono.empty())
        return out;
    const int sr = audio.sampleRate;
    const int win = std::max(256, std::min(1024, sr / 16));
    const int hop = win / 4;
    Eigen::MatrixXf V = stftMag(audio.mono, sr, win, hop);
    if (V.size() == 0)
        return out;

    const int components = 4;
    Eigen::MatrixXf W, H;
    nmfDecompose(V, components, 30, W, H);

    Json::Value tracks(Json::arrayValue);
    std::vector<float> compEnergy(components, 0.0f);
    std::vector<std::vector<float>> compActivation(components);

    for (int k = 0; k < components; ++k)
    {
        // 频谱质心与连续性指标
        float freqSum = 0.0f, magSum = 0.0f;
        for (int f = 0; f < W.rows(); ++f)
        {
            float m = W(f, k);
            float freq = (float)f * sr / win;
            freqSum += freq * m;
            magSum += m;
        }
        float centroid = magSum > 0 ? freqSum / magSum : 0.0f;
        float continuity = 0.0f;
        if (H.cols() > 1)
        {
            float flux = 0.0f;
            for (int t = 1; t < H.cols(); ++t)
            {
                float d = H(k, t) - H(k, t - 1);
                flux += d * d;
            }
            continuity = 1.0f / (1.0f + std::sqrt(std::max(0.0f, flux)) / H.cols());
        }

        // 时间激活转为向量
        compActivation[k].resize(H.cols());
        for (int t = 0; t < H.cols(); ++t)
        {
            compActivation[k][t] = H(k, t);
            compEnergy[k] += H(k, t) * H(k, t);
        }

        Json::Value t;
        t["name"] = "track_" + std::to_string(k);
        t["spectralCentroid"] = centroid;
        t["continuity"] = continuity;
        t["energy"] = compEnergy[k];
        tracks.append(t);
    }

    double totalEnergy = 0.0;
    for (float e : compEnergy)
        totalEnergy += e;
    totalEnergy = std::max(1e-9, totalEnergy);

    Json::Value relations;
    Json::Value energyShare(Json::objectValue);
    for (int k = 0; k < components; ++k)
    {
        energyShare["track_" + std::to_string(k)] = compEnergy[k] / totalEnergy;
    }
    relations["energy_share"] = energyShare;

    Json::Value corr(Json::objectValue);
    for (int i = 0; i < components; ++i)
    {
        for (int j = i + 1; j < components; ++j)
        {
            corr["track_" + std::to_string(i) + "_track_" + std::to_string(j)] = corrNorm(compActivation[i], compActivation[j]);
        }
    }
    relations["activation_correlation"] = corr;

    out["tracks"] = tracks;
    out["relations"] = relations;
    out["meta"] = Json::Value(Json::objectValue);
    out["meta"]["method"] = "stft_nmf";
    out["meta"]["components"] = components;
    out["meta"]["win"] = win;
    out["meta"]["hop"] = hop;
    return out;
}

// 将浮点单声道样本编码为 PCM16 WAV。
// 调用方式：传入 AudioData，返回完整 WAV 字节数组。
// 实现思路：写入 44 字节头并量化样本到 int16。
// 注意事项：当前固定输出单声道 16bit。
std::vector<uint8_t> SpeakIO::encodeWavPcm16(const AudioData &audio)
{
    const int sr = audio.sampleRate;
    const int channels = 1;
    const int bits = 16;
    const size_t frames = audio.mono.size();
    const size_t dataBytes = frames * channels * (bits / 8);
    const size_t totalBytes = 44 + dataBytes;
    std::vector<uint8_t> out(totalBytes, 0);

    auto writeU32 = [&](size_t off, uint32_t v)
    {
        out[off + 0] = (uint8_t)(v & 0xFF);
        out[off + 1] = (uint8_t)((v >> 8) & 0xFF);
        out[off + 2] = (uint8_t)((v >> 16) & 0xFF);
        out[off + 3] = (uint8_t)((v >> 24) & 0xFF);
    };
    auto writeU16 = [&](size_t off, uint16_t v)
    {
        out[off + 0] = (uint8_t)(v & 0xFF);
        out[off + 1] = (uint8_t)((v >> 8) & 0xFF);
    };

    std::memcpy(out.data() + 0, "RIFF", 4);
    writeU32(4, (uint32_t)(totalBytes - 8));
    std::memcpy(out.data() + 8, "WAVE", 4);
    std::memcpy(out.data() + 12, "fmt ", 4);
    writeU32(16, 16);
    writeU16(20, 1);
    writeU16(22, (uint16_t)channels);
    writeU32(24, (uint32_t)sr);
    writeU32(28, (uint32_t)(sr * channels * (bits / 8)));
    writeU16(32, (uint16_t)(channels * (bits / 8)));
    writeU16(34, (uint16_t)bits);
    std::memcpy(out.data() + 36, "data", 4);
    writeU32(40, (uint32_t)dataBytes);

    int16_t *pcm = reinterpret_cast<int16_t *>(out.data() + 44);
    for (size_t i = 0; i < frames; ++i)
    {
        float v = std::max(-1.0f, std::min(1.0f, audio.mono[i]));
        pcm[i] = (int16_t)std::lround(v * 32767.0f);
    }
    return out;
}

// 对二进制数据做 Base64 编码。
// 调用方式：用于把 WAV 字节封装到 JSON 字段。
// 实现思路：按 3 字节分组映射到编码表并处理尾部补位。
// 注意事项：输出不带换行，适合直接作为 HTTP JSON 字段。
std::string SpeakIO::base64Encode(const std::vector<uint8_t> &data)
{
    static const char kEnc[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 2 < data.size())
    {
        uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | uint32_t(data[i + 2]);
        out.push_back(kEnc[(v >> 18) & 0x3F]);
        out.push_back(kEnc[(v >> 12) & 0x3F]);
        out.push_back(kEnc[(v >> 6) & 0x3F]);
        out.push_back(kEnc[v & 0x3F]);
        i += 3;
    }
    if (i < data.size())
    {
        uint32_t v = uint32_t(data[i]) << 16;
        out.push_back(kEnc[(v >> 18) & 0x3F]);
        if (i + 1 < data.size())
        {
            v |= uint32_t(data[i + 1]) << 8;
            out.push_back(kEnc[(v >> 12) & 0x3F]);
            out.push_back(kEnc[(v >> 6) & 0x3F]);
            out.push_back('=');
        }
        else
        {
            out.push_back(kEnc[(v >> 12) & 0x3F]);
            out.push_back('=');
            out.push_back('=');
        }
    }
    return out;
}

// 文本到语音的简化合成实现。
// 调用方式：输入文本与采样率/语速/音高，返回 WAV Base64。
// 实现思路：字符映射到基频并叠加谐波与包络生成样本。
// 注意事项：该实现偏教学和占位用途，不等同专业 TTS 效果。
Json::Value SpeakIO::synthesizeText(const std::string &text, int sampleRate, float speed, float pitch)
{
    if (speakLogger().enabled())
        speakLogger().log(LoggerCXX::Type::COMPUTE, std::string("speak synthesizeText chars=") + std::to_string(text.size()) + " sr=" + std::to_string(sampleRate));
    Json::Value out;
    if (text.empty())
    {
        out["ok"] = false;
        out["error"] = "empty text";
        return out;
    }
    int sr = std::max(8000, std::min(48000, sampleRate));
    float sp = std::max(0.5f, std::min(3.0f, speed));
    float pitchMul = std::max(0.5f, std::min(2.0f, pitch));
    float baseDur = 0.09f / sp;

    std::vector<float> samples;
    samples.reserve(text.size() * (size_t)(baseDur * sr * 1.2f));
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);

    auto charFreq = [&](char c)
    {
        int v = (int)(unsigned char)c;
        int h = (v * 37 + 17) % 140;
        return (180.0f + h) * pitchMul;
    };
    auto isVowel = [&](char c)
    {
        c = (char)std::tolower((unsigned char)c);
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
    };

    for (size_t idx = 0; idx < text.size(); ++idx)
    {
        char c = text[idx];
        float dur = (std::isspace((unsigned char)c) || c == '.' || c == ',' || c == ';' || c == '!') ? 0.05f : baseDur;
        int count = std::max(1, (int)(dur * sr));
        float f = charFreq(c);
        float amp = isVowel(c) ? 0.22f : 0.16f;
        int attack = std::max(1, (int)(0.01f * sr));
        int release = std::max(1, (int)(0.01f * sr));

        for (int n = 0; n < count; ++n)
        {
            float t = (float)n / (float)sr;
            float env = 1.0f;
            if (n < attack)
                env = (float)n / attack;
            else if (n > count - release)
                env = (float)(count - n) / release;
            float s = std::sin(2.0f * kPi * f * t) + 0.5f * std::sin(2.0f * kPi * 2.0f * f * t) + 0.25f * std::sin(2.0f * kPi * 3.0f * f * t);
            if (!isVowel(c))
                s += noise(rng) * 0.05f;
            samples.push_back(amp * env * s);
        }
        if (std::isspace((unsigned char)c))
        {
            int silence = (int)(0.03f * sr);
            samples.insert(samples.end(), silence, 0.0f);
        }
    }

    AudioData audio;
    audio.sampleRate = sr;
    audio.channels = 1;
    audio.mono = std::move(samples);

    auto wav = encodeWavPcm16(audio);
    out["ok"] = true;
    out["sampleRate"] = sr;
    out["durationMs"] = (Json::Int64)((audio.mono.size() * 1000LL) / sr);
    out["audioBase64"] = base64Encode(wav);
    out["mime"] = "audio/wav";
    return out;
}

// 根据简化特征给出环境聚类标签。
// 调用方式：传入 rms/zcr/centroid，返回环境类别字符串。
// 实现思路：补齐到固定维度后调用 AudioLearner。
// 注意事项：特征维度为兼容在线学习器而设计。
std::string SpeakIO::classifyEnvironment(float rms, float zcr, float centroid)
{
    Eigen::VectorXf feat(9);
    feat << rms, zcr, centroid, centroid, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f;
    return audioLearner().classifyEnv(feat);
}

// 根据简化特征给出情绪聚类标签。
// 调用方式：传入 rms/pitch，返回情绪类别字符串。
// 实现思路：构造 8 维占位特征并调用 AudioLearner。
// 注意事项：分类结果是聚类索引标签，解释需结合上下文。
std::string SpeakIO::classifyEmotion(float rms, float pitch)
{
    Eigen::VectorXf feat(8);
    feat << rms, pitch, pitch, 0.0f, rms, 0.0f, 0.0f, 0.0f;
    return audioLearner().classifyEmo(feat);
}
