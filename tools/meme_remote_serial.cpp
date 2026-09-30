/* meme_remote_serial.cpp -- 远程串行复制实验（隔代 wipe-ingest）。

   与 tools/phoenix_gnn_sig_serial_dialogue.ps1 完全同语义的独立静态 exe，
   用于无 Python/无 pip 的裸 Windows 机器：
   - 载体在本地预合成（candidates.json 的 seedRag 字段），远端不需要语料库
   - 每组 r1..R：detectIn(冻结图) -> llama /completion -> continuation 截断
     -> exam/echo 检测 -> dirty 停 -> detect -> wipe-ingest（全新 live 图，
     冻结图作参照）-> collapse 拦截 -> judge(r>=judgeAfter) 放宽口径
     (alphaSame 或 typicalHit>=need/2)，累计 2 次失败才停
     -> hopeless-00 早停 -> current = recomposeText（或 fallback 用 output）
   - 输出 schema 与 ps1 产物一致，manifold_batch_classify.py 直接分类
   - jsonl journal 断点续跑（修复了 ps1 resume 在 dirty/collapse/hopeless
     终止记录后会重跑的问题：本工具把这些记录视为终止点）
   - 结果文件原子写（tmp + MoveFileEx 替换）

   仅依赖：winsock2（系统）、nlohmann/json（ollama vendor 头）、项目内
   memetic_existence。静态链接，无 dll 依赖。

   编译（Windows，仓库根目录 phoenix/ 下）：
     g++ -std=c++20 -O2 -static -static-libgcc -static-libstdc++ -I. \
       -I outsides/ollama/llama/llama.cpp/vendor \
       tools/meme_remote_serial.cpp memetic_existence.cpp \
       graph_diffusion_summarizer.cpp addons/ThePlugInForSecurity/meme_tensor.cpp \
       -lws2_32 -o build/meme_remote_serial.exe

   编译（Linux / RDK X5 aarch64，本机或交叉编译；POSIX socket 自动切换）：
     g++ -std=c++20 -O2 -static -I. \
       -I outsides/ollama/llama/llama.cpp/vendor \
       tools/meme_remote_serial.cpp memetic_existence.cpp \
       graph_diffusion_summarizer.cpp addons/ThePlugInForSecurity/meme_tensor.cpp \
       -o meme_remote_serial
     # 交叉编译（x86_64 宿主机）：把 g++ 换成 aarch64-linux-gnu-g++

   运行（远端，llama-server 已在本机 --port 8080 --parallel 1 启动）：
     meme_remote_serial.exe --snap snap.txt --candidates candidates.json \
       --out results.json --port 8080
*/
#include "memetic_existence.hpp"

#include <nlohmann/json.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else /* POSIX (Linux / RDK X5 aarch64 等) */
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
using SOCKET = int;
constexpr SOCKET INVALID_SOCKET = -1;
constexpr int SOCKET_ERROR = -1;
#define closesocket ::close
#define WSAStartup(a, b) 0
#define WSACleanup() ((void)0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;
using phoenix::memetic::InMemoryStore;
using phoenix::memetic::detectMemeInText;
using phoenix::memetic::ingestCorpusText;
using phoenix::memetic::mappingMergeStopWord;

namespace {

/* ------------------------------------------------------------------ */
/* 文本健康统计（与 meme_barrier_instrument.cpp 的 TextStats 一致）      */
/* ------------------------------------------------------------------ */
struct TextStats {
  int tokens{0};
  double entropy{0.0};
  double top1Frac{0.0};
  int maxRun{0};
  bool collapsed{false};
};

TextStats textStatsOf(const std::string &text) {
  TextStats st;
  std::vector<std::string> toks;
  std::string cur;
  for (unsigned char c : text) {
    if (c >= 'A' && c <= 'Z')
      cur += static_cast<char>(c - 'A' + 'a');
    else if (c >= 'a' && c <= 'z')
      cur += static_cast<char>(c);
    else if (!cur.empty()) {
      toks.push_back(cur);
      cur.clear();
    }
  }
  if (!cur.empty())
    toks.push_back(cur);
  st.tokens = static_cast<int>(toks.size());
  if (toks.empty())
    return st;
  std::unordered_map<std::string, int> freq;
  for (const auto &t : toks)
    ++freq[t];
  double h = 0.0;
  int top = 0;
  for (const auto &kv : freq) {
    const double p = static_cast<double>(kv.second) / toks.size();
    h -= p * std::log2(p);
    top = std::max(top, kv.second);
  }
  st.entropy = h;
  st.top1Frac = static_cast<double>(top) / toks.size();
  int run = 1;
  for (size_t i = 1; i < toks.size(); ++i) {
    if (toks[i] == toks[i - 1]) {
      ++run;
      st.maxRun = std::max(st.maxRun, run);
    } else {
      run = 1;
    }
  }
  if (st.maxRun < 1)
    st.maxRun = 1;
  st.collapsed = st.maxRun >= 8 || st.top1Frac >= 0.5;
  return st;
}

bool isWhite(const std::string &s) {
  for (unsigned char c : s) {
    if (!std::isspace(c))
      return false;
  }
  return true;
}

std::string collapseWs(const std::string &s) {
  std::string out;
  bool inWs = false;
  for (unsigned char c : s) {
    if (std::isspace(c)) {
      inWs = true;
    } else {
      if (inWs && !out.empty())
        out += ' ';
      out += static_cast<char>(c);
      inWs = false;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* exam / continuation / echo —— 移植自 experiment_ha.ps1               */
/* ------------------------------------------------------------------ */
const std::vector<std::regex> &examPats() {
  static const std::vector<std::regex> pats = {
      std::regex("what (is|can|does) (the main subject|be inferred|the "
                 "author)",
                 std::regex_constants::icase),
      std::regex("what is the name of", std::regex_constants::icase),
      std::regex("who (is|was|sang) the", std::regex_constants::icase),
      std::regex("answer rationale|the best answer is|this question requires",
                 std::regex_constants::icase),
      std::regex("\\bAnswer:\\s*[A-D]\\b", std::regex_constants::icase),
      std::regex("step-by-step solution|final answer is:",
                 std::regex_constants::icase),
  };
  return pats;
}

/* (?m)^\s*[A-D]\s*\)\s —— 逐行检查（.NET 该模式不带 icase，仅大写 A-D）。
   返回首个匹配的全局偏移，无匹配返回 npos。 */
size_t optionLineMatch(const std::string &text) {
  static const std::regex linePat("^\\s*[A-D]\\s*\\)\\s");
  size_t off = 0;
  while (off <= text.size()) {
    size_t nl = text.find('\n', off);
    const std::string line =
        text.substr(off, nl == std::string::npos ? std::string::npos
                                                 : nl - off);
    std::smatch m;
    if (std::regex_search(line, m, linePat))
      return off + static_cast<size_t>(m.position(0));
    if (nl == std::string::npos)
      break;
    off = nl + 1;
  }
  return std::string::npos;
}

bool testExamDump(const std::string &text) {
  if (isWhite(text))
    return false;
  for (const auto &p : examPats()) {
    if (std::regex_search(text, p))
      return true;
  }
  return optionLineMatch(text) != std::string::npos;
}

std::string serialContinuation(const std::string &text) {
  if (isWhite(text))
    return "";
  static const std::vector<std::regex> markers = {
      std::regex("what is the main subject", std::regex_constants::icase),
      std::regex("what is the name of", std::regex_constants::icase),
      std::regex("who (is|was|sang) the", std::regex_constants::icase),
      std::regex("what can be inferred", std::regex_constants::icase),
      std::regex("answer rationale", std::regex_constants::icase),
      std::regex("this question requires", std::regex_constants::icase),
      std::regex("the best answer is", std::regex_constants::icase),
      std::regex("step-by-step solution", std::regex_constants::icase),
      std::regex("the final answer is", std::regex_constants::icase),
  };
  size_t idx = text.size();
  for (const auto &p : markers) {
    std::smatch m;
    if (std::regex_search(text, m, p) &&
        static_cast<size_t>(m.position(0)) < idx)
      idx = static_cast<size_t>(m.position(0));
  }
  const size_t optIdx = optionLineMatch(text);
  if (optIdx < idx)
    idx = optIdx;
  std::string cut = idx < text.size() ? text.substr(0, idx) : text;
  return collapseWs(cut);
}

bool testEcho(const std::string &prompt, const std::string &output) {
  if (isWhite(prompt) || isWhite(output))
    return false;
  const std::string p = collapseWs(prompt);
  const std::string o = collapseWs(output);
  if (p.size() < 24 || o.size() < 24)
    return o.find(p) != std::string::npos || p.find(o) != std::string::npos;
  for (const int win : {80, 32}) {
    const size_t w = std::min<size_t>(static_cast<size_t>(win), p.size());
    if (w < 16)
      continue;
    for (size_t i = 0; i + w <= p.size(); i += 8) {
      if (o.find(p.substr(i, w)) != std::string::npos)
        return true;
    }
  }
  const size_t pre = std::min<size_t>(48, p.size());
  if (pre >= 24 && o.find(p.substr(0, pre)) != std::string::npos)
    return true;
  const size_t suf = std::min<size_t>(48, p.size());
  if (suf >= 24 && o.find(p.substr(p.size() - suf, suf)) != std::string::npos)
    return true;
  if (o.size() >= 72) {
    const size_t span = 24;
    std::unordered_map<std::string, int> seen;
    for (size_t i = 0; i + span <= o.size(); i += 8) {
      if (++seen[o.substr(i, span)] >= 3)
        return true;
    }
  }
  return false;
}

/* ------------------------------------------------------------------ */
/* Winsock2 HTTP POST（Connection: close，读直到关闭；支持 chunked）     */
/* ------------------------------------------------------------------ */
struct WsaGuard {
  WsaGuard() {
#ifdef _WIN32
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
      throw std::runtime_error("WSAStartup failed");
#endif
  }
  ~WsaGuard() { WSACleanup(); }
};

std::string decodeChunked(const std::string &raw) {
  std::string out;
  size_t pos = 0;
  while (true) {
    const size_t eol = raw.find("\r\n", pos);
    if (eol == std::string::npos)
      throw std::runtime_error("chunked: missing size line");
    const long sz = std::strtol(raw.substr(pos, eol - pos).c_str(), nullptr, 16);
    if (sz < 0)
      throw std::runtime_error("chunked: bad size");
    pos = eol + 2;
    if (sz == 0)
      return out;
    if (pos + static_cast<size_t>(sz) > raw.size())
      throw std::runtime_error("chunked: truncated");
    out.append(raw, pos, static_cast<size_t>(sz));
    pos += static_cast<size_t>(sz) + 2; /* skip trailing CRLF */
  }
}

std::string httpPost(const std::string &host, int port,
                     const std::string &path, const std::string &body,
                     int timeoutMs) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  addrinfo *res = nullptr;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) !=
      0)
    throw std::runtime_error("getaddrinfo failed for " + host);
  SOCKET s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (s == INVALID_SOCKET) {
    freeaddrinfo(res);
    throw std::runtime_error("socket() failed");
  }
#ifdef _WIN32
  DWORD tv = static_cast<DWORD>(timeoutMs);
#else
  timeval tv{};
  tv.tv_sec = timeoutMs / 1000;
  tv.tv_usec = (timeoutMs % 1000) * 1000;
#endif
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&tv),
             sizeof(tv));
  setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&tv),
             sizeof(tv));
  if (connect(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) ==
      SOCKET_ERROR) {
    freeaddrinfo(res);
    closesocket(s);
    throw std::runtime_error("connect failed to " + host + ":" +
                             std::to_string(port));
  }
  freeaddrinfo(res);

  std::ostringstream req;
  req << "POST " << path << " HTTP/1.1\r\n"
      << "Host: " << host << ":" << port << "\r\n"
      << "Content-Type: application/json; charset=utf-8\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << body;
  const std::string wire = req.str();
  size_t sent = 0;
  while (sent < wire.size()) {
    const int n = send(s, wire.data() + sent,
                       static_cast<int>(wire.size() - sent), 0);
    if (n <= 0) {
      closesocket(s);
      throw std::runtime_error("send failed");
    }
    sent += static_cast<size_t>(n);
  }
  std::string raw;
  char buf[16384];
  while (true) {
    const int n = recv(s, buf, sizeof(buf), 0);
    if (n == 0)
      break; /* Connection: close -> server closed */
    if (n < 0) {
      closesocket(s);
      throw std::runtime_error("recv failed/timeout");
    }
    raw.append(buf, static_cast<size_t>(n));
  }
  closesocket(s);

  const size_t split = raw.find("\r\n\r\n");
  if (split == std::string::npos)
    throw std::runtime_error("bad http response (no header end)");
  std::string header = raw.substr(0, split);
  std::string payload = raw.substr(split + 4);
  if (header.find(" 200") == std::string::npos) {
    std::ostringstream first;
    first << header.substr(0, header.find("\r\n"));
    throw std::runtime_error("http non-200: " + first.str());
  }
  std::string headerLower = header;
  std::transform(headerLower.begin(), headerLower.end(), headerLower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (headerLower.find("transfer-encoding: chunked") != std::string::npos)
    payload = decodeChunked(payload);
  return payload;
}

/* 两种推理后端：
   - llama  : llama.cpp llama-server 的 POST /completion（与本地 ps1 完全
              同协议同参数；随包 exe 即此后端，CPU）
   - ollama : ollama 的 POST /api/generate（raw=true 绕过 chat 模板，
              参数映射 num_predict/temperature/top_p；ollama 自动用 GPU，
              适合远端 1660 Super）。模型必须是同一个 llama3.1:8b。 */
std::string llamaComplete(const std::string &prompt, int port,
                          const std::string &engine,
                          const std::string &modelName) {
  if (isWhite(prompt))
    throw std::runtime_error("empty carrier (do not continue dirty serial)");
  std::string resp;
  if (engine == "ollama") {
    json body;
    body["model"] = modelName;
    body["prompt"] = prompt;
    body["raw"] = true; /* 不套 chat 模板，与 llama-server /completion 对齐 */
    body["stream"] = false;
    body["options"] = {{"num_predict", 96},
                       {"temperature", 0.2},
                       {"top_p", 0.9}};
    resp = httpPost("127.0.0.1", port, "/api/generate", body.dump(), 1800000);
  } else {
    json body;
    body["prompt"] = prompt;
    body["n_predict"] = 96;
    body["stream"] = false;
    body["cache_prompt"] = false;
    body["temperature"] = 0.2;
    body["top_p"] = 0.9;
    resp = httpPost("127.0.0.1", port, "/completion", body.dump(), 1800000);
  }
  if (isWhite(resp))
    throw std::runtime_error("empty llama http body; stop dirty serial");
  const auto j = json::parse(resp);
  std::string text;
  if (j.contains("content") && j["content"].is_string())
    text = j["content"].get<std::string>();
  else if (j.contains("response") && j["response"].is_string())
    text = j["response"].get<std::string>(); /* ollama /api/generate */
  else if (j.contains("choices") && j["choices"].is_array() &&
           !j["choices"].empty() && j["choices"][0].contains("message") &&
           j["choices"][0]["message"].contains("content"))
    text = j["choices"][0]["message"]["content"].get<std::string>();
  return text;
}

/* ------------------------------------------------------------------ */
/* 冻结图 detect / wipe-ingest —— 内联移植自 meme_barrier_instrument.cpp */
/* ------------------------------------------------------------------ */
json detectJson(const InMemoryStore &store, const std::string &text,
                const std::string &memeId) {
  const auto d = detectMemeInText(store, text, memeId, {});
  return json{{"present", d.present},     {"activation", d.memeScore},
              {"peak", d.peakScore},      {"rank", d.rank},
              {"lift", d.lift},           {"nullScore", d.nullScore}};
}

json wipeIngestJson(const InMemoryStore &store, const std::string &memeId,
                    const std::string &text, int maxUnits) {
  const auto frozenWords = store.wordsOfMeme(memeId);
  const auto frozenW = store.weightsOfMeme(memeId);
  const auto ingestDet = detectMemeInText(store, text, memeId, {});
  const auto frozenActs =
      phoenix::memetic::detectActivatedMemes(store, text, {}, 8);
  int targetActRank = ingestDet.rank;
  int otherFrozenPresent = 0;
  for (const auto &a : frozenActs) {
    if (a.present && a.id != memeId)
      ++otherFrozenPresent;
  }
  const bool targetDominant =
      ingestDet.present &&
      (ingestDet.rank == 0 ||
       (ingestDet.peakScore > 1e-12 &&
        ingestDet.memeScore >= 0.5 * ingestDet.peakScore));
  InMemoryStore live;
  const bool emptyBefore = live.empty() && live.nodeCount() == 0;
  const int liveUnits = ingestCorpusText(live, text, maxUnits, {});
  std::unordered_set<std::string> frozenContent;
  for (const auto &w : frozenWords) {
    if (!w.empty() && !mappingMergeStopWord(w))
      frozenContent.insert(w);
  }
  int overlapContent = 0;
  std::string bestId;
  int bestOverlap = -1;
  int bestSize = 1 << 30;
  phoenix::memetic::AlphaIdentity bestAlpha;
  for (const auto &id : live.nodeIds()) {
    int ov = 0;
    const auto ws = live.wordsOfMeme(id);
    for (const auto &w : ws) {
      if (frozenContent.count(w))
        ++ov;
    }
    const auto idn = phoenix::memetic::mappingAlphaIdentity(
        frozenW, live.weightsOfMeme(id));
    const bool betterAlpha =
        idn.typicalHit > bestAlpha.typicalHit ||
        (idn.typicalHit == bestAlpha.typicalHit &&
         idn.massCover > bestAlpha.massCover);
    const bool betterOv =
        ov > bestOverlap ||
        (ov == bestOverlap && ov > 0 &&
         static_cast<int>(ws.size()) < bestSize);
    if (betterAlpha || (idn.typicalHit == bestAlpha.typicalHit && betterOv)) {
      bestAlpha = idn;
      bestOverlap = ov;
      bestSize = static_cast<int>(ws.size());
      bestId = id;
      overlapContent = ov;
    }
  }
  phoenix::memetic::RagCarrier rag;
  phoenix::memetic::MemeDetection reDet{};
  if (!bestId.empty() && (bestAlpha.typicalHit > 0 || overlapContent > 0)) {
    rag = phoenix::memetic::composeCarrierRag(live, bestId, text, maxUnits, 2);
    reDet = detectMemeInText(store, rag.text, memeId, {});
  }
  const TextStats inStats = textStatsOf(text);
  const auto typFrozen = phoenix::memetic::mappingTypicalAlpha(frozenW);
  std::vector<std::string> hitWords;
  std::vector<std::string> missWords;
  if (!bestId.empty()) {
    std::unordered_set<std::string> liveW;
    for (const auto &wp : live.weightsOfMeme(bestId)) {
      if (!wp.first.empty() && wp.second > 0.0)
        liveW.insert(wp.first);
    }
    for (const auto &kv : typFrozen) {
      if (liveW.count(kv.first))
        hitWords.push_back(kv.first);
      else
        missWords.push_back(kv.first);
    }
  } else {
    for (const auto &kv : typFrozen)
      missWords.push_back(kv.first);
  }
  std::vector<std::string> typFrozenWords;
  typFrozenWords.reserve(typFrozen.size());
  for (const auto &kv : typFrozen)
    typFrozenWords.push_back(kv.first);
  std::sort(typFrozenWords.begin(), typFrozenWords.end());

  return json{
      {"emptyBefore", emptyBefore},
      {"liveUnits", liveUnits},
      {"liveNodes", live.nodeCount()},
      {"frozenContent", frozenContent.size()},
      {"overlapContent", overlapContent},
      {"typicalNeed", bestAlpha.typicalNeed},
      {"typicalHit", bestAlpha.typicalHit},
      {"massCover", bestAlpha.massCover},
      {"alphaCos", bestAlpha.cosine},
      {"alphaSame", bestAlpha.same},
      {"typicalFrozen", typFrozenWords},
      {"typicalHitWords", hitWords},
      {"typicalMissWords", missWords},
      {"outEntropy", inStats.entropy},
      {"outTop1Frac", inStats.top1Frac},
      {"outMaxRun", inStats.maxRun},
      {"collapsed", inStats.collapsed},
      {"decayWarn", liveUnits <= 1 || inStats.collapsed},
      {"targetDominant", targetDominant},
      {"targetActRank", targetActRank},
      {"otherFrozenPresent", otherFrozenPresent},
      {"bestLiveId", bestId},
      {"ingestPresent", ingestDet.present},
      {"ingestLift", ingestDet.lift},
      {"ingestRank", ingestDet.rank},
      {"recomposePresent", reDet.present},
      {"recomposeLift", reDet.lift},
      {"recomposeRank", reDet.rank},
      {"recomposeMethod", rag.method},
      {"recomposeText", rag.text},
  };
}

/* ------------------------------------------------------------------ */
/* journal / 结果保存                                                   */
/* ------------------------------------------------------------------ */
std::unordered_map<std::string, json> readJournal(const std::string &path) {
  std::unordered_map<std::string, json> done;
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return done;
  std::string line;
  while (std::getline(in, line)) {
    if (isWhite(line))
      continue;
    try {
      auto o = json::parse(line);
      const std::string key = o.value("mode", std::string()) + "|" +
                              o.value("id", std::string()) + "|" +
                              std::to_string(o.value("round", 0));
      done[key] = std::move(o);
    } catch (...) {
      /* 跳过损坏行（上次中断时写了一半） */
    }
  }
  return done;
}

void appendJournal(const std::string &path, const json &rec) {
  std::ofstream out(path, std::ios::binary | std::ios::app);
  out << rec.dump() << "\n";
}

void saveJsonAtomic(const std::string &path, const json &j) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out << j.dump(2);
  }
#ifdef _WIN32
  if (!MoveFileExA(tmp.c_str(), path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    std::remove(path.c_str());
    std::rename(tmp.c_str(), path.c_str());
  }
#else
  /* POSIX rename 在同文件系统内原子 */
  if (std::rename(tmp.c_str(), path.c_str()) != 0)
    throw std::runtime_error("atomic rename failed for " + path);
#endif
}

std::string utcNow() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

} // namespace

int main(int argc, char **argv) {
  std::string snapFile;
  std::string candidatesFile;
  std::string outFile;
  std::string modelNote;
  std::string engine = "llama"; /* llama | ollama */
  std::string modelName = "llama3.1:8b"; /* ollama 引擎用 */
  int port = 8080;
  int rounds = 6;
  int judgeAfter = 3;
  int units = 256;
  int limit = 0; /* 0 = 全部 */
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--snap" && i + 1 < argc)
      snapFile = argv[++i];
    else if (a == "--candidates" && i + 1 < argc)
      candidatesFile = argv[++i];
    else if (a == "--out" && i + 1 < argc)
      outFile = argv[++i];
    else if (a == "--port" && i + 1 < argc)
      port = std::atoi(argv[++i]);
    else if (a == "--engine" && i + 1 < argc)
      engine = argv[++i];
    else if (a == "--model-name" && i + 1 < argc)
      modelName = argv[++i];
    else if (a == "--rounds" && i + 1 < argc)
      rounds = std::atoi(argv[++i]);
    else if (a == "--judge-after" && i + 1 < argc)
      judgeAfter = std::atoi(argv[++i]);
    else if (a == "--units" && i + 1 < argc)
      units = std::atoi(argv[++i]);
    else if (a == "--limit" && i + 1 < argc)
      limit = std::atoi(argv[++i]);
    else if (a == "--model-note" && i + 1 < argc)
      modelNote = argv[++i];
  }
  std::cout.setf(std::ios::unitbuf); /* 重定向下也实时刷进度 */
  if (snapFile.empty() || candidatesFile.empty() || outFile.empty()) {
    std::cerr << "usage: meme_remote_serial --snap SNAP --candidates C.json "
                 "--out OUT.json [--port 8080] [--engine llama|ollama] "
                 "[--model-name llama3.1:8b] [--rounds 6] [--judge-after 3] "
                 "[--units 256] [--limit N] [--model-note TEXT]\n";
    return 2;
  }

  InMemoryStore store;
  if (!store.loadText(snapFile)) {
    std::cerr << "cannot load frozen snap " << snapFile << "\n";
    return 2;
  }
  if (store.empty()) {
    std::cerr << "empty frozen snap\n";
    return 2;
  }

  json candidates;
  {
    std::ifstream in(candidatesFile, std::ios::binary);
    if (!in) {
      std::cerr << "cannot read candidates " << candidatesFile << "\n";
      return 2;
    }
    std::ostringstream raw;
    raw << in.rdbuf();
    candidates = json::parse(raw.str());
  }
  const auto memes = candidates["memes"].get<std::vector<json>>();
  if (memes.empty()) {
    std::cerr << "empty candidates\n";
    return 1;
  }

  const std::string journalFile = [&] {
    std::string j = outFile;
    const size_t dot = j.find_last_of('.');
    if (dot != std::string::npos)
      j = j.substr(0, dot);
    return j + ".jsonl";
  }();
  auto done = readJournal(journalFile);

  std::cout << "==== remote dialogue serial: wipe GNN, ingest output only, "
               "recompose next utterance ====\n"
            << "snap=" << snapFile << " nodes=" << store.nodeCount()
            << " memes=" << memes.size() << " engine=" << engine
            << " port=" << port
            << (engine == "ollama" ? (" model=" + modelName) : "")
            << " resumedRounds=" << done.size() << "\n";

  json doc;
  doc["startedAtUtc"] = utcNow();
  doc["method"] =
      "readme 单智能体隔代再摄入: empty GNN + ingest last output; recover "
      "same (word, α) typical set, not bag-hash id; llama raw /completion "
      "of short carrier only (remote serial, same semantics as "
      "phoenix_gnn_sig_serial_dialogue.ps1)";
  doc["note"] =
      "remote run via meme_remote_serial.exe; model note: " + modelNote;
  doc["snap"] = snapFile;
  doc["candidates"] = candidatesFile;
  doc["engine"] = engine;
  doc["modelName"] = engine == "ollama" ? modelName : "";
  doc["modelNote"] = modelNote;
  doc["units"] = units;
  doc["rounds"] = rounds;
  doc["judgeAfter"] = judgeAfter;
  doc["groups"] = json::array();

  WsaGuard wsa;
  int ran = 0;

  for (const auto &m : memes) {
    const std::string id = m.value("id", std::string());
    const std::string pole = m.value("pole", std::string());
    if (limit > 0 && ran >= limit)
      break;
    /* 该组是否已在 journal 中全部完成（有终止记录或满轮）？若是则重建行。 */
    std::cout << "==== " << pole << " " << id
              << " sig=" << m.value("significance", 0.0) << " ====\n";

    json row;
    row["id"] = id;
    row["pole"] = pole;
    row["rankMost"] = m.value("rankMost", 0);
    row["rankLeast"] = m.value("rankLeast", 0);
    row["significance"] = m.value("significance", 0.0);
    row["wordCount"] = m.value("wordCount", 0);
    row["seedBag"] = m.value("seedBag", std::string());
    row["seedRag"] = m.value("seedRag", std::string());
    row["compose"] = m.value("compose", json::object());
    row["rounds"] = json::array();
    row["presentRounds"] = 0;
    row["echoRounds"] = 0;
    row["wipeEmptyRounds"] = 0;
    row["recomposePresentRounds"] = 0;
    row["alphaSameRounds"] = 0;
    row["judgeMiss"] = 0; /* judge 降噪：累计 judge 失败次数，>=2 才终止 */

    std::string current = m.value("seedRag", std::string());
    bool rowPushed = false;
    bool ranAny = false;

    auto pushRow = [&] {
      doc["groups"].push_back(row);
      rowPushed = true;
    };
    auto saveProgress = [&] {
      if (rowPushed) {
        saveJsonAtomic(outFile, doc);
      } else {
        json snapDoc = doc;
        snapDoc["groups"].push_back(row);
        saveJsonAtomic(outFile, snapDoc);
      }
    };

    for (int r = 1; r <= rounds; ++r) {
      const std::string key = pole + "|" + id + "|" + std::to_string(r);
      auto dit = done.find(key);
      if (dit != done.end()) {
        const json &prev = dit->second;
        row["rounds"].push_back(prev);
        if (prev.value("detect", json::object()).value("present", false))
          row["presentRounds"] = row["presentRounds"].get<int>() + 1;
        if (prev.value("echo", false))
          row["echoRounds"] = row["echoRounds"].get<int>() + 1;
        if (prev.contains("wipe") && prev["wipe"].is_object()) {
          if (prev["wipe"].value("emptyBefore", false))
            row["wipeEmptyRounds"] = row["wipeEmptyRounds"].get<int>() + 1;
          if (prev["wipe"].value("recomposePresent", false))
            row["recomposePresentRounds"] =
                row["recomposePresentRounds"].get<int>() + 1;
          if (prev["wipe"].value("alphaSame", false))
            row["alphaSameRounds"] = row["alphaSameRounds"].get<int>() + 1;
        }
        current = prev.value("nextUtterance", std::string());
        std::cout << "  resume round " << r << " present="
                  << prev.value("detect", json::object()).value("present",
                                                                false)
                  << "\n";
        /* judgeMiss 重建（2026-09 审计降噪口径：通过 = alphaSame 或
           typicalHit 达 typicalNeed 一半；累计 2 次失败才终止）。
           旧文件无 judgeStop 字段的 judge 失败轮在此按新口径重算：
           若按新口径算 pass 则不计 miss——旧被单轮冤杀的组 resume 后
           自然复活续跑。 */
        if (prev.value("judge", false)) {
          const json &pw = prev.contains("wipe") && prev["wipe"].is_object()
                               ? prev["wipe"]
                               : json::object();
          const bool pPass =
              pw.value("alphaSame", false) ||
              pw.value("typicalHit", 0) * 2 >= pw.value("typicalNeed", 1);
          if (!pPass)
            row["judgeMiss"] = row["judgeMiss"].get<int>() + 1;
        }
        /* 终止记录：dirty / collapse 拦截 / hopeless00 / judgeStop */
        if (prev.value("dirty", false) || prev.value("intercepted", false) ||
            prev.value("hopeless00", false) ||
            prev.value("judgeStop", false)) {
          pushRow();
          break;
        }
        continue;
      }

      ranAny = true;
      const json inHit = detectJson(store, current, id);
      std::cout << "  round " << r
                << " inPresent=" << inHit["present"].get<bool>()
                << " lift=" << inHit["lift"].get<double>()
                << " inChars=" << current.size() << "\n";
      if (r == 1 && !inHit["present"].get<bool>()) {
        std::cout << "  stop " << id
                  << ": seed carrier lost id before generate\n";
        break; /* 与 ps1 一致：0 轮行不入档 */
      }

      /* 并发排队下偶发超时/连接失败：重试 3 次（生成无服务端状态，
         重试安全；journal 只记真正入档的输出） */
      std::string rawOut;
      std::string lastErr;
      bool genOk = false;
      for (int attempt = 0; attempt < 3 && !genOk; ++attempt) {
        try {
          rawOut = llamaComplete(current, port, engine, modelName);
          if (isWhite(rawOut))
            throw std::runtime_error("empty llama output");
          genOk = true;
        } catch (const std::exception &e) {
          lastErr = e.what();
          fprintf(stderr, "[retry] %s round %d attempt %d: %s\n",
                  id.c_str(), r, attempt + 1, lastErr.c_str());
        }
      }
      if (!genOk) {
        /* 重试耗尽（qwen 偶发持续空输出）：不再当中止切片的致命错误。
           置空输出，走下方 isWhite -> dirty 行 -> break 的既有温柔路径：
           该模因记一个 dirty 轮入档后结束，切片内其余模因继续。 */
        fprintf(stderr,
                "[giveup] %s round %d: %s; record dirty round, stop meme only\n",
                id.c_str(), r, lastErr.c_str());
        rawOut.clear();
      }
      const std::string output = serialContinuation(rawOut);
      const bool exam = testExamDump(output);
      const bool echo = testEcho(current, output);

      if (isWhite(output) || exam) {
        std::cout << "  stop " << id
                  << ": do not ingest empty/exam dump exam=" << exam << "\n";
        const json det =
            detectJson(store, output.empty() ? rawOut : output, id);
        json rec;
        rec["mode"] = pole;
        rec["id"] = id;
        rec["round"] = r;
        rec["prompt"] = current;
        rec["output"] = rawOut;
        rec["detectIn"] = inHit;
        rec["detect"] = det;
        rec["echo"] = echo;
        rec["exam"] = exam;
        rec["dirty"] = true;
        rec["nextUtterance"] = "";
        row["rounds"].push_back(rec);
        pushRow();
        appendJournal(journalFile, rec);
        saveJsonAtomic(outFile, doc);
        break;
      }

      const json det = detectJson(store, output, id);
      const json wipe = wipeIngestJson(store, id, output, units);
      if (!wipe["emptyBefore"].get<bool>())
        throw std::runtime_error("live GNN was not empty before ingest at " +
                                 id + " round " + std::to_string(r));
      const bool judge = r >= judgeAfter;
      std::string next = wipe["recomposeText"].get<std::string>();
      bool recomposeFallback = false;
      const bool collapsed = wipe["collapsed"].get<bool>();
      if (collapsed) {
        /* collapse 拦截：token 循环/单 token 支配的输出不得引导下一代；
           只有 live 图自己的重组载体可以续链，没有就停。 */
        if (isWhite(next)) {
          std::cout << "  stop " << id << ": collapsed output (maxRun="
                    << wipe["outMaxRun"].get<int>() << " top1="
                    << wipe["outTop1Frac"].get<double>()
                    << ") and no healthy recompose carrier; intercepted\n";
          json rec;
          rec["mode"] = pole;
          rec["id"] = id;
          rec["round"] = r;
          rec["prompt"] = current;
          rec["output"] = output;
          rec["detectIn"] = inHit;
          rec["detect"] = det;
          rec["wipe"] = wipe;
          rec["echo"] = echo;
          rec["exam"] = false;
          rec["dirty"] = false;
          rec["judge"] = judge;
          rec["collapsed"] = true;
          rec["intercepted"] = true;
          rec["recomposeFallback"] = false;
          rec["nextUtterance"] = "";
          row["rounds"].push_back(rec);
          pushRow();
          appendJournal(journalFile, rec);
          saveJsonAtomic(outFile, doc);
          break;
        }
        std::cout << "  intercept " << id << ": collapsed output (maxRun="
                  << wipe["outMaxRun"].get<int>()
                  << " top1=" << wipe["outTop1Frac"].get<double>()
                  << "); continue on recompose carrier only\n";
      } else if (isWhite(next)) {
        next = output;
        recomposeFallback = true;
      }

      if (det["present"].get<bool>())
        row["presentRounds"] = row["presentRounds"].get<int>() + 1;
      if (echo)
        row["echoRounds"] = row["echoRounds"].get<int>() + 1;
      row["wipeEmptyRounds"] = row["wipeEmptyRounds"].get<int>() + 1;
      if (wipe["recomposePresent"].get<bool>())
        row["recomposePresentRounds"] =
            row["recomposePresentRounds"].get<int>() + 1;
      if (wipe["alphaSame"].get<bool>())
        row["alphaSameRounds"] = row["alphaSameRounds"].get<int>() + 1;

      json rec;
      rec["mode"] = pole;
      rec["id"] = id;
      rec["round"] = r;
      rec["prompt"] = current;
      rec["output"] = output;
      rec["detectIn"] = inHit;
      rec["detect"] = det;
      rec["wipe"] = wipe;
      rec["echo"] = echo;
      rec["exam"] = false;
      rec["dirty"] = false;
      rec["judge"] = judge;
      rec["collapsed"] = collapsed;
      rec["decayWarn"] = wipe["decayWarn"].get<bool>();
      rec["recomposeFallback"] = recomposeFallback;
      rec["nextUtterance"] = next;

      /* hopeless-00 早停：r1/r2 typicalHit 均 0 的组历史上从未恢复
         （41/41）。在写 journal 前打上标记，保证 resume 不重跑。 */
      bool hopeless = false;
      if (r == 2 && !judge) {
        int h1 = 0;
        if (!row["rounds"].empty() &&
            row["rounds"][0].contains("wipe") &&
            row["rounds"][0]["wipe"].is_object())
          h1 = row["rounds"][0]["wipe"].value("typicalHit", 0);
        const int h2 = wipe["typicalHit"].get<int>();
        if (h1 == 0 && h2 == 0)
          hopeless = true;
      }
      if (hopeless)
        rec["hopeless00"] = true;

      /* judge 降噪（2026-09 远端审计：单轮 alphaSame 一票否决噪声过
         大——同模型 test-retest phi 仅 0.238，逐模因判定噪声主导）。
         1) 通过口径放宽：alphaSame 或 typicalHit 达到 typicalNeed 一半
            （大词袋 an 组合 typical 集难全重现，比例命中即算活）；
         2) 累计 2 次 judge 失败才终止（judgeStop 显式入档，resume 按
            此标记/新口径重建）。 */
      const bool judgePass =
          wipe["alphaSame"].get<bool>() ||
          wipe["typicalHit"].get<int>() * 2 >= wipe["typicalNeed"].get<int>();
      bool judgeStop = false;
      if (judge) {
        rec["judgePass"] = judgePass;
        if (!judgePass) {
          row["judgeMiss"] = row["judgeMiss"].get<int>() + 1;
          if (row["judgeMiss"].get<int>() >= 2) {
            judgeStop = true;
            rec["judgeStop"] = true;
          }
        }
      }

      row["rounds"].push_back(rec);
      appendJournal(journalFile, rec);
      std::cout << "  round " << r
                << " outPresent=" << det["present"].get<bool>()
                << " lift=" << det["lift"].get<double>() << " echo=" << echo
                << " judge=" << judge
                << " liveNodes=" << wipe["liveNodes"].get<int>()
                << " overlap=" << wipe["overlapContent"].get<int>()
                << " alphaSame=" << wipe["alphaSame"].get<bool>()
                << " typical=" << wipe["typicalHit"].get<int>() << "/"
                << wipe["typicalNeed"].get<int>()
                << " recompose=" << wipe["recomposePresent"].get<bool>()
                << " fallback=" << recomposeFallback
                << " collapsed=" << collapsed
                << " decayWarn=" << wipe["decayWarn"].get<bool>()
                << " nextChars=" << next.size() << "\n";

      if (judgeStop) {
        std::cout << "  stop " << id
                  << ": judged twice - wiped GNN did not regrow the typical "
                     "set (miss " << row["judgeMiss"].get<int>() << ")\n";
        pushRow();
        saveJsonAtomic(outFile, doc);
        break;
      }
      if (hopeless) {
        std::cout << "  stop " << id
                  << ": hopeless-00 - no typical word reingested in r1/r2 "
                     "(41/41 historical never recovered)\n";
        pushRow();
        saveJsonAtomic(outFile, doc);
        break;
      }
      current = next;
      saveProgress();
    }

    if (!row["rounds"].empty() && !rowPushed) {
      pushRow();
      saveJsonAtomic(outFile, doc);
    }
    if (ranAny)
      ++ran;
  }

  doc["finishedAtUtc"] = utcNow();
  saveJsonAtomic(outFile, doc);
  std::cout << "Done " << outFile << "\n";
  return 0;
}
