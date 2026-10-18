/* hard_embed.cpp - Included by security_core.cpp so the host link picks
   this up with the security module. Do not also add this file to the
   link line or the symbols will be duplicated. */

#ifndef PHOENIX_HARD_EMBED_CPP
#define PHOENIX_HARD_EMBED_CPP

#include "hard_embed.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>

/* MinGW hides these behind a partition guard. Same signatures as the SDK. */
extern "C" {
WINBASEAPI WINBOOL WINAPI SetProcessMitigationPolicy(PROCESS_MITIGATION_POLICY MitigationPolicy,
                                                     PVOID lpBuffer, SIZE_T dwLength);
WINBASEAPI WINBOOL WINAPI GetProcessMitigationPolicy(HANDLE hProcess,
                                                     PROCESS_MITIGATION_POLICY MitigationPolicy,
                                                     PVOID lpBuffer, SIZE_T dwLength);
}
#endif

namespace phoenix {
namespace secamp {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kDefaultHashCap = 32u * 1024u * 1024u;

std::once_flag g_once;
ProcessMitigationReport g_report;
std::atomic<bool> g_forceUnsealed{false};
std::atomic<bool> g_barrier{false};
std::atomic<HardEmbedProbeFn> g_latched{nullptr};
std::atomic<HardEmbedProbeFn> g_handler{nullptr};
std::atomic<std::size_t> g_hashCap{kDefaultHashCap};
std::mutex g_mu;
std::string g_root;
PathDecision g_audit;

std::string lowerCopy(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string slashToBack(std::string s) {
  for (char &c : s)
    if (c == '/')
      c = '\\';
  return s;
}

std::string trimTrailingSep(std::string s) {
  while (s.size() > 1 && (s.back() == '\\' || s.back() == '/'))
    s.pop_back();
  return s;
}

std::string stripExtended(std::string s) {
  s = slashToBack(std::move(s));
  if (s.rfind("\\\\?\\", 0) == 0)
    s = s.substr(4);
  return s;
}

bool hexNibble(char c, unsigned &out) {
  if (c >= '0' && c <= '9') {
    out = static_cast<unsigned>(c - '0');
    return true;
  }
  if (c >= 'a' && c <= 'f') {
    out = static_cast<unsigned>(c - 'a' + 10);
    return true;
  }
  if (c >= 'A' && c <= 'F') {
    out = static_cast<unsigned>(c - 'A' + 10);
    return true;
  }
  return false;
}

std::string toHex(const unsigned char *p, std::size_t n) {
  static const char *k = "0123456789abcdef";
  std::string o(n * 2, '0');
  for (std::size_t i = 0; i < n; ++i) {
    o[i * 2] = k[p[i] >> 4];
    o[i * 2 + 1] = k[p[i] & 0x0f];
  }
  return o;
}

bool ctHexEq(const std::string &a, const std::string &b) {
  if (a.size() != 64 || b.size() != 64)
    return false;
  unsigned diff = 0;
  for (std::size_t i = 0; i < 64; ++i)
    diff |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
  return diff == 0;
}

std::string normalizePin(const std::string &pin, bool *ok) {
  *ok = false;
  if (pin.size() != 64)
    return {};
  std::string o = pin;
  for (char &c : o) {
    unsigned n = 0;
    if (!hexNibble(c, n))
      return {};
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  *ok = true;
  return o;
}

struct Sha256 {
  std::uint32_t s[8]{};
  std::uint64_t bits{0};
  unsigned char buf[64]{};
  std::size_t n{0};

  static std::uint32_t rotr(std::uint32_t x, std::uint32_t r) {
    return (x >> r) | (x << (32 - r));
  }

  void init() {
    s[0] = 0x6a09e667u;
    s[1] = 0xbb67ae85u;
    s[2] = 0x3c6ef372u;
    s[3] = 0xa54ff53au;
    s[4] = 0x510e527fu;
    s[5] = 0x9b05688cu;
    s[6] = 0x1f83d9abu;
    s[7] = 0x5be0cd19u;
    bits = 0;
    n = 0;
  }

  void block(const unsigned char b[64]) {
    static const std::uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (std::uint32_t)b[i * 4] << 24 | (std::uint32_t)b[i * 4 + 1] << 16 |
             (std::uint32_t)b[i * 4 + 2] << 8 | (std::uint32_t)b[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 =
          rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 =
          rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = s[0], b0 = s[1], c = s[2], d = s[3], e = s[4], f = s[5],
                  g = s[6], h = s[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
      const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b0) ^ (a & c) ^ (b0 & c);
      const std::uint32_t t2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b0;
      b0 = a;
      a = t1 + t2;
    }
    s[0] += a;
    s[1] += b0;
    s[2] += c;
    s[3] += d;
    s[4] += e;
    s[5] += f;
    s[6] += g;
    s[7] += h;
  }

  void update(const unsigned char *p, std::size_t len) {
    bits += static_cast<std::uint64_t>(len) * 8u;
    while (len) {
      const std::size_t take = std::min(len, static_cast<std::size_t>(64 - n));
      std::memcpy(buf + n, p, take);
      n += take;
      p += take;
      len -= take;
      if (n == 64) {
        block(buf);
        n = 0;
      }
    }
  }

  void final(unsigned char out[32]) {
    buf[n++] = 0x80;
    if (n > 56) {
      while (n < 64)
        buf[n++] = 0;
      block(buf);
      n = 0;
    }
    while (n < 56)
      buf[n++] = 0;
    for (int i = 7; i >= 0; --i)
      buf[n++] = static_cast<unsigned char>((bits >> (i * 8)) & 0xffu);
    block(buf);
    for (int i = 0; i < 8; ++i) {
      out[i * 4] = static_cast<unsigned char>(s[i] >> 24);
      out[i * 4 + 1] = static_cast<unsigned char>(s[i] >> 16);
      out[i * 4 + 2] = static_cast<unsigned char>(s[i] >> 8);
      out[i * 4 + 3] = static_cast<unsigned char>(s[i]);
    }
  }
};

bool looksDeviceOrUnc(const std::string &raw) {
  const std::string s = slashToBack(raw);
  if (s.rfind("\\\\", 0) == 0)
    return true;
  if (s.find("\\\\.\\") != std::string::npos || s.find("\\\\?\\") != std::string::npos)
    return true;
  return false;
}

bool badSegment(const std::string &seg, std::string *reason) {
  if (seg.empty()) {
    *reason = "bad-segment";
    return true;
  }
  if (seg == "..") {
    *reason = "path-escape";
    return true;
  }
  if (seg == ".")
    return false;
  if (seg.find(':') != std::string::npos) {
    *reason = "alternate-stream";
    return true;
  }
  if (seg.back() == '.' || seg.back() == ' ') {
    *reason = "bad-segment";
    return true;
  }
  const std::string u = [&] {
    std::string t = seg;
    for (char &c : t)
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return t;
  }();
  if (u == "CON" || u == "PRN" || u == "AUX" || u == "NUL") {
    *reason = "device-name";
    return true;
  }
  if (u.size() == 4 && (u.compare(0, 3, "COM") == 0 || u.compare(0, 3, "LPT") == 0) &&
      u[3] >= '1' && u[3] <= '9') {
    *reason = "device-name";
    return true;
  }
  return false;
}

std::vector<std::string> splitRel(const std::string &rel, std::string *reason) {
  std::vector<std::string> segs;
  std::string cur;
  for (char c : rel) {
    if (c == '/' || c == '\\') {
      if (cur.empty()) {
        *reason = "bad-segment";
        return {};
      }
      if (badSegment(cur, reason))
        return {};
      segs.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) {
    if (badSegment(cur, reason))
      return {};
    segs.push_back(cur);
  }
  return segs;
}

std::string displayPath(const fs::path &p) {
  std::string s = stripExtended(p.lexically_normal().string());
  return trimTrailingSep(lowerCopy(s));
}

#ifdef _WIN32
bool finalPathOfExisting(const fs::path &p, fs::path *out) {
  HANDLE h = CreateFileW(p.wstring().c_str(), FILE_READ_ATTRIBUTES,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (h == INVALID_HANDLE_VALUE)
    return false;
  wchar_t buf[4096];
  const DWORD n = GetFinalPathNameByHandleW(h, buf, 4096, FILE_NAME_NORMALIZED);
  CloseHandle(h);
  if (n == 0 || n >= 4096)
    return false;
  std::wstring ws(buf, n);
  if (ws.rfind(L"\\\\?\\", 0) == 0)
    ws = ws.substr(4);
  *out = fs::path(ws);
  return true;
}
#endif

bool insideRoot(const fs::path &root, const fs::path &target) {
  const std::string r = displayPath(root);
  const std::string t = displayPath(target);
  if (r.empty() || t.empty())
    return false;
  if (t == r)
    return true;
  if (t.size() > r.size() && t.compare(0, r.size(), r) == 0 &&
      (t[r.size()] == '\\' || t[r.size()] == '/'))
    return true;
  return false;
}

std::vector<std::string> sensitivePrefixes() {
  std::vector<std::string> out;
#ifdef _WIN32
  char buf[MAX_PATH];
  const UINT wn = GetWindowsDirectoryA(buf, MAX_PATH);
  if (wn > 0 && wn < MAX_PATH)
    out.push_back(trimTrailingSep(lowerCopy(buf)));
  const UINT sn = GetSystemDirectoryA(buf, MAX_PATH);
  if (sn > 0 && sn < MAX_PATH)
    out.push_back(trimTrailingSep(lowerCopy(buf)));
  if (GetEnvironmentVariableA("ProgramFiles", buf, MAX_PATH) > 0)
    out.push_back(trimTrailingSep(lowerCopy(buf)));
  if (GetEnvironmentVariableA("ProgramFiles(x86)", buf, MAX_PATH) > 0)
    out.push_back(trimTrailingSep(lowerCopy(buf)));
  if (GetEnvironmentVariableA("ProgramData", buf, MAX_PATH) > 0)
    out.push_back(trimTrailingSep(lowerCopy(std::string(buf) + "\\Microsoft")));
#else
  out = {"/etc", "/usr", "/bin", "/sbin", "/lib", "/proc", "/sys", "/boot", "/dev"};
#endif
  return out;
}

bool prefixHit(const std::string &pathLower, const std::string &prefixLower) {
  if (prefixLower.empty() || pathLower.size() < prefixLower.size())
    return false;
  if (pathLower.compare(0, prefixLower.size(), prefixLower) != 0)
    return false;
  if (pathLower.size() == prefixLower.size())
    return true;
  const char n = pathLower[prefixLower.size()];
  return n == '\\' || n == '/';
}

std::uint16_t readMappedDllChars(const void *base, bool *ok) {
  *ok = false;
  if (!base)
    return 0;
#ifdef _WIN32
  const auto *bytes = static_cast<const unsigned char *>(base);
  const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(bytes);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE)
    return 0;
  if (dos->e_lfanew <= 0)
    return 0;
  const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(bytes + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE)
    return 0;
  *ok = true;
  return nt->OptionalHeader.DllCharacteristics;
#else
  (void)base;
  return 0;
#endif
}

bool readFileDllChars(const std::string &path, std::uint16_t *chars, std::string *err) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *err = "unreadable";
    return false;
  }
  std::vector<unsigned char> buf((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  if (buf.size() < 0x40) {
    *err = "truncated";
    return false;
  }
  const int peOff = static_cast<int>(buf[0x3c] | (buf[0x3d] << 8) | (buf[0x3e] << 16) |
                                     (buf[0x3f] << 24));
  if (peOff <= 0 || static_cast<std::size_t>(peOff) + 24 + 72 > buf.size()) {
    *err = "bad-pe";
    return false;
  }
  if (buf[peOff] != 'P' || buf[peOff + 1] != 'E' || buf[peOff + 2] != 0 || buf[peOff + 3] != 0) {
    *err = "bad-pe";
    return false;
  }
  const std::size_t opt = static_cast<std::size_t>(peOff) + 24;
  const unsigned magic = buf[opt] | (buf[opt + 1] << 8);
  if (magic != 0x10b && magic != 0x20b) {
    *err = "bad-pe";
    return false;
  }
  const std::size_t dllAt = opt + 70;
  *chars = static_cast<std::uint16_t>(buf[dllAt] | (buf[dllAt + 1] << 8));
  return true;
}

void applyChars(std::uint16_t dll, ProcessMitigationReport *r) {
  r->depImage = (dll & 0x0100) != 0;
  r->aslrImage = (dll & 0x0040) != 0;
  r->highEntropyImage = (dll & 0x0020) != 0;
  r->cfgImage = (dll & 0x4000) != 0;
}

ProcessMitigationReport applyMitigations() {
  ProcessMitigationReport r;
  std::string miss;
#ifdef _WIN32
  bool ok = false;
  const std::uint16_t exeChars = readMappedDllChars(GetModuleHandleA(nullptr), &ok);
  if (!ok)
    miss += "exe-pe-unreadable ";
  else
    applyChars(exeChars, &r);
  HMODULE self = nullptr;
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                         GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCSTR>(&ensureProcessHardEmbed), &self);
  if (self && self != GetModuleHandleA(nullptr)) {
    bool sok = false;
    const std::uint16_t sc = readMappedDllChars(self, &sok);
    if (!sok)
      miss += "module-pe-unreadable ";
    else {
      ProcessMitigationReport m;
      applyChars(sc, &m);
      r.depImage = r.depImage && m.depImage;
      r.aslrImage = r.aslrImage && m.aslrImage;
      r.highEntropyImage = r.highEntropyImage && m.highEntropyImage;
    }
  }
  if (!r.depImage)
    miss += "dep-image ";
  if (!r.aslrImage)
    miss += "aslr-image ";
#if defined(_WIN64)
  if (!r.highEntropyImage)
    miss += "high-entropy-image ";
#endif

  PROCESS_MITIGATION_DEP_POLICY dep{};
  if (GetProcessMitigationPolicy(GetCurrentProcess(), ProcessDEPPolicy, &dep, sizeof dep))
    r.depPolicy = dep.Enable || dep.Permanent;
  if (!r.depPolicy)
    miss += "dep-policy ";

  PROCESS_MITIGATION_EXTENSION_POINT_DISABLE_POLICY ext{};
  ext.DisableExtensionPoints = 1;
  SetProcessMitigationPolicy(ProcessExtensionPointDisablePolicy, &ext, sizeof ext);
  PROCESS_MITIGATION_EXTENSION_POINT_DISABLE_POLICY extGot{};
  if (GetProcessMitigationPolicy(GetCurrentProcess(), ProcessExtensionPointDisablePolicy,
                                 &extGot, sizeof extGot))
    r.extensionPointsDisabled = extGot.DisableExtensionPoints != 0;
  if (!r.extensionPointsDisabled)
    miss += "extension-points ";

  PROCESS_MITIGATION_IMAGE_LOAD_POLICY img{};
  img.NoRemoteImages = 1;
  img.NoLowMandatoryLabelImages = 1;
  img.PreferSystem32Images = 1;
  SetProcessMitigationPolicy(ProcessImageLoadPolicy, &img, sizeof img);
  PROCESS_MITIGATION_IMAGE_LOAD_POLICY imgGot{};
  if (GetProcessMitigationPolicy(GetCurrentProcess(), ProcessImageLoadPolicy, &imgGot,
                                 sizeof imgGot)) {
    r.noRemoteImages = imgGot.NoRemoteImages != 0;
    r.noLowMandatoryLabelImages = imgGot.NoLowMandatoryLabelImages != 0;
    r.preferSystem32Images = imgGot.PreferSystem32Images != 0;
  }
  if (!r.noRemoteImages)
    miss += "no-remote-images ";
  if (!r.noLowMandatoryLabelImages)
    miss += "no-low-integrity-images ";
  if (!r.preferSystem32Images)
    miss += "prefer-system32 ";

  if (HeapSetInformation(nullptr, HeapEnableTerminationOnCorruption, nullptr, 0))
    r.heapTerminateOnCorruption = true;
  if (!r.heapTerminateOnCorruption)
    miss += "heap-terminate ";
#else
  miss = "non-windows ";
#endif
  if (!miss.empty() && miss.back() == ' ')
    miss.pop_back();
  r.sealed = miss.empty();
  r.detail = r.sealed ? "ok" : miss;
  return r;
}

} // namespace

std::string sha256Bytes(const void *data, std::size_t len) {
  Sha256 h;
  h.init();
  h.update(static_cast<const unsigned char *>(data), len);
  unsigned char out[32];
  h.final(out);
  return toHex(out, 32);
}

std::string sha256File(const std::string &path, std::string *error) {
  if (error)
    error->clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error)
      *error = "missing";
    return {};
  }
  Sha256 h;
  h.init();
  const std::size_t cap = g_hashCap.load(std::memory_order_acquire);
  std::size_t total = 0;
  char buf[1 << 15];
  while (in) {
    in.read(buf, sizeof buf);
    const std::streamsize got = in.gcount();
    if (got <= 0)
      break;
    total += static_cast<std::size_t>(got);
    if (total > cap) {
      if (error)
        *error = "too-large";
      return {};
    }
    h.update(reinterpret_cast<const unsigned char *>(buf),
             static_cast<std::size_t>(got));
  }
  if (in.bad()) {
    if (error)
      *error = "unreadable";
    return {};
  }
  unsigned char out[32];
  h.final(out);
  return toHex(out, 32);
}

const ProcessMitigationReport &ensureProcessHardEmbed() {
  std::call_once(g_once, [] { g_report = applyMitigations(); });
  return g_report;
}

ProcessMitigationReport hardEmbedReport() { return ensureProcessHardEmbed(); }

bool mitigationsSealed() {
  if (g_forceUnsealed.load(std::memory_order_acquire))
    return false;
  return ensureProcessHardEmbed().sealed;
}

void bindEstopProbes(HardEmbedProbeFn latched, HardEmbedProbeFn handlerInstalled) {
  g_latched.store(latched, std::memory_order_release);
  g_handler.store(handlerInstalled, std::memory_order_release);
}

void notifyMemeBarrierAttached() { g_barrier.store(true, std::memory_order_release); }

bool memeBarrierAttached() { return g_barrier.load(std::memory_order_acquire); }

bool estopProbeBound() {
  return g_latched.load(std::memory_order_acquire) != nullptr &&
         g_handler.load(std::memory_order_acquire) != nullptr;
}

bool estopLatched() {
  const HardEmbedProbeFn fn = g_latched.load(std::memory_order_acquire);
  if (!fn)
    return false;
  return fn();
}

bool estopHandlerInstalled() {
  const HardEmbedProbeFn fn = g_handler.load(std::memory_order_acquire);
  if (!fn)
    return false;
  return fn();
}

bool estopRequiresBlock() { return estopLatched(); }

bool safetyWiringClosed() {
  return mitigationsSealed() && memeBarrierAttached() && estopProbeBound() &&
         !estopLatched();
}

bool isSensitivePath(const std::string &path) {
  if (path.empty() || looksDeviceOrUnc(path))
    return true;
  std::error_code ec;
  fs::path abs = fs::absolute(fs::path(path), ec);
  if (ec)
    abs = fs::path(path);
  std::string shown = displayPath(abs.lexically_normal());
#ifdef _WIN32
  if (shown.size() >= 2 && shown[1] == ':') {
    /* drive-relative or absolute already normalized */
  }
#endif
  for (const auto &pre : sensitivePrefixes()) {
    if (prefixHit(shown, lowerCopy(slashToBack(pre))))
      return true;
  }
  return false;
}

PathDecision auditWorkspaceRoot(const std::string &root) {
  PathDecision d;
  if (!mitigationsSealed()) {
    d.reason = "mitigations-not-sealed";
    return d;
  }
  if (root.empty()) {
    d.reason = "empty-path";
    return d;
  }
  if (root.find("..") != std::string::npos) {
    d.reason = "path-escape";
    return d;
  }
  if (looksDeviceOrUnc(root) || isSensitivePath(root)) {
    d.reason = "sensitive-path";
    return d;
  }
  std::error_code ec;
  fs::path abs = fs::absolute(fs::path(root), ec);
  d.canonical = ec ? root : abs.lexically_normal().string();
  d.allowed = true;
  d.reason = "ok";
  return d;
}

void noteWorkspaceRoot(const std::string &root) {
  const PathDecision audit = auditWorkspaceRoot(root);
  std::lock_guard<std::mutex> lock(g_mu);
  g_root = root;
  g_audit = audit;
}

PathDecision workspaceRootAudit() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_root.empty() && g_audit.reason.empty()) {
    PathDecision d;
    d.reason = "not-noted";
    return d;
  }
  return g_audit;
}

PathDecision confineToWorkspace(const std::string &workspaceRoot,
                                const std::string &relative) {
  PathDecision d;
  if (!mitigationsSealed()) {
    d.reason = "mitigations-not-sealed";
    return d;
  }
  if (!estopProbeBound() || !memeBarrierAttached()) {
    d.reason = "safety-unwired";
    return d;
  }
  if (estopLatched()) {
    d.reason = "emergency-stop";
    return d;
  }
  if (workspaceRoot.empty() || relative.empty()) {
    d.reason = "empty-path";
    return d;
  }
  if (relative.back() == '/' || relative.back() == '\\') {
    d.reason = "bad-segment";
    return d;
  }
  if (!relative.empty() && (relative[0] == '/' || relative[0] == '\\' ||
                            (relative.size() > 1 && relative[1] == ':'))) {
    d.reason = "absolute-path";
    return d;
  }
  if (looksDeviceOrUnc(workspaceRoot) || isSensitivePath(workspaceRoot)) {
    d.reason = "sensitive-path";
    return d;
  }
  std::string segReason;
  const std::vector<std::string> segs = splitRel(relative, &segReason);
  if (!segReason.empty()) {
    d.reason = segReason;
    return d;
  }
  if (segs.empty()) {
    d.reason = "empty-path";
    return d;
  }
  std::error_code ec;
  const fs::path root = fs::weakly_canonical(fs::absolute(fs::path(workspaceRoot)), ec);
  if (ec || !fs::is_directory(root, ec)) {
    d.reason = "root-missing";
    return d;
  }
  fs::path cur = root;
  for (const auto &seg : segs) {
    if (seg == ".")
      continue;
    const fs::path next = cur / seg;
#ifdef _WIN32
    const DWORD attr = GetFileAttributesW(next.wstring().c_str());
    if (attr != INVALID_FILE_ATTRIBUTES) {
      fs::path resolved;
      if (!finalPathOfExisting(next, &resolved) || !insideRoot(root, resolved)) {
        d.reason = "path-escape";
        return d;
      }
      cur = resolved;
    } else {
#else
    std::error_code sec;
    const bool exists = fs::exists(next, sec);
    if (!sec && exists) {
      const fs::path resolved = fs::weakly_canonical(next, sec);
      if (sec || !insideRoot(root, resolved)) {
        d.reason = "path-escape";
        return d;
      }
      cur = resolved;
    } else {
#endif
      cur = next;
      if (!insideRoot(root, cur.lexically_normal())) {
        d.reason = "path-escape";
        return d;
      }
    }
  }
  if (!insideRoot(root, cur) || displayPath(cur) == displayPath(root)) {
    d.reason = displayPath(cur) == displayPath(root) ? "empty-path" : "path-escape";
    return d;
  }
  if (isSensitivePath(cur.string())) {
    d.reason = "sensitive-path";
    return d;
  }
  d.allowed = true;
  d.reason = "ok";
  d.canonical = cur.string();
  return d;
}

IntegrityDecision verifyPinnedFile(const std::string &path, const std::string &expectedHex) {
  IntegrityDecision d;
  if (path.empty()) {
    d.reason = "empty-path";
    return d;
  }
  if (path.find("..") != std::string::npos) {
    d.reason = "path-escape";
    return d;
  }
  if (looksDeviceOrUnc(path) || isSensitivePath(path)) {
    d.reason = "sensitive-path";
    return d;
  }
  bool pinOk = false;
  const std::string pin = normalizePin(expectedHex, &pinOk);
  if (expectedHex.empty()) {
    d.reason = "unpinned";
    return d;
  }
  if (!pinOk) {
    d.reason = "bad-pin";
    return d;
  }
  std::string err;
  const std::string got = sha256File(path, &err);
  if (!err.empty() || got.empty()) {
    d.reason = err.empty() ? "missing" : err;
    return d;
  }
  d.sha256 = got;
  if (!ctHexEq(got, pin)) {
    d.reason = "mismatch";
    return d;
  }
  d.allowed = true;
  d.reason = "ok";
  return d;
}

IntegrityDecision allowArtifactLoad(const std::string &path, const std::string &expectedHex) {
  IntegrityDecision d;
  if (!mitigationsSealed()) {
    d.reason = "mitigations-not-sealed";
    return d;
  }
  if (path.empty()) {
    d.reason = "empty-path";
    return d;
  }
  if (path.find("..") != std::string::npos) {
    d.reason = "path-escape";
    return d;
  }
  if (looksDeviceOrUnc(path) || isSensitivePath(path)) {
    d.reason = "sensitive-path";
    return d;
  }
  if (!estopProbeBound() || !memeBarrierAttached()) {
    d.reason = "safety-unwired";
    return d;
  }
  if (estopLatched()) {
    d.reason = "emergency-stop";
    return d;
  }
  return verifyPinnedFile(path, expectedHex);
}

PeFileAudit auditPeFile(const std::string &path) {
  PeFileAudit a;
#ifndef _WIN32
  (void)path;
  a.detail = "non-windows";
  return a;
#else
  std::uint16_t chars = 0;
  std::string err;
  if (!readFileDllChars(path, &chars, &err)) {
    a.detail = err.empty() ? "bad-pe" : err;
    return a;
  }
  a.ok = true;
  a.dep = (chars & 0x0100) != 0;
  a.aslr = (chars & 0x0040) != 0;
  a.highEntropy = (chars & 0x0020) != 0;
  a.cfg = (chars & 0x4000) != 0;
  a.detail = "ok";
  return a;
#endif
}

void testOnlyForceUnsealed(bool on) {
  g_forceUnsealed.store(on, std::memory_order_release);
}

void testOnlyResetWiring() {
  g_forceUnsealed.store(false, std::memory_order_release);
  g_barrier.store(false, std::memory_order_release);
  g_latched.store(nullptr, std::memory_order_release);
  g_handler.store(nullptr, std::memory_order_release);
  g_hashCap.store(kDefaultHashCap, std::memory_order_release);
  std::lock_guard<std::mutex> lock(g_mu);
  g_root.clear();
  g_audit = PathDecision{};
}

void testOnlySetHashCap(std::size_t bytes) {
  g_hashCap.store(bytes == 0 ? kDefaultHashCap : bytes, std::memory_order_release);
}

} // namespace secamp
} // namespace phoenix

#ifdef _WIN32
#ifdef ERROR
#undef ERROR
#endif
#endif

#endif
